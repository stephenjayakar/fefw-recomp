// SPDX-License-Identifier: GPL-3.0-or-later
// Pipelines linked from separately compiled parts (S24, VK_EXT_graphics_pipeline_library).
//
// A pipeline made whole costs 5 to 50 ms on the render thread the first time the driver sees its shaders, because the draw is the first
// moment its blend state, attachment formats and vertex layout are known. Almost all of that time is shader compilation, and the shader code
// depends on none of those. So a pipeline is split in four parts:
//
//   vertex input     attribute formats, strides' rate, topology class          (cheap, made at the draw)
//   pre-raster       vertex (+ geometry) code, raster flags, program layout    (compiled ahead on background threads)
//   fragment shader  fragment code, program layout                             (compiled ahead on background threads)
//   fragment output  blend, attachment formats, samples                        (cheap, made at the draw)
//
// When the game sets a program's shaders (nvnProgramSetShaders, at load time) the two shader parts are queued for the background threads;
// the draw makes the cheap parts and links the four, which takes a fraction of a millisecond. A part that is not ready is built (or waited
// for) by the draw, so the worst case is the cost of one shader part. Nothing is guessed that could end in a second compile: a part is
// keyed by exactly what it was compiled with (module, layout, raster flags), a draw whose flags differ from the prebuilt ones builds its own
// part and teaches the prediction (the flags are added to the set that later programs get ahead).
//
// Pipelines whose exact key is in the key file are built whole by the prewarm (nvn_prewarm.cpp) and never reach this file.
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <unordered_set>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "nxrt/config.h"
#include "nxrt/ftrace.h"
#include "nxrt/gauge.h"
#include "nxrt/log.h"
#include "nxrt/overlay.h"
#include "nvn_gpu_internal.h"
#include "nvn_pipeline.h"

namespace nxrt::nvn::be::gpu {
namespace {

constexpr const char* kTag = "nvn.gpl";

// The raster flags the pre-raster part is compiled with. The default is what the console API starts a draw with (fill, depth clamp on,
// clip space as the console has it); flags that draws show later are added.
struct RasterKey {
    uint8_t poly = 0, clamp = 1, neg = 1, pad = 0;
    bool operator==(const RasterKey& o) const { return poly == o.poly && clamp == o.clamp && neg == o.neg; }
};

struct Job {
    const ShaderInfo* st[3] = {};
    RasterKey r;
    bool fragment = false;      // the fragment shader part (r unused) instead of the pre-raster part
    };

struct Entry {
    int state = 0;              // 1 building, 2 done (pipe may be null: the build failed)
    VkPipeline pipe = VK_NULL_HANDLE;
    void* owner = nullptr;      // the background thread building it (Windows thread handle), for the boost below
    std::atomic<bool> boosted{false};
};

std::mutex g_mu;
std::condition_variable g_cv;
bool g_inited = false, g_enabled = false, g_stop = false, g_lto = false, g_cold = true;
std::atomic<bool> g_on{false};
std::atomic<bool> g_discard{false};
std::atomic<bool> g_prepare_cache{false};      // shader preparation keeps what it compiled in the pipeline cache file ([nvn] prepare_cache)
std::atomic<uint64_t> g_prep_fast{0};          // shader preparation: parts that took under 2 ms (the driver found them in its cache)          // shader preparation: a part is thrown away as soon as it is made (the driver keeps what it compiled)
std::unordered_map<std::string, Entry> g_libs;             // node based: a reference to an entry stays valid while others are added
std::deque<std::function<void()>> g_exact_q;   // [nvn] gpl_async: parts a deferred draw needs, exactly as it needs them (first in line)
std::unordered_set<std::string> g_exact_pending;
std::atomic<int64_t> g_frame_seen_ns{0};      // when the replay last saw the present count move
std::deque<std::function<void()>> g_pre_q;     // parts of the key file's pipelines (gpl_prebuild): after the hot programs, before the cold ones
std::deque<Job> g_hot_q, g_cold_q;    // hot: programs a command buffer was recorded with (oldest first); cold: set but not yet bound (newest first)
std::vector<RasterKey> g_rasters;
std::deque<std::array<const ShaderInfo*, 3>> g_programs;   // the most recent programs, for the raster flags that show up later
std::set<std::array<const ShaderInfo*, 3>> g_seen, g_hot, g_covered;
std::vector<std::thread> g_workers;
// Optimised replacements for linked pipelines (gpl_upgrade): the draw links first, a background thread then makes the whole pipeline (or links the
// same parts with link-time optimisation) and the pipeline lookup swaps it in.
enum class UpMode { Off, Whole, Lto };
struct UpJob {
    GraphicsKey key;
    const ProgLayout* layout = nullptr;
    VkPipeline parts[4] = {};
    std::shared_ptr<PipeUpgrade> up;
};
UpMode g_up_mode = UpMode::Off;
bool g_retain = false;                     // parts keep what a link-time optimised link needs
std::deque<UpJob> g_up_q;
std::condition_variable g_up_cv;
std::vector<std::thread> g_up_workers;
int g_up_running = 0;
GplStats g_stats;
uint64_t g_logged_sig = 0;
int g_fail_streak = 0;
int g_running = 0;                         // jobs being built by the workers right now

// Tells the overlay how much shader work is waiting (queued + running). Caller holds g_mu.
void publish_backlog_locked() { nxrt::gfx::set_shader_backlog(static_cast<int>(g_exact_q.size() + g_hot_q.size() + g_pre_q.size() + g_cold_q.size() + g_up_q.size()) + g_running + g_up_running); }

thread_local const GraphicsKey* t_link_key = nullptr;      // the draw being linked on this thread, for the log line of a slow part

void predict_int_attributes(GraphicsKey& k);

// Measurement only ([nvn] gpl_ab_pause = N presents): the background threads work in odd windows of N presents and rest in even ones, so the frame time of the two
// kinds of window shows what the compile threads cost the game. 0 = always work.
bool ab_pause_on() { return config_get_int("nvn", "gpl_ab_pause", 0) > 0; }
bool workers_paused() {
    static const int64_t frames = config_get_int("nvn", "gpl_ab_pause", 0);
    if (frames <= 0) return false;
    static std::atomic<int64_t> last{-1};
    const int64_t w = static_cast<int64_t>(counters().presents.load(std::memory_order_relaxed)) / frames;
    if (last.exchange(w) != w) NXLOG_I(kTag, "ab: present %lld starts a window with the compile threads %s", static_cast<long long>(w * frames), (w & 1) ? "working" : "resting");
    return (w & 1) == 0;
}

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

template <class T>
std::string bytes_of(char tag, const T& v) {
    std::string s(1, tag);
    s.append(reinterpret_cast<const char*>(&v), sizeof v);
    return s;
}

// ---- keys: plain zero-initialised structs, compared as bytes ----------------------------------------------------------------------
struct PreKey { uint64_t vs, gs, layout; uint8_t poly, clamp, neg, pad; uint32_t pad2; };
struct FsKey { uint64_t fs, layout; };
struct ViKey { uint32_t count; uint32_t pad; KeyAttr attr[16]; uint8_t instance[16]; uint8_t topology; uint8_t pad2[7]; };
struct FoKey {
    uint32_t color_format[8], color_count, depth_format, stencil_format, samples;
    uint8_t blend_enable[8], blend_src_rgb[8], blend_dst_rgb[8], blend_src_a[8], blend_dst_a[8], blend_op_rgb[8], blend_op_a[8], write_mask[8];
    uint8_t logic_op_enable, logic_op, alpha_to_coverage, pad[5];
};

std::string pre_key(const StageModules& sm, const ProgLayout& pl, const RasterKey& r) {
    PreKey k;
    std::memset(&k, 0, sizeof k);
    k.vs = module_content_id(sm.m[0]);           // by content: the parts built for the key file's stand-in shaders serve the game's own
    k.gs = sm.m[2] ? module_content_id(sm.m[2]) : 0;
    k.layout = reinterpret_cast<uint64_t>(pl.layout);
    k.poly = r.poly; k.clamp = r.clamp; k.neg = r.neg;
    return bytes_of('P', k);
}

std::string fs_key(const StageModules& sm, const ProgLayout& pl) {
    FsKey k = {module_content_id(sm.m[1]), reinterpret_cast<uint64_t>(pl.layout)};
    return bytes_of('F', k);
}

std::string vi_key(const GraphicsKey& key) {
    ViKey k;
    std::memset(&k, 0, sizeof k);
    k.count = std::min<uint32_t>(key.attr_count, 16);
    for (uint32_t i = 0; i < k.count; ++i) k.attr[i] = key.attr[i];
    std::memcpy(k.instance, key.binding_instance, 16);
    k.topology = key.topology;
    return bytes_of('V', k);
}

std::string fo_key(const GraphicsKey& key) {
    FoKey k;
    std::memset(&k, 0, sizeof k);
    const uint32_t n = std::min<uint32_t>(key.color_count, 8);
    k.color_count = n;
    for (uint32_t i = 0; i < n; ++i) {
        k.color_format[i] = key.color_format[i];
        k.blend_enable[i] = key.blend_enable[i]; k.blend_src_rgb[i] = key.blend_src_rgb[i]; k.blend_dst_rgb[i] = key.blend_dst_rgb[i];
        k.blend_src_a[i] = key.blend_src_a[i]; k.blend_dst_a[i] = key.blend_dst_a[i]; k.blend_op_rgb[i] = key.blend_op_rgb[i];
        k.blend_op_a[i] = key.blend_op_a[i]; k.write_mask[i] = key.write_mask[i];
    }
    k.depth_format = key.depth_format; k.stencil_format = key.stencil_format; k.samples = key.samples ? key.samples : 1;
    k.logic_op_enable = key.logic_op_enable; k.logic_op = key.logic_op; k.alpha_to_coverage = key.alpha_to_coverage;
    return bytes_of('O', k);
}

// ---- the four builders ------------------------------------------------------------------------------------------------------------
VkPipeline create_part(VkGraphicsPipelineCreateInfo& gi, const char* what) {
    gi.flags |= VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
    if (g_retain) gi.flags |= VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT;
    VkPipeline p = VK_NULL_HANDLE;
    const auto t0 = Clock::now();
    // shader preparation keeps nothing in the app cache file (it would grow to hundreds of MB); the driver's own cache holds the result
    const VkResult r = create_graphics_pipeline_tracked(vkdev(), (g_discard.load(std::memory_order_relaxed) && !g_prepare_cache.load(std::memory_order_relaxed)) ? VK_NULL_HANDLE : pipeline_cache_handle(), &gi, &p);
    if (r != VK_SUCCESS) {
        NXLOG_E(kTag, "pipeline part (%s) failed (%d)", what, static_cast<int>(r));
        return VK_NULL_HANDLE;
    }
    note_pipeline_created(ms_since(t0), what);
    return p;
}

VkPipeline build_pre(VkShaderModule vs, VkShaderModule gs, const ProgLayout& pl, const RasterKey& r) {
    VkPipelineShaderStageCreateInfo st[2] = {};
    uint32_t n = 0;
    auto add = [&](VkShaderStageFlagBits bit, VkShaderModule m) {
        if (!m) return;
        st[n] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, ubo_robustness()};
        st[n].stage = bit;
        st[n].module = m;
        st[n].pName = "main";
        ++n;
    };
    add(VK_SHADER_STAGE_VERTEX_BIT, vs);
    add(VK_SHADER_STAGE_GEOMETRY_BIT, gs);
    VkPipelineViewportDepthClipControlCreateInfoEXT dcc = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT};
    dcc.negativeOneToOne = r.neg ? VK_TRUE : VK_FALSE;
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    if (device().features().depth_clip_control) vp.pNext = &dcc;
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.depthClampEnable = r.clamp ? VK_TRUE : VK_FALSE;
    rs.polygonMode = static_cast<VkPolygonMode>(r.poly);
    rs.lineWidth = 1.0f;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkGraphicsPipelineLibraryCreateInfoEXT li = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, &ri,
                                                 VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT};
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &li};
    gi.stageCount = n;
    gi.pStages = st;
    gi.pViewportState = &vp;
    gi.pRasterizationState = &rs;
    gi.pDynamicState = pipeline_dynamic_state();
    gi.layout = pl.layout;
    return create_part(gi, "pre-raster part");
}

VkPipeline build_fs(VkShaderModule fs, const ProgLayout& pl) {
    VkPipelineShaderStageCreateInfo st = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, ubo_robustness()};
    st.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    st.module = fs;
    st.pName = "main";
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.maxDepthBounds = 1.0f;
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkGraphicsPipelineLibraryCreateInfoEXT li = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, &ri,
                                                 VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT};
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &li};
    gi.stageCount = 1;
    gi.pStages = &st;
    gi.pDepthStencilState = &ds;
    gi.pDynamicState = pipeline_dynamic_state();
    gi.layout = pl.layout;
    return create_part(gi, "fragment shader part");
}

VkPipeline build_vi(const GraphicsKey& key) {
    VkVertexInputBindingDescription vb[16];
    VkVertexInputAttributeDescription va[16];
    uint32_t nvb = 0, nva = 0;
    bool seen[16] = {};
    for (uint32_t i = 0; i < key.attr_count && i < 16; ++i) {
        const KeyAttr& a = key.attr[i];
        va[nva++] = {a.location, a.binding, static_cast<VkFormat>(a.format), a.offset};
        if (!seen[a.binding]) {
            seen[a.binding] = true;
            vb[nvb++] = {a.binding, 0, key.binding_instance[a.binding] ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
        }
    }
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = nvb;
    vi.pVertexBindingDescriptions = vb;
    vi.vertexAttributeDescriptionCount = nva;
    vi.pVertexAttributeDescriptions = va;
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = static_cast<VkPrimitiveTopology>(key.topology);
    VkGraphicsPipelineLibraryCreateInfoEXT li = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, nullptr,
                                                 VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT};
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &li};
    gi.pVertexInputState = &vi;
    gi.pInputAssemblyState = &ia;
    gi.pDynamicState = pipeline_dynamic_state();
    return create_part(gi, "vertex input part");
}

VkPipeline build_fo(const GraphicsKey& key) {
    VkPipelineColorBlendAttachmentState cba[8];
    VkFormat cf[8];
    for (uint32_t i = 0; i < key.color_count && i < 8; ++i) {
        cba[i] = {};
        cba[i].blendEnable = key.blend_enable[i] ? VK_TRUE : VK_FALSE;
        cba[i].srcColorBlendFactor = static_cast<VkBlendFactor>(key.blend_src_rgb[i]);
        cba[i].dstColorBlendFactor = static_cast<VkBlendFactor>(key.blend_dst_rgb[i]);
        cba[i].colorBlendOp = static_cast<VkBlendOp>(key.blend_op_rgb[i]);
        cba[i].srcAlphaBlendFactor = static_cast<VkBlendFactor>(key.blend_src_a[i]);
        cba[i].dstAlphaBlendFactor = static_cast<VkBlendFactor>(key.blend_dst_a[i]);
        cba[i].alphaBlendOp = static_cast<VkBlendOp>(key.blend_op_a[i]);
        cba[i].colorWriteMask = key.write_mask[i];
        cf[i] = static_cast<VkFormat>(key.color_format[i]);
    }
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.logicOpEnable = key.logic_op_enable ? VK_TRUE : VK_FALSE;
    cb.logicOp = static_cast<VkLogicOp>(key.logic_op);
    cb.attachmentCount = key.color_count;
    cb.pAttachments = cba;
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = static_cast<VkSampleCountFlagBits>(key.samples ? key.samples : 1);
    ms.alphaToCoverageEnable = key.alpha_to_coverage ? VK_TRUE : VK_FALSE;
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = key.color_count;
    ri.pColorAttachmentFormats = cf;
    ri.depthAttachmentFormat = static_cast<VkFormat>(key.depth_format);
    ri.stencilAttachmentFormat = static_cast<VkFormat>(key.stencil_format);
    VkGraphicsPipelineLibraryCreateInfoEXT li = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, &ri,
                                                 VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT};
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &li};
    gi.pColorBlendState = &cb;
    gi.pMultisampleState = &ms;
    gi.pDynamicState = pipeline_dynamic_state();
    return create_part(gi, "fragment output part");
}

// A draw that needs a part a background thread is still building waits for that thread, which runs at below-normal priority and in efficiency mode:
// on a busy PC it can starve for a long time, and the frame with it. The waiting draw lends the builder normal priority and full speed until it
// is done (priority inheritance).
#if defined(_WIN32)
struct SelfHandle {
    HANDLE h = nullptr;
    ~SelfHandle() { if (h) CloseHandle(h); }
    HANDLE get() {
        if (!h) h = OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION, FALSE, GetCurrentThreadId());
        return h;
    }
};
thread_local SelfHandle t_self;
void boost_thread(void* handle) {
    if (!handle) return;
    HANDLE h = static_cast<HANDLE>(handle);
    SetThreadPriority(h, THREAD_PRIORITY_ABOVE_NORMAL);
    THREAD_POWER_THROTTLING_STATE st = {};
    st.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    st.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    st.StateMask = 0;                                    // explicitly not throttled
    SetThreadInformation(h, ThreadPowerThrottling, &st, sizeof st);
}
void* self_handle() { return t_self.get(); }
#else
void boost_thread(void*) {}
void* self_handle() { return nullptr; }
#endif

// ---- the part cache -------------------------------------------------------------------------------------------------------------
// The part for `key`: from the cache, from a build in flight (waits for it) or built here. `render` = called by a draw (counted apart from
// the workers' builds). A part that failed to build is cached as null so it is not retried at every draw.
VkPipeline get_part(const std::string& key, bool render, const std::function<VkPipeline()>& build) {
    std::unique_lock<std::mutex> lk(g_mu);
    auto it = g_libs.find(key);
    if (it != g_libs.end()) {
        Entry& e = it->second;
        if (e.state != 2) {
            const auto t0 = Clock::now();
            if (render && e.owner && !e.boosted.exchange(true)) boost_thread(e.owner);
            {
                NXFT("pipe.libwait");
                g_cv.wait(lk, [&] { return e.state == 2 || g_stop; });
            }
            if (render) {
                ++g_stats.parts_waited;
                g_stats.wait_ms += ms_since(t0);
            }
            return e.state == 2 ? e.pipe : VK_NULL_HANDLE;
        }
        if (render) ++g_stats.parts_hit;
        return e.pipe;
    }
    Entry& e = g_libs[key];
    e.state = 1;
    if (!render) e.owner = self_handle();
    lk.unlock();
    const auto t0 = Clock::now();
    VkPipeline p;
    if (render) {
        NXFT("pipe.lib");
        p = build();
    } else {
        NXFT("lib.build");
        p = build();
    }
    const double ms = ms_since(t0);
    if (e.boosted.load()) set_compile_thread_priority();                 // a draw lent this thread its speed: back to the background settings
    if (p && g_discard.load(std::memory_order_relaxed)) {
        if (ms < 2.0) g_prep_fast.fetch_add(1, std::memory_order_relaxed);
        vkDestroyPipeline(vkdev(), p, nullptr);
        p = VK_NULL_HANDLE;
    }
    lk.lock();
    e.pipe = p;
    e.state = 2;
    if (render && ms > 8.0 && key[0] != 'V' && key[0] != 'O' && t_link_key) {      // g_mu is held here
        const std::array<const ShaderInfo*, 3> prog = {t_link_key->stage[0], t_link_key->stage[1], t_link_key->stage[2]};
        std::string act, pred;
        for (const SpvIntSlot& sl : key_int_slots(*t_link_key)) act += " " + std::to_string(sl.location) + (sl.is_signed ? "s" : "u");
        { GraphicsKey pk; pk.stage[0] = t_link_key->stage[0]; predict_int_attributes(pk);
          for (const SpvIntSlot& sl : key_int_slots(pk)) pred += " " + std::to_string(sl.location) + (sl.is_signed ? "s" : "u"); }
        NXLOG_W(kTag, "program was set %d, bound %d; queue %zu hot %zu cold, %d running; integer slots of the draw:%s, predicted:%s",
                static_cast<int>(g_seen.count(prog)), static_cast<int>(g_hot.count(prog)), g_hot_q.size(), g_cold_q.size(), g_running, act.c_str(), pred.c_str());
    }
    if (render && ms > 8.0 && key[0] != 'V' && key[0] != 'O' && t_link_key)
        NXLOG_W(kTag, "a %s part took %.1f ms at the draw (vs %s fs %s, polygon mode %d, depth clamp %d, neg one %d, %u attributes)",
                key[0] == 'P' ? "pre-raster" : "fragment", ms, t_link_key->stage[0] ? t_link_key->stage[0]->hash.c_str() : "-",
                t_link_key->stage[1] ? t_link_key->stage[1]->hash.c_str() : "-", t_link_key->polygon_mode, t_link_key->depth_clamp, t_link_key->depth_neg_one, t_link_key->attr_count);
    if (render) {
        ++g_stats.parts_at_draw;
        g_stats.draw_part_ms += ms;
    } else {
        ++g_stats.parts_ahead;
        g_stats.worker_ms += ms;
    }
    g_cv.notify_all();
    return p;
}

// Vertex shaders that read integer attributes (skinned meshes, instances) run as a patched variant, which depends on the vertex formats of the draw. The
// code of the shader says where the integers go in (spirv_predict_int_inputs); the sign comes from the last draw that fed integers (the engine's
// vertex data is unsigned unless a draw showed otherwise). The key gets those attributes, so resolve_stage_modules makes the variant a draw with
// them would use. A draw whose slots differ builds its own part.
std::atomic<bool> g_pred_signed{false};
std::mutex g_pred_mu;
std::unordered_map<const ShaderInfo*, std::vector<SpvIntSlot>> g_pred_cache;

void predict_int_attributes(GraphicsKey& k) {
    if (!k.stage[0]) return;
    std::vector<SpvIntSlot> slots;
    {
        std::lock_guard<std::mutex> lk(g_pred_mu);
        auto it = g_pred_cache.find(k.stage[0]);
        if (it == g_pred_cache.end()) it = g_pred_cache.emplace(k.stage[0], spirv_predict_int_inputs(k.stage[0]->spirv)).first;
        slots = it->second;
    }
    const bool sg = g_pred_signed.load(std::memory_order_relaxed);
    for (const SpvIntSlot& sl : slots)
        if (k.attr_count < 16) {
            KeyAttr& a = k.attr[k.attr_count++];
            a.location = static_cast<uint8_t>(sl.location);
            a.format = sg ? VK_FORMAT_R32_SINT : VK_FORMAT_R32_UINT;
        }
}

void run_job(const Job& j) {
    const ShaderInfo* st6[6] = {j.st[0], j.st[1], j.st[2], nullptr, nullptr, nullptr};
    const ProgLayout* pl = prog_layout(st6);
    if (!pl) {
        std::lock_guard<std::mutex> lk(g_mu);
        ++g_stats.jobs_no_layout;
        return;
    }
    GraphicsKey k;
    for (int i = 0; i < 3; ++i) k.stage[i] = j.st[i];
    if (!j.fragment) predict_int_attributes(k);
    StageModules sm;
    if (!resolve_stage_modules(k, &sm)) {
        std::lock_guard<std::mutex> lk(g_mu);
        ++g_stats.jobs_no_modules;
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_mu);
        ++g_stats.jobs_run;
    }
    if (j.fragment) {
        get_part(fs_key(sm, *pl), false, [&] { return build_fs(sm.m[1], *pl); });
    } else {
        get_part(pre_key(sm, *pl, j.r), false, [&] { return build_pre(sm.m[0], sm.m[2], *pl, j.r); });
    }
}

void worker_main() {
    log_set_thread_name("nvn-gpl");
    set_compile_thread_priority();
    for (;;) {
        Job j;
        {
            std::unique_lock<std::mutex> lk(g_mu);
            if (!ab_pause_on()) {
                g_cv.wait(lk, [] { return g_stop || !g_exact_q.empty() || !g_hot_q.empty() || !g_pre_q.empty() || (g_cold && !g_cold_q.empty()); });
                if (g_stop) return;
            } else {
                for (;;) {
                    if (g_stop) return;
                    if ((!g_exact_q.empty() || !g_hot_q.empty() || !g_pre_q.empty() || (g_cold && !g_cold_q.empty())) && !workers_paused()) break;
                    g_cv.wait_for(lk, std::chrono::milliseconds(20));
                }
            }
            if (!g_exact_q.empty() || (g_hot_q.empty() && !g_pre_q.empty())) {
                auto& q = !g_exact_q.empty() ? g_exact_q : g_pre_q;
                std::function<void()> fn = std::move(q.front());
                q.pop_front();
                ++g_running;
                publish_backlog_locked();
                lk.unlock();
                fn();
                lk.lock();
                --g_running;
                publish_backlog_locked();
                continue;
            }
            if (!g_hot_q.empty()) {
                j = g_hot_q.front();
                g_hot_q.pop_front();
            } else {                                 // the programs set last belong to the area being loaded now, which is drawn next
                j = g_cold_q.back();
                g_cold_q.pop_back();
            }
            ++g_running;
            publish_backlog_locked();
        }
        run_job(j);
        {
            std::lock_guard<std::mutex> lk(g_mu);
            --g_running;
            publish_backlog_locked();
        }
    }
}

void upgrade_main() {
    log_set_thread_name("nvn-gpl-up");
    set_compile_thread_priority();
    for (;;) {
        UpJob j;
        {
            std::unique_lock<std::mutex> lk(g_mu);
            g_up_cv.wait(lk, [] { return g_stop || !g_up_q.empty(); });
            if (g_stop) return;
            j = std::move(g_up_q.front());
            g_up_q.pop_front();
            ++g_up_running;
            publish_backlog_locked();
        }
        VkPipeline p = VK_NULL_HANDLE;
        const auto t0 = Clock::now();
        {
            NXFT("pipe.upgrade");
            if (g_up_mode == UpMode::Lto) {
                VkPipelineLibraryCreateInfoKHR lci = {VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR};
                lci.libraryCount = 4;
                lci.pLibraries = j.parts;
                VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &lci};
                gi.flags |= VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT;
                gi.layout = j.layout->layout;
                if (create_graphics_pipeline_tracked(vkdev(), pipeline_cache_handle(), &gi, &p) != VK_SUCCESS) p = VK_NULL_HANDLE;
                else note_pipeline_created(ms_since(t0), "optimised link");
            } else {
                p = build_graphics_pipeline(j.key, *j.layout);
            }
        }
        const double ms = ms_since(t0);
        j.up->pipe = p;
        j.up->state.store(1, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(g_mu);
            --g_up_running;
            if (p) { ++g_stats.upgrades_done; g_stats.upgrade_ms += ms; } else ++g_stats.upgrades_failed;
            publish_backlog_locked();
        }
    }
}

void init_once() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_inited) return;
    g_inited = true;
    g_stop = false;
    g_enabled = config_get_bool("nvn", "gpl", true) && device().features().graphics_pipeline_library;
    g_lto = config_get_bool("nvn", "gpl_lto", false);
    g_cold = config_get_bool("nvn", "gpl_cold", true);
    {
        const std::string m = config_get_str("nvn", "gpl_upgrade", "whole");
        g_up_mode = (m == "off" || m == "false" || m == "0") ? UpMode::Off : (m == "lto") ? UpMode::Lto : UpMode::Whole;
        g_retain = g_lto || g_up_mode == UpMode::Lto;
    }
    g_on.store(g_enabled, std::memory_order_release);
    if (!g_enabled) {
        NXLOG_I(kTag, "pipeline libraries off (%s)", device().features().graphics_pipeline_library ? "[nvn] gpl is false" : "the GPU or driver does not support them");
        return;
    }
    g_rasters.assign(1, RasterKey{});
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    // a quarter of the hardware threads, 1 to 6 (was a sixth, 1 to 4: 2 threads on an 8-core / 16-thread CPU); they run below normal priority
    const int64_t want = config_get_int("nvn", "gpl_threads", std::clamp<int64_t>(hw / 4, 1, 6));
    for (int64_t i = 0; i < want; ++i) g_workers.emplace_back(worker_main);
    if (g_up_mode != UpMode::Off) {
        const int64_t nup = config_get_int("nvn", "gpl_upgrade_threads", 1);
        for (int64_t i = 0; i < nup; ++i) g_up_workers.emplace_back(upgrade_main);
    }
    NXLOG_I(kTag, "pipeline libraries on: %zu background threads%s; linked pipelines are replaced by %s", g_workers.size(), g_lto ? ", link-time optimisation requested" : "",
            g_up_mode == UpMode::Off ? "nothing (gpl_upgrade is off)" : g_up_mode == UpMode::Lto ? "link-time optimised links" : "whole pipelines");
}

// A draw used raster flags the prebuilt parts were not made for: later programs get them ahead too, and so do the recent ones.
void learn_raster(const RasterKey& r) {
    std::lock_guard<std::mutex> lk(g_mu);
    for (const RasterKey& k : g_rasters)
        if (k == r) return;
    g_rasters.push_back(r);
    for (const auto& p : g_programs) {
        Job j;
        for (int i = 0; i < 3; ++i) j.st[i] = p[i];
        j.r = r;
        g_cold_q.push_back(j);
        ++g_stats.jobs;
    }
    publish_backlog_locked();
    g_cv.notify_all();
}

}  // namespace

#if defined(_WIN32)
namespace {
// A CPU with cores of more than one efficiency class (Intel P and E cores): where EcoQoS moves work to the slow cores. On the others it only
// lowers the clock of the thread, which makes the background builds slower without freeing a core for the game.
bool cpu_is_hybrid() {
    static const bool hybrid = [] {
        ULONG len = 0;
        GetSystemCpuSetInformation(nullptr, 0, &len, GetCurrentProcess(), 0);
        if (len == 0) return false;
        std::vector<uint8_t> buf(len);
        if (!GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data()), len, &len, GetCurrentProcess(), 0)) return false;
        int lo = 256, hi = -1;
        for (ULONG off = 0; off + sizeof(DWORD) * 2 <= len;) {
            const auto* e = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buf.data() + off);
            if (e->Size == 0) break;
            if (e->Type == CpuSetInformation) {
                lo = std::min<int>(lo, e->CpuSet.EfficiencyClass);
                hi = std::max<int>(hi, e->CpuSet.EfficiencyClass);
            }
            off += e->Size;
        }
        return hi > lo;
    }();
    return hybrid;
}
}  // namespace
#endif

void set_compile_thread_priority(bool background) {
#if defined(_WIN32)
    static const std::string mode = config_get_str("nvn", "compile_priority", "below");
    int p = THREAD_PRIORITY_BELOW_NORMAL;
    if (mode == "normal") p = THREAD_PRIORITY_NORMAL;
    else if (mode == "idle") p = THREAD_PRIORITY_IDLE;
    SetThreadPriority(GetCurrentThread(), p);
    // Efficiency mode (EcoQoS): Windows runs the thread on the efficiency cores of a hybrid CPU and at a lower clock elsewhere, so a burst of
    // shader compiles does not take the fast cores from the game's own threads. Not for the shader preparation, which wants all the speed.
    // [nvn] compile_eco: auto (default) = on only for a hybrid CPU; true / false force it.
    static const bool eco = [] {
        std::string v = config_get_str("nvn", "compile_eco", "auto");
        for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (v.empty() || v == "auto") {
            const bool h = cpu_is_hybrid();
            NXLOG_I(kTag, "[nvn] compile_eco = auto: background pipeline builds %s (%s CPU)", h ? "in efficiency mode" : "at full speed", h ? "hybrid" : "not a hybrid");
            return h;
        }
        return !(v == "false" || v == "0" || v == "off" || v == "no");
    }();
    if (background && eco) {
        THREAD_POWER_THROTTLING_STATE st = {};
        st.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        st.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        st.StateMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &st, sizeof st);
    }
#else
    (void)background;
#endif
}

bool gpl_active() {
    if (!g_on.load(std::memory_order_acquire)) {
        {
            std::lock_guard<std::mutex> lk(g_mu);
            if (g_inited) return false;
        }
        init_once();
    }
    return g_on.load(std::memory_order_acquire);
}

// Jobs of one program, shader parts first. `front` = urgent (a draw recorded with the program): ahead of everything queued speculatively.
static void queue_program(const std::array<const ShaderInfo*, 3>& prog, bool front) {
    std::vector<Job> jobs;
    Job f;
    for (int i = 0; i < 3; ++i) f.st[i] = prog[i];
    f.fragment = true;
    jobs.push_back(f);
    for (const RasterKey& r : g_rasters) {
        Job p = f;
        p.fragment = false;
        p.r = r;
        jobs.push_back(p);
    }
    g_stats.jobs += jobs.size();
    auto& q = front ? g_hot_q : g_cold_q;
    q.insert(q.end(), jobs.begin(), jobs.end());
    publish_backlog_locked();
    g_cv.notify_all();
}

void gpl_program(const ShaderInfo* const stages[6], bool covered) {
    if (!stages || stages[3] || stages[4] || !gpl_active()) return;   // tessellated programs are built whole
    std::array<const ShaderInfo*, 3> prog = {stages[0], stages[1], stages[2]};
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_stop) return;
    ++g_stats.prog_set;
    // A program of one stage (Pokopia sets its vertex and fragment programs apart) says nothing about the pair a draw will use; the parts of
    // the pairs the key file knows are built from boot (gpl_prebuild), new pairs when a draw first needs them.
    if (!stages[0] || !stages[1]) {
        ++g_stats.prog_single;
        return;
    }
    if (covered) {                      // the key file has pipelines for it: the prewarm builds those whole
        ++g_stats.prog_covered;
        g_covered.insert(prog);
        return;
    }
    if (!g_seen.insert(prog).second) return;
    ++g_stats.prog_new;
    g_programs.push_back(prog);
    if (g_programs.size() > 512) g_programs.pop_front();
    queue_program(prog, false);
}

// A command buffer was recorded with this program bound: a draw with it is at most a few frames away. Its shader parts go to the front of
// the queue (once per program), whether or not they were queued when the game set the shaders.
void gpl_program_bound(const ShaderInfo* const stages[6]) {
    if (!stages || !stages[0] || !stages[1] || stages[3] || stages[4] || !gpl_active()) return;
    std::array<const ShaderInfo*, 3> prog = {stages[0], stages[1], stages[2]};
    std::lock_guard<std::mutex> lk(g_mu);
    ++g_stats.prog_bound;
    if (g_stop || g_covered.count(prog) || !g_hot.insert(prog).second) return;
    ++g_stats.prog_bound_new;
    if (g_seen.insert(prog).second) {
        g_programs.push_back(prog);
        if (g_programs.size() > 512) g_programs.pop_front();
    }
    queue_program(prog, true);
}

void gpl_note_presents(uint64_t presents) {
    static uint64_t last = 0;
    if (presents == last) return;
    last = presents;
    g_frame_seen_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(), std::memory_order_relaxed);
}

namespace {
bool frames_flowing() {
    const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
    const int64_t seen = g_frame_seen_ns.load(std::memory_order_relaxed);
    return seen != 0 && now - seen < 100000000;
}
bool part_done_locked(const std::string& key) {
    auto it = g_libs.find(key);
    return it != g_libs.end() && it->second.state == 2;
}
// g_mu held. The part `key` goes first in line for the background threads, unless it is built, being built or already queued.
void queue_exact_locked(const std::string& key, std::function<VkPipeline()> build) {
    if (g_libs.count(key) || !g_exact_pending.insert(key).second) return;
    ++g_stats.parts_async;
    g_exact_q.push_back([key, build = std::move(build)] {
        get_part(key, false, build);
        std::lock_guard<std::mutex> lk(g_mu);
        g_exact_pending.erase(key);
    });
    publish_backlog_locked();
    g_cv.notify_all();
}
}  // namespace

VkPipeline gpl_link(const GraphicsKey& key, const ProgLayout& layout, std::shared_ptr<PipeUpgrade>* up, bool* pending) {
    if (pending) *pending = false;
    if (!gpl_active()) return VK_NULL_HANDLE;
    // stream-out and tessellated draws, and anything without both a vertex and a fragment shader, are built whole
    if (!key.stage[0] || !key.stage[1] || key.xfb || key.patch_control_points || key.tess[0] || key.tess[1]) return VK_NULL_HANDLE;
    NXFT("pipe.link");
    const auto t0 = Clock::now();
    StageModules sm;
    if (!resolve_stage_modules(key, &sm)) return VK_NULL_HANDLE;
    RasterKey r;
    r.poly = key.polygon_mode;
    r.clamp = key.depth_clamp;
    r.neg = key.depth_neg_one;
    learn_raster(r);
    if (const std::vector<SpvIntSlot> ints = key_int_slots(key); !ints.empty()) g_pred_signed.store(ints[0].is_signed, std::memory_order_relaxed);
    static const bool async = config_get_bool("nvn", "gpl_async", true);
    if (async && pending && frames_flowing()) {
        const std::string pk = pre_key(sm, layout, r), fk = fs_key(sm, layout);
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_stop && !(part_done_locked(pk) && part_done_locked(fk))) {
            const ProgLayout* pl = &layout;               // layouts live as long as the back end
            const StageModules m = sm;
            if (!part_done_locked(pk)) queue_exact_locked(pk, [m, pl, r] { return build_pre(m.m[0], m.m[2], *pl, r); });
            if (!part_done_locked(fk)) queue_exact_locked(fk, [m, pl] { return build_fs(m.m[1], *pl); });
            ++g_stats.draws_deferred;
            *pending = true;
            return VK_NULL_HANDLE;
        }
    }
    t_link_key = &key;
    VkPipeline parts[4];
    parts[0] = get_part(vi_key(key), true, [&] { return build_vi(key); });
    parts[1] = get_part(pre_key(sm, layout, r), true, [&] { return build_pre(sm.m[0], sm.m[2], layout, r); });
    parts[2] = get_part(fs_key(sm, layout), true, [&] { return build_fs(sm.m[1], layout); });
    parts[3] = get_part(fo_key(key), true, [&] { return build_fo(key); });
    t_link_key = nullptr;
    bool ok = true;
    for (VkPipeline p : parts) ok = ok && p != VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    if (ok) {
        VkPipelineLibraryCreateInfoKHR lci = {VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR};
        lci.libraryCount = 4;
        lci.pLibraries = parts;
        VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &lci};
        if (g_lto) gi.flags |= VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT;
        gi.layout = layout.layout;
        const VkResult res = create_graphics_pipeline_tracked(vkdev(), pipeline_cache_handle(), &gi, &pipe);
        if (res != VK_SUCCESS) {
            pipe = VK_NULL_HANDLE;
            ok = false;
            NXLOG_W(kTag, "linking the parts failed (%d): vs %s fs %s", static_cast<int>(res), key.stage[0]->hash.c_str(), key.stage[1]->hash.c_str());
        }
    }
    const double ms = ms_since(t0);
    std::lock_guard<std::mutex> lk(g_mu);
    if (!ok) {
        ++g_stats.link_failed;
        if (++g_fail_streak >= 16 && g_stats.linked == 0) {          // never worked on this driver: stop trying
            NXLOG_W(kTag, "pipeline libraries never linked on this driver: switched off");
            g_on.store(false, std::memory_order_release);
        }
        return VK_NULL_HANDLE;
    }
    g_fail_streak = 0;
    ++g_stats.linked;
    g_stats.link_ms += ms;
    note_pipeline_created(ms, "linked pipeline");
    if (up && g_up_mode != UpMode::Off && !g_stop) {
        UpJob j;
        j.key = key;
        j.layout = &layout;
        for (int i = 0; i < 4; ++i) j.parts[i] = parts[i];
        j.up = std::make_shared<PipeUpgrade>();
        *up = j.up;
        g_up_q.push_back(std::move(j));
        ++g_stats.upgrades_queued;
        publish_backlog_locked();
        g_up_cv.notify_one();
    }
    return pipe;
}

namespace {
bool link_eligible(const GraphicsKey& key) {
    return key.stage[0] && key.stage[1] && !key.xfb && !key.patch_control_points && !key.tess[0] && !key.tess[1];
}
RasterKey raster_of(const GraphicsKey& key) {
    RasterKey r;
    r.poly = key.polygon_mode;
    r.clamp = key.depth_clamp;
    r.neg = key.depth_neg_one;
    return r;
}
// Background thread: the four parts of a key, each unless it is built, being built or queued for a draw already.
void prebuild_parts(const GraphicsKey& key, const ProgLayout& pl) {
    if (!link_eligible(key)) return;
    StageModules sm;
    if (!resolve_stage_modules(key, &sm) || !sm.m[0] || !sm.m[1]) {
        std::lock_guard<std::mutex> lk(g_mu);
        ++g_stats.jobs_no_modules;
        return;
    }
    const RasterKey r = raster_of(key);
    const std::string keys[4] = {pre_key(sm, pl, r), fs_key(sm, pl), vi_key(key), fo_key(key)};
    const std::function<VkPipeline()> builds[4] = {
        [&] { return build_pre(sm.m[0], sm.m[2], pl, r); }, [&] { return build_fs(sm.m[1], pl); },
        [&] { return build_vi(key); }, [&] { return build_fo(key); }};
    for (int i = 0; i < 4; ++i) {
        {
            std::lock_guard<std::mutex> lk(g_mu);
            if (g_stop) return;
            if (g_libs.count(keys[i]) || g_exact_pending.count(keys[i])) continue;
        }
        get_part(keys[i], false, builds[i]);
    }
    std::lock_guard<std::mutex> lk(g_mu);
    bool known = false;                       // the raster flags join the set later programs get ahead
    for (const RasterKey& k : g_rasters) known = known || k == r;
    if (!known) g_rasters.push_back(r);
}
}  // namespace

void gpl_prebuild(std::function<bool(GraphicsKey&, const ProgLayout*&)> resolve, bool front) {
    if (!gpl_active()) return;
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_stop) return;
    if (!front) ++g_stats.prebuild_jobs;
    auto fn = [resolve = std::move(resolve)] {
        GraphicsKey k;
        const ProgLayout* pl = nullptr;
        if (!resolve(k, pl) || !pl) return;
        {
            std::lock_guard<std::mutex> lk2(g_mu);
            ++g_stats.prebuild_run;
        }
        prebuild_parts(k, *pl);
    };
    if (front) g_pre_q.push_front(std::move(fn));
    else g_pre_q.push_back(std::move(fn));
    publish_backlog_locked();
    g_cv.notify_one();
}

bool gpl_ready(const GraphicsKey& key, const ProgLayout& layout) {
    if (!g_on.load(std::memory_order_acquire) || !link_eligible(key)) return false;
    StageModules sm;
    if (!resolve_stage_modules(key, &sm)) return false;
    const std::string pk = pre_key(sm, layout, raster_of(key)), fk = fs_key(sm, layout);
    std::lock_guard<std::mutex> lk(g_mu);
    return part_done_locked(pk) && part_done_locked(fk);
}

bool gpl_ab_whole() {
    static const int64_t frames = config_get_int("nvn", "gpl_ab", 0);
    if (frames <= 0) return false;
    static std::atomic<int64_t> last{-1};
    const int64_t w = static_cast<int64_t>(counters().presents.load(std::memory_order_relaxed)) / frames;
    if (last.exchange(w) != w) NXLOG_I(kTag, "ab: present %lld starts a window of %s pipelines", static_cast<long long>(w * frames), (w & 1) ? "optimised" : "linked");
    return (w & 1) != 0;
}

void gpl_take_upgrade(VkPipeline& current, VkPipeline& alt, std::shared_ptr<PipeUpgrade>& up) {
    if (up->state.load(std::memory_order_acquire) != 1) return;
    VkPipeline fresh = up->pipe;
    if (fresh) {
        up->pipe = VK_NULL_HANDLE;
        if (config_get_int("nvn", "gpl_ab", 0) > 0) {
            alt = fresh;                           // measurement: keep both and let gpl_ab_whole pick
        } else {
            const VkPipeline old = current;
            current = fresh;
            defer_destroy([old] { vkDestroyPipeline(vkdev(), old, nullptr); });      // the frame being recorded may have bound it
        }
        std::lock_guard<std::mutex> lk(g_mu);
        ++g_stats.upgrades_taken;
    }
    up->state.store(2, std::memory_order_release);
    up.reset();
}

void gpl_progress_log() {
    GplStats s;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_inited || !g_enabled) return;
        s = g_stats;
        const uint64_t sig = s.linked * 3 + s.parts_ahead * 5 + s.parts_at_draw * 7 + s.parts_waited * 11 + s.link_failed * 13 + s.upgrades_done * 17 + s.upgrades_taken * 19;
        if (sig == g_logged_sig) return;
        g_logged_sig = sig;
    }
    NXLOG_I(kTag, "libraries so far: %llu pipelines linked (%.1f ms in all), parts built ahead %llu (%.0f ms on workers), at the draw %llu (%.1f ms), "
                  "waited for %llu (%.1f ms), ready %llu, %llu link failures; upgrades queued %llu, built %llu (%.0f ms), taken %llu, failed %llu",
            static_cast<unsigned long long>(s.linked), s.link_ms, static_cast<unsigned long long>(s.parts_ahead), s.worker_ms, static_cast<unsigned long long>(s.parts_at_draw),
            s.draw_part_ms, static_cast<unsigned long long>(s.parts_waited), s.wait_ms, static_cast<unsigned long long>(s.parts_hit), static_cast<unsigned long long>(s.link_failed),
            static_cast<unsigned long long>(s.upgrades_queued), static_cast<unsigned long long>(s.upgrades_done), s.upgrade_ms, static_cast<unsigned long long>(s.upgrades_taken),
            static_cast<unsigned long long>(s.upgrades_failed));
    NXLOG_I(kTag, "libraries ahead: programs set %llu (covered by the key file %llu, new %llu), bound %llu (new %llu); jobs queued %llu, run %llu, "
                  "no layout %llu, no modules %llu; draws deferred %llu for %llu parts; single-stage programs %llu; key-file keys queued %llu, run %llu",
            static_cast<unsigned long long>(s.prog_set), static_cast<unsigned long long>(s.prog_covered), static_cast<unsigned long long>(s.prog_new),
            static_cast<unsigned long long>(s.prog_bound), static_cast<unsigned long long>(s.prog_bound_new), static_cast<unsigned long long>(s.jobs),
            static_cast<unsigned long long>(s.jobs_run), static_cast<unsigned long long>(s.jobs_no_layout), static_cast<unsigned long long>(s.jobs_no_modules),
            static_cast<unsigned long long>(s.draws_deferred), static_cast<unsigned long long>(s.parts_async), static_cast<unsigned long long>(s.prog_single),
            static_cast<unsigned long long>(s.prebuild_jobs), static_cast<unsigned long long>(s.prebuild_run));
}

void program_bound(const ShaderInfo* const stages[6]) { gpl_program_bound(stages); }

GplStats gpl_stats() {
    std::lock_guard<std::mutex> lk(g_mu);
    GplStats s = g_stats;
    s.jobs_pending = g_exact_q.size() + g_hot_q.size() + g_pre_q.size() + (g_cold ? g_cold_q.size() : 0) + static_cast<uint64_t>(g_running);
    return s;
}

void gpl_stop() {
    std::vector<std::thread> workers, ups;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_inited) return;
        g_stop = true;
        workers.swap(g_workers);
        ups.swap(g_up_workers);
        g_hot_q.clear();
        g_cold_q.clear();
        g_pre_q.clear();
        g_exact_q.clear();
        g_exact_pending.clear();
        g_up_q.clear();
        publish_backlog_locked();
    }
    g_cv.notify_all();
    g_up_cv.notify_all();
    for (auto& t : workers) t.join();
    for (auto& t : ups) t.join();
}

// ---- shader preparation ----------------------------------------------------------------------------------------------------------
namespace {
struct ProgHashes { uint64_t vs, fs, gs; };

bool read_programs_file(const std::string& path, std::vector<ProgHashes>* out, std::string* error) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { *error = "cannot open " + path; return false; }
    char head[16];
    bool ok = std::fread(head, 1, 16, f) == 16 && std::memcmp(head, "MPRPRG01", 8) == 0;
    uint32_t count = 0;
    if (ok) std::memcpy(&count, head + 8, 4);
    if (ok) {
        out->resize(count);
        ok = count == 0 || std::fread(out->data(), sizeof(ProgHashes), count, f) == count;
    }
    std::fclose(f);
    if (!ok) *error = path + " is not a program list";
    return ok;
}

std::string hex16(uint64_t h) {
    char b[20];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(h));
    return b;
}
}  // namespace

// The state file next to the program list: which GPU and driver the preparation ran on and whether it finished. The key changes with the
// GPU, the driver version, the pipeline cache UUID (a driver update changes it) and the version of the part scheme in this file.
constexpr int kPrepareScheme = 1;
std::string prepare_key(uint64_t programs) {
    const VkPhysicalDeviceProperties& p = device().properties();
    char uuid[40];
    for (int i = 0; i < 16; ++i) std::snprintf(uuid + 2 * i, 3, "%02x", p.pipelineCacheUUID[i]);
    char b[512];
    std::snprintf(b, sizeof b, "scheme=%d programs=%llu gpu=%s driver=%u api=%u cache=%s salt=%s", kPrepareScheme, static_cast<unsigned long long>(programs),
                  p.deviceName, p.driverVersion, p.apiVersion, uuid, config_get_str("nvn", "spirv_salt", "").c_str());
    return b;
}

bool prepare_state_ready(const std::string& path, const std::string& key) {
    std::ifstream f(path);
    if (!f) return false;
    std::string l;
    bool key_ok = false, done = false;
    while (std::getline(f, l)) {
        while (!l.empty() && l.back() == '\r') l.pop_back();
        if (l == "key: " + key) key_ok = true;
        if (l == "complete: 1") done = true;
    }
    return key_ok && done;
}

void write_prepare_state(const std::string& path, const std::string& key, const PrepareResult& r) {
    std::ofstream f(path, std::ios::trunc);
    if (!f) return;
    const char nl = 10;
    f << "key: " << key << nl << "programs: " << r.programs << nl << "prepared: " << r.prepared << nl << "missing: " << r.missing_shaders << nl
      << "failed: " << r.failed << nl << "seconds: " << r.seconds << nl << "complete: " << (r.cancelled ? 0 : 1) << nl;
}

bool gpl_prepare(const PrepareOptions& o, PrepareResult* res) {
    PrepareResult local;
    PrepareResult& r = res ? *res : local;
    r = PrepareResult{};
    if (!device().features().graphics_pipeline_library) { r.error = "the GPU or driver does not support pipeline libraries"; return false; }
    if (!shader_cache().is_open()) { r.error = "no shader cache is open"; return false; }
    std::string file = o.programs_file;
    if (file.empty()) file = (std::filesystem::path(shader_cache().dir()).parent_path() / "programs.bin").string();
    std::vector<ProgHashes> progs;
    if (!read_programs_file(file, &progs, &r.error)) return false;
    r.programs = progs.size();
    const std::string state_path = (std::filesystem::path(file).parent_path() / "prepare.state").string();
    const std::string key = prepare_key(r.programs);
    if (o.only_if_needed && !o.force && prepare_state_ready(state_path, key)) {
        r.already_ready = true;
        return true;
    }
    // fragment shader first: programs that share one are made together, so the part of a shared fragment shader is made once per batch
    std::sort(progs.begin(), progs.end(), [](const ProgHashes& a, const ProgHashes& b) {
        return a.fs != b.fs ? a.fs < b.fs : a.vs != b.vs ? a.vs < b.vs : a.gs < b.gs;
    });
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int threads = o.threads > 0 ? o.threads : static_cast<int>(hw > 3 ? hw - 2 : 1);
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_rasters.assign(1, RasterKey{});
        g_stop = false;
    }
    g_prep_fast.store(0);
    g_prepare_cache.store(config_get_bool("nvn", "prepare_cache", true));
    g_discard.store(true);
    const auto t0 = Clock::now();
    constexpr size_t kBatch = 192;                       // programs per batch; the shader modules and SPIR-V of a batch are released after it
    std::atomic<uint64_t> done{0}, prepared{0}, missing{0}, failed{0}, parts{0};
    std::atomic<bool> cancel{false};
    std::mutex progress_mu;
    auto report = [&] {
        if (!o.progress) return;
        std::unique_lock<std::mutex> lk(progress_mu, std::try_to_lock);
        if (!lk.owns_lock()) return;
        if (!o.progress(done.load(), r.programs)) cancel.store(true);
    };
    for (size_t base = 0; base < progs.size() && !cancel.load(); base += kBatch) {
        const size_t end = std::min(progs.size(), base + kBatch);
        std::atomic<size_t> next{base};
        std::vector<const ShaderInfo*> used;
        std::mutex used_mu;
        auto work = [&] {
            set_compile_thread_priority(false);
            std::vector<const ShaderInfo*> mine;
            for (;;) {
                if (cancel.load()) break;
                const size_t i = next.fetch_add(1);
                if (i >= end) break;
                const ProgHashes& ph = progs[i];
                const ShaderInfo* vs = shader_cache().load_stub(hex16(ph.vs));
                const ShaderInfo* fs = shader_cache().load_stub(hex16(ph.fs));
                const ShaderInfo* gs = ph.gs ? shader_cache().load_stub(hex16(ph.gs)) : nullptr;
                if (!vs || !fs || (ph.gs && !gs)) {
                    missing.fetch_add(1);
                } else {
                    mine.push_back(vs); mine.push_back(fs);
                    if (gs) mine.push_back(gs);
                    const ShaderInfo* st6[6] = {vs, fs, gs, nullptr, nullptr, nullptr};
                    const ProgLayout* pl = prog_layout(st6);
                    GraphicsKey k;
                    k.stage[0] = vs; k.stage[1] = fs; k.stage[2] = gs;
                    predict_int_attributes(k);
                    StageModules sm;
                    bool ok = pl && resolve_stage_modules(k, &sm) && sm.m[0] && sm.m[1];
                    if (ok) {
                        const auto tp = Clock::now();
                        std::vector<RasterKey> rs;
                        { std::lock_guard<std::mutex> lk(g_mu); rs = g_rasters; }
                        for (const RasterKey& rk : rs) {
                            get_part(pre_key(sm, *pl, rk), false, [&] { return build_pre(sm.m[0], sm.m[2], *pl, rk); });
                            parts.fetch_add(1);
                        }
                        get_part(fs_key(sm, *pl), false, [&] { return build_fs(sm.m[1], *pl); });
                        parts.fetch_add(1);
                        (void)tp;
                    }
                    if (ok) prepared.fetch_add(1); else failed.fetch_add(1);
                }
                done.fetch_add(1);
                report();
            }
            std::lock_guard<std::mutex> lk(used_mu);
            used.insert(used.end(), mine.begin(), mine.end());
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < threads; ++t) pool.emplace_back(work);
        work();
        for (auto& t : pool) t.join();
        // between batches nothing is creating pipelines: drop the batch's modules, parts and SPIR-V
        module_cache_clear();
        {
            std::lock_guard<std::mutex> lk(g_mu);
            g_libs.clear();
        }
        for (const ShaderInfo* si : used) shader_cache().release_stub_spirv(si);
    }
    g_discard.store(false);
    if (g_prepare_cache.load() && !cancel.load()) pipeline_cache_save();
    if (o.progress) o.progress(done.load(), r.programs);
    r.fast_parts = g_prep_fast.load();
    r.prepared = prepared.load(); r.missing_shaders = missing.load(); r.failed = failed.load(); r.parts = parts.load();
    r.cancelled = cancel.load();
    r.seconds = ms_since(t0) / 1000.0;
    if (!r.cancelled && r.failed == 0) write_prepare_state(state_path, key, r);
    else if (r.cancelled) write_prepare_state(state_path, key, r);
    return true;
}

void gpl_shutdown() {
    gpl_stop();
    GplStats s;
    size_t n_rasters = 0, n_programs_seen = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_inited) return;
        s = g_stats;
        n_rasters = g_rasters.size();
        n_programs_seen = g_seen.size();
        for (auto& [k, e] : g_libs)
            if (e.pipe) vkDestroyPipeline(vkdev(), e.pipe, nullptr);
        g_libs.clear();
        g_programs.clear();
        g_seen.clear();
        g_hot.clear();
        g_covered.clear();
        g_rasters.clear();
        g_inited = false;
        g_enabled = false;
        g_on.store(false, std::memory_order_release);
        g_stats = GplStats{};
        g_logged_sig = 0;
        g_fail_streak = 0;
    }
    if (s.linked || s.parts_ahead || s.parts_at_draw)
        NXLOG_I(kTag, "pipeline libraries: %llu pipelines linked (%.1f ms in all), parts built ahead %llu (%.0f ms on workers), at the draw %llu (%.1f ms), waited for %llu (%.1f ms), ready %llu, %llu link failures; upgrades queued %llu, built %llu (%.0f ms), taken %llu, failed %llu; %zu programs set, %zu raster variants",
                static_cast<unsigned long long>(s.linked), s.link_ms, static_cast<unsigned long long>(s.parts_ahead), s.worker_ms, static_cast<unsigned long long>(s.parts_at_draw),
                s.draw_part_ms, static_cast<unsigned long long>(s.parts_waited), s.wait_ms, static_cast<unsigned long long>(s.parts_hit), static_cast<unsigned long long>(s.link_failed),
                static_cast<unsigned long long>(s.upgrades_queued), static_cast<unsigned long long>(s.upgrades_done), s.upgrade_ms, static_cast<unsigned long long>(s.upgrades_taken),
                static_cast<unsigned long long>(s.upgrades_failed), n_programs_seen, n_rasters);
}

// Gauges (nxrt/gauge.h). Each callback takes the lock that guards what it counts; the callbacks are removed before those containers are destroyed.
namespace {
struct GplGauges {
    GplGauges() {
        nxrt::gauge_register("nvn.gpl.parts", [] {
            std::lock_guard<std::mutex> lk(g_mu);
            return static_cast<int64_t>(g_libs.size());
        });
        nxrt::gauge_register("nvn.gpl.hot_queue", [] {
            std::lock_guard<std::mutex> lk(g_mu);
            return static_cast<int64_t>(g_hot_q.size());
        });
        nxrt::gauge_register("nvn.gpl.cold_queue", [] {
            std::lock_guard<std::mutex> lk(g_mu);
            return static_cast<int64_t>(g_cold_q.size());
        });
        nxrt::gauge_register("nvn.gpl.upgrade_queue", [] {
            std::lock_guard<std::mutex> lk(g_mu);
            return static_cast<int64_t>(g_up_q.size());
        });
        nxrt::gauge_register("nvn.gpl.programs_seen", [] {
            std::lock_guard<std::mutex> lk(g_mu);
            return static_cast<int64_t>(g_seen.size());
        });
        nxrt::gauge_register("nvn.gpl.hot_programs", [] {
            std::lock_guard<std::mutex> lk(g_mu);
            return static_cast<int64_t>(g_hot.size());
        });
        nxrt::gauge_register("nvn.gpl.covered_programs", [] {
            std::lock_guard<std::mutex> lk(g_mu);
            return static_cast<int64_t>(g_covered.size());
        });
        nxrt::gauge_register("nvn.gpl.recent_programs", [] {
            std::lock_guard<std::mutex> lk(g_mu);
            return static_cast<int64_t>(g_programs.size());
        });
        nxrt::gauge_register("nvn.gpl.rasters", [] {
            std::lock_guard<std::mutex> lk(g_mu);
            return static_cast<int64_t>(g_rasters.size());
        });
        nxrt::gauge_register("nvn.gpl.pred_cache", [] {
            std::lock_guard<std::mutex> lk(g_pred_mu);
            return static_cast<int64_t>(g_pred_cache.size());
        });
    }
    ~GplGauges() {
        for (const char* n : {"nvn.gpl.parts", "nvn.gpl.hot_queue", "nvn.gpl.cold_queue", "nvn.gpl.upgrade_queue", "nvn.gpl.programs_seen",
                              "nvn.gpl.hot_programs", "nvn.gpl.covered_programs", "nvn.gpl.recent_programs", "nvn.gpl.rasters", "nvn.gpl.pred_cache"})
            nxrt::gauge_unregister(n);
    }
};
GplGauges g_gpl_gauges;
}  // namespace

}  // namespace nxrt::nvn::be::gpu
