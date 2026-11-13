// SPDX-License-Identifier: GPL-3.0-or-later
// Pipeline layouts, shader modules and pipelines (see nvn_pipeline.h). Pipeline creation is synchronous on the render thread; a
// VkPipelineCache persists between runs ([nvn] pipeline_cache, default "pipeline_cache.bin").
#include "nvn_pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <tuple>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "nxrt/config.h"
#include "nxrt/ftrace.h"
#include "nxrt/gauge.h"
#include "nxrt/log.h"
#include "nvn_enums.h"
#include "nvn_format.h"
#include "nvn_gpu_internal.h"
#include "nvn_spirv_patch.h"
#include "nvn_subgroup.h"
#include "vk_gauges.h"
#include "extras/hdr_nvn.h"
#include "nxrt/extras.h"

namespace nxrt::nvn::be::gpu {
namespace {

constexpr const char* kTag = "nvn.pipe";

struct LayoutKey {
    const ShaderInfo* s[6];
    bool operator<(const LayoutKey& o) const { return std::memcmp(s, o.s, sizeof s) < 0; }
};

// Shader modules and layouts are also made by the prewarm workers (nvn_prewarm.cpp), so they have locks; the pipeline buckets below stay
// render thread only.
std::mutex g_layout_mu;
std::recursive_mutex g_module_mu;   // every module cache below; the variant makers call shader_module
std::map<LayoutKey, ProgLayout> g_layouts;
// Descriptor set layouts and the pipeline layout built from them, shared by every program whose descriptors are the same (S24): two programs with
// the same signature get the same VkPipelineLayout handle, so the shader parts compiled for one are found again for the other.
struct SharedLayout { VkDescriptorSetLayout sets[5] = {}; VkPipelineLayout layout = VK_NULL_HANDLE; };
std::map<std::string, SharedLayout> g_shared_layouts;
std::unordered_map<const ShaderInfo*, VkShaderModule> g_modules;
std::mutex g_module_id_mu;
std::unordered_map<VkShaderModule, SubgroupShader> g_module_subgroups;
std::unordered_map<VkShaderModule, uint64_t> g_module_ids;   // every module create_module made: the hash of its words (module_content_id)
std::atomic<uint64_t> g_built_at_draw{0};                    // whole pipelines built by the render thread at a draw
std::atomic<bool> g_cache_dirty{false};
std::atomic<uint32_t> g_new_since_save{0};      // pipelines and parts created since the cache file was last written
bool g_cache_big = false;                       // the file loaded at start was large (a shader preparation made it): rewritten rarely, see pipeline_cache_dirty
// Producer modules with extra outputs (nvn_spirv_patch.h): (shader, slots as location * 4 + scalar type).
std::map<std::pair<const ShaderInfo*, std::vector<uint32_t>>, VkShaderModule> g_variants;
std::unordered_map<const ShaderInfo*, VkShaderModule> g_dual_src;
std::unordered_map<const ShaderInfo*, std::pair<VkShaderModule, bool>> g_no_out;   // (module, owned by this map)      // fragment shaders with private outputs (passes without colour attachments)
std::unordered_map<const ShaderInfo*, std::pair<VkShaderModule, uint32_t>> g_xfb_mod;   // vertex shaders with the captured outputs, their stride       // fragment shaders with the second blend source at Location 0, Index 1
// One pipeline per key. `up`: a linked pipeline (nvn_gpl.cpp) that a background thread is replacing with an optimised one; the lookup swaps it in.
struct GfxEntry {
    GraphicsKey first;
    VkPipeline second = VK_NULL_HANDLE;
    VkPipeline alt = VK_NULL_HANDLE;                 // measurement only ([nvn] gpl_ab): the optimised pipeline, kept next to the linked one
    std::shared_ptr<PipeUpgrade> up;
};
std::unordered_map<uint64_t, std::vector<GfxEntry>> g_gfx;
// Compute pipelines by (shader, layout, integer samplers: set << 17 | binding << 1 | signed, see spirv_int_samplers; empty for the translated module).
std::map<std::tuple<const ShaderInfo*, VkPipelineLayout, std::vector<uint32_t>>, VkPipeline> g_compute;
std::map<std::pair<const ShaderInfo*, std::vector<uint32_t>>, VkShaderModule> g_int_samplers;   // their modules (null: could not be made)
VkPipelineCache g_cache = VK_NULL_HANDLE;
bool g_cache_tried = false;
std::string g_cache_path;
nxrt::GaugeCounter g_pipeline_count{"nvn.pipelines"};   // live VkPipelines (graphics and compute, linked ones included); render thread
std::unordered_map<const ShaderInfo*, std::pair<gfx::Buffer, VkDeviceSize>> g_const_bufs;
gfx::Buffer g_driver_const;
bool g_driver_const_made = false;
// Counters mirrored at the sites that change the maps above (they are render thread only, so the gauge reads these instead).
nxrt::GaugeCounter g_pipes_created{"nvn.pipelines_created"};     // ever; same events as counters().pipelines_created (shader parts included)
nxrt::GaugeCounter g_gfx_keys{"nvn.pipeline_keys"};              // entries in g_gfx: one per graphics key tried (a failed build keeps its entry)
nxrt::GaugeCounter g_compute_keys{"nvn.compute_pipeline_keys"};  // entries in g_compute
nxrt::GaugeCounter g_const_n{"nvn.const_buffers"};               // entries in g_const_bufs
nxrt::GaugeCounter g_const_bytes{"nvn.const_buffers.bytes"};     // bytes of the constant buffers in g_const_bufs

VkShaderStageFlagBits stage_bit(ShaderStage s) {
    switch (s) {
        case ShaderStage::Vertex: return VK_SHADER_STAGE_VERTEX_BIT;
        case ShaderStage::Fragment: return VK_SHADER_STAGE_FRAGMENT_BIT;
        case ShaderStage::Geometry: return VK_SHADER_STAGE_GEOMETRY_BIT;
        case ShaderStage::TessControl: return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        case ShaderStage::TessEval: return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        case ShaderStage::Compute: return VK_SHADER_STAGE_COMPUTE_BIT;
        default: return VK_SHADER_STAGE_VERTEX_BIT;
    }
}

std::mutex g_cache_mu;
std::atomic<bool> g_cache_ready{false};

void ensure_cache() {
    if (g_cache_ready.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lk(g_cache_mu);       // the prewarm workers and the render thread may get here together
    if (g_cache_tried) return;
    g_cache_tried = true;
    g_cache_path = config_get_str("nvn", "pipeline_cache", "pipeline_cache.bin");
    std::vector<char> data;
    if (!g_cache_path.empty()) {
        if (FILE* f = std::fopen(g_cache_path.c_str(), "rb")) {          // one read: the file can be hundreds of MB after a shader preparation
            std::fseek(f, 0, SEEK_END);
            const long len = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            if (len > 0) {
                data.resize(static_cast<size_t>(len));
                if (std::fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
            }
            std::fclose(f);
        }
    }
    g_cache_big = data.size() >= (64u << 20);
    VkPipelineCacheCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = data.size();
    ci.pInitialData = data.empty() ? nullptr : data.data();
    if (vkCreatePipelineCache(vkdev(), &ci, nullptr, &g_cache) != VK_SUCCESS) {
        ci.initialDataSize = 0;
        ci.pInitialData = nullptr;
        vkCreatePipelineCache(vkdev(), &ci, nullptr, &g_cache);
    } else if (!data.empty()) {
        NXLOG_I(kTag, "pipeline cache loaded: %zu bytes from %s", data.size(), g_cache_path.c_str());
    }
    g_cache_ready.store(true, std::memory_order_release);
}

void save_cache() {
    if (!g_cache || g_cache_path.empty()) return;
    NXFT("pipe.cache.save");
    g_cache_dirty.store(false);
    g_new_since_save.store(0);
    size_t n = 0;
    if (vkGetPipelineCacheData(vkdev(), g_cache, &n, nullptr) != VK_SUCCESS || !n) return;
    std::vector<char> data(n);
    if (vkGetPipelineCacheData(vkdev(), g_cache, &n, data.data()) != VK_SUCCESS) return;
    // written to a temporary file and renamed, so a process that ends in the middle of it leaves the old cache intact
    const std::string tmp = g_cache_path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return;
        f.write(data.data(), static_cast<std::streamsize>(n));
        if (!f) return;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, g_cache_path, ec);
    if (ec) {
        std::filesystem::remove(g_cache_path, ec);
        std::filesystem::rename(tmp, g_cache_path, ec);
    }
}

}  // namespace

// A cache of hundreds of MB (the shader preparation's) is rewritten only after thousands of new pipelines: the new ones are few next to it and
// the write takes seconds. The ordinary small cache is written whenever something new appeared, as before.
bool pipeline_cache_dirty() {
    if (!g_cache_dirty.load(std::memory_order_relaxed)) return false;
    return !g_cache_big || g_new_since_save.load(std::memory_order_relaxed) >= 3000;
}
void pipeline_cache_save() { save_cache(); }
// The write at the end of a session: skipped when the cache is the big one of a preparation and little was added.
void pipeline_cache_flush() {
    if (!g_cache_big || g_new_since_save.load(std::memory_order_relaxed) >= 300) save_cache();
}

bool GraphicsKey::operator==(const GraphicsKey& o) const { return std::memcmp(this, &o, sizeof *this) == 0; }
// Word by word: byte by byte it was ~300 dependent multiplies a lookup, and the render thread looks up a key for most draws (~1,700 a
// frame in Dagsion; rt agent, notes/rt.md). Only the in-memory map g_gfx uses it. Keys are zeroed before they are filled (padding too).
uint64_t GraphicsKey::hash() const {
    static_assert(sizeof(GraphicsKey) % 8 == 0, "GraphicsKey is hashed in 8-byte words");
    const uint8_t* p = reinterpret_cast<const uint8_t*>(this);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < sizeof *this; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * 0x9e3779b97f4a7c15ull;
        h ^= h >> 29;
    }
    return h;
}

// A module for SPIR-V words. The translator declares OpCapability Float64 in every module although no module uses a 64-bit float;
// devices without shaderFloat64 (Intel iGPUs) reject that, so the capability is dropped when no OpTypeFloat 64 exists.
static VkShaderModule create_module(const std::vector<uint32_t>& words, const char* what) {
    std::vector<uint32_t> fixed;
    const std::vector<uint32_t>* code = &words;
    {
        const auto& w = words;
        size_t cap_at = 0;
        bool uses64 = false;
        for (size_t i = 5; i < w.size();) {
            const uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
            if (len == 0) break;
            if (op == 17 && w[i + 1] == 10) cap_at = i;                       // OpCapability Float64
            if (op == 22 && i + 2 < w.size() && w[i + 2] == 64) uses64 = true;  // OpTypeFloat 64
            i += len;
        }
        if (cap_at && !uses64) {
            fixed = w;
            fixed.erase(fixed.begin() + static_cast<std::ptrdiff_t>(cap_at), fixed.begin() + static_cast<std::ptrdiff_t>(cap_at) + 2);
            code = &fixed;
        }
    }
    // Every Location output of a vertex / tessellation evaluation / geometry shader starts at (0,0,0,1) as on the console: a component the
    // shader never writes is undefined in Vulkan, and RADV hands NaN where AMD's Windows driver hands 0 (spirv_init_outputs). Every
    // module goes through here, the ones spirv_add_outputs / spirv_gs_passthrough rewrote included. [nvn] init_outputs = false: off (A/B).
    static const bool init_outputs = config_get_bool("nvn", "init_outputs", true);
    std::vector<uint32_t> initialised;
    if (init_outputs && spirv_init_outputs(*code, initialised)) code = &initialised;
    // [nvn] spirv_salt (measurement only): an OpSourceExtension with the salt in every module gives the driver a new shader hash, so a run
    // compiles every shader from scratch without touching the driver's own cache on disk.
    static const std::string salt = config_get_str("nvn", "spirv_salt", "");
    std::vector<uint32_t> salted;
    if (!salt.empty() && code->size() > 5) {
        salted = *code;
        size_t at = 5;
        while (at < salted.size()) {
            const uint32_t op = salted[at] & 0xffff, len = salted[at] >> 16;
            if (len == 0) break;
            if (op != 17 && op != 10 && op != 11 && op != 14 && op != 15 && op != 16 && op != 331) break;   // capability .. execution mode
            at += len;
        }
        std::vector<uint32_t> ins((salt.size() + 1 + 3) / 4 + 1, 0);
        ins[0] = (static_cast<uint32_t>(ins.size()) << 16) | 4u;                 // OpSourceExtension
        std::memcpy(&ins[1], salt.c_str(), salt.size());
        salted.insert(salted.begin() + static_cast<std::ptrdiff_t>(at), ins.begin(), ins.end());
        code = &salted;
    }
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code->size() * 4;
    ci.pCode = code->data();
    VkShaderModule m = VK_NULL_HANDLE;
    const VkResult r = vkCreateShaderModule(vkdev(), &ci, nullptr, &m);
    if (r != VK_SUCCESS) {
        NXLOG_E(kTag, "vkCreateShaderModule failed (%d) for %s", static_cast<int>(r), what);
        return m;
    }
    uint64_t id = 1469598103934665603ull ^ (static_cast<uint64_t>(code->size()) * 0x9e3779b97f4a7c15ull);
    for (uint32_t w : *code) id = (id ^ w) * 1099511628211ull;
    {
        std::lock_guard<std::mutex> lk(g_module_id_mu);
        const auto subgroup = subgroup_shader(*code);
        // GPL keys include the guest subgroup policy. Capabilities are immutable
        // for the lifetime of these device-local caches.
        g_module_ids[m] = subgroup.used ? id ^ 0x6775657374333201ull : id;
        g_module_subgroups[m] = subgroup;
    }
    return m;
}

VkShaderModule shader_module(const ShaderInfo* s) {
    if (!s) return VK_NULL_HANDLE;
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    auto it = g_modules.find(s);
    if (it != g_modules.end()) return it->second;
    VkShaderModule m = create_module(s->spirv, s->hash.c_str());
    g_modules[s] = m;
    return m;
}

static const std::pair<VkShaderModule, uint32_t>& xfb_variant(const ShaderInfo* vs) {
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    auto it = g_xfb_mod.find(vs);
    if (it != g_xfb_mod.end()) return it->second;
    std::vector<uint32_t> patched;
    std::vector<SpvXfbOutput> lay;
    uint32_t stride = 0;
    VkShaderModule m = VK_NULL_HANDLE;
    if (spirv_add_xfb(vs->spirv, patched, &lay, &stride)) {
        m = create_module(patched, vs->hash.c_str());
        std::string desc;
        for (const SpvXfbOutput& o : lay) desc += " loc" + std::to_string(o.location) + "x" + std::to_string(o.components) + "@" + std::to_string(o.offset);
        NXLOG_I(kTag, "%s: transform feedback variant, %u bytes per vertex:%s", vs->hash.c_str(), stride, desc.c_str());
    } else {
        NXLOG_W(kTag, "%s: no transform feedback variant could be made (no Location outputs?)", vs->hash.c_str());
    }
    return g_xfb_mod.emplace(vs, std::make_pair(m, stride)).first->second;
}

uint32_t xfb_stride(const ShaderInfo* vs) { return vs ? xfb_variant(vs).second : 0; }

static VkShaderModule no_output_module(const ShaderInfo* fs) {
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    auto it = g_no_out.find(fs);
    if (it != g_no_out.end()) return it->second.first;
    std::vector<uint32_t> patched;
    std::pair<VkShaderModule, bool> e = {VK_NULL_HANDLE, false};
    if (spirv_outputs_to_private(fs->spirv, patched)) e = {create_module(patched, fs->hash.c_str()), true};
    else e = {shader_module(fs), false};
    g_no_out[fs] = e;
    return e.first;
}

// The tone-map fragment shader runs as its HDR variant while the HDR scene is active (extras/hdr_nvn.h); every other fragment shader is the translated one.
static VkShaderModule hdr_fragment_module(const ShaderInfo* fs) {
    if (!extras::hdr_scene_active() || fs->spirv.size() > 65536) return shader_module(fs);   // Pokopia's composite tone map is about 24.5k words
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    static std::unordered_map<const ShaderInfo*, VkShaderModule> cache;
    auto it = cache.find(fs);
    if (it != cache.end()) return it->second;
    std::vector<uint32_t> patched;
    VkShaderModule m = extras::hdr_tonemap_variant(fs->spirv, fs->hash, patched) ? create_module(patched, fs->hash.c_str()) : shader_module(fs);
    cache[fs] = m;
    return m;
}

static VkShaderModule dual_source_module(const ShaderInfo* fs) {
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    auto it = g_dual_src.find(fs);
    if (it != g_dual_src.end()) return it->second;
    VkShaderModule m = VK_NULL_HANDLE;
    std::vector<uint32_t> patched;
    if (spirv_dual_source_output(fs->spirv, patched)) {
        m = create_module(patched, fs->hash.c_str());
    } else {
        NXLOG_W(kTag, "%s: dual-source blending is on but the fragment shader has no second output; the unpatched module is used", fs->hash.c_str());
        m = shader_module(fs);
    }
    g_dual_src[fs] = m;
    return m;
}

// The slots `consumer` reads that `producer` does not write: the console gives them undefined values, Vulkan needs an output.
static std::vector<SpvOutputSlot> missing_outputs(const ShaderInfo* producer, const ShaderInfo* consumer) {
    std::vector<SpvOutputSlot> missing;
    if (!producer || !consumer) return missing;
    bool covered[64] = {};
    for (const ShaderIo& o : producer->refl.outputs)
        for (uint32_t e = 0; e < (o.array ? o.array : 1); ++e)
            if (o.location + e < 64) covered[o.location + e] = true;
    for (const ShaderIo& in : consumer->refl.inputs) {
        for (uint32_t e = 0; e < (in.array ? in.array : 1); ++e) {
            const uint32_t loc = in.location + e;
            if (loc >= 64 || covered[loc]) continue;
            covered[loc] = true;
            SpvOutputSlot slot;
            slot.location = loc;
            slot.type = in.type == ScalarType::Sint ? SpvScalar::Sint : in.type == ScalarType::Uint ? SpvScalar::Uint : SpvScalar::Float;
            missing.push_back(slot);
        }
    }
    return missing;
}

// The module of `producer` to use in a pipeline whose next stage is `consumer`: the translated one, or a copy with the outputs the
// consumer reads and the producer lacks. Null on a Vulkan error.
static VkShaderModule producer_module(const ShaderInfo* producer, const ShaderInfo* consumer, const std::vector<SpvIntSlot>& ints = {}) {
    const std::vector<SpvOutputSlot> missing = missing_outputs(producer, consumer);
    if (missing.empty() && ints.empty()) return shader_module(producer);
    std::vector<uint32_t> id;
    for (const SpvOutputSlot& m : missing) id.push_back(m.location * 4 + static_cast<uint32_t>(m.type));
    for (const SpvIntSlot& n : ints) id.push_back(0x80000000u | (n.location * 2 + (n.is_signed ? 1u : 0u)));
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    auto it = g_variants.find({producer, id});
    if (it != g_variants.end()) return it->second;
    VkShaderModule m = VK_NULL_HANDLE;
    std::vector<uint32_t> base_words;
    const std::vector<uint32_t>* words = &producer->spirv;
    if (!ints.empty()) {
        // integer vertex attributes: the shader gets the raw bits the console delivers (nvn_spirv_patch.h)
        if (!spirv_int_attributes(producer->spirv, ints, base_words)) {
            NXLOG_E(kTag, "%s: the integer vertex attributes could not be patched in; draws with this shader are skipped", producer->hash.c_str());
            g_variants[{producer, id}] = VK_NULL_HANDLE;
            return VK_NULL_HANDLE;
        }
        words = &base_words;
        if (missing.empty()) {
            m = create_module(base_words, producer->hash.c_str());
            g_variants[{producer, id}] = m;
            return m;
        }
    }
    std::vector<uint32_t> patched;
    // a passthrough geometry shader: the hardware forwards the attributes of the input vertices, the translation has no code for it
    const bool pass = producer->stage == ShaderStage::Geometry && spirv_gs_passthrough(*words, missing, patched);
    if (pass || spirv_add_outputs(*words, missing, patched)) {
        m = create_module(patched, producer->hash.c_str());
        NXLOG_D(kTag, "%s: %zu outputs added for the next stage%s", producer->hash.c_str(), missing.size(), pass ? " (copied from the input vertices)" : "");
    } else {
        NXLOG_W(kTag, "%s: the %zu varyings the next stage reads could not be added; the unpatched module is used", producer->hash.c_str(), missing.size());
        m = shader_module(producer);
    }
    g_variants[{producer, id}] = m;
    return m;
}

// The device's push descriptor limit (0 without VK_KHR_push_descriptor).
uint32_t push_descriptor_max() {
    static const uint32_t n = [] {
        if (!device().features().push_descriptor || !vkCmdPushDescriptorSetKHR) return 0u;
        VkPhysicalDevicePushDescriptorPropertiesKHR pp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &pp};
        vkGetPhysicalDeviceProperties2(device().physical(), &p2);
        return pp.maxPushDescriptors;
    }();
    return n;
}

// [nvn] push_uniforms (default true since 0.1.7, #79): the uniform buffers of a draw are pushed (vkCmdPushDescriptorSetKHR) rather than written into a descriptor
// set. A crowd draws thousands of times a frame with the same textures and its own uniform offsets; every such draw missed the set cache (the
// offsets are part of the set) and paid a set allocation, its writes and a bind: about half of the render thread in a crowded town (#79).
// false: descriptor sets as before. Read once (the pipeline layouts depend on it; changing it rebuilds the pipelines once).
bool push_uniforms_on() {
    static const bool on = [] {
        const bool want = nxrt::config_get_bool("nvn", "push_uniforms", true);
        const uint32_t max = push_descriptor_max();
        const bool ok = want && max >= 8;
        NXLOG_I(kTag, "uniform buffers: %s ([nvn] push_uniforms = %s, VK_KHR_push_descriptor %s, %u push descriptors a set)",
                ok ? "pushed with each draw" : "descriptor sets", want ? "true" : "false", device().features().push_descriptor ? "yes" : "no", max);
        return ok;
    }();
    return on;
}

const ProgLayout* prog_layout(const ShaderInfo* const stages[6]) {
    LayoutKey key;
    for (int i = 0; i < 6; ++i) key.s[i] = stages[i];
    std::lock_guard<std::mutex> lk(g_layout_mu);
    auto it = g_layouts.find(key);
    if (it != g_layouts.end()) return &it->second;

    ProgLayout pl;
    std::map<std::pair<uint32_t, uint32_t>, size_t> index;
    for (int st = 0; st < 6; ++st) {
        const ShaderInfo* s = stages[st];
        if (!s) continue;
        for (size_t i = 0; i < s->refl.bindings.size(); ++i) {
            const DescriptorBinding& b = s->refl.bindings[i];
            if (!b.used || (b.set > 3 && b.set != kPoolSet)) continue;
            const auto k = std::make_pair(b.set, b.binding);
            auto f = index.find(k);
            if (f == index.end()) {
                BindingDesc d;
                d.set = b.set;
                d.binding = b.binding;
                d.type = static_cast<VkDescriptorType>(b.type);
                d.stages = stage_bit(s->stage);
                d.shader = s;
                d.index = static_cast<int>(i);
                d.nvn_stage = st;
                index[k] = pl.bindings.size();
                pl.bindings.push_back(d);
            } else {
                pl.bindings[f->second].stages |= stage_bit(s->stage);   // shared binding (the support buffer)
            }
        }
    }
    std::vector<VkDescriptorSetLayoutBinding> lbs[4];
    std::string sig;                                   // the descriptors of the program, as bytes: the key of the shared layout
    for (int set = 0; set < 4; ++set) {
        for (const BindingDesc& d : pl.bindings) {
            if (d.set != static_cast<uint32_t>(set)) continue;
            VkDescriptorSetLayoutBinding b = {};
            b.binding = d.binding;
            b.descriptorType = d.type;
            b.descriptorCount = std::max<uint32_t>(1, d.shader->refl.bindings[d.index].count);
            b.stageFlags = d.stages;
            lbs[set].push_back(b);
            pl.has_set[set] = true;
            const uint32_t w[4] = {b.binding, static_cast<uint32_t>(b.descriptorType), b.descriptorCount, b.stageFlags};
            sig.append(reinterpret_cast<const char*>(w), sizeof w);
        }
        sig.append("|", 1);
    }
    // set 0 pushed: every binding a single buffer, no more of them than the device pushes in one set
    if (pl.has_set[0] && push_uniforms_on()) {
        bool ok = lbs[0].size() <= push_descriptor_max();
        for (const VkDescriptorSetLayoutBinding& b : lbs[0])
            ok = ok && b.descriptorCount == 1 && (b.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || b.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        pl.push_set0 = ok;
        if (!ok) {
            static int logged = 0;
            if (logged++ < 4)
                NXLOG_I(kTag, "a program's uniform set (%zu bindings) is not pushed: more than the device's %u push descriptors, or not only buffers",
                        lbs[0].size(), push_descriptor_max());
        }
        if (ok) sig.append("push0", 5);
    }
    // the pool arrays: the shared layout of kPoolSet (pool_set_layout), whatever arrays the program's stages use
    for (const BindingDesc& d : pl.bindings) pl.has_set[kPoolSet] |= d.set == kPoolSet;
    VkDescriptorSetLayout pool_layout = VK_NULL_HANDLE;
    if (pl.has_set[kPoolSet]) {
        pool_layout = pool_set_layout();
        if (!pool_layout) {
            static bool logged = false;
            if (!logged) { logged = true; NXLOG_E(kTag, "a program samples through the texture pool but the device has no pool set layout: its draws are skipped"); }
            return nullptr;
        }
        sig.append("pools", 5);
    }
    // push constants of generated stages (the pass-through tessellation control shader): one range from offset 0 over those stages
    VkPushConstantRange pcr = {0, 0, 0};
    for (int st = 0; st < 6; ++st)
        if (stages[st] && stages[st]->push_bytes) {
            pcr.stageFlags |= stage_bit(stages[st]->stage);
            pcr.size = std::max(pcr.size, stages[st]->push_bytes);
        }
    if (pcr.size) {
        const uint32_t w[2] = {pcr.stageFlags, pcr.size};
        sig.append("P", 1);
        sig.append(reinterpret_cast<const char*>(w), sizeof w);
    }
    auto sh = g_shared_layouts.find(sig);
    if (sh == g_shared_layouts.end()) {
        SharedLayout nl;
        for (int set = 0; set < 4; ++set) {
            VkDescriptorSetLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            if (set == 0 && pl.push_set0) ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            ci.bindingCount = static_cast<uint32_t>(lbs[set].size());
            ci.pBindings = lbs[set].data();
            if (vkCreateDescriptorSetLayout(vkdev(), &ci, nullptr, &nl.sets[set]) != VK_SUCCESS) {
                NXLOG_E(kTag, "descriptor set layout %d failed", set);
                return nullptr;
            }
        }
        nl.sets[kPoolSet] = pool_layout;            // not owned (pool_sets_shutdown destroys it)
        VkPipelineLayoutCreateInfo pci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pci.setLayoutCount = pool_layout ? 5 : 4;
        pci.pSetLayouts = nl.sets;
        pci.pushConstantRangeCount = pcr.size ? 1 : 0;
        pci.pPushConstantRanges = pcr.size ? &pcr : nullptr;
        if (vkCreatePipelineLayout(vkdev(), &pci, nullptr, &nl.layout) != VK_SUCCESS) {
            NXLOG_E(kTag, "pipeline layout failed");
            return nullptr;
        }
        sh = g_shared_layouts.emplace(sig, nl).first;
    }
    for (int set = 0; set < 5; ++set) pl.sets[set] = sh->second.sets[set];
    pl.layout = sh->second.layout;
    pl.push_stages = pcr.stageFlags;
    pl.push_bytes = pcr.size;
    return &g_layouts.emplace(key, std::move(pl)).first->second;
}

std::vector<SpvIntSlot> key_int_slots(const GraphicsKey& key) {
    // vertex inputs the shader declares as floats but the game feeds with integer attributes (key formats UINT / SINT)
    std::vector<SpvIntSlot> int_slots;
    if (key.stage[0] && !key.xfb)
        for (uint32_t k = 0; k < key.attr_count && k < 16; ++k) {
            const VkFormat f = static_cast<VkFormat>(key.attr[k].format);
            if (!vk_format_is_integer(f)) continue;
            for (const ShaderIo& in : key.stage[0]->refl.inputs)
                if (in.type == ScalarType::Float && key.attr[k].location >= in.location && key.attr[k].location < in.location + (in.array ? in.array : 1)) {
                    SpvIntSlot sl;
                    sl.location = key.attr[k].location;
                    sl.is_signed = vk_format_is_signed_int(f);
                    int_slots.push_back(sl);
                    break;
                }
        }
    return int_slots;
}

bool resolve_stage_modules(const GraphicsKey& key, StageModules* out) {
    *out = StageModules{};
    // the stage that feeds the fragment shader gets outputs for the varyings the fragment shader reads and it does not write
    // (geometry, else tessellation evaluation, else vertex)
    const int feeder = key.stage[2] ? 2 : key.tess[1] ? 4 : 0;
    const std::vector<SpvIntSlot> int_slots = key_int_slots(key);
    for (int i = 0; i < 3; ++i) {
        if (!key.stage[i]) continue;
        if (key.xfb && i != 0) continue;     // stream-out draws run the vertex stage only
        if (key.xfb) {
            VkShaderModule m = xfb_variant(key.stage[0]).first;
            if (!m) return false;
            out->m[0] = m;
            continue;
        }
        VkShaderModule m = (i == feeder && key.stage[1]) ? producer_module(key.stage[i], key.stage[1], i == 0 ? int_slots : std::vector<SpvIntSlot>())
                           : (i == 0 && !int_slots.empty()) ? producer_module(key.stage[i], nullptr, int_slots)
                           : (i == 1 && key.no_out)      ? no_output_module(key.stage[i])
                           : (i == 1 && key.dual_src)    ? dual_source_module(key.stage[i])
                           : (i == 1)                    ? hdr_fragment_module(key.stage[i])
                                                         : shader_module(key.stage[i]);
        if (!m) return false;
        out->m[i] = m;
    }
    // tessellation stages (never under stream-out: the replay does not build such keys)
    for (int t = 0; t < 2 && !key.xfb; ++t) {
        const ShaderInfo* s = key.tess[t];
        if (!s) continue;
        VkShaderModule m = (t == 1 && feeder == 4 && key.stage[1]) ? producer_module(s, key.stage[1], std::vector<SpvIntSlot>()) : shader_module(s);
        if (!m) return false;
        out->m[3 + t] = m;
    }
    return true;
}

const VkPipelineDynamicStateCreateInfo* pipeline_dynamic_state() {
    static const VkDynamicState dyn[] = {
        VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,
        VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE, VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE, VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_DEPTH_BOUNDS, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_LINE_WIDTH};
    static const VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0,
                                                        sizeof(dyn) / sizeof(dyn[0]), dyn};
    return &dy;
}

// A constant bank read past its bound size gives zero on the console; translated shaders index banks with values read from them (the
// light loops of the game's lighting compute shaders run past their 896-byte block), and without robust access those reads go anywhere.
const void* ubo_robustness() {
    auto make = [](VkPipelineRobustnessBufferBehaviorEXT ubo) {
        VkPipelineRobustnessCreateInfoEXT r = {VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO_EXT};
        r.storageBuffers = r.vertexInputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DEVICE_DEFAULT_EXT;
        r.uniformBuffers = ubo;
        r.images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DEVICE_DEFAULT_EXT;
        return r;
    };
    static const VkPipelineRobustnessCreateInfoEXT v1 = make(VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_EXT),
                                                   v2 = make(VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT);
    static const bool on = nxrt::config_get_bool("nvn", "robust_ubo", true);
    const int level = on ? device().features().ubo_robustness : 0;
    return level == 2 ? &v2 : level == 1 ? &v1 : nullptr;
}

VkPipelineCache pipeline_cache_handle() {
    ensure_cache();
    return g_cache;
}

void note_pipeline_created(double ms, const char* what) {
    counters().pipelines_created.fetch_add(1);
    g_pipes_created.add(1);
    g_cache_dirty.store(true, std::memory_order_relaxed);
    if (ms >= 2.0) g_new_since_save.fetch_add(1, std::memory_order_relaxed);      // a part the cache already had comes back in well under that
    NXLOG_D(kTag, "%s created in %.1f ms", what, ms);
}

void configure_guest_subgroup(VkPipelineShaderStageCreateInfo& stage,
                              VkPipelineShaderStageRequiredSubgroupSizeCreateInfo& size) {
    SubgroupShader shader;
    {
        std::lock_guard<std::mutex> lk(g_module_id_mu);
        auto it = g_module_subgroups.find(stage.module);
        if (it == g_module_subgroups.end()) return;
        shader = it->second;
    }
    if (!shader.used) return;
    if (!require_guest_subgroup(stage, size, shader, device().features(), device().limits_ext())) {
        static std::atomic<uint32_t> warned{0};
        if (!(warned.fetch_or(stage.stage) & stage.stage))
            NXLOG_W(kTag, "cannot require guest subgroup size 32 for stage 0x%x (device support or compute workgroup limit); native size %u",
                    stage.stage, device().limits_ext().subgroup_size);
    }
}

VkPipeline build_graphics_pipeline(const GraphicsKey& key, const ProgLayout& layout) {
    ensure_cache();
    const auto t0 = std::chrono::steady_clock::now();

    VkPipelineShaderStageCreateInfo stages[5] = {};
    uint32_t ns = 0;
    const VkShaderStageFlagBits bits[5] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_GEOMETRY_BIT,
                                           VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT};
    StageModules sm;
    if (!resolve_stage_modules(key, &sm)) return VK_NULL_HANDLE;
    for (int i = 0; i < 5; ++i) {
        if (!sm.m[i]) continue;
        stages[ns] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, ubo_robustness()};
        stages[ns].stage = bits[i];
        stages[ns].module = sm.m[i];
        stages[ns].pName = "main";
        ++ns;
    }

    // vertex input
    VkVertexInputBindingDescription vb[16];
    VkVertexInputAttributeDescription va[16];
    uint32_t nvb = 0, nva = 0;
    bool binding_seen[16] = {};
    for (uint32_t i = 0; i < key.attr_count && i < 16; ++i) {
        const KeyAttr& a = key.attr[i];
        va[nva++] = {a.location, a.binding, static_cast<VkFormat>(a.format), a.offset};
        if (!binding_seen[a.binding]) {
            binding_seen[a.binding] = true;
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
    VkPipelineTessellationStateCreateInfo ts = {VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
    ts.patchControlPoints = key.patch_control_points;

    VkPipelineViewportDepthClipControlCreateInfoEXT dcc = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT};
    dcc.negativeOneToOne = key.depth_neg_one ? VK_TRUE : VK_FALSE;
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    if (device().features().depth_clip_control) vp.pNext = &dcc;   // viewport and scissor counts are dynamic

    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.depthClampEnable = key.depth_clamp ? VK_TRUE : VK_FALSE;
    rs.polygonMode = static_cast<VkPolygonMode>(key.polygon_mode);
    rs.lineWidth = 1.0f;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = static_cast<VkSampleCountFlagBits>(key.samples ? key.samples : 1);
    ms.alphaToCoverageEnable = key.alpha_to_coverage ? VK_TRUE : VK_FALSE;

    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.maxDepthBounds = 1.0f;

    VkPipelineColorBlendAttachmentState cba[8];
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
    }
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.logicOpEnable = key.logic_op_enable ? VK_TRUE : VK_FALSE;
    cb.logicOp = static_cast<VkLogicOp>(key.logic_op);
    cb.attachmentCount = key.color_count;
    cb.pAttachments = cba;

    VkFormat cf[8];
    for (uint32_t i = 0; i < key.color_count && i < 8; ++i) cf[i] = static_cast<VkFormat>(key.color_format[i]);
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = key.color_count;
    ri.pColorAttachmentFormats = cf;
    ri.depthAttachmentFormat = static_cast<VkFormat>(key.depth_format);
    ri.stencilAttachmentFormat = static_cast<VkFormat>(key.stencil_format);

    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri};
    gi.stageCount = ns;
    gi.pStages = stages;
    gi.pVertexInputState = &vi;
    gi.pInputAssemblyState = &ia;
    gi.pTessellationState = key.patch_control_points ? &ts : nullptr;
    gi.pViewportState = &vp;
    gi.pRasterizationState = &rs;
    gi.pMultisampleState = &ms;
    gi.pDepthStencilState = &ds;
    gi.pColorBlendState = &cb;
    gi.pDynamicState = pipeline_dynamic_state();
    gi.layout = layout.layout;
    VkPipeline pipe = VK_NULL_HANDLE;
    const VkResult r = create_graphics_pipeline_tracked(vkdev(), g_cache, &gi, &pipe);
    const double ms_taken = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (r != VK_SUCCESS) {
        NXLOG_E(kTag, "graphics pipeline failed (%d): vs %s fs %s, %u colour targets", static_cast<int>(r), key.stage[0] ? key.stage[0]->hash.c_str() : "-",
                key.stage[1] ? key.stage[1]->hash.c_str() : "-", key.color_count);
        pipe = VK_NULL_HANDLE;
    } else {
        counters().pipelines_created.fetch_add(1);
        g_cache_dirty.store(true, std::memory_order_relaxed);
    g_new_since_save.fetch_add(1, std::memory_order_relaxed);
        g_pipes_created.add(1);
        NXLOG_D(kTag, "graphics pipeline created in %.1f ms (vs %s fs %s)", ms_taken, key.stage[0] ? key.stage[0]->hash.c_str() : "-",
                key.stage[1] ? key.stage[1]->hash.c_str() : "-");
    }
    return pipe;
}

VkPipeline graphics_pipeline(const GraphicsKey& key, const ProgLayout& layout) {
    ensure_cache();
    const uint64_t h = key.hash();
    auto& bucket = g_gfx[h];
    for (auto& e : bucket)
        if (e.first == key) {
            counters().pipeline_cache_hits.fetch_add(1, std::memory_order_relaxed);
            if (e.up && e.up->state.load(std::memory_order_acquire) != 0) gpl_take_upgrade(e.second, e.alt, e.up);
            if (e.alt && gpl_ab_whole()) return e.alt;
            return e.second;
        }
    prewarm_init();
    VkPipeline pipe = VK_NULL_HANDLE;
    std::shared_ptr<PipeUpgrade> later;          // the key file's whole pipeline, still being built: it replaces the linked one
    // compiled ahead on a worker; one being compiled is linked from its parts meanwhile when they are built (else waited for)
    if (prewarm_take(key, &pipe, [&] { return gpl_ready(key, layout); }, &later)) {
        ++g_pipeline_count;
        g_gfx_keys.add(1);
        bucket.push_back({key, pipe, VK_NULL_HANDLE, nullptr});
        return pipe;
    }
    std::shared_ptr<PipeUpgrade> up;
    bool pending = false;
    pipe = gpl_link(key, layout, later ? nullptr : &up, &pending); // linked from shader parts compiled ahead (null: not eligible or a part failed)
    if (later) up = later;
    prewarm_record(key);                         // every key a draw asked for: the next run builds it from boot
    if (pending && !later) return VK_NULL_HANDLE;   // its parts are compiling in the background: the draw is skipped, the next one asks again
    g_gfx_keys.add(1);                           // every path below adds one entry to the bucket
    if (pipe) {
        ++g_pipeline_count;
        bucket.push_back({key, pipe, VK_NULL_HANDLE, up});
        return pipe;
    }
    if (pending) {                               // (rare) parts gone missing between the check and the link: skipped until the whole pipeline is in
        bucket.push_back({key, VK_NULL_HANDLE, VK_NULL_HANDLE, up});
        return VK_NULL_HANDLE;
    }
    NXFT("pipe.create");
    pipe = build_graphics_pipeline(key, layout);
    if (pipe) {
        ++g_pipeline_count;
        g_built_at_draw.fetch_add(1, std::memory_order_relaxed);
    }
    bucket.push_back({key, pipe, VK_NULL_HANDLE, up});   // up: the key file's build, when one is under way, replaces this one (same pipeline)
    return pipe;
}

// The compute shader with the samplers in `ints` reading integer images (nvn_spirv_patch.h); null when the rewrite is not possible.
static VkShaderModule int_sampler_module(const ShaderInfo* cs, const std::vector<SpvIntSampler>& ints, const std::vector<uint32_t>& key) {
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    const auto k = std::make_pair(cs, key);
    auto it = g_int_samplers.find(k);
    if (it != g_int_samplers.end()) return it->second;
    std::vector<uint32_t> patched;
    VkShaderModule m = VK_NULL_HANDLE;
    std::string desc;
    for (const SpvIntSampler& s : ints) desc += " set " + std::to_string(s.set) + " binding " + std::to_string(s.binding) + (s.is_signed ? " (int)" : " (uint)");
    if (spirv_int_samplers(cs->spirv, ints, patched)) {
        m = create_module(patched, cs->hash.c_str());
        NXLOG_I(kTag, "%s: integer sampler variant:%s", cs->hash.c_str(), desc.c_str());
    } else {
        NXLOG_W(kTag, "%s: the samplers%s read integer textures but could not be retyped; they get the dummy view", cs->hash.c_str(), desc.c_str());
    }
    g_int_samplers[k] = m;
    return m;
}

VkPipeline compute_pipeline(const ShaderInfo* cs, const ProgLayout& layout, const std::vector<SpvIntSampler>& ints) {
    ensure_cache();
    std::vector<uint32_t> ikey;
    for (const SpvIntSampler& s : ints) ikey.push_back(s.set << 17 | s.binding << 1 | (s.is_signed ? 1u : 0u));
    const auto k = std::make_tuple(cs, layout.layout, ikey);
    auto it = g_compute.find(k);
    if (it != g_compute.end()) return it->second;
    VkShaderModule m = ints.empty() ? shader_module(cs) : int_sampler_module(cs, ints, ikey);
    VkPipeline pipe = VK_NULL_HANDLE;
    if (m) {
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, ubo_robustness()};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, ubo_robustness()};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = m;
        ci.stage.pName = "main";
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size{};
        configure_guest_subgroup(ci.stage, subgroup_size);
        ci.layout = layout.layout;
        const uint64_t tp0 = nxrt::hitch_ticks();
        const VkResult cr = vkCreateComputePipelines(vkdev(), g_cache, 1, &ci, nullptr, &pipe);
        if (nxrt::hitch_on_render_thread()) nxrt::hitch_phase_add(nxrt::kHpRenderPipeline, nxrt::hitch_ticks() - tp0);
        if (cr != VK_SUCCESS) {
            NXLOG_E(kTag, "compute pipeline failed for %s", cs->hash.c_str());
            pipe = VK_NULL_HANDLE;
        } else {
            counters().pipelines_created.fetch_add(1);
            ++g_pipeline_count;
            g_pipes_created.add(1);
            g_cache_dirty.store(true, std::memory_order_relaxed);
    g_new_since_save.fetch_add(1, std::memory_order_relaxed);
        }
    }
    g_compute[k] = pipe;                         // new key: the lookup above missed
    g_compute_keys.add(1);
    return pipe;
}

VkBuffer constant_data_buffer(const ShaderInfo* s, VkDeviceSize* range) {
    auto it = g_const_bufs.find(s);
    if (it == g_const_bufs.end()) {
        const VkDeviceSize size = std::max<VkDeviceSize>(256, (s->constants.size() + 15) & ~15ull);
        gfx::Buffer b;
        if (device().create_buffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, gfx::MemoryUse::Upload, &b, "nvn c1") != VK_SUCCESS) {
            *range = 0;
            return VK_NULL_HANDLE;
        }
        std::memset(b.mapped, 0, size);
        if (!s->constants.empty()) std::memcpy(b.mapped, s->constants.data(), s->constants.size());
        it = g_const_bufs.emplace(s, std::make_pair(b, size)).first;
        g_const_n.add(1);
        g_const_bytes.add(static_cast<int64_t>(size));
    }
    *range = it->second.second;
    return it->second.first.buffer;
}

VkBuffer driver_constants_buffer(VkDeviceSize* range) {
    if (!g_driver_const_made) {
        g_driver_const_made = true;
        if (device().create_buffer(65536, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, gfx::MemoryUse::Upload, &g_driver_const, "nvn c0") == VK_SUCCESS) {
            std::memset(g_driver_const.mapped, 0, 65536);
            // c[0]+0x610: maximum point size read by the agl vertex shaders (gl_PointSize = min(size, c0[0x610]))
            const float max_point = 64.0f;
            std::memcpy(static_cast<uint8_t*>(g_driver_const.mapped) + 0x610, &max_point, 4);
        }
    }
    *range = 65536;
    return g_driver_const.buffer;
}

// ---- texture and sampler pools as descriptor arrays (kPoolSet) -------------------------------------------------------------------
// The GPU may still run submissions that read a pool set when the game registers something new, so each pool pair keeps a few sets and
// only writes one that no submission in flight reads (or one that only the submission being recorded has bound: the bindings are update
// after bind). A set is brought up to date with the ids registered since it was last current (the pools' change lists); a new set is
// written whole once. An element whose texture is not in its sampled layout yet (an attachment of the open pass) gets the dummy and is
// written again once it is. Render thread, except pool_set_layout (pipeline layouts are also made by the prewarm workers).
namespace {

constexpr VkShaderStageFlags kPoolStages = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT;
constexpr uint64_t kPoolIdle = 2000;            // submissions after which an unused pool pair or texture pool record is dropped

std::mutex g_pool_layout_mu;
VkDescriptorSetLayout g_pool_layout = VK_NULL_HANDLE;
bool g_pool_layout_tried = false;
uint32_t g_pool_copies = 2;                     // sets per pool pair (2..8, from the device's update-after-bind budget)

struct PoolCopy {
    VkDescriptorSet set = VK_NULL_HANDLE;
    bool written = false;                       // every element written once
    uint64_t tex_version = 0, smp_version = 0;  // the registrations it holds
    uint64_t used = 0, used_before = 0;         // serial of the last submission that bound it, and of the one before that
    std::vector<uint32_t> pending;              // texture ids holding the dummy until their texture is in its sampled layout
    std::vector<uint32_t> held;                 // the same for textures whose first upload waits for a later frame (pool_hold): looked at once a frame
};
struct PoolPair {
    Ref<TexturePool> tp;
    Ref<SamplerPool> sp;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    std::vector<PoolCopy> copies;
    int cur = -1;
    uint64_t last_use = 0;
};
// What pool_textures_to_prepare knows of a texture pool.
struct PoolRes {
    Ref<TexturePool> tp;
    uint64_t version = ~0ull;                   // registrations reflected in by_id / reg
    std::vector<Texture*> by_id;                // the texture at each id (compared only; reg holds the references)
    std::unordered_map<Texture*, std::pair<Ref<Texture>, uint32_t>> reg;   // registered textures, the number of ids each sits at
    std::vector<Ref<Texture>> watch;            // to look at next time
    uint64_t epoch = 0;                         // g_tex_scan_epoch when every texture was last looked at
    uint64_t scan_serial = 0;                   // submission of the last full scan made for g_tex_scan_epoch (at most one a submission)
    uint64_t moves = 0;                         // g_pool_moves when the textures were last looked at
    uint64_t watch_serial = 0;                  // submission in which the watch list was last looked at
    bool all = true;                            // look at every texture next time
    uint64_t last_use = 0;
};
std::vector<std::unique_ptr<PoolPair>> g_pool_pairs;
std::vector<std::unique_ptr<PoolRes>> g_pool_res;
std::vector<Ref<Texture>> g_layout_moves;       // pool_note_layout, drained by pool_textures_to_prepare
// pool_hold ([nvn] pool_upload_ms): textures whose first upload waits for a later frame. Until the frame ends they are left out of the watch
// lists and the sets' pending lists, which every draw that indexes the pools looks through (thousands of them while a load-in streams in);
// at the first look of the next frame they are looked at again (one full scan, and back into the pending lists). Render thread.
std::unordered_set<Texture*> g_held;
uint64_t g_held_present = 0;
constexpr size_t kPoolWatchMax = 4096;          // a longer watch list becomes "look at every texture" (PoolRes::all)
// Bumped by everything that may change whether a watched texture is ready to be sampled (a layout change, a CPU write). The end of a pass,
// after which the replay no longer leaves its attachments alone, needs none: end_pass marks every binding dirty, so the next draw looks anyway.
// pool_state_moved compares it with the value the pool's last look saw: O(1), where polling "is the watch list empty" redid every draw's
// descriptors for as long as one texture of it stayed not ready (#79).
std::atomic<uint64_t> g_pool_moves{1};
// Textures the CPU wrote (pool_note_cpu_write, any thread), drained into the watch lists of the pools they are registered in. Compared only,
// never dereferenced: the pool's own reference (PoolRes::reg) is what goes into the watch list.
std::mutex g_cpu_writes_mu;
std::vector<Texture*> g_cpu_writes;
bool g_cpu_writes_over = false;                 // more than kPoolWatchMax since the last drain: every pool looks at every texture
std::atomic<bool> g_cpu_writes_any{false};
uint64_t g_pool_full_scans = 0, g_pool_looks = 0;   // render thread: replay_stats (pool_scan_counts)
nxrt::GaugeCounter g_pool_full_scan_gauge{"nvn.pool_full_scans"};
nxrt::GaugeCounter g_pool_writes{"nvn.pool_descriptors_written"};
nxrt::GaugeCounter g_pool_sets{"nvn.pool_sets"};

template <class Pool>
uint64_t version_of(Pool* p) {
    if (!p) return 0;
    std::shared_lock<std::shared_mutex> lk(p->lock);
    return p->version;
}

// Texture id `id` as an element of the 2D or cube array: a view of that shape when the texture (or its view) has it and it is in its
// sampled layout, else the dummy (*not_ready: it may become usable, write it again later).
VkDescriptorImageInfo pool_image(const TexturePool::Entry& e, VkImageViewType type, bool* not_ready) {
    VkDescriptorImageInfo ii = {VK_NULL_HANDLE, dummy_view(type, false), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    Texture* t = e.used ? e.tex.get() : nullptr;
    if (!t || !t->fmt) return ii;
    // finalized and still registered: the image of a live texture on the same storage (tex_gpu takes it), else the dummy, not a new image
    if (t->destroyed.load(std::memory_order_relaxed) && !t->gpu && !texture_registry_find_alias(t)) return ii;
    const int target = e.has_view && e.view.has_target ? e.view.target : t->b.target;
    // a square 2D array of 6n layers is also a cube (its image is made cube compatible, shape_of): the game registers its environment
    // maps that way and samples them through the cube array, as it binds them to cube samplers
    const bool cube_array = target == NVN_TEXTURE_TARGET_2D_ARRAY && t->b.width == t->b.height && t->b.samples <= 1 && t->layers >= 6 &&
                            t->layers % 6 == 0;
    const bool cube = target == NVN_TEXTURE_TARGET_CUBEMAP || target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY || cube_array;
    // a cube read through the 2D array gives its first face: Maxwell keeps a cube's faces as layers and a non-array sample reads layer 0, as
    // tex_sampled_view does for bound slots (nxnative-penis6 6b0cffbb5e); the dummy read zero
    const bool flat = target == NVN_TEXTURE_TARGET_2D || target == NVN_TEXTURE_TARGET_2D_ARRAY || target == NVN_TEXTURE_TARGET_RECTANGLE ||
                      target == NVN_TEXTURE_TARGET_CUBEMAP || target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY;
    if (type == VK_IMAGE_VIEW_TYPE_CUBE ? !cube : !flat) return ii;
    if (!tex_gpu(t)) return ii;
    if (tex_layout(t) != tex_sampled_layout(t)) {
        *not_ready = true;
        return ii;
    }
    ViewRequest q;
    q.type = type;
    int sw[4] = {t->b.swizzle[0], t->b.swizzle[1], t->b.swizzle[2], t->b.swizzle[3]};
    int dsmode = t->b.depth_stencil_mode;
    if (e.has_view) {
        const TextureViewInfo& v = e.view;
        if (v.has_levels) { q.base_level = static_cast<uint32_t>(v.min_level); q.level_count = static_cast<uint32_t>(v.num_levels); }
        if (v.has_layers) { q.base_layer = static_cast<uint32_t>(v.min_layer); q.layer_count = static_cast<uint32_t>(v.num_layers); }
        if (v.has_format)
            if (const FormatInfo* fi = format_info(v.format)) q.format = fi->vk;
        if (v.has_swizzle) for (int i = 0; i < 4; ++i) sw[i] = v.swizzle[i];
        if (v.has_depth_stencil_mode) dsmode = v.depth_stencil_mode;
    }
    q.swizzle = {vk_swizzle(sw[0]), vk_swizzle(sw[1]), vk_swizzle(sw[2]), vk_swizzle(sw[3])};
    q.depth_aspect = dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH;
    q.component = 1;                            // the arrays are float images
    // An 8- or 16-bit integer texture (the light probes' RGBA8UI cubes) has no float view with its bits, and no translated shader reads
    // integers through the pool arrays (none bitcasts what it samples there): such ids stay the dummy, without a warning per texture.
    if (tex_int_sampler_component(t, q.format)) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            NXLOG_I(kTag, "pool arrays: integer textures (texture %u, %s, first) stay the dummy in the float arrays", t->id, t->fmt->name);
        return ii;
    }
    if (VkImageView v = tex_sampled_view(t, q)) {
        ii.imageView = v;
        ii.imageLayout = tex_sampled_layout(t);
    }
    return ii;
}

// Writes the elements of `ids` (all ids when null) of one set from the pools as they are now.
void write_pool_textures(PoolCopy& c, TexturePool* tp, const std::vector<uint32_t>* ids) {
    static std::vector<VkDescriptorImageInfo> flat, cube;
    std::vector<uint32_t> pending, held;
    std::vector<VkWriteDescriptorSet> writes;
    auto element = [&](const TexturePool::Entry* e, uint32_t id, size_t at) {
        bool not_ready = false;
        static const TexturePool::Entry kNone;
        flat[at] = pool_image(e ? *e : kNone, VK_IMAGE_VIEW_TYPE_2D, &not_ready);
        cube[at] = pool_image(e ? *e : kNone, VK_IMAGE_VIEW_TYPE_CUBE, &not_ready);
        if (not_ready) {
            Texture* t = e && e->used ? e->tex.get() : nullptr;
            if (t && !g_held.empty() && g_held.count(t)) held.push_back(id);
            else pending.push_back(id);
        }
    };
    std::shared_lock<std::shared_mutex> lk;
    if (tp) lk = std::shared_lock<std::shared_mutex>(tp->lock);
    const size_t count = tp ? std::min<size_t>(tp->entries.size(), kPoolTextureCount) : 0;
    if (!ids) {
        flat.resize(kPoolTextureCount);
        cube.resize(kPoolTextureCount);
        for (uint32_t id = 0; id < kPoolTextureCount; ++id) element(id < count ? &tp->entries[id] : nullptr, id, id);
        size_t used = 0, n2d = 0, ncube = 0;
        const VkImageView d2 = dummy_view(VK_IMAGE_VIEW_TYPE_2D, false), dc = dummy_view(VK_IMAGE_VIEW_TYPE_CUBE, false);
        for (uint32_t id = 0; id < count; ++id) {
            used += tp->entries[id].used ? 1 : 0;
            n2d += flat[id].imageView != d2 ? 1 : 0;
            ncube += cube[id].imageView != dc ? 1 : 0;
        }
        NXLOG_I(kTag, "pool arrays: set written whole: %zu of %zu ids registered, %zu sampled as 2D, %zu as cubes, %zu waiting for their sampled layout",
                used, count, n2d, ncube, pending.size());
        for (uint32_t b : {uint32_t(kPoolTex2D), uint32_t(kPoolTexCube)}) {
            VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = c.set;
            w.dstBinding = b;
            w.descriptorCount = kPoolTextureCount;
            w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            w.pImageInfo = b == kPoolTex2D ? flat.data() : cube.data();
            writes.push_back(w);
        }
        c.pending.clear();
        c.held.clear();
    } else {
        flat.resize(ids->size());
        cube.resize(ids->size());
        for (size_t k = 0; k < ids->size(); ++k) {
            const uint32_t id = (*ids)[k];
            element(id < count ? &tp->entries[id] : nullptr, id, k);
            for (uint32_t b : {uint32_t(kPoolTex2D), uint32_t(kPoolTexCube)}) {
                VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w.dstSet = c.set;
                w.dstBinding = b;
                w.dstArrayElement = id;
                w.descriptorCount = 1;
                w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                w.pImageInfo = b == kPoolTex2D ? &flat[k] : &cube[k];
                writes.push_back(w);
            }
        }
        // the ids written now are no longer pending (unless they still are); sorted copy: a load-in streams in thousands of them
        std::vector<uint32_t> written(ids->begin(), ids->end());
        std::sort(written.begin(), written.end());
        auto was_written = [&](uint32_t id) { return std::binary_search(written.begin(), written.end(), id); };
        c.pending.erase(std::remove_if(c.pending.begin(), c.pending.end(), was_written), c.pending.end());
        c.held.erase(std::remove_if(c.held.begin(), c.held.end(), was_written), c.held.end());
    }
    lk = {};
    vkUpdateDescriptorSets(vkdev(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    g_pool_writes.add(static_cast<int64_t>(ids ? ids->size() * 2 : 2ull * kPoolTextureCount));
    c.pending.insert(c.pending.end(), pending.begin(), pending.end());
    c.held.insert(c.held.end(), held.begin(), held.end());
}

void write_pool_samplers(PoolCopy& c, SamplerPool* sp, const std::vector<uint32_t>* ids) {
    static std::vector<VkDescriptorImageInfo> info;
    std::vector<VkWriteDescriptorSet> writes;
    std::shared_lock<std::shared_mutex> lk;
    if (sp) lk = std::shared_lock<std::shared_mutex>(sp->lock);
    auto element = [&](uint32_t id) {
        VkDescriptorImageInfo ii = {dummy_sampler(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        if (sp && id < sp->used.size() && sp->used[id]) ii.sampler = sampler_for(sp->entries[id]);
        return ii;
    };
    if (!ids) {
        info.resize(kPoolSamplerCount);
        for (uint32_t id = 0; id < kPoolSamplerCount; ++id) info[id] = element(id);
        VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = c.set;
        w.dstBinding = kPoolSamplers;
        w.descriptorCount = kPoolSamplerCount;
        w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        w.pImageInfo = info.data();
        writes.push_back(w);
    } else {
        info.resize(ids->size());
        for (size_t k = 0; k < ids->size(); ++k) {
            info[k] = element((*ids)[k]);
            VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = c.set;
            w.dstBinding = kPoolSamplers;
            w.dstArrayElement = (*ids)[k];
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            w.pImageInfo = &info[k];
            writes.push_back(w);
        }
    }
    lk = {};
    vkUpdateDescriptorSets(vkdev(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    g_pool_writes.add(static_cast<int64_t>(ids ? ids->size() : kPoolSamplerCount));
}

// The ids registered after `from` up to the pool's current version (sorted, unique); false when the change list no longer reaches back
// that far (write every id).
template <class Pool>
bool changed_since(Pool* p, uint64_t from, uint64_t* now, std::vector<uint32_t>& ids) {
    ids.clear();
    if (!p) { *now = 0; return true; }
    std::shared_lock<std::shared_mutex> lk(p->lock);
    *now = p->version;
    const uint64_t n = p->version - from;
    if (from > p->version || n > p->changes.size()) return false;
    ids.assign(p->changes.end() - static_cast<std::ptrdiff_t>(n), p->changes.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    ids.erase(std::remove_if(ids.begin(), ids.end(), [](uint32_t id) { return id >= kPoolTextureCount; }), ids.end());
    return true;
}

// The held textures of the frame that ended go back to being looked at (pool_hold). True when there were any.
bool release_held() {
    if (g_held.empty()) return false;
    if (counters().presents.load(std::memory_order_relaxed) == g_held_present) return false;
    g_held.clear();
    for (auto& r : g_pool_res) r->all = true;
    for (auto& pp : g_pool_pairs)
        for (PoolCopy& c : pp->copies) {
            c.pending.insert(c.pending.end(), c.held.begin(), c.held.end());
            c.held.clear();
        }
    return true;
}

// True when some pending element of the set has become writable.
bool pending_ready(const PoolCopy& c, TexturePool* tp) {
    if (c.pending.empty() || !tp) return false;
    std::shared_lock<std::shared_mutex> lk(tp->lock);
    for (uint32_t id : c.pending) {
        if (id >= tp->entries.size()) continue;
        Texture* t = tp->entries[id].used ? tp->entries[id].tex.get() : nullptr;
        if (!t || (t->gpu && tex_layout(t) == tex_sampled_layout(t))) return true;
    }
    return false;
}

void update_copy(PoolCopy& c, TexturePool* tp, SamplerPool* sp) {
    std::vector<uint32_t> ids;
    uint64_t tv = 0, sv = 0;
    if (!c.written) {
        changed_since(tp, 0, &tv, ids);
        changed_since(sp, 0, &sv, ids);
        write_pool_textures(c, tp, nullptr);
        write_pool_samplers(c, sp, nullptr);
        c.written = true;
    } else {
        if (changed_since(tp, c.tex_version, &tv, ids)) {
            for (uint32_t id : c.pending) ids.push_back(id);
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            if (!ids.empty()) write_pool_textures(c, tp, &ids);
        } else {
            write_pool_textures(c, tp, nullptr);
        }
        if (changed_since(sp, c.smp_version, &sv, ids)) {
            ids.erase(std::remove_if(ids.begin(), ids.end(), [](uint32_t id) { return id >= kPoolSamplerCount; }), ids.end());
            if (!ids.empty()) write_pool_samplers(c, sp, &ids);
        } else {
            write_pool_samplers(c, sp, nullptr);
        }
    }
    c.tex_version = tv;
    c.smp_version = sv;
}

PoolPair* pool_pair(TexturePool* tp, SamplerPool* sp) {
    for (auto& p : g_pool_pairs)
        if (p->tp.get() == tp && p->sp.get() == sp) return p.get();
    auto p = std::make_unique<PoolPair>();
    p->tp = Ref<TexturePool>::share(tp);
    p->sp = Ref<SamplerPool>::share(sp);
    const VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_SAMPLER, g_pool_copies * kPoolSamplerCount},
                                           {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, g_pool_copies * 2 * kPoolTextureCount}};
    VkDescriptorPoolCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    ci.maxSets = g_pool_copies;
    ci.poolSizeCount = 2;
    ci.pPoolSizes = sizes;
    const VkResult r = vkCreateDescriptorPool(vkdev(), &ci, nullptr, &p->dpool);
    if (r != VK_SUCCESS) {
        NXLOG_E(kTag, "pool arrays: descriptor pool for %u sets failed (%d)", g_pool_copies, static_cast<int>(r));
        return nullptr;
    }
    vk_gauge_descriptor_pools().add(1);
    g_pool_pairs.push_back(std::move(p));
    return g_pool_pairs.back().get();
}

PoolRes& pool_res(TexturePool* tp) {
    for (auto& r : g_pool_res)
        if (r->tp.get() == tp) return *r;
    auto r = std::make_unique<PoolRes>();
    r->tp = Ref<TexturePool>::share(tp);
    g_pool_res.push_back(std::move(r));
    return *g_pool_res.back();
}

// Drops the pairs and texture pool records nothing used for kPoolIdle submissions (their sets once the GPU is done with them).
void pool_prune(uint64_t serial) {
    static uint64_t last = 0;
    if (serial - last < 256) return;
    last = serial;
    for (auto it = g_pool_pairs.begin(); it != g_pool_pairs.end();) {
        if (serial - (*it)->last_use < kPoolIdle) { ++it; continue; }
        const VkDescriptorPool dp = (*it)->dpool;
        g_pool_sets.sub(static_cast<int64_t>((*it)->copies.size()));
        defer_destroy([dp] { vkDestroyDescriptorPool(vkdev(), dp, nullptr); vk_gauge_descriptor_pools().sub(1); });
        it = g_pool_pairs.erase(it);
    }
    g_pool_res.erase(std::remove_if(g_pool_res.begin(), g_pool_res.end(), [&](const std::unique_ptr<PoolRes>& r) { return serial - r->last_use >= kPoolIdle; }),
                     g_pool_res.end());
}

}  // namespace

VkDescriptorSetLayout pool_set_layout() {
    std::lock_guard<std::mutex> lk(g_pool_layout_mu);
    if (g_pool_layout_tried) return g_pool_layout;
    g_pool_layout_tried = true;
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
    vkGetPhysicalDeviceFeatures2(device().physical(), &f2);
    VkPhysicalDeviceVulkan12Properties p12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p12};
    vkGetPhysicalDeviceProperties2(device().physical(), &p2);
    // the pool arrays plus room for a program's own textures and samplers (sets 2 and 3) in the same stage
    const uint64_t images = 2ull * kPoolTextureCount, per_set = images + kPoolSamplerCount;
    const bool feats = f12.descriptorIndexing && f12.descriptorBindingPartiallyBound && f12.descriptorBindingSampledImageUpdateAfterBind &&
                       f12.shaderSampledImageArrayNonUniformIndexing;
    const bool limits = p12.maxDescriptorSetUpdateAfterBindSampledImages >= images + 512 && p12.maxPerStageDescriptorUpdateAfterBindSampledImages >= images + 256 &&
                        p12.maxDescriptorSetUpdateAfterBindSamplers >= kPoolSamplerCount + 512 && p12.maxPerStageDescriptorUpdateAfterBindSamplers >= kPoolSamplerCount + 256 &&
                        p12.maxPerStageUpdateAfterBindResources >= per_set + 512 && p12.maxUpdateAfterBindDescriptorsInAllPools >= 2 * per_set;
    g_pool_copies = static_cast<uint32_t>(std::clamp<uint64_t>(p12.maxUpdateAfterBindDescriptorsInAllPools / (2 * per_set), 2, 8));
    NXLOG_I(kTag, "pool arrays: descriptor indexing %d, partially bound %d, sampled images update after bind %d, non-uniform indexing %d; update-after-bind limits: "
            "%u sampled images per set, %u per stage, %u samplers per set, %u per stage, %u resources per stage, %u in all pools: %s",
            f12.descriptorIndexing, f12.descriptorBindingPartiallyBound, f12.descriptorBindingSampledImageUpdateAfterBind, f12.shaderSampledImageArrayNonUniformIndexing,
            p12.maxDescriptorSetUpdateAfterBindSampledImages, p12.maxPerStageDescriptorUpdateAfterBindSampledImages, p12.maxDescriptorSetUpdateAfterBindSamplers,
            p12.maxPerStageDescriptorUpdateAfterBindSamplers, p12.maxPerStageUpdateAfterBindResources, p12.maxUpdateAfterBindDescriptorsInAllPools,
            feats && limits ? "usable" : "NOT usable: programs that sample through the pools are not drawn");
    if (!feats || !limits) return VK_NULL_HANDLE;
    const VkDescriptorSetLayoutBinding b[kPoolBindingCount] = {
        {kPoolSamplers, VK_DESCRIPTOR_TYPE_SAMPLER, kPoolSamplerCount, kPoolStages, nullptr},
        {kPoolTex2D, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kPoolTextureCount, kPoolStages, nullptr},
        {kPoolTexCube, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kPoolTextureCount, kPoolStages, nullptr}};
    const VkDescriptorBindingFlags flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    const VkDescriptorBindingFlags fl[kPoolBindingCount] = {flags, flags, flags};
    VkDescriptorSetLayoutBindingFlagsCreateInfo bf = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bf.bindingCount = kPoolBindingCount;
    bf.pBindingFlags = fl;
    VkDescriptorSetLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &bf};
    ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    ci.bindingCount = kPoolBindingCount;
    ci.pBindings = b;
    if (vkCreateDescriptorSetLayout(vkdev(), &ci, nullptr, &g_pool_layout) != VK_SUCCESS) {
        NXLOG_E(kTag, "pool arrays: descriptor set layout failed");
        g_pool_layout = VK_NULL_HANDLE;
    }
    return g_pool_layout;
}

VkDescriptorSet pool_set(TexturePool* tp, SamplerPool* sp, Frame& f) {
    if (!pool_set_layout()) return VK_NULL_HANDLE;
    pool_prune(f.serial);
    PoolPair* pp = pool_pair(tp, sp);
    if (!pp) return VK_NULL_HANDLE;
    pp->last_use = f.serial;
    auto bind = [&](int i) {
        PoolCopy& c = pp->copies[static_cast<size_t>(i)];
        if (c.used != f.serial) { c.used_before = c.used; c.used = f.serial; }
        pp->cur = i;
        return c.set;
    };
    const uint64_t tv = version_of(tp), sv = version_of(sp);
    if (pp->cur >= 0) {
        const PoolCopy& c = pp->copies[static_cast<size_t>(pp->cur)];
        if (c.written && c.tex_version == tv && c.smp_version == sv && !pending_ready(c, tp)) return bind(pp->cur);
    }
    // a set no submission in flight reads: the current one when only this submission has bound it, else the most up to date idle one
    const uint64_t done = poll_completed();
    auto idle = [&](const PoolCopy& c) { return (c.used == f.serial ? c.used_before : c.used) <= done; };
    int pick = pp->cur >= 0 && idle(pp->copies[static_cast<size_t>(pp->cur)]) ? pp->cur : -1;
    for (size_t i = 0; pick < 0 && i < pp->copies.size(); ++i)
        if (idle(pp->copies[i]) && pp->copies[i].written) pick = static_cast<int>(i);
    if (pick < 0 && pp->copies.size() < g_pool_copies) {
        PoolCopy c;
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pp->dpool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &g_pool_layout;
        if (vkAllocateDescriptorSets(vkdev(), &ai, &c.set) == VK_SUCCESS) {
            device().set_name(VK_OBJECT_TYPE_DESCRIPTOR_SET, reinterpret_cast<uint64_t>(c.set), "nvn pool arrays");
            pp->copies.push_back(std::move(c));
            g_pool_sets.add(1);
            pick = static_cast<int>(pp->copies.size() - 1);
        }
    }
    if (pick < 0) {
        // every set is being read by submissions in flight: wait for the one bound longest ago
        size_t oldest = 0;
        for (size_t i = 1; i < pp->copies.size(); ++i)
            if (pp->copies[i].used < pp->copies[oldest].used) oldest = i;
        static int logged = 0;
        if (logged++ < 10) NXLOG_W(kTag, "pool arrays: all %zu sets of a pool pair are in flight; waiting for submission %llu", pp->copies.size(),
                                   static_cast<unsigned long long>(pp->copies[oldest].used));
        submit_recorded();
        while (poll_completed() < pp->copies[oldest].used) std::this_thread::yield();
        pick = static_cast<int>(oldest);
    }
    update_copy(pp->copies[static_cast<size_t>(pick)], tp, sp);
    return bind(pick);
}

void pool_note_layout(Texture* t, VkImageLayout layout) {
    g_pool_moves.fetch_add(1, std::memory_order_relaxed);                      // into its sampled layout too: a watched texture may be ready now
    if (!t || g_pool_res.empty() || layout == tex_sampled_layout(t)) return;   // nothing samples through a pool yet
    if (g_layout_moves.size() >= 4096) {
        for (auto& r : g_pool_res) r->all = true;
        g_layout_moves.clear();
    }
    g_layout_moves.push_back(Ref<Texture>::share(t));
}

void pool_note_cpu_write(Texture* t) {
    if (!t) return;
    {
        std::lock_guard<std::mutex> lk(g_cpu_writes_mu);
        if (!g_cpu_writes_over) {
            if (g_cpu_writes.size() >= kPoolWatchMax) {
                g_cpu_writes_over = true;
                g_cpu_writes.clear();
            } else {
                g_cpu_writes.push_back(t);
            }
        }
    }
    g_cpu_writes_any.store(true, std::memory_order_release);
    g_pool_moves.fetch_add(1, std::memory_order_relaxed);
}

void pool_scan_counts(uint64_t* full_scans, uint64_t* looks) {
    if (full_scans) *full_scans = g_pool_full_scans;
    if (looks) *looks = g_pool_looks;
}

void pool_texture_remade(Texture* t) {
    if (!t) return;
    const TexGpu* g = t->gpu;   // textures sharing the image hold the same views
    for (auto& pp : g_pool_pairs) {
        TexturePool* tp = pp->tp.get();
        if (!tp) continue;
        if (!g) {   // released (many at a scene change): skip the scan when the pool's record says t is not registered in it
            const PoolRes* r = nullptr;
            for (auto& x : g_pool_res)
                if (x->tp.get() == tp) r = x.get();
            if (r && !r->reg.count(t) && r->version == version_of(tp)) continue;
        }
        std::unique_lock<std::shared_mutex> lk(tp->lock);
        const size_t n = std::min<size_t>(tp->entries.size(), kPoolTextureCount);
        for (size_t id = 0; id < n; ++id) {
            Texture* x = tp->entries[id].used ? tp->entries[id].tex.get() : nullptr;
            if (x && (x == t || (g && x->gpu == g))) pool_note_change(*tp, static_cast<int>(id));
        }
    }
}

void pool_hold(Texture* t) {
    if (!t) return;
    g_held.insert(t);
    g_held_present = counters().presents.load(std::memory_order_relaxed);
}

void pool_textures_to_prepare(TexturePool* tp, std::vector<Texture*>& out) {
    if (!tp) return;
    release_held();
    PoolRes& r = pool_res(tp);
    const uint64_t serial = current_serial();
    r.last_use = serial;
    r.moves = g_pool_moves.load(std::memory_order_relaxed);   // first: a move after this is seen by the next pool_state_moved
    r.watch_serial = serial;
    ++g_pool_looks;
    if (g_cpu_writes_any.exchange(false, std::memory_order_acq_rel)) {   // written by the CPU since the last call: the pools they are registered in look at them
        static std::vector<Texture*> written;
        bool over = false;
        {
            std::lock_guard<std::mutex> lk(g_cpu_writes_mu);
            written.swap(g_cpu_writes);
            over = g_cpu_writes_over;
            g_cpu_writes_over = false;
        }
        for (auto& pr : g_pool_res) {
            if (over) { pr->all = true; pr->watch.clear(); continue; }
            if (pr->all) continue;
            for (Texture* t : written) {
                const auto it = pr->reg.find(t);
                if (it == pr->reg.end()) continue;
                if (pr->watch.size() >= kPoolWatchMax) {
                    pr->all = true;
                    pr->watch.clear();
                    break;
                }
                pr->watch.push_back(it->second.first);
            }
        }
        written.clear();
    }
    for (const Ref<Texture>& t : g_layout_moves)        // moved since the last call: the pools they are registered in look at them
        for (auto& pr : g_pool_res) {
            if (pr->all || !pr->reg.count(t.get())) continue;   // all: every registered texture is looked at next time anyway
            // A pool nothing draws with any more is never drained here: its list would grow with every layout change for good.
            if (pr->watch.size() >= kPoolWatchMax) {
                pr->all = true;
                pr->watch.clear();
                continue;
            }
            pr->watch.push_back(t);
        }
    g_layout_moves.clear();
    {
        std::shared_lock<std::shared_mutex> lk(tp->lock);
        if (r.version != tp->version) {
            const uint64_t n = tp->version - r.version;
            const bool partial = r.version != ~0ull && r.version <= tp->version && n <= tp->changes.size() && r.by_id.size() == tp->entries.size();
            auto set_id = [&](uint32_t id) {
                const TexturePool::Entry& e = tp->entries[id];
                Texture* now = e.used ? e.tex.get() : nullptr;
                Texture*& was = r.by_id[id];
                if (was == now) return;
                if (was) {
                    auto it = r.reg.find(was);
                    if (it != r.reg.end() && --it->second.second == 0) r.reg.erase(it);
                }
                if (now) {
                    auto& slot = r.reg[now];
                    if (!slot.second++) slot.first = Ref<Texture>::share(now);
                    if (partial) r.watch.push_back(slot.first);
                }
                was = now;
            };
            if (partial) {
                for (size_t k = tp->changes.size() - n; k < tp->changes.size(); ++k)
                    if (tp->changes[k] < tp->entries.size()) set_id(tp->changes[k]);
            } else {
                r.by_id.assign(tp->entries.size(), nullptr);
                r.reg.clear();
                for (uint32_t id = 0; id < tp->entries.size(); ++id) set_id(id);
                r.all = true;
            }
            r.version = tp->version;
        }
    }
    // Textures initialized or registered, or written by the CPU, since the last full scan. The CPU writes are in the watch list already
    // (pool_note_cpu_write) and registrations come with the pool's version, so the full scan is only the safety net (a new texture over the
    // storage of a finalized one that is still registered, which pool_image draws through texture_registry_find_alias): at most one a submission.
    // It used to run whenever the epoch moved, which in a crowded town was many times a submission over thousands of textures.
    const uint64_t ep = g_tex_scan_epoch.load(std::memory_order_relaxed);
    if (ep != r.epoch && r.scan_serial != serial) {
        r.all = true;
        r.epoch = ep;
        r.scan_serial = serial;
    }
    if (r.all) {
        ++g_pool_full_scans;
        g_pool_full_scan_gauge.add(1);
    }
    const size_t first = out.size();
    static const bool depth_readiness = config_get_bool("nvn", "depth_pool_readiness", true);
    auto look = [&](Texture* t) {
        if (!g_held.empty() && g_held.count(t)) return;            // held back for this frame (pool_hold)
        if ((t->destroyed.load(std::memory_order_relaxed) && !t->gpu && !texture_registry_find_alias(t)) || !tex_gpu(t)) return;
        const bool pending = depth_readiness ? tex_pool_preparation_pending(t)
            : t->cpu_dirty.load(std::memory_order_relaxed) || tex_layout(t) != tex_sampled_layout(t);
        if (pending) out.push_back(t);
    };
    if (r.all) {
        for (auto& [t, e] : r.reg) look(t);
        r.all = false;
    } else {
        for (const Ref<Texture>& t : r.watch) look(t.get());
    }
    // Each texture once. A texture that stays not ready (an attachment of the open pass, which the replay leaves as it is) is in the
    // watch list and is added again by every layout change it goes through; without this the list, and with it the work of every draw
    // that samples through the pool, grew for as long as the scene kept such a texture (a battle map left idle: 30 fps down to 9 in
    // two minutes, back at once when the picture changed).
    std::sort(out.begin() + static_cast<std::ptrdiff_t>(first), out.end());
    out.erase(std::unique(out.begin() + static_cast<std::ptrdiff_t>(first), out.end()), out.end());
    // not ready now: looked at again next time
    r.watch.clear();
    for (size_t i = first; i < out.size(); ++i) r.watch.push_back(Ref<Texture>::share(out[i]));
}

bool pool_state_moved(TexturePool* tp, SamplerPool* sp) {
    if (release_held()) return true;
    if (!g_layout_moves.empty()) return true;
    PoolRes* r = nullptr;
    for (auto& x : g_pool_res)
        if (x->tp.get() == tp) r = x.get();
    if (tp && !r) return true;
    if (tp) {
        const uint64_t serial = current_serial();
        // a watched texture becomes ready (or preparable) only through what bumps g_pool_moves; once a submission it is looked at anyway,
        // in case a path made it ready without (an upload that left it in its sampled layout needs no preparing, so nothing is lost there)
        if (r->all || r->moves != g_pool_moves.load(std::memory_order_relaxed) || (!r->watch.empty() && r->watch_serial != serial) ||
            (r->epoch != g_tex_scan_epoch.load(std::memory_order_relaxed) && r->scan_serial != serial) || r->version != version_of(tp))
            return true;
    }
    for (auto& p : g_pool_pairs) {
        if (p->tp.get() != tp || p->sp.get() != sp) continue;
        if (p->cur < 0) return true;
        const PoolCopy& c = p->copies[static_cast<size_t>(p->cur)];
        return c.tex_version != version_of(tp) || c.smp_version != version_of(sp) || pending_ready(c, tp);
    }
    return true;
}

void pool_sets_shutdown() {
    VkDevice d = vkdev();
    for (auto& p : g_pool_pairs) {
        if (p->dpool) { vkDestroyDescriptorPool(d, p->dpool, nullptr); vk_gauge_descriptor_pools().sub(1); }
        g_pool_sets.sub(static_cast<int64_t>(p->copies.size()));
    }
    g_pool_pairs.clear();
    g_pool_res.clear();
    g_layout_moves.clear();
    {
        std::lock_guard<std::mutex> wl(g_cpu_writes_mu);
        g_cpu_writes.clear();
        g_cpu_writes_over = false;
    }
    std::lock_guard<std::mutex> lk(g_pool_layout_mu);
    if (g_pool_layout) vkDestroyDescriptorSetLayout(d, g_pool_layout, nullptr);
    g_pool_layout = VK_NULL_HANDLE;
    g_pool_layout_tried = false;
}

size_t pipeline_count() { return static_cast<size_t>(g_pipeline_count.get()); }

void module_cache_clear() {
    VkDevice d = vkdev();
    std::lock_guard<std::recursive_mutex> lk(g_module_mu);
    for (auto& [k, m] : g_modules) if (m) vkDestroyShaderModule(d, m, nullptr);
    g_modules.clear();
    for (auto& [k, m] : g_variants) if (m) vkDestroyShaderModule(d, m, nullptr);
    g_variants.clear();
    for (auto& [k, m] : g_dual_src) if (m) vkDestroyShaderModule(d, m, nullptr);
    g_dual_src.clear();
    for (auto& [k, e] : g_no_out) if (e.first && e.second) vkDestroyShaderModule(d, e.first, nullptr);
    g_no_out.clear();
    for (auto& [k, v] : g_xfb_mod) if (v.first) vkDestroyShaderModule(d, v.first, nullptr);
    g_xfb_mod.clear();
    std::lock_guard<std::mutex> lk2(g_module_id_mu);
    g_module_ids.clear();
    g_module_subgroups.clear();
}

uint64_t module_content_id(VkShaderModule m) {
    std::lock_guard<std::mutex> lk(g_module_id_mu);
    auto it = g_module_ids.find(m);
    return it != g_module_ids.end() ? it->second : reinterpret_cast<uint64_t>(m);   // not made by create_module: the handle stands for itself
}

uint64_t pipelines_built_at_draw() { return g_built_at_draw.load(std::memory_order_relaxed); }

void pipeline_shutdown() {
    VkDevice d = vkdev();
    prewarm_shutdown();
    gpl_stop();
    if (!g_cache_big || g_new_since_save.load() >= 300) save_cache();
    for (auto& [h, bucket] : g_gfx)
        for (auto& e : bucket) {
            if (e.second) vkDestroyPipeline(d, e.second, nullptr);
            if (e.alt) vkDestroyPipeline(d, e.alt, nullptr);
            if (e.up && e.up->state.load() == 1 && e.up->pipe) vkDestroyPipeline(d, e.up->pipe, nullptr);     // built and never taken
        }
    g_gfx.clear();
    g_gfx_keys.sub(g_gfx_keys.get());
    gpl_shutdown();                              // the parts the pipelines were linked from
    for (auto& [k, p] : g_compute)
        if (p) vkDestroyPipeline(d, p, nullptr);
    g_compute.clear();
    for (auto& [k, m] : g_int_samplers)
        if (m) vkDestroyShaderModule(d, m, nullptr);
    g_int_samplers.clear();
    g_compute_keys.sub(g_compute_keys.get());
    g_layouts.clear();
    for (auto& [k, l] : g_shared_layouts) {          // the handles in g_layouts were copies of these
        if (l.layout) vkDestroyPipelineLayout(d, l.layout, nullptr);
        for (int set = 0; set < 4; ++set)            // kPoolSet's is pool_sets_shutdown's
            if (l.sets[set]) vkDestroyDescriptorSetLayout(d, l.sets[set], nullptr);
    }
    g_shared_layouts.clear();
    pool_sets_shutdown();
    for (auto& [k, m] : g_modules)
        if (m) vkDestroyShaderModule(d, m, nullptr);
    g_modules.clear();
    for (auto& [k, m] : g_variants)
        if (m) vkDestroyShaderModule(d, m, nullptr);
    g_variants.clear();
    for (auto& [k, m] : g_dual_src)
        if (m) vkDestroyShaderModule(d, m, nullptr);
    g_dual_src.clear();
    for (auto& [k, e] : g_no_out)
        if (e.first && e.second) vkDestroyShaderModule(d, e.first, nullptr);
    g_no_out.clear();
    for (auto& [k, v] : g_xfb_mod)
        if (v.first) vkDestroyShaderModule(d, v.first, nullptr);
    g_xfb_mod.clear();
    {
        std::lock_guard<std::mutex> lk(g_module_id_mu);
        g_module_ids.clear();
        g_module_subgroups.clear();
    }
    for (auto& [k, b] : g_const_bufs) device().destroy_buffer(b.first);
    g_const_bufs.clear();
    g_const_n.sub(g_const_n.get());
    g_const_bytes.sub(g_const_bytes.get());
    if (g_driver_const_made) device().destroy_buffer(g_driver_const);
    g_driver_const_made = false;
    if (g_cache) vkDestroyPipelineCache(d, g_cache, nullptr);
    g_cache = VK_NULL_HANDLE;
    g_cache_tried = false;
    g_cache_ready.store(false);
    g_cache_dirty.store(false);
    g_pipeline_count.sub(g_pipeline_count.get());
}

// Gauges (nxrt/gauge.h). The callbacks read the layout and module maps under the locks that guard them; they are removed before those maps
// are destroyed.
namespace {
struct PipelineGauges {
    PipelineGauges() {
        nxrt::gauge_register("nvn.pipeline_layouts", [] {
            std::lock_guard<std::mutex> lk(g_layout_mu);
            return static_cast<int64_t>(g_layouts.size());
        });
        nxrt::gauge_register("nvn.shared_layouts", [] {
            std::lock_guard<std::mutex> lk(g_layout_mu);
            return static_cast<int64_t>(g_shared_layouts.size());
        });
        nxrt::gauge_register("nvn.shader_modules", [] {
            std::lock_guard<std::recursive_mutex> lk(g_module_mu);
            return static_cast<int64_t>(g_modules.size());
        });
        nxrt::gauge_register("nvn.shader_modules.variants", [] {
            std::lock_guard<std::recursive_mutex> lk(g_module_mu);
            return static_cast<int64_t>(g_variants.size());
        });
        nxrt::gauge_register("nvn.shader_modules.dual_src", [] {
            std::lock_guard<std::recursive_mutex> lk(g_module_mu);
            return static_cast<int64_t>(g_dual_src.size());
        });
        nxrt::gauge_register("nvn.shader_modules.no_out", [] {
            std::lock_guard<std::recursive_mutex> lk(g_module_mu);
            return static_cast<int64_t>(g_no_out.size());
        });
        nxrt::gauge_register("nvn.shader_modules.xfb", [] {
            std::lock_guard<std::recursive_mutex> lk(g_module_mu);
            return static_cast<int64_t>(g_xfb_mod.size());
        });
    }
    ~PipelineGauges() {
        for (const char* n : {"nvn.pipeline_layouts", "nvn.shared_layouts", "nvn.shader_modules", "nvn.shader_modules.variants",
                              "nvn.shader_modules.dual_src", "nvn.shader_modules.no_out", "nvn.shader_modules.xfb"})
            nxrt::gauge_unregister(n);
    }
};
PipelineGauges g_pipeline_gauges;
}  // namespace

}  // namespace nxrt::nvn::be::gpu
