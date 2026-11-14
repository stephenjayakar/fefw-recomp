// SPDX-License-Identifier: GPL-3.0-or-later
// The command list interpreter of the NVN back end (render thread). Walks recorded lists in order, keeps the GPU state the
// console keeps (it persists across lists and submissions), and turns it into Vulkan: dynamic-rendering passes opened lazily on the
// first draw or clear of a render target set, graphics pipelines keyed by the baked state, descriptor sets built from the
// translated shaders' binding tables (nvn_shaders.h), vertex and index data straight from the imported pool memory.
// Lessons applied from SuperMarioWonderRecomp docs/nvn/DEBUG.md and DEBUG-2.md (see work notes in docs/S4d.md): viewport flip by a
// negative-height viewport, a memory barrier after every pass, textures sampled by a draw are made current before the pass,
// handles never dropped by count, deferred bind arrays read at record time.
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <tuple>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <intrin.h>
#include <bit>

#include "nxrt/config.h"
#include "nxrt/frame_limiter.h"
#include "nxrt/ftrace.h"
#include "nxrt/gauge.h"
#include "nxrt/gfx_tuning.h"
#include "nxrt/log.h"
#include "api_dlss.h"
#include "nvn_decoded.h"
#include "nvn_enums.h"
#include "nvn_framegen.h"
#include "nvn_format.h"
#include "nvn_mask_clear.h"
#include "nvn_gpu_internal.h"
#include "nvn_texprep.h"
#include "nvn_tess_gen.h"
#include "nxrt/lowlat.h"
#include "extras/hdr_nvn.h"
#include "extras/fsr1_spatial.h"
#include "extras/spatial_upscaler.h"
#include "extras/temporal.h"
#include "nvn_cmdq.h"
#include "nvn_pipeline.h"
#include "nvn_shaders.h"

namespace nxrt::nvn::be::gpu {
namespace {

constexpr const char* kTag = "nvn.replay";
constexpr int kMaxDepth = 8;

enum Stage { kVertex = 0, kFragment = 1, kGeometry = 2, kTessCtl = 3, kTessEval = 4, kCompute = 5 };

struct GState {
    const ShaderInfo* stage[6] = {};
    struct Bind { uint64_t addr = 0, size = 0; };
    Bind vtx[16], ubo[6][16], ssbo[6][16], xfb[4];
    // bit k: slot k is bound (addr != 0); kept by BindBuffers, the only op that writes the slots (out_draw_bindings walks only these)
    uint32_t vtx_mask = 0, ubo_mask[6] = {}, ssbo_mask[6] = {}, xfb_mask = 0;
    uint64_t tex[6][64] = {};
    uint64_t img[6][16] = {};
    Ref<TexturePool> tpool;
    Ref<SamplerPool> spool;
    VertexAttribInfo attr[16];
    uint32_t attr_count = 0;
    VertexStreamInfo stream[16];
    uint32_t stream_count = 0;
    BlendInfo blend[8];
    ColorInfo color;
    ChannelMaskInfo chmask;
    DepthStencilInfo ds;
    PolygonInfo poly;
    MultisampleInfo ms;
    float vp[16][4] = {};
    uint32_t nvp = 0;
    int32_t sc[16][4] = {};
    uint32_t nsc = 0;
    float dr[16][2] = {};
    bool depth_bounds = false;
    float db_near = 0, db_far = 1;
    uint32_t stencil_ref[2] = {0, 0}, stencil_vmask[2] = {0xff, 0xff}, stencil_wmask[2] = {0xff, 0xff};
    float blend_color[4] = {};
    float line_width = 1, point_size = 1, alpha_ref = 0;
    float bias_factor = 0, bias_units = 0, bias_clamp = 0;
    uint32_t sample_mask = 0xffffffff;
    bool rasterizer_discard = false, depth_clamp = false, prim_restart = false, render_enable = true;
    uint32_t restart_index = 0;
    // tessellation: nvnCommandBufferSetPatchSize (control points per patch; GL default 3) and the fixed levels of
    // nvnCommandBufferSetInner/OuterTessellationLevels (used by NVN only when the program has no control shader)
    uint32_t patch_size = 3;
    float tess_outer[4] = {1, 1, 1, 1}, tess_inner[2] = {1, 1};
    // render targets
    uint32_t ncolor = 0;
    Ref<Texture> color_tex[8];
    TextureViewInfo color_view[8];
    bool color_has_view[8] = {};
    uint32_t color_level[8] = {}, color_layer[8] = {}, color_layers[8] = {};   // level, first layer, layer count each colour target is attached with (trace)
    Ref<Texture> depth_tex;
    TextureViewInfo depth_view;
    bool depth_has_view = false;
    int window_origin = NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT;
    int depth_mode = NVN_DEPTH_MODE_NEAR_IS_MINUS_W;
    // NVN2 push constants (Op::PushConstants): graphics slot n is constant bank 24 + n; compute slot n sits in c[0] at 0x10 + 0x100 * n
    uint8_t push_gfx[8][256] = {};
    uint8_t push_cmp[8 * 256] = {};
};

GState S;
UploadAlloc g_push_buf[8];          // this submission's copy of each graphics push constant slot (made when a draw binds it)
UploadAlloc g_cs_c0;                // the c[0] of the dispatch being recorded (NVN2 compute: work-group counts, push constants)

struct Target {
    Texture* tex = nullptr;
    uint32_t level = 0, layer = 0, layers = 1;
    VkFormat view_format = VK_FORMAT_UNDEFINED;
    bool operator==(const Target& o) const { return tex == o.tex && level == o.level && layer == o.layer && layers == o.layers && view_format == o.view_format; }
};

struct Pass {
    bool open = false;
    uint32_t ncolor = 0;
    Target color[8];
    Target depth;
    VkExtent2D extent = {0, 0};
    uint32_t layers = 1;         // common layer range attached to the active rendering pass
    VkFormat color_fmt[8] = {};
    VkFormat depth_fmt = VK_FORMAT_UNDEFINED;
    bool has_stencil = false;
    bool depth_ro = false;      // the depth attachment is in DEPTH_STENCIL_READ_ONLY_OPTIMAL (it is also sampled by the draws of this pass)
    uint32_t samples = 1;
    float scale = 1.0f;         // resolution scale of the attachments ([gfx] render_scale): viewports and scissors are multiplied by it
};

Pass P;
// [nvn] pass_barriers = tracked (S38). A pass whose draws can only have written its attachments (no storage buffers or images, no stream-out)
// does not end with a barrier: its written attachments are remembered here, and the next command that could depend on them gets one. A
// layout change of such an image is that barrier for the image (narrow masks, set_narrow_barriers); a later pass that renders to it again in
// the same layout, and every command other than a pass or a layout change, get a full barrier first (end_pass).
struct PendingWrite { Ref<Texture> tex; VkImageLayout layout; };
std::vector<PendingWrite> g_pending;
bool g_tracked = false;         // this submission uses tracked barriers
bool g_pass_unsafe = false;     // a draw of the open pass may write memory other than its attachments
bool g_prog_writes = false;     // the bound program has storage buffer or storage image bindings
bool g_mirror_stale = false;    // a GPU write made pages of a pool copy stale while a draw was being set up: bind again at the next draw

uint32_t g_pass_draws = 0;      // draws recorded in the open pass (GPU pass profiler)
uint64_t g_pass_verts = 0;      // vertices (indices for indexed draws) times instances of those draws
Frame* F = nullptr;
// This submission's descriptor sets by content key: open addressing, emptied by moving the generation (an unordered_map's node per set
// and its lookups were ~1.7 % of the render thread; rt agent, notes/rt.md). Grows when half full.
struct SetCache {
    struct Slot { uint64_t key = 0, gen = 0; VkDescriptorSet ds = VK_NULL_HANDLE; };
    std::vector<Slot> slots = std::vector<Slot>(8192);
    uint64_t gen = 1;
    size_t used = 0;
    static size_t mix(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33; return static_cast<size_t>(k); }
    VkDescriptorSet find(uint64_t key) const {
        const size_t m = slots.size() - 1;
        for (size_t i = mix(key) & m;; i = (i + 1) & m) {
            const Slot& sl = slots[i];
            if (sl.gen != gen) return VK_NULL_HANDLE;
            if (sl.key == key) return sl.ds;
        }
    }
    void put(uint64_t key, VkDescriptorSet ds) {
        if ((used + 1) * 2 > slots.size()) {
            std::vector<Slot> old;
            old.swap(slots);
            slots.assign(old.size() * 2, Slot());
            used = 0;
            for (const Slot& sl : old) if (sl.gen == gen) put(sl.key, sl.ds);
        }
        const size_t m = slots.size() - 1;
        for (size_t i = mix(key) & m;; i = (i + 1) & m) {
            Slot& sl = slots[i];
            if (sl.gen != gen) { sl = {key, gen, ds}; ++used; return; }
            if (sl.key == key) { sl.ds = ds; return; }
        }
    }
    void clear() { ++gen; used = 0; }
};
SetCache g_set_cache;
VkPipeline g_bound_pipeline = VK_NULL_HANDLE;
UploadAlloc g_support;
UploadAlloc g_draw_support;     // the support buffer the next descriptor sets bind: g_support, or one with render scales (support_for_draw)
std::unordered_map<uint64_t, UploadAlloc> g_support_scaled;   // this submission's support buffers with render scales, by content
MaskClear g_mask;
bool g_mask_ready = false, g_mask_ok = false;

// What changed since the last draw. Ops set bits (mark_dirty); a draw handles and clears the groups it needs. A new submission, a
// new pass and anything that binds other pipelines sets them all (a new command buffer has no dynamic state, a new pass has other
// formats and extents).
enum : uint32_t {
    // kDescBuf: a buffer binding (or push constants) changed; kDescTex: a texture, image or pool binding changed. kDesc = either (every
    // other cause marks both). Only bind_sets tells them apart (texture sets left bound, [nvn] tex_set_skip).
    kPipe = 1, kDescBuf = 2, kVb = 4, kDynVp = 8, kDynRaster = 16, kDynDepth = 32, kDynStencil = 64, kDynBias = 128, kDynBlend = 256, kDescTex = 512,
    kDesc = kDescBuf | kDescTex, kAllDirty = 0x3ff
};
uint32_t g_dirty = kAllDirty;
// A command may write [addr, addr + size) of pool memory: device-local copies of those pages are stale, and bindings that point at them are
// made again (vertex buffers and descriptor sets).
// The ranges of pool memory the GPU writes, with the serial of the submission that did (a ring). The replay records commands: a range written
// by a submission that has not completed yet does not hold its result in game memory, so the CPU must not read it for a decision that needs
// the data (the arguments of an indirect draw it converts, cpu_view). A range that fell out of the ring while its submission was still running
// makes every range unreadable until that submission is done.
struct GpuWrite { uint64_t lo, hi, serial; };
constexpr size_t kGpuWriteRing = 1024;
GpuWrite g_gw[kGpuWriteRing];
size_t g_gw_next = 0, g_gw_count = 0;
uint64_t g_gw_lost_serial = 0;      // the newest serial among ranges dropped from the ring
void note_gw(uint64_t addr, uint64_t size) {
    if (!size || !F) return;
    GpuWrite& last = g_gw[(g_gw_next + kGpuWriteRing - 1) % kGpuWriteRing];
    if (g_gw_count && last.lo == addr && last.hi == addr + size) { last.serial = F->serial; return; }
    GpuWrite& w = g_gw[g_gw_next];
    if (g_gw_count == kGpuWriteRing) g_gw_lost_serial = std::max(g_gw_lost_serial, w.serial);
    else ++g_gw_count;
    w = {addr, addr + size, F->serial};
    g_gw_next = (g_gw_next + 1) % kGpuWriteRing;
}
// True when no submission still in flight (the one being recorded included) may write [addr, addr + size).
bool gw_settled(uint64_t addr, uint64_t size) {
    const uint64_t done = poll_completed();
    if (g_gw_lost_serial > done) return false;
    for (size_t i = 0; i < g_gw_count; ++i)
        if (g_gw[i].serial > done && addr < g_gw[i].hi && g_gw[i].lo < addr + size) return false;
    return true;
}

void gpu_wrote(uint64_t addr, uint64_t size, int use) {
    note_gw(addr, size);
    if (note_gpu_write(addr, size, use)) {
        g_dirty |= kVb | kDesc;
        g_mirror_stale = true;
    }
}
const ProgLayout* g_pl = nullptr;
VkPipeline g_pipe = VK_NULL_HANDLE;
VkPrimitiveTopology g_topo = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
VkPrimitiveTopology g_pipe_topo = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
bool g_restart = false, g_restart_valid = false;
uint32_t g_key_attr_count = 0;
uint8_t g_key_binding[16] = {};
// Occlusion counter (NVN_COUNTER_TYPE_SAMPLES_PASSED): ResetCounter starts a Vulkan occlusion query outside any pass; it stays active across
// the passes that follow (the console counts continuously); ReportCounter ends it and copies its 64-bit result into the report.
constexpr uint32_t kOccSlots = 256;
VkQueryPool g_occ_pool = VK_NULL_HANDLE;
bool g_occ_tried = false;
uint32_t g_occ_next = 0;
int32_t g_occ_active = -1, g_occ_last = -1;

bool occ_ready() {
    if (g_occ_pool) return true;
    if (g_occ_tried || !device().features().occlusion_query) return false;
    g_occ_tried = true;
    VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_OCCLUSION;
    qi.queryCount = kOccSlots;
    if (vkCreateQueryPool(vkdev(), &qi, nullptr, &g_occ_pool) != VK_SUCCESS) g_occ_pool = VK_NULL_HANDLE;
    return g_occ_pool != VK_NULL_HANDLE;
}

uint8_t g_key_dummy_binding = 0;      // 1 + the binding that feeds the defaults of unset vertex attributes (0: none)
bool g_want_depth_ro = false;         // some texture this draw samples is the current depth attachment
bool g_vb_bound_in_cb = false;        // vkCmdBindVertexBuffers2 was called in this command buffer (dynamic strides need one)

// [nvn] trace_frame = N (NXRT_NVN_TRACE_FRAME): every draw, clear, copy and dispatch of the frame that is presented as number N is logged
// at Info ("trace ..." lines): programs, render targets, sampled textures, blend and depth state. For finding out what a frame does.
bool g_xfb_active = false;
bool g_pipe_xfb = false;             // the bound pipeline is a stream-out variant
bool g_xfb_began = false;            // vkCmdBeginTransformFeedbackEXT is in effect in the open pass
bool g_xfb_have_counter = false;     // a counter buffer holds the bytes written so far: the next begin continues after them
gfx::Buffer g_xfb_counter;
// One byte counter per feedback control buffer (NVN's Begin/End/Draw take its address): slot * 16 bytes into g_xfb_counter. vkCmdEndTransformFeedbackEXT
// writes the bytes of the section there; DrawTransformFeedback reads them (vkCmdDrawIndirectByteCountEXT). written: the slot holds a count.
constexpr uint32_t kXfbSlots = 256, kXfbSlotBytes = 16;
std::unordered_map<uint64_t, uint32_t> g_xfb_slot_of;
uint32_t g_xfb_slot_next = 0, g_xfb_cur = 0;
bool g_xfb_slot_written[kXfbSlots] = {};
// Windowed GPU counters for the primitive counters (13 = primitives generated, 14 = feedback primitives written): a ResetCounter whose window
// (up to the matching ReportCounter, in the same list) holds an indirect or feedback draw starts a query the GPU counts those draws with;
// the CPU counts the other draws exactly (count_primitives) but cannot know the primitives of a draw whose arguments the GPU holds.
// [0] pipeline statistics (input assembly primitives; a converted quad counts as its two triangles), [1] transform feedback stream 0 query.
struct CounterWindow { VkQueryPool pool = VK_NULL_HANDLE; bool tried = false; uint32_t next = 0; int32_t active = -1, last = -1; };
constexpr uint32_t kCounterSlots = 64;
CounterWindow g_cw[2];
int g_dbg_depth = 0;                  // debug labels open in this submission (PushDebugGroup / PopDebugGroup)
// Primitive counters (NVN_COUNTER_TYPE_PRIMITIVES_GENERATED = 13, TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN = 14): counted on the CPU by every
// draw (count_primitives), process-wide like the console's, zeroed by ResetCounter of their type, dropped at shutdown.
uint64_t g_prims_generated = 0, g_prims_xfb = 0;
// Conditional render (SetRenderEnableConditional): the console compares the 64-bit counter words of two reports, at the address and
// 16 bytes after it (yuzu Maxwell3D::ProcessQueryCondition: IfEqual / IfNotEqual), when the command runs. The replay runs in GPU
// order, so the counters it writes itself (ReportCounter other than samples passed, ReportValue) are known here: g_reports keeps
// them by address. kReportUnknown marks an address the GPU writes later (samples passed): a condition on it renders, as before. An
// address the replay never wrote (the game's own memory) is read from the pool.
constexpr uint64_t kReportUnknown = ~0ull;
std::unordered_map<uint64_t, uint64_t> g_reports;
nxrt::GaugeCounter g_gauge_pipeline_binds("nvn.cnt.pipeline_binds");   // graphics pipeline binds (cumulative)
VkQueryPool g_xq_pool = VK_NULL_HANDLE;     // transform feedback stream queries (counters 13 and 14)
bool g_trace = false;
struct TraceSnap { int index; int draw; std::string name; std::string desc; };
std::vector<TraceSnap> g_trace_snaps;
std::vector<int> g_trace_snap_draws;   // [nvn] trace_snap: draw numbers after which the targets are copied
bool g_trace_snap_read = false;
std::vector<std::string> g_trace_snap_fs;
bool g_trace_snap_sampled = false;           // [nvn] trace_snap_sampled = 1: a snapshot draw also copies the textures that draw samples
std::vector<Texture*> g_trace_sampled;       // textures sampled by the last traced draw
std::vector<Ref<Texture>> g_trace_targets;   // the colour and depth targets drawn to in the traced frame, in order of first use
int64_t g_trace_frame = -2;
std::vector<int64_t> g_trace_extra;   // more frames to trace
// [nvn] trace_after_fs = "67d504": the frame after the first one that draws with a fragment shader of this hash prefix is traced
// (a frame number is not stable from run to run: the load times differ).
std::string g_trigger_fs;

// Debug: [nvn] skip_fs_file = FILE (env NXRT_NVN_SKIP_FS_FILE). FILE holds fragment shader hash prefixes separated by blanks; draws whose fragment
// shader starts with one of them are not drawn. The file is read again twice a second, so it can change while the game runs.
std::vector<std::string> g_skip_fs;
int g_draw_idx = 0;   // draws begun in this frame
bool skip_fs_draw(const std::string& hash) {
    static const std::string file = nxrt::config_get_str("nvn", "skip_fs_file", "");
    if (file.empty()) return false;
    static std::chrono::steady_clock::time_point last{};
    const auto now = std::chrono::steady_clock::now();
    if (last.time_since_epoch().count() == 0 || now - last > std::chrono::milliseconds(500)) {
        last = now;
        g_skip_fs.clear();
        std::ifstream f(file);
        std::string w;
        while (f >> w) g_skip_fs.push_back(w);
    }
    for (const std::string& p : g_skip_fs) {
        if (!p.empty() && p[0] == '#') {   // "#198" or "#198-210": the draw numbers of the frame (the first draw is #1, as in the trace)
            const int a = std::atoi(p.c_str() + 1);
            const size_t dash = p.find('-');
            const int b = dash == std::string::npos ? a : std::atoi(p.c_str() + dash + 1);
            if (g_draw_idx >= a && g_draw_idx <= b) return true;
        } else if (hash.compare(0, p.size(), p) == 0) return true;
    }
    return false;
}
bool g_trigger_read = false, g_trigger_seen = false, g_trigger_done = false;
int64_t g_trigger_present = -1;
int g_trace_n = 0;
std::atomic<bool> g_trace_req{false};   // backend_trace_next_frame(): trace the first whole frame after this flag is seen
#define TRACE(...) do { if (g_trace) NXLOG_I("nvn.trace", __VA_ARGS__); } while (0)

// ---- [nvn] replay_stats = N (debug, default 0 = off): the replay's cost per draw (notes/rt.md) ---------------------------------
// The phases of begin_draw, the draw call itself, out_before_op, dispatches and the whole replay are timed with the TSC; every N presents
// one log line (nvn.rstats) gives the averages per frame and per draw, with the counts behind them. Render thread only.
enum RsZone : int { kRsHead, kRsPrep, kRsFb, kRsPass, kRsPipe, kRsDyn, kRsSets, kRsVb, kRsCall, kRsOut, kRsDispatch, kRsAll, kRsAlloc, kRsN };
struct RStats {
    uint64_t cyc[kRsN] = {};
    uint64_t draws = 0, draws_desc = 0, draws_pipe = 0, sets = 0, set_hits = 0, set_writes = 0, res_tex = 0, res_img = 0, res_memo = 0;
    uint64_t binds = 0, binds_same = 0, layouts = 0, keys = 0, dispatches = 0, sets_skipped = 0, preps_skipped = 0;
    uint64_t miss_first = 0, miss_uboofs = 0, miss_ubo = 0, miss_tex = 0, miss_other = 0;   // set cache misses by what differs from the last set of that layout
};
RStats g_rs;
uint64_t g_rs_t = 0;   // TSC at the last phase mark
// [nvn] rt_check = 1 (debug, default 0): every memo answer of the rt changes (resolve_memo, out_memo; pool_cache in nvn_objects.cpp, the
// view and sampler caches in nvn_gpu_texture.cpp) is compared with the answer computed without it. A mismatch is logged (nvn.rtcheck,
// first 20); with replay_stats the stats line counts the checks. No mismatch = the same Vulkan commands as without the memos.
bool rt_check() {
    static const bool on = nxrt::config_get_bool("nvn", "rt_check", false);
    return on;
}
uint64_t g_chk_n = 0, g_chk_bad = 0;
void chk(bool ok, const char* what) {
    ++g_chk_n;
    if (!ok && g_chk_bad++ < 20) NXLOG_W("nvn.rtcheck", "mismatch: %s", what);
}
int64_t rs_every() {
    static const int64_t n = nxrt::config_get_int("nvn", "replay_stats", 0);
    return n;
}
inline bool rs_on() { return rs_every() > 0; }
inline void rs_start() { if (rs_on()) g_rs_t = __rdtsc(); }
inline void rs_mark(int zone) {
    if (!rs_on()) return;
    const uint64_t t = __rdtsc();
    g_rs.cyc[zone] += t - g_rs_t;
    g_rs_t = t;
}
struct RsScope {
    int zone; uint64_t t0;
    explicit RsScope(int z) : zone(z), t0(rs_on() ? __rdtsc() : 0) {}
    ~RsScope() { if (t0) g_rs.cyc[zone] += __rdtsc() - t0; }
};
void rs_report() {
    using clk = std::chrono::steady_clock;
    static uint64_t last_p = 0, last_tsc = 0;
    static clk::time_point last_t{};
    const int64_t every = rs_every();
    if (every <= 0) return;
    const uint64_t p = counters().presents.load(std::memory_order_relaxed);
    static uint64_t last_scans = 0, last_looks = 0;      // pool_textures_to_prepare: full scans of every registered texture, and all calls
    uint64_t scans = 0, looks = 0;
    pool_scan_counts(&scans, &looks);
    static uint64_t last_sv_h = 0, last_sv_m = 0;
    uint64_t sv_h = 0, sv_m = 0, sv_bad = 0;
    servable_memo_stats(&sv_h, &sv_m, &sv_bad);
    if (!last_tsc) { last_p = p; last_tsc = __rdtsc(); last_t = clk::now(); g_rs = RStats(); last_scans = scans; last_looks = looks; return; }
    if (p < last_p + static_cast<uint64_t>(every)) return;
    const uint64_t tsc = __rdtsc();
    const clk::time_point now = clk::now();
    const double sec = std::chrono::duration<double>(now - last_t).count();
    const double us_per_cyc = sec > 0 ? sec * 1e6 / static_cast<double>(tsc - last_tsc) : 0.0;
    const double frames = static_cast<double>(p - last_p);
    const RStats& s = g_rs;
    auto fms = [&](int z) { return static_cast<double>(s.cyc[z]) * us_per_cyc / 1000.0 / frames; };     // ms per frame
    auto dus = [&](int z) { return s.draws ? static_cast<double>(s.cyc[z]) * us_per_cyc / static_cast<double>(s.draws) : 0.0; };   // us per draw
    double setup = 0;
    for (int z = kRsHead; z <= kRsVb; ++z) setup += fms(z);
    const double pf = 1.0 / frames;
    NXLOG_I("nvn.rstats", "%.0f frames: replay %.2f ms/frame = draw setup %.2f + call %.2f + out %.2f + dispatch %.2f + rest %.2f | %.0f draws/frame "
            "(%.0f desc, %.0f prep left out, %.0f pipe, %.0f layouts, %.0f keys), %.0f dispatches | per draw us: head %.2f prep %.2f fb %.2f pass %.2f pipe %.2f dyn %.2f "
            "sets %.2f (alloc+update %.2f) vb %.2f call %.2f out %.2f | sets/frame %.0f (%.1f%% cached, %.0f writes, %.0f left bound) | resolves/frame tex %.0f img %.0f (memo %.0f) | "
            "binds/frame %.0f (%.1f%% same) | set misses/frame: new layout or submission %.0f, only uniform offsets %.0f, only uniform buffers %.0f, "
            "textures %.0f, other %.0f | pool looks/frame %.1f, full scans/frame %.2f | servable memo/frame hits %.0f misses %.0f (%llu mismatches) | "
            "rt_check %llu checked, %llu mismatches",
            frames, fms(kRsAll), setup, fms(kRsCall), fms(kRsOut), fms(kRsDispatch),
            fms(kRsAll) - setup - fms(kRsCall) - fms(kRsOut) - fms(kRsDispatch), s.draws * pf, s.draws_desc * pf, s.preps_skipped * pf, s.draws_pipe * pf, s.layouts * pf,
            s.keys * pf, s.dispatches * pf, dus(kRsHead), dus(kRsPrep), dus(kRsFb), dus(kRsPass), dus(kRsPipe), dus(kRsDyn), dus(kRsSets), dus(kRsAlloc), dus(kRsVb),
            dus(kRsCall), dus(kRsOut), s.sets * pf, s.sets ? 100.0 * static_cast<double>(s.set_hits) / static_cast<double>(s.sets) : 0.0,
            s.set_writes * pf, s.sets_skipped * pf, s.res_tex * pf, s.res_img * pf, s.res_memo * pf, s.binds * pf,
            s.binds ? 100.0 * static_cast<double>(s.binds_same) / static_cast<double>(s.binds) : 0.0, s.miss_first * pf, s.miss_uboofs * pf,
            s.miss_ubo * pf, s.miss_tex * pf, s.miss_other * pf, static_cast<double>(looks - last_looks) * pf, static_cast<double>(scans - last_scans) * pf,
            static_cast<double>(sv_h - last_sv_h) * pf, static_cast<double>(sv_m - last_sv_m) * pf, static_cast<unsigned long long>(sv_bad),
            static_cast<unsigned long long>(g_chk_n), static_cast<unsigned long long>(g_chk_bad));
    last_scans = scans;
    last_looks = looks;
    last_sv_h = sv_h;
    last_sv_m = sv_m;
    last_p = p;
    last_tsc = tsc;
    last_t = now;
    g_rs = RStats();
}

const char* tex_desc(const Texture* t, char* buf, size_t n) {
    if (!t) { std::snprintf(buf, n, "-"); return buf; }
    std::snprintf(buf, n, "t%u:%s %dx%dx%d L%u/%u%s%s", t->id, t->fmt ? t->fmt->name : "?", t->b.width, t->b.height, t->b.depth, t->levels, t->layers,
                  t->b.samples > 1 ? " ms" : "", t->b.target == NVN_TEXTURE_TARGET_3D ? " 3D" : "");
    return buf;
}

// Percentage of the texture's storage (in the game's memory, first 16 MiB) that holds a non-zero 8-byte word; -1 without storage. Trace only.
int storage_nonzero_pct(const Texture* t) {
    if (!t || !t->mem || !t->storage_size) return -1;
    const uint64_t n = std::min<uint64_t>(t->storage_size, 16ull << 20) / 8;
    if (!n) return -1;
    uint64_t nz = 0;
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t w;
        std::memcpy(&w, t->mem + i * 8, 8);
        nz += w != 0;
    }
    return static_cast<int>(nz * 100 / n);
}

// One warning per call site: a flag of its own, where a map keyed by a std::string built and hashed the key on every call (the alpha test
// one ran at every pipeline key build of a crowded scene, #79). Every key was a distinct literal, so nothing is merged that was not before.
#define NVN_LOG_ONCE(key, msg) do { static bool once_ = false; if (!once_) { once_ = true; NXLOG_W(kTag, "%s", (msg)); } } while (0)

bool first_time_key(const std::string& key) {   // render thread only
    static std::unordered_map<std::string, bool> seen;
    return seen.size() < 4096 && seen.emplace(key, true).second;
}

// first_time_key for the paths a draw can take every time (a shader binding that keeps getting the dummy, a 2D texture read as texels):
// the key is a hash of what identifies the case (site, shader, binding, texture), so the draw builds no string. Render thread only.
bool first_time_id(uint64_t site, const void* shader, uint64_t a, uint64_t b) {
    static std::unordered_set<uint64_t> seen;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t v : {site, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(shader)), a, b}) h = (h ^ v) * 1099511628211ull;
    return seen.size() < 4096 && seen.insert(h).second;
}

// ---- helpers ----------------------------------------------------------------------------------------------------------------
VkFormat scaled_for(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8_UINT: return VK_FORMAT_R8_USCALED;
        case VK_FORMAT_R8G8_UINT: return VK_FORMAT_R8G8_USCALED;
        case VK_FORMAT_R8G8B8_UINT: return VK_FORMAT_R8G8B8_USCALED;
        case VK_FORMAT_R8G8B8A8_UINT: return VK_FORMAT_R8G8B8A8_USCALED;
        case VK_FORMAT_R16_UINT: return VK_FORMAT_R16_USCALED;
        case VK_FORMAT_R16G16_UINT: return VK_FORMAT_R16G16_USCALED;
        case VK_FORMAT_R16G16B16_UINT: return VK_FORMAT_R16G16B16_USCALED;
        case VK_FORMAT_R16G16B16A16_UINT: return VK_FORMAT_R16G16B16A16_USCALED;
        case VK_FORMAT_A2B10G10R10_UINT_PACK32: return VK_FORMAT_A2B10G10R10_USCALED_PACK32;
        case VK_FORMAT_R8_SINT: return VK_FORMAT_R8_SSCALED;
        case VK_FORMAT_R8G8_SINT: return VK_FORMAT_R8G8_SSCALED;
        case VK_FORMAT_R8G8B8_SINT: return VK_FORMAT_R8G8B8_SSCALED;
        case VK_FORMAT_R8G8B8A8_SINT: return VK_FORMAT_R8G8B8A8_SSCALED;
        case VK_FORMAT_R16_SINT: return VK_FORMAT_R16_SSCALED;
        case VK_FORMAT_R16G16_SINT: return VK_FORMAT_R16G16_SSCALED;
        case VK_FORMAT_R16G16B16_SINT: return VK_FORMAT_R16G16B16_SSCALED;
        case VK_FORMAT_R16G16B16A16_SINT: return VK_FORMAT_R16G16B16A16_SSCALED;
        case VK_FORMAT_A2B10G10R10_SINT_PACK32: return VK_FORMAT_A2B10G10R10_SSCALED_PACK32;
        default: return f;
    }
}

bool is_int_format(VkFormat f) { return scaled_for(f) != f; }

VkImageViewType view_type_for(const DescriptorBinding& b) {
    const ImageShape& s = b.image;
    switch (s.dim) {
        case 0: return s.arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
        case 1: return s.arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
        case 2: return VK_IMAGE_VIEW_TYPE_3D;
        case 3: return s.arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
        default: return VK_IMAGE_VIEW_TYPE_2D;
    }
}

// ---- passes ----------------------------------------------------------------------------------------------------------------
// Ends transform feedback in the open pass, remembering the bytes written in the counter buffer (a pass break must not restart the buffer).
void xfb_end_in_pass() {
    if (!g_xfb_began) return;
    VkBuffer cbuf = g_xfb_counter.buffer;
    const VkDeviceSize off = static_cast<VkDeviceSize>(g_xfb_cur) * kXfbSlotBytes;
    vkCmdEndTransformFeedbackEXT(F->cb, 0, 1, &cbuf, &off);
    g_xfb_began = false;
    g_xfb_have_counter = true;
    g_xfb_slot_written[g_xfb_cur] = true;
}

// GPU pass profiler: closes the interval since the previous stamp under this label (no-op unless [prof] gpu_passes).
void gstamp(const char* label) {
    if (gpass_on()) gpass_stamp(*F, label, 0);
}

// Two texture objects over one image (as shares_image further down, which this code cannot see yet).
bool same_image(Texture* a, Texture* b) { return a == b || (a && b && a->gpu && a->gpu == b->gpu); }

// to_pass: the caller only starts another pass or changes image layouts next (ensure_pass, prepare_textures); with tracked barriers the pass
// then ends without a barrier when it only wrote its attachments. Any other caller gets a full barrier, also for passes ended earlier
// without one.
void end_pass(bool to_pass = false) {
    if (!P.open) {
        if (!to_pass && !g_pending.empty()) {
            memory_barrier(F->cb);
            g_pending.clear();
        }
        return;
    }
    xfb_end_in_pass();
    vkCmdEndRendering(F->cb);
    P.open = false;
    if (gpass_on()) {
        char b[112];
        std::snprintf(b, sizeof b, "pass %ux%u c%u:%d d%d%s", P.extent.width, P.extent.height, P.ncolor, static_cast<int>(P.color_fmt[0]), static_cast<int>(P.depth_fmt),
                      P.depth_ro ? " ro" : "");
        gpass_stamp(*F, b, g_pass_draws);
        char vb[128];
        std::snprintf(vb, sizeof vb, "gv:%s", b);
        ft::mark(vb, static_cast<uint32_t>(std::min<uint64_t>(g_pass_verts, 0xffffffffull)));
    }
    if (to_pass && g_tracked && !g_pass_unsafe) {
        auto remember = [](Texture* t, VkImageLayout l) {
            if (!t || !tex_gpu(t)) return;
            for (const PendingWrite& w : g_pending) if (w.tex.get() == t) return;
            g_pending.push_back({Ref<Texture>::share(t), l});
        };
        for (uint32_t i = 0; i < P.ncolor; ++i) remember(P.color[i].tex, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        if (!P.depth_ro) remember(P.depth.tex, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    } else {
        memory_barrier(F->cb);
        g_pending.clear();
    }
    g_pass_unsafe = false;
    g_bound_pipeline = VK_NULL_HANDLE;
    g_dirty = kAllDirty;
}

// A pass is about to use `t` as an attachment in `layout`. An earlier pass that ended without a barrier wrote it, and no layout change
// (which would carry the dependency) happens in between: the two passes need a barrier.
bool pending_conflict(Texture* t, VkImageLayout layout) {
    if (!t || g_pending.empty()) return false;
    for (const PendingWrite& w : g_pending)
        if (same_image(w.tex.get(), t) && tex_layout(t) == w.layout && layout == w.layout) return true;
    return false;
}

Target make_target(Texture* t, const TextureViewInfo& v, bool has_view) {
    Target tg;
    tg.tex = t;
    if (has_view) {
        if (v.has_levels) tg.level = static_cast<uint32_t>(v.min_level);
        if (v.has_layers) {
            tg.layer = static_cast<uint32_t>(v.min_layer);
            // A zero layer count spans the remaining texture layers, as for sampled views.
            tg.layers = v.num_layers == 0 && t && tg.layer < t->layers
                ? t->layers - tg.layer : std::max(1, v.num_layers);
        }
        else if (t && t->layers > 1 && t->b.target != NVN_TEXTURE_TARGET_3D) tg.layers = t->layers;   // a view without layers
        if (v.has_format) {
            const FormatInfo* fi = format_info(v.format);
            if (fi && fi->vk != VK_FORMAT_UNDEFINED) tg.view_format = fi->vk;
        }
    } else if (t && t->layers > 1 && t->b.target != NVN_TEXTURE_TARGET_3D) {
        // no view: an array or cube map target is attached with all its layers (layered rendering: a geometry shader picks the layer with
        // gl_Layer; without it the shaders draw to layer 0)
        tg.layers = t->layers;
    }
    return tg;
}

bool targets_match() {
    if (P.ncolor != S.ncolor) return false;
    for (uint32_t i = 0; i < S.ncolor; ++i)
        if (!(P.color[i] == make_target(S.color_tex[i].get(), S.color_view[i], S.color_has_view[i]))) return false;
    return P.depth == make_target(S.depth_tex.get(), S.depth_view, S.depth_has_view);
}

// The current render targets get their images (scaled when they qualify, tex_gpu_attachment). With `fix` (no pass open), as Ryujinx does
// when it binds render targets: a target whose image was made unscaled (first used another way) is scaled now, and the targets of one
// pass are brought to one scale (the lowest when one of them cannot be scaled). Returns the scale of the targets.
float attach_targets(bool fix) {
    Texture* ts[9];
    uint32_t n = 0;
    for (uint32_t i = 0; i < S.ncolor && i < 8; ++i) if (S.color_tex[i]) ts[n++] = S.color_tex[i].get();
    if (S.depth_tex) ts[n++] = S.depth_tex.get();
    float lo = 0.0f, hi = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        if (!tex_gpu_attachment(ts[i])) continue;
        if (fix) {
            const float want = tex_scale_wanted(ts[i]);
            if (want != tex_scale(ts[i])) tex_rescale(ts[i], *F, want, "used as a render target");
        }
        const float sc = tex_scale(ts[i]);
        lo = lo == 0.0f ? sc : std::min(lo, sc);
        hi = std::max(hi, sc);
    }
    if (lo == 0.0f) return 1.0f;
    if (lo != hi && fix) {
        for (uint32_t i = 0; i < n; ++i)
            if (tex_gpu(ts[i]) && tex_scale(ts[i]) != lo) tex_rescale(ts[i], *F, lo, "a render target of the same pass cannot be scaled");
    }
    return lo;
}

// The area the viewports cover, at most the device's framebuffer size.
VkExtent2D viewport_extent() {
    float w = 0.0f, h = 0.0f;
    for (uint32_t i = 0; i < S.nvp; ++i) {
        w = std::max(w, S.vp[i][0] + std::fabs(S.vp[i][2]));
        h = std::max(h, S.vp[i][1] + std::fabs(S.vp[i][3]));
    }
    const VkPhysicalDeviceLimits& lim = device().properties().limits;
    auto px = [](float v, uint32_t max) { return v > 0.0f ? static_cast<uint32_t>(std::min(std::ceil(v), static_cast<float>(max))) : 0u; };
    return {px(w, lim.maxFramebufferWidth), px(h, lim.maxFramebufferHeight)};
}

// [nvn] split_submit_ms = N (default 0 = off): once N ms of a submission's recording have passed (and some draws were recorded) since it began
// or was last split, the commands so far go to the GPU at the next render pass boundary (split_submission), so the GPU starts on a long
// submission while the render thread still records the rest: the GPU's work of a frame ends sooner after the game submitted it, and the
// game waits less on its fences of two frames back. Each split costs one vkQueueSubmit2. Render thread.
uint64_t g_split_t0 = 0;
uint32_t g_split_draws = 0;
uint64_t split_ticks() {
    static const uint64_t t = [] {
        const double ms = nxrt::config_get_double("nvn", "split_submit_ms", 0.0);
        if (ms > 0) NXLOG_I(kTag, "split_submit_ms %.2f: long submissions go to the GPU in parts at pass boundaries", ms);
        return ms > 0 ? static_cast<uint64_t>(ms * 1e6) : 0;     // ns
    }();
    return t;
}
uint64_t qpc_now() {     // ns, steady
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
void maybe_split() {
    const uint64_t ticks = split_ticks();
    static const uint32_t min_draws = static_cast<uint32_t>(std::max<int64_t>(0, nxrt::config_get_int("nvn", "split_min_draws", 16)));   // tests use 0
    if (!ticks || P.open || g_split_draws < min_draws) return;
    if (g_occ_active >= 0 || g_cw[0].active >= 0 || g_cw[1].active >= 0 || g_xfb_active || g_xfb_began) return;   // queries, stream-out: one buffer
    const uint64_t now = qpc_now();
    if (now - g_split_t0 < ticks) return;
    const int labels = g_dbg_depth;
    for (int i = 0; i < labels; ++i) vkCmdEndDebugUtilsLabelEXT(F->cb);     // labels cannot outlive their command buffer (later pops close nothing)
    if (labels) g_dbg_depth = 0;
    if (!split_submission(*F)) return;
    // the new command buffer has no state: everything is bound again
    g_bound_pipeline = VK_NULL_HANDLE;
    g_dirty = kAllDirty;
    g_topo = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
    g_restart_valid = false;
    g_vb_bound_in_cb = false;
    g_split_t0 = now;
    g_split_draws = 0;
}

// Starts a pass on the current render targets if none is open for them. False when there is nothing to render to. `writes`: the program
// writes images or buffers, so a draw without render targets still has an effect; it gets a pass without attachments over its viewports
// (Fortune's Weave's SSAO deinterleaves depth and stores its occlusion into 2D arrays from such draws).
bool ensure_pass(bool depth_ro = false, bool writes = false) {
    const VkExtent2D bare = writes && !S.ncolor && !S.depth_tex ? viewport_extent() : VkExtent2D{0, 0};
    if (P.open && targets_match() && P.depth_ro == depth_ro && bare.width <= P.extent.width && bare.height <= P.extent.height) return true;
    end_pass(true);
    if (!g_pending.empty()) {
        bool conflict = false;
        for (uint32_t i = 0; i < S.ncolor && !conflict; ++i) conflict = pending_conflict(S.color_tex[i].get(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        if (!conflict && S.depth_tex)
            conflict = pending_conflict(S.depth_tex.get(), depth_ro ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        if (conflict) {
            memory_barrier(F->cb);
            g_pending.clear();
        }
    }
    maybe_split();
    if (gpass_on()) gpass_stamp(*F, "gap", 0);      // copies, dispatches, barriers and layout changes since the previous pass
    g_pass_draws = 0;
    g_pass_verts = 0;
    const float pass_scale = attach_targets(true);
    P = Pass();
    P.depth_ro = depth_ro;
    P.scale = pass_scale;
    VkRenderingAttachmentInfo colors[8] = {};
    VkRenderingAttachmentInfo depth = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    VkRenderingAttachmentInfo stencil = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    bool any = false;
    VkExtent2D extent = {UINT32_MAX, UINT32_MAX};
    P.ncolor = S.ncolor;
    for (uint32_t i = 0; i < S.ncolor; ++i) {
        colors[i] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        Texture* t = S.color_tex[i].get();
        P.color[i] = make_target(t, S.color_view[i], S.color_has_view[i]);
        if (!t || !tex_gpu(t)) continue;
        const bool fresh = tex_layout(t) == VK_IMAGE_LAYOUT_UNDEFINED;
        tex_transition(t, *F, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        tex_mark_gpu_written(t);
        VkImageView v = tex_attachment_view(t, P.color[i].level, P.color[i].layer, P.color[i].layers, P.color[i].view_format);
        if (!v) continue;
        colors[i].imageView = v;
        colors[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colors[i].loadOp = fresh ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        colors[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        P.color_fmt[i] = tex_attachment_format(t, P.color[i].view_format);
        const VkExtent3D e = tex_image_extent(t, P.color[i].level);
        extent.width = std::min(extent.width, e.width);
        extent.height = std::min(extent.height, e.height);
        any = true;
    }
    Texture* dt = S.depth_tex.get();
    P.depth = make_target(dt, S.depth_view, S.depth_has_view);
    bool have_depth = false;
    if (dt && tex_gpu(dt)) {
        const bool fresh = tex_layout(dt) == VK_IMAGE_LAYOUT_UNDEFINED;
        const VkImageLayout dlayout = depth_ro ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        tex_transition(dt, *F, dlayout);
        if (!depth_ro) tex_mark_gpu_written(dt);
        VkImageView v = tex_attachment_view(dt, P.depth.level, P.depth.layer, P.depth.layers, P.depth.view_format);
        if (v) {
            depth.imageView = stencil.imageView = v;
            depth.imageLayout = stencil.imageLayout = dlayout;
            depth.loadOp = stencil.loadOp = (fresh && !depth_ro) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            depth.storeOp = stencil.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth.clearValue.depthStencil = {1.0f, 0};
            stencil.clearValue.depthStencil = {1.0f, 0};
            P.depth_fmt = tex_vk_format(dt);
            P.has_stencil = (tex_aspect(dt) & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
            const VkExtent3D e = tex_image_extent(dt, P.depth.level);
            extent.width = std::min(extent.width, e.width);
            extent.height = std::min(extent.height, e.height);
            have_depth = true;
            any = true;
        }
    }
    if (!any && bare.width && bare.height) {
        extent = bare;
        any = true;
    }
    if (!any && g_xfb_active && device().features().transform_feedback) {   // a stream-out pass without attachments
        extent = {1, 1};
        any = true;
    }
    if (!any || extent.width == UINT32_MAX) return false;
    P.extent = extent;
    P.samples = std::max<uint32_t>(1, S.color_tex[0] ? static_cast<uint32_t>(S.color_tex[0]->b.samples) : 1);
    if (P.samples == 0) P.samples = 1;
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, extent};
    uint32_t layer_count = UINT32_MAX;     // layered rendering: the layers every attachment has in common
    for (uint32_t i = 0; i < S.ncolor; ++i)
        if (colors[i].imageView) layer_count = std::min(layer_count, std::max<uint32_t>(1, P.color[i].layers));
    if (have_depth) layer_count = std::min(layer_count, std::max<uint32_t>(1, P.depth.layers));
    ri.layerCount = layer_count == UINT32_MAX ? 1 : layer_count;
    P.layers = ri.layerCount;
    uint32_t last = 0;
    for (uint32_t i = 0; i < S.ncolor; ++i)
        if (colors[i].imageView) last = i + 1;
    ri.colorAttachmentCount = S.ncolor;
    ri.pColorAttachments = colors;
    if (have_depth) {
        ri.pDepthAttachment = &depth;
        if (P.has_stencil) ri.pStencilAttachment = &stencil;
    }
    (void)last;
    gstamp("pass setup");
    vkCmdBeginRendering(F->cb, &ri);
    P.open = true;
    counters().passes.fetch_add(1, std::memory_order_relaxed);
    g_bound_pipeline = VK_NULL_HANDLE;
    return true;
}

// ---- viewport conversion -------------------------------------------------------------------------------------------------
VkRect2D scissor_rect(const int32_t* r) {
    int32_t x = r[0], y = r[1], w = r[2], h = r[3];
    if (P.scale != 1.0f) {     // the edges scale (a rectangle covering the target still covers it)
        const float sc = P.scale;
        const int32_t x1 = static_cast<int32_t>(std::ceil((static_cast<float>(x) + static_cast<float>(w)) * sc - 0.001f));
        const int32_t y1 = static_cast<int32_t>(std::ceil((static_cast<float>(y) + static_cast<float>(h)) * sc - 0.001f));
        x = static_cast<int32_t>(std::floor(static_cast<float>(x) * sc + 0.001f));
        y = static_cast<int32_t>(std::floor(static_cast<float>(y) * sc + 0.001f));
        w = x1 - x;
        h = y1 - y;
    }
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    w = std::max(0, std::min<int32_t>(w, static_cast<int32_t>(P.extent.width) - x));
    h = std::max(0, std::min<int32_t>(h, static_cast<int32_t>(P.extent.height) - y));
    return {{x, y}, {static_cast<uint32_t>(w), static_cast<uint32_t>(h)}};
}

void set_viewports_and_scissors() {
    VkCommandBuffer cb = F->cb;
    VkViewport vp[16];
    VkRect2D sc[16];
    const uint32_t nvp = S.nvp ? S.nvp : 1;
    for (uint32_t i = 0; i < nvp; ++i) {
        float x = S.vp[i][0], y = S.vp[i][1], w = S.vp[i][2], h = S.vp[i][3];
        if (!S.nvp) { x = 0; y = 0; w = static_cast<float>(P.extent.width); h = static_cast<float>(P.extent.height); }
        else if (P.scale != 1.0f) { x *= P.scale; y *= P.scale; w *= P.scale; h *= P.scale; }
        // Upper-left origin: window y is the image row from the top and NDC +y is the top edge of the rectangle (negative height).
        // Lower-left origin (the GL convention): window y 0 is memory row 0 and NDC -y is the first row, so no flip.
        const bool lower_left = S.window_origin == NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT;
        vp[i] = lower_left ? VkViewport{x, y, w, h, S.dr[i][0], S.dr[i][1] == 0.0f && S.dr[i][0] == 0.0f ? 1.0f : S.dr[i][1]}
                           : VkViewport{x, y + h, w, -h, S.dr[i][0], S.dr[i][1] == 0.0f && S.dr[i][0] == 0.0f ? 1.0f : S.dr[i][1]};
        if (i < S.nsc) sc[i] = scissor_rect(S.sc[i]);
        else sc[i] = {{0, 0}, P.extent};
    }
    vkCmdSetViewportWithCount(cb, nvp, vp);
    vkCmdSetScissorWithCount(cb, nvp, sc);
}

// ---- descriptor sets ---------------------------------------------------------------------------------------------------------
struct TexResolved {
    Texture* tex = nullptr;
    TextureViewInfo view;
    bool has_view = false;
    SamplerInfo smp;
    bool has_sampler = false;
};

// The scale nxrt/gfx_tuning.h asks for on uniform block `idx` of this binding's shader (1 = none): only the chosen fragment shader's block.
float ufs_scale_for(const BindingDesc& d, int idx) {
    const nxrt::gfx::Tuning& t = nxrt::gfx::tuning();
    const uint64_t want = t.ufs_hash.load(std::memory_order_relaxed);
    if (!want || d.nvn_stage != kFragment || idx != t.ufs_block.load(std::memory_order_relaxed)) return 1.0f;
    const float sc = t.ufs_scale.load(std::memory_order_relaxed);
    if (sc == 1.0f || !d.shader || std::strtoull(d.shader->hash.c_str(), nullptr, 16) != want) return 1.0f;
    return sc;
}

uint32_t read_handle_word(int stage, int bank, int word) {
    // a texture handle the game wrote into uniform block (bank - 3)
    const int ub = bank - 3;
    if (ub < 0 || ub >= 16) return 0;
    const GState::Bind& b = S.ubo[stage][ub];
    if (!b.addr || static_cast<uint64_t>(word) * 4 + 4 > b.size) return 0;
    uint32_t v = 0;
    std::memcpy(&v, host_ptr(b.addr) + static_cast<size_t>(word) * 4, 4);
    return v;
}

// [nvn] resolve_memo (default on): within one draw or dispatch each texture and image binding is resolved once. samples_depth_target,
// vol3d_prepare, prepare_textures, feedback_prepare, support_for_draw and build_set each resolved every binding again, each time under the
// pool locks (notes/rt.md). Nothing between those calls changes what a binding resolves to: no op runs inside a draw's setup.
bool memo_on() {
    static const bool on = nxrt::config_get_bool("nvn", "resolve_memo", true);
    return on;
}
struct ResMemo { uint64_t te = 0, ie = 0; TexResolved t, i; };
std::vector<ResMemo> g_memo;
const BindingDesc* g_memo_base = nullptr;   // the bindings of the draw or dispatch being set up (null: no memo)
size_t g_memo_n = 0;
uint64_t g_memo_epoch = 0;
struct MemoScope {
    const BindingDesc* base; size_t n;
    explicit MemoScope(const ProgLayout& pl) : base(g_memo_base), n(g_memo_n) {
        if (!memo_on()) return;
        g_memo_base = pl.bindings.data();
        g_memo_n = pl.bindings.size();
        if (g_memo.size() < g_memo_n) g_memo.resize(g_memo_n);
        ++g_memo_epoch;
    }
    ~MemoScope() {
        if (!memo_on()) return;
        g_memo_base = base;
        g_memo_n = n;
        ++g_memo_epoch;    // an outer scope (none today) resolves again
    }
};
ResMemo* memo_for(const BindingDesc& d) {
    const BindingDesc* p = &d;
    if (!g_memo_base || p < g_memo_base || p >= g_memo_base + g_memo_n) return nullptr;
    return &g_memo[static_cast<size_t>(p - g_memo_base)];
}

TexResolved resolve_texture_raw(const BindingDesc& d);
TexResolved resolve_image_raw(const BindingDesc& d);
bool same_resolved(const TexResolved& a, const TexResolved& b) {
    if (a.tex != b.tex || a.has_view != b.has_view || a.has_sampler != b.has_sampler) return false;
    if (a.has_sampler && !(a.smp == b.smp)) return false;
    if (!a.has_view) return true;
    const TextureViewInfo& x = a.view; const TextureViewInfo& y = b.view;
    return x.has_levels == y.has_levels && x.has_layers == y.has_layers && x.has_format == y.has_format && x.has_swizzle == y.has_swizzle &&
           x.has_depth_stencil_mode == y.has_depth_stencil_mode && x.has_target == y.has_target && x.min_level == y.min_level &&
           x.num_levels == y.num_levels && x.min_layer == y.min_layer && x.num_layers == y.num_layers && x.format == y.format &&
           std::memcmp(x.swizzle, y.swizzle, sizeof x.swizzle) == 0 && x.depth_stencil_mode == y.depth_stencil_mode && x.target == y.target;
}
TexResolved resolve_texture(const BindingDesc& d) {
    ResMemo* m = memo_for(d);
    if (!m) return resolve_texture_raw(d);
    if (m->te != g_memo_epoch) { m->t = resolve_texture_raw(d); m->te = g_memo_epoch; }
    else {
        ++g_rs.res_memo;
        if (rt_check()) chk(same_resolved(m->t, resolve_texture_raw(d)), "resolve_texture memo");
    }
    return m->t;
}
TexResolved resolve_image(const BindingDesc& d) {
    ResMemo* m = memo_for(d);
    if (!m) return resolve_image_raw(d);
    if (m->ie != g_memo_epoch) { m->i = resolve_image_raw(d); m->ie = g_memo_epoch; }
    else {
        ++g_rs.res_memo;
        if (rt_check()) chk(same_resolved(m->i, resolve_image_raw(d)), "resolve_image memo");
    }
    return m->i;
}

TexResolved resolve_texture_raw(const BindingDesc& d) {
    ++g_rs.res_tex;
    TexResolved r;
    const DescriptorBinding& b = d.shader->refl.bindings[d.index];
    uint64_t handle = 0;
    if (b.nvn.source == NvnSource::SamplerBinding) {
        const int idx = b.nvn.index;
        if (idx >= 0 && idx < 64) handle = S.tex[d.nvn_stage][idx];
    } else if (b.nvn.source == NvnSource::HandleInUniform) {
        handle = read_handle_word(d.nvn_stage, b.nvn.bank, b.nvn.word);
    } else if (b.nvn.source == NvnSource::ImageBinding) {     // NVN2 compute: the bound image read through its handle (sampler id 0)
        const int idx = b.nvn.index;
        if (idx >= 0 && idx < 16) handle = S.img[d.nvn_stage][idx];
    }
    const uint32_t tid = handle_texture_id(handle), sid = handle_sampler_id(handle);
    if (S.tpool && static_cast<int>(tid) < S.tpool->count) {
        std::shared_lock<std::shared_mutex> lk(S.tpool->lock);
        const TexturePool::Entry& e = S.tpool->entries[tid];
        if (e.used && e.tex) {
            r.tex = e.tex.get();
            r.view = e.view;
            r.has_view = e.has_view;
        }
    }
    if (S.spool && static_cast<int>(sid) < S.spool->count) {
        std::shared_lock<std::shared_mutex> lk(S.spool->lock);
        if (S.spool->used[sid]) {
            r.smp = S.spool->entries[sid];
            r.has_sampler = true;
        }
    }
    return r;
}

ViewRequest make_request(const DescriptorBinding& b, const TexResolved& r) {
    ViewRequest q;
    q.type = view_type_for(b);
    q.depth_compare = b.image.depth;
    const Texture& t = *r.tex;
    int sw[4] = {t.b.swizzle[0], t.b.swizzle[1], t.b.swizzle[2], t.b.swizzle[3]};
    int dsmode = t.b.depth_stencil_mode;
    if (r.has_view) {
        const TextureViewInfo& v = r.view;
        if (v.has_levels) { q.base_level = static_cast<uint32_t>(v.min_level); q.level_count = static_cast<uint32_t>(v.num_levels); }
        if (v.has_layers) { q.base_layer = static_cast<uint32_t>(v.min_layer); q.layer_count = static_cast<uint32_t>(v.num_layers); }
        if (v.has_format) {
            const FormatInfo* fi = format_info(v.format);
            if (fi) q.format = fi->vk;
        }
        if (v.has_swizzle) for (int i = 0; i < 4; ++i) sw[i] = v.swizzle[i];
        if (v.has_depth_stencil_mode) dsmode = v.depth_stencil_mode;
    }
    q.swizzle.r = vk_swizzle(sw[0]);
    q.swizzle.g = vk_swizzle(sw[1]);
    q.swizzle.b = vk_swizzle(sw[2]);
    q.swizzle.a = vk_swizzle(sw[3]);
    q.depth_aspect = dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH;
    switch (b.image.component) {
        case ScalarType::Float: q.component = 1; break;
        case ScalarType::Sint: q.component = 2; break;
        case ScalarType::Uint: q.component = 3; break;
        default: break;
    }
    return q;
}

// Integer textures read through float-declared samplers (spirv_int_samplers): do_dispatch sets, for the dispatch's layout, the sampler
// bindings (index into pl.bindings) whose texture holds 8- or 16-bit integers (2 sint, 3 uint) when the compute pipeline is the variant
// that reads them as integer images. build_set then binds an integer view and an unfiltered sampler there. Null / empty otherwise.
const ProgLayout* g_int_pl = nullptr;
std::vector<uint8_t> g_int_comp;
uint8_t int_sampler_for(const ProgLayout& pl, const BindingDesc& d) {
    if (g_int_pl != &pl || &d < pl.bindings.data() || &d >= pl.bindings.data() + pl.bindings.size()) return 0;
    const size_t i = static_cast<size_t>(&d - pl.bindings.data());
    return i < g_int_comp.size() ? g_int_comp[i] : 0;
}
// The sampler of an integer texture: no filtering (Vulkan has no linear filtering of integer formats; Maxwell takes the nearest texel).
SamplerInfo unfiltered(SamplerInfo s) {
    s.mag_filter = NVN_MAG_FILTER_NEAREST;
    s.min_filter = min_filter_uses_mips(s.min_filter) ? NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST : NVN_MIN_FILTER_NEAREST;
    s.max_anisotropy = 1;
    return s;
}

// The image a draw reads or writes through an image binding (nvnCommandBufferBindImage): the handle is a texture pool id.
TexResolved resolve_image_raw(const BindingDesc& d) {
    ++g_rs.res_img;
    TexResolved r;
    const DescriptorBinding& b = d.shader->refl.bindings[d.index];
    uint64_t handle = 0;
    if (b.nvn.source == NvnSource::ImageBinding) {
        const int idx = b.nvn.index;
        if (idx >= 0 && idx < 16) handle = S.img[d.nvn_stage][idx];
    } else if (b.nvn.source == NvnSource::HandleInUniform) {
        handle = read_handle_word(d.nvn_stage, b.nvn.bank, b.nvn.word);
    }
    const uint32_t tid = handle_texture_id(handle);
    if (S.tpool && static_cast<int>(tid) < S.tpool->count) {
        std::shared_lock<std::shared_mutex> lk(S.tpool->lock);
        const TexturePool::Entry& e = S.tpool->entries[tid];
        if (e.used && e.tex) {
            r.tex = e.tex.get();
            r.view = e.view;
            r.has_view = e.has_view;
        }
    }
    return r;
}

// True when two texture objects are the same image: the object itself, or another object over the same storage (S4f: objects over one piece of
// memory share one image, so the one the pass renders to and the one a shader samples can be two objects).
bool shares_image(Texture* a, const Texture* b) {
    if (!a || !b) return false;
    if (a == b) return true;
    if (!tex_gpu(a)) return false;
    return a->gpu != nullptr && a->gpu == b->gpu;
}

// True when a texture the program samples is the depth attachment of the current target set: the draw then runs in a pass whose depth
// attachment is read-only (DEPTH_STENCIL_READ_ONLY_OPTIMAL serves as attachment and as sampled layout).
bool samples_depth_target(const ProgLayout& pl) {
    if (!S.depth_tex) return false;
    for (const BindingDesc& d : pl.bindings) {
        if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        if (b.nvn.source != NvnSource::SamplerBinding && b.nvn.source != NvnSource::HandleInUniform) continue;
        if (shares_image(resolve_texture(d).tex, S.depth_tex.get())) return true;
    }
    return false;
}

// A colour attachment cannot be sampled in the pass that renders to it (no feedback loops): the binding gets the dummy view.
bool is_color_feedback(Texture* t) {
    if (!t) return false;
    static const bool fast = nxrt::config_get_bool("nvn", "feedback_fast", true);
    if (!fast) {
        for (uint32_t i = 0; i < S.ncolor; ++i) {
            Texture* a = S.color_tex[i].get();
            if (a && !a->gpu) tex_gpu_attachment(a);   // an attachment gets its image (or the one it shares) here, not later in ensure_pass: the check runs before the pass begins (S4g)
            if (shares_image(t, a)) return true;
        }
        return false;
    }
    // [nvn] feedback_fast (default on): the same answer with one image lookup of t instead of one per attachment (shares_image looked t's
    // image up for every colour target; with five G-buffer targets and three calls a binding per draw that was ~15 lookups of a cold
    // object). Every attachment gets its image first, so t finds an attachment it aliases exactly as before.
    bool any = false;
    for (uint32_t i = 0; i < S.ncolor; ++i) {
        Texture* a = S.color_tex[i].get();
        if (!a) continue;
        if (!a->gpu) tex_gpu_attachment(a);   // an attachment gets its image (or the one it shares) here, not later in ensure_pass (S4g)
        if (a == t) return true;
        any = true;
    }
    if (!any || !tex_gpu(t) || !t->gpu) return false;
    for (uint32_t i = 0; i < S.ncolor; ++i) {
        const Texture* a = S.color_tex[i].get();
        if (a && a->gpu == t->gpu) return true;
    }
    return false;
}

// A draw that samples its own colour attachment (the bloom composite of some frames reads the tone-mapped picture while it writes it back, one
// pixel at a time): the console allows that, Vulkan has no such loop. The draw samples a copy of the image made just before it, which holds the
// same values a pointwise read sees on the hardware (and the right ones when the shader reads another mip level of the attachment).
struct FbClone {
    Ref<Texture> tex;
    uint64_t last = 0;                                     // serial of the last submission that used the copy
};
std::unordered_map<Texture*, FbClone> g_fb_clone;          // attachment -> its copy (same shape, no storage in game memory)
std::unordered_map<Texture*, Texture*> g_fb_use;           // attachment -> the copy the current draw samples
bool g_fb_seen = false;                                    // the last check found a feedback binding: check every draw until it goes away
std::atomic<int64_t> g_fb_clone_n{0};                      // g_fb_clone's size, mirrored for the gauge (the map itself is render thread only)

Texture* feedback_clone(Texture* t) {
    FbClone& fe = g_fb_clone[t];
    fe.last = F ? F->serial : 0;
    Ref<Texture>& c = fe.tex;
    g_fb_clone_n.store(static_cast<int64_t>(g_fb_clone.size()), std::memory_order_relaxed);
    if (c && (c->b.width != t->b.width || c->b.height != t->b.height || c->b.depth != t->b.depth || c->fmt != t->fmt || c->levels != t->levels || c->layers != t->layers ||
              (c->gpu && tex_scale(c.get()) != tex_scale(t))))
        c = Ref<Texture>();                                 // the game made another texture at this address
    if (!c) {
        Texture* n = new Texture();
        n->b = t->b;
        n->fmt = t->fmt;
        n->levels = t->levels;
        n->layers = t->layers;
        n->block_linear = t->block_linear;
        n->cpu_dirty.store(false);
        c = Ref<Texture>(n);
        static uint32_t next_id = 0x7f000000;
        n->id = next_id++;
    }
    return tex_gpu_scaled(c.get(), tex_scale(t)) ? c.get() : nullptr;
}

// [nvn] feedback_levels (default false): the copy holds only the levels the draw's views sample (all bindings of the texture together)
// instead of every level. A draw that renders a small mip while it samples the next larger one (a bloom or blur chain over one mipmapped
// texture) then copies that one small level, not the whole chain from the full-size level 0 down.
bool fb_levels_on() {
    static const bool on = nxrt::config_get_bool("nvn", "feedback_levels", false);
    return on;
}

// The image levels [lo, hi) of `t` that the draw's bindings sample through it (every level when one binding has no level range).
void fb_level_range(const ProgLayout& pl, Texture* t, uint32_t* lo, uint32_t* hi) {
    const uint32_t levels = tex_levels(t);
    *lo = 0;
    *hi = levels;
    if (!fb_levels_on() || tex_level_drop(t) != 0 || levels <= 1) return;
    uint32_t a = levels, z = 0;
    for (const BindingDesc& d : pl.bindings) {
        if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        if (b.nvn.source != NvnSource::SamplerBinding && b.nvn.source != NvnSource::HandleInUniform) continue;
        const TexResolved r = resolve_texture(d);
        if (!r.tex || !shares_image(r.tex, t)) continue;
        if (r.tex != t || !r.has_view || !r.view.has_levels || r.view.min_level < 0) return;
        const uint32_t l0 = std::min<uint32_t>(static_cast<uint32_t>(r.view.min_level), levels - 1);
        const uint32_t n = r.view.num_levels > 0 ? static_cast<uint32_t>(r.view.num_levels) : levels - l0;
        a = std::min(a, l0);
        z = std::max(z, std::min(levels, l0 + n));
    }
    if (a < z) { *lo = a; *hi = z; }
}

void feedback_prepare(const ProgLayout& pl) {
    g_fb_use.clear();
    bool any = false;
    for (const BindingDesc& d : pl.bindings) {
        if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        if (b.nvn.source != NvnSource::SamplerBinding && b.nvn.source != NvnSource::HandleInUniform) continue;
        const TexResolved r = resolve_texture(d);
        if (!r.tex || !tex_gpu(r.tex) || !is_color_feedback(r.tex) || r.tex->b.samples > 1 || g_fb_use.count(r.tex)) continue;
        any = true;
        Texture* c = feedback_clone(r.tex);
        if (!c) continue;
        uint32_t lv_lo = 0, lv_hi = 0;
        fb_level_range(pl, r.tex, &lv_lo, &lv_hi);
        end_pass();
        tex_ensure_exact(r.tex, *F);                                      // storage the game wrote since the image was made reaches the copy too
        tex_transition(r.tex, *F, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);   // the next pass moves it back to the attachment layout and loads it
        tex_transition(c, *F, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        std::vector<VkImageCopy> regions;
        for (uint32_t lv = lv_lo; lv < lv_hi; ++lv) {
            const VkExtent3D e = tex_image_extent(r.tex, lv);
            VkImageCopy rc = {};
            rc.srcSubresource = {tex_aspect(r.tex), lv, 0, tex_layers(r.tex)};
            rc.dstSubresource = {tex_aspect(c), lv, 0, tex_layers(c)};
            rc.extent = e;
            regions.push_back(rc);
        }
        vkCmdCopyImage(F->cb, tex_image(r.tex), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex_image(c), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<uint32_t>(regions.size()), regions.data());
        tex_mark_gpu_written(c);
        tex_transition(c, *F, tex_sampled_layout(c));
        g_fb_use[r.tex] = c;
        counters().copies.fetch_add(1, std::memory_order_relaxed);
        gstamp("feedback copy");
    }
    g_fb_seen = any;
}

void flush_compute_out(int why = kOutGpuReader);       // compute shadows (below): results of direct compute bindings go back to the pool
void compute_out_guard(uint64_t addr, uint64_t size, int why = kOutGpuReader);
// [addr, +size) of game memory a texture upload reads: held compute results there go back first (compute_out_lazy; else all of them).
void out_texture_read(Texture* t) {
    if (!out_lazy()) { flush_compute_out(kOutTexture); return; }
    if (t && t->mem && t->cpu_dirty.load()) compute_out_guard(reinterpret_cast<uint64_t>(t->mem), t->storage_size ? t->storage_size : 4096, kOutTexture);
}

// A sampler3D reading a texture whose storage is 2D (a 2D texture the game left bound to the slot, or a 3D texture view of 2D storage): the
// console samples it as a volume one slice deep. Vulkan cannot view a 2D image as 3D, so the draw samples a 3D copy (w x h x 1, every level) of the
// texture's layer, made before the pass and refreshed once per submission (and whenever the CPU wrote the texture).
struct Vol3dCopy {
    Ref<Texture> src;                                      // keeps the key alive while the copy exists
    Ref<Texture> tex;
    uint32_t layer = 0;
    uint64_t copied = UINT64_MAX;                          // serial of the submission of the last copy
    uint64_t last = 0;                                     // serial of the last submission that used it
};
std::unordered_map<Texture*, Vol3dCopy> g_vol3d;           // 2D texture -> its 3D copy
std::unordered_map<Texture*, Texture*> g_vol3d_use;        // 2D texture -> the copy the current draw samples

bool wants_vol3d(const DescriptorBinding& b, const TexResolved& r) {
    if (!r.tex || view_type_for(b) != VK_IMAGE_VIEW_TYPE_3D) return false;
    const int target = r.tex->b.target;
    return target != NVN_TEXTURE_TARGET_3D && target != NVN_TEXTURE_TARGET_BUFFER && !target_is_multisample(target) && r.tex->b.samples <= 1 &&
           !tex_is_depth(r.tex);
}

void vol3d_prepare(const ProgLayout& pl) {
    g_vol3d_use.clear();
    for (const BindingDesc& d : pl.bindings) {
        if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        if (b.nvn.source != NvnSource::SamplerBinding && b.nvn.source != NvnSource::HandleInUniform) continue;
        const TexResolved r = resolve_texture(d);
        if (!wants_vol3d(b, r) || g_vol3d_use.count(r.tex) || !tex_gpu(r.tex)) continue;
        const uint32_t layer = r.has_view && r.view.has_layers ? static_cast<uint32_t>(r.view.min_layer) : 0;
        if (layer >= tex_layers(r.tex)) continue;
        Vol3dCopy& e = g_vol3d[r.tex];
        if (e.src.get() != r.tex) e.src = Ref<Texture>::share(r.tex);
        e.last = F ? F->serial : 0;
        Ref<Texture>& c = e.tex;
        if (c && (c->b.width != r.tex->b.width || c->b.height != r.tex->b.height || c->fmt != r.tex->fmt || c->levels != r.tex->levels || e.layer != layer ||
                  (c->gpu && tex_scale(c.get()) != tex_scale(r.tex))))
            c = Ref<Texture>();
        if (!c) {
            Texture* n = new Texture();
            n->b = r.tex->b;
            n->b.target = NVN_TEXTURE_TARGET_3D;
            n->b.depth = 1;
            n->fmt = r.tex->fmt;
            n->levels = r.tex->levels;
            n->layers = 1;
            n->block_linear = r.tex->block_linear;
            n->cpu_dirty.store(false);
            static uint32_t next_id = 0x7e000000;
            n->id = next_id++;
            c = Ref<Texture>(n);
            e.layer = layer;
            e.copied = UINT64_MAX;
        }
        if (!tex_gpu_scaled(c.get(), tex_scale(r.tex))) continue;
        const uint64_t serial = F ? F->serial : 0;
        if (e.copied != serial || r.tex->cpu_dirty.load()) {
            out_texture_read(r.tex);
            end_pass();
            tex_ensure_exact(r.tex, *F);
            tex_transition(r.tex, *F, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            tex_transition(c.get(), *F, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            std::vector<VkImageCopy> regions;
            for (uint32_t lv = 0; lv < tex_levels(r.tex); ++lv) {
                VkImageCopy rc = {};
                rc.srcSubresource = {tex_aspect(r.tex), lv, layer, 1};
                rc.dstSubresource = {tex_aspect(c.get()), lv, 0, 1};
                rc.extent = tex_image_extent(r.tex, lv);
                rc.extent.depth = 1;
                regions.push_back(rc);
            }
            vkCmdCopyImage(F->cb, tex_image(r.tex), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex_image(c.get()), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<uint32_t>(regions.size()), regions.data());
            tex_mark_gpu_written(c.get());
            tex_transition(c.get(), *F, tex_sampled_layout(c.get()));
            tex_transition(r.tex, *F, tex_sampled_layout(r.tex));   // the same draw may sample it as 2D too
            e.copied = serial;
            counters().copies.fetch_add(1, std::memory_order_relaxed);
            gstamp("3D copy of a 2D texture");
            if (first_time_key("vol3d " + std::to_string(r.tex->id)))
                NXLOG_I(kTag, "texture %u (%s %dx%d, target %d) is sampled as 3D by %s binding %s: it gets a 3D copy one slice deep", r.tex->id,
                        r.tex->fmt ? r.tex->fmt->name : "?", r.tex->b.width, r.tex->b.height, r.tex->b.target, d.shader->hash.c_str(), b.name.c_str());
        }
        g_vol3d_use[r.tex] = c.get();
    }
}

std::atomic<uint64_t> g_pool_uploads_deferred{0};
std::atomic<uint64_t> g_pacing_on_present{0};   // the present at which pacing last came back on after a load (note_frame_for_pacing)
const bool g_pool_deferred_hitch = (nxrt::hitch_add_counter("pool_uploads_deferred", []() -> uint64_t { return g_pool_uploads_deferred.load(std::memory_order_relaxed); }), true);

// Makes the textures a draw samples current (uploaded, in their sampled layout) and the storage images it binds writable (GENERAL).
// Ends the pass when it has to touch images. The depth attachment itself is left to the pass.
void prepare_textures(const ProgLayout& pl) {
    vol3d_prepare(pl);
    // kept from call to call (render thread; never re-entered): three heap allocations a draw otherwise
    static std::vector<Texture*> need, need_storage, storage_bound;
    need.clear();
    need_storage.clear();
    storage_bound.clear();
    for (const BindingDesc& d : pl.bindings) {
        if (d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
            TexResolved r = resolve_image(d);
            if (!r.tex || !tex_gpu(r.tex) || !tex_has_storage(r.tex)) continue;
            storage_bound.push_back(r.tex);
            if (r.tex->cpu_dirty.load() || tex_layout(r.tex) != VK_IMAGE_LAYOUT_GENERAL || tex_scale(r.tex) != 1.0f) need_storage.push_back(r.tex);
            continue;
        }
        if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        if (b.nvn.source != NvnSource::SamplerBinding && b.nvn.source != NvnSource::HandleInUniform && b.nvn.source != NvnSource::ImageBinding) continue;
        TexResolved r = resolve_texture(d);
        if (!r.tex) continue;
        if (!r.tex->gpu && tex_first_upload_deferrable(r.tex)) {   // an asset with no image yet: made when its upload runs (it may wait)
            need.push_back(r.tex);
            continue;
        }
        if (!tex_gpu(r.tex)) continue;
        // owned by the pass; a compute dispatch runs outside every pass, so the graphics attachments do not own what it samples
        if (shares_image(r.tex, S.depth_tex.get()) || (d.nvn_stage != kCompute && is_color_feedback(r.tex))) continue;
        if (r.tex->cpu_dirty.load() || tex_layout(r.tex) != tex_sampled_layout(r.tex)) {
            need.push_back(r.tex);
            if (r.tex->cpu_dirty.load() && ft::on()) ft::mark(b.nvn.source == NvnSource::SamplerBinding ? "tx.dirty_bind" : "tx.dirty_uniform", r.tex->id);
        }
    }
    size_t pool_from = need.size();   // need[pool_from..]: textures only needed because the pools hold them
    if (pl.has_set[kPoolSet]) {
        // a shader indexing the pools may sample any registered texture: those not ready, except what the pass renders to (their pool
        // elements hold the dummy meanwhile, nvn_pipeline.cpp)
        const bool compute = std::any_of(pl.bindings.begin(), pl.bindings.end(), [](const BindingDesc& d) { return d.nvn_stage == kCompute; });
        const size_t from = need.size();
        pool_from = from;
        pool_textures_to_prepare(S.tpool.get(), need);
        need.erase(std::remove_if(need.begin() + static_cast<std::ptrdiff_t>(from), need.end(),
                                  [&](Texture* t) {
                                      if (!t->gpu && tex_first_upload_deferrable(t)) return false;   // an asset with no image yet: not an attachment
                                      return shares_image(t, S.depth_tex.get()) || (!compute && is_color_feedback(t));
                                  }),
                   need.end());
    }
    // an image this op also writes (an NVN2 compute shader sampling its own image) stays in GENERAL, the layout both descriptors name
    need.erase(std::remove_if(need.begin(), need.end(),
                              [&](Texture* t) {
                                  if (!t->gpu && tex_first_upload_deferrable(t)) return false;
                                  return std::any_of(storage_bound.begin(), storage_bound.end(), [&](Texture* s) { return shares_image(t, s); });
                              }),
               need.end());
    // [nvn] pool_upload_ms (default 24; 0 = no limit): the first uploads of asset textures that a draw needs only because a texture pool holds
    // them (not bound by the draw itself) take at most this much render-thread time per presented frame; the rest stay not ready (their pool
    // elements are the dummy) and go in the next frames. A load-in registers thousands of textures and uploaded them all before its first
    // draw: 4378 uploads, 3.8 GB, an 8 s frame at 4K (x2-explore). Bound textures, rewrites of uploaded textures, render targets and
    // uncompressed data are never held back.
    // [nvn] pool_upload_ms (default 6) is the budget while only a few textures wait, so the frame stays near its usual time; with a backlog of
    // more than [nvn] pool_upload_backlog (64) held back in the previous frame (a load-in, a fast travel) it is [nvn] pool_upload_burst_ms
    // (24): the area streams in within a second or so at a lower frame rate. A flat 24 ms made walking through a cold area show twice the
    // frames over 20 ms (s1-stutter1, hub_cold).
    static const int64_t pool_ms = config_get_int("nvn", "pool_upload_ms", 6);
    static const int64_t burst_ms = config_get_int("nvn", "pool_upload_burst_ms", 24);
    static const int64_t backlog_n = config_get_int("nvn", "pool_upload_backlog", 64);
    static uint64_t budget_present = ~0ull;
    static std::chrono::steady_clock::duration spent{}, cap{};
    static bool burst = false;
    static std::unordered_set<Texture*> firsts_frame;   // distinct textures whose first upload the draws of this frame needed
    const uint64_t now_p = counters().presents.load(std::memory_order_relaxed);
    if (now_p != budget_present) {
        budget_present = now_p;
        spent = std::chrono::steady_clock::duration::zero();
        firsts_frame.clear();
        burst = static_cast<int64_t>(pool_held_backlog()) > backlog_n;
        cap = std::chrono::milliseconds(std::max<int64_t>(0, burst ? std::max(pool_ms, burst_ms) : pool_ms));
    }
    // What the budget covers: the first uploads of pool-only asset textures (held for later frames, pool_hold), and the swaps of a stand-in
    // on screen for its finished result ([nvn] pool_swap_budget, default on), bound or pool: the image already holds valid (approximate) data,
    // so a swap can wait a frame anywhere; at a load-in a thousand results land within a second or two and their copies piled into one frame.
    static const bool swap_budget = config_get_bool("nvn", "pool_swap_budget", true);
    // [nvn] bound_upload_defer (default on): while a backlog streams in (burst), first uploads of asset textures a draw binds itself wait too;
    // the draw samples the dummy texture until theirs is in (descriptor resolution below). A load-in's first frame still made 583 uploads, most
    // of them bound, in 582 ms (s2-default). Outside a burst bound textures always go at once.
    static const bool bound_defer = config_get_bool("nvn", "bound_upload_defer", true);
    // more than pool_upload_burst_firsts first uploads needed by this frame's draws so far (the first frame after a loading screen, spread
    // over many draws) start the burst at once; s3-default's load-in frame still made 576 uploads with a per-draw count
    // [nvn] pool_upload_burst_firsts (default 256): the threshold of this count. 64 also fired while walking into a cold street (s4-default:
    // a minute of 24 ms upload frames, 531 frames over 33 ms); a load-in needs several hundred at once.
    static const int64_t firsts_n = config_get_int("nvn", "pool_upload_burst_firsts", 256);
    // ... and in the first [nvn] pool_upload_loadin_frames (30) presents after a loading screen (pacing back on) pool_upload_backlog (64)
    // is enough: the load-in frame starts the burst early (s5-default's, with 256, made 426 uploads in 361 ms; s4's, with 64, 173 in 295)
    static const int64_t loadin_frames = config_get_int("nvn", "pool_upload_loadin_frames", 30);
    const uint64_t on_at = g_pacing_on_present.load(std::memory_order_relaxed);
    // The load-in frame itself is the present at which pacing comes back on: note_frame_for_pacing() runs at that present, after this
    // frame's prepare_textures(), so frames while pacing is still off (a loading screen and its last frame) count as load-in too.
    const bool loadin = (loadin_frames > 0 && nxrt::frame_limiter_unpaced()) ||
                        (on_at && now_p >= on_at && now_p - on_at < static_cast<uint64_t>(std::max<int64_t>(0, loadin_frames)));
    if (pool_ms > 0 && !burst) {
        const size_t lim = static_cast<size_t>(std::max<int64_t>(0, loadin ? std::min(firsts_n, backlog_n) : firsts_n));
        for (Texture* t : need)
            if (tex_first_upload_deferrable(t)) {
                firsts_frame.insert(t);
                if (firsts_frame.size() > lim) break;
            }
        if (firsts_frame.size() > lim) {
            burst = true;
            cap = std::chrono::milliseconds(std::max(pool_ms, burst_ms));
        }
    }
    // [nvn] burst_defer_uncompressed (default off): during a burst, pool-only first uploads of uncompressed textures (UI images, look-up
    // tables: their pool element is the dummy meanwhile, which can show as a wrong colour for a few frames) wait as well
    static const bool burst_uncompressed = config_get_bool("nvn", "burst_defer_uncompressed", false);
    auto budgeted = [&](Texture* t, size_t i) {
        return pool_ms > 0 && (((i >= pool_from || (burst && bound_defer)) && tex_first_upload_deferrable(t)) || (swap_budget && tex_standin_swap_pending(t)) ||
                               (burst && burst_uncompressed && i >= pool_from && tex_first_upload_uncompressed_deferrable(t)));
    };
    if (pool_ms > 0 && !need.empty()) {
        if (spent >= cap) {
            // the frame's budget is spent: what it covers leaves the list before the pass is ended for it (each draw that indexes the pools
            // would otherwise break the render pass for uploads it then skips); pool-only first uploads are held until the frame ends
            size_t kept = 0, kept_before_pool = 0, dropped = 0;
            for (size_t i = 0; i < need.size(); ++i) {
                Texture* t = need[i];
                if (budgeted(t, i)) {
                    if (i >= pool_from) pool_hold(t);   // out of the lists the next draws of this frame look through
                    ++dropped;
                    continue;
                }
                if (i < pool_from) ++kept_before_pool;
                need[kept++] = t;
            }
            need.resize(kept);
            pool_from = kept_before_pool;
            g_pool_uploads_deferred.fetch_add(dropped, std::memory_order_relaxed);
        } else if (pool_from + 1 < need.size()) {
            // the textures whose background result is ready go first: an upload buffer to adopt costs far less than a stand-in or a CPU untile,
            // and the others get more time for their jobs
            std::stable_partition(need.begin() + static_cast<std::ptrdiff_t>(pool_from), need.end(), [](Texture* t) { return texprep::has_result(t); });
        }
    }
    if (need.empty() && need_storage.empty()) return;
    for (Texture* t : need) out_texture_read(t);           // an upload reads game memory: compute results held there go back first
    for (Texture* t : need_storage) out_texture_read(t);
    end_pass(true);     // layout changes and uploads follow: each carries the dependency for its own image
    counters().pass_breaks.fetch_add(1, std::memory_order_relaxed);
    for (size_t i = 0; i < need.size(); ++i) {
        Texture* t = need[i];
        const bool slice = budgeted(t, i);
        if (slice && spent >= cap) {
            g_pool_uploads_deferred.fetch_add(1, std::memory_order_relaxed);
            if (i >= pool_from) pool_hold(t);
            continue;
        }
        const auto t0 = slice ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        tex_ensure_current(t, *F);
        tex_transition(t, *F, tex_sampled_layout(t));
        if (slice) spent += std::chrono::steady_clock::now() - t0;
    }
    for (Texture* t : need_storage) {
        if (tex_scale(t) != 1.0f) tex_rescale(t, *F, 1.0f, "bound as a storage image");   // imageStore coordinates are not scaled (Ryujinx: ResScaleUnsupported)
        tex_ensure_current(t, *F);
        tex_transition(t, *F, VK_IMAGE_LAYOUT_GENERAL);
        tex_mark_gpu_written(t);
    }
    gstamp("uploads/layouts");
}

// SPIR-V ImageFormat of a storage image -> VkFormat (UNDEFINED: Unknown or unmapped).
VkFormat image_format_vk(uint32_t f) {
    switch (f) {
        case 1: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case 2: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case 3: return VK_FORMAT_R32_SFLOAT;
        case 4: return VK_FORMAT_R8G8B8A8_UNORM;
        case 5: return VK_FORMAT_R8G8B8A8_SNORM;
        case 6: return VK_FORMAT_R32G32_SFLOAT;
        case 7: return VK_FORMAT_R16G16_SFLOAT;
        case 8: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case 9: return VK_FORMAT_R16_SFLOAT;
        case 10: return VK_FORMAT_R16G16B16A16_UNORM;
        case 11: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case 12: return VK_FORMAT_R16G16_UNORM;
        case 13: return VK_FORMAT_R8G8_UNORM;
        case 14: return VK_FORMAT_R16_UNORM;
        case 15: return VK_FORMAT_R8_UNORM;
        case 16: return VK_FORMAT_R16G16B16A16_SNORM;
        case 17: return VK_FORMAT_R16G16_SNORM;
        case 18: return VK_FORMAT_R8G8_SNORM;
        case 19: return VK_FORMAT_R16_SNORM;
        case 20: return VK_FORMAT_R8_SNORM;
        case 21: return VK_FORMAT_R32G32B32A32_SINT;
        case 22: return VK_FORMAT_R16G16B16A16_SINT;
        case 23: return VK_FORMAT_R8G8B8A8_SINT;
        case 24: return VK_FORMAT_R32_SINT;
        case 25: return VK_FORMAT_R32G32_SINT;
        case 26: return VK_FORMAT_R16G16_SINT;
        case 27: return VK_FORMAT_R8G8_SINT;
        case 28: return VK_FORMAT_R16_SINT;
        case 29: return VK_FORMAT_R8_SINT;
        case 30: return VK_FORMAT_R32G32B32A32_UINT;
        case 31: return VK_FORMAT_R16G16B16A16_UINT;
        case 32: return VK_FORMAT_R8G8B8A8_UINT;
        case 33: return VK_FORMAT_R32_UINT;
        case 34: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
        case 35: return VK_FORMAT_R32G32_UINT;
        case 36: return VK_FORMAT_R16G16_UINT;
        case 37: return VK_FORMAT_R8G8_UINT;
        case 38: return VK_FORMAT_R16_UINT;
        case 39: return VK_FORMAT_R8_UINT;
        default: return VK_FORMAT_UNDEFINED;
    }
}

int storage_slot_for(int nvn_stage, int c0_offset) {
    // c[0] storage buffer descriptors (byte offsets, as the translator reports them): 0x110 + 0x100 * hardware stage + 0x10 * binding
    // with hardware stage 0 vertex, 3 geometry, 4 fragment; compute: 0x310 + 0x10 * binding. Checked against the translated shaders of the
    // game (vertex 0x110, geometry 0x410, fragment 0x510, compute 0x310 and 0x320).
    int base = 0;
    switch (nvn_stage) {
        case kVertex: base = 0x110; break;
        case kTessCtl: base = 0x210; break;     // tessellation: Ryujinx's hardware stage order (1, 2), not checked on a game shader
        case kTessEval: base = 0x310; break;
        case kGeometry: base = 0x410; break;
        case kFragment: base = 0x510; break;
        case kCompute: base = 0x310; break;
        default: return 0;
    }
    if (c0_offset < base) return 0;
    return (c0_offset - base) / 0x10;
}

// ---- compute shadows: storage and texel buffers of a dispatch in device-local memory ---------------------------------------------
// Pools are the game's own memory imported into Vulkan, so a shader reaches storage buffers and texel buffers over PCIe. Compute shaders
// that work on such buffers (the exposure histogram's atomics, the 4 MiB simulation buffers) took 9 ms each on an RTX 3080. [nvn]
// compute_shadows (default on): each storage or texel buffer range a dispatch binds (up to 16 MiB) is bound from device-local memory.
//  * Direct ([nvn] compute_direct, default on): the range is in a pool with a device-local copy (nvn_gpu_pool.cpp, the pool mirror). The
//    dispatch binds the copy itself. Pages the copy does not hold (the CPU wrote them since the GPU last used them) are copied in from the
//    pool first (mirror_fill), so only the CPU's writes cross PCIe. A range the shader may write is written in the copy; it goes back to the
//    pool once, when something other than a compute binding of the same range could look at it: before any command other than a dispatch,
//    a state change or a barrier, before a dispatch reads it another way (uniform block, indirect arguments, another range on its pages,
//    a texture upload), and at the end of the submission, before the serial the CPU waits for can signal. A run of dispatches over the
//    same buffers thus copies each result out once instead of once per dispatch.
//  * Shadow (otherwise: no copy for the pool, hot or banned pages, a range over two chunks): the range is copied into a buffer of its own
//    before the dispatch and copied back after it when the shader can write it (and into the pool copy, when there is one).
// Game memory holds every result by the end of the submission, in queue order, as before. Ranges that overlap another range of the same
// dispatch stay in the pool.
struct Shadow { gfx::Buffer buf; VkDeviceSize size = 0; uint64_t last = 0; };
std::unordered_map<uint64_t, Shadow> g_shadows;            // by GPU address and size
struct ShadowUse { uint64_t addr; VkDeviceSize size; BufRef host; VkBuffer buf; VkDeviceSize off; bool writes; bool direct; MirrorSpan mirror; };
std::vector<ShadowUse> g_shadow_uses;                       // this dispatch
// Results of direct bindings still only in the pool copy: nvn_gpu_pool.cpp (out_add ... out_end_submit, "compute out").
bool g_shadowing = false;
bool g_sh_barrier = false, g_sh_copied = false;             // this dispatch: the barrier before the copies in is recorded / some copy in is
uint64_t g_sh_uses = 0, g_sh_from_mirror = 0, g_sh_bytes_mirror = 0, g_sh_bytes_pool = 0, g_sh_bytes_out = 0, g_sh_no_span = 0;   // summary line
uint64_t g_sh_direct = 0, g_sh_direct_filled = 0, g_sh_bytes_fill = 0, g_sh_bytes_dout = 0, g_sh_merged = 0, g_sh_flushes = 0;
constexpr VkDeviceSize kShadowMax = 16ull << 20;

bool compute_shadows_on() {
    static const bool on = nxrt::config_get_bool("nvn", "compute_shadows", true);
    return on;
}
bool compute_direct_on() {
    static const bool on = nxrt::config_get_bool("nvn", "compute_direct", true);
    return on;
}

// Every result still only in the pool copy goes back to the pool (in queue order, before what is recorded next).
void flush_compute_out(int why) {
    if (!out_any()) return;
    end_pass();
    if (out_flush(F->cb, 0, UINT64_MAX, why)) {
        ++g_sh_flushes;
        gstamp("compute out");
    }
}

// [addr, addr + size) is about to be read or written some other way than a direct binding of exactly a held range: the results sharing a
// page with it go back first (a page granular test: the pool copy is filled and tracked by whole pages).
void compute_out_guard(uint64_t addr, uint64_t size, int why) {
    if (!size || !out_overlaps(addr, size)) return;
    end_pass();
    if (out_flush(F->cb, addr, size, why)) {
        ++g_sh_flushes;
        gstamp("compute out");
    }
}

// Ops after which the pending results may stay in the pool copy: state that touches no memory, barriers (the copies carry their own),
// dispatches (they check their own ranges) and calls of other lists (their ops are checked one by one). Anything else may read the pool.
bool compute_out_may_wait(Op op) {
    if (op == Op::Dispatch || op == Op::DispatchIndirect || op == Op::Barrier || op == Op::CallCommands || op == Op::PushDebugGroup || op == Op::PopDebugGroup ||
        op == Op::PushConstants) return true;
    return op < Op::DrawArrays && op != Op::SetRenderEnable && op != Op::SetRenderTargets;
}

// A buffer a draw reads: results it shares a page with go back first, unless the draw binds the pool copy for it (vertex, index and aligned
// uniform buffers whose pages are all current there, resolve_read).
void out_draw_read(uint64_t addr, uint64_t size, bool copy_ok, int why) {
    if (!addr) return;
    if (!size) size = 65536;
    if (!out_overlaps(addr, size)) return;
    if (copy_ok && out_servable_memo(addr, size)) return;   // [nvn] servable_memo (default off: out_servable)
    compute_out_guard(addr, size, why);
}

// The buffer for [addr, +size) that a command reads through a plain buffer binding (storage buffers, indirect arguments, copy sources): the
// pool copy when it holds a compute result there and is current for the whole range, else the pool, after the results there went back
// (flush: false inside a pass, where out_before_op has done that already).
BufRef held_or_pool(uint64_t addr, uint64_t size, int why, bool flush) {
    if (size && out_overlaps(addr, size)) {
        if (out_lazy() && out_servable(addr, size)) {
            const MirrorSpan ms = mirror_span(addr, size);
            BufRef r = resolve_buffer(addr, size);
            if (ms.buf && r.ok) {
                r.buffer = ms.buf;
                r.offset = ms.offset;
                r.avail = ms.avail;
                return r;
            }
        }
        if (flush) compute_out_guard(addr, size, why);
    }
    return resolve_buffer(addr, size);
}

// out_draw_bindings' memo, one slot per binding it checks: the range last found clear of held results, and out_gen() then. The same
// range at the same generation is clear again (out_overlaps depends on nothing else). The checks were ~7 % of the render thread: ~180
// slots a draw, most of them bound to the same range as at the draw before (notes/rt.md). [nvn] out_memo (default on).
bool out_memo_on() {
    static const bool on = nxrt::config_get_bool("nvn", "out_memo", true);
    return on;
}
struct OutSlot { uint64_t addr = 0, size = 0, gen = ~0ull; };
constexpr int kOutSlots = 16 + kCompute * 32 + 4;
OutSlot g_out_slot[kOutSlots];
// [nvn] out_memo_dirty (default on): on top of the memo, out_draw_bindings only looks at the slots a BindBuffers changed since the last draw
// while out_gen() is the same and no slot is unmemoised; any program bind, generation change or reset checks every slot as before. The full
// pass was 1.1 us per draw (2.8 ms a frame at 2600 draws on the street), most of it confirming unchanged slots (run B, nvn.rstats).
bool g_out_all = true;                 // the next draw checks every slot
uint64_t g_out_last_gen = ~0ull;       // out_gen() at the last check
bool g_out_unmemo = false;             // a slot ended unmemoised (it held a result): the next draw checks every slot
uint8_t g_out_dirty[kOutSlots] = {};
uint16_t g_out_dirty_list[kOutSlots];
int g_out_dirty_n = 0;
bool out_memo_dirty_on() {
    static const bool on = nxrt::config_get_bool("nvn", "out_memo_dirty", true);
    return on;
}
void out_slot_changed(int idx) {
    if (idx < 0 || idx >= kOutSlots || g_out_dirty[idx]) return;
    g_out_dirty[idx] = 1;
    g_out_dirty_list[g_out_dirty_n++] = static_cast<uint16_t>(idx);
}
void out_dirty_clear() {
    for (int i = 0; i < g_out_dirty_n; ++i) g_out_dirty[g_out_dirty_list[i]] = 0;
    g_out_dirty_n = 0;
}
void out_draw_read_slot(OutSlot& m, const GState::Bind& b, bool copy_ok, int why, uint64_t gen) {
    if (!b.addr) return;
    if (m.gen != ~0ull && m.addr == b.addr && m.size == b.size &&
        (m.gen == gen || out_clear_since(b.addr, b.size ? b.size : 65536, m.gen))) {   // clear then, and nothing held there was added since
        if (rt_check()) chk(!out_overlaps(b.addr, b.size ? b.size : 65536), "out memo");
        m.gen = gen;
        return;
    }
    if (!out_overlaps(b.addr, b.size ? b.size : 65536)) {
        m.addr = b.addr;
        m.size = b.size;
        m.gen = gen;
        return;
    }
    m.gen = ~0ull;
    out_draw_read(b.addr, b.size, copy_ok, why);
}

bool out_masks_on() {
    static const bool on = nxrt::config_get_bool("nvn", "out_masks", true);
    return on;
}
// [nvn] rt_check: the bound-slot masks agree with the slots
void out_masks_check() {
    auto mask_of = [](const GState::Bind* b, int n) { uint32_t m = 0; for (int i = 0; i < n; ++i) if (b[i].addr) m |= 1u << i; return m; };
    bool ok = mask_of(S.vtx, 16) == S.vtx_mask && mask_of(S.xfb, 4) == S.xfb_mask;
    for (int st = 0; st < 6; ++st) ok = ok && mask_of(S.ubo[st], 16) == S.ubo_mask[st] && mask_of(S.ssbo[st], 16) == S.ssbo_mask[st];
    chk(ok, "out masks");
}

// One slot of out_draw_bindings by index (the same order as its full pass).
void out_check_slot(int idx, uint64_t gen, VkDeviceSize ua) {
    const GState::Bind* b;
    bool copy_ok;
    int why;
    if (idx < 16) {
        b = &S.vtx[idx], copy_ok = true, why = kOutDrawBuffer;
    } else if (idx < 16 + kCompute * 32) {
        const int st = (idx - 16) / 32, j = (idx - 16) % 32;
        if (!S.stage[st]) return;                   // as the full pass: an inactive stage's slots are not read by the draw
        if (j < 16) b = &S.ubo[st][j], copy_ok = ua && (b->addr % ua) == 0, why = kOutDrawBuffer;
        else b = &S.ssbo[st][j - 16], copy_ok = true, why = kOutStorageXfb;
    } else {
        b = &S.xfb[idx - 16 - kCompute * 32], copy_ok = false, why = kOutStorageXfb;
    }
    OutSlot& m = g_out_slot[idx];
    out_draw_read_slot(m, *b, copy_ok, why, gen);
    if (b->addr && m.gen == ~0ull) g_out_unmemo = true;
}

// The reads of a draw's bound buffers (out_before_op).
void out_draw_bindings() {
    if (out_memo_on()) {
        const uint64_t gen = out_gen();
        const VkDeviceSize ua = device().properties().limits.minUniformBufferOffsetAlignment;
        if (out_memo_dirty_on() && !g_out_all && !g_out_unmemo && gen == g_out_last_gen) {
            for (int i = 0; i < g_out_dirty_n; ++i) out_check_slot(g_out_dirty_list[i], gen, ua);
            if (rt_check()) {     // [nvn] rt_check: a slot the fast path skipped must have nothing held in its range
                for (int idx = 0; idx < kOutSlots; ++idx) {
                    if (g_out_dirty[idx]) continue;
                    const GState::Bind* b = idx < 16 ? &S.vtx[idx] : idx >= 16 + kCompute * 32 ? &S.xfb[idx - 16 - kCompute * 32]
                                          : !S.stage[(idx - 16) / 32] ? nullptr : (idx - 16) % 32 < 16 ? &S.ubo[(idx - 16) / 32][(idx - 16) % 32]
                                                                                                        : &S.ssbo[(idx - 16) / 32][(idx - 16) % 32 - 16];
                    if (b && b->addr) chk(!out_overlaps(b->addr, b->size ? b->size : 65536), "out memo dirty");
                }
            }
            out_dirty_clear();
            if (out_gen() != gen) g_out_all = true;     // a flush in it moved the generation
            return;
        }
        g_out_unmemo = false;
        if (out_masks_on()) {
            // [nvn] out_masks (default on): only the bound slots (S.*_mask), in the same order; an unbound slot is a no-op in the walk below.
            // The walk over every slot was ~11 % of the render thread at 2600 draws (most of ~84 slots a draw are unbound).
            if (rt_check()) out_masks_check();
            OutSlot* m0 = g_out_slot;
            for (uint32_t mk = S.vtx_mask; mk; mk &= mk - 1) {
                const unsigned k = static_cast<unsigned>(std::countr_zero(mk));
                OutSlot& m = m0[k];
                out_draw_read_slot(m, S.vtx[k], true, kOutDrawBuffer, gen);
                g_out_unmemo |= m.gen == ~0ull;
            }
            for (int st = 0; st < kCompute; ++st) {
                if (!S.stage[st]) continue;
                OutSlot* ms = m0 + 16 + st * 32;
                for (uint32_t mk = S.ubo_mask[st]; mk; mk &= mk - 1) {
                    const unsigned k = static_cast<unsigned>(std::countr_zero(mk));
                    const GState::Bind& b = S.ubo[st][k];
                    out_draw_read_slot(ms[k], b, ua && (b.addr % ua) == 0, kOutDrawBuffer, gen);
                    g_out_unmemo |= ms[k].gen == ~0ull;
                }
                for (uint32_t mk = S.ssbo_mask[st]; mk; mk &= mk - 1) {
                    const unsigned k = static_cast<unsigned>(std::countr_zero(mk));
                    out_draw_read_slot(ms[16 + k], S.ssbo[st][k], true, kOutStorageXfb, gen);
                    g_out_unmemo |= ms[16 + k].gen == ~0ull;
                }
            }
            OutSlot* mx = m0 + 16 + kCompute * 32;
            for (uint32_t mk = S.xfb_mask; mk; mk &= mk - 1) {
                const unsigned k = static_cast<unsigned>(std::countr_zero(mk));
                out_draw_read_slot(mx[k], S.xfb[k], false, kOutStorageXfb, gen);
                g_out_unmemo |= mx[k].gen == ~0ull;
            }
        } else {
            OutSlot* m = g_out_slot;
            for (const GState::Bind& b : S.vtx) { out_draw_read_slot(*m, b, true, kOutDrawBuffer, gen); g_out_unmemo |= b.addr && m->gen == ~0ull; ++m; }
            for (int st = 0; st < kCompute; ++st) {
                if (!S.stage[st]) { m += 32; continue; }
                for (const GState::Bind& b : S.ubo[st]) { out_draw_read_slot(*m, b, ua && (b.addr % ua) == 0, kOutDrawBuffer, gen); g_out_unmemo |= b.addr && m->gen == ~0ull; ++m; }
                for (const GState::Bind& b : S.ssbo[st]) { out_draw_read_slot(*m, b, true, kOutStorageXfb, gen); g_out_unmemo |= b.addr && m->gen == ~0ull; ++m; }
            }
            for (const GState::Bind& b : S.xfb) { out_draw_read_slot(*m, b, false, kOutStorageXfb, gen); g_out_unmemo |= b.addr && m->gen == ~0ull; ++m; }
        }
        g_out_all = false;
        g_out_last_gen = out_gen();     // after the pass: a flush in it may have moved the generation
        if (g_out_last_gen != gen) g_out_all = true;
        out_dirty_clear();
        return;
    }
    for (const GState::Bind& b : S.vtx) out_draw_read(b.addr, b.size, true, kOutDrawBuffer);
    const VkDeviceSize ualign = device().properties().limits.minUniformBufferOffsetAlignment;
    for (int st = 0; st < kCompute; ++st) {
        if (!S.stage[st]) continue;
        for (const GState::Bind& b : S.ubo[st]) out_draw_read(b.addr, b.size, ualign && (b.addr % ualign) == 0, kOutDrawBuffer);
        for (const GState::Bind& b : S.ssbo[st]) out_draw_read(b.addr, b.size, true, kOutStorageXfb);   // bound from the copy (build_set)
    }
    for (const GState::Bind& b : S.xfb) out_draw_read(b.addr, b.size, false, kOutStorageXfb);
}

const uint8_t* cpu_view(uint64_t addr, uint64_t size);   // below

// The bytes from index_address an indexed indirect draw reads: max over its draws of (first index + count) * index size, from the
// arguments (and draw count) when the CPU can read them (not a held result, not written by a submission that has not completed).
// Otherwise the GPU writes them too and nobody knows the count here: everything from index_address to the end of its pool.
uint64_t indirect_index_span(const P_DrawIndirect& p) {
    const uint32_t isz0 = index_size_bytes(static_cast<int>(p.type));
    const uint64_t isz = isz0 ? isz0 : 4;
    const uint64_t stride = p.stride ? static_cast<uint64_t>(p.stride) : 20;
    const auto whole = [&] { const BufRef b = resolve_buffer(p.index_address, 1); return b.ok && b.avail ? static_cast<uint64_t>(b.avail) : uint64_t{1} << 20; };
    uint64_t n = 1;
    if (p.multi && p.count_address) {
        const uint8_t* c = out_overlaps(p.count_address, 4) ? nullptr : cpu_view(p.count_address, 4);
        if (!c) return whole();
        uint32_t v = 0;
        std::memcpy(&v, c, 4);
        n = std::min<uint64_t>(v, p.max_count);
    }
    if (!n) return 0;
    const uint64_t args = stride * (n - 1) + 20;
    const uint8_t* a = out_overlaps(p.indirect_address, args) ? nullptr : cpu_view(p.indirect_address, args);
    if (!a) return whole();
    uint64_t end = 0;
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t v[3];
        std::memcpy(v, a + i * stride, sizeof v);   // count, instance count, first index
        if (v[0] && v[1]) end = std::max(end, (static_cast<uint64_t>(v[2]) + v[0]) * isz);
    }
    return end;
}

// [nvn] compute_out_lazy: before op, the results it may read or overwrite through the pool go back (instead of all of them before any
// command that is not a dispatch). Ops that only touch images, state and syncs leave them in the copy; a fence's serial is the
// submission's, and out_end_submit decides at its end.
void out_before_op(const RecordReader& r) {
    if (!out_any()) return;
    switch (r.op()) {
        case Op::Dispatch: case Op::DispatchIndirect: case Op::Barrier: case Op::CallCommands: case Op::PushDebugGroup: case Op::PopDebugGroup:
        case Op::ClearColor: case Op::ClearDepthStencil: case Op::ClearTexture: case Op::CopyTextureToTexture: case Op::Downsample:
        case Op::DrawTexture: case Op::DiscardColor: case Op::DiscardDepthStencil: case Op::ResolveDepth: case Op::FenceSync: case Op::WaitSync:
        case Op::ResetCounter: case Op::WaitEvent: case Op::SetRenderEnable: case Op::EvaluateDlss:
        case Op::PushConstants:                                        // the Deferred form reads its data through cpu_view (never a held result)
            return;
        case Op::CopyBufferToTexture: case Op::CopyTextureToBuffer:    // do_copy_buffer_texture: its own range (it knows the size)
            return;
        case Op::DrawArrays:
        case Op::DrawElements: {
            out_draw_bindings();
            if (r.op() == Op::DrawElements) {
                const auto& p = r.payload<P_DrawElements>();
                const uint32_t isz = index_size_bytes(static_cast<int>(p.type));
                out_draw_read(p.index_address, static_cast<uint64_t>(p.count) * (isz ? isz : 4), true, kOutDrawBuffer);
            }
            return;
        }
        case Op::DrawIndirect: {
            out_draw_bindings();
            const auto& p = r.payload<P_DrawIndirect>();
            const bool indexed = p.type != 0xffffffffu;
            const uint64_t cmd = indexed ? 20 : 16;
            const uint64_t stride = p.stride ? static_cast<uint64_t>(p.stride) : cmd;
            const uint64_t n = p.multi && p.count_address ? std::max<uint64_t>(1, p.max_count) : 1;
            out_draw_read(p.indirect_address, stride * (n - 1) + cmd, true, kOutIndirect);   // do_draw_indirect binds the copy for them
            if (p.multi && p.count_address) out_draw_read(p.count_address, 4, true, kOutIndirect);
            if (indexed) compute_out_guard(p.index_address, indirect_index_span(p), kOutIndirect);   // the draw binds the pool for its indices
            return;
        }
        case Op::ClearBuffer: {
            const auto& p = r.payload<P_ClearBuffer>();
            compute_out_guard(p.address, p.size, kOutBufferWrite);
            return;
        }
        case Op::CopyBufferToBuffer: {
            const auto& p = r.payload<P_CopyBufferToBuffer>();
            out_draw_read(p.src, p.size, true, kOutBufferRead);       // the copy reads the pool copy (held_or_pool)
            compute_out_guard(p.dst, p.size, kOutBufferWrite);
            return;
        }
        case Op::UpdateUniformBuffer: {
            const auto& p = r.payload<P_UpdateUniformBuffer>();
            compute_out_guard(p.address + static_cast<uint64_t>(p.offset), p.size, kOutBufferWrite);
            return;
        }
        case Op::ReportCounter: compute_out_guard(r.payload<P_Counter>().address, 32, kOutBufferWrite); return;
        case Op::ReportValue: compute_out_guard(r.payload<P_ReportValue>().address, 32, kOutBufferWrite); return;
        case Op::SignalEvent: {
            const auto& p = r.payload<P_Event>();
            if (p.event && p.event->pool && p.event->pool->base)
                compute_out_guard(p.event->pool->gpu_base + static_cast<uint64_t>(p.event->offset), 4, kOutEvent);
            return;
        }
        case Op::DrawXfb: flush_compute_out(kOutIndirect); return;
        default:
            if (r.op() < Op::DrawArrays) return;    // state
            flush_compute_out(kOutOther);           // feedback begin/end and the rest: everything first
            return;
    }
}

void sh_barrier_once() {
    if (g_sh_barrier) return;
    memory_barrier(F->cb);       // earlier writes to these ranges (any command) before the copies read them
    g_sh_barrier = true;
}

// The buffer a dispatch binds for [addr, addr + size): the pool copy, the shadow (its copy-in recorded now), or `host` when it cannot have
// either.
BufRef shadow_of(uint64_t addr, VkDeviceSize size, const BufRef& host, bool writes) {
    if (!g_shadowing || !host.ok || size == 0) return host;
    for (ShadowUse& u : g_shadow_uses) {
        if (u.addr == addr && u.size == size) {           // the same range bound twice: one binding
            u.writes |= writes;
            BufRef r = host; r.buffer = u.buf; r.offset = u.off; r.avail = size;
            return r;
        }
        if (addr < u.addr + u.size && u.addr < addr + size) {   // overlapping ranges stay coherent in the pool
            compute_out_guard(addr, size, kOutDispatch);
            return host;
        }
    }
    // a result an earlier dispatch left in the pool copy: this dispatch binds it there again (the pages are not valid for the tracker
    // until the result goes back, so mirror_span would not say so)
    {
        VkBuffer ob = VK_NULL_HANDLE;
        VkDeviceSize oo = 0;
        if (out_exact(addr, size, &ob, &oo)) {
            ++g_sh_uses;
            ++g_sh_direct;
            g_shadow_uses.push_back({addr, size, host, ob, oo, writes, true, MirrorSpan{ob, oo, true}});
            BufRef r = host; r.buffer = ob; r.offset = oo; r.avail = size;
            return r;
        }
    }
    if (!out_lazy()) compute_out_guard(addr, size, kOutDispatch);
    if (size > kShadowMax) { compute_out_guard(addr, size, kOutDispatch); return host; }
    const MirrorSpan ms = mirror_span(addr, size);    // pages holding a compute result count as current there (compute_out_lazy)
    if (ms.buf && compute_direct_on()) {
        bool ok = ms.valid;
        if (!ok) {
            sh_barrier_once();
            uint64_t filled = 0;
            ok = mirror_fill(F->cb, addr, size, &filled);
            if (filled) { g_sh_copied = true; g_sh_bytes_fill += filled; }
            if (ok) ++g_sh_direct_filled;
        }
        if (ok) {
            ++g_sh_uses;
            ++g_sh_direct;
            g_shadow_uses.push_back({addr, size, host, ms.buf, ms.offset, writes, true, ms});
            BufRef r = host; r.buffer = ms.buf; r.offset = ms.offset; r.avail = size;
            return r;
        }
    }
    compute_out_guard(addr, size, kOutDispatch);      // the shadow reads the pool: held results there go back first
    const uint64_t key = addr ^ (static_cast<uint64_t>(size) << 40) ^ (static_cast<uint64_t>(size) >> 24);
    Shadow& sh = g_shadows[key];
    if (!sh.buf || sh.size != size) {
        if (sh.buf) { gfx::Buffer old = sh.buf; forget_buffer_views(old.buffer); defer_destroy([old]() mutable { device().destroy_buffer(old); }); sh.buf = gfx::Buffer{}; }
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (device().create_buffer(size, usage, gfx::MemoryUse::GpuOnly, &sh.buf, "nvn compute shadow") != VK_SUCCESS) {
            sh.buf = gfx::Buffer{};
            g_shadows.erase(key);
            return host;
        }
        sh.size = size;
    }
    sh.last = F->serial;
    sh_barrier_once();
    // The device-local pool copy, when it holds the range, is the cheap source (video memory to video memory instead of over PCIe); after a
    // dispatch that writes the range, the result goes to both, so the next dispatch finds the copy valid again (mirror_span_written).
    ++g_sh_uses;
    if (!ms.buf) ++g_sh_no_span;
    if (ms.valid) { ++g_sh_from_mirror; g_sh_bytes_mirror += size; } else g_sh_bytes_pool += size;
    VkBufferCopy c = {ms.valid ? ms.offset : host.offset, 0, size};
    vkCmdCopyBuffer(F->cb, ms.valid ? ms.buf : host.buffer, sh.buf.buffer, 1, &c);
    g_sh_copied = true;
    g_shadow_uses.push_back({addr, size, host, sh.buf.buffer, 0, writes, false, ms});
    BufRef r = host;
    r.buffer = sh.buf.buffer;
    r.offset = 0;
    r.avail = size;
    return r;
}

// After the dispatch: the written shadows go back to the pool; written direct ranges wait in g_out_pending. Shadows unused for a while
// are freed.
void shadows_after_dispatch() {
    bool any = false;
    for (const ShadowUse& u : g_shadow_uses) {
        if (!u.writes) continue;
        if (u.direct) {
            if (out_add(u.addr, u.size, u.buf, u.off, u.host)) ++g_sh_merged;   // written again before it went back: one copy out for both
            continue;
        }
        if (!any) { memory_barrier(F->cb); any = true; }
        VkBufferCopy c = {0, u.host.offset, u.size};
        vkCmdCopyBuffer(F->cb, u.buf, u.host.buffer, 1, &c);
        g_sh_bytes_out += u.size;
        if (u.mirror.buf) {
            VkBufferCopy mc = {0, u.mirror.offset, u.size};
            vkCmdCopyBuffer(F->cb, u.buf, u.mirror.buf, 1, &mc);
            mirror_span_written(u.addr, u.size);
        }
    }
    if (any) gstamp("shadow out");
    g_shadow_uses.clear();
    g_shadowing = false;
    static uint64_t last_sweep = 0;
    if (F->serial - last_sweep > 600) {
        last_sweep = F->serial;
        NXLOG_I("nvn.replay", "compute shadows: %llu bound (%llu from the pool copy, %llu with no copy), in from the copy %llu MiB, in from the pool %llu MiB, out %llu MiB; "
                "direct to the pool copy %llu (%llu after filling pages, in %llu MiB, out %llu MiB in %llu flushes, %llu writes merged)",
                (unsigned long long)g_sh_uses, (unsigned long long)g_sh_from_mirror, (unsigned long long)g_sh_no_span, (unsigned long long)(g_sh_bytes_mirror >> 20),
                (unsigned long long)(g_sh_bytes_pool >> 20), (unsigned long long)(g_sh_bytes_out >> 20), (unsigned long long)g_sh_direct,
                (unsigned long long)g_sh_direct_filled, (unsigned long long)(g_sh_bytes_fill >> 20), (unsigned long long)(out_bytes_total() >> 20),
                (unsigned long long)g_sh_flushes, (unsigned long long)g_sh_merged);
        for (auto it = g_shadows.begin(); it != g_shadows.end();) {
            if (F->serial - it->second.last > 600) {
                gfx::Buffer old = it->second.buf;
                forget_buffer_views(old.buffer);
                defer_destroy([old]() mutable { device().destroy_buffer(old); });
                it = g_shadows.erase(it);
            } else {
                ++it;
            }
        }
    }
}

Texture* g_su_tex = nullptr;                 // scene upscaler override (S14), see scene_upscale_for_draw
VkImageView g_su_view = VK_NULL_HANDLE;
Ref<Texture> g_tsr_depth;                   // temporal upscaling: the depth target of the G-buffer draw that gave this frame's camera
uint64_t g_tsr_frame = 0;                    // temporal upscaling: tone-map draws so far (the jitter phase of the frame being drawn)
// Temporal upscaling: a copy of a vertex slot 0 camera block with this frame's jitter added (extras::temporal_jitter_block). False: left alone.
bool tsr_jitter(uint32_t* words, size_t n_words) {
    Texture* d = S.depth_tex.get();
    if (!d || !tex_gpu(d)) return false;
    const VkExtent3D e = tex_image_extent(d, 0);
    return extras::temporal_jitter_block(words, n_words, e.width, e.height, g_tsr_frame);
}

// build_set's vectors keep their capacity from call to call (allocated and freed for every set of every draw, they were ~10 % of the render
// thread in Dagsion, notes/perf.md); a call made while another one builds takes the next scratch.
struct SetScratch {
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> binfo;
    std::vector<VkDescriptorImageInfo> iinfo;
    std::vector<VkBufferView> tinfo;
};
thread_local std::vector<std::unique_ptr<SetScratch>> t_set_scratch;
thread_local size_t t_set_depth = 0;

// Set 0 of a ProgLayout::push_set0 program: build_set leaves its writes here and bind_sets pushes them once every set is built (building a
// later set may record copies, or start another command buffer; the push goes into the one the draw is recorded in). Render thread only.
struct PushSet0 {
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> binfo;
    bool ready = false;
};
PushSet0 g_push0;
std::vector<std::unique_ptr<PushSet0>> g_push_stack;   // bind_sets takes set 0's writes out of g_push0 at once, into the one of its depth
size_t g_bind_depth = 0;

VkDescriptorSet build_set(const ProgLayout& pl, uint32_t set) {
    if (t_set_scratch.size() <= t_set_depth) t_set_scratch.push_back(std::make_unique<SetScratch>());
    SetScratch& sc = *t_set_scratch[t_set_depth];
    struct Depth { Depth() { ++t_set_depth; } ~Depth() { --t_set_depth; } } depth;
    std::vector<VkWriteDescriptorSet>& writes = sc.writes;
    std::vector<VkDescriptorBufferInfo>& binfo = sc.binfo;
    std::vector<VkDescriptorImageInfo>& iinfo = sc.iinfo;
    std::vector<VkBufferView>& tinfo = sc.tinfo;
    writes.clear();
    binfo.clear();
    iinfo.clear();
    tinfo.clear();
    const size_t nb = pl.bindings.size();
    binfo.reserve(nb);
    iinfo.reserve(nb);
    tinfo.reserve(nb);
    uint64_t key = 1469598103934665603ull + set;
    auto mix = [&](uint64_t v) { key = (key ^ v) * 1099511628211ull; };
    mix(reinterpret_cast<uint64_t>(pl.layout));
    for (const BindingDesc& d : pl.bindings) {
        if (d.set != set) continue;
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstBinding = d.binding;
        w.descriptorCount = 1;
        w.descriptorType = d.type;
        if (d.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || d.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            VkDescriptorBufferInfo bi = {dummy_buffer(), 0, 65536};
            switch (b.nvn.source) {
                case NvnSource::SupportBuffer:
                    bi = g_draw_support ? VkDescriptorBufferInfo{g_draw_support.buffer, g_draw_support.offset, sizeof(SupportBuffer)}
                                        : VkDescriptorBufferInfo{g_support.buffer, g_support.offset, sizeof(SupportBuffer)};
                    break;
                case NvnSource::DriverConstants: {
                    if (d.nvn_stage == kCompute && g_cs_c0) {        // NVN2 compute: this dispatch's own (do_dispatch)
                        bi = {g_cs_c0.buffer, g_cs_c0.offset, g_cs_c0.size};
                        break;
                    }
                    VkDeviceSize range = 0;
                    VkBuffer buf = driver_constants_buffer(&range);
                    if (buf) bi = {buf, 0, range};
                    break;
                }
                case NvnSource::PushConstants: {                     // graphics push constant slot n (Op::PushConstants)
                    const int slot = b.nvn.index;
                    if (slot < 0 || slot >= 8) break;
                    UploadAlloc& u = g_push_buf[slot];
                    if (!u) {
                        u = F->upload(256, std::max<VkDeviceSize>(device().properties().limits.minUniformBufferOffsetAlignment, 256));
                        if (u) std::memcpy(u.ptr, S.push_gfx[slot], 256);
                    }
                    if (u) bi = {u.buffer, u.offset, 256};
                    break;
                }
                case NvnSource::ConstantData: {
                    VkDeviceSize range = 0;
                    VkBuffer buf = constant_data_buffer(d.shader, &range);
                    if (buf) bi = {buf, 0, range};
                    break;
                }
                case NvnSource::UniformBlock: {
                    const int idx = b.nvn.index;
                    if (idx >= 0 && idx < 16 && S.ubo[d.nvn_stage][idx].addr) {
                        const GState::Bind& bind = S.ubo[d.nvn_stage][idx];
                        if (d.nvn_stage == kCompute || !out_lazy()) {                   // draws: checked before the op (out_before_op)
                            const VkDeviceSize ua = device().properties().limits.minUniformBufferOffsetAlignment;
                            out_draw_read(bind.addr, bind.size ? bind.size : 65536, ua && (bind.addr % ua) == 0, kOutDrawBuffer);   // resolve_read serves the copy
                        }
                        BufRef br = resolve_read(bind.addr, bind.size, kReadUniform);
                        const VkDeviceSize align = device().properties().limits.minUniformBufferOffsetAlignment;
                        const float ufs = ufs_scale_for(d, idx);
                        const bool tsr = d.nvn_stage == kVertex && idx == 0 && bind.size >= 272 && extras::temporal_wanted() && S.depth_tex;
                        if (br.ok && (ufs != 1.0f || tsr)) {
                            // the game's scale on some floats of this block (nxrt/gfx_tuning.h): bound from a patched copy in the upload ring
                            const VkDeviceSize n = std::min<VkDeviceSize>({bind.size ? bind.size : 65536, br.avail, 65536});
                            UploadAlloc u = F->upload(n, std::max<VkDeviceSize>(align, 256));
                            if (u) {
                                std::memcpy(u.ptr, host_ptr(bind.addr), n);
                                if (tsr && n >= 272) tsr_jitter(reinterpret_cast<uint32_t*>(u.ptr), static_cast<size_t>(n / 4));   // temporal upscaling
                                const uint32_t mask = ufs != 1.0f ? nxrt::gfx::tuning().ufs_mask.load(std::memory_order_relaxed) : 0u;
                                for (uint32_t i = 0; i < 32; ++i) {
                                    if (!(mask & (1u << i)) || (i + 1) * 4 > n) continue;
                                    float v;
                                    std::memcpy(&v, static_cast<uint8_t*>(u.ptr) + i * 4, 4);
                                    v *= ufs;
                                    std::memcpy(static_cast<uint8_t*>(u.ptr) + i * 4, &v, 4);
                                }
                                bi = {u.buffer, u.offset, n};
                            }
                        } else if (br.ok && (br.offset % align) == 0) {
                            bi = {br.buffer, br.offset, std::min<VkDeviceSize>({bind.size ? bind.size : 65536, br.avail, 65536})};
                        } else if (br.ok) {
                            // unaligned binding: copy into the upload ring
                            const VkDeviceSize n = std::min<VkDeviceSize>(bind.size ? bind.size : 65536, 65536);
                            UploadAlloc u = F->upload(n, std::max<VkDeviceSize>(align, 256));
                            if (u) {
                                std::memcpy(u.ptr, host_ptr(bind.addr), n);
                                bi = {u.buffer, u.offset, n};
                            }
                        }
                    }
                    break;
                }
                case NvnSource::StorageBuffer: {
                    const int slot = b.nvn.index >= 0 ? b.nvn.index : storage_slot_for(d.nvn_stage, b.nvn.c0_offset);   // NVN2 compute: index from the c[0] table
                    if (slot >= 0 && slot < 16 && S.ssbo[d.nvn_stage][slot].addr) {
                        const GState::Bind& bind = S.ssbo[d.nvn_stage][slot];
                        BufRef br = resolve_buffer(bind.addr, bind.size);
                        bi = {br.buffer, br.offset, std::min<VkDeviceSize>(bind.size ? bind.size : br.avail, br.avail)};
                        if (!g_shadowing && br.ok && out_lazy() && out_overlaps(bind.addr, bi.range)) {   // a draw over a held compute result: the copy
                            const BufRef hb = held_or_pool(bind.addr, bi.range, kOutStorageXfb, false);
                            if (hb.buffer != br.buffer) {
                                bi.buffer = hb.buffer;
                                bi.offset = hb.offset;
                                gpu_wrote(bind.addr, bi.range, kWriteStorage);
                                out_add(bind.addr, bi.range, hb.buffer, hb.offset, br);   // what the draw writes there is held like a dispatch's
                                break;
                            }
                        }
                        if (g_shadowing && br.ok) {                                  // before gpu_wrote: the shadow reads the pool copy while valid
                            const BufRef sh = shadow_of(bind.addr, bi.range, br, true);
                            bi.buffer = sh.buffer;
                            bi.offset = sh.offset;
                        }
                        if (br.ok) gpu_wrote(bind.addr, bi.range, kWriteStorage);      // a storage buffer may be written by the shader
                    }
                    break;
                }
                case NvnSource::UniformAsStorage: {
                    // a compute shader that reads uniform block n through its c[0] descriptor: the block's buffer as a storage buffer
                    const int idx = b.nvn.index;
                    if (idx >= 0 && idx < 16 && S.ubo[d.nvn_stage][idx].addr) {
                        const GState::Bind& bind = S.ubo[d.nvn_stage][idx];
                        const BufRef br0 = resolve_buffer(bind.addr, bind.size);
                        const VkDeviceSize range = std::min<VkDeviceSize>(bind.size ? bind.size : br0.avail, br0.avail);
                        const BufRef br = held_or_pool(bind.addr, range, kOutDispatch, true);
                        bi = {br.buffer, br.offset, range};
                    }
                    break;
                }
                default:
                    note_unsupported("shader buffer binding of an unknown NVN source");
                    break;
            }
            binfo.push_back(bi);
            w.pBufferInfo = &binfo.back();
            mix(reinterpret_cast<uint64_t>(bi.buffer)); mix(bi.offset); mix(bi.range);
        } else if (d.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            const uint8_t int_comp = int_sampler_for(pl, d);    // the integer sampler variant of a compute shader (do_dispatch)
            const int component = int_comp ? int_comp : b.image.component == ScalarType::Sint ? 2 : b.image.component == ScalarType::Uint ? 3 : 1;
            VkDescriptorImageInfo ii = {dummy_sampler(), dummy_view(view_type_for(b), b.image.depth, component), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            TexResolved r = resolve_texture(d);
            if (r.tex && !g_fb_use.empty()) {
                const auto fb = g_fb_use.find(r.tex);
                if (fb != g_fb_use.end()) r.tex = fb->second;      // a feedback draw samples the copy made before it
            }
            if (r.tex && !g_vol3d_use.empty() && wants_vol3d(b, r)) {   // a sampler3D over 2D storage samples the 3D copy (vol3d_prepare)
                const auto vc = g_vol3d_use.find(r.tex);
                if (vc != g_vol3d_use.end()) {
                    r.tex = vc->second;
                    r.view.has_layers = false;                     // the copy is the chosen layer, one slice deep
                }
            }
            // [nvn] debug_null_format = "BPTC_UFLOAT" or "RGBA8:256x64": every sampled texture of that NVN format (and size) gets the dummy view
            // (term isolation, debugging only)
            static const std::string null_fmt = nxrt::config_get_str("nvn", "debug_null_format", "");
            if (r.tex && !null_fmt.empty() && r.tex->fmt) {
                const std::string sized = std::string(r.tex->fmt->name) + ":" + std::to_string(r.tex->b.width) + "x" + std::to_string(r.tex->b.height);
                if (null_fmt == r.tex->fmt->name || null_fmt == sized) r.tex = nullptr;
            }
            // a compute dispatch is never inside the graphics pass: a colour attachment of the last pass is an ordinary texture to it
            const bool feedback = r.tex && d.nvn_stage != kCompute && is_color_feedback(r.tex);
            // a first upload held back for a later frame ([nvn] bound_upload_defer): its image holds nothing yet, the dummy stands in
            const bool waiting = r.tex && tex_layout(r.tex) == VK_IMAGE_LAYOUT_UNDEFINED && tex_first_upload_deferrable(r.tex);
            if (r.tex && !waiting && tex_gpu(r.tex) && !feedback) {
                ViewRequest q = make_request(b, r);
                if (int_comp) q.component = int_comp;
                VkImageView v = tex_sampled_view(r.tex, q);
                if (v) {
                    ii.imageView = v;
                    // GENERAL: the op also writes it as a storage image (prepare_textures left it there)
                    ii.imageLayout = tex_layout(r.tex) == VK_IMAGE_LAYOUT_GENERAL ? VK_IMAGE_LAYOUT_GENERAL : tex_sampled_layout(r.tex);
                    if (r.has_sampler) ii.sampler = sampler_for(int_comp ? unfiltered(r.smp) : r.smp);
                    if (r.tex == g_su_tex && g_su_view) {            // the upscaled scene replaces the small one for this draw (S14)
                        ii.imageView = g_su_view;
                        ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    }
                } else if (first_time_id(1, d.shader, static_cast<uint64_t>(d.index), r.tex->id)) {
                    NXLOG_W(kTag, "sampled texture %u (%s, target %d, %dx%dx%d) has no view of the shader's view type %d, sampled type %d: shader %s binding %s (NVN slot %d, set %u binding %u) gets the dummy view",
                            r.tex->id, r.tex->fmt ? r.tex->fmt->name : "?", r.tex->b.target, r.tex->b.width, r.tex->b.height, r.tex->b.depth,
                            static_cast<int>(view_type_for(b)), static_cast<int>(q.component), d.shader->hash.c_str(), b.name.c_str(), b.nvn.index, d.set, d.binding);
                }
            } else if (feedback) {
                NVN_LOG_ONCE("color-feedback", "a texture is sampled while it is a colour attachment of the pass: the dummy texture is bound");
                static int reported = 0;
                if (reported < 40) {   // which draw: the texture, the attachment's subresource, the sampled view, the shaders (S4g)
                    ++reported;
                    char tb[96];
                    uint32_t lvl = 0, lay = 0, nlay = 0;
                    for (uint32_t i = 0; i < S.ncolor; ++i) if (S.color_tex[i].get() == r.tex) { lvl = S.color_level[i]; lay = S.color_layer[i]; nlay = S.color_layers[i]; }
                    NXLOG_I("nvn.replay", "feedback: %s binding %s vs %s fs %s | attachment level %u layer %u layers %u | sampled view levels %d+%d layers %d+%d", tex_desc(r.tex, tb, sizeof tb),
                            b.name.c_str(), S.stage[kVertex] ? S.stage[kVertex]->hash.c_str() : "-", S.stage[kFragment] ? S.stage[kFragment]->hash.c_str() : "-", lvl, lay, nlay,
                            r.has_view && r.view.has_levels ? r.view.min_level : -1, r.has_view && r.view.has_levels ? r.view.num_levels : -1,
                            r.has_view && r.view.has_layers ? r.view.min_layer : -1, r.has_view && r.view.has_layers ? r.view.num_layers : -1);
                }
            }
            iinfo.push_back(ii);
            w.pImageInfo = &iinfo.back();
            mix(reinterpret_cast<uint64_t>(ii.imageView));
            mix(reinterpret_cast<uint64_t>(ii.sampler));
            mix(static_cast<uint64_t>(ii.imageLayout));
        } else if (d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
            const VkFormat shader_fmt = image_format_vk(b.image.format);
            VkDescriptorImageInfo ii = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL};
            TexResolved r = resolve_image(d);
            if (r.tex && tex_gpu(r.tex) && tex_has_storage(r.tex)) {
                ViewRequest q = make_request(b, r);
                q.swizzle = {};
                if (shader_fmt != VK_FORMAT_UNDEFINED) q.format = shader_fmt;
                q.format = tex_storage_format(r.tex, q.format);   // an sRGB image is stored to through its UNORM view (raw bits, as on the console)
                ii.imageView = tex_sampled_view(r.tex, q);
            }
            if (!ii.imageView) {
                if (r.tex) NXLOG_W(kTag, "storage image: texture %u (%s %dx%dx%d) has no storage view (gpu %d, storage usage %d)", r.tex->id, r.tex->fmt ? r.tex->fmt->name : "?",
                                   r.tex->b.width, r.tex->b.height, r.tex->b.depth, tex_gpu(r.tex) ? 1 : 0, tex_has_storage(r.tex) ? 1 : 0);
                NVN_LOG_ONCE("storage-image-dummy", "a storage image binding has no usable texture: a dummy storage image is bound");
                ii.imageView = dummy_storage_view(view_type_for(b), shader_fmt);
            }
            iinfo.push_back(ii);
            w.pImageInfo = &iinfo.back();
            mix(reinterpret_cast<uint64_t>(ii.imageView));
        } else if (d.type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || d.type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
            // NVN BUFFER-target textures (samplerBuffer / imageBuffer): a texel buffer view over the pool memory that holds the texels
            const bool storage = d.type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
            const VkFormat shader_fmt = storage ? image_format_vk(b.image.format) : VK_FORMAT_UNDEFINED;
            const int component = storage ? 0 : b.image.component == ScalarType::Float ? 1 : b.image.component == ScalarType::Sint ? 2
                                  : b.image.component == ScalarType::Uint ? 3 : 0;
            TexResolved r = storage ? resolve_image(d) : resolve_texture(d);
            uint64_t addr = 0, bytes = 0;
            VkFormat vfmt = VK_FORMAT_UNDEFINED;
            VkBufferView v = r.tex ? tex_buffer_view(r.tex, r.has_view ? &r.view : nullptr, component, shader_fmt, storage, &addr, &bytes, &vfmt) : VK_NULL_HANDLE;
            // a 2D/3D texture the game left in a samplerBuffer slot (e.g. KTGL's 8x8 default texture 3 in the foliage wind shader's
            // interaction buffers): read as a run of texels over its storage, as the console does, instead of the dummy
            if (v && r.tex->b.target != NVN_TEXTURE_TARGET_BUFFER && first_time_id(2, d.shader, static_cast<uint64_t>(d.index), r.tex->id))
                NXLOG_I(kTag, "texture %u (%s %dx%dx%d, target %d) is read as a texel buffer by %s binding %s: it gets a buffer view over its storage (%llu bytes)",
                        r.tex->id, r.tex->fmt ? r.tex->fmt->name : "?", r.tex->b.width, r.tex->b.height, r.tex->b.depth, r.tex->b.target,
                        d.shader->hash.c_str(), b.name.c_str(), static_cast<unsigned long long>(bytes));
            if (v && g_shadowing) {                                        // before gpu_wrote: the shadow reads the pool copy while valid
                const BufRef host = resolve_buffer(addr, bytes);
                const BufRef sh = shadow_of(addr, bytes, host, storage);
                if (sh.buffer != host.buffer)
                    if (VkBufferView sv = make_buffer_view(sh.buffer, sh.offset, bytes, vfmt)) v = sv;   // offset: non-zero in the pool copy
            }
            if (v && !g_shadowing && out_lazy() && out_overlaps(addr, bytes)) {   // a draw reads a held compute result as texels: from the copy
                const MirrorSpan ms = out_servable(addr, bytes) ? mirror_span(addr, bytes) : MirrorSpan{};
                VkBufferView mv = ms.buf ? make_buffer_view(ms.buf, ms.offset, bytes, vfmt) : VK_NULL_HANDLE;
                if (mv) v = mv;
                else NVN_LOG_ONCE("out-texel", "a draw reads a compute result held in the pool copy as a texel buffer the copy cannot serve: it reads the pool");
            }
            if (v && storage) gpu_wrote(addr, bytes, kWriteStorage);      // an image buffer may be written by the shader
            if (!v) {
                if (r.tex && first_time_id(3, d.shader, static_cast<uint64_t>(d.index), r.tex->id))
                    NXLOG_W(kTag, "texel buffer: texture %u (%s, target %d, %dx%dx%d, storage %llu bytes at %p) has no buffer view: shader %s binding %s (NVN slot %d, set %u binding %u)",
                            r.tex->id, r.tex->fmt ? r.tex->fmt->name : "?", r.tex->b.target, r.tex->b.width, r.tex->b.height, r.tex->b.depth,
                            static_cast<unsigned long long>(r.tex->storage_size), static_cast<void*>(r.tex->mem), d.shader->hash.c_str(), b.name.c_str(),
                            b.nvn.index, d.set, d.binding);
                NVN_LOG_ONCE("texel-buffer-dummy", "a texel buffer binding has no usable buffer texture: a dummy buffer view is bound");
                v = dummy_buffer_view(shader_fmt, storage);
            }
            if (!v) {
                note_unsupported("texel buffer descriptor without any view (the binding stays unwritten)");
                continue;
            }
            tinfo.push_back(v);
            w.pTexelBufferView = &tinfo.back();
            mix(reinterpret_cast<uint64_t>(v));
        } else {
            note_unsupported("descriptor of an unhandled type (the binding stays unwritten)");
            continue;
        }
        writes.push_back(w);
    }
    ++g_rs.sets;
    if (set == 0 && pl.push_set0) {   // pushed by bind_sets: no set to find or make (every write of set 0 is a buffer, prog_layout)
        g_push0.binfo.assign(binfo.begin(), binfo.end());
        g_push0.writes.assign(writes.begin(), writes.end());
        for (VkWriteDescriptorSet& w : g_push0.writes) {
            w.dstSet = VK_NULL_HANDLE;
            if (w.pBufferInfo) w.pBufferInfo = g_push0.binfo.data() + (w.pBufferInfo - binfo.data());
        }
        g_push0.ready = true;
        g_rs.set_writes += writes.size();
        return VK_NULL_HANDLE;
    }
    if (const VkDescriptorSet hit = g_set_cache.find(key)) { ++g_rs.set_hits; return hit; }
    if (rs_on()) {   // replay_stats: what differs from the last set made for this layout
        struct D { uint32_t binding, type; uint64_t a, b, c; };
        static std::unordered_map<VkDescriptorSetLayout, std::vector<D>> last;
        static std::vector<D> cur;                       // reused: the stats line must not allocate per miss
        cur.clear();
        for (const VkWriteDescriptorSet& w : writes) {
            D d{w.dstBinding, static_cast<uint32_t>(w.descriptorType), 0, 0, 0};
            if (w.pBufferInfo) { d.a = reinterpret_cast<uint64_t>(w.pBufferInfo->buffer); d.b = w.pBufferInfo->offset; d.c = w.pBufferInfo->range; }
            else if (w.pImageInfo) { d.a = reinterpret_cast<uint64_t>(w.pImageInfo->imageView); d.b = reinterpret_cast<uint64_t>(w.pImageInfo->sampler); d.c = w.pImageInfo->imageLayout; }
            else if (w.pTexelBufferView) d.a = reinterpret_cast<uint64_t>(*w.pTexelBufferView);
            cur.push_back(d);
        }
        std::vector<D>& l = last[pl.sets[set]];
        if (l.size() != cur.size()) ++g_rs.miss_first;
        else {
            bool ofs = false, buf = false, tex = false, other = false;
            for (size_t i = 0; i < cur.size(); ++i) {
                const D& x = cur[i]; const D& y = l[i];
                if (x.binding != y.binding || x.type != y.type) { other = true; continue; }
                if (x.a == y.a && x.b == y.b && x.c == y.c) continue;
                if (x.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) (x.a == y.a ? ofs : buf) = true;
                else if (x.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) tex = true;
                else other = true;
            }
            if (other) ++g_rs.miss_other;
            else if (tex) ++g_rs.miss_tex;
            else if (buf) ++g_rs.miss_ubo;
            else if (ofs) ++g_rs.miss_uboofs;
            else ++g_rs.miss_first;            // the same set as the last one (a new submission)
        }
        l.assign(cur.begin(), cur.end());
    }
    RsScope rs_alloc(kRsAlloc);   // replay_stats: the miss path (draw sets and dispatch sets)
    VkDescriptorSet ds = F->alloc_set(pl.sets[set]);
    if (!ds) return VK_NULL_HANDLE;
    g_rs.set_writes += writes.size();
    for (auto& w : writes) w.dstSet = ds;
    vkUpdateDescriptorSets(vkdev(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    g_set_cache.put(key, ds);
    return ds;
}

// ---- resolution scale: the support buffer's render_scale for a draw or dispatch (Ryujinx TextureBindingsManager.UpdateRenderScale) ----
// The translator scales texelFetch / imageLoad coordinates by render_scale[1 + i] and divides textureSize and gl_FragCoord.xy by the scale
// of texture i and of the render target ([0]). Texture i is the shader's i-th texture descriptor: the harness numbers single textures
// with one counter per shader (binding = stage slot * 128 + counter, + 64 for texel buffers), images follow the textures. Vertex shaders
// read their scales after the fragment shader's (frag_scale_count). Storage images are never scaled (tex_rescale), so images read 1.
void support_for_draw(const ProgLayout& pl, bool compute) {
    g_draw_support = UploadAlloc();
    if (render_scale() == 1.0f) return;
    float sc[73];
    for (float& v : sc) v = 1.0f;
    int frag_count = 0;
    bool any = false;
    if (!compute && P.scale != 1.0f) { sc[0] = P.scale; any = true; }
    // the fragment shader's textures first, then the vertex shader's after them
    const ShaderInfo* fs = compute ? nullptr : S.stage[kFragment];
    const ShaderInfo* vs = compute ? S.stage[kCompute] : S.stage[kVertex];
    auto tex_count = [](const ShaderInfo* sh) {
        int n = 0;
        for (const DescriptorBinding& b : sh->refl.bindings)
            if (b.count == 1 && (b.type == DescriptorType::CombinedImageSampler || b.type == DescriptorType::SampledImage || b.type == DescriptorType::UniformTexelBuffer ||
                                 b.type == DescriptorType::StorageImage || b.type == DescriptorType::StorageTexelBuffer))
                ++n;
        return n;
    };
    if (fs) frag_count = tex_count(fs);
    // the native size of the render target, for the fragment rule below
    Texture* rt = S.ncolor && S.color_tex[0] ? S.color_tex[0].get() : S.depth_tex.get();
    for (const BindingDesc& d : pl.bindings) {
        if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        if (b.count != 1 || b.binding >= 640) continue;
        const bool is_fs = d.shader == fs;
        if (!is_fs && d.shader != vs) continue;
        if (b.nvn.source != NvnSource::SamplerBinding && b.nvn.source != NvnSource::HandleInUniform) continue;
        TexResolved r = resolve_texture(d);
        if (!r.tex) continue;
        if (!g_fb_use.empty()) {
            const auto fb = g_fb_use.find(r.tex);
            if (fb != g_fb_use.end()) r.tex = fb->second;
        }
        float s = tex_scale(r.tex);
        if (s == 1.0f) continue;
        const int idx = static_cast<int>(b.binding % 64u);
        const int slot = is_fs ? 1 + idx : (compute ? 1 + idx : 1 + frag_count + idx);
        if (slot >= 73) continue;
        if (is_fs && rt) {
            // Ryujinx: a texture whose size is the render target's size times one factor on both axes interpolates the scaled pixels
            // with gl_FragCoord (a negative scale tells TexelFetchScale to)
            const float rw = static_cast<float>(rt->b.width), rh = static_cast<float>(rt->b.height);
            if (r.tex->b.width > 0 && r.tex->b.height > 0 && rw / static_cast<float>(r.tex->b.width) == rh / static_cast<float>(r.tex->b.height)) s = -s;
        }
        sc[slot] = s;
        any = true;
    }
    if (!any) return;
    uint64_t key = 1469598103934665603ull + static_cast<uint64_t>(frag_count);
    for (float v : sc) { uint32_t u; std::memcpy(&u, &v, 4); key = (key ^ u) * 1099511628211ull; }
    auto it = g_support_scaled.find(key);
    if (it != g_support_scaled.end()) { g_draw_support = it->second; return; }
    UploadAlloc u = F->upload(sizeof(SupportBuffer), 256);
    if (!u) return;
    SupportBuffer sb = support_buffer_defaults();
    sb.frag_scale_count[0] = frag_count;
    for (int i = 0; i < 73; ++i) sb.render_scale[i][0] = sc[i];
    std::memcpy(u.ptr, &sb, sizeof sb);
    g_support_scaled[key] = u;
    g_draw_support = u;
}

// [nvn] tex_set_skip (default on): a draw whose texture inputs did not change since the last draw bound its sets (same layout, same command
// buffer, no texture / image / pool bind, nothing that marks everything dirty: a pass end, an upload or layout change, a dispatch, a clear
// drawn with its own pipeline, a new command buffer; the pools' registration versions the same) leaves the sets bound that hold only plain
// bound textures (SamplerBinding combined image samplers: no buffer, no handle read from a uniform block, no side effect in build_set). Set 0
// (buffers) is built and pushed as before. Most street draws change only uniform buffers between draws of the same material.
// [nvn] rt_check builds the skipped sets anyway and checks that the content cache gives the very set that is bound.
struct LastGfxSets {
    bool valid = false;
    const ProgLayout* pl = nullptr;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    TexturePool* tp = nullptr;
    SamplerPool* sp = nullptr;
    uint64_t tv = 0, sv = 0;
    VkDescriptorSet ds[kPoolSet + 1] = {};
};
LastGfxSets g_last_gfx;
struct SkipMaskSlot { const ProgLayout* pl = nullptr; uint32_t mask = 0; bool uniform_handles = false; };
SkipMaskSlot g_skip_mask[256];
bool tex_set_skip_on() {
    static const bool on = nxrt::config_get_bool("nvn", "tex_set_skip", true);
    return on;
}
uint64_t pool_version(TexturePool* p) {
    if (!p) return 0;
    std::shared_lock<std::shared_mutex> lk(p->lock);
    return p->version;
}
uint64_t pool_version(SamplerPool* p) {
    if (!p) return 0;
    std::shared_lock<std::shared_mutex> lk(p->lock);
    return p->version;
}
// bit s: set s of the layout holds only SamplerBinding combined image samplers (kPoolSet: the shared pool arrays, which pool_state_moved
// watches: it marks kDesc when they change)
uint32_t skippable_sets(const ProgLayout& pl) {
    SkipMaskSlot& sl = g_skip_mask[(reinterpret_cast<uintptr_t>(&pl) >> 4) & 255];
    if (sl.pl == &pl) return sl.mask;
    uint32_t ok = 0, bad = 0;
    bool uh = false;
    for (const BindingDesc& d : pl.bindings) {
        if (d.set > kPoolSet) continue;
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        uh |= b.nvn.source == NvnSource::HandleInUniform;
        const bool plain = d.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && b.nvn.source == NvnSource::SamplerBinding;
        (plain ? ok : bad) |= 1u << d.set;
    }
    uint32_t mask = ok & ~bad & ~1u;                 // never set 0 (the buffers)
    if (pl.has_set[kPoolSet]) mask |= 1u << kPoolSet;
    sl = {&pl, mask, uh};
    return mask;
}
// The layout reads a texture or image handle from a uniform block (a buffer bind can change what it samples).
bool uniform_handles(const ProgLayout& pl) {
    skippable_sets(pl);
    return g_skip_mask[(reinterpret_cast<uintptr_t>(&pl) >> 4) & 255].uniform_handles;
}
// [nvn] prep_skip (default on): the texture preparation of a draw (attachments, samples_depth_target, prepare_textures, the scene
// upscaler check, feedback_prepare) is left out when only buffer bindings changed since the last draw that ran it with the same layout in the
// same command buffer: textures, pools, programs and targets are what it looks at, and any change to them (or a pass end, an upload, a
// dispatch: kAllDirty) runs it again. A draw with nothing dirty already skipped it; this extends that to draws that rebind uniform buffers.
// Not for layouts that read handles from uniform blocks.
const ProgLayout* g_prep_pl = nullptr;
VkCommandBuffer g_prep_cb = VK_NULL_HANDLE;
bool prep_skip_on() {
    static const bool on = nxrt::config_get_bool("nvn", "prep_skip", true);
    return on;
}

void bind_sets(const ProgLayout& pl, VkPipelineBindPoint bp) {
    VkDescriptorSet ds[kPoolSet + 1] = {};
    // the texture sets this draw may leave bound (graphics only)
    uint32_t skip = 0;
    TexturePool* const tp = S.tpool.get();
    SamplerPool* const sp = S.spool.get();
    uint64_t tv = 0, sv = 0;
    const bool gfx = bp == VK_PIPELINE_BIND_POINT_GRAPHICS && tex_set_skip_on();
    if (gfx) {
        tv = pool_version(tp);
        sv = pool_version(sp);
        if (g_last_gfx.valid && !(g_dirty & kDescTex) && g_last_gfx.pl == &pl && g_last_gfx.cb == F->cb && g_last_gfx.tp == tp && g_last_gfx.sp == sp &&
            g_last_gfx.tv == tv && g_last_gfx.sv == sv && g_fb_use.empty() && g_vol3d_use.empty() && !g_su_tex && !g_fb_seen)
            skip = skippable_sets(pl);
    }
    if (g_push_stack.size() <= g_bind_depth) g_push_stack.push_back(std::make_unique<PushSet0>());
    PushSet0& push0 = *g_push_stack[g_bind_depth];
    struct Depth { Depth() { ++g_bind_depth; } ~Depth() { --g_bind_depth; } } depth;
    push0.ready = false;
    for (uint32_t set = 0; set < 4; ++set) {
        if (!pl.has_set[set]) continue;
        if (skip & (1u << set)) {
            ++g_rs.sets_skipped;
            if (rt_check()) chk(build_set(pl, set) == g_last_gfx.ds[set], "tex set skip");
            ds[set] = VK_NULL_HANDLE;                // still bound from the last draw
            continue;
        }
        g_push0.ready = false;
        ds[set] = build_set(pl, set);
        if (set == 0 && g_push0.ready) {      // swapped, not copied: the writes keep pointing into their own buffer infos
            push0.writes.swap(g_push0.writes);
            push0.binfo.swap(g_push0.binfo);
            push0.ready = true;
            g_push0.ready = false;
        }
    }
    if (pl.has_set[kPoolSet] && (skip & (1u << kPoolSet))) {
        ++g_rs.sets_skipped;
        if (rt_check()) chk(pool_set(S.tpool.get(), S.spool.get(), *F) == g_last_gfx.ds[kPoolSet], "pool set skip");
    } else if (pl.has_set[kPoolSet]) {
        ds[kPoolSet] = pool_set(S.tpool.get(), S.spool.get(), *F);   // the texture and sampler pools as arrays (nvn_pipeline.cpp)
    }
    if (skip && F->cb != g_last_gfx.cb) {   // defensive: the command buffer changed while the sets were built: bind every set in the new one
        for (uint32_t set = 0; set < 4; ++set)
            if ((skip & (1u << set)) && pl.has_set[set]) ds[set] = build_set(pl, set);
        if (pl.has_set[kPoolSet] && (skip & (1u << kPoolSet))) ds[kPoolSet] = pool_set(S.tpool.get(), S.spool.get(), *F);
        skip = 0;
    }
    if (gfx) {
        for (uint32_t i = 0; i <= kPoolSet; ++i)
            if (!(skip & (1u << i))) g_last_gfx.ds[i] = ds[i];
        g_last_gfx.valid = true;
        g_last_gfx.pl = &pl;
        g_last_gfx.cb = F->cb;
        g_last_gfx.tp = tp;
        g_last_gfx.sp = sp;
        g_last_gfx.tv = tv;
        g_last_gfx.sv = sv;
    }
    if (push0.ready && !push0.writes.empty()) {   // a push of no writes is not allowed
        vkCmdPushDescriptorSetKHR(F->cb, bp, pl.layout, 0, static_cast<uint32_t>(push0.writes.size()), push0.writes.data());
        push0.ready = false;
    }
    // consecutive sets in one call (bound sets are command buffer state: the copies build_set may record in between do not touch them)
    for (uint32_t i = 0; i <= kPoolSet;) {
        if (!ds[i]) { ++i; continue; }
        uint32_t j = i + 1;
        while (j <= kPoolSet && ds[j]) ++j;
        vkCmdBindDescriptorSets(F->cb, bp, pl.layout, i, j - i, ds + i, 0, nullptr);
        i = j;
    }
}

// ---- draws -------------------------------------------------------------------------------------------------------------------
const ShaderInfo* tcs_now();   // the program's control shader, or the generated pass-through one (below)
bool build_key(GraphicsKey& key, VkPrimitiveTopology topo) {
    std::memset(&key, 0, sizeof key);
    key.stage[0] = S.stage[kVertex];
    key.stage[1] = S.stage[kFragment];
    key.stage[2] = S.stage[kGeometry];
    if (topo == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST) {      // begin_draw checked: both tessellation stages, patches
        key.tess[0] = tcs_now();
        key.tess[1] = S.stage[kTessEval];
        key.patch_control_points = static_cast<uint8_t>(std::clamp<uint32_t>(S.patch_size, 1, 32));
    }
    const ShaderInfo* vs = key.stage[0];
    // attributes: those the vertex shader reads. An input array takes one location per element. A location the game did not set up
    // (no attribute, or a disabled one) still has to exist in the pipeline; it reads (0, 0, 0, 1) from a dummy binding.
    struct Missing { uint32_t location; ScalarType type; };
    Missing missing[32];
    uint32_t nmissing = 0;
    for (const ShaderIo& in : vs->refl.inputs) {
        const uint32_t elems = in.array ? in.array : 1;
        for (uint32_t e = 0; e < elems; ++e) {
            const uint32_t loc = in.location + e;
            if (loc >= 32) break;
            VkFormat f = VK_FORMAT_UNDEFINED;
            const VertexAttribInfo* a = nullptr;
            if (loc < S.attr_count && loc < 16) {
                a = &S.attr[loc];
                f = vertex_format(a->format);
            }
            if (f == VK_FORMAT_UNDEFINED) {
                if (nmissing < 32) missing[nmissing++] = {loc, in.type};
                continue;
            }
            // a float input fed by an integer attribute gets the raw bits: the integer format stays and the pipeline patches the shader
            // (spirv_int_attributes); stream-out draws use the shader's stream-out variant, which is not patched: scaled floats there
            if (in.type == ScalarType::Float && is_int_format(f) && g_xfb_active) f = scaled_for(f);
            else if ((in.type == ScalarType::Uint || in.type == ScalarType::Sint) && !is_int_format(f))
                NVN_LOG_ONCE("attr-type-mismatch", "a vertex attribute with a float format feeds an integer shader input (needs a SPIR-V variant)");
            if (key.attr_count >= 16) break;
            KeyAttr& k = key.attr[key.attr_count++];
            k.location = static_cast<uint8_t>(loc);
            k.binding = static_cast<uint8_t>(a->stream & 15);
            k.offset = static_cast<uint16_t>(a->relative_offset);
            k.format = f;
            key.binding_instance[k.binding] = S.stream[k.binding].divisor > 0 ? 1 : 0;
        }
    }
    if (nmissing) {
        uint32_t used = 0;
        for (uint32_t i = 0; i < key.attr_count; ++i) used |= 1u << key.attr[i].binding;
        uint32_t free_binding = 0;
        while (free_binding < 16 && (used >> free_binding & 1)) ++free_binding;
        if (free_binding < 16) {
            key.dummy_binding = static_cast<uint8_t>(free_binding + 1);
            key.binding_instance[free_binding] = 0;
            for (uint32_t i = 0; i < nmissing && key.attr_count < 16; ++i) {
                KeyAttr& k = key.attr[key.attr_count++];
                k.location = static_cast<uint8_t>(missing[i].location);
                k.binding = static_cast<uint8_t>(free_binding);
                const bool is_float = missing[i].type != ScalarType::Uint && missing[i].type != ScalarType::Sint;
                k.offset = is_float ? 0 : 16;
                k.format = is_float ? VK_FORMAT_R32G32B32A32_SFLOAT : (missing[i].type == ScalarType::Uint ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SINT);
            }
        }
    }
    key.color_count = P.ncolor;
    if (!P.ncolor && S.stage[kFragment] && !S.stage[kFragment]->refl.outputs.empty()) key.no_out = 1;   // a depth-only pass: no outputs
    if (!key.no_out && device().features().dynamic_rendering_unused_attachments) {
        // the fragment shader may write outputs the pass has no attachment for: they are listed as unused (format UNDEFINED)
        if (const ShaderInfo* fs = S.stage[kFragment])
            for (const ShaderIo& o : fs->refl.outputs)
                for (uint32_t e = 0; e < (o.array ? o.array : 1); ++e)
                    if (o.location + e < 8 && o.location + e >= key.color_count) key.color_count = o.location + e + 1;
    }
    for (uint32_t i = 0; i < key.color_count && i < 8; ++i) {
        if (i >= P.ncolor) {
            key.color_format[i] = VK_FORMAT_UNDEFINED;
            key.write_mask[i] = 0;
            continue;
        }
        key.color_format[i] = P.color_fmt[i];
        key.blend_enable[i] = (S.color.blend_enable_mask >> i) & 1;
        const BlendInfo& b = S.blend[i];
        key.blend_src_rgb[i] = b.src_rgb ? vk_blend_factor(b.src_rgb) : VK_BLEND_FACTOR_ONE;
        key.blend_dst_rgb[i] = b.dst_rgb ? vk_blend_factor(b.dst_rgb) : VK_BLEND_FACTOR_ZERO;
        key.blend_src_a[i] = b.src_a ? vk_blend_factor(b.src_a) : VK_BLEND_FACTOR_ONE;
        key.blend_dst_a[i] = b.dst_a ? vk_blend_factor(b.dst_a) : VK_BLEND_FACTOR_ZERO;
        key.blend_op_rgb[i] = b.eq_rgb ? vk_blend_op(b.eq_rgb) : VK_BLEND_OP_ADD;
        key.blend_op_a[i] = b.eq_a ? vk_blend_op(b.eq_a) : VK_BLEND_OP_ADD;
        key.write_mask[i] = S.chmask.mask[i] & 0xf;
        if (P.color_fmt[i] == VK_FORMAT_UNDEFINED) key.write_mask[i] = 0;
        auto uses_src1 = [](VkBlendFactor f) {
            return f == VK_BLEND_FACTOR_SRC1_COLOR || f == VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR || f == VK_BLEND_FACTOR_SRC1_ALPHA ||
                   f == VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        };
        if (key.blend_enable[i] && (uses_src1(static_cast<VkBlendFactor>(key.blend_src_rgb[i])) || uses_src1(static_cast<VkBlendFactor>(key.blend_dst_rgb[i])) ||
                                    uses_src1(static_cast<VkBlendFactor>(key.blend_src_a[i])) || uses_src1(static_cast<VkBlendFactor>(key.blend_dst_a[i]))))
            key.dual_src = 1;
    }
    key.xfb = g_xfb_active && device().features().transform_feedback && S.stage[kVertex] && xfb_stride(S.stage[kVertex]) ? 1 : 0;
    key.depth_format = P.depth_fmt;
    key.stencil_format = P.has_stencil ? P.depth_fmt : VK_FORMAT_UNDEFINED;
    key.samples = P.samples;
    key.topology = static_cast<uint8_t>(topo);
    key.polygon_mode = static_cast<uint8_t>(vk_polygon_mode(S.poly.polygon_mode));
    if (key.polygon_mode != VK_POLYGON_MODE_FILL && !device().features().fill_mode_non_solid) key.polygon_mode = VK_POLYGON_MODE_FILL;
    key.depth_clamp = S.depth_clamp && device().features().depth_clamp;
    key.alpha_to_coverage = S.ms.alpha_to_coverage && S.ms.enable ? 1 : 0;
    key.depth_neg_one = S.depth_mode == NVN_DEPTH_MODE_NEAR_IS_MINUS_W ? 1 : 0;
    if (S.color.logic_op != 0 && S.color.logic_op != NVN_LOGIC_OP_COPY && device().features().logic_op) {
        key.logic_op_enable = 1;
        key.logic_op = static_cast<uint8_t>(vk_logic_op(S.color.logic_op));
        for (uint32_t i = 0; i < 8; ++i) key.blend_enable[i] = 0;
    }
    if (S.color.alpha_test != 0 && S.color.alpha_test != NVN_ALPHA_FUNC_ALWAYS) NVN_LOG_ONCE("alpha-test", "alpha test is enabled by the game: not implemented (needs a shader variant)");
    return true;
}

void set_dynamic_state(uint32_t dirty, VkPrimitiveTopology topo, bool indexed, int index_type) {
    VkCommandBuffer cb = F->cb;
    if (dirty & kDynVp) set_viewports_and_scissors();
    if (dirty & kDynRaster) {
        vkCmdSetCullMode(cb, vk_cull_mode(S.poly.cull_face));
        VkFrontFace ff = vk_front_face(S.poly.front_face);
        if (S.window_origin == NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT) ff = ff == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;   // unflipped viewport
        vkCmdSetFrontFace(cb, ff);
        vkCmdSetRasterizerDiscardEnable(cb, S.rasterizer_discard);
        vkCmdSetLineWidth(cb, device().features().wide_lines ? std::max(1.0f, S.line_width) : 1.0f);
    }
    if (topo != g_topo) {
        vkCmdSetPrimitiveTopology(cb, topo);
        g_topo = topo;
    }
    const bool restart = S.prim_restart && indexed &&
                         ((index_type == NVN_INDEX_TYPE_UNSIGNED_INT && S.restart_index == 0xffffffffu) || (index_type == NVN_INDEX_TYPE_UNSIGNED_SHORT && S.restart_index == 0xffffu) ||
                          (index_type == NVN_INDEX_TYPE_UNSIGNED_BYTE && S.restart_index == 0xffu));
    if (!g_restart_valid || restart != g_restart || (dirty & kDynRaster)) {
        vkCmdSetPrimitiveRestartEnable(cb, restart);
        g_restart = restart;
        g_restart_valid = true;
    }
    const bool have_depth = P.depth_fmt != VK_FORMAT_UNDEFINED;
    if (dirty & kDynDepth) {
        vkCmdSetDepthTestEnable(cb, have_depth && S.ds.depth_test);
        vkCmdSetDepthWriteEnable(cb, have_depth && !P.depth_ro && S.ds.depth_test && S.ds.depth_write);
        vkCmdSetDepthCompareOp(cb, vk_compare_op(S.ds.depth_func ? S.ds.depth_func : NVN_DEPTH_FUNC_LESS));
        vkCmdSetDepthBoundsTestEnable(cb, have_depth && S.depth_bounds && device().features().depth_bounds);
        if (S.depth_bounds && device().features().depth_bounds) vkCmdSetDepthBounds(cb, S.db_near, S.db_far);
    }
    if (dirty & kDynStencil) {
        vkCmdSetStencilTestEnable(cb, P.has_stencil && S.ds.stencil_test);
        for (int f = 0; f < 2; ++f) {
            const VkStencilFaceFlags face = f == 0 ? VK_STENCIL_FACE_FRONT_BIT : VK_STENCIL_FACE_BACK_BIT;
            vkCmdSetStencilOp(cb, face, vk_stencil_op(S.ds.stencil_fail[f]), vk_stencil_op(S.ds.depth_pass[f]), vk_stencil_op(S.ds.depth_fail[f]),
                              vk_compare_op(S.ds.stencil_func[f] ? S.ds.stencil_func[f] : NVN_STENCIL_FUNC_ALWAYS));
            vkCmdSetStencilReference(cb, face, S.stencil_ref[f]);
            vkCmdSetStencilCompareMask(cb, face, S.stencil_vmask[f]);
            vkCmdSetStencilWriteMask(cb, face, P.depth_ro ? 0u : S.stencil_wmask[f]);
        }
    }
    if (dirty & kDynBias) {
        const int off = S.poly.offset_enables;
        const int mode = S.poly.polygon_mode;
        const bool bias = (mode == NVN_POLYGON_MODE_FILL && (off & NVN_POLYGON_OFFSET_ENABLE_FILL)) ||
                          (mode == NVN_POLYGON_MODE_LINE && (off & NVN_POLYGON_OFFSET_ENABLE_LINE)) ||
                          (mode == NVN_POLYGON_MODE_POINT && (off & NVN_POLYGON_OFFSET_ENABLE_POINT));
        vkCmdSetDepthBiasEnable(cb, bias);
        vkCmdSetDepthBias(cb, S.bias_units, device().features().depth_bias_clamp ? S.bias_clamp : 0.0f, S.bias_factor);
    }
    if (dirty & kDynBlend) vkCmdSetBlendConstants(cb, S.blend_color);
}

// Prepares everything a draw needs and binds it. False: skip the draw.
// [nvn] trace_min_sec = 250: trace_after_fs only starts that many seconds after the first draw (the scene to look at comes late).
bool trigger_time_ok() {
    static const auto start = std::chrono::steady_clock::now();
    static const double min_sec = static_cast<double>(nxrt::config_get_int("nvn", "trace_min_sec", 0));
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= min_sec;
}

// Scene upscaler (S14, src/gfx/extras/hdr_nvn.h): before a tone-map draw whose HDR scene input is smaller than its target, FSR 1 enlarges the scene
// to the target's size and the draw samples that instead. The override lives until the next draw.
// Temporal upscaling (FSR 3 / DLSS / DLAA, src/gfx/extras/temporal.h): a G-buffer draw (4+ colour targets and a depth target) offers its vertex
// uniform slot 0 as the camera block; the first valid one of the frame is kept with its depth target.
void temporal_gbuffer_note() {
    if (S.ncolor < 4 || !S.depth_tex || !S.stage[kVertex] || extras::temporal_camera_ready()) return;
    const GState::Bind& u = S.ubo[kVertex][0];
    if (!u.addr || u.size < 272) return;
    uint32_t w[68];
    std::memcpy(w, host_ptr(u.addr), sizeof w);
    tsr_jitter(w, 68);                                                    // the camera as the draw sees it (the bound copy is jittered the same way)
    extras::temporal_note_gbuffer(w, 68);
    if (extras::temporal_camera_ready()) g_tsr_depth = S.depth_tex;
}

// The tone-map draw with a temporal mode: the largest 2D colour input is the HDR scene; the SDK's result (target size) replaces it for this draw.
void temporal_for_draw(const ProgLayout& pl) {
    ++g_tsr_frame;                                                        // one tone-map draw per frame: the next frame's jitter phase
    Texture* target = S.color_tex[0].get();
    Texture* depth = g_tsr_depth.get();
    static bool told_intercept = false, told_input = false, told_depth = false;
    if (!told_intercept) {
        told_intercept = true;
        NXLOG_I(kTag, "temporal: tone-map draw %s intercepted (target %ux%u)", S.stage[kFragment]->hash.c_str(), target ? target->b.width : 0,
                target ? target->b.height : 0);
    }
    if (!extras::temporal_camera_ready()) {                              // counts the frames without a camera (warns after 120)
        extras::temporal_evaluate(nullptr, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, {}, {}, false);
        return;
    }
    if (!tex_gpu(target) || !depth || !tex_gpu(depth)) {
        if (!told_depth) {
            told_depth = true;
            NXLOG_W(kTag, "temporal: camera found but no usable target / G-buffer depth texture at the tone-map draw");
        }
        return;
    }
    const BindingDesc* best = nullptr;
    TexResolved best_r{};
    uint64_t best_area = 0;
    for (const BindingDesc& d : pl.bindings) {
        if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        if (b.nvn.source != NvnSource::SamplerBinding && b.nvn.source != NvnSource::HandleInUniform) continue;
        if (view_type_for(b) != VK_IMAGE_VIEW_TYPE_2D) continue;
        TexResolved r = resolve_texture(d);
        if (!r.tex || !tex_gpu(r.tex) || tex_is_depth(r.tex) || shares_image(r.tex, depth)) continue;
        const VkExtent3D e = tex_image_extent(r.tex, 0);
        const uint64_t area = static_cast<uint64_t>(e.width) * e.height;
        if (area > best_area) { best_area = area; best = &d; best_r = r; }
    }
    if (!best) {
        if (!told_input) {
            told_input = true;
            NXLOG_W(kTag, "temporal: the tone-map draw samples no 2D colour texture to use as the scene");
        }
        return;
    }
    const DescriptorBinding& b = best->shader->refl.bindings[best->index];
    const VkExtent3D ce = tex_image_extent(best_r.tex, 0), de = tex_image_extent(depth, 0), te = tex_image_extent(target, 0);
    const VkExtent2D render{std::min(ce.width, de.width), std::min(ce.height, de.height)};
    end_pass();                                                           // compute work cannot run inside a render pass
    tex_ensure_current(best_r.tex, *F);
    tex_transition(best_r.tex, *F, tex_sampled_layout(best_r.tex));
    tex_transition(depth, *F, tex_sampled_layout(depth));
    VkImageView src = tex_sampled_view(best_r.tex, make_request(b, best_r));
    ViewRequest dr;
    dr.depth_aspect = true;
    VkImageView dv = tex_sampled_view(depth, dr);
    if (!src || !dv) return;
    VkImageView out;
    {
        cmdq::Direct direct;   // DLSS / FSR 2 / Streamline record through their own entry points
        out = extras::temporal_evaluate(&device(), F->cb, src, dv, tex_sampled_layout(depth), render, VkExtent2D{te.width, te.height},
                                        S.depth_mode == NVN_DEPTH_MODE_NEAR_IS_MINUS_W);
    }
    g_dirty = kAllDirty;                                                  // the SDKs bind their own compute state
    if (out) {
        g_su_tex = best_r.tex;
        g_su_view = out;
    }
}

void scene_upscale_for_draw(const ProgLayout& pl) {
    g_su_tex = nullptr;
    g_su_view = VK_NULL_HANDLE;
    const ShaderInfo* fs = S.stage[kFragment];
    if (fs && extras::temporal_wanted() && S.ncolor > 0 && S.color_tex[0] && extras::hdr_is_tonemap(fs->spirv, fs->hash)) {
        temporal_for_draw(pl);
        return;
    }
    if (!fs || !extras::scene_upscale_wanted() || S.ncolor == 0 || !S.color_tex[0] || !extras::hdr_is_tonemap(fs->spirv, fs->hash)) return;
    Texture* target = S.color_tex[0].get();
    if (!tex_gpu(target)) return;
    const VkExtent3D te = tex_extent(target, 0);
    for (const BindingDesc& d : pl.bindings) {
        if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        if (b.nvn.source != NvnSource::SamplerBinding && b.nvn.source != NvnSource::HandleInUniform) continue;
        if (view_type_for(b) != VK_IMAGE_VIEW_TYPE_2D) continue;
        TexResolved r = resolve_texture(d);
        if (!r.tex || !tex_gpu(r.tex) || shares_image(r.tex, S.depth_tex.get())) continue;
        const VkExtent3D se = tex_extent(r.tex, 0);
        if (se.width >= te.width && se.height >= te.height) return;          // nothing to enlarge
        extras::SpatialUpscaler* su = extras::scene_upscaler(&device(), VkExtent2D{se.width, se.height}, VkExtent2D{te.width, te.height});
        if (!su) return;
        end_pass();                                                           // compute work cannot run inside a render pass
        tex_ensure_current(r.tex, *F);
        tex_transition(r.tex, *F, tex_sampled_layout(r.tex));
        VkImageView src = tex_sampled_view(r.tex, make_request(b, r));
        if (!src) return;
        VkImageView up;
        {
            cmdq::Direct direct;   // the upscalers may record through their SDK's entry points
            up = su->upscale(F->cb, src, VkExtent2D{se.width, se.height}, VkExtent2D{se.width, se.height}, VkExtent2D{te.width, te.height}, true,
                             extras::scene_upscale_prescale(), extras::upscale_sharpness());
        }
        if (up) {
            g_su_tex = r.tex;
            g_su_view = up;
        }
        return;
    }
}

// Evaluation-only programs (NVN tessellates with the fixed SetInner/OuterTessellationLevels): a generated pass-through control shader
// (nvn_tess_gen.h), one per (vertex shader, evaluation shader, patch size). It copies each per-vertex input of the evaluation shader from the
// vertex shader output at the same location (zero where the vertex shader writes none) and gl_Position, zeroes the per-patch inputs, and reads
// the levels from push constants (pushed before each such draw, so level changes need no new pipeline). Null when an interface variable has a
// type the generator does not handle.
const ShaderInfo* g_gen_tcs = nullptr;   // the generated control shader of the current draw (null: the program has its own, or none)
const ShaderInfo* passthrough_tcs_for(const ShaderInfo* vs, const ShaderInfo* tes) {
    struct Key {
        const ShaderInfo* vs; const ShaderInfo* tes; uint32_t n;
        bool operator<(const Key& o) const { return std::tie(vs, tes, n) < std::tie(o.vs, o.tes, o.n); }
    };
    static std::map<Key, std::unique_ptr<ShaderInfo>> cache;
    const uint32_t n = std::clamp<uint32_t>(S.patch_size, 1, 32);
    const Key key{vs, tes, n};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second.get();
    auto to_io = [](const ShaderIo& v, tessgen::Io& o) {
        if (v.components < 1 || v.components > 4) return false;
        if (v.type == ScalarType::Float) o.kind = tessgen::Kind::Float;
        else if (v.type == ScalarType::Sint) o.kind = tessgen::Kind::Sint;
        else if (v.type == ScalarType::Uint) o.kind = tessgen::Kind::Uint;
        else return false;
        o.location = v.location;
        o.component = v.component;
        o.components = v.components;
        return true;
    };
    std::vector<tessgen::Io> vs_out, per_vertex, per_patch;
    bool ok = true;
    for (const ShaderIo& o : vs->refl.outputs) {   // an output array (several locations) is not matched: its readers get zero
        tessgen::Io io;
        if (o.array <= 1 && to_io(o, io)) vs_out.push_back(io);
    }
    for (const ShaderIo& in : tes->refl.inputs) {
        tessgen::Io io;
        if (!to_io(in, io)) { ok = false; break; }
        (in.array > 1 ? per_vertex : per_patch).push_back(io);   // in an evaluation shader a non-array input is per patch
    }
    std::unique_ptr<ShaderInfo> si;
    if (ok) {
        const std::vector<tessgen::Copy> copies = tessgen::match(vs_out, per_vertex);
        size_t copied = 0;
        for (const tessgen::Copy& c : copies) copied += c.has_src;
        si = std::make_unique<ShaderInfo>();
        si->hash = "tcs-pass-" + vs->hash + "-" + tes->hash + "-" + std::to_string(n);
        si->stage = ShaderStage::TessControl;
        // gl_ClipDistance / gl_CullDistance: copied when the vertex shader writes them and the evaluation shader reads gl_in[]'s
        auto uses = [](const std::vector<uint32_t>& v, uint32_t b) { return std::find(v.begin(), v.end(), b) != v.end(); };
        constexpr uint32_t kBuiltInClip = 3, kBuiltInCull = 4;
        const bool clip = uses(vs->refl.builtins_used_out, kBuiltInClip) && uses(tes->refl.builtins_used_in, kBuiltInClip);
        const bool cull = uses(vs->refl.builtins_used_out, kBuiltInCull) && uses(tes->refl.builtins_used_in, kBuiltInCull);
        si->spirv = tessgen::passthrough_tcs(copies, per_patch, n, 7, 1, clip, cull);
        si->push_bytes = tessgen::kPushBytes;
        std::string err;
        if (!reflect_spirv(si->spirv.data(), si->spirv.size(), si->refl, &err)) {
            NXLOG_W(kTag, "generated control shader for %s does not reflect: %s", tes->hash.c_str(), err.c_str());
            si.reset();
        } else {
            NXLOG_I(kTag, "evaluation shader %s (vs %s): pass-through control shader, %u vertices, %zu per-vertex locations (%zu from the vertex shader, %zu zero), %zu per-patch%s%s",
                    tes->hash.c_str(), vs->hash.c_str(), n, copies.size(), copied, copies.size() - copied, per_patch.size(), clip ? ", clip distances" : "",
                    cull ? ", cull distances" : "");
        }
    } else {
        NXLOG_W(kTag, "evaluation shader %s: an input is not a 1..4 component float/int, no pass-through control shader", tes->hash.c_str());
    }
    const ShaderInfo* out = si.get();
    cache.emplace(key, std::move(si));
    return out;
}
const ShaderInfo* tcs_now() { return S.stage[kTessCtl] ? S.stage[kTessCtl] : g_gen_tcs; }

// [nvn] depth_ro_sticky (default false): a draw that does not sample the depth target and writes neither depth nor stencil stays in an
// open pass whose depth attachment is read-only, instead of ending it for a writable one. The transparent and particle draws of a frame
// (depth test on, depth writes off) where only some of them sample the depth (soft particles) then stay in one pass, without a layout
// change of the full-size depth buffer back and forth at each switch.
void cs_check_draw(const ProgLayout& pl);                 // dispatch barriers (do_dispatch, further down)
void cs_draw_index_range(uint64_t addr, uint64_t size);

bool depth_ro_for_draw() {
    if (g_want_depth_ro) return true;
    static const bool sticky = nxrt::config_get_bool("nvn", "depth_ro_sticky", false);
    if (!sticky || !P.open || !P.depth_ro || !targets_match()) return false;
    const bool writes_depth = S.ds.depth_test && S.ds.depth_write;
    const bool writes_stencil = P.has_stencil && S.ds.stencil_test && (S.stencil_wmask[0] != 0 || S.stencil_wmask[1] != 0);
    return !writes_depth && !writes_stencil;
}

// [nvn] layout_cache (default on): the layouts of recently drawn programs by their stages, direct mapped, without prog_layout's lock and map
// (~2 % of the render thread: a street frame switches programs ~1100 times). Layouts live until pipeline_shutdown (replay_shutdown clears it).
struct LayoutCacheSlot { const ShaderInfo* st[6] = {}; const ProgLayout* pl = nullptr; };
LayoutCacheSlot g_layout_cache[256];
const ProgLayout* layout_cached(const ShaderInfo* const stages[6]) {
    static const bool on = nxrt::config_get_bool("nvn", "layout_cache", true);
    if (!on) return prog_layout(stages);
    uint64_t h = 0;
    for (int i = 0; i < 6; ++i) h = (h ^ reinterpret_cast<uint64_t>(stages[i])) * 0x9e3779b97f4a7c15ull;
    LayoutCacheSlot& sl = g_layout_cache[(h >> 56) & 255];
    if (sl.pl && std::memcmp(sl.st, stages, sizeof sl.st) == 0) {
        if (rt_check()) chk(prog_layout(stages) == sl.pl, "layout cache");
        return sl.pl;
    }
    const ProgLayout* pl = prog_layout(stages);
    if (pl) {
        std::memcpy(sl.st, stages, sizeof sl.st);
        sl.pl = pl;
    }
    return pl;
}

bool begin_draw(int mode, bool indexed, int index_type, VkPrimitiveTopology* topo_out) {
    rs_start();
    const ShaderInfo* vs = S.stage[kVertex];
    if (!vs) { NVN_LOG_ONCE("draw-no-vs", "draw without a vertex shader skipped"); return false; }
    if (!S.render_enable) return false;
    ++g_draw_idx;
    if (S.stage[kFragment] && skip_fs_draw(S.stage[kFragment]->hash)) return false;
    // Tessellation: Vulkan runs it only with both shaders and a patch list. NVN also allows an evaluation shader alone (fixed levels from
    // SetInner/OuterTessellationLevels): those draws get a generated pass-through control shader (passthrough_tcs_for).
    if (S.stage[kTessCtl] || S.stage[kTessEval] || mode == NVN_DRAW_PRIMITIVE_PATCHES) {
        if (mode != NVN_DRAW_PRIMITIVE_PATCHES) { note_unsupported("tessellation draw: tessellation program drawn without patches"); return false; }
        if (!S.stage[kTessEval]) { note_unsupported("tessellation draw: patches without a tessellation evaluation shader"); return false; }
        if (!S.stage[kTessCtl]) {
            const ShaderInfo* gen = passthrough_tcs_for(vs, S.stage[kTessEval]);
            if (!gen) { note_unsupported("tessellation draw: evaluation shader without a control shader, no pass-through could be made"); return false; }
            if (gen != g_gen_tcs) g_dirty |= kPipe;
            g_gen_tcs = gen;
        }
    } else {
        g_gen_tcs = nullptr;
    }
    if (g_dirty & (kPipe | kDesc)) {
        const ShaderInfo* stages[6] = {vs, S.stage[kFragment], S.stage[kGeometry], tcs_now(), S.stage[kTessEval], nullptr};
        // the layout of the same stages as at the last lookup is the same layout (layouts live until pipeline_shutdown): no lock, no map
        static const ShaderInfo* last_stages[6] = {};
        if (!g_pl || std::memcmp(stages, last_stages, sizeof stages) != 0) {
            g_pl = layout_cached(stages);
            std::memcpy(last_stages, stages, sizeof stages);
            ++g_rs.layouts;
        }
    }
    if (!g_trigger_fs.empty() && !g_trigger_done && trigger_time_ok() && S.stage[kFragment] && S.stage[kFragment]->hash.compare(0, g_trigger_fs.size(), g_trigger_fs) == 0) {
        g_trigger_seen = true;
        g_trigger_present = static_cast<int64_t>(counters().presents.load());
    }
    const ProgLayout* pl = g_pl;
    if (!pl) return false;
    MemoScope memo(*pl);
    cs_check_draw(*pl);     // before this draw records anything (uploads, layout changes, feedback copies)
    if (pl->has_set[kPoolSet] && pool_state_moved(S.tpool.get(), S.spool.get())) g_dirty |= kDesc;   // registrations or layout moves since
    if (extras::temporal_wanted()) temporal_gbuffer_note();
    rs_mark(kRsHead);
    const bool full_prep = !prep_skip_on() || (g_dirty & (kDescTex | kPipe)) || pl != g_prep_pl || F->cb != g_prep_cb || g_trace || g_fb_seen ||
                           uniform_handles(*pl);
    if ((g_dirty & (kDesc | kPipe)) && !full_prep) ++g_rs.preps_skipped;
    if ((g_dirty & (kDesc | kPipe)) && full_prep) {
        g_prep_pl = pl;
        g_prep_cb = F->cb;
        ++g_rs.draws_desc;
        g_prog_writes = false;
        for (const BindingDesc& d : pl->bindings) g_prog_writes |= d.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        attach_targets(false);
        g_want_depth_ro = samples_depth_target(*pl);
        prepare_textures(*pl);
        if (g_trace) {   // [nvn] trace_dump_tex: also the block-compressed level 0 as the GPU holds it, to compare with the prepared data
            static const std::string dump_dir = nxrt::config_get_str("nvn", "trace_dump_tex", "");
            static std::unordered_map<uint32_t, bool> raw_done;
            if (!dump_dir.empty()) {
                for (const BindingDesc& d : pl->bindings) {
                    if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
                    const TexResolved r = resolve_texture(d);
                    if (!r.tex || !raw_done.emplace(r.tex->id, true).second) continue;
                    end_pass();
                    const int ri = snapshot_record_raw(r.tex, *F);
                    if (ri >= 0) g_trace_snaps.push_back({ri, g_trace_n, dump_dir + "/gpu_t" + std::to_string(r.tex->id) + ".bin", "raw"});
                }
            }
        }
        scene_upscale_for_draw(*pl);
    }
    rs_mark(kRsPrep);
    if (S.ncolor && fg_hudless_candidate(S.color_tex[0].get(), (S.color.blend_enable_mask & 1u) != 0)) {   // frame generation: the scene before the UI
        end_pass();
        cmdq::Direct direct;   // frame generation SDKs record through their own entry points
        fg_hudless_capture(*F, S.color_tex[0].get());
    }
    if (g_fb_seen || (full_prep && (g_dirty & (kDesc | kPipe)))) feedback_prepare(*pl);
    rs_mark(kRsFb);
    if (!ensure_pass(depth_ro_for_draw(), g_prog_writes)) {
        // which draw, and what its targets are (the first of each vertex/fragment shader pair)
        const std::string vs = S.stage[kVertex] ? S.stage[kVertex]->hash : "-", fs = S.stage[kFragment] ? S.stage[kFragment]->hash : "-";
        if (first_time_key("draw-no-target " + vs + " " + fs)) {
            std::string tg;
            for (uint32_t i = 0; i < S.ncolor; ++i) {
                char b[96];
                const Target t = make_target(S.color_tex[i].get(), S.color_view[i], S.color_has_view[i]);
                tg += " rt" + std::to_string(i) + " " + tex_desc(t.tex, b, sizeof b) + " layer " + std::to_string(t.layer) + "+" + std::to_string(t.layers) +
                      (t.tex && !tex_gpu(t.tex) ? " (no GPU image)" : "");
            }
            char b[96];
            NXLOG_W(kTag, "draw without a usable render target skipped: vs %s fs %s, %u colour targets%s, depth %s", vs.c_str(), fs.c_str(), S.ncolor, tg.c_str(),
                    tex_desc(S.depth_tex.get(), b, sizeof b));
        }
        return false;
    }
    if (g_prog_writes || g_xfb_active) g_pass_unsafe = true;
    rs_mark(kRsPass);
    const VkPrimitiveTopology topo = vk_topology(mode);
    if (g_dirty & kPipe) ++g_rs.draws_pipe;
    if ((g_dirty & kPipe) || topo != g_pipe_topo || !g_pipe) {
        ++g_rs.keys;
        GraphicsKey key;
        build_key(key, topo);
        g_pipe = graphics_pipeline(key, *pl);
        g_pipe_xfb = key.xfb != 0;
        g_pipe_topo = topo;
        g_key_attr_count = key.attr_count;
        g_key_dummy_binding = key.dummy_binding;
        for (uint32_t i = 0; i < key.attr_count; ++i) g_key_binding[i] = key.attr[i].binding;
    }
    if (!g_pipe) return false;
    // a draw inside a stream-out section that has no stream-out variant and nothing to render to has no effect
    if (g_xfb_active && !g_pipe_xfb && !P.ncolor && !S.depth_tex) return false;
    if (g_trace) {
        char a[96], b[96], c[96];
        std::string line;
        auto remember = [](Texture* t) {
            if (!t) return;
            for (const Ref<Texture>& r : g_trace_targets) if (r.get() == t) return;
            g_trace_targets.push_back(Ref<Texture>::share(t));
        };
        for (uint32_t i = 0; i < P.ncolor; ++i) remember(S.color_tex[i].get());
        remember(S.depth_tex.get());
        for (uint32_t i = 0; i < P.ncolor; ++i) { line += tex_desc(S.color_tex[i].get(), a, sizeof a); line += " | "; }
        TRACE("draw #%d vs %s fs %s gs %s mode %d idx %d | rt %s ds %s%s | blend %u mask %x depth t%d w%d f%d cull %d | ubo0 %llx/%llu",
              ++g_trace_n, S.stage[kVertex] ? S.stage[kVertex]->hash.c_str() : "-", S.stage[kFragment] ? S.stage[kFragment]->hash.c_str() : "-",
              S.stage[kGeometry] ? S.stage[kGeometry]->hash.c_str() : "-", mode, indexed ? 1 : 0, line.c_str(), tex_desc(S.depth_tex.get(), b, sizeof b),
              P.depth_ro ? " (ro)" : "", S.color.blend_enable_mask, S.chmask.mask[0], S.ds.depth_test ? 1 : 0, S.ds.depth_write ? 1 : 0, S.ds.depth_func,
              S.poly.cull_face, static_cast<unsigned long long>(S.ubo[kVertex][0].addr), static_cast<unsigned long long>(S.ubo[kVertex][0].size));
        TRACE("    poly mode %d offset %x factor %g units %g clamp %g | depth mode %d origin %d clamp %d", S.poly.polygon_mode, S.poly.offset_enables, S.bias_factor, S.bias_units,
              S.bias_clamp, S.depth_mode, S.window_origin, S.depth_clamp ? 1 : 0);
        if (S.color.blend_enable_mask)
            TRACE("    blend rt0 rgb %d,%d,%d a %d,%d,%d (nvn enums src,dst,eq)", S.blend[0].src_rgb, S.blend[0].dst_rgb, S.blend[0].eq_rgb, S.blend[0].src_a, S.blend[0].dst_a, S.blend[0].eq_a);
        // [nvn] trace_ubo_bytes = N (NXRT_NVN_TRACE_UBO_BYTES, default 768, at most 65536): how many leading bytes of each uniform block the trace prints
        static const uint64_t ubo_bytes = static_cast<uint64_t>(std::clamp<int64_t>(nxrt::config_get_int("nvn", "trace_ubo_bytes", 768), 16, 65536));
        for (int stg = kVertex; stg <= kFragment; ++stg)
        for (uint32_t slot = 0; slot < 16; ++slot) {    // the first words of every uniform block bound to the vertex and the fragment stage
            const GState::Bind& ub = S.ubo[stg][slot];
            if (!ub.addr) continue;
            BufRef ur = resolve_buffer(ub.addr, std::min<uint64_t>(ub.size, ubo_bytes));
            std::string words;
            if (ur.ok) {
                const uint32_t* w = reinterpret_cast<const uint32_t*>(ub.addr);
                for (uint32_t i = 0; i * 4 < ubo_bytes && i * 4 < ub.size; ++i) { char t[16]; std::snprintf(t, sizeof t, " %08x", w[i]); words += t; }
            }
            TRACE("    %s ubo%u %llx/%llu%s:%s", stg == kVertex ? "vs" : "fs", slot, static_cast<unsigned long long>(ub.addr), static_cast<unsigned long long>(ub.size), ur.ok ? "" : " (outside every pool)", words.c_str());
        }
        for (int stg = kVertex; stg <= kFragment; ++stg)   // storage buffers of the vertex and the fragment stage: where they are, not what they hold
            for (uint32_t slot = 0; slot < 16; ++slot)
                if (S.ssbo[stg][slot].addr)
                    TRACE("    %s ssbo%u %llx/%llu", stg == kVertex ? "vs" : "fs", slot, static_cast<unsigned long long>(S.ssbo[stg][slot].addr),
                          static_cast<unsigned long long>(S.ssbo[stg][slot].size));
        TRACE("    viewport %u: %g %g %g %g depth %g..%g scissor %u: %d %d %d %d", S.nvp, S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3], S.dr[0][0], S.dr[0][1], S.nsc,
              S.sc[0][0], S.sc[0][1], S.sc[0][2], S.sc[0][3]);
        for (uint32_t i = 0; i < P.ncolor; ++i)
            if (Texture* ct = S.color_tex[i].get()) {
                const FormatInfo* vf = (S.color_has_view[i] && S.color_view[i].has_format) ? format_info(S.color_view[i].format) : nullptr;
                TRACE("    rt%u t%u mem %p storage %llu cpu-dirty %d view level %u layer %u layers %u format %s", i, ct->id, static_cast<void*>(ct->mem),
                      static_cast<unsigned long long>(ct->storage_size), ct->cpu_dirty.load() ? 1 : 0, S.color_level[i], S.color_layer[i], S.color_layers[i],
                      vf && vf->name ? vf->name : "-");
            }
        g_trace_sampled.clear();
        for (const BindingDesc& d : pl->bindings) {   // texel buffers (light lists and other BUFFER-target textures): the view the draw gets
            if (d.type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER && d.type != VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) continue;
            const bool storage = d.type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
            const TexResolved r = storage ? resolve_image(d) : resolve_texture(d);
            if (!r.tex) { TRACE("    texel set%u b%u -> (no texture)", d.set, d.binding); continue; }
            uint64_t addr = 0, bytes = 0;
            const VkBufferView v = tex_buffer_view(r.tex, r.has_view ? &r.view : nullptr, 0, VK_FORMAT_UNDEFINED, storage, &addr, &bytes);
            std::string words;
            if (r.tex->mem) {
                const uint32_t* w = reinterpret_cast<const uint32_t*>(r.tex->mem);
                const size_t nw = std::min<size_t>(32, static_cast<size_t>(r.tex->b.width) * (r.tex->fmt ? r.tex->fmt->bytes : 4) / 4);
                char wb[12];
                for (size_t i = 0; i < nw; ++i) { std::snprintf(wb, sizeof wb, " %08x", w[i]); words += wb; }
                const size_t all = static_cast<size_t>(r.tex->b.width) * (r.tex->fmt ? r.tex->fmt->bytes : 4) / 4;
                size_t nz = 0;
                for (size_t i = 0; i < all; ++i) nz += w[i] != 0;
                std::snprintf(wb, sizeof wb, " |nz %zu", nz);
                words = std::string(wb) + "/" + std::to_string(all) + words;
            }
            TRACE("    texel set%u b%u -> t%u %s width %d @%p view %s addr %llx bytes %llu view-format %s:%s", d.set, d.binding, r.tex->id, r.tex->fmt ? r.tex->fmt->name : "?", r.tex->b.width,
                  static_cast<void*>(r.tex->mem), v ? "ok" : "NONE (dummy)", static_cast<unsigned long long>(addr), static_cast<unsigned long long>(bytes),
                  (r.has_view && r.view.has_format) ? "set" : "-", words.c_str());
        }
        for (const BindingDesc& d : pl->bindings) {
            if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && d.type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) continue;
            const TexResolved r = d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? resolve_image(d) : resolve_texture(d);
            if (r.tex) g_trace_sampled.push_back(r.tex);
            if (r.tex) {
                static const std::string dump_dir = nxrt::config_get_str("nvn", "trace_dump_tex", "");
                if (!dump_dir.empty()) trace_dump_prepared(r.tex, dump_dir);
            }
            char smp[200] = "", vw[160] = "";
            if (r.has_sampler)
                std::snprintf(smp, sizeof smp, " | sampler min %d mag %d wrap %d,%d,%d lod %g..%g bias %g aniso %g cmp %d func %d", r.smp.min_filter, r.smp.mag_filter, r.smp.wrap_s, r.smp.wrap_t,
                              r.smp.wrap_r, r.smp.lod_min, r.smp.lod_max, r.smp.lod_bias, r.smp.max_anisotropy, r.smp.compare_mode, r.smp.compare_func);
            if (r.has_view) {
                // the view's own format by name (a view naming the sRGB partner of its texture's format, or the reverse, changes the decode)
                const FormatInfo* vf = r.view.has_format ? format_info(r.view.format) : nullptr;
                char vfn[48] = "";
                if (r.view.has_format) std::snprintf(vfn, sizeof vfn, " format %s", vf && vf->name ? vf->name : "?");
                std::snprintf(vw, sizeof vw, " | view levels %d+%d layers %d+%d%s", r.view.has_levels ? r.view.min_level : -1, r.view.has_levels ? r.view.num_levels : -1,
                              r.view.has_layers ? r.view.min_layer : -1, r.view.has_layers ? r.view.num_layers : -1, vfn);
            }
            if (r.tex) {
                const int* sw = (r.has_view && r.view.has_swizzle) ? r.view.swizzle : r.tex->b.swizzle;
                const size_t n = std::strlen(vw);
                std::snprintf(vw + n, sizeof vw - n, " swz %d%d%d%d", sw[0], sw[1], sw[2], sw[3]);
            }
            TRACE("    %s set%u b%u %s -> %s @%p%s mem %d%%%s%s%s", d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? "image" : "tex", d.set, d.binding,
                  d.shader->refl.bindings[d.index].name.c_str(), tex_desc(r.tex, c, sizeof c), r.tex ? static_cast<void*>(r.tex->mem) : nullptr, r.has_sampler ? "" : " (no sampler)", storage_nonzero_pct(r.tex),
                  r.tex && r.tex->cpu_dirty.load() ? " cpu-dirty" : "", smp, vw);
        }
    }
    if (g_xfb_active) {
        static std::unordered_map<std::string, bool> seen;
        if (seen.emplace(S.stage[kVertex]->hash, true).second)
            NXLOG_I("nvn.xfb", "draw under transform feedback: vs %s mode %d attrs %u xfb0 %llx/%llu", S.stage[kVertex]->hash.c_str(), mode, S.attr_count,
                    static_cast<unsigned long long>(S.xfb[0].addr), static_cast<unsigned long long>(S.xfb[0].size));
    }
    if (g_pipe != g_bound_pipeline) {
        xfb_end_in_pass();     // the pipeline cannot change while feedback is active; the byte counter carries the position over
        vkCmdBindPipeline(F->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipe);
        g_gauge_pipeline_binds.add(1);
        g_bound_pipeline = g_pipe;
    }
    if (g_xfb_active && g_pipe_xfb && !g_xfb_began && device().features().transform_feedback && S.xfb[0].addr) {
        BufRef xb = resolve_buffer(S.xfb[0].addr, S.xfb[0].size);
        if (xb.ok) {
            gpu_wrote(S.xfb[0].addr, S.xfb[0].size ? S.xfb[0].size : xb.avail, kWriteXfb);
            if (!g_xfb_counter) {
                if (device().create_buffer(kXfbSlots * kXfbSlotBytes, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, gfx::MemoryUse::Upload,
                                           &g_xfb_counter, "nvn xfb counter") != VK_SUCCESS)
                    g_xfb_counter = gfx::Buffer();
            }
            const VkDeviceSize size = std::min<VkDeviceSize>(S.xfb[0].size ? S.xfb[0].size : xb.avail, xb.avail) & ~static_cast<VkDeviceSize>(3);
            vkCmdBindTransformFeedbackBuffersEXT(F->cb, 0, 1, &xb.buffer, &xb.offset, &size);
            VkBuffer cbuf = g_xfb_counter.buffer;
            const VkDeviceSize coff = static_cast<VkDeviceSize>(g_xfb_cur) * kXfbSlotBytes;
            if (g_xfb_have_counter && cbuf) vkCmdBeginTransformFeedbackEXT(F->cb, 0, 1, &cbuf, &coff);
            else vkCmdBeginTransformFeedbackEXT(F->cb, 0, 0, nullptr, nullptr);
            g_xfb_began = true;
        } else {
            note_unsupported("transform feedback into an address outside every pool");
        }
    }
    rs_mark(kRsPipe);
    set_dynamic_state(g_dirty, topo, indexed, index_type);
    rs_mark(kRsDyn);
    if (g_dirty & kDesc) {
        support_for_draw(*pl, false);
        bind_sets(*pl, VK_PIPELINE_BIND_POINT_GRAPHICS);
    }
    rs_mark(kRsSets);
    if (pl->push_bytes && g_gen_tcs) {
        // the pass-through control shader's tessellation levels (NVN's fixed SetInner/OuterTessellationLevels), every draw: cheap, and
        // the layout may have changed since the last push
        tessgen::TessPush push = {};
        std::memcpy(push.outer, S.tess_outer, sizeof push.outer);
        std::memcpy(push.inner, S.tess_inner, sizeof push.inner);
        vkCmdPushConstants(F->cb, pl->layout, pl->push_stages, 0, std::min<uint32_t>(pl->push_bytes, sizeof push), &push);
        static std::set<const ShaderInfo*> logged;
        if (logged.insert(g_gen_tcs).second)
            NXLOG_I(kTag, "tessellation draw: evaluation shader %s drawn with pass-through control shader %s, patch size %u, levels %g %g %g %g / %g %g",
                    S.stage[kTessEval]->hash.c_str(), g_gen_tcs->hash.c_str(), static_cast<unsigned>(std::clamp<uint32_t>(S.patch_size, 1, 32)),
                    S.tess_outer[0], S.tess_outer[1], S.tess_outer[2], S.tess_outer[3], S.tess_inner[0], S.tess_inner[1]);
    }
    if (g_dirty & (kVb | kPipe)) {
        // each binding's buffer, then consecutive bindings in one call
        VkBuffer vbuf[16];
        VkDeviceSize voff[16], vsize[16], vstride[16];
        bool vset[16] = {};
        for (uint32_t i = 0; i < g_key_attr_count; ++i) {
            const uint32_t binding = g_key_binding[i];
            if (binding >= 16 || vset[binding]) continue;
            vset[binding] = true;
            if (g_key_dummy_binding && binding == static_cast<uint32_t>(g_key_dummy_binding - 1)) {
                vbuf[binding] = dummy_buffer();
                voff[binding] = kDummyAttrFloatOffset;
                vsize[binding] = 32;
                vstride[binding] = 0;
                continue;
            }
            const GState::Bind& vb = S.vtx[binding];
            BufRef br = vb.addr ? resolve_read(vb.addr, vb.size, kReadVertex) : BufRef{dummy_buffer(), 0, 65536, false};
            vbuf[binding] = br.buffer;
            voff[binding] = br.offset;
            vsize[binding] = vb.size ? std::min<VkDeviceSize>(vb.size, br.avail) : br.avail;
            vstride[binding] = static_cast<VkDeviceSize>(S.stream[binding].stride);
        }
        for (uint32_t b = 0; b < 16;) {
            if (!vset[b]) { ++b; continue; }
            uint32_t e = b + 1;
            while (e < 16 && vset[e]) ++e;
            vkCmdBindVertexBuffers2(F->cb, b, e - b, vbuf + b, voff + b, vsize + b, vstride + b);
            g_vb_bound_in_cb = true;
            b = e;
        }
        if (!g_vb_bound_in_cb) {
            // the pipeline's vertex binding strides are dynamic: a command buffer that never bound a vertex buffer must still bind one
            VkBuffer dummy = dummy_buffer();
            VkDeviceSize off = 0, size = 65536, stride = 0;
            vkCmdBindVertexBuffers2(F->cb, 0, 1, &dummy, &off, &size, &stride);
            g_vb_bound_in_cb = true;
        }
    }
    g_dirty = g_mirror_stale ? (kVb | kDesc) : 0;     // a storage buffer of this draw made copied pages stale: the next draw binds again
    g_mirror_stale = false;
    *topo_out = topo;
    rs_mark(kRsVb);
    ++g_rs.draws;
    ++g_split_draws;
    return true;
}


// [nvn] trace_snap = "253,300": after those draws of the traced frame the pass is closed and the colour targets and the depth target are
// copied (png after the frame: snap_d<draw>_t<id>_<format>.png).
void trace_snap_after_draw() {
    if (!g_trace) return;
    if (!g_trace_snap_read) {
        g_trace_snap_read = true;
        g_trace_snap_sampled = nxrt::config_get_int("nvn", "trace_snap_sampled", 0) != 0;
        {   // [nvn] trace_snap_fs = "f77f9d,e72811": every draw whose fragment shader hash starts with one of these
            const std::string fl = nxrt::config_get_str("nvn", "trace_snap_fs", "");
            size_t k = 0;
            while (k < fl.size()) {
                size_t j = fl.find(',', k);
                if (j == std::string::npos) j = fl.size();
                if (j > k) g_trace_snap_fs.push_back(fl.substr(k, j - k));
                k = j + 1;
            }
        }
        const std::string list = nxrt::config_get_str("nvn", "trace_snap", "");
        size_t i = 0;
        while (i < list.size()) {
            size_t j = list.find(',', i);
            if (j == std::string::npos) j = list.size();
            if (j > i) g_trace_snap_draws.push_back(std::atoi(list.substr(i, j - i).c_str()));
            i = j + 1;
        }
    }
    bool hit = false;
    for (int d : g_trace_snap_draws) hit |= d == g_trace_n;
    if (!hit && S.stage[kFragment] && !g_trace_snap_fs.empty()) {
        for (const std::string& pre : g_trace_snap_fs) hit |= S.stage[kFragment]->hash.compare(0, pre.size(), pre) == 0;
    }
    if (!hit) return;
    end_pass();
    auto take = [&](Texture* t) {
        if (!t) return;
        const int idx = snapshot_record(t, *F);
        if (idx < 0) return;
        char buf[96];
        g_trace_snaps.push_back({idx, g_trace_n, std::string("snap_d") + std::to_string(g_trace_n) + "_t" + std::to_string(t->id) + "_" + (t->fmt ? t->fmt->name : "x") + ".png", tex_desc(t, buf, sizeof buf)});
    };
    for (uint32_t i = 0; i < S.ncolor; ++i) take(S.color_tex[i].get());
    take(S.depth_tex.get());
    if (g_trace_snap_sampled) {
        for (Texture* t : g_trace_sampled) take(t);
    }
}

// Counts a draw for the primitive counters, before the replay decides whether to skip it: the console generates the primitives of every
// draw it runs, and a draw of a feedback section also writes them (14). A draw with rendering disabled is not run by the console: not counted.
void count_primitives(uint32_t mode, uint32_t count, uint32_t instances) {
    if (!S.render_enable) return;
    const uint64_t per = mode == NVN_DRAW_PRIMITIVE_PATCHES ? count / std::max<uint32_t>(1, S.patch_size) : primitives_for(static_cast<int>(mode), count);   // patches: input patches
    const uint64_t prims = per * std::max<uint32_t>(1, instances);
    g_prims_generated += prims;
    if (g_xfb_active) g_prims_xfb += prims;
}

// Quads, quad strips, polygons and line loops, which Vulkan has no topology for, as the triangle (or line strip) index list of the same vertices.
// `in` holds the vertex indices of the draw in order (for DrawArrays: first, first + 1, ...). Shared by the direct and the indirect draws.
std::vector<uint32_t> convert_prim_indices(PrimConversion conv, const std::vector<uint32_t>& in) {
    std::vector<uint32_t> idx;
    const size_t n = in.size();
    switch (conv) {
        case PrimConversion::Quads:
            for (size_t q = 0; q + 3 < n; q += 4) idx.insert(idx.end(), {in[q], in[q + 1], in[q + 2], in[q], in[q + 2], in[q + 3]});
            break;
        case PrimConversion::QuadStrip:
            for (size_t q = 0; q + 3 < n; q += 2) idx.insert(idx.end(), {in[q], in[q + 1], in[q + 2], in[q + 2], in[q + 1], in[q + 3]});
            break;
        case PrimConversion::Polygon:
            for (size_t k = 1; k + 1 < n; ++k) idx.insert(idx.end(), {in[0], in[k], in[k + 1]});
            break;
        case PrimConversion::LineLoop:
            idx = in;
            if (!in.empty()) idx.push_back(in[0]);
            break;
        default: idx = in; break;
    }
    return idx;
}

// Binds an index list of 32-bit indices from the upload buffer. False: no room.
bool bind_u32_indices(const std::vector<uint32_t>& idx) {
    UploadAlloc u = F->upload(idx.size() * 4, 4);
    if (!u) return false;
    std::memcpy(u.ptr, idx.data(), idx.size() * 4);
    vkCmdBindIndexBuffer(F->cb, u.buffer, u.offset, VK_INDEX_TYPE_UINT32);
    return true;
}

// [nvn] draw_log_vs = <comma list of vertex shader hash prefixes> (debug, empty = off): every draw with one of them is logged at Info under
// nvn.drawlog with the presented frame, the counts and the vertex stage's first four uniform blocks (notes/fps120.md, typewriter hunt).
void draw_log(uint32_t count, uint32_t instances, bool indexed) {
    static const std::vector<std::string> prefixes = [] {
        std::vector<std::string> v;
        const std::string s = nxrt::config_get_str("nvn", "draw_log_vs", "");
        for (size_t i = 0; i < s.size();) {
            size_t j = s.find(',', i);
            if (j == std::string::npos) j = s.size();
            if (j > i) v.push_back(s.substr(i, j - i));
            i = j + 1;
        }
        return v;
    }();
    if (prefixes.empty() || !S.stage[kVertex]) return;
    const std::string& h = S.stage[kVertex]->hash;
    bool hit = false;
    for (const std::string& p : prefixes) hit = hit || h.compare(0, p.size(), p) == 0;
    if (!hit) return;
    NXLOG_I("nvn.drawlog", "frame %lld vs %s fs %s %s count %u instances %u ubo %llx/%llu %llx/%llu %llx/%llu %llx/%llu",
            static_cast<long long>(counters().presents.load()), h.c_str(), S.stage[kFragment] ? S.stage[kFragment]->hash.c_str() : "-",
            indexed ? "indexed" : "arrays", count, instances, static_cast<unsigned long long>(S.ubo[kVertex][0].addr),
            static_cast<unsigned long long>(S.ubo[kVertex][0].size), static_cast<unsigned long long>(S.ubo[kVertex][1].addr),
            static_cast<unsigned long long>(S.ubo[kVertex][1].size), static_cast<unsigned long long>(S.ubo[kVertex][2].addr),
            static_cast<unsigned long long>(S.ubo[kVertex][2].size), static_cast<unsigned long long>(S.ubo[kVertex][3].addr),
            static_cast<unsigned long long>(S.ubo[kVertex][3].size));
}

void do_draw_arrays(const P_DrawArrays& p) {
    count_primitives(p.mode, p.count, p.instance_count);
    VkPrimitiveTopology topo;
    if (!begin_draw(static_cast<int>(p.mode), false, 0, &topo)) { counters().draws_skipped.fetch_add(1); return; }
    draw_log(p.count, p.instance_count, false);
    const PrimConversion conv = prim_conversion(static_cast<int>(p.mode));
    if (conv == PrimConversion::None) {
        TRACE("    vertices %u instances %u first %u", p.count, p.instance_count, p.first);
        vkCmdDraw(F->cb, p.count, std::max<uint32_t>(1, p.instance_count), p.first, p.base_instance);
    } else {
        // QUADS / QUAD_STRIP / POLYGON / LINE_LOOP: generate an index buffer
        std::vector<uint32_t> in(p.count);
        for (uint32_t k = 0; k < p.count; ++k) in[k] = p.first + k;
        const std::vector<uint32_t> idx = convert_prim_indices(conv, in);
        if (idx.empty()) return;
        if (!bind_u32_indices(idx)) return;
        vkCmdDrawIndexed(F->cb, static_cast<uint32_t>(idx.size()), std::max<uint32_t>(1, p.instance_count), 0, 0, p.base_instance);
    }
    rs_mark(kRsCall);
    counters().draws.fetch_add(1, std::memory_order_relaxed);
    g_pass_draws++;
    g_pass_verts += static_cast<uint64_t>(p.count) * std::max<uint32_t>(1, p.instance_count);
    trace_snap_after_draw();
}

void do_draw_elements(const P_DrawElements& p) {
    count_primitives(p.mode, p.count, p.instance_count);
    VkPrimitiveTopology topo;
    cs_draw_index_range(p.index_address, static_cast<uint64_t>(p.count) * index_size_bytes(static_cast<int>(p.type)));
    const bool began = begin_draw(static_cast<int>(p.mode), true, static_cast<int>(p.type), &topo);
    cs_draw_index_range(0, 0);
    if (!began) { counters().draws_skipped.fetch_add(1); return; }
    draw_log(p.count, p.instance_count, true);
    if (g_trace) {   // the indices and the first attributes of the first vertices
        const uint32_t isz0 = index_size_bytes(static_cast<int>(p.type));
        const uint8_t* ix = host_ptr(p.index_address);
        TRACE("    indexed: count %u type %d instances %u base_vertex %d base_instance %u index address %llx", p.count, static_cast<int>(p.type), p.instance_count, static_cast<int>(p.base_vertex),
              p.base_instance, static_cast<unsigned long long>(p.index_address));
        for (uint32_t i = 0; ix && i < std::min<uint32_t>(p.count, 12); ++i) {
            uint32_t v = 0;
            if (isz0 == 1) v = ix[i]; else if (isz0 == 2) { uint16_t t; std::memcpy(&t, ix + i * 2, 2); v = t; } else std::memcpy(&v, ix + i * 4, 4);
            std::string a;
            for (uint32_t k = 0; k < S.attr_count && k < 3; ++k) {
                const VertexAttribInfo& at = S.attr[k];
                const auto& st = S.stream[at.stream];
                const GState::Bind& vb = S.vtx[at.stream];
                if (!vb.addr) continue;
                const float* f = reinterpret_cast<const float*>(vb.addr + static_cast<uint64_t>(static_cast<int64_t>(v) + p.base_vertex) * st.stride + at.relative_offset);
                char t[96]; std::snprintf(t, sizeof t, " a%u(fmt %d)=(%g %g %g)", k, at.format, f[0], f[1], f[2]);
                a += t;
            }
            TRACE("      index %u:%s", v, a.c_str());
        }
    }
    if (g_trace) {   // bounds of the first attributes over all indices of the draw (half and float formats decoded), to see what part of a texture a mesh maps
        const uint32_t isz1 = index_size_bytes(static_cast<int>(p.type));
        const uint8_t* ix = host_ptr(p.index_address);
        if (ix && p.count) {
            std::string a;
            for (uint32_t k = 0; k < S.attr_count && k < 3; ++k) {
                const VertexAttribInfo& at = S.attr[k];
                const auto& st = S.stream[at.stream];
                const GState::Bind& vb = S.vtx[at.stream];
                if (!vb.addr) continue;
                float lo[4] = {1e30f, 1e30f, 1e30f, 1e30f}, hi[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
                int nc = at.format == 41 ? 4 : at.format == 34 ? 3 : at.format == 17 ? 2 : at.format == 22 ? 2 : 0;
                for (uint32_t i = 0; nc && i < p.count && i < 20000; ++i) {
                    uint32_t v = 0;
                    if (isz1 == 1) v = ix[i]; else if (isz1 == 2) { uint16_t t; std::memcpy(&t, ix + i * 2, 2); v = t; } else std::memcpy(&v, ix + i * 4, 4);
                    const uint8_t* ve = reinterpret_cast<const uint8_t*>(vb.addr + static_cast<uint64_t>(static_cast<int64_t>(v) + p.base_vertex) * st.stride + at.relative_offset);
                    for (int c = 0; c < nc; ++c) {
                        float f;
                        if (at.format == 41 || at.format == 17) {
                            uint16_t h; std::memcpy(&h, ve + c * 2, 2);
                            const int e = (h >> 10) & 31, m = h & 1023; const float sg = (h & 0x8000) ? -1.f : 1.f;
                            f = e == 0 ? sg * m * (1.f / 16777216.f) : e == 31 ? sg * 65504.f : sg * (1.f + m / 1024.f) * std::ldexp(1.f, e - 15);
                        } else std::memcpy(&f, ve + c * 4, 4);
                        lo[c] = std::min(lo[c], f); hi[c] = std::max(hi[c], f);
                    }
                }
                char t[200];
                if (nc) std::snprintf(t, sizeof t, " a%u[%g..%g, %g..%g, %g..%g, %g..%g]", k, lo[0], hi[0], lo[1], hi[1], nc > 2 ? lo[2] : 0.f, nc > 2 ? hi[2] : 0.f, nc > 3 ? lo[3] : 0.f, nc > 3 ? hi[3] : 0.f);
                else std::snprintf(t, sizeof t, " a%u(fmt %d: not decoded)", k, at.format);
                a += t;
            }
            TRACE("    bounds:%s", a.c_str());
        }
    }
    const uint32_t isz = index_size_bytes(static_cast<int>(p.type));
    const PrimConversion conv = prim_conversion(static_cast<int>(p.mode));
    const uint8_t* src = host_ptr(p.index_address);
    if (conv != PrimConversion::None || isz == 1) {
        // widen to 32 bit and/or convert the primitive
        std::vector<uint32_t> in(p.count);
        for (uint32_t i = 0; i < p.count; ++i) {
            uint32_t v = 0;
            if (isz == 1) v = src[i];
            else if (isz == 2) { uint16_t t; std::memcpy(&t, src + i * 2, 2); v = t; }
            else std::memcpy(&v, src + i * 4, 4);
            in[i] = v;
        }
        const std::vector<uint32_t> idx = convert_prim_indices(conv, in);
        if (idx.empty()) return;
        if (!bind_u32_indices(idx)) return;
        vkCmdDrawIndexed(F->cb, static_cast<uint32_t>(idx.size()), std::max<uint32_t>(1, p.instance_count), 0, p.base_vertex, p.base_instance);
    } else {
        BufRef br = resolve_read(p.index_address, static_cast<uint64_t>(p.count) * isz, kReadIndex);
        vkCmdBindIndexBuffer(F->cb, br.buffer, br.offset, vk_index_type(static_cast<int>(p.type)));
        vkCmdDrawIndexed(F->cb, p.count, std::max<uint32_t>(1, p.instance_count), 0, p.base_vertex, p.base_instance);
    }
    rs_mark(kRsCall);
    counters().draws.fetch_add(1, std::memory_order_relaxed);
    g_pass_draws++;
    g_pass_verts += static_cast<uint64_t>(p.count) * std::max<uint32_t>(1, p.instance_count);
    trace_snap_after_draw();
}

// The bytes at a GPU address as the CPU sees them, when they are current: inside a pool, and not a range a submission that has not completed
// may write (gw_settled). Null otherwise. For decisions that need data the GPU would otherwise hold (indirect arguments, draw counts).
// Compute results held in the device copy ([nvn] compute_out_lazy) are newer than the pool: they are fetched back first (out_fetch: a
// completed submission's result, or one whose copy-back is recorded but has not run); one the submission being recorded writes cannot be
// read (null). Without the fetch an unguarded held result would be read stale, and a guarded one through a fault.
const uint8_t* cpu_view(uint64_t addr, uint64_t size) {
    if (!addr || !size) return nullptr;
    const PoolHit hit = pool_lookup(addr);
    if (!hit.pool || hit.pool->is_virtual || !hit.pool->base || hit.offset + size > hit.pool->size) return nullptr;
    if (!gw_settled(addr, size)) return nullptr;
    if (out_any()) {
        out_fetch(addr, size, false);
        if (out_overlaps(addr, size)) return nullptr;
    }
    return host_ptr(addr);
}

// An indirect draw Vulkan cannot take as it is (quads, quad strips, polygons and line loops have no topology; 8-bit indices need
// VK_EXT_index_type_uint8): each draw's arguments (and its indices) are read by the CPU and the draw is made as a direct indexed draw of
// the converted 32-bit index list, like DrawArrays / DrawElements. That needs the arguments in memory the CPU can read: arguments a compute
// shader or a copy wrote and no completed submission has handed back are not (skipped, logged). A multi-draw reads its count the same way.
void draw_indirect_cpu(const P_DrawIndirect& p, PrimConversion conv, uint32_t isz, bool indexed) {
    static bool noted = false;
    if (!noted) {
        noted = true;
        NXLOG_I("nvn.replay", "replay: indirect draw by CPU index conversion (first): mode %u, %s, index size %u, %s", p.mode, indexed ? "indexed" : "arrays", isz,
                p.multi ? "multi-draw" : "single");
    }
    const uint64_t cmd_size = indexed ? 20 : 16;
    const uint64_t stride = p.stride > 0 ? static_cast<uint64_t>(p.stride) : cmd_size;
    uint32_t n = 1;
    if (p.multi && p.count_address) {
        const uint8_t* c = cpu_view(p.count_address, 4);
        if (!c) { note_unsupported("indirect draw conversion: the draw count is not readable by the CPU (written by the GPU)"); counters().draws_skipped.fetch_add(1); return; }
        std::memcpy(&n, c, 4);
        n = std::min(n, p.max_count);
    }
    bool drew = false;
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t* a = cpu_view(p.indirect_address + static_cast<uint64_t>(i) * stride, cmd_size);
        if (!a) { note_unsupported("indirect draw conversion: the arguments are not readable by the CPU (written by the GPU)"); break; }
        uint32_t v[5] = {};
        std::memcpy(v, a, cmd_size);
        const uint32_t count = v[0], instances = v[1];
        if (!count || !instances) continue;
        std::vector<uint32_t> in(count);
        int32_t base_vertex = 0;
        uint32_t base_instance = v[3];
        if (indexed) {
            base_vertex = static_cast<int32_t>(v[3]);
            base_instance = v[4];
            const uint8_t* src = cpu_view(p.index_address + static_cast<uint64_t>(v[2]) * isz, static_cast<uint64_t>(count) * isz);
            if (!src) { note_unsupported("indirect draw conversion: the indices are not readable by the CPU (written by the GPU)"); break; }
            for (uint32_t k = 0; k < count; ++k) {
                uint32_t x = 0;
                if (isz == 1) x = src[k];
                else if (isz == 2) { uint16_t t; std::memcpy(&t, src + k * 2, 2); x = t; }
                else std::memcpy(&x, src + k * 4, 4);
                in[k] = x;
            }
        } else {
            for (uint32_t k = 0; k < count; ++k) in[k] = v[2] + k;
        }
        count_primitives(p.mode, count, instances);
        const std::vector<uint32_t> idx = convert_prim_indices(conv, in);
        if (idx.empty() || !bind_u32_indices(idx)) continue;
        vkCmdDrawIndexed(F->cb, static_cast<uint32_t>(idx.size()), instances, 0, base_vertex, base_instance);
        drew = true;
    }
    if (drew) {
        rs_mark(kRsCall);
    counters().draws.fetch_add(1, std::memory_order_relaxed);
        g_pass_draws++;
    } else {
        counters().draws_skipped.fetch_add(1);
    }
    trace_snap_after_draw();
}

void do_draw_indirect(const P_DrawIndirect& p) {
    const bool indexed = p.type != 0xffffffffu;
    VkPrimitiveTopology topo;
    if (!begin_draw(static_cast<int>(p.mode), indexed, indexed ? static_cast<int>(p.type) : 0, &topo)) { counters().draws_skipped.fetch_add(1); return; }
    const PrimConversion conv = prim_conversion(static_cast<int>(p.mode));
    const uint32_t isz = indexed ? index_size_bytes(static_cast<int>(p.type)) : 0;
    if (conv != PrimConversion::None || (indexed && isz == 1 && !device().features().index_type_uint8)) {
        draw_indirect_cpu(p, conv, isz, indexed);
        return;
    }
    // GPU path. The primitives of these draws are not known to the CPU: the primitive counters see them through a query window (ResetCounter).
    static bool noted_indirect = false, noted_u8 = false;
    if (!noted_indirect) { noted_indirect = true; NXLOG_I("nvn.replay", "replay: indirect draw (first): mode %u, %s%s", p.mode, indexed ? "indexed" : "arrays", p.multi ? ", multi-draw" : ""); }
    if (indexed && isz == 1 && !noted_u8) { noted_u8 = true; NXLOG_I("nvn.replay", "replay: indirect draw with 8-bit indices (first): VK_EXT_index_type_uint8"); }
    const uint64_t cmd_size = indexed ? 20 : 16;
    const uint64_t n_cmds = p.multi && p.count_address ? std::max<uint64_t>(1, p.max_count) : 1;
    BufRef ib = held_or_pool(p.indirect_address, (p.stride ? static_cast<uint64_t>(p.stride) : cmd_size) * (n_cmds - 1) + cmd_size, kOutIndirect, false);
    if (indexed) {
        BufRef xb = resolve_buffer(p.index_address, 4);
        vkCmdBindIndexBuffer(F->cb, xb.buffer, xb.offset, vk_index_type(static_cast<int>(p.type)));
    }
    if (p.multi && p.count_address) {
        BufRef cb = held_or_pool(p.count_address, 4, kOutIndirect, false);
        const uint32_t stride = static_cast<uint32_t>(p.stride ? p.stride : cmd_size);
        if (indexed) vkCmdDrawIndexedIndirectCount(F->cb, ib.buffer, ib.offset, cb.buffer, cb.offset, p.max_count, stride);
        else vkCmdDrawIndirectCount(F->cb, ib.buffer, ib.offset, cb.buffer, cb.offset, p.max_count, stride);
    } else if (indexed) {
        vkCmdDrawIndexedIndirect(F->cb, ib.buffer, ib.offset, 1, 20);
    } else {
        vkCmdDrawIndirect(F->cb, ib.buffer, ib.offset, 1, 16);
    }
    rs_mark(kRsCall);
    counters().draws.fetch_add(1, std::memory_order_relaxed);
    g_pass_draws++;
    trace_snap_after_draw();
}

// DrawTransformFeedback(mode, control): draws what the feedback section begun with this control buffer wrote, as many vertices as it wrote
// (vkCmdDrawIndirectByteCountEXT over the section's byte counter, divided by the stride of the vertex stream that reads the feedback buffer). The game
// binds the feedback buffer as an ordinary vertex buffer. The feedback writes are ordered before the vertex fetch by ending the pass first.
void do_draw_xfb(const P_DrawXfb& p) {
    static bool noted = false;
    if (!noted) { noted = true; NXLOG_I("nvn.xfb", "replay: transform feedback draw (first): mode %u control %llx", p.mode, static_cast<unsigned long long>(p.control)); }
    if (!S.render_enable) return;
    if (!device().features().transform_feedback) { note_unsupported("DrawTransformFeedback (the device has no VK_EXT_transform_feedback)"); counters().draws_skipped.fetch_add(1); return; }
    const auto it = g_xfb_slot_of.find(p.control);
    if (!g_xfb_counter || it == g_xfb_slot_of.end() || !g_xfb_slot_written[it->second]) {
        TRACE("DrawTransformFeedback control %llx: the section wrote nothing", static_cast<unsigned long long>(p.control));   // zero bytes: zero vertices
        counters().draws_skipped.fetch_add(1);
        return;
    }
    if (prim_conversion(static_cast<int>(p.mode)) != PrimConversion::None) { note_unsupported("DrawTransformFeedback of a quad/polygon/line-loop primitive"); return; }
    const uint32_t slot = it->second;
    end_pass();
    memory_barrier(F->cb);   // the section's writes (feedback buffer, counter) before the vertex fetch and the indirect read
    VkPrimitiveTopology topo;
    if (!begin_draw(static_cast<int>(p.mode), false, 0, &topo)) { counters().draws_skipped.fetch_add(1); return; }
    uint32_t stride = g_key_attr_count ? S.stream[g_key_binding[0]].stride : 0;
    for (uint32_t b = 0; !stride && b < 16; ++b) stride = S.stream[b].stride;
    if (!stride) { note_unsupported("DrawTransformFeedback with no vertex stream stride"); counters().draws_skipped.fetch_add(1); return; }
    vkCmdDrawIndirectByteCountEXT(F->cb, 1, 0, g_xfb_counter.buffer, static_cast<VkDeviceSize>(slot) * kXfbSlotBytes, 0, stride);
    rs_mark(kRsCall);
    counters().draws.fetch_add(1, std::memory_order_relaxed);
    g_pass_draws++;
    trace_snap_after_draw();
}

std::string g_cs_missing;   // trace: the untranslated block(s) of the bound compute program ("cs -" dispatches)

// [nvn] trace_snap_cs = "98acab99,e8d64471": after every dispatch of the traced frame whose compute shader hash starts with one of these,
// the textures and images it binds are copied (png after the frame: snap_c<dispatch>_t<id>_<format>.png; float formats also as .bin).
void trace_snap_after_dispatch(const ShaderInfo& cs, const ProgLayout& pl) {
    static const std::vector<std::string> list = [] {
        std::vector<std::string> v;
        const std::string s = nxrt::config_get_str("nvn", "trace_snap_cs", "");
        for (size_t k = 0; k < s.size();) {
            size_t j = s.find(',', k);
            if (j == std::string::npos) j = s.size();
            if (j > k) v.push_back(s.substr(k, j - k));
            k = j + 1;
        }
        return v;
    }();
    static int n = 0;
    ++n;
    bool hit = false;
    for (const std::string& pre : list) hit |= cs.hash.compare(0, pre.size(), pre) == 0;
    if (!hit) return;
    std::vector<Texture*> done;
    for (const BindingDesc& d : pl.bindings) {
        const bool img = d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if (!img && d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        Texture* t = (img ? resolve_image(d) : resolve_texture(d)).tex;
        if (!t || std::find(done.begin(), done.end(), t) != done.end()) continue;
        done.push_back(t);
        const int idx = snapshot_record(t, *F);
        if (idx < 0) continue;
        char buf[96];
        g_trace_snaps.push_back({idx, -n, "snap_c" + std::to_string(n) + "_" + cs.hash.substr(0, 8) + "_t" + std::to_string(t->id) + "_" +
                                                 (t->fmt ? t->fmt->name : "x") + ".png", tex_desc(t, buf, sizeof buf)});
    }
}

// ---- dispatch barriers ([nvn] dispatch_barriers) ----------------------------------------------------------------------------
// `full` (default): a full barrier after every dispatch, as before. `tracked`: a dispatch leaves its barrier open, and the next dispatch
// gets one only when it touches what an open dispatch may write, or may write what one reads: buffers by their game address range (storage
// buffers count as written), textures and storage images by their image. Any other command closes the open dispatches with the full
// barrier first, and so does the end of the submission. Dispatches that share nothing (one per character, separate buffers) then overlap
// on the GPU instead of each one draining it. A dispatch with a binding this cannot follow (texel buffers, an unknown source) keeps the
// full barrier after it.
struct CsRange { uint64_t a, b; };
// An image use: the image and the NVN levels [l0, l1) and layers [a0, a1) its view covers.
struct CsImg {
    const void* img;
    uint32_t l0, l1, a0, a1;
};
struct CsSet {
    std::vector<CsRange> rd, wr;
    std::vector<CsImg> img_rd, img_wr;
    bool pool = false;                                     // samples through the texture pool: any texture
    void clear() { rd.clear(); wr.clear(); img_rd.clear(); img_wr.clear(); pool = false; }
    size_t size() const { return rd.size() + wr.size() + img_rd.size() + img_wr.size(); }
};
CsSet g_cs_open_set, g_cs_new;
bool g_cs_open = false;                                    // dispatches ended without their barrier
bool g_cs_known = false;                                   // g_cs_new holds everything the current dispatch binds
VkCommandBuffer g_cs_cb = VK_NULL_HANDLE;
uint64_t g_cs_mark = 0;                                    // memory_barrier_count(g_cs_cb) when the barrier was left open
uint64_t g_cs_n = 0, g_cs_skipped = 0, g_cs_conflicts = 0, g_cs_untracked = 0;

// `tracked_draws`: as `tracked`, and draws stay outside the check too: a draw (DrawArrays / DrawElements) that binds nothing an open
// dispatch may write, and renders to no image one writes, runs without the barrier, so the dispatch overlaps the graphics work after it
// (the probe-filter dispatches of Fortune's Weave sit between unrelated passes). At most 32 draws are checked per dispatch; then, or at a
// draw that depends on an open dispatch, the pass ends and the barrier is recorded.
int cs_mode() {
    static const int m = [] {
        const std::string v = nxrt::config_get_str("nvn", "dispatch_barriers", "full");
        return v == "tracked_draws" ? 2 : v == "tracked" ? 1 : 0;
    }();
    return m;
}
bool cs_tracked_on() { return cs_mode() >= 1; }
bool cs_draws_on() { return cs_mode() >= 2; }
uint32_t g_cs_draws_checked = 0;
uint64_t g_cs_draw_overlaps = 0, g_cs_draw_conflicts = 0;
std::vector<CsRange> g_draw_extra_rd;                      // the draw being set up: its index range (do_draw_elements)

const void* cs_img_key(Texture* t) { return t->gpu ? static_cast<const void*>(t->gpu) : static_cast<const void*>(t); }

// The subresources a binding's view covers (all of them without a view, and always for 3D images, whose layers are slices).
CsImg cs_img(Texture* t, const TextureViewInfo* v) {
    CsImg u{cs_img_key(t), 0, UINT32_MAX, 0, UINT32_MAX};
    if (!v || t->b.target == NVN_TEXTURE_TARGET_3D) return u;
    if (v->has_levels && v->min_level >= 0) {
        u.l0 = static_cast<uint32_t>(v->min_level);
        u.l1 = v->num_levels > 0 ? u.l0 + static_cast<uint32_t>(v->num_levels) : UINT32_MAX;
    }
    if (v->has_layers && v->min_layer >= 0) {
        u.a0 = static_cast<uint32_t>(v->min_layer);
        u.a1 = v->num_layers > 0 ? u.a0 + static_cast<uint32_t>(v->num_layers) : UINT32_MAX;
    }
    return u;
}

// Bindings of a module that the shader only queries (textureSize / textureQueryLevels / samples), never reads: (set << 16 | binding).
// A dispatch that only asks an image for its size does not depend on what writes it (the probe filters of Fortune's Weave bind the
// image they write as a texture too, for its size).
bool spv_query_only(const ShaderInfo* sh, uint32_t set, uint32_t binding) {
    static std::unordered_map<const ShaderInfo*, std::vector<uint32_t>> cache;
    auto it = cache.find(sh);
    if (it == cache.end()) {
        std::vector<uint32_t> out;
        const std::vector<uint32_t>& w = sh->spirv;
        std::unordered_map<uint32_t, uint32_t> src, sets, binds;
        std::unordered_map<uint32_t, int> use;          // root variable -> 1 queried only, 2 read
        std::vector<std::pair<uint32_t, int>> ops;      // (image operand id, kind)
        for (size_t i = 5; i < w.size();) {
            const uint32_t n = w[i] >> 16, op = w[i] & 0xffffu;
            if (n == 0 || i + n > w.size()) break;
            const uint32_t* a = &w[i + 1];
            if (op == 71 && n >= 4 && (a[1] == 33 || a[1] == 34)) (a[1] == 33 ? binds : sets)[a[0]] = a[2];   // Decorate Binding / DescriptorSet
            else if ((op == 61 || op == 86 || op == 100 || op == 65 || op == 66) && n >= 4) src[a[1]] = a[2];  // Load, SampledImage, Image, AccessChain
            else if (op >= 103 && op <= 107 && n >= 4) ops.push_back({a[2], 1});          // ImageQuery*
            else if (op >= 87 && op <= 98 && n >= 4) ops.push_back({a[2], 2});            // ImageSample*, Fetch, Gather, Read, ...
            else if (op == 99 && n >= 2) ops.push_back({a[0], 2});                        // ImageWrite
            else if (((op >= 305 && op <= 320) || op == 60 || op >= 4500) && n >= 4) ops.push_back({a[2], 2});   // sparse, texel pointer, newer
            i += n;
        }
        auto root = [&](uint32_t x) {
            for (int k = 0; k < 16; ++k) {
                auto f = src.find(x);
                if (f == src.end()) break;
                x = f->second;
            }
            return x;
        };
        for (const auto& o : ops) {
            int& u = use[root(o.first)];
            u = std::max(u, o.second);
        }
        for (const auto& b : binds) {
            const auto u = use.find(b.first);
            if (u != use.end() && u->second == 1) out.push_back(((sets.count(b.first) ? sets[b.first] : 0u) << 16) | b.second);
        }
        it = cache.emplace(sh, std::move(out)).first;
    }
    const uint32_t key = set << 16 | binding;
    return std::find(it->second.begin(), it->second.end(), key) != it->second.end();
}

// Everything the dispatch about to run binds. False: a binding this does not follow.
bool cs_collect(const ProgLayout& pl, uint64_t indirect, CsSet& o) {
    o.clear();
    o.pool = pl.has_set[kPoolSet];
    if (indirect) o.rd.push_back({indirect, indirect + 12});
    for (const BindingDesc& d : pl.bindings) {
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        const int st = d.nvn_stage;
        if (st < 0 || st >= 6) return false;
        switch (d.type) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                switch (b.nvn.source) {
                    case NvnSource::SupportBuffer: case NvnSource::DriverConstants: case NvnSource::ConstantData: case NvnSource::PushConstants:
                        break;                                     // written by the CPU (upload memory, constant data)
                    case NvnSource::UniformBlock: case NvnSource::UniformAsStorage: {
                        const int idx = b.nvn.index;
                        if (idx >= 0 && idx < 16 && S.ubo[st][idx].addr) {
                            const GState::Bind& bd = S.ubo[st][idx];
                            o.rd.push_back({bd.addr, bd.addr + (bd.size ? bd.size : 65536)});
                        }
                        break;
                    }
                    case NvnSource::StorageBuffer: {
                        const int slot = b.nvn.index >= 0 ? b.nvn.index : storage_slot_for(st, b.nvn.c0_offset);
                        if (slot < 0 || slot >= 16) return false;
                        if (S.ssbo[st][slot].addr) {
                            const GState::Bind& bd = S.ssbo[st][slot];
                            o.wr.push_back({bd.addr, bd.size ? bd.addr + bd.size : UINT64_MAX});
                        }
                        break;
                    }
                    default: return false;
                }
                break;
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: {
                if (b.nvn.source == NvnSource::TexturePoolArray) { o.pool = true; break; }
                if (b.nvn.source == NvnSource::SamplerPoolArray) break;
                if (spv_query_only(d.shader, b.set, b.binding)) break;        // only its size is asked
                const TexResolved r = resolve_texture(d);
                if (r.tex) o.img_rd.push_back(cs_img(r.tex, r.has_view ? &r.view : nullptr));
                break;
            }
            case VK_DESCRIPTOR_TYPE_SAMPLER: break;
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
                const TexResolved r = resolve_image(d);
                if (r.tex) o.img_wr.push_back(cs_img(r.tex, r.has_view ? &r.view : nullptr));
                break;
            }
            default: return false;                                 // texel buffers (pool memory seen as texels) and anything else
        }
    }
    return true;
}

bool cs_overlap(const std::vector<CsRange>& a, const std::vector<CsRange>& b) {
    for (const CsRange& x : a)
        for (const CsRange& y : b)
            if (x.a < y.b && y.a < x.b) return true;
    return false;
}
bool cs_has(const std::vector<CsImg>& a, const std::vector<CsImg>& b) {
    for (const CsImg& x : a)
        for (const CsImg& y : b)
            if (x.img == y.img && x.l0 < y.l1 && y.l0 < x.l1 && x.a0 < y.a1 && y.a0 < x.a1) return true;
    return false;
}
bool cs_conflict(const CsSet& open, const CsSet& n) {
    return cs_overlap(n.wr, open.rd) || cs_overlap(n.wr, open.wr) || cs_overlap(n.rd, open.wr) || cs_has(n.img_wr, open.img_rd) ||
           cs_has(n.img_wr, open.img_wr) || cs_has(n.img_rd, open.img_wr) || (n.pool && !open.img_wr.empty()) || (open.pool && !n.img_wr.empty());
}

// The open dispatches get their barrier now (unless one was recorded since).
void cs_close() {
    if (!g_cs_open) return;
    g_cs_open = false;
    if (F && F->cb == g_cs_cb && memory_barrier_count(F->cb) == g_cs_mark) memory_barrier(F->cb);
    g_cs_open_set.clear();
}

// Right before the dispatch command: the barrier the open dispatches need, if the new one depends on them.
void cs_before_dispatch(const ProgLayout& pl, uint64_t indirect) {
    if (!cs_tracked_on()) return;
    g_cs_known = cs_collect(pl, indirect, g_cs_new);
    if (!g_cs_open) return;
    if (F->cb != g_cs_cb || memory_barrier_count(F->cb) != g_cs_mark) {      // a full barrier lies between: nothing is open any more
        g_cs_open = false;
        g_cs_open_set.clear();
    } else if (!g_cs_known || cs_conflict(g_cs_open_set, g_cs_new)) {
        memory_barrier(F->cb);
        ++g_cs_conflicts;
        g_cs_open = false;
        g_cs_open_set.clear();
    } else {
        ++g_cs_skipped;
        counters().dispatch_barriers_left_out.fetch_add(1, std::memory_order_relaxed);
    }
}

// After the dispatch (and its shadow copies): the barrier now, or left open.
void cs_after_dispatch() {
    if (!cs_tracked_on()) { memory_barrier(F->cb); return; }
    if (++g_cs_n % 20000 == 0)
        NXLOG_I("nvn.replay", "dispatch barriers (tracked): %llu dispatches, %llu barriers left out, %llu kept for a dependency, %llu dispatches not tracked; "
                "draws after open dispatches: %llu ran without the barrier, %llu got it",
                static_cast<unsigned long long>(g_cs_n), static_cast<unsigned long long>(g_cs_skipped), static_cast<unsigned long long>(g_cs_conflicts),
                static_cast<unsigned long long>(g_cs_untracked), static_cast<unsigned long long>(g_cs_draw_overlaps),
                static_cast<unsigned long long>(g_cs_draw_conflicts));
    if (!g_cs_known || g_cs_open_set.size() + g_cs_new.size() > 256) {
        if (!g_cs_known) ++g_cs_untracked;
        memory_barrier(F->cb);
        g_cs_open = false;
        g_cs_open_set.clear();
        return;
    }
    if (!g_cs_open) g_cs_open_set.clear();
    CsSet& o = g_cs_open_set;
    o.rd.insert(o.rd.end(), g_cs_new.rd.begin(), g_cs_new.rd.end());
    o.wr.insert(o.wr.end(), g_cs_new.wr.begin(), g_cs_new.wr.end());
    o.img_rd.insert(o.img_rd.end(), g_cs_new.img_rd.begin(), g_cs_new.img_rd.end());
    o.img_wr.insert(o.img_wr.end(), g_cs_new.img_wr.begin(), g_cs_new.img_wr.end());
    o.pool = o.pool || g_cs_new.pool;
    g_cs_open = true;
    g_cs_draws_checked = 0;
    g_cs_cb = F->cb;
    g_cs_mark = memory_barrier_count(F->cb);
}

// Ops that record no GPU work (state and bindings) or that check the open dispatches themselves keep them open.
bool game_barriers_tracked() {
    static const bool tracked = nxrt::config_get_str("nvn", "game_barriers", "full") == "tracked";
    return tracked;
}

bool cs_keeps_open(Op op) {
    if (op < Op::DrawArrays) return true;
    if (op == Op::Barrier && game_barriers_tracked()) return true;      // the dispatches' dependencies are followed here (case Op::Barrier)
    if ((op == Op::DrawArrays || op == Op::DrawElements) && cs_draws_on()) return true;    // begin_draw checks (cs_check_draw)
    switch (op) {
        case Op::Dispatch: case Op::DispatchIndirect: case Op::FenceSync: case Op::WaitSync: case Op::CallCommands: case Op::PushDebugGroup:
        case Op::PopDebugGroup: case Op::PushConstants: case Op::DiscardColor: case Op::DiscardDepthStencil:
            return true;
        default: return false;
    }
}

void cs_draw_index_range(uint64_t addr, uint64_t size) {
    g_draw_extra_rd.clear();
    if (addr) g_draw_extra_rd.push_back({addr, size ? addr + size : UINT64_MAX});
}

// A draw with dispatches open (dispatch_barriers = tracked_draws): the barrier now (ending the pass) if the draw depends on one of them.
void cs_check_draw(const ProgLayout& pl) {
    if (!g_cs_open) return;
    if (F->cb != g_cs_cb || memory_barrier_count(F->cb) != g_cs_mark) {      // a full barrier lies between
        g_cs_open = false;
        g_cs_open_set.clear();
        return;
    }
    bool dep = !cs_draws_on() || ++g_cs_draws_checked > 32 || g_xfb_active;
    if (!dep) {
        CsSet& d = g_cs_new;
        dep = !cs_collect(pl, 0, d);
        if (!dep) {
            for (const GState::Bind& v : S.vtx)
                if (v.addr) d.rd.push_back({v.addr, v.size ? v.addr + v.size : UINT64_MAX});
            d.rd.insert(d.rd.end(), g_draw_extra_rd.begin(), g_draw_extra_rd.end());
            for (uint32_t i = 0; i < S.ncolor && i < 8; ++i)
                if (S.color_tex[i]) d.img_wr.push_back(cs_img(S.color_tex[i].get(), nullptr));
            if (S.depth_tex) d.img_wr.push_back(cs_img(S.depth_tex.get(), nullptr));
            dep = cs_conflict(g_cs_open_set, d);
        }
    }
    if (dep) {
        ++g_cs_draw_conflicts;
        end_pass();
        cs_close();
    } else {
        ++g_cs_draw_overlaps;
    }
}

void do_dispatch(uint32_t x, uint32_t y, uint32_t z, uint64_t indirect) {
    const ShaderInfo* cs = S.stage[kCompute];
    TRACE("dispatch %u %u %u indirect %llx cs %s%s%s", x, y, z, static_cast<unsigned long long>(indirect), cs ? cs->hash.c_str() : "-",
          !cs && !g_cs_missing.empty() ? " missing " : "", !cs ? g_cs_missing.c_str() : "");
    if (!cs) return;
    const ShaderInfo* stages[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, cs};
    const ProgLayout* pl = layout_cached(stages);
    if (!pl) return;
    MemoScope memo(*pl);
    prepare_textures(*pl);
    end_pass();
    if (g_trace) {   // the compute bindings: buffers (source, slot, address), texel buffers and textures, and the first words of each uniform block
        for (const BindingDesc& d : pl->bindings) {
            const DescriptorBinding& b = d.shader->refl.bindings[d.index];
            if (d.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || d.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                const int slot = b.nvn.source == NvnSource::StorageBuffer ? (b.nvn.index >= 0 ? b.nvn.index : storage_slot_for(d.nvn_stage, b.nvn.c0_offset)) : b.nvn.index;
                const bool ssbo = b.nvn.source == NvnSource::StorageBuffer;
                const GState::Bind* bd = (slot >= 0 && slot < 16) ? (ssbo ? &S.ssbo[d.nvn_stage][slot] : &S.ubo[d.nvn_stage][slot]) : nullptr;
                std::string words;
                if (bd && bd->addr && resolve_buffer(bd->addr, 4).ok) {
                    const uint32_t* w = reinterpret_cast<const uint32_t*>(host_ptr(bd->addr));
                    for (uint32_t i = 0; i < 128 && i * 4 < bd->size; ++i) { char t[16]; std::snprintf(t, sizeof t, " %08x", w[i]); words += t; }
                }
                TRACE("    cs b%u %s src %d slot %d c0 %d addr %llx/%llu:%s", d.binding, b.name.c_str(), static_cast<int>(b.nvn.source), slot, b.nvn.c0_offset,
                      bd ? static_cast<unsigned long long>(bd->addr) : 0ull, bd ? static_cast<unsigned long long>(bd->size) : 0ull, words.c_str());
            } else {
                const bool img = d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || d.type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                const TexResolved r = img ? resolve_image(d) : resolve_texture(d);
                char c[96];
                std::string words;
                if (r.tex && r.tex->mem) {
                    const uint32_t* w = reinterpret_cast<const uint32_t*>(r.tex->mem);
                    for (uint32_t i = 0; i < 64; ++i) { char t[16]; std::snprintf(t, sizeof t, " %08x", w[i]); words += t; }
                }
                TRACE("    cs b%u %s type %d -> %s view L%d+%d A%d+%d%s @%p:%s", d.binding, b.name.c_str(), static_cast<int>(d.type), tex_desc(r.tex, c, sizeof c),
                      r.has_view && r.view.has_levels ? r.view.min_level : -1, r.has_view && r.view.has_levels ? r.view.num_levels : -1,
                      r.has_view && r.view.has_layers ? r.view.min_layer : -1, r.has_view && r.view.has_layers ? r.view.num_layers : -1,
                      spv_query_only(d.shader, b.set, b.binding) ? " (size query only)" : "", r.tex ? static_cast<void*>(r.tex->mem) : nullptr, words.c_str());
            }
        }
    }
    // Float-declared samplers over 8- or 16-bit integer textures (the light probe shader's RGBA8UI cubes): the variant that reads them
    // as integer images, so the shader gets the integer bits as on the console (spirv_int_samplers). 32-bit integer textures keep the
    // translated module: their float alias gives the bits (tex_sampled_view).
    std::vector<SpvIntSampler> ints;
    g_int_pl = nullptr;
    g_int_comp.assign(pl->bindings.size(), 0);
    for (size_t i = 0; i < pl->bindings.size(); ++i) {
        const BindingDesc& d = pl->bindings[i];
        if (d.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        const DescriptorBinding& b = d.shader->refl.bindings[d.index];
        if (b.image.component != ScalarType::Float || b.image.depth || b.count != 1) continue;
        const TexResolved r = resolve_texture(d);
        if (!r.tex || !tex_gpu(r.tex)) continue;
        const uint8_t c = tex_int_sampler_component(r.tex, make_request(b, r).format);
        if (!c) continue;
        g_int_comp[i] = c;
        ints.push_back({b.set, b.binding, c == 2});
    }
    VkPipeline pipe = compute_pipeline(cs, *pl, ints);
    if (pipe && !ints.empty()) g_int_pl = pl;
    else if (!pipe && !ints.empty()) pipe = compute_pipeline(cs, *pl);     // no variant: the translated module, float views (or the dummy)
    if (!pipe) return;
    vkCmdBindPipeline(F->cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    g_bound_pipeline = VK_NULL_HANDLE;
    support_for_draw(*pl, true);
    const BufRef ind = indirect ? held_or_pool(indirect, 12, kOutIndirect, true) : BufRef{};   // arguments a dispatch wrote: from the copy
    g_shadowing = compute_shadows_on();
    g_shadow_uses.clear();
    g_sh_barrier = g_sh_copied = false;
    // NVN2 compute c[0]: the work-group counts at 0x4 (x, y, z; the linearised group index of ee541a5d reads x and y there), the push
    // constant words from 0x10 (control +0xa9), then the resource table, whose descriptor words stay 0: storage buffers are bound at their
    // exact address, so the translated code's base - (c[0] word & ~alignment) offset must see 0 there. Indirect counts are not known here.
    g_cs_c0 = UploadAlloc();
    if (cs->control.nvn2 && std::any_of(cs->refl.bindings.begin(), cs->refl.bindings.end(),
                                        [](const DescriptorBinding& b) { return b.used && b.nvn.source == NvnSource::DriverConstants; })) {
        const VkDeviceSize n = std::max<VkDeviceSize>(0x400, (cs->control.c0_table_size + 0xffu) & ~0xffu);
        g_cs_c0 = F->upload(n, std::max<VkDeviceSize>(device().properties().limits.minUniformBufferOffsetAlignment, 256));
        if (g_cs_c0) {
            std::memset(g_cs_c0.ptr, 0, static_cast<size_t>(n));
            const uint32_t groups[3] = {x, y, z};
            std::memcpy(g_cs_c0.ptr + 4, groups, sizeof groups);
            std::memcpy(g_cs_c0.ptr + 0x10, S.push_cmp, std::min<size_t>(4u * cs->control.c0_push_words, sizeof S.push_cmp));
            if (indirect) NVN_LOG_ONCE("cs-c0-indirect", "indirect dispatch of an NVN2 compute shader that reads c[0]: its work-group counts there read 0");
        }
    }
    bind_sets(*pl, VK_PIPELINE_BIND_POINT_COMPUTE);
    g_int_pl = nullptr;
    if (g_sh_copied) {
        memory_barrier(F->cb);     // the copies into the shadows before the dispatch reads them
        gstamp("shadow in");
    }
    cs_before_dispatch(*pl, indirect);
    if (indirect) {
        vkCmdDispatchIndirect(F->cb, ind.buffer, ind.offset);
    } else {
        vkCmdDispatch(F->cb, x, y, z);
        if (gpass_on()) {
            char lb[64];
            std::snprintf(lb, sizeof lb, "cs %.8s g%ux%ux%u", cs->hash.c_str(), x, y, z);
            gstamp(lb);
        }
    }
    shadows_after_dispatch();
    cs_after_dispatch();
    if (g_trace) trace_snap_after_dispatch(*cs, *pl);
    g_cs_c0 = UploadAlloc();
    g_dirty = kAllDirty;
    counters().dispatches.fetch_add(1, std::memory_order_relaxed);
}

// ---- conditional render ---------------------------------------------------------------------------------------------------------
void note_report(uint64_t address, uint64_t counter) {
    if (!address) return;
    if (g_reports.size() > 65536) g_reports.clear();   // bounded: a forgotten address reads from memory again
    g_reports[address] = counter;
}

// The counter word at a report address; false when it is not known before the GPU runs.
bool report_counter(uint64_t address, uint64_t* out) {
    auto it = g_reports.find(address);
    if (it != g_reports.end()) {
        if (it->second == kReportUnknown) return false;
        *out = it->second;
        return true;
    }
    const PoolHit hit = pool_lookup(address);
    if (!hit.pool || hit.pool->is_virtual || !hit.pool->base || hit.offset + 8 > hit.pool->size) return false;
    std::memcpy(out, host_ptr(address), 8);
    return true;
}

bool conditional_render_enabled(uint32_t mode, uint64_t address) {
    uint64_t a = 0, b = 0;
    if (!report_counter(address, &a) || !report_counter(address + 16, &b)) return true;
    if (mode == NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_EQUAL) return a == b;
    if (mode == NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_NOT_EQUAL) return a != b;
    return true;
}

// ---- clears -------------------------------------------------------------------------------------------------------------------
VkClearRect full_rect() {
    VkClearRect r = {};
    r.rect = S.nsc ? scissor_rect(S.sc[0]) : VkRect2D{{0, 0}, P.extent};
    r.baseArrayLayer = 0;
    r.layerCount = P.layers;
    return r;
}

void do_clear_color(const P_ClearColor& p) {
    if (g_trace) {   // the value (as floats for kind 0, else as words) and the rectangle the clear covers (the scissor when one is set)
        char a[96], v[96];
        if (p.kind == 0) {
            float f[4];
            std::memcpy(f, p.value, sizeof f);
            std::snprintf(v, sizeof v, "%g %g %g %g", f[0], f[1], f[2], f[3]);
        } else {
            std::snprintf(v, sizeof v, "%08x %08x %08x %08x", p.value[0], p.value[1], p.value[2], p.value[3]);
        }
        const VkClearRect cr = full_rect();
        TRACE("clear color rt%u mask %x kind %u %s value %s rect %d %d %u %u", p.target, p.mask, p.kind, tex_desc(p.target < 8 ? S.color_tex[p.target].get() : nullptr, a, sizeof a), v,
              cr.rect.offset.x, cr.rect.offset.y, cr.rect.extent.width, cr.rect.extent.height);
    }
    if (!S.render_enable) return;   // render enable gates clears as it gates draws (yuzu DrawManager::Clear)
    if (p.target >= 8 || !S.color_tex[p.target]) return;
    if (p.mask == 0) return;
    if (!ensure_pass()) return;
    if (!P.color_fmt[p.target]) return;
    if (p.mask != 0xf) {
        // vkCmdClearAttachments ignores the colour write mask: a masked clear is a draw with that mask
        if (!g_mask_ready) { g_mask_ok = g_mask.init(device()); g_mask_ready = true; }
        const VkClearRect cr = full_rect();
        if (g_mask_ok && g_mask.record(F->cb, P.color_fmt[p.target], P.ncolor, p.target, p.mask, p.kind, p.value, cr.rect, P.samples)) {
            static bool noted = false;
            if (!noted) { noted = true; NXLOG_I("nvn.replay", "replay: ClearColor with a partial channel mask (first): rt%u mask %x, drawn with that colour write mask", p.target, p.mask); }
            counters().clears.fetch_add(1, std::memory_order_relaxed);
            g_bound_pipeline = VK_NULL_HANDLE;
            g_dirty = kAllDirty;
            g_topo = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;   // binding a pipeline without these dynamic states invalidates them
            g_restart_valid = false;
            return;
        }
        note_unsupported("ClearColor with a partial channel mask (all channels are cleared)");
    }
    VkClearAttachment ca = {};
    ca.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ca.colorAttachment = p.target;
    std::memcpy(ca.clearValue.color.uint32, p.value, 16);
    VkClearRect r = full_rect();
    vkCmdClearAttachments(F->cb, 1, &ca, 1, &r);
    counters().clears.fetch_add(1, std::memory_order_relaxed);
}

void do_clear_depth_stencil(const P_ClearDepthStencil& p) {
    TRACE("clear depth/stencil d %g (mask %d) s %d (mask %x)", p.depth, p.depth_mask, p.stencil, p.stencil_mask);
    if (!S.render_enable) return;
    if (!S.depth_tex) return;
    if (!ensure_pass()) return;
    VkClearAttachment ca = {};
    if (p.depth_mask) ca.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (p.stencil_mask && P.has_stencil) ca.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    if (!ca.aspectMask) return;
    if (p.stencil_mask && p.stencil_mask != 0xff && P.has_stencil && !P.depth_ro) {
        // vkCmdClearAttachments has no stencil write mask: a masked stencil clear is a draw with that write mask (depth goes along)
        if (!g_mask_ready) { g_mask_ok = g_mask.init(device()); g_mask_ready = true; }
        const VkClearRect cr = full_rect();
        if (g_mask_ok && g_mask.record_depth_stencil(F->cb, P.color_fmt, P.ncolor, P.depth_fmt, true, P.samples, p.depth_mask != 0, p.depth, p.stencil_mask,
                                                    static_cast<uint32_t>(p.stencil), cr.rect)) {
            static bool noted = false;
            if (!noted) { noted = true; NXLOG_I("nvn.replay", "replay: ClearDepthStencil with a partial stencil mask (first): mask %x, drawn with that stencil write mask", p.stencil_mask); }
            counters().clears.fetch_add(1, std::memory_order_relaxed);
            g_bound_pipeline = VK_NULL_HANDLE;
            g_dirty = kAllDirty;
            g_topo = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;   // binding a pipeline without these dynamic states invalidates them
            g_restart_valid = false;
            return;
        }
        note_unsupported("ClearDepthStencil with a partial stencil mask (the whole stencil is cleared)");
    }
    ca.clearValue.depthStencil = {p.depth, static_cast<uint32_t>(p.stencil)};
    VkClearRect r = full_rect();
    vkCmdClearAttachments(F->cb, 1, &ca, 1, &r);
    counters().clears.fetch_add(1, std::memory_order_relaxed);
}

// ---- copies ------------------------------------------------------------------------------------------------------------------
struct SubRes { uint32_t level = 0, layer = 0, layers = 1; };
SubRes sub_of(const TextureViewInfo& v, bool has_view) {
    SubRes s;
    if (has_view) {
        if (v.has_levels) s.level = static_cast<uint32_t>(v.min_level);
        if (v.has_layers) { s.layer = static_cast<uint32_t>(v.min_layer); s.layers = 1; }
    }
    return s;
}

void do_clear_texture(const P_ClearTexture& p) {
    Texture* t = p.tex;
    if (!t || !tex_gpu(t)) return;
    end_pass();
    SubRes s = sub_of(p.view, p.has_view);
    tex_ensure_exact(t, *F);
    tex_transition(t, *F, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    tex_mark_gpu_written(t);
    if (p.has_region) {
        const VkExtent3D e = tex_extent(t, s.level);
        if (p.region.x != 0 || p.region.y != 0 || (p.region.w && static_cast<uint32_t>(p.region.w) < e.width) || (p.region.h && static_cast<uint32_t>(p.region.h) < e.height))
            note_unsupported("ClearTexture of a sub-region (the whole level is cleared)");
    }
    VkImageSubresourceRange rng = {tex_aspect(t), s.level, 1, s.layer, p.has_view && p.view.has_layers ? static_cast<uint32_t>(std::max(1, p.view.num_layers)) : tex_layers(t)};
    if (tex_is_depth(t)) {
        VkClearDepthStencilValue v = {};
        std::memcpy(&v.depth, &p.value[0], 4);
        vkCmdClearDepthStencilImage(F->cb, tex_image(t), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &rng);
    } else {
        VkClearColorValue v = {};
        std::memcpy(v.uint32, p.value, 16);
        vkCmdClearColorImage(F->cb, tex_image(t), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &rng);
    }
    counters().clears.fetch_add(1, std::memory_order_relaxed);
    gstamp("clear texture");
}

void do_copy_buffer_texture(const P_CopyBufTex& p, bool to_texture) {
    Texture* t = p.tex;
    if (t && t->b.target == NVN_TEXTURE_TARGET_3D) {
        static int logged = 0;
        if (logged < 30) {
            ++logged;
            char a[96];
            NXLOG_I("nvn.copy", "%s 3D %s region %s (%d,%d,%d %dx%dx%d) buffer %llx row %lld image %lld", to_texture ? "buffer -> texture" : "texture -> buffer", tex_desc(t, a, sizeof a),
                    p.has_region ? "given" : "whole", p.has_region ? p.region.x : 0, p.has_region ? p.region.y : 0, p.has_region ? p.region.z : 0,
                    p.has_region ? p.region.w : 0, p.has_region ? p.region.h : 0, p.has_region ? p.region.d : 0, static_cast<unsigned long long>(p.address),
                    static_cast<long long>(p.row_stride), static_cast<long long>(p.image_stride));
        }
    }
    if (!t || !tex_gpu(t)) return;
    end_pass();
    if (tex_scale(t) != 1.0f) {
        static int logged = 0;
        if (logged++ < 10) {
            char a[96];
            NXLOG_W(kTag, "%s a scaled render target %s: the copy uses NVN texel coordinates (wrong part of the image)", to_texture ? "buffer copied into" : "copy from",
                    tex_desc(t, a, sizeof a));
        }
    }
    SubRes s = sub_of(p.view, p.has_view);
    const VkExtent3D ext = tex_extent(t, s.level);
    P_CopyRegion r = p.has_region ? p.region : P_CopyRegion{0, 0, 0, static_cast<int32_t>(ext.width), static_cast<int32_t>(ext.height), static_cast<int32_t>(ext.depth)};
    if (const uint32_t drop = tex_level_drop(t)) {   // an image stored without its top level: that level has no texels to copy
        if (s.level < drop) { NVN_LOG_ONCE("copy-dropped-level", "a buffer copy names the level a texture is stored without ([gfx] texture_detail): skipped"); return; }
        s.level -= drop;
    }
    const FormatInfo* fi = t->fmt;
    const uint32_t bw = fi ? fi->block_w : 1, bh = fi ? fi->block_h : 1, bpb = fi ? fi->bytes : 4;
    uint32_t rw = static_cast<uint32_t>(r.w), rh = static_cast<uint32_t>(r.h), rd = std::max<int32_t>(1, r.d);
    VkBufferImageCopy c = {};
    const uint64_t row_blocks = (rw + bw - 1) / bw;
    const uint64_t row_bytes = row_blocks * bpb;
    const uint64_t stride = p.row_stride > 0 ? static_cast<uint64_t>(p.row_stride) : row_bytes;
    c.bufferRowLength = static_cast<uint32_t>(stride / bpb * bw);
    c.bufferImageHeight = p.image_stride > 0 ? static_cast<uint32_t>(static_cast<uint64_t>(p.image_stride) / stride * bh) : 0;
    c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, s.level, s.layer, 1};
    if (tex_is_depth(t)) c.imageSubresource.aspectMask = tex_aspect(t) & VK_IMAGE_ASPECT_DEPTH_BIT;
    c.imageOffset = {r.x, r.y, r.z};
    c.imageExtent = {rw, rh, rd};
    if (t->b.target != NVN_TEXTURE_TARGET_3D && t->layers > 1 && !p.has_view) { c.imageSubresource.layerCount = tex_layers(t); c.imageExtent.depth = 1; }
    if (t->b.target != NVN_TEXTURE_TARGET_3D && rd > 1) { c.imageSubresource.layerCount = rd; c.imageSubresource.baseArrayLayer = s.layer + static_cast<uint32_t>(std::max(0, r.z)); c.imageOffset.z = 0; c.imageExtent.depth = 1; }
    const uint64_t buf_bytes = stride * ((rh + bh - 1) / bh) * rd;
    if (!to_texture) compute_out_guard(p.address, buf_bytes, kOutBufferWrite);     // written: held results there go back first
    BufRef br = to_texture ? held_or_pool(p.address, buf_bytes, kOutBufferRead, true) : resolve_buffer(p.address, buf_bytes);
    c.bufferOffset = br.offset;
    if (to_texture) {
        tex_ensure_exact(t, *F);
        tex_transition(t, *F, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        tev(t, "copy_buffer_to_texture", p.address, p.has_view ? 1 : 0);
        for (uint32_t i = 0; i < c.imageSubresource.layerCount; ++i) tex_sparse_before_copy(t, c.imageSubresource.baseArrayLayer + i, s.level);
        tex_mark_gpu_written(t);
        vkCmdCopyBufferToImage(F->cb, br.buffer, tex_image(t), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    } else {
        tex_ensure_exact(t, *F);
        tex_transition(t, *F, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        vkCmdCopyImageToBuffer(F->cb, tex_image(t), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, br.buffer, 1, &c);
        gpu_wrote(p.address, stride * ((rh + bh - 1) / bh) * rd, kWriteImageCopy);
    }
    memory_barrier(F->cb);
    counters().copies.fetch_add(1, std::memory_order_relaxed);
    gstamp(to_texture ? "copy buffer->texture" : "copy texture->buffer");
}

void do_copy_texture_texture(const P_CopyTexTex& p) {
    Texture *s = p.src, *d = p.dst;
    if (!s || !d || !tex_gpu(s) || !tex_gpu(d)) return;
    end_pass();
    SubRes ss = sub_of(p.src_view, p.has_src_view), ds = sub_of(p.dst_view, p.has_dst_view);
    const VkExtent3D se = tex_extent(s, ss.level), de = tex_extent(d, ds.level);
    if (ss.level < tex_level_drop(s) || ds.level < tex_level_drop(d)) {   // an image stored without its top level: no texels to copy there
        NVN_LOG_ONCE("copy-dropped-level-tex", "a texture copy names the level a texture is stored without ([gfx] texture_detail): skipped");
        return;
    }
    P_CopyRegion sr = p.has_src_region ? p.src_region : P_CopyRegion{0, 0, 0, static_cast<int32_t>(se.width), static_cast<int32_t>(se.height), static_cast<int32_t>(se.depth)};
    P_CopyRegion dr = p.has_dst_region ? p.dst_region : P_CopyRegion{0, 0, 0, static_cast<int32_t>(de.width), static_cast<int32_t>(de.height), static_cast<int32_t>(de.depth)};
    // scaled images: the NVN regions in image texels (x, y and the size times each image's scale, clipped to the image)
    auto scale_region = [](P_CopyRegion& r, Texture* t, uint32_t level) {
        const float sc = tex_scale(t);
        if (sc == 1.0f) return;
        const VkExtent3D ie = tex_image_extent(t, level);
        const int32_t x1 = std::min<int32_t>(static_cast<int32_t>(std::lround((r.x + r.w) * sc)), static_cast<int32_t>(ie.width));
        const int32_t y1 = std::min<int32_t>(static_cast<int32_t>(std::lround((r.y + r.h) * sc)), static_cast<int32_t>(ie.height));
        r.x = static_cast<int32_t>(std::lround(r.x * sc));
        r.y = static_cast<int32_t>(std::lround(r.y * sc));
        r.w = std::max(0, x1 - r.x);
        r.h = std::max(0, y1 - r.y);
    };
    scale_region(sr, s, ss.level);
    scale_region(dr, d, ds.level);
    ss.level -= tex_level_drop(s);   // image levels from here on
    ds.level -= tex_level_drop(d);
    if (sr.w <= 0 || sr.h <= 0 || dr.w <= 0 || dr.h <= 0) return;
    if (g_trace) {
        char a1[96], a2[96];
        TRACE("copy texture %s level %u layer %u region %d,%d %dx%dx%d -> %s level %u layer %u region %d,%d %dx%dx%d flags %x", tex_desc(s, a1, sizeof a1), ss.level,
              ss.layer, sr.x, sr.y, sr.w, sr.h, sr.d, tex_desc(d, a2, sizeof a2), ds.level, ds.layer, dr.x, dr.y, dr.w, dr.h, dr.d, p.flags);
    }
    tex_ensure_exact(s, *F);
    if (s != d) tex_ensure_exact(d, *F);
    const bool same = s == d;
    tex_transition(s, *F, same ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    if (!same) tex_transition(d, *F, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    tev(d, "copy_texture_to_texture", reinterpret_cast<uint64_t>(s), 0);
    for (int32_t i = 0; i < std::max(1, d->b.target != NVN_TEXTURE_TARGET_3D ? dr.d : 1); ++i) tex_sparse_before_copy(d, ds.layer + static_cast<uint32_t>(i), ds.level);
    tex_mark_gpu_written(d);
    const VkImageLayout sl = same ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const VkImageLayout dl = same ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    const VkImageAspectFlags sa = tex_aspect(s), da = tex_aspect(d);
    const bool same_size = sr.w == dr.w && sr.h == dr.h && sr.d == dr.d;
    const bool same_fmt = tex_vk_format(s) == tex_vk_format(d);
    // MIRROR_X/Y/Z: the 2D engine writes the region reversed along that axis (a blit with the destination corners swapped)
    const uint32_t mirror = p.flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT | NVN_COPY_FLAGS_MIRROR_Z_BIT);
    if (mirror) NVN_LOG_ONCE("copy-mirror", "CopyTextureToTexture with mirror flags: copied as a mirrored blit");
    if (same_size && same_fmt && !mirror) {
        VkImageCopy c = {};
        c.srcSubresource = {sa, ss.level, ss.layer, 1};
        c.dstSubresource = {da, ds.level, ds.layer, 1};
        c.srcOffset = {sr.x, sr.y, sr.z};
        c.dstOffset = {dr.x, dr.y, dr.z};
        c.extent = {static_cast<uint32_t>(sr.w), static_cast<uint32_t>(sr.h), static_cast<uint32_t>(std::max(1, sr.d))};
        if (s->b.target != NVN_TEXTURE_TARGET_3D && sr.d > 1) { c.srcSubresource.layerCount = c.dstSubresource.layerCount = sr.d; c.extent.depth = 1; c.srcOffset.z = c.dstOffset.z = 0; }
        vkCmdCopyImage(F->cb, tex_image(s), sl, tex_image(d), dl, 1, &c);
    } else {
        VkImageBlit b = {};
        b.srcSubresource = {sa, ss.level, ss.layer, 1};
        b.dstSubresource = {da, ds.level, ds.layer, 1};
        b.srcOffsets[0] = {sr.x, sr.y, sr.z};
        b.srcOffsets[1] = {sr.x + sr.w, sr.y + sr.h, sr.z + std::max(1, sr.d)};
        b.dstOffsets[0] = {dr.x, dr.y, dr.z};
        b.dstOffsets[1] = {dr.x + dr.w, dr.y + dr.h, dr.z + std::max(1, dr.d)};
        if (mirror & NVN_COPY_FLAGS_MIRROR_X_BIT) std::swap(b.dstOffsets[0].x, b.dstOffsets[1].x);
        if (mirror & NVN_COPY_FLAGS_MIRROR_Y_BIT) std::swap(b.dstOffsets[0].y, b.dstOffsets[1].y);
        if (mirror & NVN_COPY_FLAGS_MIRROR_Z_BIT) std::swap(b.dstOffsets[0].z, b.dstOffsets[1].z);
        const bool linear = (p.flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) || (tex_scale(s) != tex_scale(d) && !tex_is_depth(s) && !tex_is_depth(d));
        vkCmdBlitImage(F->cb, tex_image(s), sl, tex_image(d), dl, 1, &b, linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }
    memory_barrier(F->cb);
    counters().copies.fetch_add(1, std::memory_order_relaxed);
    if (gpass_on()) {
        char lb[96];
        std::snprintf(lb, sizeof lb, "copy texture %dx%dx%d %s%s", sr.w, sr.h, std::max(1, sr.d), s->fmt ? s->fmt->name : "?", same_size && same_fmt && !mirror ? "" : " blit");
        gstamp(lb);
    }
}

void do_downsample(const P_Downsample& p) {
    Texture *s = p.src, *d = p.dst;
    if (!s || !d || !tex_gpu(s) || !tex_gpu(d)) return;
    end_pass();
    if (tex_is_depth(s)) { note_unsupported("Downsample of a depth texture"); return; }
    tex_transition(s, *F, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    tex_transition(d, *F, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    tex_mark_gpu_written(d);
    VkImageResolve rs = {};
    rs.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    rs.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    if (tex_scale(s) != tex_scale(d)) NVN_LOG_ONCE("downsample-scale", "Downsample between a scaled and an unscaled texture: only the common part is resolved");
    const VkExtent3D e = tex_image_extent(d, 0), es = tex_image_extent(s, 0);
    rs.extent = {std::min(e.width, es.width), std::min(e.height, es.height), 1};
    vkCmdResolveImage(F->cb, tex_image(s), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex_image(d), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rs);
    memory_barrier(F->cb);
    counters().copies.fetch_add(1, std::memory_order_relaxed);
    gstamp("downsample");
}

// [nvn] batch_ubo_updates (default false): the game updates uniform buffers in runs (~15 nvnCommandBufferUpdateUniformBuffer back to
// back, four runs a frame), and each update had a full barrier before and after it. With this on, an update directly after another one
// (nothing recorded in between, ranges apart) needs no barrier before it, and the barrier after it waits for the next command that is not
// an update (wm_close from the interpreter): a run costs two barriers instead of two per update.
bool g_wm_open = false;                                    // updates recorded without their trailing barrier
VkCommandBuffer g_wm_cb = VK_NULL_HANDLE;
uint64_t g_wm_mark = 0;                                    // memory_barrier_count(g_wm_cb) after the last update
std::vector<std::pair<uint64_t, uint64_t>> g_wm_ranges;     // the open updates' [addr, end)

bool wm_batch_on() {
    static const bool on = nxrt::config_get_bool("nvn", "batch_ubo_updates", false);
    return on;
}

void wm_close() {
    if (!g_wm_open) return;
    g_wm_open = false;
    if (F && F->cb == g_wm_cb && memory_barrier_count(F->cb) == g_wm_mark) memory_barrier(F->cb);
    g_wm_ranges.clear();
}

bool wm_keeps_open(Op op) {
    if (op < Op::DrawArrays) return true;
    switch (op) {
        case Op::UpdateUniformBuffer: case Op::FenceSync: case Op::WaitSync: case Op::PushDebugGroup: case Op::PopDebugGroup: case Op::PushConstants:
        case Op::DiscardColor: case Op::DiscardDepthStencil:
            return true;
        default: return false;
    }
}

// `batch`: an UpdateUniformBuffer (see batch_ubo_updates); the other writes (counters, events) are followed by more commands of their op.
void do_write_memory(uint64_t gpu_addr, const void* data, uint32_t size, bool batch = false) {
    // a GPU-ordered write of up to 64 KiB into pool memory (UpdateUniformBuffer, counters, events)
    batch = batch && wm_batch_on();
    const uint64_t before = memory_barrier_count(F->cb);
    end_pass();
    bool lead = memory_barrier_count(F->cb) == before;     // end_pass recorded none
    if (lead && batch && g_wm_open && F->cb == g_wm_cb && g_wm_mark == before) {
        lead = false;                                       // only other updates since the last full barrier: they do not overlap this one?
        for (const auto& r : g_wm_ranges)
            if (gpu_addr < r.second && r.first < gpu_addr + size) { lead = true; break; }
    }
    if (lead) {
        memory_barrier(F->cb);
        g_wm_open = false;
        g_wm_ranges.clear();
    }
    BufRef b = resolve_buffer(gpu_addr, size);
    if (!b.ok) {
        if (g_wm_open) g_wm_mark = memory_barrier_count(F->cb);
        return;
    }
    gpu_wrote(gpu_addr, size, kWriteUpdate);
    vkCmdUpdateBuffer(F->cb, b.buffer, b.offset, size, data);
    if (batch) {
        if (!g_wm_open) g_wm_ranges.clear();
        g_wm_ranges.emplace_back(gpu_addr, gpu_addr + size);
        g_wm_open = true;
        g_wm_cb = F->cb;
        g_wm_mark = memory_barrier_count(F->cb);
    } else {
        memory_barrier(F->cb);
        g_wm_open = false;
        g_wm_ranges.clear();
    }
    gstamp("write memory");
}

// ---- the interpreter ----------------------------------------------------------------------------------------------------------
void apply_bind_program(const P_BindProgram& p) {
    g_out_all = true;     // out_draw_bindings: the active stages may change
    for (int st = 0; st < 6; ++st)
        if (p.stages & (1u << st)) S.stage[st] = p.program ? p.program->stage[st] : nullptr;
    if (g_trace && (p.stages & (1u << kCompute))) {
        g_cs_missing = (p.program && !p.program->stage[kCompute]) ? std::string(p.program->missing) : std::string();
        if (!g_cs_missing.empty()) TRACE("bind program: compute block(s) %s not in the shader cache", g_cs_missing.c_str());
    }
}

// [nvn] same_binds (default on): a bind that sets what is bound already marks nothing dirty. The game binds its program, buffers,
// textures and state objects again before most draws; each such bind made the next draw look up its layout and pipeline and build its
// descriptor sets and vertex buffers again from the same state (notes/rt.md). What the draw's bindings read at replay time (pool entries,
// uniform data) is read at the same time either way; whatever else must rebind (a new pass or submission, GPU writes to copied pages,
// dispatches, DLSS) sets the bits itself.
bool same_binds_on() {
    static const bool on = nxrt::config_get_bool("nvn", "same_binds", true);
    return on;
}
struct RawLast { bool valid = false; uint8_t b[sizeof(P_RawState::bytes)] = {}; };
RawLast g_raw_state[6], g_raw_blend[8];   // the raw state blocks S was last set from (BindColor .. BindMultisample; BindBlend per target)
uint32_t g_raw_attr_n = UINT32_MAX, g_raw_stream_n = UINT32_MAX;
uint32_t g_raw_attr[16] = {};
uint64_t g_raw_stream[16] = {};

bool raw_same(RawLast& l, const uint8_t* bytes) {
    const bool same = l.valid && std::memcmp(l.b, bytes, sizeof l.b) == 0;
    std::memcpy(l.b, bytes, sizeof l.b);
    l.valid = true;
    return same;
}
template <typename T> bool raw_list_same(uint32_t& last_n, T* last, uint32_t count, const uint8_t* blob) {
    const uint32_t n = std::min<uint32_t>(count, 16);
    const bool same = last_n == count && std::memcmp(last, blob, n * sizeof(T)) == 0;
    last_n = count;
    std::memcpy(last, blob, n * sizeof(T));
    return same;
}
void raw_binds_reset() {
    for (RawLast& l : g_raw_state) l.valid = false;
    for (RawLast& l : g_raw_blend) l.valid = false;
    g_raw_attr_n = g_raw_stream_n = UINT32_MAX;
}

// Is the bind op `r` (before it is applied) a no-op on S? Also keeps the raw state blocks up to date: call it for every bind op or never.
bool bind_is_same(const RecordReader& r) {
    switch (r.op()) {
        case Op::BindProgram: {
            const auto& p = r.payload<P_BindProgram>();
            for (int st = 0; st < 6; ++st)
                if ((p.stages & (1u << st)) && S.stage[st] != (p.program ? p.program->stage[st] : nullptr)) return false;
            return true;
        }
        case Op::BindBuffers: {
            const auto& p = r.payload<P_BindBuffers>();
            if (p.kind == kBufXfb) return false;
            const NVNbufferRange* rg = reinterpret_cast<const NVNbufferRange*>(r.blob<P_BindBuffers>());
            for (uint32_t i = 0; i < p.count; ++i) {
                const uint32_t k = p.first + i;
                const GState::Bind* b = nullptr;
                if (p.kind == kBufVertex && k < 16) b = &S.vtx[k];
                else if (p.kind == kBufUniform && p.stage < 6 && k < 16) b = &S.ubo[p.stage][k];
                else if (p.kind == kBufStorage && p.stage < 6 && k < 16) b = &S.ssbo[p.stage][k];
                if (b && (b->addr != rg[i].address || b->size != rg[i].size)) return false;
            }
            return true;
        }
        case Op::BindTextures: case Op::BindImages: {
            const auto& p = r.payload<P_BindTextures>();
            const uint64_t* h = reinterpret_cast<const uint64_t*>(r.blob<P_BindTextures>());
            const bool tex = r.op() == Op::BindTextures;
            for (uint32_t i = 0; i < p.count; ++i) {
                if (p.stage >= 6 || p.first + i >= (tex ? 64u : 16u)) continue;
                if ((tex ? S.tex[p.stage][p.first + i] : S.img[p.stage][p.first + i]) != h[i]) return false;
            }
            return true;
        }
        case Op::SetTexturePool: return S.tpool.get() == static_cast<TexturePool*>(r.payload<P_Pool>().pool);
        case Op::SetSamplerPool: return S.spool.get() == static_cast<SamplerPool*>(r.payload<P_Pool>().pool);
        case Op::BindColor: return raw_same(g_raw_state[0], r.payload<P_RawState>().bytes);
        case Op::BindChannelMask: return raw_same(g_raw_state[1], r.payload<P_RawState>().bytes);
        case Op::BindDepthStencil: return raw_same(g_raw_state[2], r.payload<P_RawState>().bytes);
        case Op::BindPolygon: return raw_same(g_raw_state[3], r.payload<P_RawState>().bytes);
        case Op::BindMultisample: return raw_same(g_raw_state[4], r.payload<P_RawState>().bytes);
        case Op::BindBlend: {
            NVNblendState st;
            std::memcpy(&st, r.payload<P_RawState>().bytes, sizeof st);
            const int t = decode(&st).target;
            if (t < 0 || t >= 8) return false;
            return raw_same(g_raw_blend[t], r.payload<P_RawState>().bytes);
        }
        case Op::BindVertexAttribs:
            return raw_list_same(g_raw_attr_n, g_raw_attr, r.payload<P_BindVertexAttribs>().count, r.blob<P_BindVertexAttribs>());
        case Op::BindVertexStreams:
            return raw_list_same(g_raw_stream_n, g_raw_stream, r.payload<P_BindVertexStreams>().count, r.blob<P_BindVertexStreams>());
        default: return false;
    }
}

bool is_bind_op(Op op) {
    switch (op) {
        case Op::BindProgram: case Op::BindBuffers: case Op::BindTextures: case Op::BindImages: case Op::SetTexturePool: case Op::SetSamplerPool:
        case Op::BindColor: case Op::BindChannelMask: case Op::BindDepthStencil: case Op::BindPolygon: case Op::BindMultisample: case Op::BindBlend:
        case Op::BindVertexAttribs: case Op::BindVertexStreams:
            return true;
        default: return false;
    }
}

void mark_dirty(const RecordReader& r) {
    if ((same_binds_on() || rs_on()) && is_bind_op(r.op())) {
        ++g_rs.binds;
        if (bind_is_same(r)) {
            ++g_rs.binds_same;
            if (same_binds_on()) return;
        }
    }
    switch (r.op()) {
        case Op::BindProgram: g_dirty |= kPipe | kDesc | kVb; break;
        case Op::BindBuffers: g_dirty |= r.payload<P_BindBuffers>().kind == kBufVertex ? kVb : kDescBuf; break;
        case Op::BindTextures: case Op::BindImages: case Op::SetTexturePool: case Op::SetSamplerPool: g_dirty |= kDescTex; break;
        case Op::BindVertexAttribs: case Op::BindVertexStreams: g_dirty |= kPipe | kVb; break;
        case Op::BindBlend: case Op::BindColor: case Op::BindChannelMask: case Op::BindMultisample: case Op::SetDepthClamp: g_dirty |= kPipe; break;
        case Op::BindPolygon: g_dirty |= kPipe | kDynRaster | kDynBias; break;
        case Op::BindDepthStencil: case Op::SetDepthBounds: g_dirty |= kDynDepth | kDynStencil; break;
        case Op::SetViewports: case Op::SetScissors: case Op::SetDepthRanges: g_dirty |= kDynVp; break;
        case Op::SetStencil: g_dirty |= kDynStencil; break;
        case Op::SetBlendColor: g_dirty |= kDynBlend; break;
        case Op::SetLineWidth: case Op::SetRasterizerDiscard: case Op::SetPrimitiveRestart: g_dirty |= kDynRaster; break;
        case Op::SetPatchSize: g_dirty |= kPipe; break;   // patch control points are part of the pipeline
        case Op::SetPolygonOffsetClamp: g_dirty |= kDynBias; break;
        case Op::SetRenderTargets: g_dirty = kAllDirty; break;
        default: break;
    }
}

// ---- windowed GPU counters (see g_cw) ----------------------------------------------------------------------------------------
bool cw_ready(CounterWindow& w, bool xfb) {
    if (w.pool) return true;
    if (w.tried) return false;
    w.tried = true;
    const gfx::DeviceFeatures& f = device().features();
    if (xfb ? !(f.transform_feedback && f.transform_feedback_queries) : !f.pipeline_statistics_query) return false;
    VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = xfb ? VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT : VK_QUERY_TYPE_PIPELINE_STATISTICS;
    qi.queryCount = kCounterSlots;
    if (!xfb) qi.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT;
    if (vkCreateQueryPool(vkdev(), &qi, nullptr, &w.pool) != VK_SUCCESS) w.pool = VK_NULL_HANDLE;
    return w.pool != VK_NULL_HANDLE;
}

void cw_end_query(CounterWindow& w, bool xfb) {
    if (w.active < 0) return;
    if (xfb) vkCmdEndQueryIndexedEXT(F->cb, w.pool, static_cast<uint32_t>(w.active), 0);
    else vkCmdEndQuery(F->cb, w.pool, static_cast<uint32_t>(w.active));
}

// Does the window that a ResetCounter of `type` opens (up to the next report or reset of that type in the same list) hold a draw whose
// primitives the CPU cannot count: an indirect draw or a DrawTransformFeedback? The record reader is copied; the list itself is not touched.
bool counter_window_has_gpu_draw(RecordReader look, uint32_t type) {
    while (look.next()) {
        const Op op = look.op();
        if ((op == Op::ReportCounter || op == Op::ResetCounter) && look.payload<P_Counter>().type == type) return false;
        if (op == Op::DrawIndirect || op == Op::DrawXfb) return true;
    }
    return false;
}

void run_list(const CmdList& list, int depth) {
    if (depth > kMaxDepth) { NVN_LOG_ONCE("depth", "CallCommands nesting too deep"); return; }
    counters().lists_replayed.fetch_add(1, std::memory_order_relaxed);
    for (RecordReader r(list); r.next();) {
        mark_dirty(r);
        if (g_cs_open && !cs_keeps_open(r.op())) cs_close();
        if (g_wm_open && !wm_keeps_open(r.op())) wm_close();
        {
            RsScope rs(kRsOut);
            if (out_lazy()) out_before_op(r);                                               // compute shadows: results back to the pool first
            else if (out_any() && !compute_out_may_wait(r.op())) flush_compute_out(kOutOther);
        }
        switch (r.op()) {
            case Op::Nop: break;
            case Op::BindProgram: apply_bind_program(r.payload<P_BindProgram>()); break;
            case Op::BindBuffers: {
                const auto& p = r.payload<P_BindBuffers>();
                const NVNbufferRange* rg = reinterpret_cast<const NVNbufferRange*>(r.blob<P_BindBuffers>());
                for (uint32_t i = 0; i < p.count; ++i) {
                    const uint32_t k = p.first + i;
                    const uint32_t bit = 1u << (k & 31), on = rg[i].address ? bit : 0u;
                    if (p.kind == kBufVertex && k < 16) {
                        S.vtx[k] = {rg[i].address, rg[i].size};
                        S.vtx_mask = (S.vtx_mask & ~bit) | on;
                        out_slot_changed(static_cast<int>(k));
                    } else if (p.kind == kBufUniform && p.stage < 6 && k < 16) {
                        S.ubo[p.stage][k] = {rg[i].address, rg[i].size};
                        S.ubo_mask[p.stage] = (S.ubo_mask[p.stage] & ~bit) | on;
                        if (p.stage < kCompute) out_slot_changed(16 + static_cast<int>(p.stage) * 32 + static_cast<int>(k));
                    } else if (p.kind == kBufStorage && p.stage < 6 && k < 16) {
                        S.ssbo[p.stage][k] = {rg[i].address, rg[i].size};
                        S.ssbo_mask[p.stage] = (S.ssbo_mask[p.stage] & ~bit) | on;
                        if (p.stage < kCompute) out_slot_changed(16 + static_cast<int>(p.stage) * 32 + 16 + static_cast<int>(k));
                    } else if (p.kind == kBufXfb && k < 4) {
                        S.xfb[k] = {rg[i].address, rg[i].size};
                        S.xfb_mask = (S.xfb_mask & ~bit) | on;
                        out_slot_changed(16 + kCompute * 32 + static_cast<int>(k));
                    }
                }
                break;
            }
            case Op::BindTextures: {
                const auto& p = r.payload<P_BindTextures>();
                const uint64_t* h = reinterpret_cast<const uint64_t*>(r.blob<P_BindTextures>());
                for (uint32_t i = 0; i < p.count; ++i)
                    if (p.stage < 6 && p.first + i < 64) S.tex[p.stage][p.first + i] = h[i];
                break;
            }
            case Op::BindImages: {
                const auto& p = r.payload<P_BindTextures>();
                const uint64_t* h = reinterpret_cast<const uint64_t*>(r.blob<P_BindTextures>());
                for (uint32_t i = 0; i < p.count; ++i)
                    if (p.stage < 6 && p.first + i < 16) S.img[p.stage][p.first + i] = h[i];
                break;
            }
            case Op::BindVertexAttribs: {
                const auto& p = r.payload<P_BindVertexAttribs>();
                const NVNvertexAttribState* a = reinterpret_cast<const NVNvertexAttribState*>(r.blob<P_BindVertexAttribs>());
                S.attr_count = std::min<uint32_t>(p.count, 16);
                for (uint32_t i = 0; i < S.attr_count; ++i) S.attr[i] = decode(&a[i]);
                break;
            }
            case Op::BindVertexStreams: {
                const auto& p = r.payload<P_BindVertexStreams>();
                const NVNvertexStreamState* a = reinterpret_cast<const NVNvertexStreamState*>(r.blob<P_BindVertexStreams>());
                S.stream_count = std::min<uint32_t>(p.count, 16);
                for (uint32_t i = 0; i < S.stream_count; ++i) S.stream[i] = decode(&a[i]);
                break;
            }
            case Op::BindBlend: {
                NVNblendState st; std::memcpy(&st, r.payload<P_RawState>().bytes, sizeof st);
                BlendInfo b = decode(&st);
                if (b.target >= 0 && b.target < 8) S.blend[b.target] = b;
                break;
            }
            case Op::BindColor: { NVNcolorState st; std::memcpy(&st, r.payload<P_RawState>().bytes, sizeof st); S.color = decode(&st); break; }
            case Op::BindChannelMask: { NVNchannelMaskState st; std::memcpy(&st, r.payload<P_RawState>().bytes, sizeof st); S.chmask = decode(&st); break; }
            case Op::BindDepthStencil: { NVNdepthStencilState st; std::memcpy(&st, r.payload<P_RawState>().bytes, sizeof st); S.ds = decode(&st); break; }
            case Op::BindPolygon: { NVNpolygonState st; std::memcpy(&st, r.payload<P_RawState>().bytes, sizeof st); S.poly = decode(&st); break; }
            case Op::BindMultisample: { NVNmultisampleState st; std::memcpy(&st, r.payload<P_RawState>().bytes, sizeof st); S.ms = decode(&st); break; }
            case Op::SetTexturePool: S.tpool = Ref<TexturePool>::share(static_cast<TexturePool*>(r.payload<P_Pool>().pool)); break;
            case Op::SetSamplerPool: S.spool = Ref<SamplerPool>::share(static_cast<SamplerPool*>(r.payload<P_Pool>().pool)); break;
            case Op::SetViewports: {
                const auto& p = r.payload<P_Rects>();
                const float* v = reinterpret_cast<const float*>(r.blob<P_Rects>());
                for (uint32_t i = 0; i < p.count && p.first + i < 16; ++i) std::memcpy(S.vp[p.first + i], v + 4 * i, 16);
                S.nvp = std::max(S.nvp, std::min<uint32_t>(p.first + p.count, 16));
                break;
            }
            case Op::SetScissors: {
                const auto& p = r.payload<P_Rects>();
                const int32_t* v = reinterpret_cast<const int32_t*>(r.blob<P_Rects>());
                for (uint32_t i = 0; i < p.count && p.first + i < 16; ++i) std::memcpy(S.sc[p.first + i], v + 4 * i, 16);
                S.nsc = std::max(S.nsc, std::min<uint32_t>(p.first + p.count, 16));
                break;
            }
            case Op::SetDepthRanges: {
                const auto& p = r.payload<P_DepthRanges>();
                const float* v = reinterpret_cast<const float*>(r.blob<P_DepthRanges>());
                for (uint32_t i = 0; i < p.count && p.first + i < 16; ++i) std::memcpy(S.dr[p.first + i], v + 2 * i, 8);
                break;
            }
            case Op::SetDepthBounds: { const auto& p = r.payload<P_DepthBounds>(); S.depth_bounds = p.enable; S.db_near = p.near_; S.db_far = p.far_; break; }
            case Op::SetStencil: {
                const auto& p = r.payload<P_Stencil>();
                for (int f = 0; f < 2; ++f) {
                    const bool hit = p.face == NVN_FACE_FRONT_AND_BACK || (f == 0 && p.face == NVN_FACE_FRONT) || (f == 1 && p.face == NVN_FACE_BACK);
                    if (!hit) continue;
                    if (p.what == 0) S.stencil_ref[f] = p.value;
                    else if (p.what == 1) S.stencil_vmask[f] = p.value;
                    else S.stencil_wmask[f] = p.value;
                }
                break;
            }
            case Op::SetBlendColor: std::memcpy(S.blend_color, r.payload<P_BlendColor>().c, 16); break;
            case Op::SetPointSize: S.point_size = r.payload<P_Float>().v; break;
            case Op::SetLineWidth: S.line_width = r.payload<P_Float>().v; break;
            case Op::SetAlphaRef: S.alpha_ref = r.payload<P_Float>().v; break;
            case Op::SetPolygonOffsetClamp: { const auto& p = r.payload<P_PolygonOffsetClamp>(); S.bias_factor = p.factor; S.bias_units = p.units; S.bias_clamp = p.clamp; break; }
            case Op::SetSampleMask: S.sample_mask = r.payload<P_U32>().v; break;
            case Op::SetRasterizerDiscard: S.rasterizer_discard = r.payload<P_U32>().v != 0; break;
            case Op::SetDepthClamp: S.depth_clamp = r.payload<P_U32>().v != 0; break;
            case Op::SetPrimitiveRestart: { const auto& p = r.payload<P_PrimitiveRestart>(); S.prim_restart = p.enable != 0; S.restart_index = p.index; break; }
            case Op::SetPatchSize: S.patch_size = r.payload<P_U32>().v; g_dirty |= kPipe; break;   // the patch size is in the pipeline key
            case Op::SetTessLevels: {
                const auto& p = r.payload<P_TessLevels>();
                if (p.outer) std::memcpy(S.tess_outer, p.v, sizeof S.tess_outer);
                else std::memcpy(S.tess_inner, p.v, sizeof S.tess_inner);
                break;
            }
            case Op::SetRenderEnable:
                if (r.size() >= sizeof(RecHeader) + sizeof(P_RenderEnableCond)) {
                    const auto& p = r.payload<P_RenderEnableCond>();
                    S.render_enable = conditional_render_enabled(p.mode, p.address);
                } else {
                    S.render_enable = r.payload<P_U32>().v != 0;
                }
                break;
            case Op::SetCoverageModulationTable: case Op::SetConservativeRaster: break;
            case Op::SetRenderTargets: {
                const auto& p = r.payload<P_RenderTargets>();
                S.ncolor = p.count;
                for (uint32_t i = 0; i < 8; ++i) {
                    S.color_tex[i] = i < p.count ? Ref<Texture>::share(p.color[i].tex) : Ref<Texture>();
                    if (i < p.count) { S.color_view[i] = p.color[i].view; S.color_has_view[i] = p.color[i].has_view != 0; }
                    else S.color_has_view[i] = false;
                    const TextureViewInfo& v = S.color_view[i];
                    const bool hv = S.color_has_view[i];
                    S.color_level[i] = hv && v.has_levels ? static_cast<uint32_t>(v.min_level) : 0;
                    S.color_layer[i] = hv && v.has_layers ? static_cast<uint32_t>(v.min_layer) : 0;
                    S.color_layers[i] = hv && v.has_layers ? static_cast<uint32_t>(v.num_layers) : (S.color_tex[i] ? static_cast<uint32_t>(S.color_tex[i]->layers) : 0u);
                }
                S.depth_tex = Ref<Texture>::share(p.depth.tex);
                S.depth_view = p.depth.view;
                S.depth_has_view = p.depth.has_view != 0;
                break;
            }
            case Op::DrawArrays: do_draw_arrays(r.payload<P_DrawArrays>()); break;
            case Op::DrawElements: do_draw_elements(r.payload<P_DrawElements>()); break;
            case Op::DrawIndirect: do_draw_indirect(r.payload<P_DrawIndirect>()); break;
            case Op::DrawXfb: do_draw_xfb(r.payload<P_DrawXfb>()); break;
            case Op::PushDebugGroup:
                if (device().features().debug_labels && vkCmdBeginDebugUtilsLabelEXT) {
                    VkDebugUtilsLabelEXT l = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
                    l.pLabelName = reinterpret_cast<const char*>(r.blob<P_DebugGroup>());
                    vkCmdBeginDebugUtilsLabelEXT(F->cb, &l);
                    ++g_dbg_depth;
                }
                break;
            case Op::PopDebugGroup:
                if (g_dbg_depth > 0 && vkCmdEndDebugUtilsLabelEXT) { vkCmdEndDebugUtilsLabelEXT(F->cb); --g_dbg_depth; }   // a pop whose push was in an earlier submission has nothing to close
                break;
            case Op::DrawTexture: note_unsupported("DrawTexture"); break;
            case Op::ClearColor: do_clear_color(r.payload<P_ClearColor>()); break;
            case Op::ClearDepthStencil: do_clear_depth_stencil(r.payload<P_ClearDepthStencil>()); break;
            case Op::ClearTexture: do_clear_texture(r.payload<P_ClearTexture>()); break;
            case Op::ClearBuffer: {
                const auto& p = r.payload<P_ClearBuffer>();
                end_pass();
                BufRef b = resolve_buffer(p.address, p.size);
                if (b.ok) vkCmdFillBuffer(F->cb, b.buffer, b.offset, p.size & ~3ull, p.value);
                if (b.ok) gpu_wrote(p.address, p.size, kWriteFill);
                memory_barrier(F->cb);
                counters().clears.fetch_add(1);
                gstamp("fill buffer");
                break;
            }
            case Op::CopyBufferToBuffer: {
                const auto& p = r.payload<P_CopyBufferToBuffer>();
                end_pass();
                BufRef s = held_or_pool(p.src, p.size, kOutBufferRead, true), d = resolve_buffer(p.dst, p.size);
                if (s.ok && d.ok) {
                    VkBufferCopy c = {s.offset, d.offset, p.size};
                    vkCmdCopyBuffer(F->cb, s.buffer, d.buffer, 1, &c);
                    gpu_wrote(p.dst, p.size, kWriteCopy);
                }
                memory_barrier(F->cb);
                counters().copies.fetch_add(1);
                gstamp("copy buffer");
                break;
            }
            case Op::CopyBufferToTexture: do_copy_buffer_texture(r.payload<P_CopyBufTex>(), true); break;
            case Op::CopyTextureToBuffer: do_copy_buffer_texture(r.payload<P_CopyBufTex>(), false); break;
            case Op::CopyTextureToTexture: do_copy_texture_texture(r.payload<P_CopyTexTex>()); break;
            case Op::Downsample: do_downsample(r.payload<P_Downsample>()); break;
            case Op::UpdateUniformBuffer: {
                const auto& p = r.payload<P_UpdateUniformBuffer>();
                const uint8_t* data = r.blob<P_UpdateUniformBuffer>();
                const uint32_t n = static_cast<uint32_t>(p.size & ~3ull);
                if (n) do_write_memory(p.address + static_cast<uint64_t>(p.offset), data, n, true);
                break;
            }
            case Op::PushConstants: {
                const auto& p = r.payload<P_PushConstants>();
                const uint8_t* src = p.address ? cpu_view(p.address, p.size) : r.blob<P_PushConstants>();
                if (!src) {
                    NVN_LOG_ONCE("push-deferred", "a Deferred push constant reads memory the GPU has not finished writing: skipped");
                    break;
                }
                const uint32_t off = std::min<uint32_t>(p.offset, 256), n = std::min<uint32_t>(p.size, 256 - off);
                if (g_trace) {
                    std::string words;
                    for (uint32_t i = 0; i + 4 <= n; i += 4) { uint32_t v; std::memcpy(&v, src + i, 4); char t[12]; std::snprintf(t, sizeof t, " %08x", v); words += t; }
                    TRACE("push constants %s slot %u offset %u size %u:%s", p.compute ? "compute" : "graphics", p.slot, p.offset, p.size, words.c_str());
                }
                if (p.compute) {
                    std::memcpy(S.push_cmp + 256 * (p.slot & 7) + off, src, n);
                } else {
                    std::memcpy(S.push_gfx[p.slot & 7] + off, src, n);
                    g_push_buf[p.slot & 7] = UploadAlloc();
                }
                g_dirty |= kDescBuf;   // a buffer binding of set 0 (build_set: NvnSource::PushConstants)
                break;
            }
            case Op::Dispatch: { RsScope rs(kRsDispatch); ++g_rs.dispatches; const auto& p = r.payload<P_Dispatch>(); do_dispatch(p.x, p.y, p.z, 0); break; }
            case Op::DispatchIndirect: { RsScope rs(kRsDispatch); ++g_rs.dispatches; do_dispatch(0, 0, 0, r.payload<P_DispatchIndirect>().address); break; }
            case Op::ReportCounter: {
                const auto& p = r.payload<P_Counter>();
                struct { uint64_t counter, timestamp; } rep = {1, timestamp_ns()};   // counters other than the ones handled below are reported as non-zero
                if (p.type == NVN_COUNTER_TYPE_SAMPLES_PASSED && occ_ready()) {
                    end_pass();
                    if (g_occ_active >= 0) {
                        vkCmdEndQuery(F->cb, g_occ_pool, static_cast<uint32_t>(g_occ_active));
                        g_occ_last = g_occ_active;
                        g_occ_active = -1;
                    }
                    if (p.address) {
                        rep.counter = 0;
                        g_reports[p.address] = kReportUnknown;   // the query result lands on the GPU
                        do_write_memory(p.address, &rep, sizeof rep);
                        BufRef b = resolve_buffer(p.address, sizeof rep);
                        if (b.ok && g_occ_last >= 0) {
                            vkCmdCopyQueryPoolResults(F->cb, g_occ_pool, static_cast<uint32_t>(g_occ_last), 1, b.buffer, b.offset, 16,
                                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                            memory_barrier(F->cb);
                        }
                    }
                    break;
                }
                if (p.type == NVN_COUNTER_TYPE_PRIMITIVES_GENERATED || p.type == NVN_COUNTER_TYPE_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN) {
                    CounterWindow& cw = g_cw[p.type == NVN_COUNTER_TYPE_PRIMITIVES_GENERATED ? 0 : 1];
                    if (cw.active >= 0) {   // the GPU counted this window (it holds indirect or feedback draws): the report is the query's result
                        end_pass();
                        cw_end_query(cw, &cw == &g_cw[1]);
                        cw.last = cw.active;
                        cw.active = -1;
                        if (p.address) {
                            rep.counter = 0;
                            g_reports[p.address] = kReportUnknown;   // the result lands on the GPU
                            do_write_memory(p.address, &rep, sizeof rep);
                            BufRef b = resolve_buffer(p.address, sizeof rep);
                            if (b.ok) {
                                // the feedback query has two results (written, needed): the second would land on the timestamp word, which is written again after it
                                vkCmdCopyQueryPoolResults(F->cb, cw.pool, static_cast<uint32_t>(cw.last), 1, b.buffer, b.offset, 16, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                                memory_barrier(F->cb);
                                if (&cw == &g_cw[1]) do_write_memory(p.address + 8, &rep.timestamp, 8);
                            }
                        }
                        break;
                    }
                    rep.counter = p.type == NVN_COUNTER_TYPE_PRIMITIVES_GENERATED ? g_prims_generated : g_prims_xfb;
                    note_report(p.address, rep.counter);
                    if (p.address) do_write_memory(p.address, &rep, sizeof rep);
                    break;
                }
                // TIMESTAMP_TOP is a timestamp too (taken when the command reaches the top of the pipe): its counter word is 0 like TIMESTAMP's
                const bool stamp = p.type == NVN_COUNTER_TYPE_TIMESTAMP || p.type == NVN_COUNTER_TYPE_TIMESTAMP_TOP;
                if (!stamp) note_unsupported("statistics counters other than samples passed, primitives generated and feedback primitives written are reported as non-zero");
                if (stamp) rep.counter = 0;
                note_report(p.address, rep.counter);
                if (p.address) do_write_memory(p.address, &rep, sizeof rep);
                break;
            }
            case Op::ReportValue: {
                const auto& p = r.payload<P_ReportValue>();
                struct { uint64_t counter, timestamp; } rep = {p.value, timestamp_ns()};
                note_report(p.address, rep.counter);
                if (p.address) do_write_memory(p.address, &rep, sizeof rep);
                break;
            }
            case Op::ResetCounter: {
                const auto& p = r.payload<P_Counter>();
                if (p.type == NVN_COUNTER_TYPE_SAMPLES_PASSED && occ_ready()) {
                    end_pass();
                    if (g_occ_active >= 0) vkCmdEndQuery(F->cb, g_occ_pool, static_cast<uint32_t>(g_occ_active));
                    const uint32_t slot = g_occ_next++ % kOccSlots;
                    vkCmdResetQueryPool(F->cb, g_occ_pool, slot, 1);
                    vkCmdBeginQuery(F->cb, g_occ_pool, slot, device().features().occlusion_query_precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
                    g_occ_active = static_cast<int32_t>(slot);
                    g_occ_last = -1;
                } else if (p.type == NVN_COUNTER_TYPE_PRIMITIVES_GENERATED || p.type == NVN_COUNTER_TYPE_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN) {
                    const bool gen = p.type == NVN_COUNTER_TYPE_PRIMITIVES_GENERATED;
                    (gen ? g_prims_generated : g_prims_xfb) = 0;
                    CounterWindow& cw = g_cw[gen ? 0 : 1];
                    if (cw.active >= 0) { end_pass(); cw_end_query(cw, !gen); cw.active = -1; }   // an unreported window is over
                    if (counter_window_has_gpu_draw(r, p.type) && cw_ready(cw, !gen)) {
                        static bool noted = false;
                        if (!noted) { noted = true; NXLOG_I("nvn.replay", "replay: primitive counter %u counted by a GPU query (first): its window holds an indirect or feedback draw", p.type); }
                        end_pass();
                        const uint32_t slot = cw.next++ % kCounterSlots;
                        vkCmdResetQueryPool(F->cb, cw.pool, slot, 1);
                        if (gen) vkCmdBeginQuery(F->cb, cw.pool, slot, 0);
                        else vkCmdBeginQueryIndexedEXT(F->cb, cw.pool, slot, 0, 0);
                        cw.active = static_cast<int32_t>(slot);
                    }
                }
                break;
            }
            case Op::Barrier: {
                counters().game_barriers.fetch_add(1, std::memory_order_relaxed);
                // [nvn] game_barriers = tracked (default full): with tracked pass barriers, everything recorded since the last full barrier
                // that has no barrier of its own is attachment writes of passes (a pass whose draws wrote other memory ended with a full
                // barrier, copies and fills carry one, open dispatches are excluded here). Every later use of those attachments gets its
                // own dependency (a layout change, or pending_conflict for the next pass on the same image), so the game's barrier adds
                // nothing: it neither ends the pass nor drains the GPU.
                // Open dispatches (dispatch_barriers = tracked) stay open: every later command either checks its dependency on them or
                // closes them first.
                if (game_barriers_tracked() && g_tracked && !g_pass_unsafe && !g_xfb_active && (!g_cs_open || cs_tracked_on())) {
                    counters().game_barriers_left_out.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                end_pass();
                memory_barrier(F->cb);
                break;
            }
            case Op::FenceSync: {
                const auto& p = r.payload<P_Sync>();
                if (p.sync) { p.sync->serial.store(F->serial); p.sync->armed.store(true); }
                break;
            }
            case Op::WaitSync: break;
            case Op::SignalEvent: {
                const auto& p = r.payload<P_Event>();
                if (p.event && p.event->pool && p.event->pool->base) {
                    Event* ev = p.event;
                    uint32_t v = p.value;
                    if (p.mode != NVN_EVENT_SIGNAL_MODE_WRITE) {
                        // ADD: the GPU cannot add into memory; the sum is computed here from the value after the signals recorded so far
                        uint32_t base = 0;
                        if (ev->shadow_valid.load()) base = ev->shadow.load();
                        else std::memcpy(&base, ev->pool->base + ev->offset, 4);
                        v = base + p.value;
                    }
                    ev->shadow.store(v);
                    ev->shadow_valid.store(true);
                    do_write_memory(ev->pool->gpu_base + static_cast<uint64_t>(ev->offset), &v, 4);
                }
                break;
            }
            case Op::WaitEvent: note_unsupported("GPU-side WaitEvent"); break;
            case Op::CallCommands: {
                const auto& p = r.payload<P_CallCommands>();
                const uint64_t* h = reinterpret_cast<const uint64_t*>(r.blob<P_CallCommands>());
                for (uint32_t i = 0; i < p.count; ++i) {
                    Ref<CmdList> l = handle_lookup(h[i]);
                    if (l) run_list(*l, depth + 1);
                    else NVN_LOG_ONCE("missing-handle", "CallCommands of an unknown handle skipped");
                }
                break;
            }
            case Op::DiscardColor: case Op::DiscardDepthStencil: break;
            case Op::ResolveDepth: note_unsupported("ResolveDepthBuffer"); break;
            case Op::EvaluateDlss:                                                    // the host upscaler (api_dlss_replay.cpp) binds its own compute state
                end_pass();
                TRACE("dlss evaluation (the scene is upscaled here; frame generation's inputs)");
                {
                    cmdq::Direct direct;   // NGX / Streamline record through their own entry points
                    dlss_replay(*F, r.payload<P_EvaluateDlss>().eval);
                }
                g_dirty = kAllDirty;
                gstamp("dlss");
                break;
            case Op::BeginXfb: case Op::EndXfb: case Op::PauseXfb: case Op::ResumeXfb: {
                const auto& p = r.payload<P_Counter>();
                static std::unordered_map<std::string, bool> seen;
                const std::string key = std::string(op_name(r.op())) + (S.stage[kVertex] ? S.stage[kVertex]->hash : "-");
                if (seen.emplace(key, true).second)
                    NXLOG_I("nvn.xfb", "%s control %llx vs %s xfb buffer0 %llx/%llu", op_name(r.op()), static_cast<unsigned long long>(p.address),
                            S.stage[kVertex] ? S.stage[kVertex]->hash.c_str() : "-", static_cast<unsigned long long>(S.xfb[0].addr),
                            static_cast<unsigned long long>(S.xfb[0].size));
                TRACE("%s control %llx", op_name(r.op()), static_cast<unsigned long long>(p.address));
                if (!device().features().transform_feedback) {
                    note_unsupported("transform feedback (the device has no VK_EXT_transform_feedback)");
                    break;
                }
                if (r.op() == Op::BeginXfb) {
                    xfb_end_in_pass();    // a section still open in the pass ends where the new one starts
                    auto sit = g_xfb_slot_of.find(p.address);
                    if (sit == g_xfb_slot_of.end()) {
                        if (g_xfb_slot_next >= kXfbSlots) {   // all counters in use: start over (a DrawTransformFeedback of an older control buffer then draws nothing)
                            g_xfb_slot_of.clear();
                            g_xfb_slot_next = 0;
                        }
                        sit = g_xfb_slot_of.emplace(p.address, g_xfb_slot_next++).first;
                    }
                    g_xfb_cur = sit->second;
                    g_xfb_slot_written[g_xfb_cur] = false;
                    g_xfb_active = true;
                    g_xfb_have_counter = false;
                } else if (r.op() == Op::EndXfb) {
                    xfb_end_in_pass();
                    g_xfb_active = false;
                    g_xfb_have_counter = false;
                }
                break;
            }
            default: note_unsupported(op_name(r.op())); break;
        }
    }
}

}  // namespace

// [display] unpaced_loading (default on): the limiter stops waiting while the game loads, since its loading code advances once per
// presented frame. Loading = presented frames with fewer than 16 draws (boot's black screen draws 3: exposure, bloom, tone map; the
// loading screen a few more) AND the game still stalling: a present gap over 100 ms in the last 1.5 s. A light screen that presents
// steadily (the title screen, a menu) is paced again within 1.5 s, so it does not run at 1000 fps; so is any frame with 16+ draws.
void note_frame_for_pacing() {
    static const bool on = nxrt::config_get_bool("display", "unpaced_loading", true);
    if (!on) return;
    using clk = std::chrono::steady_clock;
    static uint64_t last_p = 0, last_d = 0;
    static int low = 0;
    static clk::time_point last_present{}, last_stall{};
    const uint64_t p = counters().presents.load(std::memory_order_relaxed);
    if (p == last_p) return;
    const auto now = clk::now();
    const uint64_t d = counters().draws.load(std::memory_order_relaxed) + counters().draws_skipped.load(std::memory_order_relaxed);
    const uint64_t per = (d - last_d) / (p - last_p);
    last_p = p;
    last_d = d;
    if (last_present == clk::time_point{} || now - last_present > std::chrono::milliseconds(100)) last_stall = now;
    last_present = now;
    low = per < 16 ? low + 1 : 0;
    const bool want = low >= 3 && now - last_stall < std::chrono::milliseconds(1500);
    if (want != nxrt::frame_limiter_unpaced()) {
        nxrt::frame_limiter_set_unpaced(want);
        if (!want) g_pacing_on_present.store(p, std::memory_order_relaxed);   // a load just ended (prepare_textures' load-in burst)
        ft::mark(want ? "pace.loading" : "pace.playing", static_cast<uint32_t>(per));
        NXLOG_I(kTag, "pacing %s (present %llu, %llu draws)", want ? "off: loading" : "on", static_cast<unsigned long long>(p),
                static_cast<unsigned long long>(per));
    }
    static bool full = false;
    if (!full && per >= 16) {
        full = true;
        NXLOG_I(kTag, "first frame with 16+ draws (present %llu, %llu draws)", static_cast<unsigned long long>(p), static_cast<unsigned long long>(per));
    }
}

void replay_begin_submit(Frame& f) {
    gpl_note_presents(counters().presents.load(std::memory_order_relaxed));   // [nvn] gpl_async defers draws only while frames are flowing
    note_frame_for_pacing();
    rs_report();
    F = &f;
    tex_poll_cpu_writes();
    if (extras::LowLatency* ll = extras::lowlat()) ll->submit_begin();     // RENDERSUBMIT_START of the frame (S14)
    if (g_trace_frame == -2) {
        // [nvn] trace_frame = "6050" or "6050,9300": every listed presented frame is traced (the first one is the main one)
        const std::string list = nxrt::config_get_str("nvn", "trace_frame", "");
        g_trace_frame = -1;
        size_t i = 0;
        while (i < list.size()) {
            size_t j = list.find(',', i);
            if (j == std::string::npos) j = list.size();
            const long long v = std::atoll(list.substr(i, j - i).c_str());
            if (g_trace_frame < 0) g_trace_frame = v; else g_trace_extra.push_back(v);
            i = j + 1;
        }
    }
    if (!g_trigger_read) {
        g_trigger_read = true;
        g_trigger_fs = nxrt::config_get_str("nvn", "trace_after_fs", "");
    }
    if (!g_trigger_fs.empty() && g_trigger_seen && !g_trigger_done && static_cast<int64_t>(counters().presents.load()) > g_trigger_present) {
        g_trigger_done = true;
        g_trace_frame = static_cast<int64_t>(counters().presents.load());
    }
    {   // [nvn] trace_sec = N (NXRT_NVN_TRACE_SEC): trace the first frame presented N seconds after the first draw (frame numbers differ between runs)
        static const double trace_sec = static_cast<double>(nxrt::config_get_int("nvn", "trace_sec", 0));
        static const auto t0 = std::chrono::steady_clock::now();
        static bool trace_sec_done = false;
        if (trace_sec > 0 && !trace_sec_done && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() >= trace_sec) {
            trace_sec_done = true;
            g_trace_frame = static_cast<int64_t>(counters().presents.load());
        }
    }
    if (g_trace_req.exchange(false)) g_trace_frame = static_cast<int64_t>(counters().presents.load()) + 1;   // armed from a script (@shot nvntrace_*)
    g_trace = g_trace_frame >= 0 && static_cast<int64_t>(counters().presents.load()) == g_trace_frame;
    for (int64_t e : g_trace_extra) g_trace |= static_cast<int64_t>(counters().presents.load()) == e;
    if (g_trace) { g_trace_n = 0; NXLOG_I("nvn.trace", "---- frame %lld ----", static_cast<long long>(counters().presents.load())); }
    {   // S38: device-local pool copies and tracked barriers. [nvn] gpuperf_ab = N: both are off in even windows of N presents and on in odd
        // ones (the paired-window measurement of S29); without it they follow their own settings.
        static const int64_t ab = config_get_int("nvn", "gpuperf_ab", 0);
        static const bool tracked = config_get_str("nvn", "pass_barriers", "tracked") != "full";
        static int last_on = -1;
        bool on = true;
        if (ab > 0) on = ((counters().presents.load() / static_cast<uint64_t>(ab)) & 1) != 0;
        if (static_cast<int>(on) != last_on) {
            last_on = on ? 1 : 0;
            if (ab > 0) ft::mark(on ? "gpuperf.on" : "gpuperf.off", static_cast<uint32_t>(counters().presents.load()));
        }
        g_tracked = on && tracked;
        set_narrow_barriers(g_tracked);
        g_pending.clear();
        g_pass_unsafe = false;
        g_cs_open = false;
        g_cs_open_set.clear();
        g_wm_open = false;
        g_wm_ranges.clear();
        mirror_begin_submit(f, on);
    }
    S.window_origin = window_origin_mode();
    S.depth_mode = depth_mode();
    g_draw_idx = 0;
    P = Pass();
    g_set_cache.clear();
    g_bound_pipeline = VK_NULL_HANDLE;
    g_dirty = kAllDirty;
    g_pl = nullptr;
    g_pipe = VK_NULL_HANDLE;
    g_topo = g_pipe_topo = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
    g_restart_valid = false;
    g_vb_bound_in_cb = false;
    g_want_depth_ro = false;
    g_fb_use.clear();
    g_fb_seen = false;
    g_key_dummy_binding = 0;
    g_split_t0 = qpc_now();
    g_split_draws = 0;
    for (UploadAlloc& u : g_push_buf) u = UploadAlloc();   // the upload memory of the last submission
    g_cs_c0 = UploadAlloc();
    // one support buffer per submission (Ryujinx SupportBuffer: render scales 1.0, no BGRA remap)
    g_support = f.upload(sizeof(SupportBuffer), 256);
    g_support_scaled.clear();
    g_draw_support = UploadAlloc();
    if (g_support) {
        SupportBuffer sb = support_buffer_defaults();
        std::memcpy(g_support.ptr, &sb, sizeof sb);
    }
}

void replay_lists(Frame& f, const std::vector<Ref<CmdList>>& lists) {
    F = &f;
    RsScope rs(kRsAll);
    for (const Ref<CmdList>& l : lists)
        if (l) run_list(*l, 0);
}

void replay_request_trace() { g_trace_req.store(true); }

void trace_dump_targets() {
    for (const TraceSnap& sn : g_trace_snaps) {
        std::string stats;
        const bool ok = snapshot_write(sn.index, sn.name, &stats);
        NXLOG_I("nvn.trace", "snapshot after draw %d: %s -> %s: %s", sn.draw, sn.desc.c_str(), ok ? sn.name.c_str() : "(not written)", ok ? stats.c_str() : "");
    }
    g_trace_snaps.clear();
    if (g_trace_targets.empty()) return;
    std::vector<Ref<Texture>> list;
    list.swap(g_trace_targets);
    for (const Ref<Texture>& t : list) {
        char buf[96];
        std::string stats;
        const std::string png = std::string("trace_t") + std::to_string(t->id) + "_" + (t->fmt ? t->fmt->name : "x") + ".png";
        const bool ok = dump_texture(t.get(), png, &stats);
        NXLOG_I("nvn.trace", "target %s -> %s: %s", tex_desc(t.get(), buf, sizeof buf), ok ? png.c_str() : "(not dumped)", ok ? stats.c_str() : "");
    }
}

// The copies of feedback attachments are keyed by the attachment's address. A texture the game finalizes (every area load makes new render
// targets) leaves its copy behind, a full-size image each (S34 soak: 500 an hour, about 3 GB an hour of GPU memory). A copy that no submission
// has used for kFbIdleSubmits is released; one that is still in use stays (and one released too early is made again on its next use).
constexpr uint64_t kFbIdleSubmits = 600;
void fb_sweep(uint64_t serial) {
    g_fb_use.clear();       // raw pointers into the copies: none is in use between two submissions
    for (auto it = g_fb_clone.begin(); it != g_fb_clone.end();) {
        if (serial > it->second.last + kFbIdleSubmits) it = g_fb_clone.erase(it);
        else ++it;
    }
    g_fb_clone_n.store(static_cast<int64_t>(g_fb_clone.size()), std::memory_order_relaxed);
}

void replay_end_submit(Frame& f) {
    F = &f;
    cs_close();
    wm_close();
    if (out_any()) {         // every compute result is in game memory, or held and guarded, before the submission's serial signals
        end_pass();
        out_end_submit(F->cb, F->serial);
    }
    end_pass();
    if ((f.serial & 127) == 0 && !g_fb_clone.empty()) fb_sweep(f.serial);
    if ((f.serial & 127) == 0 && !g_vol3d.empty()) {   // 3D copies of 2D textures no submission used for kFbIdleSubmits are released
        g_vol3d_use.clear();
        for (auto it = g_vol3d.begin(); it != g_vol3d.end();) {
            if (f.serial > it->second.last + kFbIdleSubmits) it = g_vol3d.erase(it);
            else ++it;
        }
    }
    if (g_occ_active >= 0) {   // a query cannot outlive its command buffer: the count of an unreported counter is dropped
        vkCmdEndQuery(f.cb, g_occ_pool, static_cast<uint32_t>(g_occ_active));
        g_occ_active = -1;
    }
    for (int k = 0; k < 2; ++k) {
        cw_end_query(g_cw[k], k == 1);
        g_cw[k].active = -1;
    }
    for (; g_dbg_depth > 0; --g_dbg_depth) vkCmdEndDebugUtilsLabelEXT(f.cb);   // labels cannot outlive their command buffer either
    F = nullptr;
}

// A recording that threw: its command buffer is thrown away, so nothing is ended in it. The state that would otherwise make the next
// submission end a pass, query or label it never began is cleared.
void replay_abort_submit() {
    P.open = false;
    g_wm_open = false;
    g_wm_ranges.clear();
    g_cs_open = false;
    g_cs_open_set.clear();
    g_xfb_began = false;
    g_pending.clear();
    g_occ_active = -1;
    for (int k = 0; k < 2; ++k) g_cw[k].active = -1;
    g_dbg_depth = 0;
    g_fb_use.clear();
    F = nullptr;
}

void replay_shutdown() {
    if (g_xfb_counter) device().destroy_buffer(g_xfb_counter);
    g_xfb_counter = gfx::Buffer();
    g_xfb_began = g_xfb_active = g_xfb_have_counter = false;
    g_xfb_slot_of.clear();
    g_xfb_slot_next = g_xfb_cur = 0;
    std::memset(g_xfb_slot_written, 0, sizeof g_xfb_slot_written);
    for (CounterWindow& w : g_cw) {
        if (w.pool) vkDestroyQueryPool(vkdev(), w.pool, nullptr);
        w = CounterWindow();
    }
    g_dbg_depth = 0;
    g_gw_next = g_gw_count = 0;
    g_gw_lost_serial = 0;
    if (g_occ_pool) vkDestroyQueryPool(vkdev(), g_occ_pool, nullptr);
    g_occ_pool = VK_NULL_HANDLE;
    g_occ_tried = false;
    g_occ_active = g_occ_last = -1;
    g_prims_generated = g_prims_xfb = 0;
    g_mask.shutdown();
    g_mask_ready = false;
    dlss_replay_shutdown();
    pipeline_shutdown();
    g_pl = nullptr;                // the layouts are gone with the pipelines
    for (LayoutCacheSlot& sl : g_layout_cache) sl = LayoutCacheSlot();
    for (SkipMaskSlot& sl : g_skip_mask) sl = SkipMaskSlot();
    g_last_gfx = LastGfxSets();
    g_prep_pl = nullptr;
    g_prep_cb = VK_NULL_HANDLE;
    S = GState();
    g_out_all = true;
    raw_binds_reset();
    P = Pass();
    g_pending.clear();
    out_reset();
    for (auto& kv : g_shadows) {   // left at exit otherwise (the validation layer reports them at vkDestroyDevice)
        if (!kv.second.buf) continue;
        forget_buffer_views(kv.second.buf.buffer);
        device().destroy_buffer(kv.second.buf);
    }
    g_shadows.clear();
    g_set_cache.clear();
    g_fb_use.clear();
    g_fb_clone.clear();
    g_vol3d_use.clear();
    g_vol3d.clear();
    g_fb_clone_n.store(0, std::memory_order_relaxed);
}

// Gauge (nxrt/gauge.h): reads the atomic mirror above, never the map. Removed before the globals are destroyed.
namespace {
struct ReplayGauges {
    ReplayGauges() {
        nxrt::gauge_register("nvn.replay.fb_clones", [] { return g_fb_clone_n.load(std::memory_order_relaxed); });
    }
    ~ReplayGauges() { nxrt::gauge_unregister("nvn.replay.fb_clones"); }
};
ReplayGauges g_replay_gauges;
}  // namespace

}  // namespace nxrt::nvn::be::gpu
