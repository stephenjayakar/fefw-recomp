// SPDX-License-Identifier: GPL-3.0-or-later
// Texture images of the NVN back end (render thread only). Each be::Texture gets one VkImage on first use (tex_gpu). The image is
// created without a view: every view (sampled, attachment, present alias) is made here and cached on the TexGpu. Formats: the NVN
// format when the GPU has it, its vk_fallback when the GPU refuses it, ASTC as BC7 when the GPU has no ASTC.
// Uploads: prepare_texture (nvn_texdata.cpp) gives the bytes, they go through a staging buffer into one buffer-to-image copy per
// level. Layouts are tracked per image (all levels and layers share one layout). Samplers are cached by their NVN state.
// Readback (download_texture) copies a render target back and tiles it into the game's storage.
#include "texture_upload_readiness.h"
#include <mutex>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "nxrt/config.h"
#include "nxrt/ftrace.h"
#include "nxrt/gfx_tuning.h"
#include "nxrt/gauge.h"
#include "nxrt/log.h"
#include "nxrt/memory.h"
#include "nxrt/write_watch.h"
#include "vk_gauges.h"
#include "nvn_enums.h"
#include "nvn_format.h"
#include "nvn_gpu_internal.h"
#include "extras/hdr_nvn.h"
#include "nvn_texdata.h"
#include "nvn_texprep.h"
#include "nvn_texup.h"
#include "nvn_texture_layout.h"
#include "tegra_tiling.h"
#include "astc2bc7.h"

namespace nxrt::nvn::be {

// The Vulkan side of a texture: Texture::gpu points to it (created by gpu::tex_gpu, freed by gpu::tex_destroy).
static std::mutex g_live_mu;
static std::set<struct TexGpu*> g_live_set;

struct TexGpu {
    bool promoted = false;                              // HDR scene pipeline: an 8-bit sRGB render target kept as RGBA16F (src/gfx/extras/hdr_nvn.h)
    gfx::Image img;                                     // the image; created without a view (views are cached below)
    const FormatInfo* fmt = nullptr;
    VkFormat format = VK_FORMAT_UNDEFINED;              // the image format: the NVN one, its fallback, or BC7 for ASTC
    VkImageType type = VK_IMAGE_TYPE_2D;
    VkExtent3D base = {1, 1, 1};                        // level 0 extent (depth for 3D images)
    uint32_t levels = 1, layers = 1;
    VkImageUsageFlags usage = 0;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;   // tracked layout (the same for every level and layer)
    bool failed = false;                                // no image: the texture is not usable on the GPU (logged once)
    uint32_t retry_at = 0;                              // failed for want of memory: present count from which tex_or_null makes it again (0: never)
    uint32_t retries = 0;                               // times it was made again after such a failure
    bool transcoded = false;                            // ASTC stored as BC7
    uint32_t drop = 0;                                  // NVN levels left out at the top ([gfx] texture_detail): image level i is NVN level i + drop
    bool mutable_fmt = false;                           // sRGB and UNORM views of the image are allowed
    bool cube = false;                                  // created with CUBE_COMPATIBLE
    bool depth = false;                                 // depth and/or stencil aspect
    bool uploaded = false;                              // the image holds the storage's content (or the black fallback)
    bool gpu_written = false;                           // rendered by the GPU since the CPU last wrote the storage
    bool presented = false;                             // also read by the renderer, outside the NVN submission timeline
    float scale = 1.0f;                                 // resolution scale: base is the NVN size times this ([gfx] render_scale, render targets only)
    bool scale_blocked = false;                         // was scaled, brought back to 1 (bound as a storage image, or paired with an unscaled target)
    std::map<std::array<uint32_t, 12>, VkImageView> views;
    std::map<VkFormat, bool> sampled_formats;
    std::array<uint32_t, 12> last_key = {};             // view_for's last answer (render thread): most lookups repeat it (rt agent)
    VkImageView last_view = VK_NULL_HANDLE;
    int refs = 1;                                      // texture objects over the same storage share one image (render thread)
    struct SparseState* sparse = nullptr;               // a sparse-residency image (a big texture of a virtual pool), else null
    // CPU-streamed textures (tex_ensure_current): uploads for CPU writes that came at most kStreamGap presents apart, in a row
    uint32_t cpu_streak = 0, cpu_upload_at = 0;
    bool streamed = false;                              // rewritten by the CPU frame after frame: not write-watched, read again once per present
    uint32_t streamed_at = 0;                           // present count of the last read in that mode
    bool untiled = false;                               // the last upload went through gpu_untile (cheap enough to repeat every frame)
    // the nvn.tex.* gauges (tex_census): what the image costs, what its ASTC source would cost on a GPU that samples ASTC, and when a
    // draw last asked for a sampled view of it
    uint64_t est_bytes = 0, est_native_bytes = 0;
    std::atomic<uint32_t> sampled_at{0};                // present count + 1 at the last tex_sampled_view; 0: never
};

// A virtual-pool texture too big to back fully (Mario Kart World's terrain virtual texture caches: 8192x8192 arrays of 68 layers, 1.4 to
// 5.8 GB, of which a few hundred MB of tiles are mapped) is a Vulkan sparse-residency image. Host sparse blocks get device memory when a
// mapped guest tile lands on them; the mip tail and the levels whose guest tiles are not whole GOB-block runs are bound at creation.
struct SparseState {
    VkDeviceSize page = 0x10000;                        // bytes of one sparse block (the image's memory alignment)
    uint32_t mem_type = 0;
    VkExtent3D gran = {1, 1, 1};                        // texels of one sparse block
    uint32_t tail_first = 0;                            // first level of the mip tail
    VkDeviceSize tail_size = 0, tail_offset = 0, tail_stride = 0;
    bool single_tail = false;
    std::vector<VkDeviceMemory> chunks;                 // device memory the blocks are suballocated from
    VkDeviceSize chunk_used = 0, chunk_size = 0;
    std::set<uint64_t> bound;                           // (layer, level, block y, block x) of the bound blocks
    TexLayout L;                                        // the guest layout
    std::vector<VkSparseImageMemoryBind> binds;         // waiting for the next flush
    std::vector<VkSparseMemoryBind> opaque;
    uint64_t bytes = 0;                                 // device memory bound
    uint64_t tiles_uploaded = 0;
};

Texture::~Texture() {
    lifetime_texture_freed(this);
    if (gpu) gpu::destroy_texture_gpu(gpu);
    gpu = nullptr;
}

namespace gpu {
namespace {

constexpr const char* kTag = "nvn.tex";
std::atomic<uint64_t> g_live_images{0};
nxrt::GaugeCounter g_gauge_textures_live("nvn.textures_live");               // TexGpu objects alive (g_live_set)
nxrt::GaugeCounter g_gauge_texture_snapshots("nvn.texture_snapshots");       // debug snapshots kept (g_snaps; the list never shrinks)

// The hitch line: where the render thread's upload time goes (bytes it prepared itself, of them stand-ins; images it made; GPU untiles).
std::atomic<uint64_t> g_rt_prep_bytes{0}, g_rt_standin_bytes{0}, g_rt_untile_bytes{0}, g_rt_images{0}, g_rt_gpu_untiles{0};
const bool g_rt_hitch_registered =
    (nxrt::hitch_add_counter("rt_prep_kb", []() -> uint64_t { return g_rt_prep_bytes.load(std::memory_order_relaxed) >> 10; }),
     nxrt::hitch_add_counter("rt_standin_kb", []() -> uint64_t { return g_rt_standin_bytes.load(std::memory_order_relaxed) >> 10; }),
     nxrt::hitch_add_counter("rt_untile_kb", []() -> uint64_t { return g_rt_untile_bytes.load(std::memory_order_relaxed) >> 10; }),
     nxrt::hitch_add_counter("rt_images", []() -> uint64_t { return g_rt_images.load(std::memory_order_relaxed); }),
     nxrt::hitch_add_counter("gpu_untiles", []() -> uint64_t { return g_rt_gpu_untiles.load(std::memory_order_relaxed); }), true);
// A phase timer of the hitch line that counts only on the render thread.
struct RtPhase {
    int phase;
    bool on;
    uint64_t t0;
    explicit RtPhase(int p) : phase(p), on(nxrt::hitch_on_render_thread()), t0(on ? nxrt::hitch_ticks() : 0) {}
    ~RtPhase() { if (on) nxrt::hitch_phase_add(phase, nxrt::hitch_ticks() - t0); }
    RtPhase(const RtPhase&) = delete;
    RtPhase& operator=(const RtPhase&) = delete;
};

// Logs a message only the first time its key is seen (render thread only).
bool first_time(const std::string& key) {
    static std::set<std::string> seen;
    return seen.insert(key).second;
}

uint64_t div_up(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

// ---- formats -------------------------------------------------------------------------------------------------------------
bool is_block_format(VkFormat f) {
    return (f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_BC7_SRGB_BLOCK) ||
           (f >= VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK && f <= VK_FORMAT_EAC_R11G11_SNORM_BLOCK) ||
           (f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK);
}

bool is_srgb(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
        case VK_FORMAT_BC2_SRGB_BLOCK:
        case VK_FORMAT_BC3_SRGB_BLOCK:
        case VK_FORMAT_BC7_SRGB_BLOCK:
            return true;
        default:
            break;
    }
    if (f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK)
        return ((static_cast<int>(f) - static_cast<int>(VK_FORMAT_ASTC_4x4_UNORM_BLOCK)) & 1) != 0;
    return false;
}

// The sRGB / UNORM partner of a format (the two may view the same image when it is mutable), UNDEFINED when there is none.
VkFormat srgb_partner(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_SRGB;
        case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_SRGB;
        case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK: return VK_FORMAT_BC1_RGB_SRGB_BLOCK;
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK: return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case VK_FORMAT_BC2_UNORM_BLOCK: return VK_FORMAT_BC2_SRGB_BLOCK;
        case VK_FORMAT_BC2_SRGB_BLOCK: return VK_FORMAT_BC2_UNORM_BLOCK;
        case VK_FORMAT_BC3_UNORM_BLOCK: return VK_FORMAT_BC3_SRGB_BLOCK;
        case VK_FORMAT_BC3_SRGB_BLOCK: return VK_FORMAT_BC3_UNORM_BLOCK;
        case VK_FORMAT_BC7_UNORM_BLOCK: return VK_FORMAT_BC7_SRGB_BLOCK;
        case VK_FORMAT_BC7_SRGB_BLOCK: return VK_FORMAT_BC7_UNORM_BLOCK;
        default: break;
    }
    if (f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK) {
        const int base = static_cast<int>(VK_FORMAT_ASTC_4x4_UNORM_BLOCK);
        return static_cast<VkFormat>(((static_cast<int>(f) - base) ^ 1) + base);   // the ASTC UNORM/SRGB pairs are consecutive
    }
    return VK_FORMAT_UNDEFINED;
}

// The float member of a 32/64/128-bit integer format's class and back: Maxwell moves the raw bits, whatever the shader's declared
// type (the translated shaders declare every sampled value and colour output as float).
VkFormat float_alias(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: return VK_FORMAT_R32_SFLOAT;
        case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: return VK_FORMAT_R32G32_SFLOAT;
        case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: return VK_FORMAT_R32G32B32A32_SFLOAT;
        default: return VK_FORMAT_UNDEFINED;
    }
}
VkFormat uint_alias(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_SINT: return VK_FORMAT_R32_UINT;
        case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_SINT: return VK_FORMAT_R32G32_UINT;
        case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_SINT: return VK_FORMAT_R32G32B32A32_UINT;
        default: return VK_FORMAT_UNDEFINED;
    }
}
VkFormat sint_alias(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT: return VK_FORMAT_R32_SINT;
        case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: return VK_FORMAT_R32G32_SINT;
        case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: return VK_FORMAT_R32G32B32A32_SINT;
        default: return VK_FORMAT_UNDEFINED;
    }
}
bool bit_alias_class(VkFormat f) { return float_alias(f) != VK_FORMAT_UNDEFINED || uint_alias(f) != VK_FORMAT_UNDEFINED; }

// Same image family: the format itself, its sRGB / UNORM partner, or a raw-bit alias of the same size, on a mutable image.
bool format_view_ok(const TexGpu& g, VkFormat fmt) {
    if (fmt == g.format) return true;
    if (!g.mutable_fmt) return false;
    if (srgb_partner(g.format) == fmt) return true;
    return bit_alias_class(g.format) && (fmt == float_alias(g.format) || fmt == uint_alias(g.format) || fmt == sint_alias(g.format));
}

// A promoted image answers the 8-bit RGBA formats the game asks for (its sRGB and UNORM views) with its own format: the values are linear floats.
VkFormat promoted_view_format(const TexGpu& g, VkFormat f) {
    // An image made in its format's fallback (D24S8 as D32_SFLOAT_S8_UINT on GPUs without D24S8, e.g. AMD) answers views of the
    // NVN format with its own: before, every shader sampling such a depth buffer got the dummy view.
    if (g.fmt && f == g.fmt->vk && g.format == g.fmt->vk_fallback && f != g.format) return g.format;
    if (g.promoted && (f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_R8G8B8A8_UNORM)) return g.format;
    // An ASTC view (a texture view naming its ASTC format, sRGB or not) of an image stored transcoded (BC7, or the disk cache's RGBA8/BC5)
    // reads the image's own format, as its sRGB or UNORM partner like the view asks. Mario Kart World's character and course materials
    // name their format in the view; before, they sampled the black dummy.
    if (g.transcoded && f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK) {
        const bool want_srgb = ((f - VK_FORMAT_ASTC_4x4_UNORM_BLOCK) & 1) != 0;
        const VkFormat partner = srgb_partner(g.format);
        if (partner == VK_FORMAT_UNDEFINED) return g.format;
        const bool img_srgb = is_srgb(g.format);
        return want_srgb == img_srgb ? g.format : partner;
    }
    return f;
}

// The format a colour attachment of this image is rendered through: integer 32-bit formats as their float alias, because the
// translated fragment shaders write float outputs and the console stores the register bits unconverted.
VkFormat attachment_format_of(const TexGpu& g, VkFormat requested) {
    const VkFormat f = requested == VK_FORMAT_UNDEFINED ? g.format : requested;
    if (g.mutable_fmt && bit_alias_class(f)) {
        const VkFormat a = float_alias(f);
        if (a != VK_FORMAT_UNDEFINED && format_view_ok(g, a)) return a;
    }
    return f;
}

VkImageAspectFlags aspect_bits(const FormatInfo* f) {
    switch (f->aspect) {
        case kAspectDepth: return VK_IMAGE_ASPECT_DEPTH_BIT;
        case kAspectStencil: return VK_IMAGE_ASPECT_STENCIL_BIT;
        case kAspectDepthStencil: return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        default: return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

// ---- shapes --------------------------------------------------------------------------------------------------------------
struct Shape {
    VkImageType type = VK_IMAGE_TYPE_2D;
    VkExtent3D base = {1, 1, 1};
    uint32_t levels = 1, layers = 1;
    bool cube = false;
    uint32_t samples = 1;                      // multisampled render target / resolve source
};

Shape shape_of(const Texture& t) {
    const TextureBuilderInfo& b = t.b;
    Shape s;
    const bool is1d = b.target == NVN_TEXTURE_TARGET_1D || b.target == NVN_TEXTURE_TARGET_1D_ARRAY;
    const bool is3d = b.target == NVN_TEXTURE_TARGET_3D;
    s.type = is3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    s.base.width = static_cast<uint32_t>(std::max(1, b.width));
    s.base.height = is1d ? 1u : static_cast<uint32_t>(std::max(1, b.height));
    s.base.depth = is3d ? static_cast<uint32_t>(std::max(1, b.depth)) : 1u;
    s.levels = std::max<uint32_t>(1, t.levels);
    s.layers = std::max<uint32_t>(1, t.layers);
    // a square 2D array of 6n layers may be sampled as cubes (Fortune's Weave's environment compute shaders do): cube-compatible too
    s.cube = target_is_cube(b.target) ||
             (b.target == NVN_TEXTURE_TARGET_2D_ARRAY && s.layers >= 6 && s.layers % 6 == 0 && s.base.width == s.base.height && b.samples <= 1);
    if (target_is_multisample(b.target) || b.samples > 1) {
        s.samples = static_cast<uint32_t>(vk_sample_count(b.samples));
        s.levels = 1;
    }
    return s;
}

VkExtent3D level_extent(const VkExtent3D& base, uint32_t level) {
    auto shr = [level](uint32_t v) { return level >= 32 ? 1u : std::max<uint32_t>(1, v >> level); };
    return {shr(base.width), shr(base.height), shr(base.depth)};
}

gfx::ImageDesc make_desc(const Shape& s, VkFormat fmt, VkImageUsageFlags usage, VkImageCreateFlags flags) {
    gfx::ImageDesc d;
    d.type = s.type;
    d.format = fmt;
    d.width = s.base.width;
    d.height = s.base.height;
    d.depth = s.base.depth;
    d.mip_levels = s.levels;
    d.array_layers = s.layers;
    d.usage = usage;
    d.flags = flags;
    d.samples = static_cast<VkSampleCountFlagBits>(s.samples);
    d.create_view = false;
    d.name = "nvn texture";
    return d;
}

VkImageCreateFlags base_flags(const Shape& s, VkFormat fmt) {
    VkImageCreateFlags f = 0;
    if (s.cube) f |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    if (s.type == VK_IMAGE_TYPE_3D) f |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;   // 2D views of the slices
    if (srgb_partner(fmt) != VK_FORMAT_UNDEFINED || bit_alias_class(fmt)) f |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    return f;
}

// ---- sparse residency images --------------------------------------------------------------------------------------------------
constexpr uint64_t kGuestTile = 0x10000;   // NVN sparse tile: 64 KiB of storage

// A level whose 64 KiB pieces of storage are each a whole run of GOB blocks of one block row, so that a piece is a rectangle of texels.
bool sparse_level_chunked(const SparseState& sp, uint32_t level) {
    const TexLayout& L = sp.L;
    if (level >= L.layout.levels.size() || level >= sp.tail_first) return false;
    const tegra::LevelLayout& lv = L.layout.levels[level];
    const uint64_t bpb = L.desc.bytes_per_block, row = static_cast<uint64_t>(lv.blocks_x) * bpb;
    const uint64_t block_bytes = 512ull * lv.gob_blocks_y;
    if (row % 64 || kGuestTile % block_bytes || lv.gob_blocks_z != 1) return false;
    const uint64_t per = kGuestTile / block_bytes, gob_cols = row / 64;
    return gob_cols % per == 0 && lv.blocks_y % (8ull * lv.gob_blocks_y) == 0 && lv.tiled_offset % kGuestTile == 0 && lv.tiled_size % kGuestTile == 0 &&
           L.layout.tiled_layer_stride % kGuestTile == 0;
}

bool sparse_alloc(SparseState& sp, VkDeviceSize bytes, VkDeviceMemory* mem, VkDeviceSize* off) {
    bytes = (bytes + sp.page - 1) / sp.page * sp.page;
    if (sp.chunks.empty() || sp.chunk_used + bytes > sp.chunk_size) {
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = std::max<VkDeviceSize>(sp.chunk_size, bytes);
        ai.memoryTypeIndex = sp.mem_type;
        VkDeviceMemory m = VK_NULL_HANDLE;
        if (vkAllocateMemory(vkdev(), &ai, nullptr, &m) != VK_SUCCESS) return false;
        sp.chunks.push_back(m);
        sp.chunk_used = 0;
    }
    *mem = sp.chunks.back();
    *off = sp.chunk_used;
    sp.chunk_used += bytes;
    sp.bytes += bytes;
    return true;
}

// Binds the block (bx, by) of (layer, level) when it is not bound yet.
bool sparse_bind_block(TexGpu& g, uint32_t layer, uint32_t level, uint32_t bx, uint32_t by) {
    SparseState& sp = *g.sparse;
    const uint64_t key = (static_cast<uint64_t>(layer) << 48) | (static_cast<uint64_t>(level) << 40) | (static_cast<uint64_t>(by) << 20) | bx;
    if (sp.bound.count(key)) return true;
    VkDeviceMemory m;
    VkDeviceSize off;
    if (!sparse_alloc(sp, sp.page, &m, &off)) return false;
    const uint32_t lw = std::max(1u, g.base.width >> level), lh = std::max(1u, g.base.height >> level);
    VkSparseImageMemoryBind b = {};
    b.subresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer};
    b.offset = {static_cast<int32_t>(bx * sp.gran.width), static_cast<int32_t>(by * sp.gran.height), 0};
    b.extent = {std::min(sp.gran.width, lw - bx * sp.gran.width), std::min(sp.gran.height, lh - by * sp.gran.height), 1};
    b.memory = m;
    b.memoryOffset = off;
    sp.binds.push_back(b);
    sp.bound.insert(key);
    return true;
}

// Binds the texel rectangle of (layer, level).
bool sparse_bind_rect(TexGpu& g, uint32_t layer, uint32_t level, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h) {
    const SparseState& sp = *g.sparse;
    for (uint32_t by = y0 / sp.gran.height; by <= (y0 + h - 1) / sp.gran.height; ++by)
        for (uint32_t bx = x0 / sp.gran.width; bx <= (x0 + w - 1) / sp.gran.width; ++bx)
            if (!sparse_bind_block(g, layer, level, bx, by)) return false;
    return true;
}

// Hands the waiting binds to the queue and waits for them (binds are rare: new blocks only).
bool sparse_flush_binds(TexGpu& g) {
    SparseState& sp = *g.sparse;
    if (sp.binds.empty() && sp.opaque.empty()) return true;
    VkSparseImageMemoryBindInfo ib = {g.img.image, static_cast<uint32_t>(sp.binds.size()), sp.binds.data()};
    VkSparseImageOpaqueMemoryBindInfo ob = {g.img.image, static_cast<uint32_t>(sp.opaque.size()), sp.opaque.data()};
    VkBindSparseInfo bi = {VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
    if (!sp.binds.empty()) { bi.imageBindCount = 1; bi.pImageBinds = &ib; }
    if (!sp.opaque.empty()) { bi.imageOpaqueBindCount = 1; bi.pImageOpaqueBinds = &ob; }
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(vkdev(), &fi, nullptr, &fence) != VK_SUCCESS) return false;
    if (on_render_thread()) submit_recorded();     // binds come after the submissions recorded before them, as without a batch
    VkResult r;
    {
        std::lock_guard<std::mutex> lk(device().queue_mutex());
        r = vkQueueBindSparse(device().queue(), 1, &bi, fence);
    }
    if (r == VK_SUCCESS) r = vkWaitForFences(vkdev(), 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(vkdev(), fence, nullptr);
    sp.binds.clear();
    sp.opaque.clear();
    if (r != VK_SUCCESS) NXLOG_E(kTag, "vkQueueBindSparse failed (%d)", static_cast<int>(r));
    return r == VK_SUCCESS;
}

bool create_sparse_texture(Texture* t, TexGpu* g, const Shape& s, std::string* why) {
    const gfx::DeviceFeatures& ft = device().features();
    const bool to_bc7 = g->fmt->astc && !ft.astc_ldr;
    const VkFormat fmt = to_bc7 ? (g->fmt->srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK) : g->fmt->vk;
    if (fmt == VK_FORMAT_UNDEFINED) { *why = "no Vulkan format"; return false; }
    auto sp = std::make_unique<SparseState>();
    sp->L = compute_texture_layout(t->b);
    if (!sp->L.valid || !sp->L.block_linear || sp->L.layout.levels.size() < s.levels) { *why = "no block-linear layout"; return false; }

    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.flags = base_flags(s, fmt) | VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {s.base.width, s.base.height, 1};
    ci.mipLevels = s.levels;
    ci.arrayLayers = s.layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage img = VK_NULL_HANDLE;
    VkResult r = vkCreateImage(vkdev(), &ci, nullptr, &img);
    if (r != VK_SUCCESS) { *why = "vkCreateImage " + std::to_string(r); return false; }

    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(vkdev(), img, &mr);
    uint32_t n = 0;
    vkGetImageSparseMemoryRequirements(vkdev(), img, &n, nullptr);
    std::vector<VkSparseImageMemoryRequirements> sr(n);
    vkGetImageSparseMemoryRequirements(vkdev(), img, &n, sr.data());
    const VkSparseImageMemoryRequirements* req = nullptr;
    for (const auto& q : sr)
        if (q.formatProperties.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) req = &q;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(device().physical(), &mp);
    int type = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount && type < 0; ++i)
        if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) type = static_cast<int>(i);
    if (!req || type < 0 || (req->formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT)) {
        vkDestroyImage(vkdev(), img, nullptr);
        *why = !req ? "no colour sparse requirements" : type < 0 ? "no device-local memory type" : "non-standard sparse block size";
        return false;
    }
    sp->page = mr.alignment;
    sp->mem_type = static_cast<uint32_t>(type);
    sp->chunk_size = std::max<VkDeviceSize>(64ull << 20, sp->page);
    sp->gran = req->formatProperties.imageGranularity;
    sp->tail_first = std::min(req->imageMipTailFirstLod, s.levels);
    sp->tail_size = req->imageMipTailSize;
    sp->tail_offset = req->imageMipTailOffset;
    sp->tail_stride = req->imageMipTailStride;
    sp->single_tail = (req->formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) != 0;

    g->img.image = img;
    g->img.format = fmt;
    g->img.extent = ci.extent;
    g->img.mip_levels = s.levels;
    g->img.array_layers = s.layers;
    g->format = fmt;
    g->transcoded = to_bc7;
    g->mutable_fmt = srgb_partner(fmt) != VK_FORMAT_UNDEFINED || bit_alias_class(fmt);
    g->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    g->usage = ci.usage;
    g->sparse = sp.release();
    SparseState& st = *g->sparse;

    // always resident: the mip tail and the levels that are not cut into whole-rectangle guest tiles
    bool ok = true;
    if (st.tail_first < s.levels && st.tail_size) {
        const uint32_t tails = st.single_tail ? 1 : s.layers;
        for (uint32_t l = 0; l < tails && ok; ++l) {
            VkDeviceMemory m;
            VkDeviceSize off;
            ok = sparse_alloc(st, st.tail_size, &m, &off);
            if (ok) st.opaque.push_back({st.tail_offset + l * st.tail_stride, st.tail_size, m, off, 0});
        }
    }
    for (uint32_t lvl = 0; lvl < st.tail_first && ok; ++lvl) {
        if (sparse_level_chunked(st, lvl)) continue;
        const uint32_t lw = std::max(1u, s.base.width >> lvl), lh = std::max(1u, s.base.height >> lvl);
        for (uint32_t l = 0; l < s.layers && ok; ++l) ok = sparse_bind_rect(*g, l, lvl, 0, 0, lw, lh);
    }
    if (ok) ok = sparse_flush_binds(*g);
    std::string chunked;
    for (uint32_t lvl = 0; lvl < st.tail_first; ++lvl) chunked += sparse_level_chunked(st, lvl) ? "T" : "w";
    {
        std::string lay;
        char buf[96];
        for (uint32_t lvl = 0; lvl < s.levels && lvl < st.L.layout.levels.size(); ++lvl) {
            const tegra::LevelLayout& lv = st.L.layout.levels[lvl];
            snprintf(buf, sizeof buf, " L%u@%llx+%llx gob%u %ux%u", lvl, static_cast<unsigned long long>(lv.tiled_offset), static_cast<unsigned long long>(lv.tiled_size),
                     lv.gob_blocks_y, lv.blocks_x, lv.blocks_y);
            lay += buf;
        }
        NXLOG_I(kTag, "texture %u: guest layout: layer stride %llx, storage %llx,%s", t->id, static_cast<unsigned long long>(st.L.layout.tiled_layer_stride),
                static_cast<unsigned long long>(t->storage_size), lay.c_str());
    }
    NXLOG_I(kTag, "texture %u: %s %ux%u x%u layers, %u levels as a sparse-residency image (block %ux%u, %llu KB; tail from level %u, %llu KB%s; "
            "levels %s: T by guest tile, w whole); %llu MB bound at creation%s", t->id, g->fmt->name, s.base.width, s.base.height, s.layers, s.levels,
            st.gran.width, st.gran.height, static_cast<unsigned long long>(st.page >> 10), st.tail_first, static_cast<unsigned long long>(st.tail_size >> 10),
            st.single_tail ? " single" : " per layer", chunked.c_str(), static_cast<unsigned long long>(st.bytes >> 20), ok ? "" : " (FAILED)");
    if (!ok) { *why = "binding the always-resident levels failed"; return false; }
    g_live_images.fetch_add(1);
    counters().textures_created.fetch_add(1);
    return true;
}


// ---- texture detail --------------------------------------------------------------------------------------------------------
// The game's ASTC textures are 8x8 blocks (2 bits per texel); stored as BC7 (8 bits) they take four times the console's memory, about
// 2.5 GB in the capital. [gfx] texture_detail = half leaves out their largest level: the image starts at the second one, a quarter of the
// memory (what the console spends). Only mipmapped 2D textures from kDropMinSize texels up (UI and other single-level textures keep every
// texel). auto (the default): half on a GPU whose video memory heap is under kDropAutoHeap, full otherwise. Read once: images keep their shape.
constexpr uint32_t kDropMinSize = 512;
constexpr uint64_t kDropAutoHeap = 9ull << 30;
bool texture_detail_half() {
    static const bool half = [] {
        std::string v = nxrt::config_get_str("gfx", "texture_detail", "auto");
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(device().physical(), &mp);
        uint64_t heap = 0;
        for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
            if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) heap = std::max<uint64_t>(heap, mp.memoryHeaps[i].size);
        bool h = false;
        if (v == "half") h = true;
        else if (v == "full") h = false;
        else {
            if (v != "auto") NXLOG_W(kTag, "[gfx] texture_detail '%s' is not auto, full or half: using auto", v.c_str());
            v = "auto";
            h = heap != 0 && heap < kDropAutoHeap;
        }
        NXLOG_I(kTag, "texture detail: %s ([gfx] texture_detail = %s, video memory heap %.1f GiB)%s", h ? "half" : "full", v.c_str(),
                static_cast<double>(heap) / (1024.0 * 1024.0 * 1024.0),
                h ? ": transcoded ASTC textures of 512 texels and more are stored without their largest level" : "");
        return h;
    }();
    return half;
}

// ---- creation -------------------------------------------------------------------------------------------------------------
// Makes the image of t (or the failed marker when no format works). Stores the result in t->gpu, which is never null afterwards.
// ---- the texture census (gauges nvn.tex.*, read by the soak sampler) ---------------------------------------------------------------
// Estimated device memory of an image: every level, layer and sample at the format's size (alignment and allocator overhead not counted).
// `block_w`, `block_h`, `unit`: texels per block and bytes per block (1, 1 and the bytes of a texel for an uncompressed format).
uint64_t estimate_image_bytes(VkExtent3D e, uint32_t levels, uint32_t layers, uint32_t samples, uint32_t block_w, uint32_t block_h, uint32_t unit) {
    if (!block_w || !block_h) return 0;
    uint64_t total = 0;
    for (uint32_t l = 0; l < levels; ++l) {
        const uint64_t w = std::max<uint32_t>(1, e.width >> l), h = std::max<uint32_t>(1, e.height >> l), d = std::max<uint32_t>(1, e.depth >> l);
        total += ((w + block_w - 1) / block_w) * ((h + block_h - 1) / block_h) * d * unit;
    }
    return total * std::max<uint32_t>(1, layers) * std::max<uint32_t>(1, samples);
}

struct TexCensus {
    int64_t astc = 0;          // bytes of the images that stand in for ASTC textures (BC7, or RGBA8 / BC5 from a cache entry)
    int64_t astc_native = 0;   // what those textures occupy as ASTC (the console's cost, and this GPU's if it sampled ASTC)
    int64_t compressed = 0;    // other block-compressed textures
    int64_t plain = 0;         // uncompressed colour images (render targets among them)
    int64_t depth = 0;
    int64_t never = 0;         // images no draw has asked a sampled view of
    int64_t idle_600 = 0;      // sampled, but not in the last 600 presents (10 s at 60 fps)
    int64_t idle_3600 = 0;     // ... not in the last 3600 presents (a minute)
    int64_t count_never = 0, count_idle_3600 = 0;
};

// One walk of the live images serves all the gauges of a sampler pass (they are read back to back).
TexCensus tex_census() {
    static std::mutex mu;
    static TexCensus cached;
    static std::chrono::steady_clock::time_point at{};
    std::lock_guard<std::mutex> lk(mu);
    const auto now = std::chrono::steady_clock::now();
    if (at != std::chrono::steady_clock::time_point{} && now - at < std::chrono::seconds(1)) return cached;
    TexCensus c;
    if (!counters_ready()) return c;   // a gauge read (soak sampler) before the GPU is set up
    const uint32_t presents = static_cast<uint32_t>(counters().presents.load(std::memory_order_relaxed));
    {
        std::lock_guard<std::mutex> lk2(g_live_mu);
        for (const TexGpu* g : g_live_set) {
            if (g->failed || !g->img) continue;
            const int64_t b = static_cast<int64_t>(g->est_bytes);
            if (g->transcoded) { c.astc += b; c.astc_native += static_cast<int64_t>(g->est_native_bytes); }
            else if (g->depth) c.depth += b;
            else if (g->fmt && g->fmt->compressed) c.compressed += b;
            else c.plain += b;
            const uint32_t sa = g->sampled_at.load(std::memory_order_relaxed);
            if (!sa) { c.never += b; ++c.count_never; }
            else {
                const uint32_t age = presents + 1 - sa;
                if (age > 600) c.idle_600 += b;
                if (age > 3600) { c.idle_3600 += b; ++c.count_idle_3600; }
            }
        }
    }
    cached = c;
    at = now;
    return c;
}

struct TexCensusGauges {
    TexCensusGauges() {
        nxrt::gauge_register("nvn.tex.astc_mb", [] { return tex_census().astc >> 20; });
        nxrt::gauge_register("nvn.tex.astc_native_mb", [] { return tex_census().astc_native >> 20; });
        nxrt::gauge_register("nvn.tex.compressed_mb", [] { return tex_census().compressed >> 20; });
        nxrt::gauge_register("nvn.tex.plain_mb", [] { return tex_census().plain >> 20; });
        nxrt::gauge_register("nvn.tex.depth_mb", [] { return tex_census().depth >> 20; });
        nxrt::gauge_register("nvn.tex.never_sampled_mb", [] { return tex_census().never >> 20; });
        nxrt::gauge_register("nvn.tex.idle_600_mb", [] { return tex_census().idle_600 >> 20; });
        nxrt::gauge_register("nvn.tex.idle_3600_mb", [] { return tex_census().idle_3600 >> 20; });
        nxrt::gauge_register("nvn.tex.never_sampled", [] { return tex_census().count_never; });
        nxrt::gauge_register("nvn.tex.idle_3600", [] { return tex_census().count_idle_3600; });
    }
    ~TexCensusGauges() {
        for (const char* n : {"nvn.tex.astc_mb", "nvn.tex.astc_native_mb", "nvn.tex.compressed_mb", "nvn.tex.plain_mb", "nvn.tex.depth_mb",
                              "nvn.tex.never_sampled_mb", "nvn.tex.idle_600_mb", "nvn.tex.idle_3600_mb", "nvn.tex.never_sampled", "nvn.tex.idle_3600"})
            nxrt::gauge_unregister(n);
    }
};
TexCensusGauges g_tex_census_gauges;

constexpr uint32_t kTexRetryPresents = 300;   // a texture whose image failed for want of memory is made again after this many presents (doubling, up to 16 times)

TexGpu* create_texture_gpu(Texture* t, float want_scale) {
    RtPhase ht_image(nxrt::kHpRenderImage);
    if (ht_image.on) g_rt_images.fetch_add(1, std::memory_order_relaxed);
    auto* g = new TexGpu();
    { std::lock_guard<std::mutex> lk(g_live_mu); g_live_set.insert(g); }
    g_gauge_textures_live.add(1);
    t->gpu = g;
    const Shape s = shape_of(*t);
    g->fmt = t->fmt;
    g->type = s.type;
    g->base = s.base;
    g->levels = s.levels;
    g->layers = s.layers;
    g->cube = s.cube;
    auto fail = [&](const std::string& why) -> TexGpu* {
        NXLOG_E(kTag, "texture %u: no image (%s): the texture is not usable on the GPU", t->id, why.c_str());
        g->failed = true;
        counters().textures_failed.fetch_add(1);
        return g;
    };
    if (!g->fmt) return fail("no format");
    // A texture of a virtual (sparse) pool would be a fully backed image here: Mario Kart World's virtual texture caches are 8192x8192 arrays
    // of 68 layers (5.8 GB each as BC7) of which the console maps a few hundred MB of tiles. Until sparse residency images exist, such a
    // texture over [nvn] sparse_image_max_mb gets no image (it samples as missing) instead of exhausting the host's memory.
    if (t->vpool) {
        static const uint64_t max_mb = static_cast<uint64_t>(std::max<int64_t>(0, config_get_int("nvn", "sparse_image_max_mb", 1024)));
        if (max_mb && (t->storage_size >> 20) > max_mb && s.type == VK_IMAGE_TYPE_2D && !s.cube && s.samples == 1 &&
            device().features().sparse_binding && device().features().sparse_residency_image2d && config_get_bool("nvn", "sparse_residency", true)) {
            std::string why;
            if (create_sparse_texture(t, g, s, &why)) return g;
            return fail("sparse residency image: " + why);
        }
        if (max_mb && (t->storage_size >> 20) > max_mb)
            return fail("sparse texture of " + std::to_string(t->storage_size >> 20) + " MB (" + std::to_string(s.base.width) + "x" + std::to_string(s.base.height) +
                        ", " + std::to_string(s.layers) + " layers, " + std::to_string(s.levels) + " levels, NVN format " + std::to_string(t->b.format) +
                        ") is over [nvn] sparse_image_max_mb; needs sparse residency");
    }

    const gfx::DeviceFeatures& ft = device().features();
    const bool to_bc7 = g->fmt->astc && !ft.astc_ldr;
    if (to_bc7 && !ft.bc) return fail("ASTC is stored as BC7 and the GPU has no BC compression");
    const VkFormat bc7_format = g->fmt->srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
    // A disk cache entry that is not BC7 (RGBA8 for a texture BC7 cannot reproduce, BC5 for a two-channel normal map) makes the image in the
    // entry's format; prepare_texture is then told the image's format and delivers the entry's data (docs/S4f.md).
    VkFormat entry = VK_FORMAT_UNDEFINED;
    // the background job already found the cache entry; else the disk ([nvn] tex_async = standin: what was learned, no file opened on the render thread)
    if (to_bc7 && !texprep::peek_entry_format(t, entry)) entry = cache_entry_format(*t, texprep::async_standin());
    const VkFormat want = to_bc7 ? (entry != VK_FORMAT_UNDEFINED ? entry : bc7_format) : g->fmt->vk;
    const VkFormat alt = to_bc7 ? (entry != VK_FORMAT_UNDEFINED ? bc7_format : VK_FORMAT_UNDEFINED) : g->fmt->vk_fallback;

    std::string why = std::string("no Vulkan format for NVN format ") + g->fmt->name;
    VkFormat chosen = VK_FORMAT_UNDEFINED;
    for (VkFormat c : {want, alt}) {
        if (c == VK_FORMAT_UNDEFINED) continue;
        std::string r;
        if (device().image_supported(make_desc(s, c, VK_IMAGE_USAGE_SAMPLED_BIT, base_flags(s, c)), &r)) {
            chosen = c;
            break;
        }
        why = r;
    }
    if (chosen == VK_FORMAT_UNDEFINED) return fail(why);

    if (const VkFormat hf = extras::hdr_promote_target(chosen, t->b.flags, s.type == VK_IMAGE_TYPE_2D, s.levels, s.layers, s.samples, g->fmt->srgb)) {
        if (device().image_supported(make_desc(s, hf, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, base_flags(s, hf)))) {
            NXLOG_I(kTag, "texture %u: %s %ux%u (flags 0x%x) kept as float16 for the HDR scene", t->id, g->fmt->name, s.base.width, s.base.height, static_cast<unsigned>(t->b.flags));
            chosen = hf;
            g->promoted = true;
        }
    }

    g->format = chosen;
    g->transcoded = to_bc7;
    if (to_bc7 && (chosen == VK_FORMAT_R8G8B8A8_UNORM || chosen == VK_FORMAT_R8G8B8A8_SRGB) && astc_quality() == AstcQuality::Uncompressed) note_uncompressed_image(*t);
    g->mutable_fmt = srgb_partner(chosen) != VK_FORMAT_UNDEFINED || bit_alias_class(chosen);
    g->aspect = aspect_bits(g->fmt);
    g->depth = (g->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0;

    // Required usage is sampling. Each optional usage is kept only when the GPU accepts the image with it.
    VkImageCreateFlags flags = base_flags(s, chosen);
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    const auto add = [&](VkImageUsageFlags bit) {
        if (device().image_supported(make_desc(s, chosen, usage | bit, flags))) usage |= bit;
    };
    add(VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    add(VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    if (!g->depth && !is_block_format(chosen)) add(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    if (g->depth) add(VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    // The console reads and writes any colour texture through an image handle; the builder's IMAGE flag is a hint the game does not always
    // set (the froxel volume of the fog is written by a vertex shader without it). Every uncompressed colour image therefore keeps the
    // storage usage when the GPU accepts the combination.
    if (!g->depth && !is_block_format(chosen)) {
        add(VK_IMAGE_USAGE_STORAGE_BIT);
        // sRGB formats have no storage support. The console's image stores write the raw bits (no sRGB encode), which is what a UNORM view
        // of the same image does: storage usage is then allowed through EXTENDED_USAGE when the UNORM partner can be stored to.
        const VkFormat unorm = is_srgb(chosen) ? srgb_partner(chosen) : VK_FORMAT_UNDEFINED;
        if (!(usage & VK_IMAGE_USAGE_STORAGE_BIT) && unorm != VK_FORMAT_UNDEFINED && (flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) &&
            device().image_supported(make_desc(s, unorm, usage | VK_IMAGE_USAGE_STORAGE_BIT, flags))) {
            flags |= VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
            usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        }
        if ((t->b.flags & NVN_TEXTURE_FLAGS_IMAGE) && !(usage & VK_IMAGE_USAGE_STORAGE_BIT))
            NXLOG_W(kTag, "texture %u (%s, image flag set): the GPU refuses storage usage with the other usages (flags 0x%x, usage 0x%x)", t->id, t->fmt ? t->fmt->name : "?",
                    static_cast<unsigned>(flags), static_cast<unsigned>(usage));
    }
    g->usage = usage;

    // Resolution scale: the image is the NVN size times the scale. Other modules keep the NVN size (tex_extent) and ask tex_scale /
    // tex_image_extent where the image size matters (render area, viewports, copies, the support buffer's render_scale).
    Shape is = s;
    if (want_scale != 1.0f && !g->transcoded && !is_block_format(chosen)) {
        const uint32_t lim = device().properties().limits.maxImageDimension2D;
        const uint32_t w = static_cast<uint32_t>(std::lround(s.base.width * want_scale)), h = static_cast<uint32_t>(std::lround(s.base.height * want_scale));
        if (w >= 1 && h >= 1 && w <= lim && h <= lim) {
            is.base.width = w;
            is.base.height = h;
            g->scale = want_scale;
            g->base = is.base;
            static int logged = 0;
            if (logged++ < 60)
                NXLOG_I(kTag, "texture %u: %s %ux%u (flags 0x%x, %u levels, %u layers, %u samples) rendered at %ux%u (scale %.3f)", t->id, g->fmt->name, s.base.width,
                        s.base.height, static_cast<unsigned>(t->b.flags), s.levels, s.layers, s.samples, w, h, want_scale);
        }
    }
    if (g->transcoded && !t->vpool && s.type == VK_IMAGE_TYPE_2D && s.samples == 1 && s.levels >= 2 &&
        std::max(s.base.width, s.base.height) >= kDropMinSize && texture_detail_half()) {
        g->drop = 1;
        is.base = level_extent(s.base, 1);
        is.levels = s.levels - 1;
        g->base = is.base;
        g->levels = is.levels;
    }
    std::string why2;
    const VkResult cr = device().create_image(make_desc(is, chosen, usage, flags), &g->img, &why2);
    if (cr == VK_ERROR_OUT_OF_DEVICE_MEMORY || cr == VK_ERROR_OUT_OF_HOST_MEMORY) {
        // Out of memory is not the texture's fault: tex_or_null makes it again later (kTexRetryPresents). Under pressure thousands of
        // textures can fail in a row: the first few are logged, then one in 256.
        static std::atomic<uint32_t> n_oom{0};
        const uint32_t k = n_oom.fetch_add(1, std::memory_order_relaxed);
        if (k < 16 || (k - 16) % 256 == 255)
            NXLOG_E(kTag, "texture %u: no image (out of memory: %s, %ux%u, %u levels, %u layers): not usable on the GPU until a retry (%u such failures)",
                    t->id, why2.c_str(), is.base.width, is.base.height, is.levels, s.layers, k + 1);
        g->failed = true;
        g->retry_at = static_cast<uint32_t>(counters().presents.load(std::memory_order_relaxed)) + kTexRetryPresents;
        counters().textures_failed.fetch_add(1);
        return g;
    }
    if (cr != VK_SUCCESS) return fail("create_image: " + why2);
    {   // the census: the image's size in its format, and for an ASTC stand-in the size of the ASTC it replaces
        const bool bc8 = chosen >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && chosen <= VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
        const bool bc4 = chosen == VK_FORMAT_BC4_UNORM_BLOCK || chosen == VK_FORMAT_BC4_SNORM_BLOCK;
        if (is_block_format(chosen) && (g->transcoded || !g->fmt->compressed))
            g->est_bytes = estimate_image_bytes(is.base, s.levels, s.layers, s.samples, 4, 4, bc8 || bc4 ? 8 : 16);
        else if (g->transcoded || g->promoted)   // RGBA8 for an ASTC texture BC7 cannot reproduce; RGBA16F for a promoted target
            g->est_bytes = estimate_image_bytes(is.base, s.levels, s.layers, s.samples, 1, 1, g->promoted ? 8 : 4);
        else
            g->est_bytes = estimate_image_bytes(is.base, s.levels, s.layers, s.samples, g->fmt->block_w, g->fmt->block_h, g->fmt->bytes);
        if (g->transcoded) g->est_native_bytes = estimate_image_bytes(is.base, s.levels, s.layers, s.samples, g->fmt->block_w, g->fmt->block_h, g->fmt->bytes);
    }
    g_live_images.fetch_add(1);
    counters().textures_created.fetch_add(1);
    return g;
}

// The texture's image, created on first use. Null when the texture has no usable image (the failure is counted and logged once).
TexGpu* tex_or_null(Texture* t, float want_scale = 1.0f) {
    if (!t || !active()) return nullptr;
    if (t->gpu) {
        TexGpu* old = t->gpu;
        if (!old->failed) return old;
        // An image that failed for want of memory is made again once its wait is over: the old object (no image, no views) goes after the
        // submissions recorded so far, and the texture pools that hold the texture get its new views.
        const uint32_t now = static_cast<uint32_t>(counters().presents.load(std::memory_order_relaxed));
        if (!old->retry_at || old->refs != 1 || static_cast<int32_t>(now - old->retry_at) < 0 || !on_render_thread()) return nullptr;
        const uint32_t retries = old->retries + 1;
        old->retry_at = 0;
        t->gpu = nullptr;
        destroy_texture_gpu(old);
        counters().textures_failed.fetch_sub(1);
        NXFT_MIN("tex.create", 30000);
        TexGpu* g = create_texture_gpu(t, want_scale);
        g->retries = retries;
        if (g->failed) {
            if (g->retry_at) g->retry_at = now + (kTexRetryPresents << std::min<uint32_t>(retries, 4));
            return nullptr;
        }
        static int logged = 0;
        if (logged++ < 60) NXLOG_I(kTag, "texture %u: image made on retry %u after running out of memory", t->id, retries);
        pool_texture_remade(t);
        return g;
    }
    if (Ref<Texture> alias = texture_registry_find_alias(t)) {
        TexGpu* ag = alias->gpu;
        if (ag && !ag->failed) {
            // the same bytes under another texture object: one image (the object rendered to and the one that is sampled are not the same
            // object); what the image holds is as current for this object as for the one it was made for
            t->gpu = ag;
            ++ag->refs;
            if (ag->uploaded || ag->gpu_written) t->cpu_dirty.store(false);
            static int logged = 0;
            if (logged++ < 20) NXLOG_I(kTag, "texture %u shares the image of texture %u (the same storage)", t->id, alias->id);
            return ag;
        }
    }
    NXFT_MIN("tex.create", 30000);
    TexGpu* g = create_texture_gpu(t, want_scale);
    return g->failed ? nullptr : g;
}

// The resolution scale: the game's play settings ([display] render_scale, turned into a factor by the game module and handed over in
// gfx::tuning) or, for tests, [gfx] render_scale (a number, NXRT_GFX_RENDER_SCALE), which wins. Fixed once known: images keep their size.
// Stays 1 (not fixed) until one of them is there.
float g_render_scale = 0.0f;
float render_scale_now() {
    if (g_render_scale > 0.0f) return g_render_scale;
    static const std::string v = nxrt::config_get_str("gfx", "render_scale", "");   // once: every draw asks (support_for_draw), and a lookup is a getenv
    float sc = 1.0f;
    if (!v.empty()) {
        sc = static_cast<float>(std::atof(v.c_str()));
    } else {
        sc = nxrt::gfx::tuning().render_scale.load();
        if (!(sc > 0.0f)) return 1.0f;
    }
    if (!(sc > 0.0f)) sc = 1.0f;
    sc = std::clamp(sc, 0.25f, 4.0f);
    sc = std::round(sc * 1080.0f) / 1080.0f;   // a whole number of lines at 1080 (1.333 -> 4/3: 1440, not 1439)
    if (std::fabs(sc - 1.0f) < 0.01f) sc = 1.0f;
    g_render_scale = sc;
    NXLOG_I(kTag, "render scale %.3f (%s)", sc, v.empty() ? "play settings" : "[gfx] render_scale");
    return sc;
}

// The scale a texture gets when its image is made for a render target (after Ryujinx's TextureScaling rules): 2D colour or depth targets
// of at least 32x32, not block-compressed, not pitch-linear, and not square powers of two (shadow
// maps, LUTs). Everything else, and every texture first used another way, keeps the NVN size.
float attachment_scale(const Texture& t) {
    const float sc = render_scale_now();
    if (sc == 1.0f) return 1.0f;
    const TextureBuilderInfo& b = t.b;
    if (!t.fmt || t.fmt->compressed || t.fmt->block_w != 1 || t.fmt->block_h != 1) return 1.0f;
    if (b.target != NVN_TEXTURE_TARGET_2D && b.target != NVN_TEXTURE_TARGET_2D_ARRAY && b.target != NVN_TEXTURE_TARGET_2D_MULTISAMPLE &&
        b.target != NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY && b.target != NVN_TEXTURE_TARGET_RECTANGLE)
        return 1.0f;
    // not the IMAGE flag: the game sets it on every render target; storage use brings a scaled image back instead (tex_rescale)
    if (b.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_VIDEO_DECODE)) return 1.0f;
    const int w = b.width, h = b.height;
    if (w < 32 || h < 32) return 1.0f;
    if (w == h && (w & (w - 1)) == 0) return 1.0f;
    return sc;
}

// ---- barriers and copies ----------------------------------------------------------------------------------------------------
void move_layout(Frame& f, TexGpu& g, VkImageLayout to) {
    if (g.layout == to) return;
    // A 3D image created with 2D_ARRAY_COMPATIBLE: without maintenance9 the layer range of a barrier means all depth slices.
    const uint32_t layer_count = g.type == VK_IMAGE_TYPE_3D ? VK_REMAINING_ARRAY_LAYERS : g.layers;
    image_barrier(f.cb, g.img.image, g.layout, to, g.aspect, 0, g.levels, 0, layer_count);
    g.layout = to;
}

// Copies buffer regions (already offset to the staging buffer) into every level of the image, then leaves it readable by shaders.
void copy_regions_to_image(Frame& f, TexGpu& g, VkBuffer buf, const std::vector<VkBufferImageCopy>& regions) {
    move_layout(f, g, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdCopyBufferToImage(f.cb, buf, g.img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<uint32_t>(regions.size()), regions.data());
    move_layout(f, g, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}


// ---- sparse residency: uploads --------------------------------------------------------------------------------------------------
// The dirty ranges of a sparse texture's storage are uploaded tile by tile: a 64 KiB guest tile of a tiled level is a rectangle of texels
// (a run of GOB blocks of one block row); its host blocks are bound, it is untiled (and transcoded to BC7 for ASTC) and copied in. Levels
// that are not tiled that way (small levels and the host mip tail) are uploaded whole per layer. At most [nvn] sparse_tiles_per_frame
// tiles go per call; the rest stays dirty for the next frame.
void sparse_update(Texture* t, TexGpu& g, Frame& f) {
    SparseState& sp = *g.sparse;
    const TexLayout& L = sp.L;
    const uint64_t stride = L.layout.tiled_layer_stride;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    {
        std::lock_guard<std::mutex> lk(t->sparse_mu);
        if (t->sparse_all) {
            t->sparse_all = false;
            t->sparse_dirty.clear();
            ranges.emplace_back(0, t->storage_size);
        } else {
            ranges.swap(t->sparse_dirty);
        }
    }
    if (ranges.empty()) return;
    if (g.gpu_written) return;   // the GPU fills it (copies into the cache): the pool bytes are not the content, its tiles are bound by the copies
    std::sort(ranges.begin(), ranges.end());

    // the tiles and whole (layer, level) pieces the ranges touch
    std::set<uint64_t> tiles;                           // storage offsets of guest tiles
    std::set<std::pair<uint32_t, uint32_t>> whole;      // (layer, level)
    for (const auto& r : ranges) {
        uint64_t a = r.first, b = std::min(r.second, t->storage_size);
        while (a < b) {
            const uint32_t layer = static_cast<uint32_t>(a / stride);
            if (layer >= g.layers) break;
            const uint64_t in = a - static_cast<uint64_t>(layer) * stride;
            uint32_t level = 0;
            while (level + 1 < g.levels && in >= L.layout.levels[level + 1].tiled_offset) ++level;
            const tegra::LevelLayout& lv = L.layout.levels[level];
            const uint64_t lvl_end = static_cast<uint64_t>(layer) * stride + lv.tiled_offset + lv.tiled_size;
            if (a >= lvl_end) {   // padding after the level: the next level, or the next layer
                a = level + 1 < g.levels ? static_cast<uint64_t>(layer) * stride + L.layout.levels[level + 1].tiled_offset : static_cast<uint64_t>(layer + 1) * stride;
                continue;
            }
            if (sparse_level_chunked(sp, level)) {
                const uint64_t tile = a / kGuestTile * kGuestTile;
                tiles.insert(tile);
                a = tile + kGuestTile;
            } else {
                whole.insert({layer, level});
                a = lvl_end;
            }
        }
    }

    static const int64_t max_tiles = std::max<int64_t>(1, config_get_int("nvn", "sparse_tiles_per_frame", 256));
    std::vector<std::pair<uint64_t, uint64_t>> later;
    a2b_options opt{};
    opt.quality = static_cast<int>(config_get_int("nvn", "sparse_bc7_quality", 1));
    opt.threads = 0;
    a2b_input in{};
    in.format = g.fmt->srgb ? A2B_ASTC_SRGB : A2B_ASTC_UNORM;
    in.block_w = L.desc.block_w;
    in.block_h = L.desc.block_h;
    const uint32_t bw = L.desc.block_w, bh = L.desc.block_h, bpb = L.desc.bytes_per_block;

    struct Up { UploadAlloc st; VkBufferImageCopy r; };
    std::vector<Up> ups;
    std::vector<uint8_t> raw(kGuestTile), lin, out;
    uint64_t staged = 0;
    static const uint64_t max_staged = static_cast<uint64_t>(std::max<int64_t>(1, config_get_int("nvn", "sparse_upload_mb", 48))) << 20;
    // Untiles (and transcodes) a block-linear piece of w x h texels (whole blocks) with block height gby; the result goes to a staging allocation.
    auto convert = [&](const uint8_t* tiled, uint32_t w, uint32_t h, uint32_t gby, uint32_t level, uint32_t layer, uint32_t x0, uint32_t y0,
                       uint32_t cw, uint32_t ch) -> bool {
        tegra::TextureDesc d = L.desc;
        d.width = w;
        d.height = h;
        d.depth = 1;
        d.levels = 1;
        d.layers = 1;
        d.gob_blocks_y = gby;
        d.gob_blocks_z = 1;
        d.linear_stride_align = 1;
        const tegra::Layout dl = tegra::compute_layout(d);
        if (!dl.valid || dl.levels.empty()) return false;
        lin.assign(static_cast<size_t>(dl.linear_size), 0);
        tegra::untile_level(d, dl.levels[0], tiled, lin.data());
        const uint8_t* data = lin.data();
        uint64_t bytes = lin.size();
        if (g.transcoded) {
            in.width = w;
            in.height = h;
            out.resize(static_cast<size_t>(a2b_output_size(&in)));
            a2b_format of;
            if (a2b_transcode(&in, lin.data(), lin.size(), &opt, &of, out.data(), out.size(), nullptr) != A2B_OK) return false;
            data = out.data();
            bytes = out.size();
        }
        staged += bytes;
        UploadAlloc st = f.staging(bytes);
        if (!st) return false;
        std::memcpy(st.ptr, data, static_cast<size_t>(bytes));
        VkBufferImageCopy r = {};
        r.bufferOffset = st.offset;
        r.bufferRowLength = w;
        r.bufferImageHeight = h;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1};
        r.imageOffset = {static_cast<int32_t>(x0), static_cast<int32_t>(y0), 0};
        r.imageExtent = {cw, ch, 1};
        ups.push_back({st, r});
        return true;
    };

    // A physical tile mapped at many places of the cache is the game's filler for unused slots (nothing samples them through the page
    // tables): those places stay non-resident (they read as zero) instead of each getting memory and an upload.
    std::vector<const uint8_t*> ptrs;
    texture_virtual_tile_ptrs(*t, ptrs);
    std::unordered_map<const uint8_t*, uint32_t> uses;
    for (const uint8_t* p : ptrs)
        if (p) ++uses[p];
    static const uint32_t filler_min = static_cast<uint32_t>(std::max<int64_t>(2, config_get_int("nvn", "sparse_filler_min", 4)));
    uint64_t skipped_filler = 0, skipped_zero = 0;

    int64_t done = 0;
    uint64_t skipped_unmapped = 0, failed = 0;
    for (uint64_t off : tiles) {
        {
            const size_t ti = static_cast<size_t>(off / kGuestTile);
            const uint8_t* p = ti < ptrs.size() ? ptrs[ti] : nullptr;
            if (!p) { ++skipped_unmapped; continue; }   // not mapped: nothing to read
            if (uses[p] >= filler_min) { ++skipped_filler; continue; }
        }
        if (done >= max_tiles || staged >= max_staged) { later.emplace_back(off, off + kGuestTile); continue; }
        const uint32_t layer = static_cast<uint32_t>(off / stride);
        const uint64_t in_layer = off - static_cast<uint64_t>(layer) * stride;
        uint32_t level = 0;
        while (level + 1 < g.levels && in_layer >= L.layout.levels[level + 1].tiled_offset) ++level;
        const tegra::LevelLayout& lv = L.layout.levels[level];
        if (!texture_read_virtual(*t, off, kGuestTile, raw.data())) { ++skipped_unmapped; continue; }
        {   // a mapped tile the game has not filled yet is zeros (an ASTC error block: magenta once decoded): leave it unbound
            const uint64_t* q = reinterpret_cast<const uint64_t*>(raw.data());
            bool zero = true;
            for (size_t i = 0; i < kGuestTile / 8 && zero; ++i) zero = q[i] == 0;
            if (zero) {
                ++skipped_zero;
                continue;
            }
        }
        // the tile's rectangle: tiles run along a block row of the level
        const uint64_t block_bytes = 512ull * lv.gob_blocks_y, per = kGuestTile / block_bytes;
        const uint64_t gob_cols = static_cast<uint64_t>(lv.blocks_x) * bpb / 64;
        const uint64_t first = (in_layer - lv.tiled_offset) / block_bytes;
        const uint64_t brow = first / gob_cols, bcol = first % gob_cols;
        const uint32_t nbx = static_cast<uint32_t>(per * 64 / bpb), nby = 8u * lv.gob_blocks_y;
        const uint32_t x0 = static_cast<uint32_t>(bcol * 64 / bpb) * bw, y0 = static_cast<uint32_t>(brow) * nby * bh;
        const uint32_t w = nbx * bw, h = nby * bh;
        const uint32_t lw = std::max(1u, g.base.width >> level), lh = std::max(1u, g.base.height >> level);
        if (x0 >= lw || y0 >= lh) continue;
        const uint32_t cw = std::min(w, lw - x0), ch = std::min(h, lh - y0);
        if (!sparse_bind_rect(g, layer, level, x0, y0, cw, ch) || !convert(raw.data(), w, h, lv.gob_blocks_y, level, layer, x0, y0, cw, ch)) {
            ++failed;
            continue;
        }
        ++done;
    }
    std::vector<uint8_t> lvbuf;
    for (const auto& wl : whole) {
        const uint32_t layer = wl.first, level = wl.second;
        const tegra::LevelLayout& lv = L.layout.levels[level];
        lvbuf.assign(static_cast<size_t>(lv.tiled_size), 0);
        texture_read_virtual(*t, static_cast<uint64_t>(layer) * stride + lv.tiled_offset, lv.tiled_size, lvbuf.data());   // unmapped parts are zeros
        const uint32_t w = lv.blocks_x * bw, h = lv.blocks_y * bh;
        const uint32_t lw = std::max(1u, g.base.width >> level), lh = std::max(1u, g.base.height >> level);
        if (!convert(lvbuf.data(), w, h, lv.gob_blocks_y, level, layer, 0, 0, std::min(w, lw), std::min(h, lh))) ++failed;
    }
    if (!sparse_flush_binds(g)) return;
    if (!ups.empty()) {
        move_layout(f, g, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        for (const Up& u : ups) vkCmdCopyBufferToImage(f.cb, u.st.buffer, g.img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &u.r);
        move_layout(f, g, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    } else if (g.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        move_layout(f, g, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    sp.tiles_uploaded += static_cast<uint64_t>(done);
    g.uploaded = true;
    if (!later.empty()) {
        std::lock_guard<std::mutex> lk(t->sparse_mu);
        t->sparse_dirty.insert(t->sparse_dirty.end(), later.begin(), later.end());
        t->cpu_dirty.store(true);
    }
    static int logged = 0;
    if (logged++ < 40 || failed || logged % 50 == 0) {
        NXLOG_I(kTag, "texture %u: sparse upload: %lld tiles (%llu total), %zu whole levels, %llu unmapped, %llu zero and %llu filler skipped, %llu failed, %zu left; %llu MB bound",
                t->id, static_cast<long long>(done), static_cast<unsigned long long>(sp.tiles_uploaded), whole.size(),
                static_cast<unsigned long long>(skipped_unmapped), static_cast<unsigned long long>(skipped_zero), static_cast<unsigned long long>(skipped_filler), static_cast<unsigned long long>(failed), later.size(),
                static_cast<unsigned long long>(sp.bytes >> 20));
    }
}


// The image gets zeros (transparent black). Block formats are filled from a zeroed staging buffer; uncompressed ones are cleared.
void fill_zero(Frame& f, TexGpu& g) {
    if (!(g.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return;
    if (is_block_format(g.format)) {
        uint32_t bw = 0, bh = 0, bytes = 0;
        if (g.transcoded) {
            bw = 4; bh = 4; bytes = 16;
        } else if (g.fmt && g.format == g.fmt->vk) {
            bw = g.fmt->block_w; bh = g.fmt->block_h; bytes = g.fmt->bytes;
        } else {
            return;
        }
        std::vector<VkBufferImageCopy> regions;
        uint64_t total = 0;
        for (uint32_t l = 0; l < g.levels; ++l) {
            const VkExtent3D e = level_extent(g.base, l);
            const uint64_t size = div_up(e.width, bw) * div_up(e.height, bh) * bytes * e.depth * g.layers;
            VkBufferImageCopy r = {};
            r.bufferOffset = total;
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, g.layers};
            r.imageExtent = e;
            regions.push_back(r);
            total += size;
        }
        UploadAlloc st = f.staging(total);
        if (!st) return;
        std::memset(st.ptr, 0, static_cast<size_t>(total));
        for (VkBufferImageCopy& r : regions) r.bufferOffset += st.offset;
        copy_regions_to_image(f, g, st.buffer, regions);
        return;
    }
    move_layout(f, g, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkClearColorValue zero = {};
    const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, g.levels, 0, g.layers};
    vkCmdClearColorImage(f.cb, g.img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
    move_layout(f, g, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

// ---- views ----------------------------------------------------------------------------------------------------------------
bool rt_check_on() {
    static const bool on = nxrt::config_get_bool("nvn", "rt_check", false);
    return on;
}

// The view with these fields, created once and cached on the image.
VkImageView view_for(TexGpu& g, VkImageViewType type, uint32_t base_level, uint32_t level_count, uint32_t base_layer, uint32_t layer_count,
                     VkFormat fmt, VkImageAspectFlags aspect, const VkComponentMapping& swz) {
    const std::array<uint32_t, 12> key = {static_cast<uint32_t>(type), base_level, level_count, base_layer, layer_count,
                                          static_cast<uint32_t>(fmt), static_cast<uint32_t>(aspect), static_cast<uint32_t>(swz.r),
                                          static_cast<uint32_t>(swz.g), static_cast<uint32_t>(swz.b), static_cast<uint32_t>(swz.a), 0u};
    if (g.last_view && g.last_key == key) {
        if (rt_check_on()) {   // [nvn] rt_check: the map holds the same view (notes/rt.md)
            const auto c = g.views.find(key);
            if (c == g.views.end() || c->second != g.last_view) NXLOG_W("nvn.rtcheck", "mismatch: view_for last view");
        }
        return g.last_view;
    }
    auto it = g.views.find(key);
    if (it != g.views.end()) {
        g.last_key = key;
        g.last_view = it->second;
        return it->second;
    }
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = g.img.image;
    vi.viewType = type;
    vi.format = fmt;
    vi.components = swz;
    vi.subresourceRange = {aspect, base_level, level_count, base_layer, layer_count};
    // A view takes the usages of its image; a format of the view that cannot do one of them (an sRGB view cannot be stored to) restricts it.
    VkImageViewUsageCreateInfo uc = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    {
        constexpr VkImageUsageFlags kViewUsages = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        VkFormatProperties fp = {};
        vkGetPhysicalDeviceFormatProperties(device().physical(), fmt, &fp);
        VkImageUsageFlags u = g.usage & kViewUsages;
        if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) u &= ~VK_IMAGE_USAGE_SAMPLED_BIT;
        if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) u &= ~VK_IMAGE_USAGE_STORAGE_BIT;
        if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) u &= ~VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (u != (g.usage & kViewUsages) && !(aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) {
            uc.usage = u;
            vi.pNext = &uc;
        }
    }
    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(vkdev(), &vi, nullptr, &view) != VK_SUCCESS) {
        NXLOG_W(kTag, "vkCreateImageView failed for a view of format %d", static_cast<int>(fmt));
        return VK_NULL_HANDLE;
    }
    g.views.emplace(key, view);
    vk_gauge_image_views().add(1);
    return view;
}

VkImageView bad_view(const Texture& t, const char* what) {
    if (first_time("view " + std::to_string(t.id) + " " + what))
        NXLOG_W(kTag, "texture %u (%s, target %d, %dx%dx%d, %u layers): view not possible (%s): the binding gets the dummy view", t.id,
                t.fmt ? t.fmt->name : "?", t.b.target, t.b.width, t.b.height, t.b.depth, t.layers, what);
    return VK_NULL_HANDLE;
}

const VkComponentMapping kIdentity = {};

// Dummy views: 1x1 (x1) textures of each view type, black (or depth 1.0), made the first time they are asked for.
struct Dummy {
    gfx::Image img;
    VkImageView view = VK_NULL_HANDLE;
    bool tried = false;           // made (set on success only)
    uint32_t next_try = 0;        // after a failure: present count of the next attempt
};

// A dummy whose image failed (out of memory) is tried again 60 presents later; until then the caller gets a null view.
bool dummy_due(Dummy& d) {
    if (d.tried) return false;
    const uint32_t now = static_cast<uint32_t>(counters().presents.load(std::memory_order_relaxed));
    return !d.next_try || static_cast<int32_t>(now - d.next_try) >= 0;
}
void dummy_failed(Dummy& d) { d.next_try = static_cast<uint32_t>(counters().presents.load(std::memory_order_relaxed)) + 60; }
constexpr VkImageViewType kDummyTypes[6] = {VK_IMAGE_VIEW_TYPE_1D, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                                            VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, VK_IMAGE_VIEW_TYPE_3D};

int dummy_index(VkImageViewType type) {
    for (int i = 0; i < 6; ++i)
        if (kDummyTypes[i] == type) return i;
    return 1;   // anything else: a 2D dummy
}

struct Shared {
    std::map<uint64_t, std::vector<std::pair<SamplerInfo, VkSampler>>> samplers;
    VkSampler dummy_sampler = VK_NULL_HANDLE;
    Dummy color[6];
    Dummy sint[6], uint[6];
    Dummy depth[6];
    std::map<std::pair<int, int>, Dummy> storage;   // (view type, format): GENERAL-layout 1x1 storage images
};
Shared g_sh;

bool make_dummy(Dummy& d, VkImageViewType type, bool depth, int component) {
    const VkFormat fmt = depth ? VK_FORMAT_D32_SFLOAT : component == 2 ? VK_FORMAT_R8G8B8A8_SINT :
                         component == 3 ? VK_FORMAT_R8G8B8A8_UINT : VK_FORMAT_R8G8B8A8_UNORM;
    gfx::ImageDesc id;
    id.format = fmt;
    id.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    id.create_view = false;
    id.name = depth ? "nvn dummy depth" : "nvn dummy";
    switch (type) {
        case VK_IMAGE_VIEW_TYPE_1D: id.type = VK_IMAGE_TYPE_1D; break;
        case VK_IMAGE_VIEW_TYPE_3D: id.type = VK_IMAGE_TYPE_3D; break;
        case VK_IMAGE_VIEW_TYPE_CUBE:
        case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
            id.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
            id.array_layers = 6;
            break;
        default: break;
    }
    std::string why;
    if (device().create_image(id, &d.img, &why) != VK_SUCCESS) {
        static int logged = 0;
        if (logged++ < 8) NXLOG_W(kTag, "dummy view image failed: %s", why.c_str());
        dummy_failed(d);
        return false;
    }
    const VkImageAspectFlags asp = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    const uint32_t layers = id.array_layers;
    const VkImage img = d.img.image;
    const bool ok = device().submit_once([&](VkCommandBuffer cmd) {
        image_barrier(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, asp, 0, 1, 0, layers);
        const VkImageSubresourceRange range = {asp, 0, 1, 0, layers};
        if (depth) {
            VkClearDepthStencilValue v = {1.0f, 0};
            vkCmdClearDepthStencilImage(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
        } else {
            VkClearColorValue c = {};
            vkCmdClearColorImage(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &range);
        }
        image_barrier(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, asp, 0, 1, 0, layers);
    });
    if (!ok) {
        NXLOG_W(kTag, "dummy view clear failed");
        device().destroy_image(d.img);
        dummy_failed(d);
        return false;
    }
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img;
    vi.viewType = type;
    vi.format = fmt;
    vi.subresourceRange = {asp, 0, 1, 0, layers};
    if (vkCreateImageView(vkdev(), &vi, nullptr, &d.view) != VK_SUCCESS) {
        d.view = VK_NULL_HANDLE;
        device().destroy_image(d.img);
        dummy_failed(d);
        return false;
    }
    vk_gauge_image_views().add(1);
    d.tried = true;
    return true;
}

// ---- samplers -------------------------------------------------------------------------------------------------------------
uint64_t sampler_hash(const SamplerInfo& s) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
        h ^= h >> 29;
    };
    auto bits = [](float f) {
        uint32_t u = 0;
        std::memcpy(&u, &f, 4);
        return static_cast<uint64_t>(u);
    };
    mix(static_cast<uint32_t>(s.min_filter));
    mix(static_cast<uint32_t>(s.mag_filter));
    mix(static_cast<uint32_t>(s.wrap_s));
    mix(static_cast<uint32_t>(s.wrap_t));
    mix(static_cast<uint32_t>(s.wrap_r));
    mix(bits(s.lod_min));
    mix(bits(s.lod_max));
    mix(bits(s.lod_bias));
    mix(bits(s.lod_snap));
    mix(static_cast<uint32_t>(s.compare_mode));
    mix(static_cast<uint32_t>(s.compare_func));
    for (float c : s.border_color) mix(bits(c));
    mix(static_cast<uint32_t>(s.border_kind));
    mix(bits(s.max_anisotropy));
    mix(static_cast<uint32_t>(s.reduction));
    return h;
}

VkSamplerAddressMode wrap_mode(int nvn_wrap) {
    const VkSamplerAddressMode m = vk_address_mode(nvn_wrap);
    if (m == VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE && !device().features().sampler_mirror_clamp_to_edge)
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    return m;
}

// Without custom border colours: the nearest of transparent black, opaque black and opaque white.
VkBorderColor standard_border(const SamplerInfo& s) {
    float c[4];
    for (int k = 0; k < 4; ++k) {
        if (s.border_kind == 0) {
            c[k] = s.border_color[k];
        } else {
            uint32_t u = 0;
            std::memcpy(&u, &s.border_color[k], 4);
            c[k] = s.border_kind == 1 ? static_cast<float>(static_cast<int32_t>(u)) : static_cast<float>(u);
        }
    }
    const float cand[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 1, 1}};
    int best = 0;
    float best_d = 1e30f;
    for (int i = 0; i < 3; ++i) {
        float d = 0;
        for (int k = 0; k < 4; ++k) d += (c[k] - cand[i][k]) * (c[k] - cand[i][k]);
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    if (s.border_kind == 0) {
        static const VkBorderColor f[3] = {VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK, VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE};
        return f[best];
    }
    static const VkBorderColor i[3] = {VK_BORDER_COLOR_INT_TRANSPARENT_BLACK, VK_BORDER_COLOR_INT_OPAQUE_BLACK, VK_BORDER_COLOR_INT_OPAQUE_WHITE};
    return i[best];
}

// Null when the device refuses the sampler (the caller then uses the dummy sampler).
VkSampler create_sampler(const SamplerInfo& s) {
    const gfx::DeviceFeatures& ft = device().features();
    const bool mips = min_filter_uses_mips(s.min_filter);
    VkSamplerCustomBorderColorCreateInfoEXT cbc = {VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
    VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter = vk_mag_filter(s.mag_filter);
    ci.minFilter = vk_min_filter(s.min_filter);
    ci.mipmapMode = mips ? vk_mipmap_mode(s.min_filter) : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = wrap_mode(s.wrap_s);
    ci.addressModeV = wrap_mode(s.wrap_t);
    ci.addressModeW = wrap_mode(s.wrap_r);
    ci.mipLodBias = s.lod_bias;
    ci.minLod = mips ? s.lod_min : 0.0f;
    ci.maxLod = mips ? s.lod_max : 0.0f;
    if (s.max_anisotropy > 1.0f && ft.sampler_anisotropy) {
        ci.anisotropyEnable = VK_TRUE;
        ci.maxAnisotropy = std::min(s.max_anisotropy, device().properties().limits.maxSamplerAnisotropy);
    }
    if (s.compare_mode == NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE) {
        ci.compareEnable = VK_TRUE;
        ci.compareOp = vk_compare_op(s.compare_func);
    }
    if (s.reduction != NVN_SAMPLER_REDUCTION_AVERAGE)
        note_unsupported("sampler reduction (min/max filtering): not enabled on the device, ignored");
    if (ft.custom_border_color) {
        // The border words are stored as the game gave them: float bits, or int / uint bits (border_kind).
        if (s.border_kind == 0) {
            ci.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
            std::memcpy(cbc.customBorderColor.float32, s.border_color, sizeof(float) * 4);
        } else {
            ci.borderColor = VK_BORDER_COLOR_INT_CUSTOM_EXT;
            std::memcpy(cbc.customBorderColor.uint32, s.border_color, sizeof(uint32_t) * 4);
        }
        cbc.format = VK_FORMAT_UNDEFINED;
        ci.pNext = &cbc;
    } else {
        ci.borderColor = standard_border(s);
    }
    VkSampler out = VK_NULL_HANDLE;
    if (vkCreateSampler(vkdev(), &ci, nullptr, &out) != VK_SUCCESS) {
        NXLOG_W(kTag, "vkCreateSampler failed: the dummy sampler is used");
        return VK_NULL_HANDLE;
    }
    vk_gauge_samplers().add(1);
    return out;
}

}  // namespace

namespace {
// ---- CPU writes without a flush --------------------------------------------------------------------------------------------
// A texture in a CPU_UNCACHED pool is coherent on the console: the game writes it and the GPU sees the bytes without FlushMappedRange.
// ACNH's text renderer rasterizes the glyphs it needs into such a texture (an R8 2048x4096 glyph cache) as new characters come up; an
// image uploaded once kept the old glyph set and every later new glyph drew as nothing. The storage pages of an uploaded coherent texture
// are therefore armed in one write-watch region over the heap (nxrt/write_watch.h), before the storage is read for the upload. At the start
// of every submission the pages the CPU wrote since are handed to textures_cpu_wrote, as a flush of them would be: the textures there are
// uploaded again before their next use (and armed again). A texture the GPU renders to is not armed. [nvn] tex_write_watch = false: off.
mem::ww::Region* g_tex_ww = nullptr;
bool g_tex_ww_tried = false;
std::vector<uint64_t> g_tex_ww_pages;

mem::ww::Region* tex_ww() {
    if (!g_tex_ww_tried) {
        g_tex_ww_tried = true;
        if (nxrt::config_get_bool("nvn", "tex_write_watch", true) && mem::heap_exists())
            g_tex_ww = mem::ww::region_create(mem::kHeapBase, mem::heap_reserved(), "texture heap");
    }
    return g_tex_ww;
}

bool tex_ww_wanted(const Texture& t, const TexGpu& g) { return t.cpu_coherent && t.mem && t.storage_size && !g.gpu_written && tex_ww(); }

void tex_ww_arm(const Texture& t, const TexGpu& g) {
    if (!tex_ww_wanted(t, g)) return;
    mem::ww::Region* r = tex_ww();
    const uint64_t a = reinterpret_cast<uint64_t>(t.mem), base = mem::ww::region_base(r);
    if (a < base) return;
    const uint64_t first = (a - base) >> 12, last = (a + t.storage_size - 1 - base) >> 12;
    mem::ww::arm(r, first, last - first + 1);   // false: a page it cannot protect (another view, a file read in progress): not watched
}

// ---- arming off the render thread ([nvn] tex_ww_async, default on) ---------------------------------------------------------------
// arm() is a VirtualQuery walk and a VirtualProtect under the region's exclusive lock. While the game streams an area those calls took the
// render thread 60 to 490 ms a frame (the hitch line's "write-watch arming", inside "uploads"). Here the render thread hashes the storage it
// uploads from and hands the arm to one worker thread. The worker arms first and hashes the storage again: a write after the arm faults and
// is reported as before (tex_poll_cpu_writes); a write between the upload's read and the arm shows as a different hash and is reported the
// same way (textures_cpu_wrote), so the texture is uploaded again. Nothing the CPU writes is missed; the only cost is that write's re-upload.
struct ArmJob {
    Ref<Texture> t;
    uint64_t first = 0, count = 0, hash = 0, seq = 0;
};
struct ArmQueue {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<ArmJob> q;
    std::unordered_map<Texture*, uint64_t> latest;     // newest job of a texture (older ones are skipped)
    uint64_t seq = 0;
    bool started = false;
};
ArmQueue* g_armq = nullptr;     // never destroyed: the worker runs until the process ends
std::atomic<uint64_t> g_arm_async_jobs{0}, g_arm_async_rewritten{0};
// the hitch log: arms the worker made, and uploads it found written before its arm (uploaded again), during a slow frame
const bool g_arm_hitch_registered =
    (nxrt::hitch_add_counter("tex_arms_async", []() -> uint64_t { return g_arm_async_jobs.load(std::memory_order_relaxed); }),
     nxrt::hitch_add_counter("tex_rewritten_before_arm", []() -> uint64_t { return g_arm_async_rewritten.load(std::memory_order_relaxed); }), true);

bool tex_ww_async_on() {
    static const bool on = nxrt::config_get_bool("nvn", "tex_ww_async", true);
    return on;
}

void arm_worker() {
    log_set_thread_name("nvn-ww-arm");
    ArmQueue& aq = *g_armq;
    std::unique_lock<std::mutex> lk(aq.mu);
    for (;;) {
        aq.cv.wait(lk, [&] { return !aq.q.empty(); });
        ArmJob j = std::move(aq.q.front());
        aq.q.pop_front();
        Texture* t = j.t.get();
        auto it = aq.latest.find(t);
        if (it == aq.latest.end() || it->second != j.seq) continue;   // a newer upload queued its own arm
        lk.unlock();
        if (!t->destroyed.load()) {
            mem::ww::arm(g_tex_ww, j.first, j.count);   // false: a page it cannot protect: not watched (as the synchronous arm)
            uint64_t h = 0;
            const bool same = texprep::hash_storage(*t, h) && h == j.hash;   // unreadable now: reported, as a write would be
            lk.lock();
            const bool newest = aq.latest.count(t) && aq.latest[t] == j.seq;
            if (newest) aq.latest.erase(t);
            lk.unlock();
            if (!same && newest) {      // written since the upload read it (or not readable): upload it again
                g_arm_async_rewritten.fetch_add(1, std::memory_order_relaxed);
                textures_cpu_wrote(t->mem, t->storage_size);
            }
        } else {
            lk.lock();
            if (aq.latest.count(t) && aq.latest[t] == j.seq) aq.latest.erase(t);
            lk.unlock();
        }
        g_arm_async_jobs.fetch_add(1, std::memory_order_relaxed);
        j.t = Ref<Texture>();     // the texture may be destroyed here
        lk.lock();
    }
}

// Render thread, after the storage was read for an upload of content `hash` (have_hash false: unknown). Arms the pages on the worker.
void tex_ww_arm_later(Texture* t, const TexGpu& g, bool have_hash, uint64_t hash) {
    if (!tex_ww_wanted(*t, g)) return;
    if (!have_hash) { tex_ww_arm(*t, g); return; }      // nothing to compare with after the arm (storage not readable): as before
    mem::ww::Region* r = tex_ww();
    const uint64_t a = reinterpret_cast<uint64_t>(t->mem), base = mem::ww::region_base(r);
    if (a < base) return;
    if (!g_armq) g_armq = new ArmQueue();
    ArmQueue& aq = *g_armq;
    std::lock_guard<std::mutex> lk(aq.mu);
    if (!aq.started) {
        aq.started = true;
        std::thread(arm_worker).detach();
    }
    ArmJob j;
    j.t = Ref<Texture>::share(t);
    j.first = (a - base) >> 12;
    j.count = ((a + t->storage_size - 1 - base) >> 12) - j.first + 1;
    j.hash = hash;
    j.seq = ++aq.seq;
    aq.latest[t] = j.seq;
    aq.q.push_back(std::move(j));
    aq.cv.notify_one();
}

// Before a GPU copy into level `level` of layer `layer` of a sparse texture: binds the host blocks of the guest tiles the game mapped
// there (a copy lands only on mapped tiles; on unbound blocks it is dropped, as on the console's unmapped pages).
void sparse_bind_for_copy(Texture* t, TexGpu& g, uint32_t layer, uint32_t level) {
    SparseState& sp = *g.sparse;
    const TexLayout& L = sp.L;
    if (layer >= g.layers || level >= g.levels || !sparse_level_chunked(sp, level)) return;   // other levels are bound at creation
    std::vector<const uint8_t*> ptrs;
    texture_virtual_tile_ptrs(*t, ptrs);
    const tegra::LevelLayout& lv = L.layout.levels[level];
    const uint32_t bw = L.desc.block_w, bh = L.desc.block_h, bpb = L.desc.bytes_per_block;
    const uint64_t base = static_cast<uint64_t>(layer) * L.layout.tiled_layer_stride + lv.tiled_offset;
    const uint64_t block_bytes = 512ull * lv.gob_blocks_y, per = kGuestTile / block_bytes;
    const uint64_t gob_cols = static_cast<uint64_t>(lv.blocks_x) * bpb / 64;
    const uint32_t nbx = static_cast<uint32_t>(per * 64 / bpb), nby = 8u * lv.gob_blocks_y;
    const uint32_t lw = std::max(1u, g.base.width >> level), lh = std::max(1u, g.base.height >> level);
    uint32_t mapped = 0;
    for (uint64_t off = base; off < base + lv.tiled_size; off += kGuestTile) {
        const size_t ti = static_cast<size_t>(off / kGuestTile);
        if (ti >= ptrs.size() || !ptrs[ti]) continue;
        const uint64_t first = (off - base) / block_bytes;
        const uint32_t x0 = static_cast<uint32_t>((first % gob_cols) * 64 / bpb) * bw, y0 = static_cast<uint32_t>(first / gob_cols) * nby * bh;
        if (x0 >= lw || y0 >= lh) continue;
        sparse_bind_rect(g, layer, level, x0, y0, std::min(nbx * bw, lw - x0), std::min(nby * bh, lh - y0));
        ++mapped;
    }
    sparse_flush_binds(g);
    static int logged = 0;
    if (logged < 12) {
        ++logged;
        NXLOG_I(kTag, "texture %u: copy into layer %u level %u: %u mapped tiles of %llu bound (%zu tile pointers); %llu MB bound", t->id, layer, level, mapped,
                static_cast<unsigned long long>(lv.tiled_size / kGuestTile), ptrs.size(), static_cast<unsigned long long>(sp.bytes >> 20));
    }
}

}  // namespace

// ---- texture interface (gpu::*, internal.h) -----------------------------------------------------------------------------
TexGpu* tex_gpu(Texture* t) { return tex_or_null(t); }

TexGpu* tex_gpu_attachment(Texture* t) {
    if (!t || t->gpu || !active()) return tex_or_null(t);
    if (texture_registry_find_alias(t)) return tex_or_null(t);   // shares the image (and its scale) of the texture it aliases
    return tex_or_null(t, attachment_scale(*t));
}

TexGpu* tex_gpu_scaled(Texture* t, float scale) { return tex_or_null(t, scale); }

float render_scale() { return render_scale_now(); }

float tex_scale(Texture* t) {
    return (t && t->gpu && !t->gpu->failed) ? t->gpu->scale : 1.0f;
}

VkExtent3D tex_image_extent(Texture* t, uint32_t level) {
    TexGpu* g = tex_or_null(t);
    return g ? level_extent(g->base, level > g->drop ? level - g->drop : 0) : tex_extent(t, level);
}

uint32_t tex_level_drop(Texture* t) { return (t && t->gpu && !t->gpu->failed) ? t->gpu->drop : 0; }

float tex_scale_wanted(Texture* t) {
    TexGpu* g = tex_or_null(t);
    if (!g || g->scale_blocked || g->transcoded || is_block_format(g->format)) return g ? g->scale : 1.0f;
    return attachment_scale(*t);
}

// Remakes the image at another scale (Ryujinx scales a texture when it becomes a render target and blacklists one that cannot be
// scaled). The content is blitted over when the format allows it (otherwise it starts undefined). Going to 1 is for good. The old image
// and its views are destroyed after the frames that use them.
bool tex_rescale(Texture* t, Frame& f, float to, const char* why) {
    TexGpu* g = tex_or_null(t);
    if (!g || g->scale == to) return false;
    if (to != 1.0f && (g->scale_blocked || g->transcoded || is_block_format(g->format))) return false;
    Shape s = shape_of(*t);
    if (to != 1.0f) {
        s.base.width = static_cast<uint32_t>(std::lround(s.base.width * to));
        s.base.height = static_cast<uint32_t>(std::lround(s.base.height * to));
    }
    gfx::Image ni;
    std::string err;
    if (device().create_image(make_desc(s, g->format, g->usage, base_flags(s, g->format)), &ni, &err) != VK_SUCCESS) {
        NXLOG_W(kTag, "texture %u: cannot remake it at %ux%u (%s): it keeps scale %.3f", t->id, s.base.width, s.base.height, err.c_str(), g->scale);
        if (to == 1.0f) g->scale_blocked = true;
        return false;
    }
    static int logged = 0;
    if (logged++ < 60)
        NXLOG_I(kTag, "texture %u: %s %dx%d now rendered at %ux%u (%s)", t->id, g->fmt ? g->fmt->name : "?", t->b.width, t->b.height, s.base.width, s.base.height, why);
    VkFormatProperties fp = {};
    vkGetPhysicalDeviceFormatProperties(device().physical(), g->format, &fp);
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    const bool can_blit = g->layout != VK_IMAGE_LAYOUT_UNDEFINED && s.samples <= 1 && (fp.optimalTilingFeatures & need) == need &&
                          (g->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) && (g->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    VkImageLayout nl = VK_IMAGE_LAYOUT_UNDEFINED;
    if (can_blit) {
        move_layout(f, *g, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        image_barrier(f.cb, ni.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, g->aspect, 0, s.levels, 0, s.layers);
        std::vector<VkImageBlit> regions;
        for (uint32_t l = 0; l < s.levels; ++l) {
            const VkExtent3D a = level_extent(g->base, l), b = level_extent(s.base, l);
            VkImageBlit r = {};
            r.srcSubresource = {g->aspect, l, 0, s.layers};
            r.dstSubresource = {g->aspect, l, 0, s.layers};
            r.srcOffsets[1] = {static_cast<int32_t>(a.width), static_cast<int32_t>(a.height), 1};
            r.dstOffsets[1] = {static_cast<int32_t>(b.width), static_cast<int32_t>(b.height), 1};
            regions.push_back(r);
        }
        vkCmdBlitImage(f.cb, g->img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ni.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<uint32_t>(regions.size()),
                       regions.data(), g->depth ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
        nl = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    }
    gfx::Image old = g->img;
    // The renderer samples window images in its own submission. Finishing the
    // NVN serial alone does not retire those reads (notably boot's window swap).
    if (g->presented) device().wait_idle();
    g->presented = false;
    std::vector<VkImageView> views;
    for (auto& kv : g->views) views.push_back(kv.second);
    g->views.clear();
    g->last_view = VK_NULL_HANDLE;
    defer_destroy([old, views]() mutable {
        for (VkImageView v : views) { vkDestroyImageView(vkdev(), v, nullptr); vk_gauge_image_views().sub(1); }
        device().destroy_image(old);
    });
    g->img = ni;
    g->base = s.base;
    g->layout = nl;
    g->scale = to;
    if (to == 1.0f) g->scale_blocked = true;
    pool_texture_remade(t);
    return true;
}

void tex_poll_cpu_writes() {
    if (!g_tex_ww) return;
    g_tex_ww_pages.clear();
    mem::ww::take_written(g_tex_ww, g_tex_ww_pages);
    if (g_tex_ww_pages.empty()) return;
    std::sort(g_tex_ww_pages.begin(), g_tex_ww_pages.end());
    const uint64_t base = mem::ww::region_base(g_tex_ww);
    for (size_t i = 0; i < g_tex_ww_pages.size();) {
        size_t j = i + 1;
        while (j < g_tex_ww_pages.size() && g_tex_ww_pages[j] <= g_tex_ww_pages[j - 1] + 1) ++j;   // a run of written pages: one call
        textures_cpu_wrote(reinterpret_cast<const void*>(base + (g_tex_ww_pages[i] << 12)), (g_tex_ww_pages[j - 1] - g_tex_ww_pages[i] + 1) << 12);
        i = j;
    }
}

// ---- GPU untile ([nvn] gpu_untile, default on) ----------------------------------------------------------------------------------
// A block-linear texture the CPU rewrites every frame (Pokopia: three 1024x1024 RGBA8 textures in turn) cost the render thread ~3 ms
// per upload: untile 4 MiB on the CPU, copy it into staging, copy that to the image. Instead a compute shader untiles straight from
// the pool (the game's memory, imported) into a device-local linear buffer, which is copied to the image: no CPU work. Like a sampler
// on the console, it reads the storage when the GPU runs the command. One level, one layer, 2D, block depth 1, rows of whole 16-byte
// pieces, the image in the storage's own format, 64 KiB and more; anything else takes the CPU path.
namespace {
#include "nvn_untile_shader.inc"
struct UntilePush { uint32_t base, ppr, total, gobs_x, ys, sw2; };
struct Untiler {
    bool tried = false, ok = false;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    static constexpr uint32_t kRing = 16;
    struct Entry { VkDescriptorSet set = VK_NULL_HANDLE; gfx::Buffer scratch; uint64_t serial = 0; };
    Entry ring[kRing];
    uint32_t next = 0;
    VkDeviceSize ssbo_align = 256;
};
Untiler g_untile;

bool untiler_init() {
    Untiler& u = g_untile;
    if (u.tried) return u.ok;
    u.tried = true;
    VkDevice d = vkdev();
    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(device().physical(), &props);
    u.ssbo_align = std::max<VkDeviceSize>(16, props.limits.minStorageBufferOffsetAlignment);
    const VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                               {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo sli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 2;
    sli.pBindings = b;
    if (vkCreateDescriptorSetLayout(d, &sli, nullptr, &u.set_layout) != VK_SUCCESS) return false;
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UntilePush)};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &u.set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(d, &pli, nullptr, &u.layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = sizeof kUntileComp;
    mi.pCode = kUntileComp;
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(d, &mi, nullptr, &m) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
    ci.layout = u.layout;
    const VkResult pr = vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &ci, nullptr, &u.pipe);
    vkDestroyShaderModule(d, m, nullptr);
    if (pr != VK_SUCCESS) return false;
    const VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * Untiler::kRing};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = Untiler::kRing;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(d, &dpi, nullptr, &u.pool) != VK_SUCCESS) return false;
    for (Untiler::Entry& e : u.ring) {
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = u.pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &u.set_layout;
        if (vkAllocateDescriptorSets(d, &ai, &e.set) != VK_SUCCESS) return false;
    }
    u.ok = true;
    NXLOG_I(kTag, "GPU untile on ([nvn] gpu_untile)");
    return true;
}

// A pitch-linear texture whose rows are whole texels: the copy to the image reads the pool directly (no CPU copy, no staging).
bool gpu_pitch_copy(Texture* t, TexGpu& g, Frame& f, const TexLayout& L, bool may_import) {
    const uint32_t bpb = L.fmt->bytes;
    if (!bpb || L.pitch % bpb || L.fmt->block_w != 1 || L.fmt->block_h != 1 || g.base.depth != 1) return false;
    const uint64_t bytes = L.pitch * g.base.height;
    if (bytes > t->storage_size) return false;
    const BufRef br = may_import ? resolve_buffer(reinterpret_cast<uint64_t>(t->mem), bytes) : resolve_buffer_imported(reinterpret_cast<uint64_t>(t->mem), bytes);
    if (!br.ok || br.offset % std::max<uint32_t>(4, bpb)) return false;
    memory_barrier(f.cb);                          // earlier GPU writes to the storage
    VkBufferImageCopy r = {};
    r.bufferOffset = br.offset;
    r.bufferRowLength = static_cast<uint32_t>(L.pitch / bpb);
    r.bufferImageHeight = g.base.height;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {g.base.width, g.base.height, 1};
    copy_regions_to_image(f, g, br.buffer, {r});
    return true;
}

// True when the image now gets the storage's content from a GPU untile (or pitch copy) recorded into f.
bool gpu_untile_impl(Texture* t, TexGpu& g, Frame& f, const char** why, bool may_import);
// may_import: the pool part holding the storage may be imported for it now (a streamed texture, read by the GPU every frame). Otherwise only
// an import that already exists is used ([nvn] untile_import = false, default): a first upload untiled on the CPU costs well under a
// millisecond, the import of the 64 MiB chunk around it 16 to 90 ms on the render thread (s1-stutter1, a 246 ms frame).
bool gpu_untile(Texture* t, TexGpu& g, Frame& f, bool may_import = true) {
    static const bool untile_import = nxrt::config_get_bool("nvn", "untile_import", false);
    const char* why = "";
    if (gpu_untile_impl(t, g, f, &why, may_import || untile_import)) return true;
    static int logged = 0;
    if (t->storage_size >= (1u << 20) && logged < 12) {
        ++logged;
        NXLOG_I(kTag, "texture %u: GPU untile not used (%s): CPU upload", t->id, why);
    }
    return false;
}

bool gpu_untile_impl(Texture* t, TexGpu& g, Frame& f, const char** why, bool may_import) {
    static const bool on = nxrt::config_get_bool("nvn", "gpu_untile", true);
    if (!on) return *why = "off", false;
    if (!t->mem || t->vpool || t->levels != 1 || t->layers != 1 || t->storage_size < (64u << 10)) return *why = "storage, levels or layers", false;
    if (g.levels != 1 || g.layers != 1 || g.transcoded || !g.fmt || g.format != g.fmt->vk || !(g.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        return *why = "image format or usage", false;
    const TexLayout L = compute_texture_layout(t->b);
    if (!L.valid || !L.fmt || L.fmt->astc || L.levels != 1 || L.layers != 1) return *why = "layout", false;
    if (!L.block_linear) return gpu_pitch_copy(t, g, f, L, may_import) || (*why = "pitch", false);
    if (!t->block_linear || L.layout.levels.size() != 1) return *why = "layout levels", false;
    const tegra::LevelLayout& lv = L.layout.levels[0];
    const uint64_t row_bytes = static_cast<uint64_t>(lv.blocks_x) * L.desc.bytes_per_block;
    if (lv.depth != 1 || lv.gob_blocks_z != 1 || row_bytes % 16 || lv.tiled_offset != 0) return *why = "3D or row size", false;
    const uint64_t lin = row_bytes * lv.blocks_y;
    const uint64_t total = lin / 16;
    if (total == 0 || total > 64ull * 65535) return false;
    const uint32_t gobs_x = static_cast<uint32_t>((row_bytes + 63) / 64);
    uint32_t ys = 0;
    while ((1u << ys) < lv.gob_blocks_y) ++ys;
    // the GOBs of every block row, whole blocks: what the shader may read
    const uint64_t tiled = static_cast<uint64_t>((lv.blocks_y + 8 * lv.gob_blocks_y - 1) / (8 * lv.gob_blocks_y)) * gobs_x * lv.gob_blocks_y * 512;
    if (tiled > t->storage_size) return *why = "storage size", false;
    const BufRef br = may_import ? resolve_buffer(reinterpret_cast<uint64_t>(t->mem), tiled) : resolve_buffer_imported(reinterpret_cast<uint64_t>(t->mem), tiled);
    if (!br.ok) return *why = may_import ? "not in a pool" : "pool part not imported", false;
    if (!untiler_init()) return false;
    Untiler& u = g_untile;
    const VkDeviceSize boff = br.offset & ~(u.ssbo_align - 1);
    const VkDeviceSize delta = br.offset - boff;
    if (delta % 16) return false;
    Untiler::Entry& e = u.ring[u.next];
    if (e.serial && completed_serial() < e.serial) return false;     // its set and scratch may still be in use
    u.next = (u.next + 1) % Untiler::kRing;
    if (!e.scratch || e.scratch.size < lin) {
        if (e.scratch) {
            gfx::Buffer old = e.scratch;
            defer_destroy([old]() mutable { device().destroy_buffer(old); });
            e.scratch = gfx::Buffer{};
        }
        if (device().create_buffer(lin, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, gfx::MemoryUse::GpuOnly, &e.scratch, "nvn untile") != VK_SUCCESS) {
            e.scratch = gfx::Buffer{};
            return false;
        }
    }
    const VkDescriptorBufferInfo bi[2] = {{br.buffer, boff, delta + tiled}, {e.scratch.buffer, 0, lin}};
    VkWriteDescriptorSet w[2] = {};
    for (int i = 0; i < 2; ++i) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = e.set;
        w[i].dstBinding = static_cast<uint32_t>(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(vkdev(), 2, w, 0, nullptr);
    const UntilePush pc = {static_cast<uint32_t>(delta / 16), static_cast<uint32_t>(row_bytes / 16), static_cast<uint32_t>(total), gobs_x, ys,
                           L.desc.gob_switch2 ? 1u : 0u};
    memory_barrier(f.cb);                          // earlier writes to the storage, and earlier reads of the scratch
    vkCmdBindPipeline(f.cb, VK_PIPELINE_BIND_POINT_COMPUTE, u.pipe);
    vkCmdBindDescriptorSets(f.cb, VK_PIPELINE_BIND_POINT_COMPUTE, u.layout, 0, 1, &e.set, 0, nullptr);
    vkCmdPushConstants(f.cb, u.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(f.cb, static_cast<uint32_t>((total + 63) / 64), 1, 1);
    memory_barrier(f.cb);
    VkBufferImageCopy r = {};
    r.bufferRowLength = lv.blocks_x * L.desc.block_w;
    r.bufferImageHeight = lv.blocks_y * L.desc.block_h;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {g.base.width, g.base.height, 1};
    copy_regions_to_image(f, g, e.scratch.buffer, {r});
    e.serial = f.serial;
    return true;
}
}  // namespace

// The caller copies or clears the image, so it needs the real content now: a stand-in would end up in the copy ([nvn] tex_async).
void tex_ensure_exact(Texture* t, Frame& f) {
    const bool was = texprep::set_exact_thread(true);
    // the image holds a stand-in of the current content: the real data goes in first (not over what the GPU rendered into it since)
    if (TexGpu* g = tex_or_null(t)) {
        if (g->uploaded && !g->gpu_written && t->prep_prov.load(std::memory_order_acquire) == (t->prep_gen.load(std::memory_order_acquire) | 0x80000000u))
            t->cpu_dirty.store(true);
    }
    tex_ensure_current(t, f);
    texprep::set_exact_thread(was);
}

void tex_ensure_current(Texture* t, Frame& f) {
    NXFT_MIN("tex.ensure", 30000);
    TexGpu* g = tex_or_null(t);
    if (!g || g->depth) return;
    if (t->b.samples > 1 || target_is_multisample(t->b.target)) return;   // multisampled images are only ever rendered
    if (t->overflow) {   // host-backed storage (nvn_core.h): no bytes in the game's memory; the image starts cleared, then only the GPU writes it
        t->cpu_dirty.store(false);
        if (!g->uploaded && !g->gpu_written) fill_zero(f, *g);
        g->uploaded = true;
        return;
    }
    // A coherent texture the CPU rewrites every frame (Pokopia: three 1024x1024 RGBA8 textures in turn) cost one write-watch fault per 4 KiB
    // page per rewrite (~3500 a second on the main thread, inside the game's writes) because each upload armed its pages again. Once its CPU
    // uploads come kStreamGap presents apart or closer kStreamAfter times in a row, and the GPU untile path takes it, it is "streamed": never
    // armed again, and read from the pool once per present at its first use (a compute untile; the bytes the CPU wrote that frame, as a
    // console GPU sampling the memory would see them). [nvn] tex_stream = false: off.
    constexpr uint32_t kStreamGap = 4, kStreamAfter = 8;   // gap 4: Pokopia rewrites its three in turn (every third present each)
    static const bool stream_on = nxrt::config_get_bool("nvn", "tex_stream", true);
    const uint32_t now = static_cast<uint32_t>(counters().presents.load(std::memory_order_relaxed));
    if (g->streamed) {
        if (g->streamed_at == now && g->uploaded) return;
        t->cpu_dirty.exchange(false);
        if (gpu_untile(t, *g, f)) {
            g->uploaded = true;
            g->gpu_written = false;
            g->streamed_at = now;
            return;
        }
        g->streamed = false;                     // the untile path no longer takes it: back to write-watched uploads below
        t->cpu_dirty.store(true);
    }
    const bool dirty = t->cpu_dirty.load();
    if (g->uploaded && !dirty) return;           // the image already holds the storage's content
    nxrt::HitchTimer ht_upload(nxrt::kHpRenderUpload);   // the hitch line: an upload (or its GPU untile) on the render thread
    if (dirty && g->uploaded && stream_on && t->cpu_coherent) {   // an upload for a CPU write (not the first one)
        g->cpu_streak = now - g->cpu_upload_at <= kStreamGap ? g->cpu_streak + 1 : 1;
        g->cpu_upload_at = now;
        if (g->cpu_streak >= kStreamAfter && g->untiled) {
            g->streamed = true;
            g->streamed_at = now;
            NXLOG_I(kTag, "texture %u (%ux%u, %llu KiB): rewritten by the CPU every frame: read once per present, no longer write-watched", t->id,
                    g->base.width, g->base.height, static_cast<unsigned long long>(t->storage_size >> 10));
            t->cpu_dirty.exchange(false);
            if (gpu_untile(t, *g, f)) { g->uploaded = true; g->gpu_written = false; return; }
            g->streamed = false;
            t->cpu_dirty.store(true);
        }
    }
    if (g->gpu_written && !dirty) return;        // the GPU rendered it and the CPU did not write since
    if (g->sparse) {                             // a sparse-residency image: the dirty tiles only
        t->cpu_dirty.exchange(false);
        sparse_update(t, *g, f);
        return;
    }
    if (g->scale != 1.0f && dirty && (g->uploaded || g->gpu_written)) {
        static int logged = 0;
        if (logged++ < 20) NXLOG_W(kTag, "texture %u: the CPU wrote a scaled render target: the write is not uploaded", t->id);
    }
    // A float16 render target of the HDR scene, or a scaled one: the NVN bytes in memory do not fit the image. It starts cleared.
    if (g->promoted || g->scale != 1.0f) {
        t->cpu_dirty.exchange(false);
        if (!g->uploaded) fill_zero(f, *g);
        g->uploaded = true;
        return;
    }
    t->cpu_dirty.exchange(false);
    // before the storage is read: a write from now on makes it dirty again. Off the render thread (tex_ww_arm_later): the content hash of what
    // the upload reads is taken here instead, and the worker compares it after arming.
    const bool arm_later = tex_ww_async_on() && tex_ww_wanted(*t, *g);
    if (!arm_later) tex_ww_arm(*t, *g);
    uint64_t content_hash = 0;
    bool have_hash = false;
    if (g->gpu_written) {
        static int logged = 0;
        if (logged++ < 20) NXLOG_W(kTag, "texture %u: the GPU wrote it, the game flushed memory it overlaps: its storage replaces the image content", t->id);
    }

    g->untiled = false;
    // the untile reads the pool when the GPU runs it: hashed now, armed after (only when the untile can take it: the image in the storage's format)
    if (arm_later && t->levels == 1 && t->layers == 1 && t->storage_size >= (64u << 10) && !g->transcoded && g->fmt && g->format == g->fmt->vk)
        have_hash = texprep::hash_storage(*t, content_hash);
    if (gpu_untile(t, *g, f, false)) {
        g_rt_gpu_untiles.fetch_add(1, std::memory_order_relaxed);
        if (arm_later) tex_ww_arm_later(t, *g, have_hash, content_hash);
        g->untiled = true;
        ft::mark("tex.untile", t->id);
        be::tx_mark("tx.use", t);
        g->uploaded = true;
        g->gpu_written = false;
        texprep::note_uploaded(t);
        counters().texture_uploads.fetch_add(1);
        return;
    }

    const gfx::DeviceFeatures& ft = device().features();
    TexCaps caps;
    caps.bc = ft.bc;
    caps.astc = ft.astc_ldr;
    caps.bc7_transcode = true;
    caps.upgrade = true;
    caps.image_format = g->format;
    // Reused from upload to upload (render thread): prepare_texture writes into the data buffer it already has, so a first visit's hundreds
    // of megabytes of untiles and stand-ins stop paging in fresh memory per texture (page faults and zeroing on the render thread). A buffer
    // grown past 64 MiB is given back.
    static thread_local PreparedTexture tl_prep;
    if (tl_prep.data.capacity() > (64ull << 20)) std::vector<uint8_t>().swap(tl_prep.data);
    PreparedTexture& prep = tl_prep;
    prep.provisional = false;
    prep.compact = false;
    bool ok = false;
    bool from_job = false;
    const uint64_t t_prep0 = ft::now_ns();
    gfx::Buffer staged;                        // a big background result comes with its upload buffer already filled
    uint64_t take_hash = 0;
    bool took = false;
    {
        nxrt::HitchTimer ht_take(nxrt::kHpRenderTexTake);
        took = texprep::take(t, prep, &staged, &take_hash);
    }
    if (took) {     // prepared in the background from the current storage content (nvn_texprep.h)
        if (prep.format == g->format) {
            content_hash = take_hash;
            have_hash = true;
            ok = from_job = true;
        } else {                               // another image format than the one it was made for (a fallback format): prepare it again
            prep = PreparedTexture();
            if (staged) device().destroy_buffer(staged);
            staged = gfx::Buffer{};
        }
    }
    // The image already holds a stand-in of this content and its job has no result yet (still queued or running, or its result was lost): the
    // stand-in stays and the job goes (again) to the front of the queue; its end marks the texture dirty. Before, this prepared the texture here
    // (a cache read or a transcode on the render thread, 3.4 s of them in one frame of a 4K first visit). Not for a copy or a clear (exact).
    // [nvn] standin_keep = false: prepared here as before.
    static const bool standin_keep = nxrt::config_get_bool("nvn", "standin_keep", true);
    if (!ok && standin_keep && g->uploaded && !g->gpu_written && !texprep::exact_thread() && texprep::standin_on_screen(t)) {
        texprep::standin_kept(t);
        return;
    }
    if (!ok) {
        static const uint16_t zid = ft::zone_id("tex.prepare");
        ft::Scope zs(zid, 30000, t->id);
        if (arm_later && !have_hash) have_hash = texprep::hash_storage(*t, content_hash);   // before the read the upload uses
        // ASTC without a cache entry, needed now: a quick stand-in (preset 0) instead of the full transcode; the background job makes the real data
        caps.provisional = texprep::wants_provisional(t);
        // [nvn] tex_async = standin: no cache read or transcode here, also when an entry exists (on a hard disk each read stalled the frame for
        // 30 ms to over a second, thousands of them on a first visit). Not for a copy or a clear, which would keep the stand-in's pixels.
        // A sparse texture written tile by tile is only given one at its first upload: later uploads follow its writes and would flicker.
        caps.standin_first = caps.provisional && !texprep::exact_thread() && (!t->vpool || !g->uploaded) && texprep::async_standin();
        // [nvn] standin_rows (default on): the stand-in's big solid levels come as their distinct block rows, copied into the image row by row
        static const bool standin_rows = nxrt::config_get_bool("nvn", "standin_rows", true);
        caps.compact_standin = caps.standin_first && standin_rows && !g->sparse;
        nxrt::HitchTimer ht_prep(nxrt::kHpRenderTexPrep);
        ok = prepare_texture(*t, caps, prep);
        g_rt_prep_bytes.fetch_add(prep.data.size(), std::memory_order_relaxed);
        if (prep.provisional) g_rt_standin_bytes.fetch_add(prep.data.size(), std::memory_order_relaxed);
        else if (ok && !prep.transcoded) g_rt_untile_bytes.fetch_add(prep.data.size(), std::memory_order_relaxed);   // BCn / uncompressed: a CPU untile
    }
    be::tx_mark("tx.use", t);
    {
        static const bool log_uploads = nxrt::config_get_bool("nvn", "tex_prep_log", false);      // [nvn] tex_prep_log = 1: one line per upload with where the data came from
        if (log_uploads)
            NXLOG_I(kTag, "upload t%u %s %ux%ux%u L%u storage %llu KB%s sched 0x%x gen %u: %s %.2f ms", t->id, t->fmt ? t->fmt->name : "?", t->b.width, t->b.height, t->b.depth, t->levels,
                    static_cast<unsigned long long>(t->storage_size >> 10), t->vpool ? " sparse" : "", static_cast<unsigned>(t->prep_sched.load()), static_cast<unsigned>(t->prep_gen.load()),
                    from_job ? "background" : "render thread", static_cast<double>(ft::now_ns() - t_prep0) * 1e-6);
    }
    if (!ok || prep.format != g->format || !(g->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        NXLOG_E(kTag, "texture %u: not uploaded (%s): the image is cleared", t->id,
                !ok ? prep.error.c_str() : "the image format differs from the storage format or the image cannot be written");
        if (staged) device().destroy_buffer(staged);
        if (arm_later) tex_ww_arm_later(t, *g, have_hash, content_hash);   // as the synchronous arm: a later write tries again
        fill_zero(f, *g);
        g->uploaded = true;
        g->gpu_written = false;
        return;
    }

    {
        uint64_t w0 = 0;
        if (!prep.data.empty()) std::memcpy(&w0, prep.data.data(), std::min<size_t>(8, prep.data.size()));
        tev(t, "upload", w0, dirty ? 1 : 0);
    }
    const uint64_t bytes = staged ? staged.size : prep.data.size();
    const uint64_t t_stage0 = nxrt::hitch_ticks();
    UploadAlloc st = staged ? f.adopt_staging(staged) : f.staging(bytes);
    if (!st) {
        t->cpu_dirty.store(true);                // try again next time
        return;
    }
    if (!staged) std::memcpy(st.ptr, prep.data.data(), static_cast<size_t>(bytes));
    nxrt::hitch_phase_add(nxrt::kHpRenderTexStaging, nxrt::hitch_ticks() - t_stage0);
    std::vector<VkBufferImageCopy> regions;
    regions.reserve(prep.level.size());
    for (uint32_t i = g->drop; i < prep.level.size() && i - g->drop < g->levels; ++i) {   // the levels the image has ([gfx] texture_detail)
        const PreparedLevel& pl = prep.level[i];
        if (!pl.rows.empty()) {                  // a compact stand-in level: each block row from its (shared) row of blocks
            const uint32_t bh = (pl.height + 3) / 4;
            for (uint32_t layer = 0; layer < g->layers && static_cast<size_t>(layer + 1) * bh <= pl.rows.size(); ++layer)
                for (uint32_t by = 0; by < bh; ++by) {
                    VkBufferImageCopy r = {};
                    r.bufferOffset = st.offset + pl.rows[static_cast<size_t>(layer) * bh + by];
                    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - g->drop, layer, 1};
                    r.imageOffset = {0, static_cast<int32_t>(by * 4), 0};
                    r.imageExtent = {pl.width, std::min<uint32_t>(4, pl.height - by * 4), 1};
                    regions.push_back(r);
                }
            continue;
        }
        VkBufferImageCopy r = {};
        r.bufferOffset = st.offset + pl.offset;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - g->drop, 0, g->layers};
        r.imageExtent = {pl.width, pl.height, pl.depth};
        regions.push_back(r);
    }
    copy_regions_to_image(f, *g, st.buffer, regions);
    if (arm_later) tex_ww_arm_later(t, *g, have_hash, content_hash);
    g->uploaded = true;
    g->gpu_written = false;
    texprep::note_uploaded(t);
    if (prep.provisional) texprep::provisional_uploaded(t);
    else texprep::final_uploaded(t);
    counters().texture_uploads.fetch_add(1);
    counters().texture_upload_bytes.fetch_add(bytes);
    if (prep.transcoded) counters().bc7_transcodes.fetch_add(1);
}

void tex_transition(Texture* t, Frame& f, VkImageLayout layout) {
    TexGpu* g = tex_or_null(t);
    if (g && g->layout != layout) pool_note_layout(t, layout);
    if (g) move_layout(f, *g, layout);
}

VkImageLayout tex_layout(Texture* t) {
    return (t && t->gpu && !t->gpu->failed) ? t->gpu->layout : VK_IMAGE_LAYOUT_UNDEFINED;
}

void tex_sparse_before_copy(Texture* t, uint32_t layer, uint32_t level) {
    TexGpu* g = tex_or_null(t);
    if (g && g->sparse) sparse_bind_for_copy(t, *g, layer, level);
}

void tex_mark_gpu_written(Texture* t) {
    TexGpu* g = tex_or_null(t);
    if (!g) return;
    g->gpu_written = true;
    t->cpu_dirty.store(false);                   // the GPU data is the newer one until the CPU writes again
}

VkImage tex_image(Texture* t) {
    TexGpu* g = tex_or_null(t);
    return g ? g->img.image : VK_NULL_HANDLE;
}

VkFormat tex_vk_format(Texture* t) {
    TexGpu* g = tex_or_null(t);
    return g ? g->format : VK_FORMAT_UNDEFINED;
}

VkImageAspectFlags tex_aspect(Texture* t) {
    TexGpu* g = tex_or_null(t);
    return g ? g->aspect : VK_IMAGE_ASPECT_COLOR_BIT;
}

VkExtent3D tex_extent(Texture* t, uint32_t level) { return level_extent(shape_of(*t).base, level); }

uint32_t tex_levels(Texture* t) { return t->levels; }

uint32_t tex_layers(Texture* t) { return t->layers; }

bool tex_is_depth(Texture* t) { return t->fmt && t->fmt->aspect != kAspectColor; }

bool tex_first_upload_deferrable(Texture* t) {
    if (!t || !t->fmt || !(t->fmt->astc || t->fmt->bc) || !(t->mem || t->vpool) || !t->cpu_dirty.load(std::memory_order_relaxed)) return false;
    const TexGpu* g = t->gpu;
    if (!g) {
        // no image yet ([nvn] pool_image_lazy): it is made when the upload's turn comes, not at the pool scan (a load-in made 3093 images in one
        // frame, 151 ms). Not when another texture's image over the same storage would be shared, and not for a texture already finalized.
        static const bool lazy = nxrt::config_get_bool("nvn", "pool_image_lazy", true);
        return lazy && active() && !t->destroyed.load(std::memory_order_relaxed) && t->fmt->aspect == kAspectColor && t->b.samples <= 1 && !texture_registry_find_alias(t);
    }
    return !g->failed && !g->uploaded && !g->gpu_written && !g->depth && !g->promoted && g->scale == 1.0f;
}

bool tex_standin_swap_pending(Texture* t) {
    const TexGpu* g = t ? t->gpu : nullptr;
    return g && !g->failed && g->uploaded && !g->gpu_written && t->cpu_dirty.load(std::memory_order_relaxed) && texprep::standin_on_screen(t);
}

bool tex_pool_preparation_pending(Texture* t) {
    const TexGpu* g = t ? t->gpu : nullptr;
    if (!g || g->failed) return true; // caller normally initializes/checks the image first
    return pool_preparation_required(t->cpu_dirty.load(std::memory_order_relaxed), g->depth,
                                     g->layout == tex_sampled_layout(t));
}

VkImageView tex_sampled_view(Texture* t, const ViewRequest& req) {
    TexGpu* g = tex_or_null(t);
    if (!g) return VK_NULL_HANDLE;
    g->sampled_at.store(static_cast<uint32_t>(counters().presents.load(std::memory_order_relaxed)) + 1, std::memory_order_relaxed);
    const bool is3d = g->type == VK_IMAGE_TYPE_3D;
    const uint32_t total_layers = is3d ? g->base.depth : g->layers;   // 3D images: the slices are the layers of 2D views
    // NVN levels to image levels: an image stored without its top level ([gfx] texture_detail) answers a view that starts there from
    // its own first level, with as many levels as are left of the range
    uint32_t bl = req.base_level, want_nl = req.level_count;
    if (g->drop) {
        const uint32_t end = want_nl ? bl + want_nl : 0;          // NVN level past the range (0: to the last one)
        bl = bl > g->drop ? bl - g->drop : 0;
        if (want_nl) want_nl = end > g->drop + bl ? end - g->drop - bl : 1;
    }
    if (bl >= g->levels) return bad_view(*t, "base level out of range");
    const uint32_t nl = want_nl ? want_nl : g->levels - bl;
    if (nl == 0 || bl + nl > g->levels) return bad_view(*t, "level range out of range");
    const uint32_t bly = req.base_layer;
    uint32_t nly = 1;
    switch (req.type) {
        case VK_IMAGE_VIEW_TYPE_2D:
            // A 2D sampler over an array texture (or a texture view of several layers) reads the first layer of the range
            // (inferred for a 2D TEX over an array header; before, the binding got the black dummy).
            if (bly >= total_layers) return bad_view(*t, "2D view of a layer out of range");
            break;
        case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
            nly = req.layer_count ? req.layer_count : total_layers - bly;
            if (nly == 0 || bly + nly > total_layers) return bad_view(*t, "2D array layer range out of range");
            break;
        case VK_IMAGE_VIEW_TYPE_CUBE:
            if (!g->cube || is3d || bly % 6 != 0 || bly + 6 > g->layers) return bad_view(*t, "cube needs a cube-compatible image of 6 layers");
            nly = 6;
            break;
        case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
            nly = req.layer_count ? req.layer_count : g->layers - bly;
            if (!g->cube || is3d || nly == 0 || nly % 6 != 0 || bly % 6 != 0 || bly + nly > g->layers)
                return bad_view(*t, "cube array needs a cube-compatible image with a multiple of 6 layers");
            break;
        case VK_IMAGE_VIEW_TYPE_3D:
            if (!is3d || bly != 0) return bad_view(*t, "3D view of an image that is not 3D");
            break;
        default:
            return bad_view(*t, "1D views are not supported (the image is 2D with height 1)");
    }
    VkFormat fmt = req.format == VK_FORMAT_UNDEFINED ? g->format : promoted_view_format(*g, req.format);
    if (req.component != 0 && g->mutable_fmt) {
        // The shader's sampled type decides how the bits read: float shaders over an integer texture (and the reverse) see the raw bits.
        // Apply this to explicit NVN views too. A UINT view of float storage is compatible with
        // the image, but not with the pool's float OpTypeImage (VUID-format-07753).
        const VkFormat a = req.component == 1 ? float_alias(fmt) : req.component == 2 ? sint_alias(fmt) : uint_alias(fmt);
        if (a != VK_FORMAT_UNDEFINED && format_view_ok(*g, a)) fmt = a;
    }
    if (!format_view_ok(*g, fmt)) {
        char what[160];
        snprintf(what, sizeof what, "view format %d is not compatible with the image's %d (NVN texture format %d, image mutable %d)", static_cast<int>(fmt),
                 static_cast<int>(g->format), static_cast<int>(t->b.format), g->mutable_fmt ? 1 : 0);
        return bad_view(*t, what);
    }
    VkImageAspectFlags asp = g->aspect;
    if (g->aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) asp = req.depth_aspect ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_STENCIL_BIT;
    const int component = asp == VK_IMAGE_ASPECT_STENCIL_BIT ? 3 :
                          vk_format_is_integer(fmt) ? (vk_format_is_signed_int(fmt) ? 2 : 3) : 1;
    if (req.component && req.component != component) return bad_view(*t, "sampled numeric type does not match the shader");
    if (req.depth_compare && !(asp & VK_IMAGE_ASPECT_DEPTH_BIT)) return bad_view(*t, "depth comparison needs a depth aspect");
    // Check the actual alias format, rather than only the image's original format. view_for
    // can restrict inherited usages, but a descriptor must still have sampled-image usage.
    // Cache per image/device; this is also called for every pool descriptor refresh.
    auto [feature, inserted] = g->sampled_formats.try_emplace(fmt, false);
    if (inserted) {
        VkFormatProperties fp = {};
        vkGetPhysicalDeviceFormatProperties(device().physical(), fmt, &fp);
        feature->second = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
    }
    if (!feature->second) return bad_view(*t, "view format does not support sampling");
    return view_for(*g, req.type, bl, nl, bly, nly, fmt, asp, req.swizzle);
}

uint8_t tex_int_sampler_component(Texture* t, VkFormat requested) {
    TexGpu* g = tex_or_null(t);
    if (!g || g->aspect != VK_IMAGE_ASPECT_COLOR_BIT) return 0;
    const VkFormat fmt = requested == VK_FORMAT_UNDEFINED ? g->format : promoted_view_format(*g, requested);
    if (!vk_format_is_integer(fmt) || !format_view_ok(*g, fmt)) return 0;
    if (g->mutable_fmt) {                        // a 32-bit integer format: its float alias already gives a float shader the bits
        const VkFormat a = float_alias(fmt);
        if (a != VK_FORMAT_UNDEFINED && format_view_ok(*g, a)) return 0;
    }
    return vk_format_is_signed_int(fmt) ? 2 : 3;
}

VkImageView tex_attachment_view(Texture* t, uint32_t level, uint32_t layer, uint32_t layer_count, VkFormat view_format) {
    TexGpu* g = tex_or_null(t);
    if (!g) return VK_NULL_HANDLE;
    if (level >= g->levels) return bad_view(*t, "attachment level out of range");
    const uint32_t total_layers = g->type == VK_IMAGE_TYPE_3D ? g->base.depth : g->layers;
    const uint32_t n = layer_count > 1 ? layer_count : 1;
    if (layer >= total_layers || layer + n > total_layers) return bad_view(*t, "attachment layers out of range");
    const VkFormat fmt = attachment_format_of(*g, promoted_view_format(*g, view_format));
    if (!format_view_ok(*g, fmt)) return bad_view(*t, "attachment format is not compatible with the image");
    const VkImageViewType type = n > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    return view_for(*g, type, level, 1, layer, n, fmt, g->aspect, kIdentity);
}

VkFormat tex_attachment_format(Texture* t, VkFormat view_format) {
    TexGpu* g = tex_or_null(t);
    return g ? attachment_format_of(*g, promoted_view_format(*g, view_format)) : VK_FORMAT_UNDEFINED;
}

bool tex_has_storage(Texture* t) {
    TexGpu* g = tex_or_null(t);
    return g && (g->usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0;
}

VkFormat tex_storage_format(Texture* t, VkFormat requested) {
    TexGpu* g = tex_or_null(t);
    const VkFormat f = requested != VK_FORMAT_UNDEFINED ? requested : g ? g->format : VK_FORMAT_UNDEFINED;
    if (!g || !g->mutable_fmt || !is_srgb(f) || srgb_partner(f) == VK_FORMAT_UNDEFINED) return requested;
    return srgb_partner(f);
}

VkImageLayout tex_sampled_layout(Texture* t) {
    return (t && t->fmt && t->fmt->aspect != kAspectColor) ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

VkImageView tex_present_view(Texture* t) {
    TexGpu* g = tex_or_null(t);
    if (!g) return VK_NULL_HANDLE;
    g->presented = true;
    // The UNORM alias of an sRGB image; the plain view when the image cannot be viewed as UNORM.
    const VkFormat fmt = (g->mutable_fmt && is_srgb(g->format)) ? srgb_partner(g->format) : g->format;
    return view_for(*g, VK_IMAGE_VIEW_TYPE_2D, 0, 1, 0, 1, fmt, g->aspect, kIdentity);
}

VkImageView dummy_view(VkImageViewType type, bool depth_compare) { return dummy_view(type, depth_compare, 1); }

VkImageView dummy_view(VkImageViewType type, bool depth_compare, int component) {
    const int idx = dummy_index(type);
    Dummy& d = (depth_compare ? g_sh.depth : component == 2 ? g_sh.sint : component == 3 ? g_sh.uint : g_sh.color)[idx];
    if (dummy_due(d)) make_dummy(d, kDummyTypes[idx], depth_compare, component);
    return d.view;
}

VkImageView dummy_storage_view(VkImageViewType type, VkFormat format) {
    if (format == VK_FORMAT_UNDEFINED) format = VK_FORMAT_R8G8B8A8_UNORM;
    if (type == VK_IMAGE_VIEW_TYPE_CUBE || type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    const std::pair<int, int> key = {static_cast<int>(type), static_cast<int>(format)};
    auto& d = g_sh.storage[key];
    if (!dummy_due(d)) return d.view;
    gfx::ImageDesc id;
    id.format = format;
    id.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    id.create_view = false;
    id.name = "nvn dummy storage";
    if (type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_1D_ARRAY) id.type = VK_IMAGE_TYPE_1D;
    if (type == VK_IMAGE_VIEW_TYPE_3D) id.type = VK_IMAGE_TYPE_3D;
    if (type == VK_IMAGE_VIEW_TYPE_2D_ARRAY || type == VK_IMAGE_VIEW_TYPE_1D_ARRAY) id.array_layers = 6;
    std::string why;
    if (device().create_image(id, &d.img, &why) != VK_SUCCESS) {
        static int logged = 0;
        if (logged++ < 8) NXLOG_W(kTag, "dummy storage image failed: %s", why.c_str());
        dummy_failed(d);
        return d.view;
    }
    const VkImage img = d.img.image;
    const uint32_t layers = id.array_layers;
    const bool ok = device().submit_once([&](VkCommandBuffer cmd) {
        image_barrier(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers);
        const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
        VkClearColorValue c = {};
        vkCmdClearColorImage(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &range);
        image_barrier(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers);
    });
    if (!ok) {
        device().destroy_image(d.img);
        dummy_failed(d);
        return d.view;
    }
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img;
    vi.viewType = type;
    vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    if (vkCreateImageView(vkdev(), &vi, nullptr, &d.view) != VK_SUCCESS) {
        d.view = VK_NULL_HANDLE;
        device().destroy_image(d.img);
        dummy_failed(d);
    } else {
        vk_gauge_image_views().add(1);
        d.tried = true;
    }
    return d.view;
}

VkSampler dummy_sampler() {
    if (g_sh.dummy_sampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.magFilter = VK_FILTER_NEAREST;
        ci.minFilter = VK_FILTER_NEAREST;
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.addressModeU = ci.addressModeV = ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        if (vkCreateSampler(vkdev(), &ci, nullptr, &g_sh.dummy_sampler) != VK_SUCCESS) g_sh.dummy_sampler = VK_NULL_HANDLE;
        else vk_gauge_samplers().add(1);
    }
    return g_sh.dummy_sampler;
}

// The sampler as the display settings want it (nxrt/gfx_tuning.h): more anisotropy on filtering samplers, a LOD bias on every sampler
// that samples colour. Nearest-filtered samplers, shadow comparison samplers and clamped single-level samplers are left as the game set them.
SamplerInfo tuned_sampler(const SamplerInfo& in) {
    SamplerInfo s = in;
    const int aniso = gfx::tuning().anisotropy.load(std::memory_order_relaxed);
    const float bias = gfx::tuning().lod_bias.load(std::memory_order_relaxed);
    if (s.compare_mode == NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE) return s;
    const bool mips = min_filter_uses_mips(s.min_filter);
    if (aniso > 1 && mips && vk_min_filter(s.min_filter) == VK_FILTER_LINEAR && vk_mag_filter(s.mag_filter) == VK_FILTER_LINEAR)
        s.max_anisotropy = std::max(s.max_anisotropy, static_cast<float>(aniso));
    if (bias != 0.0f && mips) s.lod_bias += bias;
    return s;
}

// sampler_for's last few answers, for the game's sampler and the tuning tuned_sampler applies (render thread, as the map): a draw's
// samplers repeat from draw to draw (rt agent, notes/rt.md). Emptied with the map (tex_shutdown).
struct SamplerLast { SamplerInfo in; int aniso = 0; float bias = 0.0f; VkSampler out = VK_NULL_HANDLE; };
static SamplerLast g_sampler_last[4];
static unsigned g_sampler_last_next = 0;

VkSampler sampler_for(const SamplerInfo& game_sampler) {
    SamplerLast* last = g_sampler_last;
    unsigned& last_next = g_sampler_last_next;
    const int aniso = gfx::tuning().anisotropy.load(std::memory_order_relaxed);
    const float bias = gfx::tuning().lod_bias.load(std::memory_order_relaxed);
    for (int i = 0; i < 4; ++i) {
        const SamplerLast& l = last[i];
        if (l.out && l.aniso == aniso && l.bias == bias && l.in == game_sampler) {
            if (rt_check_on()) {   // [nvn] rt_check: the map gives the same sampler
                const SamplerInfo s = tuned_sampler(game_sampler);
                VkSampler m = VK_NULL_HANDLE;
                for (const auto& e : g_sh.samplers[sampler_hash(s)]) if (e.first == s) m = e.second;
                if (m != l.out) NXLOG_W("nvn.rtcheck", "mismatch: sampler_for last sampler");
            }
            return l.out;
        }
    }
    const SamplerInfo s = tuned_sampler(game_sampler);
    auto& bucket = g_sh.samplers[sampler_hash(s)];
    for (const auto& e : bucket)
        if (e.first == s) {
            last[last_next++ % 4] = {game_sampler, aniso, bias, e.second};
            return e.second;
        }
    const VkSampler v = create_sampler(s);
    if (v == VK_NULL_HANDLE) return dummy_sampler();
    bucket.emplace_back(s, v);
    last[last_next++ % 4] = {game_sampler, aniso, bias, v};
    return v;
}

void tex_destroy(TexGpu* g) {
    if (!g) return;
    if (--g->refs > 0) return;   // another texture object over the same storage still uses the image
    {
        std::lock_guard<std::mutex> lk(g_live_mu);
        if (!g_live_set.erase(g)) return;   // already destroyed by tex_shutdown
    }
    g_gauge_textures_live.sub(1);
    if (g->presented) device().wait_idle();
    for (auto& kv : g->views) { vkDestroyImageView(vkdev(), kv.second, nullptr); vk_gauge_image_views().sub(1); }
    g->views.clear();
    if (g->img) g_live_images.fetch_sub(1);
    device().destroy_image(g->img);
    if (g->sparse) {
        for (VkDeviceMemory m : g->sparse->chunks) vkFreeMemory(vkdev(), m, nullptr);
        delete g->sparse;
    }
    delete g;
}

TexStats tex_stats() {
    TexStats s;
    s.images = g_live_images.load();
    s.uploads = counters().texture_uploads.load();
    s.upload_bytes = counters().texture_upload_bytes.load();
    s.transcodes = counters().bc7_transcodes.load();
    s.failed = counters().textures_failed.load();
    return s;
}

// Frees every cached object this module made (samplers, dummy views and images). Render thread, device idle, no texture alive:
// called by the tests before shutdown_for_test (the back end's teardown does not call it yet).
void tex_shutdown() {
    texup::shutdown();                // the cache upgrade threads stop (their work is dropped, never half written)
    texprep::release_staging();       // upload buffers of finished background results belong to the device that is going away
    std::vector<TexGpu*> left;
    {
        std::lock_guard<std::mutex> lk(g_live_mu);
        left.assign(g_live_set.begin(), g_live_set.end());
    }
    // Textures that are still alive (the game never finalized them, or a test left them behind) keep their place in the registry. They
    // must not keep a pointer to the image object that is about to be freed: texture_registry_find_alias would hand it to the next texture
    // made over the same memory, and that texture would use freed memory (a barrier on VK_NULL_HANDLE, or a crash).
    texture_registry_detach_gpu();
    for (TexGpu* g : left) { g->refs = 1; tex_destroy(g); }
    for (auto& kv : g_sh.samplers)
        for (auto& e : kv.second) { vkDestroySampler(vkdev(), e.second, nullptr); vk_gauge_samplers().sub(1); }
    g_sh.samplers.clear();
    for (SamplerLast& l : g_sampler_last) l = SamplerLast();
    if (g_sh.dummy_sampler != VK_NULL_HANDLE) { vkDestroySampler(vkdev(), g_sh.dummy_sampler, nullptr); vk_gauge_samplers().sub(1); }
    for (auto& kv : g_sh.storage) {
        if (kv.second.view != VK_NULL_HANDLE) { vkDestroyImageView(vkdev(), kv.second.view, nullptr); vk_gauge_image_views().sub(1); }
        device().destroy_image(kv.second.img);
    }
    for (Dummy* set : {g_sh.color, g_sh.depth, g_sh.sint, g_sh.uint}) {
        for (int i = 0; i < 6; ++i) {
            if (set[i].view != VK_NULL_HANDLE) { vkDestroyImageView(vkdev(), set[i].view, nullptr); vk_gauge_image_views().sub(1); }
            device().destroy_image(set[i].img);
        }
    }
    g_sh = Shared();
}

namespace {
float half_to_float(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = std::ldexp(static_cast<float>(m), -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(static_cast<float>(m + 1024), static_cast<int>(e) - 25);
    return s ? -v : v;
}
float small_float(uint32_t bits, int mant_bits) {   // unsigned 11 / 10 bit float (5 bit exponent)
    const uint32_t e = bits >> mant_bits, m = bits & ((1u << mant_bits) - 1);
    if (e == 0) return std::ldexp(static_cast<float>(m), -14 - mant_bits);
    if (e == 31) return m ? NAN : INFINITY;
    return std::ldexp(1.0f + static_cast<float>(m) / static_cast<float>(1u << mant_bits), static_cast<int>(e) - 15);
}
uint8_t to8(float v, bool hdr) {
    if (!(v == v)) return 255;
    if (hdr) v = v / (1.0f + v);          // display of a high dynamic range value
    return static_cast<uint8_t>(std::min(1.0f, std::max(0.0f, v)) * 255.0f + 0.5f);
}
}  // namespace

// The decode of a dump: texel bytes of a level to RGBA8 (saved as a PNG when path is not empty) plus per-channel statistics.
static bool convert_png_impl(const uint8_t* src, VkFormat f, VkExtent3D e, uint32_t bpp, VkImageAspectFlags aspect, const std::string& path, std::string* stats) {
    std::vector<uint8_t> rgba(static_cast<size_t>(e.width) * e.height * 4, 255);
    double sum[4] = {0, 0, 0, 0};
    float mn[4] = {1e30f, 1e30f, 1e30f, 1e30f}, mx[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    uint64_t nonzero = 0;
    for (uint32_t i = 0; i < e.width * e.height; ++i) {
        const uint8_t* p = src + static_cast<size_t>(i) * bpp;
        float c[4] = {0, 0, 0, 1};
        bool hdr = false;
        auto u32 = [&](int k) { uint32_t v; std::memcpy(&v, p + 4 * k, 4); return v; };
        auto f32 = [&](int k) { float v; std::memcpy(&v, p + 4 * k, 4); return v; };
        auto u16 = [&](int k) { uint16_t v; std::memcpy(&v, p + 2 * k, 2); return v; };
        switch (f) {
            case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: for (int k = 0; k < 4; ++k) c[k] = p[k] / 255.0f; break;
            case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: c[0] = p[2] / 255.0f; c[1] = p[1] / 255.0f; c[2] = p[0] / 255.0f; c[3] = p[3] / 255.0f; break;
            case VK_FORMAT_A2B10G10R10_UNORM_PACK32: { const uint32_t v = u32(0); c[0] = (v & 1023) / 1023.0f; c[1] = ((v >> 10) & 1023) / 1023.0f; c[2] = ((v >> 20) & 1023) / 1023.0f; c[3] = (v >> 30) / 3.0f; break; }
            case VK_FORMAT_R8G8_UNORM: c[0] = p[0] / 255.0f; c[1] = p[1] / 255.0f; c[2] = 0; break;
            case VK_FORMAT_R8_UNORM: c[0] = c[1] = c[2] = p[0] / 255.0f; break;
            case VK_FORMAT_R16_UNORM: c[0] = c[1] = c[2] = u16(0) / 65535.0f; break;
            case VK_FORMAT_R16_SFLOAT: c[0] = c[1] = c[2] = half_to_float(u16(0)); hdr = true; break;
            case VK_FORMAT_R16G16_SFLOAT: c[0] = half_to_float(u16(0)); c[1] = half_to_float(u16(1)); c[2] = 0; hdr = true; break;
            case VK_FORMAT_R16G16B16A16_SFLOAT: for (int k = 0; k < 4; ++k) c[k] = half_to_float(u16(k)); hdr = true; break;
            case VK_FORMAT_R16G16B16A16_UNORM: for (int k = 0; k < 4; ++k) c[k] = u16(k) / 65535.0f; break;
            case VK_FORMAT_R32_SFLOAT: c[0] = c[1] = c[2] = f32(0); hdr = true; break;
            case VK_FORMAT_R32G32_SFLOAT: c[0] = f32(0); c[1] = f32(1); c[2] = 0; hdr = true; break;
            case VK_FORMAT_R32G32B32A32_SFLOAT: for (int k = 0; k < 4; ++k) c[k] = f32(k); hdr = true; break;
            case VK_FORMAT_B10G11R11_UFLOAT_PACK32: { const uint32_t v = u32(0); c[0] = small_float(v & 2047, 6); c[1] = small_float((v >> 11) & 2047, 6); c[2] = small_float(v >> 22, 5); hdr = true; break; }
            case VK_FORMAT_R32_UINT: { const uint32_t v = u32(0); c[0] = (v & 255) / 255.0f; c[1] = ((v >> 8) & 255) / 255.0f; c[2] = ((v >> 16) & 255) / 255.0f; c[3] = (v >> 24) / 255.0f; break; }
            case VK_FORMAT_R32G32_UINT: { const uint32_t v = u32(0), w2 = u32(1); c[0] = (v & 255) / 255.0f; c[1] = ((v >> 8) & 255) / 255.0f; c[2] = (w2 & 255) / 255.0f; c[3] = ((w2 >> 8) & 255) / 255.0f; break; }
            case VK_FORMAT_R32G32B32A32_UINT: for (int k = 0; k < 4; ++k) c[k] = (u32(k) & 255) / 255.0f; break;
            case VK_FORMAT_D16_UNORM: c[0] = c[1] = c[2] = u16(0) / 65535.0f; break;
            case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_D32_SFLOAT_S8_UINT: c[0] = c[1] = c[2] = f32(0); break;
            case VK_FORMAT_D24_UNORM_S8_UINT: c[0] = c[1] = c[2] = (u32(0) & 0xffffff) / 16777215.0f; break;
            default: break;
        }
        bool any = false;
        for (int k = 0; k < 4; ++k) {
            if (c[k] == c[k]) {
                sum[k] += c[k];
                mn[k] = std::min(mn[k], c[k]);
                mx[k] = std::max(mx[k], c[k]);
            }
            if (c[k] != 0.0f) any = true;
            rgba[static_cast<size_t>(i) * 4 + k] = (k == 3 && (aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) ? 255 : to8(c[k], hdr);
        }
        if (any) ++nonzero;
    }
    if (stats) {
        char b[400];
        const double n = static_cast<double>(e.width) * e.height;
        std::snprintf(b, sizeof b, "min %g %g %g %g max %g %g %g %g mean %g %g %g %g nonzero %.1f%%", mn[0], mn[1], mn[2], mn[3], mx[0], mx[1], mx[2], mx[3],
                      sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n, 100.0 * static_cast<double>(nonzero) / n);
        *stats = b;
    }
    return path.empty() || gfx::save_png_rgba(path, rgba.data(), e.width, e.height);
}

// ---- mid-frame snapshots (debug): a copy of a texture recorded into the frame being replayed, written out after the frame
static uint32_t dump_bpp(VkFormat f, VkImageAspectFlags* aspect) {
    *aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_R16G16_SFLOAT:
            return 4;
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: return 8;
        case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: return 16;
        case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: return 2;
        case VK_FORMAT_R8_UNORM: return 1;
        case VK_FORMAT_D16_UNORM: *aspect = VK_IMAGE_ASPECT_DEPTH_BIT; return 2;
        case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_D32_SFLOAT_S8_UINT: case VK_FORMAT_D24_UNORM_S8_UINT: *aspect = VK_IMAGE_ASPECT_DEPTH_BIT; return 4;
        default: return 0;
    }
}

namespace {
struct Snapshot {
    gfx::Buffer buf;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent3D e = {0, 0, 1};
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t bpp = 0;
    bool raw = false;                  // a block-compressed image copied as it is (the buffer is written to a file as it is)
};
std::vector<Snapshot> g_snaps;
}  // namespace

int snapshot_record(Texture* t, Frame& fr) {
    TexGpu* g = t ? t->gpu : nullptr;
    if (!g || g->failed || g->layout == VK_IMAGE_LAYOUT_UNDEFINED) return -1;
    Snapshot s;
    s.format = g->format;
    s.bpp = dump_bpp(g->format, &s.aspect);
    if (!s.bpp) return -1;
    const VkExtent3D full = level_extent(g->base, 0);
    const bool is3d = g->type == VK_IMAGE_TYPE_3D;
    const uint32_t slices = is3d ? full.depth : std::max<uint32_t>(1, g->layers);   // a 3D texture's slices or an array's layers, stacked into one tall image
    s.e = {full.width, full.height * slices, 1};
    const uint64_t size = static_cast<uint64_t>(s.e.width) * s.e.height * s.bpp;
    if (device().create_buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gfx::MemoryUse::Readback, &s.buf, "nvn snapshot") != VK_SUCCESS) return -1;
    const VkImageLayout prev = g->layout;
    move_layout(fr, *g, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy r = {};
    r.imageSubresource = {s.aspect, 0, 0, is3d ? 1u : slices};
    r.imageExtent = {full.width, full.height, is3d ? slices : 1u};
    vkCmdCopyImageToBuffer(fr.cb, g->img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.buf.buffer, 1, &r);
    move_layout(fr, *g, prev);
    g_snaps.push_back(std::move(s));
    g_gauge_texture_snapshots.add(1);
    return static_cast<int>(g_snaps.size()) - 1;
}

// A block-compressed image: every level and layer, copied as blocks (for comparing what the GPU holds with trace_dump_prepared's data).
int snapshot_record_raw(Texture* t, Frame& fr) {
    TexGpu* g = t ? t->gpu : nullptr;
    if (!g || g->failed || g->layout == VK_IMAGE_LAYOUT_UNDEFINED || g->depth) return -1;
    uint32_t bytes = 0;
    if (g->format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && g->format <= VK_FORMAT_BC7_SRGB_BLOCK)
        bytes = (g->format <= VK_FORMAT_BC1_RGBA_SRGB_BLOCK || g->format == VK_FORMAT_BC4_UNORM_BLOCK || g->format == VK_FORMAT_BC4_SNORM_BLOCK) ? 8 : 16;
    if (!bytes) return -1;
    // every level, tight, level after level with all layers: the layout prepare_texture produces for block formats
    std::vector<VkBufferImageCopy> regions;
    uint64_t size = 0;
    for (uint32_t l = 0; l < g->levels; ++l) {
        const VkExtent3D e = level_extent(g->base, l);
        VkBufferImageCopy r = {};
        r.bufferOffset = size;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, g->layers};
        r.imageExtent = {e.width, e.height, 1};
        regions.push_back(r);
        size += static_cast<uint64_t>((e.width + 3) / 4) * ((e.height + 3) / 4) * bytes * g->layers;
    }
    Snapshot s;
    s.raw = true;
    s.format = g->format;
    s.e = {static_cast<uint32_t>(size), 1, 1};
    if (device().create_buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gfx::MemoryUse::Readback, &s.buf, "nvn raw snapshot") != VK_SUCCESS) return -1;
    const VkImageLayout prev = g->layout;
    move_layout(fr, *g, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdCopyImageToBuffer(fr.cb, g->img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.buf.buffer, static_cast<uint32_t>(regions.size()), regions.data());
    move_layout(fr, *g, prev);
    g_snaps.push_back(std::move(s));
    g_gauge_texture_snapshots.add(1);
    return static_cast<int>(g_snaps.size()) - 1;
}

void trace_dump_prepared(Texture* t, const std::string& dir) {
    static std::unordered_set<uint32_t> done;
    if (!t || !done.insert(t->id).second) return;
    TexCaps caps;
    caps.bc = device().features().bc;
    caps.astc = device().features().astc_ldr;
    caps.bc7_transcode = true;
    PreparedTexture prep;
    const bool ok = prepare_texture(*t, caps, prep);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string base = dir + "/t" + std::to_string(t->id);
    if (FILE* f = std::fopen((base + ".txt").c_str(), "w")) {
        std::fprintf(f, "ok %d format %d levels %u layers %u block %ux%u bytes %u transcoded %d error %s\n", ok ? 1 : 0, static_cast<int>(prep.format), prep.levels, prep.layers,
                     prep.block_w, prep.block_h, prep.block_bytes, prep.transcoded ? 1 : 0, prep.error.c_str());
        for (const PreparedLevel& l : prep.level)
            std::fprintf(f, "level %u %u %u offset %llu size %llu row_blocks %u\n", l.width, l.height, l.depth, static_cast<unsigned long long>(l.offset),
                         static_cast<unsigned long long>(l.size), l.row_blocks);
        std::fclose(f);
    }
    if (ok)
        if (FILE* f = std::fopen((base + ".bin").c_str(), "wb")) {
            std::fwrite(prep.data.data(), 1, prep.data.size(), f);
            std::fclose(f);
        }
}

bool snapshot_write(int index, const std::string& path, std::string* stats) {
    if (index < 0 || index >= static_cast<int>(g_snaps.size())) return false;
    Snapshot& s = g_snaps[static_cast<size_t>(index)];
    if (!s.buf) return false;
    device().invalidate_buffer(s.buf);
    if (s.raw) {
        bool wrote = false;
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            wrote = std::fwrite(s.buf.mapped, 1, static_cast<size_t>(s.e.width), f) == s.e.width;   // e.width holds the byte count of a raw copy
            std::fclose(f);
        }
        if (stats) *stats = "raw " + std::to_string(s.e.width) + " bytes";
        device().destroy_buffer(s.buf);
        s.buf = gfx::Buffer();
        return wrote;
    }
    const bool ok = convert_png_impl(static_cast<const uint8_t*>(s.buf.mapped), s.format, s.e, s.bpp, s.aspect, path, stats);
    const bool hdr = s.format == VK_FORMAT_R16G16B16A16_SFLOAT || s.format == VK_FORMAT_R16_SFLOAT || s.format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ||
                     s.format == VK_FORMAT_R32_SFLOAT || s.format == VK_FORMAT_R16G16_SFLOAT;
    if ((s.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || s.format == VK_FORMAT_R16G16B16A16_UNORM || hdr) {   // depth, 16-bit colour, float: the texel bytes too (<path>.bin), an 8-bit PNG loses the precision
        if (FILE* f = std::fopen((path + ".bin").c_str(), "wb")) {
            std::fwrite(s.buf.mapped, 1, static_cast<size_t>(s.e.width) * s.e.height * s.bpp, f);
            std::fclose(f);
        }
    }
    device().destroy_buffer(s.buf);
    s.buf = gfx::Buffer();
    return ok;
}

bool dump_texture(Texture* t, const std::string& path, std::string* stats) {
    TexGpu* g = t ? t->gpu : nullptr;
    if (!g || g->failed || g->layout == VK_IMAGE_LAYOUT_UNDEFINED) return false;
    const VkFormat f = g->format;
    uint32_t bpp = 0;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            bpp = 4; break;
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: bpp = 8; break;
        case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: bpp = 16; break;
        case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: bpp = 2; break;
        case VK_FORMAT_R8_UNORM: bpp = 1; break;
        case VK_FORMAT_D16_UNORM: bpp = 2; aspect = VK_IMAGE_ASPECT_DEPTH_BIT; break;
        case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_D32_SFLOAT_S8_UINT: bpp = 4; aspect = VK_IMAGE_ASPECT_DEPTH_BIT; break;
        case VK_FORMAT_D24_UNORM_S8_UINT: bpp = 4; aspect = VK_IMAGE_ASPECT_DEPTH_BIT; break;
        default: return false;
    }
    const VkExtent3D e = level_extent(g->base, 0);
    const uint64_t size = static_cast<uint64_t>(e.width) * e.height * bpp;
    finish();
    gfx::Buffer rb;
    run_frame_sync([&](Frame& fr) {
        if (device().create_buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gfx::MemoryUse::Readback, &rb, "nvn texture dump") != VK_SUCCESS) return;
        const VkImageLayout prev = g->layout;
        move_layout(fr, *g, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy r = {};
        r.imageSubresource = {aspect, 0, 0, 1};
        r.imageExtent = {e.width, e.height, 1};
        vkCmdCopyImageToBuffer(fr.cb, g->img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buffer, 1, &r);
        move_layout(fr, *g, prev);
    });
    if (!rb) return false;
    device().invalidate_buffer(rb);
    const bool ok = convert_png_impl(static_cast<const uint8_t*>(rb.mapped), f, e, bpp, aspect, path, stats);
    device().destroy_buffer(rb);
    return ok;
}

void mark_texture_cpu_dirty(Texture* t) {
    if (!t) return;
    texprep::on_flushed(t);        // before the flag: a reader that sees the flag also sees the new generation
    t->cpu_dirty.store(true);
}

void texture_bound(Texture* t) { texprep::on_bound(t); }
void texture_registered(Texture* t) { texprep::on_registered(t); }

// Copies every level of a render target back into the game's storage (block-linear, tiled with tegra::tile). Only when the GPU
// image holds the newest data: it exists, it was written by the GPU (or uploaded) and the CPU did not write since. Pitch-linear,
// transcoded, depth and format-fallback textures are not read back.
bool download_texture(Texture* t) {
    if (!t || !active() || !t->mem || !t->fmt || !t->block_linear) return false;
    if (t->cpu_dirty.load()) return false;
    const TexLayout L = compute_texture_layout(t->b);
    if (!L.valid || !L.block_linear) return false;
    const uint64_t size = L.layout.linear_size;

    finish();   // earlier submissions must be done before the copy reads the image
    gfx::Buffer rb;
    run_frame_sync([&](Frame& f) {
        // The image's state (made again, rescaled, its layout) belongs to the render thread: it is looked at here, in the job.
        TexGpu* g = t->gpu;
        if (!g || g->failed || g->depth || g->transcoded || g->sparse || g->format != t->fmt->vk) return;
        if (g->scale != 1.0f) {
            static int logged = 0;
            if (logged++ < 10) NXLOG_W(kTag, "texture %u: a scaled render target is not read back into game memory", t->id);
            return;
        }
        if (!(g->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || g->layout == VK_IMAGE_LAYOUT_UNDEFINED) return;
        if (device().create_buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gfx::MemoryUse::Readback, &rb, "nvn texture readback") != VK_SUCCESS)
            return;
        const VkImageLayout prev = g->layout;
        move_layout(f, *g, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        std::vector<VkBufferImageCopy> regions;
        for (uint32_t l = 0; l < L.levels; ++l) {
            const tegra::LevelLayout& lv = L.layout.levels[l];
            VkBufferImageCopy r = {};
            r.bufferOffset = lv.linear_offset;
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, g->layers};
            r.imageExtent = {lv.width, lv.height, lv.depth};
            regions.push_back(r);
        }
        vkCmdCopyImageToBuffer(f.cb, g->img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buffer, static_cast<uint32_t>(regions.size()),
                               regions.data());
        move_layout(f, *g, prev);
    });
    if (!rb) return false;
    device().invalidate_buffer(rb);
    std::vector<uint8_t> lin(static_cast<size_t>(size));
    std::memcpy(lin.data(), rb.mapped, static_cast<size_t>(size));
    device().destroy_buffer(rb);
    tegra::tile(L.desc, L.layout, lin.data(), t->mem);
    return true;
}


// ---- BUFFER-target textures: texel buffer views -------------------------------------------------------------------------------
namespace {
struct BufViewKey {
    VkBuffer buffer;
    VkDeviceSize offset, range;
    VkFormat format;
    bool operator<(const BufViewKey& o) const {
        if (buffer != o.buffer) return buffer < o.buffer;
        if (offset != o.offset) return offset < o.offset;
        if (range != o.range) return range < o.range;
        return format < o.format;
    }
};
std::mutex g_bufview_mu;
std::map<BufViewKey, VkBufferView> g_bufviews;

bool buffer_format_ok(VkFormat f, bool storage) {
    static std::mutex mu;
    static std::map<VkFormat, VkFormatFeatureFlags> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(f);
    if (it == cache.end()) {
        VkFormatProperties p = {};
        vkGetPhysicalDeviceFormatProperties(device().physical(), f, &p);
        it = cache.emplace(f, p.bufferFeatures).first;
    }
    return (it->second & (storage ? VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT : VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT)) != 0;
}
}  // namespace

VkBufferView make_buffer_view(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range, VkFormat format) {
    std::lock_guard<std::mutex> lk(g_bufview_mu);
    const BufViewKey key{buffer, offset, range, format};
    if (auto it = g_bufviews.find(key); it != g_bufviews.end()) return it->second;
    VkBufferViewCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};
    ci.buffer = buffer;
    ci.format = format;
    ci.offset = offset;
    ci.range = range;
    VkBufferView v = VK_NULL_HANDLE;
    if (vkCreateBufferView(vkdev(), &ci, nullptr, &v) != VK_SUCCESS) v = VK_NULL_HANDLE;
    g_bufviews.emplace(key, v);   // a failure is remembered too (null), so it is not retried every draw
    return v;
}

void forget_buffer_views(VkBuffer buffer) {
    std::vector<VkBufferView> gone;
    {
        std::lock_guard<std::mutex> lk(g_bufview_mu);
        for (auto it = g_bufviews.lower_bound(BufViewKey{buffer, 0, 0, VK_FORMAT_UNDEFINED}); it != g_bufviews.end() && it->first.buffer == buffer;) {
            if (it->second) gone.push_back(it->second);
            it = g_bufviews.erase(it);
        }
    }
    if (!gone.empty())
        defer_destroy([gone]() { for (VkBufferView v : gone) vkDestroyBufferView(vkdev(), v, nullptr); });
}

VkBufferView tex_buffer_view(Texture* t, const TextureViewInfo* view, int component, VkFormat shader_format, bool storage,
                             uint64_t* gpu_address, uint64_t* bytes, VkFormat* format) {
    if (!t || !t->fmt || t->b.width <= 0) return VK_NULL_HANDLE;
    // A buffer fetch reads the texture's memory as a run of texels, whatever target the storage was built with (a texture view of BUFFER target,
    // or a 2D/3D texture the game left bound to a samplerBuffer slot): the run is the BUFFER texture's width, else the whole storage.
    const bool buffer_target = t->b.target == NVN_TEXTURE_TARGET_BUFFER;
    const FormatInfo* fi = t->fmt;
    if (view && view->has_format)
        if (const FormatInfo* vf = format_info(view->format)) fi = vf;
    VkFormat fmt = storage && shader_format != VK_FORMAT_UNDEFINED ? shader_format : fi->vk;
    if (fmt == VK_FORMAT_UNDEFINED) return VK_NULL_HANDLE;
    if (component != 0) {
        const VkFormat a = component == 1 ? float_alias(fmt) : component == 2 ? sint_alias(fmt) : uint_alias(fmt);
        if (a != VK_FORMAT_UNDEFINED) fmt = a;
        const int actual = vk_format_is_integer(fmt) ? (vk_format_is_signed_int(fmt) ? 2 : 3) : 1;
        if (actual != component) {
            bad_view(*t, "texel buffer numeric type does not match the shader");
            return VK_NULL_HANDLE;
        }
    }
    if (!buffer_format_ok(fmt, storage)) return VK_NULL_HANDLE;
    uint64_t addr = 0;
    if (t->vpool) addr = t->vpool->gpu_base + t->voffset;
    else if (t->mem) addr = reinterpret_cast<uint64_t>(t->mem);
    if (!addr) return VK_NULL_HANDLE;
    const VkPhysicalDeviceLimits& lim = device().properties().limits;
    if (!fi->bytes) return VK_NULL_HANDLE;
    uint64_t texels = static_cast<uint64_t>(t->b.width);
    if (!buffer_target) {
        texels = t->storage_size / fi->bytes;
        if (!texels) texels = static_cast<uint64_t>(t->b.width) * static_cast<uint64_t>(std::max(1, t->b.height)) * static_cast<uint64_t>(std::max(1, t->b.depth));
    }
    uint64_t n = std::min<uint64_t>(texels, lim.maxTexelBufferElements);
    const uint64_t size = n * fi->bytes;
    BufRef br = resolve_buffer(addr, size);
    if (!br.ok) return VK_NULL_HANDLE;
    const VkDeviceSize align = std::max<VkDeviceSize>(1, lim.minTexelBufferOffsetAlignment);
    if (br.offset % align) return VK_NULL_HANDLE;
    const VkDeviceSize range = std::min<VkDeviceSize>(size, br.avail) / fi->bytes * fi->bytes;
    if (!range) return VK_NULL_HANDLE;
    if (gpu_address) *gpu_address = addr;
    if (bytes) *bytes = range;
    if (format) *format = fmt;
    return make_buffer_view(br.buffer, br.offset, range, fmt);
}

VkBufferView dummy_buffer_view(VkFormat format, bool storage) {
    if (format == VK_FORMAT_UNDEFINED || !buffer_format_ok(format, storage)) format = VK_FORMAT_R32_SFLOAT;
    return make_buffer_view(dummy_buffer(), 0, 65536 - 32, format);   // the last 32 bytes are the unset-attribute defaults
}

void buffer_views_forget(VkBuffer buffer) {
    std::lock_guard<std::mutex> lk(g_bufview_mu);
    for (auto it = g_bufviews.lower_bound(BufViewKey{buffer, 0, 0, VK_FORMAT_UNDEFINED}); it != g_bufviews.end() && it->first.buffer == buffer;) {
        if (it->second) vkDestroyBufferView(vkdev(), it->second, nullptr);
        it = g_bufviews.erase(it);
    }
}

}  // namespace gpu
}  // namespace nxrt::nvn::be
