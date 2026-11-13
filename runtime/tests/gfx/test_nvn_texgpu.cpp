// SPDX-License-Identifier: GPL-3.0-or-later
// NVN texture GPU module (src/gfx/nvn/nvn_gpu_texture.cpp) on a headless device with validation: upload and readback of every level
// (bit exact), BC and ASTC (ASTC to BC7 when the GPU has no ASTC), views (cube, array, 3D, cache), depth and colour targets,
// layout transitions, samplers and border colours, CPU and GPU write rules, the download of a render target, the failure path and
// the dummy views. Each case builds its own device and textures and checks that validation reports nothing and that no VMA
// allocation is left after shutdown.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <nxrt/config.h>

#include "nvn_core.h"
#include "nvn_enums.h"
#include "nvn_format.h"
#include "nvn_gpu.h"
#include "nvn_gpu_internal.h"
#include "nvn_texdata.h"
#include "nvn_texprep.h"
#include "nvn_texture_layout.h"
#include "nxrt/gfx.h"
#include "nxrt/gfx_tuning.h"
#include "tegra_tiling.h"
#include "test_framework.h"
NXTEST_GROUP(gfx_nvn_texgpu)

namespace nxrt::nvn::be::gpu {
// Frees the samplers, dummy views and dummy images (defined in nvn_gpu_texture.cpp). Render thread, no texture alive.
void tex_shutdown();
}  // namespace nxrt::nvn::be::gpu

using namespace nxrt;
using namespace nxrt::gfx;
using namespace nxrt::nvn;
using nxrt::nvn::be::Ref;
using nxrt::nvn::be::Texture;
using nxrt::nvn::be::TexLayout;
using nxrt::nvn::be::compute_texture_layout;
using nxrt::nvn::be::gpu::Frame;
using nxrt::nvn::be::gpu::ViewRequest;
namespace gpu = nxrt::nvn::be::gpu;

namespace {

uint32_t g_next_id = 100;

// Texture data is cached on disk by prepare_texture: the tests keep the cache off (the env var wins over the config).
void no_cache() {
#ifdef _WIN32
    _putenv_s("NXRT_NVN_BC7_CACHE", "");
#else
    unsetenv("NXRT_NVN_BC7_CACHE");
#endif
    config_set("nvn", "bc7_cache", "");
}

TextureBuilderInfo info(int target, int w, int h, int depth, int levels, int format, int flags = 0) {
    TextureBuilderInfo b;
    b.target = target;
    b.width = w;
    b.height = h;
    b.depth = depth;
    b.levels = levels;
    b.format = format;
    b.flags = flags;
    return b;
}

// A texture built by hand: its layout, the linear image (ground truth, levels back to back) and the tiled storage the game holds.
struct Tex {
    Ref<Texture> t;
    TexLayout L;
    std::vector<uint8_t> lin;
    std::vector<uint8_t> mem;
};

bool make_tex(Tex& x, const TextureBuilderInfo& b) {
    x.L = compute_texture_layout(b);
    if (!x.L.valid || !x.L.block_linear) {
        std::printf("    layout of the test texture is invalid\n");
        return false;
    }
    x.lin.assign(static_cast<size_t>(x.L.layout.linear_size), 0);
    x.mem.assign(static_cast<size_t>(x.L.storage_size), 0);
    x.t = Ref<Texture>(new Texture());
    Texture& t = *x.t;
    t.b = b;
    t.fmt = x.L.fmt;
    t.mem = x.mem.data();
    t.storage_size = x.L.storage_size;
    t.block_linear = true;
    t.block_height_gobs = x.L.layout.levels[0].gob_blocks_y;
    t.levels = x.L.levels;
    t.layers = x.L.layers;
    t.id = g_next_id++;
    t.is_display = (b.flags & NVN_TEXTURE_FLAGS_DISPLAY) != 0;
    return true;
}

void fill_random(std::vector<uint8_t>& v, uint32_t seed) {
    std::mt19937 rng(seed);
    for (uint8_t& b : v) b = static_cast<uint8_t>(rng());
}

// Writes the linear image into the tiled storage (what the game does with its memory).
void tile_into_storage(Tex& x) { tegra::tile(x.L.desc, x.L.layout, x.lin.data(), x.mem.data()); }

// Copies every level and layer of the image into a host buffer (tight rows: the same layout as the linear image). The image is moved
// to TRANSFER_SRC and back to the layout it had.
bool read_back(Device& dev, Texture* t, const TexLayout& L, std::vector<uint8_t>& out) {
    const uint64_t size = L.layout.linear_size;
    out.assign(static_cast<size_t>(size), 0);
    gfx::Buffer rb;
    gpu::finish();
    gpu::run_frame_sync([&](Frame& f) {
        if (dev.create_buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUse::Readback, &rb, "test readback") != VK_SUCCESS) return;
        const VkImageLayout prev = gpu::tex_layout(t);
        gpu::tex_transition(t, f, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        std::vector<VkBufferImageCopy> regions;
        for (uint32_t l = 0; l < L.levels; ++l) {
            const tegra::LevelLayout& lv = L.layout.levels[l];
            VkBufferImageCopy r = {};
            r.bufferOffset = lv.linear_offset;
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, L.layers};
            r.imageExtent = {lv.width, lv.height, lv.depth};
            regions.push_back(r);
        }
        vkCmdCopyImageToBuffer(f.cb, gpu::tex_image(t), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buffer, static_cast<uint32_t>(regions.size()),
                               regions.data());
        if (prev != VK_IMAGE_LAYOUT_UNDEFINED) gpu::tex_transition(t, f, prev);
    });
    if (!rb) return false;
    dev.invalidate_buffer(rb);
    std::memcpy(out.data(), rb.mapped, static_cast<size_t>(size));
    dev.destroy_buffer(rb);
    return true;
}

// Index of the first differing byte, or -1 when equal.
long first_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size()) return 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return static_cast<long>(i);
    return -1;
}

// Upload through the module and read the image back: must equal the source bytes exactly.
bool roundtrip(Device& dev, const char* what, const TextureBuilderInfo& b, uint32_t seed) {
    Tex x;
    if (!make_tex(x, b)) {
        CHECK_MSG(false, what);
        return false;
    }
    fill_random(x.lin, seed);
    tile_into_storage(x);
    Texture* t = x.t.get();
    gpu::run_frame_sync([&](Frame& f) { gpu::tex_ensure_current(t, f); });
    std::vector<uint8_t> got;
    const bool read = read_back(dev, t, x.L, got);
    CHECK_MSG(read, what);
    const long diff = read ? first_diff(got, x.lin) : 0;
    CHECK_MSG(diff < 0, (std::string(what) + ": first differing byte " + std::to_string(diff)).c_str());
    std::printf("    %s: %zu bytes, %s\n", what, x.lin.size(), diff < 0 ? "bit exact" : "DIFFERS");
    return read && diff < 0;
}

// Per-case device: created with validation, torn down after every texture of the case is gone.
struct Rig {
    std::unique_ptr<Device> dev;
    uint64_t baseline_allocs = 0;
};

bool bring_up(Rig& r) {
    no_cache();
    DeviceDesc dd;
    dd.validation = 1;
    dd.headless = 1;
    std::string err;
    r.dev = Device::create(dd, &err);
    if (!r.dev) {
        std::printf("    device: %s\n", err.c_str());
        return false;
    }
    CHECK(r.dev->validation_enabled());
    r.baseline_allocs = r.dev->memory_stats().allocations;
    if (!gpu::init_for_test(*r.dev, nullptr)) {
        std::printf("    back end did not start\n");
        return false;
    }
    return true;
}

// Frees what the module keeps, shuts the back end down, then checks validation and the allocation count.
void teardown(Rig& r) {
    if (!r.dev) return;
    gpu::finish();
    gpu::run_on_render_thread([] { gpu::tex_shutdown(); });
    gpu::shutdown_for_test();
    const ValidationStats v = r.dev->validation_stats();
    std::printf("    validation: %llu errors, %llu warnings\n", static_cast<unsigned long long>(v.errors), static_cast<unsigned long long>(v.warnings));
    if (!v.first_error.empty()) std::printf("      first error: %s\n", v.first_error.c_str());
    CHECK_EQ(v.errors, 0ull);
    CHECK_EQ(v.warnings, 0ull);
    CHECK_EQ(r.dev->memory_stats().allocations, r.baseline_allocs);
    r.dev.reset();
}

}  // namespace

// ---- 1: RGBA8, four levels, every level bit exact -----------------------------------------------------------------------
NXTEST(upload_readback_rgba8_levels) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    {
        const bool ok = roundtrip(*r.dev, "RGBA8 64x64, 4 levels", info(NVN_TEXTURE_TARGET_2D, 64, 64, 1, 4, NVN_FORMAT_RGBA8), 1);
        CHECK(ok);
    }
    teardown(r);
}

// ---- 2: other uncompressed and block formats, bit exact ------------------------------------------------------------------
NXTEST(upload_readback_formats) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    {
        CHECK(roundtrip(*r.dev, "R8 32x16, 3 levels", info(NVN_TEXTURE_TARGET_2D, 32, 16, 1, 3, NVN_FORMAT_R8), 2));
        CHECK(roundtrip(*r.dev, "RGBA16F 40x24, 3 levels", info(NVN_TEXTURE_TARGET_2D, 40, 24, 1, 3, NVN_FORMAT_RGBA16F), 3));
        if (!r.dev->features().bc) {
            nxtest::skip("textureCompressionBC is not available on this GPU");
        } else {
            CHECK(roundtrip(*r.dev, "BC1 64x48, 3 levels", info(NVN_TEXTURE_TARGET_2D, 64, 48, 1, 3, NVN_FORMAT_RGBA_DXT1), 4));
            CHECK(roundtrip(*r.dev, "BC7 64x64, 4 levels", info(NVN_TEXTURE_TARGET_2D, 64, 64, 1, 4, NVN_FORMAT_BPTC_UNORM), 5));
        }
    }
    teardown(r);
}

// ---- 3: ASTC 5x4 becomes BC7 when the GPU has no ASTC -----------------------------------------------------------------------
NXTEST(astc_becomes_bc7) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    if (!r.dev->features().bc && !r.dev->features().astc_ldr) {
        nxtest::skip("neither BC nor ASTC is available on this GPU");
    } else {
        Tex x;
        if (make_tex(x, info(NVN_TEXTURE_TARGET_2D, 40, 32, 1, 1, NVN_FORMAT_RGBA_ASTC_5x4))) {
            // Constant-colour void-extent blocks (valid ASTC of every footprint, with a known colour).
            const tegra::LevelLayout& lv = x.L.layout.levels[0];
            std::mt19937 rng(7);
            for (uint32_t y = 0; y < lv.blocks_y; ++y)
                for (uint32_t bx = 0; bx < lv.blocks_x; ++bx) {
                    uint8_t* blk = x.lin.data() + lv.linear_offset + y * lv.linear_row_stride + bx * 16;
                    blk[0] = 0xFC;
                    blk[1] = 0xFD;
                    std::memset(blk + 2, 0xFF, 6);
                    uint16_t c[4];
                    for (uint16_t& v : c) v = static_cast<uint16_t>(rng());
                    std::memcpy(blk + 8, c, 8);
                }
            tile_into_storage(x);
            Texture* t = x.t.get();
            uint64_t uploads0 = 0, transcodes0 = 0, uploads1 = 0, transcodes1 = 0;
            VkFormat vk = VK_FORMAT_UNDEFINED;
            gpu::run_on_render_thread([&] {
                uploads0 = gpu::tex_stats().uploads;
                transcodes0 = gpu::tex_stats().transcodes;
            });
            gpu::run_frame_sync([&](Frame& f) {
                gpu::tex_ensure_current(t, f);
                vk = gpu::tex_vk_format(t);
            });
            gpu::run_on_render_thread([&] {
                uploads1 = gpu::tex_stats().uploads;
                transcodes1 = gpu::tex_stats().transcodes;
            });
            CHECK_EQ(uploads1 - uploads0, 1ull);
            if (r.dev->features().astc_ldr) {
                std::printf("    this GPU has ASTC: the image is native ASTC\n");
                CHECK_EQ(vk, VK_FORMAT_ASTC_5x4_UNORM_BLOCK);
                CHECK_EQ(transcodes1 - transcodes0, 0ull);
            } else {
                std::printf("    image format %d (BC7 UNORM is %d), transcodes %llu\n", static_cast<int>(vk), static_cast<int>(VK_FORMAT_BC7_UNORM_BLOCK),
                            static_cast<unsigned long long>(transcodes1 - transcodes0));
                CHECK(vk == VK_FORMAT_BC7_UNORM_BLOCK || vk == VK_FORMAT_BC7_SRGB_BLOCK);
                CHECK_EQ(transcodes1 - transcodes0, 1ull);
            }
        } else {
            CHECK(false);
        }
    }
    teardown(r);
}

// ---- 4: cube, 2D array and 3D views ------------------------------------------------------------------------------------------
NXTEST(views_cube_array_3d) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    {
        Tex cube, arr, plain;
        CHECK(make_tex(cube, info(NVN_TEXTURE_TARGET_CUBEMAP, 16, 16, 1, 1, NVN_FORMAT_RGBA8)));
        CHECK(make_tex(arr, info(NVN_TEXTURE_TARGET_2D_ARRAY, 16, 16, 4, 1, NVN_FORMAT_RGBA8)));
        CHECK(make_tex(plain, info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_RGBA8)));
        Texture* tc = cube.t.get();
        Texture* ta = arr.t.get();
        Texture* tp = plain.t.get();
        gpu::run_on_render_thread([&] {
            ViewRequest q;
            q.type = VK_IMAGE_VIEW_TYPE_CUBE;
            CHECK(gpu::tex_sampled_view(tc, q) != VK_NULL_HANDLE);
            q.type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            CHECK(gpu::tex_sampled_view(tc, q) != VK_NULL_HANDLE);
            CHECK(gpu::tex_sampled_view(ta, q) != VK_NULL_HANDLE);
            q.base_layer = 2;
            q.layer_count = 2;
            CHECK(gpu::tex_sampled_view(ta, q) != VK_NULL_HANDLE);
            q = ViewRequest();
            q.type = VK_IMAGE_VIEW_TYPE_2D;
            q.base_layer = 2;
            CHECK(gpu::tex_sampled_view(ta, q) != VK_NULL_HANDLE);
            q = ViewRequest();
            q.type = VK_IMAGE_VIEW_TYPE_3D;
            CHECK(gpu::tex_sampled_view(tp, q) == VK_NULL_HANDLE);
            q.type = VK_IMAGE_VIEW_TYPE_CUBE;
            CHECK(gpu::tex_sampled_view(tp, q) == VK_NULL_HANDLE);
            q.type = VK_IMAGE_VIEW_TYPE_2D;
            CHECK(gpu::tex_sampled_view(tp, q) != VK_NULL_HANDLE);
        });
    }
    {
        // A real 3D image: a 3D view and a 2D view of one slice.
        Tex vol;
        CHECK(make_tex(vol, info(NVN_TEXTURE_TARGET_3D, 8, 8, 4, 1, NVN_FORMAT_RGBA8)));
        Texture* tv = vol.t.get();
        gpu::run_on_render_thread([&] {
            ViewRequest q;
            q.type = VK_IMAGE_VIEW_TYPE_3D;
            CHECK(gpu::tex_sampled_view(tv, q) != VK_NULL_HANDLE);
            q.type = VK_IMAGE_VIEW_TYPE_2D;
            CHECK(gpu::tex_sampled_view(tv, q) != VK_NULL_HANDLE);
        });
    }
    teardown(r);
}

// ---- 5: the sampled view cache -------------------------------------------------------------------------------------------------
NXTEST(sampled_view_cache) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    {
        Tex x;
        CHECK(make_tex(x, info(NVN_TEXTURE_TARGET_2D, 32, 32, 1, 3, NVN_FORMAT_RGBA8)));
        Texture* t = x.t.get();
        gpu::run_on_render_thread([&] {
            ViewRequest q;
            q.type = VK_IMAGE_VIEW_TYPE_2D;
            const VkImageView a = gpu::tex_sampled_view(t, q);
            const VkImageView b = gpu::tex_sampled_view(t, q);
            CHECK(a != VK_NULL_HANDLE);
            CHECK(a == b);
            q.base_level = 1;
            q.level_count = 2;
            const VkImageView c = gpu::tex_sampled_view(t, q);
            CHECK(c != VK_NULL_HANDLE);
            CHECK(c != a);
            CHECK(gpu::tex_sampled_view(t, q) == c);
        });
    }
    teardown(r);
}

// ---- 6: depth and depth-stencil textures: no upload, attachment views -------------------------------------------------------------
NXTEST(depth_textures) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    {
        Tex d32, d24s8;
        CHECK(make_tex(d32, info(NVN_TEXTURE_TARGET_2D, 32, 32, 1, 1, NVN_FORMAT_DEPTH32F)));
        CHECK(make_tex(d24s8, info(NVN_TEXTURE_TARGET_2D, 32, 32, 1, 1, NVN_FORMAT_DEPTH24_STENCIL8)));
        Texture* td = d32.t.get();
        Texture* ts = d24s8.t.get();
        uint64_t uploads0 = 0, uploads1 = 0;
        gpu::run_on_render_thread([&] { uploads0 = gpu::tex_stats().uploads; });
        gpu::run_frame_sync([&](Frame& f) {
            gpu::tex_ensure_current(td, f);
            gpu::tex_ensure_current(ts, f);
        });
        gpu::run_on_render_thread([&] {
            uploads1 = gpu::tex_stats().uploads;
            CHECK(gpu::tex_is_depth(td));
            CHECK_EQ(gpu::tex_vk_format(td), VK_FORMAT_D32_SFLOAT);
            CHECK_EQ(gpu::tex_aspect(td), VK_IMAGE_ASPECT_DEPTH_BIT);
            CHECK(gpu::tex_attachment_view(td, 0, 0, 1, VK_FORMAT_UNDEFINED) != VK_NULL_HANDLE);
            const VkFormat f24 = gpu::tex_vk_format(ts);
            CHECK(f24 == VK_FORMAT_D24_UNORM_S8_UINT || f24 == VK_FORMAT_D32_SFLOAT_S8_UINT);
            CHECK_EQ(gpu::tex_aspect(ts), VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
            CHECK(gpu::tex_attachment_view(ts, 0, 0, 1, VK_FORMAT_UNDEFINED) != VK_NULL_HANDLE);
            ViewRequest q;
            q.type = VK_IMAGE_VIEW_TYPE_2D;
            q.depth_aspect = true;
            CHECK(gpu::tex_sampled_view(ts, q) != VK_NULL_HANDLE);
            q.depth_aspect = false;
            CHECK(gpu::tex_sampled_view(ts, q) != VK_NULL_HANDLE);
        });
        CHECK_EQ(uploads1 - uploads0, 0ull);
    }
    teardown(r);
}

// ---- 7: colour target transitions -----------------------------------------------------------------------------------------------
NXTEST(color_target_transitions) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    {
        Tex x;
        CHECK(make_tex(x, info(NVN_TEXTURE_TARGET_2D, 64, 32, 1, 1, NVN_FORMAT_RGBA8, NVN_TEXTURE_FLAGS_DISPLAY)));
        Texture* t = x.t.get();
        VkImageLayout after_first = VK_IMAGE_LAYOUT_UNDEFINED, after_back = VK_IMAGE_LAYOUT_UNDEFINED;
        gpu::run_frame_sync([&](Frame& f) {
            gpu::tex_transition(t, f, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            after_first = gpu::tex_layout(t);
            gpu::tex_transition(t, f, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            after_back = gpu::tex_layout(t);
        });
        CHECK_EQ(after_first, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        CHECK_EQ(after_back, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    teardown(r);
}

// ---- 8: samplers: distinct states, equal states share, border colours and anisotropy --------------------------------------------
NXTEST(samplers) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    gpu::run_on_render_thread([&] {
        std::set<VkSampler> distinct;
        for (int i = 0; i < 10; ++i) {
            SamplerInfo s;
            s.min_filter = i % 6;
            s.wrap_s = i < 6 ? NVN_WRAP_MODE_REPEAT : NVN_WRAP_MODE_CLAMP;
            distinct.insert(gpu::sampler_for(s));
        }
        CHECK_EQ(distinct.size(), 10u);
        SamplerInfo a, b;
        a.min_filter = NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR;
        b = a;
        CHECK(gpu::sampler_for(a) == gpu::sampler_for(b));
        CHECK(gpu::sampler_for(a) != VK_NULL_HANDLE);

        // Border colours: transparent black, opaque white, custom floats, and an integer border.
        SamplerInfo tb;
        tb.wrap_s = NVN_WRAP_MODE_CLAMP_TO_BORDER;
        SamplerInfo ow = tb;
        for (float& c : ow.border_color) c = 1.0f;
        SamplerInfo custom[4];
        const float vals[4] = {0.25f, 0.5f, 0.75f, 1.0f};
        for (int i = 0; i < 4; ++i) {
            custom[i] = tb;
            for (float& c : custom[i].border_color) c = vals[i];
        }
        SamplerInfo ints = tb;
        const int32_t iv[4] = {1, 2, 3, 4};
        std::memcpy(ints.border_color, iv, sizeof iv);
        ints.border_kind = 1;
        SamplerInfo aniso = tb;
        aniso.max_anisotropy = 16.0f;
        aniso.min_filter = NVN_MIN_FILTER_LINEAR;
        aniso.mag_filter = 1;

        const VkSampler dummy = gpu::dummy_sampler();
        CHECK(dummy != VK_NULL_HANDLE);
        std::set<VkSampler> bords;
        bords.insert(gpu::sampler_for(tb));
        bords.insert(gpu::sampler_for(ow));
        for (int i = 0; i < 4; ++i) bords.insert(gpu::sampler_for(custom[i]));
        bords.insert(gpu::sampler_for(ints));
        CHECK(bords.count(VK_NULL_HANDLE) == 0);
        CHECK(gpu::sampler_for(aniso) != VK_NULL_HANDLE);
        CHECK(gpu::sampler_for(aniso) == gpu::sampler_for(aniso));
    });
    teardown(r);
}

// ---- 8b: display tuning on the samplers: anisotropy and LOD bias only where the rules allow ---------------------------------------
NXTEST(sampler_tuning) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    const bool aniso_ok = r.dev->features().sampler_anisotropy;
    std::printf("    sampler_anisotropy on this device: %s\n", aniso_ok ? "yes" : "no");
    gpu::run_on_render_thread([&] {
        // Puts the tuning back to its defaults when the case ends, also when a check above failed.
        struct Reset {
            ~Reset() {
                gfx::tuning().anisotropy.store(0);
                gfx::tuning().lod_bias.store(0.0f);
            }
        } reset;
        auto set_tuning = [](int aniso, float bias) {
            gfx::tuning().anisotropy.store(aniso);
            gfx::tuning().lod_bias.store(bias);
        };

        // 1 and 2: linear-mipmap-linear filters take both the anisotropy and the LOD bias.
        SamplerInfo L;
        L.min_filter = NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR;
        L.mag_filter = NVN_MAG_FILTER_LINEAR;
        L.max_anisotropy = 1;
        set_tuning(0, 0.0f);
        const VkSampler base = gpu::sampler_for(L);
        CHECK(base != VK_NULL_HANDLE);

        set_tuning(8, 0.0f);
        const VkSampler aniso8 = gpu::sampler_for(L);
        if (aniso_ok) {
            CHECK(aniso8 != base);
        } else {
            std::printf("    anisotropy is not available on this GPU: the anisotropy handle check is skipped\n");
        }
        set_tuning(0, 0.0f);
        CHECK(gpu::sampler_for(L) == base);

        set_tuning(0, 0.5f);
        const VkSampler bias05 = gpu::sampler_for(L);
        CHECK(bias05 != base);
        CHECK(bias05 != aniso8);
        set_tuning(0, 0.0f);
        CHECK(gpu::sampler_for(L) == base);

        // 3: nearest filtering is never changed.
        SamplerInfo N;
        N.min_filter = NVN_MIN_FILTER_NEAREST;
        N.mag_filter = NVN_MAG_FILTER_NEAREST;
        set_tuning(0, 0.0f);
        const VkSampler nearest0 = gpu::sampler_for(N);
        CHECK(nearest0 != VK_NULL_HANDLE);
        set_tuning(16, -1.0f);
        CHECK(gpu::sampler_for(N) == nearest0);
        set_tuning(0, 0.0f);

        // 4: comparison samplers are never changed.
        SamplerInfo C = L;
        C.compare_mode = NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE;
        set_tuning(0, 0.0f);
        const VkSampler cmp0 = gpu::sampler_for(C);
        CHECK(cmp0 != VK_NULL_HANDLE);
        set_tuning(16, 0.0f);
        CHECK(gpu::sampler_for(C) == cmp0);
        set_tuning(0, 1.0f);
        CHECK(gpu::sampler_for(C) == cmp0);
        set_tuning(0, 0.0f);

        // 5: linear minification without mipmaps is changed by neither the anisotropy nor the LOD bias.
        SamplerInfo Lin;
        Lin.min_filter = NVN_MIN_FILTER_LINEAR;
        Lin.mag_filter = NVN_MAG_FILTER_LINEAR;
        set_tuning(0, 0.0f);
        const VkSampler lin0 = gpu::sampler_for(Lin);
        CHECK(lin0 != VK_NULL_HANDLE);
        set_tuning(16, 0.0f);
        CHECK(gpu::sampler_for(Lin) == lin0);
        set_tuning(0, 1.0f);
        CHECK(gpu::sampler_for(Lin) == lin0);
    });
    teardown(r);
}

// ---- 9: CPU rewrites are uploaded; a GPU write suppresses uploads until the CPU writes again ----------------------------------
NXTEST(cpu_dirty_and_gpu_written) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    {
        Tex x;
        CHECK(make_tex(x, info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_RGBA8)));
        Texture* t = x.t.get();
        fill_random(x.lin, 11);
        tile_into_storage(x);
        gpu::run_frame_sync([&](Frame& f) { gpu::tex_ensure_current(t, f); });
        std::vector<uint8_t> got;
        CHECK(read_back(*r.dev, t, x.L, got));
        CHECK(first_diff(got, x.lin) < 0);
        uint64_t u0 = 0, u1 = 0, u2 = 0;
        gpu::run_on_render_thread([&] { u0 = gpu::tex_stats().uploads; });

        // The CPU writes new data and marks it: the next use uploads it.
        fill_random(x.lin, 12);
        tile_into_storage(x);
        gpu::mark_texture_cpu_dirty(t);
        gpu::run_frame_sync([&](Frame& f) { gpu::tex_ensure_current(t, f); });
        gpu::run_on_render_thread([&] { u1 = gpu::tex_stats().uploads; });
        CHECK_EQ(u1 - u0, 1ull);
        CHECK(read_back(*r.dev, t, x.L, got));
        CHECK(first_diff(got, x.lin) < 0);

        // The GPU writes (marked as written): new CPU data without a mark is not uploaded, the GPU content stays.
        std::vector<uint8_t> gpu_content = x.lin;
        gpu::run_on_render_thread([&] { gpu::tex_mark_gpu_written(t); });
        fill_random(x.lin, 13);
        tile_into_storage(x);
        gpu::run_frame_sync([&](Frame& f) { gpu::tex_ensure_current(t, f); });
        gpu::run_on_render_thread([&] { u2 = gpu::tex_stats().uploads; });
        CHECK_EQ(u2 - u1, 0ull);
        CHECK(read_back(*r.dev, t, x.L, got));
        CHECK(first_diff(got, gpu_content) < 0);
        std::printf("    uploads: first %llu, after CPU write %llu, after GPU write %llu\n", static_cast<unsigned long long>(u0),
                    static_cast<unsigned long long>(u1), static_cast<unsigned long long>(u2));
    }
    teardown(r);
}

// ---- 10: download of a render target after a clear, untiled back to the clear colour ---------------------------------------------
NXTEST(download_render_target_after_clear) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    {
        Tex x;
        CHECK(make_tex(x, info(NVN_TEXTURE_TARGET_2D, 64, 64, 1, 1, NVN_FORMAT_RGBA8)));
        Texture* t = x.t.get();
        const float rgba[4] = {64.0f / 255.0f, 128.0f / 255.0f, 200.0f / 255.0f, 1.0f};
        bool downloaded = false;
        gpu::run_frame_sync([&](Frame& f) {
            gpu::tex_transition(t, f, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearColorValue c = {};
            std::memcpy(c.float32, rgba, sizeof rgba);
            const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(f.cb, gpu::tex_image(t), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &range);
            gpu::tex_transition(t, f, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            gpu::tex_mark_gpu_written(t);
        });
        downloaded = gpu::download_texture(t);
        CHECK(downloaded);
        if (downloaded) {
            std::vector<uint8_t> lin(static_cast<size_t>(x.L.layout.linear_size), 0);
            tegra::untile(x.L.desc, x.L.layout, x.mem.data(), lin.data());
            const uint8_t want[4] = {64, 128, 200, 255};
            size_t bad = 0;
            for (size_t i = 0; i + 4 <= static_cast<size_t>(64 * 64 * 4); i += 4)
                if (std::memcmp(&lin[i], want, 4) != 0) ++bad;
            CHECK_MSG(bad == 0, (std::to_string(bad) + " texels differ from the clear colour").c_str());
            std::printf("    downloaded: %s\n", bad == 0 ? "every texel is the clear colour" : "DIFFERS");
        }
    }
    teardown(r);
}

// ---- 11: a format the device cannot make: null, counted once --------------------------------------------------------------------
NXTEST(failure_no_format) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    {
        Tex x;
        CHECK(make_tex(x, info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_RGBA8)));
        static FormatInfo bad;
        bad = *format_info(NVN_FORMAT_RGBA8);
        bad.vk = VK_FORMAT_UNDEFINED;
        bad.vk_fallback = VK_FORMAT_UNDEFINED;
        x.t->fmt = &bad;
        Texture* t = x.t.get();
        gpu::run_on_render_thread([&] {
            const uint64_t f0 = gpu::tex_stats().failed;
            const uint64_t c0 = gpu::tex_stats().images;
            CHECK(gpu::tex_gpu(t) == nullptr);
            CHECK(gpu::tex_gpu(t) == nullptr);
            ViewRequest q;
            CHECK(gpu::tex_sampled_view(t, q) == VK_NULL_HANDLE);
            CHECK_EQ(gpu::tex_stats().failed - f0, 1ull);
            CHECK_EQ(gpu::tex_stats().images - c0, 0ull);
        });
        gpu::run_frame_sync([&](Frame& f) { gpu::tex_ensure_current(t, f); });
    }
    teardown(r);
}

// ---- 12: dummy views and the dummy sampler exist and are stable -------------------------------------------------------------------
NXTEST(dummy_views_and_sampler) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    gpu::run_on_render_thread([&] {
        const VkImageViewType types[] = {VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY,
                                         VK_IMAGE_VIEW_TYPE_3D};
        for (VkImageViewType ty : types) {
            const VkImageView v = gpu::dummy_view(ty, false);
            CHECK(v != VK_NULL_HANDLE);
            CHECK(gpu::dummy_view(ty, false) == v);
            for (int component : {2, 3}) {
                const VkImageView integer = gpu::dummy_view(ty, false, component);
                CHECK(integer != VK_NULL_HANDLE);
                CHECK(integer != v);
                CHECK(gpu::dummy_view(ty, false, component) == integer);
            }
        }
        CHECK(gpu::dummy_view(VK_IMAGE_VIEW_TYPE_2D, true) != VK_NULL_HANDLE);
        CHECK(gpu::dummy_sampler() != VK_NULL_HANDLE);
        CHECK(gpu::dummy_sampler() == gpu::dummy_sampler());
    });
    teardown(r);
}

// ---- 13: a 3D image: layout barriers cover every slice, no validation warnings ------------------------------------------------
NXTEST(transition_3d_texture_no_warnings) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    {
        Tex x;
        CHECK(make_tex(x, info(NVN_TEXTURE_TARGET_3D, 8, 8, 4, 3, NVN_FORMAT_RGBA8)));
        fill_random(x.lin, 21);
        tile_into_storage(x);
        Texture* t = x.t.get();
        VkImageLayout after_upload = VK_IMAGE_LAYOUT_UNDEFINED, after_src = VK_IMAGE_LAYOUT_UNDEFINED, after_back = VK_IMAGE_LAYOUT_UNDEFINED;
        gpu::run_frame_sync([&](Frame& f) {
            gpu::tex_ensure_current(t, f);
            after_upload = gpu::tex_layout(t);
            gpu::tex_transition(t, f, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            gpu::tex_transition(t, f, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            after_src = gpu::tex_layout(t);
            gpu::tex_transition(t, f, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            after_back = gpu::tex_layout(t);
        });
        CHECK_EQ(after_upload, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        CHECK_EQ(after_src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        CHECK_EQ(after_back, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const ValidationStats v = r.dev->validation_stats();
        std::printf("    3D 8x8x4 RGBA8, 3 levels: validation %llu errors, %llu warnings after the transitions\n",
                    static_cast<unsigned long long>(v.errors), static_cast<unsigned long long>(v.warnings));
        CHECK_EQ(v.errors, 0ull);
        CHECK_EQ(v.warnings, 0ull);
    }
    teardown(r);
}

// ---- 14: raw-bit aliases of 32-bit integer and float formats -------------------------------------------------------------------
NXTEST(sampled_view_float_alias_of_uint_texture) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    {
        Tex u, fl, rgba;
        CHECK(make_tex(u, info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_R32UI)));
        CHECK(make_tex(fl, info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_R32F)));
        CHECK(make_tex(rgba, info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_RGBA8)));
        Texture* tu = u.t.get();
        Texture* tf = fl.t.get();
        Texture* tr = rgba.t.get();
        bool explicit_float_ok = false, explicit_uint_ok = false, incompatible_rejected = false;
        gpu::run_on_render_thread([&] {
            ViewRequest q;
            q.component = 1;   // float shader over the uint texture: R32_SFLOAT alias
            const VkImageView vfloat = gpu::tex_sampled_view(tu, q);
            q.component = 3;   // uint shader: the image's own R32_UINT
            const VkImageView vuint = gpu::tex_sampled_view(tu, q);
            q.component = 0;   // unknown: the image's own format
            const VkImageView vown = gpu::tex_sampled_view(tu, q);
            CHECK(vfloat != VK_NULL_HANDLE);
            CHECK(vuint != VK_NULL_HANDLE);
            CHECK(vown != VK_NULL_HANDLE);
            CHECK(vfloat != vuint);
            q.component = 1;
            CHECK(gpu::tex_sampled_view(tu, q) == vfloat);
            // The reverse: a float texture read by a uint shader gets the R32_UINT alias.
            q.component = 3;
            const VkImageView fuint = gpu::tex_sampled_view(tf, q);
            q.component = 0;
            const VkImageView fown = gpu::tex_sampled_view(tf, q);
            CHECK(fuint != VK_NULL_HANDLE);
            CHECK(fown != VK_NULL_HANDLE);
            CHECK(fuint != fown);
            // Explicit views must also agree with the shader's sampled type.
            q.format = VK_FORMAT_R32_UINT;
            q.component = 1;
            explicit_float_ok = gpu::tex_sampled_view(tf, q) == fown;
            q.format = VK_FORMAT_R32_SFLOAT;
            q.component = 3;
            explicit_uint_ok = gpu::tex_sampled_view(tu, q) == vuint;
            // No raw-bit alias is implemented for this class: never return a float view to UINT.
            q.format = VK_FORMAT_UNDEFINED;
            incompatible_rejected = gpu::tex_sampled_view(tr, q) == VK_NULL_HANDLE;
            // Colour attachments of the uint texture render through the float alias; RGBA8 is unchanged.
            CHECK_EQ(gpu::tex_attachment_format(tu, VK_FORMAT_UNDEFINED), VK_FORMAT_R32_SFLOAT);
            CHECK(gpu::tex_attachment_view(tu, 0, 0, 1, VK_FORMAT_UNDEFINED) != VK_NULL_HANDLE);
            CHECK_EQ(gpu::tex_attachment_format(tr, VK_FORMAT_UNDEFINED), VK_FORMAT_R8G8B8A8_UNORM);
        });
        CHECK(explicit_float_ok);
        CHECK(explicit_uint_ok);
        CHECK(incompatible_rejected);
    }
    teardown(r);
}

// ---- 15: sampled layout by format: colour vs depth and stencil ----------------------------------------------------------------
NXTEST(sampled_layout_by_format) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    {
        Tex rgba, d32, d24s8;
        CHECK(make_tex(rgba, info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_RGBA8)));
        CHECK(make_tex(d32, info(NVN_TEXTURE_TARGET_2D, 32, 32, 1, 1, NVN_FORMAT_DEPTH32F)));
        CHECK(make_tex(d24s8, info(NVN_TEXTURE_TARGET_2D, 32, 32, 1, 1, NVN_FORMAT_DEPTH24_STENCIL8)));
        Texture* tr = rgba.t.get();
        Texture* td = d32.t.get();
        Texture* ts = d24s8.t.get();
        gpu::run_on_render_thread([&] {
            CHECK_EQ(gpu::tex_sampled_layout(tr), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            CHECK_EQ(gpu::tex_sampled_layout(td), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
            CHECK_EQ(gpu::tex_sampled_layout(ts), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        });
        VkImageLayout after = VK_IMAGE_LAYOUT_UNDEFINED;
        gpu::run_frame_sync([&](Frame& f) {
            gpu::tex_transition(td, f, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
            after = gpu::tex_layout(td);
        });
        CHECK_EQ(after, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        CHECK_EQ(after, gpu::tex_sampled_layout(td));
    }
    teardown(r);
}

// ---- 16: dummy storage views: types, formats, cache, cube as 2D array ----------------------------------------------------------
NXTEST(dummy_storage_view_types_and_cache) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    gpu::run_on_render_thread([&] {
        const VkImageViewType types[] = {VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_VIEW_TYPE_1D};
        const VkFormat formats[] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SFLOAT};
        std::vector<VkImageView> seen;
        for (VkImageViewType ty : types)
            for (VkFormat fm : formats) {
                const VkImageView v = gpu::dummy_storage_view(ty, fm);
                CHECK(v != VK_NULL_HANDLE);
                CHECK(gpu::dummy_storage_view(ty, fm) == v);
                seen.push_back(v);
            }
        // Each (type, format) pair is its own view.
        const std::set<VkImageView> distinct(seen.begin(), seen.end());
        CHECK_EQ(distinct.size(), seen.size());
        CHECK_EQ(distinct.size(), 12u);
        // Cube requests use the 2D array dummy of the same format.
        const VkImageView arr_u = gpu::dummy_storage_view(VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_FORMAT_R32_UINT);
        CHECK(gpu::dummy_storage_view(VK_IMAGE_VIEW_TYPE_CUBE, VK_FORMAT_R32_UINT) == arr_u);
        const VkImageView arr_c = gpu::dummy_storage_view(VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_FORMAT_R8G8B8A8_UNORM);
        CHECK(gpu::dummy_storage_view(VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, VK_FORMAT_R8G8B8A8_UNORM) == arr_c);
        // UNDEFINED format means RGBA8 UNORM.
        CHECK(gpu::dummy_storage_view(VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_UNDEFINED) == gpu::dummy_storage_view(VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM));
    });
    teardown(r);
}

// ---- 17: tex_has_storage: every uncompressed colour image has the storage usage ------------------------------------------------
NXTEST(has_storage_for_colour_images) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    {
        Tex img_on, img_off;
        CHECK(make_tex(img_on, info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_RGBA8, NVN_TEXTURE_FLAGS_IMAGE)));
        CHECK(make_tex(img_off, info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_RGBA8)));
        Texture* ton = img_on.t.get();
        Texture* toff = img_off.t.get();
        gpu::run_on_render_thread([&] {
            CHECK(gpu::tex_has_storage(ton));
            // the IMAGE flag is only a hint (the game does not set it on the fog volume a vertex shader writes): every uncompressed
            // colour image keeps the storage usage
            CHECK(gpu::tex_has_storage(toff));
        });
    }
    teardown(r);
}

// ---- 18: ASTC with a disk cache entry of RGBA8 or BC5: the image is made in the entry's format ----------------------------------

namespace fs = std::filesystem;

namespace {

namespace be = nxrt::nvn::be;

void put32(std::vector<uint8_t>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i)));
}
void put64(std::vector<uint8_t>& v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i)));
}
uint64_t get64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

// A new cache folder for one case, set as the cache folder (the env var wins over the config; no_cache() undoes it).
fs::path new_cache_dir(const char* tag) {
    const fs::path dir = fs::temp_directory_path() /
                         (std::string("nxrt_texgpu_") + tag + "_" +
                          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code ec;
    fs::create_directories(dir, ec);
#ifdef _WIN32
    _putenv_s("NXRT_NVN_BC7_CACHE", dir.string().c_str());
#else
    setenv("NXRT_NVN_BC7_CACHE", dir.string().c_str(), 1);
#endif
    return dir;
}

// Level table of a tight cache entry: RGBA8 texels (4 bytes each) or 4x4 blocks (16 bytes each); all layers of a level back to back.
std::vector<be::PreparedLevel> tight_table(const TexLayout& L, bool rgba8) {
    std::vector<be::PreparedLevel> v;
    uint64_t off = 0;
    for (const tegra::LevelLayout& lv : L.layout.levels) {
        be::PreparedLevel p{};
        p.width = lv.width;
        p.height = lv.height;
        p.depth = lv.depth;
        const uint64_t per_slice = rgba8 ? static_cast<uint64_t>(lv.width) * lv.height * 4
                                         : static_cast<uint64_t>((lv.width + 3) / 4) * ((lv.height + 3) / 4) * 16;
        p.offset = off;
        p.size = per_slice * lv.depth * L.layers;
        off += p.size;
        v.push_back(p);
    }
    return v;
}

// The RGBA8 image of a 2D texture (one layer, one slice): texel (x, y) of level i is (x, y, i, 255).
std::vector<uint8_t> rgba8_pattern(const TexLayout& L) {
    const std::vector<be::PreparedLevel> t = tight_table(L, true);
    std::vector<uint8_t> v(static_cast<size_t>(t.back().offset + t.back().size));
    for (size_t i = 0; i < t.size(); ++i)
        for (uint32_t y = 0; y < t[i].height; ++y)
            for (uint32_t x = 0; x < t[i].width; ++x) {
                uint8_t* px = v.data() + t[i].offset + (static_cast<uint64_t>(y) * t[i].width + x) * 4;
                px[0] = static_cast<uint8_t>(x);
                px[1] = static_cast<uint8_t>(y);
                px[2] = static_cast<uint8_t>(i);
                px[3] = 255;
            }
    return v;
}

// Overwrites a cache file with an entry in the layout cache_write uses: magic, version 1, key, VkFormat, level and layer counts,
// the level table, then the data.
void write_entry(const fs::path& file, uint64_t key, VkFormat format, const std::vector<be::PreparedLevel>& level, uint32_t layers,
                 const std::vector<uint8_t>& data) {
    std::vector<uint8_t> head;
    const uint8_t magic[4] = {'N', 'X', 'B', '7'};
    head.insert(head.end(), magic, magic + 4);
    put32(head, 1);
    put64(head, key);
    put32(head, static_cast<uint32_t>(format));
    put32(head, static_cast<uint32_t>(level.size()));
    put32(head, layers);
    for (const be::PreparedLevel& l : level) {
        put32(head, l.width);
        put32(head, l.height);
        put32(head, l.depth);
        put64(head, l.offset);
        put64(head, l.size);
    }
    std::ofstream f(file, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(head.data()), static_cast<std::streamsize>(head.size()));
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

// The one .bc7 file of a folder and its key (read from the header). False when the folder does not hold exactly one file.
bool single_entry(const fs::path& dir, fs::path& file, uint64_t& key) {
    std::vector<fs::path> files;
    std::error_code ec;
    for (const auto& e : fs::recursive_directory_iterator(dir, ec))
        if (e.is_regular_file(ec) && e.path().extension() == ".bc7") files.push_back(e.path());
    if (files.size() != 1) return false;
    file = files[0];
    std::ifstream f(file, std::ios::binary);
    uint8_t head[16];
    if (!f.read(reinterpret_cast<char*>(head), sizeof head)) return false;
    key = get64(head + 8);
    return true;
}

// Copies every level of the image into a host buffer at the offsets of `table` (tight texel rows). As read_back, the image is moved
// to TRANSFER_SRC and back to the layout it had.
bool read_back_tight(Device& dev, Texture* t, const std::vector<be::PreparedLevel>& table, uint32_t layers, std::vector<uint8_t>& out) {
    const uint64_t size = table.back().offset + table.back().size;
    out.assign(static_cast<size_t>(size), 0);
    gfx::Buffer rb;
    gpu::finish();
    gpu::run_frame_sync([&](Frame& f) {
        if (dev.create_buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUse::Readback, &rb, "test readback") != VK_SUCCESS) return;
        const VkImageLayout prev = gpu::tex_layout(t);
        gpu::tex_transition(t, f, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        std::vector<VkBufferImageCopy> regions;
        for (uint32_t l = 0; l < table.size(); ++l) {
            VkBufferImageCopy r = {};
            r.bufferOffset = table[l].offset;
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, layers};
            r.imageExtent = {table[l].width, table[l].height, table[l].depth};
            regions.push_back(r);
        }
        vkCmdCopyImageToBuffer(f.cb, gpu::tex_image(t), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buffer, static_cast<uint32_t>(regions.size()),
                               regions.data());
        if (prev != VK_IMAGE_LAYOUT_UNDEFINED) gpu::tex_transition(t, f, prev);
    });
    if (!rb) return false;
    dev.invalidate_buffer(rb);
    std::memcpy(out.data(), rb.mapped, static_cast<size_t>(size));
    dev.destroy_buffer(rb);
    return true;
}

// One ASTC 4x4 UNORM texture (40x32, 3 levels, constant-colour void-extent blocks) whose cache folder holds an entry of `entry` format
// (RGBA8 or BC5). prepare_texture writes the BC7 entry first; the hand-made entry replaces it; the upload must make the image in the
// entry's format. For RGBA8 the image is read back and must equal the entry's pattern. For BC5 the format and the upload are checked
// (validation is checked at teardown).
void cache_entry_case(const char* what, VkFormat entry) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    if (r.dev->features().astc_ldr) {
        nxtest::skip("this GPU has ASTC: the cache entries are not used");
    } else if (!r.dev->features().bc) {
        nxtest::skip("textureCompressionBC is not available on this GPU");
    } else {
        std::printf("    %s\n", what);
        const bool rgba8 = entry == VK_FORMAT_R8G8B8A8_UNORM;
        const fs::path dir = new_cache_dir(rgba8 ? "rgba8" : "bc5");
        Tex x;
        if (make_tex(x, info(NVN_TEXTURE_TARGET_2D, 40, 32, 1, 3, NVN_FORMAT_RGBA_ASTC_4x4))) {
            for (size_t l = 0; l < x.L.layout.levels.size(); ++l) {
                const tegra::LevelLayout& lv = x.L.layout.levels[l];
                std::mt19937 rng(static_cast<uint32_t>(7 + l));
                for (uint32_t y = 0; y < lv.blocks_y; ++y)
                    for (uint32_t bx = 0; bx < lv.blocks_x; ++bx) {
                        uint8_t* blk = x.lin.data() + lv.linear_offset + y * lv.linear_row_stride + bx * 16;
                        blk[0] = 0xFC;
                        blk[1] = 0xFD;
                        std::memset(blk + 2, 0xFF, 6);
                        uint16_t c[4];
                        for (uint16_t& v : c) v = static_cast<uint16_t>(rng());
                        std::memcpy(blk + 8, c, 8);
                    }
            }
            tile_into_storage(x);
            Texture* t = x.t.get();

            be::PreparedTexture prep;
            CHECK_MSG(be::prepare_texture(*t, be::TexCaps(), prep), prep.error);
            fs::path file;
            uint64_t key = 0;
            if (!single_entry(dir, file, key)) {
                CHECK_MSG(false, "expected one BC7 cache file after the first prepare");
            } else {
                const std::vector<be::PreparedLevel> table = tight_table(x.L, rgba8);
                std::vector<uint8_t> data;
                if (rgba8) {
                    data = rgba8_pattern(x.L);
                } else {
                    data.assign(static_cast<size_t>(table.back().offset + table.back().size), 0);
                    fill_random(data, 901);
                }
                write_entry(file, key, entry, table, x.L.layers, data);
                CHECK_EQ(be::cache_entry_format(*t), entry);

                uint64_t uploads0 = 0, uploads1 = 0;
                VkFormat vk = VK_FORMAT_UNDEFINED;
                gpu::run_on_render_thread([&] { uploads0 = gpu::tex_stats().uploads; });
                gpu::run_frame_sync([&](Frame& f) {
                    gpu::tex_ensure_current(t, f);
                    vk = gpu::tex_vk_format(t);
                });
                gpu::run_on_render_thread([&] { uploads1 = gpu::tex_stats().uploads; });
                CHECK_EQ(uploads1 - uploads0, 1ull);
                CHECK_EQ(vk, entry);
                std::printf("    image format %d (entry %d)\n", static_cast<int>(vk), static_cast<int>(entry));
                if (rgba8) {
                    std::vector<uint8_t> got;
                    CHECK_MSG(read_back_tight(*r.dev, t, table, x.L.layers, got), "RGBA8 readback failed");
                    CHECK_MSG(got == data, "RGBA8 image differs from the entry");
                }
            }
            std::error_code ec;
            fs::remove_all(dir, ec);
        }
        no_cache();
    }
    teardown(r);
}

}  // namespace

NXTEST(astc_with_rgba8_and_bc5_cache_entries) {
    cache_entry_case("RGBA8 cache entry: the image is R8G8B8A8_UNORM and reads back as the entry", VK_FORMAT_R8G8B8A8_UNORM);
    cache_entry_case("BC5 cache entry: the image is BC5_UNORM and the upload completes", VK_FORMAT_BC5_UNORM_BLOCK);
}

// ---- shutdown with a texture still alive: no pointer to a freed image stays behind --------------------------------------------------
// Regression for the validation error "vkCmdPipelineBarrier2: pImageMemoryBarriers[0].image is VK_NULL_HANDLE" (and the crashes behind it)
// in gfx_nvn_scene_state: a texture the game (or a test) never finalizes stays in the registry when the back end shuts down, and
// gpu::tex_shutdown frees every image. The texture kept its pointer to the freed image; the next texture made over the same memory with
// the same shape found it as an alias and used freed memory.
NXTEST(shutdown_detaches_live_textures_from_freed_images) {
    const TextureBuilderInfo b = info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_RGBA8);
    Tex a;
    {
        Rig r;
        if (!bring_up(r)) {
            CHECK(false);
            return;
        }
        REQUIRE(make_tex(a, b));
        fill_random(a.lin, 7);
        tile_into_storage(a);
        be::texture_registry_add(a.t.get());
        gpu::run_frame_sync([&](Frame& f) { gpu::tex_ensure_current(a.t.get(), f); });
        CHECK(a.t->gpu != nullptr);
        teardown(r);   // a is still alive and registered
        CHECK(a.t->gpu == nullptr);
    }
    {
        Rig r;
        if (!bring_up(r)) {
            CHECK(false);
            return;
        }
        Tex c;
        REQUIRE(make_tex(c, b));
        c.t->mem = a.mem.data();   // the same memory and shape as a: the registry offers a as an alias
        c.lin = a.lin;
        be::texture_registry_add(c.t.get());
        gpu::run_frame_sync([&](Frame& f) { gpu::tex_ensure_current(c.t.get(), f); });
        std::vector<uint8_t> got;
        CHECK(read_back(*r.dev, c.t.get(), c.L, got));
        CHECK_MSG(first_diff(got, a.lin) < 0, "the new texture holds the content of the storage");
        be::texture_registry_remove(c.t.get());
        c.t = Ref<Texture>();
        teardown(r);
    }
    be::texture_registry_remove(a.t.get());
}

// ---- background preparation (S21, src/gfx/nvn/nvn_texprep.cpp) -------------------------------------------------------------
// A bound texture is prepared on a worker; the render thread uses the result when the storage is unchanged and prepares the texture itself when the storage
// changed under the job (content hash) or the CPU reported a write (generation). Either way the image holds exactly what the storage holds at the first use.
NXTEST(texprep_background_upload) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    if (!r.dev->features().bc) {
        nxtest::skip("textureCompressionBC is not available on this GPU");
        teardown(r);
        return;
    }
    namespace tp = nxrt::nvn::be::texprep;
    {
        const TextureBuilderInfo b = info(NVN_TEXTURE_TARGET_2D, 256, 256, 1, 5, NVN_FORMAT_BPTC_UNORM);
        auto upload_and_compare = [&](Tex& x, const char* what) {
            Texture* t = x.t.get();
            gpu::run_frame_sync([&](Frame& f) { gpu::tex_ensure_current(t, f); });
            std::vector<uint8_t> got;
            const bool read = read_back(*r.dev, t, x.L, got);
            CHECK_MSG(read, what);
            const long diff = read ? first_diff(got, x.lin) : 0;
            CHECK_MSG(diff < 0, (std::string(what) + ": first differing byte " + std::to_string(diff)).c_str());
        };
        // 1. bound, prepared in the background, used as prepared
        Tex a;
        CHECK(make_tex(a, b));
        fill_random(a.lin, 11);
        tile_into_storage(a);
        const tp::Stats s0 = tp::stats();
        gpu::texture_bound(a.t.get());
        tp::wait_idle_for_test();
        upload_and_compare(a, "bound, prepared in the background");
        const tp::Stats s1 = tp::stats();
        CHECK_EQ(s1.hits - s0.hits, 1ull);
        // 2. the storage changes after the job without any notification: the content hash catches it
        Tex c;
        CHECK(make_tex(c, b));
        fill_random(c.lin, 12);
        tile_into_storage(c);
        gpu::texture_bound(c.t.get());
        tp::wait_idle_for_test();
        fill_random(c.lin, 13);
        tile_into_storage(c);
        upload_and_compare(c, "storage changed after the job, no notification");
        const tp::Stats s2 = tp::stats();
        CHECK_EQ(s2.hits - s1.hits, 0ull);
        CHECK(s2.stale > s1.stale);
        // 3. the CPU reports a write after the job: the generation catches it
        Tex d;
        CHECK(make_tex(d, b));
        fill_random(d.lin, 14);
        tile_into_storage(d);
        gpu::texture_bound(d.t.get());
        tp::wait_idle_for_test();
        fill_random(d.lin, 15);
        tile_into_storage(d);
        gpu::mark_texture_cpu_dirty(d.t.get());
        upload_and_compare(d, "write reported after the job");
        const tp::Stats s3 = tp::stats();
        CHECK_EQ(s3.hits - s2.hits, 0ull);
        // 4. a rewrite after the first upload is prepared in the background again once the texture is bound
        fill_random(a.lin, 16);
        tile_into_storage(a);
        gpu::mark_texture_cpu_dirty(a.t.get());
        gpu::texture_bound(a.t.get());
        tp::wait_idle_for_test();
        upload_and_compare(a, "rewritten, bound again");
        const tp::Stats s4 = tp::stats();
        CHECK_EQ(s4.hits - s3.hits, 1ull);
        // 5. many textures at once, each used exactly as the storage holds it
        std::vector<Tex> many(24);
        for (size_t i = 0; i < many.size(); ++i) {
            CHECK(make_tex(many[i], b));
            fill_random(many[i].lin, 100 + static_cast<uint32_t>(i));
            tile_into_storage(many[i]);
            gpu::texture_bound(many[i].t.get());
        }
        for (size_t i = 0; i < many.size(); ++i) upload_and_compare(many[i], "burst of 24 textures");
        // 6. a big texture comes with an upload buffer that the job already filled
        Tex big;
        CHECK(make_tex(big, info(NVN_TEXTURE_TARGET_2D, 1024, 1024, 1, 5, NVN_FORMAT_BPTC_UNORM)));
        fill_random(big.lin, 200);
        tile_into_storage(big);
        const tp::Stats s6 = tp::stats();
        gpu::texture_bound(big.t.get());
        tp::wait_idle_for_test();
        upload_and_compare(big, "big texture, upload buffer filled by the job");
        CHECK_EQ(tp::stats().hits - s6.hits, 1ull);
    }
    teardown(r);
}

// [nvn] tex_async = standin with standin_rows: a texture with no background result gets a compact stand-in (its big levels copied in block row by
// block row from their distinct rows); the image must hold exactly the whole stand-in. Its job's result then replaces it; an upload asked for while
// the job is not done keeps the stand-in (standin_keep) instead of preparing the texture on the render thread.
NXTEST(compact_standin_upload_and_replacement) {
    Rig r;
    if (!bring_up(r)) {
        CHECK(false);
        return;
    }
    if (!r.dev->features().bc || r.dev->features().astc_ldr) {
        nxtest::skip("needs BC and no native ASTC (the stand-in path)");
        teardown(r);
        return;
    }
    no_cache();
    namespace tp = nxrt::nvn::be::texprep;
    for (int target : {NVN_TEXTURE_TARGET_2D, NVN_TEXTURE_TARGET_2D_ARRAY}) {
        const bool arr = target == NVN_TEXTURE_TARGET_2D_ARRAY;
        Tex x;
        CHECK(make_tex(x, info(target, arr ? 300 : 1024, arr ? 200 : 512, arr ? 2 : 1, arr ? 7 : 11, NVN_FORMAT_RGBA_ASTC_8x8)));
        std::mt19937 rng(arr ? 77 : 78);
        for (uint32_t l = 0; l < x.L.layout.levels.size(); ++l) {
            const tegra::LevelLayout& lv = x.L.layout.levels[l];
            for (uint32_t layer = 0; layer < x.L.layers; ++layer)
                for (uint32_t y = 0; y < lv.blocks_y; ++y)
                    for (uint32_t bx = 0; bx < lv.blocks_x; ++bx) {
                        uint8_t* blk = x.lin.data() + lv.linear_offset + layer * lv.linear_slice_size + y * lv.linear_row_stride + bx * 16;
                        blk[0] = 0xFC;
                        blk[1] = 0xFD;
                        std::memset(blk + 2, 0xFF, 6);
                        uint16_t c[4];
                        for (uint16_t& v : c) v = static_cast<uint16_t>(rng());
                        std::memcpy(blk + 8, c, 8);
                    }
        }
        tile_into_storage(x);
        Texture* t = x.t.get();
        // what the image must hold: the whole stand-in, and later the full transcode
        be::TexCaps si;
        si.provisional = true;
        si.standin_first = true;
        si.image_format = VK_FORMAT_BC7_UNORM_BLOCK;
        be::PreparedTexture want_si, want_full;
        CHECK_MSG(be::prepare_texture(*t, si, want_si), want_si.error);
        CHECK_MSG(be::prepare_texture(*t, be::TexCaps(), want_full), want_full.error);
        uint64_t up0 = 0, up1 = 0, up2 = 0;
        gpu::run_on_render_thread([&] { up0 = gpu::tex_stats().uploads; });
        gpu::run_frame_sync([&](Frame& f) { gpu::tex_ensure_current(t, f); });
        gpu::run_on_render_thread([&] { up1 = gpu::tex_stats().uploads; });
        CHECK_EQ(up1 - up0, 1ull);
        CHECK(tp::standin_on_screen(t));
        std::vector<uint8_t> got;
        CHECK_MSG(read_back_tight(*r.dev, t, want_si.level, x.L.layers, got), "stand-in readback failed");
        const long d0 = first_diff(got, want_si.data);
        CHECK_MSG(d0 < 0, (std::string(arr ? "array" : "2D") + ": the image differs from the whole stand-in at byte " + std::to_string(d0)).c_str());
        // asked again before the job is done (cpu_dirty without a write): the stand-in stays, or the job's result is already there
        t->cpu_dirty.store(true);
        gpu::run_frame_sync([&](Frame& f) { gpu::tex_ensure_current(t, f); });
        gpu::run_on_render_thread([&] { up2 = gpu::tex_stats().uploads; });
        std::printf("    %s: second upload %s\n", arr ? "array" : "2D", up2 == up1 ? "kept the stand-in" : "took the job's result");
        tp::wait_idle_for_test();
        gpu::run_frame_sync([&](Frame& f) { gpu::tex_ensure_current(t, f); });
        CHECK(!tp::standin_on_screen(t));
        CHECK_MSG(read_back_tight(*r.dev, t, want_full.level, x.L.layers, got), "final readback failed");
        const long d1 = first_diff(got, want_full.data);
        CHECK_MSG(d1 < 0, (std::string(arr ? "array" : "2D") + ": the image differs from the full transcode at byte " + std::to_string(d1)).c_str());
    }
    teardown(r);
}

// ---- sparse texture helpers used by the background preparation (nvn_texture_registry.cpp) ----------------------------------------------
// gather copies the mapped tiles (unmapped bytes are zero), direct hands out a pointer when the range is one piece of one physical pool, and
// the hash follows both the bytes and where they sit.
NXTEST(sparse_texture_segments) {
    using nxrt::nvn::be::MemoryPool;
    using nxrt::nvn::be::texture_gather_virtual;
    using nxrt::nvn::be::texture_hash_virtual;
    using nxrt::nvn::be::texture_virtual_direct;
    std::vector<uint8_t> phys_a(1 << 20), phys_b(1 << 20);
    fill_random(phys_a, 21);
    fill_random(phys_b, 22);
    Ref<MemoryPool> pa(new MemoryPool()), pb(new MemoryPool()), vp(new MemoryPool());
    pa->base = phys_a.data();
    pa->size = phys_a.size();
    pb->base = phys_b.data();
    pb->size = phys_b.size();
    vp->is_virtual = true;
    vp->size = 8u << 20;
    Texture t;
    t.vpool = vp;
    t.voffset = 256 * 1024;
    t.storage_size = 512 * 1024;
    auto map = [&](uint64_t voff, uint64_t size, Ref<MemoryPool> phys, int64_t poff) {
        MemoryPool::VirtualMap m;
        m.voff = voff;
        m.size = size;
        m.phys = phys;
        m.poff = poff;
        vp->vmaps.push_back(m);
    };
    std::vector<uint8_t> got;
    Ref<MemoryPool> keep;
    uint64_t h0 = 0, h1 = 0, h2 = 0;
    // nothing mapped: all zeros, no direct pointer
    CHECK(texture_gather_virtual(t, got));
    CHECK_EQ(got.size(), static_cast<size_t>(512 * 1024));
    CHECK(std::all_of(got.begin(), got.end(), [](uint8_t b) { return b == 0; }));
    CHECK(texture_virtual_direct(t, keep) == nullptr);
    CHECK(texture_hash_virtual(t, h0));
    // the whole range from one pool: direct points into it, gather equals it
    map(0, 4u << 20, pa, 4096);
    const uint8_t* d = texture_virtual_direct(t, keep);
    CHECK(d != nullptr);
    CHECK(d == phys_a.data() + 4096 + 256 * 1024);
    CHECK(texture_gather_virtual(t, got));
    CHECK(std::memcmp(got.data(), d, got.size()) == 0);
    CHECK(texture_hash_virtual(t, h1));
    CHECK(h1 != h0);
    // a byte changes: the hash moves
    phys_a[4096 + 256 * 1024 + 1000] ^= 0x5a;
    CHECK(texture_hash_virtual(t, h2));
    CHECK(h2 != h1);
    // two pieces: no direct pointer, gather stitches them, the hash differs from the one-piece mapping
    vp->vmaps.clear();
    map(0, 256 * 1024 + 128 * 1024, pa, 0);
    map(256 * 1024 + 128 * 1024, 4u << 20, pb, 0);
    CHECK(texture_virtual_direct(t, keep) == nullptr);
    CHECK(texture_gather_virtual(t, got));
    CHECK(std::memcmp(got.data(), phys_a.data() + 256 * 1024, 128 * 1024) == 0);
    CHECK(std::memcmp(got.data() + 128 * 1024, phys_b.data(), 384 * 1024) == 0);
    uint64_t h3 = 0;
    CHECK(texture_hash_virtual(t, h3));
    CHECK(h3 != h2);
    CHECK(h3 != h1);
    // a texture that is not sparse has no virtual hash
    Texture plain;
    uint64_t hp = 0;
    CHECK(!texture_hash_virtual(plain, hp));
}
