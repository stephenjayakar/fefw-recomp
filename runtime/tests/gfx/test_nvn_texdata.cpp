// SPDX-License-Identifier: GPL-3.0-or-later
// Tests of the CPU-side texture data preparation (src/gfx/nvn/nvn_texdata.cpp): untiling, pass-through of BCn, the pitch-linear
// copy, ASTC transcode to BC7 and decode to RGBA8 (through the astc2bc7 library), the disk cache, the caps and the failure paths.
// Textures are built by hand (no NVN pool): the image is tiled with tegra::tile into the storage, as the game's data sits in
// memory, and prepare_texture must give the image back.
// ASTC test blocks are constant-colour void-extent blocks (0xFC 0xFD 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF, then the RGBA16 colour):
// valid blocks of every footprint, with a known colour. The BC7 output is checked for format, sizes, layout and cache, and
// the RGBA8 decode is checked against the library's own decode. No BC7 pixel decoder is available here, so the BC7 pixels
// themselves are not compared (the tools/textures oracle does that).
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <nxrt/config.h>

#include "astc2bc7.h"
#include "nvn_core.h"
#include "nvn_format.h"
#include "nvn_texdata.h"
#include "nvn_texture_layout.h"
#ifdef NXRT_HAVE_BC7_REF
#include "bc7decomp.h"
#endif
#include "tegra_tiling.h"
#include "test_framework.h"
NXTEST_GROUP(gfx_nvn_texdata)

using namespace nxrt;
using namespace nxrt::nvn;
using namespace nxrt::nvn::be;
namespace fs = std::filesystem;

namespace {

uint64_t div_up(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

void fill_random(std::vector<uint8_t>& v, uint32_t seed) {
    std::mt19937 rng(seed);
    for (uint8_t& b : v) b = static_cast<uint8_t>(rng());
}

// Sets an environment variable; an empty value removes it (both on Windows and POSIX).
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (*value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

// The disk cache off for a test: no file may be written into the tree. A developer's environment setting is removed too.
void no_cache() {
    set_env("NXRT_NVN_BC7_CACHE", "");
    config_set("nvn", "bc7_cache", "");
}

TextureBuilderInfo make_info(int target, int w, int h, int depth, int levels, int format, int flags = 0, ptrdiff_t stride = 0) {
    TextureBuilderInfo b;
    b.target = target;
    b.width = w;
    b.height = h;
    b.depth = depth;
    b.levels = levels;
    b.format = format;
    b.flags = flags;
    b.stride = stride;
    return b;
}

// A texture built by hand: its layout, the untiled image (ground truth) and the storage the game would hold.
struct Built {
    Ref<Texture> tex;
    TexLayout L;
    std::vector<uint8_t> lin;   // the image: levels back to back, layers, z slices, tight rows
    std::vector<uint8_t> mem;   // storage: tiled (block-linear) or pitch-padded (pitch-linear)
};

void make_texture(Built& bt, const TextureBuilderInfo& info) {
    bt.tex = Ref<Texture>(new Texture());
    Texture& t = *bt.tex;
    t.b = info;
    t.fmt = bt.L.fmt;
    t.mem = bt.mem.data();
    t.storage_size = bt.L.storage_size;
    t.block_linear = bt.L.block_linear;
    t.levels = bt.L.levels;
    t.layers = bt.L.layers;
    t.block_height_gobs = bt.L.block_linear ? bt.L.layout.levels[0].gob_blocks_y : 0;
}

// Block-linear texture: the image bt.lin is zero, the caller fills it, then finish_block_linear tiles it into the storage.
bool begin_block_linear(Built& bt, const TextureBuilderInfo& info) {
    bt.L = compute_texture_layout(info);
    if (!bt.L.valid || !bt.L.block_linear) return false;
    bt.lin.assign(bt.L.layout.linear_size, 0);
    return true;
}

void finish_block_linear(Built& bt, const TextureBuilderInfo& info) {
    bt.mem.assign(bt.L.storage_size, 0);
    tegra::tile(bt.L.desc, bt.L.layout, bt.lin.data(), bt.mem.data());
    make_texture(bt, info);
}

// Pitch-linear texture (2D, one level): the storage is random, padding included; the image is its tight rows.
void build_pitch(Built& bt, const TextureBuilderInfo& info, uint32_t seed) {
    bt.L = compute_texture_layout(info);
    CHECK(bt.L.valid && !bt.L.block_linear);
    bt.mem.assign(bt.L.storage_size, 0);
    fill_random(bt.mem, seed);
    const uint64_t bx = div_up(static_cast<uint64_t>(info.width), bt.L.fmt->block_w);
    const uint64_t by = div_up(static_cast<uint64_t>(info.height), bt.L.fmt->block_h);
    const uint64_t row = bx * bt.L.fmt->bytes;
    bt.lin.assign(row * by * bt.L.layers, 0);
    for (uint32_t layer = 0; layer < bt.L.layers; ++layer)
        for (uint64_t y = 0; y < by; ++y)
            std::memcpy(bt.lin.data() + layer * row * by + y * row,
                        bt.mem.data() + layer * bt.L.pitch * by + y * bt.L.pitch, static_cast<size_t>(row));
    make_texture(bt, info);
}

// A texture with storage but no image, for the failure paths.
Ref<Texture> bare_texture(const TextureBuilderInfo& info, std::vector<uint8_t>& buf) {
    buf.assign(1 << 16, 0);
    Ref<Texture> t(new Texture());
    t->b = info;
    t->mem = buf.data();
    t->storage_size = buf.size();
    return t;
}

// The image comes back: out holds the same levels and layers, each layer at its place in the untiled buffer.
void expect_image(const Built& bt, const PreparedTexture& out, const char* what) {
    CHECK_MSG(out.error.empty(), out.error);
    CHECK_EQ(out.levels, bt.L.levels);
    CHECK_EQ(out.layers, bt.L.layers);
    CHECK(!out.transcoded);
    CHECK(!out.from_cache);
    CHECK_MSG(out.data.size() == bt.lin.size(), what);
    for (size_t i = 0; i < bt.L.layout.levels.size() && i < out.level.size(); ++i) {
        const tegra::LevelLayout& lv = bt.L.layout.levels[i];
        CHECK_EQ(out.level[i].offset, lv.linear_offset);
        CHECK_EQ(out.level[i].size, lv.linear_slice_size * bt.L.layers);
        CHECK_EQ(out.level[i].width, lv.width);
        CHECK_EQ(out.level[i].height, lv.height);
        CHECK_EQ(out.level[i].depth, lv.depth);
        for (uint32_t layer = 0; layer < bt.L.layers; ++layer) {
            const uint64_t at = lv.linear_offset + layer * lv.linear_slice_size;
            CHECK_MSG(std::memcmp(out.data.data() + at, bt.lin.data() + at, static_cast<size_t>(lv.linear_slice_size)) == 0,
                      what);
        }
    }
}

// ASTC: fills layer `layer` of level `level` with constant-colour void-extent blocks; colours gets the RGBA16 of each block.
void fill_astc_void_level(Built& bt, size_t level, uint32_t layer, std::vector<uint16_t>& colours, uint32_t seed) {
    const tegra::LevelLayout& lv = bt.L.layout.levels[level];
    std::mt19937 rng(seed);
    colours.assign(static_cast<size_t>(lv.blocks_x) * lv.blocks_y * 4, 0);
    for (uint32_t y = 0; y < lv.blocks_y; ++y)
        for (uint32_t x = 0; x < lv.blocks_x; ++x) {
            uint8_t* blk = bt.lin.data() + lv.linear_offset + layer * lv.linear_slice_size + y * lv.linear_row_stride + x * 16;
            blk[0] = 0xFC;
            blk[1] = 0xFD;
            std::memset(blk + 2, 0xFF, 6);
            uint16_t c[4];
            for (uint16_t& v : c) v = static_cast<uint16_t>(rng());
            std::memcpy(blk + 8, c, 8);
            for (int i = 0; i < 4; ++i) colours[(static_cast<size_t>(y) * lv.blocks_x + x) * 4 + i] = c[i];
        }
}

// The RGBA8 decode of every level and layer must equal the library's own decode of the same ASTC slice.
void expect_decode_matches_library(const Built& bt, const PreparedTexture& d) {
    const FormatInfo* fmt = bt.L.fmt;
    for (size_t i = 0; i < bt.L.layout.levels.size(); ++i) {
        const tegra::LevelLayout& lv = bt.L.layout.levels[i];
        a2b_input in{};
        in.format = fmt->srgb ? A2B_ASTC_SRGB : A2B_ASTC_UNORM;
        in.width = lv.width;
        in.height = lv.height;
        in.block_w = fmt->block_w;
        in.block_h = fmt->block_h;
        const size_t slice_bytes = static_cast<size_t>(lv.linear_slice_size / lv.depth);
        const size_t pixel_bytes = static_cast<size_t>(lv.width) * lv.height * 4;
        std::vector<uint8_t> ref(pixel_bytes);
        for (uint32_t layer = 0; layer < bt.L.layers; ++layer)
            for (uint32_t z = 0; z < lv.depth; ++z) {
                const uint64_t src = lv.linear_offset + layer * lv.linear_slice_size + z * slice_bytes;
                CHECK_EQ(a2b_decode_astc_rgba8(&in, bt.lin.data() + src, slice_bytes, 0, ref.data(), nullptr), A2B_OK);
                const uint64_t dst = d.level[i].offset + (static_cast<uint64_t>(layer) * lv.depth + z) * pixel_bytes;
                CHECK_MSG(std::memcmp(d.data.data() + dst, ref.data(), pixel_bytes) == 0, "RGBA8 decode differs from library");
            }
    }
}

// Each BC7 slice must be the library's transcode of the same ASTC slice (placement of every slice in the output). The
// options are the defaults of the back end: [nvn] bc7_quality (bc7_fast_preset), all threads, no perceptual weights.
void expect_bc7_matches_library(const Built& bt, const PreparedTexture& out) {
    const FormatInfo* fmt = bt.L.fmt;
    a2b_options opt{};
    opt.quality = bc7_fast_preset();
    opt.threads = 0;
    opt.perceptual = 0;
    opt.allow_opaque_mode6 = 0;
    for (size_t i = 0; i < bt.L.layout.levels.size(); ++i) {
        const tegra::LevelLayout& lv = bt.L.layout.levels[i];
        a2b_input in{};
        in.format = fmt->srgb ? A2B_ASTC_SRGB : A2B_ASTC_UNORM;
        in.width = lv.width;
        in.height = lv.height;
        in.block_w = fmt->block_w;
        in.block_h = fmt->block_h;
        const size_t in_slice = static_cast<size_t>(lv.linear_slice_size / lv.depth);
        const size_t out_slice = static_cast<size_t>(a2b_output_size(&in));
        std::vector<uint8_t> ref(out_slice);
        for (uint32_t layer = 0; layer < bt.L.layers; ++layer)
            for (uint32_t z = 0; z < lv.depth; ++z) {
                const uint64_t src = lv.linear_offset + layer * lv.linear_slice_size + z * in_slice;
                a2b_format of{};
                CHECK_EQ(a2b_transcode(&in, bt.lin.data() + src, in_slice, &opt, &of, ref.data(), out_slice, nullptr), A2B_OK);
                const uint64_t dst = out.level[i].offset + (static_cast<uint64_t>(layer) * lv.depth + z) * out_slice;
                CHECK_MSG(std::memcmp(out.data.data() + dst, ref.data(), out_slice) == 0, "BC7 slice differs from library");
            }
    }
}

// Constant-colour blocks decode to their colour at the origin pixel of each block (within one 8-bit step, linear formats).
void expect_void_colours(const uint8_t* rgba, uint32_t width, uint32_t bw, uint32_t bh, uint32_t bx, uint32_t by,
                         const std::vector<uint16_t>& col) {
    for (uint32_t y = 0; y < by; ++y)
        for (uint32_t x = 0; x < bx; ++x) {
            const size_t px = ((static_cast<size_t>(y) * bh) * width + static_cast<size_t>(x) * bw) * 4;
            for (int c = 0; c < 4; ++c) {
                const int want = (col[(static_cast<size_t>(y) * bx + x) * 4 + c] + 128) / 257;
                const int got = rgba[px + c];
                CHECK_MSG(std::abs(got - want) <= 1, "void-extent colour differs");
            }
        }
}

std::vector<fs::path> cache_files(const fs::path& dir) {
    std::vector<fs::path> files;
    std::error_code ec;
    if (!fs::exists(dir, ec)) return files;
    for (const auto& e : fs::recursive_directory_iterator(dir, ec))
        if (e.is_regular_file(ec) && e.path().extension() == ".bc7") files.push_back(e.path());
    return files;
}

}  // namespace

// ---- untiling and pass-through ----------------------------------------------------------------------------------------------

NXTEST(untile_rgba8_r8_2d_array_six_levels) {
    no_cache();
    const int formats[] = {NVN_FORMAT_RGBA8, NVN_FORMAT_R8};
    for (int f : formats) {
        // 2D array with 4 layers, and a plain 2D texture: 100x37, 6 levels (odd sizes, the tail levels are tiny)
        for (int target : {NVN_TEXTURE_TARGET_2D_ARRAY, NVN_TEXTURE_TARGET_2D}) {
            const TextureBuilderInfo info = make_info(target, 100, 37, target == NVN_TEXTURE_TARGET_2D_ARRAY ? 4 : 1, 6, f);
            Built bt;
            CHECK_MSG(begin_block_linear(bt, info), "layout");
            fill_random(bt.lin, 1000 + static_cast<uint32_t>(f) + static_cast<uint32_t>(target));
            finish_block_linear(bt, info);
            PreparedTexture out;
            const bool ok = prepare_texture(*bt.tex, TexCaps(), out);
            CHECK_MSG(ok, out.error);
            CHECK_EQ(out.format, bt.L.fmt->vk);
            CHECK_EQ(out.block_w, 1u);
            CHECK_EQ(out.block_bytes, bt.L.fmt->bytes);
            expect_image(bt, out, "untile differs");
        }
    }
}

// A sparse texture (no storage of its own, mem null, vpool set): its storage is gathered from the tiles mapped into its virtual pool. The image is
// tiled as usual into a 64 KiB physical tile, which the virtual pool maps at virtual 0, and prepare_texture gives the untiled image back.
NXTEST(sparse_texture_prepares_from_mapped_tiles) {
    no_cache();
    const TextureBuilderInfo info = make_info(NVN_TEXTURE_TARGET_2D, 64, 64, 1, 1, NVN_FORMAT_RGBA8);
    Built bt;
    CHECK_MSG(begin_block_linear(bt, info), "layout");
    fill_random(bt.lin, 4242);
    finish_block_linear(bt, info);
    // the physical pool: a 64 KiB host buffer that holds the tiled storage at its start (the storage is smaller than one tile)
    std::vector<uint8_t> phys(0x10000, 0);
    std::memcpy(phys.data(), bt.mem.data(), bt.mem.size());
    be::Ref<be::MemoryPool> pp(new be::MemoryPool());
    pp->base = phys.data();
    pp->size = phys.size();
    // the virtual pool: one mapped tile at virtual 0, backed by the physical pool at offset 0
    be::Ref<be::MemoryPool> vp(new be::MemoryPool());
    vp->is_virtual = true;
    vp->size = 1u << 20;
    be::MemoryPool::VirtualMap map;
    map.voff = 0;
    map.size = 0x10000;
    map.phys = be::Ref<be::MemoryPool>::share(pp.get());
    map.poff = 0;
    vp->vmaps.push_back(std::move(map));
    // the texture reads its storage through the virtual pool, not through mem
    Texture& t = *bt.tex;
    t.mem = nullptr;
    t.vpool = be::Ref<be::MemoryPool>::share(vp.get());
    t.voffset = 0;
    PreparedTexture out;
    const bool ok = prepare_texture(t, TexCaps(), out);
    CHECK_MSG(ok, out.error);
    expect_image(bt, out, "sparse untile differs");
}

NXTEST(untile_cube_and_cube_array) {
    no_cache();
    const TextureBuilderInfo cube = make_info(NVN_TEXTURE_TARGET_CUBEMAP, 64, 64, 0, 4, NVN_FORMAT_RGBA8);
    // a cube array's depth counts layer-faces, 6 per cube (nvn_texture_layout.cpp; Pokopia sets 6 * cubes): two cubes = 12
    const TextureBuilderInfo cube_array = make_info(NVN_TEXTURE_TARGET_CUBEMAP_ARRAY, 40, 40, 12, 3, NVN_FORMAT_RGBA8);
    for (const TextureBuilderInfo& info : {cube, cube_array}) {
        Built bt;
        REQUIRE(begin_block_linear(bt, info));   // without a layout the rest would read an empty image
        CHECK_EQ(bt.L.layers, info.target == NVN_TEXTURE_TARGET_CUBEMAP ? 6u : 12u);
        fill_random(bt.lin, 77);
        finish_block_linear(bt, info);
        PreparedTexture out;
        CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), out), out.error);
        expect_image(bt, out, "cube data differs");
    }
}

NXTEST(untile_3d_volume_33_cubed) {
    no_cache();
    const TextureBuilderInfo info = make_info(NVN_TEXTURE_TARGET_3D, 33, 33, 33, 6, NVN_FORMAT_RGBA8);
    Built bt;
    CHECK_MSG(begin_block_linear(bt, info), "layout");
    fill_random(bt.lin, 33);
    finish_block_linear(bt, info);
    PreparedTexture out;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), out), out.error);
    CHECK_EQ(out.levels, 6u);
    CHECK_EQ(out.layers, 1u);
    expect_image(bt, out, "3D data differs");
}

NXTEST(passthrough_bc1_and_bc7_unchanged) {
    no_cache();
    const struct {
        int format;
        int w, h;
    } cases[] = {{NVN_FORMAT_RGBA_DXT1, 64, 48}, {NVN_FORMAT_BPTC_UNORM, 37, 21}, {NVN_FORMAT_BPTC_UNORM_SRGB, 64, 64}};
    for (const auto& c : cases) {
        const TextureBuilderInfo info = make_info(NVN_TEXTURE_TARGET_2D, c.w, c.h, 1, 3, c.format);
        Built bt;
        CHECK_MSG(begin_block_linear(bt, info), "layout");
        fill_random(bt.lin, 500 + static_cast<uint32_t>(c.format));   // random blocks: any bytes are a valid BCn block for pass-through
        finish_block_linear(bt, info);
        PreparedTexture out;
        CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), out), out.error);
        CHECK_EQ(out.format, bt.L.fmt->vk);
        CHECK_EQ(out.block_w, 4u);
        CHECK_EQ(out.block_h, 4u);
        CHECK_EQ(out.block_bytes, bt.L.fmt->bytes);
        CHECK_EQ(out.level[0].row_blocks, static_cast<uint32_t>(div_up(c.w, 4)));
        expect_image(bt, out, "BCn data differs");
    }
}

NXTEST(pitch_linear_copies_tight_rows) {
    no_cache();
    // 37 texels of RGBA8 are 148 bytes, the pitch is 160 (32-byte aligned): the copy must drop the padding
    const TextureBuilderInfo info = make_info(NVN_TEXTURE_TARGET_2D, 37, 9, 1, 1, NVN_FORMAT_RGBA8, NVN_TEXTURE_FLAGS_LINEAR);
    Built bt;
    build_pitch(bt, info, 5);
    CHECK_EQ(bt.L.pitch, 160u);
    PreparedTexture out;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), out), out.error);
    CHECK(out.data == bt.lin);
    CHECK_EQ(out.level[0].row_blocks, 37u);
    CHECK_EQ(out.level[0].size, 37u * 4 * 9);
    CHECK(!out.transcoded);
}

// ---- ASTC -------------------------------------------------------------------------------------------------------------------

NXTEST(astc_transcode_bc7_keep_and_decode) {
    no_cache();
    const int formats[] = {NVN_FORMAT_RGBA_ASTC_5x4,      NVN_FORMAT_RGBA_ASTC_5x4_SRGB, NVN_FORMAT_RGBA_ASTC_5x5,
                           NVN_FORMAT_RGBA_ASTC_5x5_SRGB, NVN_FORMAT_RGBA_ASTC_8x5,      NVN_FORMAT_RGBA_ASTC_8x5_SRGB};
    for (int f : formats) {
        const TextureBuilderInfo info = make_info(NVN_TEXTURE_TARGET_2D, 64, 64, 1, 3, f);
        Built bt;
        CHECK_MSG(begin_block_linear(bt, info), "layout");
        std::vector<uint16_t> col0, scratch;
        for (uint32_t i = 0; i < 3; ++i) fill_astc_void_level(bt, i, 0, i == 0 ? col0 : scratch, 900 + i + static_cast<uint32_t>(f));
        finish_block_linear(bt, info);
        const bool srgb = bt.L.fmt->srgb;
        const char* name = bt.L.fmt->name;

        // default caps: BC7 transcode, timed
        PreparedTexture out;
        const auto t0 = std::chrono::steady_clock::now();
        CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), out), out.error);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("texdata: %s 64x64, 3 levels, ASTC -> BC7 in %.2f ms\n", name, ms);
        CHECK_EQ(out.format, srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK);
        CHECK(out.transcoded);
        CHECK(!out.from_cache);
        CHECK_EQ(out.levels, 3u);
        CHECK_EQ(out.layers, 1u);
        CHECK_EQ(out.block_w, 4u);
        CHECK_EQ(out.block_h, 4u);
        CHECK_EQ(out.block_bytes, 16u);
        uint64_t pos = 0;
        uint32_t w = 64, h = 64;
        for (size_t i = 0; i < out.level.size(); ++i) {
            const uint64_t want = div_up(w, 4) * div_up(h, 4) * 16;
            CHECK_EQ(out.level[i].width, w);
            CHECK_EQ(out.level[i].height, h);
            CHECK_EQ(out.level[i].size, want);
            CHECK_EQ(out.level[i].offset, pos);
            CHECK_EQ(out.level[i].row_blocks, static_cast<uint32_t>(div_up(w, 4)));
            pos += want;
            w = std::max(1u, w / 2);
            h = std::max(1u, h / 2);
        }
        CHECK_EQ(out.data.size(), pos);
        expect_bc7_matches_library(bt, out);

        // keep ASTC when the GPU samples it: the data is the untiled source, unchanged
        TexCaps keep;
        keep.astc = true;
        PreparedTexture k;
        CHECK_MSG(prepare_texture(*bt.tex, keep, k), k.error);
        CHECK_EQ(k.format, bt.L.fmt->vk);
        CHECK(!k.transcoded);
        CHECK(k.data == bt.lin);

        // no BC7 transcode: RGBA8 decode, equal to the library's decode, and the constant colours at the block origins
        TexCaps rgba;
        rgba.bc7_transcode = false;
        PreparedTexture d;
        CHECK_MSG(prepare_texture(*bt.tex, rgba, d), d.error);
        CHECK_EQ(d.format, srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM);
        CHECK(d.transcoded);
        CHECK_EQ(d.block_bytes, 4u);
        expect_decode_matches_library(bt, d);
        if (!srgb) expect_void_colours(d.data.data() + d.level[0].offset, 64, bt.L.fmt->block_w, bt.L.fmt->block_h,
                                       static_cast<uint32_t>(bt.L.layout.levels[0].blocks_x),
                                       static_cast<uint32_t>(bt.L.layout.levels[0].blocks_y), col0);
    }
}

NXTEST(astc_odd_size_two_layers) {
    no_cache();
    // 61x37 with 8x5 blocks (partial blocks at both edges), 2 layers, 2 levels
    const TextureBuilderInfo info = make_info(NVN_TEXTURE_TARGET_2D_ARRAY, 61, 37, 2, 2, NVN_FORMAT_RGBA_ASTC_8x5);
    Built bt;
    CHECK_MSG(begin_block_linear(bt, info), "layout");
    std::vector<uint16_t> scratch;
    for (uint32_t layer = 0; layer < 2; ++layer)
        for (uint32_t i = 0; i < 2; ++i) fill_astc_void_level(bt, i, layer, scratch, 40 + layer * 7 + i);
    finish_block_linear(bt, info);

    PreparedTexture out;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), out), out.error);
    CHECK_EQ(out.layers, 2u);
    // level 0: 61x37 -> 16x10 BC7 blocks, 2 layers; level 1: 30x18 -> 8x5 blocks, 2 layers
    CHECK_EQ(out.level[0].size, 16u * 10 * 16 * 2);
    CHECK_EQ(out.level[1].size, 8u * 5 * 16 * 2);
    CHECK_EQ(out.level[1].offset, out.level[0].size);
    CHECK_EQ(out.data.size(), out.level[0].size + out.level[1].size);
    expect_bc7_matches_library(bt, out);

    TexCaps rgba;
    rgba.bc7_transcode = false;
    PreparedTexture d;
    CHECK_MSG(prepare_texture(*bt.tex, rgba, d), d.error);
    expect_decode_matches_library(bt, d);
}

// ---- disk cache -------------------------------------------------------------------------------------------------------------

NXTEST(bc7_disk_cache_hit_and_truncated_file) {
    const fs::path dir = fs::temp_directory_path() /
                         ("nxrt_texdata_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code ec;
    fs::create_directories(dir, ec);
    set_env("NXRT_NVN_BC7_CACHE", dir.string().c_str());
    CHECK_EQ(config_get_str("nvn", "bc7_cache", "x"), dir.string());

    const TextureBuilderInfo info = make_info(NVN_TEXTURE_TARGET_2D, 64, 64, 1, 3, NVN_FORMAT_RGBA_ASTC_5x4_SRGB);
    Built bt;
    CHECK_MSG(begin_block_linear(bt, info), "layout");
    std::vector<uint16_t> scratch;
    for (uint32_t i = 0; i < 3; ++i) fill_astc_void_level(bt, i, 0, scratch, 300 + i);
    finish_block_linear(bt, info);

    PreparedTexture a;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), a), a.error);
    CHECK(!a.from_cache);
    const std::vector<fs::path> files = cache_files(dir);
    CHECK_EQ(files.size(), 1u);
    if (files.size() != 1) {
        set_env("NXRT_NVN_BC7_CACHE", "");
        fs::remove_all(dir, ec);
        return;
    }
    // file = 28-byte header + 28 bytes per level + the data
    CHECK_EQ(fs::file_size(files[0]), 28u + 3 * 28 + a.data.size());

    PreparedTexture b;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), b), b.error);
    CHECK(b.from_cache);
    CHECK(b.data == a.data);
    CHECK(b.level.size() == a.level.size());

    // a truncated file is ignored: the texture is transcoded again and the file is rewritten whole
    fs::resize_file(files[0], fs::file_size(files[0]) / 2, ec);
    PreparedTexture c;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), c), c.error);
    CHECK(!c.from_cache);
    CHECK(c.data == a.data);
    CHECK_EQ(fs::file_size(files[0]), 28u + 3 * 28 + a.data.size());

    // a file with a wrong magic is ignored the same way
    {
        std::fstream f(files[0], std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(0);
        f.put('X');
    }
    PreparedTexture e;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), e), e.error);
    CHECK(!e.from_cache);
    CHECK(e.data == a.data);

    // with the cache disabled nothing is read or written
    set_env("NXRT_NVN_BC7_CACHE", "");
    config_set("nvn", "bc7_cache", "");
    PreparedTexture g;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), g), g.error);
    CHECK(!g.from_cache);
    CHECK_EQ(cache_files(dir).size(), 1u);

    fs::remove_all(dir, ec);
    no_cache();
}

// [nvn] tex_async = standin (TexCaps::standin_first): with a valid cache entry on disk the render thread still gets the quick stand-in, never the
// entry, and the stand-in is the same data as the one made when there is no entry (it untiles only the levels it encodes).
NXTEST(standin_first_skips_the_cache_entry) {
    const fs::path dir = fs::temp_directory_path() /
                         ("nxrt_texdata_si_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code ec;
    fs::create_directories(dir, ec);
    set_env("NXRT_NVN_BC7_CACHE", dir.string().c_str());

    const TextureBuilderInfo info = make_info(NVN_TEXTURE_TARGET_2D, 256, 256, 1, 4, NVN_FORMAT_RGBA_ASTC_5x4_SRGB);
    Built bt;
    CHECK_MSG(begin_block_linear(bt, info), "layout");
    std::vector<uint16_t> scratch;
    for (uint32_t i = 0; i < 4; ++i) fill_astc_void_level(bt, i, 0, scratch, 700 + i);
    finish_block_linear(bt, info);

    PreparedTexture full;   // writes the entry
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), full), full.error);
    CHECK_EQ(cache_files(dir).size(), 1u);
    PreparedTexture hit;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), hit), hit.error);
    CHECK(hit.from_cache);

    TexCaps si;
    si.provisional = true;
    si.standin_first = true;
    si.image_format = VK_FORMAT_BC7_SRGB_BLOCK;
    PreparedTexture a;
    CHECK_MSG(prepare_texture(*bt.tex, si, a), a.error);
    CHECK(a.provisional);
    CHECK(!a.from_cache);
    CHECK_EQ(a.format, VK_FORMAT_BC7_SRGB_BLOCK);
    CHECK_EQ(a.data.size(), full.data.size());
    CHECK_EQ(a.level.size(), full.level.size());

    // an image that is not BC7 (an RGBA8 entry made it) is not given a BC7 stand-in
    TexCaps rgba = si;
    rgba.image_format = VK_FORMAT_R8G8B8A8_SRGB;
    PreparedTexture c;
    CHECK_MSG(prepare_texture(*bt.tex, rgba, c), c.error);
    CHECK(!c.provisional);
    CHECK_EQ(c.format, VK_FORMAT_R8G8B8A8_SRGB);

    // the stand-in of a texture without an entry (the old path, whole untile) is byte for byte the same
    set_env("NXRT_NVN_BC7_CACHE", "");
    config_set("nvn", "bc7_cache", "");
    TexCaps prov;
    prov.provisional = true;
    prov.image_format = VK_FORMAT_BC7_SRGB_BLOCK;
    PreparedTexture b;
    CHECK_MSG(prepare_texture(*bt.tex, prov, b), b.error);
    CHECK(b.provisional);
    CHECK(a.data == b.data);

    fs::remove_all(dir, ec);
    no_cache();
}

// [nvn] standin_copy: the stand-in's solid blocks are encoded once per source texel and copied into the big levels (rows that map to the same
// source row copied whole). The result must be byte for byte the block-by-block one, for big, odd-sized and layered textures; the times of both
// are printed (a 4096x4096 stand-in is the case that costs the render thread on a first visit).
NXTEST(standin_copy_equals_block_by_block) {
    no_cache();
    struct Case { int target, w, h, depth, levels, format; };
    const Case cases[] = {
        {NVN_TEXTURE_TARGET_2D, 4096, 4096, 1, 13, NVN_FORMAT_RGBA_ASTC_8x8_SRGB},
        {NVN_TEXTURE_TARGET_2D, 1000, 600, 1, 10, NVN_FORMAT_RGBA_ASTC_8x8},
        {NVN_TEXTURE_TARGET_2D_ARRAY, 300, 200, 3, 6, NVN_FORMAT_RGBA_ASTC_4x4},
        {NVN_TEXTURE_TARGET_2D, 2048, 512, 1, 12, NVN_FORMAT_RGBA_ASTC_5x4_SRGB},
    };
    for (const Case& c : cases) {
        const TextureBuilderInfo info = make_info(c.target, c.w, c.h, c.depth, c.levels, c.format);
        Built bt;
        CHECK_MSG(begin_block_linear(bt, info), "layout");
        std::vector<uint16_t> scratch;
        for (uint32_t layer = 0; layer < bt.L.layers; ++layer)
            for (uint32_t i = 0; i < bt.L.layout.levels.size(); ++i) fill_astc_void_level(bt, i, layer, scratch, 1300 + layer * 31 + i);
        finish_block_linear(bt, info);
        TexCaps si;
        si.provisional = true;
        si.standin_first = true;
        si.image_format = bt.L.fmt->srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
        PreparedTexture fast, slow;
        double ms_fast = 1e9, ms_slow = 1e9;
        for (int rep = 0; rep < 3; ++rep) {   // the best of three of each, alternating (the first call pays for fresh memory)
            config_set("nvn", "standin_copy", "1");
            auto t0 = std::chrono::steady_clock::now();
            CHECK_MSG(prepare_texture(*bt.tex, si, fast), fast.error);
            ms_fast = std::min(ms_fast, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            config_set("nvn", "standin_copy", "0");
            t0 = std::chrono::steady_clock::now();
            CHECK_MSG(prepare_texture(*bt.tex, si, slow), slow.error);
            ms_slow = std::min(ms_slow, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        config_set("nvn", "standin_copy", "1");
        std::printf("texdata: stand-in %dx%d x%d, %d levels (%zu KB): %.2f ms copied, %.2f ms block by block\n", c.w, c.h, c.depth, c.levels,
                    fast.data.size() >> 10, ms_fast, ms_slow);
        CHECK(fast.provisional);
        CHECK(slow.provisional);
        CHECK_EQ(fast.data.size(), slow.data.size());
        CHECK(fast.data == slow.data);

        // TexCaps::compact_standin: the solid levels as their distinct block rows. Expanded row by row (as the GPU copy does) it is the same image.
        TexCaps cs = si;
        cs.compact_standin = true;
        PreparedTexture compact;
        double ms_compact = 1e9;
        for (int rep = 0; rep < 3; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            CHECK_MSG(prepare_texture(*bt.tex, cs, compact), compact.error);
            ms_compact = std::min(ms_compact, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::printf("texdata: compact stand-in %dx%d: %zu KB in %.2f ms (whole: %zu KB)\n", c.w, c.h, compact.data.size() >> 10, ms_compact,
                    fast.data.size() >> 10);
        CHECK(compact.provisional);
        CHECK(compact.compact);
        CHECK_EQ(compact.level.size(), fast.level.size());
        CHECK(compact.data.size() < fast.data.size());
        bool same = compact.level.size() == fast.level.size();
        for (size_t i = 0; same && i < fast.level.size(); ++i) {
            const PreparedLevel& f = fast.level[i];
            const PreparedLevel& k = compact.level[i];
            const uint64_t row_bytes = static_cast<uint64_t>(f.row_blocks) * 16;
            const uint32_t bh = (f.height + 3) / 4;
            if (k.rows.empty()) {
                same = k.size == f.size && std::memcmp(compact.data.data() + k.offset, fast.data.data() + f.offset, static_cast<size_t>(f.size)) == 0;
                continue;
            }
            same = k.rows.size() == static_cast<size_t>(bh) * bt.L.layers;
            for (uint32_t layer = 0; same && layer < bt.L.layers; ++layer)
                for (uint32_t by = 0; same && by < bh; ++by) {
                    const uint64_t want = f.offset + (static_cast<uint64_t>(layer) * bh + by) * row_bytes;
                    const uint64_t got = k.rows[static_cast<size_t>(layer) * bh + by];
                    same = got + row_bytes <= compact.data.size() && got % 16 == 0 &&
                           std::memcmp(compact.data.data() + got, fast.data.data() + want, static_cast<size_t>(row_bytes)) == 0;
                }
        }
        CHECK_MSG(same, "the compact stand-in expands to the whole one");
    }
}

// ---- disk cache entries of other formats ------------------------------------------------------------------------------------
// An entry the setup tool makes has the layout of the runtime's BC7 entry, with another VkFormat (nvn_texdata.cpp). These tests
// replace the BC7 entry of a texture with a hand-made one and check which entries prepare_texture uses.

namespace {

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

// A new cache folder for one test, set as the [nvn] bc7_cache folder (the env var wins over the config).
fs::path new_cache_dir(const char* tag) {
    const fs::path dir = fs::temp_directory_path() /
                         (std::string("nxrt_texdata_") + tag + "_" +
                          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code ec;
    fs::create_directories(dir, ec);
    set_env("NXRT_NVN_BC7_CACHE", dir.string().c_str());
    return dir;
}

// Level table of a tight entry: RGBA8 texels (4 bytes each) or 4x4 blocks (16 bytes each); all layers of a level back to back.
std::vector<PreparedLevel> tight_table(const TexLayout& L, bool rgba8) {
    std::vector<PreparedLevel> v;
    uint64_t off = 0;
    for (const tegra::LevelLayout& lv : L.layout.levels) {
        PreparedLevel p{};
        p.width = lv.width;
        p.height = lv.height;
        p.depth = lv.depth;
        const uint64_t per_slice = rgba8 ? static_cast<uint64_t>(lv.width) * lv.height * 4
                                         : div_up(lv.width, 4) * div_up(lv.height, 4) * 16;
        p.offset = off;
        p.size = per_slice * lv.depth * L.layers;
        off += p.size;
        v.push_back(p);
    }
    return v;
}

// The RGBA8 image of a 2D texture (one layer, one slice): texel (x, y) of level i is (x, y, i, 255).
std::vector<uint8_t> rgba8_pattern(const TexLayout& L) {
    const std::vector<PreparedLevel> t = tight_table(L, true);
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
void write_entry(const fs::path& file, uint64_t key, VkFormat format, const std::vector<PreparedLevel>& level, uint32_t layers,
                 const std::vector<uint8_t>& data) {
    std::vector<uint8_t> head;
    const uint8_t magic[4] = {'N', 'X', 'B', '7'};
    head.insert(head.end(), magic, magic + 4);
    put32(head, 1);
    put64(head, key);
    put32(head, static_cast<uint32_t>(format));
    put32(head, static_cast<uint32_t>(level.size()));
    put32(head, layers);
    for (const PreparedLevel& l : level) {
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

// The one cache file of a folder and its key (read from the header). False when the folder does not hold exactly one file.
bool single_entry(const fs::path& dir, fs::path& file, uint64_t& key) {
    const std::vector<fs::path> files = cache_files(dir);
    if (files.size() != 1) return false;
    file = files[0];
    std::ifstream f(file, std::ios::binary);
    uint8_t head[16];
    if (!f.read(reinterpret_cast<char*>(head), sizeof head)) return false;
    key = get64(head + 8);
    return true;
}

// An ASTC texture of `info` with constant-colour void-extent blocks in every level; the image is tiled into the storage.
void build_astc(Built& bt, const TextureBuilderInfo& info, uint32_t seed) {
    CHECK_MSG(begin_block_linear(bt, info), "layout");
    std::vector<uint16_t> scratch;
    for (uint32_t i = 0; i < bt.L.layout.levels.size(); ++i) fill_astc_void_level(bt, i, 0, scratch, seed + i);
    finish_block_linear(bt, info);
}

}  // namespace

// ASTC 4x4 UNORM with a hand-made RGBA8 entry: the entry is used (format, from_cache, data); caps that want BC7 ignore it.
NXTEST(cache_entry_rgba8_is_used) {
    std::error_code ec;
    const fs::path dir = new_cache_dir("rgba8");
    const TextureBuilderInfo info = make_info(NVN_TEXTURE_TARGET_2D, 64, 64, 1, 3, NVN_FORMAT_RGBA_ASTC_4x4);
    Built bt;
    build_astc(bt, info, 400);
    CHECK_EQ(bt.L.layers, 1u);

    // the default caps write the BC7 entry of this texture
    PreparedTexture a;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), a), a.error);
    CHECK_EQ(a.format, VK_FORMAT_BC7_UNORM_BLOCK);
    fs::path file;
    uint64_t key = 0;
    REQUIRE(single_entry(dir, file, key));

    // the hand-made RGBA8 entry replaces it
    const std::vector<PreparedLevel> table = tight_table(bt.L, true);
    const std::vector<uint8_t> pattern = rgba8_pattern(bt.L);
    write_entry(file, key, VK_FORMAT_R8G8B8A8_UNORM, table, bt.L.layers, pattern);
    CHECK_EQ(cache_entry_format(*bt.tex), VK_FORMAT_R8G8B8A8_UNORM);

    PreparedTexture r;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), r), r.error);
    CHECK(r.from_cache);
    CHECK_EQ(r.format, VK_FORMAT_R8G8B8A8_UNORM);
    CHECK_EQ(r.block_w, 1u);
    CHECK_EQ(r.block_bytes, 4u);
    CHECK_EQ(r.levels, 3u);
    REQUIRE(r.level.size() == table.size());
    for (size_t i = 0; i < table.size(); ++i) {
        CHECK_EQ(r.level[i].width, table[i].width);
        CHECK_EQ(r.level[i].height, table[i].height);
        CHECK_EQ(r.level[i].offset, table[i].offset);
        CHECK_EQ(r.level[i].size, static_cast<uint64_t>(table[i].width) * table[i].height * 4 * bt.L.layers);
    }
    CHECK_MSG(r.data == pattern, "RGBA8 entry data differs from the entry");

    // caps that want a BC7 image do not use the RGBA8 entry: the ASTC is transcoded (and the file is rewritten as BC7)
    TexCaps want_bc7;
    want_bc7.image_format = VK_FORMAT_BC7_UNORM_BLOCK;
    PreparedTexture t7;
    CHECK_MSG(prepare_texture(*bt.tex, want_bc7, t7), t7.error);
    CHECK_EQ(t7.format, VK_FORMAT_BC7_UNORM_BLOCK);
    CHECK(!t7.from_cache);
    CHECK(t7.transcoded);
    expect_bc7_matches_library(bt, t7);

    fs::remove_all(dir, ec);
    no_cache();
}

// ASTC 4x4 UNORM with a hand-made BC5 entry: used for default caps and for a BC5 image; not for an RGBA8 image (decoded instead).
// An sRGB texture does not use a BC5 entry (BC5 is for non-sRGB data only): it stays BC7.
NXTEST(cache_entry_bc5_is_used) {
    std::error_code ec;
    const fs::path dir = new_cache_dir("bc5");
    const TextureBuilderInfo info = make_info(NVN_TEXTURE_TARGET_2D, 64, 64, 1, 3, NVN_FORMAT_RGBA_ASTC_4x4);
    Built bt;
    build_astc(bt, info, 500);
    PreparedTexture a;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), a), a.error);
    fs::path file;
    uint64_t key = 0;
    REQUIRE(single_entry(dir, file, key));

    // a BC5 entry: 16-byte blocks with the level geometry of BC7; the data is any bytes
    const std::vector<PreparedLevel> table = tight_table(bt.L, false);
    std::vector<uint8_t> data(static_cast<size_t>(table.back().offset + table.back().size));
    fill_random(data, 501);
    write_entry(file, key, VK_FORMAT_BC5_UNORM_BLOCK, table, bt.L.layers, data);
    CHECK_EQ(cache_entry_format(*bt.tex), VK_FORMAT_BC5_UNORM_BLOCK);

    PreparedTexture r;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), r), r.error);
    CHECK(r.from_cache);
    CHECK_EQ(r.format, VK_FORMAT_BC5_UNORM_BLOCK);
    CHECK_EQ(r.block_w, 4u);
    CHECK_EQ(r.block_h, 4u);
    CHECK_EQ(r.block_bytes, 16u);
    REQUIRE(r.level.size() == table.size());
    for (size_t i = 0; i < table.size(); ++i) {
        CHECK_EQ(r.level[i].width, table[i].width);
        CHECK_EQ(r.level[i].offset, table[i].offset);
        CHECK_EQ(r.level[i].size, div_up(table[i].width, 4) * div_up(table[i].height, 4) * 16 * bt.L.layers);
    }
    CHECK_MSG(r.data == data, "BC5 entry data differs from the entry");

    // a BC5 image uses the entry too
    TexCaps want_bc5;
    want_bc5.image_format = VK_FORMAT_BC5_UNORM_BLOCK;
    PreparedTexture r5;
    CHECK_MSG(prepare_texture(*bt.tex, want_bc5, r5), r5.error);
    CHECK(r5.from_cache);
    CHECK_EQ(r5.format, VK_FORMAT_BC5_UNORM_BLOCK);
    CHECK_MSG(r5.data == data, "BC5 entry data differs from the entry (BC5 image)");

    // an RGBA8 image does not use the BC5 entry: the ASTC is decoded to RGBA8, and the entry is left in place
    TexCaps want_rgba8;
    want_rgba8.image_format = VK_FORMAT_R8G8B8A8_UNORM;
    PreparedTexture d;
    CHECK_MSG(prepare_texture(*bt.tex, want_rgba8, d), d.error);
    CHECK(!d.from_cache);
    CHECK_EQ(d.format, VK_FORMAT_R8G8B8A8_UNORM);
    expect_decode_matches_library(bt, d);
    PreparedTexture still;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), still), still.error);
    CHECK(still.from_cache);
    CHECK_MSG(still.data == data, "BC5 entry was changed by the RGBA8 decode");
    fs::remove_all(dir, ec);

    // an sRGB texture with a BC5 entry: the entry is not used, the texture is transcoded to BC7_SRGB
    const fs::path sdir = new_cache_dir("bc5_srgb");
    Built sb;
    build_astc(sb, make_info(NVN_TEXTURE_TARGET_2D, 64, 64, 1, 3, NVN_FORMAT_RGBA_ASTC_4x4_SRGB), 600);
    PreparedTexture s0;
    CHECK_MSG(prepare_texture(*sb.tex, TexCaps(), s0), s0.error);
    fs::path sfile;
    uint64_t skey = 0;
    REQUIRE(single_entry(sdir, sfile, skey));
    const std::vector<PreparedLevel> stable = tight_table(sb.L, false);
    std::vector<uint8_t> sdata(static_cast<size_t>(stable.back().offset + stable.back().size));
    fill_random(sdata, 601);
    write_entry(sfile, skey, VK_FORMAT_BC5_UNORM_BLOCK, stable, sb.L.layers, sdata);
    CHECK_EQ(cache_entry_format(*sb.tex), VK_FORMAT_UNDEFINED);
    PreparedTexture s;
    CHECK_MSG(prepare_texture(*sb.tex, TexCaps(), s), s.error);
    CHECK(!s.from_cache);
    CHECK_EQ(s.format, VK_FORMAT_BC7_SRGB_BLOCK);
    fs::remove_all(sdir, ec);
    no_cache();
}

// The RGBA8 entry of an sRGB texture is the _SRGB variant: an entry declared UNORM is not used, the _SRGB one is.
NXTEST(cache_entry_srgb_rgba8) {
    std::error_code ec;
    const fs::path dir = new_cache_dir("srgb");
    Built bt;
    build_astc(bt, make_info(NVN_TEXTURE_TARGET_2D, 64, 64, 1, 3, NVN_FORMAT_RGBA_ASTC_4x4_SRGB), 700);
    PreparedTexture a;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), a), a.error);
    CHECK_EQ(a.format, VK_FORMAT_BC7_SRGB_BLOCK);
    fs::path file;
    uint64_t key = 0;
    REQUIRE(single_entry(dir, file, key));
    const std::vector<PreparedLevel> table = tight_table(bt.L, true);
    const std::vector<uint8_t> pattern = rgba8_pattern(bt.L);

    // UNORM entry: not the entry of an sRGB texture
    write_entry(file, key, VK_FORMAT_R8G8B8A8_UNORM, table, bt.L.layers, pattern);
    CHECK_EQ(cache_entry_format(*bt.tex), VK_FORMAT_UNDEFINED);
    PreparedTexture u;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), u), u.error);
    CHECK(!u.from_cache);
    CHECK_EQ(u.format, VK_FORMAT_BC7_SRGB_BLOCK);

    // _SRGB entry: used
    write_entry(file, key, VK_FORMAT_R8G8B8A8_SRGB, table, bt.L.layers, pattern);
    CHECK_EQ(cache_entry_format(*bt.tex), VK_FORMAT_R8G8B8A8_SRGB);
    PreparedTexture s;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), s), s.error);
    CHECK(s.from_cache);
    CHECK_EQ(s.format, VK_FORMAT_R8G8B8A8_SRGB);
    CHECK_MSG(s.data == pattern, "sRGB RGBA8 entry data differs from the entry");

    fs::remove_all(dir, ec);
    no_cache();
}

// ---- hash -------------------------------------------------------------------------------------------------------------------

NXTEST(hash_bytes_separates_inputs) {
    std::vector<uint8_t> v(100);
    fill_random(v, 7);
    const uint64_t h = hash_bytes(v.data(), v.size());
    CHECK_EQ(h, hash_bytes(v.data(), v.size()));
    // every prefix length hashes differently
    std::vector<uint64_t> prefixes;
    for (size_t n = 0; n <= v.size(); ++n) prefixes.push_back(hash_bytes(v.data(), n));
    std::sort(prefixes.begin(), prefixes.end());
    CHECK(std::adjacent_find(prefixes.begin(), prefixes.end()) == prefixes.end());
    // flipping one bit anywhere changes the hash
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] ^= 0x10;
        CHECK_MSG(hash_bytes(v.data(), v.size()) != h, "one bit did not change the hash");
        v[i] ^= 0x10;
    }
}

// ---- caps and failures ------------------------------------------------------------------------------------------------------

NXTEST(failures_return_false_without_crashing) {
    no_cache();
    std::vector<uint8_t> buf;
    PreparedTexture out;

    // no storage
    {
        Built bt;
        CHECK(begin_block_linear(bt, make_info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_RGBA8)));
        finish_block_linear(bt, make_info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_RGBA8));
        bt.tex->mem = nullptr;
        CHECK(!prepare_texture(*bt.tex, TexCaps(), out));
        CHECK(!out.error.empty());
    }
    // storage smaller than the layout
    {
        const TextureBuilderInfo info = make_info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_RGBA8);
        Ref<Texture> t = bare_texture(info, buf);
        t->storage_size = 16 * 16 * 4 - 1;
        CHECK(!prepare_texture(*t, TexCaps(), out));
        CHECK(!out.error.empty());
    }
    // layouts that do not exist (a BUFFER target has one now: a single tight row, nvn_texture_layout.cpp)
    const TextureBuilderInfo bad[] = {
        make_info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_NONE),
        make_info(NVN_TEXTURE_TARGET_3D, 33, 33, 0, 1, NVN_FORMAT_RGBA8),
        make_info(NVN_TEXTURE_TARGET_BUFFER, 0, 1, 1, 1, NVN_FORMAT_RGBA8),
        make_info(NVN_TEXTURE_TARGET_2D, 0, 16, 1, 1, NVN_FORMAT_RGBA8),
        make_info(NVN_TEXTURE_TARGET_2D_ARRAY, 16, 16, 0, 1, NVN_FORMAT_RGBA8),
    };
    for (const TextureBuilderInfo& info : bad) {
        Ref<Texture> t = bare_texture(info, buf);
        CHECK(!prepare_texture(*t, TexCaps(), out));
        CHECK(!out.error.empty());
    }
    // multisample with samples: a real layout, but not supported
    {
        TextureBuilderInfo info = make_info(NVN_TEXTURE_TARGET_2D_MULTISAMPLE, 16, 16, 1, 1, NVN_FORMAT_RGBA8);
        info.samples = 4;
        Ref<Texture> t = bare_texture(info, buf);
        CHECK(!prepare_texture(*t, TexCaps(), out));
        CHECK(!out.error.empty());
    }
    // BCn without BCn support in the GPU (there is no CPU decoder)
    {
        Ref<Texture> t = bare_texture(make_info(NVN_TEXTURE_TARGET_2D, 16, 16, 1, 1, NVN_FORMAT_RGBA_DXT1), buf);
        TexCaps none;
        none.bc = false;
        CHECK(!prepare_texture(*t, none, out));
        CHECK(!out.error.empty());
    }
    // a bad texture object (never initialised) is refused
    {
        Texture empty;
        CHECK(!prepare_texture(empty, TexCaps(), out));
        CHECK(!out.error.empty());
    }
}

NXTEST(caps_select_output_format) {
    no_cache();
    const TextureBuilderInfo info = make_info(NVN_TEXTURE_TARGET_2D, 32, 32, 1, 1, NVN_FORMAT_RGBA_ASTC_5x5);
    Built bt;
    CHECK(begin_block_linear(bt, info));
    std::vector<uint16_t> scratch;
    fill_astc_void_level(bt, 0, 0, scratch, 11);
    finish_block_linear(bt, info);

    PreparedTexture out;
    CHECK_MSG(prepare_texture(*bt.tex, TexCaps(), out), out.error);
    CHECK_EQ(out.format, VK_FORMAT_BC7_UNORM_BLOCK);
    TexCaps rgba;
    rgba.bc7_transcode = false;
    CHECK_MSG(prepare_texture(*bt.tex, rgba, out), out.error);
    CHECK_EQ(out.format, VK_FORMAT_R8G8B8A8_UNORM);
    CHECK_EQ(out.level[0].size, 32u * 32 * 4);
    TexCaps astc;
    astc.astc = true;
    CHECK_MSG(prepare_texture(*bt.tex, astc, out), out.error);
    CHECK_EQ(out.format, VK_FORMAT_ASTC_5x5_UNORM_BLOCK);
    CHECK_EQ(out.block_w, 5u);
    CHECK_EQ(out.block_h, 5u);
    CHECK_EQ(out.block_bytes, 16u);
}

// ---- [gfx] astc_quality: the high preset ---------------------------------------------------------------------------------------

namespace {
// A test image with smooth ramps, a hard edge and some noise.
std::vector<uint8_t> sample_rgba(uint32_t w, uint32_t h) {
    std::vector<uint8_t> px(static_cast<size_t>(w) * h * 4);
    std::mt19937 rng(7);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            uint8_t* p = &px[(static_cast<size_t>(y) * w + x) * 4];
            const bool edge = x > w / 2;
            p[0] = static_cast<uint8_t>(edge ? 230 - y * 3 : x * 9 + (rng() & 7));
            p[1] = static_cast<uint8_t>(y * 11 + (rng() & 15));
            p[2] = static_cast<uint8_t>(edge ? 40 + (rng() & 31) : 200 - x * 5);
            p[3] = 255;
        }
    return px;
}
}  // namespace

NXTEST(astc_quality_setting_parses) {
    config_set("gfx", "astc_quality", "fast");
    CHECK(astc_quality() == AstcQuality::Fast);
    config_set("gfx", "astc_quality", "Uncompressed");
    CHECK(astc_quality() == AstcQuality::Uncompressed);
    config_set("gfx", "astc_quality", "high");
    CHECK(astc_quality() == AstcQuality::High);
    config_set("gfx", "astc_quality", "nonsense");
    CHECK(astc_quality() == astc_quality_default());
    config_set("gfx", "astc_quality", "");
    CHECK(astc_quality() == astc_quality_default());
    // the default follows the encoder: high with bc7e's ISPC build, fast with its C++ build
    CHECK(astc_quality_default() == (a2b_bc7_encoder(nullptr) == A2B_ENCODER_CPP ? AstcQuality::Fast : AstcQuality::High));
    config_set("gfx", "astc_quality", "high");
}

// The banded encoder (used by the background upgrade, so it can stop between bands) gives the same blocks as one whole-image encode.
NXTEST(banded_bc7_encode_equals_whole_image_encode) {
    const uint32_t w = 22, h = 70;
    const std::vector<uint8_t> px = sample_rgba(w, h);
    const size_t bytes = static_cast<size_t>(div_up(w, 4)) * div_up(h, 4) * 16;
    std::vector<uint8_t> whole(bytes), banded(bytes);
    a2b_options opt{};
    opt.quality = kHighPreset;
    opt.threads = 1;
    CHECK_EQ(static_cast<int>(a2b_encode_bc7(w, h, px.data(), &opt, whole.data(), whole.size(), nullptr)), static_cast<int>(A2B_OK));
    CHECK(encode_bc7_banded(w, h, px.data(), kHighPreset, banded.data(), banded.size(), nullptr));
    CHECK(whole == banded);
    std::atomic<bool> stop{true};
    CHECK(!encode_bc7_banded(w, h, px.data(), kHighPreset, banded.data(), banded.size(), &stop));
}

#ifdef NXRT_HAVE_BC7_REF
// One sample image through the fast and the high preset, decoded with the reference decoder: the high preset must not be worse (squared error).
NXTEST(high_preset_bc7_error_not_above_fast_preset) {
    const uint32_t w = 24, h = 20;
    const std::vector<uint8_t> px = sample_rgba(w, h);
    const size_t bytes = static_cast<size_t>(div_up(w, 4)) * div_up(h, 4) * 16;
    const auto error_of = [&](int quality) {
        std::vector<uint8_t> blocks(bytes);
        CHECK(encode_bc7_banded(w, h, px.data(), quality, blocks.data(), blocks.size(), nullptr));
        double e = 0;
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) {
                bc7decomp::color_rgba blk[16];
                bc7decomp_ref::unpack_bc7(&blocks[(static_cast<size_t>(y / 4) * div_up(w, 4) + x / 4) * 16], blk);
                const bc7decomp::color_rgba& c = blk[(y % 4) * 4 + (x % 4)];
                const uint8_t* s = &px[(static_cast<size_t>(y) * w + x) * 4];
                e += double(c.r - s[0]) * (c.r - s[0]) + double(c.g - s[1]) * (c.g - s[1]) + double(c.b - s[2]) * (c.b - s[2]);
            }
        return e;
    };
    const double fast = error_of(3), high = error_of(kHighPreset);
    std::printf("    squared RGB error: preset 3 %.0f, preset %d %.0f\n", fast, kHighPreset, high);
    CHECK(high <= fast * 1.01);
}
#endif
