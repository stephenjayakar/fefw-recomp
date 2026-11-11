// SPDX-License-Identifier: GPL-3.0-or-later
// CPU-side texture data preparation (S4d). See nvn_texdata.h for what comes out. No Vulkan calls in this file.
// Libraries: tegra_tiling (untile) and astc2bc7 (ASTC -> BC7 transcode, ASTC -> RGBA8 decode; tools/textures).
// Config: [gfx] astc_quality (fast / high / uncompressed, default high, fast with the C++ encoder; nvn_texdata.h), [nvn] bc7_quality (env NXRT_NVN_BC7_QUALITY,
// default 3, 1 with the C++ encoder), [nvn] bc7_cache (env NXRT_NVN_BC7_CACHE, default "bc7_cache" in the current directory, "" disables the cache).
#include "nvn_texdata.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>

#include <nxrt/config.h>
#include <nxrt/diag.h>
#include <nxrt/log.h>

#include "astc2bc7.h"
#include "nvn_format.h"
#include "nvn_texture_layout.h"
#include "nvn_texup.h"
#include "tegra_tiling.h"

namespace nxrt::nvn::be {
namespace {

constexpr const char* kTag = "nvn.texdata";
constexpr uint32_t kCacheVersion = 1;
constexpr uint8_t kCacheMagic[4] = {'N', 'X', 'B', '7'};
constexpr size_t kCacheHeadBytes = 4 + 4 + 8 + 4 + 4 + 4;   // magic, version, key hash, VkFormat, level count, layer count
constexpr size_t kCacheLevelBytes = 3 * 4 + 2 * 8;          // per level: width, height, depth, offset, size

// The defaults follow the encoder. bc7e's ISPC build makes preset 3 and the background re-encode cheap; its C++ build is about 3 to 4 times
// slower, so it starts at preset 1 (about 3 times faster than 3, under 1.5 dB lower on the game's ASTC 8x8 textures) in fast mode.
bool slow_encoder() {
    static const bool slow = a2b_bc7_encoder(nullptr) == A2B_ENCODER_CPP;
    return slow;
}

uint64_t div_up(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ---- hash -----------------------------------------------------------------------------------------------------------------
// xxHash64-style mix: four lanes over 32-byte stripes, then a final avalanche. Not byte compatible with xxHash; it only has
// to be fast and to separate textures. The cache also compares its header, so a collision cannot return wrong data silently.
constexpr uint64_t kP1 = 0x9E3779B185EBCA87ull, kP2 = 0xC2B2AE3D27D4EB4Full, kP3 = 0x165667B19E3779F9ull,
                   kP4 = 0x85EBCA77C2B2AE63ull, kP5 = 0x27D4EB2F165667C5ull;

inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t read64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline uint32_t read32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t lane(uint64_t acc, uint64_t in) { acc += in * kP2; acc = rotl64(acc, 31); return acc * kP1; }
inline uint64_t merge_lane(uint64_t h, uint64_t v) { h ^= lane(0, v); return h * kP1 + kP4; }

uint64_t hash_seeded(const uint8_t* p, size_t n, uint64_t seed) {
    const uint8_t* const end = p + n;
    uint64_t h;
    if (n >= 32) {
        uint64_t v1 = seed + kP1 + kP2, v2 = seed + kP2, v3 = seed, v4 = seed - kP1;
        do {
            v1 = lane(v1, read64(p));
            v2 = lane(v2, read64(p + 8));
            v3 = lane(v3, read64(p + 16));
            v4 = lane(v4, read64(p + 24));
            p += 32;
        } while (end - p >= 32);
        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = merge_lane(h, v1);
        h = merge_lane(h, v2);
        h = merge_lane(h, v3);
        h = merge_lane(h, v4);
    } else {
        h = seed + kP5;
    }
    h += n;
    while (end - p >= 8) {
        h ^= lane(0, read64(p));
        h = rotl64(h, 27) * kP1 + kP4;
        p += 8;
    }
    if (end - p >= 4) {
        h ^= static_cast<uint64_t>(read32(p)) * kP1;
        h = rotl64(h, 23) * kP2 + kP3;
        p += 4;
    }
    while (p < end) {
        h ^= static_cast<uint64_t>(*p) * kP5;
        h = rotl64(h, 11) * kP1;
        ++p;
    }
    h ^= h >> 33;
    h *= kP2;
    h ^= h >> 29;
    h *= kP3;
    h ^= h >> 32;
    return h;
}

// ---- little-endian helpers for the cache file ------------------------------------------------------------------------------
void put32(std::vector<uint8_t>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i)));
}
void put64(std::vector<uint8_t>& v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i)));
}
uint32_t get32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}
uint64_t get64(const uint8_t* p) { return static_cast<uint64_t>(get32(p)) | (static_cast<uint64_t>(get32(p + 4)) << 32); }

std::string cache_dir() { return config_get_str("nvn", "bc7_cache", "bc7_cache"); }

int bc7_quality() {
    const int def = slow_encoder() ? 1 : 3;
    const int64_t q = config_get_int("nvn", "bc7_quality", def);
    if (q < 0 || q > 6) {
        NXLOG_W(kTag, "[nvn] bc7_quality %lld is outside 0..6: using %d", static_cast<long long>(q), def);
        return def;
    }
    return static_cast<int>(q);
}

// With the C++ encoder the re-encode of high mode (which the play settings choose by default) uses preset 3: about 1 dB over preset 1 on
// the game's textures at a third of the cost of preset 5, and the entries are the fast ones of an ISPC build.
int high_preset() { return std::max(slow_encoder() ? 3 : kHighPreset, bc7_quality()); }

std::string cache_path_in(const std::string& dir, uint64_t hash) {
    if (dir.empty()) return std::string();
    char name[32];
    std::snprintf(name, sizeof name, "%016llx", static_cast<unsigned long long>(hash));
    return (std::filesystem::path(dir) / std::string(name, 2) / (std::string(name) + ".bc7")).string();
}

// Writes the header and data of `out` under `key`. Atomic: the file is written under a temporary name and renamed, so a reader
// never sees a partial file and concurrent writers of the same key both leave a complete file. A failure leaves no file.
bool cache_write(const std::string& path, uint64_t key, const PreparedTexture& out) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path final_path(path);
    fs::create_directories(final_path.parent_path(), ec);
    if (ec) return false;

    std::vector<uint8_t> head;
    head.reserve(kCacheHeadBytes + out.level.size() * kCacheLevelBytes);
    head.insert(head.end(), kCacheMagic, kCacheMagic + 4);
    put32(head, kCacheVersion);
    put64(head, key);
    put32(head, static_cast<uint32_t>(out.format));
    put32(head, static_cast<uint32_t>(out.level.size()));
    put32(head, out.layers);
    for (const PreparedLevel& l : out.level) {
        put32(head, l.width);
        put32(head, l.height);
        put32(head, l.depth);
        put64(head, l.offset);
        put64(head, l.size);
    }

    static std::atomic<uint64_t> seq{0};
    const std::string tmp_name = path + ".tmp." +
                                 std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + "." +
                                 std::to_string(seq.fetch_add(1));
    const fs::path tmp(tmp_name);
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(head.data()), static_cast<std::streamsize>(head.size()));
        f.write(reinterpret_cast<const char*>(out.data.data()), static_cast<std::streamsize>(out.data.size()));
        f.close();
        if (f.fail()) {
            fs::remove(tmp, ec);
            return false;
        }
    }
    fs::rename(tmp, final_path, ec);
    if (ec) {
        std::error_code ignore;
        fs::remove(tmp, ignore);
        return false;
    }
    return true;
}

// Reads the cache file of `key` into out.data. out.format, out.level and out.layers already hold what the file must describe;
// a missing file is a quiet miss, a file of the wrong size, magic, key or layout (for example a truncated write) is ignored
// with a warning, and the caller transcodes again.
bool cache_read(const std::string& path, uint64_t key, PreparedTexture& out) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) return false;
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    const size_t head_bytes = kCacheHeadBytes + out.level.size() * kCacheLevelBytes;
    const uint64_t data_bytes = out.level.empty() ? 0 : out.level.back().offset + out.level.back().size;
    f.seekg(0, std::ios::end);
    const std::streamoff file_bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    if (file_bytes < 0 || static_cast<uint64_t>(file_bytes) != head_bytes + data_bytes) {
        NXLOG_W(kTag, "cache file %s has the wrong size (truncated?): ignored", path.c_str());
        return false;
    }

    std::vector<uint8_t> head(head_bytes);
    if (!f.read(reinterpret_cast<char*>(head.data()), static_cast<std::streamsize>(head_bytes))) return false;
    const uint8_t* p = head.data();
    bool ok = std::memcmp(p, kCacheMagic, 4) == 0 && get32(p + 4) == kCacheVersion && get64(p + 8) == key &&
              get32(p + 16) == static_cast<uint32_t>(out.format) && get32(p + 20) == out.level.size() &&
              get32(p + 24) == out.layers;
    for (size_t i = 0; ok && i < out.level.size(); ++i) {
        const uint8_t* e = p + kCacheHeadBytes + i * kCacheLevelBytes;
        const PreparedLevel& want = out.level[i];
        ok = get32(e) == want.width && get32(e + 4) == want.height && get32(e + 8) == want.depth &&
             get64(e + 12) == want.offset && get64(e + 20) == want.size;
    }
    if (!ok) {
        NXLOG_W(kTag, "cache file %s does not match its key or the texture: ignored", path.c_str());
        return false;
    }

    out.data.resize(static_cast<size_t>(data_bytes));
    if (!f.read(reinterpret_cast<char*>(out.data.data()), static_cast<std::streamsize>(data_bytes))) {
        out.data.clear();
        return false;
    }
    return true;
}

// ---- level geometry --------------------------------------------------------------------------------------------------------
// One level of the tight linear buffer (the untiled image, or the pitch-linear copy).
// One BC7 block of a single colour: mode 6, both endpoints the colour (7 bits + a 0 p-bit, so within 1/255), every index 0.
// Bits 0-6 the mode, then R0 R1 G0 G1 B0 B1 A0 A1 at 7 bits each (bits 7-62), all in the low half. Built with shifts: a stand-in fills
// tens of thousands of these per texture on the render thread, and a bit-by-bit loop made that most of a cold-cache load (0.1.7 hub load).
void bc7_solid_block(const uint8_t* rgba, uint8_t* out) {
    uint64_t lo = uint64_t{1} << 6;   // mode 6
    for (int c = 0; c < 4; ++c) {
        const uint64_t v = rgba[c] >> 1;
        lo |= (v << (7 + 14 * c)) | (v << (14 + 14 * c));
    }
    const uint64_t hi = 0;   // p-bits 0, indices 0
    std::memcpy(out, &lo, 8);
    std::memcpy(out + 8, &hi, 8);
}

struct LevelGeom {
    uint32_t w = 1, h = 1, d = 1;   // texels
    uint32_t bx = 1, by = 1;        // blocks per row and rows, in the source format's blocks
    uint64_t lin_off = 0;           // start of the level in the linear buffer (layer 0)
    uint64_t slice = 0;             // bytes of one layer of the level (all z slices)
};

std::vector<LevelGeom> level_geometry(const TextureBuilderInfo& b, const TexLayout& L) {
    std::vector<LevelGeom> g;
    if (L.block_linear) {
        for (const tegra::LevelLayout& lv : L.layout.levels) {
            LevelGeom e;
            e.w = lv.width;
            e.h = lv.height;
            e.d = lv.depth;
            e.bx = lv.blocks_x;
            e.by = lv.blocks_y;
            e.lin_off = lv.linear_offset;
            e.slice = lv.linear_slice_size;
            g.push_back(e);
        }
    } else {
        // pitch-linear: one level; the 1D targets have height 1 (as compute_texture_layout uses them)
        LevelGeom e;
        e.w = static_cast<uint32_t>(b.width);
        e.h = (b.target == NVN_TEXTURE_TARGET_1D || b.target == NVN_TEXTURE_TARGET_1D_ARRAY) ? 1u : static_cast<uint32_t>(b.height);
        e.d = b.target == NVN_TEXTURE_TARGET_3D ? static_cast<uint32_t>(b.depth) : 1u;
        e.bx = static_cast<uint32_t>(div_up(e.w, L.fmt->block_w));
        e.by = static_cast<uint32_t>(div_up(e.h, L.fmt->block_h));
        e.lin_off = 0;
        e.slice = static_cast<uint64_t>(e.bx) * L.fmt->bytes * e.by * e.d;
        g.push_back(e);
    }
    return g;
}

enum class Mode { Passthrough, Bc7, Rgba8, Bc5 };

// The output level table of a mode (offsets and sizes in PreparedTexture::data; all layers of a level together).
void build_level_table(Mode mode, const std::vector<LevelGeom>& geo, uint32_t layers, PreparedTexture& out) {
    out.level.clear();
    out.level.reserve(geo.size());
    for (const LevelGeom& g : geo) {
        PreparedLevel pl{};
        pl.width = g.w;
        pl.height = g.h;
        pl.depth = g.d;
        uint64_t layer_bytes = 0;
        switch (mode) {
            case Mode::Passthrough:
                pl.row_blocks = g.bx;
                pl.offset = g.lin_off;
                layer_bytes = g.slice;
                break;
            case Mode::Bc7:
            case Mode::Bc5:
                pl.row_blocks = static_cast<uint32_t>(div_up(g.w, 4));
                layer_bytes = static_cast<uint64_t>(pl.row_blocks) * div_up(g.h, 4) * 16 * g.d;
                pl.offset = out.level.empty() ? 0 : out.level.back().offset + out.level.back().size;
                break;
            case Mode::Rgba8:
                pl.row_blocks = g.w;
                layer_bytes = static_cast<uint64_t>(g.w) * g.h * 4 * g.d;
                pl.offset = out.level.empty() ? 0 : out.level.back().offset + out.level.back().size;
                break;
        }
        pl.size = layer_bytes * layers;
        out.level.push_back(pl);
    }
}

// The source image as a tight linear buffer: untiled (block-linear) or copied row by row (pitch-linear).
bool make_linear(const TexLayout& L, const std::vector<LevelGeom>& geo, const FormatInfo& fmt, const uint8_t* src_mem, std::vector<uint8_t>& lin) {
    const LevelGeom& last = geo.back();
    lin.assign(static_cast<size_t>(last.lin_off + last.slice * L.layers), 0);
    if (L.block_linear) return tegra::untile(L.desc, L.layout, src_mem, lin.data());
    const LevelGeom& g = geo[0];
    const uint64_t row = static_cast<uint64_t>(g.bx) * fmt.bytes;
    for (uint32_t layer = 0; layer < L.layers; ++layer)
        for (uint32_t z = 0; z < g.d; ++z)
            for (uint32_t y = 0; y < g.by; ++y) {
                const uint64_t r = static_cast<uint64_t>(z) * g.by + y;   // row index in the level
                const uint8_t* src = src_mem + static_cast<uint64_t>(layer) * (L.pitch * g.by * g.d) + r * L.pitch;
                std::memcpy(lin.data() + static_cast<uint64_t>(layer) * g.slice + r * row, src, static_cast<size_t>(row));
            }
    return true;
}

}  // namespace

AstcQuality astc_quality_default() { return slow_encoder() ? AstcQuality::Fast : AstcQuality::High; }

int bc7_fast_preset() { return bc7_quality(); }

AstcQuality astc_quality() {
    std::string v = config_get_str("gfx", "astc_quality", "");
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (v.empty()) return astc_quality_default();
    if (v == "fast") return AstcQuality::Fast;
    if (v == "high") return AstcQuality::High;
    if (v == "uncompressed" || v == "rgba8") return AstcQuality::Uncompressed;
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true))
        NXLOG_W(kTag, "[gfx] astc_quality '%s' is not fast, high or uncompressed: using %s", v.c_str(), astc_quality_name(astc_quality_default()));
    return astc_quality_default();
}

const char* astc_quality_name(AstcQuality q) {
    switch (q) {
        case AstcQuality::Fast: return "fast";
        case AstcQuality::High: return "high";
        case AstcQuality::Uncompressed: return "uncompressed";
    }
    return "high";
}

bool encode_bc7_banded(uint32_t w, uint32_t h, const uint8_t* rgba, int quality, uint8_t* dst, size_t dst_cap, const std::atomic<bool>* stop) {
    constexpr uint32_t kBandRows = 32;   // a multiple of 4: bands start on block rows, so their blocks equal those of a whole-image call
    if (!w || !h || !rgba || !dst) return false;
    const uint64_t bx = div_up(w, 4);
    if (dst_cap < bx * div_up(h, 4) * 16) return false;
    a2b_options opt{};
    opt.quality = quality;
    opt.threads = 1;
    opt.perceptual = 0;
    opt.allow_opaque_mode6 = 0;
    for (uint32_t y = 0; y < h; y += kBandRows) {
        if (stop && stop->load(std::memory_order_relaxed)) return false;
        const uint32_t bh = std::min(kBandRows, h - y);
        const size_t off = static_cast<size_t>(y / 4) * bx * 16;
        if (a2b_encode_bc7(w, bh, rgba + static_cast<size_t>(y) * w * 4, &opt, dst + off, static_cast<size_t>(bx * div_up(bh, 4) * 16), nullptr) != A2B_OK) return false;
    }
    return true;
}


uint64_t hash_bytes(const void* data, size_t size) {
    return hash_seeded(static_cast<const uint8_t*>(data), size, 0);
}

namespace {

// The cache key of a texture: the hash of the source bytes, seeded with what else decides the output (format, size, layout, quality).
uint64_t cache_key_of(uint64_t storage_size, const uint8_t* mem, const TexLayout& L, const std::vector<LevelGeom>& geo, const FormatInfo& fmt, int quality) {
    const LevelGeom& g0 = geo[0];
    const uint64_t block_height = L.block_linear ? L.layout.levels[0].gob_blocks_y : 0;
    const std::vector<uint64_t> meta = {kCacheVersion,
                                        static_cast<uint64_t>(fmt.nvn),
                                        g0.w,
                                        g0.h,
                                        g0.d,
                                        L.levels,
                                        L.layers,
                                        block_height,
                                        L.pitch,
                                        static_cast<uint64_t>(quality)};
    const uint64_t meta_hash = hash_bytes(meta.data(), meta.size() * sizeof(uint64_t));
    return hash_seeded(mem, static_cast<size_t>(storage_size), meta_hash);
}

// The VkFormat a cache file declares (magic, version and key checked), VK_FORMAT_UNDEFINED when there is no valid file.
VkFormat cache_peek_format(const std::string& path, uint64_t key) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) return VK_FORMAT_UNDEFINED;
    std::ifstream f(path, std::ios::binary);
    uint8_t head[kCacheHeadBytes];
    if (!f || !f.read(reinterpret_cast<char*>(head), sizeof head)) return VK_FORMAT_UNDEFINED;
    if (std::memcmp(head, kCacheMagic, 4) != 0 || get32(head + 4) != kCacheVersion || get64(head + 8) != key) return VK_FORMAT_UNDEFINED;
    return static_cast<VkFormat>(get32(head + 16));
}

}  // namespace
namespace {

// The BC7 entry of the high preset for a texture, made on a worker thread from a snapshot of the texture's storage (tiled bytes as the game holds them).
// The key is worked out from the snapshot, so the entry is right for the bytes it was made from whatever the game wrote since.
constexpr uint64_t kUpgradeMaxBytes = 48ull << 20;

bool upgrade_job(const std::vector<uint8_t>& snap, const TextureBuilderInfo& b, int quality, const std::string& dir, const std::atomic<bool>& stop) {
    const TexLayout L = compute_texture_layout(b);
    if (!L.valid || !L.fmt || !L.fmt->astc || b.samples > 1 || snap.size() < L.storage_size) return false;
    const FormatInfo& fmt = *L.fmt;
    const std::vector<LevelGeom> geo = level_geometry(b, L);
    if (geo.size() != L.levels) return false;
    const uint64_t key = cache_key_of(snap.size(), snap.data(), L, geo, fmt, quality);
    const std::string path = cache_path_in(dir, key);
    if (path.empty()) return false;
    if (cache_peek_format(path, key) != VK_FORMAT_UNDEFINED) return true;   // made meanwhile (another run, or the same data under another texture)

    PreparedTexture hi;
    hi.format = fmt.srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
    hi.block_w = 4;
    hi.block_h = 4;
    hi.block_bytes = 16;
    hi.transcoded = true;
    hi.levels = L.levels;
    hi.layers = L.layers;
    build_level_table(Mode::Bc7, geo, L.layers, hi);
    hi.data.assign(static_cast<size_t>(hi.level.back().offset + hi.level.back().size), 0);

    std::vector<uint8_t> lin;
    if (!make_linear(L, geo, fmt, snap.data(), lin)) return false;
    a2b_input in{};
    in.format = fmt.srgb ? A2B_ASTC_SRGB : A2B_ASTC_UNORM;
    in.block_w = fmt.block_w;
    in.block_h = fmt.block_h;
    std::vector<uint8_t> rgba;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < geo.size(); ++i) {
        const LevelGeom& g = geo[i];
        const PreparedLevel& pl = hi.level[i];
        in.width = g.w;
        in.height = g.h;
        const uint64_t out_slice = pl.size / L.layers / g.d;
        const uint64_t in_slice = g.slice / g.d;
        rgba.resize(static_cast<size_t>(g.w) * g.h * 4);
        for (uint32_t layer = 0; layer < L.layers; ++layer)
            for (uint32_t z = 0; z < g.d; ++z) {
                if (stop.load(std::memory_order_relaxed)) return false;
                const uint8_t* src = lin.data() + g.lin_off + static_cast<uint64_t>(layer) * g.slice + z * in_slice;
                uint8_t* dst = hi.data.data() + pl.offset + (static_cast<uint64_t>(layer) * g.d + z) * out_slice;
                if (a2b_decode_astc_rgba8(&in, src, static_cast<size_t>(in_slice), 1, rgba.data(), nullptr) != A2B_OK) return false;
                if (!encode_bc7_banded(g.w, g.h, rgba.data(), quality, dst, static_cast<size_t>(out_slice), &stop)) return false;
            }
    }
    if (stop.load(std::memory_order_relaxed)) return false;
    if (!cache_write(path, key, hi)) return false;
    NXLOG_I(kTag, "texture cache upgrade: %ux%u, %u levels, %u layers at preset %d: %.0f ms", geo[0].w, geo[0].h, L.levels, L.layers, quality, ms_since(t0));
    return true;
}

// Hands the texture to the background upgrade unless it is queued already. lookup_key: the high-preset key the render thread looked for (identifies the job).
void schedule_upgrade(const Texture& t, const uint8_t* src_mem, uint64_t lookup_key, int quality, const std::string& dir) {
    if (!t.storage_size || t.storage_size > kUpgradeMaxBytes || !texup::wants(lookup_key)) return;
    std::vector<uint8_t> snap(src_mem, src_mem + static_cast<size_t>(t.storage_size));
    const TextureBuilderInfo b = t.b;
    const uint64_t bytes = snap.size();
    texup::submit(lookup_key, bytes, [snap = std::move(snap), b, quality, dir](const std::atomic<bool>& stop) { return upgrade_job(snap, b, quality, dir, stop); });
}

}  // namespace

void note_uncompressed_image(const Texture& t) {
    static std::atomic<uint64_t> n{0}, rgba_bytes{0}, bc7_bytes{0}, astc_bytes{0}, logged_step{0};
    const TexLayout L = compute_texture_layout(t.b);
    if (!L.valid || !L.fmt || !L.fmt->astc) return;
    const std::vector<LevelGeom> geo = level_geometry(t.b, L);
    uint64_t rgba = 0, bc7 = 0;
    for (const LevelGeom& g : geo) {
        rgba += static_cast<uint64_t>(g.w) * g.h * 4 * g.d * L.layers;
        bc7 += div_up(g.w, 4) * div_up(g.h, 4) * 16 * g.d * L.layers;
    }
    const uint64_t cnt = n.fetch_add(1) + 1;
    const uint64_t r = rgba_bytes.fetch_add(rgba) + rgba, b7 = bc7_bytes.fetch_add(bc7) + bc7, as = astc_bytes.fetch_add(geo.empty() ? 0 : t.storage_size) + t.storage_size;
    if (cnt == 1)
        NXLOG_I(kTag, "[gfx] astc_quality = uncompressed: ASTC textures are decoded to RGBA8 (4 times the size of BC7, 4 to 16 times the size of ASTC, in video memory)");
    const uint64_t step = r >> 28;   // every 256 MB
    uint64_t prev = logged_step.load();
    if (step > prev && logged_step.compare_exchange_strong(prev, step))
        NXLOG_I(kTag, "uncompressed ASTC textures so far: %llu images, %.0f MB as RGBA8 (BC7 would be %.0f MB, the game's ASTC %.0f MB)",
                static_cast<unsigned long long>(cnt), static_cast<double>(r) / 1048576.0, static_cast<double>(b7) / 1048576.0, static_cast<double>(as) / 1048576.0);
}

std::string bc7_cache_path(const Texture& t, uint64_t hash) {
    (void)t;
    return cache_path_in(cache_dir(), hash);
}

namespace {
// The presets whose cache entries a texture may use, best first: the high preset (when [gfx] astc_quality = high and it is above the fast one), then the fast
// preset ([nvn] bc7_quality). Returns how many.
int entry_qualities(AstcQuality am, int q[2]) {
    int n = 0;
    const int fq = bc7_quality();
    if (am == AstcQuality::High && high_preset() > fq) q[n++] = high_preset();
    q[n++] = fq;
    return n;
}
}  // namespace

VkFormat cache_entry_format(const Texture& t) {
    if (!t.mem) return VK_FORMAT_UNDEFINED;
    const TexLayout L = compute_texture_layout(t.b);
    if (!L.valid || !L.fmt || !L.fmt->astc || t.b.samples > 1 || t.storage_size < L.storage_size) return VK_FORMAT_UNDEFINED;
    const VkFormat rgba = L.fmt->srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    const AstcQuality am = astc_quality();
    if (am == AstcQuality::Uncompressed) return rgba;   // every ASTC texture is RGBA8; the cache is not used
    if (cache_dir().empty()) return VK_FORMAT_UNDEFINED;
    const std::vector<LevelGeom> geo = level_geometry(t.b, L);
    if (geo.size() != L.levels) return VK_FORMAT_UNDEFINED;
    int q[2];
    const int n = entry_qualities(am, q);
    for (int i = 0; i < n; ++i) {
        const uint64_t key = cache_key_of(t.storage_size, t.mem, L, geo, *L.fmt, q[i]);
        const VkFormat f = cache_peek_format(bc7_cache_path(t, key), key);
        if (f == VK_FORMAT_UNDEFINED) continue;
        return (f == rgba || (f == VK_FORMAT_BC5_UNORM_BLOCK && !L.fmt->srgb)) ? f : VK_FORMAT_UNDEFINED;   // the first valid entry decides
    }
    return VK_FORMAT_UNDEFINED;
}

bool prepare_texture(const Texture& t, const TexCaps& caps, PreparedTexture& out) {
    out = PreparedTexture();
    const auto fail = [&out, &t](std::string msg) {
        NXLOG_W(kTag, "texture %u: %s", t.id, msg.c_str());
        out.error = std::move(msg);
        return false;
    };

    // The source bytes: the texture's storage in the game's memory, or for a sparse texture the tiles mapped into its virtual pool.
    const uint8_t* src_mem = t.mem;
    std::vector<uint8_t> gathered;
    Ref<MemoryPool> direct_keep;
    if (!src_mem && t.vpool) {
        if (const uint8_t* d = texture_virtual_direct(t, direct_keep)) src_mem = d;      // one piece of one pool: untile straight from it
        else if (texture_gather_virtual(t, gathered)) src_mem = gathered.data();
    }
    if (!src_mem) return fail("no storage (mem is null)");
    const TexLayout L = compute_texture_layout(t.b);
    if (!L.valid || !L.fmt) return fail("layout is invalid (format, size, target or layout)");
    if (t.b.samples > 1) return fail("multisample textures are not supported");
    if (t.storage_size < L.storage_size) return fail("storage_size is smaller than the layout needs");
    const FormatInfo& fmt = *L.fmt;

    const std::vector<LevelGeom> geo = level_geometry(t.b, L);
    if (geo.size() != L.levels) return fail("internal: level count mismatch");

    // ---- what the GPU gets ------------------------------------------------------------------------------------------------
    Mode mode = Mode::Passthrough;
    std::string cache_file;   // empty: cache off (or not a BC7 output)
    uint64_t key = 0;
    const VkFormat rgba8_format = fmt.srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    bool entry_applies = false;   // the cache file of the key holds the output format the mode needs
    VkFormat entry_format = VK_FORMAT_UNDEFINED;   // the format a valid cache file of the key declares
    // [gfx] astc_quality (nvn_texdata.h). The cache entries of a texture are looked up at the high preset first (high mode), then at the fast preset; the first
    // valid entry the image can take is used. upgrade_*: no entry at the high preset exists, so the texture goes to the background upgrade.
    const AstcQuality aq = astc_quality();
    const bool uncompressed = fmt.astc && !caps.astc && aq == AstcQuality::Uncompressed;
    const VkFormat bc7_format = fmt.srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
    bool upgrade_possible = false;
    uint64_t upgrade_key = 0;
    int upgrade_quality = 0;
    // [nvn] tex_async = standin: a BC7 image whose data is not ready gets a stand-in at once. No cache key, no file lookup, no read: the disk
    // is only touched by the background job. Host pixel edits keep the synchronous path (their data never comes from the cache).
    const bool standin_now = caps.standin_first && caps.provisional && fmt.astc && !caps.astc && caps.bc7_transcode && !uncompressed &&
                             caps.image_format == bc7_format && !tex_edit_active();
    if (fmt.astc) {
        if (caps.astc) mode = Mode::Passthrough;
        else if (caps.bc7_transcode && !uncompressed) mode = Mode::Bc7;
        else mode = Mode::Rgba8;
        if (!caps.astc && !uncompressed && !cache_dir().empty() && !standin_now) {
            // A cache entry may hold the texture as BC7 (the default), as RGBA8 (a texture BC7 cannot reproduce well) or as BC5 (a two-channel
            // normal map: blue 0 and alpha 1 on sampling). The image the data goes into is made in the entry's format (caps.image_format).
            const VkFormat bc5_format = VK_FORMAT_BC5_UNORM_BLOCK;
            const VkFormat want = caps.image_format;   // UNDEFINED: whatever the entry has
            int q[2];
            const int nq = entry_qualities(aq, q);
            struct Cand { uint64_t key; std::string path; VkFormat ef; } cand[2];
            int chosen = -1, filled = 0;
            for (int i = 0; i < nq && chosen < 0; ++i) {
                cand[i].key = cache_key_of(t.storage_size, src_mem, L, geo, fmt, q[i]);
                cand[i].path = bc7_cache_path(t, cand[i].key);
                cand[i].ef = cache_peek_format(cand[i].path, cand[i].key);
                filled = i + 1;
                const VkFormat ef = cand[i].ef;
                if (ef == rgba8_format && (want == VK_FORMAT_UNDEFINED || want == rgba8_format)) { mode = Mode::Rgba8; chosen = i; }
                else if (ef == bc5_format && !fmt.srgb && (want == VK_FORMAT_UNDEFINED || want == bc5_format)) { mode = Mode::Bc5; chosen = i; }
                else if (ef == bc7_format && (want == VK_FORMAT_UNDEFINED || want == bc7_format)) { mode = Mode::Bc7; chosen = i; }
            }
            entry_applies = chosen >= 0;
            const int use = entry_applies ? chosen : nq - 1;   // no usable entry: the fast preset's file is where a transcode goes
            key = cand[use].key;
            cache_file = cand[use].path;
            entry_format = cand[use].ef;
            if (!entry_applies) {
                if (want == rgba8_format) mode = Mode::Rgba8;     // an RGBA8 image and no entry for it: decode the ASTC
                else if (want == bc5_format) return fail("the image is BC5 but its cache entry is gone");
                else mode = Mode::Bc7;
            }
            // Upgrade: the high preset has no entry for this texture (the loop looked at it first and it was not there), and what the image takes is BC7.
            if (caps.upgrade && nq == 2 && filled >= 1 && q[0] > q[1] && cand[0].ef == VK_FORMAT_UNDEFINED && mode == Mode::Bc7 &&
                (want == VK_FORMAT_UNDEFINED || want == bc7_format) && (!entry_applies || chosen == 1) && (entry_format == VK_FORMAT_UNDEFINED || entry_format == bc7_format)) {
                upgrade_possible = true;
                upgrade_key = cand[0].key;
                upgrade_quality = q[0];
            }
        }
    } else if (fmt.bc) {
        if (!caps.bc) return fail(std::string("BCn format ") + fmt.name + ", and the GPU does not sample BCn");
    } else if (fmt.compressed) {
        return fail(std::string("compressed format ") + fmt.name + " is neither BCn nor ASTC");
    }
    if (mode == Mode::Passthrough && fmt.vk == VK_FORMAT_UNDEFINED)
        return fail(std::string("format ") + fmt.name + " has no Vulkan format");

    switch (mode) {
        case Mode::Passthrough:
            out.format = fmt.vk;
            out.block_w = fmt.block_w;
            out.block_h = fmt.block_h;
            out.block_bytes = fmt.bytes;
            break;
        case Mode::Bc7:
            out.format = fmt.srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
            out.block_w = 4;
            out.block_h = 4;
            out.block_bytes = 16;
            break;
        case Mode::Bc5:
            out.format = VK_FORMAT_BC5_UNORM_BLOCK;
            out.block_w = 4;
            out.block_h = 4;
            out.block_bytes = 16;
            break;
        case Mode::Rgba8:
            out.format = fmt.srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
            out.block_w = 1;
            out.block_h = 1;
            out.block_bytes = 4;
            break;
    }
    out.transcoded = mode != Mode::Passthrough;
    out.levels = L.levels;
    out.layers = L.layers;

    // ---- output level table ---------------------------------------------------------------------------------------------
    build_level_table(mode, geo, L.layers, out);
    const uint64_t out_total = out.level.back().offset + out.level.back().size;

    // Debug: [nvn] dump_astc = DIR writes level 0 of every ASTC texture of width [nvn] dump_astc_w (default any) as tight ASTC blocks to
    // DIR/astc_<key>_<w>x<h>_<bw>x<bh>_<srgb>.bin, to decode them with another decoder.
    {
        static const std::string dump_dir = config_get_str("nvn", "dump_astc", "");
        if (!dump_dir.empty() && fmt.astc && L.block_linear) {
            static const int64_t want_w = config_get_int("nvn", "dump_astc_w", 0);
            if (want_w == 0 || want_w == static_cast<int64_t>(geo[0].w)) {
                char nm[160];
                std::snprintf(nm, sizeof nm, "/astc_t%u_%016llx_%ux%u_%ux%u_%d.bin", t.id, static_cast<unsigned long long>(key), geo[0].w, geo[0].h,
                              fmt.block_w, fmt.block_h, fmt.srgb ? 1 : 0);
                const std::string path = dump_dir + nm;
                if (!std::filesystem::exists(path)) {
                    const LevelGeom& lg = geo.back();
                    std::vector<uint8_t> dl(static_cast<size_t>(lg.lin_off + lg.slice * L.layers));
                    if (tegra::untile(L.desc, L.layout, src_mem, dl.data())) {
                        std::ofstream f(path, std::ios::binary);
                        f.write(reinterpret_cast<const char*>(dl.data() + geo[0].lin_off), static_cast<std::streamsize>(geo[0].slice));
                        NXLOG_I(kTag, "texture %u: ASTC level 0 written to %s", t.id, path.c_str());
                    }
                }
            }
        }
    }

    // ---- host pixel edit (nxrt/tex_edit.h): the kept edited data, or this call makes it ------------------------------------
    TexEditKey edit;
    bool edit_make = false;
    if (fmt.astc && (mode == Mode::Bc7 || mode == Mode::Rgba8) && L.layers == 1 && geo[0].d == 1 && t.b.target == NVN_TEXTURE_TARGET_2D &&
        tex_edit_active()) {
        edit = tex_edit_match(t, src_mem);
        if (edit.id) {
            std::vector<uint8_t> kept;
            if (tex_edit_lookup(edit, out.format, kept) && kept.size() == out_total) {
                out.data = std::move(kept);
                tex_edit_note(t);
                NXLOG_I(kTag, "texture %u: edited data (host id %016llx) reused", t.id, static_cast<unsigned long long>(edit.id));
                return true;
            }
            edit_make = true;
        }
    }
    struct EditGuard {   // a failure below gives up the data this call was to make, so a waiting thread makes it itself
        const TexEditKey* k = nullptr;
        VkFormat f = VK_FORMAT_UNDEFINED;
        ~EditGuard() {
            if (k) tex_edit_abandon(*k, f);
        }
    } edit_guard;
    if (edit_make) {
        edit_guard.k = &edit;
        edit_guard.f = out.format;
    }

    // ---- disk cache ---------------------------------------------------------------------------------------------------------
    const bool bc7 = mode == Mode::Bc7;
    const int quality = bc7 ? bc7_quality() : 0;
    (void)quality;
    if (entry_applies && !edit_make) {
        if (cache_read(cache_file, key, out)) {
            out.from_cache = true;
            NXLOG_I(kTag, "texture %u: %s data from cache %s", t.id, mode == Mode::Bc5 ? "BC5" : mode == Mode::Rgba8 ? "RGBA8" : "BC7", cache_file.c_str());
            if (upgrade_possible && !edit_make) schedule_upgrade(t, src_mem, upgrade_key, upgrade_quality, cache_dir());   // a fast entry served now; the high one is made behind
            return true;
        }
        if (mode == Mode::Bc5) return fail("the BC5 cache entry could not be read");
    }
    // (uncompressed mode: the decode to RGBA8 is short and needs no cache, so a background job does it too)
    if (caps.cache_only && mode != Mode::Passthrough && !edit_make && !uncompressed) {   // background job: a transcode is a long CPU job on data that may not be final; the render thread does it if it is needed
        out.error = "no cache entry";
        return false;
    }
    // only BC7 results are written to the cache, and never over a valid entry of another format (RGBA8 / BC5 entries come from the setup tool)
    if (!bc7 || (entry_format != VK_FORMAT_UNDEFINED && entry_format != out.format)) cache_file.clear();
    if (edit_make) cache_file.clear();   // edited data stays out of the shared cache
    // needed this frame and no entry: the quickest preset now (not cached, no upgrade); the background job makes the entry and the real data
    const bool provisional = caps.provisional && bc7 && !edit_make && (quality > 0 || standin_now);
    if (provisional) {
        cache_file.clear();
        out.provisional = true;
    }

    // A stand-in only has to look right for the half second until the background job lands: the big levels are filled with
    // solid BC7 blocks taken from a small level (decoded once per layer) instead of being encoded, so it costs about 1 ms, not 50-150.
    size_t solid_src = geo.size();
    if (provisional && geo[0].d == 1 && geo.size() > 1) {
        // [nvn] standin_full_px: levels up to this size are encoded (preset 0), the bigger ones are solid blocks taken from the first of them. 256 cost
        // 5-10 s of render thread per Dagsion load (580 stand-ins at 10-40 ms, mostly the 256 level's encode); 64 costs 0.5 s.
        static const uint32_t solid_max = static_cast<uint32_t>(std::max<int64_t>(16, config_get_int("nvn", "standin_full_px", 64)));
        for (size_t i = 0; i < geo.size(); ++i)
            if (std::max(geo[i].w, geo[i].h) <= solid_max) { solid_src = i; break; }
        if (solid_src == 0 || solid_src >= geo.size()) solid_src = geo.size();
    }

    // ---- the source image: tight linear buffer --------------------------------------------------------------------------
    // A stand-in reads only the small levels it encodes (solid_src and after): untiling the big ones, which it never looks at, was most of its cost.
    std::vector<uint8_t> lin;
    if (solid_src < geo.size() && L.block_linear) {
        const LevelGeom& last = geo.back();
        lin.assign(static_cast<size_t>(last.lin_off + last.slice * L.layers), 0);
        for (size_t i = solid_src; i < L.layout.levels.size(); ++i) {
            const tegra::LevelLayout& lv = L.layout.levels[i];
            for (uint32_t layer = 0; layer < L.layers; ++layer)
                tegra::untile_level(L.desc, lv, src_mem + layer * L.layout.tiled_layer_stride + lv.tiled_offset,
                                    lin.data() + lv.linear_offset + layer * lv.linear_slice_size);
        }
    } else if (!make_linear(L, geo, fmt, src_mem, lin)) {
        return fail("untile failed");
    }

    if (mode == Mode::Passthrough) {
        // button prompt icons and the like: a host hook may replace the pixels of a small single-level BC7 sRGB texture
        if (fmt.vk == VK_FORMAT_BC7_SRGB_BLOCK && L.levels == 1 && L.layers == 1 && geo[0].d == 1 && geo[0].w <= 128 &&
            geo[0].h <= 128 && texture_override_active() && texture_override_apply(t, geo[0].w, geo[0].h, lin))
            NXLOG_I(kTag, "texture %u: content replaced by the host (%ux%u BC7)", t.id, geo[0].w, geo[0].h);
        out.data = std::move(lin);
        return true;
    }

    // ---- transcode or decode, one 2D slice at a time (the library threads inside a slice) ------------------------------
    static std::atomic<bool> encoder_logged{false};
    if (!encoder_logged.exchange(true)) {
        const char* name = "?";
        a2b_bc7_encoder(&name);
        NXLOG_I(kTag, "ASTC transcoder: %s; [nvn] bc7_quality %d, [gfx] astc_quality %s", name, bc7_quality(), astc_quality_name(aq));
    }
    out.data.assign(static_cast<size_t>(out_total), 0);
    a2b_options opt{};
    opt.quality = provisional ? 0 : quality;
    opt.threads = provisional ? 1 : caps.encode_threads;   // 0: all hardware threads; a stand-in is small work: thread start-up per level cost more than the encode
    opt.perceptual = 0;
    opt.allow_opaque_mode6 = 0;
    a2b_input in{};
    in.format = fmt.srgb ? A2B_ASTC_SRGB : A2B_ASTC_UNORM;
    in.block_w = fmt.block_w;
    in.block_h = fmt.block_h;

    const auto t0 = std::chrono::steady_clock::now();
    if (edit_make) {
        // decode each level to RGBA8, let the host edit it, encode it again (one layer, one slice: checked above)
        a2b_options eopt = opt;
        eopt.quality = tex_edit_quality();
        std::vector<uint8_t> rgba;
        for (size_t i = 0; i < geo.size(); ++i) {
            const LevelGeom& g = geo[i];
            const PreparedLevel& pl = out.level[i];
            in.width = g.w;
            in.height = g.h;
            uint8_t* dst = out.data.data() + pl.offset;
            uint8_t* px = dst;
            if (bc7) {
                rgba.resize(static_cast<size_t>(g.w) * g.h * 4);
                px = rgba.data();
            }
            a2b_status st = a2b_decode_astc_rgba8(&in, lin.data() + g.lin_off, static_cast<size_t>(g.slice), 0, px, nullptr);
            if (st != A2B_OK) return fail(std::string("ASTC decode for the edit failed: ") + a2b_status_string(st));
            tex_edit_level(edit.id, static_cast<uint32_t>(i), g.w, g.h, px);
            if (bc7) {
                st = a2b_encode_bc7(g.w, g.h, px, &eopt, dst, static_cast<size_t>(pl.size), nullptr);
                if (st != A2B_OK) return fail(std::string("BC7 encode of the edit failed: ") + a2b_status_string(st));
            }
            diag_progress();
        }
        const double ms = ms_since(t0);
        tex_edit_store(edit, out.format, out.data, ms);
        edit_guard.k = nullptr;
        tex_edit_note(t);
        NXLOG_I(kTag, "texture %u: edited by the host (id %016llx), %s %ux%u, %u levels: %.1f ms", t.id, static_cast<unsigned long long>(edit.id),
                bc7 ? "BC7" : "RGBA8", geo[0].w, geo[0].h, L.levels, ms);
        return true;
    }
    std::vector<uint8_t> solid_rgba;
    uint32_t solid_layer = ~0u;
    for (size_t i = 0; i < geo.size(); ++i) {
        const LevelGeom& g = geo[i];
        const PreparedLevel& pl = out.level[i];
        if (i < solid_src && solid_src < geo.size()) {
            const LevelGeom& sg = geo[solid_src];
            const uint64_t out_slice = pl.size / L.layers;
            const uint32_t bw = (g.w + 3) / 4, bh = (g.h + 3) / 4;
            if (out_slice != static_cast<uint64_t>(bw) * bh * 16) return fail("internal: BC7 stand-in size mismatch");
            for (uint32_t layer = 0; layer < L.layers; ++layer) {
                if (solid_layer != layer) {
                    a2b_input sin = in;
                    sin.width = sg.w;
                    sin.height = sg.h;
                    solid_rgba.resize(static_cast<size_t>(sg.w) * sg.h * 4);
                    const a2b_status st = a2b_decode_astc_rgba8(&sin, lin.data() + sg.lin_off + static_cast<uint64_t>(layer) * sg.slice,
                                                                static_cast<size_t>(sg.slice), 1, solid_rgba.data(), nullptr);
                    if (st != A2B_OK) return fail(std::string("ASTC decode for the stand-in failed: ") + a2b_status_string(st));
                    solid_layer = layer;
                }
                uint8_t* dst = out.data.data() + pl.offset + static_cast<uint64_t>(layer) * out_slice;
                for (uint32_t by = 0; by < bh; ++by) {
                    const uint32_t sy = std::min(sg.h - 1, static_cast<uint32_t>((static_cast<uint64_t>(by) * 4 + 2) * sg.h / g.h));
                    for (uint32_t bx = 0; bx < bw; ++bx) {
                        const uint32_t sx = std::min(sg.w - 1, static_cast<uint32_t>((static_cast<uint64_t>(bx) * 4 + 2) * sg.w / g.w));
                        bc7_solid_block(solid_rgba.data() + (static_cast<size_t>(sy) * sg.w + sx) * 4, dst + (static_cast<size_t>(by) * bw + bx) * 16);
                    }
                }
            }
            continue;
        }
        in.width = g.w;
        in.height = g.h;
        const uint64_t out_slice = pl.size / L.layers / g.d;   // bytes of one z slice of the output
        const uint64_t in_slice = g.slice / g.d;               // bytes of one z slice of the ASTC data (tight rows)
        if (bc7 && a2b_output_size(&in) != out_slice) return fail("internal: BC7 slice size mismatch");
        for (uint32_t layer = 0; layer < L.layers; ++layer)
            for (uint32_t z = 0; z < g.d; ++z) {
                const uint8_t* src = lin.data() + g.lin_off + static_cast<uint64_t>(layer) * g.slice + z * in_slice;
                uint8_t* dst = out.data.data() + pl.offset + (static_cast<uint64_t>(layer) * g.d + z) * out_slice;
                a2b_status st;
                if (bc7) {
                    a2b_format of;
                    st = a2b_transcode(&in, src, static_cast<size_t>(in_slice), &opt, &of, dst, static_cast<size_t>(out_slice),
                                       nullptr);
                } else {
                    st = a2b_decode_astc_rgba8(&in, src, static_cast<size_t>(in_slice), 0, dst, nullptr);
                }
                if (st != A2B_OK)
                    return fail(std::string("ASTC ") + (bc7 ? "transcode" : "decode") + " failed: " + a2b_status_string(st));
                diag_progress();   // a cold cache transcodes on the render thread for minutes: slow, not frozen (debug note 421)
            }
    }
    NXLOG_I(kTag, "texture %u: %s %ux%u, %u levels, %u layers%s: %.1f ms", t.id, bc7 ? "ASTC -> BC7" : "ASTC -> RGBA8",
            geo[0].w, geo[0].h, L.levels, L.layers, provisional ? " (quick stand-in, preset 0)" : "", ms_since(t0));

    if (!cache_file.empty() && !cache_write(cache_file, key, out))
        NXLOG_W(kTag, "texture %u: could not write cache file %s", t.id, cache_file.c_str());
    if (upgrade_possible && !edit_make && !provisional) schedule_upgrade(t, src_mem, upgrade_key, upgrade_quality, cache_dir());   // the fast transcode serves now; the high preset is made behind
    return true;
}

}  // namespace nxrt::nvn::be
