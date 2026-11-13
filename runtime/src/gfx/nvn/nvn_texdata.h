// SPDX-License-Identifier: GPL-3.0-or-later
// CPU-side texture data preparation (S4d). No Vulkan calls: the output is plain bytes plus the VkFormat that the GPU
// layer copies them into. Implementation: nvn_texdata.cpp.
//
// What prepare_texture does with the bytes of a texture in the game's memory:
//   * block-linear (Tegra X1 layout) is untiled with tegra::untile into a tight linear buffer; pitch-linear rows are copied
//     into tight rows. Either way the result is the image, levels back to back, inside a level all layers, inside a layer
//     the z slices, then rows of row_blocks blocks.
//   * BCn (BC1..BC7, BC6H, BC4/5) blocks are passed through unchanged.
//   * ASTC is kept when caps.astc, otherwise transcoded to BC7 once (caps.bc7_transcode; the result is cached on disk, keyed
//     by a hash of the source bytes), otherwise decoded to RGBA8.
//   * The disk cache entry decides the output format: BC7 as the runtime writes it, or RGBA8 / BC5 entries made by the setup tool
//     (cache_entry_format; docs/S4f.md "Texture cache entries").
//   * Output sizes: BC7 levels are rounded up to whole 4x4 blocks, i.e. (w+3)/4 * (h+3)/4 * 16 bytes per layer and slice.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <volk.h>

#include "nvn_core.h"

namespace nxrt::nvn::be {

struct TexCaps {
    bool bc = true;              // the GPU samples BCn; BCn data is passed through
    bool astc = false;           // the GPU samples ASTC; ASTC data is kept
    bool bc7_transcode = true;   // when astc is false: transcode ASTC to BC7 (else decode it to RGBA8)
    VkFormat image_format = VK_FORMAT_UNDEFINED;   // the format of the image the data goes into, when it exists already: the data comes out in it
    bool cache_only = false;     // background preparation: an ASTC texture without a disk cache entry fails quietly (error "no cache entry") instead of being transcoded
    bool upgrade = false;        // [gfx] astc_quality = high: a texture without an entry of the high preset hands its source to the background upgrade (nvn_texup.h)
    int encode_threads = 0;      // threads of an ASTC -> BC7 transcode (0: all hardware threads; background jobs use 1)
    bool provisional = false;    // render thread, texture needed now: a BC7 transcode without a cache entry uses the quickest preset (0), is not cached and
                                 // comes out with PreparedTexture::provisional set; a background job makes the real one, which replaces it (nvn_texprep.h)
    bool standin_first = false;  // with provisional, render thread ([nvn] tex_async = standin): an ASTC texture going into a BC7 image gets the quick stand-in
                                 // straight away, without looking at the disk cache (no file open or read: on a hard disk those cost 30 ms to over a
                                 // second each). The background job reads the entry (or transcodes) and its result replaces the stand-in.
    bool compact_standin = false;   // with standin_first ([nvn] standin_rows): the stand-in's big levels, solid blocks copied from a small level, come
                                    // as one row of blocks per distinct source row (PreparedLevel::rows) instead of whole: the caller copies each
                                    // block row into the image from its row, a tenth of the bytes and none of the filling of a whole 2048x2048 level.
};

struct PreparedLevel {
    uint32_t width, height, depth;   // texels of the level
    uint64_t offset, size;           // bytes in PreparedTexture::data; the size covers all layers of the level
    uint32_t row_blocks;             // blocks per row of the output data (texels per row for uncompressed formats)
    // Compact stand-in levels only (TexCaps::compact_standin; empty otherwise): entry layer * block rows + y is the offset in data of block row y
    // of that layer (row_blocks blocks); rows that repeat share one offset. `size` is then the bytes of the distinct rows.
    std::vector<uint64_t> rows;
};

struct PreparedTexture {
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t levels = 0, layers = 0;
    uint32_t block_w = 1, block_h = 1;   // texels per block of the output format
    uint32_t block_bytes = 0;            // bytes per block (texel bytes for uncompressed formats)
    bool transcoded = false;             // ASTC was transcoded to BC7 or decoded to RGBA8 (the data is not the source encoding)
    bool from_cache = false;             // the data came from the BC7 disk cache
    bool provisional = false;            // a quick stand-in (TexCaps::provisional): replaced when the background job has the real data
    bool compact = false;                // some levels are given as rows (PreparedLevel::rows; TexCaps::compact_standin)
    std::vector<PreparedLevel> level;
    std::vector<uint8_t> data;
    std::string error;                   // set when prepare_texture returns false
};

// Prepares the data of t for the GPU described by caps. Returns false with out.error set on failure (no storage, bad layout,
// no Vulkan format, a codec error, ...). Never crashes on a bad texture.
bool prepare_texture(const Texture& t, const TexCaps& caps, PreparedTexture& out);

// The format the disk cache entry of an ASTC texture holds when that is not BC7: R8G8B8A8 (UNORM or SRGB as the texture) or BC5_UNORM (a
// two-channel normal map). VK_FORMAT_UNDEFINED: no entry, or a BC7 entry. The GPU layer makes the image in this format (docs/S4f.md).
// no_disk ([nvn] tex_async = standin, render thread): no file is opened; only the entries this process has already read, peeked or written count
// (an unseen RGBA8 / BC5 entry then gets a BC7 image, and its texture a BC7 transcode in the background: a little less exact, never a stall).
VkFormat cache_entry_format(const Texture& t, bool no_disk = false);
// Texture content override (nxrt/tex_override.h). apply: when a hook is set and it answers for the texture, replaces bc7 (the tight
// linear BC7 blocks of the w x h texture) with the encoded answer.
bool texture_override_active();
bool texture_override_apply(const Texture& t, uint32_t w, uint32_t h, std::vector<uint8_t>& bc7);

// Texture pixel edit (nxrt/tex_edit.h; nvn_tex_edit.cpp). prepare_texture asks match for 2D ASTC textures it transcodes or decodes; for a
// non-zero id it serves the kept data of (id, generation, format) or makes it (decode, the host's edit per level, encode at tex_edit_quality)
// and keeps it; edited data never reaches the BC7 disk cache.
struct TexEditKey {
    uint64_t id = 0, gen = 0;
};
bool tex_edit_active();
TexEditKey tex_edit_match(const Texture& t, const uint8_t* src);
// true: data holds the kept data. false: this thread makes it and must call tex_edit_store or tex_edit_abandon (another thread making the same
// data is waited for first).
bool tex_edit_lookup(const TexEditKey& k, VkFormat f, std::vector<uint8_t>& data);
void tex_edit_store(const TexEditKey& k, VkFormat f, const std::vector<uint8_t>& data, double ms);
void tex_edit_abandon(const TexEditKey& k, VkFormat f);
void tex_edit_level(uint64_t id, uint32_t level, uint32_t w, uint32_t h, uint8_t* rgba);
void tex_edit_note(const Texture& t);   // the texture was given edited data (uploaded again when the answers change)
int tex_edit_quality();                 // [nvn] tex_edit_quality, bc7e preset 0..6 (default 1)

// [gfx] astc_quality (env NXRT_GFX_ASTC_QUALITY), read at every call so a test can change it; the GPU layer reads it once per run (it applies on restart):
//   fast         ASTC -> BC7 at the preset [nvn] bc7_quality (default 3; 1 with the C++ build of bc7e), written to the disk cache, as before.
//                The default with the C++ build (a2b_bc7_encoder: A2B_ENCODER_CPP, about 3 to 4 times slower than the ISPC build).
//   high         (default with the ISPC build) the same on the first use of a texture, and a background thread re-encodes it with the slow preset (kHighPreset; 3 with the C++ build; or
//                [nvn] bc7_quality when that is higher) into the cache; the next start finds the high entry. Entries of the fast preset
//                (including the BC5 / RGBA8 entries of the setup tool) stay valid and are used until the high entry exists.
//   uncompressed ASTC -> RGBA8 for every texture (4 to 16 times the size of ASTC in video memory); the disk cache is not used.
enum class AstcQuality { Fast, High, Uncompressed };
AstcQuality astc_quality();
AstcQuality astc_quality_default();   // when the key is unset or not one of the three: high, or fast with the C++ build of bc7e
const char* astc_quality_name(AstcQuality q);
int bc7_fast_preset();                // [nvn] bc7_quality as the transcodes use it (default 3, 1 with the C++ build of bc7e)
constexpr int kHighPreset = 5;   // bc7e "veryslow", the preset the setup tool can pre-build with (--bc7-quality 5)

// Uncompressed mode: counts a texture whose image was made as RGBA8 and logs the running video memory estimate (RGBA8 now against BC7 and ASTC).
void note_uncompressed_image(const Texture& t);

// Encodes a w x h RGBA8 image to BC7 in bands of rows on the calling thread, checking stop between bands (false: stopped or error; dst is then partial).
// The blocks are identical to one a2b_encode_bc7 call over the whole image (bc7e blocks are independent), which is what the test checks.
bool encode_bc7_banded(uint32_t w, uint32_t h, const uint8_t* rgba, int quality, uint8_t* dst, size_t dst_cap, const std::atomic<bool>* stop);

// Fast 64-bit hash of a byte range (xxHash64-style mix). Used for the cache key.
uint64_t hash_bytes(const void* data, size_t size);

// Cache file of a key: <[nvn] bc7_cache dir>/<first 2 hex digits>/<16 hex digits>.bc7. Empty when the cache is disabled.
std::string bc7_cache_path(const Texture& t, uint64_t hash);

}  // namespace nxrt::nvn::be
