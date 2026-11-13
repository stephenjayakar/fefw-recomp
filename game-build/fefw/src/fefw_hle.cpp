// SPDX-License-Identifier: GPL-3.0-or-later
// Native routines that the lifted game code calls in place of a lifted guest function (GameConfig.inline_replace bodies; see
// recomp/games/fefw/config.py). Linked into fefw.exe with the lifted libraries (tools/build/fefw_build.py), so no runtime change is
// needed. Each one keeps the guest function's results exactly or hands the call back to the lifted original (`orig`).
#include <stdint.h>
#include <stdlib.h>
#include <nxrecomp/ctx.h>
#include "zstd.h"

namespace {

// FEFW_HLE_ZSTD=0 turns the native zstd decoder off (every call goes to the lifted original).
int zstd_enabled() {
    static int on = -1;
    int v = __atomic_load_n(&on, __ATOMIC_RELAXED);
    if (v < 0) {
        const char* e = getenv("FEFW_HLE_ZSTD");
        v = !(e && e[0] == '0');
        __atomic_store_n(&on, v, __ATOMIC_RELAXED);
    }
    return v;
}

}  // namespace

// The game's zstd (1.5.x, statically linked in main): ZSTD_decompressMultiFrame(dctx, dst, dstCapacity, src, srcSize, dict, dictSize,
// ddict), which ZSTD_decompressStream's single-pass shortcut and the one-shot API call when whole frames are in memory (the loading /
// area-change hot path: 50-75 % of some loading frames). Decoding is defined by the format, so any conforming decoder writes the same
// bytes and returns the same size. Without a dictionary the guest dctx contributes only parameters that make a frame fail (window
// limit, magicless format, checksum policy), and its internal tables are reset before the next frame: a native error therefore
// re-runs the call in the lifted original, and so does any call with a dictionary. The registers the call leaves are x0 = the size;
// x19-x29 and SP are untouched (callee-saved), the rest are free under the AAPCS64.
// The guest dctx fields read (this build's layout, as the lifted ZSTD_decompressMultiFrame / ZSTD_decompressStream use them): the
// frame format at +0x7598 (0 = ZSTD_f_zstd1, with magic) and maxWindowSize at +0x7638 (the default, 2^27 + 1): with anything else
// set, the call stays with the original, so a window limit or format the native decoder does not share cannot change a result.
extern "C" void nx_hle_zstd_multiframe(Ctx* c, void (*orig)(Ctx*)) {
    const uint64_t dctx = c->x[0];
    if (!zstd_enabled() || c->x[5] || c->x[6] || c->x[7] || rd32(dctx + 0x7598) != 0 || rd64(dctx + 0x7638) != (1ull << 27) + 1) {
        orig(c);
        return;
    }
    static thread_local ZSTD_DCtx* d = nullptr;
    if (!d && !(d = ZSTD_createDCtx())) {
        orig(c);
        return;
    }
    const size_t r = ZSTD_decompressDCtx(d, reinterpret_cast<void*>(c->x[1]), static_cast<size_t>(c->x[2]),
                                         reinterpret_cast<const void*>(c->x[3]), static_cast<size_t>(c->x[4]));
    if (ZSTD_isError(r)) {
        orig(c);
        return;
    }
    c->x[0] = r;
}
