# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-game configuration of Fire Emblem: Fortune's Weave (Nintendo Switch 2, title 04007EF01F8AE000).

Supported builds of main are listed in BUILDS (build id -> title version). main of this build is ZBIC-compressed (NSO flag bit 7, a
patched zstd) on the console; the lifter and the runtime read the uncompressed NSO made from it (segment hashes unchanged).
Modules: main at 0x7100000000; sdk (nnSdk 22.2.0), subsdk0 (dlss: NVIDIA DLSS/DLAA for NVN2), subsdk1 (cuda: NVIDIA CUDA 12.9),
subsdk2 (movie: nn::movie) and rtld in the runtime's SDK module region (nxrt/memory.h: kSdkBase up to kHeapBase). sdk, subsdk0 and
rtld sit where Pokopia has them; the subsdk modules are 1 GiB apart from 0x7600000000.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nxrecomp.config import GameConfig, ModuleSpec          # noqa: E402

BUILDS = {
    'DB93B0125F4850601D61C3EDDE012F8EB2ECFE77': '131072',
}


class Config(GameConfig):
    name = 'fefw'
    title_id = '04007EF01F8AE000'
    version = '131072'            # title version of the update (the display version is not in the ExeFS)
    build_id = 'DB93B0125F4850601D61C3EDDE012F8EB2ECFE77'       # of main (first 20 bytes; the rest of the 32-byte field is zero)
    modules = [
        ModuleSpec('main', 0x7100000000, 'main'),
        ModuleSpec('sdk', 0x7500000000, 'sdk'),
        ModuleSpec('subsdk0', 0x7600000000, 'subsdk0'),
        ModuleSpec('subsdk1', 0x7640000000, 'subsdk1'),
        ModuleSpec('subsdk2', 0x7680000000, 'subsdk2'),
        ModuleSpec('rtld', 0x7700000000, 'rtld'),
    ]
    overrides = {}
    # nxrecomp/opt.py (perf/codegen; fewer and shorter x86 instructions, the same guest behaviour): NZCV in function locals, and
    # scalar results written to a V register as one whole-register store (perf/codegen2: every V register write). 'tregs' (x8..x17 in
    # function locals, opt.temps_local) is off for now: it changes the control flow of ~40 % of the functions it touches as clang sees
    # them, so their PGO profile no longer applies; it needs a fresh profile (pgogen run) to be measured fairly.
    opt = ('flags', 'vstore')
    # nnSdk 22.2.0's C string routines run natively (cxx.h hle_*; same results, checked against the lifted routines by
    # recomp/tests/hle_libc_test.cpp). Address: (body, the routine's first instruction words).
    inline_replace = {
        0x75007bbc80: ('nx::hle_memmove(c);', (0x8b020024, 0x8b020005, 0xf102005f, 0x540007a8)),   # __memcpy_aarch64 (overlap-safe)
        0x750078df20: ('nx::hle_memmove(c);', (0xeb01001f, 0x54000ba0, 0x8b020028, 0xeb00011f)),   # memmove
        0x750078e1e0: ('nx::hle_memset(c);', (0xb40006c2, 0x8b020008, 0x39000001, 0x381ff101)),    # memset
        0x75007a6c20: ('nx::hle_strlen(c);', (0xaa0003e8, 0xf240081f, 0x54000560, 0x39400008)),    # strlen
        0x75007bc140: ('nx::hle_memcmp(c);', (0xb4000162, 0x4a000028, 0x92400908, 0xb4000168)),    # memcmp
        0x75007bc240: ('nx::hle_strcmp(c);', (0x4a000028, 0x92400908, 0xb4000288, 0x39400008)),    # strcmp
        # main's zstd 1.5.x ZSTD_decompressMultiFrame (loading / area changes): native zstd, the lifted original as the fallback
        # (game-build/fefw/src/fefw_hle.cpp; FEFW_HLE_ZSTD=0 turns it off)
        0x71000c6a60: ('nx_hle_zstd_multiframe(c, NX_ORIG);', (0xd503237f, 0xd10483ff, 0xa90c7bfd, 0xa90d6ffc)),
    }
    # hot code packed together (guest entries from the perf push's profiles, hottest first; see hot.txt's header)
    hot_order = 'hot.txt'
    # small hot callees copied into their hot callers' translation units (inline.txt's header)
    inline_edges = 'inline.txt'
    # rtld's process entry (0x0): this rtld has an FDE there, Pokopia's has none; listed so discovery roots it either way
    extra_entries = {'rtld': [0x0]}
    patches = None
