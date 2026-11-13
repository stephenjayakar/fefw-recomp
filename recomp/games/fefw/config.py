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
    # nxrecomp/opt.py: NZCV in function locals (perf/codegen; fewer and shorter x86 instructions, the same guest behaviour)
    opt = ('flags',)
    # rtld's process entry (0x0): this rtld has an FDE there, Pokopia's has none; listed so discovery roots it either way
    extra_entries = {'rtld': [0x0]}
    patches = None
