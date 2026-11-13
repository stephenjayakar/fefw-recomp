# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-game configuration seam.

Everything game-specific that the generic lifter must not know lives in a `GameConfig` subclass under
recomp/games/<name>/config.py (and is selected with --game <name>). smo-recomp kept this knowledge inside its
lifter (sdklift_rules.txt, "nerve" RTTI naming, NPDM / version checks, base-to-update symbol matching); here the
lifter only asks the config. The default config is a neutral one that lifts a module as it is.
"""
import importlib
import importlib.util
import os


class ModuleSpec:
    """One module of the program: where it is mapped and how it is lifted."""

    def __init__(self, name, base, source=None, kind='nso'):
        self.name = name              # 'main', 'sdk', 'subsdk0', ...
        self.base = base              # guest address of the module's first byte
        self.source = source          # default input path (relative to the game's work dir) or None
        self.kind = kind              # 'nso' | 'nx_elf' | 'static_elf'


class GameConfig:
    name = 'generic'

    # modules of the program, in load order
    modules = []                      # [ModuleSpec]

    # guest address -> name of a host handler `h_<name>` that REPLACES the guest function (called as h_<name>(Ctx *c)).
    # Direct calls, indirect calls (call_indirect) and tail calls to that address all go to the handler.
    overrides = {}

    # extra function entries the discovery cannot see (offsets are module-relative), per module name
    extra_entries = {}

    # instruction patches: file of `<address> <original word> <new word>` lines (see program.load_patches), or None
    patches = None

    # a function with more than this many instructions is compiled with optnone (bounded compile time); the build's
    # time-limit fallback (cmake/nx_cc_launcher.py) lowers it when a file still takes too long
    optnone_insns = 20000

    # guest address -> (C++ statements, first instruction words): the lifted function's body is these statements instead of its
    # translation (a native routine with the same results, e.g. cxx.h's hle_memmove for the guest's memmove). The words are checked
    # against the module, so another build of it is lifted as usual (and reported).
    inline_replace = {}

    # file (relative to recomp/games/<name>/) listing guest function entries, hottest first: each listed function goes into its own
    # `.text$nxh<rank>` section, which the linker places together in rank order (hot code packed for the instruction cache and TLB)
    hot_order = None

    # file (relative to recomp/games/<name>/) of `<caller> <callee>` guest entries: the callee gets a `static` copy (fi_<callee>) in
    # the caller's translation unit and the caller calls that, so the host compiler can inline it across shards. Same code.
    inline_edges = None

    # function-level optimisation of the generated C++ (nxrecomp/opt.py), a tuple of: 'flags' (NZCV in locals and never stored
    # across calls and returns, which the AAPCS64 leaves undefined; 'flagsafe': in locals but kept in the context too), 'vstore'
    # (a scalar result into a V register as one whole-register store), 'xregs' / 'vregs' (X / V registers in locals, loaded at the
    # first use after each call: experimental). Empty: the output is as before. The environment variable NXRECOMP_OPT overrides it.
    opt = ()

    # host function names to treat as noreturn-like terminators (none by default)
    noreturn_imports = ()

    # a hook the game config may use to name functions in the generated code's comments: addr -> str
    def function_name(self, module, addr):
        return None


def load_game(name):
    """The GameConfig instance of recomp/games/<name>/config.py (class `Config`)."""
    if not name or name == 'generic':
        return GameConfig()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, 'games', name, 'config.py')
    spec = importlib.util.spec_from_file_location('nxrecomp_game_' + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.Config()
    cfg.dir = os.path.dirname(path)
    return cfg
