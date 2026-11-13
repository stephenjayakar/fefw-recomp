// SPDX-License-Identifier: GPL-3.0-or-later
// Registry of the lifter's module descriptors (NxModule, recomp/include/nxrecomp/ctx.h, ABI version 3).
//
// Every lifted module is registered once, at start-up, before any guest code runs:
//
//     auto r = nxrt::nxmodule_register(image, &nxmod_main, NXRECOMP_ABI_VERSION);   // or the macro below
//     if (r.status != nxrt::RegStatus::Ok) ...                                      // refused: nothing was registered
//
// Registration validates the descriptor, refuses a module whose ABI version is not nxrt::kAbiVersion, enters its functions in the
// call_indirect table and makes its tables available to:
//   * host_import(c, import_id): NX_IMPORT_ID(module index, symbol) is resolved through the registry (no guessing which module a
//     stub belongs to). Resolution order: a symbol forced to the host by module_override_symbol, then a function another registered
//     module defines under that name (DynSym.defined), then the HLE handler or logging stub of that name (dispatch.h).
//   * the C++ unwinder (eh.h): frames are found through the module whose [base, text_end) contains the pc.
//
// The ABI version of a lifted module cannot be read from the NxModule itself (the struct has no version field). Since ABI 3 every
// generated module_<name>.cpp also defines `extern "C" const uint32_t nxmod_<name>_abi`, the NXRECOMP_ABI_VERSION of the ctx.h the
// library was built with: the caller passes that constant (nxmodule_register(image, &nxmod_main, nxmod_main_abi)), so a stale prebuilt
// library is refused. NXRT_REGISTER_NXMODULE passes the runtime's own ctx.h version instead (programs that compile the module file
// themselves, the tests).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nxrt/abi.h"

namespace nxrt {

class Module;

enum class RegStatus {
    Ok,
    AbiMismatch,      // abi_version != kAbiVersion
    Invalid,          // null/inconsistent descriptor (null name, negative counts, unsorted tables, ranges outside the image, ...)
    DuplicateIndex,   // another registered module has the same NxModule.index
    DuplicateRange,   // the [base, text_end) range overlaps a registered module
};

struct RegResult {
    RegStatus status = RegStatus::Ok;
    size_t functions = 0;       // entries added to the call table
    std::string message;        // why a registration was refused (also logged)
    bool ok() const { return status == RegStatus::Ok; }
};

// `image` is the mapped ELF (functions outside it are rejected), or null to trust the descriptor's own [base, text_end) range.
RegResult nxmodule_register(Module* image, const NxModule* nx, uint32_t abi_version);
#define NXRT_REGISTER_NXMODULE(image, nxmod) ::nxrt::nxmodule_register((image), &(nxmod), NXRECOMP_ABI_VERSION)

const NxModule* nxmodule_by_index(uint32_t index);
const NxModule* nxmodule_by_name(const char* name);
const NxModule* nxmodule_at(uint64_t addr);        // the registered module whose [base, text_end) contains addr, or null
std::vector<const NxModule*> nxmodules();          // registration order
void nxmodules_clear();                            // tests (also done by modules_unload_all)

// The name behind an import id ("" for an unknown id).
const char* import_name(uint32_t import_id);
// Guest address host_import(c, import_id) dispatches to: a lifted function of another module or an HLE thunk. 0 for an unknown id.
// The answer is cached per (module, symbol); registering a module or overriding a symbol invalidates the cache.
uint64_t import_resolve(uint32_t import_id);
// import_resolve and import_name in one lookup (host_import's path: every cross-module call of the game goes through it).
uint64_t import_resolve_named(uint32_t import_id, const char** name);
// Guest address of a DynSym of a registered module (ABI 3: DynSym.value as is), 0 if it is not defined.
uint64_t nxmodule_symbol_address(const NxModule* nx, int sym);
// The typeinfo address behind filter `filter` (> 0) of an FDE of module `nx`: types[fde.type_first + filter - 1], or, when that slot
// is 0 and NxEh.type_syms names an imported symbol, the address of that symbol in the module that defines it (resolved once and
// cached). *out = 0 means catch(...). False (and *out = 0) when the import is defined by no registered module (logged once): the
// clause then matches nothing.
bool nxmodule_type_address(const NxModule* nx, const NxFde* fde, uint32_t filter, uint64_t* out);
// Absolute address of the function a registered module defines under `name` (any module, registration order), 0 if none.
uint64_t nxmodule_find_export(const char* name, const NxModule* except = nullptr);
void import_cache_invalidate();                    // forget every resolved import (module_override_symbol calls it)
// Number of host_import calls made through the registry (all ids): end-of-run report. Calls that ended in a logging stub are
// counted by dispatch.h (hle_list: stub entries with calls).
uint64_t import_calls();
namespace detail {
void count_import();      // abi_glue.cpp: host_import bookkeeping
}

}  // namespace nxrt
