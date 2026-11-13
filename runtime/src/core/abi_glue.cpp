// SPDX-License-Identifier: GPL-3.0-or-later
// The runtime-provided half of the generated-code ABI (recomp/include/nxrecomp/ctx.h, ABI version 2, section "runtime-provided"):
// call_indirect, host_import, guest_trap, guest_unimpl, guest_svc, guest_mrs, guest_msr, stx_diag. The rest of the section lives next
// to its subsystem: nx_site_inline / nx_site_midhook / nx_hook_entry in hooks.cpp, nx_catch_call in eh/catch_call.cpp, and the
// registration of the lifted module tables (NxModule) in nxmodule.cpp.
// Pattern from smo-recomp runtime/core/hostrt.c (host_import with in_import bookkeeping, stx_diag, guest_trap / guest_unimpl).
#include <windows.h>

#include <cstdio>
#include <mutex>
#include <set>

#include "nxrt/dispatch.h"
#include "nxrt/kernel.h"
#include "nxrt/log.h"
#include "nxrt/module.h"
#include "nxrt/nxmodule.h"
#include "nxrt/svc.h"

namespace nxrt {
namespace {
std::mutex g_sysreg_lock;
std::set<uint32_t> g_sysregs_seen;
}  // namespace
}  // namespace nxrt

extern "C" {

void call_indirect(Ctx* c, uint64_t addr) { nxrt::dispatch_call(c, addr); }

// A call to a symbol another module provides: the lifted PLT stub passes NX_IMPORT_ID(module index, symbol index) of its own
// module. The registry (nxmodule.cpp) names the import and finds its target: a function another lifted module defines, else the HLE
// handler or logging stub of that name.
void host_import(Ctx* c, uint32_t import_id) {
    const char* name = nullptr;
    const uint64_t target = nxrt::import_resolve_named(import_id, &name);
    if (!target) {
        char what[96];
        snprintf(what, sizeof what, "host_import with an unknown import id (module %u, symbol %u)", NX_IMPORT_MODULE(import_id), NX_IMPORT_SYM(import_id));
        nxrt::trap(what, import_id);
    }
    nxrt::detail::count_import();
    const char* saved_name = c->in_import;
    const uint64_t saved_from = c->import_from;
    c->in_import = name;
    c->import_from = c->x[30] - 4;
    nxrt::dispatch_call(c, target);
    c->in_import = saved_name;
    c->import_from = saved_from;
}

void guest_trap(const char* what, uint64_t addr) { nxrt::trap(what, addr); }

void guest_unimpl(uint64_t pc, uint32_t insn, const char* what) {
    NXLOG_E("guest", "UNTRANSLATED INSTRUCTION at 0x%llx: %08x (%s)", static_cast<unsigned long long>(pc), insn, what);
    nxrt::log_flush();
    RaiseException(0xE0000002, EXCEPTION_NONCONTINUABLE, 0, nullptr);
    TerminateProcess(GetCurrentProcess(), 3);
}

void guest_svc(Ctx* c, uint32_t number) { nxrt::svc_dispatch(c, number, 0); }

// System registers the lifter does not model inline: the generic timer counters are real (19.2 MHz system tick), other reads give 0, writes are ignored; each register is reported once.
uint64_t guest_mrs(Ctx* c, uint32_t sysreg) {
    switch (sysreg) {                       // the generic timer: the SDK reads CNTPCT_EL0 33 times (nn::os::GetSystemTick and friends)
        case 0xdf00: return nxrt::kernel::tick_hz();                 // CNTFRQ_EL0: 19.2 MHz (Switch 1) or 31.25 MHz (Switch 2)
        case 0xdf01:                                                // CNTPCT_EL0
        case 0xdf02: return nxrt::kernel::system_tick();            // CNTVCT_EL0
        default: break;
    }
    std::lock_guard<std::mutex> lk(nxrt::g_sysreg_lock);
    if (nxrt::g_sysregs_seen.insert(sysreg).second) NXLOG_W("guest", "MRS of system register 0x%04x (not modelled): returns 0 (in f_%llx)", sysreg, static_cast<unsigned long long>(c->fn));
    return 0;
}

void guest_msr(Ctx* c, uint32_t sysreg, uint64_t value) {
    std::lock_guard<std::mutex> lk(nxrt::g_sysreg_lock);
    if (nxrt::g_sysregs_seen.insert(sysreg | 0x10000u).second) NXLOG_W("guest", "MSR to system register 0x%04x = 0x%llx (not modelled): ignored (in f_%llx)", sysreg, static_cast<unsigned long long>(value), static_cast<unsigned long long>(c->fn));
}

// Called on every failed exclusive store; reports a loop that never succeeds.
void stx_diag(Ctx* c, uint64_t a, uint64_t expected, uint64_t seen, uint64_t v, int bits) {
    uint64_t n = ++c->ex_fail;
    if (n == 1000 || n == 100000)
        NXLOG_W("guest", "STXR failing %llu times in f_%llx: addr=%llx expected=%llx seen=%llx new=%llx (%s%d)", static_cast<unsigned long long>(n),
                static_cast<unsigned long long>(c->fn), static_cast<unsigned long long>(a), static_cast<unsigned long long>(expected),
                static_cast<unsigned long long>(seen), static_cast<unsigned long long>(v), bits < 0 ? "monitor lost, " : "", bits < 0 ? -bits : bits);
}

}  // extern "C"
