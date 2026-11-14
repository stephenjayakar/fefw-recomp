// SPDX-License-Identifier: GPL-3.0-or-later
// [kernel] thread_log (diagnostic, off by default): how a game sizes its thread pools. The nn::os imports that create threads, name them
// and ask for cores are wrapped: each call is logged with its arguments and the guest backtrace of the caller, then the SDK's own code
// runs unchanged (the wrapper calls the lifted SDK function). svcCreateThread tags the guest thread it makes with the nn::os call that
// asked for it, so the log ties ThreadType pointers, names and the runtime's guest-<id> names together (notes/threads.md).
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>

#include "kernel_internal.h"
#include "nxrt/config.h"
#include "nxrt/dispatch.h"
#include "nxrt/eh.h"
#include "nxrt/kernel.h"
#include "nxrt/log.h"
#include "nxrt/module.h"

namespace nxrt::kernel {
namespace {

struct Wrapped {
    const char* sym;
    HleFn fn;
    std::atomic<uint64_t> orig{0};
    std::atomic<uint32_t> calls{0};
};

bool g_log = false;                       // [kernel] thread_log; the wrappers also run (silently) for [kernel] occupancy
bool g_names = false;                     // [kernel] wait_stats / spin_threads / lock_spin_threads: names go to the kernel (kernel_thread_named)
std::mutex g_tt_mu;
std::unordered_map<uint64_t, uint32_t> g_tt_handle;   // ThreadType* -> kernel handle, for SetThreadName

thread_local uint64_t t_creating = 0;     // ThreadType* of the nn::os::CreateThread running on this host thread
thread_local uint32_t t_created = 0;      // the handle svcCreateThread made inside it

uint64_t original(Wrapped& w) {
    uint64_t a = w.orig.load(std::memory_order_relaxed);
    if (!a) {
        a = module_find_symbol(w.sym);
        w.orig.store(a, std::memory_order_relaxed);
    }
    return a;
}

std::string bt(const ::Ctx* c) {
    std::string s = eh::guest_backtrace(c, 10);
    for (char& ch : s)
        if (ch == '\n') ch = '|';
    return s;
}

std::string guest_str(uint64_t p) {
    if (!p || IsBadReadPtr(reinterpret_cast<const void*>(p), 1)) return "(null)";
    std::string s;
    for (int i = 0; i < 64; i++) {
        const char ch = *reinterpret_cast<const char*>(p + i);
        if (!ch) break;
        s += ch;
    }
    return s;
}

const char* self_name() {
    KThread* t = current_thread();
    return t ? t->name.c_str() : "?";
}

void call_original(Wrapped& w, ::Ctx* c) {
    const uint64_t a = original(w);
    if (!a) {
        NXLOG_E("threads", "%s: no SDK definition to call", w.sym);
        c->x[0] = 0;
        return;
    }
    dispatch_call(c, a);
}

// nn::os::CreateThread(ThreadType*, ThreadFunction, void* arg, void* stack, size_t size, int prio[, int idealCore])
void log_create(Wrapped& w, ::Ctx* c, bool with_core) {
    const uint64_t tt = c->x[0], fn = c->x[1], arg = c->x[2], size = c->x[4];
    const int prio = static_cast<int32_t>(c->x[5]);
    const int core = with_core ? static_cast<int32_t>(c->x[6]) : -100;
    const std::string where = g_log ? bt(c) : std::string();
    t_creating = tt;
    t_created = 0;
    call_original(w, c);
    const uint32_t rc = static_cast<uint32_t>(c->x[0]);
    std::string gname = "?";
    if (t_created)
        if (auto t = handle_as<KThread>(t_created)) gname = t->name;
    t_creating = 0;
    if (t_created) {
        std::lock_guard<std::mutex> lk(g_tt_mu);
        g_tt_handle[tt] = t_created;
    }
    if (g_log) NXLOG_I("threads", "CreateThread #%u by %s: ThreadType 0x%llx -> %s (handle 0x%x), entry %s, arg 0x%llx, stack %llu KB, nn prio %d, ideal core %d%s, rc 0x%x; caller %s",
            w.calls.fetch_add(1) + 1, self_name(), (unsigned long long)tt, gname.c_str(), t_created, module_describe(fn).c_str(),
            (unsigned long long)arg, (unsigned long long)(size >> 10), prio, core, with_core ? "" : " (default)", rc, where.c_str());
}

Wrapped g_create6{"_ZN2nn2os12CreateThreadEPNS0_10ThreadTypeEPFvPvES3_S3_mi", nullptr};
Wrapped g_create7{"_ZN2nn2os12CreateThreadEPNS0_10ThreadTypeEPFvPvES3_S3_mii", nullptr};
Wrapped g_name{"_ZN2nn2os13SetThreadNameEPNS0_10ThreadTypeEPKc", nullptr};
Wrapped g_name_ptr{"_ZN2nn2os20SetThreadNamePointerEPNS0_10ThreadTypeEPKc", nullptr};
Wrapped g_core_mask{"_ZN2nn2os17SetThreadCoreMaskEPNS0_10ThreadTypeEim", nullptr};
Wrapped g_avail{"_ZN2nn2os26GetThreadAvailableCoreMaskEv", nullptr};
Wrapped g_cur_core{"_ZN2nn2os20GetCurrentCoreNumberEv", nullptr};

void create6(::Ctx* c) { log_create(g_create6, c, false); }
void create7(::Ctx* c) { log_create(g_create7, c, true); }

void set_name(Wrapped& w, ::Ctx* c) {
    const uint64_t tt = c->x[0];
    const std::string name = guest_str(c->x[1]);
    const std::string where = g_log ? bt(c) : std::string();
    call_original(w, c);
    uint32_t h = 0;
    {
        std::lock_guard<std::mutex> lk(g_tt_mu);
        auto it = g_tt_handle.find(tt);
        if (it != g_tt_handle.end()) h = it->second;
    }
    if (h)
        if (auto t = handle_as<KThread>(h)) {
            occ_name(*t, name.c_str());
            if (g_names) {
                kernel_thread_named(*t, name.c_str());
                NXLOG_I("threads", "%s is \"%s\": nn prio %d, ideal core %d, core mask 0x%llx%s%s", t->name.c_str(), name.c_str(), t->prio - 28, t->ideal_core,
                        (unsigned long long)t->core_mask, t->spin_ok ? ", spins before sleeping" : "", t->lock_spin_ok ? ", polls held mutexes" : "");
            }
        }
    if (g_log) NXLOG_I("threads", "%s by %s: ThreadType 0x%llx = \"%s\"; caller %s", w.sym + 9, self_name(), (unsigned long long)tt, name.c_str(), where.c_str());
}
void name1(::Ctx* c) { set_name(g_name, c); }
void name2(::Ctx* c) { set_name(g_name_ptr, c); }

void core_mask(::Ctx* c) {
    const uint64_t tt = c->x[0], mask = c->x[2];
    const int ideal = static_cast<int32_t>(c->x[1]);
    const std::string where = bt(c);
    call_original(g_core_mask, c);
    NXLOG_I("threads", "SetThreadCoreMask by %s: ThreadType 0x%llx ideal %d mask 0x%llx; caller %s", self_name(), (unsigned long long)tt, ideal,
            (unsigned long long)mask, where.c_str());
}

void avail(::Ctx* c) {
    const std::string where = bt(c);
    call_original(g_avail, c);
    const uint32_t n = g_avail.calls.fetch_add(1) + 1;
    if (n <= 40) NXLOG_I("threads", "GetThreadAvailableCoreMask #%u by %s -> 0x%llx; caller %s", n, self_name(), (unsigned long long)c->x[0], where.c_str());
}

// [sched] core_number_at = <return address>:<core> (experiment, notes/threads.md): nn::os::GetCurrentCoreNumber called from that return
// address answers <core>. fefw: 0x710197d0d8:5 makes its Basic job pool (sized from the cores of 0x1f other than the caller's) start
// five workers on cores 0..4 instead of four on 1..4.
uint64_t g_core_at_ret = 0;
int g_core_at_val = 0;

void cur_core(::Ctx* c) {
    if (g_core_at_ret && c->x[30] == g_core_at_ret) {
        call_original(g_cur_core, c);
        NXLOG_I("sched", "core_number_at: GetCurrentCoreNumber from 0x%llx answers %d (the SDK says %llu)", (unsigned long long)g_core_at_ret, g_core_at_val,
                (unsigned long long)c->x[0]);
        c->x[0] = static_cast<uint64_t>(g_core_at_val);
        return;
    }
    if (!g_log) return call_original(g_cur_core, c);
    const uint32_t n = g_cur_core.calls.fetch_add(1) + 1;
    if (n <= 20 || (n & (n - 1)) == 0) {
        const std::string where = bt(c);
        call_original(g_cur_core, c);
        NXLOG_I("threads", "GetCurrentCoreNumber #%u by %s -> %llu; caller %s", n, self_name(), (unsigned long long)c->x[0], where.c_str());
        return;
    }
    call_original(g_cur_core, c);
}

}  // namespace

void thread_diag_note_created(uint32_t handle) {
    if (t_creating) t_created = handle;
}

void install_thread_diag() {
    g_log = config_get_bool("kernel", "thread_log", false);
    g_names = kernel_names_wanted();
    install_occupancy();
    const std::string at = config_get_str("sched", "core_number_at", "");
    if (!at.empty()) {
        const size_t colon = at.find(':');
        g_core_at_ret = _strtoui64(at.c_str(), nullptr, 16);
        g_core_at_val = colon == std::string::npos ? 0 : std::atoi(at.c_str() + colon + 1);
    }
    if (!g_log && !occupancy_wanted() && !g_core_at_ret && !g_names) return;
    g_create6.fn = create6;
    g_create7.fn = create7;
    g_name.fn = name1;
    g_name_ptr.fn = name2;
    g_core_mask.fn = core_mask;
    g_avail.fn = avail;
    g_cur_core.fn = cur_core;
    for (Wrapped* w : {&g_create6, &g_create7, &g_name, &g_name_ptr, &g_core_mask, &g_avail, &g_cur_core}) {
        if (w == &g_cur_core ? !(g_log || g_core_at_ret)
                             : (!g_log && w != &g_create6 && w != &g_create7 && w != &g_name && w != &g_name_ptr)) continue;   // occupancy needs names only
        hle_register(w->sym, w->fn);
        module_override_symbol(w->sym);
    }
    if (g_log) NXLOG_I("threads", "[kernel] thread_log: nn::os thread creation, names and core queries are logged with their callers");
}

}  // namespace nxrt::kernel
