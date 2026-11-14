// SPDX-License-Identifier: GPL-3.0-or-later
// Guest threads: one Win32 thread per guest thread, a TLS region per thread (TPIDRRO_EL0) and a static TLS block (TPIDR_EL0).
// Pattern from SuperMarioWonderRecomp runtime/src/kernel.cpp new_thread / thread_main (MIT, Copyright (c) 2026 BelmanteGu) and
// smo-recomp runtime/os/os_threads.c (thread start with console core model).
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>

#include "kernel_internal.h"
#include "nxrt/game.h"
#include "nxrt/config.h"
#include "nxrt/diag.h"
#include "nxrt/dispatch.h"
#include "nxrt/gauge.h"
#include "nxrt/kernel.h"
#include "nxrt/log.h"
#include "nxrt/memory.h"
#include "nxrt/module.h"

namespace nxrt::kernel {
namespace {

SRWLOCK g_list_lock = SRWLOCK_INIT;
std::vector<std::shared_ptr<KThread>> g_all;
std::atomic<uint64_t> g_next_id{0x50};
thread_local KThread* t_self = nullptr;
thread_local std::shared_ptr<KThread> t_keepalive;   // released explicitly before ExitThread (which skips destructors)

// Removes its gauge when the file's statics are torn down (declared after g_all, so it goes first).
struct GaugeHook {
    const char* name;
    GaugeHook(const char* n, std::function<int64_t()> f) : name(n) { nxrt::gauge_register(n, std::move(f)); }
    ~GaugeHook() { nxrt::gauge_unregister(name); }
};
// Thread objects in the list: live threads plus exited ones that something still refers to (a game handle, a waiter). thread_create drops the
// exited, unreferenced ones, and kernel_reset() drops all of them.
GaugeHook gauge_threads("kernel.threads", [] {
    AcquireSRWLockShared(&g_list_lock);
    const int64_t n = static_cast<int64_t>(g_all.size());
    ReleaseSRWLockShared(&g_list_lock);
    return n;
});

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

void finish_current(KThread& t) {
    diag_unregister_thread();
    detail::thread_mark_exited(t);
    t_self = nullptr;
}

DWORD WINAPI host_thread_main(LPVOID param) {
    auto* box = static_cast<std::shared_ptr<KThread>*>(param);
    t_keepalive = std::move(*box);
    delete box;
    KThread* t = t_keepalive.get();
    t_self = t;
    t->host_tid = GetCurrentThreadId();
    log_set_thread_name(t->name.c_str());
    diag_register_thread(t->name.c_str(), &t->ctx, t->id);
    NXLOG_D("thread", "%s (id 0x%llx) starts at 0x%llx, core mask %llx prio %d", t->name.c_str(), (unsigned long long)t->id,
            (unsigned long long)t->entry, (unsigned long long)t->core_mask, t->prio);
    dispatch_call(&t->ctx, t->entry);
    NXLOG_D("thread", "%s returned from its entry function", t->name.c_str());
    finish_current(*t);
    t_keepalive.reset();
    return 0;
}

}  // namespace

// The TLS region (tls) and the static TLS block (tp_block) are not freed here: they come from mem::alloc_kernel, a bump allocator with no free,
// so there is nothing to give back. They cost 0x1000 bytes plus the block (at least 0x1000) per thread ever created, a
// few hundred KiB an hour at the S34 soak rate, against the 4 GiB kernel region.
KThread::~KThread() {
    if (host_handle) CloseHandle(static_cast<HANDLE>(host_handle));
}

KThread* current_thread() { return t_self; }
::Ctx* current_ctx() { return t_self ? &t_self->ctx : nullptr; }

uint32_t thread_create(const ThreadParams& p) {
    auto t = std::make_shared<KThread>();
    t->self = t;
    t->id = g_next_id++;
    t->name = p.name.empty() ? "guest-" + std::to_string(t->id) : p.name;
    t->entry = p.entry;
    t->arg = p.arg;
    t->stack_top = p.stack_top;
    t->prio = p.prio;
    int core = p.core == -2 ? 0 : p.core;
    const int top = app_core_count() - 1;
    t->ideal_core = core > top ? top : core;             // Switch 1: the fourth (system) core maps onto core 2
    t->core_mask = 1ull << (t->ideal_core < 0 ? 0 : t->ideal_core);

    // TLS region (read through TPIDRRO_EL0) and the static TLS block (TPIDR_EL0). The nn::os convention keeps the thread pointer at
    // TLS +0x1F0. The block holds the 16-byte TCB, then the initial TLS image of the first module that has a PT_TLS segment.
    const TlsInfo& tpl = module_tls_template();
    const uint64_t block_size = align_up(0x1000 > tpl.tp_offset + tpl.mem_size + 0x100 ? 0x1000 : tpl.tp_offset + tpl.mem_size + 0x100, 0x1000);
    t->tls = mem::alloc_kernel(0x1000);
    t->tp_block = mem::alloc_kernel(block_size);
    if (!t->tls || !t->tp_block) {
        NXLOG_E("thread", "out of kernel memory for the TLS of a new thread");
        return 0;
    }
    if (tpl.file_size) std::memcpy(reinterpret_cast<void*>(t->tp_block + tpl.tp_offset), reinterpret_cast<const void*>(tpl.init_addr), static_cast<size_t>(tpl.file_size));
    *reinterpret_cast<uint64_t*>(t->tls + 0x1F0) = t->tp_block;

    std::memset(&t->ctx, 0, sizeof t->ctx);
    t->ctx.ex_addr = ~0ull;
    t->ctx.x[0] = p.arg;
    t->ctx.x[31] = p.stack_top;
    t->ctx.tpidr = t->tp_block;
    t->ctx.tpidrro = t->tls;
    t->handle = handle_add(t);
    if (!t->handle) {
        NXLOG_E("thread", "handle table full");
        return 0;
    }
    // The kernel's ThreadLocalRegion current_thread_handle (TLS +0x110): SDK 22's InternalCriticalSection compares a lock's owner with it, so
    // a zero there makes every free mutex look held by its caller (Eden-NS2 k_process.cpp Run, svc_thread.cpp CreateThread write it too).
    *reinterpret_cast<uint32_t*>(t->tls + 0x110) = t->handle;
    AcquireSRWLockExclusive(&g_list_lock);
    // Thread objects of threads that have exited and that nothing else refers to (the game closed the handle, the host thread dropped its
    // own reference) leave the list. Without this the list grew by every thread the game ever made (S34 soak: 60 an hour).
    g_all.erase(std::remove_if(g_all.begin(), g_all.end(), [](const std::shared_ptr<KThread>& o) { return o->exited && o.use_count() == 1; }), g_all.end());
    g_all.push_back(t);
    ReleaseSRWLockExclusive(&g_list_lock);
    return t->handle;
}

uint32_t thread_start(uint32_t handle) {
    auto t = handle_as<KThread>(handle);
    if (!t) return kInvalidHandle;
    {
        AcquireSRWLockExclusive(&detail::g_k);
        const bool already = t->started;
        t->started = true;
        ReleaseSRWLockExclusive(&detail::g_k);
        if (already) return kInvalidState;
    }
    SIZE_T stack = 0;
    int64_t mb = config_get_int("threads", "host_stack_mb", 64);
    stack = static_cast<SIZE_T>((mb < 1 ? 64 : mb) << 20);
    auto* box = new std::shared_ptr<KThread>(t);
    DWORD tid = 0;
    HANDLE h = CreateThread(nullptr, stack, host_thread_main, box, CREATE_SUSPENDED | STACK_SIZE_PARAM_IS_A_RESERVATION, &tid);
    if (!h) {
        NXLOG_E("thread", "CreateThread failed for %s (error %lu)", t->name.c_str(), GetLastError());
        delete box;
        t->started = false;
        return kOutOfResource;
    }
    t->host_handle = h;
    t->host_tid = tid;
    sched_apply(*t);
    ResumeThread(h);
    return kOk;
}

void thread_exit() {
    KThread* t = t_self;
    if (t) {
        NXLOG_D("thread", "%s exits (svcExitThread)", t->name.c_str());
        finish_current(*t);
        t_keepalive.reset();     // the object may die here: t must not be touched again
    }
    ExitThread(0);
    for (;;) {}
}

uint32_t main_thread_adopt(uint64_t stack_size, const char* name) {
    ThreadParams p;
    p.name = name ? name : "main";
    p.prio = 44;
    p.core = 0;
    const uint64_t stack = mem::alloc_kernel(stack_size, mem::kAllocGranularity);
    if (!stack) return 0;
    mem::vm_set(stack, stack_size, mem::kMemStack, 3);
    p.stack_top = stack + stack_size;
    uint32_t h = thread_create(p);
    if (!h) return 0;
    auto t = handle_as<KThread>(h);
    HANDLE real = nullptr;
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &real, THREAD_ALL_ACCESS, FALSE, 0);
    t->host_handle = real;
    t->host_tid = GetCurrentThreadId();
    t->started = true;
    t_self = t.get();
    log_set_thread_name(t->name.c_str());
    diag_register_thread(t->name.c_str(), &t->ctx, t->id);
    sched_apply(*t);
    occ_tag_main(*t);
    kernel_thread_named(*t, "main");
    NXLOG_I("thread", "host thread %lu is now the guest main thread (stack 0x%llx..0x%llx)", GetCurrentThreadId(), (unsigned long long)stack, (unsigned long long)p.stack_top);
    return h;
}

void main_thread_release() {
    if (KThread* t = t_self) finish_current(*t);
}

std::vector<std::shared_ptr<KThread>> threads() {
    AcquireSRWLockShared(&g_list_lock);
    auto v = g_all;
    ReleaseSRWLockShared(&g_list_lock);
    return v;
}

bool thread_join(uint32_t handle, uint32_t timeout_ms) {
    auto t = handle_as<KThread>(handle);
    if (!t || !t->started) return true;
    HANDLE h = static_cast<HANDLE>(t->host_handle);
    if (!h) return true;
    return WaitForSingleObject(h, timeout_ms) == WAIT_OBJECT_0;
}

uint64_t thread_count_running() {
    uint64_t n = 0;
    for (auto& t : threads()) n += t->started && !t->exited;
    return n;
}

bool thread_set_priority(KThread& t, int hos_prio) {
    if (hos_prio < 0 || hos_prio > 63) return false;
    t.prio = hos_prio;
    sched_apply(t);
    return true;
}

// Application cores: 3 on Switch 1 (cores 0..2), 6 on Switch 2 (cores 0..5, the game's NPDM ThreadInfo).
// [sched] app_cores = N (experiment, notes/threads.md): the process is told it has N cores (GetInfo CoreMask, core numbers up to N - 1),
// 1..16; a game that sizes its thread pools from the core mask then makes more threads. Cores past the host's physical cores get ideal
// processors on SMT siblings (sched.cpp). Read once.
int app_core_count() {
    static const int n = [] {
        const int def = game::info().platform == game::Platform::Switch2 ? 6 : 3;
        const int64_t v = config_get_int("sched", "app_cores", 0);
        if (v <= 0) return def;
        const int k = static_cast<int>(v > 16 ? 16 : v);
        if (k != def) NXLOG_I("sched", "[sched] app_cores = %d: the process is told it has %d cores (mask 0x%llx), not %d", k, k, (unsigned long long)((1ull << k) - 1), def);
        return k;
    }();
    return n;
}

bool thread_set_core_mask(KThread& t, int ideal, uint64_t mask) {
    const int n = app_core_count();
    uint64_t usable = mask & ((1ull << n) - 1);
    if (n == 3 && (mask & 8)) usable |= 4;   // Switch 1: the system core maps onto core 2
    if (!usable) return false;
    if (ideal == -2) ideal = 0;
    if (ideal == -1) {                                 // IdealCoreDontCare: the first core of the mask
        ideal = 0;
        for (int k = 0; k < n; k++)
            if (usable >> k & 1) { ideal = k; break; }
    }
    if (ideal >= 0) {
        if (ideal > n - 1) ideal = n - 1;
        if (!(usable >> ideal & 1)) {                 // ideal core outside the mask: take the first core of the mask
            for (int k = 0; k < n; k++)
                if (usable >> k & 1) { ideal = k; break; }
        }
        t.ideal_core = ideal;
    }
    t.core_mask = usable;
    sched_apply(t);
    return true;
}

void kernel_reset() {
    std::vector<std::shared_ptr<KThread>> all = threads();
    for (auto& t : all) {
        if (t.get() == t_self || !t->started || t->exited || !t->host_handle) continue;
        if (WaitForSingleObject(static_cast<HANDLE>(t->host_handle), 5000) != WAIT_OBJECT_0)
            NXLOG_E("thread", "kernel_reset: %s is still running after 5 s", t->name.c_str());
    }
    AcquireSRWLockExclusive(&g_list_lock);
    g_all.clear();
    ReleaseSRWLockExclusive(&g_list_lock);
    handles_reset();
    detail::sync_reset();
}

}  // namespace nxrt::kernel
