// SPDX-License-Identifier: GPL-3.0-or-later
// Kernel-level synchronisation with the semantics of the Horizon SVCs: ArbitrateLock/Unlock (guest mutex words),
// WaitProcessWideKeyAtomic/SignalProcessWideKey (condition variables), WaitForAddress/SignalToAddress, WaitSynchronization and
// CancelSynchronization. The recompiled SDK's nn::os mutexes and condition variables sit on these.
// Logic ported from SuperMarioWonderRecomp runtime/src/kernel.cpp (MIT, Copyright (c) 2026 BelmanteGu; see THIRD_PARTY_NOTICES.md),
// reworked: one SRWLOCK kernel lock, a condition variable per thread, waiters woken in priority order, a pending
// CancelSynchronization is remembered until the next wait.
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>

#include "kernel_internal.h"
#include "nxrt/config.h"
#include "nxrt/gauge.h"
#include "nxrt/kernel.h"
#include "nxrt/log.h"

namespace nxrt::kernel {
namespace detail {

SRWLOCK g_k = SRWLOCK_INIT;

namespace {
std::vector<KThread*> g_sync_waiters;    // in WaitSynchronization: woken by any change of waitable state
std::vector<KThread*> g_mutex_waiters;   // waiting for a guest mutex word to be handed over
std::vector<KThread*> g_cv_waiters;      // waiting on a process-wide key
std::vector<KThread*> g_addr_waiters;    // WaitForAddress

struct Guard {
    Guard() {
        if (!g_occ_on) AcquireSRWLockExclusive(&g_k);
        else if (!TryAcquireSRWLockExclusive(&g_k)) occ_lock_contended();   // [kernel] occupancy: counts g_k contention
    }
    ~Guard() { ReleaseSRWLockExclusive(&g_k); }
};

void remove_from(std::vector<KThread*>& v, KThread* t) { v.erase(std::remove(v.begin(), v.end(), t), v.end()); }

uint32_t load32(uint64_t a) { return __atomic_load_n(reinterpret_cast<uint32_t*>(a), __ATOMIC_SEQ_CST); }
void store32(uint64_t a, uint32_t v) { __atomic_store_n(reinterpret_cast<uint32_t*>(a), v, __ATOMIC_SEQ_CST); }
bool cas32(uint64_t a, uint32_t expected, uint32_t v) {
    return __atomic_compare_exchange_n(reinterpret_cast<uint32_t*>(a), &expected, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

// The calling guest thread, or a per-host-thread stand-in (tests and host threads that call kernel functions directly).
KThread& self() {
    if (KThread* t = current_thread()) return *t;
    static thread_local KThread host;
    return host;
}

// Highest priority (lowest number) first, FIFO among equals: the order in which Horizon hands a mutex over or signals a key.
KThread* pick_best(std::vector<KThread*>& v, uint64_t KThread::*field, uint64_t value) {
    KThread* best = nullptr;
    for (KThread* t : v)
        if (t->*field == value && (!best || t->prio < best->prio)) best = t;
    return best;
}

// Hands the mutex word at addr to its best waiter (or clears it). g_k held.
void unlock_mutex(uint64_t addr) {
    KThread* next = pick_best(g_mutex_waiters, &KThread::wait_mutex, addr);
    if (!next) {
        store32(addr, 0);
        return;
    }
    int more = 0;
    for (KThread* t : g_mutex_waiters)
        if (t != next && t->wait_mutex == addr) more++;
    remove_from(g_mutex_waiters, next);
    store32(addr, next->wait_tag | (more ? kHandleWaitMask : 0));
    wake_thread(*next);
}

// Monotonic nanoseconds from the registered console's tick frequency, split so the multiplication cannot overflow.
int64_t now_ns() {
    const uint64_t t = system_tick();
    const uint64_t hz = tick_hz();
    return static_cast<int64_t>((t / hz) * 1000000000ull + (t % hz) * 1000000000ull / hz);
}

// set_blocked_hook: the host thread that runs a hook while it is blocked in a kernel wait, and the hook.
std::atomic<DWORD> g_blocked_tid{0};
std::atomic<void (*)()> g_blocked_fn{nullptr};

int64_t qpc_now() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return c.QuadPart;
}

// [kernel] wait_stats = <seconds> (env NXRT_KERNEL_WAIT_STATS, default 0 = off): per guest thread, how often it blocked in WaitForAddress,
// ArbitrateLock and WaitProcessWideKeyAtomic, for how long, and the wake latency (wake_thread to the thread running again with the kernel
// lock: the host's wake-up plus the lock), logged every <seconds>. Counted under g_k.
struct WaitStats {
    std::string name;
    int prio = 0, ideal = 0;
    uint64_t mask = 0;
    uint64_t waits[3] = {}, blocked[3] = {};           // per kind: blocks, QPC ticks blocked
    uint64_t woken = 0, spun = 0, lat_sum = 0, lat_max = 0, hist[8] = {};
};
constexpr int64_t kLatBoundsUs[7] = {5, 10, 20, 50, 100, 200, 500};
std::atomic<std::unordered_map<const KThread*, WaitStats>*> g_stats{nullptr};
std::unordered_map<const KThread*, int64_t> g_woken_at;   // g_k held; only while wait_stats is on

// [kernel] spin_us (env NXRT_KERNEL_SPIN_US, default 0 = off): a thread about to sleep in one of those waits first polls its wake flag
// this long with g_k released, so a signal that comes within it reaches the thread without the host's sleep and wake-up.
// [kernel] spin_threads / lock_spin_threads (comma lists of game thread name prefixes, `main` = the main thread; empty = every thread): only
// those threads spin (spin_us) or poll a held mutex (lock_spin_us). spin_us for every thread was a loss in Dagsion (cpu-1 H: the job workers
// and the GFX workers spin each time they go idle, on the processors and SMT siblings the game and render threads need); the waits that the
// frame waits for are main's and the job workers' it waits on.
struct WaitTuning {
    int64_t hz = 1, spin = 0, stats_s = 0, lock_spin = 0;
    std::vector<std::string> spin_names, lock_names;      // empty: every thread
};
std::vector<std::string> name_list(const std::string& csv) {
    std::vector<std::string> v;
    size_t i = 0;
    while (i <= csv.size()) {
        size_t j = csv.find(',', i);
        if (j == std::string::npos) j = csv.size();
        std::string w = csv.substr(i, j - i);
        while (!w.empty() && w.front() == ' ') w.erase(w.begin());
        while (!w.empty() && w.back() == ' ') w.pop_back();
        if (!w.empty()) v.push_back(w);
        i = j + 1;
    }
    return v;
}
bool name_listed(const std::vector<std::string>& v, const char* name) {
    for (const std::string& w : v)
        if (!std::strncmp(name, w.c_str(), w.size())) return true;
    return false;
}
void stats_report_loop(int64_t seconds);
const WaitTuning& wait_tuning() {      // first called before g_k is taken (config reads log)
    static const WaitTuning w = [] {
        WaitTuning v;
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        v.hz = f.QuadPart;
        const int64_t us = config_get_int("kernel", "spin_us", 0);
        v.spin = us > 0 ? us * v.hz / 1000000 : 0;
        v.stats_s = config_get_int("kernel", "wait_stats", 0);
        // [kernel] lock_spin_us (experiment, notes/threads.md): ArbitrateLock first polls the mutex word this long without the kernel lock and
        // returns (the guest retries its compare-and-swap) when the owner has released it, before it queues as a waiter. A queued waiter is
        // handed the mutex and owns it until the host wakes it (5-10 us), so contended locks convoy; a spinner never makes that hand-over.
        const int64_t lus = config_get_int("kernel", "lock_spin_us", 0);
        v.lock_spin = lus > 0 ? lus * v.hz / 1000000 : 0;
        const std::string sl = config_get_str("kernel", "spin_threads", ""), ll = config_get_str("kernel", "lock_spin_threads", "");
        v.spin_names = name_list(sl);
        v.lock_names = name_list(ll);
        if (v.lock_spin) NXLOG_I("kernel", "ArbitrateLock polls the mutex word %lld us before queueing ([kernel] lock_spin_us), threads: %s", static_cast<long long>(lus),
                                 ll.empty() ? "all" : ll.c_str());
        if (v.spin) NXLOG_I("kernel", "waits spin %lld us before sleeping ([kernel] spin_us), threads: %s", static_cast<long long>(us), sl.empty() ? "all" : sl.c_str());
        if (v.stats_s > 0) {
            g_stats.store(new std::unordered_map<const KThread*, WaitStats>, std::memory_order_release);
            std::thread(stats_report_loop, v.stats_s).detach();
        }
        return v;
    }();
    return w;
}

void stats_report_loop(int64_t seconds) {
    const double us_per_tick = 1e6 / static_cast<double>(wait_tuning().hz);
    for (;;) {
        Sleep(static_cast<DWORD>(seconds * 1000));
        std::unordered_map<const KThread*, WaitStats> snap;
        AcquireSRWLockExclusive(&g_k);
        snap.swap(*g_stats.load(std::memory_order_acquire));
        ReleaseSRWLockExclusive(&g_k);
        std::vector<const WaitStats*> rows;
        for (auto& kv : snap) rows.push_back(&kv.second);
        std::sort(rows.begin(), rows.end(), [](const WaitStats* a, const WaitStats* b) {
            return a->blocked[0] + a->blocked[1] + a->blocked[2] > b->blocked[0] + b->blocked[1] + b->blocked[2];
        });
        for (const WaitStats* s : rows) {
            const uint64_t n = s->waits[0] + s->waits[1] + s->waits[2];
            if (!n) continue;
            char hist[160];
            int len = 0;
            for (int i = 0; i < 8 && s->woken; i++)
                len += std::snprintf(hist + len, sizeof hist - len, "%s%s%lld %.0f%%", i ? ", " : "", i < 7 ? "<" : ">=",
                                     static_cast<long long>(kLatBoundsUs[i < 7 ? i : 6]), 100.0 * s->hist[i] / s->woken);
            NXLOG_I("kernel", "waits in %llds: %s (prio %d, core %d of %llx) blocked %llu times %.1f ms (address %llu %.1f ms, mutex %llu %.1f ms, cv %llu %.1f ms), "
                    "spin caught %llu; woken %llu, wake latency mean %.1f us max %.0f us (us: %s)", static_cast<long long>(seconds), s->name.c_str(), s->prio,
                    s->ideal, static_cast<unsigned long long>(s->mask),
                    static_cast<unsigned long long>(n), (s->blocked[0] + s->blocked[1] + s->blocked[2]) * us_per_tick / 1000.0,
                    static_cast<unsigned long long>(s->waits[0]), s->blocked[0] * us_per_tick / 1000.0, static_cast<unsigned long long>(s->waits[1]),
                    s->blocked[1] * us_per_tick / 1000.0, static_cast<unsigned long long>(s->waits[2]), s->blocked[2] * us_per_tick / 1000.0,
                    static_cast<unsigned long long>(s->spun), static_cast<unsigned long long>(s->woken),
                    s->woken ? s->lat_sum * us_per_tick / static_cast<double>(s->woken) : 0.0, s->lat_max * us_per_tick, s->woken ? hist : "-");
        }
    }
}

}  // namespace

// One sleep on t's condition variable (g_k held, released while sleeping). On the thread set_blocked_hook names it sleeps 10 ms at most and,
// when that runs out, calls the hook with g_k released. Every caller loops and re-checks its condition, so a wake that comes while the hook
// runs is not lost.
void sleep_locked(KThread& t, DWORD ms) {
    void (*fn)() = g_blocked_fn.load(std::memory_order_acquire);
    if (fn && GetCurrentThreadId() == g_blocked_tid.load(std::memory_order_relaxed)) {
        if (!SleepConditionVariableSRW(thread_cv(t), &g_k, std::min<DWORD>(ms, 10), 0)) {
            ReleaseSRWLockExclusive(&g_k);
            fn();
            AcquireSRWLockExclusive(&g_k);
        }
        return;
    }
    SleepConditionVariableSRW(thread_cv(t), &g_k, ms, 0);
}

// Waits (g_k held, released while sleeping) until t.wake or the timeout; false on timeout.
bool block_on_wake(KThread& t, int64_t timeout_ns, int kind) {
    const int64_t until = timeout_ns < 0 ? 0 : now_ns() + timeout_ns;
    if (t.wake) return true;
    const WaitTuning& wt = wait_tuning();
    auto* stats = g_stats.load(std::memory_order_relaxed);
    const bool spin_me = wt.spin && (wt.spin_names.empty() || t.spin_ok);
    const int64_t t0 = stats || spin_me ? qpc_now() : 0;
    bool spun = false, slept = false;
    if (g_occ_on) occ_block(t, kind);
    if (spin_me && timeout_ns != 0) {
        int64_t end = t0 + wt.spin;
        if (timeout_ns > 0 && timeout_ns < 1000000000) end = std::min(end, t0 + timeout_ns * wt.hz / 1000000000);
        ReleaseSRWLockExclusive(&g_k);
        while (!__atomic_load_n(&t.wake, __ATOMIC_ACQUIRE) && qpc_now() < end) YieldProcessor();
        AcquireSRWLockExclusive(&g_k);
        spun = t.wake;
    }
    auto done = [&](bool ok) {
        if (g_occ_on) occ_unblock(t);
        if (stats) {
            const int64_t now = qpc_now();
            WaitStats& s = (*stats)[&t];
            if (s.name.empty() || (t.game_name[0] && s.name == t.name)) s.name = t.game_name[0] ? std::string(t.game_name) + " (" + t.name + ")" : t.name;
            s.prio = t.prio, s.ideal = t.ideal_core, s.mask = t.core_mask;
            s.waits[kind]++;
            s.blocked[kind] += static_cast<uint64_t>(now - t0);
            s.spun += spun;
            auto w = g_woken_at.find(&t);
            if (ok && slept && w != g_woken_at.end()) {
                const uint64_t lat = static_cast<uint64_t>(std::max<int64_t>(0, now - w->second));
                const double us = static_cast<double>(lat) * 1e6 / static_cast<double>(wt.hz);
                int b = 0;
                while (b < 7 && us >= static_cast<double>(kLatBoundsUs[b])) b++;
                s.woken++, s.lat_sum += lat, s.lat_max = std::max(s.lat_max, lat), s.hist[b]++;
            }
            if (w != g_woken_at.end()) g_woken_at.erase(w);
        }
        return ok;
    };
    while (!t.wake) {
        DWORD ms = INFINITE;
        if (timeout_ns >= 0) {
            int64_t rem = until - now_ns();
            if (rem <= 0) return done(false);
            int64_t m = (rem + 999999) / 1000000;
            ms = static_cast<DWORD>(m > 0x7FFFFFFE ? 0x7FFFFFFE : m);
        }
        sleep_locked(t, ms);
        slept = true;
    }
    return done(true);
}

void wake_thread(KThread& t, uint32_t result) {
    t.wait_result = result;                                // before the flag: a spinner reads the result once it sees the flag
    __atomic_store_n(&t.wake, true, __ATOMIC_RELEASE);     // read without g_k by a thread spinning in block_on_wake
    if (g_stats.load(std::memory_order_relaxed)) g_woken_at[&t] = qpc_now();
    if (g_occ_on && t.occ_group == 1) occ_main_woken();
    WakeConditionVariable(thread_cv(t));
}

void thread_mark_exited(KThread& t) {
    Guard g;
    t.exited = true;
    for (KThread* w : g_sync_waiters) WakeConditionVariable(thread_cv(*w));
}

void sync_reset() {
    Guard g;
    g_sync_waiters.clear();
    g_mutex_waiters.clear();
    g_cv_waiters.clear();
    g_addr_waiters.clear();
}

}  // namespace detail

using namespace detail;

std::string sync_wait_report() {
    // Called from the watchdog thread while the process may be frozen: never block on the kernel lock.
    if (!TryAcquireSRWLockExclusive(&g_k)) return "kernel sync report: the kernel lock is held (a thread is inside a sync call)\n";
    struct Release { ~Release() { ReleaseSRWLockExclusive(&g_k); } } release;
    std::string out = "kernel sync waiters:\n";
    char b[256];
    auto list = [&](const char* kind, const std::vector<KThread*>& v, uint64_t KThread::*field) {
        for (KThread* t : v) {
            const uint64_t a = t->*field;
            std::snprintf(b, sizeof b, "  %-18s %s (guest %llu) address 0x%llx value 0x%08x tag 0x%x wake %d\n", kind, t->name.c_str(),
                          static_cast<unsigned long long>(t->id), static_cast<unsigned long long>(a), a ? load32(a) : 0u, t->wait_tag, t->wake ? 1 : 0);
            out += b;
        }
    };
    list("WaitForAddress", g_addr_waiters, &KThread::wait_addr);
    list("process-wide key", g_cv_waiters, &KThread::wait_key);
    list("mutex hand-over", g_mutex_waiters, &KThread::wait_mutex);
    std::snprintf(b, sizeof b, "  WaitSynchronization waiters: %zu\n", g_sync_waiters.size());
    out += b;
    return out;
}

uint32_t arbitrate_lock(uint32_t owner, uint64_t addr, uint32_t tag) {
    if (addr & 3) return kInvalidAddress;
    const WaitTuning& wt = wait_tuning();
    if (wt.lock_spin && (wt.lock_names.empty() || self().lock_spin_ok)) {
        const int64_t end = qpc_now() + wt.lock_spin;
        for (uint32_t i = 0; load32(addr) == (owner | kHandleWaitMask); i++) {
            if ((i & 15) == 15 && qpc_now() >= end) break;
            YieldProcessor();
        }
    }
    Guard g;
    KThread& me = self();
    if (load32(addr) != (owner | kHandleWaitMask)) return kOk;      // the owner already released it: the caller retries in guest code
    me.wake = false;
    me.wait_mutex = addr;
    me.wait_tag = tag;
    g_mutex_waiters.push_back(&me);
    block_on_wake(me, -1, 1);
    return me.wait_result;
}

uint32_t arbitrate_unlock(uint64_t addr) {
    if (addr & 3) return kInvalidAddress;
    Guard g;
    unlock_mutex(addr);
    return kOk;
}

uint32_t wait_process_wide_key(uint64_t mutex, uint64_t key, uint32_t tag, int64_t timeout_ns) {
    if (mutex & 3 || key & 3) return kInvalidAddress;
    wait_tuning();
    Guard g;
    KThread& me = self();
    // Like Horizon: the condition variable's "has waiters" word is written BEFORE the mutex is released. The SDK's Signal/Broadcast
    // read that word in guest code and skip the SVC when it is 0; the other order lost wake-ups (Wonder: a thread took the freed mutex,
    // posted, saw 0, and the waiter then slept forever).
    me.wake = false;
    me.wait_mutex = mutex;
    me.wait_key = key;
    me.wait_tag = tag;
    me.wait_result = 0;
    g_cv_waiters.push_back(&me);
    store32(key, 1);
    unlock_mutex(mutex);
    if (timeout_ns == 0 || !block_on_wake(me, timeout_ns, 2)) {
        // Timed out (not handed the mutex): leave the condition variable or the mutex queue a signal moved us to, return without
        // the mutex (the SDK re-locks it itself: TimedWait -> InternalCriticalSection::Enter).
        remove_from(g_cv_waiters, &me);
        remove_from(g_mutex_waiters, &me);
        bool others = std::any_of(g_cv_waiters.begin(), g_cv_waiters.end(), [&](KThread* t) { return t->wait_key == key; });
        if (!others) store32(key, 0);
        return kTimedOut;
    }
    return me.wait_result;
}

uint32_t signal_process_wide_key(uint64_t key, int32_t count) {
    if (key & 3) return kInvalidAddress;
    Guard g;
    int n = 0;
    while (count <= 0 || n < count) {
        KThread* t = pick_best(g_cv_waiters, &KThread::wait_key, key);
        if (!t) break;
        remove_from(g_cv_waiters, t);
        n++;
        // Try to give it the mutex right away.
        for (;;) {
            uint32_t v = load32(t->wait_mutex);
            if (v == 0) {
                if (!cas32(t->wait_mutex, 0, t->wait_tag)) continue;
                wake_thread(*t);
                break;
            }
            if (!(v & kHandleWaitMask) && !cas32(t->wait_mutex, v, v | kHandleWaitMask)) continue;
            g_mutex_waiters.push_back(t);
            break;
        }
    }
    bool others = std::any_of(g_cv_waiters.begin(), g_cv_waiters.end(), [&](KThread* t) { return t->wait_key == key; });
    if (!others) store32(key, 0);
    return kOk;
}

uint32_t wait_for_address(uint64_t addr, uint32_t type, int32_t value, int64_t timeout_ns) {
    if (addr & 3) return kInvalidAddress;
    if (type > 2) return kInvalidEnum;
    wait_tuning();
    Guard g;
    KThread& me = self();
    int32_t cur = static_cast<int32_t>(load32(addr));
    switch (type) {
        case 0:  // WaitIfLessThan
            if (!(cur < value)) return kInvalidState;
            break;
        case 1:  // DecrementAndWaitIfLessThan: one atomic step, as the kernel's exclusive load/store loop (guest threads change the word
                 // with their own atomics without the kernel lock; a plain load and store lost their updates)
            for (;;) {
                if (!(cur < value)) return kInvalidState;
                if (cas32(addr, static_cast<uint32_t>(cur), static_cast<uint32_t>(cur) - 1u)) break;
                cur = static_cast<int32_t>(load32(addr));
            }
            break;
        case 2:  // WaitIfEqual
            if (cur != value) return kInvalidState;
            break;
    }
    if (timeout_ns == 0) return kTimedOut;
    me.wake = false;
    me.wait_addr = addr;
    me.wait_result = 0;
    g_addr_waiters.push_back(&me);
    if (!block_on_wake(me, timeout_ns)) {
        remove_from(g_addr_waiters, &me);
        return kTimedOut;
    }
    return me.wait_result;
}

uint32_t signal_to_address(uint64_t addr, uint32_t type, int32_t value, int32_t count) {
    if (addr & 3) return kInvalidAddress;
    if (type > 2) return kInvalidEnum;
    Guard g;
    int waiters = static_cast<int>(std::count_if(g_addr_waiters.begin(), g_addr_waiters.end(), [&](KThread* t) { return t->wait_addr == addr; }));
    switch (type) {
        case 0:  // Signal
            break;
        case 1:  // SignalAndIncrementIfEqual
            if (!cas32(addr, static_cast<uint32_t>(value), static_cast<uint32_t>(value + 1))) return kInvalidState;
            break;
        case 2: {  // SignalAndModifyByWaitingCountIfEqual
            int32_t updated;
            if (count <= 0)
                updated = waiters ? value - 2 : value + 1;
            else if (waiters)
                updated = waiters <= count ? value - 1 : value;
            else
                updated = value + 1;
            if (!cas32(addr, static_cast<uint32_t>(value), static_cast<uint32_t>(updated))) return kInvalidState;
            break;
        }
    }
    int n = 0;
    while (count <= 0 || n < count) {
        KThread* t = pick_best(g_addr_waiters, &KThread::wait_addr, addr);
        if (!t) break;
        remove_from(g_addr_waiters, t);
        wake_thread(*t);
        n++;
    }
    return kOk;
}

uint32_t wait_synchronization(const uint32_t* handles, int32_t count, int64_t timeout_ns, int32_t* index_out) {
    if (count < 0 || count > 0x40) return kOutOfRange;
    std::vector<std::shared_ptr<KObject>> objs;
    for (int i = 0; i < count; i++) {
        auto o = handle_get(handles[i]);
        if (!o) {
            if (index_out) *index_out = -1;
            return kInvalidHandle;
        }
        objs.push_back(std::move(o));
    }
    auto ready = [&]() -> int {
        for (int i = 0; i < count; i++)
            if (objs[i]->signaled()) return i;
        return -1;
    };
    const int64_t until = timeout_ns <= 0 ? 0 : now_ns() + timeout_ns;
    Guard g;
    KThread& me = self();
    g_sync_waiters.push_back(&me);
    struct Leave {
        KThread* t;
        ~Leave() { remove_from(g_sync_waiters, t); }
    } leave{&me};
    for (;;) {
        if (me.cancel_sync) {
            me.cancel_sync = false;
            return kCancelled;
        }
        int i = ready();
        if (i >= 0) {
            if (index_out) *index_out = i;
            objs[i]->consume();
            return kOk;
        }
        if (timeout_ns == 0) return kTimedOut;
        DWORD ms = INFINITE;
        if (timeout_ns > 0) {
            int64_t rem = until - now_ns();
            if (rem <= 0) return kTimedOut;
            int64_t m = (rem + 999999) / 1000000;
            ms = static_cast<DWORD>(m > 0x7FFFFFFE ? 0x7FFFFFFE : m);
        }
        sleep_locked(me, ms);
    }
}

namespace detail {
void (*blocked_hook_here())() {
    void (*fn)() = g_blocked_fn.load(std::memory_order_acquire);
    return fn && GetCurrentThreadId() == g_blocked_tid.load(std::memory_order_relaxed) ? fn : nullptr;
}
}  // namespace detail

void set_blocked_hook(uint32_t host_tid, void (*fn)()) {
    g_blocked_tid.store(host_tid, std::memory_order_relaxed);
    g_blocked_fn.store(fn, std::memory_order_release);
}

uint32_t cancel_synchronization(uint32_t thread_handle) {
    auto t = handle_as<KThread>(thread_handle);
    if (!t) return kInvalidHandle;
    Guard g;
    t->cancel_sync = true;
    if (std::find(g_sync_waiters.begin(), g_sync_waiters.end(), t.get()) != g_sync_waiters.end()) WakeConditionVariable(thread_cv(*t));
    return kOk;
}

void notify_waitable_changed() {
    Guard g;
    for (KThread* t : g_sync_waiters) WakeConditionVariable(thread_cv(*t));
}

bool kernel_names_wanted() {
    const WaitTuning& wt = wait_tuning();
    return wt.stats_s > 0 || !wt.spin_names.empty() || !wt.lock_names.empty();
}

void kernel_thread_named(KThread& t, const char* game_name) {
    if (!game_name) return;
    const WaitTuning& wt = wait_tuning();
    Guard g;
    std::snprintf(t.game_name, sizeof t.game_name, "%s", game_name);
    t.spin_ok = wt.spin && name_listed(wt.spin_names, game_name);
    t.lock_spin_ok = wt.lock_spin && name_listed(wt.lock_names, game_name);
}

// ---- gauges (include/nxrt/gauge.h): wait-queue lengths ----
namespace {
// Removes its gauge when the file's statics are torn down (declared after the wait queues, so it goes first).
struct GaugeHook {
    const char* name;
    GaugeHook(const char* n, std::function<int64_t()> f) : name(n) { nxrt::gauge_register(n, std::move(f)); }
    ~GaugeHook() { nxrt::gauge_unregister(name); }
};

// Length of one wait queue under the kernel lock. The sampler never waits for the kernel lock: -1 while a thread holds it.
int64_t queue_len(const std::vector<KThread*>& v) {
    if (!TryAcquireSRWLockExclusive(&g_k)) return -1;
    const int64_t n = static_cast<int64_t>(v.size());
    ReleaseSRWLockExclusive(&g_k);
    return n;
}

GaugeHook gauge_wait_sync("kernel.waiters_sync", [] { return queue_len(g_sync_waiters); });
GaugeHook gauge_wait_mutex("kernel.waiters_mutex", [] { return queue_len(g_mutex_waiters); });
GaugeHook gauge_wait_cv("kernel.waiters_cv", [] { return queue_len(g_cv_waiters); });
GaugeHook gauge_wait_addr("kernel.waiters_addr", [] { return queue_len(g_addr_waiters); });
}  // namespace

}  // namespace nxrt::kernel
