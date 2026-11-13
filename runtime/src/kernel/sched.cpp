// SPDX-License-Identifier: GPL-3.0-or-later
// Console core model. The application cores of the console (3 on Switch 1, 6 on Switch 2) are as many host processors; every guest
// thread's affinity is its core mask mapped onto them and its HOS priority is mapped onto six Windows priority bands with the dynamic
// boost off. One host processor per console core matters: threads of one core never run at the same time, and GetCurrentProcessorNumber
// names the thread's real core (Mario Kart World keeps per-core allocator state; with Switch 2's cores 3..5 folded onto 0..2 two cores
// shared that state and its loads crashed at a different place every run).
// Measured in smo-recomp (runtime/os/os_threads.c): free scheduling failed 4 of 6 boots in sead heap code, the console model 0 of 6.
// Config: [sched] console_cores (default 1; env NXRT_SCHED_CONSOLE_CORES), [sched] host_cpus = hexadecimal mask of host processors to use,
// [sched] pick = auto (the three fastest free physical cores, one hardware thread each) | top (the old rule: every second processor from the top of the mask).
// [sched] affinity = soft (default) | strict. strict pins each thread to its core's processor (the model above). soft makes that processor the
// thread's ideal processor and lets it run on any processor of a wide pool (every usable processor except the physical core of processor 0, where
// most DPC and interrupt work lands): the p2 soak's 100+ ms stalls were ready threads pinned to one busy processor while the machine sat idle
// (debug note 422: idle 2000+ ms of CPU in the 100 ms window, one thread of ours holding the processor the whole window). Under soft, threads
// of one console core may run at the same time and GetCurrentProcessorNumber answers the thread's ideal core.
#include <windows.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

#include "nxrt/config.h"
#include "nxrt/game.h"
#include "nxrt/kernel.h"
#include "nxrt/log.h"

namespace nxrt::kernel {
namespace {

std::mutex g_lock;
bool g_done = false;
SchedInfo g_info;
HANDLE g_slot_mutex = nullptr;

// The physical cores of processor group 0 inside `proc`, fastest first (a hybrid CPU reports a higher EfficiencyClass for its performance cores), each
// as the mask of ONE hardware thread of the core (its lowest): a guest thread never shares a core with another guest thread, and never lands on an
// efficiency core while a performance core is free. Empty when the system call fails.
std::vector<uint64_t> cores_fastest_first(uint64_t proc, bool full = false) {
    std::vector<uint64_t> out;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (!len) return out;
    std::vector<uint8_t> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) return out;
    struct C { int eff; uint64_t first; };
    std::vector<C> cores;
    for (DWORD off = 0; off < len;) {
        auto* e = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
        if (e->Relationship == RelationProcessorCore && e->Processor.GroupCount >= 1 && e->Processor.GroupMask[0].Group == 0) {
            const uint64_t m = static_cast<uint64_t>(e->Processor.GroupMask[0].Mask) & proc;
            if (m) cores.push_back({e->Processor.EfficiencyClass, full ? m : m & (~m + 1)});
        }
        off += e->Size;
    }
    std::stable_sort(cores.begin(), cores.end(), [](const C& a, const C& b) { return a.eff > b.eff || (a.eff == b.eff && (a.first & (~a.first + 1)) < (b.first & (~b.first + 1))); });
    for (const C& c : cores) out.push_back(c.first);
    return out;
}

// The processors of the physical core that holds processor 0 of group 0 (processor 0 and its SMT siblings). 1 when the call fails.
uint64_t cpu0_core_mask() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (!len) return 1;
    std::vector<uint8_t> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) return 1;
    for (DWORD off = 0; off < len;) {
        auto* e = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
        if (e->Relationship == RelationProcessorCore && e->Processor.GroupCount >= 1 && e->Processor.GroupMask[0].Group == 0 &&
            (e->Processor.GroupMask[0].Mask & 1))
            return static_cast<uint64_t>(e->Processor.GroupMask[0].Mask);
        off += e->Size;
    }
    return 1;
}

int lowest_bit(uint64_t m) {
    for (int b = 0; b < 64; b++)
        if (m >> b & 1) return b;
    return 0;
}

// Slot k = the n fastest cores not taken by a lower slot (a named mutex per slot, held while the process lives). Slots of 3 and of 6
// cores have separate names, so a Switch 1 and a Switch 2 process side by side may overlap; two of one kind never do.
uint64_t take_core_slot(const std::vector<uint64_t>& cores, size_t n, int* slot_out) {
    for (size_t k = 0; k + n <= cores.size(); k += n) {
        char name[64];
        if (n == 3) snprintf(name, sizeof name, "Local\\nxrt_core_slot_%zu", k / n);
        else snprintf(name, sizeof name, "Local\\nxrt_core_slot%zu_%zu", n, k / n);
        HANDLE m = CreateMutexA(nullptr, FALSE, name);
        if (!m) break;
        if (GetLastError() == ERROR_ALREADY_EXISTS) { CloseHandle(m); continue; }
        if (g_slot_mutex) CloseHandle(g_slot_mutex);
        g_slot_mutex = m;
        uint64_t mask = 0;
        for (size_t i = 0; i < n; ++i) mask |= cores[k + i];
        *slot_out = static_cast<int>(k / n);
        NXLOG_I("sched", "console cores: slot %zu of %zu, the %zu-th to %zu-th fastest physical cores (mask %llx); other nxrt processes take other slots", k / n, cores.size() / n, k + 1, k + n, (unsigned long long)mask);
        return mask;
    }
    return 0;
}

// The first free slot of six processors counted from the top of `proc` (a named mutex per slot, held while the process lives), so
// several games or test processes running side by side do not share processors. One process alone takes slot 0.
uint64_t take_slot(uint64_t proc, int* slot_out) {
    int bits[64], nb = 0;
    for (int b = 63; b >= 0; b--)
        if (proc >> b & 1) bits[nb++] = b;
    const int per = 2 * (game::info().platform == game::Platform::Switch2 ? 6 : 3);   // two hardware threads per console core
    for (int k = 0; k < nb / per; k++) {
        char name[64];
        snprintf(name, sizeof name, "Local\\nxrt_console_cores_%d", k);
        HANDLE m = CreateMutexA(nullptr, FALSE, name);
        if (!m) break;
        if (GetLastError() == ERROR_ALREADY_EXISTS) { CloseHandle(m); continue; }
        if (g_slot_mutex) CloseHandle(g_slot_mutex);
        g_slot_mutex = m;
        uint64_t mask = 0;
        for (int i = per * k; i < per * k + per; i++) mask |= 1ull << bits[i];
        *slot_out = k;
        NXLOG_I("sched", "console cores in processor slot %d of %d (mask %llx); other nxrt processes take other slots", k, nb / per, (unsigned long long)mask);
        return mask;
    }
    NXLOG_W("sched", "every processor slot is held by other processes: sharing the top of the affinity mask");
    *slot_out = -1;
    return proc;
}

// [sched] render_place = reserve (host_place.cpp): one physical core that holds no console core's processor (and is not CPU 0's core) is
// kept for the render thread, and the SMT sibling of console core 0's processor (main's) is kept free, so the two threads on the frame's critical
// path never share a core with a guest worker or a background job (on Zen 2 a busy sibling costs the other hardware thread a third of its speed).
// The free core nearest to main's is taken (on a Ryzen 3700X: core 7 next to main's core 6, the same CCX, so the command lists main writes stay
// in the L3 the render thread reads). Both leave the guest pool; the placer moves the runtime's and the driver's host threads off them too.
void apply_reserve() {
    std::string how = config_get_str("sched", "render_place", "off");
    for (char& ch : how) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (how != "reserve" || !g_info.active || !g_info.soft) return;
    const std::vector<uint64_t> full = cores_fastest_first(g_info.pool, true);   // physical cores of the pool, all their hardware threads
    uint64_t used = 0;
    for (int i = 0; i < g_info.cores; ++i) used |= g_info.host_masks[i];
    const int main_p = lowest_bit(g_info.host_masks[0]);
    uint64_t best = 0;
    int best_dist = 1 << 30;
    for (uint64_t c : full) {
        if (c & used) continue;
        const int d = std::abs(lowest_bit(c) - main_p);
        if (d < best_dist) best_dist = d, best = c;
    }
    if (!best) {
        NXLOG_W("sched", "[sched] render_place = reserve: no free physical core left beside the console cores; render thread placement as `apart`");
        return;
    }
    uint64_t main_sib = 0;
    for (uint64_t c : full)
        if (c & g_info.host_masks[0]) main_sib = c & ~g_info.host_masks[0];
    g_info.render_core = best;
    g_info.reserved = best | main_sib;
    g_info.pool &= ~g_info.reserved;
    NXLOG_I("sched", "render_place = reserve: processors %llx kept for the render thread, main's sibling %llx kept free; guest pool %llx",
            (unsigned long long)best, (unsigned long long)main_sib, (unsigned long long)g_info.pool);
}

void log_model() {
    std::string m;
    char buf[32];
    for (int i = 0; i < g_info.cores; ++i) {
        snprintf(buf, sizeof buf, " %llx", (unsigned long long)g_info.host_masks[i]);
        m += buf;
    }
    if (g_info.soft)
        NXLOG_I("sched", "console core model ON, affinity soft: ideal processors (as masks)%s for cores 0..%d, every guest thread may run on processors %llx (%d of them)",
                m.c_str(), g_info.cores - 1, (unsigned long long)g_info.pool, __builtin_popcountll(g_info.pool));
    else
        NXLOG_I("sched", "console core model ON, affinity strict: host masks%s for cores 0..%d", m.c_str(), g_info.cores - 1);
}

void init_locked() {
    if (g_done) return;
    g_done = true;
    g_info = SchedInfo{};
    g_info.cores = app_core_count();            // 3 / 6, or [sched] app_cores
    const int nc = g_info.cores;
    if (!config_get_bool("sched", "console_cores", true)) {
        NXLOG_I("sched", "console core model off ([sched] console_cores = 0)");
        return;
    }
    std::string aff = config_get_str("sched", "affinity", "soft");
    for (char& ch : aff) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (aff != "soft" && aff != "strict") NXLOG_W("sched", "[sched] affinity = %s is not soft or strict: soft", aff.c_str());
    g_info.soft = aff != "strict";
    DWORD_PTR proc = 0, sys = 0;
    GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys);
    uint64_t usable = proc;
    std::string cpus = config_get_str("sched", "host_cpus", "");
    int slot = -1;
    if (!cpus.empty()) {
        uint64_t want = _strtoui64(cpus.c_str(), nullptr, 16);
        if (want & usable) usable &= want;
        NXLOG_I("sched", "host_cpus %s: using processors %llx of %llx", cpus.c_str(), (unsigned long long)usable, (unsigned long long)proc);
    } else if (g_info.soft) {                   // soft: leave the physical core of processor 0 (DPCs, interrupts) out when enough remain
        const uint64_t trimmed = usable & ~cpu0_core_mask();
        if (cores_fastest_first(trimmed).size() >= static_cast<size_t>(nc)) usable = trimmed;
    }
    g_info.pool = usable;                       // soft: every guest thread's affinity; strict: the widened affinity of a starvation boost
    if (cpus.empty()) {
        bool top = config_get_str("sched", "pick", "auto") == "top";
        uint64_t chosen = 0;
        if (!top) {
            const std::vector<uint64_t> cores = cores_fastest_first(usable);
            if (cores.size() >= static_cast<size_t>(nc)) chosen = take_core_slot(cores, static_cast<size_t>(nc), &slot);
            else if (g_info.soft && cores.size() >= 2 && nc <= static_cast<int>(2 * cores.size())) {
                // [sched] app_cores above the physical cores: cores 0..P-1 take the first hardware thread of each physical core as before
                // (core 0 the highest), cores P.. the SMT siblings, starting with the sibling of the last core's processor (core P-1: the
                // console's system-side core, lightly loaded) and ending with main's (core 0). Ideal processors only: the affinity is the pool.
                const std::vector<uint64_t> full = cores_fastest_first(usable, true);
                const int P = static_cast<int>(cores.size());
                uint64_t firsts = 0;
                for (uint64_t c : cores) firsts |= c;
                int k = 0;
                for (int b = 63; b >= 0 && k < P; b--)
                    if (firsts >> b & 1) g_info.host_masks[k++] = 1ull << b;
                for (int i = P; i < nc; i++) {
                    const uint64_t first = g_info.host_masks[P - 1 - (i - P) % P];
                    uint64_t sib = first;
                    for (uint64_t f : full)
                        if (f & first) { const uint64_t rest = f & ~first; if (rest) sib = rest & (~rest + 1); }
                    g_info.host_masks[i] = sib;
                }
                g_info.active = true;
                NXLOG_I("sched", "[sched] app_cores = %d on %d physical cores: cores %d..%d have SMT siblings as ideal processors", nc, P, P, nc - 1);
                log_model();
                return;
            }
        }
        if (chosen) {
            int k = 0;
            for (int b = 63; b >= 0 && k < nc; b--)
                if (chosen >> b & 1) g_info.host_masks[k++] = 1ull << b;
            g_info.slot = slot;
            g_info.active = k == nc;
            apply_reserve();
            log_model();
            return;
        }
        usable = take_slot(usable, &slot);
    }
    int n = 0;
    for (int b = 63; b >= 0 && n < nc; b--)     // from the top, every second processor: not two hardware threads of one core
        if (usable >> b & 1) {
            g_info.host_masks[n++] = 1ull << b;
            b--;
        }
    g_info.slot = slot;
    g_info.active = n == nc;
    if (!g_info.active) {                       // second chance without skipping: any processors
        n = 0;
        for (int b = 63; b >= 0 && n < nc; b--)
            if (usable >> b & 1) g_info.host_masks[n++] = 1ull << b;
        g_info.active = n == nc;
    }
    if (g_info.active)
        log_model();
    else
        NXLOG_W("sched", "console core model NOT possible: fewer than %d usable host processors", nc);
}

}  // namespace

SchedInfo sched_info() {
    std::lock_guard<std::mutex> lk(g_lock);
    init_locked();
    return g_info;
}

void sched_reset_for_tests() {
    std::lock_guard<std::mutex> lk(g_lock);
    g_done = false;
    g_info = SchedInfo{};
    if (g_slot_mutex) { CloseHandle(g_slot_mutex); g_slot_mutex = nullptr; }     // so the next sched_info() can take the same slot again
}

// nn::os priorities run 0 (highest) .. 31 (lowest); HOS priority = nn::os + 28. Six bands under the model:
// <= 4, 5..8, 9..15, 16, 17, 18..31 (smo: 17 against 18 on one core decided a use-after-free race in 2 of 91 stage-entry runs).
int sched_host_priority(int hos_prio) {
    const int nn = hos_prio - 28;
    if (sched_info().active) {
        if (nn <= 4) return THREAD_PRIORITY_TIME_CRITICAL;
        if (nn <= 8) return THREAD_PRIORITY_HIGHEST;
        if (nn <= 15) return THREAD_PRIORITY_ABOVE_NORMAL;
        if (nn == 16) return THREAD_PRIORITY_NORMAL;
        if (nn == 17) return THREAD_PRIORITY_BELOW_NORMAL;
        return THREAD_PRIORITY_LOWEST;
    }
    if (nn <= 4) return THREAD_PRIORITY_HIGHEST;
    if (nn <= 15) return THREAD_PRIORITY_ABOVE_NORMAL;
    if (nn == 16) return THREAD_PRIORITY_NORMAL;
    if (nn <= 27) return THREAD_PRIORITY_BELOW_NORMAL;
    return THREAD_PRIORITY_LOWEST;
}

void sched_set_process_priority(bool interactive) {
    std::string v = config_get_str("sched", "priority_class", "auto");
    for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    DWORD cls = 0;
    if (v == "high") cls = HIGH_PRIORITY_CLASS;
    else if (v == "above_normal" || v == "above") cls = ABOVE_NORMAL_PRIORITY_CLASS;
    else if (v == "normal") cls = NORMAL_PRIORITY_CLASS;
    else if (v == "auto") cls = interactive ? HIGH_PRIORITY_CLASS : 0;
    else NXLOG_W("sched", "[sched] priority_class = %s is not auto, high, above_normal or normal: left alone", v.c_str());
    if (!cls) return;
    if (SetPriorityClass(GetCurrentProcess(), cls)) NXLOG_I("sched", "process priority class: %s", cls == HIGH_PRIORITY_CLASS ? "high" : cls == ABOVE_NORMAL_PRIORITY_CLASS ? "above normal" : "normal");
    else NXLOG_W("sched", "SetPriorityClass failed, error %lu", static_cast<unsigned long>(GetLastError()));
}

void yield_host() {
    KThread* t = current_thread();
    if (t && t->aged.load(std::memory_order_relaxed)) YieldProcessor();   // aged: a hand-over would give its time slice to the program that starves it
    else SwitchToThread();
}

int sched_current_core() {
    SchedInfo si = sched_info();
    KThread* t = current_thread();
    if (si.active && si.soft) {                     // soft: the thread's ideal core (a core of its mask), whatever processor it is on
        if (!t) return 0;
        if (t->ideal_core >= 0 && (t->core_mask >> t->ideal_core & 1)) return t->ideal_core;
        return t->core_mask ? lowest_bit(t->core_mask) : 0;
    }
    if (si.active) {
        PROCESSOR_NUMBER pn{};
        GetCurrentProcessorNumberEx(&pn);
        if (pn.Group == 0)
            for (int i = 0; i < si.cores; i++)
                if (si.host_masks[i] >> pn.Number & 1) return i;
    }
    return t ? (t->ideal_core >= 0 ? t->ideal_core : 0) : 0;
}

void sched_apply(KThread& t) {
    HANDLE h = static_cast<HANDLE>(t.host_handle);
    if (!h) return;
    SchedInfo si = sched_info();
    if (si.active && si.soft) {
        // The pool as affinity, the console core's processor as the ideal one: Windows runs the thread there when it is free and on any idle
        // processor of the pool otherwise, instead of leaving it ready behind whatever holds that one processor.
        const int c = t.ideal_core >= 0 && (t.core_mask >> t.ideal_core & 1) ? t.ideal_core : (t.core_mask ? lowest_bit(t.core_mask) : 0);
        const DWORD_PTR m = static_cast<DWORD_PTR>(si.pool);
        SetThreadAffinityMask(h, m);
        PROCESSOR_NUMBER pn{};
        pn.Group = 0;
        pn.Number = static_cast<BYTE>(lowest_bit(si.host_masks[c % si.cores]));
        SetThreadIdealProcessorEx(h, &pn, nullptr);
        SetThreadPriorityBoost(h, TRUE);            // Windows' dynamic boost off
        t.host_affinity.store(m, std::memory_order_relaxed);
    } else if (si.active) {
        DWORD_PTR m = 0;
        for (int i = 0; i < 16; i++)                // a core the model lacks (core 3..5 under a 3-core model) shares a host processor
            if (t.core_mask >> i & 1) m |= static_cast<DWORD_PTR>(si.host_masks[i % si.cores]);
        if (!m) m = static_cast<DWORD_PTR>(si.host_masks[0]);
        SetThreadAffinityMask(h, m);
        SetThreadPriorityBoost(h, TRUE);            // Windows' dynamic boost off
        t.host_affinity.store(m, std::memory_order_relaxed);
    }
    SetThreadPriority(h, sched_host_priority(t.prio));
}

}  // namespace nxrt::kernel
