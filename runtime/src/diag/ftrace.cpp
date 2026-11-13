// SPDX-License-Identifier: GPL-3.0-or-later
// Frame trace recorder (include/nxrt/ftrace.h). Events go into a preallocated array claimed with one atomic add; a writer thread
// appends them to the trace file once a second. File format, one record per line:
//   # zone <id> <name>        # thread <index> <os tid> <name>        <t_ns> <dur_ns> <thread index> <zone id> <arg>
#include "nxrt/ftrace.h"

#include <windows.h>

#include <intrin.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "nxrt/config.h"
#include "nxrt/diag.h"
#include "nxrt/log.h"

namespace nxrt::ft {

std::atomic<bool> g_on{false};

namespace {

struct Ev {
    uint64_t t0;
    uint32_t dur;      // ns, saturated
    uint16_t zone;
    uint16_t thread;
    uint32_t arg;
};

constexpr uint32_t kMaxThreads = 1024;

Ev* g_ev = nullptr;
size_t g_cap = 0;
std::atomic<size_t> g_n{0};
std::mutex g_mu;                                   // names and the file
std::vector<std::string> g_zone_names;
std::atomic<uint16_t> g_next_thread{0};
std::string g_thread_names[kMaxThreads];
uint32_t g_thread_tids[kMaxThreads];
std::atomic<bool> g_thread_named[kMaxThreads];
std::atomic<bool> g_stop{false};
std::thread g_writer;
FILE* g_file = nullptr;
size_t g_written = 0;
size_t g_zones_written = 0;
size_t g_threads_written = 0;
bool g_started = false;
thread_local int t_index = -1;

uint64_t qpc_freq() {
    static const uint64_t f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return static_cast<uint64_t>(x.QuadPart); }();
    return f;
}

int thread_index() {
    if (t_index >= 0) return t_index;
    const uint16_t i = g_next_thread.fetch_add(1);
    if (i >= kMaxThreads) { t_index = static_cast<int>(kMaxThreads - 1); return t_index; }
    g_thread_tids[i] = GetCurrentThreadId();
    t_index = i;
    return t_index;
}

void flush_locked() {
    if (!g_file) return;
    // names first so the reader knows every id it meets
    for (; g_zones_written < g_zone_names.size(); ++g_zones_written) std::fprintf(g_file, "# zone %zu %s\n", g_zones_written, g_zone_names[g_zones_written].c_str());
    const size_t n = std::min(g_n.load(std::memory_order_acquire), g_cap);
    const size_t nt = std::min<size_t>(g_next_thread.load(), kMaxThreads);
    for (; g_threads_written < nt; ++g_threads_written) {
        const char* nm = g_thread_names[g_threads_written].empty() ? "-" : g_thread_names[g_threads_written].c_str();
        std::fprintf(g_file, "# thread %zu %u %s\n", g_threads_written, g_thread_tids[g_threads_written], nm);
    }
    for (; g_written < n; ++g_written) {
        const Ev& e = g_ev[g_written];
        std::fprintf(g_file, "%llu %u %u %u %u\n", static_cast<unsigned long long>(e.t0), e.dur, e.thread, e.zone, e.arg);
    }
    std::fflush(g_file);
}

// The writer thread's flush: the new zone and thread names are taken under the lock, the events (appended lock-free) are formatted and
// written outside it. Holding the lock while fprintf wrote a second of events (tens of thousands of lines) stalled every zone_id() / mark()
// caller for that long, the render thread among them: a 30-45 ms frame once a second in every traced run.
void flush_unlocked() {
    std::string head;
    size_t n = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_file) return;
        char b[320];
        for (; g_zones_written < g_zone_names.size(); ++g_zones_written) {
            std::snprintf(b, sizeof b, "# zone %zu %s\n", g_zones_written, g_zone_names[g_zones_written].c_str());
            head += b;
        }
        n = std::min(g_n.load(std::memory_order_acquire), g_cap);
        const size_t nt = std::min<size_t>(g_next_thread.load(), kMaxThreads);
        for (; g_threads_written < nt; ++g_threads_written) {
            const char* nm = g_thread_names[g_threads_written].empty() ? "-" : g_thread_names[g_threads_written].c_str();
            std::snprintf(b, sizeof b, "# thread %zu %u %s\n", g_threads_written, g_thread_tids[g_threads_written], nm);
            head += b;
        }
    }
    if (!head.empty()) std::fwrite(head.data(), 1, head.size(), g_file);
    std::string out;
    out.reserve(1 << 20);
    char b[96];
    for (; g_written < n; ++g_written) {
        const Ev& e = g_ev[g_written];
        const int k = std::snprintf(b, sizeof b, "%llu %u %u %u %u\n", static_cast<unsigned long long>(e.t0), e.dur, e.thread, e.zone, e.arg);
        if (k > 0) out.append(b, static_cast<size_t>(k));
        if (out.size() >= (1u << 20)) {
            std::fwrite(out.data(), 1, out.size(), g_file);
            out.clear();
        }
    }
    if (!out.empty()) std::fwrite(out.data(), 1, out.size(), g_file);
    std::fflush(g_file);
}

void writer_main() {
    // [prof] trace_flush_locked = true: the old flush under the names lock (to compare)
    static const bool locked = config_get_bool("prof", "trace_flush_locked", false);
    while (!g_stop.load()) {
        for (int i = 0; i < 10 && !g_stop.load(); ++i) Sleep(100);
        if (locked) {
            std::lock_guard<std::mutex> lk(g_mu);
            flush_locked();
        } else {
            flush_unlocked();
        }
    }
}

}  // namespace

uint64_t now_ns() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    const uint64_t q = static_cast<uint64_t>(c.QuadPart), f = qpc_freq();
    return (q / f) * 1000000000ull + ((q % f) * 1000000000ull) / f;
}

uint16_t zone_id(const char* name) {
    std::lock_guard<std::mutex> lk(g_mu);
    for (size_t i = 0; i < g_zone_names.size(); ++i)
        if (g_zone_names[i] == name) return static_cast<uint16_t>(i);
    g_zone_names.emplace_back(name);
    return static_cast<uint16_t>(g_zone_names.size() - 1);
}

void event(uint16_t zone, uint64_t t0, uint64_t t1, uint32_t arg) {
    if (!g_on.load(std::memory_order_relaxed)) return;
    const size_t i = g_n.fetch_add(1, std::memory_order_relaxed);
    if (i >= g_cap) return;
    const int ti = thread_index();
    if (!g_thread_named[ti].load(std::memory_order_relaxed)) {
        const char* n = log_thread_name();
        g_thread_names[ti] = n ? n : "";
        g_thread_named[ti].store(true, std::memory_order_relaxed);
    }
    const uint64_t d = t1 > t0 ? t1 - t0 : 0;
    g_ev[i] = {t0, d > 0xffffffffull ? 0xffffffffu : static_cast<uint32_t>(d), zone, static_cast<uint16_t>(ti), arg};
}

void mark(const char* name, uint32_t arg) {
    if (!on()) return;
    const uint16_t z = zone_id(name);
    const uint64_t t = now_ns();
    event(z, t, t, arg);
}

namespace {
std::atomic<uint64_t> g_last_frame{0};
std::atomic<uint32_t> g_stall_ms{0};
std::thread g_sampler;

// [prof] stall_ms = N: when no frame was presented for N ms, dump every thread's stack to <trace>.stacks (up to 4 times per stall).
struct CpuSnap {
    std::vector<ThreadRecord> th;
    std::vector<uint64_t> cyc;
};

CpuSnap take_snap() {
    CpuSnap s;
    std::vector<ThreadRecord> recs = diag_threads_dup();   // private handles: an unregistering thread cannot close ours mid-query
    for (const ThreadRecord& r : recs) {
        ULONG64 c = 0;
        if (r.host_handle) QueryThreadCycleTime(static_cast<HANDLE>(r.host_handle), &c);
        s.cyc.push_back(c);
    }
    s.th = recs;
    for (ThreadRecord& r : s.th) r.host_handle = nullptr;   // the snapshot keeps tid/name only
    diag_threads_release(recs);
    return s;
}

double tsc_per_ns() {
    LARGE_INTEGER a, b;
    QueryPerformanceCounter(&a);
    const unsigned long long c0 = __rdtsc();
    Sleep(60);
    QueryPerformanceCounter(&b);
    const unsigned long long c1 = __rdtsc();
    const double ns = static_cast<double>(b.QuadPart - a.QuadPart) * 1e9 / static_cast<double>(qpc_freq());
    return ns > 0 ? static_cast<double>(c1 - c0) / ns : 3.0;
}

void sampler_main(std::string path) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    const double tpn = tsc_per_ns();
    uint64_t handled_frame = 0, prev_frame = 0;
    int dumps = 0;
    CpuSnap snap = take_snap();               // taken just after a frame was seen
    while (!g_stop.load()) {
        Sleep(3);
        const uint64_t last = g_last_frame.load();
        if (!last) continue;
        if (last != handled_frame) {
            // a new frame: if the gap before it was a stall, say what every thread's CPU time was during it
            if (dumps > 0) {
                const CpuSnap now = take_snap();
                std::fprintf(f, "--- stall ended: %.1f ms between frames; CPU ms per thread in that time:\n", (last - prev_frame) / 1e6);
                std::vector<std::pair<double, std::string>> v;
                for (size_t i = 0; i < now.th.size(); ++i)
                    for (size_t j = 0; j < snap.th.size(); ++j)
                        if (now.th[i].host_tid == snap.th[j].host_tid && now.cyc[i] >= snap.cyc[j]) {
                            v.emplace_back(static_cast<double>(now.cyc[i] - snap.cyc[j]) / tpn / 1e6, now.th[i].name);
                            break;
                        }
                std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first; });
                for (size_t i = 0; i < v.size() && i < 6; ++i) std::fprintf(f, "    cpu %-12s %7.1f ms\n", v[i].second.c_str(), v[i].first);
                std::fflush(f);
            }
            prev_frame = last;
            handled_frame = last;
            dumps = 0;
            snap = take_snap();
        }
        const uint64_t idle = now_ns() - last;
        const uint64_t want = static_cast<uint64_t>(g_stall_ms.load()) * 1000000ull + static_cast<uint64_t>(dumps) * 12000000ull;
        if (dumps < 4 && idle >= want) {
            const uint64_t t = now_ns();
            const std::string d = diag_thread_dump();
            std::fprintf(f, "=== stall sample %d at t=%llu ns, %.1f ms since the last frame\n%s\n", dumps, static_cast<unsigned long long>(t), idle / 1e6, d.c_str());
            std::fflush(f);
            ++dumps;
        }
    }
    std::fclose(f);
}
}  // namespace

void frame() {
    if (!on()) return;
    static const uint16_t z = zone_id("frame");
    const uint64_t t = now_ns();
    event(z, t, t, 0);
    g_last_frame.store(t);
}

void start_from_config() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_started) return;
    const std::string path = config_get_str("prof", "trace", "");
    if (path.empty()) return;
    g_cap = static_cast<size_t>(config_get_int("prof", "trace_events", 6000000));
    g_ev = static_cast<Ev*>(VirtualAlloc(nullptr, g_cap * sizeof(Ev), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!g_ev) return;
    g_file = std::fopen(path.c_str(), "wb");
    if (!g_file) { VirtualFree(g_ev, 0, MEM_RELEASE); g_ev = nullptr; return; }
    g_started = true;
    g_on.store(true);
    g_writer = std::thread(writer_main);
    if (const int64_t sm = config_get_int("prof", "stall_ms", 0); sm > 0) {
        g_stall_ms.store(static_cast<uint32_t>(sm));
        g_sampler = std::thread(sampler_main, path + ".stacks");
    }
    NXLOG_I("prof", "frame trace on: %s (%zu events)", path.c_str(), g_cap);
}

void stop() {
    if (!g_started) return;
    g_on.store(false);
    g_stop.store(true);
    if (g_writer.joinable()) g_writer.join();
    if (g_sampler.joinable()) g_sampler.join();
    std::lock_guard<std::mutex> lk(g_mu);
    flush_locked();
    if (g_file) { std::fclose(g_file); g_file = nullptr; }
}

}  // namespace nxrt::ft
