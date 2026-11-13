// SPDX-License-Identifier: GPL-3.0-or-later
// Gauges and the soak sampler (declared in nxrt/gauge.h).
#include "nxrt/gauge.h"

#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <thread>
#include <cstring>
#include <intrin.h>

#include "nxrt/config.h"
#include "nxrt/diag.h"
#include "nxrt/log.h"

namespace nxrt {
namespace {

struct Registry {
    std::mutex m;
    std::map<std::string, std::function<int64_t()>> g;
};
Registry& reg() {
    static Registry* r = new Registry;     // never destroyed: counters in other statics may unregister at exit
    return *r;
}

// ---- frame times of the window -------------------------------------------------------------------------------------------
constexpr size_t kRing = 1 << 16;           // frames kept between two samples (enough for 6,500 fps at a 10 s window)
std::atomic<uint64_t> g_ring_n{0};
std::atomic<uint64_t> g_ring[kRing];        // QueryPerformanceCounter ticks of each presented frame

uint64_t qpc() {
    LARGE_INTEGER v;
    QueryPerformanceCounter(&v);
    return (uint64_t)v.QuadPart;
}
double qpc_freq() {
    static double f = [] { LARGE_INTEGER v; QueryPerformanceFrequency(&v); return (double)v.QuadPart; }();
    return f;
}

// ---- hitch log -------------------------------------------------------------------------------------------------------------
constexpr size_t kMaxSources = 32;
struct Source {
    const char* label = nullptr;
    uint64_t (*counter)() = nullptr;     // cumulative; shown as growth during the frame
    int64_t (*level)() = nullptr;        // instantaneous; shown when non-zero
};
Source g_src[kMaxSources];
std::atomic<size_t> g_nsrc{0};
uint64_t g_prev_val[kMaxSources];                 // the sources at the previous frame; the presenting thread only
std::atomic<bool> g_hitch_on{false};
std::atomic<uint64_t> g_hitch_ticks{0};           // a frame longer than this is a hitch
std::atomic<uint64_t> g_t0_ticks{0};

// phase timers (HitchPhase): QPC ticks summed over all threads
std::atomic<uint64_t> g_phase[kHpCount];
uint64_t g_prev_phase[kHpCount];                  // at the previous frame; the presenting thread only
std::atomic<uint64_t> g_game_present_last{0};     // QPC of the game's latest present call
std::atomic<uint64_t> g_game_gap_max{0};          // longest gap between two game present calls since the last presented frame
thread_local bool t_render_thread = false;
thread_local bool t_game_present_thread = false;  // this thread has called hitch_note_game_present

struct HitchEvent {
    double ms = 0;
    uint64_t frame = 0;
    double at_s = 0;                              // seconds since the monitor started
    int64_t val[kMaxSources] = {};
    uint64_t phase[kHpCount] = {};                // ticks of each phase inside the frame
    uint64_t game_gap = 0;                        // ticks: longest gap between the game's present calls
};
std::mutex g_ev_mu;                               // taken by the presenting thread only for a frame over the limit
std::vector<HitchEvent> g_pending;

// ---- frame budget ([diag] budget, default true) ------------------------------------------------------------------------------
// Per presented frame (render thread) and per game present (the game's presenting thread), what each thread spent, so the 10 s summary can
// say whether the render thread and the game thread fit in the frame, with percentiles (the hitch line only sees frames over 50 ms).
// Render thread work = replay - GPU slot waits - lock waits + present - limiter - acquire; its CPU time comes from QueryThreadCycleTime (TSC
// cycles on the core). Game work = the gap between two presents - the present's back pressure (waiting for the render thread).
struct RFrame { float dt = 0, work = 0, gpu = 0, idle = 0, lim = 0, kcyc = 0; };
struct GFrame { float gap = 0, bp = 0, sync = 0, wait = 0, kcyc = 0; };
constexpr size_t kBRing = 8192;
std::atomic<bool> g_budget_on{false};
RFrame g_rf[kBRing];
std::atomic<uint64_t> g_rf_n{0};
GFrame g_gf[kBRing];
std::atomic<uint64_t> g_gf_n{0};

uint64_t thread_cycles() {
    ULONG64 c = 0;
    QueryThreadCycleTime(GetCurrentThread(), &c);
    return c;
}

// Called for every presented frame at tick t (previous frame's tick prev_t, 0 for the first), frame number n.
void hitch_on_frame(uint64_t t, uint64_t prev_t, uint64_t n) {
    if (!g_hitch_on.load(std::memory_order_relaxed)) return;
    const size_t ns = g_nsrc.load(std::memory_order_acquire);
    uint64_t cur[kMaxSources];
    for (size_t i = 0; i < ns; i++) cur[i] = g_src[i].counter ? g_src[i].counter() : (uint64_t)g_src[i].level();
    uint64_t curp[kHpCount];
    for (int i = 0; i < kHpCount; i++) curp[i] = g_phase[i].load(std::memory_order_relaxed);
    if (g_budget_on.load(std::memory_order_relaxed)) {
        static thread_local uint64_t prev_cyc = 0;
        const uint64_t cyc = thread_cycles();
        if (prev_t && t > prev_t && prev_cyc) {
            const double tm = 1000.0 / qpc_freq();
            auto d = [&](int p) { return (double)(curp[p] - g_prev_phase[p]) * tm; };
            RFrame f;
            f.dt = (float)((double)(t - prev_t) * tm);
            const double work = d(kHpRenderReplay) - d(kHpRenderGpuWait) - d(kHpRenderLockWait) + d(kHpRenderPresent) - d(kHpRenderLimiter) - d(kHpRenderAcquire);
            f.work = (float)std::max(0.0, work);
            f.gpu = (float)(d(kHpRenderGpuWait) + d(kHpRenderAcquire));
            f.idle = (float)d(kHpRenderIdle);
            f.lim = (float)d(kHpRenderLimiter);
            f.kcyc = (float)((double)(cyc - prev_cyc) / 1000.0);
            const uint64_t k = g_rf_n.load(std::memory_order_relaxed);
            g_rf[k % kBRing] = f;
            g_rf_n.store(k + 1, std::memory_order_release);
        }
        prev_cyc = cyc;
    }
    const uint64_t gap = g_game_gap_max.exchange(0, std::memory_order_relaxed);
    const uint64_t thr = g_hitch_ticks.load(std::memory_order_relaxed);
    if (thr && prev_t && t > prev_t && t - prev_t > thr) {
        HitchEvent ev;
        ev.ms = (double)(t - prev_t) * 1000.0 / qpc_freq();
        ev.frame = n;
        ev.at_s = (double)(t - g_t0_ticks.load(std::memory_order_relaxed)) / qpc_freq();
        for (size_t i = 0; i < ns; i++) ev.val[i] = g_src[i].counter ? (int64_t)(cur[i] - g_prev_val[i]) : (int64_t)cur[i];
        for (int i = 0; i < kHpCount; i++) ev.phase[i] = curp[i] - g_prev_phase[i];
        ev.game_gap = gap;
        std::lock_guard<std::mutex> lk(g_ev_mu);
        if (g_pending.size() < 256) g_pending.push_back(ev);
    }
    for (size_t i = 0; i < ns; i++) g_prev_val[i] = cur[i];
    for (int i = 0; i < kHpCount; i++) g_prev_phase[i] = curp[i];
}

struct FrameStats {
    size_t frames = 0;
    double span_s = 0, mean_fps = 0, low1_fps = 0, worst_ms = 0, median_ms = 0;
    size_t over2x = 0;
};
// dt: frame intervals in ms (sorted here).
FrameStats stats_of(std::vector<double>& dt) {
    FrameStats st;
    st.frames = dt.size();
    if (dt.empty()) return st;
    double sum = 0;
    for (double d : dt) sum += d;
    std::sort(dt.begin(), dt.end());
    st.span_s = sum / 1000.0;
    st.mean_fps = sum > 0 ? (double)dt.size() * 1000.0 / sum : 0;
    st.median_ms = dt[dt.size() / 2];
    const double p99 = dt[std::min(dt.size() - 1, (size_t)((double)dt.size() * 0.99))];
    st.low1_fps = p99 > 0 ? 1000.0 / p99 : 0;
    st.worst_ms = dt.back();
    for (double d : dt) if (d > 2.0 * st.median_ms) st.over2x++;
    return st;
}

std::string format_event(const HitchEvent& ev, size_t others) {
    char b[400];
    snprintf(b, sizeof b, "frame %llu took %.1f ms (t=%.1f s)", (unsigned long long)ev.frame, ev.ms, ev.at_s);
    std::string out = b;
    const size_t ns = g_nsrc.load(std::memory_order_acquire);
    bool any = false;
    for (size_t i = 0; i < ns; i++) {
        if (ev.val[i] == 0) continue;
        out += any ? ", " : "; during it: ";
        out += g_src[i].label;
        snprintf(b, sizeof b, " %s%lld", g_src[i].counter ? "+" : "=", (long long)ev.val[i]);
        out += b;
        any = true;
    }
    if (!any) out += "; nothing tracked (compiles, uploads, write faults) changed in it";
    // where the frame's time went (ms inside the frame). Render thread: idle + replay + present + other = the frame.
    auto ms = [&](int p) { return (double)ev.phase[p] * 1000.0 / qpc_freq(); };
    auto part = [&](std::string& s, const char* name, int p) {
        const double v = ms(p);
        if (v < 0.05) return;
        snprintf(b, sizeof b, "%s%s %.1f", s.empty() ? "" : ", ", name, v);
        s += b;
    };
    std::string rep, pre, game;
    part(rep, "gpu slot wait", kHpRenderGpuWait);
    part(rep, "submit", kHpRenderSubmit);
    part(rep, "uploads", kHpRenderUpload);
    part(rep, "pipeline creation", kHpRenderPipeline);
    part(rep, "lock waits", kHpRenderLockWait);
    part(pre, "limiter", kHpRenderLimiter);
    part(pre, "acquire", kHpRenderAcquire);
    part(pre, "end/queue present", kHpRenderPresentEnd);
    const double other = ev.ms - ms(kHpRenderIdle) - ms(kHpRenderReplay) - ms(kHpRenderPresent);
    snprintf(b, sizeof b, "; render thread ms: idle %.1f, replay %.1f%s%s%s, present %.1f%s%s%s, other %.1f", ms(kHpRenderIdle), ms(kHpRenderReplay),
             rep.empty() ? "" : " (", rep.c_str(), rep.empty() ? "" : ")", ms(kHpRenderPresent), pre.empty() ? "" : " (", pre.c_str(), pre.empty() ? "" : ")",
             other > 0 ? other : 0.0);
    out += b;
    snprintf(b, sizeof b, "; game ms: longest gap between presents %.1f", (double)ev.game_gap * 1000.0 / qpc_freq());
    game = b;
    part(game, "present back pressure", kHpGameBackpressure);
    part(game, "GPU sync waits (all threads)", kHpGameSyncWait);
    {   // the presenting game thread's system calls (IPC = services and file reads, waits); the rest of its gap is guest code on the CPU
        std::string sv;
        part(sv, "IPC", kHpGameIpc);
        part(sv, "waits", kHpGameWait);
        const double v = ms(kHpGameSvc);
        if (v >= 0.05) {
            snprintf(b, sizeof b, ", presenting thread in system calls %.1f%s%s%s", v, sv.empty() ? "" : " (", sv.c_str(), sv.empty() ? "" : ")");
            game += b;
        }
    }
    out += game;
    std::string any_thr;
    part(any_thr, "write-fault handler", kHpFaults);
    part(any_thr, "write-watch arming", kHpWwArm);
    part(any_thr, "memory pool imports", kHpPoolImport);
    if (!any_thr.empty()) out += "; any thread ms: " + any_thr;
    if (others) {
        snprintf(b, sizeof b, " (+%zu more slow frames in the last second)", others);
        out += b;
    }
    return out;
}

// ---- frame budget report -----------------------------------------------------------------------------------------------------
float pct(std::vector<float>& v, double p) {     // v sorted
    if (v.empty()) return 0;
    return v[std::min(v.size() - 1, (size_t)((double)v.size() * p))];
}
std::string pcts(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    char b[96];
    snprintf(b, sizeof b, "p50 %.1f p95 %.1f p99 %.1f max %.1f", pct(v, 0.5), pct(v, 0.95), pct(v, 0.99), v.empty() ? 0.0f : v.back());
    return b;
}
float median_of(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return pct(v, 0.5);
}

// TSC cycles per ms, measured between two reports (QueryThreadCycleTime counts TSC cycles).
double g_bud_tsc0 = 0, g_bud_qpc0 = 0;
double tsc_per_ms_now() {
    const double tsc = (double)__rdtsc(), q = (double)qpc();
    double r = 0;
    if (g_bud_qpc0 > 0 && q > g_bud_qpc0) r = (tsc - g_bud_tsc0) / ((q - g_bud_qpc0) * 1000.0 / qpc_freq());
    g_bud_tsc0 = tsc, g_bud_qpc0 = q;
    return r;
}

uint64_t g_bud_rf_last = 0, g_bud_gf_last = 0;

void budget_report() {
    const double tpm = tsc_per_ms_now();
    const uint64_t rn = g_rf_n.load(std::memory_order_acquire), gn = g_gf_n.load(std::memory_order_acquire);
    const uint64_t r0 = std::max(g_bud_rf_last, rn > kBRing - 64 ? rn - (kBRing - 64) : 0), g0 = std::max(g_bud_gf_last, gn > kBRing - 64 ? gn - (kBRing - 64) : 0);
    g_bud_rf_last = rn, g_bud_gf_last = gn;
    if (!(tpm > 0) || rn <= r0) return;
    std::vector<float> dt, work, cpu, gpu, idle, lim;
    for (uint64_t i = r0; i < rn; i++) {
        const RFrame& f = g_rf[i % kBRing];
        dt.push_back(f.dt), work.push_back(f.work), gpu.push_back(f.gpu), idle.push_back(f.idle), lim.push_back(f.lim);
        cpu.push_back((float)((double)f.kcyc * 1000.0 / tpm));
    }
    std::vector<float> gap, gwork, gcpu, gwait, gsync;
    for (uint64_t i = g0; i < gn; i++) {
        const GFrame& f = g_gf[i % kBRing];
        gap.push_back(f.gap);
        gwork.push_back(std::max(0.0f, f.gap - f.bp));
        gcpu.push_back((float)((double)f.kcyc * 1000.0 / tpm));
        gwait.push_back(f.wait);
        gsync.push_back(f.sync);
    }
    // slow frames: over 1.2x the median and 2 ms over it; the cause is the part that grew most against its own median
    const float mdt = median_of(dt), mwork = median_of(work), mgpu = median_of(gpu), midle = median_of(idle);
    size_t slow = 0, by_game = 0, by_render = 0, by_gpu = 0;
    for (size_t i = 0; i < dt.size(); i++) {
        if (!(dt[i] > 1.2f * mdt && dt[i] - mdt > 2.0f)) continue;
        ++slow;
        const float a = idle[i] - midle, b = work[i] - mwork, c = gpu[i] - mgpu;
        if (a >= b && a >= c) ++by_game;
        else if (b >= c) ++by_render;
        else ++by_gpu;
    }
    NXLOG_I("budget", "%zu frames, median %.2f ms | render thread ms: work %s; cpu %s; GPU/swapchain waits %s; limiter p50 %.1f; idle p50 %.1f",
            dt.size(), mdt, pcts(work).c_str(), pcts(cpu).c_str(), pcts(gpu).c_str(), median_of(lim), midle);
    if (!gap.empty())
        NXLOG_I("budget", "game thread ms (%zu presents): frame %s; work (minus present back pressure) %s; cpu %s; waits p50 %.1f p99 %.1f; GPU sync waits p99 %.1f | "
                "slow frames (over 1.2x the median): %zu = waiting for the game %zu, render thread work %zu, GPU/swapchain %zu",
                gap.size(), pcts(gap).c_str(), pcts(gwork).c_str(), pcts(gcpu).c_str(), median_of(gwait),
                [&] { std::vector<float> v = gwait; std::sort(v.begin(), v.end()); return pct(v, 0.99); }(),
                [&] { std::vector<float> v = gsync; std::sort(v.begin(), v.end()); return pct(v, 0.99); }(), slow, by_game, by_render, by_gpu);
}

// CPU time of every thread of the process over the window, grouped by name: who uses the cores (oversubscription, driver threads).
struct ThrSlot { HANDLE h = nullptr; uint64_t last = 0; std::string name; bool seen = false; };
std::map<DWORD, ThrSlot> g_thr;
uint64_t g_sys_idle = 0, g_sys_total = 0, g_thr_qpc = 0, g_thr_tsc = 0;

std::string thread_name_of(HANDLE h, DWORD tid, const std::map<uint32_t, std::string>& reg_names) {
    auto it = reg_names.find(tid);
    if (it != reg_names.end() && !it->second.empty()) return it->second;
    using GetDescFn = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    static const GetDescFn get_desc = reinterpret_cast<GetDescFn>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription")));
    if (get_desc) {
        PWSTR w = nullptr;
        if (SUCCEEDED(get_desc(h, &w)) && w) {
            std::string s;
            for (PWSTR p = w; *p && s.size() < 48; ++p) s += (*p < 128) ? (char)*p : '?';
            LocalFree(w);
            if (!s.empty()) return s;
        }
    }
    // the module its start address is in: driver and system threads
    using NtQit = LONG(WINAPI*)(HANDLE, int, PVOID, ULONG, PULONG);
    static const NtQit qit = reinterpret_cast<NtQit>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread")));
    void* start = nullptr;
    if (qit && qit(h, 9 /* ThreadQuerySetWin32StartAddress */, &start, sizeof start, nullptr) == 0 && start) {
        HMODULE m = nullptr;
        char path[MAX_PATH];
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)start, &m) &&
            GetModuleFileNameA(m, path, sizeof path)) {
            const char* base = strrchr(path, '\\');
            return std::string("[") + (base ? base + 1 : path) + "]";
        }
    }
    return "[?]";
}

void thread_cpu_report() {
    const uint64_t q = qpc(), tsc = __rdtsc();
    const double win_ms = g_thr_qpc ? (double)(q - g_thr_qpc) * 1000.0 / qpc_freq() : 0;
    const double tpm = win_ms > 0 ? (double)(tsc - g_thr_tsc) / win_ms : 0;
    g_thr_qpc = q, g_thr_tsc = tsc;
    FILETIME fi, fk, fu;
    uint64_t idle = 0, total = 0;
    if (GetSystemTimes(&fi, &fk, &fu)) {
        auto u = [](const FILETIME& f) { return (uint64_t)f.dwHighDateTime << 32 | f.dwLowDateTime; };
        idle = u(fi), total = u(fk) + u(fu);     // kernel time includes idle
    }
    const double sys_busy = (total > g_sys_total && g_sys_total) ? 100.0 * (1.0 - (double)(idle - g_sys_idle) / (double)(total - g_sys_total)) : -1;
    g_sys_idle = idle, g_sys_total = total;
    std::map<uint32_t, std::string> reg_names;
    for (const ThreadRecord& r : diag_threads()) reg_names[r.host_tid] = r.name;
    for (auto& kv : g_thr) kv.second.seen = false;
    std::map<std::string, std::pair<double, int>> groups;     // name -> (cpu ms, threads)
    double total_ms = 0;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap != INVALID_HANDLE_VALUE) {
        THREADENTRY32 te;
        te.dwSize = sizeof te;
        const DWORD pid = GetCurrentProcessId();
        if (Thread32First(snap, &te)) do {
            if (te.th32OwnerProcessID != pid) continue;
            ThrSlot& s = g_thr[te.th32ThreadID];
            const bool fresh = !s.h;
            if (fresh) {
                s.h = OpenThread(THREAD_QUERY_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION, FALSE, te.th32ThreadID);
                if (!s.h) s.h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, te.th32ThreadID);
                if (!s.h) { g_thr.erase(te.th32ThreadID); continue; }
            }
            s.seen = true;
            s.name = thread_name_of(s.h, te.th32ThreadID, reg_names);
            ULONG64 c = 0;
            QueryThreadCycleTime(s.h, &c);
            if (!fresh && tpm > 0 && c >= s.last) {
                const double ms = (double)(c - s.last) / tpm;
                auto& gr = groups[s.name];
                gr.first += ms, gr.second++;
                total_ms += ms;
            }
            s.last = c;
        } while (Thread32Next(snap, &te));
        CloseHandle(snap);
    }
    for (auto it = g_thr.begin(); it != g_thr.end();) {
        if (!it->second.seen) { CloseHandle(it->second.h); it = g_thr.erase(it); }
        else ++it;
    }
    if (!(win_ms > 0) || groups.empty()) return;
    std::vector<std::pair<double, std::string>> v;
    for (auto& kv : groups) {
        char b[96];
        if (kv.second.second > 1) snprintf(b, sizeof b, "%s x%d", kv.first.c_str(), kv.second.second);
        else snprintf(b, sizeof b, "%s", kv.first.c_str());
        v.emplace_back(kv.second.first, b);
    }
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::string line;
    char b[128];
    for (size_t i = 0; i < v.size() && i < 16; i++) {
        if (v[i].first / win_ms < 0.005) break;
        snprintf(b, sizeof b, "%s%s %.0f", i ? ", " : "", v[i].second.c_str(), 100.0 * v[i].first / win_ms);
        line += b;
    }
    NXLOG_I("budget", "cpu %% of one core over %.1f s: process %.0f, machine busy %.0f%% of %u processors | %s", win_ms / 1000.0, 100.0 * total_ms / win_ms, sys_busy,
            (unsigned)std::thread::hardware_concurrency(), line.c_str());
}

struct Monitor {
    std::mutex ctl;
    std::thread th;
    std::mutex m;
    std::condition_variable cv;
    bool stop = false;
};
Monitor& mon() {
    static Monitor* s = new Monitor;
    return *s;
}

void monitor_run(double summary_s) {
    Monitor& s = mon();
    log_set_thread_name("hitch");
    using Clock = std::chrono::steady_clock;
    const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(summary_s > 0 ? summary_s : 1e6));
    auto next_summary = Clock::now() + period;
    auto last_line = Clock::now() - std::chrono::seconds(10);
    uint64_t last_n = g_ring_n.load(std::memory_order_acquire);
    bool seen_frames = last_n > 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(s.m);
            if (s.cv.wait_for(lk, std::chrono::milliseconds(100), [&] { return s.stop; })) return;
        }
        const auto now = Clock::now();
        // at most one hitch line per second: the worst frame waiting, with the count of the others
        if (now - last_line >= std::chrono::seconds(1)) {
            HitchEvent worst;
            size_t cnt = 0;
            {
                std::lock_guard<std::mutex> lk(g_ev_mu);
                for (const auto& e : g_pending) if (cnt++ == 0 || e.ms > worst.ms) worst = e;
                g_pending.clear();
            }
            if (cnt) {
                NXLOG_W("hitch", "%s", format_event(worst, cnt - 1).c_str());
                last_line = now;
            }
        }
        if (summary_s > 0 && now >= next_summary) {
            next_summary = now + period;
            const uint64_t n = g_ring_n.load(std::memory_order_acquire);
            const uint64_t first = (n > last_n + kRing - 1) ? n - (kRing - 1) : last_n;
            std::vector<double> dt;
            uint64_t prev = first > 0 ? g_ring[(first - 1) % kRing].load(std::memory_order_relaxed) : 0;
            for (uint64_t i = first; i < n; i++) {
                const uint64_t t = g_ring[i % kRing].load(std::memory_order_relaxed);
                if (prev) dt.push_back((double)(t - prev) * 1000.0 / qpc_freq());
                prev = t;
            }
            const uint64_t window_frames = n - last_n;
            last_n = n;
            if (n > 0) seen_frames = true;
            if (!dt.empty()) {
                const FrameStats st = stats_of(dt);
                NXLOG_I("hitch", "%llu frames in %.1f s: mean %.1f fps, 1%% low %.1f fps, worst %.1f ms, %zu over 2x the median (%.1f ms)",
                        (unsigned long long)window_frames, st.span_s, st.mean_fps, st.low1_fps, st.worst_ms, st.over2x, st.median_ms);
            } else if (seen_frames) {
                NXLOG_W("hitch", "no frame presented in the last %.1f s", summary_s);
            }
            if (g_budget_on.load(std::memory_order_relaxed)) {
                if (!dt.empty()) budget_report();
                thread_cpu_report();
            }
        }
    }
}

std::string frame_times_report() { return "\n" + frame_times_summary(30.0); }

struct Sampler {
    std::mutex ctl;
    std::thread th;
    std::mutex m;
    std::condition_variable cv;
    bool stop = false;
    FILE* f = nullptr;
};
Sampler& smp() {
    static Sampler* s = new Sampler;
    return *s;
}

uint32_t thread_count() {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return 0;
    THREADENTRY32 te;
    te.dwSize = sizeof te;
    uint32_t n = 0;
    const DWORD pid = GetCurrentProcessId();
    if (Thread32First(snap, &te)) do { if (te.th32OwnerProcessID == pid) n++; } while (Thread32Next(snap, &te));
    CloseHandle(snap);
    return n;
}

void run(double interval_s, double t0) {
    Sampler& s = smp();
    uint64_t last_n = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(s.m);
            if (s.cv.wait_for(lk, std::chrono::duration<double>(interval_s), [&] { return s.stop; })) return;
        }
        const double now = (double)qpc() / qpc_freq();
        const uint64_t n = g_ring_n.load(std::memory_order_acquire);
        std::vector<double> dt;
        const uint64_t first = (n > last_n + kRing) ? n - kRing : last_n;
        uint64_t prev = first > 0 ? g_ring[(first - 1) % kRing].load(std::memory_order_relaxed) : 0;
        for (uint64_t i = first; i < n; i++) {
            const uint64_t t = g_ring[i % kRing].load(std::memory_order_relaxed);
            if (prev) dt.push_back((double)(t - prev) * 1000.0 / qpc_freq());
            prev = t;
        }
        const uint64_t window_frames = n - last_n;
        last_n = n;
        double p50 = 0, p99 = 0, mx = 0;
        if (!dt.empty()) {
            std::sort(dt.begin(), dt.end());
            p50 = dt[dt.size() / 2];
            p99 = dt[std::min(dt.size() - 1, (size_t)(dt.size() * 0.99))];
            mx = dt.back();
        }
        PROCESS_MEMORY_COUNTERS_EX pm{};
        pm.cb = sizeof pm;
        GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pm, sizeof pm);
        DWORD handles = 0;
        GetProcessHandleCount(GetCurrentProcess(), &handles);
        FILETIME c, e, k, u;
        GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
        const double cpu_ms = (((uint64_t)k.dwHighDateTime << 32 | k.dwLowDateTime) + ((uint64_t)u.dwHighDateTime << 32 | u.dwLowDateTime)) / 10000.0;
        char buf[640];
        snprintf(buf, sizeof buf,
                 "{\"t\":%.1f,\"frame\":%llu,\"win_frames\":%llu,\"ft_p50\":%.3f,\"ft_p99\":%.3f,\"ft_max\":%.2f,\"ws\":%llu,\"private\":%llu,\"peak_ws\":%llu,"
                 "\"pagefile\":%llu,\"handles\":%lu,\"threads\":%u,\"gdi\":%lu,\"user\":%lu,\"cpu_ms\":%.0f",
                 now - t0, (unsigned long long)diag_frame_count(), (unsigned long long)window_frames, p50, p99, mx, (unsigned long long)pm.WorkingSetSize,
                 (unsigned long long)pm.PrivateUsage, (unsigned long long)pm.PeakWorkingSetSize, (unsigned long long)pm.PagefileUsage, (unsigned long)handles,
                 thread_count(), (unsigned long)GetGuiResources(GetCurrentProcess(), GR_GDIOBJECTS), (unsigned long)GetGuiResources(GetCurrentProcess(), GR_USEROBJECTS),
                 cpu_ms);
        std::string line = buf;
        for (auto& kv : gauge_read_all()) {
            snprintf(buf, sizeof buf, ",\"%s\":%lld", kv.first.c_str(), (long long)kv.second);
            line += buf;
        }
        line += "}\n";
        fwrite(line.data(), 1, line.size(), s.f);
        fflush(s.f);
    }
}

}  // namespace

void gauge_register(const char* name, std::function<int64_t()> fn) {
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.m);
    r.g[name] = std::move(fn);
}

void gauge_unregister(const char* name) {
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.m);
    r.g.erase(name);
}

std::vector<std::pair<std::string, int64_t>> gauge_read_all() {
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.m);
    std::vector<std::pair<std::string, int64_t>> out;
    out.reserve(r.g.size());
    for (auto& kv : r.g) out.emplace_back(kv.first, kv.second());
    return out;
}

GaugeCounter::GaugeCounter(const char* name) : name_(name) {
    gauge_register(name, [this] { return get(); });
}
GaugeCounter::~GaugeCounter() { gauge_unregister(name_.c_str()); }

void soak_note_frame() {
    const uint64_t n = g_ring_n.load(std::memory_order_relaxed);
    const uint64_t t = qpc();
    const uint64_t prev_t = n > 0 ? g_ring[(n - 1) % kRing].load(std::memory_order_relaxed) : 0;
    g_ring[n % kRing].store(t, std::memory_order_relaxed);
    g_ring_n.store(n + 1, std::memory_order_release);
    hitch_on_frame(t, prev_t, n + 1);
}

size_t frame_times_recent(float* ms, size_t n) {
    const uint64_t total = g_ring_n.load(std::memory_order_acquire);
    if (!ms || n == 0 || total < 2) return 0;
    const size_t k = static_cast<size_t>(std::min<uint64_t>({static_cast<uint64_t>(n), total - 1, kRing - 2}));
    const double to_ms = 1000.0 / qpc_freq();
    for (size_t i = 0; i < k; i++) {   // frame (total - k + i) minus the one before it
        const uint64_t idx = total - k + i;
        const uint64_t a = g_ring[(idx - 1) % kRing].load(std::memory_order_relaxed), b = g_ring[idx % kRing].load(std::memory_order_relaxed);
        ms[i] = b > a ? static_cast<float>(static_cast<double>(b - a) * to_ms) : 0.f;
    }
    return k;
}

double hitch_phase_total_ms(int phase) {
    if (phase < 0 || phase >= kHpCount) return 0;
    return static_cast<double>(g_phase[phase].load(std::memory_order_relaxed)) * 1000.0 / qpc_freq();
}

uint64_t hitch_ticks() { return qpc(); }
void hitch_phase_add(int phase, uint64_t ticks) {
    if (phase >= 0 && phase < kHpCount) g_phase[phase].fetch_add(ticks, std::memory_order_relaxed);
}
void hitch_set_render_thread() { t_render_thread = true; }
bool hitch_on_render_thread() { return t_render_thread; }
bool hitch_on_game_present_thread() { return t_game_present_thread; }
void hitch_note_game_present() {
    t_game_present_thread = true;
    const uint64_t t = qpc();
    const uint64_t prev = g_game_present_last.exchange(t, std::memory_order_relaxed);
    if (g_budget_on.load(std::memory_order_relaxed)) {
        static thread_local uint64_t p_cyc = 0, p_bp = 0, p_sync = 0, p_wait = 0;
        const uint64_t cyc = thread_cycles();
        const uint64_t bp = g_phase[kHpGameBackpressure].load(std::memory_order_relaxed), sy = g_phase[kHpGameSyncWait].load(std::memory_order_relaxed),
                       wt = g_phase[kHpGameWait].load(std::memory_order_relaxed);
        if (prev && t > prev && p_cyc) {
            const double tm = 1000.0 / qpc_freq();
            GFrame f;
            f.gap = (float)((double)(t - prev) * tm);
            f.bp = (float)((double)(bp - p_bp) * tm);
            f.sync = (float)((double)(sy - p_sync) * tm);
            f.wait = (float)((double)(wt - p_wait) * tm);
            f.kcyc = (float)((double)(cyc - p_cyc) / 1000.0);
            const uint64_t k = g_gf_n.load(std::memory_order_relaxed);
            g_gf[k % kBRing] = f;
            g_gf_n.store(k + 1, std::memory_order_release);
        }
        p_cyc = cyc, p_bp = bp, p_sync = sy, p_wait = wt;
    }
    if (!prev || t <= prev) return;
    const uint64_t d = t - prev;
    uint64_t m = g_game_gap_max.load(std::memory_order_relaxed);
    while (d > m && !g_game_gap_max.compare_exchange_weak(m, d, std::memory_order_relaxed)) {}
}

void hitch_add_counter(const char* label, uint64_t (*fn)()) {
    const size_t i = g_nsrc.load();
    if (i >= kMaxSources || !fn) return;
    g_src[i] = {label, fn, nullptr};
    g_nsrc.store(i + 1, std::memory_order_release);
}

void hitch_add_level(const char* label, int64_t (*fn)()) {
    const size_t i = g_nsrc.load();
    if (i >= kMaxSources || !fn) return;
    g_src[i] = {label, nullptr, fn};
    g_nsrc.store(i + 1, std::memory_order_release);
}

void hitch_start() {
    if (!config_get_bool("diag", "hitch_log", true)) return;
    Monitor& s = mon();
    std::lock_guard<std::mutex> g(s.ctl);
    if (s.th.joinable()) return;
    const double summary_s = config_get_double("diag", "hitch_summary_s", 10.0);
    const double hitch_ms = config_get_double("diag", "hitch_ms", 50.0);
    g_t0_ticks.store(qpc());
    g_hitch_ticks.store(hitch_ms > 0 ? (uint64_t)(hitch_ms / 1000.0 * qpc_freq()) : 0);
    {
        std::lock_guard<std::mutex> lk(g_ev_mu);
        g_pending.clear();
    }
    {
        const size_t ns = g_nsrc.load(std::memory_order_acquire);
        for (size_t i = 0; i < ns; i++) g_prev_val[i] = g_src[i].counter ? g_src[i].counter() : (uint64_t)g_src[i].level();
        for (int i = 0; i < kHpCount; i++) g_prev_phase[i] = g_phase[i].load(std::memory_order_relaxed);
    }
    g_budget_on.store(config_get_bool("diag", "budget", true));
    g_hitch_on.store(true);
    s.stop = false;
    s.th = std::thread(monitor_run, summary_s);
    static const bool report_registered = (diag_add_report(frame_times_report), true);   // the freeze dump shows the last 30 s of frame times
    (void)report_registered;
    NXLOG_I("diag", "hitch log: frame summary every %.0f s, frames over %.0f ms logged (at most one line per second)", summary_s, hitch_ms);
}

void hitch_stop() {
    Monitor& s = mon();
    std::lock_guard<std::mutex> g(s.ctl);
    g_hitch_on.store(false);
    if (!s.th.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(s.m);
        s.stop = true;
    }
    s.cv.notify_all();
    s.th.join();
}

std::string frame_times_summary(double window_s) {
    if (!(window_s > 0)) window_s = 30.0;
    const uint64_t now = qpc();
    const uint64_t n = g_ring_n.load(std::memory_order_acquire);
    if (n == 0) return "frame times: no frame was presented yet\n";
    std::vector<uint64_t> ts;       // ticks of the frames inside the window, newest first
    const uint64_t win_ticks = (uint64_t)(window_s * qpc_freq());
    for (uint64_t k = 0; k < n && k < kRing - 1; k++) {
        const uint64_t t = g_ring[(n - 1 - k) % kRing].load(std::memory_order_relaxed);
        if (t > now || now - t > win_ticks) break;
        ts.push_back(t);
    }
    char b[256];
    const double last_age = (double)(now - g_ring[(n - 1) % kRing].load(std::memory_order_relaxed)) / qpc_freq();
    if (ts.size() < 2) {
        snprintf(b, sizeof b, "frame times, last %.0f s: %zu frame(s); the last frame was presented %.1f s ago (%llu frames in all)\n", window_s, ts.size(), last_age, (unsigned long long)n);
        return b;
    }
    std::reverse(ts.begin(), ts.end());     // oldest first
    const size_t nb = (size_t)std::ceil(window_s);
    std::vector<unsigned> fps(nb, 0);
    std::vector<double> bworst(nb, 0.0);
    std::vector<double> dt;
    struct W { double ms, ago; };
    std::vector<W> frames;
    for (size_t i = 1; i < ts.size(); i++) {
        const double d = (double)(ts[i] - ts[i - 1]) * 1000.0 / qpc_freq();
        const double ago = (double)(now - ts[i]) / qpc_freq();
        dt.push_back(d);
        frames.push_back({d, ago});
        const size_t bi = std::min(nb - 1, (size_t)ago);
        fps[bi]++;
        bworst[bi] = std::max(bworst[bi], d);
    }
    const FrameStats st = stats_of(dt);
    std::string out;
    snprintf(b, sizeof b, "frame times, last %.0f s (%llu frames in all; the last one was presented %.1f s ago):\n", window_s, (unsigned long long)n, last_age);
    out += b;
    snprintf(b, sizeof b, "  %zu frames: mean %.1f fps, 1%% low %.1f fps, worst %.1f ms, %zu over 2x the median (%.1f ms)\n", st.frames, st.mean_fps, st.low1_fps, st.worst_ms, st.over2x, st.median_ms);
    out += b;
    out += "  fps per second (oldest first):";
    for (size_t i = nb; i-- > 0;) { snprintf(b, sizeof b, " %u", fps[i]); out += b; }
    out += "\n  worst frame ms per second (oldest first):";
    for (size_t i = nb; i-- > 0;) { snprintf(b, sizeof b, " %.0f", bworst[i]); out += b; }
    out += "\n";
    std::sort(frames.begin(), frames.end(), [](const W& a, const W& c) { return a.ms > c.ms; });
    out += "  slowest frames (ms@seconds before the end):";
    for (size_t i = 0; i < frames.size() && i < 6; i++) { snprintf(b, sizeof b, " %.1f@-%.1f", frames[i].ms, frames[i].ago); out += b; }
    out += "\n";
    return out;
}

void soak_start() {
    const std::string path = config_get_str("diag", "soak_csv", "");
    if (path.empty()) return;
    Sampler& s = smp();
    std::lock_guard<std::mutex> g(s.ctl);
    if (s.th.joinable()) return;
    s.f = fopen(path.c_str(), "ab");
    if (!s.f) {
        NXLOG_W("diag", "soak: cannot open %s", path.c_str());
        return;
    }
    const double iv = std::max(0.5, config_get_double("diag", "soak_interval_s", 10.0));
    s.stop = false;
    const double t0 = (double)qpc() / qpc_freq();
    s.th = std::thread(run, iv, t0);
    NXLOG_I("diag", "soak sampler: %s every %.1f s", path.c_str(), iv);
}

void soak_stop() {
    Sampler& s = smp();
    std::lock_guard<std::mutex> g(s.ctl);
    if (!s.th.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(s.m);
        s.stop = true;
    }
    s.cv.notify_all();
    s.th.join();
    if (s.f) { fclose(s.f); s.f = nullptr; }
}

}  // namespace nxrt
