// SPDX-License-Identifier: GPL-3.0-or-later
// Gauges and the soak sampler (declared in nxrt/gauge.h).
#include "nxrt/gauge.h"

#include <windows.h>
#include <intrin.h>
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
// The game's presenting thread (a handle opened at its first present) and the TSC rate, for its CPU time inside a slow frame: a frame where
// the render thread idles and the game thread is "guest CPU" but was not on the CPU (page faults, preempted) shows it here.
std::atomic<HANDLE> g_game_thread{nullptr};
uint64_t g_prev_game_cycles = 0;                  // the presenting thread only
uint64_t g_prev_faults = 0;
uint64_t g_tsc0 = 0, g_qpc0 = 0;
uint64_t game_cycles() {
    HANDLE h = g_game_thread.load(std::memory_order_acquire);
    ULONG64 c = 0;
    if (h) QueryThreadCycleTime(h, &c);
    return c;
}
uint64_t process_faults() {
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof pmc;
    return GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc) ? pmc.PageFaultCount : 0;
}

struct HitchEvent {
    double ms = 0;
    uint64_t frame = 0;
    double at_s = 0;                              // seconds since the monitor started
    int64_t val[kMaxSources] = {};
    uint64_t phase[kHpCount] = {};                // ticks of each phase inside the frame
    uint64_t game_gap = 0;                        // ticks: longest gap between the game's present calls
    double game_cpu_ms = -1;                      // CPU time of the game's presenting thread inside the frame (-1: unknown)
    uint64_t faults = 0;                          // page faults of the process inside the frame (soft and hard)
};
std::mutex g_ev_mu;                               // taken by the presenting thread only for a frame over the limit
std::vector<HitchEvent> g_pending;

// Called for every presented frame at tick t (previous frame's tick prev_t, 0 for the first), frame number n.
void hitch_on_frame(uint64_t t, uint64_t prev_t, uint64_t n) {
    if (!g_hitch_on.load(std::memory_order_relaxed)) return;
    const size_t ns = g_nsrc.load(std::memory_order_acquire);
    uint64_t cur[kMaxSources];
    for (size_t i = 0; i < ns; i++) cur[i] = g_src[i].counter ? g_src[i].counter() : (uint64_t)g_src[i].level();
    uint64_t curp[kHpCount];
    for (int i = 0; i < kHpCount; i++) curp[i] = g_phase[i].load(std::memory_order_relaxed);
    const uint64_t gap = g_game_gap_max.exchange(0, std::memory_order_relaxed);
    const uint64_t thr = g_hitch_ticks.load(std::memory_order_relaxed);
    const uint64_t gc = game_cycles(), pf = process_faults();
    if (thr && prev_t && t > prev_t && t - prev_t > thr) {
        HitchEvent ev;
        ev.ms = (double)(t - prev_t) * 1000.0 / qpc_freq();
        ev.frame = n;
        ev.at_s = (double)(t - g_t0_ticks.load(std::memory_order_relaxed)) / qpc_freq();
        for (size_t i = 0; i < ns; i++) ev.val[i] = g_src[i].counter ? (int64_t)(cur[i] - g_prev_val[i]) : (int64_t)cur[i];
        for (int i = 0; i < kHpCount; i++) ev.phase[i] = curp[i] - g_prev_phase[i];
        ev.game_gap = gap;
        if (gc && g_prev_game_cycles && gc >= g_prev_game_cycles && g_tsc0) {   // cycles -> ms at the TSC rate measured since the monitor started
            const double qms = static_cast<double>(t - g_qpc0) * 1000.0 / qpc_freq();
            const double tsc_per_ms = qms > 100 ? static_cast<double>(__rdtsc() - g_tsc0) / qms : 0;
            if (tsc_per_ms > 0) ev.game_cpu_ms = static_cast<double>(gc - g_prev_game_cycles) / tsc_per_ms;
        }
        ev.faults = pf >= g_prev_faults ? pf - g_prev_faults : 0;
        std::lock_guard<std::mutex> lk(g_ev_mu);
        if (g_pending.size() < 256) g_pending.push_back(ev);
    }
    for (size_t i = 0; i < ns; i++) g_prev_val[i] = cur[i];
    for (int i = 0; i < kHpCount; i++) g_prev_phase[i] = curp[i];
    g_prev_game_cycles = gc;
    g_prev_faults = pf;
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
    {   // the uploads' parts: [take, prepare, staging]
        std::string up;
        part(up, "take", kHpRenderTexTake);
        part(up, "prepare", kHpRenderTexPrep);
        part(up, "staging", kHpRenderTexStaging);
        if (!up.empty()) rep += " [" + up + "]";
    }
    part(rep, "image creation", kHpRenderImage);
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
    if (ev.game_cpu_ms >= 0) {
        snprintf(b, sizeof b, ", presenting thread on the CPU %.1f", ev.game_cpu_ms);
        game += b;
    }
    if (ev.faults) {
        snprintf(b, sizeof b, ", page faults +%llu", static_cast<unsigned long long>(ev.faults));
        game += b;
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

bool hitch_frame_age(uint64_t* frames, double* age_ms) {
    const uint64_t n = g_ring_n.load(std::memory_order_acquire);
    if (n == 0) return false;
    const uint64_t t = g_ring[(n - 1) % kRing].load(std::memory_order_relaxed);
    const uint64_t now = qpc();
    if (frames) *frames = n;
    if (age_ms) *age_ms = now > t ? static_cast<double>(now - t) * 1000.0 / qpc_freq() : 0.0;
    return true;
}

uint64_t hitch_ticks() { return qpc(); }
void hitch_phase_add(int phase, uint64_t ticks) {
    if (phase >= 0 && phase < kHpCount) g_phase[phase].fetch_add(ticks, std::memory_order_relaxed);
}
void hitch_set_render_thread() { t_render_thread = true; }
bool hitch_on_render_thread() { return t_render_thread; }
bool hitch_on_game_present_thread() { return t_game_present_thread; }
void hitch_note_game_present() {
    if (!t_game_present_thread && !g_game_thread.load(std::memory_order_relaxed)) {
        HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, GetCurrentThreadId());
        HANDLE expected = nullptr;
        if (h && !g_game_thread.compare_exchange_strong(expected, h)) CloseHandle(h);
    }
    t_game_present_thread = true;
    const uint64_t t = qpc();
    const uint64_t prev = g_game_present_last.exchange(t, std::memory_order_relaxed);
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
    g_qpc0 = qpc();
    g_tsc0 = __rdtsc();
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
