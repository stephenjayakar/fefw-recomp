// SPDX-License-Identifier: GPL-3.0-or-later
// SVC table and dispatcher (see include/nxrt/svc.h). Dispatch pattern from SuperMarioWonderRecomp runtime/src/kernel.cpp (MIT).
#include "nxrt/svc.h"
#include <windows.h>
#include "nxrt/tl_counts.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <functional>
#include <map>
#include <mutex>
#include <vector>
#include <tuple>
#include <cstring>
#include <string>

#include "nxrt/config.h"
#include "nxrt/diag.h"
#include "nxrt/dispatch.h"
#include "nxrt/gauge.h"
#include "nxrt/kernel.h"
#include "nxrt/ftrace.h"
#include "nxrt/log.h"
#include "nxrt/memory.h"
#include "nxrt/module.h"
#include "nxrt/tt.h"

namespace nxrt {
namespace {

struct NameEntry { uint32_t num; const char* name; };
#define NXRT_SVC(n, name) {n, name},
const NameEntry kNames[] = {
#include "svc_names.inc"
};
#undef NXRT_SVC

SvcHandler g_handlers[kSvcCount];
// Per-thread counts (tl_counts.h): the wait and lock SVCs come from every guest thread at a high rate, and the shared counters (eight to a
// cache line) moved between cores on each call.
struct SvcCountTag {};
TlCounts<kSvcCount, SvcCountTag> g_calls;
std::once_flag g_init_once;

std::mutex g_unk_lock;
std::map<uint32_t, SvcUnknownRecord> g_unknown;
std::atomic<uint64_t> g_unknown_total{0};
std::atomic<int64_t> g_log_limit{-1};

// [diag] wait_sites (env NXRT_DIAG_WAIT_SITES=1): every svc that takes 20 us or more is counted against its thread, its number and the first two
// return addresses in main (frame-pointer walk); the 12 sites with the most time are logged every 10 s (time since the previous dump).
struct WaitSite { uint64_t n = 0, ns = 0, req_ns = 0; };
std::mutex g_ws_lock;
std::map<std::tuple<std::string, uint32_t, uint64_t, uint64_t>, WaitSite> g_ws;
uint64_t g_ws_dump_ns = 0;
bool wait_sites_on() {
    static const bool on = config_get_bool("diag", "wait_sites", false);
    return on;
}
void wait_site_record(::Ctx* c, uint32_t number, uint64_t req, uint64_t t0, uint64_t t1) {
    static Module* main_mod = module_by_name("main");
    uint64_t site[2] = {0, 0};
    int k = 0;
    if (main_mod && main_mod->contains(c->x[30])) site[k++] = c->x[30];
    uint64_t fp = c->x[29];
    const uint64_t sp = c->x[31];
    for (int depth = 0; depth < 12 && k < 2 && fp && (fp & 7) == 0; ++depth) {
        if (fp < sp || fp - sp > (256u << 10)) break;     // frames live just above sp on the thread's own stack: no VirtualQuery per step
        uint64_t next, lr;
        std::memcpy(&next, reinterpret_cast<const void*>(fp), 8);
        std::memcpy(&lr, reinterpret_cast<const void*>(fp + 8), 8);
        if (main_mod && main_mod->contains(lr) && (k == 0 || site[k - 1] != lr)) site[k++] = lr;
        if (next <= fp) break;
        fp = next;
    }
    const kernel::KThread* th = kernel::current_thread();
    std::lock_guard<std::mutex> lk(g_ws_lock);
    WaitSite& w = g_ws[{th ? th->name : std::string("?"), number, site[0], site[1]}];
    w.n++; w.ns += t1 - t0; w.req_ns += req;
    if (!g_ws_dump_ns) g_ws_dump_ns = t1;
    if (t1 - g_ws_dump_ns < 10000000000ull) return;
    const double secs = static_cast<double>(t1 - g_ws_dump_ns) / 1e9;
    std::vector<std::pair<uint64_t, decltype(g_ws)::const_iterator>> v;
    for (auto it = g_ws.cbegin(); it != g_ws.cend(); ++it) v.push_back({it->second.ns, it});
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    NXLOG_I("svc", "wait sites over %.1f s (svc >= 20 us): the 8 largest, then the guest main thread's 8 largest", secs);
    size_t shown = 0, main_shown = 0;
    for (size_t i = 0; i < v.size() && (shown < 8 || main_shown < 8); ++i) {
        const auto& [key, w2] = *v[i].second;
        const bool is_main = std::get<0>(key) == "main";
        if (shown >= 8 && !(is_main && main_shown < 8)) continue;
        ++shown;
        if (is_main) ++main_shown;
        NXLOG_I("svc", "  %-14s %-26s %6.1f ms/s %7.0f/s avg %6.0f us (req %6.0f us)  %s <- %s", std::get<0>(key).c_str(), svc_name(std::get<1>(key)),
                w2.ns / secs / 1e6, w2.n / secs, w2.ns / 1e3 / w2.n, w2.req_ns / 1e3 / w2.n, module_describe(std::get<2>(key)).c_str(),
                module_describe(std::get<3>(key)).c_str());
    }
    g_ws.clear();
    g_ws_dump_ns = t1;
}

// ---- gauges (include/nxrt/gauge.h) ----
// Removes its gauge when the file's statics are torn down (declared after the tables above, so it goes first).
struct GaugeHook {
    const char* name;
    GaugeHook(const char* n, std::function<int64_t()> f) : name(n) { gauge_register(n, std::move(f)); }
    ~GaugeHook() { gauge_unregister(name); }
};
GaugeHook gauge_svc_calls("kernel.svc_calls", [] {   // SVCs answered since start (all numbers)
    return static_cast<int64_t>(g_calls.total());
});
GaugeHook gauge_svc_unknown_kinds("kernel.svc_unknown_kinds", [] {   // distinct unknown SVC numbers seen (the map only grows)
    std::lock_guard<std::mutex> lk(g_unk_lock);
    return static_cast<int64_t>(g_unknown.size());
});

}  // namespace

const char* svc_name(uint32_t number) {
    for (auto& e : kNames)
        if (e.num == number) return e.name;
    return "?";
}

void svc_register(uint32_t number, SvcHandler fn) {
    if (number < kSvcCount) g_handlers[number] = fn;
}

SvcHandler svc_handler(uint32_t number) { return number < kSvcCount ? g_handlers[number] : nullptr; }

void svc_init() {
    std::call_once(g_init_once, [] {
        svc_register_memory_group();
        svc_register_info_group();
        svc_register_thread_group();
        svc_register_ipc_group();
        size_t n = 0;
        for (auto* h : g_handlers) n += h != nullptr;
        NXLOG_I("svc", "%zu SVC handlers registered", n);
    });
}

static void report_unknown(::Ctx* c, uint32_t number, uint64_t pc) {
    uint64_t count;
    {
        std::lock_guard<std::mutex> lk(g_unk_lock);
        SvcUnknownRecord& r = g_unknown[number];
        if (r.count == 0) {
            r.number = number;
            for (int i = 0; i < 8; i++) r.first_args[i] = c->x[i];
            r.first_lr = c->x[30];
            r.first_fn = c->fn;
            r.first_pc = pc;
        }
        count = ++r.count;
    }
    g_unknown_total.fetch_add(1, std::memory_order_relaxed);
    if (g_log_limit.load() < 0) g_log_limit = config_get_int("svc", "unknown_log_limit", 100);
    if (static_cast<int64_t>(count) <= g_log_limit.load() || count % 1000 == 0) {
        const kernel::KThread* t = kernel::current_thread();
        NXLOG_W("svc", "UNKNOWN SVC 0x%02X (%s) #%llu x0=%llx x1=%llx x2=%llx x3=%llx x4=%llx x5=%llx x6=%llx x7=%llx caller: lr=%llx pc=%llx in f_%llx thread %s",
                number, svc_name(number), static_cast<unsigned long long>(count), (unsigned long long)c->x[0], (unsigned long long)c->x[1],
                (unsigned long long)c->x[2], (unsigned long long)c->x[3], (unsigned long long)c->x[4], (unsigned long long)c->x[5],
                (unsigned long long)c->x[6], (unsigned long long)c->x[7], (unsigned long long)c->x[30], (unsigned long long)pc,
                (unsigned long long)c->fn, t ? t->name.c_str() : "?");
    }
}

namespace {
std::atomic<void (*)()> g_busy_hook{nullptr};

// The hitch line's split of the game's presenting thread: time in system calls, and of it IPC (services, file reads) and waits.
struct GameSvcTimer {
    uint64_t t0;
    uint32_t number;
    explicit GameSvcTimer(uint32_t n) : t0(hitch_on_game_present_thread() ? hitch_ticks() : 0), number(n) {}
    ~GameSvcTimer() {
        if (!t0) return;
        const uint64_t d = hitch_ticks() - t0;
        hitch_phase_add(kHpGameSvc, d);
        if (number >= 0x21 && number <= 0x23) hitch_phase_add(kHpGameIpc, d);   // SendSyncRequest(WithUserBuffer), SendAsyncRequestWithUserBuffer
        else if (number == 0x0B || number == 0x18 || number == 0x1A || number == 0x1C || number == 0x34) hitch_phase_add(kHpGameWait, d);
    }
    GameSvcTimer(const GameSvcTimer&) = delete;
    GameSvcTimer& operator=(const GameSvcTimer&) = delete;
};
}
void svc_set_busy_hook(void (*fn)()) { g_busy_hook.store(fn, std::memory_order_release); }

void svc_dispatch(::Ctx* c, uint32_t number, uint64_t pc) {
    GameSvcTimer game_timer(number);
    if (kernel::KThread* self = kernel::current_thread()) self->last_proc.store(static_cast<int>(GetCurrentProcessorNumber()), std::memory_order_relaxed);   // the starvation line
    if (auto fn = g_busy_hook.load(std::memory_order_relaxed)) fn();
    if (tt::detail::svc_every.load(std::memory_order_relaxed)) tt::detail::on_svc(c);      // test tooling: frame_source = svc:N
    SvcHandler h = svc_handler(number);
    if (number < kSvcCount) g_calls.add(number);
    if (!h) {
        report_unknown(c, number, pc);
        c->x[0] = kernel::kNotImplemented;
        return;
    }
    if (wait_sites_on()) {
        const uint64_t req = number == 0x0B ? c->x[0] : number == 0x18 || number == 0x1C ? c->x[3] : 0;   // SleepThread / WaitSynchronization / WaitProcessWideKeyAtomic timeout
        const uint64_t t0 = ft::now_ns();
        h(c);
        const uint64_t t1 = ft::now_ns();
        if (t1 - t0 >= 20000) wait_site_record(c, number, req, t0, t1);
        if (ft::on() && t1 - t0 >= 100000) { static const uint16_t z = ft::zone_id("svc"); ft::event(z, t0, t1, number); }
        return;
    }
    if (ft::on()) {
        const uint64_t t0 = ft::now_ns();
        h(c);
        const uint64_t t1 = ft::now_ns();
        if (t1 - t0 >= 100000) { static const uint16_t z = ft::zone_id("svc"); ft::event(z, t0, t1, number); }
        return;
    }
    h(c);
}

std::vector<SvcUnknownRecord> svc_unknown_records() {
    std::lock_guard<std::mutex> lk(g_unk_lock);
    std::vector<SvcUnknownRecord> v;
    for (auto& kv : g_unknown) v.push_back(kv.second);
    return v;
}

uint64_t svc_unknown_total() { return g_unknown_total.load(); }
uint64_t svc_call_count(uint32_t number) { return number < kSvcCount ? g_calls.sum(number) : 0; }

void svc_reset_counters() {
    std::lock_guard<std::mutex> lk(g_unk_lock);
    g_unknown.clear();
    g_unknown_total = 0;
    g_log_limit = -1;
    g_calls.clear();
}

std::string svc_summary() {
    uint64_t total = 0;
    size_t distinct = 0;
    std::vector<uint64_t> counts(kSvcCount);
    for (uint32_t n = 0; n < kSvcCount; n++) counts[n] = g_calls.sum(n);
    for (uint64_t c : counts)
        if (c) { total += c; distinct++; }
    std::string s = "svc: " + std::to_string(total) + " calls, " + std::to_string(distinct) + " distinct, " + std::to_string(g_unknown_total.load()) + " unknown";
    // The busiest three, so a guest spinning on one SVC shows which.
    std::vector<std::pair<uint64_t, uint32_t>> top;
    for (uint32_t n = 0; n < kSvcCount; n++)
        if (const uint64_t k = counts[n]) top.emplace_back(k, n);
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    if (!top.empty()) {
        s += " (most:";
        for (size_t i = 0; i < top.size() && i < 3; i++) {
            char b[64];
            snprintf(b, sizeof b, " %s x%llu", svc_name(top[i].second), static_cast<unsigned long long>(top[i].first));
            s += b;
        }
        s += ")";
    }
    std::lock_guard<std::mutex> lk(g_unk_lock);
    for (auto& kv : g_unknown) {
        char b[96];
        snprintf(b, sizeof b, " [0x%02X %s x%llu]", kv.first, svc_name(kv.first), static_cast<unsigned long long>(kv.second.count));
        s += b;
    }
    return s;
}

}  // namespace nxrt
