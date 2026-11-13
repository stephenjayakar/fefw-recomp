// SPDX-License-Identifier: GPL-3.0-or-later
// Function table + HLE thunks. Pattern from smo-recomp runtime/core/hostrt.c (open-addressing hash, hostfn thunks) and
// SuperMarioWonderRecomp runtime/src/loader.cpp (thunks, logging stub for unknown imports).
#include "nxrt/dispatch.h"

#include <windows.h>

#include <algorithm>
#include <bit>
#include <atomic>
#include "nxrt/tl_counts.h"
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "nxrt/ftrace.h"
#include "nxrt/config.h"
#include "nxrt/log.h"
#include "nxrt/memory.h"

// The flat table of the main module's functions, under C names: one entry per 4-byte guest slot of [kMainBase, kMainBase +
// kFlatSpan), the function as an offset from nx_fn_flat_host (0: not a registered function). Generated code built with a cxx.h that
// has nx_call_indirect reads it at the call site (a range test and a load), so every site has its own indirect branch; anything else
// goes through call_indirect. nx_fn_flat_n stays 0 until the table exists.
extern "C" {
uint32_t* nx_fn_flat = nullptr;
uint64_t nx_fn_flat_n = 0;
uintptr_t nx_fn_flat_host = 0;
}

namespace nxrt {
namespace {

// ---- function hash: open addressing, power of two, load < 1/2. Readers never lock; a writer publishes the fn first and the
// address last (release), so a reader that sees the address sees the function. Growth builds a new table and swaps the
// pointer; retired tables stay allocated until fn_clear because a reader may still be walking one.
struct Slot {
    std::atomic<uint64_t> addr{0};
    std::atomic<GuestFn> fn{nullptr};
};
struct Table {
    explicit Table(size_t n) : mask(n - 1), slots(new Slot[n]) {}
    size_t mask;
    std::unique_ptr<Slot[]> slots;
    size_t used = 0;
};

std::mutex g_write;
std::atomic<Table*> g_table{nullptr};

// Per-thread direct-mapped cache in front of the function hash (S34). Indirect calls (virtual calls, function pointers) are the most
// frequent thing the guest does, and the hash table of 138,000 functions is 4 MiB: a lookup is a cache miss. The same few hundred
// targets repeat, so a 1,024-entry table per thread (16 KiB) answers most of them. A cache holds only functions that were found; a
// registration that replaces a function, and fn_clear, bump the epoch and every thread drops its cache before the next lookup.
std::atomic<uint32_t> g_fn_epoch{0};
struct FnCacheEntry {
    uint64_t addr;
    GuestFn fn;
};
constexpr size_t kFnCache = 1024;
thread_local FnCacheEntry t_fn_cache[kFnCache];
thread_local uint32_t t_fn_epoch = 0;

// Flat table (above): 64 MiB of guest addresses, 64 MiB reserved and committed once; a page stays untouched until an entry is written.
constexpr uint64_t kFlatSpan = 0x4000000;

inline uint64_t flat_index(uint64_t a) { return std::rotr(a - mem::kMainBase, 2); }   // a misaligned address is out of range

void flat_put(uint64_t guest_addr, GuestFn fn) {   // under g_write
    if (!nx_fn_flat) {
        void* p = VirtualAlloc(nullptr, kFlatSpan, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!p) return;
        nx_fn_flat_host = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
        nx_fn_flat = static_cast<uint32_t*>(p);
        __atomic_store_n(&nx_fn_flat_n, kFlatSpan / 4, __ATOMIC_RELEASE);
    }
    const uint64_t i = flat_index(guest_addr);
    if (i >= nx_fn_flat_n) return;
    // a function outside the image (none today) stays out of the table: its callers take call_indirect
    const uintptr_t h = reinterpret_cast<uintptr_t>(fn);
    const uint32_t off = h > nx_fn_flat_host && h - nx_fn_flat_host <= UINT32_MAX ? uint32_t(h - nx_fn_flat_host) : 0;
    __atomic_store_n(&nx_fn_flat[i], off, __ATOMIC_RELEASE);
}
std::vector<std::unique_ptr<Table>> g_all_tables;   // owns current + retired tables
std::atomic<size_t> g_fn_count{0};

inline size_t hash_addr(uint64_t a) { return static_cast<size_t>(((a >> 2) * 0x9E3779B97F4A7C15ull) >> 24); }

Table* table_for_write() {
    Table* t = g_table.load(std::memory_order_relaxed);
    if (t && (t->used + 1) * 2 <= t->mask + 1) return t;
    size_t n = t ? (t->mask + 1) * 2 : 4096;
    auto nt = std::make_unique<Table>(n);
    if (t) {
        for (size_t i = 0; i <= t->mask; i++) {
            uint64_t a = t->slots[i].addr.load(std::memory_order_relaxed);
            if (!a) continue;
            size_t h = hash_addr(a) & nt->mask;
            while (nt->slots[h].addr.load(std::memory_order_relaxed)) h = (h + 1) & nt->mask;
            nt->slots[h].fn.store(t->slots[i].fn.load(std::memory_order_relaxed), std::memory_order_relaxed);
            nt->slots[h].addr.store(a, std::memory_order_relaxed);
            nt->used++;
        }
    }
    Table* raw = nt.get();
    g_all_tables.push_back(std::move(nt));
    g_table.store(raw, std::memory_order_release);
    return raw;
}

// host-address order for guest_function_of: rebuilt lazily when the table changed
struct ByHost {
    std::vector<std::pair<uint64_t, uint64_t>> v;   // (host fn address, guest address) sorted by host address
    size_t built_for = ~size_t(0);
};
std::mutex g_byhost_lock;
ByHost g_byhost;

// ---- HLE thunks
struct Thunk {
    std::string name;
    HleFn fn = nullptr;       // null: logging stub
    std::atomic<uint64_t> calls{0};
    std::atomic<int> ft_zone{-1};   // frame trace zone ("hle:<name>"), registered on the first slow call
};
// Calls of handled thunks are counted per thread (tl_counts.h: every guest thread calls the hot ones, and a shared counter bounced between
// cores); thunks past the per-thread table and logging stubs count in Thunk::calls. hle_calls() adds both.
struct ThunkCountTag {};
constexpr size_t kTlThunks = 4096;
TlCounts<kTlThunks, ThunkCountTag> g_thunk_tl;
std::mutex g_hle_lock;
std::vector<std::unique_ptr<Thunk>> g_thunks;        // index = (addr - kHleBase) / 16
std::atomic<size_t> g_thunk_count{0};                // published size (thunks are never removed outside hle_clear)
std::unordered_map<std::string, size_t> g_thunk_by_name;
thread_local const char* t_hle_name = nullptr;

// [hle] trace = <name prefix> (env NXRT_HLE_TRACE): every call of an HLE thunk whose name starts with the prefix is logged in order
// ("hle-trace #seq name(x0..x3) lr"), up to [hle] trace_limit calls (default 20000). Diagnostics for the boot: the order in which a
// game calls an API (NVN).
struct HleTrace {
    std::string prefix;
    int64_t limit = 0;
    std::atomic<int64_t> seq{0};
};
HleTrace& hle_trace() {
    static HleTrace* t = [] {
        auto* h = new HleTrace;
        h->prefix = config_get_str("hle", "trace", "");
        h->limit = config_get_int("hle", "trace_limit", 20000);
        return h;
    }();
    return *t;
}

constexpr size_t kMaxThunks = mem::kHleSize / 16;

size_t thunk_index(uint64_t addr) { return static_cast<size_t>((addr - mem::kHleBase) / 16); }

}  // namespace

void fn_register(uint64_t guest_addr, GuestFn fn) {
    if (!guest_addr || !fn) return;
    std::lock_guard<std::mutex> lk(g_write);
    Table* t = table_for_write();
    size_t h = hash_addr(guest_addr) & t->mask;
    for (;;) {
        uint64_t a = t->slots[h].addr.load(std::memory_order_relaxed);
        if (a == guest_addr) {
            t->slots[h].fn.store(fn, std::memory_order_release);
            flat_put(guest_addr, fn);
            g_fn_epoch.fetch_add(1, std::memory_order_release);      // a replaced function: the per-thread caches may hold the old one
            return;
        }
        if (!a) break;
        h = (h + 1) & t->mask;
    }
    t->slots[h].fn.store(fn, std::memory_order_relaxed);
    t->slots[h].addr.store(guest_addr, std::memory_order_release);
    flat_put(guest_addr, fn);
    t->used++;
    g_fn_count.fetch_add(1, std::memory_order_relaxed);
}

GuestFn fn_lookup(uint64_t guest_addr) {
    Table* t = g_table.load(std::memory_order_acquire);
    if (!t) return nullptr;
    size_t h = hash_addr(guest_addr) & t->mask;
    for (;;) {
        uint64_t a = t->slots[h].addr.load(std::memory_order_acquire);
        if (a == guest_addr) return t->slots[h].fn.load(std::memory_order_acquire);
        if (!a) return nullptr;
        h = (h + 1) & t->mask;
    }
}

size_t fn_count() { return g_fn_count.load(std::memory_order_relaxed); }

void fn_clear() {
    std::lock_guard<std::mutex> lk(g_write);
    g_table.store(nullptr);
    if (nx_fn_flat) std::memset(nx_fn_flat, 0, kFlatSpan);
    g_fn_epoch.fetch_add(1, std::memory_order_release);
    g_all_tables.clear();
    g_fn_count = 0;
    std::lock_guard<std::mutex> l2(g_byhost_lock);
    g_byhost = {};
}

uint64_t guest_function_of(uint64_t host_rip) {
    std::lock_guard<std::mutex> lk(g_byhost_lock);
    size_t n = g_fn_count.load(std::memory_order_relaxed);
    if (g_byhost.built_for != n) {
        g_byhost.v.clear();
        Table* t = g_table.load(std::memory_order_acquire);
        if (t)
            for (size_t i = 0; i <= t->mask; i++) {
                uint64_t a = t->slots[i].addr.load(std::memory_order_acquire);
                GuestFn f = a ? t->slots[i].fn.load(std::memory_order_acquire) : nullptr;
                if (f) g_byhost.v.emplace_back(reinterpret_cast<uint64_t>(f), a);
            }
        std::sort(g_byhost.v.begin(), g_byhost.v.end());
        g_byhost.built_for = n;
    }
    auto& v = g_byhost.v;
    auto it = std::upper_bound(v.begin(), v.end(), std::make_pair(host_rip, ~uint64_t(0)));
    if (it == v.begin()) return 0;
    --it;
    if (host_rip - it->first > 0x400000) return 0;
    return it->second;
}

// ---- HLE thunks -----------------------------------------------------------------------------------------------------

uint64_t hle_register(const char* name, HleFn fn) {
    std::lock_guard<std::mutex> lk(g_hle_lock);
    if (g_thunks.capacity() < kMaxThunks) g_thunks.reserve(kMaxThunks);   // never reallocates: dispatch reads without the lock
    auto it = g_thunk_by_name.find(name);
    size_t i;
    if (it != g_thunk_by_name.end()) {
        i = it->second;
    } else {
        if (g_thunks.size() >= kMaxThunks) {
            NXLOG_E("dispatch", "out of HLE thunks (%zu)", kMaxThunks);
            return 0;
        }
        i = g_thunks.size();
        g_thunks.push_back(std::make_unique<Thunk>());
        g_thunks[i]->name = name;
        g_thunk_by_name.emplace(name, i);
        g_thunk_count.store(g_thunks.size(), std::memory_order_release);
    }
    g_thunks[i]->fn = fn;
    return mem::kHleBase + i * 16;
}

uint64_t hle_address(const char* name) {
    {
        std::lock_guard<std::mutex> lk(g_hle_lock);
        auto it = g_thunk_by_name.find(name);
        if (it != g_thunk_by_name.end()) return mem::kHleBase + it->second * 16;
    }
    return hle_register(name, nullptr);   // no handler: a logging stub
}

HleFn hle_lookup(const char* name) {
    std::lock_guard<std::mutex> lk(g_hle_lock);
    auto it = g_thunk_by_name.find(name);
    return it == g_thunk_by_name.end() ? nullptr : g_thunks[it->second]->fn;
}

bool hle_is_thunk(uint64_t addr) {
    return addr - mem::kHleBase < mem::kHleSize && thunk_index(addr) < g_thunk_count.load(std::memory_order_acquire);
}

const char* hle_name_at(uint64_t addr) {
    if (!hle_is_thunk(addr)) return nullptr;
    return g_thunks[thunk_index(addr)]->name.c_str();
}

size_t hle_count() { return g_thunk_count.load(std::memory_order_acquire); }

size_t hle_stub_count() {
    std::lock_guard<std::mutex> lk(g_hle_lock);
    size_t n = 0;
    for (auto& t : g_thunks) n += t->fn == nullptr;
    return n;
}

uint64_t hle_calls(uint64_t addr) {
    if (!hle_is_thunk(addr)) return 0;
    const size_t i = thunk_index(addr);
    return g_thunks[i]->calls.load() + (i < kTlThunks ? g_thunk_tl.sum(i) : 0);
}

std::vector<HleInfo> hle_list() {
    std::lock_guard<std::mutex> lk(g_hle_lock);
    std::vector<HleInfo> v;
    for (size_t i = 0; i < g_thunks.size(); ++i)
        v.push_back({g_thunks[i]->name, g_thunks[i]->calls.load() + (i < kTlThunks ? g_thunk_tl.sum(i) : 0), g_thunks[i]->fn == nullptr});
    return v;
}

void hle_clear() {
    std::lock_guard<std::mutex> lk(g_hle_lock);
    g_thunk_count.store(0);
    g_thunks.clear();
    g_thunk_tl.clear();
    g_thunk_by_name.clear();
}

const char* hle_current_name() { return t_hle_name; }
void hle_restore_current(const char* name) { t_hle_name = name; }

namespace {
thread_local uint32_t t_host_cb_flags = 0;
}
uint32_t host_callback_flags() { return t_host_cb_flags; }
void host_callback_restore(uint32_t flags) { t_host_cb_flags = flags; }

// ---- dispatch ---------------------------------------------------------------------------------------------------------

void trap(const char* what, uint64_t addr) {
    NXLOG_E("guest", "GUEST TRAP: %s (0x%llx)", what, static_cast<unsigned long long>(addr));
    log_flush();
    RaiseException(0xE0000001, EXCEPTION_NONCONTINUABLE, 0, nullptr);
    TerminateProcess(GetCurrentProcess(), 2);
    for (;;) {}
}

void dispatch_call(::Ctx* c, uint64_t addr) {
    // HLE thunks live in their own address range, away from every lifted function: no hash lookup for them
    const bool in_hle_range = addr - mem::kHleBase < mem::kHleSize;
    if (!in_hle_range) {
        const uint64_t i = flat_index(addr);
        if (i < __atomic_load_n(&nx_fn_flat_n, __ATOMIC_ACQUIRE)) {
            if (const uint32_t off = __atomic_load_n(&nx_fn_flat[i], __ATOMIC_ACQUIRE)) {
                reinterpret_cast<GuestFn>(nx_fn_flat_host + off)(c);
                return;
            }
        }
        const uint32_t ep = g_fn_epoch.load(std::memory_order_acquire);
        if (t_fn_epoch != ep) {
            std::memset(t_fn_cache, 0, sizeof t_fn_cache);
            t_fn_epoch = ep;
        }
        FnCacheEntry& e = t_fn_cache[((addr >> 2) ^ (addr >> 12)) & (kFnCache - 1)];
        // An empty entry is {0, nullptr}: without the fn test a guest call to address 0 would "hit" it and jump to host address 0.
        if (e.fn && e.addr == addr) {
            e.fn(c);
            return;
        }
        if (GuestFn fn = fn_lookup(addr)) {
            e.addr = addr;
            e.fn = fn;
            fn(c);
            return;
        }
    }
    if (hle_is_thunk(addr)) {
        const size_t ti = thunk_index(addr);
        Thunk& t = *g_thunks[ti];
        if (t.fn && ti < kTlThunks) g_thunk_tl.add(ti);
        else t.calls.fetch_add(1, std::memory_order_relaxed);
        const char* saved = t_hle_name;
        t_hle_name = t.name.c_str();
        // [hle] trace: the line is written after the handler ran, so it carries the result (x0) next to the arguments
        const HleTrace& tr = hle_trace();
        const bool traced = !tr.prefix.empty() && t.name.compare(0, tr.prefix.size(), tr.prefix) == 0;
        uint64_t targs[4] = {};
        int64_t tseq = 0;
        if (traced) {
            tseq = const_cast<HleTrace&>(tr).seq.fetch_add(1, std::memory_order_relaxed) + 1;
            for (int i = 0; i < 4; i++) targs[i] = reg(c, i);
        }
        if (t.fn) {
            if (ft::on()) {
                const uint64_t t0 = ft::now_ns();
                t.fn(c);
                const uint64_t t1 = ft::now_ns();
                if (t1 - t0 >= 100000) {       // frame trace: HLE calls (services, NVN entry points) of at least 100 us
                    int z = t.ft_zone.load(std::memory_order_relaxed);
                    if (z < 0) { z = ft::zone_id(("hle:" + t.name).c_str()); t.ft_zone.store(z, std::memory_order_relaxed); }
                    ft::event(static_cast<uint16_t>(z), t0, t1, 0);
                }
            } else {
                t.fn(c);
            }
        } else {
            // No handler: the console would fault on an unresolved import. A silent 0 is a fake answer, so say so (every
            // call up to 20, then every 1000th) and return 0.
            uint64_t n = t.calls.load(std::memory_order_relaxed);
            if (n <= 20 || n % 1000 == 0)
                NXLOG_W("hle", "UNIMPLEMENTED import %s called (call #%llu, lr=0x%llx, x0=0x%llx x1=0x%llx): returning 0",
                        t.name.c_str(), static_cast<unsigned long long>(n), static_cast<unsigned long long>(lr(c)),
                        static_cast<unsigned long long>(reg(c, 0)), static_cast<unsigned long long>(reg(c, 1)));
            reg(c, 0) = 0;
        }
        if (traced && tseq <= tr.limit)
            NXLOG_I("hle-trace", "#%lld %s(x0=0x%llx x1=0x%llx x2=0x%llx x3=0x%llx) lr=0x%llx -> 0x%llx", static_cast<long long>(tseq), t.name.c_str(),
                    static_cast<unsigned long long>(targs[0]), static_cast<unsigned long long>(targs[1]), static_cast<unsigned long long>(targs[2]),
                    static_cast<unsigned long long>(targs[3]), static_cast<unsigned long long>(lr(c)), static_cast<unsigned long long>(reg(c, 0)));
        t_hle_name = saved;
        return;
    }
    trap("indirect call to an address that is not a known function", addr);
}

}  // namespace nxrt
