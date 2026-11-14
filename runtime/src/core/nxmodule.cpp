// SPDX-License-Identifier: GPL-3.0-or-later
// Registry of lifted modules (nxmodule.h) and import resolution for host_import(c, import_id).
// Pattern: smo-recomp runtime/core/hostrt.c (module tables registered at start-up, imports bound by name); the (module index, symbol)
// addressing is the lifter's ABI 2 (recomp/include/nxrecomp/ctx.h NX_IMPORT_ID).
#include "nxrt/nxmodule.h"

#include <windows.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include "nxrt/tl_counts.h"
#include <unordered_map>

#include "nxrt/dispatch.h"
#include "nxrt/gauge.h"
#include "nxrt/log.h"
#include "nxrt/module.h"

namespace nxrt {
namespace {

constexpr uint32_t kMaxModules = 1u << (32 - 20);   // NX_IMPORT_ID keeps the module index in the top 12 bits

struct Entry {
    const NxModule* nx = nullptr;
    Module* image = nullptr;
    std::unique_ptr<std::atomic<uint64_t>[]> cache;   // per DynSym: resolved target, 0 = not resolved yet
    size_t ncache = 0;
};

constexpr size_t kMaxListed = 64;
std::mutex g_lock;                                    // registration and clearing; lookups are lock free
std::unique_ptr<Entry> g_owned[kMaxListed];           // owns the entries, registration order
std::atomic<Entry*> g_list[kMaxListed];               // the same, for lock-free readers: entries [0, g_count) are valid
std::atomic<size_t> g_count{0};
std::atomic<Entry*> g_by_index[kMaxModules];
// Imports through the registry, counted per thread (tl_counts.h: a relaxed load and store into the thread's own block): every guest thread calls
// imports at a high rate, and one shared counter made each call a cache-line transfer between cores (8.9 % of the game thread's busy samples,
// 46-52 % of the job workers', sat on its fetch_add in stutter-1's slow-frame samples).
struct ImportCountTag {};
TlCounts<1, ImportCountTag> g_import_counts;
uint64_t import_calls_sum() { return g_import_counts.total(); }
void import_calls_zero() { g_import_counts.clear(); }
std::mutex g_type_lock;                               // resolved imported typeinfo slots (ABI 3 NxEh.type_syms), keyed by slot address
std::unordered_map<const uint64_t*, uint64_t> g_type_cache;

// ---- gauges (include/nxrt/gauge.h) ----
// Removes its gauge when this file's statics are torn down (declared after them, so it goes first).
struct GaugeHook {
    const char* name;
    GaugeHook(const char* n, std::function<int64_t()> f) : name(n) { nxrt::gauge_register(n, std::move(f)); }
    ~GaugeHook() { nxrt::gauge_unregister(name); }
};
GaugeHook gauge_nx_modules("core.nx_modules", [] { return static_cast<int64_t>(g_count.load(std::memory_order_acquire)); });
GaugeHook gauge_import_calls("core.import_calls", [] { return static_cast<int64_t>(import_calls_sum()); });
GaugeHook gauge_type_cache("core.type_cache", [] {   // resolved typeinfo slots (only grows)
    std::lock_guard<std::mutex> lk(g_type_lock);
    return static_cast<int64_t>(g_type_cache.size());
});

Entry* by_index(uint32_t index) { return index < kMaxModules ? g_by_index[index].load(std::memory_order_acquire) : nullptr; }

void invalidate_caches_locked() {
    for (size_t k = 0; k < g_count.load(); k++) {
        Entry* e = g_list[k].load();
        for (size_t i = 0; i < e->ncache; i++) e->cache[i].store(0, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> lk(g_type_lock);
    g_type_cache.clear();
}

RegResult refuse(RegStatus st, const NxModule* nx, const char* fmt, ...) {
    char buf[384];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    RegResult r;
    r.status = st;
    r.message = buf;
    NXLOG_E("loader", "lifted module %s refused: %s", nx && nx->name ? nx->name : "(null)", buf);
    return r;
}

// Cheap structural checks of a descriptor (the arrays' lengths are only known for fns, syms and resume).
const char* validate(const NxModule* nx) {
    if (!nx->name) return "null name";
    if (nx->nfns < 0 || nx->nsyms < 0 || nx->nresume < 0) return "negative table length";
    if ((nx->nfns && !nx->fns) || (nx->nsyms && !nx->syms) || (nx->nresume && !nx->resume)) return "null table with a non-zero length";
    if (nx->text_end < nx->base) return "text_end below base";
    if (nx->index >= kMaxModules) return "module index does not fit an import id (12 bits)";
    for (int i = 0; i < nx->nfns; i++) {
        if (!nx->fns[i].fn) return "null function pointer in the function table";
        if (nx->fns[i].addr - nx->base >= nx->text_end - nx->base) return "function address outside [base, text_end)";
        if (i && nx->fns[i - 1].addr >= nx->fns[i].addr) return "function table not strictly sorted by address";
    }
    for (int i = 1; i < nx->nresume; i++)
        if (nx->resume[i - 1].pad > nx->resume[i].pad) return "resume table not sorted by pad";
    if (nx->eh)
        for (int i = 1; i < nx->eh->nfdes; i++)
            if (nx->eh->fdes[i - 1].start > nx->eh->fdes[i].start) return "FDE table not sorted";
    return nullptr;
}

uint64_t find_export(const char* name, const NxModule* except) {
    for (size_t k = 0, n = g_count.load(std::memory_order_acquire); k < n; k++) {
        Entry* e = g_list[k].load(std::memory_order_acquire);
        if (e->nx == except) continue;
        for (int i = 0; i < e->nx->nsyms; i++) {
            const DynSym& s = e->nx->syms[i];
            if (s.defined && s.name && std::strcmp(s.name, name) == 0) return nxmodule_symbol_address(e->nx, i);
        }
    }
    return 0;
}

}  // namespace

RegResult nxmodule_register(Module* image, const NxModule* nx, uint32_t abi_version) {
    if (!nx) return refuse(RegStatus::Invalid, nx, "null descriptor");
    if (abi_version != kAbiVersion)
        return refuse(RegStatus::AbiMismatch, nx, "lifted against ABI version %u, this runtime implements version %u (re-lift the module)", abi_version, kAbiVersion);
    if (const char* why = validate(nx)) return refuse(RegStatus::Invalid, nx, "%s", why);

    std::lock_guard<std::mutex> lk(g_lock);
    if (by_index(nx->index)) return refuse(RegStatus::DuplicateIndex, nx, "module index %u is already registered (%s)", nx->index, g_by_index[nx->index].load()->nx->name);
    for (size_t k = 0; k < g_count.load(); k++) {
        const NxModule* o = g_list[k].load()->nx;
        if (nx->base < o->text_end && o->base < nx->text_end)
            return refuse(RegStatus::DuplicateRange, nx, "code range 0x%llx..0x%llx overlaps module %s", static_cast<unsigned long long>(nx->base),
                          static_cast<unsigned long long>(nx->text_end), o->name);
    }
    if (g_count.load() >= kMaxListed) return refuse(RegStatus::Invalid, nx, "more than %zu modules", kMaxListed);

    RegResult r;
    if (image) {
        r.functions = module_register_functions(image, nx->fns, static_cast<size_t>(nx->nfns));
    } else {
        for (int i = 0; i < nx->nfns; i++) {
            fn_register(nx->fns[i].addr, nx->fns[i].fn);
            r.functions++;
        }
    }
    auto e = std::make_unique<Entry>();
    e->nx = nx;
    e->image = image;
    e->ncache = static_cast<size_t>(nx->nsyms);
    e->cache = std::make_unique<std::atomic<uint64_t>[]>(e->ncache ? e->ncache : 1);
    g_by_index[nx->index].store(e.get(), std::memory_order_release);
    size_t n = g_count.load();
    g_list[n].store(e.get(), std::memory_order_release);
    g_owned[n] = std::move(e);
    g_count.store(n + 1, std::memory_order_release);
    invalidate_caches_locked();
    NXLOG_I("loader", "registered lifted module %s (index %u): %zu of %d functions, %d dynamic symbols, %d resume pads, unwind tables %s", nx->name, nx->index,
            r.functions, nx->nfns, nx->nsyms, nx->nresume, nx->eh ? "yes" : "no");
    return r;
}

size_t module_register_nxmodule(Module* m, const NxModule* nx) { return nxmodule_register(m, nx, kAbiVersion).functions; }

const NxModule* nxmodule_by_index(uint32_t index) {
    Entry* e = by_index(index);
    return e ? e->nx : nullptr;
}

const NxModule* nxmodule_by_name(const char* name) {
    for (size_t k = 0, n = g_count.load(std::memory_order_acquire); k < n; k++) {
        const NxModule* nx = g_list[k].load(std::memory_order_acquire)->nx;
        if (nx->name && std::strcmp(nx->name, name) == 0) return nx;
    }
    return nullptr;
}

const NxModule* nxmodule_at(uint64_t addr) {
    for (size_t k = 0, n = g_count.load(std::memory_order_acquire); k < n; k++) {
        const NxModule* nx = g_list[k].load(std::memory_order_acquire)->nx;
        if (addr - nx->base < nx->text_end - nx->base) return nx;
    }
    return nullptr;
}

std::vector<const NxModule*> nxmodules() {
    std::vector<const NxModule*> v;
    for (size_t k = 0, n = g_count.load(std::memory_order_acquire); k < n; k++) v.push_back(g_list[k].load(std::memory_order_acquire)->nx);
    return v;
}

void nxmodules_clear() {
    std::lock_guard<std::mutex> lk(g_lock);
    for (auto& s : g_by_index) s.store(nullptr);
    g_count.store(0);
    for (size_t k = 0; k < kMaxListed; k++) {
        g_list[k].store(nullptr);
        g_owned[k].reset();
    }
    import_calls_zero();
    std::lock_guard<std::mutex> tk(g_type_lock);
    g_type_cache.clear();
}

uint64_t nxmodule_symbol_address(const NxModule* nx, int sym) {
    if (!nx || sym < 0 || sym >= nx->nsyms || !nx->syms[sym].defined) return 0;
    return nx->syms[sym].value;      // ABI 3: a guest address for every defined symbol (TLS offsets and SHN_ABS numbers stay raw)
}

bool nxmodule_type_address(const NxModule* nx, const NxFde* fde, uint32_t filter, uint64_t* out) {
    *out = 0;
    if (!nx || !nx->eh || !fde || filter == 0) return false;
    const NxEh* eh = nx->eh;
    const uint64_t* slot = eh->types + fde->type_first + filter - 1;
    if (*slot || !eh->type_syms) { *out = *slot; return true; }        // final address, or 0 = catch(...)
    const int32_t sym = eh->type_syms[fde->type_first + filter - 1];
    if (sym < 0) return true;                                          // 0 with no symbol: catch(...)
    if (sym >= nx->nsyms || !nx->syms[sym].name) return false;
    std::lock_guard<std::mutex> lk(g_type_lock);
    auto it = g_type_cache.find(slot);
    if (it != g_type_cache.end()) { *out = it->second; return it->second != 0; }
    uint64_t a = nxmodule_find_export(nx->syms[sym].name, nx);         // the typeinfo is an import: the module that defines it
    if (!a) NXLOG_W("loader", "typeinfo %s imported by %s is defined by no registered module: the clause matches nothing", nx->syms[sym].name, nx->name);
    g_type_cache.emplace(slot, a);
    *out = a;
    return a != 0;
}

uint64_t nxmodule_find_export(const char* name, const NxModule* except) { return find_export(name, except); }

const char* import_name(uint32_t import_id) {
    const NxModule* nx = nxmodule_by_index(NX_IMPORT_MODULE(import_id));
    uint32_t sym = NX_IMPORT_SYM(import_id);
    return nx && sym < static_cast<uint32_t>(nx->nsyms) && nx->syms[sym].name ? nx->syms[sym].name : "";
}

uint64_t import_resolve_named(uint32_t import_id, const char** name_out) {
    Entry* e = by_index(NX_IMPORT_MODULE(import_id));
    const uint32_t sym = NX_IMPORT_SYM(import_id);
    if (e && sym < e->ncache) {
        const char* name = e->nx->syms[sym].name;
        if (name) {
            if (const uint64_t hit = e->cache[sym].load(std::memory_order_acquire)) {
                *name_out = name;
                return hit;
            }
        }
    }
    *name_out = import_name(import_id);
    return import_resolve(import_id);
}

uint64_t import_resolve(uint32_t import_id) {
    Entry* e = by_index(NX_IMPORT_MODULE(import_id));
    uint32_t sym = NX_IMPORT_SYM(import_id);
    if (!e || sym >= e->ncache || !e->nx->syms[sym].name) return 0;
    uint64_t hit = e->cache[sym].load(std::memory_order_acquire);
    if (hit) return hit;
    const char* name = e->nx->syms[sym].name;
    uint64_t target = 0;
    if (module_symbol_overridden(name)) {
        target = hle_address(name);                                    // the host replaces this symbol everywhere
    } else {
        target = nxmodule_find_export(name, e->nx);                    // a lifted function of another module
        if (!target) {
            uint64_t a = module_find_symbol(name);                     // an ELF symbol of another mapped image that is a registered function
            if (a && fn_lookup(a)) target = a;
        }
        if (!target) target = hle_address(name);                       // HLE handler, else a logging stub
    }
    e->cache[sym].store(target, std::memory_order_release);
    return target;
}

void import_cache_invalidate() {
    std::lock_guard<std::mutex> lk(g_lock);
    invalidate_caches_locked();
}

uint64_t import_calls() { return import_calls_sum(); }

namespace detail {
void count_import() { g_import_counts.add(0); }
}  // namespace detail

}  // namespace nxrt
