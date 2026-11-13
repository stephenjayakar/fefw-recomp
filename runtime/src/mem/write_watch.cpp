// SPDX-License-Identifier: GPL-3.0-or-later
// Page write tracking by page protection (include/nxrt/write_watch.h). Four bitmaps per region, one bit per 4 KiB page:
//   prot    the page is read-only because of us (set by arm, cleared by whoever makes it writable again)
//   valid   armed and not written since (the owner's copy of the page is current)
//   written the CPU wrote it since the owner last asked (take_written)
//   ban     never arm (another view of the page exists)
// The fault handler and before_os_write may run on any thread at any time; they only use atomics, VirtualProtect and the region's
// slim reader/writer lock: whoever makes pages writable holds it shared, arm() and region_destroy() hold it exclusive, so a page's bit
// and its protection change together as seen by everybody else (no page ends up read-only without its bit, or armed and writable). Regions are
// published in stable, dynamically appended slot blocks. Readers count themselves in g_busy while looking at a region;
// region_destroy takes the region out of the table, then waits for that count to drop.
#include "nxrt/write_watch.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <unordered_map>
#include <vector>

#include "nxrt/config.h"
#include "nxrt/gauge.h"
#include "nxrt/log.h"
#include "nxrt/memory.h"

namespace nxrt::mem::ww {

struct Region {
    const char* owner = "unspecified"; // static lifetime name supplied by owner
    uint64_t base = 0;          // address of page 0
    uint64_t npages = 0;
    uint64_t nwords = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> prot, valid, written, ban;
    std::unique_ptr<std::atomic<uint64_t>[]> guard;     // PAGE_NOACCESS because of guard() (a subset of prot)
    std::atomic<Region*>* slot = nullptr;
    SRWLOCK lock = SRWLOCK_INIT;
    std::unique_ptr<std::atomic<uint16_t>[]> releasing;   // per page: releasers that saw it armed and have not finished its VirtualProtect (release_page)
};

// Protection belongs to the physical page, not a Region. Arming must exclude
// transitions in every overlapping region, including destruction and unguarding.
// Lock order: this lock, then Region::lock. Guard callbacks run outside both.
SRWLOCK g_transitions = SRWLOCK_INIT;
struct SharedLock {
    explicit SharedLock(Region* r) : r_(r) { AcquireSRWLockShared(&g_transitions); AcquireSRWLockShared(&r_->lock); }
    ~SharedLock() { ReleaseSRWLockShared(&r_->lock); ReleaseSRWLockShared(&g_transitions); }
    Region* r_;
};
struct ExclusiveLock {
    explicit ExclusiveLock(Region* r, bool arming = false) : r_(r), arming_(arming) {
        if (arming_) AcquireSRWLockExclusive(&g_transitions);
        else AcquireSRWLockShared(&g_transitions);
        AcquireSRWLockExclusive(&r_->lock);
    }
    ~ExclusiveLock() {
        ReleaseSRWLockExclusive(&r_->lock);
        if (arming_) ReleaseSRWLockExclusive(&g_transitions);
        else ReleaseSRWLockShared(&g_transitions);
    }
    Region* r_;
    bool arming_;
};

namespace {

constexpr const char* kTag = "mem.ww";
// Blocks never move or disappear while fault handlers can run. Growth allocates only
// on region_create, under g_mu; readers need neither allocation nor a table lock.
// Empty slots are reused, bounding table storage by peak live regions (2 KiB/block).
constexpr int kSlotsPerBlock = 256;
struct SlotBlock {
    std::atomic<Region*> slots[kSlotsPerBlock]{};
    std::atomic<int> hi{0};
    std::atomic<SlotBlock*> next{nullptr};
};
SlotBlock g_slots;

// Caller participates in g_busy until done using the returned Region pointers.
struct RegionCursor {
    SlotBlock* block = &g_slots;
    int index = 0;
    int hi = block->hi.load(std::memory_order_seq_cst);
    Region* next() {
        while (block) {
            while (index < hi)
                if (Region* r = block->slots[index++].load(std::memory_order_seq_cst)) return r;
            block = block->next.load(std::memory_order_acquire);
            index = 0;
            if (block) hi = block->hi.load(std::memory_order_seq_cst);
        }
        return nullptr;
    }
};
std::atomic<int> g_busy{0};            // fault handlers and os-write calls looking at the table
std::atomic<int> g_live{0};
std::atomic<uint64_t> g_generation{0};
std::atomic<uint64_t> g_faults{0};
std::atomic<uint64_t> g_waits{0};      // rounds a fault waited for another releaser (statistics, tests)
std::atomic<uint64_t> g_guard_faults{0};   // accesses that hit a guarded page
std::atomic<GuardHandler> g_guard_fn{nullptr};
std::mutex g_mu;                       // create / destroy
PVOID g_veh = nullptr;

// Operating-system writes in progress (OsWriteScope): arm() leaves these ranges alone, so a page cannot become read-only between the
// release in before_os_write and the end of the ReadFile that fills it. When every slot is taken, g_os_overflow stops all arming.
constexpr int kOsSlots = 32;
struct OsRange { std::atomic<uint64_t> lo{0}, hi{0}; };
OsRange g_os[kOsSlots];
std::atomic<int> g_os_overflow{0};

bool os_write_overlaps(uint64_t lo, uint64_t hi) {
    if (g_os_overflow.load(std::memory_order_seq_cst)) return true;
    for (OsRange& o : g_os) {
        const uint64_t h = o.hi.load(std::memory_order_seq_cst);
        if (h && o.lo.load(std::memory_order_seq_cst) < hi && lo < h) return true;
    }
    return false;
}

// Import freezes (ImportFreeze): ranges whose CPU writes wait for a Vulkan import of the same pages. g_scope_mu orders them against
// OsWriteScope: a file read never starts in a frozen range, and a freeze never starts over a file read in progress.
constexpr int kFreezeSlots = 16;
OsRange g_frz[kFreezeSlots];
std::atomic<int> g_frz_n{0};
std::mutex g_scope_mu;

std::atomic<uint64_t> g_freeze_table_hits{0};   // ImportFreeze constructions that found their runs from the page table (no VirtualQuery)

bool freeze_overlaps(uint64_t lo, uint64_t hi) {
    if (!g_frz_n.load(std::memory_order_seq_cst)) return false;
    for (OsRange& f : g_frz) {
        const uint64_t h = f.hi.load(std::memory_order_seq_cst);
        if (h && f.lo.load(std::memory_order_seq_cst) < hi && lo < h) return true;
    }
    return false;
}

uint64_t bit(uint64_t page) { return uint64_t{1} << (page & 63); }

bool test(const std::atomic<uint64_t>* bm, uint64_t page) { return (bm[page >> 6].load(std::memory_order_acquire) & bit(page)) != 0; }

// Sets or clears the bits of pages [first, first + count) word by word.
void set_bits(std::atomic<uint64_t>* bm, uint64_t first, uint64_t count, bool on) {
    uint64_t p = first;
    const uint64_t end = first + count;
    while (p < end) {
        const uint64_t w = p >> 6;
        const uint64_t lo = p & 63;
        const uint64_t hi = std::min<uint64_t>(64, lo + (end - p));
        const uint64_t mask = (hi - lo == 64) ? ~uint64_t{0} : (((uint64_t{1} << (hi - lo)) - 1) << lo);
        if (on) bm[w].fetch_or(mask, std::memory_order_acq_rel);
        else bm[w].fetch_and(~mask, std::memory_order_acq_rel);
        p += hi - lo;
    }
}

bool all_set(const std::atomic<uint64_t>* bm, uint64_t first, uint64_t count) {
    uint64_t p = first;
    const uint64_t end = first + count;
    while (p < end) {
        const uint64_t w = p >> 6;
        const uint64_t lo = p & 63;
        const uint64_t hi = std::min<uint64_t>(64, lo + (end - p));
        const uint64_t mask = (hi - lo == 64) ? ~uint64_t{0} : (((uint64_t{1} << (hi - lo)) - 1) << lo);
        if ((bm[w].load(std::memory_order_acquire) & mask) != mask) return false;
        p += hi - lo;
    }
    return true;
}

bool any_set(const std::atomic<uint64_t>* bm, uint64_t first, uint64_t count) {
    uint64_t p = first;
    const uint64_t end = first + count;
    while (p < end) {
        const uint64_t w = p >> 6;
        const uint64_t lo = p & 63;
        const uint64_t hi = std::min<uint64_t>(64, lo + (end - p));
        const uint64_t mask = (hi - lo == 64) ? ~uint64_t{0} : (((uint64_t{1} << (hi - lo)) - 1) << lo);
        if (bm[w].load(std::memory_order_acquire) & mask) return true;
        p += hi - lo;
    }
    return false;
}

DWORD page_protect(uint64_t addr) {
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(reinterpret_cast<void*>(addr), &mbi, sizeof mbi) != sizeof mbi) return 0;
    if (mbi.State != MEM_COMMIT) return 0;
    return mbi.Protect;
}

// A fault can be delivered after its last owner has gone out of the table.
// Retain only a bit per retired heap page (384 KiB for the maximum heap), never
// Region pointers. Retry only if the *current* protection permits the access;
// foreign read-only/no-access/decommitted pages still pass to the crash handler.
std::atomic<uint64_t> g_retired[kHeapMax / kPage / 64]{};
bool retired_retry(uint64_t addr, ULONG_PTR kind) {
    if (addr < kHeapBase || addr - kHeapBase >= kHeapMax || !test(g_retired, (addr - kHeapBase) / kPage)) return false;
    const DWORD pr = page_protect(addr);
    if (pr & PAGE_GUARD) return false;
    const DWORD p = pr & 0xff;
    if (p == PAGE_READWRITE || p == PAGE_EXECUTE_READWRITE) return true;
    return kind == 0 && (p == PAGE_READONLY || p == PAGE_EXECUTE_READ);
}

// A final rejection must not sample a replacement arm halfway through its
// protection change (its generation is published before releasing this lock).
int final_retry(uint64_t addr, ULONG_PTR kind, uint64_t generation) {
    AcquireSRWLockShared(&g_transitions);
    const bool accessible = retired_retry(addr, kind);
    const bool changed = g_generation.load(std::memory_order_seq_cst) != generation;
    ReleaseSRWLockShared(&g_transitions);
    return accessible ? 1 : changed ? -1 : 0;
}

// One page of a region becomes writable and counts as written. True when the page is ours to handle (the faulting write may retry).
// The caller holds the region's lock shared.
bool release_page(Region* r, uint64_t page) {
    const uint64_t b = bit(page);
    const uint64_t w = page >> 6;
    // Announce the release before the bit goes: whoever finds the bit already clear can then tell "somebody is making this page writable"
    // (the count is up from before the bit went until after the VirtualProtect) from "nobody is": a foreign read-only page.
    std::atomic<uint16_t>& rel = r->releasing[page];
    const bool armed = (r->prot[w].load(std::memory_order_acquire) & b) != 0;
    if (armed) rel.fetch_add(1, std::memory_order_seq_cst);
    const uint64_t old = r->prot[w].fetch_and(~b, std::memory_order_acq_rel);
    r->valid[w].fetch_and(~b, std::memory_order_acq_rel);
    r->written[w].fetch_or(b, std::memory_order_acq_rel);
    const uint64_t addr = r->base + page * kPage;
    if (old & b) {
        bool ok = true;
        DWORD was = 0;
        if (!VirtualProtect(reinterpret_cast<void*>(addr), kPage, PAGE_READWRITE, &was)) {
            ok = false;
        } else if ((was & 0xff) != PAGE_READONLY) {    // somebody else set this protection: give it back and let the fault through
            DWORD tmp = 0;
            VirtualProtect(reinterpret_cast<void*>(addr), kPage, was, &tmp);
            ok = false;
        }
        rel.fetch_sub(1, std::memory_order_seq_cst);
        return ok;
    }
    if (armed) rel.fetch_sub(1, std::memory_order_seq_cst);   // lost the race for the bit: not a releaser (and must not make others wait for it)
    // Not armed (any more): another releaser (also holding the lock shared) may be making it writable right now. Wait for it only while
    // such a release is in progress; with none, a read-only page is somebody else's and the fault goes on at once. The count is read before
    // the protection: a releaser that cleared the bit before our fetch_and has counted itself in before that, so a zero count means every
    // such release has finished and its effect is visible in the protection we read next.
    // The wait has no round limit: a releaser that is counted in will finish (it only has its VirtualProtect left), but it may be descheduled
    // for a long time on a busy machine, and giving up early would pass a fault on that is ours (seen as an unhandled access violation).
    for (int i = 0;; ++i) {
        const uint16_t busy = rel.load(std::memory_order_seq_cst);
        const DWORD pr = page_protect(addr) & 0xff;
        if (pr == PAGE_READWRITE || pr == PAGE_EXECUTE_READWRITE) return true;
        if (pr != PAGE_READONLY || busy == 0) return false;
        g_waits.fetch_add(1, std::memory_order_relaxed);
        if (i < 50) YieldProcessor();
        else if (i < 1000) Sleep(0);
        else Sleep(1);
    }
}

// [diag] ww_stats (env NXRT_DIAG_WW_STATS=1): faults per region and per 64 KiB block, the 10 busiest blocks logged every 10 s.
std::mutex g_fs_lock;
std::map<std::pair<uint64_t, uint64_t>, uint64_t> g_fs_blocks;   // (region base, 64 KiB block address) -> faults
std::unordered_map<uint64_t, uint64_t> g_fs_regions;     // region base -> faults
ULONGLONG g_fs_since = 0;
void fault_stat(const Region* r, uint64_t addr) {
    static const bool on = config_get_bool("diag", "ww_stats", false);
    if (!on) return;
    std::lock_guard<std::mutex> lk(g_fs_lock);
    g_fs_blocks[{r->base, addr & ~0xffffull}]++;
    g_fs_regions[r->base]++;
    const ULONGLONG now = GetTickCount64();
    if (!g_fs_since) g_fs_since = now;
    if (now - g_fs_since < 10000) return;
    const double secs = static_cast<double>(now - g_fs_since) / 1000.0;
    std::vector<std::pair<std::pair<uint64_t, uint64_t>, uint64_t>> v(g_fs_blocks.begin(), g_fs_blocks.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    for (const auto& [base, n] : g_fs_regions) NXLOG_I("ww", "faults over %.1f s: region %llx: %.0f/s", secs, static_cast<unsigned long long>(base), n / secs);
    for (size_t i = 0; i < v.size() && i < 10; ++i)
        NXLOG_I("ww", "  region %llx block %llx: %.0f faults/s", static_cast<unsigned long long>(v[i].first.first), static_cast<unsigned long long>(v[i].first.second),
                v[i].second / secs);
    g_fs_blocks.clear();
    g_fs_regions.clear();
    g_fs_since = now;
}

// The guarded pages among [first, first + count) become read-only (armed) again. The caller holds the region's lock exclusive.
void unguard_locked(Region* r, uint64_t first, uint64_t count) {
    for (uint64_t p = first; p < first + count;) {
        if (!test(r->guard.get(), p)) { ++p; continue; }
        uint64_t q = p;
        while (q < first + count && test(r->guard.get(), q)) ++q;
        // protection first, then the bit: a fault that finds the bit clear must find the page accessible (the read path takes no lock)
        DWORD was = 0;
        VirtualProtect(reinterpret_cast<void*>(r->base + p * kPage), static_cast<SIZE_T>((q - p) * kPage), PAGE_READONLY, &was);
        set_bits(r->guard.get(), p, q - p, false);
        p = q;
    }
}

bool guarded_at(uint64_t addr) {
    bool any = false;
    g_busy.fetch_add(1, std::memory_order_seq_cst);
    for (RegionCursor cursor; Region* r = cursor.next();) {
        if (any) break;
        if (addr < r->base || addr >= r->base + r->npages * kPage) continue;
        any = test(r->guard.get(), (addr - r->base) / kPage);
    }
    g_busy.fetch_sub(1, std::memory_order_seq_cst);
    return any;
}

void drop_guard(uint64_t addr, uint64_t size);   // below

LONG CALLBACK on_fault(EXCEPTION_POINTERS* ep) {
    if (!ep || !ep->ExceptionRecord) return EXCEPTION_CONTINUE_SEARCH;
    const EXCEPTION_RECORD* rec = ep->ExceptionRecord;
    if (rec->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || rec->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    const ULONG_PTR kind = rec->ExceptionInformation[0];     // 0 read, 1 write, 8 execute
    if (kind != 0 && kind != 1) return EXCEPTION_CONTINUE_SEARCH;
    const uint64_t addr = static_cast<uint64_t>(rec->ExceptionInformation[1]);
    // A write into a range an import has frozen (ImportFreeze): wait for the import, then let the write run again (the page is
    // read-write again, or armed and handled below on the next fault).
    if (kind == 1 && freeze_overlaps(addr, addr + 1)) {
        for (int i = 0; freeze_overlaps(addr, addr + 1); ++i) {
            if (i < 64) YieldProcessor();
            else Sleep(i < 1000 ? 0 : 1);
        }
        return EXCEPTION_CONTINUE_EXECUTION;
    }
retry:
    const uint64_t generation = g_generation.load(std::memory_order_seq_cst);
    nxrt::HitchTimer ht(nxrt::kHpFaults);      // the hitch line: time in this handler (a wait for arm's exclusive lock included)
    // A guarded page (read or write): the owner makes it current first (outside every lock: it may wait for the GPU). A write then faults
    // again on the read-only page and is tracked as usual.
    if (guarded_at(addr)) {
        g_guard_faults.fetch_add(1, std::memory_order_relaxed);
        if (GuardHandler fn = g_guard_fn.load(std::memory_order_acquire)) fn(addr & ~(kPage - 1));
        drop_guard(addr & ~(kPage - 1), kPage);   // still guarded (no handler, or it found nothing): read-only again, as it is
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (kind != 1) {
        // a guard dropped by another thread since the access: the page is readable again, let the read retry
        const DWORD pr = page_protect(addr) & 0xff;
        if (pr == PAGE_READONLY || pr == PAGE_READWRITE || pr == PAGE_EXECUTE_READ || pr == PAGE_EXECUTE_READWRITE) {
            bool in_region = false;
            g_busy.fetch_add(1, std::memory_order_seq_cst);
            for (RegionCursor cursor; Region* r = cursor.next();) {
                if (addr >= r->base && addr < r->base + r->npages * kPage) { in_region = true; break; }
            }
            g_busy.fetch_sub(1, std::memory_order_seq_cst);
            if (in_region) return EXCEPTION_CONTINUE_EXECUTION;
        }
        if (pr == PAGE_NOACCESS && guarded_at(addr)) {    // dropped and guarded again since the check above
            g_guard_faults.fetch_add(1, std::memory_order_relaxed);
            if (GuardHandler fn = g_guard_fn.load(std::memory_order_acquire)) fn(addr & ~(kPage - 1));
            drop_guard(addr & ~(kPage - 1), kPage);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        const int decision = final_retry(addr, kind, generation);
        if (decision == 1) return EXCEPTION_CONTINUE_EXECUTION;
        if (decision == -1) goto retry;
        return EXCEPTION_CONTINUE_SEARCH;
    }
    bool ours = false, guarded_now = false;
    g_busy.fetch_add(1, std::memory_order_seq_cst);
    // Pass 0 releases the page in the regions that armed it (usually one). Only when none did (another thread released it meanwhile) does
    // pass 1 ask release_page's slow path in every region: a VirtualQuery over the heap view (milliseconds), which the whole-heap texture
    // region paid for every fault on a pool copy's page, under its shared lock (notes/pacing.md).
    for (int pass = 0; pass < 2 && !ours && !guarded_now; ++pass) {
        for (RegionCursor cursor; Region* r = cursor.next();) {
            if (addr < r->base || addr >= r->base + r->npages * kPage) continue;
            SharedLock lk(r);
            const uint64_t pg = (addr - r->base) / kPage;
            // guard() (exclusive lock) may have guarded the page since guarded_at above: under the shared lock the bit is final. Releasing a
            // guarded page cleared its armed bit and left it no-access with nobody to claim the fault (a crash in Dagsion, notes/pacing.md).
            if (test(r->guard.get(), pg)) { guarded_now = true; continue; }
            if (pass == 0 && !test(r->prot.get(), pg)) continue;
            const bool mine = release_page(r, pg);
            if (mine) fault_stat(r, addr);
            ours |= mine;
        }
    }
    if (guarded_now) {     // the guard fault as above; the write then faults again on the armed page and is released as usual
        g_busy.fetch_sub(1, std::memory_order_seq_cst);
        g_guard_faults.fetch_add(1, std::memory_order_relaxed);
        if (GuardHandler fn = g_guard_fn.load(std::memory_order_acquire)) fn(addr & ~(kPage - 1));
        drop_guard(addr & ~(kPage - 1), kPage);
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (!ours) {
        const int decision = final_retry(addr, kind, generation);
        ours = decision == 1;
        if (decision == -1) {
            g_busy.fetch_sub(1, std::memory_order_seq_cst);
            goto retry; // a replacement region may now own the protection
        }
    }
    if (!ours) {   // a write fault in a tracked region that no region claims: what each one holds for the page (the crash report follows)
        const DWORD pr = page_protect(addr);
        for (RegionCursor cursor; Region* r = cursor.next();) {
            if (addr < r->base || addr >= r->base + r->npages * kPage) continue;
            const uint64_t pg = (addr - r->base) / kPage;
            NXLOG_E(kTag, "unclaimed write fault at %llx (protection %lx): region %llx (+%llu pages) prot %d valid %d guard %d ban %d releasing %d",
                    static_cast<unsigned long long>(addr), static_cast<unsigned long>(pr), static_cast<unsigned long long>(r->base),
                    static_cast<unsigned long long>(r->npages), test(r->prot.get(), pg), test(r->valid.get(), pg), test(r->guard.get(), pg),
                    test(r->ban.get(), pg), static_cast<int>(r->releasing[pg].load()));
        }
    }
    g_busy.fetch_sub(1, std::memory_order_seq_cst);
    if (!ours) return EXCEPTION_CONTINUE_SEARCH;
    g_faults.fetch_add(1, std::memory_order_relaxed);
    return EXCEPTION_CONTINUE_EXECUTION;
}

// Calls fn(region, first page, count) for the part of [addr, addr + size) inside every published region.
template <class Fn>
void for_regions(uint64_t addr, uint64_t size, Fn&& fn) {
    if (g_live.load(std::memory_order_acquire) == 0 || size == 0) return;
    g_busy.fetch_add(1, std::memory_order_seq_cst);
    for (RegionCursor cursor; Region* r = cursor.next();) {
        const uint64_t lo = std::max(addr, r->base), end = std::min(addr + size, r->base + r->npages * kPage);
        if (lo >= end) continue;
        const uint64_t first = (lo - r->base) / kPage, last = (end - 1 - r->base) / kPage;
        SharedLock lk(r);
        fn(r, first, last - first + 1);
    }
    g_busy.fetch_sub(1, std::memory_order_seq_cst);
}

void drop_guard(uint64_t addr, uint64_t size) {
    if (g_live.load(std::memory_order_acquire) == 0 || size == 0) return;
    g_busy.fetch_add(1, std::memory_order_seq_cst);
    for (RegionCursor cursor; Region* r = cursor.next();) {
        const uint64_t lo = std::max(addr, r->base), end = std::min(addr + size, r->base + r->npages * kPage);
        if (lo >= end) continue;
        const uint64_t first = (lo - r->base) / kPage, last = (end - 1 - r->base) / kPage;
        if (!any_set(r->guard.get(), first, last - first + 1)) continue;
        ExclusiveLock lk(r);
        unguard_locked(r, first, last - first + 1);
    }
    g_busy.fetch_sub(1, std::memory_order_seq_cst);
}

struct Gauges {
    Gauges() {
        nxrt::gauge_register("mem.ww.faults", [] { return static_cast<int64_t>(g_faults.load(std::memory_order_relaxed)); });
        nxrt::gauge_register("mem.ww.regions", [] { return static_cast<int64_t>(g_live.load(std::memory_order_relaxed)); });
    }
    ~Gauges() {
        nxrt::gauge_unregister("mem.ww.faults");
        nxrt::gauge_unregister("mem.ww.regions");
    }
};
Gauges g_gauges;

}  // namespace

Region* region_create(uint64_t addr, uint64_t size, const char* owner) {
    if (!size || !heap_exists()) return nullptr;
    const uint64_t start = addr & ~(kPage - 1), end = (addr + size + kPage - 1) & ~(kPage - 1);
    if (start < kHeapBase || end > kHeapBase + heap_reserved()) return nullptr;
    auto r = std::make_unique<Region>();
    r->owner = owner ? owner : "unspecified";
    r->base = start;
    r->npages = (end - start) / kPage;
    r->nwords = (r->npages + 63) / 64;
    for (auto* bm : {&r->prot, &r->valid, &r->written, &r->ban, &r->guard}) {
        bm->reset(new std::atomic<uint64_t>[r->nwords]);
        for (uint64_t i = 0; i < r->nwords; ++i) (*bm)[i].store(0, std::memory_order_relaxed);
    }
    r->releasing.reset(new std::atomic<uint16_t>[r->npages]);
    for (uint64_t i = 0; i < r->npages; ++i) r->releasing[i].store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_veh) g_veh = AddVectoredExceptionHandler(1, on_fault);
        if (!g_veh) {
            NXLOG_W(kTag, "AddVectoredExceptionHandler failed: no write tracking");
            return nullptr;
        }
        SlotBlock* block = &g_slots;
        for (;;) {
            int slot = 0;
            while (slot < kSlotsPerBlock && block->slots[slot].load(std::memory_order_seq_cst)) ++slot;
            if (slot < kSlotsPerBlock) {
                r->slot = &block->slots[slot];
                r->slot->store(r.get(), std::memory_order_seq_cst);
                if (slot >= block->hi.load()) block->hi.store(slot + 1, std::memory_order_seq_cst);
                break;
            }
            SlotBlock* next = block->next.load(std::memory_order_acquire);
            if (!next) {
                next = new SlotBlock;
                block->next.store(next, std::memory_order_release);
            }
            block = next;
        }
        g_live.fetch_add(1, std::memory_order_seq_cst);
        g_generation.fetch_add(1, std::memory_order_seq_cst);
    }
    // Pages that another view shows can never be tracked. Published first, so an alias made from now on bans them through ban_section.
    std::vector<std::pair<uint64_t, uint64_t>> views;
    heap_alias_ranges(views);
    for (const auto& v : views) {
        const uint64_t lo = std::max(start, kHeapBase + v.first), hi = std::min(end, kHeapBase + v.first + v.second);
        if (lo < hi) set_bits(r->ban.get(), (lo - start) / kPage, (hi - lo) / kPage, true);
    }
    return r.release();
}

void region_destroy(Region* r) {
    if (!r) return;
    // Writable again first (faults on armed pages wait for the lock and then find the page writable), then out of the table.
    {
    ExclusiveLock lk(r);
    // Every page this region protects: armed (read-only) or guarded (no access). A guarded page can have lost its armed bit (release_page
    // on a page that guard() took again after the caller dropped its guard, as before_os_write does outside the lock): skipped here, it
    // stayed no-access after the region was gone, with nobody to claim the fault when the memory was used again.
    const auto held = [r](uint64_t i) { return test(r->prot.get(), i) || test(r->guard.get(), i); };
    for (uint64_t p = 0; p < r->npages;) {
        if (!held(p)) { ++p; continue; }
        uint64_t q = p;
        while (q < r->npages && held(q)) ++q;
        set_bits(g_retired, (r->base - kHeapBase) / kPage + p, q - p, true);
        set_bits(r->prot.get(), p, q - p, false);
        set_bits(r->guard.get(), p, q - p, false);
        DWORD was = 0;
        VirtualProtect(reinterpret_cast<void*>(r->base + p * kPage), static_cast<SIZE_T>((q - p) * kPage), PAGE_READWRITE, &was);
        p = q;
    }
    }
    {
        std::lock_guard<std::mutex> lk(g_mu);
        r->slot->store(nullptr, std::memory_order_seq_cst);
        g_live.fetch_sub(1, std::memory_order_seq_cst);
        g_generation.fetch_add(1, std::memory_order_seq_cst);
    }
    while (g_busy.load(std::memory_order_seq_cst) != 0) std::this_thread::yield();
    delete r;
}

std::string describe_page(uint64_t addr) {
    std::string out;
    g_busy.fetch_add(1, std::memory_order_seq_cst);
    unsigned matches = 0;
    for (RegionCursor cursor; Region* r = cursor.next();) {
        if (addr < r->base || addr - r->base >= r->npages * kPage) continue;
        const uint64_t pg = (addr - r->base) / kPage;
        char line[256];
        std::snprintf(line, sizeof line, "  write-watch owner=%s base=0x%llx pages=%llu page=%llu prot=%d valid=%d guard=%d ban=%d releasing=%u\n",
            r->owner, (unsigned long long)r->base, (unsigned long long)r->npages, (unsigned long long)pg,
            test(r->prot.get(), pg), test(r->valid.get(), pg), test(r->guard.get(), pg), test(r->ban.get(), pg),
            (unsigned)r->releasing[pg].load());
        out += line;
        ++matches;
    }
    g_busy.fetch_sub(1, std::memory_order_seq_cst);
    if (!matches) out += "  write-watch owner=none\n";
    if (addr >= kHeapBase && addr - kHeapBase < kHeapMax && test(g_retired, (addr - kHeapBase) / kPage))
        out += "  write-watch retired-page history=1\n";
    return out;
}

uint64_t region_base(const Region* r) { return r ? r->base : 0; }
uint64_t region_pages(const Region* r) { return r ? r->npages : 0; }

bool arm(Region* r, uint64_t first, uint64_t count) {
    if (!r || !count || first >= r->npages || count > r->npages - first) return false;
    nxrt::HitchTimer ht(nxrt::kHpWwArm);
    const uint64_t a = r->base + first * kPage, n = count * kPage;
    // The heap's page table says whether every page is committed read-write (memory.h heap_rw_pages). A VirtualQuery walk over the heap
    // view costs a walk of every page of the run of identical pages after the address (5 to 85 ms on a 10 GiB heap), under the exclusive
    // lock that the game's write faults wait for: the 70 ms frames of the hitch line's "write-watch arming" (notes/pacing.md). The table
    // is read before the lock (memory.cpp takes its lock first and region locks inside it); a change since shows in the epoch.
    const uint64_t epoch = heap_epoch();
    const int table = heap_rw_pages(a, n);
    ExclusiveLock lk(r, true);
    if (any_set(r->ban.get(), first, count)) return false;
    // A file read (or a mapping change in memory.cpp) into these pages is in progress: leave them alone. Its scope is published before it
    // releases pages under the shared lock, so either this sees it, or its release waits for this arm and undoes it.
    if (os_write_overlaps(a, a + n) || freeze_overlaps(a, a + n)) return false;
    if (table == 1 && heap_epoch() == epoch) {
        // what the walk below checks besides the table: no page read-guarded here, or read-only because another region armed it
        if (any_set(r->guard.get(), first, count)) return false;
        bool other = false;
        g_busy.fetch_add(1, std::memory_order_seq_cst);
        for (RegionCursor cursor; Region* o = cursor.next();) {
            if (other) break;
            if (o == r) continue;
            const uint64_t lo = std::max(a, o->base), end = std::min(a + n, o->base + o->npages * kPage);
            if (lo < end) other = any_set(o->prot.get(), (lo - o->base) / kPage, (end - lo) / kPage);
        }
        g_busy.fetch_sub(1, std::memory_order_seq_cst);
        if (other) return false;
    } else if (table == 0) {
        return false;
    } else {
        // every page committed read-write (the guest may have changed a protection; a page it made read-only is not ours to take)
        for (uint64_t cur = a; cur < a + n;) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (VirtualQuery(reinterpret_cast<void*>(cur), &mbi, sizeof mbi) != sizeof mbi || mbi.State != MEM_COMMIT) return false;
            const DWORD pr = mbi.Protect & 0xff;
            const bool already = pr == PAGE_READONLY && all_set(r->prot.get(), (cur - r->base) / kPage,
                                                                (std::min<uint64_t>(a + n, reinterpret_cast<uint64_t>(mbi.BaseAddress) + mbi.RegionSize) - cur) / kPage);
            if ((pr != PAGE_READWRITE && !already) || (mbi.Protect & PAGE_GUARD)) return false;
            cur = std::min<uint64_t>(a + n, reinterpret_cast<uint64_t>(mbi.BaseAddress) + mbi.RegionSize);
        }
    }
    // Under the exclusive lock no fault is handled until both bits and protection are in place; a write that faults meanwhile waits and
    // then releases the page. A write that lands before the protection does not fault, and the owner's copy (made later) still sees it.
    set_bits(r->prot.get(), first, count, true);
    set_bits(r->valid.get(), first, count, true);
    DWORD was = 0;
    if (!VirtualProtect(reinterpret_cast<void*>(a), static_cast<SIZE_T>(n), PAGE_READONLY, &was)) {
        set_bits(r->valid.get(), first, count, false);
        set_bits(r->prot.get(), first, count, false);
        return false;
    }
    g_generation.fetch_add(1, std::memory_order_seq_cst);
    return true;
}

bool all_valid(const Region* r, uint64_t first, uint64_t count) {
    if (!r || !count || first >= r->npages || count > r->npages - first) return false;
    return all_set(r->valid.get(), first, count);
}

bool page_valid(const Region* r, uint64_t page) { return r && page < r->npages && test(r->valid.get(), page); }
bool any_valid(const Region* r, uint64_t first, uint64_t count) {
    if (!r || !count || first >= r->npages) return false;
    const uint64_t end = first + std::min(count, r->npages - first);
    for (uint64_t p = first; p < end;) {
        const uint64_t lo = p & 63;
        const uint64_t hi = std::min<uint64_t>(64, lo + (end - p));
        const uint64_t mask = (hi - lo == 64) ? ~uint64_t{0} : (((uint64_t{1} << (hi - lo)) - 1) << lo);
        if (r->valid[p >> 6].load(std::memory_order_acquire) & mask) return true;
        p += hi - lo;
    }
    return false;
}
bool page_banned(const Region* r, uint64_t page) { return !r || page >= r->npages || test(r->ban.get(), page); }

void invalidate(Region* r, uint64_t first, uint64_t count) {
    if (!r || first >= r->npages) return;
    count = std::min(count, r->npages - first);
    if (count) set_bits(r->valid.get(), first, count, false);
}

void set_guard_handler(GuardHandler fn) { g_guard_fn.store(fn, std::memory_order_release); }

bool guard(Region* r, uint64_t first, uint64_t count) {
    if (!r || !count || first >= r->npages || count > r->npages - first) return false;
    ExclusiveLock lk(r);
    if (!all_set(r->prot.get(), first, count) || any_set(r->ban.get(), first, count)) return false;
    const uint64_t a = r->base + first * kPage, n = count * kPage;
    if (os_write_overlaps(a, a + n) || freeze_overlaps(a, a + n)) return false;
    set_bits(r->guard.get(), first, count, true);
    DWORD was = 0;
    if (!VirtualProtect(reinterpret_cast<void*>(a), static_cast<SIZE_T>(n), PAGE_NOACCESS, &was)) {
        VirtualProtect(reinterpret_cast<void*>(a), static_cast<SIZE_T>(n), PAGE_READONLY, &was);
        set_bits(r->guard.get(), first, count, false);
        return false;
    }
    return true;
}

void unguard(Region* r, uint64_t first, uint64_t count) {
    if (!r || first >= r->npages) return;
    count = std::min(count, r->npages - first);
    if (!count || !any_set(r->guard.get(), first, count)) return;
    ExclusiveLock lk(r);
    unguard_locked(r, first, count);
}

bool all_armed(const Region* r, uint64_t first, uint64_t count) {
    if (!r || !count || first >= r->npages || count > r->npages - first) return false;
    return all_set(r->prot.get(), first, count);
}

bool all_valid_or_armed(const Region* r, uint64_t first, uint64_t count, const std::atomic<uint64_t>* alt) {
    if (!r || !alt || !count || first >= r->npages || count > r->npages - first) return false;
    uint64_t p = first;
    const uint64_t end = first + count;
    while (p < end) {
        const uint64_t w = p >> 6;
        const uint64_t lo = p & 63;
        const uint64_t hi = std::min<uint64_t>(64, lo + (end - p));
        const uint64_t mask = (hi - lo == 64) ? ~uint64_t{0} : (((uint64_t{1} << (hi - lo)) - 1) << lo);
        const uint64_t ok = r->valid[w].load(std::memory_order_acquire) |
                            (r->prot[w].load(std::memory_order_acquire) & alt[w].load(std::memory_order_acquire));
        if ((ok & mask) != mask) return false;
        p += hi - lo;
    }
    return true;
}

uint64_t guard_faults() { return g_guard_faults.load(std::memory_order_relaxed); }

void take_written(Region* r, std::vector<uint64_t>& out) {
    if (!r) return;
    for (uint64_t w = 0; w < r->nwords; ++w) {
        if (!r->written[w].load(std::memory_order_relaxed)) continue;
        uint64_t v = r->written[w].exchange(0, std::memory_order_acq_rel);
        while (v) {
            const unsigned long b = static_cast<unsigned long>(__builtin_ctzll(v));
            v &= v - 1;
            const uint64_t page = w * 64 + b;
            if (page < r->npages) out.push_back(page);
        }
    }
}

namespace {
// An OS-call slot over [p, p + n) (OsWriteScope, OsReadScope): arm() and guard() leave the range alone while it is held.
int os_slot_acquire(const void* p, uint64_t n) {
    const uint64_t lo = reinterpret_cast<uint64_t>(p) & ~(kPage - 1), hi = (reinterpret_cast<uint64_t>(p) + n + kPage - 1) & ~(kPage - 1);
    for (int i = 0; i < kOsSlots; ++i) {
        uint64_t expect = 0;
        if (g_os[i].hi.load(std::memory_order_relaxed) == 0 && g_os[i].lo.compare_exchange_strong(expect, lo | 1, std::memory_order_seq_cst)) {
            g_os[i].lo.store(lo, std::memory_order_seq_cst);
            g_os[i].hi.store(hi, std::memory_order_seq_cst);
            return i;
        }
    }
    g_os_overflow.fetch_add(1, std::memory_order_seq_cst);
    return kOsSlots;
}

void os_slot_release(int slot) {
    if (slot < 0) return;
    if (slot == kOsSlots) {
        g_os_overflow.fetch_sub(1, std::memory_order_seq_cst);
        return;
    }
    g_os[slot].hi.store(0, std::memory_order_seq_cst);
    g_os[slot].lo.store(0, std::memory_order_seq_cst);
}
}  // namespace

OsWriteScope::OsWriteScope(const void* p, uint64_t n) {
    if (!n) return;
    const uint64_t lo = reinterpret_cast<uint64_t>(p) & ~(kPage - 1), hi = (reinterpret_cast<uint64_t>(p) + n + kPage - 1) & ~(kPage - 1);
    for (int i = 0;; ++i) {      // the operating system's write would fail on a page an import froze: wait for the import
        {
            std::lock_guard<std::mutex> lk(g_scope_mu);
            if (!freeze_overlaps(lo, hi)) {
                slot_ = os_slot_acquire(p, n);
                break;
            }
        }
        Sleep(i < 1000 ? 0 : 1);
    }
    before_os_write(p, n);
}

ImportFreeze::ImportFreeze(const void* p, uint64_t n) {
    if (!n || !heap_exists()) return;
    const uint64_t lo = reinterpret_cast<uint64_t>(p) & ~(kPage - 1), hi = (reinterpret_cast<uint64_t>(p) + n + kPage - 1) & ~(kPage - 1);
    if (lo < kHeapBase || hi > kHeapBase + heap_reserved()) return;      // only the heap's primary view (the import goes through another)
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_veh) g_veh = AddVectoredExceptionHandler(1, on_fault);
        if (!g_veh) return;
    }
    for (int i = 0; slot_ < 0; ++i) {   // after the file reads in progress there; then no new one starts until the end
        {
            std::lock_guard<std::mutex> lk(g_scope_mu);
            if (!os_write_overlaps(lo, hi)) {
                for (int k = 0; k < kFreezeSlots; ++k) {
                    uint64_t expect = 0;
                    if (g_frz[k].hi.load(std::memory_order_relaxed) == 0 && g_frz[k].lo.compare_exchange_strong(expect, lo | 1, std::memory_order_seq_cst)) {
                        g_frz[k].lo.store(lo, std::memory_order_seq_cst);
                        g_frz[k].hi.store(hi, std::memory_order_seq_cst);
                        g_frz_n.fetch_add(1, std::memory_order_seq_cst);
                        slot_ = k;
                        break;
                    }
                }
            }
        }
        if (slot_ < 0) Sleep(i < 1000 ? 0 : 1);
    }
    // [memory] freeze_table (default on): when the heap's page table says every page of the range is committed read-write (the guest's own
    // protection), the read-write runs are the pages no region has armed or guarded (our read-only pages, by their bits), and no VirtualQuery
    // is needed. Each VirtualQuery on the heap's section view walks the whole run of identical pages after the address (5 to 85 ms on a 10 GiB
    // heap), and this loop ran under the exclusive lock that arm() and every write fault wait for: render-thread frames of 200-500 ms with
    // "write-watch arming" 180 ms next to each pool import. The table is read before the lock (memory.cpp takes its lock first and the
    // tracker's inside it); a change since shows in the epoch, and then, or with another freeze over the range, the walk is done as before.
    const bool use_table = config_get_bool("memory", "freeze_table", true);   // read per freeze (imports are rare): the tests switch it
    const uint64_t epoch = use_table ? heap_epoch() : 0;
    const int table = use_table ? heap_rw_pages(lo, hi - lo) : -1;
    // Exclusive against arm() and region destruction: the read-write runs found here are ours until the destructor gives them back.
    AcquireSRWLockExclusive(&g_transitions);
    bool done = false;
    if (table == 1 && heap_epoch() == epoch) {
        bool other = false;
        for (int k = 0; k < kFreezeSlots && !other; ++k) {
            if (k == slot_) continue;
            const uint64_t h = g_frz[k].hi.load(std::memory_order_seq_cst);
            other = h && g_frz[k].lo.load(std::memory_order_seq_cst) < hi && lo < h;
        }
        if (!other) {
            const uint64_t np = (hi - lo) / kPage;
            std::vector<uint64_t> ours((np + 63) / 64, 0);   // pages read-only (or no-access) because a region armed or guarded them
            g_busy.fetch_add(1, std::memory_order_seq_cst);
            for (RegionCursor cursor; Region* r = cursor.next();) {
                const uint64_t rlo = std::max(lo, r->base), rhi = std::min(hi, r->base + r->npages * kPage);
                for (uint64_t a = rlo; a < rhi; a += kPage)
                    if (test(r->prot.get(), (a - r->base) / kPage)) {
                        const uint64_t q = (a - lo) / kPage;
                        ours[q >> 6] |= uint64_t{1} << (q & 63);
                    }
            }
            g_busy.fetch_sub(1, std::memory_order_seq_cst);
            for (uint64_t q = 0; q < np;) {
                while (q < np && (ours[q >> 6] >> (q & 63) & 1)) ++q;
                const uint64_t s = q;
                while (q < np && !(ours[q >> 6] >> (q & 63) & 1)) ++q;
                if (q > s) {
                    DWORD was = 0;
                    const uint64_t a = lo + s * kPage, e = lo + q * kPage;
                    if (VirtualProtect(reinterpret_cast<void*>(a), static_cast<SIZE_T>(e - a), PAGE_READONLY, &was)) runs_.emplace_back(a, e);
                }
            }
            done = true;
            g_freeze_table_hits.fetch_add(1, std::memory_order_relaxed);
        }
    }
    for (uint64_t a = lo; !done && a < hi;) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(reinterpret_cast<void*>(a), &mbi, sizeof mbi) != sizeof mbi) break;
        const uint64_t end = std::min<uint64_t>(hi, reinterpret_cast<uint64_t>(mbi.BaseAddress) + mbi.RegionSize);
        DWORD was = 0;
        if (mbi.State == MEM_COMMIT && (mbi.Protect & 0xff) == PAGE_READWRITE && !(mbi.Protect & PAGE_GUARD) &&
            VirtualProtect(reinterpret_cast<void*>(a), static_cast<SIZE_T>(end - a), PAGE_READONLY, &was))
            runs_.emplace_back(a, end);
        a = end;
    }
    ReleaseSRWLockExclusive(&g_transitions);
}

uint64_t freeze_table_hits() { return g_freeze_table_hits.load(std::memory_order_relaxed); }

bool any_guarded(uint64_t lo, uint64_t hi) {
    bool any = false;
    g_busy.fetch_add(1, std::memory_order_seq_cst);
    for (RegionCursor cursor; Region* r = cursor.next();) {
        if (any) break;
        const uint64_t a = std::max(lo, r->base), e = std::min(hi, r->base + r->npages * kPage);
        if (a < e) any = any_set(r->guard.get(), (a - r->base) / kPage, (e - a + kPage - 1) / kPage);
    }
    g_busy.fetch_sub(1, std::memory_order_seq_cst);
    return any;
}

ImportFreeze::~ImportFreeze() {
    if (slot_ < 0) return;
    AcquireSRWLockExclusive(&g_transitions);
    for (const auto& r : runs_) {
        DWORD was = 0;
        VirtualProtect(reinterpret_cast<void*>(r.first), static_cast<SIZE_T>(r.second - r.first), PAGE_READWRITE, &was);
    }
    ReleaseSRWLockExclusive(&g_transitions);
    g_frz[slot_].hi.store(0, std::memory_order_seq_cst);
    g_frz[slot_].lo.store(0, std::memory_order_seq_cst);
    g_frz_n.fetch_sub(1, std::memory_order_seq_cst);
}

OsWriteScope::~OsWriteScope() { os_slot_release(slot_); }

OsReadScope::OsReadScope(const void* p, uint64_t n) {
    if (!n) return;
    slot_ = os_slot_acquire(p, n);   // first: guard() cannot take a page of the range back between the fetch and the OS call
    before_os_read(p, n);
}

OsReadScope::~OsReadScope() { os_slot_release(slot_); }

void before_os_read(const void* p, uint64_t n) {
    // guarded pages only: their owner makes them current and they become read-only (armed) again, which the OS can read. Armed pages
    // stay armed: a read does not change them.
    if (!n || g_live.load(std::memory_order_acquire) == 0) return;
    const uint64_t lo = reinterpret_cast<uint64_t>(p) & ~(kPage - 1), hi = reinterpret_cast<uint64_t>(p) + n;
    for (uint64_t a = lo; a < hi; a += kPage)
        if (guarded_at(a)) {
            if (GuardHandler fn = g_guard_fn.load(std::memory_order_acquire)) fn(a);
            drop_guard(a, kPage);
        }
}

void before_os_write(const void* p, uint64_t n) {
    // guarded pages first: their owner makes them current (the write may cover only part of a page), outside the region locks
    if (n && g_live.load(std::memory_order_acquire) != 0) {
        const uint64_t lo = reinterpret_cast<uint64_t>(p) & ~(kPage - 1), hi = reinterpret_cast<uint64_t>(p) + n;
        for (uint64_t a = lo; a < hi; a += kPage)
            if (guarded_at(a)) {
                if (GuardHandler fn = g_guard_fn.load(std::memory_order_acquire)) fn(a);
                drop_guard(a, kPage);
            }
    }
    for_regions(reinterpret_cast<uint64_t>(p), n, [](Region* r, uint64_t first, uint64_t count) {
        for (uint64_t q = first; q < first + count; ++q)
            if (test(r->prot.get(), q)) release_page(r, q);
    });
}

void forget(uint64_t addr, uint64_t size, bool ban) {
    for_regions(addr, size, [ban](Region* r, uint64_t first, uint64_t count) {
        if (ban) set_bits(r->ban.get(), first, count, true);
        if (!any_set(r->prot.get(), first, count) && !any_set(r->valid.get(), first, count)) return;
        for (uint64_t q = first; q < first + count; ++q)           // guarded pages: read-only first (the owner's newer copy is dropped)
            if (test(r->guard.get(), q)) {              // protection first, then the bit (see unguard_locked)
                DWORD was = 0;
                VirtualProtect(reinterpret_cast<void*>(r->base + q * kPage), kPage, PAGE_READONLY, &was);
                r->guard[q >> 6].fetch_and(~bit(q), std::memory_order_acq_rel);
            }
        for (uint64_t q = first; q < first + count; ++q)           // armed pages go back to read-write (the caller may then set its own)
            if (test(r->prot.get(), q)) release_page(r, q);
        set_bits(r->valid.get(), first, count, false);
        set_bits(r->written.get(), first, count, true);
    });
}

void ban_section(uint64_t offset, uint64_t size) { forget(kHeapBase + offset, size, true); }

uint64_t faults() { return g_faults.load(std::memory_order_relaxed); }
uint64_t fault_waits() { return g_waits.load(std::memory_order_relaxed); }

}  // namespace nxrt::mem::ww
