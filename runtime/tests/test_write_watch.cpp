// SPDX-License-Identifier: GPL-3.0-or-later
// S38: page write tracking (include/nxrt/write_watch.h).
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "nxrt/config.h"
#include "nxrt/memory.h"
#include "nxrt/write_watch.h"
#include "test_framework.h"

NXTEST_GROUP(write_watch)

namespace {

namespace mem = nxrt::mem;
namespace ww = nxrt::mem::ww;
constexpr uint64_t kPg = 4096;
constexpr uint64_t kHeap = 64ull << 20;
constexpr uint64_t kPages = 64;

DWORD prot_of(uint64_t addr) {
    MEMORY_BASIC_INFORMATION mbi{};
    VirtualQuery(reinterpret_cast<void*>(addr), &mbi, sizeof mbi);
    return mbi.Protect & 0xff;
}

// A fresh heap with kPages committed pages at offset 0 and a region over them.
ww::Region* setup() {
    mem::shutdown();
    if (!mem::init() || !mem::heap_create(kHeap) || !mem::heap_commit(0, kPages * kPg)) return nullptr;
    return ww::region_create(mem::kHeapBase, kPages * kPg);
}

void teardown(ww::Region* r) {
    ww::region_destroy(r);
    mem::shutdown();
}

}  // namespace

NXTEST(arm_then_write_releases_the_page) {
    ww::Region* r = setup();
    REQUIRE(r != nullptr);
    CHECK_EQ(ww::region_pages(r), kPages);
    CHECK(ww::arm(r, 2, 4));
    CHECK(ww::all_valid(r, 2, 4));
    CHECK_EQ(prot_of(mem::kHeapBase + 3 * kPg), static_cast<DWORD>(PAGE_READONLY));
    std::vector<uint64_t> w;
    ww::take_written(r, w);
    CHECK(w.empty());
    const uint64_t before = ww::faults();
    *reinterpret_cast<volatile uint32_t*>(mem::kHeapBase + 3 * kPg + 8) = 0x1234;   // faults once, then goes through
    CHECK_EQ(*reinterpret_cast<volatile uint32_t*>(mem::kHeapBase + 3 * kPg + 8), 0x1234u);
    CHECK_EQ(ww::faults(), before + 1);
    CHECK(!ww::page_valid(r, 3));
    CHECK(ww::page_valid(r, 2) && ww::page_valid(r, 4) && ww::page_valid(r, 5));
    CHECK_EQ(prot_of(mem::kHeapBase + 3 * kPg), static_cast<DWORD>(PAGE_READWRITE));
    ww::take_written(r, w);
    CHECK(w.size() == 1 && w[0] == 3);
    ww::invalidate(r, 4, 1);                                  // the owner wrote it (GPU): not valid, still read-only
    CHECK(!ww::page_valid(r, 4));
    CHECK_EQ(prot_of(mem::kHeapBase + 4 * kPg), static_cast<DWORD>(PAGE_READONLY));
    CHECK(ww::arm(r, 2, 4));                                  // re-arming a mix of armed and released pages
    CHECK(ww::all_valid(r, 2, 4));
    teardown(r);
}

NXTEST(destroy_makes_pages_writable) {
    ww::Region* r = setup();
    REQUIRE(r != nullptr);
    CHECK(ww::arm(r, 0, kPages));
    ww::region_destroy(r);
    CHECK_EQ(prot_of(mem::kHeapBase), static_cast<DWORD>(PAGE_READWRITE));
    *reinterpret_cast<volatile uint32_t*>(mem::kHeapBase + 10 * kPg) = 7;      // no region: a plain write
    mem::shutdown();
}

NXTEST(file_read_into_armed_pages) {
    ww::Region* r = setup();
    REQUIRE(r != nullptr);
    char tmp[MAX_PATH], path[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp);
    GetTempFileNameA(tmp, "nww", 0, path);
    std::vector<uint8_t> data(3 * kPg);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<uint8_t>(i * 7);
    FILE* f = std::fopen(path, "wb");
    REQUIRE(f != nullptr);
    std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING * 0, nullptr);
    REQUIRE(h != INVALID_HANDLE_VALUE);
    uint8_t* dst = reinterpret_cast<uint8_t*>(mem::kHeapBase + 8 * kPg);
    CHECK(ww::arm(r, 8, 3));
    DWORD got = 0;
    SetFilePointer(h, 0, nullptr, FILE_BEGIN);
    CHECK(!ReadFile(h, dst, static_cast<DWORD>(data.size()), &got, nullptr));   // what the scope is for: the kernel cannot write read-only pages
    {
        const ww::OsWriteScope ws(dst, data.size());
        CHECK(!ww::arm(r, 8, 3));                              // no arming while the read is in progress
        SetFilePointer(h, 0, nullptr, FILE_BEGIN);
        CHECK(ReadFile(h, dst, static_cast<DWORD>(data.size()), &got, nullptr) && got == data.size());
    }
    CHECK(std::memcmp(dst, data.data(), data.size()) == 0);
    CHECK(!ww::page_valid(r, 8));
    CHECK(ww::arm(r, 8, 3));                                   // the scope ended
    CloseHandle(h);
    DeleteFileA(path);
    teardown(r);
}

NXTEST(memory_module_changes_are_respected) {
    ww::Region* r = setup();
    REQUIRE(r != nullptr);
    CHECK(ww::arm(r, 20, 2));
    CHECK(mem::protect(mem::kHeapBase + 20 * kPg, kPg, mem::Prot::R));   // the guest's own protection wins over the tracker's
    CHECK(!ww::page_valid(r, 20));
    CHECK(ww::page_valid(r, 21));
    CHECK_EQ(prot_of(mem::kHeapBase + 20 * kPg), static_cast<DWORD>(PAGE_READONLY));
    CHECK(!ww::arm(r, 20, 1));                                 // a page the guest made read-only is not armed
    CHECK(mem::protect(mem::kHeapBase + 20 * kPg, kPg, mem::Prot::RW));
    CHECK(ww::arm(r, 20, 1));
    CHECK(mem::heap_alias(0x10000, 0x10000) != 0);             // pages 16..31 get a second view: banned
    CHECK(ww::page_banned(r, 20));
    CHECK(!ww::page_valid(r, 20) && !ww::page_valid(r, 21));
    CHECK(!ww::arm(r, 16, 1));
    CHECK_EQ(prot_of(mem::kHeapBase + 21 * kPg), static_cast<DWORD>(PAGE_READWRITE));
    CHECK(ww::arm(r, 2, 1));                                   // pages outside the alias still work
    teardown(r);
}

NXTEST(writer_thread_while_arming) {
    ww::Region* r = setup();
    REQUIRE(r != nullptr);
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> writes{0};
    std::thread t([&] {
        uint32_t n = 0;
        while (!stop.load()) {
            for (uint64_t p = 0; p < 16; ++p) *reinterpret_cast<volatile uint32_t*>(mem::kHeapBase + p * kPg + (n % 1024) * 4) = n;
            ++n;
            writes.fetch_add(16);
        }
    });
    int armed = 0;
    for (int i = 0; i < 3000; ++i)
        armed += ww::arm(r, static_cast<uint64_t>(i % 13), 4) ? 1 : 0;
    for (int i = 0; i < 5000 && writes.load() == 0; ++i) Sleep(1);     // the writer may start late on a loaded machine
    stop.store(true);
    t.join();
    CHECK(armed > 0);
    CHECK(writes.load() > 0);
    // Invariant: a valid page is read-only; a page that is not armed is writable.
    for (uint64_t p = 0; p < 20; ++p) {
        const DWORD pr = prot_of(mem::kHeapBase + p * kPg);
        if (ww::page_valid(r, p)) CHECK_EQ(pr, static_cast<DWORD>(PAGE_READONLY));
        CHECK(pr == PAGE_READONLY || pr == PAGE_READWRITE);
    }
    for (uint64_t p = 0; p < 16; ++p) *reinterpret_cast<volatile uint32_t*>(mem::kHeapBase + p * kPg) = 1;   // every page still writable through faults
    for (uint64_t p = 0; p < 16; ++p) CHECK(!ww::page_valid(r, p));
    teardown(r);
}

namespace {

// The store sits in a non-inlined callee: clang-cl's SEH only models faults at call sites (see test_memory.cpp).
__declspec(noinline) void store_word(uint64_t addr) { *reinterpret_cast<volatile uint32_t*>(addr) = 0x55aa; }

// A write that may fault; true when the fault went past the tracker to the caller's own handler. (SEH needs a function without destructors.)
__declspec(noinline) bool write_passes_or_faults(uint64_t addr) {
    __try {
        store_word(addr);
        return false;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return true;
    }
}

}  // namespace

NXTEST(foreign_fault_passes_on_at_once) {
    ww::Region* r = setup();
    REQUIRE(r != nullptr);
    DWORD was = 0;
    REQUIRE(VirtualProtect(reinterpret_cast<void*>(mem::kHeapBase + 30 * kPg), kPg, PAGE_READONLY, &was));   // inside the region, not armed by us
    const uint64_t faults0 = ww::faults(), waits0 = ww::fault_waits();
    CHECK(write_passes_or_faults(mem::kHeapBase + 30 * kPg + 16));          // nobody is releasing: not ours, no waiting
    CHECK(write_passes_or_faults(mem::kHeapBase + 30 * kPg + 32));
    CHECK_EQ(ww::fault_waits(), waits0);
    CHECK_EQ(ww::faults(), faults0);
    CHECK_EQ(prot_of(mem::kHeapBase + 30 * kPg), static_cast<DWORD>(PAGE_READONLY));
    // an armed page next to it is still handled
    CHECK(ww::arm(r, 31, 1));
    CHECK(!write_passes_or_faults(mem::kHeapBase + 31 * kPg));
    CHECK_EQ(ww::faults(), faults0 + 1);
    // after a release has finished, a second fault on the same page (a late writer) finds it writable and goes on without waiting
    CHECK(!write_passes_or_faults(mem::kHeapBase + 31 * kPg + 4));
    teardown(r);
}

NXTEST(writers_race_for_the_same_armed_pages) {
    ww::Region* r = setup();
    REQUIRE(r != nullptr);
    std::atomic<bool> stop{false};
    std::atomic<int> failed{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; ++t)
        ts.emplace_back([&, t] {
            uint32_t n = static_cast<uint32_t>(t);
            while (!stop.load()) {
                for (uint64_t p = 0; p < 8; ++p)
                    if (write_passes_or_faults(mem::kHeapBase + p * kPg + (n % 1024) * 4)) failed.fetch_add(1);   // a fault the tracker did not handle
                ++n;
            }
        });
    // keep arming until the writers have really raced the arms (they start late on a loaded machine): at least 3000 arms and 20000 faults, or 5 s
    const uint64_t faults0 = ww::faults();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 3000 || (ww::faults() - faults0 < 20000 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5)); ++i) ww::arm(r, 0, 8);
    stop.store(true);
    for (auto& t : ts) t.join();
    CHECK_EQ(failed.load(), 0);
    CHECK(ww::faults() > faults0);
    teardown(r);
}

// More live pools than the old 512-entry cap, including faults and mapping changes
// in later blocks. Repeat with holes so destruction/recreation exercises slot reuse.
NXTEST(region_table_grows_and_reuses_slots) {
    mem::shutdown();
    REQUIRE(mem::init() && mem::heap_create(kHeap));
    constexpr size_t count = 1537;
    REQUIRE(mem::heap_commit(0, count * kPg));
    std::vector<ww::Region*> regions(count);
    for (size_t round = 0; round < 2; ++round) {
        for (size_t i = 0; i < count; ++i) {
            if (regions[i]) continue;
            regions[i] = ww::region_create(mem::kHeapBase + i * kPg, kPg);
            REQUIRE(regions[i] != nullptr);
        }
        for (size_t i = 0; i < count; ++i) {
            CHECK(ww::arm(regions[i], 0, 1));
            *reinterpret_cast<volatile uint32_t*>(mem::kHeapBase + i * kPg) = static_cast<uint32_t>(i);
            CHECK(!ww::page_valid(regions[i], 0));
            std::vector<uint64_t> written;
            ww::take_written(regions[i], written);
            CHECK(written.size() == 1 && written[0] == 0);
            CHECK(ww::arm(regions[i], 0, 1));
        }
        // Mapping changes must invalidate every block, not just the first 512.
        ww::forget(mem::kHeapBase, count * kPg, false);
        for (size_t i = 0; i < count; ++i) CHECK(!ww::page_valid(regions[i], 0));
        for (size_t i = 0; i < count; i += 2) {
            ww::region_destroy(regions[i]);
            regions[i] = nullptr;
        }
    }
    for (auto* r : regions) ww::region_destroy(r);
    mem::shutdown();
}

NXTEST(region_table_growth_while_faulting) {
    ww::Region* r = setup();
    REQUIRE(r != nullptr);
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> writes{0};
    std::thread writer([&] {
        while (!stop.load()) {
            *reinterpret_cast<volatile uint32_t*>(mem::kHeapBase) = 42;
            writes.fetch_add(1);
        }
    });
    // Region creation and destruction race with faults on a separate, stable page.
    std::vector<ww::Region*> regions;
    for (size_t i = 0; i < 2305; ++i) {
        regions.push_back(ww::region_create(mem::kHeapBase + kPg, kPg));
        CHECK(regions.back() != nullptr);
        ww::arm(r, 0, 1);
    }
    for (auto* region : regions) {
        ww::region_destroy(region);
        ww::arm(r, 0, 1);
    }
    stop.store(true);
    writer.join();
    CHECK(writes.load() != 0);
    teardown(r);
}

// Distinct owners (texture tracker and pool mirror) may cover the same pages.
NXTEST(overlapping_owners_arm_concurrently) {
    auto* a = setup();
    REQUIRE(a != nullptr);
    auto* b = ww::region_create(mem::kHeapBase, kPages * kPg);
    REQUIRE(b != nullptr);
    std::atomic<int> phase{0}, ready{0}, both{0};
    bool ok[2]{};
    auto run = [&](int id, ww::Region* r) {
        for (int i = 1; i <= 10000; ++i) {
            while (phase.load() != i) YieldProcessor();
            ok[id] = ww::arm(r, 0, 1);
            ready.fetch_add(1);
            while (phase.load() == i) YieldProcessor();
        }
    };
    std::thread ta(run, 0, a), tb(run, 1, b);
    for (int i = 1; i <= 10000; ++i) {
        ready.store(0);
        phase.store(i);
        while (ready.load() != 2) YieldProcessor();
        if (ok[0] && ok[1]) both.fetch_add(1);
        ww::forget(mem::kHeapBase, kPg, false);
        phase.store(-i);
        // Workers must observe the end of this phase before the next begins.
        // The next phase's different positive value also releases their wait.
    }
    ta.join(); tb.join();
    CHECK_EQ(both.load(), 0);
    ww::region_destroy(b);
    teardown(a);
}

NXTEST(destroy_and_recreate_while_writers_fault) {
    auto* r = setup();
    REQUIRE(r != nullptr);
    std::atomic<bool> stop{false};
    std::atomic<unsigned> failed{0}, writes{0};
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) writers.emplace_back([&] {
        while (!stop.load()) {
            if (write_passes_or_faults(mem::kHeapBase)) failed.fetch_add(1);
            writes.fetch_add(1);
        }
    });
    for (int i = 0; i < 3000; ++i) {
        ww::arm(r, 0, 1);
        ww::region_destroy(r);
        r = ww::region_create(mem::kHeapBase, kPages * kPg);
    }
    stop.store(true);
    for (auto& t : writers) t.join();
    CHECK_EQ(failed.load(), 0u);
    CHECK(writes.load() > 0);
    teardown(r);
}

namespace {
std::atomic<bool> delayed_entered{false}, delayed_resume{false};
LONG CALLBACK delay_write_fault(EXCEPTION_POINTERS* ep) {
    auto* e = ep->ExceptionRecord;
    if (e->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && e->NumberParameters >= 2 &&
        e->ExceptionInformation[1] == mem::kHeapBase) {
        delayed_entered.store(true);
        while (!delayed_resume.load()) YieldProcessor();
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
}
NXTEST(fault_delivered_after_last_region_destroyed) {
    auto* r = setup();
    REQUIRE(r != nullptr);
    REQUIRE(ww::arm(r, 0, 1));
    delayed_entered.store(false); delayed_resume.store(false);
    auto veh = AddVectoredExceptionHandler(1, delay_write_fault);
    REQUIRE(veh != nullptr);
    bool failed = false;
    std::thread writer([&] { failed = write_passes_or_faults(mem::kHeapBase); });
    while (!delayed_entered.load()) YieldProcessor();
    ww::region_destroy(r);
    delayed_resume.store(true);
    writer.join();
    RemoveVectoredExceptionHandler(veh);
    CHECK(!failed);
    CHECK_EQ(prot_of(mem::kHeapBase), (DWORD)PAGE_READWRITE);
    mem::shutdown();
}

NXTEST(guarded_pages_boundaries_and_mapping_changes) {
    auto* a = setup();
    REQUIRE(a != nullptr);
    auto* b = ww::region_create(mem::kHeapBase + kPg, kPg, "boundary owner");
    REQUIRE(b != nullptr);
    REQUIRE(ww::arm(a, 0, 1));
    REQUIRE(ww::arm(b, 0, 1));
    REQUIRE(ww::guard(b, 0, 1));
    CHECK(!ww::arm(a, 1, 1)); // cannot take another owner's guard
    CHECK(ww::describe_page(mem::kHeapBase + kPg).find("guard=1") != std::string::npos);
    // One native store crosses both region and page boundaries. Each fault must
    // release the correct owner, including the guard on the second page.
    *reinterpret_cast<volatile uint64_t*>(mem::kHeapBase + kPg - 4) = 0x123456789abcdef0ull;
    CHECK(!ww::page_valid(a, 0));
    CHECK(!ww::page_valid(b, 0));
    ww::region_destroy(b);
    CHECK(mem::heap_decommit(kPg, kPg));
    CHECK(write_passes_or_faults(mem::kHeapBase + kPg)); // retired does not authorize NOACCESS
    CHECK(!ww::arm(a, 1, 1));
    CHECK(mem::heap_commit(kPg, kPg));
    CHECK(!write_passes_or_faults(mem::kHeapBase + kPg));
    REQUIRE(ww::arm(a, 1, 1));
    REQUIRE(ww::guard(a, 1, 1));
    {
        ww::OsWriteScope scope(reinterpret_cast<void*>(mem::kHeapBase + kPg), kPg);
        CHECK_EQ(prot_of(mem::kHeapBase + kPg), (DWORD)PAGE_READWRITE);
        CHECK(!ww::guard(a, 1, 1));
    }
    teardown(a);
}

// ImportFreeze (issue 97): while a range is frozen for a Vulkan import, a CPU write to it waits and lands after the freeze (the driver
// rewrites each imported page and would have lost it); pages the tracker armed stay armed, and arm() leaves the range alone meanwhile.
// Both ways of finding the read-write runs: the heap's page table with the tracker's bits ([memory] freeze_table, default) and the
// VirtualQuery walk.
NXTEST(import_freeze_holds_writes_until_it_ends) {
  for (const char* table : {"1", "0"}) {
    nxrt::config_set("memory", "freeze_table", table);
    std::printf("    [memory] freeze_table = %s\n", table);
    ww::Region* r = setup();
    REQUIRE(r != nullptr);
    CHECK(ww::arm(r, 6, 1));                                   // an armed page inside the frozen range
    const uint64_t hits0 = ww::freeze_table_hits();
    std::atomic<bool> wrote{false};
    std::thread writer;
    {
        const ww::ImportFreeze fz(reinterpret_cast<void*>(mem::kHeapBase + 4 * kPg), 4 * kPg);
        CHECK_EQ(prot_of(mem::kHeapBase + 4 * kPg), static_cast<DWORD>(PAGE_READONLY));
        CHECK_EQ(prot_of(mem::kHeapBase + 6 * kPg), static_cast<DWORD>(PAGE_READONLY));    // armed, left to the tracker
        CHECK_EQ(prot_of(mem::kHeapBase + 8 * kPg), static_cast<DWORD>(PAGE_READWRITE));   // outside the range
        CHECK(!ww::arm(r, 5, 1));                              // no arming while frozen
        writer = std::thread([&] {
            *reinterpret_cast<volatile uint8_t*>(mem::kHeapBase + 5 * kPg) = 0x5a;
            *reinterpret_cast<volatile uint8_t*>(mem::kHeapBase + 6 * kPg) = 0x6b;
            wrote.store(true);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        CHECK(!wrote.load());                                  // still waiting for the freeze
        CHECK_EQ(*reinterpret_cast<volatile uint8_t*>(mem::kHeapBase + 5 * kPg), 0u);   // reads go on
    }
    writer.join();
    CHECK_EQ(ww::freeze_table_hits() - hits0, std::string(table) == "1" ? 1ull : 0ull);   // the way under test was taken
    CHECK(wrote.load());
    CHECK_EQ(*reinterpret_cast<volatile uint8_t*>(mem::kHeapBase + 5 * kPg), 0x5au);
    CHECK_EQ(*reinterpret_cast<volatile uint8_t*>(mem::kHeapBase + 6 * kPg), 0x6bu);
    CHECK_EQ(prot_of(mem::kHeapBase + 4 * kPg), static_cast<DWORD>(PAGE_READWRITE));
    std::vector<uint64_t> w;
    ww::take_written(r, w);
    CHECK(std::find(w.begin(), w.end(), 6u) != w.end());      // the armed page's write was tracked as usual
    CHECK(ww::arm(r, 5, 1));                                   // the freeze ended
    teardown(r);
  }
  nxrt::config_set("memory", "freeze_table", "1");
}
