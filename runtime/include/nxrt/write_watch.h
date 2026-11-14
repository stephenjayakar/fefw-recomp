// SPDX-License-Identifier: GPL-3.0-or-later
// Page write tracking of guest memory (S38). The owner of a region "arms" pages: they become read-only for the CPU, and the first
// write to one of them faults; a vectored exception handler makes the page writable again, marks it written and lets the write go on.
// The owner (the NVN back end keeps device-local copies of pool pages, nvn_gpu_pool.cpp) then knows which pages changed without reading
// them. Only the CPU's view is protected: the GPU reaches the same physical pages through its own import and never faults.
//
// Limits, all handled here or by the callers named:
//  * Regions lie in the heap's primary view (mem::kHeapBase). A page the guest can also reach through an alias view (svcMapMemory and
//    friends, mem::heap_alias) is banned: a write through the alias would not fault. memory.cpp calls forget() for every change of
//    mapping or protection it makes, so an armed page never ends up with a protection somebody else chose.
//  * Writes by the operating system into guest memory (ReadFile into a buffer) do not raise an exception: the call fails instead. The file
//    readers call before_os_write() on the destination first (services/fs_*.cpp).
//  * Reads by the operating system from guest memory (WriteFile from a buffer, ReadProcessMemory) fail the same way on a guarded page
//    (PAGE_NOACCESS, the read guards below). Callers hold an OsReadScope over the source first.
//  * Granularity is the 4 KiB page.
#pragma once
#include <atomic>
#include <cstdint>
#include <vector>
#include <string>
#include <utility>

namespace nxrt::mem::ww {

struct Region;

// A region over [addr, addr + size) (page aligned down/up). Null when the range is not inside the heap's primary view or the tracker is
// unavailable. owner is a diagnostic name with static lifetime. Any thread; the region has one owner thread for the calls below marked "owner".
Region* region_create(uint64_t addr, uint64_t size, const char* owner = "unspecified");
// Crash diagnostics: snapshots atomic ownership bits without taking region locks.
std::string describe_page(uint64_t addr);
// Makes every armed page writable again and frees the region. Owner.
void region_destroy(Region* r);
uint64_t region_base(const Region* r);
uint64_t region_pages(const Region* r);

// Owner: arms pages [first, first + count): read-only, and valid until written. False when a page could not be protected (not committed
// read-write, banned): nothing in the run is armed then.
bool arm(Region* r, uint64_t first, uint64_t count);
// Owner: true when every page of the run is armed and nobody wrote it since.
bool all_valid(const Region* r, uint64_t first, uint64_t count);
bool page_valid(const Region* r, uint64_t page);
// Owner: some page of [first, first + count) is valid (pages past the region are not); word by word (rt agent).
bool any_valid(const Region* r, uint64_t first, uint64_t count);
bool page_banned(const Region* r, uint64_t page);
// Owner: the owner changed the pages itself (the GPU wrote them): no longer valid, still armed (the next arm() re-validates them).
void invalidate(Region* r, uint64_t first, uint64_t count);
// Owner: appends the pages written by the CPU since the last call (page indices) and clears that record.
void take_written(Region* r, std::vector<uint64_t>& out);

// Any thread: the operating system is about to write [p, p + n) (a file read into guest memory): armed pages in the range become writable
// and count as written. Cheap when no region exists.
void before_os_write(const void* p, uint64_t n);
// The same for the lifetime of the object, and no page of the range is armed again until it ends: hold it around the ReadFile calls,
// and memory.cpp holds it around every change of mapping or protection it makes.
class OsWriteScope {
public:
    OsWriteScope(const void* p, uint64_t n);
    ~OsWriteScope();
    OsWriteScope(const OsWriteScope&) = delete;
    OsWriteScope& operator=(const OsWriteScope&) = delete;
private:
    int slot_ = -1;
};
// The CPU's writes to [p, p + n) (the heap's primary view) wait while the object lives: its pages that are read-write become read-only, a
// write to one faults and the fault handler holds it until the scope ends, and file reads into the range (OsWriteScope) and arm() wait
// too. Hold it around a Vulkan import of the same pages through another view (nvn_gpu_pool.cpp): the NVIDIA driver's vkAllocateMemory
// with VK_EXT_external_memory_host reads and writes back every page it pins, and a CPU write landing between its read and its write
// was lost (one byte at the start of a page, issue 97: a zstd frame the game then failed to decompress forever, a stuck loading screen).
// Pages that are no-access (guarded) or read-only already are left as they are; their writes fault and wait the same way.
// ImportFreezes that found their read-write runs from the heap's page table ([memory] freeze_table) rather than VirtualQuery (tests).
uint64_t freeze_table_hits();
// Any thread: some page of [lo, hi) is guarded (no access) by a region.
bool any_guarded(uint64_t lo, uint64_t hi);
class ImportFreeze {
public:
    ImportFreeze(const void* p, uint64_t n);
    ~ImportFreeze();
    ImportFreeze(const ImportFreeze&) = delete;
    ImportFreeze& operator=(const ImportFreeze&) = delete;
private:
    int slot_ = -1;
    std::vector<std::pair<uint64_t, uint64_t>> runs_;   // the runs made read-only here, made read-write again at the end
};
// Any thread: the operating system is about to read [p, p + n) (WriteFile from guest memory): guarded pages in the range are made
// current by the guard handler and become readable. Armed pages stay armed. Cheap when no region exists.
void before_os_read(const void* p, uint64_t n);
// The same for the lifetime of the object, and no page of the range is guarded (or armed) again until it ends: hold it around the OS
// call that reads guest memory.
class OsReadScope {
public:
    OsReadScope(const void* p, uint64_t n);
    ~OsReadScope();
    OsReadScope(const OsReadScope&) = delete;
    OsReadScope& operator=(const OsReadScope&) = delete;
private:
    int slot_ = -1;
};
// memory.cpp: the mapping or protection of [addr, addr + size) is about to change. Armed pages there are made read-write again and
// forgotten (no longer valid or armed); memory.cpp then sets its own protection. ban: they can never be armed again (another view of
// them exists).
void forget(uint64_t addr, uint64_t size, bool ban);
// Section offsets [offset, offset + size) got another view (memory.cpp, heap_alias): those pages are banned in every region.
void ban_section(uint64_t offset, uint64_t size);

// ---- read guards (the NVN back end's deferred compute results, nvn_gpu_pool.cpp) ----------------------------------------------
// The owner's copy of guarded pages is newer than the CPU's: they are PAGE_NOACCESS, and the first CPU access of any kind (read or
// write, any thread) calls the guard handler with the address before the access goes on. The handler makes the CPU's pages current and
// calls unguard(); if a page is still guarded when it returns, it is unguarded here (the access then sees what the pages hold). Only
// armed pages (read-only because of arm) can be guarded; unguard makes them read-only again (armed, not valid). A file read into guarded
// pages (before_os_write) calls the handler first; a change of mapping or protection (forget) drops the guard without it.
using GuardHandler = void (*)(uint64_t addr);
void set_guard_handler(GuardHandler fn);
// Owner: guards pages [first, first + count). False when a page is not armed (written since, never armed, banned) or a file read is
// in progress there: nothing is guarded then. Pages already guarded stay so.
bool guard(Region* r, uint64_t first, uint64_t count);
// Any thread: the guarded pages among [first, first + count) become read-only (armed) again.
void unguard(Region* r, uint64_t first, uint64_t count);
// Owner: true when every page of the run is still armed (read-only or guarded: the CPU did not write it since it was armed).
bool all_armed(const Region* r, uint64_t first, uint64_t count);
// Owner: true when every page of the run is valid, or still armed with its bit set in `alt` (one bit per page of the region, the caller's
// own bitmap). A word of 64 pages at a time: the NVN back end asks this for every buffer a draw binds.
bool all_valid_or_armed(const Region* r, uint64_t first, uint64_t count, const std::atomic<uint64_t>* alt);
// Moves (after the bits change) every time a page of any region stops being valid or armed: a fault releases it, invalidate, forget, a
// failed arm, region_destroy. An answer built from valid / armed bits that saw the same value before and after is still true as long as
// the value has not moved (pages only turn valid or armed in between, which keeps such an answer true). nvn's servable memo uses it.
uint64_t clear_epoch();
uint64_t guard_faults();   // accesses that hit a guarded page (statistics)

uint64_t faults();     // write faults handled so far (statistics)
uint64_t fault_waits(); // rounds a fault spent waiting for another thread's release of the same page (statistics; stays 0 for foreign faults)

}  // namespace nxrt::mem::ww
