// SPDX-License-Identifier: GPL-3.0-or-later
// A small Horizon kernel: result codes, kernel objects and the handle table, guest threads (one Win32 thread per guest thread,
// console core model), and the 19.2 MHz system tick. The SVC dispatcher lives in svc.h, IPC sessions in ipc.h.
// Pattern from SuperMarioWonderRecomp runtime/src/kernel.cpp + kernel_state.h (MIT) and smo-recomp runtime/os/os_threads.c (GPL-3.0).
#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nxrt/abi.h"

namespace nxrt::kernel {

// ---- result codes (module | description << 9; the low 9 bits are the module, kernel = 1) ----------------------------------
constexpr uint32_t make_result(uint32_t module, uint32_t desc) { return module | (desc << 9); }
// Values from Atmosphere libvapours svc_results.hpp (module 1): description numbers below.
constexpr uint32_t kOk = 0;
constexpr uint32_t kNotImplemented = 0x4201;     // 33: what a kernel answers for an SVC number it does not implement
constexpr uint32_t kInvalidSize = 0xCA01;        // 101
constexpr uint32_t kInvalidAddress = 0xCC01;     // 102
constexpr uint32_t kOutOfResource = 0xCE01;      // 103
constexpr uint32_t kOutOfMemory = 0xD001;        // 104
constexpr uint32_t kOutOfHandles = 0xD201;       // 105
constexpr uint32_t kInvalidMemoryState = 0xD401; // 106 (InvalidCurrentMemory)
constexpr uint32_t kInvalidPermission = 0xD801;  // 108
constexpr uint32_t kInvalidMemoryRegion = 0xDC01;// 110
constexpr uint32_t kInvalidPriority = 0xE001;    // 112
constexpr uint32_t kInvalidCoreId = 0xE201;      // 113
constexpr uint32_t kInvalidHandle = 0xE401;      // 114
constexpr uint32_t kInvalidPointer = 0xE601;     // 115
constexpr uint32_t kInvalidCombination = 0xE801; // 116
constexpr uint32_t kTimedOut = 0xEA01;           // 117
constexpr uint32_t kCancelled = 0xEC01;          // 118
constexpr uint32_t kOutOfRange = 0xEE01;         // 119
constexpr uint32_t kInvalidEnum = 0xF001;        // 120
constexpr uint32_t kNotFound = 0xF201;           // 121
constexpr uint32_t kSessionClosed = 0xF601;      // 123
constexpr uint32_t kInvalidState = 0xFA01;       // 125
constexpr uint32_t kStubService = make_result(0x1D5, 0x5E5);   // 0x0BCBD5: "an nxrt stub service answered": never a real Horizon result

constexpr uint32_t kPseudoCurrentThread = 0xFFFF8000;
constexpr uint32_t kPseudoCurrentProcess = 0xFFFF8001;
constexpr uint32_t kHandleWaitMask = 0x40000000;   // set in a guest mutex word while there are waiters

// ---- objects ---------------------------------------------------------------------------------------------------------------
enum class ObjType { Unknown, Thread, Event, Session, SharedMemory, TransferMemory, Process, Port };

struct KObject {
    virtual ~KObject() = default;
    virtual ObjType type() const { return ObjType::Unknown; }
    virtual bool signaled() const { return false; }        // for WaitSynchronization
    virtual void consume() {}                              // called on the object WaitSynchronization returned
};

struct EventState {
    bool signaled = false;
};
struct KEvent : KObject {                                  // readable and writable handles share `state`
    std::shared_ptr<EventState> state = std::make_shared<EventState>();
    bool writable = false;
    ObjType type() const override { return ObjType::Event; }
    bool signaled() const override { return state->signaled; }
};

struct KSharedMemory : KObject {
    uint64_t size = 0;
    uint64_t host = 0;           // backing memory (kernel region) the service writes to
    std::string name;
    ObjType type() const override { return ObjType::SharedMemory; }
};

struct KTransferMemory : KObject {
    uint64_t addr = 0, size = 0;
    ObjType type() const override { return ObjType::TransferMemory; }
};

// ---- handle table ------------------------------------------------------------------------------------------------------------
uint32_t handle_add(std::shared_ptr<KObject> o);          // 0 when the table is full; handles start at 0x10000 (Horizon-like opaque values)
std::shared_ptr<KObject> handle_get(uint32_t h);          // resolves kPseudoCurrentThread to the calling guest thread
void handle_close(uint32_t h);
size_t handle_count();
template <class T>
std::shared_ptr<T> handle_as(uint32_t h) { return std::dynamic_pointer_cast<T>(handle_get(h)); }
void handles_reset();                                      // tests

// ---- guest threads ---------------------------------------------------------------------------------------------------------
struct KThread : KObject {
    uint64_t id = 0;                    // svcGetThreadId value
    uint32_t handle = 0;
    std::string name;
    alignas(16) ::Ctx ctx{};            // guest registers of this thread
    uint64_t tls = 0;                   // TLS region (TPIDRRO_EL0), 0x200 bytes; +0x1F0 holds the thread pointer (nn::os convention)
    uint64_t tp_block = 0;              // static TLS block (TPIDR_EL0 points here)
    uint64_t entry = 0, arg = 0, stack_top = 0;
    int prio = 44;                      // HOS priority 0 (highest) .. 63; applications use 28..59 (nn::os 0..31 = prio - 28)
    int ideal_core = 0;                 // console core 0..2 the thread prefers (-1 none)
    uint64_t core_mask = 1;             // console cores the thread may run on (bit n = core n)
    void* host_handle = nullptr;        // Win32 thread handle (owned by the thread object)
    uint32_t host_tid = 0;
    bool started = false, cancel_sync = false;
    std::atomic<bool> exited{false};    // written under the kernel lock (thread_mark_exited), read under the thread-list lock and elsewhere
    std::atomic<bool> aged{false};      // starvation guard (starvation_guard.cpp): the host leaves this thread ready, so its yields keep the processor
    std::atomic<uint64_t> host_affinity{0};   // the host affinity mask sched_apply (or a starvation boost) last gave the host thread
    std::atomic<int> last_proc{-1};     // host processor the thread was on at its last SVC (svc_dispatch), for the starvation line
    int occ_group = 0;                  // [kernel] occupancy (occupancy.cpp): 0 none, 1 main, 2.. a named game pool; set before the thread starts
    // wait state, protected by the kernel lock (kernel_sync.cpp)
    bool wake = false;
    uint32_t wait_result = 0;
    uint64_t wait_mutex = 0, wait_key = 0, wait_addr = 0;
    uint32_t wait_tag = 0;
    void* cv_storage[1] = {nullptr};    // CONDITION_VARIABLE is pointer sized: this thread's own wake-up (kernel_sync.cpp)
    std::weak_ptr<KThread> self;
    // [kernel] spin_threads / lock_spin_threads (kernel_sync.cpp): this thread spins on its wake flag before it sleeps / polls a held mutex
    // word before it queues. Set when the game names the thread (main: when it is adopted). Read by the thread itself.
    bool spin_ok = false, lock_spin_ok = false;
    char game_name[32] = {};            // the name nn::os::SetThreadName gave it (thread_diag.cpp), written under the kernel lock; empty until then
    ~KThread() override;                // closes host_handle
    ObjType type() const override { return ObjType::Thread; }
    bool signaled() const override { return exited; }
};

struct ThreadParams {
    uint64_t entry = 0, arg = 0, stack_top = 0;
    int prio = 44;
    int core = -2;                       // -2: the process default core (0)
    std::string name;
    uint64_t host_stack_bytes = 0;       // 0: [threads] host_stack_mb (default 64) MiB: lifted code nests deeply
};

// Creates the thread object (TLS region and static TLS block included) without starting it. Returns its handle, 0 on failure.
uint32_t thread_create(const ThreadParams& p);
// Starts the host thread: it registers with diag, applies the console core/priority model, runs entry(arg) through dispatch_call
// with sp = stack_top and exits when the function returns (like svcExitThread).
uint32_t thread_start(uint32_t handle);                    // result code
[[noreturn]] void thread_exit();                           // svcExitThread for the calling guest thread
// Turns the calling host thread into the main guest thread (stack allocated in the kernel region). Returns its handle.
uint32_t main_thread_adopt(uint64_t stack_size, const char* name = "main");
void main_thread_release();                                // the adopted main thread is done: unregisters it from diag and marks it exited
KThread* current_thread();                                 // nullptr on a host thread that is not a guest thread
::Ctx* current_ctx();                                      // current_thread()->ctx or nullptr
std::vector<std::shared_ptr<KThread>> threads();           // every thread ever created and not yet destroyed
bool thread_join(uint32_t handle, uint32_t timeout_ms);    // host-side wait for the thread to finish (tests, shutdown)
void kernel_reset();                                       // tests: join/forget threads, clear handles, sync state (no thread may run guest code)
uint64_t thread_count_running();                           // started and not yet exited
bool thread_set_priority(KThread& t, int hos_prio);       // updates the object and the host thread
int app_core_count();                 // application cores: 3 (Switch 1) or 6 (Switch 2); [sched] app_cores overrides (threads.md)
bool thread_set_core_mask(KThread& t, int ideal_core, uint64_t mask);   // console cores; false when mask has no usable core

// ---- scheduling model (console cores) ----------------------------------------------------------------------------------------
// The console runs the highest-priority ready thread on each of its three application cores. The model gives every console core one host
// processor and maps HOS priorities onto six Windows priority bands. [sched] affinity = soft (default): that processor is the thread's ideal
// processor and its affinity is the whole pool, so a ready thread runs on any idle processor; strict: the affinity is pinned to it.
struct SchedInfo {
    bool active = false;                 // false: fewer usable host processors than console cores, or [sched] console_cores = 0
    bool soft = false;                   // [sched] affinity = soft: affinity `pool`, ideal processor = the core's processor
    int cores = 3;                       // application cores of the console: 3 (Switch 1), 6 (Switch 2)
    uint64_t host_masks[16] = {};        // host processor per console core (the first `cores` are used): the affinity (strict) or ideal processor (soft)
    uint64_t pool = 0;                   // host processors a guest thread may use under soft (and during a starvation boost under strict)
    int slot = -1;                       // processor slot taken (-1: shared / explicit mask)
    uint64_t render_core = 0;            // [sched] render_place = reserve: the physical core kept for the render thread (left out of `pool`), 0 = none
    uint64_t reserved = 0;               // processors left out of `pool` by render_place = reserve (render_core and main's SMT sibling)
};
SchedInfo sched_info();                  // initialises on first use
int sched_host_priority(int hos_prio);   // THREAD_PRIORITY_* for a HOS priority under the model (nn::os prio = hos - 28)
// Which console core the calling thread is on right now: the core of its host processor (strict), or its ideal core (soft, or off-model processor).
int sched_current_core();
// Process priority class. [sched] priority_class = auto | high | above_normal | normal (default auto: high when `interactive`, i.e. a person
// is playing in a window, otherwise left alone). The guest threads are pinned and run with Windows' wake-up boost off, so under
// background CPU load a woken thread waits a full scheduler quantum for its processor (debug 180, S10 measurements: 100 to 200 ms
// stalls and half the frame rate with every core busy); a class above the background keeps the hand-overs immediate.
void sched_set_process_priority(bool interactive);
void sched_apply(KThread& t);            // (re)applies affinity + priority of t to its host thread
std::string sync_wait_report();          // the guest threads blocked in WaitForAddress / process-wide keys / mutex hand-overs, with the word each waits on (watchdog dump; never blocks)
void yield_host();                       // the calling thread offers its processor to another ready thread (SwitchToThread), unless the starvation guard aged it
void starvation_guard_start();           // ages guest threads the host leaves ready but never runs (starvation_guard.cpp, [sched] starvation_guard)
void starvation_guard_stop();
uint64_t starvation_boosts();            // boosts given so far
void install_spin_hint();                // WFE / YIELD of the lifted code hand the host processor over (busy_mutex.cpp, [sched] spin_yield)
void install_busy_mutex_hle();           // nn::os::LockBusyMutex as host code that yields while it waits (busy_mutex.cpp, [sched] busy_mutex_hle)
void sched_reset_for_tests();            // forget the slot and re-read the config at the next sched_info()

// ---- time --------------------------------------------------------------------------------------------------------------------
uint64_t system_tick();                  // the console tick (tick_hz()), from QueryPerformanceCounter
constexpr uint64_t kTickHzSwitch1 = 19200000;   // Switch 1 system tick (CNTFRQ_EL0)
constexpr uint64_t kTickHzSwitch2 = 31250000;   // Switch 2 system tick (SDK 20.5 for Switch 2 uses 31.25 MHz)
uint64_t tick_hz();                      // the registered game's platform tick frequency
void sleep_ns(int64_t ns);               // high-resolution sleep; <= 0 yields

// ---- kernel-level synchronisation (the SVC semantics; kernel_sync.cpp) -------------------------------------------------------
// ArbitrateLock / WaitProcessWideKeyAtomic / WaitForAddress ... operate on guest memory words. Exposed for tests and for HLE code.
uint32_t arbitrate_lock(uint32_t owner_handle, uint64_t addr, uint32_t tag);
uint32_t arbitrate_unlock(uint64_t addr);
uint32_t wait_process_wide_key(uint64_t mutex, uint64_t key, uint32_t tag, int64_t timeout_ns);
uint32_t signal_process_wide_key(uint64_t key, int32_t count);
uint32_t wait_for_address(uint64_t addr, uint32_t arb_type, int32_t value, int64_t timeout_ns);     // 0 less, 1 dec+less, 2 equal
uint32_t signal_to_address(uint64_t addr, uint32_t signal_type, int32_t value, int32_t count);     // 0 signal, 1 +1 if equal, 2 modify by waiting
uint32_t wait_synchronization(const uint32_t* handles, int32_t count, int64_t timeout_ns, int32_t* index_out);
uint32_t cancel_synchronization(uint32_t thread_handle);
void notify_waitable_changed();          // wake threads blocked in wait_synchronization (call after changing an object's signaled state)
// A host function that one host thread runs while it is blocked in a kernel wait (WaitSynchronization, mutex, condition variable and address
// waits, and SleepThread longer than 10 ms): its wait is cut into 10 ms slices and fn runs between them, without the kernel lock. The play layer gives the window's thread one,
// so the window keeps working while the game waits on something the player must do (the software keyboard applet). fn null: off.
void set_blocked_hook(uint32_t host_tid, void (*fn)());

}  // namespace nxrt::kernel
