// SPDX-License-Identifier: GPL-3.0-or-later
// Gauges for long-session monitoring (S34). A gauge is a named integer that a subsystem can report on request: the number of live
// textures, the size of a cache, bytes held by a pool. The soak sampler (`[diag] soak_csv`) reads all gauges every few seconds and writes
// one JSON line per sample; nothing runs when the key is empty, and registering a gauge costs one map insert.
//
//   static nxrt::GaugeCounter g_images("vk.images");       // a live-object counter: ++ at create, -- at destroy (relaxed atomics)
//   nxrt::gauge_register("nvn.pipelines", [] { return (int64_t)g_pipelines.size(); });   // a callback, called from the sampler thread
//
// A callback runs on the sampler thread while the game runs: it must take the lock that protects what it reads, or read an atomic. A
// callback that cannot be made safe should not be registered. Names are "area.what" in lower case, unique; registering a name again
// replaces the callback. gauge_unregister removes it (call it before the object the callback reads is destroyed).
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace nxrt {

void gauge_register(const char* name, std::function<int64_t()> fn);
void gauge_unregister(const char* name);
// Every gauge as (name, value), sorted by name. Callbacks are called with the registry lock held, so they must not register gauges.
std::vector<std::pair<std::string, int64_t>> gauge_read_all();

// A live-object counter that registers itself on construction (use a namespace-scope or function-local static).
class GaugeCounter {
public:
    explicit GaugeCounter(const char* name);
    ~GaugeCounter();
    GaugeCounter(const GaugeCounter&) = delete;
    GaugeCounter& operator=(const GaugeCounter&) = delete;
    void add(int64_t n = 1) { v_.fetch_add(n, std::memory_order_relaxed); }
    void sub(int64_t n = 1) { v_.fetch_sub(n, std::memory_order_relaxed); }
    GaugeCounter& operator++() { add(1); return *this; }
    GaugeCounter& operator--() { sub(1); return *this; }
    int64_t get() const { return v_.load(std::memory_order_relaxed); }
private:
    std::string name_;
    std::atomic<int64_t> v_{0};
};

// ---- the sampler -----------------------------------------------------------------------------------------------------------
// Started by the runtime when `[diag] soak_csv` (env NXRT_DIAG_SOAK_CSV) names a file; `[diag] soak_interval_s` (default 10) is the period.
// Each line is a JSON object: t (seconds since start), frame, win_frames, ft_p50/ft_p99/ft_max (ms between presented frames in the window),
// process figures (ws, private, peak_ws, pagefile, handles, threads, gdi, user, cpu_ms), then every gauge by name.
void soak_start();
void soak_stop();
// Called by diag_frame_tick(): records the time of a presented frame for the window statistics. Cheap (two relaxed stores).
void soak_note_frame();

// ---- the hitch log (always on) ---------------------------------------------------------------------------------------------
// Every presented frame's time goes into the same ring as the soak sampler's. A monitor thread logs, every `[diag] hitch_summary_s` (default 10,
// 0 = off): frames, mean fps, 1% low fps, worst frame ms and the frames over twice the median. A frame longer than `[diag] hitch_ms` (default 50;
// 0 = off) is logged with what the registered context sources saw change during it; at most one such line per second (the worst of the second,
// with the count of the others), then the phase times below (ms of the frame). Sources are read on the presenting thread once per frame:
// they must be lock-free (an atomic load).
void hitch_add_counter(const char* label, uint64_t (*fn)());   // cumulative count / bytes: the hitch line shows the growth during the frame
void hitch_add_level(const char* label, int64_t (*fn)());      // a level (jobs in flight): shown when non-zero at the end of the frame
void hitch_start();
void hitch_stop();

// Phase timers of the hitch line: time spent in each phase is summed (any thread) and the slow frame's line shows how many ms of it fell
// inside that frame, so a slow frame names where the time went: the render thread idle (waiting for the game) or busy (replay, GPU waits,
// uploads, present), the game's threads waiting on the back end, and the write-fault handler. Cheap: two QueryPerformanceCounter reads and
// one relaxed add per timed scope. Phases nested in another (the replay's parts) are shown inside it.
enum HitchPhase : int {
    kHpRenderIdle,          // render thread: waiting for a job (nothing queued by the game)
    kHpRenderReplay,        // render thread: recording and submitting a submission (all of submit_on_render_thread)
    kHpRenderGpuWait,       //   of it: waiting for the GPU to free a submission slot
    kHpRenderSubmit,        //   of it: the queue lock and vkQueueSubmit2
    kHpRenderUpload,        //   of it: texture uploads (prepare, staging, copies; GPU untile recording)
    kHpRenderPipeline,      //   of it: pipeline creation on the render thread
    kHpRenderLockWait,      //   of it: waits for other threads (pool import, background texture job, queue lock)
    kHpWwArm,               // any thread: write-watch arming (holds the region lock exclusive, faults wait for it)
    kHpRenderPresent,       // render thread: the whole present
    kHpRenderLimiter,       //   of it: the frame limiter's wait
    kHpRenderAcquire,       //   of it: begin_frame (frame slot fence, swapchain image)
    kHpRenderPresentEnd,    //   of it: end_frame (submit and vkQueuePresent)
    kHpFaults,              // any thread: the write-watch fault handler
    kHpGameBackpressure,    // game threads: present waiting for the render thread (present_ahead)
    kHpGameSyncWait,        // game threads: sync waits on the GPU timeline (nvnSyncWait and the like)
    kHpPoolImport,          // any thread: importing a memory pool (vkAllocateMemory over the game's memory; the driver pins the pages)
    kHpGameSvc,             // the game's presenting thread: inside system calls (the rest of its time between presents is guest code)
    kHpGameIpc,             //   of it: IPC requests (service calls, file reads)
    kHpGameWait,            //   of it: waits (SleepThread, WaitSynchronization, ArbitrateLock, WaitProcessWideKeyAtomic, WaitForAddress)
    kHpRenderTexTake,       // render thread, of the uploads: texprep::take (finding a background result, waits for a running job included)
    kHpRenderTexPrep,       // render thread, of the uploads: prepare_texture run there (stand-ins, CPU untile, cache reads, transcodes)
    kHpRenderTexStaging,    // render thread, of the uploads: the upload buffer (allocation and the copy of the prepared data into it)
    kHpRenderImage,         // render thread: image creation (create_texture_gpu: format checks, vmaCreateImage), inside or outside the uploads
    kHpCount
};
uint64_t hitch_ticks();                                  // QueryPerformanceCounter
void hitch_phase_add(int phase, uint64_t ticks);
struct HitchTimer {
    int phase;
    uint64_t t0;
    explicit HitchTimer(int p) : phase(p), t0(hitch_ticks()) {}
    ~HitchTimer() { hitch_phase_add(phase, hitch_ticks() - t0); }
    HitchTimer(const HitchTimer&) = delete;
    HitchTimer& operator=(const HitchTimer&) = delete;
};
void hitch_set_render_thread();                          // the calling thread is the render thread
bool hitch_on_render_thread();
// The game's thread called present: the longest gap between two such calls inside a slow frame is shown (game-side frame time).
void hitch_note_game_present();
// The calling thread has called hitch_note_game_present at least once: svc_dispatch times its system calls for the hitch line.
bool hitch_on_game_present_thread();
// The last `window_s` seconds of frame times as a compact block of text (frames, fps, 1% low, worst, per-second fps and worst ms, the worst
// frames and the age of the last one). Allocates; never takes a lock another thread could hold: for the crash, freeze and memory reports.
std::string frame_times_summary(double window_s = 30.0);
// For a live view (the F1 menu's debug page). The times between the last presented frames in ms, oldest first, at most `n`; returns how many.
size_t frame_times_recent(float* ms, size_t n);
// A phase timer's running total (HitchPhase), in ms since start: the difference of two reads is the time spent in it between them.
double hitch_phase_total_ms(int phase);
// Frames presented so far and the ms since the last one (the frame in progress is number *frames + 1, as the hitch line counts). Lock-free.
// False before the first frame.
bool hitch_frame_age(uint64_t* frames, double* age_ms);

}  // namespace nxrt
