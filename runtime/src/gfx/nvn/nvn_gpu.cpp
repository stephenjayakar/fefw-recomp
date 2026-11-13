// SPDX-License-Identifier: GPL-3.0-or-later
// The Vulkan side of the NVN back end: the render thread, submission slots (command buffer, descriptor pools, upload memory),
// the timeline semaphore, deferred destruction, barriers and the public backend API (include/nxrt/nvn_backend.h).
// Design follows smo-recomp (record on game threads, replay on one render thread) and SuperMarioWonderRecomp docs/nvn (a submit
// is replayed under one lock and never overlaps the next); the code is new.
#include "nxrt/nvn_backend.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>

#include "nxrt/config.h"
#include "nxrt/diag.h"
#include "nxrt/frame_limiter.h"
#include "nxrt/ftrace.h"
#include "nxrt/gauge.h"
#include "nxrt/log.h"
#include "nxrt/lowlat.h"
#include "vk_gauges.h"
#include "nvn_gpu_internal.h"
#include "nvn_impl.h"
#include "nvn_pipeline.h"
#include "nvn_present_pass.h"
#include "extras/output_pass.h"

namespace nxrt::nvn::be::gpu {
namespace {

constexpr const char* kTag = "nvn.gpu";
constexpr uint32_t kSlots = 36;                  // about three frames of submissions (the game submits ~11 times a frame): with fewer, recording waits on
                                                 // the GPU for a submission of a recent frame and the CPU cannot get frames ahead
constexpr uint32_t kGpMax = 1024;      // GPU pass profiler: timestamps per submission
constexpr VkDeviceSize kUploadBytes = 8ull << 20;   // per slot; larger uploads get their own staging buffer

// Live-count gauges of this file (leak monitor, nxrt/gauge.h).
nxrt::GaugeCounter g_gauge_deferred("nvn.deferred_destroy");     // closures waiting for their submission to finish (defer_destroy, Frame::defer)
nxrt::GaugeCounter g_gauge_render_queue("nvn.render_queue");     // jobs queued for the render thread
nxrt::GaugeCounter g_gauge_barriers_mem("nvn.cnt.barriers_mem");   // full memory barriers recorded (cumulative; per frame = delta / frames)
nxrt::GaugeCounter g_gauge_barriers_img("nvn.cnt.barriers_img");   // image layout barriers recorded (cumulative)

struct Slot {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> dpools;
    size_t dpool_cur = 0;
    gfx::Buffer upload;
    VkDeviceSize upload_off = 0;
    std::vector<gfx::Buffer> extra;       // staging / overflow buffers of the submission
    uint64_t serial = 0;                  // last submission that used the slot
    bool timed = false;                   // frame trace: the submission wrote GPU timestamps (query pair index = slot index * 2)
    uint64_t cpu_t0 = 0;
    uint64_t sets_live = 0;               // descriptor sets allocated from this slot's pools since the last reset (vk.descriptor_sets)
    uint32_t gp_n = 0;                    // GPU pass profiler: timestamps written by the submission
    std::vector<std::pair<std::string, uint32_t>> gp_labels;   // label and argument of each interval (stamp i closes interval i)
    uint64_t frame = 0;                   // [prof] gpu_stats: presents counted when the submission was recorded (the frame it belongs to)
};

// [prof] gpu_stats = N: every N frames one log line (nvn.gpustats) with the GPU time per frame (the sum of each submission's first-to-last
// timestamp, the present's own command buffer excluded) and the commands and barriers per frame; with [prof] gpu_passes as well, the GPU
// time per pass label. Render thread only.
struct GpuStats {
    int64_t every = 0;
    uint64_t cur_frame = UINT64_MAX;
    double cur_ns = 0;
    std::vector<float> frame_ms;
    uint64_t presents0 = 0;
    uint64_t c0[13] = {};
    std::map<std::string, std::pair<double, uint64_t>> gp;   // label -> GPU ns, intervals
};

struct Deferred {
    uint64_t serial;
    std::function<void()> fn;
};

struct Gpu {
    gfx::Device* dev = nullptr;
    gfx::Renderer* rend = nullptr;
    FrameLimiter* limiter = nullptr;
    bool keep_aspect = false;
    bool installed_entry_points = false;
    VkSemaphore timeline = VK_NULL_HANDLE;
    std::atomic<uint64_t> completed{0};
    uint64_t submitted = 0;               // render thread: last serial recorded (handed to vkQueueSubmit2, or waiting in `batch`)
    uint64_t flushed = 0;                 // render thread: last serial handed to vkQueueSubmit2
    // [nvn] submit_batch: recorded submissions wait here and go to the GPU in one vkQueueSubmit2 (flush_batch). Render thread.
    std::vector<Slot*> batch;
    uint32_t batch_max = 1;
    // submit_batch = frame: the batch is not flushed when the render thread runs out of work; it goes with the present, in the present's
    // own vkQueueSubmit2 (take_batch_for_present), or earlier when a thread waits for one of its serials (flush_want_serial).
    bool frame_batch = false;
    std::atomic<uint64_t> flushed_pub{0};          // `flushed`, readable from any thread
    // The highest serial a thread waits for (or polls), kept as a maximum and never cleared: the batch is flushed as soon as it holds a
    // serial >= this one that is not flushed yet. A serial asked for before the render thread has recorded it is flushed when it is
    // recorded (a bool cleared by the next flush lost such a request: the waiter hung until a present that it was the one to queue).
    std::atomic<uint64_t> flush_want_serial{0};
    uint64_t recording = 0;               // render thread: serial of the submission being recorded (0 outside)
    std::atomic<uint64_t> last_submitted{0};
    // Serials are handed out when a submission is queued (the game thread gets its serial at once and goes on while the render thread
    // records), in queue order, so the timeline is signalled in increasing order. Under qmu.
    std::atomic<uint64_t> reserved{0};
    // Presents queued and not yet done; the game thread waits while [nvn] present_ahead of them are pending (back pressure).
    std::mutex pmu;
    std::condition_variable pcv;
    int presents_pending = 0;
    // The game's last present (under pmu): present_again shows it once more while the game presents nothing (it waits for a library applet).
    Ref<Window> last_window;
    int last_index = -1;
    std::chrono::steady_clock::time_point last_present{};
    Slot slots[kSlots];
    VkQueryPool qpool = VK_NULL_HANDLE;   // frame trace: two timestamps per slot
    VkQueryPool gpool = VK_NULL_HANDLE;   // GPU pass profiler ([prof] gpu_passes): kGpMax timestamps per slot
    GpuStats stats;
    uint32_t next_slot = 0;
    Counters ctr;

    // render thread
    std::thread th;
    std::mutex qmu;
    std::condition_variable qcv;
    std::deque<std::function<void()>> queue;
    bool stop = false;
    std::thread::id tid;

    // deferred destruction
    std::mutex dmu;
    std::vector<std::function<void()>> pending_destroy;   // from any thread, assigned a serial by the render thread
    std::deque<Deferred> deferred;                        // render thread

    PresentPass present;
    bool present_ok = false;
    extras::OutputPass output;                 // HDR encode and FSR 1 (S14); used instead of `present` when the extras are on
    bool output_ok = false;
    std::mutex unsupported_mu;
    std::map<std::string, uint64_t> unsupported;
};

Gpu* g = nullptr;
void flush_batch();   // [nvn] submit_batch, below
// submit_batch = frame: a thread waits for a serial of the batch (render thread; the batch only changes there).
bool batch_flush_wanted() {
    if (g->batch.empty()) return false;
    const uint64_t want = g->flush_want_serial.load();
    return want > g->flushed && g->batch.back()->serial >= want;
}
std::mutex g_install_mu;
std::atomic<uint64_t> g_copy_inflight{0};
std::mutex g_copy_done_mu;
std::condition_variable g_copy_done_cv;

uint64_t read_completed() {
    uint64_t v = 0;
    if (g && g->timeline) vkGetSemaphoreCounterValue(g->dev->device(), g->timeline, &v);
    const uint64_t copying = g_copy_inflight.load();
    if (copying && v >= copying) v = copying - 1;
    if (v > g->completed.load()) g->completed.store(v);
    return g->completed.load();
}

void drain_pending_destroy() {
    std::vector<std::function<void()>> pend;
    {
        std::lock_guard<std::mutex> lk(g->dmu);
        pend.swap(g->pending_destroy);
    }
    const uint64_t s = g->recording ? g->recording : g->submitted;
    for (auto& fn : pend) g->deferred.push_back({s, std::move(fn)});
}

void retire() {
    drain_pending_destroy();
    const uint64_t done = read_completed();
    while (!g->deferred.empty() && g->deferred.front().serial <= done) {
        auto fn = std::move(g->deferred.front().fn);
        g->deferred.pop_front();
        g_gauge_deferred.sub(1);
        fn();
    }
}

thread_local bool t_render_thread = false;
// [nvn] test_throw_every = N (tests only): every Nth recorded submission and every Nth present throws std::bad_alloc, to check that
// a failed render job costs a frame and nothing else.
void test_throw(std::atomic<uint64_t>& n) {
    static const int64_t every = nxrt::config_get_int("nvn", "test_throw_every", 0);
    if (every > 0 && (n.fetch_add(1) + 1) % static_cast<uint64_t>(every) == 0) throw std::bad_alloc();
}
std::atomic<uint64_t> g_test_records{0}, g_test_presents{0};
template <class Fn> struct OnExit { Fn fn; ~OnExit() { fn(); } };
template <class Fn> OnExit<Fn> on_exit(Fn fn) { return OnExit<Fn>{std::move(fn)}; }
std::atomic<int> g_render_throw_reports{0};

// A C++ exception thrown on the render thread is caught by render_loop, which aborts that job; the first few are logged with
// the throwing stack (the catch alone cannot say where it came from).
LONG WINAPI render_throw_report(EXCEPTION_POINTERS* ep) {
    if (t_render_thread && ep && ep->ExceptionRecord && ep->ExceptionRecord->ExceptionCode == 0xE06D7363u &&
        g_render_throw_reports.fetch_add(1) < 3) {
        MEMORYSTATUSEX ms = {sizeof(ms)};
        GlobalMemoryStatusEx(&ms);
        NXLOG_E(kTag, "render thread C++ exception (system commit free %.1f GB of %.1f GB): %s", ms.ullAvailPageFile / 1073741824.0,
                ms.ullTotalPageFile / 1073741824.0, nxrt::crash_build_report(ep).c_str());
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

void render_loop() {
    t_render_thread = true;
    static void* veh = AddVectoredExceptionHandler(0, render_throw_report);
    (void)veh;
    g->tid = std::this_thread::get_id();
    log_set_thread_name("nvn-render");
    diag_register_thread("nvn-render", nullptr);   // so the freeze watchdog and the stall sampler can see its stack
    nxrt::hitch_set_render_thread();
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lk(g->qmu);
            if (!g->batch.empty() && (batch_flush_wanted() || (!g->frame_batch && !g->stop && g->queue.empty()))) {
                lk.unlock();
                flush_batch();      // nothing more to record for now (or a thread waits for it): the GPU gets what there is
                lk.lock();
            }
            if (!g->stop && g->queue.empty()) {
                nxrt::HitchTimer ht(nxrt::kHpRenderIdle);      // the hitch line: the render thread waited for the game
                g->qcv.wait(lk, [] { return g->stop || !g->queue.empty() || batch_flush_wanted(); });
                if (g->queue.empty() && !g->stop) continue;    // woken to flush
            }
            if (g->queue.empty()) {   // stop requested and drained
                lk.unlock();
                flush_batch();
                return;
            }
            job = std::move(g->queue.front());
            g->queue.pop_front();
            g_gauge_render_queue.sub(1);
        }
        try {
            job();
        } catch (const std::exception& e) {
            NXLOG_E(kTag, "render job threw: %s", e.what());
        }
    }
}

bool create_slot(Slot& s) {
    gfx::Device& d = *g->dev;
    VkCommandPoolCreateInfo ci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    ci.queueFamilyIndex = d.queue_family();
    if (vkCreateCommandPool(d.device(), &ci, nullptr, &s.pool) != VK_SUCCESS) return false;
    vk_gauge_command_pools().add(1);
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = s.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(d.device(), &ai, &s.cb) != VK_SUCCESS) return false;
    vk_gauge_command_buffers().add(1);
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (d.create_buffer(kUploadBytes, usage, gfx::MemoryUse::Upload, &s.upload, "nvn upload") != VK_SUCCESS) return false;
    return true;
}

VkDescriptorPool new_dpool() {
    const VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u << 16},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u << 13},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1u << 16},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u << 12},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1u << 10},
        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1u << 10},
    };
    VkDescriptorPoolCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.maxSets = 1u << 14;
    ci.poolSizeCount = sizeof(sizes) / sizeof(sizes[0]);
    ci.pPoolSizes = sizes;
    VkDescriptorPool p = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(g->dev->device(), &ci, nullptr, &p) != VK_SUCCESS) return VK_NULL_HANDLE;
    vk_gauge_descriptor_pools().add(1);
    return p;
}

// Hands the recorded submissions of the batch to the GPU in one vkQueueSubmit2, in recording order, signalling the timeline with the last
// serial (the earlier ones of the batch complete with it). Every kernel submission costs a pass over all memory the device holds: on Linux
// (Proton, RADV) the multi-gigabyte host-memory pool imports are user pages the kernel looks up again on each one, about 2 ms a
// submission on a Legion Go S, so 11 submissions a frame took 20 of its 25 ms. Render thread.
void flush_batch() {
    if (g->batch.empty()) return;
    std::vector<VkCommandBufferSubmitInfo> cbi(g->batch.size(), VkCommandBufferSubmitInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO});
    for (size_t i = 0; i < g->batch.size(); i++) cbi[i].commandBuffer = g->batch[i]->cb;
    const uint64_t serial = g->batch.back()->serial;
    VkSemaphoreSubmitInfo sig = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    sig.semaphore = g->timeline;
    sig.value = serial;
    sig.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = static_cast<uint32_t>(cbi.size());
    si.pCommandBufferInfos = cbi.data();
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos = &sig;
    VkResult r;
    {
        nxrt::HitchTimer ht(nxrt::kHpRenderSubmit);
        const uint64_t tl = nxrt::hitch_ticks();
        std::lock_guard<std::mutex> lk(g->dev->queue_mutex());
        nxrt::hitch_phase_add(nxrt::kHpRenderLockWait, nxrt::hitch_ticks() - tl);
        r = vkQueueSubmit2(g->dev->queue(), 1, &si, VK_NULL_HANDLE);
    }
    if (r != VK_SUCCESS) {
        NXLOG_E(kTag, "vkQueueSubmit2 failed (%d): the device is lost or out of memory", static_cast<int>(r));
        // keep the timeline moving so waiters do not hang: signal from the host
        VkSemaphoreSignalInfo ssi = {VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        ssi.semaphore = g->timeline;
        ssi.value = serial;
        vkSignalSemaphore(g->dev->device(), &ssi);
    }
    g->flushed = serial;
    g->flushed_pub.store(serial);
    g->batch.clear();
    g->ctr.submits.fetch_add(1);
}

// submit_batch = frame: a thread outside the render thread waits for a serial that may still sit in the batch (or not be recorded yet);
// the render thread flushes it as soon as it is in the batch instead of with the next present (which that thread may be the one to queue).
// renotify: wake the render thread even when the serial was asked for already (wait_serial's backstop).
void want_flush(uint64_t serial, bool renotify = false) {
    if (!g->frame_batch || serial <= g->flushed_pub.load()) return;
    uint64_t cur = g->flush_want_serial.load();
    do {
        if (cur >= serial) {
            if (!renotify) return;   // asked for already (by whoever raised it, who notified)
            break;
        }
    } while (!g->flush_want_serial.compare_exchange_weak(cur, serial));
    // Notified under qmu, so a render thread between its predicate and its wait cannot miss it.
    std::lock_guard<std::mutex> lk(g->qmu);
    g->qcv.notify_one();
}

void wait_serial_blocking(uint64_t serial) {
    if (!serial || read_completed() >= serial) return;
    if (serial > g->flushed && on_render_thread()) flush_batch();     // a serial still in the batch would never signal
    NXFT("gpu.wait");
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &g->timeline;
    wi.pValues = &serial;
    VkResult r = vkWaitSemaphores(g->dev->device(), &wi, UINT64_MAX);
    if (r != VK_SUCCESS) NXLOG_E(kTag, "vkWaitSemaphores failed (%d)", static_cast<int>(r));
    read_completed();
}

uint64_t stats_counter(int i) {
    const Counters& c = g->ctr;
    switch (i) {
        case 0: return c.draws.load(std::memory_order_relaxed);
        case 1: return c.passes.load(std::memory_order_relaxed);
        case 2: return c.pass_breaks.load(std::memory_order_relaxed);
        case 3: return c.dispatches.load(std::memory_order_relaxed);
        case 4: return c.clears.load(std::memory_order_relaxed);
        case 5: return c.copies.load(std::memory_order_relaxed);
        case 6: return c.submits.load(std::memory_order_relaxed);
        case 7: return static_cast<uint64_t>(g_gauge_barriers_mem.get());
        case 8: return static_cast<uint64_t>(g_gauge_barriers_img.get());
        case 9: return c.texture_uploads.load(std::memory_order_relaxed);
        case 11: return c.game_barriers.load(std::memory_order_relaxed);
        case 12: return c.dispatch_barriers_left_out.load(std::memory_order_relaxed);
        default: return c.presents.load(std::memory_order_relaxed);
    }
}

// [prof] gpu_stats: one submission's GPU time, added to the frame it was recorded in; a full window of frames is logged.
void stats_add(uint64_t frame, double ns) {
    GpuStats& st = g->stats;
    if (frame != st.cur_frame) {
        if (st.cur_frame != UINT64_MAX && st.cur_ns > 0) st.frame_ms.push_back(static_cast<float>(st.cur_ns * 1e-6));
        if (st.cur_frame == UINT64_MAX) {
            for (int i = 0; i < 13; ++i) st.c0[i] = stats_counter(i);
            st.gp.clear();
        }
        st.cur_frame = frame;
        st.cur_ns = 0;
    }
    st.cur_ns += ns;
    if (st.frame_ms.size() < static_cast<size_t>(st.every)) return;
    std::vector<float> v = st.frame_ms;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (float x : v) sum += x;
    const size_t n = v.size();
    auto pct = [&](double q) { return static_cast<double>(v[std::min(n - 1, static_cast<size_t>(q * static_cast<double>(n - 1) + 0.5))]); };
    uint64_t c1[13];
    for (int i = 0; i < 13; ++i) c1[i] = stats_counter(i);
    const double frames = std::max<double>(1.0, static_cast<double>(c1[10] - st.c0[10]));
    auto pf = [&](int i) { return static_cast<double>(c1[i] - st.c0[i]) / frames; };
    NXLOG_I("nvn.gpustats", "%zu frames: GPU ms/frame avg %.2f p50 %.2f p90 %.2f p99 %.2f max %.2f | per frame: draws %.0f, passes %.1f, pass breaks %.1f, "
            "dispatches %.1f, clears %.1f, copies %.1f, submits %.1f, full barriers %.1f (game %.1f), image barriers %.1f, texture uploads %.1f, dispatch barriers left out %.1f",
            n, sum / static_cast<double>(n), pct(0.5), pct(0.9), pct(0.99), static_cast<double>(v[n - 1]), pf(0), pf(1), pf(2), pf(3), pf(4), pf(5),
            pf(6), pf(7), pf(11), pf(8), pf(9), pf(12));
    if (!st.gp.empty()) {
        // the pass profiler's intervals by label, the most expensive first ("pass WxH cN:fmt dfmt": one render pass; "cs <hash>": one dispatch)
        std::vector<std::pair<double, std::string>> by;
        double total = 0;
        for (const auto& kv : st.gp) { by.emplace_back(kv.second.first, kv.first); total += kv.second.first; }
        std::sort(by.begin(), by.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        NXLOG_I("nvn.gpustats", "pass profile (timestamps drain the GPU between intervals): %.2f ms/frame in %zu labels", total * 1e-6 / frames, by.size());
        for (size_t i = 0; i < by.size() && i < 40; ++i)
            NXLOG_I("nvn.gpustats", "  %7.3f ms/frame %6.1f/frame  %s", by[i].first * 1e-6 / frames, static_cast<double>(st.gp[by[i].second].second) / frames,
                    by[i].second.c_str());
    }
    st.frame_ms.clear();
    st.gp.clear();
    std::copy(c1, c1 + 13, st.c0);
}

Slot& acquire_slot() {
    NXFT_MIN("slot.acquire", 20000);
    Slot& s = g->slots[g->next_slot];
    g->next_slot = (g->next_slot + 1) % kSlots;
    {
        nxrt::HitchTimer ht(nxrt::kHpRenderGpuWait);
        wait_serial_blocking(s.serial);
    }
    if (s.timed) {
        s.timed = false;
        uint64_t ts[2] = {0, 0};
        const uint32_t idx = static_cast<uint32_t>(&s - g->slots) * 2;
        if (vkGetQueryPoolResults(g->dev->device(), g->qpool, idx, 2, sizeof ts, ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && ts[1] >= ts[0]) {
            static const uint16_t z = ft::zone_id("gpu.exec");
            const double ns = static_cast<double>(ts[1] - ts[0]) * g->dev->properties().limits.timestampPeriod;
            ft::event(z, s.cpu_t0, s.cpu_t0 + static_cast<uint64_t>(ns), static_cast<uint32_t>(s.serial));
            if (g->stats.every > 0) stats_add(s.frame, ns);
        }
    }
    if (s.gp_n >= 2 && g->gpool) {      // GPU pass profiler: one event per interval between two timestamps, named gp:<label>, arg = draws
        uint64_t ts[kGpMax];
        const uint32_t n = std::min(s.gp_n, kGpMax);
        const uint32_t base = static_cast<uint32_t>(&s - g->slots) * kGpMax;
        if (vkGetQueryPoolResults(g->dev->device(), g->gpool, base, n, sizeof(uint64_t) * n, ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            static std::map<std::string, uint16_t> zones;
            const double period = g->dev->properties().limits.timestampPeriod;
            uint64_t t = s.cpu_t0;
            for (uint32_t i = 1; i < n && i < s.gp_labels.size(); i++) {
                const uint64_t d = ts[i] >= ts[i - 1] ? static_cast<uint64_t>(static_cast<double>(ts[i] - ts[i - 1]) * period) : 0;
                auto it = zones.find(s.gp_labels[i].first);
                if (it == zones.end()) it = zones.emplace(s.gp_labels[i].first, ft::zone_id(("gp:" + s.gp_labels[i].first).c_str())).first;
                ft::event(it->second, t, t + d, s.gp_labels[i].second);
                if (g->stats.every > 0) {
                    auto& e = g->stats.gp[s.gp_labels[i].first];
                    e.first += static_cast<double>(d);
                    e.second++;
                }
                t += d;
            }
        }
    }
    s.gp_n = 0;
    s.gp_labels.clear();
    retire();
    vkResetCommandPool(g->dev->device(), s.pool, 0);
    for (VkDescriptorPool p : s.dpools) vkResetDescriptorPool(g->dev->device(), p, 0);
    vk_gauge_descriptor_sets().sub(static_cast<int64_t>(s.sets_live));   // the reset freed them
    s.sets_live = 0;
    s.dpool_cur = 0;
    s.upload_off = 0;
    for (auto& b : s.extra) g->dev->destroy_buffer(b);
    s.extra.clear();
    return s;
}

}  // namespace

// ---- Frame ---------------------------------------------------------------------------------------------------------------
UploadAlloc Frame::upload(VkDeviceSize size, VkDeviceSize align) {
    Slot& s = *static_cast<Slot*>(slot);
    if (align == 0) align = 16;
    VkDeviceSize off = (s.upload_off + align - 1) & ~(align - 1);
    if (off + size <= kUploadBytes) {
        s.upload_off = off + size;
        UploadAlloc a;
        a.buffer = s.upload.buffer;
        a.offset = off;
        a.ptr = static_cast<uint8_t*>(s.upload.mapped) + off;
        a.size = size;
        return a;
    }
    return staging(size + align);
}

UploadAlloc Frame::staging(VkDeviceSize size) {
    Slot& s = *static_cast<Slot*>(slot);
    gfx::Buffer b;
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (g->dev->create_buffer(std::max<VkDeviceSize>(size, 256), usage, gfx::MemoryUse::Staging, &b, "nvn staging") != VK_SUCCESS) {
        NXLOG_E(kTag, "staging buffer of %llu bytes failed", static_cast<unsigned long long>(size));
        return {};
    }
    UploadAlloc a;
    a.buffer = b.buffer;
    a.offset = 0;
    a.ptr = static_cast<uint8_t*>(b.mapped);
    a.size = size;
    s.extra.push_back(b);
    return a;
}

UploadAlloc Frame::adopt_staging(gfx::Buffer b) {
    Slot& s = *static_cast<Slot*>(slot);
    UploadAlloc a;
    a.buffer = b.buffer;
    a.offset = 0;
    a.ptr = static_cast<uint8_t*>(b.mapped);
    a.size = b.size;
    s.extra.push_back(b);
    return a;
}

void Frame::defer(std::function<void()> fn) {
    g->deferred.push_back({serial, std::move(fn)});
    g_gauge_deferred.add(1);
}

VkDescriptorSet Frame::alloc_set(VkDescriptorSetLayout layout) {
    Slot& s = *static_cast<Slot*>(slot);
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (s.dpool_cur >= s.dpools.size()) {
            VkDescriptorPool p = new_dpool();
            if (!p) return VK_NULL_HANDLE;
            s.dpools.push_back(p);
        }
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = s.dpools[s.dpool_cur];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const VkResult r = vkAllocateDescriptorSets(g->dev->device(), &ai, &set);
        if (r == VK_SUCCESS) {
            g->ctr.descriptor_sets.fetch_add(1, std::memory_order_relaxed);
            s.sets_live++;
            vk_gauge_descriptor_sets().add(1);
            return set;
        }
        ++s.dpool_cur;   // exhausted or fragmented: move on to the next pool (created on demand)
    }
    return VK_NULL_HANDLE;
}

// ---- general helpers ---------------------------------------------------------------------------------------------------
bool on_render_thread() { return g && std::this_thread::get_id() == g->tid; }
gfx::Device& device() { return *g->dev; }
VkDevice vkdev() { return g->dev->device(); }
Counters& counters() { return g->ctr; }
bool counters_ready() { return g != nullptr; }
Counters* counters_if() { return g ? &g->ctr : nullptr; }
uint64_t completed_serial() { return g ? g->completed.load() : 0; }
uint64_t poll_completed() { return g ? read_completed() : 0; }
uint64_t current_serial() { return g ? (g->recording ? g->recording : g->submitted) : 0; }

void note_unsupported(const char* what) {
    if (!g) return;
    g->ctr.unsupported_ops.fetch_add(1);
    std::lock_guard<std::mutex> lk(g->unsupported_mu);
    uint64_t& n = g->unsupported[what];
    if (n++ == 0) NXLOG_W(kTag, "not implemented, skipped: %s", what);
}

void defer_destroy(std::function<void()> fn) {
    if (!g) {
        fn();
        return;
    }
    std::lock_guard<std::mutex> lk(g->dmu);
    g->pending_destroy.push_back(std::move(fn));
    g_gauge_deferred.add(1);
}

void run_on_render_thread(const std::function<void()>& fn) {
    if (!g) return;
    if (on_render_thread()) {
        fn();
        return;
    }
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    {
        std::lock_guard<std::mutex> lk(g->qmu);
        g->queue.push_back([&] {
            auto signal = on_exit([&] {
                std::lock_guard<std::mutex> l2(m);
                done = true;
                cv.notify_all();
            });
            fn();
        });
        g_gauge_render_queue.add(1);
    }
    g->qcv.notify_one();
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&] { return done; });
}

bool gpass_on() { return g && g->gpool != VK_NULL_HANDLE; }

void gpass_stamp(Frame& f, const char* label, uint32_t arg) {
    if (!g || !g->gpool || !f.slot) return;
    Slot& s = *static_cast<Slot*>(f.slot);
    if (s.gp_n >= kGpMax) return;
    vkCmdWriteTimestamp2(f.cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, g->gpool, static_cast<uint32_t>(&s - g->slots) * kGpMax + s.gp_n);
    s.gp_labels.emplace_back(label, arg);
    s.gp_n++;
}

namespace {
// [nvn] pass_barriers = tracked (S38): a layout change waits only for the stages that can have used the image in its old layout and blocks
// only the stages that use it in the new one. The layout says how the image was used: the replay keeps every image in the layout of its
// last use (attachments in the attachment layouts, sampled images in the read-only layouts, copies in the transfer layouts); GENERAL and
// anything else keep the full barrier.
std::atomic<bool> g_narrow_barriers{false};

constexpr VkPipelineStageFlags2 kShaderStages = VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
constexpr VkPipelineStageFlags2 kDepthStages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;

// Stages and accesses of an image's use in `layout`; writes_only: the access part names the writes only (the source half of a barrier).
void layout_scope(VkImageLayout layout, bool src, VkPipelineStageFlags2* stage, VkAccessFlags2* access) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            // the old contents do not matter, but the memory may have belonged to something still in flight: wait for all earlier commands
            *stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            *access = src ? VK_ACCESS_2_NONE : (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
            return;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            *stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            *access = src ? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT : (VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            return;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            *stage = kDepthStages;
            *access = src ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                          : (VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
            return;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
            *stage = kDepthStages | kShaderStages;
            *access = src ? VK_ACCESS_2_NONE : (VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            return;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            *stage = kShaderStages;
            *access = src ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            return;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            *stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
            *access = src ? VK_ACCESS_2_NONE : VK_ACCESS_2_TRANSFER_READ_BIT;
            return;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            *stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
            *access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            return;
        default:
            *stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            *access = src ? VK_ACCESS_2_MEMORY_WRITE_BIT : (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
            return;
    }
}
}  // namespace

void set_narrow_barriers(bool on) { g_narrow_barriers.store(on, std::memory_order_relaxed); }

void image_barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to, VkImageAspectFlags aspect, uint32_t base_level,
                   uint32_t level_count, uint32_t base_layer, uint32_t layer_count) {
    g_gauge_barriers_img.add(1);
    VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    if (g_narrow_barriers.load(std::memory_order_relaxed)) {
        layout_scope(from, true, &b.srcStageMask, &b.srcAccessMask);
        layout_scope(to, false, &b.dstStageMask, &b.dstAccessMask);
    }
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, base_level, level_count, base_layer, layer_count};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}

namespace {
thread_local std::unordered_map<VkCommandBuffer, uint64_t> t_barriers_in_cb;   // full memory barriers this thread recorded, per command buffer
}
uint64_t memory_barrier_count(VkCommandBuffer cb) {
    const auto it = t_barriers_in_cb.find(cb);
    return it == t_barriers_in_cb.end() ? 0 : it->second;
}

void memory_barrier(VkCommandBuffer cb) {
    g_gauge_barriers_mem.add(1);
    ++t_barriers_in_cb[cb];
    VkMemoryBarrier2 b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}

// ---- submission --------------------------------------------------------------------------------------------------------
namespace {

// `serial`: the one reserved when the job was queued (queue_submit). 0: a submission made inside a render job (texture dumps of the trace):
// it takes the next serial when nothing queued holds one, else it signals nothing and the queue is drained after it.
uint64_t submit_on_render_thread(const std::function<void(Frame&)>& record, uint64_t serial = 0) {
    NXFT("replay");
    nxrt::HitchTimer ht_replay(nxrt::kHpRenderReplay);
    const auto t0 = std::chrono::steady_clock::now();
    Slot& s = acquire_slot();
    bool unsignalled = false;
    if (!serial) {
        std::lock_guard<std::mutex> lk(g->qmu);
        if (g->reserved.load() == g->submitted) serial = g->reserved.fetch_add(1) + 1;
        else { serial = g->submitted; unsignalled = true; }
    }
    pool_copy_begin();
    g->recording = serial;
    Frame f;
    f.cb = s.cb;
    f.serial = serial;
    f.slot = &s;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(s.cb, &bi);
    const uint32_t qi = static_cast<uint32_t>(&s - g->slots) * 2;
    if (g->gpool) {
        vkCmdResetQueryPool(s.cb, g->gpool, static_cast<uint32_t>(&s - g->slots) * kGpMax, kGpMax);
        s.gp_n = 0;
        s.gp_labels.clear();
        gpass_stamp(f, "(start)", 0);
    }
    if (g->qpool) {
        vkCmdResetQueryPool(s.cb, g->qpool, qi, 2);
        vkCmdWriteTimestamp(s.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g->qpool, qi);
        s.timed = true;
        s.cpu_t0 = ft::now_ns();
        s.frame = g->ctr.presents.load(std::memory_order_relaxed);
    }
    bool aborted = false;
    try {
        test_throw(g_test_records);
        record(f);
    } catch (const std::exception& e) {
        // An empty command buffer still signals the serial, so every wait for it (fences, the game's syncs, slot reuse) returns.
        NXLOG_E(kTag, "submission %llu: recording threw (%s): its commands are dropped, an empty submission signals it",
                static_cast<unsigned long long>(serial), e.what());
        replay_abort_submit();
        vkResetCommandPool(g->dev->device(), s.pool, 0);
        vkBeginCommandBuffer(s.cb, &bi);
        s.timed = false;
        s.gp_n = 0;
        s.gp_labels.clear();
        aborted = true;
    }
    if (g->gpool && !aborted) gpass_stamp(f, "tail", 0);
    if (g->qpool && !aborted) vkCmdWriteTimestamp(s.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g->qpool, qi + 1);
    vkEndCommandBuffer(s.cb);
    if (!unsignalled) {
        s.serial = serial;
        g->submitted = serial;
        g->last_submitted.store(serial);
        g->recording = 0;
        g->batch.push_back(&s);
        if (pool_copy_pending()) g_copy_inflight.store(serial);
        if (g->batch.size() >= g->batch_max || pool_copy_pending()) flush_batch();
        if (pool_copy_pending()) {
            wait_serial_blocking(serial);
            pool_copy_end();
            { std::lock_guard<std::mutex> lock(g_copy_done_mu); g_copy_inflight.store(0); }
            g_copy_done_cv.notify_all();
            read_completed();
        }
        // submit_batch = frame: a thread already waits for this serial (it asked before it was recorded): it goes to the GPU now.
        if (g->frame_batch && batch_flush_wanted()) flush_batch();
        g->ctr.submit_replay_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()));
        return serial;
    }
    flush_batch();      // debug path: submitted on its own, after everything recorded before it, signalling nothing
    VkCommandBufferSubmitInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = s.cb;
    VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    VkResult r;
    {
        nxrt::HitchTimer ht(nxrt::kHpRenderSubmit);
        const uint64_t tl = nxrt::hitch_ticks();
        std::lock_guard<std::mutex> lk(g->dev->queue_mutex());
        nxrt::hitch_phase_add(nxrt::kHpRenderLockWait, nxrt::hitch_ticks() - tl);
        r = vkQueueSubmit2(g->dev->queue(), 1, &si, VK_NULL_HANDLE);
        if (r == VK_SUCCESS) vkQueueWaitIdle(g->dev->queue());   // nothing else tells when it is done
    }
    if (pool_copy_pending()) pool_copy_end();
    g->recording = 0;
    s.serial = 0;
    return serial;
}

}  // namespace

void submit_recorded() {
    if (g) flush_batch();
}

bool take_batch_for_present(std::vector<VkCommandBuffer>* cmds, gfx::Renderer::SubmitPrefix* p) {
    if (!g || !g->frame_batch || g->batch.empty()) {
        submit_recorded();
        return false;
    }
    cmds->clear();
    for (Slot* s : g->batch) cmds->push_back(s->cb);
    p->cmds = cmds->data();
    p->count = static_cast<uint32_t>(cmds->size());
    p->timeline = g->timeline;
    p->value = g->batch.back()->serial;
    return true;
}

void present_batch_done(const gfx::Renderer::SubmitPrefix& p) {
    if (!p.submitted) {   // as flush_batch: keep the timeline moving so waiters do not hang
        VkSemaphoreSignalInfo ssi = {VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        ssi.semaphore = g->timeline;
        ssi.value = p.value;
        vkSignalSemaphore(g->dev->device(), &ssi);
    }
    g->flushed = p.value;
    g->flushed_pub.store(p.value);
    g->batch.clear();
    g->ctr.submits.fetch_add(1);
}

// Queues a submission with the next serial (in queue order). wait: until the render thread has run it (and, with gpu_done, the GPU too).
uint64_t queue_submit(std::function<void(Frame&)> record, bool wait, bool gpu_done) {
    if (on_render_thread()) {
        const uint64_t serial = submit_on_render_thread(record);
        if (gpu_done) { wait_serial_blocking(serial); retire(); }
        return serial;
    }
    struct Done { std::mutex m; std::condition_variable cv; bool done = false; };
    auto done = wait ? std::make_shared<Done>() : nullptr;
    uint64_t serial = 0;
    {
        std::lock_guard<std::mutex> lk(g->qmu);
        serial = g->reserved.fetch_add(1) + 1;
        g->queue.push_back([record = std::move(record), serial, done, gpu_done] {
            auto signal = on_exit([&] {
                if (!done) return;
                std::lock_guard<std::mutex> l2(done->m);
                done->done = true;
                done->cv.notify_all();
            });
            submit_on_render_thread(record, serial);
            if (gpu_done) { wait_serial_blocking(serial); retire(); }
        });
        g_gauge_render_queue.add(1);
    }
    g->qcv.notify_one();
    if (done) {
        std::unique_lock<std::mutex> lk(done->m);
        done->cv.wait(lk, [&] { return done->done; });
    }
    return serial;
}

uint64_t run_frame_sync(const std::function<void(Frame&)>& fn) {
    return queue_submit(fn, true, true);
}

// ---- gpu.h ---------------------------------------------------------------------------------------------------------------
bool active() { return g != nullptr; }
void cpu_will_read(uint64_t addr, uint64_t size) {
    if (g && out_any()) out_fetch(addr, size, false);
}
bool debug_labels() { return g && g->dev && g->dev->features().debug_labels; }

namespace {
std::atomic<uint64_t> g_idle_serial{0};   // without a back end submissions still get serials and complete at once (tests, tools)
}

// [nvn] async_submit (default on): the game thread queues its command lists and goes on; the render thread records and submits them in
// order. The serial is known at once (reserved in queue order), so sync objects and waits work as before: a wait for a serial the render
// thread has not submitted yet waits on the timeline until it has (wait-before-signal).
bool async_submit() {
    static const bool on = nxrt::config_get_bool("nvn", "async_submit", true);
    return on;
}

uint64_t submit(const std::vector<Ref<CmdList>>& lists) {
    if (!g) return ++g_idle_serial;
    NXFT("submit");
    return queue_submit([lists](Frame& f) {
        replay_begin_submit(f);
        replay_lists(f, lists);
        replay_end_submit(f);
    }, !async_submit(), false);
}

uint64_t flush() { return g ? g->reserved.load() : g_idle_serial.load(); }
uint64_t last_serial() { return g ? g->reserved.load() : g_idle_serial.load(); }

void finish() {
    if (!g) return;
    run_on_render_thread([] {
        wait_serial_blocking(g->submitted);
        retire();
    });
}

bool serial_done(uint64_t serial) {
    if (!g || !serial) return true;
    if (g->completed.load() >= serial) return true;
    if (read_completed() >= serial) return true;
    want_flush(serial);     // polled: it must reach the GPU to ever complete
    return false;
}

namespace {
std::atomic<uint64_t> g_waits_begun{0}, g_waits_done{0};   // sync waits entered and left: a live graphics thread moves both
std::atomic<int> g_waiters{0};                 // game threads inside wait_serial (freeze report)
std::atomic<uint64_t> g_wait_serial{0}, g_wait_timeout{0};   // the latest wait: serial and timeout

std::string gpu_report() {
    if (!g) return {};
    uint64_t v = 0;
    if (g->timeline) vkGetSemaphoreCounterValue(g->dev->device(), g->timeline, &v);
    char b[400];
    size_t q = 0;
    bool got = false;
    if (g->qmu.try_lock()) { q = g->queue.size(); got = true; g->qmu.unlock(); }
    std::snprintf(b, sizeof b, "NVN back end: last submitted serial %llu, timeline value %llu, %d game thread(s) in a sync wait (latest wait: serial %llu, timeout %llu ns), sync waits begun %llu done %llu, submits %llu, presents %llu, render queue %s%zu\n",
                  static_cast<unsigned long long>(g->last_submitted.load()), static_cast<unsigned long long>(v), g_waiters.load(),
                  static_cast<unsigned long long>(g_wait_serial.load()), static_cast<unsigned long long>(g_wait_timeout.load()),
                  static_cast<unsigned long long>(g_waits_begun.load()), static_cast<unsigned long long>(g_waits_done.load()),
                  static_cast<unsigned long long>(g->ctr.submits.load()), static_cast<unsigned long long>(g->ctr.presents.load()), got ? "" : "(busy) ", q);
    return b;
}
}  // namespace

int presents_since(uint64_t serial);   // below, beside the frame history

bool wait_serial(uint64_t serial, uint64_t timeout_ns) {
    if (!g || !serial) return true;
    if (serial_done(serial)) return true;
    if (timeout_ns == 0) return false;
    want_flush(serial);
    NXFT("sync.wait");
    nxrt::HitchTimer ht(nxrt::kHpGameSyncWait);
    g_wait_serial.store(serial);
    g_wait_timeout.store(timeout_ns);
    struct Counted { Counted() { g_waiters++; g_waits_begun++; } ~Counted() { g_waiters--; g_waits_done++; } } counted;
    // Where a game sync wait spends its time (log line every 10 s): entered while the render thread had not recorded the fenced work yet
    // ("replay": the wait covers replay + GPU), or after ("gpu": the GPU alone). Submissions behind = how far the fence is past what is recorded.
    static std::atomic<uint64_t> w_n[2], w_us[2], w_behind{0}, w_last_ms{0}, w_age_n[4], w_age_us[4];
    const uint64_t rec = g->last_submitted.load(std::memory_order_relaxed);
    const int kind = serial > rec ? 0 : 1;
    const int age = presents_since(serial);
    if (kind == 0) w_behind.fetch_add(serial - rec, std::memory_order_relaxed);
    const auto tw0 = std::chrono::steady_clock::now();
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &g->timeline;
    wi.pValues = &serial;
    VkResult r;
    if (!g->frame_batch) {
        r = vkWaitSemaphores(g->dev->device(), &wi, timeout_ns);
    } else {
        // submit_batch = frame: waited in slices, asking again for the flush between them, so a request the render thread missed
        // costs 50 ms instead of the game.
        constexpr uint64_t kSliceNs = 50'000'000;
        const auto deadline = timeout_ns == UINT64_MAX ? std::chrono::steady_clock::time_point::max()
                                                       : tw0 + std::chrono::nanoseconds(std::min<uint64_t>(timeout_ns, INT64_MAX / 2));
        for (;;) {
            uint64_t slice = kSliceNs;
            if (timeout_ns != UINT64_MAX) {
                const auto left = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - std::chrono::steady_clock::now()).count();
                slice = left <= 0 ? 0 : std::min<uint64_t>(kSliceNs, static_cast<uint64_t>(left));
            }
            r = vkWaitSemaphores(g->dev->device(), &wi, slice);
            if (r != VK_TIMEOUT || slice < kSliceNs) break;
            want_flush(serial, true);
        }
    }
    if (r == VK_SUCCESS && !on_render_thread()) {
        std::unique_lock<std::mutex> lock(g_copy_done_mu);
        auto ready = [serial] { const auto copying = g_copy_inflight.load(); return !copying || copying > serial; };
        if (timeout_ns == UINT64_MAX) g_copy_done_cv.wait(lock, ready);
        else {
            const auto deadline = tw0 + std::chrono::nanoseconds(std::min<uint64_t>(timeout_ns, INT64_MAX));
            if (!g_copy_done_cv.wait_until(lock, deadline, ready)) r = VK_TIMEOUT;
        }
    }
    read_completed();
    const auto tw1 = std::chrono::steady_clock::now();
    w_n[kind].fetch_add(1, std::memory_order_relaxed);
    const uint64_t us_w = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(tw1 - tw0).count());
    w_us[kind].fetch_add(us_w, std::memory_order_relaxed);
    w_age_n[age].fetch_add(1, std::memory_order_relaxed);
    w_age_us[age].fetch_add(us_w, std::memory_order_relaxed);
    const uint64_t now_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(tw1.time_since_epoch()).count());
    uint64_t last = w_last_ms.load(std::memory_order_relaxed);
    if (!last) w_last_ms.compare_exchange_strong(last, now_ms);
    else if (now_ms - last >= 10000 && w_last_ms.compare_exchange_strong(last, now_ms)) {
        const uint64_t n0 = w_n[0].exchange(0), n1 = w_n[1].exchange(0), u0 = w_us[0].exchange(0), u1 = w_us[1].exchange(0), b = w_behind.exchange(0);
        NXLOG_I(kTag, "sync waits, last %.1f s: behind the render thread %llu (%.1f ms total, %.2f submissions behind on average), on the GPU only %llu (%.1f ms total)",
                (now_ms - last) / 1000.0, static_cast<unsigned long long>(n0), u0 / 1000.0, n0 ? static_cast<double>(b) / n0 : 0.0,
                static_cast<unsigned long long>(n1), u1 / 1000.0);
        uint64_t an[4], au[4];
        for (int i = 0; i < 4; ++i) { an[i] = w_age_n[i].exchange(0); au[i] = w_age_us[i].exchange(0); }
        NXLOG_I(kTag, "sync waits by fence age (presents since it was queued): 0: %llu (%.1f ms), 1: %llu (%.1f ms), 2: %llu (%.1f ms), 3+: %llu (%.1f ms)",
                static_cast<unsigned long long>(an[0]), au[0] / 1000.0, static_cast<unsigned long long>(an[1]), au[1] / 1000.0,
                static_cast<unsigned long long>(an[2]), au[2] / 1000.0, static_cast<unsigned long long>(an[3]), au[3] / 1000.0);
    }
    return r == VK_SUCCESS;
}

namespace {
// The last serial queued before each of the game's presents (the end of that frame's GPU work), for the low-latency GPU wait (lowlat.h).
// Written by the game thread in present(), read by the main thread at the frame boundary (the same thread in Pokopia).
std::atomic<uint64_t> g_frame_work[4];
std::atomic<uint64_t> g_frame_count{0};

bool gpu_wait_frames(int frames_back, uint64_t timeout_ns) {
    const uint64_t n = g_frame_count.load(std::memory_order_acquire);
    if (!g || n <= static_cast<uint64_t>(frames_back)) return true;
    const uint64_t serial = g_frame_work[(n - 1 - static_cast<uint64_t>(frames_back)) & 3].load(std::memory_order_relaxed);
    NXFT("lowlat.gpu_wait");
    return wait_serial(serial, timeout_ns);
}
}  // namespace

// Presents the game made after `serial` was queued (0: the fence is in the frame being built; 3: three or more frames back).
int presents_since(uint64_t serial) {
    const uint64_t n = g_frame_count.load(std::memory_order_acquire);
    int k = 0;
    while (k < 3 && static_cast<uint64_t>(k) < n && g_frame_work[(n - 1 - static_cast<uint64_t>(k)) & 3].load(std::memory_order_relaxed) >= serial) ++k;
    return k;
}

// [nvn] present_ahead (default 1; 0 = wait for each present): the game thread queues the present and goes on while fewer than this many
// presents are still pending on the render thread (the frame limiter and the swapchain's frame slots pace the render thread). 1 keeps the
// render thread one present behind the game at most (2 let a second finished frame wait in the queue: one more frame between input and screen).
void present(Window* window, int index) {
    if (!g || !window) return;
    NXFT("present.call");
    nxrt::hitch_note_game_present();
    {
        std::lock_guard<std::mutex> lk(g->pmu);
        g->last_window = Ref<Window>::share(window);
        g->last_index = index;
        g->last_present = std::chrono::steady_clock::now();
    }
    {
        const uint64_t n = g_frame_count.load(std::memory_order_relaxed);
        g_frame_work[n & 3].store(g->reserved.load(), std::memory_order_relaxed);
        g_frame_count.store(n + 1, std::memory_order_release);
    }
    static const int ahead = static_cast<int>(std::clamp<int64_t>(nxrt::config_get_int("nvn", "present_ahead", 1), 0, 3));
    if (ahead == 0 || on_render_thread()) {
        Ref<Window> keep = Ref<Window>::share(window);
        run_on_render_thread([&] { present_on_render_thread(window, index); });
        return;
    }
    {
        NXFT("present.backpressure");
        nxrt::HitchTimer ht(nxrt::kHpGameBackpressure);
        std::unique_lock<std::mutex> lk(g->pmu);
        g->pcv.wait(lk, [] { return g->presents_pending < ahead; });
        ++g->presents_pending;
    }
    Ref<Window> keep = Ref<Window>::share(window);
    {
        std::lock_guard<std::mutex> lk(g->qmu);
        g->queue.push_back([keep, index] {
            auto release = on_exit([] {
                std::lock_guard<std::mutex> l2(g->pmu);
                --g->presents_pending;
                g->pcv.notify_all();
            });
            test_throw(g_test_presents);
            present_on_render_thread(keep.get(), index);
        });
        g_gauge_render_queue.add(1);
    }
    g->qcv.notify_one();
}

// The game's last frame once more, queued like a present (the overlay is drawn on top as on every frame). Only when the game has not
// presented for 50 ms and nothing is queued, so it never competes with the game's own frames.
void present_again() {
    if (!g) return;
    Ref<Window> keep;
    int index = -1;
    {
        std::lock_guard<std::mutex> lk(g->pmu);
        if (!g->last_window || g->presents_pending > 0) return;
        if (std::chrono::steady_clock::now() - g->last_present < std::chrono::milliseconds(50)) return;
        keep = g->last_window;
        index = g->last_index;
        ++g->presents_pending;
    }
    {
        std::lock_guard<std::mutex> lk(g->qmu);
        g->queue.push_back([keep, index] {
            auto release = on_exit([] {
                std::lock_guard<std::mutex> l2(g->pmu);
                --g->presents_pending;
                g->pcv.notify_all();
            });
            present_on_render_thread(keep.get(), index, true);
        });
        g_gauge_render_queue.add(1);
    }
    g->qcv.notify_one();
}

void destroy_texture_gpu(TexGpu* t) {
    if (!t) return;
    if (!g) { tex_destroy(t); return; }
    defer_destroy([t] { tex_destroy(t); });
}

void release_texture_gpu(Texture* t) {
    if (!g || !t) return;
    Ref<Texture> keep = Ref<Texture>::share(t);
    {
        std::lock_guard<std::mutex> lk(g->qmu);
        if (g->stop) return;
        g->queue.push_back([keep] {
            Texture* x = keep.get();
            if (!x->destroyed.load() || !x->gpu) return;   // initialized again meanwhile, or never got an image
            TexGpu* img = x->gpu;
            x->gpu = nullptr;
            pool_texture_remade(x);
            destroy_texture_gpu(img);
        });
        g_gauge_render_queue.add(1);
    }
    g->qcv.notify_one();
}

void destroy_pool_gpu(PoolGpu* p) {
    pool_copy_retire(p);
    if (!p) return;
    if (!g) { pool_gpu_destroy(p); return; }
    defer_destroy([p] { pool_gpu_destroy(p); });
}

namespace {
std::atomic<int> g_origin{NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT}, g_depth_mode{NVN_DEPTH_MODE_NEAR_IS_MINUS_W};
}
void set_window_modes(int origin_mode, int depth_mode) { g_origin.store(origin_mode); g_depth_mode.store(depth_mode); }
int window_origin_mode() { return g_origin.load(); }
int depth_mode() { return g_depth_mode.load(); }
gfx::Renderer* renderer() { return g ? g->rend : nullptr; }
FrameLimiter* limiter() { return g ? g->limiter : nullptr; }
PresentPass* present_pass() { return g && g->present_ok ? &g->present : nullptr; }
extras::OutputPass* output_pass() { return g && g->output_ok ? &g->output : nullptr; }
bool keep_aspect() { return g && g->keep_aspect; }

uint64_t timestamp_ns() {
    static const auto t0 = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()) + 1000000000ull;
}

std::string unsupported_report() {
    std::string out;
    if (!g) return out;
    std::lock_guard<std::mutex> lk(g->unsupported_mu);
    for (auto& [name, n] : g->unsupported) out += name + ": " + std::to_string(n) + "\n";
    return out;
}

// ---- lifecycle -----------------------------------------------------------------------------------------------------------
namespace {

void register_counter_gauges();

bool bring_up(gfx::Device& dev, gfx::Renderer* rend, FrameLimiter* limiter, bool keep_aspect) {
    std::lock_guard<std::mutex> lk(g_install_mu);
    if (g) return g->dev == &dev;
    auto* gp = new Gpu();
    gp->dev = &dev;
    gp->rend = rend;
    gp->limiter = limiter;
    gp->keep_aspect = keep_aspect;
    {   // [nvn] submit_batch: auto = frame under Wine (each kernel submission walks the imported pools again, flush_batch), 1 elsewhere.
        // frame: one kernel submission per frame, the present's; N: flush every N submissions and whenever the render thread idles.
        std::string b = nxrt::config_get_str("nvn", "submit_batch", "auto");
        HMODULE nt = GetModuleHandleW(L"ntdll.dll");
        const bool wine = nt && GetProcAddress(nt, "wine_get_version");
        const std::string how = b == "auto" ? (wine ? "auto, Wine" : "auto, not Wine") : b;
        if (b == "auto") b = wine ? "frame" : "1";
        gp->frame_batch = b == "frame";
        gp->batch_max = gp->frame_batch ? kSlots / 2 : static_cast<uint32_t>(std::clamp<int64_t>(std::atoll(b.c_str()), 1, kSlots / 2));
        NXLOG_I(kTag, "submit_batch %s (%s)", gp->frame_batch ? "frame" : std::to_string(gp->batch_max).c_str(), how.c_str());
    }
    g = gp;
    if (!dev.features().timeline_semaphore) {
        NXLOG_E(kTag, "the GPU has no timeline semaphores: the NVN back end cannot run");
        g = nullptr;
        delete gp;
        return false;
    }
    VkSemaphoreTypeCreateInfo tci = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &tci};
    if (vkCreateSemaphore(dev.device(), &sci, nullptr, &gp->timeline) != VK_SUCCESS) {
        g = nullptr;
        delete gp;
        return false;
    }
    vk_gauge_semaphores().add(1);
    for (Slot& s : gp->slots)
        if (!create_slot(s)) {
            NXLOG_E(kTag, "creating a submission slot failed");
            g = nullptr;
            delete gp;
            return false;
        }
    gp->stats.every = std::max<int64_t>(0, config_get_int("prof", "gpu_stats", 0));
    if (ft::on() || gp->stats.every > 0) {
        VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kSlots * 2;
        if (vkCreateQueryPool(dev.device(), &qi, nullptr, &gp->qpool) != VK_SUCCESS) gp->qpool = VK_NULL_HANDLE;
        else vk_gauge_query_pools().add(1);
    }
    if ((ft::on() || gp->stats.every > 0) && config_get_bool("prof", "gpu_passes", false)) {
        VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kSlots * kGpMax;
        if (vkCreateQueryPool(dev.device(), &qi, nullptr, &gp->gpool) != VK_SUCCESS) gp->gpool = VK_NULL_HANDLE;
        else vk_gauge_query_pools().add(1);
    }
    if (ft::on()) extras::lowlat_set_backlog_fn([]() -> uint64_t {      // latency probe: submissions the GPU has not finished
        if (!g || !g->timeline) return 0;
        uint64_t v = 0;
        vkGetSemaphoreCounterValue(g->dev->device(), g->timeline, &v);
        const uint64_t sub = g->last_submitted.load();
        return sub > v ? sub - v : 0;
    });
    extras::lowlat_set_gpu_wait_fn(&gpu_wait_frames);
    static const bool report_registered = (diag_add_report(gpu_report), true);   // the freeze dump shows the GPU state
    (void)report_registered;
    register_counter_gauges();
    gp->th = std::thread(render_loop);
    // wait for the thread id to be published
    for (int i = 0; i < 1000 && gp->tid == std::thread::id(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (rend) {
        run_on_render_thread([&] { gp->present_ok = gp->present.init(dev, rend->format()); });
        if (!gp->present_ok) NXLOG_E(kTag, "present pass failed to initialise: frames will not be shown");
        run_on_render_thread([&] { gp->output_ok = gp->output.init(dev); });
        if (!gp->output_ok) NXLOG_W(kTag, "output pass (HDR, FSR 1) failed to initialise: those extras are off");
    }
    NXLOG_I(kTag, "NVN back end up on %s (%u submission slots, timeline semaphore, host-memory import %s)", dev.properties().deviceName, kSlots,
            dev.features().external_memory_host ? "available" : "NOT available");
    return true;
}

const char* const kCounterGauges[] = {"nvn.cnt.draws", "nvn.cnt.passes", "nvn.cnt.pass_breaks", "nvn.cnt.dispatches", "nvn.cnt.clears", "nvn.cnt.copies",
                                      "nvn.cnt.submits", "nvn.cnt.descriptor_sets"};

void register_counter_gauges() {
    auto reg = [](const char* name, std::atomic<uint64_t> Counters::*m) {
        nxrt::gauge_register(name, [m]() -> int64_t { Gpu* gp = g; return gp ? static_cast<int64_t>((gp->ctr.*m).load(std::memory_order_relaxed)) : 0; });
    };
    reg("nvn.cnt.draws", &Counters::draws);
    reg("nvn.cnt.passes", &Counters::passes);
    reg("nvn.cnt.pass_breaks", &Counters::pass_breaks);
    reg("nvn.cnt.dispatches", &Counters::dispatches);
    reg("nvn.cnt.clears", &Counters::clears);
    reg("nvn.cnt.copies", &Counters::copies);
    reg("nvn.cnt.submits", &Counters::submits);
    reg("nvn.cnt.descriptor_sets", &Counters::descriptor_sets);
    // What the hitch log (nxrt/gauge.h) shows for a slow frame: growth of these during the frame, read lock-free on the presenting thread.
    static const bool hitch_registered = [] {
        nxrt::hitch_add_counter("pipelines_created", []() -> uint64_t { Gpu* gp = g; return gp ? gp->ctr.pipelines_created.load(std::memory_order_relaxed) : 0; });
        nxrt::hitch_add_counter("texture_uploads", []() -> uint64_t { Gpu* gp = g; return gp ? gp->ctr.texture_uploads.load(std::memory_order_relaxed) : 0; });
        nxrt::hitch_add_counter("texture_upload_kb", []() -> uint64_t { Gpu* gp = g; return gp ? gp->ctr.texture_upload_bytes.load(std::memory_order_relaxed) >> 10 : 0; });
        nxrt::hitch_add_counter("bc7_transcodes", []() -> uint64_t { Gpu* gp = g; return gp ? gp->ctr.bc7_transcodes.load(std::memory_order_relaxed) : 0; });
        nxrt::hitch_add_level("pipeline_compiles_in_flight", []() -> int64_t { return compiles_in_flight().load(std::memory_order_relaxed); });
        return true;
    }();
    (void)hitch_registered;
}

void tear_down() {
    std::lock_guard<std::mutex> lk(g_install_mu);
    if (!g) return;
    for (const char* n : kCounterGauges) nxrt::gauge_unregister(n);
    Gpu* gp = g;
    run_on_render_thread([&] {
        {
            std::lock_guard<std::mutex> l2(gp->pmu);
            gp->last_window = nullptr;
        }
        wait_serial_blocking(gp->submitted);
        retire();
        replay_shutdown();
        tex_shutdown();
        pool_shutdown();
        while (!gp->deferred.empty()) {
            auto fn = std::move(gp->deferred.front().fn);
            gp->deferred.pop_front();
            g_gauge_deferred.sub(1);
            fn();
        }
        drain_pending_destroy();
        for (auto& d : gp->deferred) { d.fn(); g_gauge_deferred.sub(1); }
        gp->deferred.clear();
        gp->present.shutdown();
        gp->output.shutdown();
    });
    {
        std::lock_guard<std::mutex> ql(gp->qmu);
        gp->stop = true;
    }
    gp->qcv.notify_all();
    if (gp->th.joinable()) gp->th.join();
    gfx::Device& d = *gp->dev;
    vkDeviceWaitIdle(d.device());
    for (Slot& s : gp->slots) {
        for (auto& b : s.extra) d.destroy_buffer(b);
        vk_gauge_descriptor_sets().sub(static_cast<int64_t>(s.sets_live));
        s.sets_live = 0;
        for (VkDescriptorPool p : s.dpools) {
            vkDestroyDescriptorPool(d.device(), p, nullptr);
            vk_gauge_descriptor_pools().sub(1);
        }
        d.destroy_buffer(s.upload);
        if (s.pool) {
            vkDestroyCommandPool(d.device(), s.pool, nullptr);
            vk_gauge_command_pools().sub(1);
            if (s.cb) vk_gauge_command_buffers().sub(1);
        }
    }
    if (gp->qpool) { vkDestroyQueryPool(d.device(), gp->qpool, nullptr); vk_gauge_query_pools().sub(1); }
    if (gp->gpool) { vkDestroyQueryPool(d.device(), gp->gpool, nullptr); vk_gauge_query_pools().sub(1); }
    if (gp->timeline) { vkDestroySemaphore(d.device(), gp->timeline, nullptr); vk_gauge_semaphores().sub(1); }
    g = nullptr;
    delete gp;
}

}  // namespace

bool init_for_test(gfx::Device& dev, gfx::Renderer* renderer) { return bring_up(dev, renderer, nullptr, false); }
void shutdown_for_test() { tear_down(); }

}  // namespace nxrt::nvn::be::gpu

namespace nxrt::nvn {

bool install_backend(const BackendDesc& desc) {
    if (!desc.device) return false;
    if (!be::gpu::bring_up(*desc.device, desc.renderer, desc.limiter, desc.keep_aspect)) return false;
    if (desc.register_entry_points) {
        install();
        register_all_impls();
        NXLOG_I("nvn", "NVN back end installed: %zu of %zu entry points implemented", implemented_count(), entry_count());
    }
    return true;
}

void shutdown_backend() { be::gpu::tear_down(); }
void present_last_frame_again() {
    if (be::gpu::active()) be::gpu::present_again();
}

bool prepare_shaders(const ShaderPrepareOptions& o, ShaderPrepareResult* r) {
    ShaderPrepareResult local;
    ShaderPrepareResult& out = r ? *r : local;
    if (!be::gpu::active()) { out = ShaderPrepareResult{}; out.error = "the graphics back end is not running"; return false; }
    return be::gpu::gpl_prepare(o, &out);
}
bool backend_installed() { return be::gpu::active(); }
void backend_flush_caches() {
    if (be::gpu::active()) be::gpu::flush_caches();
}

void backend_trace_next_frame() {
    if (be::gpu::active()) be::gpu::replay_request_trace();
}

BackendStats backend_stats() {
    BackendStats s;
    if (!be::gpu::active()) return s;
    auto& c = be::gpu::counters();
    s.submits = c.submits; s.lists_replayed = c.lists_replayed; s.draws = c.draws; s.draws_skipped = c.draws_skipped;
    s.dispatches = c.dispatches; s.clears = c.clears; s.copies = c.copies; s.presents = c.presents;
    s.passes = c.passes; s.pass_breaks = c.pass_breaks; s.pipelines_created = c.pipelines_created;
    s.pipeline_cache_hits = c.pipeline_cache_hits; s.descriptor_sets = c.descriptor_sets;
    s.textures_created = c.textures_created; s.texture_uploads = c.texture_uploads; s.texture_upload_bytes = c.texture_upload_bytes;
    s.bc7_transcodes = c.bc7_transcodes; s.textures_failed = c.textures_failed; s.pools_imported = c.pools_imported;
    s.unresolved_addresses = c.unresolved_addresses; s.missing_handles = be::handles_missing(); s.unsupported_ops = c.unsupported_ops;
    s.submit_replay_ns = c.submit_replay_ns;
    return s;
}

std::string backend_unsupported_report() { return be::gpu::unsupported_report(); }

}  // namespace nxrt::nvn
