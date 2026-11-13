// SPDX-License-Identifier: GPL-3.0-or-later
// Background preparation of texture upload data. See nvn_texprep.h for the design and the correctness argument.
#include "nvn_texprep.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "nxrt/config.h"
#include "nxrt/ftrace.h"
#include "nxrt/gauge.h"
#include "nxrt/log.h"
#include "nvn_gpu_internal.h"

namespace nxrt::nvn::be::texprep {
namespace {

constexpr const char* kTag = "nvn.texprep";
constexpr uint8_t kUploaded = 1, kWanted = 4;   // kWanted: a texture pool or a command list has named the texture
constexpr uint64_t kDebounceNs = 40ull * 1000000ull;   // sparse textures: a job starts when no write was reported for this long

struct Entry {
    enum State { Idle, Waiting, Queued, Running, Done } st = Idle;
    uint64_t due_ns = 0;                     // Waiting: when the job may start
    Ref<Texture> keep;                       // alive while a job or a result exists
    uint32_t job_gen = 0;                    // prep_gen the job read the storage at
    uint64_t src_hash = 0;                   // hash of the storage the job read (before and after its work: equal, or the result is dropped)
    uint64_t seq = 0;                        // identifies the queued job (bumped to cancel it)
    bool again = false;                      // written again while the job ran: queue another when it ends
    bool slow = false;                       // Running: the job found no cache entry and transcodes (take does not wait for it)
    uint64_t held = 0;                       // bytes accounted for this entry
    std::unique_ptr<PreparedTexture> res;
    gfx::Buffer stage;                       // big results: the data already copied into a host-visible buffer for the upload (res->data is then empty)
};

struct Job {
    Ref<Texture> t;
    uint64_t seq;
};

struct State {
    std::mutex mu;
    std::condition_variable cv_work, cv_done;
    std::unordered_map<Texture*, Entry> map;
    std::deque<Job> q;
    std::vector<Texture*> waiting;           // textures in the Waiting state
    uint64_t seq = 0;
    uint64_t held_bytes = 0;
    uint64_t budget = 0;
    uint64_t min_bytes = 0;                  // smaller textures are prepared inline: the hand-over costs more than the work
    uint64_t stage_bytes = 0;                // results from this size are copied into an upload buffer by the job (0: never)
    std::atomic<bool> quiesced{false};       // the GPU layer is shutting down: no more upload buffers
    uint64_t early_bytes = 0;                // textures from this size are also prepared when they are registered in a texture pool (the others only when bound)
    bool verify = false;
    TexCaps caps;
    Stats st;
    uint64_t last_log_ns = 0;
    std::atomic<bool> caps_ready{false};
};

std::atomic<uint64_t> s_transcoded{0};   // ASTC -> BC7 transcodes made by jobs (no cache entry)
std::atomic<uint64_t> s_provisional{0};  // quick stand-ins the render thread uploaded while a job made the real data
std::atomic<uint64_t> s_kept{0};         // uploads skipped because the image holds a stand-in of this content and its job is not done (standin_kept)
const bool g_hitch_kept_registered = (nxrt::hitch_add_counter("standins_kept", []() -> uint64_t { return s_kept.load(std::memory_order_relaxed); }), true);

// The image holds a stand-in of the texture's current content: its job's result is promised to it (not evicted for the budget).
bool promised(const Texture* t, uint32_t job_gen) {
    return t->prep_prov.load(std::memory_order_acquire) == (job_gen | 0x80000000u) && job_gen == t->prep_gen.load(std::memory_order_acquire);
}
// the hitch log: transcodes the jobs finished and stand-ins uploaded during a slow frame
const bool g_hitch_tc_registered = (nxrt::hitch_add_counter("texprep_transcodes", []() -> uint64_t { return s_transcoded.load(std::memory_order_relaxed); }),
                                    nxrt::hitch_add_counter("texture_standins", []() -> uint64_t { return s_provisional.load(std::memory_order_relaxed); }), true);

State* g_state = nullptr;
std::once_flag g_once;
std::atomic<int> g_enabled{-1};
nxrt::GaugeCounter g_staged_bytes_total("nvn.texprep.staged_bytes_total");   // bytes of prepared results copied into upload buffers, ever
// the hitch log (nxrt/gauge.h): prepared texture data copied into upload buffers during a slow frame
const bool g_hitch_registered = (nxrt::hitch_add_counter("texprep_staged_kb", []() -> uint64_t { return static_cast<uint64_t>(g_staged_bytes_total.get()) >> 10; }), true);

State& S() { return *g_state; }

// [nvn] tex_async (nvn_texprep.h): standin (default) or off. Read once.
bool async_standin_cfg() {
    static const bool on = [] {
        std::string v = config_get_str("nvn", "tex_async", "standin");
        for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        const bool off = v == "off" || v == "0" || v == "false" || v == "sync";
        NXLOG_I(kTag, "[nvn] tex_async = %s: %s", off ? "off" : "standin",
                off ? "a texture needed before its background job is done is prepared on the render thread (cache read or transcode)"
                    : "a texture needed before its background job is done gets a quick stand-in; the render thread does not read the BC7 cache");
        return !off;
    }();
    return on;
}

thread_local bool t_exact = false;   // set_exact_thread: the caller copies or clears the texture (no stand-in)

// How long the render thread waits for a job that is already running on the texture it needs. [nvn] tex_prep_wait_ms; default 5000 (the
// old behaviour) with tex_async = off, 2 ms with stand-ins (a running job is mostly a disk read: on a hard disk that is 30 ms to over a second).
std::chrono::steady_clock::duration wait_cap() {
    static const int64_t ms = config_get_int("nvn", "tex_prep_wait_ms", async_standin_cfg() ? 2 : 5000);
    return std::chrono::milliseconds(std::clamp<int64_t>(ms, 0, 5000));
}

bool prepare_guarded(const Texture& t, const TexCaps& caps, PreparedTexture& out, bool& crashed) {
    __try {
        return prepare_texture(t, caps, out);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        crashed = true;
        return false;
    }
}

// Reads the storage under SEH: a page the game unmapped must not take the process down.
bool hash_guarded(const Texture& t, uint64_t& h) {
    __try {
        if (t.mem) {
            h = hash_bytes(t.mem, static_cast<size_t>(t.storage_size));
            return true;
        }
        return texture_hash_virtual(t, h);       // a sparse texture: the mapped tiles, hashed where they are
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

uint64_t now_ns() { return ft::now_ns(); }

// Caller holds mu.
void drop_result(Entry& e) {
    S().held_bytes -= std::min(S().held_bytes, e.held);
    e.held = 0;
    e.res.reset();
    if (e.stage) gpu::device().destroy_buffer(e.stage);
    e.stage = gfx::Buffer{};
}

void log_summary_locked(bool force) {
    State& s = S();
    const uint64_t n = now_ns();
    if (!force && n - s.last_log_ns < 10ull * 1000000000ull) return;
    if (s.st.scheduled == 0) return;
    s.last_log_ns = n;
    NXLOG_I(kTag, "scheduled %llu, used %llu (waited for %llu), stale %llu, taken back by the render thread %llu, stand-in while queued or running %llu, no job %llu, failed %llu, crashed %llu, over budget %llu, evicted %llu, held %.1f MB",
            static_cast<unsigned long long>(s.st.scheduled), static_cast<unsigned long long>(s.st.hits), static_cast<unsigned long long>(s.st.waited_hits),
            static_cast<unsigned long long>(s.st.stale), static_cast<unsigned long long>(s.st.stolen), static_cast<unsigned long long>(s.st.deferred),
            static_cast<unsigned long long>(s.st.none),
            static_cast<unsigned long long>(s.st.failed), static_cast<unsigned long long>(s.st.crashes), static_cast<unsigned long long>(s.st.skipped_budget),
            static_cast<unsigned long long>(s.st.evicted), static_cast<double>(s.held_bytes) / 1048576.0);
    if (s.verify)
        NXLOG_I(kTag, "verify: %llu results compared with a fresh preparation, %llu differed", static_cast<unsigned long long>(s.st.verified), static_cast<unsigned long long>(s.st.mismatches));
}

void leave_waiting_locked(Texture* t) {
    auto& w = S().waiting;
    w.erase(std::remove(w.begin(), w.end(), t), w.end());
}

// Caller holds mu. Queues a job for t unless one is waiting; a running job is asked to run again.
// front: the texture is on screen with a stand-in now: its job goes before the textures only registered ahead of use.
// force: queued even over the budget (a stand-in waits for this job: dropping it left the stand-in in place until the next write).
// evict: finished results may be dropped for its budget (a job for a texture about to be used); a job only started ahead of use is not.
void schedule_locked(Texture* t, bool front = false, bool force = false, bool evict = true) {
    State& s = S();
    Entry& e = s.map[t];
    if (e.st == Entry::Queued) return;                // the job reads the generation when it starts
    if (e.st == Entry::Running) { e.again = true; return; }
    if (e.st == Entry::Waiting) { leave_waiting_locked(t); e.st = Entry::Idle; }
    if (e.st == Entry::Done) { drop_result(e); e.st = Entry::Idle; }   // its content is older than the new write
    uint64_t est = std::min<uint64_t>(t->storage_size, 1ull << 30);
    // [gfx] astc_quality = uncompressed: a job holds the texture as RGBA8, 4 bytes a texel against 16 per ASTC block
    if (t->fmt && t->fmt->astc && !S().caps.astc && astc_quality() == AstcQuality::Uncompressed) est = std::min<uint64_t>(est * t->fmt->block_w * t->fmt->block_h / 4, 1ull << 30);
    // Over budget: the oldest finished results go first (they belong to textures not used yet); those are prepared on the render thread if needed.
    // [nvn] tex_prep_keep_promised = true (default): a result a stand-in on screen waits for is never evicted (the 4K first visit of x2-explore
    // evicted 3623 results, and 1243 of their textures were then read from the cache by the render thread, 3.4 s in one frame), and a job only
    // started ahead of use (pool registration) never evicts a finished result: over the budget it is not started (the texture gets a stand-in
    // and a job of its own when it is drawn). false: any finished result may go, for any job (before).
    static const bool keep_promised = config_get_bool("nvn", "tex_prep_keep_promised", true);
    while (s.held_bytes + est > s.budget && (evict || !keep_promised)) {
        auto victim = s.map.end();
        for (auto it = s.map.begin(); it != s.map.end(); ++it)
            // not t itself: e refers to its node, and erasing it here had the code below write into freed memory
            if (it->first != t && it->second.st == Entry::Done && (!keep_promised || !promised(it->first, it->second.job_gen)) &&
                (victim == s.map.end() || it->second.seq < victim->second.seq)) victim = it;
        if (victim == s.map.end()) break;
        drop_result(victim->second);
        s.map.erase(victim);
        s.st.evicted++;
    }
    if (s.held_bytes + est > s.budget && !force) {
        s.st.skipped_budget++;
        e.keep = Ref<Texture>();
        s.map.erase(t);
        t->prep_sched.store(0, std::memory_order_relaxed);   // not requested after all: a bind before its use may ask again
        return;
    }
    e.keep = Ref<Texture>::share(t);
    e.st = Entry::Queued;
    e.seq = ++s.seq;
    e.held = est;
    s.held_bytes += est;
    if (front) s.q.push_front({e.keep, e.seq});
    else s.q.push_back({e.keep, e.seq});
    s.st.scheduled++;
    s.cv_work.notify_one();
}

// Caller holds mu. A write was reported for a sparse texture: the job starts when the writes stop (the game maps and fills the tiles one range at a time).
void debounce_locked(Texture* t) {
    State& s = S();
    Entry& e = s.map[t];
    const uint64_t due = now_ns() + kDebounceNs;
    switch (e.st) {
        case Entry::Waiting: e.due_ns = due; return;
        case Entry::Queued: return;                       // reads the generation when it starts
        case Entry::Running: e.again = true; return;
        case Entry::Done: drop_result(e); break;
        case Entry::Idle: break;
    }
    e.keep = Ref<Texture>::share(t);
    e.st = Entry::Waiting;
    e.due_ns = due;
    s.waiting.push_back(t);
    s.cv_work.notify_all();
}

void worker_main() {
    log_set_thread_name("nvn-texprep");
    State& s = S();
    std::unique_lock<std::mutex> lk(s.mu);
    for (;;) {
        if (!s.waiting.empty()) {                         // Waiting textures whose writes stopped: queue them now
            const uint64_t now = now_ns();
            uint64_t next = UINT64_MAX;
            for (size_t i = 0; i < s.waiting.size();) {
                Texture* wt = s.waiting[i];
                auto wit = s.map.find(wt);
                if (wit == s.map.end() || wit->second.st != Entry::Waiting) { s.waiting.erase(s.waiting.begin() + i); continue; }
                if (wit->second.due_ns <= now) {
                    s.waiting.erase(s.waiting.begin() + i);
                    wit->second.st = Entry::Idle;
                    schedule_locked(wt);
                    continue;
                }
                next = std::min(next, wit->second.due_ns);
                ++i;
            }
            if (s.q.empty() && next != UINT64_MAX) {
                s.cv_work.wait_for(lk, std::chrono::nanoseconds(next - now));
                continue;
            }
        }
        if (s.q.empty()) {
            s.cv_work.wait(lk);
            continue;
        }
        Job job = std::move(s.q.front());
        s.q.pop_front();
        Texture* t = job.t.get();
        auto it = s.map.find(t);
        if (it == s.map.end() || it->second.seq != job.seq || it->second.st != Entry::Queued) continue;   // cancelled: the render thread took the texture back
        Entry& e = it->second;
        e.st = Entry::Running;
        e.job_gen = t->prep_gen.load(std::memory_order_acquire);
        const uint32_t gen = e.job_gen;
        lk.unlock();

        auto res = std::make_unique<PreparedTexture>();
        bool crashed = false, ok = false;
        uint64_t h0 = 0, h1 = 0, staged_size = 0;
        gfx::Buffer stage;
        if (!t->destroyed.load() && hash_guarded(*t, h0)) {
            const uint16_t zid = ft::zone_id("tex.prep.job");
            ft::Scope zs(zid, 0, t->id);
            ok = prepare_guarded(*t, s.caps, *res, crashed);
            // no cache entry: the job transcodes it itself (one thread, below normal priority) and writes the entry; take does not wait for this
            if (!ok && !crashed && res->error == "no cache entry" && !t->destroyed.load()) {
                {
                    std::lock_guard<std::mutex> sl(s.mu);
                    s.map[t].slow = true;     // the entry is never erased while its job runs
                }
                // A take that began waiting while this job was still looking for a cache entry must see `slow` now and go on with a stand-in;
                // without this it slept through the whole transcode (1.5-2 s render-thread stalls choosing body types at character creation).
                s.cv_done.notify_all();
                TexCaps full = s.caps;
                full.cache_only = false;
                full.encode_threads = 1;
                const int prio = GetThreadPriority(GetCurrentThread());
                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                *res = PreparedTexture();
                ok = prepare_guarded(*t, full, *res, crashed);
                SetThreadPriority(GetCurrentThread(), prio);
                if (ok) s_transcoded.fetch_add(1, std::memory_order_relaxed);
            }
            ok = ok && hash_guarded(*t, h1) && h0 == h1;     // the storage must not have changed while the job read it
            // A big result is copied into the upload buffer here: allocating it and touching its pages costs as much as the preparation itself
            if (ok && s.stage_bytes && res->data.size() >= s.stage_bytes && !s.verify) {
                if (!s.quiesced.load() && gpu::device().create_buffer(res->data.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, gfx::MemoryUse::Upload, &stage, "nvn staging (prepared)") == VK_SUCCESS &&
                    stage.mapped) {
                    std::memcpy(stage.mapped, res->data.data(), res->data.size());
                    gpu::device().flush_buffer(stage);
                    staged_size = res->data.size();
                    g_staged_bytes_total.add(static_cast<int64_t>(staged_size));
                    std::vector<uint8_t>().swap(res->data);
                } else if (stage) {
                    gpu::device().destroy_buffer(stage);
                    stage = gfx::Buffer{};
                }
            }
        }

        lk.lock();
        Entry& e2 = s.map[t];   // the entry is never erased while a job runs (the render thread only erases Queued and Done entries)
        e2.slow = false;
        s.held_bytes -= std::min(s.held_bytes, e2.held);
        e2.held = 0;
        if (crashed) s.st.crashes++;
        else if (!ok) s.st.failed++;
        const bool fresh = ok && t->prep_gen.load(std::memory_order_acquire) == gen && !t->destroyed.load();
        if (fresh) {
            e2.src_hash = h0;
            e2.held = staged_size ? staged_size : res->data.size();
            s.held_bytes += e2.held;
            e2.res = std::move(res);
            e2.stage = stage;
            stage = gfx::Buffer{};
            e2.st = Entry::Done;
            // the image holds a quick stand-in of this content: the next use uploads this result (cpu_dirty without a new generation)
            if (t->prep_prov.load(std::memory_order_acquire) == (gen | 0x80000000u)) {
                t->cpu_dirty.store(true);
                gpu::pool_note_cpu_write(t);   // a texture drawn through a pool takes the final content too
            }
        } else {
            if (ok) s.st.stale++;
            e2.st = Entry::Idle;
            e2.res.reset();
        }
        if (stage) gpu::device().destroy_buffer(stage);      // not used (the result is stale or the job failed)
        const bool again = e2.again;
        e2.again = false;
        s.cv_done.notify_all();
        if (e2.st == Entry::Idle) {
            Ref<Texture> keep = std::move(e2.keep);
            s.map.erase(t);
            if (again && !t->destroyed.load()) {
                // a stand-in of the current content is on screen: its job goes first and is never dropped for the budget
                const bool standin = t->prep_prov.load(std::memory_order_acquire) == (t->prep_gen.load(std::memory_order_acquire) | 0x80000000u);
                if (t->vpool) debounce_locked(t);
                else schedule_locked(t, standin, standin);
            }
            lk.unlock();
            keep = Ref<Texture>();     // the texture may be destroyed here; ~Texture takes no lock of ours
            lk.lock();
        }
        // results of textures the game has finalized will never be used
        for (auto it2 = s.map.begin(); it2 != s.map.end();) {
            if (it2->second.st == Entry::Done && it2->first->destroyed.load()) {
                drop_result(it2->second);
                it2 = s.map.erase(it2);
            } else {
                ++it2;
            }
        }
        log_summary_locked(false);
    }
}

void init_once() {
    std::call_once(g_once, [] {
        g_state = new State();   // never destroyed: the workers run until the process ends
        State& s = S();
        // Gauges (nxrt/gauge.h). State is never destroyed, so the callbacks stay valid for the life of the process.
        nxrt::gauge_register("nvn.texprep.entries", [] {
            std::lock_guard<std::mutex> lk(S().mu);
            return static_cast<int64_t>(S().map.size());
        });
        nxrt::gauge_register("nvn.texprep.bytes", [] {
            std::lock_guard<std::mutex> lk(S().mu);
            return static_cast<int64_t>(S().held_bytes);
        });
        nxrt::gauge_register("nvn.texprep.queue", [] {
            std::lock_guard<std::mutex> lk(S().mu);
            return static_cast<int64_t>(S().q.size());
        });
        nxrt::gauge_register("nvn.texprep.waiting", [] {
            std::lock_guard<std::mutex> lk(S().mu);
            return static_cast<int64_t>(S().waiting.size());
        });
        const bool on = config_get_bool("nvn", "tex_prep", true);
        if (!on) {
            g_enabled.store(0);
            NXLOG_I(kTag, "background texture preparation is off");
            return;
        }
        // 1024 MB (was 512): a 4K first visit registers several GB of textures; results evicted before their first use were read again
        s.budget = static_cast<uint64_t>(std::max<int64_t>(16, config_get_int("nvn", "tex_prep_mb", 1024))) << 20;
        s.min_bytes = static_cast<uint64_t>(std::max<int64_t>(0, config_get_int("nvn", "tex_prep_min_kb", 8))) << 10;
        // 64 KB (was 1024): with stand-ins ([nvn] tex_async) a texture whose job has not run is drawn blurry for a moment; starting the jobs of the
        // middle-sized textures at registration (long before their first draw) leaves fewer of those. In a 1.7 h session 70% of the BC7 cache files
        // the render thread read itself were 256 KB to 1 MB (ASTC 8x8 storage of 64 to 256 KB): under the old threshold. Over the budget a job is
        // simply not started.
        s.early_bytes = static_cast<uint64_t>(std::max<int64_t>(0, config_get_int("nvn", "tex_prep_early_kb", 64))) << 10;
        s.stage_bytes = static_cast<uint64_t>(std::max<int64_t>(0, config_get_int("nvn", "tex_prep_stage_kb", 1024))) << 10;
        s.verify = config_get_bool("nvn", "tex_prep_verify", false);
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        int n = static_cast<int>(config_get_int("nvn", "tex_prep_threads", 0));
        if (n <= 0) n = static_cast<int>(std::clamp<unsigned>(hw / 3, 2, 8));
        for (int i = 0; i < n; ++i) std::thread(worker_main).detach();
        g_enabled.store(1);
        g_tex_prep_on.store(true);
        NXLOG_I(kTag, "background texture preparation: %d threads, budget %llu MB, textures from %llu KB%s", n, static_cast<unsigned long long>(s.budget >> 20),
                static_cast<unsigned long long>(s.min_bytes >> 10), s.verify ? ", verifying every result" : "");
    });
}

// Block-compressed textures only: they are the game's assets (ASTC, BCn). Uncompressed ones are mostly render targets and small UI data whose
// storage is not final when the texture is initialized.
// any_size: also under [nvn] tex_prep_min_kb (stand-ins of tex_async: a small texture's cache read is still a disk seek on the render thread).
bool eligible(const Texture* t, bool any_size = false) {
    if (!t || t->destroyed.load() || !t->fmt || t->is_display || (!t->mem && !t->vpool)) return false;
    if (!(t->fmt->astc || t->fmt->bc)) return false;
    if (t->b.samples > 1 || t->fmt->aspect != kAspectColor) return false;
    if ((!any_size && t->storage_size < S().min_bytes) || t->storage_size > (256ull << 20)) return false;
    return true;
}

void consider(Texture* t, bool imminent) {
    if (!eligible(t) || !gpu::active()) return;
    if (!S().caps_ready.load(std::memory_order_acquire)) {
        static std::mutex cm;
        std::lock_guard<std::mutex> cl(cm);
        if (!S().caps_ready.load()) {
            const gfx::DeviceFeatures& ft = gpu::device().features();
            S().caps.bc = ft.bc;
            S().caps.astc = ft.astc_ldr;
            S().caps.bc7_transcode = true;
            S().caps.upgrade = true;
            S().caps.cache_only = true;
            S().caps.image_format = VK_FORMAT_UNDEFINED;   // whatever the cache entry holds; the user checks the format against its image
            S().caps_ready.store(true, std::memory_order_release);
        }
    }
    std::lock_guard<std::mutex> lk(S().mu);
    schedule_locked(t, false, false, imminent);
}

void bound_or_registered(Texture* t, bool imminent);

bool is_entry_format(VkFormat f) {
    return f == VK_FORMAT_R8G8B8A8_UNORM || f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_BC5_UNORM_BLOCK;
}

}  // namespace

bool async_standin() { return enabled() && async_standin_cfg(); }

bool set_exact_thread(bool on) {
    const bool was = t_exact;
    t_exact = on;
    return was;
}

bool exact_thread() { return t_exact; }

bool enabled() {
    int v = g_enabled.load(std::memory_order_acquire);
    if (v < 0) {
        init_once();
        v = g_enabled.load(std::memory_order_acquire);
    }
    return v == 1;
}

void on_flushed(Texture* t) {
    if (!t) return;
    t->prep_gen.fetch_add(1, std::memory_order_acq_rel);
    g_tex_scan_epoch.fetch_add(1, std::memory_order_relaxed);        // handles looked at before may now lead to a dirty texture
    gpu::pool_note_cpu_write(t);                                     // the pools it is registered in look at it (no scan of every texture)
    // Sparse textures (the big volumes): tiles are mapped and written range by range, each reported here. Once a pool or a command list has named the
    // texture, a job follows the writes and starts when they stop.
    if (t->vpool && (t->prep_flags.load(std::memory_order_acquire) & kWanted) && enabled() && gpu::active() && eligible(t)) {
        std::lock_guard<std::mutex> lk(S().mu);
        debounce_locked(t);
    }
}

void on_registered(Texture* t) {
    if (!t || !enabled() || t->storage_size < S().early_bytes) return;
    bound_or_registered(t, false);
}

void on_bound(Texture* t) { bound_or_registered(t, true); }

namespace {
void bound_or_registered(Texture* t, bool imminent) {
    if (!t || !enabled()) return;
    if (S().quiesced.load(std::memory_order_relaxed)) reopen_staging();      // a new GPU layer is running (the old one was shut down)
    if (!(t->prep_flags.load(std::memory_order_relaxed) & kWanted)) t->prep_flags.fetch_or(kWanted, std::memory_order_acq_rel);
    if (!t->cpu_dirty.load(std::memory_order_relaxed)) return;      // the common case: bound again every draw, nothing to do
    const uint32_t gen = t->prep_gen.load(std::memory_order_acquire);
    const uint32_t mark = gen | 0x80000000u;
    if (t->prep_sched.load(std::memory_order_relaxed) == mark) return;                    // already requested for this content
    t->prep_sched.store(mark, std::memory_order_relaxed);
    consider(t, imminent);
}
}  // namespace

void note_uploaded(Texture* t) {
    if (t) t->prep_flags.fetch_or(kUploaded, std::memory_order_acq_rel);
}

bool peek_entry_format(Texture* t, VkFormat& entry) {
    if (!enabled() || !t) return false;
    State& s = S();
    std::unique_lock<std::mutex> lk(s.mu);
    const auto deadline = std::chrono::steady_clock::now() + wait_cap();
    for (;;) {
        auto it = s.map.find(t);
        if (it == s.map.end()) return false;
        Entry& e = it->second;
        if (e.st == Entry::Running && e.job_gen == t->prep_gen.load(std::memory_order_acquire)) {
            if (e.slow) return false;   // no cache entry: the job transcodes (seconds); the caller looks the texture up itself, as with no job
            if (s.cv_done.wait_until(lk, deadline) == std::cv_status::timeout) return false;
            continue;
        }
        if (e.st != Entry::Done || e.job_gen != t->prep_gen.load(std::memory_order_acquire) || !e.res) return false;
        entry = is_entry_format(e.res->format) ? e.res->format : VK_FORMAT_UNDEFINED;
        return true;
    }
}

bool take(Texture* t, PreparedTexture& out, gfx::Buffer* staged, uint64_t* src_hash) {
    if (!enabled() || !t) return false;
    State& s = S();
    const uint64_t t_enter = now_ns();
    // tex_async = standin and t can get a stand-in for this content: a job not done yet is neither waited for long nor done here; the caller
    // uploads the stand-in and the job's result replaces it (the job moves to the front of the queue). wants_provisional takes no lock of ours.
    // The same when the image already holds a stand-in of this content (standin_on_screen): the caller keeps it (standin_kept) rather than
    // preparing the texture itself (that was a cache read or a transcode on the render thread, or a wait of up to 5 s for a running job).
    const bool defer = async_standin_cfg() && !t_exact && (wants_provisional(t) || standin_on_screen(t));
    std::unique_lock<std::mutex> lk(s.mu);
    bool waited = false;
    const auto deadline = std::chrono::steady_clock::now() + (defer ? wait_cap() : std::chrono::steady_clock::duration(std::chrono::seconds(5)));
    for (;;) {
        auto it = s.map.find(t);
        if (it == s.map.end()) {
            s.st.none++;
            return false;
        }
        Entry& e = it->second;
        if (e.st == Entry::Idle) {
            s.map.erase(it);
            s.st.none++;
            return false;
        }
        if (defer && (e.st == Entry::Waiting || e.st == Entry::Queued)) {   // not started: to the front of the queue, a stand-in meanwhile
            if (e.st == Entry::Waiting) {
                leave_waiting_locked(t);
                e.st = Entry::Idle;
                schedule_locked(t, true, true);
            } else {
                e.seq = ++s.seq;                          // the copy further back in the queue is skipped when a worker pops it
                s.q.push_front({e.keep, e.seq});
                s.cv_work.notify_one();
            }
            s.st.deferred++;
            return false;
        }
        if (e.st == Entry::Waiting || e.st == Entry::Queued) {   // not started: cheaper to do it here than to wait for a free worker or for the writes to stop
            if (e.st == Entry::Waiting) leave_waiting_locked(t);
            e.seq = ++s.seq;                          // the queued job is skipped when a worker pops it
            drop_result(e);
            Ref<Texture> keep = std::move(e.keep);
            s.map.erase(it);
            s.st.stolen++;
            lk.unlock();
            keep = Ref<Texture>();
            return false;
        }
        if (e.st == Entry::Running) {
            if (e.job_gen != t->prep_gen.load(std::memory_order_acquire)) {   // written again after the job started: its result will be stale
                s.st.none++;
                return false;
            }
            if (e.slow) {                 // a transcode: not waited for (the caller uploads a stand-in; the result replaces it when ready)
                s.st.none++;
                return false;
            }
            waited = true;
            const uint64_t tw = nxrt::hitch_ticks();
            const bool timed_out = s.cv_done.wait_until(lk, deadline) == std::cv_status::timeout;
            if (nxrt::hitch_on_render_thread()) nxrt::hitch_phase_add(nxrt::kHpRenderLockWait, nxrt::hitch_ticks() - tw);
            if (timed_out) {
                if (defer) s.st.deferred++;
                return false;
            }
            continue;
        }
        // Done: usable when nothing was written since (generation) and the storage still holds the bytes the job read (content hash)
        bool good = e.job_gen == t->prep_gen.load(std::memory_order_acquire);
        const uint64_t want = e.src_hash;
        std::unique_ptr<PreparedTexture> res = std::move(e.res);
        gfx::Buffer stage = e.stage;
        e.stage = gfx::Buffer{};
        drop_result(e);
        Ref<Texture> keep = std::move(e.keep);
        s.map.erase(it);
        lk.unlock();
        if (good) {
            uint64_t h = 0;
            good = hash_guarded(*t, h) && h == want;
        }
        keep = Ref<Texture>();
        bool verify_fail = false;
        if (!good && stage) gpu::device().destroy_buffer(stage);
        if (good) {
            out = std::move(*res);
            if (src_hash) *src_hash = want;
            if (stage) {
                if (staged) *staged = stage;
                else gpu::device().destroy_buffer(stage);     // the caller cannot use it: it would be left without the data
            }
            if (s.verify) {
                PreparedTexture fresh;
                const bool ok = prepare_texture(*t, s.caps, fresh);
                uint64_t after = 0;
                const bool still = hash_guarded(*t, after) && after == want;     // the game may be writing the storage right now: then there is nothing to compare
                lk.lock();
                if (still) s.st.verified++;
                lk.unlock();
                if (still && (!ok || fresh.format != out.format || fresh.data != out.data || fresh.level.size() != out.level.size())) {
                    NXLOG_E(kTag, "texture %u: the background result differs from a fresh preparation (%zu vs %zu bytes)", t->id, out.data.size(), fresh.data.size());
                    verify_fail = true;
                    if (ok) out = std::move(fresh);
                    else good = false;
                }
            }
        }
        lk.lock();
        if (good) {
            s.st.hits++;
            if (waited) s.st.waited_hits++;
        } else {
            s.st.stale++;
        }
        if (verify_fail) s.st.mismatches++;
        lk.unlock();
        if (waited && ft::on()) {
            static const uint16_t zid = ft::zone_id("tex.prep.wait");
            ft::event(zid, t_enter, now_ns(), t->id);
        }
        return good;
    }
}

bool hash_storage(const Texture& t, uint64_t& h) { return hash_guarded(t, h); }

bool wants_provisional(Texture* t) {
    if (!t || !enabled() || !gpu::active() || !eligible(t, async_standin_cfg()) || !t->fmt->astc) return false;
    const uint32_t mark = t->prep_gen.load(std::memory_order_acquire) | 0x80000000u;
    return t->prep_prov.load(std::memory_order_acquire) != mark;
}

void provisional_uploaded(Texture* t) {
    if (!t || !enabled()) return;
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mu);
    const uint32_t gen = t->prep_gen.load(std::memory_order_acquire);
    t->prep_prov.store(gen | 0x80000000u, std::memory_order_release);
    s_provisional.fetch_add(1, std::memory_order_relaxed);
    auto it = s.map.find(t);
    if (it != s.map.end()) {
        Entry& e = it->second;
        if (e.st == Entry::Done && e.job_gen == gen) {   // finished meanwhile: used at the next upload
            t->cpu_dirty.store(true);
            gpu::pool_note_cpu_write(t);
            return;
        }
        if (e.st == Entry::Running && e.job_gen == gen) return;                             // the worker marks the texture when it is done
        if (e.st == Entry::Queued || e.st == Entry::Waiting) return;
    }
    // the stand-in stays until this job is done: it goes first and is not dropped for the budget
    if (!t->destroyed.load()) schedule_locked(t, true, true);
}

bool standin_on_screen(Texture* t) {
    if (!t || !enabled() || !async_standin_cfg()) return false;
    return t->prep_prov.load(std::memory_order_acquire) == (t->prep_gen.load(std::memory_order_acquire) | 0x80000000u);
}

void standin_kept(Texture* t) {
    if (!t || !enabled()) return;
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mu);
    s_kept.fetch_add(1, std::memory_order_relaxed);
    const uint32_t gen = t->prep_gen.load(std::memory_order_acquire);
    auto it = s.map.find(t);
    if (it != s.map.end()) {
        Entry& e = it->second;
        if (e.st == Entry::Done && e.job_gen == gen) {   // finished meanwhile: used at the next upload
            t->cpu_dirty.store(true);
            gpu::pool_note_cpu_write(t);
            return;
        }
        if ((e.st == Entry::Running && e.job_gen == gen) || e.st == Entry::Queued || e.st == Entry::Waiting) return;   // its end marks the texture
    }
    if (!t->destroyed.load()) schedule_locked(t, true, true);
}

void final_uploaded(Texture* t) {
    if (t && t->prep_prov.load(std::memory_order_relaxed)) t->prep_prov.store(0, std::memory_order_release);
}

void wait_idle_for_test() {
    if (!g_state || !enabled()) return;
    State& s = S();
    std::unique_lock<std::mutex> lk(s.mu);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
        bool busy = !s.q.empty();
        for (const auto& kv : s.map)
            if (kv.second.st == Entry::Running || kv.second.st == Entry::Queued) busy = true;
        if (!busy) return;
        if (s.cv_done.wait_until(lk, deadline) == std::cv_status::timeout) return;
    }
}

void release_staging() {
    if (!g_state) return;
    State& s = S();
    std::vector<Ref<Texture>> release;
    {
        std::unique_lock<std::mutex> lk(s.mu);
        s.quiesced.store(true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (;;) {
            bool busy = false;
            for (const auto& kv : s.map)
                if (kv.second.st == Entry::Running) busy = true;
            if (!busy) break;
            if (s.cv_done.wait_until(lk, deadline) == std::cv_status::timeout) break;
        }
        for (auto it = s.map.begin(); it != s.map.end();) {
            if (it->second.stage && it->second.st == Entry::Done) {
                drop_result(it->second);           // gives the upload buffer back
                release.push_back(std::move(it->second.keep));
                it = s.map.erase(it);
            } else {
                ++it;
            }
        }
    }
    release.clear();
}

void reopen_staging() {
    if (g_state) S().quiesced.store(false);
}

Stats stats() {
    if (!g_state) return Stats();
    std::lock_guard<std::mutex> lk(S().mu);
    return S().st;
}

}  // namespace nxrt::nvn::be::texprep
