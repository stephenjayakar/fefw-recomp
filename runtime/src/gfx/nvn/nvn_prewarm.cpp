// SPDX-License-Identifier: GPL-3.0-or-later
// Pipeline prewarm (S10). A graphics pipeline is created the first time a draw needs its combination of shaders and baked state,
// and on the real GPU that costs from a few to a few hundred milliseconds on the render thread. Two things keep it off the frame:
//
//  * Recording. Every key a draw asks for is appended to the key file ([nvn] pipeline_keys, default pipeline_keys.bin), whether it was built,
//    linked or had to wait for its parts: the hashes of its shaders (the cache's code hashes, stable between runs) and the rest of the key
//    (formats, blend, topology ...) as plain bytes. The file is in the order the keys were first used, so the ones boot needs come first.
//  * Prewarming. At start every recorded key is queued in that order. Worker threads build each one whole with the on-disk VkPipelineCache,
//    from stand-in shaders of the shader cache (ShaderCache::load_stub: the same SPIR-V and reflection the game's own ShaderInfo gets, and the
//    same shared pipeline layout), so nothing waits for the game to set a program's shaders. The four shader parts of each key are also queued
//    for the pipeline-library threads (nvn_gpl.cpp, gpl_prebuild). When the game sets a program's shaders, the keys that use them go first.
//    A draw finds its pipeline by content (shader hashes and key bytes): it takes the finished one, links one from the parts while the whole
//    one is still being built (the whole one then replaces it), or builds it itself.
//
// The VkPipelineCache is also written to disk here: a background thread saves it while new pipelines appear, because a window that is
// closed ends the process without running the shutdown path. The same thread logs a summary line every 30 s.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nxrt/config.h"
#include "nxrt/ftrace.h"
#include "nxrt/gauge.h"
#include "nxrt/log.h"
#include "nvn_gpu_internal.h"
#include "nvn_pipeline.h"

namespace nxrt::nvn::be::gpu {
namespace {

constexpr const char* kTag = "nvn.prewarm";
constexpr char kMagic[8] = {'N', 'X', 'P', 'K', 'E', 'Y', 'S', '1'};
constexpr size_t kStageBytes = sizeof(const ShaderInfo*) * 3;
constexpr size_t kRestBytes = offsetof(GraphicsKey, tess) - kStageBytes;   // the tessellation stages are not recorded (tessellated keys never are)
static_assert(offsetof(GraphicsKey, stage) == 0, "the shader pointers come first in GraphicsKey");
static_assert(offsetof(GraphicsKey, color_format) == kStageBytes, "the plain key bytes follow the shader pointers");
constexpr size_t kMaxRecords = 400000;

struct Rec {
    uint64_t h[3];                          // code hashes of vertex, fragment, geometry (0 = stage absent)
    uint8_t rest[kRestBytes];               // the key after the shader pointers, byte for byte
};

// One recorded key being built ahead. Guarded by g_mu.
struct Job {
    Rec rec;
    uint64_t digest = 0;
    int state = 0;                          // 0 queued, 1 building, 2 done, 3 taken over by a draw (the worker skips it)
    bool in_queue = false, promoted = false;
    VkPipeline pipe = VK_NULL_HANDLE;
    std::shared_ptr<PipeUpgrade> up;        // a draw linked the key from parts meanwhile: the whole pipeline goes here and replaces that one
};

std::mutex g_mu;
std::condition_variable g_cv;
bool g_inited = false, g_enabled = true, g_stop = false;
std::string g_path;
std::vector<Rec> g_recs;
std::vector<std::shared_ptr<Job>> g_rec_jobs;                // by index in g_recs (null: recorded this session, built at its draw)
// Programs of one stage each (Pokopia binds a vertex program and a fragment program separately): a record is found by any of its stages.
std::unordered_multimap<uint64_t, size_t> g_by_any;          // any stage's hash -> index in g_recs
std::unordered_set<uint64_t> g_digests;                      // recorded keys
std::deque<std::shared_ptr<Job>> g_queue;                    // a job can sit in it twice (moved ahead): the second visit skips it
std::unordered_map<uint64_t, std::vector<std::shared_ptr<Job>>> g_jobs;   // by digest; a job leaves when a draw takes it or takes it over
std::vector<std::thread> g_workers;
std::thread g_saver;
PrewarmStats g_stats;

uint64_t parse_hash(const std::string& hex) {
    uint64_t v = 0;
    for (char ch : hex) {
        uint64_t d;
        if (ch >= '0' && ch <= '9') d = static_cast<uint64_t>(ch - '0');
        else if (ch >= 'a' && ch <= 'f') d = static_cast<uint64_t>(ch - 'a' + 10);
        else if (ch >= 'A' && ch <= 'F') d = static_cast<uint64_t>(ch - 'A' + 10);
        else return 0;
        v = (v << 4) | d;
    }
    return v;
}

std::string hex16(uint64_t h) {
    char b[20];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(h));
    return b;
}

uint64_t digest_of(const uint64_t h[3], const uint8_t* rest) {
    uint64_t d = 1469598103934665603ull;
    auto mix = [&d](uint8_t b) { d = (d ^ b) * 1099511628211ull; };
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 8; ++k) mix(static_cast<uint8_t>(h[i] >> (8 * k)));
    for (size_t i = 0; i < kRestBytes; ++i) mix(rest[i]);
    return d;
}

// The record of a key; false when it cannot be recorded (tessellation, a stage the cache has no hash for).
bool rec_of(const GraphicsKey& key, Rec* r) {
    if (key.tess[0] || key.tess[1]) return false;
    std::memset(r, 0, sizeof *r);
    for (int i = 0; i < 3; ++i) r->h[i] = key.stage[i] ? parse_hash(key.stage[i]->hash) : 0;
    if (!r->h[0]) return false;
    std::memcpy(r->rest, reinterpret_cast<const uint8_t*>(&key) + kStageBytes, kRestBytes);
    return true;
}

// The key of a record with stand-in shaders from the shader cache, and its layout (the shared one of the same descriptors, so the same
// VkPipelineLayout the game's program gets). Any thread; takes no lock of this file.
bool stub_key(const Rec& r, GraphicsKey& k, const ProgLayout*& pl) {
    if (!shader_cache().is_open()) return false;
    std::memset(&k, 0, sizeof k);
    for (int i = 0; i < 3; ++i) {
        if (!r.h[i]) continue;
        k.stage[i] = shader_cache().load_stub(hex16(r.h[i]));
        if (!k.stage[i]) return false;
    }
    std::memcpy(reinterpret_cast<uint8_t*>(&k) + kStageBytes, r.rest, kRestBytes);
    const ShaderInfo* st6[6] = {k.stage[0], k.stage[1], k.stage[2], nullptr, nullptr, nullptr};
    pl = prog_layout(st6);
    return pl != nullptr;
}

// The parts of a record, for the pipeline-library threads.
void queue_parts(const Rec& r, bool front) {
    gpl_prebuild([r](GraphicsKey& k, const ProgLayout*& pl) { return stub_key(r, k, pl); }, front);
}

void erase_job_locked(const std::shared_ptr<Job>& j) {
    auto it = g_jobs.find(j->digest);
    if (it == g_jobs.end()) return;
    auto& vec = it->second;
    vec.erase(std::remove(vec.begin(), vec.end(), j), vec.end());
    if (vec.empty()) g_jobs.erase(it);
}

void load_file() {
    std::ifstream f(g_path, std::ios::binary);
    if (!f) return;
    char magic[8];
    uint32_t size = 0, pad = 0;
    f.read(magic, 8);
    f.read(reinterpret_cast<char*>(&size), 4);
    f.read(reinterpret_cast<char*>(&pad), 4);
    if (!f || std::memcmp(magic, kMagic, 8) != 0 || size != kRestBytes) {
        NXLOG_W(kTag, "%s is not a key file of this build (the record layout changed): ignored, a new one is started", g_path.c_str());
        f.close();
        std::error_code ec;
        std::filesystem::remove(g_path, ec);                 // so the next record writes a fresh header
        return;
    }
    Rec r;
    while (g_recs.size() < kMaxRecords && f.read(reinterpret_cast<char*>(&r), sizeof r)) {
        if (!g_digests.insert(digest_of(r.h, r.rest)).second) continue;
        for (uint64_t h : r.h)
            if (h) g_by_any.emplace(h, g_recs.size());
        g_recs.push_back(r);
    }
    g_stats.loaded = g_recs.size();
}

void append_record(const Rec& r) {
    bool fresh = false;
    {
        std::ifstream t(g_path, std::ios::binary | std::ios::ate);
        fresh = !t || t.tellg() < 16;
    }
    std::ofstream f(g_path, std::ios::binary | std::ios::app);
    if (!f) return;
    if (fresh) {
        const uint32_t size = static_cast<uint32_t>(kRestBytes), pad = 0;
        f.write(kMagic, 8);
        f.write(reinterpret_cast<const char*>(&size), 4);
        f.write(reinterpret_cast<const char*>(&pad), 4);
    }
    f.write(reinterpret_cast<const char*>(&r), sizeof r);
}

void worker_main() {
    log_set_thread_name("nvn-prewarm");
    set_compile_thread_priority();
    for (;;) {
        std::shared_ptr<Job> j;
        {
            std::unique_lock<std::mutex> lk(g_mu);
            g_cv.wait(lk, [] { return g_stop || !g_queue.empty(); });
            if (g_stop) return;
            j = std::move(g_queue.front());
            g_queue.pop_front();
            if (j->state != 0) continue;       // built already (it was moved ahead) or a draw took it over
            j->state = 1;
        }
        VkPipeline p = VK_NULL_HANDLE;
        {
            NXFT("prewarm.build");
            GraphicsKey k;
            const ProgLayout* pl = nullptr;
            if (stub_key(j->rec, k, pl)) p = build_graphics_pipeline(k, *pl);
        }
        {
            std::lock_guard<std::mutex> lk(g_mu);
            j->state = 2;
            if (p) ++g_stats.built; else ++g_stats.failed;
            if (j->up) {                       // a draw linked it meanwhile: this one replaces the linked one (gpl_take_upgrade)
                j->up->pipe = p;
                j->up->state.store(1, std::memory_order_release);
                j->up.reset();
            } else {
                j->pipe = p;
            }
        }
        g_cv.notify_all();
    }
}

// The summary line every 30 s: what was deferred, built at the draw and built in the background since the last one (session totals in brackets).
void log_summary_locked() {
    struct Snap { uint64_t deferred = 0, linked = 0, whole_draw = 0, whole_bg = 0, parts_bg = 0, upgrades = 0; };
    static Snap last;
    const GplStats g = gpl_stats();
    Snap now;
    now.deferred = g.draws_deferred;
    now.linked = g.linked;
    now.whole_draw = pipelines_built_at_draw();
    now.whole_bg = g_stats.built;
    now.parts_bg = g.parts_ahead;
    now.upgrades = g.upgrades_done;
    size_t waiting = 0;
    for (const auto& j : g_rec_jobs)
        if (j && j->state <= 1) ++waiting;
    auto d = [](uint64_t a, uint64_t b) { return static_cast<unsigned long long>(a - b); };
    auto u = [](uint64_t a) { return static_cast<unsigned long long>(a); };
    NXLOG_I("nvn.pipe", "pipelines, last 30 s (session): draws deferred %llu (%llu), linked at the draw %llu (%llu), built whole at the draw %llu (%llu); "
                        "background builds %llu (%llu): whole %llu, parts %llu, upgrades %llu; keys known %zu (%llu from the file, %llu new), %zu still to build",
            d(now.deferred, last.deferred), u(now.deferred), d(now.linked, last.linked), u(now.linked), d(now.whole_draw, last.whole_draw), u(now.whole_draw),
            d(now.whole_bg + now.parts_bg + now.upgrades, last.whole_bg + last.parts_bg + last.upgrades), u(now.whole_bg + now.parts_bg + now.upgrades),
            d(now.whole_bg, last.whole_bg), d(now.parts_bg, last.parts_bg), d(now.upgrades, last.upgrades), g_recs.size(), u(g_stats.loaded), u(g_stats.recorded),
            waiting);
    last = now;
}

void saver_main() {
    log_set_thread_name("nvn-cache-save");
    std::unique_lock<std::mutex> lk(g_mu);
    uint64_t logged = 0;
    auto next_summary = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!g_stop) {
        g_cv.wait_for(lk, std::chrono::seconds(2), [] { return g_stop; });
        if (g_stop) break;
        const uint64_t sig = g_stats.recorded + g_stats.queued * 3 + g_stats.built * 5 + g_stats.taken * 7 + g_stats.waited * 11 + g_stats.claimed * 13 +
                             g_stats.linked_meanwhile * 17 + g_stats.promoted * 19 + g_stats.skipped_busy * 23;
        if (sig != logged) {                       // the process may end without a shutdown: say what the prewarm did while it runs
            logged = sig;
            NXLOG_I(kTag, "prewarm so far: %llu recorded, %llu loaded, %llu queued (%llu moved ahead), %llu built ahead (%llu failed), %llu taken by draws (%llu waited %.1f ms), "
                          "%llu linked from parts while building, %llu built by the draw itself, %llu draws skipped while their build ran",
                    static_cast<unsigned long long>(g_stats.recorded), static_cast<unsigned long long>(g_stats.loaded), static_cast<unsigned long long>(g_stats.queued),
                    static_cast<unsigned long long>(g_stats.promoted), static_cast<unsigned long long>(g_stats.built), static_cast<unsigned long long>(g_stats.failed),
                    static_cast<unsigned long long>(g_stats.taken), static_cast<unsigned long long>(g_stats.waited), g_stats.wait_ms,
                    static_cast<unsigned long long>(g_stats.linked_meanwhile), static_cast<unsigned long long>(g_stats.claimed),
                    static_cast<unsigned long long>(g_stats.skipped_busy));
        }
        gpl_progress_log();
        compute_progress_log();
        if (std::chrono::steady_clock::now() >= next_summary) {
            next_summary += std::chrono::seconds(30);
            log_summary_locked();
        }
        if (pipeline_cache_dirty()) {
            lk.unlock();
            pipeline_cache_save();
            lk.lock();
        }
    }
}

}  // namespace

void prewarm_init() {
    std::vector<Rec> parts;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_inited) return;
        g_inited = true;
        g_stop = false;
        g_enabled = config_get_bool("nvn", "prewarm", true);
        g_path = config_get_str("nvn", "pipeline_keys", "pipeline_keys.bin");
        if (!g_enabled || g_path.empty()) {
            g_enabled = false;
            return;
        }
        load_file();
        // [nvn] prewarm_boot = false: a key waits until the game sets one of its shaders (the old behaviour)
        const bool boot = config_get_bool("nvn", "prewarm_boot", true);
        g_rec_jobs.resize(g_recs.size());
        for (size_t i = 0; i < g_recs.size(); ++i) {
            auto j = std::make_shared<Job>();
            j->rec = g_recs[i];
            j->digest = digest_of(j->rec.h, j->rec.rest);
            g_rec_jobs[i] = j;
            g_jobs[j->digest].push_back(j);
            if (boot) {
                j->in_queue = true;
                g_queue.push_back(j);
                parts.push_back(j->rec);
            }
        }
        g_stats.queued = boot ? g_recs.size() : 0;
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const int64_t want = config_get_int("nvn", "prewarm_threads", std::clamp<int64_t>(hw / 4, 1, 6));   // was hw / 6, 1 to 4 (2 on 16 threads)
        for (int64_t i = 0; i < want; ++i) g_workers.emplace_back(worker_main);
        g_saver = std::thread(saver_main);
        NXLOG_I(kTag, "prewarm: %zu recorded pipelines from %s, %zu worker threads; %s", g_recs.size(), g_path.c_str(), g_workers.size(),
                boot ? "all queued from boot, in the order they were first used" : "each queued when the game sets one of its shaders ([nvn] prewarm_boot is false)");
    }
    for (const Rec& r : parts) queue_parts(r, false);           // outside g_mu: the pipeline-library side takes its own lock
}

void prewarm_record(const GraphicsKey& key) {
    Rec r;
    if (!rec_of(key, &r)) return;                               // the record has no room for tessellation stages
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_enabled) return;
    if (g_recs.size() >= kMaxRecords || !g_digests.insert(digest_of(r.h, r.rest)).second) return;
    g_recs.push_back(r);
    g_rec_jobs.push_back(nullptr);                              // built or linked at its draw this session
    ++g_stats.recorded;
    append_record(r);
}

size_t prewarm_program(const ShaderInfo* const stages[6]) {
    if (!stages || stages[3] || stages[4]) return 0;            // tessellated programs are not recorded
    prewarm_init();
    std::vector<Rec> ahead;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_enabled || g_stop) return 0;
        // The game is about to draw with these shaders: every recorded key that has one of them goes first in line.
        for (int i = 0; i < 3; ++i) {
            if (!stages[i]) continue;
            auto range = g_by_any.equal_range(parse_hash(stages[i]->hash));
            for (auto it = range.first; it != range.second; ++it) {
                const std::shared_ptr<Job>& j = g_rec_jobs[it->second];
                if (!j || j->state != 0 || j->promoted) continue;
                j->promoted = true;
                if (!j->in_queue) {
                    j->in_queue = true;
                    ++g_stats.queued;
                }
                g_queue.push_front(j);
                ++g_stats.promoted;
                ahead.push_back(j->rec);
            }
        }
    }
    if (ahead.empty()) return 0;
    g_cv.notify_all();
    for (const Rec& r : ahead) queue_parts(r, true);
    return ahead.size();
}

bool prewarm_take(const GraphicsKey& key, VkPipeline* out, const std::function<bool()>& can_link, std::shared_ptr<PipeUpgrade>* later, bool* busy) {
    if (busy) *busy = false;
    Rec want;
    if (!rec_of(key, &want)) return false;
    const uint64_t digest = digest_of(want.h, want.rest);
    std::unique_lock<std::mutex> lk(g_mu);
    if (!g_enabled) return false;
    std::shared_ptr<Job> j;
    {
        auto it = g_jobs.find(digest);
        if (it == g_jobs.end()) return false;
        for (const auto& e : it->second)
            if (std::memcmp(&e->rec, &want, sizeof want) == 0) { j = e; break; }
    }
    if (!j) return false;
    bool asked = false;
    for (;;) {
        if (j->state == 2) {
            erase_job_locked(j);
            if (!j->pipe) return false;                          // the build failed: the draw tries itself
            *out = j->pipe;
            j->pipe = VK_NULL_HANDLE;
            ++g_stats.taken;
            return true;
        }
        if (j->state == 3) return false;
        if (!asked && can_link && later) {                       // queued or building: can the draw link it from its parts now?
            asked = true;
            lk.unlock();
            const bool link = can_link();                        // takes the module and pipeline-library locks
            lk.lock();
            if (j->state == 2 || j->state == 3) continue;
            if (link) {
                j->up = std::make_shared<PipeUpgrade>();
                *later = j->up;
                erase_job_locked(j);
                ++g_stats.linked_meanwhile;
                return false;
            }
        }
        if (j->state == 0) {                                     // still queued: build it here instead of behind the others
            j->state = 3;
            erase_job_locked(j);
            ++g_stats.claimed;
            return false;
        }
        if (busy) {                                              // being built and not linkable: the draw is skipped, the next one asks again
            *busy = true;
            ++g_stats.skipped_busy;
            return false;
        }
        const auto t0 = std::chrono::steady_clock::now();        // being built: wait for it
        {
            NXFT("prewarm.wait");
            g_cv.wait(lk, [&] { return j->state == 2 || g_stop; });
        }
        g_stats.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++g_stats.waited;
        if (j->state != 2) return false;
    }
}

void prewarm_shutdown() {
    std::vector<std::thread> workers;
    std::thread saver;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_inited) return;
        g_stop = true;
        workers.swap(g_workers);
        saver = std::move(g_saver);
        g_queue.clear();
    }
    g_cv.notify_all();
    for (auto& t : workers) t.join();
    if (saver.joinable()) saver.join();
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto& [h, vec] : g_jobs)
        for (auto& j : vec)
            if (j->state == 2 && j->pipe) vkDestroyPipeline(vkdev(), j->pipe, nullptr);     // built and never used
    g_jobs.clear();
    NXLOG_I(kTag, "prewarm: %llu recorded, %llu loaded, %llu queued, %llu built ahead (%llu failed), %llu taken (%llu waited %.1f ms), %llu linked from parts while building, "
                  "%llu taken over by a draw",
            static_cast<unsigned long long>(g_stats.recorded), static_cast<unsigned long long>(g_stats.loaded), static_cast<unsigned long long>(g_stats.queued),
            static_cast<unsigned long long>(g_stats.built), static_cast<unsigned long long>(g_stats.failed), static_cast<unsigned long long>(g_stats.taken),
            static_cast<unsigned long long>(g_stats.waited), g_stats.wait_ms, static_cast<unsigned long long>(g_stats.linked_meanwhile),
            static_cast<unsigned long long>(g_stats.claimed));
    g_inited = false;
    g_stats = PrewarmStats{};
    g_recs.clear();
    g_rec_jobs.clear();
    g_by_any.clear();
    g_digests.clear();
}

PrewarmStats prewarm_stats() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_stats;
}

// The game set a program's shaders: its recorded pipelines go first in line. Programs with both a vertex and a fragment shader also get
// their shader parts queued by the pipeline-library side when the key file has nothing for them.
void prewarm_program_shaders(const ShaderInfo* const stages[6]) {
    if (stages && stages[5]) compute_prebuild(stages[5]);       // a compute program: its pipeline on a background thread ([nvn] compute_async)
    gpl_program(stages, prewarm_program(stages) != 0);
}

void flush_caches() { pipeline_cache_flush(); }

// Gauges (nxrt/gauge.h). The callbacks take g_mu; they are removed before the containers they read are destroyed.
namespace {
struct PrewarmGauges {
    PrewarmGauges() {
        nxrt::gauge_register("nvn.prewarm.records", [] {
            std::lock_guard<std::mutex> lk(g_mu);
            return static_cast<int64_t>(g_recs.size());
        });
        nxrt::gauge_register("nvn.prewarm.records.bytes", [] {
            std::lock_guard<std::mutex> lk(g_mu);
            return static_cast<int64_t>(g_recs.size() * sizeof(Rec));
        });
        nxrt::gauge_register("nvn.prewarm.digests", [] {
            std::lock_guard<std::mutex> lk(g_mu);
            return static_cast<int64_t>(g_digests.size());
        });
        nxrt::gauge_register("nvn.prewarm.queue", [] {
            std::lock_guard<std::mutex> lk(g_mu);
            return static_cast<int64_t>(g_queue.size());
        });
        nxrt::gauge_register("nvn.prewarm.jobs", [] {
            std::lock_guard<std::mutex> lk(g_mu);
            return static_cast<int64_t>(g_jobs.size());
        });
    }
    ~PrewarmGauges() {
        for (const char* n : {"nvn.prewarm.records", "nvn.prewarm.records.bytes", "nvn.prewarm.digests", "nvn.prewarm.queue", "nvn.prewarm.jobs"})
            nxrt::gauge_unregister(n);
    }
};
PrewarmGauges g_prewarm_gauges;
}  // namespace

}  // namespace nxrt::nvn::be::gpu
