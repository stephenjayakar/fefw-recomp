// SPDX-License-Identifier: GPL-3.0-or-later
// Guest/host object binding and the memory pool registry (GPU address -> pool).
#include <algorithm>
#include <atomic>
#include <map>
#include <shared_mutex>

#include "nxrt/config.h"
#include "nxrt/gauge.h"
#include "nxrt/log.h"
#include "nvn_core.h"

namespace nxrt::nvn::be {

void bind_guest(void* guest, Obj* obj) {
    GuestHeader* h = static_cast<GuestHeader*>(guest);
    Obj* old = (h->magic == kGuestMagic) ? h->host : nullptr;
    h->host = obj;
    h->kind = static_cast<uint32_t>(obj->kind);
    h->magic = kGuestMagic;
    if (old) old->unref();
}

void unbind_guest(void* guest) {
    GuestHeader* h = static_cast<GuestHeader*>(guest);
    Obj* old = (h->magic == kGuestMagic) ? h->host : nullptr;
    h->host = nullptr;
    h->magic = 0;
    h->kind = 0;
    if (old) old->unref();
}

namespace {
std::shared_mutex g_pool_mu;
std::map<uint64_t, MemoryPool*> g_pools;   // gpu_base -> pool
std::atomic<uint64_t> g_pool_gen{0};       // moves with every registration and removal (pool_lookup's per-thread cache)

// Gauge (nxrt/gauge.h). The callback takes g_pool_mu; it is removed before the map is destroyed.
struct PoolGauges {
    PoolGauges() {
        nxrt::gauge_register("nvn.pool_registry", [] {
            std::shared_lock<std::shared_mutex> lk(g_pool_mu);
            return static_cast<int64_t>(g_pools.size());
        });
    }
    ~PoolGauges() { nxrt::gauge_unregister("nvn.pool_registry"); }
};
PoolGauges g_pool_gauges;
// Host objects the game has finalized that are still alive (something on the host side still holds a reference: a pool pair, a texture
// pool entry, a command list, a sparse texture). They should come back down to a handful a few seconds after a scene change; one that
// only grows is a host object outliving its guest object (issue #81).
nxrt::GaugeCounter g_pools_finalized_alive("nvn.memory_pools.finalized_alive");
nxrt::GaugeCounter g_pools_finalized_alive_bytes("nvn.memory_pools.finalized_alive.bytes");
nxrt::GaugeCounter g_textures_finalized_alive("nvn.textures.finalized_alive");
}  // namespace

void lifetime_pool_retired(const MemoryPool* p) {
    g_pools_finalized_alive.add(1);
    g_pools_finalized_alive_bytes.add(static_cast<int64_t>(p->size));
}
void lifetime_pool_freed(const MemoryPool* p) {
    if (!p->dead.load()) return;
    g_pools_finalized_alive.sub(1);
    g_pools_finalized_alive_bytes.sub(static_cast<int64_t>(p->size));
}
void lifetime_texture_finalized() { g_textures_finalized_alive.add(1); }
void lifetime_texture_freed(const Texture* t) {
    if (t->destroyed.load()) g_textures_finalized_alive.sub(1);
}

void pool_register(MemoryPool* p) {
    std::unique_lock<std::shared_mutex> lk(g_pool_mu);
    g_pools[p->gpu_base] = p;
    g_pool_gen.fetch_add(1, std::memory_order_release);
}

void pool_unregister(MemoryPool* p) {
    std::unique_lock<std::shared_mutex> lk(g_pool_mu);
    auto it = g_pools.find(p->gpu_base);
    if (it != g_pools.end() && it->second == p) g_pools.erase(it);
    g_pool_gen.fetch_add(1, std::memory_order_release);
}

// Each thread keeps the pools it found (64 slots by the address's MiB), valid while no pool was registered or removed (g_pool_gen): the render thread looks up
// the same few pools thousands of times a frame, and the map walk under the shared lock was ~7 % of its time (rt agent, notes/rt.md).
// A pool's gpu_base and size are set before it is registered and never change. [nvn] pool_cache (default on).
uint64_t pool_generation() { return g_pool_gen.load(std::memory_order_acquire); }

PoolHit pool_lookup(uint64_t gpu_address) {
    // direct mapped by the address's MiB (the survey's page-keyed cache, notes/survey-perf.md #6): one compare on a hit
    struct Hit { uint64_t lo = 0, len = 0; MemoryPool* p = nullptr; };
    constexpr size_t kWays = 64;
    thread_local uint64_t c_gen = ~0ull;
    thread_local Hit c_hit[kWays];
    static const bool cache_on = nxrt::config_get_bool("nvn", "pool_cache", true);
    static const bool check = nxrt::config_get_bool("nvn", "rt_check", false);   // [nvn] rt_check: the map gives the same pool
    Hit& slot = c_hit[(gpu_address >> 20) % kWays];
    if (cache_on && c_gen == g_pool_gen.load(std::memory_order_acquire))
        if (const Hit& h = slot; gpu_address - h.lo < h.len) {
            if (check) {
                std::shared_lock<std::shared_mutex> lk(g_pool_mu);
                auto it = g_pools.upper_bound(gpu_address);
                MemoryPool* m = nullptr;
                if (it != g_pools.begin() && gpu_address < std::prev(it)->second->gpu_base + std::prev(it)->second->size) m = std::prev(it)->second;
                if (m != h.p && c_gen == g_pool_gen.load()) NXLOG_W("nvn.rtcheck", "mismatch: pool_lookup cache");
            }
            return {h.p, gpu_address - h.lo};
        }
    std::shared_lock<std::shared_mutex> lk(g_pool_mu);
    auto it = g_pools.upper_bound(gpu_address);
    if (it == g_pools.begin()) return {};
    --it;
    MemoryPool* p = it->second;
    if (gpu_address >= p->gpu_base + p->size) return {};
    if (cache_on) {
        const uint64_t gen = g_pool_gen.load(std::memory_order_relaxed);   // under the lock: the map as of this generation
        if (gen != c_gen) {
            for (Hit& h : c_hit) h = Hit();
            c_gen = gen;
        }
        slot = {p->gpu_base, p->size, p};
    }
    return {p, gpu_address - p->gpu_base};
}

}  // namespace nxrt::nvn::be
