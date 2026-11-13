// SPDX-License-Identifier: GPL-3.0-or-later
// Memory pools as Vulkan memory. A pool's storage is the game's own memory (a view of the runtime's pagefile-backed heap); it is
// imported with VK_EXT_external_memory_host and wrapped in one VkBuffer, so a GPU address (= host address) resolves to
// (buffer, offset) with no copy, and anything the GPU writes (compute, copies, queries, events) is visible to the CPU through
// the same pages. Measured on the RTX 3080 and the UHD 770 with MapViewOfFile views: imports work with 4096-byte alignment,
// and GPU writes are visible through every view of the section.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <new>
#include <unordered_set>
#include <unordered_map>
#include <windows.h>

#include "nxrt/config.h"
#include "nxrt/ftrace.h"
#include "nxrt/log.h"
#include "nxrt/gauge.h"
#include "nxrt/memory.h"
#include "nxrt/write_watch.h"
#include "vk_gauges.h"
#include "nvn_gpu_internal.h"
#include "nvn_mirror_cap.h"

namespace nxrt::nvn::be {

namespace gpu { struct Mirror; }

// One Vulkan import of a byte range of a pool: its own private view, VkDeviceMemory and VkBuffer.
struct PoolChunk {
    std::shared_ptr<PoolChunk> backing; // shared import of a committed heap slab
    gfx::Buffer copy;
    std::vector<uint8_t> snapshot;
    uint8_t* host = nullptr;
    uint64_t buffer_offset = 0;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    void* view = nullptr;             // private small view of the heap section the import goes through (nullptr: imported at its own address)
    uint64_t off = 0;                 // pool offset (from PoolGpu::base) of byte 0 of the import
    VkDeviceSize size = 0;
    bool copy_used = false;
    bool copy_tried = false;
    bool tried = false;               // import attempted
    bool failed = false;
    PoolGpu* owner = nullptr;         // copy fallback: the pool, for the retried import (pool_copy_begin)
    uint32_t retry_at = 0;            // copy fallback: present count at which the real import is tried again
    uint32_t retry_wait = 0;          // ... and the presents to wait after that one fails (doubles)
};

// A pool is imported whole (one chunk; the background importer does it) or, when it is larger than one chunk ([nvn] pool_import_chunk_mb), in
// chunks imported on the render thread the first time an address in them is resolved. Chunk k covers pool bytes [k*stride, k*stride + clen):
// neighbouring chunks overlap by clen - stride bytes, so any range that short lies wholly in one chunk. Each chunk goes through its own private
// view of the section (never the same host pointer twice: the spec lets a driver refuse importing the same pointer again). A longer range
// that crosses a stride boundary gets a span: one more import over the strides it touches (rare; logged).
struct PoolGpu {
    uint8_t* base = nullptr;          // aligned start of the pool's imports
    VkDeviceSize size = 0;            // aligned size
    bool failed = false;
    bool viewed = false;              // imports go through private views of the heap section (the mirror needs that)
    int flags = 0;
    uint64_t stride = 0;              // chunk k starts at k * stride (whole import: stride = size)
    uint64_t clen = 0;                // bytes per chunk (the last one may be shorter)
    std::vector<PoolChunk> chunks;
    std::deque<PoolChunk> spans;      // imports of ranges longer than the overlap that cross a stride boundary (deque: stable addresses)
    gpu::Mirror* mirror = nullptr;  // device-local copy of the pages draws read (S38, [nvn] pool_mirror); render thread
    bool mirror_tried = false;
};

MemoryPool::~MemoryPool() {
    lifetime_pool_freed(this);
    if (gpu) gpu::destroy_pool_gpu(gpu);
    gpu = nullptr;
    if (PoolGpu* pf = prefetched.exchange(nullptr)) gpu::destroy_pool_gpu(pf);   // imported in the background and never used
}

namespace gpu {
namespace {

constexpr const char* kTag = "nvn.pool";
constexpr uint64_t kMirrorChunkBytes = 64ull << 20;   // the mirror's chunk (kChunkBytes below): import strides are a multiple of it

VkBuffer g_dummy = VK_NULL_HANDLE;
gfx::Buffer g_dummy_buf;
std::once_flag g_dummy_once;
std::mutex g_log_mu;
std::unordered_set<uint64_t> g_logged;
std::mutex g_live_mu;
std::unordered_set<PoolGpu*> g_live;
nxrt::GaugeCounter g_gauge_pools_live("nvn.pools_live");   // pools imported and not yet destroyed (g_live)

// [nvn] pool_import_chunk_mb (default 64, rounded up to a multiple of 64; 0 = import every pool whole). A pool larger than one chunk is imported
// in chunks of that many MiB plus a quarter of it overlapping the next chunk, each on the render thread the first time a draw, copy or texture
// touches it. vkAllocateMemory of an import pins every page of it (about a quarter of a millisecond per MiB) and a vkQueueSubmit2 running beside
// it waits: Pokopia's 3500 MiB graphics heap took ~0.9 s as one import, while draws read only a few MiB of it.
uint64_t import_stride() {
    static const uint64_t s = [] {
        const int64_t mb = std::max<int64_t>(0, config_get_int("nvn", "pool_import_chunk_mb", 64));
        const uint64_t b = static_cast<uint64_t>(mb) << 20;
        return b ? (b + kMirrorChunkBytes - 1) / kMirrorChunkBytes * kMirrorChunkBytes : uint64_t{0};
    }();
    return s;
}
uint64_t import_overlap() { return import_stride() / 4; }

void destroy_chunk(PoolChunk& c);
nxrt::GaugeCounter g_import_failures("nvn.pool_import_failures");
nxrt::GaugeCounter g_shared_import_hits("nvn.pool_shared_import_hits");
nxrt::GaugeCounter g_copy_fallbacks("nvn.pool_copy_fallbacks");
std::mutex g_slabs_mu;
std::unordered_map<uint64_t, std::weak_ptr<PoolChunk>> g_slabs;
std::unordered_set<uint64_t> g_rejected_slabs;
constexpr uint64_t kSlab = 16ull << 20;
std::mutex g_copies_mu;
std::unordered_set<PoolChunk*> g_copies;
std::atomic<bool> g_have_copies{false};
std::atomic<bool> g_copy_touched{false};   // the submission being recorded uses a copy buffer (render thread sets it, pool_copy_begin clears it)
// A pool part on a copy buffer tries the real import again after this many presents, then after twice that, up to 16 times as long (the
// import failed under memory pressure, which may be gone a minute later).
constexpr uint32_t kImportRetryPresents = 600;
constexpr uint32_t kImportRetryMax = kImportRetryPresents * 16;
uint32_t presents_now();
bool out_uses_buffer(VkBuffer b);

// Failures that can repeat once per resource under memory pressure: the first `first` of a kind are logged, then one in 256.
bool log_rate(std::atomic<uint32_t>& n, uint32_t first = 16) {
    const uint32_t k = n.fetch_add(1, std::memory_order_relaxed);
    return k < first || (k - first) % 256 == 255;
}
std::atomic<uint32_t> g_log_import_fail{0}, g_log_headroom{0}, g_log_fallback{0};


// Only batch committed pages. Pinning SEC_RESERVE gaps is invalid on AMD.
bool committed_slab(uint64_t start) {
    // The heap's page table answers "every page committed" without VirtualQuery, which on the section view walks the whole run of identical
    // pages after the address (up to 85 ms) under g_slabs_mu, which a render-thread import of a small pool waits for. A slab is imported through
    // a private view, so the primary view's protection does not matter here, only the commit. [memory] freeze_table = false: always the walk.
    static const bool use_table = config_get_bool("memory", "freeze_table", true);
    if (use_table && mem::heap_rw_pages(start, kSlab) == 1 && !mem::ww::any_guarded(start, start + kSlab)) return true;
    const uint64_t end = start + kSlab;
    while (start < end) {
        MEMORY_BASIC_INFORMATION m{};
        if (!VirtualQuery(reinterpret_cast<void*>(start), &m, sizeof(m)) || m.State != MEM_COMMIT ||
            (m.Protect & (PAGE_NOACCESS | PAGE_GUARD))) return false;
        start = reinterpret_cast<uint64_t>(m.BaseAddress) + m.RegionSize;
    }
    return true;
}

// Imports pool bytes [off, off + size) of g into c. False (c.failed) when it cannot.
bool import_range(PoolGpu* g, PoolChunk& c, uint64_t off, uint64_t size, const char* what, size_t index, size_t count, bool batch = true) {
    c.tried = true;
    c.off = off;
    c.size = size;
    ft::Scope zs(ft::zone_id("pool.import"), 0, static_cast<uint32_t>(g->flags) << 16 | static_cast<uint32_t>(std::min<uint64_t>(size >> 20, 0xffff)));
    nxrt::HitchTimer ht(nxrt::kHpPoolImport);      // the hitch line: an import running beside a slow vkQueueSubmit2 shows here
    gfx::Device& dev = device();
    const uint64_t start = reinterpret_cast<uint64_t>(g->base) + off;
    if (!dev.features().external_memory_host) { c.failed = true; return false; }

    // Small pools share one import rather than consuming an allocation each. The weak
    // cache releases the slab after its last pool's deferred GPU retirement.
    const uint64_t slab = start & ~(kSlab - 1);
    if (batch && g->viewed && start + size <= slab + kSlab &&
        slab >= mem::kHeapBase && slab + kSlab <= mem::kHeapBase + mem::heap_reserved()) {
        std::lock_guard<std::mutex> lock(g_slabs_mu);
        auto& entry = g_slabs[slab];
        auto shared = entry.lock();
        if (!shared && !g_rejected_slabs.contains(slab) && committed_slab(slab)) {
            shared = std::shared_ptr<PoolChunk>(new PoolChunk, [](PoolChunk* p) { destroy_chunk(*p); delete p; });
            PoolGpu region;
            region.base = reinterpret_cast<uint8_t*>(slab);
            region.viewed = true;
            region.flags = g->flags;
            if (!import_range(&region, *shared, 0, kSlab, "slab", 0, 1, false)) {
                shared.reset();
                g_rejected_slabs.insert(slab); // don't repeatedly pin a slab the driver rejected
            }
            entry = shared;
        }
        if (shared) {
            g_shared_import_hits.add(1);
            c.backing = shared;
            c.buffer = shared->buffer;
            c.buffer_offset = start - slab;
            return true;
        }
        if (entry.expired()) g_slabs.erase(slab);
    }

    // Import through a small private view of the same section pages when the range lies in the heap's primary view (see mem::heap_private_view).
    uint8_t* import_ptr = reinterpret_cast<uint8_t*>(start);
    if (g->viewed) {
        const uint64_t soff = start - mem::kHeapBase;
        const uint64_t voff = soff & ~(mem::kAllocGranularity - 1);
        const uint64_t delta = soff - voff;
        const uint64_t vsize = (delta + size + mem::kAllocGranularity - 1) & ~(mem::kAllocGranularity - 1);
        if (void* v = mem::heap_private_view(voff, std::min<uint64_t>(vsize, mem::heap_reserved() - voff))) {
            c.view = v;
            import_ptr = static_cast<uint8_t*>(v) + delta;
        }
    }

    // The driver reads and writes back every page it imports (vkAllocateMemory pins them); a write of the game's landing in between was
    // lost (issue 97: one byte of a file the game was loading next to the pool, and its zstd frame never decompressed). Through a private
    // view the game's own view can be made read-only meanwhile: its writes to these pages wait for the import (mem::ww::ImportFreeze).
    std::optional<mem::ww::ImportFreeze> freeze;
    if (c.view) freeze.emplace(reinterpret_cast<const void*>(start), size);
    VkMemoryHostPointerPropertiesEXT hp = {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult r;
    const auto tm0 = std::chrono::steady_clock::now();
    { NXFT("pool.hostprops"); r = vkGetMemoryHostPointerPropertiesEXT(dev.device(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, import_ptr, &hp); }
    if (r != VK_SUCCESS || !hp.memoryTypeBits) {
        NXLOG_E(kTag, "vkGetMemoryHostPointerPropertiesEXT failed (%d) for %p (%s %zu of %zu)", static_cast<int>(r), reinterpret_cast<void*>(start), what, index, count);
        c.failed = true;
        return false;
    }
    VkExternalMemoryBufferCreateInfo ebi = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ebi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &ebi};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
                (device().features().transform_feedback ? VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT : 0u);
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    r = vkCreateBuffer(dev.device(), &bci, nullptr, &c.buffer);
    if (r == VK_SUCCESS) vk_gauge_buffers().add(1);
    else c.buffer = VK_NULL_HANDLE;
    if (r != VK_SUCCESS) { c.failed = true; return false; }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev.device(), c.buffer, &req);
    hp.memoryTypeBits &= req.memoryTypeBits;
    if (!hp.memoryTypeBits || req.size > size) { c.failed = true; destroy_chunk(c); c.tried = c.failed = true; return false; }
    uint32_t ty = 0;
    while (!(hp.memoryTypeBits & (1u << ty))) ++ty;
    VkImportMemoryHostPointerInfoEXT imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = import_ptr;
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp};
    mai.allocationSize = size;
    mai.memoryTypeIndex = ty;
    const auto tm1 = std::chrono::steady_clock::now();
    {
        NXFT("pool.alloc");
        static const bool inject = config_get_bool("diag", "fail_small_pool_import", false);
        static std::atomic<bool> injected{false};
        const uint64_t limit = dev.limits_ext().max_memory_allocation_count;
        const bool pressure = limit > 128 && static_cast<uint64_t>(vk_gauge_memory_allocs().get()) >= limit - 128;
        if (pressure || (inject && size <= 65536 && !injected.exchange(true))) {
            r = VK_ERROR_INVALID_EXTERNAL_HANDLE;
            if (!pressure || log_rate(g_log_headroom))
                NXLOG_W(kTag, "host import skipped (%s)", pressure ? "allocation headroom" : "diagnostic fault injection");
        } else r = vkAllocateMemory(dev.device(), &mai, nullptr, &c.mem);
    }
    const auto tm2 = std::chrono::steady_clock::now();
    if (r != VK_SUCCESS) {
        if (log_rate(g_log_import_fail))
            NXLOG_E(kTag, "importing pool %p %s %zu of %zu (offset 0x%llx, 0x%llx bytes) failed (%d)", static_cast<void*>(g->base), what, index, count,
                    static_cast<unsigned long long>(off), static_cast<unsigned long long>(size), static_cast<int>(r));
        g_import_failures.add(1);
        c.mem = VK_NULL_HANDLE;
        c.failed = true;
        return false;
    }
    freeze.reset();
    if (r == VK_SUCCESS) r = vkBindBufferMemory(dev.device(), c.buffer, c.mem, 0);
    if (r != VK_SUCCESS) {
        NXLOG_E(kTag, "buffer over pool %p %s %zu of %zu failed (%d)", static_cast<void*>(g->base), what, index, count, static_cast<int>(r));
        c.failed = true;
        return false;
    }
    counters().pools_imported.fetch_add(1);
    {   // a slow import stalls a vkQueueSubmit2 running beside it (the driver): size and where the time went, to tell page pinning from the rest
        const auto tm3 = std::chrono::steady_clock::now();
        auto msd = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        const double total = msd(tm0, tm3);
        if (total >= 20.0)
            NXLOG_I(kTag, "slow pool import: %p, %llu MiB (flags 0x%x, %s %zu of %zu, %s): %.1f ms (host pointer properties %.1f, vkAllocateMemory %.1f, buffer and bind %.1f) on %s",
                    reinterpret_cast<void*>(start), static_cast<unsigned long long>(size >> 20), static_cast<unsigned>(g->flags), what, index, count,
                    c.view ? "through a private view" : "at its own address", total, msd(tm0, tm1), msd(tm1, tm2), msd(tm2, tm3),
                    nxrt::hitch_on_render_thread() ? "the render thread" : "the import worker");
    }
    NXLOG_D(kTag, "pool %p %s %zu of %zu imported: offset 0x%llx, 0x%llx bytes, memory type %u", static_cast<void*>(g->base), what, index, count,
            static_cast<unsigned long long>(off), static_cast<unsigned long long>(size), ty);
    return true;
}

void destroy_chunk(PoolChunk& c) {
    if (c.backing) { c = PoolChunk{}; return; }
    gfx::Device& dev = device();
    if (c.copy.buffer) {
        std::lock_guard<std::mutex> lock(g_copies_mu);
        g_copies.erase(&c);
        g_have_copies.store(!g_copies.empty());
        buffer_views_forget(c.copy.buffer);
        dev.destroy_buffer(c.copy);
        c.buffer = VK_NULL_HANDLE;
    }
    if (c.buffer && !c.failed) buffer_views_forget(c.buffer);
    if (c.buffer) { vkDestroyBuffer(dev.device(), c.buffer, nullptr); vk_gauge_buffers().sub(1); }
    if (c.mem) {
        vkFreeMemory(dev.device(), c.mem, nullptr);
    }
    if (c.view) mem::heap_private_view_unmap(c.view);
    c = PoolChunk{};
}

// The Vulkan side of a pool. A pool no larger than one chunk (every pool, with [nvn] pool_import_chunk_mb = 0) is imported here, whole; a
// larger one only gets its chunk table here, and its chunks are imported as they are touched (pool_chunk).
PoolGpu* import_pool(MemoryPool* p) {
    auto* g = new PoolGpu();
    g->flags = p->flags;
    gfx::Device& dev = device();
    if (!p->base || !p->size) {
        g->failed = true;
        NXLOG_E(kTag, "pool at %p size 0x%llx cannot be imported (%s)", static_cast<void*>(p->base), static_cast<unsigned long long>(p->size),
                !dev.features().external_memory_host ? "VK_EXT_external_memory_host is missing" : "no storage");
        return g;
    }
    const uint64_t align = std::max<uint64_t>(dev.limits_ext().min_imported_host_alignment, 4096);
    const uint64_t start = reinterpret_cast<uint64_t>(p->base) & ~(align - 1);
    const uint64_t end = (reinterpret_cast<uint64_t>(p->base) + p->size + align - 1) & ~(align - 1);
    VkDeviceSize size = end - start;
    static const bool use_view = config_get_bool("nvn", "pool_view", true);      // [nvn] pool_view = 0: import the pool's own address (slow)
    const bool viewed = use_view && mem::heap_has_section() && start >= mem::kHeapBase && start + size <= mem::kHeapBase + mem::heap_reserved();
    const uint64_t stride = import_stride();
    // Chunks overlap, and without private views two of them would import the same host pages: the driver refuses the second
    // (VK_ERROR_INVALID_EXTERNAL_HANDLE, seen on the RX 9060 XT with [memory] heap_section = 0). Unviewed pools are imported whole.
    const bool chunked = viewed && stride && size > stride + import_overlap();
    if (!chunked && size > dev.limits_ext().max_buffer_size && dev.limits_ext().max_buffer_size) {
        NXLOG_W(kTag, "pool of 0x%llx bytes is larger than maxBufferSize 0x%llx: the tail is not reachable by the GPU",
                static_cast<unsigned long long>(size), static_cast<unsigned long long>(dev.limits_ext().max_buffer_size));
        size = dev.limits_ext().max_buffer_size & ~(align - 1);
    }
    g->base = reinterpret_cast<uint8_t*>(start);
    g->size = size;
    g->viewed = viewed;
    if (chunked) {
        g->stride = stride;
        g->clen = stride + import_overlap();
        g->chunks.resize(static_cast<size_t>((size + stride - 1) / stride));
        NXLOG_I(kTag, "pool %p, %llu MiB (flags 0x%x): imported in %zu chunks of %llu MiB (%llu MiB overlapping the next) as they are first used",
                static_cast<void*>(g->base), static_cast<unsigned long long>(size >> 20), static_cast<unsigned>(p->flags), g->chunks.size(),
                static_cast<unsigned long long>(g->clen >> 20), static_cast<unsigned long long>(import_overlap() >> 20));
    } else {
        g->stride = size;
        g->clen = size;
        g->chunks.resize(1);
        // Failed imports become copy buffers on the render thread's first use.
        import_range(g, g->chunks[0], 0, size, "whole", 0, 1);
    }
    { std::lock_guard<std::mutex> lk(g_live_mu); g_live.insert(g); }
    g_gauge_pools_live.add(1);
    return g;
}

bool copy_fallback(PoolGpu* g, PoolChunk& c, uint64_t off, uint64_t size) {
    if (c.failed && !c.copy_tried) {
        // The snapshot is allocated before the chunk is touched: out of memory leaves it as it was (failed, and not tried again).
        std::vector<uint8_t> snap;
        try {
            snap.resize(static_cast<size_t>(size));
        } catch (const std::bad_alloc&) {
            c.copy_tried = true;
            if (log_rate(g_log_fallback))
                NXLOG_E(kTag, "pool import fallback: no memory for a %llu KiB snapshot: pool %p offset 0x%llx is not usable on the GPU",
                        static_cast<unsigned long long>(size >> 10), static_cast<void*>(g->base), static_cast<unsigned long long>(off));
            return false;
        }
        destroy_chunk(c);
        c.tried = c.copy_tried = true;
        c.off = off;
        c.size = size;
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
            VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
            (device().features().transform_feedback ? VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT : 0u);
        if (device().create_buffer(size, usage, gfx::MemoryUse::Readback, &c.copy, "pool import fallback") != VK_SUCCESS) {
            c.failed = true;
            return false;
        }
        c.copy_used = true;
        g_copy_touched.store(true, std::memory_order_relaxed);
        c.buffer = c.copy.buffer;
        c.host = g->base + off;
        c.owner = g;
        c.retry_wait = kImportRetryPresents;
        c.retry_at = presents_now() + kImportRetryPresents;
        c.snapshot = std::move(snap);
        std::memcpy(c.snapshot.data(), c.host, static_cast<size_t>(size));
        std::memcpy(c.copy.mapped, c.snapshot.data(), static_cast<size_t>(size));
        device().flush_buffer(c.copy);
        std::lock_guard<std::mutex> lock(g_copies_mu);
        g_copy_fallbacks.add(1);
        g_copies.insert(&c);
        g_have_copies.store(true);
        if (log_rate(g_log_fallback))
            NXLOG_W(kTag, "pool import fallback: %llu KiB; submissions using copy buffers are synchronized (the import is tried again in %u presents)",
                    (unsigned long long)(size >> 10), kImportRetryPresents);
    }
    return !c.failed && c.buffer;
}

void copy_touch(PoolChunk& c) {
    if (!c.copy.buffer || c.copy_used || !c.host) return;
    std::memcpy(c.snapshot.data(), c.host, static_cast<size_t>(c.size));
    std::memcpy(c.copy.mapped, c.snapshot.data(), static_cast<size_t>(c.size));
    device().flush_buffer(c.copy);
    c.copy_used = true;
    g_copy_touched.store(true, std::memory_order_relaxed);
}

// Chunk k of g, imported now if it was not tried yet (render thread). Null when it cannot be imported.
PoolChunk* pool_chunk(PoolGpu* g, uint64_t k) {
    if (k >= g->chunks.size()) return nullptr;
    PoolChunk& c = g->chunks[static_cast<size_t>(k)];
    if (!c.tried) {
        const uint64_t off = k * g->stride;
        import_range(g, c, off, std::min<uint64_t>(g->clen, g->size - off), "chunk", static_cast<size_t>(k), g->chunks.size());
    }
    copy_fallback(g, c, k * g->stride, std::min<uint64_t>(g->clen, g->size - k * g->stride));
    copy_touch(c);
    return c.failed || !c.buffer ? nullptr : &c;
}

// An import holding pool bytes [off, off + size) when no chunk does: an existing span, or a new one over the strides the range touches.
PoolChunk* pool_span(PoolGpu* g, uint64_t off, uint64_t size) {
    for (PoolChunk& s : g->spans)
        if (off >= s.off && off + size <= s.off + s.size) {
            copy_touch(s);
            return s.failed || !s.buffer ? nullptr : &s;
        }   // a failed span is not tried again
    const uint64_t s0 = off / g->stride * g->stride;
    const uint64_t s1 = std::min<uint64_t>((off + size + g->stride - 1) / g->stride * g->stride, g->size);
    const uint64_t maxb = device().limits_ext().max_buffer_size;
    if (s1 <= s0 || (maxb && s1 - s0 > maxb)) return nullptr;
    NXLOG_I(kTag, "pool %p: a range of %llu KiB at offset 0x%llx crosses a chunk boundary: span import of %llu MiB at offset 0x%llx",
            static_cast<void*>(g->base), static_cast<unsigned long long>(size >> 10), static_cast<unsigned long long>(off),
            static_cast<unsigned long long>((s1 - s0) >> 20), static_cast<unsigned long long>(s0));
    g->spans.emplace_back();
    PoolChunk& s = g->spans.back();
    if (!import_range(g, s, s0, s1 - s0, "span", g->spans.size() - 1, g->spans.size()) &&
        !copy_fallback(g, s, s0, s1 - s0)) return nullptr;
    return &s;
}

// ---- background import --------------------------------------------------------------------------------------------------
// One worker thread imports pools as they are created. The pool is kept alive by a reference while it is queued or running; the render
// thread, when it needs a pool that is still in flight, waits for it instead of importing a second time.
std::mutex g_pf_mu;
std::condition_variable g_pf_cv;
std::deque<Ref<MemoryPool>> g_pf_queue;
std::thread g_pf_thread;
bool g_pf_stop = false;

void prefetch_loop() {
    log_set_thread_name("nvn-pool-import");
    for (;;) {
        Ref<MemoryPool> p;
        {
            std::unique_lock<std::mutex> lk(g_pf_mu);
            g_pf_cv.wait(lk, [] { return g_pf_stop || !g_pf_queue.empty(); });
            if (g_pf_stop) return;
            p = std::move(g_pf_queue.front());
            g_pf_queue.pop_front();
        }
        int expect = 1;
        if (!p->prefetch.compare_exchange_strong(expect, 3)) continue;   // the render thread needed it first and imports it itself
        PoolGpu* g = nullptr;
        if (!p->dead.load()) g = import_pool(p.get());
        {
            std::lock_guard<std::mutex> lk(g_pf_mu);
            p->prefetched.store(g);
            p->prefetch.store(2);
        }
        g_pf_cv.notify_all();
    }
}

void make_dummy() {
    gfx::Device& dev = device();
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                     VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
    if (dev.create_buffer(65536, usage, gfx::MemoryUse::Upload, &g_dummy_buf, "nvn dummy buffer") == VK_SUCCESS) {
        std::memset(g_dummy_buf.mapped, 0, 65536);
        // What the console delivers for a vertex attribute the game did not set up: (0, 0, 0, 1) in the shader's own type.
        const float f1[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        const uint32_t i1[4] = {0, 0, 0, 1};
        std::memcpy(static_cast<uint8_t*>(g_dummy_buf.mapped) + kDummyAttrFloatOffset, f1, 16);
        std::memcpy(static_cast<uint8_t*>(g_dummy_buf.mapped) + kDummyAttrIntOffset, i1, 16);
        g_dummy = g_dummy_buf.buffer;
    }
}

}  // namespace

void prefetch_pool(MemoryPool* p) {
    static const bool enabled = config_get_bool("nvn", "pool_prefetch", true);   // [nvn] pool_prefetch = false: import on first use only
    if (!enabled || !p || p->is_virtual || !p->base || !p->size || !active()) return;
    // a pool larger than one chunk is imported chunk by chunk as the render thread touches it (pool_chunk), never whole
    if (import_stride() && p->size > import_stride() + import_overlap()) return;
    std::lock_guard<std::mutex> lk(g_pf_mu);
    if (g_pf_stop) return;
    int expect = 0;
    if (!p->prefetch.compare_exchange_strong(expect, 1)) return;
    if (!g_pf_thread.joinable()) g_pf_thread = std::thread(prefetch_loop);
    g_pf_queue.push_back(Ref<MemoryPool>::share(p));
    g_pf_cv.notify_all();
}

VkBuffer dummy_buffer() {
    std::call_once(g_dummy_once, make_dummy);
    return g_dummy;
}

namespace {
BufRef resolve_in_pool(uint64_t gpu_address, uint64_t size, MemoryPool** pool_out, PoolGpu** gpu_out, uint64_t* pool_off_out = nullptr);
}

BufRef resolve_buffer(uint64_t gpu_address, uint64_t size) { return resolve_in_pool(gpu_address, size, nullptr, nullptr); }

BufRef resolve_buffer_imported(uint64_t gpu_address, uint64_t size) {
    BufRef out;
    const PoolHit hit = pool_lookup(gpu_address);
    PoolGpu* pg = hit.pool ? hit.pool->gpu : nullptr;
    if (!pg || pg->failed) return out;
    const uint64_t off = gpu_address - reinterpret_cast<uint64_t>(pg->base);
    if (off >= pg->size) return out;
    const uint64_t k = off / pg->stride;
    if (k >= pg->chunks.size()) return out;
    const PoolChunk& c0 = pg->chunks[static_cast<size_t>(k)];
    if (!c0.tried || c0.failed || !c0.buffer || c0.copy.buffer) return out;   // not imported (or a copy buffer): the caller does without
    const PoolChunk* c = &c0;
    if (size && off + size > c->off + c->size) {
        c = nullptr;
        for (const PoolChunk& s : pg->spans)
            if (off >= s.off && off + size <= s.off + s.size && !s.failed && s.buffer && !s.copy.buffer) { c = &s; break; }
        if (!c) return out;
    }
    out.buffer = c->buffer;
    out.offset = off - c->off + c->buffer_offset;
    out.avail = c->off + c->size - off;
    out.ok = size <= out.avail;
    return out;
}

namespace {
// out.buffer / out.offset name the chunk (or span) import holding the range; *pool_off_out receives the offset from the pool's aligned base
// (what the mirror's pages count from). out.avail is what is left of that import, at least the chunk overlap for a chunked pool.
BufRef resolve_in_pool(uint64_t gpu_address, uint64_t size, MemoryPool** pool_out, PoolGpu** gpu_out, uint64_t* pool_off_out) {
    BufRef out;
    const PoolHit hit = pool_lookup(gpu_address);
    PoolGpu* pg = nullptr;
    if (hit.pool) {
        if (!hit.pool->gpu) {
            MemoryPool* mp = hit.pool;
            int expect = 1;
            if (mp->prefetch.load() == 1 && mp->prefetch.compare_exchange_strong(expect, 4)) {
                // still queued: import it here and now rather than behind the pools queued before it
            } else if (mp->prefetch.load() == 3) {
                NXFT("pool.wait");
                nxrt::HitchTimer ht(nxrt::kHpRenderLockWait);
                std::unique_lock<std::mutex> lk(g_pf_mu);
                g_pf_cv.wait(lk, [mp] { return mp->prefetch.load() == 2 || g_pf_stop; });
            }
            if (mp->prefetch.load() == 2) mp->gpu = mp->prefetched.exchange(nullptr);
            if (!mp->gpu) mp->gpu = import_pool(mp);
            ft::mark("pool.used", static_cast<uint32_t>(mp->flags) << 16 | static_cast<uint32_t>(std::min<uint64_t>(mp->size >> 20, 0xffff)));
        }
        pg = hit.pool->gpu;
    }
    // the import starts at the aligned base, which may be below the pool's own base
    const uint64_t off = pg ? gpu_address - reinterpret_cast<uint64_t>(pg->base) : 0;
    const PoolChunk* c = nullptr;
    if (pg && !pg->failed && off < pg->size) {
        c = pool_chunk(pg, off / pg->stride);
        // a range longer than the overlap that runs past its chunk (and not past the pool): a span import holds all of it
        if (c && size && off + size > c->off + c->size && off + size <= pg->size)
            if (const PoolChunk* sp = pool_span(pg, off, size)) c = sp;
    }
    if (!pg || pg->failed || (off < pg->size && !c)) {
        counters().unresolved_addresses.fetch_add(1);
        std::lock_guard<std::mutex> lk(g_log_mu);
        if (g_logged.insert(gpu_address >> 12).second)
            NXLOG_W(kTag, "GPU address 0x%llx (+0x%llx) is not in an importable memory pool: the dummy buffer is bound", static_cast<unsigned long long>(gpu_address),
                    static_cast<unsigned long long>(size));
        out.buffer = dummy_buffer();
        out.avail = 65536;
        return out;
    }
    if (off >= pg->size) {
        counters().unresolved_addresses.fetch_add(1);
        out.buffer = dummy_buffer();
        out.avail = 65536;
        return out;
    }
    out.buffer = c->buffer;
    out.offset = off - c->off + c->buffer_offset;
    out.avail = c->off + c->size - off;
    out.ok = size <= out.avail;
    if (pool_out) *pool_out = hit.pool;
    if (gpu_out) *gpu_out = pg;
    if (pool_off_out) *pool_off_out = off;
    return out;
}
}  // namespace

void pool_copy_retire(PoolGpu* p) {
    if (!p || !g_have_copies.load()) return;
    std::lock_guard<std::mutex> lock(g_copies_mu);
    for (PoolChunk& c : p->chunks) if (c.copy.buffer) c.host = nullptr;
    for (PoolChunk& c : p->spans) if (c.copy.buffer) c.host = nullptr;
}
// True while recording a submission that uses a copy buffer: that submission is waited for and its GPU writes are copied back (nvn_gpu.cpp).
// Submissions that touch no copy buffer are not synchronized, even while copies exist.
bool pool_copy_pending() { return g_copy_touched.load(std::memory_order_relaxed); }

namespace {
// Pool parts on copy buffers whose retry is due try the real import again (start of a submission, render thread). The copy buffer's last
// user was a synchronized submission whose writes are back in the pool, so on success the copy is dropped (destroyed after the submissions
// made so far) and the chunk is the import from now on.
void retry_imports() {
    const uint32_t now = presents_now();
    std::vector<PoolChunk*> due;
    {
        std::lock_guard<std::mutex> lock(g_copies_mu);
        for (PoolChunk* c : g_copies)
            if (c->host && c->owner && c->retry_at && static_cast<int32_t>(now - c->retry_at) >= 0) due.push_back(c);
    }
    for (PoolChunk* c : due) {
        // a compute result waiting to be copied back into this buffer (out_add) keeps it until the entry is gone
        if (out_uses_buffer(c->copy.buffer)) { c->retry_at = now + 60; continue; }
        PoolChunk fresh;
        if (!import_range(c->owner, fresh, c->off, c->size, "retry", 0, 1)) {
            destroy_chunk(fresh);
            c->retry_wait = std::min<uint32_t>(std::max<uint32_t>(c->retry_wait, kImportRetryPresents) * 2, kImportRetryMax);
            c->retry_at = now + c->retry_wait;
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(g_copies_mu);
            g_copies.erase(c);
            g_have_copies.store(!g_copies.empty());
        }
        NXLOG_I(kTag, "pool %p: offset 0x%llx, %llu KiB imported on a retry: its copy buffer is dropped", static_cast<void*>(c->owner->base),
                static_cast<unsigned long long>(c->off), static_cast<unsigned long long>(c->size >> 10));
        auto* old = new PoolChunk(std::move(*c));
        *c = std::move(fresh);
        defer_destroy([old] { destroy_chunk(*old); delete old; });
    }
}
}  // namespace

void pool_copy_begin() {
    g_copy_touched.store(false, std::memory_order_relaxed);
    if (!g_have_copies.load()) return;
    {
        std::lock_guard<std::mutex> lock(g_copies_mu);
        for (PoolChunk* c : g_copies) c->copy_used = false;
    }
    retry_imports();
}
void pool_copy_end() {
    if (!pool_copy_pending()) return;
    std::lock_guard<std::mutex> lock(g_copies_mu);
    for (PoolChunk* c : g_copies) {
        if (!c->host || !c->copy_used) continue;
        device().invalidate_buffer(c->copy);
        const auto* data = static_cast<const uint8_t*>(c->copy.mapped);
        const uint8_t* snap = c->snapshot.data();
        // Only GPU-modified bytes are copied back: preserve unrelated concurrent CPU writes. Pages the GPU did not change are skipped whole.
        const size_t n = static_cast<size_t>(c->size);
        for (size_t p = 0; p < n; p += 4096) {
            const size_t e = std::min<size_t>(n, p + 4096);
            if (!std::memcmp(data + p, snap + p, e - p)) continue;
            for (size_t i = p; i < e; ++i)
                if (data[i] != snap[i]) c->host[i] = data[i];
        }
    }
}

void pool_shutdown() {
    {   // stop the importer first: a pool it is working on finishes, queued ones are dropped
        std::unique_lock<std::mutex> lk(g_pf_mu);
        g_pf_stop = true;
        g_pf_queue.clear();
    }
    g_pf_cv.notify_all();
    if (g_pf_thread.joinable()) g_pf_thread.join();
    {
        std::lock_guard<std::mutex> lk(g_pf_mu);
        g_pf_stop = false;           // a later back end (tests bring it up and down) starts clean
    }
    std::vector<PoolGpu*> left;
    {
        std::lock_guard<std::mutex> lk(g_live_mu);
        left.assign(g_live.begin(), g_live.end());
    }
    for (PoolGpu* p : left) pool_gpu_destroy(p);
    { std::lock_guard<std::mutex> lock(g_slabs_mu); g_slabs.clear(); g_rejected_slabs.clear(); }
    if (g_dummy_buf.buffer) buffer_views_forget(g_dummy_buf.buffer);
    if (g_dummy_buf.buffer) device().destroy_buffer(g_dummy_buf);
    g_dummy = VK_NULL_HANDLE;
    g_dummy_buf = gfx::Buffer();
    new (&g_dummy_once) std::once_flag();
}

uint64_t pool_imports() { return counters().pools_imported.load(); }

void pool_gpu_destroy(PoolGpu* p) {
    if (!p) return;
    bool was_live = false;
    {
        std::lock_guard<std::mutex> lk(g_live_mu);
        was_live = g_live.erase(p) != 0;
        if (!was_live && !p->failed) return;   // already destroyed by pool_shutdown
    }
    if (was_live) g_gauge_pools_live.sub(1);
    if (p->mirror) mirror_destroy(p);
    for (PoolChunk& c : p->chunks) destroy_chunk(c);
    for (PoolChunk& c : p->spans) destroy_chunk(c);
    delete p;
}


// ---- device-local copies of pool pages (S38) ---------------------------------------------------------------------
// The game's vertex, index and uniform data live in its memory pools, which are host memory: every draw read them over PCIe (S34). A pool
// that draws read from gets a device-local buffer of the same size (the mirror). A page is copied into it, in a transfer at the start of a
// submission, after some draw wanted it; from then on draws whose whole range is in valid pages read the mirror. A page stops being valid
// when the CPU writes it (the write tracker made it read-only when it was copied, nxrt/write_watch.h) or when the GPU writes it (copies,
// UpdateUniformBuffer, reports, storage buffers: the replay calls note_gpu_write). Until it is copied again, draws that touch it read the
// pool itself, as before. A page written again soon after every copy (a ring buffer the game refills each frame) is "hot": it is left in
// host memory for a while instead of faulting and being copied every frame.
//
// The copy is made of 64 MiB chunks, each created the first time a draw wants one of its pages, so a pool of gigabytes (Pokopia's 3.5 GiB
// graphics heap) costs video memory only for the parts draws read. A bind that spans two chunks reads the pool.
//
// [nvn] pool_mirror = on (default) | off.
constexpr uint64_t kChunkBytes = 64ull << 20;
struct Mirror {
    std::vector<gfx::Buffer> chunks;    // the copy, kChunkBytes each (the last one may be shorter); empty until a draw wants a page in it
    PoolGpu* pg = nullptr;              // the pool (the copies read its chunk imports, mirror_src)
    mem::ww::Region* ww = nullptr;      // the CPU's writes to the pool's pages
    uint64_t npages = 0;
    std::vector<uint64_t> want;         // bitmap: a draw wanted the page while it was not valid
    std::vector<uint32_t> copied_at;    // present count of the last copy (0: never)
    std::vector<uint64_t> used;         // bitmap: a draw read the page from the copy since it was copied
    std::vector<uint8_t> score;         // copies in a row that the CPU overwrote before any draw read them
    std::vector<uint8_t> backoff;       // how often the page has been hot in a row (the hold doubles each time)
    std::vector<uint32_t> hot_until;    // present count until which the page is not copied
    bool any_want = false;
    int pool_flags = 0;
    int64_t st_written = 0, st_copied = 0;   // pages the CPU wrote after a copy, pages copied (summary line)
    uint64_t pool_bytes = 0;
    // compute out (below): pages where a compute result in the copy is newer than the pool, and pages the CPU has read compute results
    // from (any thread sets them; atomics)
    std::unique_ptr<std::atomic<uint64_t>[]> gpu_new, cpu_read;
};

namespace {

enum class MirrorMode { Off, On };
constexpr uint64_t kPageBytes = 4096;
constexpr uint64_t kMaxBindPages = (64ull << 20) / kPageBytes;      // a binding over more than 64 MiB stays in the pool
constexpr uint64_t kCopyBudget = 64ull << 20;                        // bytes copied per submission at most (an area load can want hundreds of MiB)
constexpr uint8_t kHotHits = 2;                                      // two copies in a row overwritten before any draw read them: the page is hot ...
constexpr uint32_t kHotHold = 600;                                   // ... and is not copied for 600 presents, twice that the next time, ...
constexpr uint8_t kHotMaxShift = 4;                                  // ... up to 16 times
constexpr uint32_t kHotWindow = 4;                                   // a write this many presents or fewer after the copy counts even when draws read it

// [nvn] mirror_hot_used (default on): a page the CPU rewrites within kHotWindow presents of its copy, kHotHits times in a row, goes hot even
// when draws read it (per-frame vertices, uniforms and texels). Each such rewrite was a write fault on the game's thread (a vectored exception
// and a VirtualProtect, tens of microseconds) and a copy; hot, the page stays writable and draws read it from the pool, like data the console
// streams, while geometry the game writes once stays in the device-local copy. It is tried again after the hold, like any hot page.
bool hot_used_on() {
    static const bool on = config_get_bool("nvn", "mirror_hot_used", true);
    return on;
}

std::vector<Mirror*> g_mirrors;      // render thread
uint64_t g_mirror_bytes = 0;         // video memory of the chunks made
bool g_mirror_on = false;            // this submission reads mirrors (mode on and, with [nvn] gpuperf_ab, an "on" window)
std::vector<uint64_t> g_written_tmp;
std::vector<VkBufferCopy> g_copy_tmp;

// Arming budget of one submission ([nvn] mirror_arm_ms, default 3; 0 = no limit). Each arm is a VirtualQuery walk and a VirtualProtect; while
// the game streams an area they took the render thread up to 490 ms a frame (hitch line "write-watch arming"). Past the budget no page is
// armed or copied in this submission: draws read those pages from the pool itself (always current), and the copies wait for the next
// submissions. Render thread.
int64_t g_arm_spent_ns = 0;
std::atomic<uint64_t> g_st_arm_deferred{0};
std::atomic<uint64_t> g_st_chunk_bytes_made{0};   // video memory of the copy chunks made, ever (the hitch line: new chunks during a slow frame)

int64_t arm_budget_ns() {
    static const int64_t ns = std::max<int64_t>(0, config_get_int("nvn", "mirror_arm_ms", 3)) * 1000000;
    return ns;
}

bool arm_in_budget() {
    const int64_t b = arm_budget_ns();
    if (!b || g_arm_spent_ns < b) return true;
    g_st_arm_deferred.fetch_add(1, std::memory_order_relaxed);
    return false;
}

bool arm_timed(mem::ww::Region* r, uint64_t first, uint64_t count) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = mem::ww::arm(r, first, count);
    g_arm_spent_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

std::atomic<int64_t> g_st_dev_binds{0}, g_st_host_binds{0}, g_st_copied{0}, g_st_hot{0}, g_st_live_bytes{0};
// statistics for the summary line (render thread): binds by use (to the copy / to the pool / not eligible), pages made stale by GPU writes by use
int64_t g_use_dev[3] = {}, g_use_host[3] = {}, g_use_skip[3] = {};
int64_t g_skip_why[5] = {};          // binds not eligible: outside a pool (or past its end), over 64 MiB, pool without a copy, two chunks, no chunk
int64_t g_wr_pages[6] = {}, g_wr_calls[6] = {};

MirrorMode mirror_mode() {
    static const MirrorMode m = [] {
        const std::string v = config_get_str("nvn", "pool_mirror", "on");
        if (v == "on" || v == "1" || v == "true") return MirrorMode::On;
        return MirrorMode::Off;
    }();
    return m;
}

// The cap of the copies ([nvn] pool_mirror_mb, or from the card's video memory when it is not set; nvn_mirror_cap.h), logged once.
uint64_t mirror_budget() {
    uint64_t vram = 0;
    bool shared = false;
    if (VkPhysicalDevice pd = device().physical()) {
        VkPhysicalDeviceProperties pp{};
        vkGetPhysicalDeviceProperties(pd, &pp);
        shared = pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        for (uint32_t h = 0; h < mp.memoryHeapCount; ++h)
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) vram = std::max<uint64_t>(vram, mp.memoryHeaps[h].size);
    }
    const int64_t set_mb = config_has("nvn", "pool_mirror_mb") ? std::max<int64_t>(0, config_get_int("nvn", "pool_mirror_mb", 0)) : -1;
    const MirrorCap c = mirror_cap(vram, set_mb, shared);
    NXLOG_I(kTag, "pool copies: cap %llu MiB (%s; largest device-local heap %llu MiB)%s", static_cast<unsigned long long>(c.bytes >> 20), c.why,
            static_cast<unsigned long long>(vram >> 20), c.bytes ? "" : ": no device-local copies");
    return c.bytes;
}

uint64_t mirror_cap_bytes() {
    static const uint64_t b = mirror_budget();
    return b;
}

uint32_t presents_now() { return static_cast<uint32_t>(counters().presents.load(std::memory_order_relaxed)); }

bool bit_get(const std::vector<uint64_t>& v, uint64_t i) { return (v[static_cast<size_t>(i >> 6)] >> (i & 63)) & 1; }
void bit_put(std::vector<uint64_t>& v, uint64_t i, bool on) {
    if (on) v[static_cast<size_t>(i >> 6)] |= uint64_t{1} << (i & 63);
    else v[static_cast<size_t>(i >> 6)] &= ~(uint64_t{1} << (i & 63));
}

bool page_valid(const Mirror& m, uint64_t p) { return mem::ww::page_valid(m.ww, p); }
void out_register(Mirror* m);
bool new_servable(const Mirror& m, uint64_t first, uint64_t last);

bool range_valid(const Mirror& m, uint64_t first, uint64_t count) { return mem::ww::all_valid(m.ww, first, count); }

Mirror* mirror_for(PoolGpu* pg, MemoryPool* mp) {
    for (const PoolChunk& c : pg->chunks) if (c.copy.buffer) return nullptr;
    for (const PoolChunk& c : pg->spans) if (c.copy.buffer) return nullptr;
    if (pg->mirror || pg->mirror_tried) return pg->mirror;
    pg->mirror_tried = true;
    if (!mp || mp->is_virtual || mp->is_shader_code || mp->dead.load() || pg->failed) return nullptr;
    // Tracked copies change the protection of the CPU's view of the pages. With [nvn] pool_view = false the GPU import is that same view;
    // that combination is not tried: no copy.
    if (!pg->viewed) return nullptr;
    const uint64_t budget = mirror_cap_bytes();
    if (budget == 0) return nullptr;

    auto m = std::make_unique<Mirror>();
    // Without tracking of the CPU's writes there is no copy: a copy that misses a write serves old vertices and indices, and after an area
    // load those can point anywhere (an untracked experiment lost the Vulkan device at the first load).
    m->ww = mem::ww::region_create(reinterpret_cast<uint64_t>(pg->base), pg->size, "pool mirror / lazy compute");
    if (!m->ww) {
        NXLOG_I(kTag, "pool %p (%llu KiB) gets no device-local copy: its writes cannot be tracked (not in the heap's primary view)",
                static_cast<void*>(pg->base), static_cast<unsigned long long>(pg->size >> 10));
        return nullptr;
    }
    m->chunks.resize(static_cast<size_t>((pg->size + kChunkBytes - 1) / kChunkBytes));
    m->pg = pg;
    m->pool_flags = mp->flags;
    m->npages = pg->size / kPageBytes;
    const size_t words = static_cast<size_t>((m->npages + 63) / 64);
    m->want.assign(words, 0);
    m->copied_at.assign(static_cast<size_t>(m->npages), 0);
    m->used.assign(words, 0);
    m->score.assign(static_cast<size_t>(m->npages), 0);
    m->backoff.assign(static_cast<size_t>(m->npages), 0);
    m->hot_until.assign(static_cast<size_t>(m->npages), 0);
    m->pool_bytes = pg->size;
    m->gpu_new.reset(new std::atomic<uint64_t>[words]);
    m->cpu_read.reset(new std::atomic<uint64_t>[words]);
    for (size_t i = 0; i < words; ++i) { m->gpu_new[i].store(0, std::memory_order_relaxed); m->cpu_read[i].store(0, std::memory_order_relaxed); }
    out_register(m.get());
    NXLOG_I(kTag, "pool %p: device-local copy (%llu KiB in %zu chunks made on demand, pool flags 0x%x, %s)", static_cast<void*>(pg->base),
            static_cast<unsigned long long>(pg->size >> 10), m->chunks.size(), static_cast<unsigned>(mp->flags), "writes tracked");
    pg->mirror = m.release();
    g_mirrors.push_back(pg->mirror);
    return pg->mirror;
}

// The CPU wrote these pages since their copy: they wait for the next draw that wants them. A copy the CPU overwrote before any draw read it
// was wasted (data the game writes for one frame, a ring buffer): after kHotHits of those in a row the page is hot and stays in the pool for
// a hold that doubles every time it comes back.
void absorb_writes(Mirror& m, uint32_t now) {
    g_written_tmp.clear();
    mem::ww::take_written(m.ww, g_written_tmp);
    m.st_written += static_cast<int64_t>(g_written_tmp.size());
    for (uint64_t p : g_written_tmp) {
        const size_t i = static_cast<size_t>(p);
        bit_put(m.want, p, false);
        if (m.copied_at[i] == 0) continue;                     // never copied: nothing was wasted
        const bool soon = hot_used_on() && now - m.copied_at[i] <= kHotWindow;   // rewritten right after its copy: per-frame data
        if (bit_get(m.used, p) && !soon) {
            m.score[i] = 0;
            m.backoff[i] = 0;
            continue;
        }
        if (++m.score[i] >= kHotHits) {
            if (m.hot_until[i] <= now) g_st_hot.fetch_add(1, std::memory_order_relaxed);
            m.hot_until[i] = now + (kHotHold << m.backoff[i]);
            if (m.backoff[i] < kHotMaxShift) ++m.backoff[i];
            m.score[i] = 0;
        }
    }
}

bool new_get(const Mirror& m, uint64_t p) { return (m.gpu_new[static_cast<size_t>(p >> 6)].load(std::memory_order_acquire) >> (p & 63)) & 1; }

bool copyable(const Mirror& m, uint64_t i, uint32_t now) {
    return bit_get(m.want, i) && !page_valid(m, i) && !new_get(m, i) && m.hot_until[static_cast<size_t>(i)] <= now && !mem::ww::page_banned(m.ww, i);
}

void log_summary(uint32_t now) {
    static uint32_t last = 0;
    if (now - last < 1200) return;
    last = now;
    uint64_t valid = 0, hot = 0;
    for (const Mirror* m : g_mirrors)
        for (uint64_t p = 0; p < m->npages; ++p) {
            valid += page_valid(*m, p) ? 1 : 0;
            hot += m->hot_until[static_cast<size_t>(p)] > now ? 1 : 0;
        }
    NXLOG_I(kTag, "pool mirrors: %zu, %llu MiB, valid %llu MiB, hot %llu KiB; binds to mirrors %lld, to pools %lld; copied %lld MiB; write faults %llu",
            g_mirrors.size(), static_cast<unsigned long long>(g_mirror_bytes >> 20), static_cast<unsigned long long>(valid * kPageBytes >> 20),
            static_cast<unsigned long long>(hot * kPageBytes >> 10), static_cast<long long>(g_st_dev_binds.load()), static_cast<long long>(g_st_host_binds.load()),
            static_cast<long long>(g_st_copied.load() >> 20), static_cast<unsigned long long>(mem::ww::faults()));
    NXLOG_I(kTag, "pool mirror binds (copy/pool/not eligible): vertex %lld/%lld/%lld index %lld/%lld/%lld uniform %lld/%lld/%lld; GPU writes (calls/pages): storage %lld/%lld "
            "stream-out %lld/%lld image copy %lld/%lld update %lld/%lld fill %lld/%lld copy %lld/%lld",
            (long long)g_use_dev[0], (long long)g_use_host[0], (long long)g_use_skip[0], (long long)g_use_dev[1], (long long)g_use_host[1], (long long)g_use_skip[1],
            (long long)g_use_dev[2], (long long)g_use_host[2], (long long)g_use_skip[2], (long long)g_wr_calls[0], (long long)g_wr_pages[0], (long long)g_wr_calls[1],
            (long long)g_wr_pages[1], (long long)g_wr_calls[2], (long long)g_wr_pages[2], (long long)g_wr_calls[3], (long long)g_wr_pages[3], (long long)g_wr_calls[4],
            (long long)g_wr_pages[4], (long long)g_wr_calls[5], (long long)g_wr_pages[5]);
    NXLOG_I(kTag, "pool mirror binds not eligible: outside a pool %lld, over 64 MiB %lld, pool without a copy %lld, across two chunks %lld, no chunk %lld",
            (long long)g_skip_why[0], (long long)g_skip_why[1], (long long)g_skip_why[2], (long long)g_skip_why[3], (long long)g_skip_why[4]);
    std::vector<const Mirror*> top(g_mirrors.begin(), g_mirrors.end());
    std::sort(top.begin(), top.end(), [](const Mirror* a, const Mirror* b) { return a->st_copied > b->st_copied; });
    for (size_t i = 0; i < top.size() && i < 6; ++i) {
        uint64_t v = 0;
        for (uint64_t p = 0; p < top[i]->npages; ++p) v += page_valid(*top[i], p) ? 1 : 0;
        NXLOG_I(kTag, "  mirror of %llu KiB (flags 0x%x): valid %llu KiB, copied %lld KiB, written by the CPU after a copy %lld KiB", (unsigned long long)(top[i]->npages * 4),
                top[i]->pool_flags, (unsigned long long)(v * 4), (long long)(top[i]->st_copied * 4), (long long)(top[i]->st_written * 4));
    }
}

struct MirrorGauges {
    MirrorGauges() {
        nxrt::gauge_register("nvn.mirror.bytes", [] { return g_st_live_bytes.load(std::memory_order_relaxed); });
        nxrt::gauge_register("nvn.mirror.copied_mb", [] { return g_st_copied.load(std::memory_order_relaxed) >> 20; });
        nxrt::gauge_register("nvn.mirror.dev_binds", [] { return g_st_dev_binds.load(std::memory_order_relaxed); });
        nxrt::gauge_register("nvn.mirror.host_binds", [] { return g_st_host_binds.load(std::memory_order_relaxed); });
        nxrt::gauge_register("nvn.mirror.hot_pages", [] { return g_st_hot.load(std::memory_order_relaxed); });
        // the hitch log (nxrt/gauge.h): growth during a slow frame
        nxrt::hitch_add_counter("pool_mirror_copied_kb", []() -> uint64_t { return static_cast<uint64_t>(g_st_copied.load(std::memory_order_relaxed)) >> 10; });
        nxrt::hitch_add_counter("write_faults", []() -> uint64_t { return mem::ww::faults(); });
        nxrt::hitch_add_counter("mirror_arms_deferred", []() -> uint64_t { return g_st_arm_deferred.load(std::memory_order_relaxed); });
        nxrt::hitch_add_counter("pool_mirror_chunks_mb", []() -> uint64_t { return g_st_chunk_bytes_made.load(std::memory_order_relaxed) >> 20; });
        nxrt::hitch_add_counter("pools_imported", []() -> uint64_t { Counters* c = counters_if(); return c ? c->pools_imported.load(std::memory_order_relaxed) : 0; });
    }
    ~MirrorGauges() {
        for (const char* n : {"nvn.mirror.bytes", "nvn.mirror.copied_mb", "nvn.mirror.dev_binds", "nvn.mirror.host_binds", "nvn.mirror.hot_pages"}) nxrt::gauge_unregister(n);
    }
};
MirrorGauges g_mirror_gauges;

}  // namespace

void out_unregister(Mirror* m);
void out_retire();
void out_log();

bool mirror_enabled() { return mirror_mode() != MirrorMode::Off; }

// The chunk of the copy that holds page `page`, made now if it is not yet (null: over the budget or out of video memory).
VkBuffer mirror_chunk(Mirror& m, uint64_t page) {
    const size_t ci = static_cast<size_t>(page * kPageBytes / kChunkBytes);
    if (ci >= m.chunks.size()) return VK_NULL_HANDLE;
    gfx::Buffer& c = m.chunks[ci];
    if (c.buffer) return c.buffer;
    const uint64_t budget = mirror_cap_bytes();
    const uint64_t bytes = std::min<uint64_t>(kChunkBytes, m.pool_bytes - ci * kChunkBytes);
    if (g_mirror_bytes + bytes > budget) {
        static bool logged = false;
        if (!logged) { logged = true; NXLOG_I(kTag, "pool copies: the budget is used up (%llu MiB): further chunks stay in the pools", static_cast<unsigned long long>(g_mirror_bytes >> 20)); }
        return VK_NULL_HANDLE;
    }
    // texel buffer and transfer source usage: compute dispatches bind the copy directly as storage and texel buffers, and their results go
    // back to the pool from it (nvn_replay.cpp, compute shadows)
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
    if (device().create_buffer(bytes, usage, gfx::MemoryUse::GpuOnly, &c, "nvn pool mirror") != VK_SUCCESS) {
        c = gfx::Buffer{};
        static bool logged = false;
        if (!logged) { logged = true; NXLOG_W(kTag, "a device-local chunk of %llu MiB could not be made: those pages stay in the pool", static_cast<unsigned long long>(bytes >> 20)); }
        return VK_NULL_HANDLE;
    }
    g_mirror_bytes += bytes;
    g_st_chunk_bytes_made.fetch_add(bytes, std::memory_order_relaxed);
    g_st_live_bytes.store(static_cast<int64_t>(g_mirror_bytes), std::memory_order_relaxed);
    return c.buffer;
}

// The pool import the copies of mirror chunk `page` belongs to read from (imported now if it is not yet), and in *base the pool offset of its
// byte 0. A mirror chunk lies wholly in one import chunk: import strides are multiples of kChunkBytes. Null: the pool part cannot be imported.
VkBuffer mirror_src(Mirror& m, uint64_t page, uint64_t* base) {
    const PoolChunk* c = pool_chunk(m.pg, page * kPageBytes / m.pg->stride);
    if (!c) return VK_NULL_HANDLE;
    *base = c->off - c->buffer_offset;
    return c->buffer;
}

BufRef resolve_read(uint64_t gpu_address, uint64_t size, int use) {
    MemoryPool* mp = nullptr;
    PoolGpu* pg = nullptr;
    uint64_t poff = 0;
    BufRef out = resolve_in_pool(gpu_address, size, &mp, &pg, &poff);
    if (!g_mirror_on) return out;
    if (!out.ok || !pg || !size) { g_use_skip[use % 3]++; g_skip_why[0]++; return out; }
    const uint64_t first = poff / kPageBytes, last = (poff + size - 1) / kPageBytes;
    if (last - first + 1 > kMaxBindPages) { g_use_skip[use % 3]++; g_skip_why[1]++; return out; }
    Mirror* m = mirror_for(pg, mp);
    if (!m || last >= m->npages) { g_use_skip[use % 3]++; g_skip_why[2]++; return out; }
    const uint64_t chunk_pages = kChunkBytes / kPageBytes;
    if (first / chunk_pages != last / chunk_pages) { g_use_skip[use % 3]++; g_skip_why[3]++; return out; }
    const VkBuffer cb = mirror_chunk(*m, first);
    if (!cb) { g_use_skip[use % 3]++; g_skip_why[4]++; return out; }
    if (range_valid(*m, first, last - first + 1) || (out_lazy() && new_servable(*m, first, last))) {
        out.buffer = cb;
        out.offset = poff - (first / chunk_pages) * kChunkBytes;
        out.avail = std::min<uint64_t>(kChunkBytes, m->pool_bytes - (first / chunk_pages) * kChunkBytes) - out.offset;
        for (uint64_t p = first; p <= last; ++p) bit_put(m->used, p, true);
        g_st_dev_binds.fetch_add(1, std::memory_order_relaxed);
        g_use_dev[use % 3]++;
        return out;
    }
    g_use_host[use % 3]++;
    for (uint64_t p = first; p <= last; ++p) bit_put(m->want, p, true);
    m->any_want = true;
    g_st_host_binds.fetch_add(1, std::memory_order_relaxed);
    return out;
}

MirrorSpan mirror_span(uint64_t gpu_address, uint64_t size) {
    MirrorSpan s;
    if (!g_mirror_on || !size) return s;
    MemoryPool* mp = nullptr;
    PoolGpu* pg = nullptr;
    uint64_t poff = 0;
    const BufRef out = resolve_in_pool(gpu_address, size, &mp, &pg, &poff);
    if (!out.ok || !pg) return s;
    const uint64_t first = poff / kPageBytes, last = (poff + size - 1) / kPageBytes;
    if (last - first + 1 > kMaxBindPages) return s;
    Mirror* m = mirror_for(pg, mp);
    if (!m || last >= m->npages) return s;
    const uint64_t chunk_pages = kChunkBytes / kPageBytes;
    if (first / chunk_pages != last / chunk_pages) return s;
    const VkBuffer cb = mirror_chunk(*m, first);
    if (!cb) return s;
    s.buf = cb;
    s.offset = poff - (first / chunk_pages) * kChunkBytes;
    s.valid = range_valid(*m, first, last - first + 1) || (out_lazy() && new_servable(*m, first, last));
    s.avail = std::min<uint64_t>(kChunkBytes, m->pool_bytes - (first / chunk_pages) * kChunkBytes) - s.offset;
    return s;
}

void mirror_span_written(uint64_t gpu_address, uint64_t size) {
    if (!g_mirror_on || !size) return;
    MemoryPool* mp = nullptr;
    PoolGpu* pg = nullptr;
    uint64_t poff = 0;
    const BufRef out = resolve_in_pool(gpu_address, size, &mp, &pg, &poff);
    if (!out.ok || !pg || !pg->mirror) return;
    Mirror& m = *pg->mirror;
    const uint64_t first = (poff + kPageBytes - 1) / kPageBytes, end = (poff + size) / kPageBytes;   // pages wholly inside
    if (first >= end || end > m.npages) return;
    const uint32_t now = presents_now();
    for (uint64_t p = first; p < end; ++p)
        if (m.hot_until[static_cast<size_t>(p)] > now || mem::ww::page_banned(m.ww, p)) return;     // per-frame CPU data: leave it untracked
    if (!arm_in_budget() || !arm_timed(m.ww, first, end - first)) return;   // not armed: the pages stay invalid and are read from the pool
    for (uint64_t p = first; p < end; ++p) {
        m.copied_at[static_cast<size_t>(p)] = now ? now : 1;
        bit_put(m.want, p, false);
    }
}

bool mirror_fill(VkCommandBuffer cb, uint64_t gpu_address, uint64_t size, uint64_t* bytes) {
    if (bytes) *bytes = 0;
    if (!g_mirror_on || !size) return false;
    MemoryPool* mp = nullptr;
    PoolGpu* pg = nullptr;
    uint64_t poff = 0;
    const BufRef out = resolve_in_pool(gpu_address, size, &mp, &pg, &poff);
    if (!out.ok || !pg || !pg->mirror) return false;
    Mirror& m = *pg->mirror;
    const uint64_t first = poff / kPageBytes, last = (poff + size - 1) / kPageBytes;
    const uint64_t chunk_pages = kChunkBytes / kPageBytes;
    if (last >= m.npages || last - first + 1 > kMaxBindPages || first / chunk_pages != last / chunk_pages) return false;
    const VkBuffer chunk = mirror_chunk(m, first);
    if (!chunk) return false;
    uint64_t src_base = 0;
    const VkBuffer src = mirror_src(m, first, &src_base);
    if (!src) return false;
    const uint32_t now = presents_now();
    for (uint64_t p = first; p <= last; ++p)      // per-frame CPU data (hot) or pages with another view (banned) stay in the pool
        if (!page_valid(m, p) && (m.hot_until[static_cast<size_t>(p)] > now || mem::ww::page_banned(m.ww, p))) return false;
    static std::vector<VkBufferCopy> copies;
    copies.clear();
    const uint64_t base = (first / chunk_pages) * kChunkBytes;
    bool ok = true;
    for (uint64_t p = first; p <= last;) {
        if (page_valid(m, p) || new_get(m, p)) { ++p; continue; }
        uint64_t q = p + 1;
        while (q <= last && !page_valid(m, q) && !new_get(m, q)) ++q;
        if (arm_in_budget() && arm_timed(m.ww, p, q - p)) {   // valid from now until the CPU writes a page; the copy below reads the pool when it runs
            for (uint64_t i = p; i < q; ++i) {
                m.copied_at[static_cast<size_t>(i)] = now ? now : 1;
                bit_put(m.used, i, true);
                bit_put(m.want, i, false);
            }
            copies.push_back(VkBufferCopy{p * kPageBytes - src_base, p * kPageBytes - base, (q - p) * kPageBytes});
            m.st_copied += static_cast<int64_t>(q - p);
            g_st_copied.fetch_add(static_cast<int64_t>((q - p) * kPageBytes), std::memory_order_relaxed);
            if (bytes) *bytes += (q - p) * kPageBytes;
        } else {
            ok = false;
        }
        p = q;
    }
    if (!copies.empty()) vkCmdCopyBuffer(cb, src, chunk, static_cast<uint32_t>(copies.size()), copies.data());
    return ok && (range_valid(m, first, last - first + 1) || new_servable(m, first, last));
}

bool note_gpu_write(uint64_t gpu_address, uint64_t size, int use) {
    if (g_mirrors.empty() || !size) return false;
    const PoolHit hit = pool_lookup(gpu_address);
    if (!hit.pool || !hit.pool->gpu || !hit.pool->gpu->mirror) return false;
    PoolGpu* pg = hit.pool->gpu;
    Mirror& m = *pg->mirror;
    const uint64_t off = gpu_address - reinterpret_cast<uint64_t>(pg->base);
    if (off >= pg->size) return false;
    const uint64_t first = off / kPageBytes, last = std::min<uint64_t>((off + size - 1) / kPageBytes, m.npages - 1);
    g_wr_calls[use % 6]++;
    g_wr_pages[use % 6] += static_cast<int64_t>(last - first + 1);
    // word by word (page by page this was ~3.5 % of the render thread: draws note their storage buffers, MiBs each; notes/rt.md)
    const bool stale = mem::ww::any_valid(m.ww, first, last - first + 1);
    mem::ww::invalidate(m.ww, first, last - first + 1);
    for (uint64_t p = first; p <= last;) {
        const uint64_t lo = p & 63, hi = std::min<uint64_t>(64, lo + (last + 1 - p));
        const uint64_t mask = (hi - lo == 64) ? ~uint64_t{0} : (((uint64_t{1} << (hi - lo)) - 1) << lo);
        m.want[static_cast<size_t>(p >> 6)] &= ~mask;
        p += hi - lo;
    }
    return stale;
}

void mirror_begin_submit(Frame& f, bool on) {
    g_mirror_on = on && mirror_mode() != MirrorMode::Off;
    out_retire();
    out_log();
    if (!g_mirror_on) {
        if (out_any()) out_flush(f.cb, 0, UINT64_MAX, kOutMirrorOff);   // draws of this submission read the pools: results go back first
        return;
    }
    const uint32_t now = presents_now();
    log_summary(now);
    uint64_t budget = kCopyBudget;
    bool barrier_done = false;
    g_arm_spent_ns = 0;          // the arming budget is per submission (mirror_fill and mirror_span_written during recording count too)
    for (Mirror* mp : g_mirrors) {
        Mirror& m = *mp;
        absorb_writes(m, now);
        if (!m.any_want) continue;
        if (!arm_in_budget()) continue;   // the wanted pages wait (any_want stays set)
        m.any_want = false;
        g_copy_tmp.clear();
        uint64_t p = 0;
        while (p < m.npages) {
            if (!m.want[static_cast<size_t>(p >> 6)]) { p = (p | 63) + 1; continue; }     // a word with nothing wanted
            if (!bit_get(m.want, p)) { ++p; continue; }
            if (!copyable(m, p, now)) { bit_put(m.want, p, false); ++p; continue; }     // valid already, hot or banned: the next draw asks again
            if (budget < kPageBytes) { m.any_want = true; break; }                      // the rest waits for the next submission
            if (!arm_in_budget()) { m.any_want = true; break; }                         // arming took its time for this submission: the same
            const uint64_t chunk_pages = kChunkBytes / kPageBytes;
            const uint64_t chunk_end = (p / chunk_pages + 1) * chunk_pages;
            uint64_t src_base = 0;
            if (!mirror_chunk(m, p) || !mirror_src(m, p, &src_base)) { bit_put(m.want, p, false); ++p; continue; }
            uint64_t q = p + 1;
            while (q < m.npages && q < chunk_end && copyable(m, q, now) && (q - p + 1) * kPageBytes <= budget) ++q;
            const bool armed = arm_timed(m.ww, p, q - p);
            for (uint64_t i = p; i < q; ++i) {
                bit_put(m.want, i, false);
                if (!armed) continue;
                m.copied_at[static_cast<size_t>(i)] = now ? now : 1;
                bit_put(m.used, i, false);
            }
            if (armed) {
                const uint64_t base = (p / chunk_pages) * kChunkBytes;
                g_copy_tmp.push_back(VkBufferCopy{p * kPageBytes, p * kPageBytes - base, (q - p) * kPageBytes});
                budget -= (q - p) * kPageBytes;
                m.st_copied += static_cast<int64_t>(q - p);
                g_st_copied.fetch_add(static_cast<int64_t>((q - p) * kPageBytes), std::memory_order_relaxed);
            }
            p = q;
        }
        if (g_copy_tmp.empty()) continue;
        if (!barrier_done) {
            memory_barrier(f.cb);      // earlier GPU writes into the pools, and draws still reading the mirrors, come first
            barrier_done = true;
        }
        // the source is the pool's own buffer: the copy reads what the CPU and earlier GPU work left in the pages. The runs are in pool
        // order and never span two chunks: one copy command per chunk.
        for (size_t i = 0; i < g_copy_tmp.size();) {
            const size_t ci = static_cast<size_t>(g_copy_tmp[i].srcOffset / kChunkBytes);
            size_t j = i + 1;
            while (j < g_copy_tmp.size() && g_copy_tmp[j].srcOffset / kChunkBytes == ci) ++j;
            uint64_t src_base = 0;
            const VkBuffer src = mirror_src(m, g_copy_tmp[i].srcOffset / kPageBytes, &src_base);   // imported when the run was made
            for (size_t k = i; k < j; ++k) g_copy_tmp[k].srcOffset -= src_base;
            if (src) vkCmdCopyBuffer(f.cb, src, m.chunks[ci].buffer, static_cast<uint32_t>(j - i), g_copy_tmp.data() + i);
            i = j;
        }
    }
    if (barrier_done) {
        VkMemoryBarrier2 b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &b;
        vkCmdPipelineBarrier2(f.cb, &di);
        if (gpass_on()) gpass_stamp(f, "mirror", 0);      // GPU pass profiler: the copies of this submission
    }
}

void mirror_destroy(PoolGpu* p) {
    Mirror* m = p->mirror;
    if (!m) return;
    p->mirror = nullptr;
    out_unregister(m);
    g_mirrors.erase(std::remove(g_mirrors.begin(), g_mirrors.end(), m), g_mirrors.end());
    if (m->ww) mem::ww::region_destroy(m->ww);
    for (gfx::Buffer& c : m->chunks)
        if (c.buffer) {
            forget_buffer_views(c.buffer);     // compute texel buffers bound to the copy
            g_mirror_bytes -= std::min<uint64_t>(g_mirror_bytes, c.size);
            device().destroy_buffer(c);
        }
    g_st_live_bytes.store(static_cast<int64_t>(g_mirror_bytes), std::memory_order_relaxed);
    delete m;
}


// ---- compute out: results of direct compute bindings held in the copy (nvn_gpu_internal.h) ---------------------------------------
// The copy-back of compute results to the pool streamed ~0.8 GB/s over PCIe, while most results are only read by later GPU work. With
// [nvn] compute_out_lazy (default on) a result stays in the copy after its submission: its pages are marked newer-in-the-copy (gpu_new)
// and guarded (PAGE_NOACCESS, mem::ww::guard). Draws bind the copy for them (resolve_read); a GPU reader of the pool gets them copied back
// first, in queue order (out_flush, called by the replay before such a command); the CPU's first touch of a guarded page faults, and the
// handler waits for the submission that wrote the result and copies it back on its own small submission (out_fetch, also called for
// nvnMemoryPool/BufferInvalidateMappedRange). A page the CPU read once (cpu_read) has its later results copied back at the end of their
// submission, as before, so a result the game reads every frame costs one wait, not one per frame.
namespace {

struct OutEntry {
    uint64_t addr = 0;
    VkDeviceSize size = 0;
    Mirror* m = nullptr;
    uint64_t first = 0, last = 0;       // pool pages it touches
    VkBuffer mbuf = VK_NULL_HANDLE;     // its place in the copy
    VkDeviceSize moff = 0;
    VkBuffer hbuf = VK_NULL_HANDLE;     // its place in the pool's own buffer
    VkDeviceSize hoff = 0;
    bool pending = false;               // written by the submission being recorded
    bool guarded = false;               // its pages are guarded
    uint64_t serial = 0;                // the submission whose result the copy holds (0: only the one being recorded)
    uint64_t flushed = 0;               // a copy-back is recorded in this submission: the guard goes when it completes
};

std::mutex g_out_mu;                    // g_out, g_out_mirrors; never held while touching guest memory
std::vector<OutEntry> g_out;
std::vector<Mirror*> g_out_mirrors;
std::atomic<size_t> g_out_n{0};
std::atomic<uint64_t> g_out_lo{UINT64_MAX}, g_out_hi{0};   // address bounds of the entries (render thread; a quick filter)

// An entry copies into b (a pool import, or a copy buffer standing in for one: retry_imports keeps that one while this holds).
bool out_uses_buffer(VkBuffer b) {
    if (!b || !g_out_n.load(std::memory_order_acquire)) return false;
    std::lock_guard<std::mutex> lk(g_out_mu);
    for (const OutEntry& e : g_out)
        if (e.hbuf == b) return true;
    return false;
}
// Bumped (under g_out_mu, before it is released) by every function that adds, removes or re-flags entries: out_overlaps rebuilds its
// index when it moved (read without the lock).
std::atomic<uint64_t> g_out_gen{0};
struct OutChanged { ~OutChanged() { ++g_out_gen; } };
// out_add's growth of the live entries, by the generation it made (out_clear_since). Every other change of g_out keeps or drops live
// entries (copy-backs, retirement, fetches, the end of a submission), so a range clear at some generation stays clear until a range
// logged here after it overlaps it. Render thread (out_add's callers), under g_out_mu.
struct OutGrow { uint64_t gen, lo, hi; };
constexpr size_t kOutGrowLog = 256;
OutGrow g_out_grow[kOutGrowLog];
uint64_t g_out_grow_n = 0;
void out_grew(uint64_t addr, uint64_t size) {
    g_out_grow[g_out_grow_n++ % kOutGrowLog] = {g_out_gen.load(std::memory_order_relaxed), addr & ~(kPageBytes - 1),
                                                (addr + size + kPageBytes - 1) & ~(kPageBytes - 1)};
}

// statistics (bytes and counts; the 10 s line)
std::atomic<uint64_t> g_ob_end{0}, g_ob_gpu{0}, g_ob_fault{0}, g_ob_inval{0};
std::atomic<uint64_t> g_why_bytes[kOutWhyCount], g_why_n[kOutWhyCount];   // GPU-reader copy-backs by reason
std::atomic<uint64_t> g_on_fault{0}, g_on_inval{0}, g_on_deferred{0}, g_on_eager_read{0}, g_on_eager_noguard{0}, g_on_gpu_flush{0};

// A word (64 pages) at a time: page by page, one locked RMW each, it was ~6 % of the render thread in Dagsion crowds (out_add on every
// draw binding a storage buffer of hundreds of pages). A word already in the wanted state is left alone (no RMW, no cache-line write).
void bits_set(std::atomic<uint64_t>* bm, uint64_t first, uint64_t last, bool on) {
    if (first > last) return;
    for (uint64_t w = first >> 6, wl = last >> 6; w <= wl; ++w) {
        const uint64_t lo = w == (first >> 6) ? (first & 63) : 0, hi = w == wl ? (last & 63) : 63;
        const uint64_t mask = (hi == 63 ? ~uint64_t{0} : ((uint64_t{1} << (hi + 1)) - 1)) & ~((uint64_t{1} << lo) - 1);
        const uint64_t cur = bm[w].load(std::memory_order_acquire);
        if (on) {
            if ((cur & mask) != mask) bm[w].fetch_or(mask, std::memory_order_acq_rel);
        } else {
            if (cur & mask) bm[w].fetch_and(~mask, std::memory_order_acq_rel);
        }
    }
}
bool bits_any(const std::atomic<uint64_t>* bm, uint64_t first, uint64_t last) {
    if (first > last) return false;
    for (uint64_t w = first >> 6, wl = last >> 6; w <= wl; ++w) {
        const uint64_t lo = w == (first >> 6) ? (first & 63) : 0, hi = w == wl ? (last & 63) : 63;
        const uint64_t mask = (hi == 63 ? ~uint64_t{0} : ((uint64_t{1} << (hi + 1)) - 1)) & ~((uint64_t{1} << lo) - 1);
        if (bm[w].load(std::memory_order_acquire) & mask) return true;
    }
    return false;
}

void out_register(Mirror* m) {
    std::lock_guard<std::mutex> lk(g_out_mu);
    g_out_mirrors.push_back(m);
    static std::once_flag once;
    std::call_once(once, [] { mem::ww::set_guard_handler([](uint64_t addr) { out_fetch(addr, kPageBytes, true); }); });
}

// every page current in the copy, or newer there (a GPU result) and armed: the CPU has not written it since. 64 pages at a time: page by page
// it was a third of the render thread in Dagsion (notes/pacing.md)
bool new_servable(const Mirror& m, uint64_t first, uint64_t last) {
    return mem::ww::all_valid_or_armed(m.ww, first, last - first + 1, m.gpu_new.get());
}

bool overlaps(const OutEntry& e, uint64_t lo, uint64_t hi) {      // page granular: [lo, hi) of addresses
    const uint64_t a0 = e.addr & ~(kPageBytes - 1), a1 = (e.addr + e.size + kPageBytes - 1) & ~(kPageBytes - 1);
    return a0 < hi && lo < a1;
}

// After entries of m covering [first, last] were removed or changed (lock held): pages no entry covers any more are not newer in the
// copy; pages no guarded entry covers lose their guard.
void settle(Mirror* m, uint64_t first, uint64_t last) {
    const uint64_t n = last - first + 1;
    std::vector<uint8_t> cov(static_cast<size_t>(n), 0);     // 1: some entry, 2: some guarded entry
    for (const OutEntry& e : g_out) {
        if (e.m != m || e.last < first || e.first > last) continue;
        const uint64_t a = std::max(e.first, first), b = std::min(e.last, last);
        for (uint64_t p = a; p <= b; ++p) cov[static_cast<size_t>(p - first)] |= static_cast<uint8_t>(1 | (e.guarded ? 2 : 0));
    }
    for (uint64_t p = first; p <= last;) {
        const uint8_t c = cov[static_cast<size_t>(p - first)];
        if (c & 2) { ++p; continue; }
        if (!(c & 1)) bits_set(m->gpu_new.get(), p, p, false);
        uint64_t q = p + 1;
        while (q <= last && !(cov[static_cast<size_t>(q - first)] & 2)) {
            if (!(cov[static_cast<size_t>(q - first)] & 1)) bits_set(m->gpu_new.get(), q, q, false);
            ++q;
        }
        mem::ww::unguard(m->ww, p, q - p);
        p = q;
    }
}

void recompute_bounds() {
    uint64_t lo = UINT64_MAX, hi = 0;
    for (const OutEntry& e : g_out) { lo = std::min(lo, e.addr); hi = std::max(hi, e.addr + e.size); }
    g_out_lo.store(lo, std::memory_order_relaxed);
    g_out_hi.store(hi, std::memory_order_relaxed);
    g_out_n.store(g_out.size(), std::memory_order_release);
}

void copy_barrier_in(VkCommandBuffer cb) { memory_barrier(cb); }   // the writes of the results (and anything earlier) before the copies

void copy_barrier_host(VkCommandBuffer cb) {                         // the copies before everything after, the host's reads included
    VkMemoryBarrier2 b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}

uint64_t record_copies(VkCommandBuffer cb, const std::vector<size_t>& idx) {
    if (idx.empty()) return 0;
    copy_barrier_in(cb);
    uint64_t bytes = 0;
    for (size_t i : idx) {
        const OutEntry& e = g_out[i];
        VkBufferCopy c = {e.moff, e.hoff, e.size};
        vkCmdCopyBuffer(cb, e.mbuf, e.hbuf, 1, &c);
        bytes += e.size;
    }
    copy_barrier_host(cb);
    return bytes;
}

// Removes the entries at idx (ascending or not) and settles their pages. Lock held.
void remove_entries(std::vector<size_t> idx) {
    if (idx.empty()) return;
    std::sort(idx.begin(), idx.end());
    struct Span { Mirror* m; uint64_t first, last; };
    std::vector<Span> spans;
    for (size_t k = idx.size(); k-- > 0;) {
        const OutEntry& e = g_out[idx[k]];
        spans.push_back({e.m, e.first, e.last});
        g_out.erase(g_out.begin() + static_cast<std::ptrdiff_t>(idx[k]));
    }
    for (const Span& s : spans) settle(s.m, s.first, s.last);
    recompute_bounds();
}

uint64_t page_size_kb(uint64_t bytes) { return bytes >> 10; }

}  // namespace

bool out_lazy() {
    static const bool on = config_get_bool("nvn", "compute_out_lazy", true);
    return on;
}

bool out_any() { return g_out_n.load(std::memory_order_acquire) != 0; }

uint64_t out_bytes_total() {
    return g_ob_end.load(std::memory_order_relaxed) + g_ob_gpu.load(std::memory_order_relaxed) + g_ob_fault.load(std::memory_order_relaxed) +
           g_ob_inval.load(std::memory_order_relaxed);
}

bool out_add(uint64_t addr, VkDeviceSize size, VkBuffer mbuf, VkDeviceSize moff, const BufRef& host) {
    if (!size || !host.ok) return false;
    const PoolHit hit = pool_lookup(addr);
    Mirror* m = hit.pool && hit.pool->gpu ? hit.pool->gpu->mirror : nullptr;
    std::lock_guard<std::mutex> lk(g_out_mu);
    // Entries are unique by (addr, size, m) (only this function adds them), so a remembered index that still matches is the entry the walk
    // would find. Draws add the same ranges again and again; the walk over ~160 entries was ~3 % of the render thread (notes/rt.md).
    static size_t recent[64];                                            // direct mapped by the range
    static bool recent_init = false;
    if (!recent_init) { std::fill(std::begin(recent), std::end(recent), SIZE_MAX); recent_init = true; }
    size_t& memo = recent[((addr >> 8) ^ (size * 0x9e3779b97f4a7c15ull >> 40)) % 64];
    auto again = [&](OutEntry& e) {
        if (e.flushed && !e.pending) { ++g_out_gen; out_grew(e.addr, e.size); }   // live again (out_overlaps); a result written again changes nothing there
        e.pending = true;
        e.mbuf = mbuf; e.moff = moff; e.hbuf = host.buffer; e.hoff = host.offset;
        if (m) bits_set(m->gpu_new.get(), e.first, e.last, true);        // a copy-back recorded earlier in this submission cleared them
        return true;
    };
    if (memo < g_out.size() && g_out[memo].addr == addr && g_out[memo].size == size && g_out[memo].m == m) return again(g_out[memo]);
    for (size_t i = 0; i < g_out.size(); ++i) {
        OutEntry& e = g_out[i];
        if (e.addr == addr && e.size == size && e.m == m) {
            memo = i;
            return again(e);
        }
    }
    OutEntry e;
    e.addr = addr; e.size = size; e.m = m;
    e.mbuf = mbuf; e.moff = moff; e.hbuf = host.buffer; e.hoff = host.offset;
    e.pending = true;
    if (m) {
        const uint64_t base = reinterpret_cast<uint64_t>(m->pg->base);
        e.first = (addr - base) / kPageBytes;
        e.last = std::min<uint64_t>((addr + size - 1 - base) / kPageBytes, m->npages - 1);
        bits_set(m->gpu_new.get(), e.first, e.last, true);
    }
    g_out.push_back(e);
    memo = g_out.size() - 1;
    ++g_out_gen;
    out_grew(addr, size);
    g_out_lo.store(std::min(g_out_lo.load(std::memory_order_relaxed), addr), std::memory_order_relaxed);
    g_out_hi.store(std::max(g_out_hi.load(std::memory_order_relaxed), addr + size), std::memory_order_relaxed);
    g_out_n.store(g_out.size(), std::memory_order_release);
    return false;
}

bool out_exact(uint64_t addr, VkDeviceSize size, VkBuffer* buf, VkDeviceSize* off) {
    if (!out_any()) return false;
    std::lock_guard<std::mutex> lk(g_out_mu);
    for (const OutEntry& e : g_out)
        if (e.addr == addr && e.size == size && e.mbuf) {
            // a copy-back recorded in this submission (flushed) leaves the copy as current as the pool: still the right place
            *buf = e.mbuf;
            *off = e.moff;
            return true;
        }
    return false;
}

bool out_overlaps(uint64_t addr, uint64_t size) {
    if (!out_any() || !size) return false;
    const uint64_t lo = addr & ~(kPageBytes - 1), hi = size >= UINT64_MAX - addr ? UINT64_MAX : (addr + size + kPageBytes - 1) & ~(kPageBytes - 1);
    if (hi <= (g_out_lo.load(std::memory_order_relaxed) & ~(kPageBytes - 1)) || lo >= g_out_hi.load(std::memory_order_relaxed) + kPageBytes) return false;
    // The live entries (not copied back, or written again) as page ranges sorted by start, with the running maximum of their ends: a binary
    // search instead of a walk over every held result for every buffer a draw binds (the walk was 15 % of the render thread in Dagsion
    // with about 160 results held, notes/pacing.md). Rebuilt when g_out_gen moved. The index is per thread and searched without the lock
    // (the lock on every query was ~3 % of the render thread, notes/perf.md); a change right after the check is the same as one right after
    // the lock was released.
    thread_local uint64_t gen = ~0ull;
    thread_local std::vector<uint64_t> a0, a1max;
    thread_local std::vector<std::pair<uint64_t, uint64_t>> iv;
    // The live pages hashed into 2^15 bits: a query of a few pages whose bits are all clear holds no result (the common answer; rt agent,
    // notes/rt.md). Empty when the live entries span too many pages to say anything.
    constexpr size_t kBloomWords = 512;
    constexpr uint64_t kBloomMaxPages = 8192;
    thread_local std::vector<uint64_t> bloom;
    auto bloom_bit = [](uint64_t page_addr) { return static_cast<size_t>(((page_addr >> 12) * 0x9e3779b97f4a7c15ull) >> 49); };
    if (gen != g_out_gen.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(g_out_mu);
        iv.clear();
        for (const OutEntry& e : g_out)
            if (!e.flushed || e.pending) iv.emplace_back(e.addr & ~(kPageBytes - 1), (e.addr + e.size + kPageBytes - 1) & ~(kPageBytes - 1));
        std::sort(iv.begin(), iv.end());
        a0.resize(iv.size());
        a1max.resize(iv.size());
        uint64_t m = 0;
        for (size_t i = 0; i < iv.size(); ++i) {
            a0[i] = iv[i].first;
            m = std::max(m, iv[i].second);
            a1max[i] = m;
        }
        bloom.assign(kBloomWords, 0);
        uint64_t pages = 0;
        for (const auto& r : iv) {
            pages += (r.second - r.first) / kPageBytes;
            if (pages > kBloomMaxPages) { bloom.clear(); break; }
            for (uint64_t pa = r.first; pa < r.second; pa += kPageBytes) {
                const size_t h = bloom_bit(pa);
                bloom[h >> 6] |= uint64_t{1} << (h & 63);
            }
        }
        gen = g_out_gen.load(std::memory_order_relaxed);
    }
    if (!bloom.empty() && hi - lo <= 16 * kPageBytes) {
        bool any = false;
        for (uint64_t pa = lo; pa < hi && !any; pa += kPageBytes) {
            const size_t h = bloom_bit(pa);
            any = (bloom[h >> 6] >> (h & 63)) & 1;
        }
        if (!any) return false;
    }
    const size_t k = static_cast<size_t>(std::lower_bound(a0.begin(), a0.end(), hi) - a0.begin());   // the entries that start below hi
    return k > 0 && a1max[k - 1] > lo;
}

uint64_t out_gen() { return g_out_gen.load(std::memory_order_acquire); }

bool out_clear_since(uint64_t addr, uint64_t size, uint64_t gen) {
    const uint64_t lo = addr & ~(kPageBytes - 1), hi = (addr + size + kPageBytes - 1) & ~(kPageBytes - 1);
    std::lock_guard<std::mutex> lk(g_out_mu);
    const uint64_t n = g_out_grow_n;
    for (uint64_t k = 0; k < n; ++k) {
        if (k >= kOutGrowLog) return false;                  // the log does not reach back that far
        const OutGrow& g = g_out_grow[(n - 1 - k) % kOutGrowLog];
        if (g.gen <= gen) return true;                       // every growth after `gen` was looked at
        if (g.lo < hi && lo < g.hi) return false;
    }
    return true;
}

bool out_servable(uint64_t addr, uint64_t size) {
    if (!out_lazy() || !g_mirror_on || !size) return false;
    MemoryPool* mp = nullptr;
    PoolGpu* pg = nullptr;
    uint64_t poff = 0;
    const BufRef r = resolve_in_pool(addr, size, &mp, &pg, &poff);
    if (!r.ok || !pg || !pg->mirror) return false;
    const Mirror& m = *pg->mirror;
    const uint64_t first = poff / kPageBytes, last = (poff + size - 1) / kPageBytes;
    const uint64_t chunk_pages = kChunkBytes / kPageBytes;
    if (last >= m.npages || last - first + 1 > kMaxBindPages || first / chunk_pages != last / chunk_pages) return false;
    if (!m.chunks[static_cast<size_t>(first / chunk_pages)].buffer) return false;
    return range_valid(m, first, last - first + 1) || new_servable(m, first, last);
}

uint64_t out_flush(VkCommandBuffer cb, uint64_t addr, uint64_t size, int why) {
    if (!out_any()) return 0;
    const uint64_t lo = size == UINT64_MAX ? 0 : addr & ~(kPageBytes - 1);
    const uint64_t hi = size == UINT64_MAX || size >= UINT64_MAX - addr ? UINT64_MAX : (addr + size + kPageBytes - 1) & ~(kPageBytes - 1);
    const uint64_t cur = current_serial();
    std::vector<std::pair<uint64_t, uint64_t>> rearm;     // results copied back in queue order: tracked again (mirror_span_written)
    uint64_t bytes = 0;
    {
        std::lock_guard<std::mutex> lk(g_out_mu);
        const OutChanged changed;
        std::vector<size_t> idx, gone;
        for (size_t i = 0; i < g_out.size(); ++i) {
            const OutEntry& e = g_out[i];
            if (e.flushed && !e.pending) continue;          // copied back already in this submission, nothing newer since
            if (overlaps(e, lo, hi)) idx.push_back(i);
        }
        if (idx.empty()) return 0;
        bytes = record_copies(cb, idx);
        for (size_t i : idx) {
            OutEntry& e = g_out[i];
            if (e.guarded) {                                // the CPU may still read old bytes until the copy has run: guarded until then;
                e.pending = false;                          // the copy stays as current as the pool for it (gpu_new kept until retired)
                e.flushed = cur;
            } else {
                rearm.emplace_back(e.addr, e.size);
                gone.push_back(i);
            }
        }
        remove_entries(gone);
    }
    for (const auto& r : rearm) mirror_span_written(r.first, r.second);
    (why == kOutSubmitEnd ? g_ob_end : g_ob_gpu).fetch_add(bytes, std::memory_order_relaxed);
    if (why != kOutSubmitEnd) g_on_gpu_flush.fetch_add(1, std::memory_order_relaxed);
    const int w = why >= 0 && why < kOutWhyCount ? why : kOutGpuReader;
    g_why_bytes[w].fetch_add(bytes, std::memory_order_relaxed);
    g_why_n[w].fetch_add(1, std::memory_order_relaxed);
    return bytes;
}

void out_end_submit(VkCommandBuffer cb, uint64_t serial) {
    if (!out_any()) return;
    std::vector<std::pair<uint64_t, uint64_t>> rearm;
    {
        std::lock_guard<std::mutex> lk(g_out_mu);
        const OutChanged changed;
        std::vector<size_t> eager, gone;
        for (size_t i = 0; i < g_out.size(); ++i) {
            OutEntry& e = g_out[i];
            if (!e.pending) continue;
            bool hold = out_lazy() && e.m && e.m->ww;
            if (hold && bits_any(e.m->cpu_read.get(), e.first, e.last)) { hold = false; g_on_eager_read.fetch_add(1, std::memory_order_relaxed); }
            else if (hold && !e.guarded && !mem::ww::guard(e.m->ww, e.first, e.last - e.first + 1)) { hold = false; g_on_eager_noguard.fetch_add(1, std::memory_order_relaxed); }
            if (hold) {
                e.guarded = true;
                e.pending = false;
                e.serial = serial;
                e.flushed = 0;
                g_on_deferred.fetch_add(1, std::memory_order_relaxed);
            } else {
                eager.push_back(i);
            }
        }
        const uint64_t bytes = record_copies(cb, eager);
        g_ob_end.fetch_add(bytes, std::memory_order_relaxed);
        for (size_t i : eager) {
            OutEntry& e = g_out[i];
            e.pending = false;
            if (e.guarded) { e.flushed = serial; e.serial = serial; }   // guarded from before: the guard goes when this submission is done
            else { rearm.emplace_back(e.addr, e.size); gone.push_back(i); }
        }
        remove_entries(gone);
    }
    for (const auto& r : rearm) mirror_span_written(r.first, r.second);
}

void out_retire() {
    if (!out_any()) return;
    const uint64_t done = poll_completed();
    std::vector<std::pair<uint64_t, uint64_t>> rearm;
    {
        std::lock_guard<std::mutex> lk(g_out_mu);
        const OutChanged changed;
        std::vector<size_t> gone;
        for (size_t i = 0; i < g_out.size(); ++i) {
            OutEntry& e = g_out[i];
            if (!e.flushed || e.pending || e.flushed > done) continue;
            e.guarded = false;
            rearm.emplace_back(e.addr, e.size);
            gone.push_back(i);
        }
        remove_entries(gone);
    }
    for (const auto& r : rearm) mirror_span_written(r.first, r.second);
}

void out_fetch(uint64_t addr, uint64_t size, bool by_fault) {
    if (!size || (!by_fault && !out_lazy())) return;
    const uint64_t lo = addr & ~(kPageBytes - 1), hi = (addr + size + kPageBytes - 1) & ~(kPageBytes - 1);
    std::unique_lock<std::mutex> lk(g_out_mu);
    const OutChanged changed;
    // later results in these pages go back at the end of their submission
    for (Mirror* m : g_out_mirrors) {
        const uint64_t base = reinterpret_cast<uint64_t>(m->pg->base), end = base + m->npages * kPageBytes;
        if (hi <= base || lo >= end) continue;
        bits_set(m->cpu_read.get(), (std::max(lo, base) - base) / kPageBytes, (std::min(hi, end) - 1 - base) / kPageBytes, true);
    }
    for (int waits = 0;; ) {
        // rebuilt after every wait: entries may have changed or gone while the lock was free
        std::vector<size_t> idx, copy;
        uint64_t need = 0;
        const bool rt = on_render_thread();
        const uint64_t cur = rt ? current_serial() : 0;
        for (size_t i = 0; i < g_out.size(); ++i) {
            const OutEntry& e = g_out[i];
            if (!overlaps(e, lo, hi) || !(e.guarded || e.serial)) continue;   // only in the submission being recorded: it goes back at its end
            idx.push_back(i);
            // a copy-back is recorded already (flushed): once that submission is done the pool is current, and later pool writes there
            // (a clear, a copy, an update) must not be overwritten with the held result. Only when this thread is the one recording it
            // (the render thread) can it not wait: then the result is copied now, before those later writes run.
            if (e.flushed && !e.pending && !(rt && e.flushed >= cur)) need = std::max(need, e.flushed);
            else { copy.push_back(i); need = std::max(need, e.serial); }
        }
        if (idx.empty()) return;
        if (need && !serial_done(need) && !(rt && need >= cur) && waits < 8) {
            lk.unlock();
            if (!wait_serial(need, 5ull * 1000 * 1000 * 1000)) NXLOG_W(kTag, "compute out: waited 5 s for submission %llu to read its result", static_cast<unsigned long long>(need));
            lk.lock();
            ++waits;
            continue;
        }
        uint64_t bytes = 0;
        if (!copy.empty()) device().submit_once([&](VkCommandBuffer cb) { bytes = record_copies(cb, copy); });
        (by_fault ? g_ob_fault : g_ob_inval).fetch_add(bytes, std::memory_order_relaxed);
        (by_fault ? g_on_fault : g_on_inval).fetch_add(1, std::memory_order_relaxed);
        std::vector<size_t> gone;
        struct Span { Mirror* m; uint64_t first, last; };
        std::vector<Span> spans;
        for (size_t i : idx) {
            OutEntry& e = g_out[i];
            if (e.pending) {                               // written again by the submission being recorded: it stays out, unguarded
                e.guarded = false;
                e.flushed = 0;
                e.serial = 0;
                if (e.m) spans.push_back({e.m, e.first, e.last});
            } else {
                gone.push_back(i);
            }
        }
        for (const Span& s : spans) settle(s.m, s.first, s.last);
        remove_entries(gone);
        return;
    }
}

void out_unregister(Mirror* m) {
    std::lock_guard<std::mutex> lk(g_out_mu);
    const OutChanged changed;
    g_out_mirrors.erase(std::remove(g_out_mirrors.begin(), g_out_mirrors.end(), m), g_out_mirrors.end());
    // the pool is going away: its held results with it (the region's destruction makes every page writable again)
    g_out.erase(std::remove_if(g_out.begin(), g_out.end(), [m](const OutEntry& e) { return e.m == m; }), g_out.end());
    recompute_bounds();
}

void out_reset() {
    std::lock_guard<std::mutex> lk(g_out_mu);
    const OutChanged changed;
    std::vector<size_t> all(g_out.size());
    for (size_t i = 0; i < all.size(); ++i) { all[i] = i; g_out[i].guarded = false; }
    remove_entries(all);
}

// One line every 10 s: what went back to game memory and why, and what the copies hold.
void out_log() {
    static auto last = std::chrono::steady_clock::now();
    static uint64_t p_end = 0, p_gpu = 0, p_fault = 0, p_inval = 0, pn_fault = 0, pn_inval = 0, pn_def = 0, pn_read = 0, pn_ng = 0, pn_gf = 0, p_gfaults = 0;
    const auto now = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(now - last).count();
    if (secs < 10.0) return;
    last = now;
    const uint64_t end = g_ob_end.load(), gpu = g_ob_gpu.load(), fault = g_ob_fault.load(), inval = g_ob_inval.load();
    const uint64_t nf = g_on_fault.load(), ni = g_on_inval.load(), nd = g_on_deferred.load(), nr = g_on_eager_read.load(), ng = g_on_eager_noguard.load(),
                   ngf = g_on_gpu_flush.load(), gfaults = mem::ww::guard_faults();
    const uint64_t total = (end - p_end) + (gpu - p_gpu) + (fault - p_fault) + (inval - p_inval);
    size_t held = 0;
    uint64_t held_bytes = 0;
    {
        std::lock_guard<std::mutex> lk(g_out_mu);
        for (const OutEntry& e : g_out)
            if (e.guarded && !e.flushed) { ++held; held_bytes += e.size; }
    }
    NXLOG_I(kTag, "compute out (%s): copied back %.0f KB/s over %.1f s (end of submission %llu KiB, for GPU readers %llu KiB in %llu flushes, on demand: "
            "CPU touches %llu KiB in %llu copies, invalidates %llu KiB in %llu copies; guard faults %llu); held in the copy %zu results, %llu KiB "
            "(held %llu, sent back at once %llu: CPU-read pages, %llu: not guardable)",
            out_lazy() ? "lazy" : "always", total / 1000.0 / secs, secs, (unsigned long long)page_size_kb(end - p_end), (unsigned long long)page_size_kb(gpu - p_gpu),
            (unsigned long long)(ngf - pn_gf), (unsigned long long)page_size_kb(fault - p_fault), (unsigned long long)(nf - pn_fault),
            (unsigned long long)page_size_kb(inval - p_inval), (unsigned long long)(ni - pn_inval), (unsigned long long)(gfaults - p_gfaults), held,
            (unsigned long long)(held_bytes >> 10), (unsigned long long)(nd - pn_def), (unsigned long long)(nr - pn_read), (unsigned long long)(ng - pn_ng));
    static uint64_t pw_b[kOutWhyCount] = {}, pw_n[kOutWhyCount] = {};
    static const char* const kWhyName[kOutWhyCount] = {"reader", "submit end", "copies off", "dispatch", "draw vertex/index/uniform", "draw storage/xfb",
                                                       "indirect", "buffer write", "buffer read", "texture upload", "event", "other"};
    std::string why;
    for (int w = 0; w < kOutWhyCount; ++w) {
        const uint64_t b = g_why_bytes[w].load(), n = g_why_n[w].load();
        if (b != pw_b[w] || n != pw_n[w]) {
            char t[96];
            snprintf(t, sizeof t, "%s%s %llu KiB/%llu", why.empty() ? "" : ", ", kWhyName[w], (unsigned long long)((b - pw_b[w]) >> 10), (unsigned long long)(n - pw_n[w]));
            why += t;
        }
        pw_b[w] = b; pw_n[w] = n;
    }
    if (!why.empty()) NXLOG_I(kTag, "compute out: copy-backs by reason (KiB/flushes over %.1f s): %s", secs, why.c_str());
    p_end = end; p_gpu = gpu; p_fault = fault; p_inval = inval; pn_fault = nf; pn_inval = ni; pn_def = nd; pn_read = nr; pn_ng = ng; pn_gf = ngf; p_gfaults = gfaults;
}

}  // namespace gpu
}  // namespace nxrt::nvn::be
