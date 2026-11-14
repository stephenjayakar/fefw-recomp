// Deferred Vulkan command recording on a worker thread: [nvn] replay_threads (nvn_cmdq.h).
#include "nvn_cmdq.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <intrin.h>

#include <volk.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <thread>
#include <tuple>

#include "nxrt/config.h"
#include "nxrt/diag.h"
#include "nxrt/log.h"

#pragma comment(lib, "synchronization.lib")   // WaitOnAddress / WakeByAddressSingle

namespace nxrt::nvn::be::cmdq {
namespace {

constexpr const char* kTag = "nvn.cmdq";

// ---- the queue: one producer (the render thread inside a Scope), one consumer (the worker) ------------------------------------------
// Records are [Hdr][payload], 16-byte aligned, in a ring addressed by ever-growing byte counters. A record never wraps: a padding record
// (exec null) fills the end of the ring first. The payload holds the call's arguments, with every array it points to copied after it and
// the pointers in the copy aimed at those copies (the ring does not move), so a record runs in place.
using Exec = void (*)(const uint8_t* payload);
struct alignas(16) Hdr {
    Exec exec;
    uint32_t size;    // whole record, header included, a multiple of 16
    uint32_t pad;
};
static_assert(sizeof(Hdr) == 16, "record header");

constexpr size_t kRingBytes = size_t{32} << 20;

struct Queue {
    uint8_t* mem = nullptr;
    uint64_t head = 0;                        // producer: bytes written
    uint64_t rec_start = 0;                   // producer: where the record being written starts
    uint64_t tail_seen = 0;                   // producer: the consumer's progress as last read (re-read only when the ring looks full)
    uint32_t unpublished = 0;                 // producer: records written since the last publish
    alignas(64) std::atomic<uint64_t> pub{0}; // bytes the consumer may run
    alignas(64) std::atomic<uint64_t> tail{0};// bytes the consumer has run
    alignas(64) std::atomic<uint32_t> sleeping{0};
    std::atomic<uint32_t> active{0};          // a Scope is open: the worker spins between records instead of sleeping
    std::atomic<bool> stop{false};
    std::thread th;
    // stats (producer side; the worker's busy time is its own)
    uint64_t records = 0, bytes = 0, syncs = 0, sync_ns = 0, fallbacks = 0, full_waits = 0;
    std::atomic<uint64_t> worker_ns{0};
};
Queue* g_q = nullptr;
bool g_running = false;                      // the worker runs (install .. shutdown)
int g_threads = 1;

thread_local bool t_q = false;               // this thread's calls are queued (render thread inside a Scope, outside a Direct)

inline size_t align16(size_t n) { return (n + 15) & ~size_t{15}; }

void wake_worker() {
    if (g_q->sleeping.load(std::memory_order_seq_cst)) WakeByAddressSingle(&g_q->pub);
}

// Room for a record of `payload` bytes; returns where the payload goes. q_commit() publishes it.
uint8_t* q_begin(Exec exec, size_t payload) {
    Queue& q = *g_q;
    const size_t total = align16(sizeof(Hdr) + payload);
    uint64_t pos = q.head % kRingBytes;
    const size_t pad = pos + total > kRingBytes ? kRingBytes - static_cast<size_t>(pos) : 0;
    // the consumer's tail is read only when the space it was last known to leave runs out (it is written by the worker: reading it for
    // every record moved its cache line between the two cores every time)
    if (q.head + pad + total - q.tail_seen > kRingBytes) {
        q.tail_seen = q.tail.load(std::memory_order_acquire);
        if (q.head + pad + total - q.tail_seen > kRingBytes) {
            ++q.full_waits;
            q.unpublished = 0;
            q.pub.store(q.head, std::memory_order_seq_cst);
            wake_worker();
            while (q.head + pad + total - (q.tail_seen = q.tail.load(std::memory_order_acquire)) > kRingBytes) { _mm_pause(); SwitchToThread(); }
        }
    }
    if (pad) {
        Hdr* h = reinterpret_cast<Hdr*>(q.mem + pos);
        h->exec = nullptr;
        h->size = static_cast<uint32_t>(pad);
        q.head += pad;
        pos = 0;
    }
    Hdr* h = reinterpret_cast<Hdr*>(q.mem + pos);
    h->exec = exec;
    h->size = static_cast<uint32_t>(total);
    q.rec_start = q.head;
    q.head += total;
    ++q.records;
    q.bytes += total;
    return reinterpret_cast<uint8_t*>(h + 1);
}

// Records are published in batches: each publish is a store the spinning worker's core must fetch (a cache line moving between cores per
// record cost about as much as the call it carried). [nvn] replay_batch (default 16): records per publish; every sync publishes the rest.
uint32_t g_batch = 16;
void q_publish() {
    g_q->unpublished = 0;
    g_q->pub.store(g_q->head, std::memory_order_seq_cst);
    wake_worker();
}
inline void q_commit() {
    if (++g_q->unpublished >= g_batch) q_publish();
}

void worker_main() {
    log_set_thread_name("nvn-rec");
    diag_register_thread("nvn-rec", nullptr);
    {   // the same priority as the render thread ([sched] render_priority): the render thread waits for it at the end of each submission
        std::string v = config_get_str("sched", "render_priority", "normal");
        for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        int p = THREAD_PRIORITY_NORMAL;
        if (v == "above" || v == "above_normal") p = THREAD_PRIORITY_ABOVE_NORMAL;
        else if (v == "highest") p = THREAD_PRIORITY_HIGHEST;
        else if (v == "time_critical") p = THREAD_PRIORITY_TIME_CRITICAL;
        if (p != THREAD_PRIORITY_NORMAL) SetThreadPriority(GetCurrentThread(), p);
    }
    Queue& q = *g_q;
    uint64_t t = 0;
    for (;;) {
        uint64_t p = q.pub.load(std::memory_order_acquire);
        if (p == t) {
            if (q.stop.load()) break;
            // idle: spin a little (inside a Scope the next record is usually a microsecond away), then sleep until a record or a Scope comes
            const uint64_t t0 = __rdtsc();
            const uint64_t spin = q.active.load(std::memory_order_relaxed) ? 400000ull : 40000ull;   // ~100 us / ~10 us at ~4 GHz
            while ((p = q.pub.load(std::memory_order_acquire)) == t && __rdtsc() - t0 < spin) _mm_pause();
            if (p == t) {
                q.sleeping.store(1, std::memory_order_seq_cst);
                if (q.pub.load(std::memory_order_seq_cst) == t && !q.stop.load()) WaitOnAddress(&q.pub, &t, sizeof t, 5);
                q.sleeping.store(0, std::memory_order_relaxed);
                continue;
            }
        }
        const auto w0 = std::chrono::steady_clock::now();
        while (t < p) {
            const Hdr* h = reinterpret_cast<const Hdr*>(q.mem + t % kRingBytes);
            if (h->exec) h->exec(reinterpret_cast<const uint8_t*>(h + 1));
            t += h->size;
        }
        q.tail.store(t, std::memory_order_release);   // once per batch: the producer reads it only at a sync or when the ring looks full
        q.worker_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w0).count()),
                              std::memory_order_relaxed);
    }
    diag_unregister_thread();
}

void do_sync() {
    Queue& q = *g_q;
    const uint64_t want = q.head;
    if (q.unpublished) q_publish();
    if (q.tail.load(std::memory_order_acquire) >= want) return;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; q.tail.load(std::memory_order_acquire) < want; ++i) {
        if (i < 4000) _mm_pause();
        else {
            wake_worker();
            SwitchToThread();
        }
    }
    ++q.syncs;
    q.sync_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}

// ---- the real entry points ------------------------------------------------------------------------------------------------------
// Plain calls: every argument is a value or a handle (copied as a tuple).
#define CMDQ_PLAIN_LIST(X)                                                                                                               \
    X(vkCmdBindPipeline) X(vkCmdDraw) X(vkCmdDrawIndexed) X(vkCmdDispatch) X(vkCmdDispatchIndirect) X(vkCmdDrawIndirect)                  \
    X(vkCmdDrawIndexedIndirect) X(vkCmdDrawIndirectCount) X(vkCmdDrawIndexedIndirectCount) X(vkCmdDrawIndirectByteCountEXT)              \
    X(vkCmdEndRendering) X(vkCmdResetQueryPool) X(vkCmdEndQuery) X(vkCmdBeginQuery) X(vkCmdEndQueryIndexedEXT) X(vkCmdBeginQueryIndexedEXT) \
    X(vkCmdWriteTimestamp) X(vkCmdWriteTimestamp2) X(vkCmdBindIndexBuffer) X(vkCmdSetStencilWriteMask) X(vkCmdSetStencilReference)        \
    X(vkCmdSetStencilCompareMask) X(vkCmdSetStencilTestEnable) X(vkCmdSetStencilOp) X(vkCmdSetRasterizerDiscardEnable)                    \
    X(vkCmdSetPrimitiveTopology) X(vkCmdSetPrimitiveRestartEnable) X(vkCmdSetLineWidth) X(vkCmdSetFrontFace) X(vkCmdSetDepthWriteEnable)  \
    X(vkCmdSetDepthTestEnable) X(vkCmdSetDepthCompareOp) X(vkCmdSetDepthBoundsTestEnable) X(vkCmdSetDepthBounds) X(vkCmdSetDepthBiasEnable) \
    X(vkCmdSetDepthBias) X(vkCmdSetCullMode) X(vkCmdCopyQueryPoolResults) X(vkCmdFillBuffer) X(vkCmdEndDebugUtilsLabelEXT)

// Calls with arrays or structures behind pointers (deep copies, below).
#define CMDQ_DEEP_LIST(X)                                                                                                                \
    X(vkCmdSetBlendConstants) X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdSetViewportWithCount) X(vkCmdSetScissorWithCount)            \
    X(vkCmdBindDescriptorSets) X(vkCmdPushDescriptorSetKHR) X(vkCmdPushConstants) X(vkCmdBindVertexBuffers2) X(vkCmdPipelineBarrier2)    \
    X(vkCmdPipelineBarrier) X(vkCmdBeginRendering) X(vkCmdCopyBuffer) X(vkCmdCopyImage) X(vkCmdBlitImage) X(vkCmdCopyBufferToImage)      \
    X(vkCmdCopyImageToBuffer) X(vkCmdCopyBufferToImage2) X(vkCmdCopyImageToBuffer2) X(vkCmdClearAttachments) X(vkCmdClearColorImage)     \
    X(vkCmdClearDepthStencilImage) X(vkCmdUpdateBuffer) X(vkCmdResolveImage) X(vkCmdBeginDebugUtilsLabelEXT)                             \
    X(vkCmdBindTransformFeedbackBuffersEXT) X(vkCmdBeginTransformFeedbackEXT) X(vkCmdEndTransformFeedbackEXT) X(vkUpdateDescriptorSets)

// Calls that need everything queued before them made first, then run on the calling thread.
#define CMDQ_SYNC_LIST(X)                                                                                                                \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer) X(vkResetCommandPool) X(vkAllocateCommandBuffers)              \
    X(vkFreeCommandBuffers) X(vkQueueSubmit) X(vkQueueSubmit2) X(vkQueueWaitIdle) X(vkDeviceWaitIdle) X(vkTrimCommandPool)              \
    X(vkDestroyCommandPool)

#define CMDQ_DECL(name) PFN_##name real_##name = nullptr;
CMDQ_PLAIN_LIST(CMDQ_DECL)
CMDQ_DEEP_LIST(CMDQ_DECL)
CMDQ_SYNC_LIST(CMDQ_DECL)
#undef CMDQ_DECL

// ---- plain wrappers ---------------------------------------------------------------------------------------------------------------
template <auto* Slot, class Sig> struct Plain;
template <auto* Slot, class R, class... A> struct Plain<Slot, R(VKAPI_PTR*)(A...)> {
    using Tup = std::tuple<A...>;
    static void exec(const uint8_t* p) { std::apply(*Slot, *reinterpret_cast<const Tup*>(p)); }
    static void VKAPI_PTR wrap(A... a) {
        if (!t_q) { (*Slot)(a...); return; }
        uint8_t* p = q_begin(&exec, sizeof(Tup));
        new (p) Tup(a...);
        q_commit();
    }
};

// ---- sync wrappers ----------------------------------------------------------------------------------------------------------------
template <auto* Slot, class Sig> struct Synced;
template <auto* Slot, class R, class... A> struct Synced<Slot, R(VKAPI_PTR*)(A...)> {
    static R VKAPI_PTR wrap(A... a) {
        if (t_q) do_sync();
        return (*Slot)(a...);
    }
};

// ---- deep copies ------------------------------------------------------------------------------------------------------------------
// A record is built in two passes: size() adds up what the copies need, then the writer fills the payload.
struct Writer {
    uint8_t* p;
    template <class T> T* one(const T& v) {
        T* d = reinterpret_cast<T*>(p);
        std::memcpy(d, &v, sizeof(T));
        p += align16(sizeof(T));
        return d;
    }
    template <class T> T* arr(const T* src, uint32_t n) {
        if (!src || !n) return nullptr;
        T* d = reinterpret_cast<T*>(p);
        std::memcpy(d, src, sizeof(T) * n);
        p += align16(sizeof(T) * n);
        return d;
    }
    void* bytes(const void* src, size_t n) {
        if (!src || !n) return nullptr;
        void* d = p;
        std::memcpy(d, src, n);
        p += align16(n);
        return d;
    }
};
template <class T> size_t sz(uint32_t n = 1) { return align16(sizeof(T) * n); }

// The fallback for anything the copies do not know (a pNext chain): run it here, after everything queued.
#define CMDQ_FALLBACK(name, ...)        \
    do {                                \
        do_sync();                      \
        ++g_q->fallbacks;               \
        real_##name(__VA_ARGS__);       \
        return;                         \
    } while (0)

// vkCmdSetBlendConstants
struct A_Blend { VkCommandBuffer cb; float c[4]; };
void x_SetBlendConstants(const uint8_t* p) { auto* a = reinterpret_cast<const A_Blend*>(p); real_vkCmdSetBlendConstants(a->cb, a->c); }
void VKAPI_PTR w_vkCmdSetBlendConstants(VkCommandBuffer cb, const float c[4]) {
    if (!t_q) return real_vkCmdSetBlendConstants(cb, c);
    Writer w{q_begin(&x_SetBlendConstants, sz<A_Blend>())};
    A_Blend a{cb, {c[0], c[1], c[2], c[3]}};
    w.one(a);
    q_commit();
}

// vkCmdSetViewport / vkCmdSetScissor (first, count, array)
struct A_Rects { VkCommandBuffer cb; uint32_t first, count; const void* data; };
void x_SetViewport(const uint8_t* p) { auto* a = reinterpret_cast<const A_Rects*>(p); real_vkCmdSetViewport(a->cb, a->first, a->count, static_cast<const VkViewport*>(a->data)); }
void x_SetScissor(const uint8_t* p) { auto* a = reinterpret_cast<const A_Rects*>(p); real_vkCmdSetScissor(a->cb, a->first, a->count, static_cast<const VkRect2D*>(a->data)); }
void x_SetViewportWithCount(const uint8_t* p) { auto* a = reinterpret_cast<const A_Rects*>(p); real_vkCmdSetViewportWithCount(a->cb, a->count, static_cast<const VkViewport*>(a->data)); }
void x_SetScissorWithCount(const uint8_t* p) { auto* a = reinterpret_cast<const A_Rects*>(p); real_vkCmdSetScissorWithCount(a->cb, a->count, static_cast<const VkRect2D*>(a->data)); }
template <class T> void put_rects(Exec x, VkCommandBuffer cb, uint32_t first, uint32_t count, const T* data) {
    Writer w{q_begin(x, sz<A_Rects>() + sz<T>(count))};
    A_Rects* a = w.one(A_Rects{cb, first, count, nullptr});
    a->data = w.arr(data, count);
    q_commit();
}
void VKAPI_PTR w_vkCmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t count, const VkViewport* v) {
    if (!t_q) return real_vkCmdSetViewport(cb, first, count, v);
    put_rects(&x_SetViewport, cb, first, count, v);
}
void VKAPI_PTR w_vkCmdSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t count, const VkRect2D* r) {
    if (!t_q) return real_vkCmdSetScissor(cb, first, count, r);
    put_rects(&x_SetScissor, cb, first, count, r);
}
void VKAPI_PTR w_vkCmdSetViewportWithCount(VkCommandBuffer cb, uint32_t count, const VkViewport* v) {
    if (!t_q) return real_vkCmdSetViewportWithCount(cb, count, v);
    put_rects(&x_SetViewportWithCount, cb, 0, count, v);
}
void VKAPI_PTR w_vkCmdSetScissorWithCount(VkCommandBuffer cb, uint32_t count, const VkRect2D* r) {
    if (!t_q) return real_vkCmdSetScissorWithCount(cb, count, r);
    put_rects(&x_SetScissorWithCount, cb, 0, count, r);
}

// vkCmdBindDescriptorSets
struct A_BindSets { VkCommandBuffer cb; VkPipelineBindPoint bp; VkPipelineLayout layout; uint32_t first, count, ndyn; const VkDescriptorSet* sets; const uint32_t* dyn; };
void x_BindSets(const uint8_t* p) {
    auto* a = reinterpret_cast<const A_BindSets*>(p);
    real_vkCmdBindDescriptorSets(a->cb, a->bp, a->layout, a->first, a->count, a->sets, a->ndyn, a->dyn);
}
void VKAPI_PTR w_vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t first, uint32_t count,
                                         const VkDescriptorSet* sets, uint32_t ndyn, const uint32_t* dyn) {
    if (!t_q) return real_vkCmdBindDescriptorSets(cb, bp, layout, first, count, sets, ndyn, dyn);
    Writer w{q_begin(&x_BindSets, sz<A_BindSets>() + sz<VkDescriptorSet>(count) + sz<uint32_t>(ndyn))};
    A_BindSets* a = w.one(A_BindSets{cb, bp, layout, first, count, ndyn, nullptr, nullptr});
    a->sets = w.arr(sets, count);
    a->dyn = w.arr(dyn, ndyn);
    q_commit();
}

// descriptor writes (push descriptors and vkUpdateDescriptorSets): the writes and the infos each one points at
// Which of a write's three pointers the descriptor type reads (the other two may hold anything; Vulkan ignores them).
enum class WKind { Image, Buffer, Texel, Other };
WKind write_kind(VkDescriptorType t) {
    switch (t) {
        case VK_DESCRIPTOR_TYPE_SAMPLER: case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            return WKind::Image;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            return WKind::Buffer;
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            return WKind::Texel;
        default:
            return WKind::Other;
    }
}
// Writes this module can copy: no pNext, a known descriptor type.
bool writes_plain(uint32_t n, const VkWriteDescriptorSet* ws) {
    for (uint32_t i = 0; i < n; ++i)
        if (ws[i].pNext || write_kind(ws[i].descriptorType) == WKind::Other) return false;
    return true;
}
size_t writes_size(uint32_t n, const VkWriteDescriptorSet* ws) {
    size_t s = sz<VkWriteDescriptorSet>(n);
    for (uint32_t i = 0; i < n; ++i) {
        const VkWriteDescriptorSet& x = ws[i];
        switch (write_kind(x.descriptorType)) {
            case WKind::Image: s += sz<VkDescriptorImageInfo>(x.descriptorCount); break;
            case WKind::Buffer: s += sz<VkDescriptorBufferInfo>(x.descriptorCount); break;
            case WKind::Texel: s += sz<VkBufferView>(x.descriptorCount); break;
            default: break;
        }
    }
    return s;
}
const VkWriteDescriptorSet* writes_copy(Writer& w, uint32_t n, const VkWriteDescriptorSet* ws) {
    VkWriteDescriptorSet* d = w.arr(ws, n);
    for (uint32_t i = 0; i < n; ++i) {
        const VkWriteDescriptorSet& x = ws[i];
        const WKind k = write_kind(x.descriptorType);
        d[i].pImageInfo = k == WKind::Image ? w.arr(x.pImageInfo, x.descriptorCount) : nullptr;
        d[i].pBufferInfo = k == WKind::Buffer ? w.arr(x.pBufferInfo, x.descriptorCount) : nullptr;
        d[i].pTexelBufferView = k == WKind::Texel ? w.arr(x.pTexelBufferView, x.descriptorCount) : nullptr;
    }
    return d;
}

struct A_Push { VkCommandBuffer cb; VkPipelineBindPoint bp; VkPipelineLayout layout; uint32_t set, n; const VkWriteDescriptorSet* ws; };
void x_Push(const uint8_t* p) { auto* a = reinterpret_cast<const A_Push*>(p); real_vkCmdPushDescriptorSetKHR(a->cb, a->bp, a->layout, a->set, a->n, a->ws); }
void VKAPI_PTR w_vkCmdPushDescriptorSetKHR(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set, uint32_t n,
                                           const VkWriteDescriptorSet* ws) {
    if (!t_q) return real_vkCmdPushDescriptorSetKHR(cb, bp, layout, set, n, ws);
    if (!writes_plain(n, ws)) CMDQ_FALLBACK(vkCmdPushDescriptorSetKHR, cb, bp, layout, set, n, ws);
    Writer w{q_begin(&x_Push, sz<A_Push>() + writes_size(n, ws))};
    A_Push* a = w.one(A_Push{cb, bp, layout, set, n, nullptr});
    a->ws = writes_copy(w, n, ws);
    q_commit();
}

struct A_Update { VkDevice dev; uint32_t n; const VkWriteDescriptorSet* ws; };
void x_Update(const uint8_t* p) { auto* a = reinterpret_cast<const A_Update*>(p); real_vkUpdateDescriptorSets(a->dev, a->n, a->ws, 0, nullptr); }
void VKAPI_PTR w_vkUpdateDescriptorSets(VkDevice dev, uint32_t n, const VkWriteDescriptorSet* ws, uint32_t nc, const VkCopyDescriptorSet* cs) {
    if (!t_q) return real_vkUpdateDescriptorSets(dev, n, ws, nc, cs);
    if (nc || !writes_plain(n, ws)) CMDQ_FALLBACK(vkUpdateDescriptorSets, dev, n, ws, nc, cs);
    if (!n) return;
    Writer w{q_begin(&x_Update, sz<A_Update>() + writes_size(n, ws))};
    A_Update* a = w.one(A_Update{dev, n, nullptr});
    a->ws = writes_copy(w, n, ws);
    q_commit();
}

// vkCmdPushConstants
struct A_PushC { VkCommandBuffer cb; VkPipelineLayout layout; VkShaderStageFlags stages; uint32_t off, size; const void* data; };
void x_PushC(const uint8_t* p) { auto* a = reinterpret_cast<const A_PushC*>(p); real_vkCmdPushConstants(a->cb, a->layout, a->stages, a->off, a->size, a->data); }
void VKAPI_PTR w_vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t off, uint32_t size, const void* data) {
    if (!t_q) return real_vkCmdPushConstants(cb, layout, stages, off, size, data);
    Writer w{q_begin(&x_PushC, sz<A_PushC>() + align16(size))};
    A_PushC* a = w.one(A_PushC{cb, layout, stages, off, size, nullptr});
    a->data = w.bytes(data, size);
    q_commit();
}

// vkCmdBindVertexBuffers2
struct A_Vb { VkCommandBuffer cb; uint32_t first, n; const VkBuffer* bufs; const VkDeviceSize *offs, *sizes, *strides; };
void x_Vb(const uint8_t* p) { auto* a = reinterpret_cast<const A_Vb*>(p); real_vkCmdBindVertexBuffers2(a->cb, a->first, a->n, a->bufs, a->offs, a->sizes, a->strides); }
void VKAPI_PTR w_vkCmdBindVertexBuffers2(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer* bufs, const VkDeviceSize* offs,
                                         const VkDeviceSize* sizes, const VkDeviceSize* strides) {
    if (!t_q) return real_vkCmdBindVertexBuffers2(cb, first, n, bufs, offs, sizes, strides);
    Writer w{q_begin(&x_Vb, sz<A_Vb>() + sz<VkBuffer>(n) + 3 * sz<VkDeviceSize>(n))};
    A_Vb* a = w.one(A_Vb{cb, first, n, nullptr, nullptr, nullptr, nullptr});
    a->bufs = w.arr(bufs, n);
    a->offs = w.arr(offs, n);
    a->sizes = w.arr(sizes, n);
    a->strides = w.arr(strides, n);
    q_commit();
}

// vkCmdPipelineBarrier2
struct A_Dep { VkCommandBuffer cb; VkDependencyInfo di; };
void x_Dep(const uint8_t* p) { auto* a = reinterpret_cast<const A_Dep*>(p); real_vkCmdPipelineBarrier2(a->cb, &a->di); }
void VKAPI_PTR w_vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo* di) {
    if (!t_q) return real_vkCmdPipelineBarrier2(cb, di);
    bool plain = !di->pNext;
    for (uint32_t i = 0; plain && i < di->memoryBarrierCount; ++i) plain = !di->pMemoryBarriers[i].pNext;
    for (uint32_t i = 0; plain && i < di->bufferMemoryBarrierCount; ++i) plain = !di->pBufferMemoryBarriers[i].pNext;
    for (uint32_t i = 0; plain && i < di->imageMemoryBarrierCount; ++i) plain = !di->pImageMemoryBarriers[i].pNext;
    if (!plain) CMDQ_FALLBACK(vkCmdPipelineBarrier2, cb, di);
    Writer w{q_begin(&x_Dep, sz<A_Dep>() + sz<VkMemoryBarrier2>(di->memoryBarrierCount) + sz<VkBufferMemoryBarrier2>(di->bufferMemoryBarrierCount) +
                                 sz<VkImageMemoryBarrier2>(di->imageMemoryBarrierCount))};
    A_Dep* a = w.one(A_Dep{cb, *di});
    a->di.pMemoryBarriers = w.arr(di->pMemoryBarriers, di->memoryBarrierCount);
    a->di.pBufferMemoryBarriers = w.arr(di->pBufferMemoryBarriers, di->bufferMemoryBarrierCount);
    a->di.pImageMemoryBarriers = w.arr(di->pImageMemoryBarriers, di->imageMemoryBarrierCount);
    q_commit();
}

// vkCmdPipelineBarrier (Vulkan 1.0 form)
struct A_Dep1 {
    VkCommandBuffer cb; VkPipelineStageFlags src, dst; VkDependencyFlags dep; uint32_t nm, nb, ni;
    const VkMemoryBarrier* m; const VkBufferMemoryBarrier* b; const VkImageMemoryBarrier* i;
};
void x_Dep1(const uint8_t* p) { auto* a = reinterpret_cast<const A_Dep1*>(p); real_vkCmdPipelineBarrier(a->cb, a->src, a->dst, a->dep, a->nm, a->m, a->nb, a->b, a->ni, a->i); }
void VKAPI_PTR w_vkCmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags dep, uint32_t nm,
                                      const VkMemoryBarrier* m, uint32_t nb, const VkBufferMemoryBarrier* b, uint32_t ni, const VkImageMemoryBarrier* im) {
    if (!t_q) return real_vkCmdPipelineBarrier(cb, src, dst, dep, nm, m, nb, b, ni, im);
    bool plain = true;
    for (uint32_t i = 0; plain && i < nm; ++i) plain = !m[i].pNext;
    for (uint32_t i = 0; plain && i < nb; ++i) plain = !b[i].pNext;
    for (uint32_t i = 0; plain && i < ni; ++i) plain = !im[i].pNext;
    if (!plain) CMDQ_FALLBACK(vkCmdPipelineBarrier, cb, src, dst, dep, nm, m, nb, b, ni, im);
    Writer w{q_begin(&x_Dep1, sz<A_Dep1>() + sz<VkMemoryBarrier>(nm) + sz<VkBufferMemoryBarrier>(nb) + sz<VkImageMemoryBarrier>(ni))};
    A_Dep1* a = w.one(A_Dep1{cb, src, dst, dep, nm, nb, ni, nullptr, nullptr, nullptr});
    a->m = w.arr(m, nm);
    a->b = w.arr(b, nb);
    a->i = w.arr(im, ni);
    q_commit();
}

// vkCmdBeginRendering
struct A_Rendering { VkCommandBuffer cb; VkRenderingInfo ri; };
void x_Rendering(const uint8_t* p) { auto* a = reinterpret_cast<const A_Rendering*>(p); real_vkCmdBeginRendering(a->cb, &a->ri); }
void VKAPI_PTR w_vkCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* ri) {
    if (!t_q) return real_vkCmdBeginRendering(cb, ri);
    bool plain = !ri->pNext && (!ri->pDepthAttachment || !ri->pDepthAttachment->pNext) && (!ri->pStencilAttachment || !ri->pStencilAttachment->pNext);
    for (uint32_t i = 0; plain && i < ri->colorAttachmentCount; ++i) plain = !ri->pColorAttachments[i].pNext;
    if (!plain) CMDQ_FALLBACK(vkCmdBeginRendering, cb, ri);
    Writer w{q_begin(&x_Rendering, sz<A_Rendering>() + sz<VkRenderingAttachmentInfo>(ri->colorAttachmentCount) + 2 * sz<VkRenderingAttachmentInfo>())};
    A_Rendering* a = w.one(A_Rendering{cb, *ri});
    a->ri.pColorAttachments = w.arr(ri->pColorAttachments, ri->colorAttachmentCount);
    a->ri.pDepthAttachment = ri->pDepthAttachment ? w.one(*ri->pDepthAttachment) : nullptr;
    a->ri.pStencilAttachment = ri->pStencilAttachment ? w.one(*ri->pStencilAttachment) : nullptr;
    q_commit();
}

// copies, blits, clears, resolves: a handful of handles and enums, then one or two arrays
struct A_CopyBuf { VkCommandBuffer cb; VkBuffer src, dst; uint32_t n; const VkBufferCopy* r; };
void x_CopyBuf(const uint8_t* p) { auto* a = reinterpret_cast<const A_CopyBuf*>(p); real_vkCmdCopyBuffer(a->cb, a->src, a->dst, a->n, a->r); }
void VKAPI_PTR w_vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer src, VkBuffer dst, uint32_t n, const VkBufferCopy* r) {
    if (!t_q) return real_vkCmdCopyBuffer(cb, src, dst, n, r);
    Writer w{q_begin(&x_CopyBuf, sz<A_CopyBuf>() + sz<VkBufferCopy>(n))};
    A_CopyBuf* a = w.one(A_CopyBuf{cb, src, dst, n, nullptr});
    a->r = w.arr(r, n);
    q_commit();
}

struct A_CopyImg { VkCommandBuffer cb; VkImage src; VkImageLayout sl; VkImage dst; VkImageLayout dl; uint32_t n; const void* r; VkFilter filter; };
void x_CopyImg(const uint8_t* p) { auto* a = reinterpret_cast<const A_CopyImg*>(p); real_vkCmdCopyImage(a->cb, a->src, a->sl, a->dst, a->dl, a->n, static_cast<const VkImageCopy*>(a->r)); }
void x_Blit(const uint8_t* p) { auto* a = reinterpret_cast<const A_CopyImg*>(p); real_vkCmdBlitImage(a->cb, a->src, a->sl, a->dst, a->dl, a->n, static_cast<const VkImageBlit*>(a->r), a->filter); }
void x_Resolve(const uint8_t* p) { auto* a = reinterpret_cast<const A_CopyImg*>(p); real_vkCmdResolveImage(a->cb, a->src, a->sl, a->dst, a->dl, a->n, static_cast<const VkImageResolve*>(a->r)); }
template <class T> void put_img2img(Exec x, VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst, VkImageLayout dl, uint32_t n, const T* r, VkFilter f) {
    Writer w{q_begin(x, sz<A_CopyImg>() + sz<T>(n))};
    A_CopyImg* a = w.one(A_CopyImg{cb, src, sl, dst, dl, n, nullptr, f});
    a->r = w.arr(r, n);
    q_commit();
}
void VKAPI_PTR w_vkCmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst, VkImageLayout dl, uint32_t n, const VkImageCopy* r) {
    if (!t_q) return real_vkCmdCopyImage(cb, src, sl, dst, dl, n, r);
    put_img2img(&x_CopyImg, cb, src, sl, dst, dl, n, r, VK_FILTER_NEAREST);
}
void VKAPI_PTR w_vkCmdBlitImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst, VkImageLayout dl, uint32_t n, const VkImageBlit* r, VkFilter f) {
    if (!t_q) return real_vkCmdBlitImage(cb, src, sl, dst, dl, n, r, f);
    put_img2img(&x_Blit, cb, src, sl, dst, dl, n, r, f);
}
void VKAPI_PTR w_vkCmdResolveImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst, VkImageLayout dl, uint32_t n, const VkImageResolve* r) {
    if (!t_q) return real_vkCmdResolveImage(cb, src, sl, dst, dl, n, r);
    put_img2img(&x_Resolve, cb, src, sl, dst, dl, n, r, VK_FILTER_NEAREST);
}

struct A_BufImg { VkCommandBuffer cb; VkBuffer buf; VkImage img; VkImageLayout layout; uint32_t n; const VkBufferImageCopy* r; };
void x_BufToImg(const uint8_t* p) { auto* a = reinterpret_cast<const A_BufImg*>(p); real_vkCmdCopyBufferToImage(a->cb, a->buf, a->img, a->layout, a->n, a->r); }
void x_ImgToBuf(const uint8_t* p) { auto* a = reinterpret_cast<const A_BufImg*>(p); real_vkCmdCopyImageToBuffer(a->cb, a->img, a->layout, a->buf, a->n, a->r); }
void VKAPI_PTR w_vkCmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer buf, VkImage img, VkImageLayout layout, uint32_t n, const VkBufferImageCopy* r) {
    if (!t_q) return real_vkCmdCopyBufferToImage(cb, buf, img, layout, n, r);
    Writer w{q_begin(&x_BufToImg, sz<A_BufImg>() + sz<VkBufferImageCopy>(n))};
    A_BufImg* a = w.one(A_BufImg{cb, buf, img, layout, n, nullptr});
    a->r = w.arr(r, n);
    q_commit();
}
void VKAPI_PTR w_vkCmdCopyImageToBuffer(VkCommandBuffer cb, VkImage img, VkImageLayout layout, VkBuffer buf, uint32_t n, const VkBufferImageCopy* r) {
    if (!t_q) return real_vkCmdCopyImageToBuffer(cb, img, layout, buf, n, r);
    Writer w{q_begin(&x_ImgToBuf, sz<A_BufImg>() + sz<VkBufferImageCopy>(n))};
    A_BufImg* a = w.one(A_BufImg{cb, buf, img, layout, n, nullptr});
    a->r = w.arr(r, n);
    q_commit();
}

struct A_BufImg2 { VkCommandBuffer cb; VkCopyBufferToImageInfo2 i; };
struct A_ImgBuf2 { VkCommandBuffer cb; VkCopyImageToBufferInfo2 i; };
void x_BufToImg2(const uint8_t* p) { auto* a = reinterpret_cast<const A_BufImg2*>(p); real_vkCmdCopyBufferToImage2(a->cb, &a->i); }
void x_ImgToBuf2(const uint8_t* p) { auto* a = reinterpret_cast<const A_ImgBuf2*>(p); real_vkCmdCopyImageToBuffer2(a->cb, &a->i); }
void VKAPI_PTR w_vkCmdCopyBufferToImage2(VkCommandBuffer cb, const VkCopyBufferToImageInfo2* i) {
    if (!t_q) return real_vkCmdCopyBufferToImage2(cb, i);
    bool plain = !i->pNext;
    for (uint32_t k = 0; plain && k < i->regionCount; ++k) plain = !i->pRegions[k].pNext;
    if (!plain) CMDQ_FALLBACK(vkCmdCopyBufferToImage2, cb, i);
    Writer w{q_begin(&x_BufToImg2, sz<A_BufImg2>() + sz<VkBufferImageCopy2>(i->regionCount))};
    A_BufImg2* a = w.one(A_BufImg2{cb, *i});
    a->i.pRegions = w.arr(i->pRegions, i->regionCount);
    q_commit();
}
void VKAPI_PTR w_vkCmdCopyImageToBuffer2(VkCommandBuffer cb, const VkCopyImageToBufferInfo2* i) {
    if (!t_q) return real_vkCmdCopyImageToBuffer2(cb, i);
    bool plain = !i->pNext;
    for (uint32_t k = 0; plain && k < i->regionCount; ++k) plain = !i->pRegions[k].pNext;
    if (!plain) CMDQ_FALLBACK(vkCmdCopyImageToBuffer2, cb, i);
    Writer w{q_begin(&x_ImgToBuf2, sz<A_ImgBuf2>() + sz<VkBufferImageCopy2>(i->regionCount))};
    A_ImgBuf2* a = w.one(A_ImgBuf2{cb, *i});
    a->i.pRegions = w.arr(i->pRegions, i->regionCount);
    q_commit();
}

struct A_ClearAtt { VkCommandBuffer cb; uint32_t na, nr; const VkClearAttachment* a; const VkClearRect* r; };
void x_ClearAtt(const uint8_t* p) { auto* a = reinterpret_cast<const A_ClearAtt*>(p); real_vkCmdClearAttachments(a->cb, a->na, a->a, a->nr, a->r); }
void VKAPI_PTR w_vkCmdClearAttachments(VkCommandBuffer cb, uint32_t na, const VkClearAttachment* at, uint32_t nr, const VkClearRect* r) {
    if (!t_q) return real_vkCmdClearAttachments(cb, na, at, nr, r);
    Writer w{q_begin(&x_ClearAtt, sz<A_ClearAtt>() + sz<VkClearAttachment>(na) + sz<VkClearRect>(nr))};
    A_ClearAtt* a = w.one(A_ClearAtt{cb, na, nr, nullptr, nullptr});
    a->a = w.arr(at, na);
    a->r = w.arr(r, nr);
    q_commit();
}

struct A_ClearImg { VkCommandBuffer cb; VkImage img; VkImageLayout layout; VkClearValue v; uint32_t n; const VkImageSubresourceRange* r; };
void x_ClearColor(const uint8_t* p) { auto* a = reinterpret_cast<const A_ClearImg*>(p); real_vkCmdClearColorImage(a->cb, a->img, a->layout, &a->v.color, a->n, a->r); }
void x_ClearDs(const uint8_t* p) { auto* a = reinterpret_cast<const A_ClearImg*>(p); real_vkCmdClearDepthStencilImage(a->cb, a->img, a->layout, &a->v.depthStencil, a->n, a->r); }
void VKAPI_PTR w_vkCmdClearColorImage(VkCommandBuffer cb, VkImage img, VkImageLayout layout, const VkClearColorValue* c, uint32_t n, const VkImageSubresourceRange* r) {
    if (!t_q) return real_vkCmdClearColorImage(cb, img, layout, c, n, r);
    Writer w{q_begin(&x_ClearColor, sz<A_ClearImg>() + sz<VkImageSubresourceRange>(n))};
    A_ClearImg v{cb, img, layout, {}, n, nullptr};
    v.v.color = *c;
    A_ClearImg* a = w.one(v);
    a->r = w.arr(r, n);
    q_commit();
}
void VKAPI_PTR w_vkCmdClearDepthStencilImage(VkCommandBuffer cb, VkImage img, VkImageLayout layout, const VkClearDepthStencilValue* c, uint32_t n,
                                             const VkImageSubresourceRange* r) {
    if (!t_q) return real_vkCmdClearDepthStencilImage(cb, img, layout, c, n, r);
    Writer w{q_begin(&x_ClearDs, sz<A_ClearImg>() + sz<VkImageSubresourceRange>(n))};
    A_ClearImg v{cb, img, layout, {}, n, nullptr};
    v.v.depthStencil = *c;
    A_ClearImg* a = w.one(v);
    a->r = w.arr(r, n);
    q_commit();
}

struct A_UpdBuf { VkCommandBuffer cb; VkBuffer buf; VkDeviceSize off, size; const void* data; };
void x_UpdBuf(const uint8_t* p) { auto* a = reinterpret_cast<const A_UpdBuf*>(p); real_vkCmdUpdateBuffer(a->cb, a->buf, a->off, a->size, a->data); }
void VKAPI_PTR w_vkCmdUpdateBuffer(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, VkDeviceSize size, const void* data) {
    if (!t_q) return real_vkCmdUpdateBuffer(cb, buf, off, size, data);
    if (size > 65536) CMDQ_FALLBACK(vkCmdUpdateBuffer, cb, buf, off, size, data);   // not allowed by Vulkan anyway
    Writer w{q_begin(&x_UpdBuf, sz<A_UpdBuf>() + align16(static_cast<size_t>(size)))};
    A_UpdBuf* a = w.one(A_UpdBuf{cb, buf, off, size, nullptr});
    a->data = w.bytes(data, static_cast<size_t>(size));
    q_commit();
}

struct A_Label { VkCommandBuffer cb; VkDebugUtilsLabelEXT l; };
void x_Label(const uint8_t* p) { auto* a = reinterpret_cast<const A_Label*>(p); real_vkCmdBeginDebugUtilsLabelEXT(a->cb, &a->l); }
void VKAPI_PTR w_vkCmdBeginDebugUtilsLabelEXT(VkCommandBuffer cb, const VkDebugUtilsLabelEXT* l) {
    if (!t_q) return real_vkCmdBeginDebugUtilsLabelEXT(cb, l);
    if (l->pNext) CMDQ_FALLBACK(vkCmdBeginDebugUtilsLabelEXT, cb, l);
    const size_t n = l->pLabelName ? std::strlen(l->pLabelName) + 1 : 0;
    Writer w{q_begin(&x_Label, sz<A_Label>() + align16(n))};
    A_Label* a = w.one(A_Label{cb, *l});
    a->l.pLabelName = static_cast<const char*>(w.bytes(l->pLabelName, n));
    q_commit();
}

struct A_Xfb { VkCommandBuffer cb; uint32_t first, n; const VkBuffer* bufs; const VkDeviceSize *offs, *sizes; };
void x_XfbBind(const uint8_t* p) { auto* a = reinterpret_cast<const A_Xfb*>(p); real_vkCmdBindTransformFeedbackBuffersEXT(a->cb, a->first, a->n, a->bufs, a->offs, a->sizes); }
void x_XfbBegin(const uint8_t* p) { auto* a = reinterpret_cast<const A_Xfb*>(p); real_vkCmdBeginTransformFeedbackEXT(a->cb, a->first, a->n, a->bufs, a->offs); }
void x_XfbEnd(const uint8_t* p) { auto* a = reinterpret_cast<const A_Xfb*>(p); real_vkCmdEndTransformFeedbackEXT(a->cb, a->first, a->n, a->bufs, a->offs); }
void put_xfb(Exec x, VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer* bufs, const VkDeviceSize* offs, const VkDeviceSize* sizes) {
    Writer w{q_begin(x, sz<A_Xfb>() + sz<VkBuffer>(n) + 2 * sz<VkDeviceSize>(n))};
    A_Xfb* a = w.one(A_Xfb{cb, first, n, nullptr, nullptr, nullptr});
    a->bufs = w.arr(bufs, n);
    a->offs = w.arr(offs, n);
    a->sizes = w.arr(sizes, n);
    q_commit();
}
void VKAPI_PTR w_vkCmdBindTransformFeedbackBuffersEXT(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer* bufs, const VkDeviceSize* offs,
                                                      const VkDeviceSize* sizes) {
    if (!t_q) return real_vkCmdBindTransformFeedbackBuffersEXT(cb, first, n, bufs, offs, sizes);
    put_xfb(&x_XfbBind, cb, first, n, bufs, offs, sizes);
}
void VKAPI_PTR w_vkCmdBeginTransformFeedbackEXT(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer* bufs, const VkDeviceSize* offs) {
    if (!t_q) return real_vkCmdBeginTransformFeedbackEXT(cb, first, n, bufs, offs);
    put_xfb(&x_XfbBegin, cb, first, n, bufs, offs, nullptr);
}
void VKAPI_PTR w_vkCmdEndTransformFeedbackEXT(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer* bufs, const VkDeviceSize* offs) {
    if (!t_q) return real_vkCmdEndTransformFeedbackEXT(cb, first, n, bufs, offs);
    put_xfb(&x_XfbEnd, cb, first, n, bufs, offs, nullptr);
}

// ---- stats ----------------------------------------------------------------------------------------------------------------------
// One log line every 10 s while deferred recording runs: records, bytes, waits for the worker, worker busy time.
std::chrono::steady_clock::time_point g_stat_t0{};
uint64_t g_stat_scopes = 0;
void maybe_log_stats() {
    Queue& q = *g_q;
    ++g_stat_scopes;
    const auto now = std::chrono::steady_clock::now();
    if (g_stat_t0 == std::chrono::steady_clock::time_point{}) { g_stat_t0 = now; return; }
    const double secs = std::chrono::duration<double>(now - g_stat_t0).count();
    if (secs < 10.0) return;
    NXLOG_I(kTag, "deferred recording: %.1f s, %llu submissions: %.0f records/s (%.1f MB/s), %llu waits for the worker (%.2f ms/s waited), "
            "worker busy %.1f ms/s, %llu ring-full waits, %llu calls run directly (unknown pNext)",
            secs, static_cast<unsigned long long>(g_stat_scopes), q.records / secs, q.bytes / secs / 1048576.0, static_cast<unsigned long long>(q.syncs),
            q.sync_ns / 1e6 / secs, q.worker_ns.exchange(0) / 1e6 / secs, static_cast<unsigned long long>(q.full_waits),
            static_cast<unsigned long long>(q.fallbacks));
    q.records = q.bytes = q.syncs = q.sync_ns = q.full_waits = q.fallbacks = 0;
    g_stat_scopes = 0;
    g_stat_t0 = now;
}

}  // namespace

bool enabled() { return g_q != nullptr && g_running; }

void install() {
    // read once; the back end may be brought up again (tests): the wrappers go in again whenever volk loaded fresh entry points
    static const int threads = static_cast<int>(std::clamp<int64_t>(config_get_int("nvn", "replay_threads", 1), 1, 2));
    g_threads = threads;
    if (g_threads < 2 || g_running) return;
    if (!vkCmdDraw || !vkUpdateDescriptorSets || !vkEndCommandBuffer) {
        NXLOG_W(kTag, "[nvn] replay_threads = 2: the Vulkan entry points are not loaded yet; recording stays on the render thread");
        return;
    }
    if (!g_q) {
        Queue* q = new Queue();
        q->mem = static_cast<uint8_t*>(VirtualAlloc(nullptr, kRingBytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        if (!q->mem) {
            delete q;
            NXLOG_W(kTag, "[nvn] replay_threads = 2: no memory for the command queue; recording stays on the render thread");
            return;
        }
        g_q = q;
    }
    Queue& q = *g_q;
    // the wrappers go in, unless they are in already (an entry point the device does not have stays null and unwrapped)
    if (vkCmdDraw != &Plain<&real_vkCmdDraw, PFN_vkCmdDraw>::wrap) {
#define CMDQ_WRAP_PLAIN(name) if (name) { real_##name = name; name = &Plain<&real_##name, PFN_##name>::wrap; }
#define CMDQ_WRAP_DEEP(name) if (name) { real_##name = name; name = &w_##name; }
#define CMDQ_WRAP_SYNC(name) if (name) { real_##name = name; name = &Synced<&real_##name, PFN_##name>::wrap; }
        CMDQ_PLAIN_LIST(CMDQ_WRAP_PLAIN)
        CMDQ_DEEP_LIST(CMDQ_WRAP_DEEP)
        CMDQ_SYNC_LIST(CMDQ_WRAP_SYNC)
#undef CMDQ_WRAP_PLAIN
#undef CMDQ_WRAP_DEEP
#undef CMDQ_WRAP_SYNC
    }
    q.head = q.rec_start = q.tail_seen = 0;
    q.unpublished = 0;
    g_batch = static_cast<uint32_t>(std::clamp<int64_t>(config_get_int("nvn", "replay_batch", 16), 1, 256));
    q.pub.store(0);
    q.tail.store(0);
    q.stop.store(false);
    q.th = std::thread(worker_main);
    g_running = true;
    NXLOG_I(kTag, "[nvn] replay_threads = 2: the render thread's Vulkan calls are made by a worker thread (nvn-rec), %zu MB queue",
            kRingBytes >> 20);
}

void shutdown() {
    if (!g_q || !g_running) return;
    Queue& q = *g_q;
    q.stop.store(true);
    WakeByAddressSingle(&q.pub);
    if (q.th.joinable()) q.th.join();
    g_running = false;
    // the wrappers stay: with no Scope open every call passes through
}

Scope::Scope() {
    if (!g_q || !g_running || t_q) return;
    on_ = true;
    t_q = true;
    g_q->active.store(1, std::memory_order_relaxed);
    if (g_q->sleeping.load(std::memory_order_seq_cst)) WakeByAddressSingle(&g_q->pub);
}

Scope::~Scope() {
    if (!on_) return;
    do_sync();
    t_q = false;
    g_q->active.store(0, std::memory_order_relaxed);
    maybe_log_stats();
}

Direct::Direct() {
    was_ = t_q;
    if (!was_) return;
    do_sync();
    t_q = false;
}

Direct::~Direct() {
    if (was_) t_q = true;
}

void sync() {
    if (g_q && t_q) do_sync();
}

bool queueing() { return g_q && t_q; }
uint8_t* record(void (*exec)(const uint8_t* payload), size_t payload) { return q_begin(exec, payload); }
void commit() { q_commit(); }

}  // namespace nxrt::nvn::be::cmdq
