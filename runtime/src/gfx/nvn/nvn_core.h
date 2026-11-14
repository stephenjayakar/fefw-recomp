// SPDX-License-Identifier: GPL-3.0-or-later
// Host objects of the NVN back end (S4d). Design (see docs/S4d.md):
//   * The game allocates every NVN object. "Heavy" objects (device, queue, command buffer, memory pool, texture, program, sync,
//     event, window, texture pool, sampler pool) keep a pointer to a host object in their first 16 bytes (GuestHeader); the
//     plain-data objects (builders, state objects, samplers, views, buffers) live entirely in guest memory (api_*.cpp).
//   * GPU addresses ARE host addresses: a memory pool's storage is the game's own memory, so an NVNbufferAddress is a host
//     pointer. The back end imports each pool as Vulkan memory (VK_EXT_external_memory_host), which makes buffers zero-copy
//     and GPU writes visible to the CPU without a copy.
//   * Game threads only record (command buffers, pool and texture bookkeeping). Everything that calls Vulkan runs on the
//     render thread (nvn_gpu.h): a submit is replayed there while the submitting game thread waits for the recording, not for
//     the GPU.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include <nvn/nvn.h>

#include "nvn_decoded.h"
#include "nvn_format.h"

namespace nxrt::nvn::be {

// ---- reference counted host objects -----------------------------------------------------------------------------------
enum class Kind : uint32_t {
    None = 0, Device, Queue, CommandBuffer, MemoryPool, Texture, Program, Sync, Event, Window, TexturePool, SamplerPool, CmdList
};

struct Obj {
    explicit Obj(Kind k) : kind(k) {}
    Obj(const Obj&) = delete;
    Obj& operator=(const Obj&) = delete;
    virtual ~Obj() = default;
    const Kind kind;
    std::atomic<int> refs{1};
    void ref() { refs.fetch_add(1, std::memory_order_relaxed); }
    void unref() {
        if (refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete this;
    }
};

template <class T>
class Ref {
public:
    Ref() = default;
    Ref(std::nullptr_t) {}
    explicit Ref(T* adopt) : p_(adopt) {}                 // takes over one reference
    static Ref share(T* p) { if (p) p->ref(); return Ref(p); }
    Ref(const Ref& o) : p_(o.p_) { if (p_) p_->ref(); }
    Ref(Ref&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    Ref& operator=(Ref o) noexcept { std::swap(p_, o.p_); return *this; }
    ~Ref() { if (p_) p_->unref(); }
    T* get() const { return p_; }
    T* operator->() const { return p_; }
    T& operator*() const { return *p_; }
    explicit operator bool() const { return p_ != nullptr; }
    T* release() { T* p = p_; p_ = nullptr; return p; }
private:
    T* p_ = nullptr;
};

// ---- the guest side of a heavy object ------------------------------------------------------------------------------------
constexpr uint32_t kGuestMagic = 0x4e585433;   // "NXT3"
struct GuestHeader {
    Obj* host;
    uint32_t magic;
    uint32_t kind;
};
static_assert(sizeof(GuestHeader) == 16, "header layout");

// The host object behind a guest object, or null when the guest memory does not hold one of this kind.
template <class T>
inline T* host_of(const void* guest) {
    if (!guest) return nullptr;
    const GuestHeader* h = static_cast<const GuestHeader*>(guest);
    if (h->magic != kGuestMagic || h->kind != static_cast<uint32_t>(T::kKind) || !h->host) return nullptr;
    return static_cast<T*>(h->host);
}
// Installs obj (one reference is handed over to the guest object). An object that was there is released first.
void bind_guest(void* guest, Obj* obj);
// Finalize: the guest object stops naming its host object (which lives on while commands still refer to it).
void unbind_guest(void* guest);

// ---- memory ----------------------------------------------------------------------------------------------------------------
struct PoolGpu;   // Vulkan side of a pool (nvn_gpu_pool.cpp): imported memory and buffer(s)

struct MemoryPool : Obj {
    static constexpr Kind kKind = Kind::MemoryPool;
    MemoryPool() : Obj(kKind) {}
    ~MemoryPool() override;
    uint8_t* base = nullptr;        // storage (host address); null for a virtual pool
    uint64_t size = 0;
    int flags = 0;                  // NVNmemoryPoolFlags
    uint64_t gpu_base = 0;          // GPU virtual address of byte 0: the host address, or a synthetic address for a virtual pool
    bool is_virtual = false;
    bool is_shader_code = false;
    std::atomic<bool> dead{false};
    // Virtual (sparse) pools: which 64 KiB tiles of the range are backed by which physical memory (nvnMemoryPoolMapVirtual). The textures
    // living in the pool read their texels from there.
    struct VirtualMap {
        uint64_t voff = 0, size = 0;    // the mapped range of this pool
        Ref<MemoryPool> phys;           // the physical pool
        int64_t poff = 0;               // offset in the physical pool of byte 0 of the range
    };
    std::mutex vmu;
    std::vector<VirtualMap> vmaps;
    PoolGpu* gpu = nullptr;         // render thread only
    // Background import (gpu::prefetch_pool): 0 not started, 1 queued, 3 running on the importer, 2 finished (result in `prefetched`, taken by
    // the render thread), 4 taken over by the render thread while it was still queued (the importer skips it).
    std::atomic<int> prefetch{0};
    std::atomic<PoolGpu*> prefetched{nullptr};
};

// A GPU address resolved to its pool.
struct PoolHit {
    MemoryPool* pool = nullptr;
    uint64_t offset = 0;
};
struct Texture;
// Lifetime gauges (nvn_objects.cpp): a pool retired by the game (retire_pool) or a texture it finalized counts until its host object is
// destroyed. Each object is counted once: pools are retired once (dead), textures finalized once (destroyed).
void lifetime_pool_retired(const MemoryPool* p);
void lifetime_pool_freed(const MemoryPool* p);
void lifetime_texture_finalized();
void lifetime_texture_freed(const Texture* t);
void pool_register(MemoryPool* p);
void pool_unregister(MemoryPool* p);
PoolHit pool_lookup(uint64_t gpu_address);          // pool is null when the address is in no pool
uint64_t pool_generation();                         // moves with every pool registration and removal
inline uint8_t* host_ptr(uint64_t gpu_address) { return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(gpu_address)); }

// The data of an NVNbuffer (0x30 bytes in the game's memory).
struct BufferData {
    uint64_t address;               // GPU address
    uint64_t size;
    uint64_t pool;                  // NVNmemoryPool*
    uint64_t magic;
};
static_assert(sizeof(BufferData) <= sizeof(NVNbuffer), "buffer layout fits NVNbuffer");

// ---- textures --------------------------------------------------------------------------------------------------------------
struct TexGpu;    // Vulkan side of a texture (nvn_gpu_texture.cpp)

struct Texture : Obj {
    static constexpr Kind kKind = Kind::Texture;
    Texture() : Obj(kKind) {}
    ~Texture() override;
    TextureBuilderInfo b;                   // as built
    const FormatInfo* fmt = nullptr;
    uint8_t* mem = nullptr;                 // storage in the game's memory (null: no storage, e.g. a render target the game never reads)
    Ref<MemoryPool> vpool;                  // a sparse texture: the virtual pool it lives in (mem is null; the texels are in the mapped tiles)
    uint64_t voffset = 0;                   // ... and its offset there
    // Host-backed storage (nvnTextureInitialize): the storage lies outside its pool's memory (placed past the end of the pool, or offset -1
    // when the game's allocator found no room) in a pool the CPU cannot access. Only the GPU image holds the texels: mem is null, nothing is
    // uploaded from or read back into the game's memory, and the image starts cleared. Texture objects over the same storage (same pool and
    // offset, the same shape) share one image, as objects over the same bytes of a pool do; offset -1 names no storage and shares nothing.
    bool overflow = false;
    const MemoryPool* overflow_pool = nullptr;   // identity of the storage only (never dereferenced)
    int64_t overflow_offset = -1;
    uint64_t storage_size = 0;
    uint32_t block_height_gobs = 0;         // level-0 block height in GOBs (block-linear layouts)
    bool block_linear = true;               // false: pitch-linear (NVN_TEXTURE_FLAGS_LINEAR)
    bool is_display = false;                // window texture
    bool cpu_coherent = false;              // in a CPU_UNCACHED pool: the CPU writes it without FlushMappedRange (watched by page protection)
    uint32_t levels = 1, layers = 1;        // levels clamped to the size; layers = array size / cube faces * count
    uint32_t id = 0;                        // unique, for logs
    std::atomic<bool> cpu_dirty{true};      // the CPU may have written the storage since the last upload
    std::atomic<bool> destroyed{false};     // finalized by the game
    // Background preparation of the upload data (nvn_texprep.cpp). prep_gen counts the CPU-write notifications; a prepared result is only used
    // when the count has not moved since the job read the storage.
    std::atomic<uint32_t> prep_gen{0};
    std::atomic<uint8_t> prep_flags{0};     // bit 0: the render thread uploaded it at least once
    std::atomic<uint32_t> prep_sched{0};    // 0x80000000 | prep_gen when a background job was requested for that generation
    std::atomic<uint32_t> prep_prov{0};     // 0x80000000 | prep_gen when the image holds a quick stand-in for that generation (texprep::provisional_uploaded)
    TexGpu* gpu = nullptr;                  // render thread only
    // Sparse textures kept as a Vulkan sparse-residency image (too big to back fully): the storage ranges [first, second) written or remapped
    // since the last upload, so only those 64 KiB tiles are uploaded again. sparse_all: everything (until the first upload).
    std::mutex sparse_mu;
    std::vector<std::pair<uint64_t, uint64_t>> sparse_dirty;
    bool sparse_all = true;
};

struct TexturePool : Obj {
    static constexpr Kind kKind = Kind::TexturePool;
    TexturePool() : Obj(kKind) {}
    struct Entry {
        Ref<Texture> tex;
        TextureViewInfo view;
        bool has_view = false;
        bool used = false;
    };
    MemoryPool* mem = nullptr;
    int64_t offset = 0;
    int count = 0;
    std::vector<Entry> entries;             // index = texture id
    // Registrations so far and the ids of the latest ones, oldest first (changes.back() is registration `version`): the pool descriptor
    // arrays (nvn_pipeline.cpp) write only the ids changed since they were last current. Under `lock`.
    uint64_t version = 0;
    std::vector<uint32_t> changes;
    std::shared_mutex lock;
};

struct SamplerPool : Obj {
    static constexpr Kind kKind = Kind::SamplerPool;
    SamplerPool() : Obj(kKind) {}
    MemoryPool* mem = nullptr;
    int64_t offset = 0;
    int count = 0;
    std::vector<SamplerInfo> entries;
    std::vector<uint8_t> used;
    uint64_t version = 0;                   // as TexturePool's
    std::vector<uint32_t> changes;
    std::shared_mutex lock;
};

// Notes registration `id` in a pool's change list (caller holds the pool's lock exclusively); the list keeps the latest 4096 to 8192.
template <class Pool>
inline void pool_note_change(Pool& p, int id) {
    ++p.version;
    p.changes.push_back(static_cast<uint32_t>(id));
    if (p.changes.size() > 8192) p.changes.erase(p.changes.begin(), p.changes.begin() + 4096);
}

// Every initialized texture is in a registry keyed by its storage range, so a flush of pool memory can mark the textures
// that live there as CPU-written (nvn_texture_registry.cpp). Host-backed textures (Texture::overflow) are kept apart, by pool and offset:
// no CPU write reaches them.
void texture_registry_add(Texture* t);
void texture_registry_remove(Texture* t);
void texture_registry_detach_gpu();   // sets Texture::gpu to null on every registered texture (the images are being freed: gpu::tex_shutdown)
void textures_cpu_wrote(const void* lo, uint64_t size);      // marks every texture whose storage overlaps [lo, lo+size) cpu_dirty and tells the GPU layer
size_t texture_registry_size();
// Frame-trace mark with the texture id as its argument (a no-op unless [prof] trace is on): tx.init, tx.write, tx.flush, tx.reg, tx.use.
void tx_mark(const char* what, const Texture* t);
// Sparse textures: registered by the virtual pool they live in. A remap of tiles (nvnMemoryPoolMapVirtual) or a write to the physical
// memory behind a tile marks the textures over it CPU-written; texture_gather_virtual builds the bytes of the storage out of the mapped
// tiles (unmapped tiles are zeros).
void texture_registry_add_virtual(Texture* t);
void texture_registry_remove_virtual(Texture* t);
// Hash of the bytes texture_gather_virtual would copy (mapped tiles and their positions), without copying them. False for a texture that is not sparse.
bool texture_hash_virtual(const Texture& t, uint64_t& hash);
// When the whole range of a sparse texture is one piece of one physical pool: a pointer to it (kept valid by `keep`), else null (use texture_gather_virtual).
const uint8_t* texture_virtual_direct(const Texture& t, Ref<MemoryPool>& keep);
// Background preparation (nvn_texprep.h): handles just bound in a command list recorded with texture pool tp; the textures behind them that the CPU wrote are
// queued for preparation. tp must stay alive for the call (the list being recorded holds it).
void textures_note_bound(TexturePool* tp, const uint64_t* handles, int count);
// Bumped whenever a texture may have become a candidate for background preparation (a texture was initialized or registered in a pool, or the CPU wrote one).
// textures_note_bound remembers, per thread, which handles it has looked at since the last bump: a command list binds thousands of handles per frame and
// nearly all of them are textures that were looked at long ago.
extern std::atomic<uint64_t> g_tex_scan_epoch;
extern std::atomic<bool> g_tex_prep_on;       // background texture preparation is running ([nvn] tex_prep); set by nvn_texprep.cpp when it starts
void virtual_textures_remapped(const MemoryPool* vpool, uint64_t voff, uint64_t size);
bool texture_gather_virtual(const Texture& t, std::vector<uint8_t>& out);
// Copies [at, at + len) of a sparse texture's storage out of the mapped tiles into dst (unmapped bytes are zeros). Returns true when the
// whole range is mapped.
bool texture_read_virtual(const Texture& t, uint64_t at, uint64_t len, uint8_t* dst);
// The physical memory behind each 64 KiB tile of a sparse texture's storage (null: not mapped as one piece); a later map wins.
void texture_virtual_tile_ptrs(const Texture& t, std::vector<const uint8_t*>& out);
// Another initialized texture whose storage starts at the same address and has the same shape (target, size, levels, format, layout),
// that already has a GPU image: the game makes several texture objects over one piece of memory (one per mip level of a cube map it
// renders, and the one the materials sample), and on the console they are the same bytes. Null when there is none. For a host-backed
// texture: one at the same pool and offset.
Ref<Texture> texture_registry_find_alias(const Texture* t);
size_t texture_registry_overflow_size();   // host-backed textures registered
// [nvn] trace_tex_events = 1: logs the life of block-compressed 128x128 cube textures (init, writes by the CPU, copies, uploads) as nvn.tev lines.
void tev(const Texture* t, const char* what, uint64_t a = 0, uint64_t b = 0);
void textures_mark_override_candidates_dirty();   // small BC7 sRGB textures: upload again (nxrt/tex_override.h)
// Every live registered texture (storage in the game's memory) for which pred answers true, referenced. pred runs under the registry lock: it must be
// quick and must not read the storage.
std::vector<Ref<Texture>> texture_registry_collect(bool (*pred)(const Texture& t, void* user), void* user);

// NVNtextureHandle = (sampler id << 20) | texture id
constexpr uint64_t make_texture_handle(uint32_t tex_id, uint32_t sampler_id) { return (static_cast<uint64_t>(sampler_id) << 20) | (tex_id & 0xfffffu); }
constexpr uint32_t handle_texture_id(uint64_t h) { return static_cast<uint32_t>(h & 0xfffffu); }
constexpr uint32_t handle_sampler_id(uint64_t h) { return static_cast<uint32_t>((h >> 20) & 0xfffu); }

// ---- programs --------------------------------------------------------------------------------------------------------------
struct ShaderInfo;   // translated shader (nvn_shaders.h)

struct Program : Obj {
    static constexpr Kind kKind = Kind::Program;
    Program() : Obj(kKind) {}
    // stages indexed by the NVNshaderStage value: 0 vertex, 1 fragment, 2 geometry, 3 tess control, 4 tess eval, 5 compute
    const ShaderInfo* stage[6] = {};
    uint64_t code_hash[6] = {};
    char missing[64] = {};                  // keys of the blocks SetShaders found no translation for (trace: "cs -" dispatches)
    bool linked = false;
    uint32_t serial = 0;
    std::atomic<uint8_t> gpl_bound{0};      // 1 once a command buffer was recorded with these shaders (the shader parts are queued first); reset by SetShaders
    std::string label;
};

// ---- synchronisation -----------------------------------------------------------------------------------------------------
struct Sync : Obj {
    static constexpr Kind kKind = Kind::Sync;
    Sync() : Obj(kKind) {}
    std::atomic<uint64_t> serial{0};        // the GPU timeline value that signals it (0: not armed)
    std::atomic<bool> armed{false};
};

struct Event : Obj {
    static constexpr Kind kKind = Kind::Event;
    Event() : Obj(kKind) {}
    MemoryPool* pool = nullptr;
    const NVNmemoryPool* guest_pool = nullptr;   // the guest pool object the builder named (nvnEventGetMemoryPool)
    int64_t offset = 0;                     // the 32-bit event value lives in pool memory at this offset
    std::atomic<uint32_t> shadow{0};        // the value after every GPU signal recorded so far (replay thread)
    std::atomic<bool> shadow_valid{false};
};

// ---- window ----------------------------------------------------------------------------------------------------------------
struct Window : Obj {
    static constexpr Kind kKind = Kind::Window;
    Window() : Obj(kKind) {}
    std::vector<Ref<Texture>> textures;
    int present_interval = 1;
    int active = 0;                         // textures in use (SetNumActiveTextures)
    int next_acquire = 0;                   // round robin
    int crop[4] = {0, 0, 0, 0};             // x, y, w, h of the part shown (w == 0: all)
    void* native_window = nullptr;
    WindowHdrInfo hdr;                      // NVN2 colour space and HDR metadata (from the builder, then the window setters)
    std::atomic<uint64_t> present_serial[4] = {};   // the GPU serial each texture's last present followed (NVN2 acquire syncs)
    std::mutex lock;
};

// ---- device and queue ------------------------------------------------------------------------------------------------
struct Device : Obj {
    static constexpr Kind kKind = Kind::Device;
    Device() : Obj(kKind) {}
    int flags = 0;
    int window_origin = NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT;
    int depth_mode = NVN_DEPTH_MODE_NEAR_IS_MINUS_W;
    std::atomic<uint32_t> next_texture_id{1};
    std::atomic<uint32_t> debug_domain{1};
};

struct Queue : Obj {
    static constexpr Kind kKind = Kind::Queue;
    Queue() : Obj(kKind) {}
    Device* device = nullptr;
    int flags = 0;
    size_t command_memory = 0, compute_memory = 0, control_memory = 0;
    std::atomic<uint64_t> last_serial{0};   // the serial of this queue's last submission (NVN2 fence syncs)
};

}  // namespace nxrt::nvn::be
