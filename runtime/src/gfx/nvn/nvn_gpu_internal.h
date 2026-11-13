// SPDX-License-Identifier: GPL-3.0-or-later
// Internal interfaces of the Vulkan side of the NVN back end (render thread only). Modules:
//   nvn_gpu.cpp          the Gpu singleton: device access, render thread, submission slots, timeline, deferred destruction, uploads
//   nvn_gpu_pool.cpp     memory pools imported as Vulkan memory, GPU address -> VkBuffer
//   nvn_gpu_texture.cpp  texture images, uploads, views, samplers, layout tracking
//   nvn_replay.cpp       command list interpreter (state, passes, draws, clears, copies)
//   nvn_pipeline.cpp     pipeline layouts, graphics/compute pipelines, descriptor sets
//   nvn_present.cpp      window texture -> swapchain
// Everything declared here must be called on the render thread (gpu::on_render_thread()) unless it says otherwise.
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <volk.h>

#include "nxrt/gfx.h"
#include "nvn_present_pass.h"

namespace nxrt::extras { class OutputPass; }
#include "nvn_core.h"
#include "nvn_gpu.h"

namespace nxrt {
class FrameLimiter;
}

namespace nxrt::nvn::be::gpu {

bool on_render_thread();
void flush_caches();              // writes the pipeline cache to disk now (the key file is appended as pipelines are created)
gfx::Device& device();
VkDevice vkdev();

struct Counters {
    std::atomic<uint64_t> submits{0}, lists_replayed{0}, draws{0}, draws_skipped{0}, dispatches{0}, clears{0}, copies{0}, presents{0};
    std::atomic<uint64_t> passes{0}, pass_breaks{0}, pipelines_created{0}, pipeline_cache_hits{0}, descriptor_sets{0};
    std::atomic<uint64_t> textures_created{0}, texture_uploads{0}, texture_upload_bytes{0}, bc7_transcodes{0}, textures_failed{0};
    std::atomic<uint64_t> pools_imported{0}, unresolved_addresses{0}, unsupported_ops{0}, submit_replay_ns{0};
    std::atomic<uint64_t> game_barriers{0}, game_barriers_left_out{0}, dispatch_barriers_left_out{0};   // nvnCommandBufferBarrier replayed (left out: [nvn] game_barriers = tracked); [nvn] dispatch_barriers = tracked
};
Counters& counters();
bool counters_ready();   // false until the GPU back end is set up (counters() would read through a null pointer)
Counters* counters_if();   // null before the GPU is up
// An operation the back end does not implement was skipped: logged the first time per name, counted for the report.
void note_unsupported(const char* what);
std::string unsupported_report();

// Hands the submissions recorded so far to the GPU ([nvn] submit_batch keeps them back). Call before a queue operation that must come after
// them: the present, sparse binds.
void submit_recorded();
// The present: with [nvn] submit_batch = frame, the batch goes in the present's own vkQueueSubmit2 (one kernel submission a frame).
// true: p is filled (cmds holds its command buffers); call present_batch_done(p) after Renderer::end_frame(p). false: the batch is
// submitted already (submit_recorded), call end_frame without a prefix.
bool take_batch_for_present(std::vector<VkCommandBuffer>* cmds, gfx::Renderer::SubmitPrefix* p);
void present_batch_done(const gfx::Renderer::SubmitPrefix& p);
// Runs fn on the render thread and waits for it. From the render thread itself it just calls fn.
void run_on_render_thread(const std::function<void()>& fn);
// Test and one-shot hook: runs fn(Frame&) on the render thread with a fresh command buffer, submits it and waits for the GPU.
// Returns the serial of the submission.
uint64_t run_frame_sync(const std::function<void(struct Frame&)>& fn);
// Brings the Vulkan layer up on a device without touching the NVN entry points (tests of nvn_gpu_*.cpp).
bool init_for_test(gfx::Device& dev, gfx::Renderer* renderer = nullptr);
void shutdown_for_test();

// ---- the recording context of one submission ---------------------------------------------------------------------------
struct UploadAlloc {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    uint8_t* ptr = nullptr;       // host pointer to the allocation (host visible, coherent)
    VkDeviceSize size = 0;
    explicit operator bool() const { return buffer != VK_NULL_HANDLE; }
};

struct Frame {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    uint64_t serial = 0;          // the timeline value this submission signals when the GPU has run it
    void* slot = nullptr;         // nvn_gpu.cpp: the submission slot (command pool, descriptor pools, upload memory)
    // Host-visible memory valid until the submission has completed: small per-draw data (support buffer, driver constants,
    // inline uniform data). align is a power of two.
    UploadAlloc upload(VkDeviceSize size, VkDeviceSize align);
    // A staging buffer for texture data (own buffer when large). Freed when the submission has completed.
    UploadAlloc staging(VkDeviceSize size);
    // A staging buffer somebody else made and filled (background texture preparation): freed like the ones above when the submission has completed.
    UploadAlloc adopt_staging(gfx::Buffer b);
    // Runs fn on the render thread after the submission has completed on the GPU (and, for objects the replay used, no
    // later submission can still be using them if they were last used in this one).
    void defer(std::function<void()> fn);
    // A descriptor set of this submission's pool (freed when the slot is reused). Null on failure.
    VkDescriptorSet alloc_set(VkDescriptorSetLayout layout);
};

// Destruction: fn runs once every submission made so far (and the one being recorded) has completed.
void defer_destroy(std::function<void()> fn);
uint64_t completed_serial();      // cached timeline value; call poll_completed() for a fresh read
uint64_t poll_completed();
uint64_t current_serial();        // the serial of the submission being recorded (or the last one)

// ---- GPU pass profiler ([prof] gpu_passes = true, needs the frame trace) --------------------------------------------------------------
// Writes a GPU timestamp after everything recorded so far; the time since the previous stamp of the submission becomes a trace event
// `gp:<label>` (arg = draws or 0). No-ops when the profiler is off.
bool gpass_on();
void gpass_stamp(Frame& f, const char* label, uint32_t arg = 0);

// ---- barriers ---------------------------------------------------------------------------------------------------------------
void image_barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to, VkImageAspectFlags aspect,
                   uint32_t base_level, uint32_t level_count, uint32_t base_layer, uint32_t layer_count);
// All commands before, all commands after, all memory (used after passes, dispatches, copies and fills).
void memory_barrier(VkCommandBuffer cb);
// Full memory barriers this thread recorded into `cb` so far: a change since a point in its recording means one lies between.
uint64_t memory_barrier_count(VkCommandBuffer cb);
// [nvn] pass_barriers = tracked (S38): image_barrier derives its stage and access masks from the two layouts instead of all commands and
// all memory. Set by the replay for each submission; any thread may record image barriers, so it is an atomic.
void set_narrow_barriers(bool on);

// ---- pools ---------------------------------------------------------------------------------------------------------------
struct BufRef {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize avail = 0;       // bytes from offset to the end of the pool buffer
    bool ok = false;
};
// The buffer and offset of `size` bytes at a GPU address. When the address is in no pool (or size runs past the pool), ok is
// false and the buffer is the dummy buffer (a zeroed 64 KiB buffer) so the draw can go on; this is logged once per address.
BufRef resolve_buffer(uint64_t gpu_address, uint64_t size);
VkBuffer dummy_buffer();
// The last 32 bytes of the dummy buffer hold the default of an unset vertex attribute: float (0,0,0,1) and integer (0,0,0,1).
constexpr VkDeviceSize kDummyAttrFloatOffset = 65504, kDummyAttrIntOffset = 65520;
uint64_t pool_imports();          // statistics
// Called from destroy_pool_gpu on the render thread.
void pool_gpu_destroy(PoolGpu* p);
void pool_shutdown();             // destroys every imported pool still alive and the dummy buffer (device idle)
// Device-local copies of pool pages (S38, [nvn] pool_mirror). resolve_read is resolve_buffer for data draws only read (vertex and index
// buffers, uniform blocks): it gives the device-local copy when every page of the range is current there, else the pool. The replay calls
// note_gpu_write for every range a command may write in pool memory (true: a copied page there went stale, so bindings must be made again),
// and mirror_begin_submit before anything else of a submission (it records the copies of the pages draws wanted; on = this submission may
// read copies).
bool mirror_enabled();
enum ReadUse { kReadVertex = 0, kReadIndex = 1, kReadUniform = 2 };
enum WriteUse { kWriteStorage = 0, kWriteXfb = 1, kWriteImageCopy = 2, kWriteUpdate = 3, kWriteFill = 4, kWriteCopy = 5 };
BufRef resolve_read(uint64_t gpu_address, uint64_t size, int use);
// The device-local copy of [gpu_address, +size) for the compute shadows (nvn_replay.cpp): its chunk and offset (buf null: no copy for it, or the
// range spans two chunks), and whether the copy holds the pool's bytes right now. Call it before note_gpu_write invalidates the range.
struct MirrorSpan { VkBuffer buf = VK_NULL_HANDLE; VkDeviceSize offset = 0; bool valid = false; VkDeviceSize avail = 0; };   // avail: bytes to the chunk's end
MirrorSpan mirror_span(uint64_t gpu_address, uint64_t size);
// The GPU wrote the same bytes of the range into the pool and into its copy (a shadow's result): the pages wholly inside the range hold the
// pool's bytes again and are tracked again (unless hot or banned), so the next dispatch reads them from video memory.
void mirror_span_written(uint64_t gpu_address, uint64_t size);
// The compute bindings that use the copy directly (nvn_replay.cpp): records into cb the copies, from the pool, of the pages of the range the
// copy does not hold, and tracks them like the copies of mirror_begin_submit. The caller orders earlier writes to the pool before cb's copies
// and the copies before their readers. True when every page of the range is valid in the copy afterwards; false (nothing or only part of it
// copied) when a page is hot, banned or cannot be tracked, or the range has no copy. *bytes (optional) receives the bytes copied.
bool mirror_fill(VkCommandBuffer cb, uint64_t gpu_address, uint64_t size, uint64_t* bytes);
bool note_gpu_write(uint64_t gpu_address, uint64_t size, int use);
void mirror_begin_submit(Frame& f, bool on);
void mirror_destroy(PoolGpu* p);

// ---- compute results held in the pool copy (nvn_gpu_pool.cpp, "compute out") ----------------------------------------------------
// A dispatch that binds the pool copy directly (nvn_replay.cpp, compute shadows) writes its result there: the range is "out" until the
// copy-back to the pool. [nvn] compute_out_lazy (default on): at the end of a submission a result is not copied back but held in the
// copy, its pages guarded (nxrt/write_watch.h: the CPU's first access waits for an on-demand copy, out_fetch); draws read it from the copy
// (resolve_read), and the other GPU readers of the pool get it copied back first (out_flush). Results in pages the CPU has read before
// (a guard fault or nvnMemoryPool/BufferInvalidateMappedRange) are copied back at the end of the submission, as without the option.
// Off: every result goes back at the end of its submission, and before any command other than a dispatch, a state change or a barrier.
// Render thread unless noted.
bool out_lazy();
bool out_any();                                                    // any thread (a hint)
// The direct binding [addr, +size) at (mbuf, moff) of the copy was written (host: the pool's buffer). True: the same range was out already.
bool out_add(uint64_t addr, VkDeviceSize size, VkBuffer mbuf, VkDeviceSize moff, const BufRef& host);
// The same range is out: its place in the copy.
bool out_exact(uint64_t addr, VkDeviceSize size, VkBuffer* buf, VkDeviceSize* off);
// Some page of [addr, +size) holds a result that is newer in the copy than in the pool.
bool out_overlaps(uint64_t addr, uint64_t size);
// Moves whenever the held results change: out_overlaps' answer for a range is the same while it stays (rt agent, notes/rt.md).
uint64_t out_gen();
// No result held now in [addr, +size) was added since out_gen() was `gen`, given none was there then (false when the log of additions does
// not reach back to `gen`). Render thread.
bool out_clear_since(uint64_t addr, uint64_t size, uint64_t gen);
// resolve_read would bind the copy for [addr, +size) (every page current there).
bool out_servable(uint64_t addr, uint64_t size);
// Why results went back (the stats line): the end of a submission, or the GPU reader of the pool that needed them.
enum OutWhy {
    kOutGpuReader = 0,      // unspecified reader
    kOutSubmitEnd,          // end of the submission (CPU-read pages, not guardable, or compute_out_lazy off)
    kOutMirrorOff,          // a submission without pool copies (gpuperf A/B off window)
    kOutDispatch,           // a compute binding over a held result that cannot bind the copy (overlapping ranges, no copy, shadow)
    kOutDrawBuffer,         // a vertex, index or uniform buffer the copy cannot serve (two chunks, unaligned uniform ...)
    kOutStorageXfb,         // a storage or feedback buffer of a draw the copy cannot serve
    kOutIndirect,           // indirect or feedback-draw arguments the copy cannot serve
    kOutBufferWrite,        // a command writing pool memory over a held result (clear, copy destination, uniform update, report)
    kOutBufferRead,         // a buffer copy or buffer-texture copy reading a held result the copy cannot serve
    kOutTexture,            // a texture upload reading a held result
    kOutEvent,              // an event signal over a held result
    kOutOther,              // anything else (feedback begin/end, ops without an address)
    kOutWhyCount
};
// Records into cb (outside any pass) the copy-back of every result sharing a page with [addr, +size) (size UINT64_MAX: all), with the
// barriers around it. Returns the bytes.
uint64_t out_flush(VkCommandBuffer cb, uint64_t addr, uint64_t size, int why);
// The end of a submission's recording (serial): each result written in it is held (guarded) or copied back.
void out_end_submit(VkCommandBuffer cb, uint64_t serial);
// Any thread: the CPU will read [addr, +size) (nvnMemoryPool/BufferInvalidateMappedRange; a guard fault calls it too): results held
// there are copied back now (waiting for the GPU), and later results there go back at the end of their submission.
void out_fetch(uint64_t addr, uint64_t size, bool by_fault);
uint64_t out_bytes_total();                                        // bytes copied back so far (statistics)
void out_reset();                                                  // replay shutdown: drops every entry (guards off)

// ---- textures --------------------------------------------------------------------------------------------------------------
// What a binding site needs to see of a texture.
struct ViewRequest {
    VkImageViewType type = VK_IMAGE_VIEW_TYPE_2D;     // the shader's image dimension (from the SPIR-V binding)
    bool depth_compare = false;
    uint32_t base_level = 0, level_count = 0;         // 0 = all levels from base
    uint32_t base_layer = 0, layer_count = 0;         // 0 = all layers from base
    VkFormat format = VK_FORMAT_UNDEFINED;            // UNDEFINED: the image's own format
    VkComponentMapping swizzle = {};                  // identity when all IDENTITY
    bool depth_aspect = true;                         // depth/stencil images: sample depth (true) or stencil (false)
    uint8_t component = 0;                            // the shader's sampled type: 0 unknown, 1 float, 2 sint, 3 uint (raw-bit aliases of 32-bit formats)
};
// Creates the VkImage if needed and returns it (null when the format cannot be created: logged once; callers then bind
// the dummy). Does not upload.
TexGpu* tex_gpu(Texture* t);
// tex_gpu for a texture about to be a render target: an image made now gets the resolution scale when the texture qualifies ([gfx]
// render_scale). tex_gpu_scaled makes the image at a given scale (a copy of a scaled target). Both return the existing image if there is one.
TexGpu* tex_gpu_attachment(Texture* t);
TexGpu* tex_gpu_scaled(Texture* t, float scale);
float render_scale();                                   // [gfx] render_scale (1 until known)
float tex_scale(Texture* t);                            // the image size over the NVN size (1 for most textures)
VkExtent3D tex_image_extent(Texture* t, uint32_t level = 0);   // the image's own size (tex_extent times tex_scale)
// The scale a render target should have: its own when it was brought back to 1 for good, else the render scale when it qualifies.
float tex_scale_wanted(Texture* t);
// Remakes the image at scale `to`, content blitted over (a render target made unscaled, storage use, a pass mixing scales). Going to 1 is
// for good. No render pass open.
bool tex_rescale(Texture* t, Frame& f, float to, const char* why);
// Records an upload when the texture's memory was not yet uploaded or the CPU wrote it since (and the GPU did not render it).
// Must be called with no render pass open on `f.cb`. Leaves the image in SHADER_READ_ONLY_OPTIMAL after an upload.
void tex_ensure_current(Texture* t, Frame& f);
// The same for a copy or clear of the image: never a stand-in ([nvn] tex_async), the real content now.
void tex_ensure_exact(Texture* t, Frame& f);
void tex_poll_cpu_writes();   // start of a submission: textures of coherent pools whose pages the CPU wrote become dirty (nvn_gpu_texture.cpp)
// Moves the whole image to `layout` (no-op if already there). Call with no render pass open.
void tex_transition(Texture* t, Frame& f, VkImageLayout layout);
// nvn_pipeline.cpp: tex_transition moves t to `layout`. A texture leaving its sampled layout is looked at again by the next draw that
// samples through a pool array (pool_textures_to_prepare).
void pool_note_layout(Texture* t, VkImageLayout layout);
// nvn_pipeline.cpp, any thread: the CPU wrote t (or its prepared content changed): the pools it is registered in look at it again at their
// next draw, without a scan of every registered texture.
void pool_note_cpu_write(Texture* t);
// nvn_pipeline.cpp, render thread: pool_textures_to_prepare calls that scanned every registered texture, and all calls (replay_stats).
void pool_scan_counts(uint64_t* full_scans, uint64_t* looks);
// nvn_pipeline.cpp: t's image and views were remade (tex_rescale) or released: the pool arrays write the ids it sits at again.
void pool_texture_remade(Texture* t);
VkImageLayout tex_layout(Texture* t);
// The texture has been (or is about to be) written by the GPU: no more uploads from memory until the CPU writes it.
void tex_mark_gpu_written(Texture* t);
void tex_sparse_before_copy(Texture* t, uint32_t layer, uint32_t level);   // a sparse texture: bind the mapped tiles a copy writes
VkImage tex_image(Texture* t);
VkFormat tex_vk_format(Texture* t);
VkImageAspectFlags tex_aspect(Texture* t);
VkExtent3D tex_extent(Texture* t, uint32_t level = 0);
uint32_t tex_levels(Texture* t);
uint32_t tex_level_drop(Texture* t);   // NVN levels the image leaves out at the top ([gfx] texture_detail; 0 for nearly every texture)
uint32_t tex_layers(Texture* t);
bool tex_is_depth(Texture* t);
// Readiness aligned with supported uploads; retains layout dependencies.
bool tex_pool_preparation_pending(Texture* t);
// A texture whose first upload may wait for a later frame when it is only needed through a texture pool ([nvn] pool_upload_ms): an asset texture
// (ASTC or BCn from the game's memory) whose image never held content, is not rendered to, and is not a depth image. Its pool element is the dummy
// until then. Render thread.
bool tex_first_upload_deferrable(Texture* t);
// Sampled image view for a binding (cached). Null when impossible (the caller binds the dummy view).
VkImageView tex_sampled_view(Texture* t, const ViewRequest& req);
// A float-declared sampler over this texture (the view format `requested`, UNDEFINED: the image's own): 0 when a float view gives the
// shader the texel bits (float formats; 32-bit integer ones through their float alias), else 2 / 3: signed / unsigned integers that only
// an integer view can read, so the shader needs its sampler retyped (spirv_int_samplers) and the binding an integer view.
uint8_t tex_int_sampler_component(Texture* t, VkFormat requested);
// Single-level, single-layer (or all layers when layer_count > 1) view for use as an attachment.
VkImageView tex_attachment_view(Texture* t, uint32_t level, uint32_t layer, uint32_t layer_count, VkFormat view_format);
// The format colour attachments of this texture are rendered through (integer 32-bit formats as their float alias).
VkFormat tex_attachment_format(Texture* t, VkFormat view_format);
// The image can be bound as a storage image (created with NVN_TEXTURE_FLAGS_IMAGE).
bool tex_has_storage(Texture* t);
VkFormat tex_storage_format(Texture* t, VkFormat requested);   // the view format a storage binding uses: the UNORM partner of an sRGB image
// The layout a texture is sampled in: SHADER_READ_ONLY_OPTIMAL, or DEPTH_STENCIL_READ_ONLY_OPTIMAL for depth and stencil formats (so
// the same layout serves a depth texture that is also the read-only depth attachment of the pass).
VkImageLayout tex_sampled_layout(Texture* t);
// A BUFFER-target texture as a texel buffer view over the pool buffer that holds its texels (the VkBuffer storage buffers and CPU writes
// use, so the view always sees current data). `view` = the texture pool entry's view (format) or null; component = the shader's sampled
// type (0 any, 1 float, 2 sint, 3 uint: a float read of a 32-bit integer format reads the raw bits); shader_format = the storage image
// format the SPIR-V declares (UNDEFINED: the texture's own). Null when impossible (the caller logs and leaves the dummy).
// *gpu_address / *bytes receive the range viewed (for gpu_wrote). Views are kept until their pool's buffer is destroyed.
VkBufferView tex_buffer_view(Texture* t, const TextureViewInfo* view, int component, VkFormat shader_format, bool storage,
                             uint64_t* gpu_address = nullptr, uint64_t* bytes = nullptr, VkFormat* format = nullptr);
// make_buffer_view: a cached texel buffer view (null when Vulkan refuses it, remembered).
VkBufferView make_buffer_view(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range, VkFormat format);
// forget_buffer_views: drop (and destroy once the GPU is done) the cached views of a buffer about to be destroyed.
void forget_buffer_views(VkBuffer buffer);
// Destroys the texel buffer views over `buffer` (pool_gpu_destroy, before the buffer goes).
void buffer_views_forget(VkBuffer buffer);
// A texel buffer view of `format` over the zeroed dummy buffer, for texel buffer descriptors that cannot be resolved.
VkBufferView dummy_buffer_view(VkFormat format, bool storage);
// A 1x1 storage image view in GENERAL layout for descriptors that cannot be resolved (UNDEFINED format: RGBA8 UNORM).
VkImageView dummy_storage_view(VkImageViewType type, VkFormat format);
// A UNORM (non-sRGB) view of the whole texture for the present pass.
VkImageView tex_present_view(Texture* t);
// Dummy 1x1 views used when a binding cannot be resolved: black texture of the requested type.
VkImageView dummy_view(VkImageViewType type, bool depth_compare);
VkImageView dummy_view(VkImageViewType type, bool depth_compare, int component);
VkSampler dummy_sampler();
// Sampler for the NVN sampler state (cached forever). Never null.
VkSampler sampler_for(const SamplerInfo& s);
// Buffer <-> image copies with the NVN conventions (used by the replay for CopyBufferToTexture etc.).
// Debug: the content of level 0 / layer 0 of a texture the GPU has written, as a PNG (HDR values squashed with x / (1 + x), integer
// formats show their low bytes as colour) and its per-channel statistics. Render thread; waits for the GPU. False: not dumpable.
bool dump_texture(Texture* t, const std::string& png_path, std::string* stats);
// Debug: records, into the frame being replayed, a copy of level 0 / layer 0 of the texture (no pass may be open) and keeps it until
// snapshot_write(): returns an index (-1: not copyable). snapshot_write writes it as dump_texture does and frees the buffer.
int snapshot_record(Texture* t, Frame& f);
bool snapshot_write(int index, const std::string& png_path, std::string* stats);
// [nvn] trace_dump_tex = <dir>: writes the data prepare_texture builds for t (raw blocks of every level and layer) as t<id>.bin with a t<id>.txt describing it, once per texture.
void trace_dump_prepared(Texture* t, const std::string& dir);
int snapshot_record_raw(Texture* t, Frame& f);   // block-compressed level 0 as the GPU holds it; snapshot_write writes the bytes
void tex_destroy(TexGpu* g);      // render thread
void tex_shutdown();              // frees samplers and dummy resources (device idle)
// Image formats chosen: bool descriptor for tests and logs
struct TexStats { uint64_t images = 0, uploads = 0, upload_bytes = 0, transcodes = 0, failed = 0; };
TexStats tex_stats();

// ---- replay (nvn_replay.cpp) ----------------------------------------------------------------------------------------------
// Called by nvn_gpu.cpp inside a submission: begin_submit resets per-submit tracking, replay_lists interprets the lists in
// order (state persists across lists and submissions), end_submit ends the open render pass and emits the closing barrier.
void replay_begin_submit(Frame& f);
void replay_lists(Frame& f, const std::vector<Ref<CmdList>>& lists);
void replay_end_submit(Frame& f);
void replay_abort_submit();     // after a recording threw (its command buffer is discarded)
void replay_request_trace();      // trace the next whole presented frame (any thread)
void replay_shutdown();           // destroys pipelines and layouts (render thread, device idle)

// ---- shared facts (nvn_gpu.cpp) ----------------------------------------------------------------------------------------
int window_origin_mode();
int depth_mode();
gfx::Renderer* renderer();
FrameLimiter* limiter();
PresentPass* present_pass();
extras::OutputPass* output_pass();
bool keep_aspect();

// After a traced frame ([nvn] trace_frame): dumps the targets it drew to (PNG next to the log: trace_<texture id>_<format>.png) and logs their statistics.
void trace_dump_targets();

// Coherent copy fallback for rejected host imports. Completion includes copy-back to guest memory.
void pool_copy_retire(PoolGpu*);
bool pool_copy_pending();
void pool_copy_begin();
void pool_copy_end();

// ---- present -----------------------------------------------------------------------------------------------------------
void present_on_render_thread(Window* w, int index, bool again = false);   // again: present_again (not counted as a game present)
void present_again();

}  // namespace nxrt::nvn::be::gpu
