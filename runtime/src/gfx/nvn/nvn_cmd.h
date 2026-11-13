// SPDX-License-Identifier: GPL-3.0-or-later
// Recorded commands (S4d). A command buffer appends records to a CmdList; nvnCommandBufferEndRecording turns it into
// an NVNcommandHandle that the game submits (any number of times) or calls from another list; nvnQueueSubmitCommands replays
// the lists on the render thread (nvn_replay.cpp).
//
// A record is { u16 op, u16 flags, u32 size } + a payload struct (8-byte aligned) + an optional blob (8-byte aligned);
// `size` covers all of it. Pointers to host objects in a payload (Texture*, Program*, pools ...) are strong references: the
// recorder passes them to CmdList::hold(), so they stay valid while the list lives. State objects (blend, depth, ...) are
// copied by value (raw bytes of the guest object), because the game reuses its structs right after the call. Memory
// addresses (NVNbufferAddress) are stored as given; the replay resolves them against the pool registry at submit time.
#pragma once
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#include "nvn_core.h"

namespace nxrt::nvn::be {

enum class Op : uint16_t {
    Nop = 0,
    // bindings
    BindProgram, BindBuffers, BindTextures, BindImages, BindVertexAttribs, BindVertexStreams,
    BindBlend, BindColor, BindChannelMask, BindDepthStencil, BindPolygon, BindMultisample,
    SetTexturePool, SetSamplerPool,
    // fixed-function state
    SetViewports, SetScissors, SetDepthRanges, SetDepthBounds, SetStencil, SetBlendColor, SetPointSize, SetLineWidth,
    SetPolygonOffsetClamp, SetAlphaRef, SetSampleMask, SetRasterizerDiscard, SetDepthClamp, SetPrimitiveRestart,
    SetPatchSize, SetTessLevels, SetRenderEnable, SetRenderTargets, SetCoverageModulationTable, SetConservativeRaster,
    // work
    DrawArrays, DrawElements, DrawIndirect, DrawTexture, ClearColor, ClearDepthStencil, ClearTexture, ClearBuffer,
    CopyBufferToBuffer, CopyBufferToTexture, CopyTextureToBuffer, CopyTextureToTexture, Downsample, UpdateUniformBuffer,
    Dispatch, DispatchIndirect, ReportCounter, ReportValue, ResetCounter, Barrier, FenceSync, WaitSync, SignalEvent, WaitEvent,
    CallCommands, DiscardColor, DiscardDepthStencil, ResolveDepth,
    BeginXfb, EndXfb, PauseXfb, ResumeXfb,            // transform feedback: P_Counter (address = the control buffer)
    DrawXfb,                                          // DrawTransformFeedback: P_DrawXfb (draws what the control buffer's feedback section wrote)
    PushDebugGroup, PopDebugGroup,                    // debug labels, recorded only when the back end has VK_EXT_debug_utils: P_DebugGroup (+ blob: the name, NUL ended)
    EvaluateDlss,                                     // nvnCommandBufferEvaluateDlss: P_EvaluateDlss (api_dlss.h; replayed by api_dlss_replay.cpp)
    PushConstants,                                    // NVN2 nvnCommandBufferPush{Graphics,Compute}Constants[Deferred]: P_PushConstants (+ blob: the data)
    Count
};
const char* op_name(Op op);

// ---- payloads (all trivially copyable) ---------------------------------------------------------------------------------------
struct P_BindProgram { Program* program; uint32_t stages; uint32_t pad; };                  // stages: NVN shader stage bits
enum BufferKind : uint32_t { kBufVertex = 0, kBufUniform = 1, kBufStorage = 2, kBufXfb = 3 };
struct P_BindBuffers { uint32_t kind, stage, first, count; };                                 // blob: NVNbufferRange[count]
struct P_BindTextures { uint32_t stage, first, count, pad; };                                 // blob: u64 handles[count]
struct P_BindVertexAttribs { uint32_t count, pad; };                                           // blob: u32 raw NVNvertexAttribState[count]
struct P_BindVertexStreams { uint32_t count, pad; };                                           // blob: 8-byte raw NVNvertexStreamState[count]
struct P_RawState { uint8_t bytes[0x18]; uint32_t size, pad; };                               // BindBlend/Color/ChannelMask/DepthStencil/Polygon/Multisample
struct P_Pool { void* pool; };                                                                 // SetTexturePool / SetSamplerPool: TexturePool* / SamplerPool* (null: none)
struct P_Rects { uint32_t first, count; };                                                    // SetViewports (blob float x,y,w,h per viewport) / SetScissors (blob i32 x,y,w,h)
struct P_DepthRanges { uint32_t first, count; };                                              // blob: float near,far per viewport
struct P_DepthBounds { uint32_t enable; float near_, far_; uint32_t pad; };
struct P_Stencil { uint32_t what; uint32_t face; uint32_t value; uint32_t pad; };              // what: 0 ref, 1 value mask, 2 write mask
struct P_BlendColor { float c[4]; };
struct P_Float { float v; uint32_t pad; };                                                    // SetPointSize, SetLineWidth, SetAlphaRef
struct P_PolygonOffsetClamp { float factor, units, clamp; uint32_t pad; };
struct P_U32 { uint32_t v; uint32_t pad; };                                                   // SetSampleMask, SetRasterizerDiscard, SetDepthClamp, SetPatchSize, SetRenderEnable, SetConservativeRaster
struct P_PrimitiveRestart { uint32_t enable; uint32_t index; };
struct P_TessLevels { uint32_t outer; float v[4]; };                                          // outer: 1 = outer levels (4 floats), 0 = inner (2 floats)
struct P_RenderEnableCond { uint32_t mode; uint32_t pad; uint64_t address; };
struct TargetRef {
    Texture* tex;                 // null: no target
    TextureViewInfo view;         // valid if has_view
    uint32_t has_view;
    uint32_t pad;
};
struct P_RenderTargets { uint32_t count; uint32_t pad; TargetRef color[8]; TargetRef depth; };

struct P_DrawArrays { uint32_t mode, first, count, base_instance, instance_count, pad; };
struct P_DrawElements { uint32_t mode, type, count, instance_count; int32_t base_vertex; uint32_t base_instance; uint64_t index_address; };
struct P_DrawIndirect {
    uint32_t mode, type;            // type: ~0u for DrawArrays
    uint32_t multi;                 // 0 single draw, 1 multi-draw with a count buffer
    uint32_t max_count;
    int64_t stride;
    uint64_t index_address, indirect_address, count_address;
};
struct P_DrawTexture { uint64_t handle; float dst[4]; float src[4]; };                       // x0,y0,x1,y1 regions

struct P_ClearColor { uint32_t target, kind, mask, pad; uint32_t value[4]; };                  // kind: 0 float, 1 int, 2 uint (value holds the 32-bit words)
struct P_ClearDepthStencil { float depth; uint32_t depth_mask; int32_t stencil; uint32_t stencil_mask; };
struct P_CopyRegion { int32_t x, y, z, w, h, d; };
struct P_ClearTexture { Texture* tex; TextureViewInfo view; uint32_t has_view; uint32_t kind; P_CopyRegion region; uint32_t has_region; uint32_t mask; uint32_t value[4]; };
struct P_ClearBuffer { uint64_t address; uint64_t size; uint32_t value; uint32_t pad; };
struct P_CopyBufferToBuffer { uint64_t src, dst, size; uint32_t flags, pad; };
struct P_CopyBufTex {                                 // CopyBufferToTexture / CopyTextureToBuffer
    uint64_t address; Texture* tex; TextureViewInfo view; uint32_t has_view; uint32_t flags; P_CopyRegion region; uint32_t has_region; uint32_t pad;
    int64_t row_stride, image_stride;
};
struct P_CopyTexTex {
    Texture* src; TextureViewInfo src_view; uint32_t has_src_view; uint32_t flags; P_CopyRegion src_region; uint32_t has_src_region; uint32_t pad0;
    Texture* dst; TextureViewInfo dst_view; uint32_t has_dst_view; uint32_t pad1;       P_CopyRegion dst_region; uint32_t has_dst_region; uint32_t pad2;
};
struct P_Downsample { Texture* src; Texture* dst; TextureViewInfo src_view, dst_view; uint32_t has_views; uint32_t pad; };
struct P_UpdateUniformBuffer { uint64_t address; uint64_t buffer_size; int64_t offset; uint64_t size; };   // blob: size bytes
struct P_Dispatch { uint32_t x, y, z, pad; };
struct P_DispatchIndirect { uint64_t address; };
struct P_Counter { uint32_t type, pad; uint64_t address; };                                    // ReportCounter / ResetCounter(address 0)
struct P_ReportValue { uint32_t value, pad; uint64_t address; };
struct P_DrawXfb { uint32_t mode, pad; uint64_t control; };                                    // DrawTransformFeedback: the primitive and the feedback control buffer address
struct P_DebugGroup { uint32_t id, length; };                                                  // PushDebugGroup (blob: length + 1 bytes) / PopDebugGroup (length 0)
struct P_Barrier { uint32_t bits, pad; };
struct P_Sync { Sync* sync; uint32_t condition, flags; };                                      // FenceSync (condition, flags) / WaitSync
struct P_Event { Event* event; uint32_t mode, location, index, value, pad; };                 // SignalEvent: mode (NVNeventSignalMode), location, index, value; WaitEvent: mode = NVNeventWaitMode, value
struct P_CallCommands { uint32_t count, copy; };                                               // blob: u64 handles[count]
struct P_EvaluateDlss { Obj* eval; };                                                          // the be::DlssEval (held by the list)
// PushConstants: compute 0 graphics / 1 compute, slot 0..7, offset 0..255 and size 4..256 in bytes; blob: the data. address != 0 (the
// Deferred forms): no blob, the data is read at that GPU address when the command runs.
struct P_PushConstants { uint32_t compute, slot, offset, size; uint64_t address; };

// ---- lists --------------------------------------------------------------------------------------------------------------------
struct RecHeader { uint16_t op; uint16_t flags; uint32_t size; };

struct CommandBuffer;
// Console command / control memory model (api_cmdbuf.cpp): charges one record of the buffer being recorded, before it is
// appended. Caller holds the buffer's lock (or has exclusive use of it). Running out queues the game's memory callback, which
// runs when the NVN call returns (cmdmem_run_pending, from the thunk in nvn_abi.h).
void cmdmem_charge(CommandBuffer* cb, Op op, uint32_t blob_size);

class CmdList : public Obj {
public:
    static constexpr Kind kKind = Kind::CmdList;
    CmdList() : Obj(kKind) {}
    std::vector<uint8_t> bytes;
    std::vector<Ref<Obj>> held;           // objects the payloads point to
    uint64_t handle = 0;                  // NVNcommandHandle (0 until EndRecording)
    uint32_t record_count = 0;
    uint32_t draw_count = 0;
    uint64_t mem_start = 0, mem_end = 0;  // command memory the game believes this list occupies (GPU addresses)
    uint64_t hw_bytes = 0;                // command memory the console would have used for it (the model's estimate)
    CommandBuffer* meter = nullptr;       // the buffer recording it (charged per record), null once recorded and for host-made lists
    std::atomic<uint64_t> calls{0};

    // Appends one record. `blob` (may be null) is copied after the payload.
    template <class P>
    void emit(Op op, const P& payload, const void* blob = nullptr, uint32_t blob_size = 0) {
        static_assert(std::is_trivially_copyable_v<P>, "payloads are plain data");
        append(op, &payload, sizeof(P), blob, blob_size);
    }
    void emit_empty(Op op) { append(op, nullptr, 0, nullptr, 0); }
    void hold(Obj* o) { if (o) { o->ref(); held.emplace_back(o); } }
    void reset() { bytes.clear(); held.clear(); record_count = draw_count = 0; handle = 0; }

private:
    void append(Op op, const void* payload, uint32_t payload_size, const void* blob, uint32_t blob_size) {
        if (meter) cmdmem_charge(meter, op, blob_size);
        const uint32_t p = (payload_size + 7u) & ~7u, b = (blob_size + 7u) & ~7u;
        const uint32_t total = static_cast<uint32_t>(sizeof(RecHeader)) + p + b;
        const size_t at = bytes.size();
        bytes.resize(at + total);   // zero-initialised padding
        RecHeader h{static_cast<uint16_t>(op), 0, total};
        std::memcpy(&bytes[at], &h, sizeof h);
        if (payload_size) std::memcpy(&bytes[at + sizeof h], payload, payload_size);
        if (blob_size) std::memcpy(&bytes[at + sizeof h + p], blob, blob_size);
        ++record_count;
    }
};

// Walks the records of a list: for (RecordReader r(list); r.next(); ) { r.op(); r.payload<P>(); r.blob(); }
class RecordReader {
public:
    explicit RecordReader(const CmdList& l) : data_(l.bytes.data()), end_(l.bytes.data() + l.bytes.size()) {}
    bool next() {
        if (cur_ == nullptr) cur_ = data_; else cur_ += hdr().size;
        return cur_ < end_;
    }
    Op op() const { return static_cast<Op>(hdr().op); }
    template <class P>
    const P& payload() const { return *reinterpret_cast<const P*>(cur_ + sizeof(RecHeader)); }
    // The blob after a payload of type P.
    template <class P>
    const uint8_t* blob() const { return cur_ + sizeof(RecHeader) + ((sizeof(P) + 7u) & ~7u); }
    uint32_t size() const { return hdr().size; }
private:
    RecHeader hdr() const { RecHeader h; std::memcpy(&h, cur_, sizeof h); return h; }
    const uint8_t* data_;
    const uint8_t* end_;
    const uint8_t* cur_ = nullptr;
};

// ---- command buffers and handles ------------------------------------------------------------------------------------
struct CommandBuffer : Obj {
    static constexpr Kind kKind = Kind::CommandBuffer;
    CommandBuffer() : Obj(kKind) {}
    Ref<CmdList> rec;                 // the list being recorded (null when not recording)
    bool recording = false;
    // The sizes of this buffer's last list: the next list reserves them up front, so recording a big list does not reallocate and copy its
    // byte vector a dozen times as it grows (a fifth of the game thread's NVN recording time went to vector growth and its memcpy).
    size_t bytes_hint = 0, held_hint = 0;
    // NVN2 nvnCommandBufferSetRecordingMode (the sdk keeps it at cmdbuf+0x18). It picks how the driver encodes the commands (a per-mode
    // header word ahead of each packet; a mode-2 buffer drops Barrier), not what they do, so the replay only keeps the value.
    int32_t recording_mode = 0;
    TexturePool* tpool = nullptr;     // the texture pool set in the list being recorded (the list holds it); background texture preparation resolves handles with it
    int64_t copy_row_stride = 0, copy_image_stride = 0;
    uint64_t callback = 0, callback_data = 0;       // guest function pointer and its data
    uint64_t guest = 0;                              // the guest NVNcommandBuffer (first argument of the callback)
    // command / control memory the game added
    struct Block { uint64_t pool_gpu; uint64_t offset; uint64_t size; };
    std::vector<Block> command_blocks;
    // Console model (api_cmdbuf.cpp): positions in the command block and the control memory added last, as the console's
    // write pointers would be, from per-record estimates of the command words and the control entries the sdk writes.
    uint64_t command_total = 0, control_total = 0;   // size of the command block / control memory added last
    uint64_t command_used = 0, control_used = 0;     // write positions in them
    uint64_t last_block_addr = 0;                    // GPU address of the command memory added last
    uint64_t seg_start = 0;                          // command_used at the last control entry (the sdk's command segment start)
    bool ctl_group = false, seg_contig = false;      // a control group is open / the next segment continues the last one
    bool cmd_full = false, ctl_full = false;         // ran out and nothing was added: lists go to the overflow ring
    uint64_t cmd_serial = 0, ctl_serial = 0;         // AddCommandMemory / AddControlMemory calls
    uint64_t pend_cmd = 0, pend_ctl = 0;             // bytes charged while the callback is queued (they go in the memory it adds)
    uint64_t rec_start = 0, rec_block_end = 0, rec_serial = 0;   // where the current recording began
    bool rec_unbacked = false;                       // the recording began without room in the game's command memory
    // Overflow memory: a recording that finds no room left in the command memory the game added (the estimate reached the
    // block's end and the game added nothing since, because on the console the out-of-memory callback would hand over more)
    // is placed in a ring of this buffer's own made-up addresses (kOverflowAddrBase), so the overlap rule still ends it.
    uint64_t overflow_base = 0, overflow_pos = 0;
    std::vector<uint64_t> handles;                   // handles of the lists recorded here (released at Finalize)
    Device* device = nullptr;
    std::string label;
    std::mutex lock;
};

// Made-up command memory addresses of the overflow rings (above any GPU address a pool can have; one 64 GiB slot per buffer).
constexpr uint64_t kOverflowAddrBase = 0xFF00000000000000ull;
inline bool is_overflow_addr(uint64_t a) { return a >= kOverflowAddrBase; }

// Handle table: handle -> list. Lists live until the game records over their command memory, the memory pool goes away, or
// the command buffer is finalized (nvn_cmd_handles.cpp).
uint64_t handle_register(Ref<CmdList> list);
Ref<CmdList> handle_lookup(uint64_t handle);
void handle_release(uint64_t handle);            // the game finalized the handle (nvnDeviceFinalizeCommandHandle)
void handle_release_unbacked(uint64_t handle);   // drops the handle only when its list has no region in the game's command memory
void handles_release_pool(const MemoryPool* pool);
size_t handles_live();
uint64_t handles_missing();      // lookups of unknown handles so far

}  // namespace nxrt::nvn::be
