// SPDX-License-Identifier: GPL-3.0-or-later
// NVNcommandBuffer lifecycle: initialize/finalize, command and control memory bookkeeping, Begin/EndRecording, handles,
// Call/CopyCommands. The recorders of the commands themselves are api_cmd_state.cpp and api_cmd_work.cpp.
//
// Command memory model. The game hands over memory (AddCommandMemory / AddControlMemory), reads how much is used and free, and
// registers a memory callback that the console's driver calls when a recording runs out: the game then adds the next chunk
// itself (Pokopia moves its own command ring there). The back end records into host lists, so the console's usage is modelled:
//   command memory: every record costs an estimate of the command words it would write (8 bytes plus its inline data, rounded
//     to the 4-byte command alignment), charged before it is written. The estimate stays below the console's, so the callback
//     never fires where the console's would not (several of Pokopia's buffers have a callback whose handler is null).
//   control memory: the entries the sdk writes (sdk 0x5C4D20 BeginRecording, 0x5C4DE0 EndRecording, 0x5C49C0 AddCommandMemory,
//     0x5C63F8 FenceSync, 0x5C6640 WaitSync, 0x5C5740 CallCommands): BeginRecording 16 bytes, EndRecording 8, a command segment
//     32 bytes when it opens a group, 24 more in an open group, nothing when it continues the last segment; FenceSync and WaitSync
//     16 bytes and they close the group. The last 16 bytes of the control memory are kept for the jump AddControlMemory writes.
// When a charge does not fit, the callback is queued: (cmdbuf, NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY or
// _CONTROL_MEMORY, minimum size, callback data) as in the sdk, the minimum being max(need, 512) for commands
// (NVN_DEVICE_INFO_COMMAND_BUFFER_MIN_COMMAND_SIZE) and max(need + 16, 256) for control. It runs on the recording thread when
// the NVN call returns (the thunk, nvn_abi.h), with no lock held, so the game's handler can add memory, query usage or wait.
// The charge then lands in the memory it added. A buffer without a callback, or whose callback added nothing, keeps the old
// fallback: its lists go in the buffer's overflow ring (CommandBuffer::overflow_base), so every list has a region and the
// overlap rule of nvn_cmd_handles.cpp ends it. [nvn] cmdmem_callback = false never calls the game and always falls back.
#include <algorithm>
#include <atomic>

#include "nvn_cmd.h"
#include "nvn_impl.h"
#include "nxrt/config.h"
#include "nxrt/dispatch.h"
#include "nxrt/gauge.h"
#include "nxrt/log.h"

namespace nxrt::nvn {
using namespace be;
namespace {

constexpr uint64_t kCtlJumpReserve = 16;               // control bytes AddControlMemory keeps for its jump entry
constexpr uint64_t kMinCommandSize = 512, kMinControlSize = 256;
constexpr int kEventCommand = 0, kEventControl = 1;     // NVNcommandBufferMemoryEvent

CommandBuffer* cb_of(NVNcommandBuffer* c) { return host_of<CommandBuffer>(c); }
const CommandBuffer* cb_of(const NVNcommandBuffer* c) { return host_of<CommandBuffer>(c); }

// Live counts (nxrt/gauge.h). A command buffer counts from its Initialize to its Finalize. Command memory counts the bytes the game handed over
// and that are still in the buffer's block list: AddCommandMemory adds a block, the oldest one is dropped past 8 blocks, and leaving the
// buffer (Finalize, or Initialize over it) drops the rest.
nxrt::GaugeCounter g_command_buffers("nvn.command_buffers");
nxrt::GaugeCounter g_command_memory("nvn.command_memory.bytes");
std::atomic<uint64_t> g_overflow_serial{0};   // numbers the overflow rings (CommandBuffer::overflow_base)
std::atomic<uint64_t> g_cb_calls{0}, g_cb_added{0}, g_cb_fallbacks{0};   // callbacks run, that added memory, run-outs that fell back

struct CmdMemGauges {
    CmdMemGauges() {
        nxrt::gauge_register("nvn.cmdmem_callbacks", [] { return static_cast<int64_t>(g_cb_calls.load(std::memory_order_relaxed)); });
        nxrt::gauge_register("nvn.cmdmem_callbacks_added", [] { return static_cast<int64_t>(g_cb_added.load(std::memory_order_relaxed)); });
        nxrt::gauge_register("nvn.cmdmem_fallbacks", [] { return static_cast<int64_t>(g_cb_fallbacks.load(std::memory_order_relaxed)); });
    }
    ~CmdMemGauges() {
        nxrt::gauge_unregister("nvn.cmdmem_callbacks");
        nxrt::gauge_unregister("nvn.cmdmem_callbacks_added");
        nxrt::gauge_unregister("nvn.cmdmem_fallbacks");
    }
};
CmdMemGauges g_cmdmem_gauges;

bool callback_enabled() {
    static const bool on = config_get_bool("nvn", "cmdmem_callback", true);
    return on;
}

// The queued callback of this thread (one buffer at a time: the thunk runs it when the NVN call returns).
struct Pending {
    CommandBuffer* cb = nullptr;
    bool cmd = false, ctl = false;
    uint64_t cmd_min = 0, ctl_min = 0;
};
thread_local Pending t_pending;
// In the guest's command-memory callback: the flag lives in nxrt::host_callback_flags() so a guest exception that unwinds out of the
// callback (a longjmp past CallbackScope) does not leave it set: catch_call.cpp restores it at the landing.
bool in_callback() { return (nxrt::host_callback_flags() & nxrt::kHostCbCmdMem) != 0; }
struct CallbackScope {
    uint32_t saved = nxrt::host_callback_flags();
    CallbackScope() { nxrt::host_callback_restore(saved | nxrt::kHostCbCmdMem); }
    ~CallbackScope() { nxrt::host_callback_restore(saved); }
    CallbackScope(const CallbackScope&) = delete;
    CallbackScope& operator=(const CallbackScope&) = delete;
};

// A run-out with no callback to call: the recording continues in the overflow ring (EndRecording).
void fall_back(CommandBuffer* cb, int event) {
    if (event == kEventCommand) { cb->cmd_full = true; cb->pend_cmd = 0; }
    else { cb->ctl_full = true; cb->pend_ctl = 0; }
    g_cb_fallbacks.fetch_add(1, std::memory_order_relaxed);
}

// Queues the callback for `event`. Caller holds the buffer's lock (or has exclusive use of it).
void request(CommandBuffer* cb, int event, uint64_t min_size) {
    if (!cb->callback || !cb->guest || !callback_enabled() || in_callback() || (t_pending.cb && t_pending.cb != cb)) {
        fall_back(cb, event);
        return;
    }
    t_pending.cb = cb;
    if (event == kEventCommand) { if (!t_pending.cmd) t_pending.cmd_min = min_size; t_pending.cmd = true; }
    else { if (!t_pending.ctl) t_pending.ctl_min = min_size; t_pending.ctl = true; }
    abi::t_cmdmem_pending = true;
}

uint64_t control_end(const CommandBuffer* cb) { return cb->control_total > kCtlJumpReserve ? cb->control_total - kCtlJumpReserve : 0; }

void ctl_need(CommandBuffer* cb, uint64_t n) {
    if (!cb->control_total || cb->ctl_full || !n) return;            // no control memory added: nothing to model
    if (cb->pend_ctl || cb->control_used + n > control_end(cb)) {
        cb->pend_ctl += n;
        request(cb, kEventControl, std::max<uint64_t>(n + kCtlJumpReserve, kMinControlSize));
        return;
    }
    cb->control_used += n;
}

void cmd_need(CommandBuffer* cb, uint64_t n) {
    if (!cb->last_block_addr || cb->cmd_full || !n) return;          // no command memory added: the list goes in the ring
    if (cb->pend_cmd || cb->command_used + n > cb->command_total) {
        cb->pend_cmd += n;
        request(cb, kEventCommand, std::max<uint64_t>(n, kMinCommandSize));
        return;
    }
    cb->command_used += n;
}

// The control entry for the commands written since the last one (the sdk's segment flush).
void flush_segment(CommandBuffer* cb) {
    if (cb->command_used == cb->seg_start) return;
    const uint64_t n = cb->ctl_group ? (cb->seg_contig ? 0 : 24) : 32;
    cb->seg_start = cb->command_used;
    cb->ctl_group = true;
    cb->seg_contig = true;
    ctl_need(cb, n);
}

// Command words the console would write for one record: a lower bound (see the top of the file).
uint64_t record_cost(Op op, uint32_t blob_size) {
    switch (op) {
    case Op::Nop: case Op::CallCommands: return 0;   // a call is a control entry (small lists are copied in: not counted)
    case Op::PushDebugGroup: case Op::PopDebugGroup: return 8;
    default: return 8 + ((static_cast<uint64_t>(blob_size) + 3u) & ~3ull);
    }
}

// Logs the first callback of each buffer family (callback, event, label).
void note_family(const CommandBuffer* cb, uint64_t callback, int event, uint64_t min_size, uint64_t lr) {
    static std::mutex mu;
    static std::vector<std::string> seen;
    const std::string key = std::to_string(callback) + "/" + std::to_string(event) + "/" + cb->label;
    {
        std::lock_guard<std::mutex> lk(mu);
        if (seen.size() >= 64 || std::find(seen.begin(), seen.end(), key) != seen.end()) return;
        seen.push_back(key);
    }
    NXLOG_I("nvn", "command buffer memory callback (first for this family): callback 0x%llx data 0x%llx, %s, min %llu, cmdbuf 0x%llx \"%s\", "
            "recorded from 0x%llx; command block %llu bytes, control %llu bytes",
            static_cast<unsigned long long>(callback), static_cast<unsigned long long>(cb->callback_data),
            event == kEventCommand ? "out of command memory" : "out of control memory", static_cast<unsigned long long>(min_size),
            static_cast<unsigned long long>(cb->guest), cb->label.c_str(), static_cast<unsigned long long>(lr),
            static_cast<unsigned long long>(cb->command_total), static_cast<unsigned long long>(cb->control_total));
}

// Calls a guest function with four arguments from inside a thunk; the thunk's registers (its result in x0) are kept.
void gcall4(::Ctx* c, uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    uint64_t save[9];
    std::memcpy(save, c->x, sizeof save);
    const uint64_t lr = c->x[30], f = c->fn;
    c->x[0] = a0; c->x[1] = a1; c->x[2] = a2; c->x[3] = a3;
    dispatch_call(c, fn);
    std::memcpy(c->x, save, sizeof save);
    c->x[30] = lr;
    c->fn = f;
}

// Caller holds the buffer's lock (or has exclusive use of it).
void command_buffer_gone(const CommandBuffer* cb) {
    g_command_buffers.sub(1);
    for (const CommandBuffer::Block& b : cb->command_blocks) g_command_memory.sub(static_cast<int64_t>(b.size));
}

NVNboolean nvnCommandBufferInitialize(NVNcommandBuffer* c, NVNdevice* device) {
    if (CommandBuffer* old = cb_of(c)) command_buffer_gone(old);   // initialised over a live one: the old one is replaced
    auto* cb = new CommandBuffer();
    cb->device = host_of<Device>(device);
    cb->guest = reinterpret_cast<uint64_t>(c);
    bind_guest(c, cb);
    g_command_buffers.add(1);
    return 1;
}

void nvnCommandBufferFinalize(NVNcommandBuffer* c) {
    CommandBuffer* cb = cb_of(c);
    if (!cb) return;
    {
        std::lock_guard<std::mutex> lk(cb->lock);
        for (uint64_t h : cb->handles) handle_release_unbacked(h);   // lists in command memory outlive the buffer (nvn_cmd_handles.cpp)
        cb->handles.clear();
        command_buffer_gone(cb);
    }
    unbind_guest(c);
}

void nvnCommandBufferSetDebugLabel(NVNcommandBuffer* c, const char* label) {
    if (CommandBuffer* cb = cb_of(c)) cb->label = label ? label : "";
}

// NVN2 only (not in the NVN1 table): nvnCommandBufferSetRecordingMode(cmd, mode). The sdk's body is one store, mode -> cmdbuf+0x18 (the game passes
// 0, 1 and 2). What reads it (nvnCommandBufferBarrier, BindProgram, BeginDeviceCommands, ... in the sdk): every command writer takes the word
// from a 5-entry table indexed by the mode and puts it in front of the packet it writes (the command stream's engine or queue header), and
// nvnCommandBufferBarrier does nothing at all in mode 2. So the mode changes how the driver encodes the stream, not what the commands do;
// Barrier in mode 2 is the one visible difference and the replay keeps the barrier (an extra barrier is never wrong). The value is
// kept on the buffer; each mode is logged the first time it is set, so a run shows which ones the game uses.
void nvnCommandBufferSetRecordingMode(NVNcommandBuffer* c, int mode) {
    CommandBuffer* cb = cb_of(c);
    if (!cb) return;
    {
        std::lock_guard<std::mutex> lk(cb->lock);
        cb->recording_mode = mode;
    }
    static std::atomic<uint32_t> seen{0};
    const uint32_t bit = 1u << (static_cast<uint32_t>(mode) & 31u);
    if (!(seen.fetch_or(bit) & bit)) NXLOG_I("nvn", "nvnCommandBufferSetRecordingMode(%d): kept on the buffer, the replay records the same commands in every mode (first)", mode);
}

void nvnCommandBufferSetMemoryCallback(NVNcommandBuffer* c, PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC cb_fn) {
    if (CommandBuffer* cb = cb_of(c)) cb->callback = reinterpret_cast<uint64_t>(cb_fn);
}
void nvnCommandBufferSetMemoryCallbackData(NVNcommandBuffer* c, void* data) {
    if (CommandBuffer* cb = cb_of(c)) cb->callback_data = reinterpret_cast<uint64_t>(data);
}
PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC nvnCommandBufferGetMemoryCallback(const NVNcommandBuffer* c) {
    const CommandBuffer* cb = cb_of(c);
    return cb ? reinterpret_cast<PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC>(cb->callback) : nullptr;
}
void* nvnCommandBufferGetMemoryCallbackData(const NVNcommandBuffer* c) {
    const CommandBuffer* cb = cb_of(c);
    return cb ? reinterpret_cast<void*>(cb->callback_data) : nullptr;
}

void nvnCommandBufferAddCommandMemory(NVNcommandBuffer* c, const NVNmemoryPool* pool, ptrdiff_t offset, size_t size) {
    CommandBuffer* cb = cb_of(c);
    if (!cb) return;
    const MemoryPool* mp = host_of<MemoryPool>(pool);
    std::lock_guard<std::mutex> lk(cb->lock);
    const uint64_t addr = mp ? mp->gpu_base + static_cast<uint64_t>(offset) : static_cast<uint64_t>(offset);
    cb->command_blocks.push_back({addr, static_cast<uint64_t>(offset), size});
    g_command_memory.add(static_cast<int64_t>(size));
    if (cb->command_blocks.size() > 8) {
        g_command_memory.sub(static_cast<int64_t>(cb->command_blocks.front().size));
        cb->command_blocks.erase(cb->command_blocks.begin());
    }
    if (cb->recording) flush_segment(cb);   // the commands written so far in the old block get their control entry
    cb->command_total = size;
    cb->command_used = 0;
    cb->seg_start = 0;
    cb->seg_contig = false;
    cb->cmd_full = false;
    cb->last_block_addr = addr;
    ++cb->cmd_serial;
}

void nvnCommandBufferAddControlMemory(NVNcommandBuffer* c, void* memory, size_t size) {
    CommandBuffer* cb = cb_of(c);
    if (!cb) return;
    std::lock_guard<std::mutex> lk(cb->lock);
    (void)memory;
    cb->control_total = size;
    cb->control_used = 0;
    cb->ctl_full = false;
    if (cb->recording) cb->ctl_group = false;   // the jump entry ends the group
    ++cb->ctl_serial;
}

size_t nvnCommandBufferGetCommandMemorySize(const NVNcommandBuffer* c) { const CommandBuffer* cb = cb_of(c); return cb ? cb->command_total : 0; }
size_t nvnCommandBufferGetCommandMemoryUsed(const NVNcommandBuffer* c) { const CommandBuffer* cb = cb_of(c); return cb ? cb->command_used : 0; }
size_t nvnCommandBufferGetCommandMemoryFree(const NVNcommandBuffer* c) { const CommandBuffer* cb = cb_of(c); return cb ? cb->command_total - std::min(cb->command_used, cb->command_total) : 0; }
size_t nvnCommandBufferGetControlMemorySize(const NVNcommandBuffer* c) { const CommandBuffer* cb = cb_of(c); return cb ? cb->control_total : 0; }
size_t nvnCommandBufferGetControlMemoryUsed(const NVNcommandBuffer* c) { const CommandBuffer* cb = cb_of(c); return cb ? cb->control_used : 0; }
size_t nvnCommandBufferGetControlMemoryFree(const NVNcommandBuffer* c) { const CommandBuffer* cb = cb_of(c); return cb ? cb->control_total - std::min(cb->control_used, cb->control_total) : 0; }

}  // namespace

// Charges one record (nvn_cmd.h).
void be::cmdmem_charge(CommandBuffer* cb, Op op, uint32_t blob_size) {
    switch (op) {
    case Op::FenceSync: case Op::WaitSync:
        flush_segment(cb);
        ctl_need(cb, 16);
        cb->ctl_group = false;
        break;
    case Op::CallCommands:
        flush_segment(cb);
        break;
    default: break;
    }
    const uint64_t n = record_cost(op, blob_size);
    cmd_need(cb, n);
    if (cb->rec) cb->rec->hw_bytes += n;
}

// Runs the callback this thread queued (from the thunk, after the NVN call, no lock held).
void abi::cmdmem_run_pending(::Ctx* c) {
    abi::t_cmdmem_pending = false;
    const Pending p = t_pending;
    t_pending = Pending{};
    if (!p.cb) return;
    Ref<CommandBuffer> hold = Ref<CommandBuffer>::share(p.cb);   // the handler may finalize the buffer
    CommandBuffer* cb = p.cb;
    for (int event : {kEventCommand, kEventControl}) {
        if (event == kEventCommand ? !p.cmd : !p.ctl) continue;
        uint64_t fn = 0, data = 0, serial = 0;
        {
            std::lock_guard<std::mutex> lk(cb->lock);
            fn = cb->callback;
            data = cb->callback_data;
            serial = event == kEventCommand ? cb->cmd_serial : cb->ctl_serial;
            if (!fn || (event == kEventCommand ? !cb->pend_cmd : !cb->pend_ctl)) continue;
        }
        const uint64_t min_size = event == kEventCommand ? p.cmd_min : p.ctl_min;
        note_family(cb, fn, event, min_size, c->x[30]);
        g_cb_calls.fetch_add(1, std::memory_order_relaxed);
        {
            const CallbackScope scope;
            gcall4(c, fn, cb->guest, static_cast<uint64_t>(event), min_size, data);
        }
        std::lock_guard<std::mutex> lk(cb->lock);
        if (event == kEventCommand) {
            const uint64_t n = cb->pend_cmd;
            cb->pend_cmd = 0;
            if (cb->cmd_serial != serial && cb->command_used + n <= cb->command_total) {
                cb->command_used += n;   // the commands that did not fit are in the new block
                g_cb_added.fetch_add(1, std::memory_order_relaxed);
            } else {
                fall_back(cb, event);
            }
        } else {
            const uint64_t n = cb->pend_ctl;
            cb->pend_ctl = 0;
            if (cb->ctl_serial != serial && cb->control_used + n <= control_end(cb)) {
                cb->control_used += n;
                g_cb_added.fetch_add(1, std::memory_order_relaxed);
            } else {
                fall_back(cb, event);
            }
        }
    }
}

namespace {

NVNboolean nvnCommandBufferIsRecording(const NVNcommandBuffer* c) {
    const CommandBuffer* cb = cb_of(c);
    return cb && cb->recording ? 1 : 0;
}

void nvnCommandBufferBeginRecording(NVNcommandBuffer* c) {
    CommandBuffer* cb = cb_of(c);
    if (!cb) return;
    std::lock_guard<std::mutex> lk(cb->lock);
    cb->rec = Ref<CmdList>(new CmdList());
    cb->rec->meter = cb;
    if (cb->bytes_hint) cb->rec->bytes.reserve(cb->bytes_hint + cb->bytes_hint / 8);
    if (cb->held_hint) cb->rec->held.reserve(cb->held_hint + cb->held_hint / 8);
    cb->recording = true;
    cb->tpool = nullptr;
    cb->ctl_group = false;
    cb->seg_start = cb->command_used;
    cb->rec_start = cb->last_block_addr + cb->command_used;
    cb->rec_block_end = cb->last_block_addr + cb->command_total;
    cb->rec_serial = cb->cmd_serial;
    cb->rec_unbacked = !cb->last_block_addr || cb->cmd_full;
    ctl_need(cb, 16);
}

NVNcommandHandle nvnCommandBufferEndRecording(NVNcommandBuffer* c) {
    CommandBuffer* cb = cb_of(c);
    if (!cb) return 0;
    Ref<CmdList> list;
    {
        std::lock_guard<std::mutex> lk(cb->lock);
        if (!cb->recording || !cb->rec) return 0;
        list = std::move(cb->rec);
        cb->recording = false;
        // the next list of this buffer starts with room for one like this; a much smaller one lets the hint decay (lists of one buffer vary)
        constexpr size_t kHintMax = 64u << 20;
        cb->bytes_hint = std::min(kHintMax, std::max(list->bytes.size(), cb->bytes_hint - cb->bytes_hint / 4));
        cb->held_hint = std::min<size_t>(1u << 20, std::max(list->held.size(), cb->held_hint - cb->held_hint / 4));
        cb->tpool = nullptr;
        list->meter = nullptr;
        flush_segment(cb);
        ctl_need(cb, 8);
        cb->ctl_group = false;
        const uint64_t want = std::max<uint64_t>(list->hw_bytes, 1);
        if (!cb->rec_unbacked && !cb->cmd_full && cb->last_block_addr) {
            // in the memory the game added: from where the recording began to where it ended, or to the end of that block
            // when the callback moved it to another one (the game reuses the older block first). An empty list takes one
            // byte (its handle still names control memory), which the next recording, starting at the same place, overlaps.
            const bool moved = cb->rec_serial != cb->cmd_serial;
            if (moved && cb->rec_start < cb->rec_block_end) {
                list->mem_start = cb->rec_start;
                list->mem_end = cb->rec_block_end;
            } else {
                list->mem_start = moved ? cb->last_block_addr : cb->rec_start;
                list->mem_end = std::max(cb->last_block_addr + cb->command_used, list->mem_start + 1);
            }
        } else {
            // no room left (or none added) and no memory from the callback: the list goes in the buffer's overflow ring, as
            // large as the last block (at least 1 MiB), oldest first.
            const uint64_t ring = std::max<uint64_t>(cb->command_total, 1ull << 20);
            const uint64_t size = std::clamp<uint64_t>(want, 1, ring);
            if (!cb->overflow_base) cb->overflow_base = kOverflowAddrBase + ((g_overflow_serial.fetch_add(1) & 0xFFFFF) << 36);
            if (cb->overflow_pos + size > ring) cb->overflow_pos = 0;
            list->mem_start = cb->overflow_base + cb->overflow_pos;
            list->mem_end = list->mem_start + size;
            cb->overflow_pos += size;
        }
    }
    const uint64_t h = handle_register(list);
    std::lock_guard<std::mutex> lk(cb->lock);
    cb->handles.push_back(h);
    if (cb->handles.size() > 4096) cb->handles.erase(cb->handles.begin(), cb->handles.begin() + 2048);   // old ones are dropped by region anyway
    return h;
}

void nvnCommandBufferCallCommands(NVNcommandBuffer* c, int count, const NVNcommandHandle* handles) {
    CommandBuffer* cb = cb_of(c);
    if (!cb || count <= 0 || !handles) return;
    std::lock_guard<std::mutex> lk(cb->lock);
    if (!cb->recording || !cb->rec) return;
    P_CallCommands p{static_cast<uint32_t>(count), 0};
    cb->rec->emit(Op::CallCommands, p, handles, static_cast<uint32_t>(count) * 8u);
}

// CopyCommands: the other lists' records are copied into this one now (the console copies the command words).
void nvnCommandBufferCopyCommands(NVNcommandBuffer* c, int count, const NVNcommandHandle* handles) {
    CommandBuffer* cb = cb_of(c);
    if (!cb || count <= 0 || !handles) return;
    std::lock_guard<std::mutex> lk(cb->lock);
    if (!cb->recording || !cb->rec) return;
    for (int i = 0; i < count; ++i) {
        Ref<CmdList> src = handle_lookup(handles[i]);
        if (!src) continue;
        cmd_need(cb, src->hw_bytes);   // the console copies the command words
        cb->rec->hw_bytes += src->hw_bytes;
        cb->rec->bytes.insert(cb->rec->bytes.end(), src->bytes.begin(), src->bytes.end());
        cb->rec->record_count += src->record_count;
        cb->rec->draw_count += src->draw_count;
        for (auto& h : src->held) cb->rec->held.push_back(h);
    }
}

void nvnCommandBufferSetCopyRowStride(NVNcommandBuffer* c, ptrdiff_t stride) { if (CommandBuffer* cb = cb_of(c)) cb->copy_row_stride = stride; }
void nvnCommandBufferSetCopyImageStride(NVNcommandBuffer* c, ptrdiff_t stride) { if (CommandBuffer* cb = cb_of(c)) cb->copy_image_stride = stride; }
ptrdiff_t nvnCommandBufferGetCopyRowStride(const NVNcommandBuffer* c) { const CommandBuffer* cb = cb_of(c); return cb ? static_cast<ptrdiff_t>(cb->copy_row_stride) : 0; }
ptrdiff_t nvnCommandBufferGetCopyImageStride(const NVNcommandBuffer* c) { const CommandBuffer* cb = cb_of(c); return cb ? static_cast<ptrdiff_t>(cb->copy_image_stride) : 0; }

}  // namespace

void register_api_cmdbuf() {
    NVN_REG(nvnCommandBufferInitialize);
    NVN_REG(nvnCommandBufferFinalize);
    NVN_REG(nvnCommandBufferSetDebugLabel);
    NVN_REG(nvnCommandBufferSetMemoryCallback);
    NVN_REG(nvnCommandBufferSetMemoryCallbackData);
    NVN_REG(nvnCommandBufferGetMemoryCallback);
    NVN_REG(nvnCommandBufferGetMemoryCallbackData);
    NVN_REG(nvnCommandBufferAddCommandMemory);
    NVN_REG(nvnCommandBufferAddControlMemory);
    NVN_REG(nvnCommandBufferGetCommandMemorySize);
    NVN_REG(nvnCommandBufferGetCommandMemoryUsed);
    NVN_REG(nvnCommandBufferGetCommandMemoryFree);
    NVN_REG(nvnCommandBufferGetControlMemorySize);
    NVN_REG(nvnCommandBufferGetControlMemoryUsed);
    NVN_REG(nvnCommandBufferGetControlMemoryFree);
    NVN_REG(nvnCommandBufferIsRecording);
    NVN_REG(nvnCommandBufferBeginRecording);
    NVN_REG(nvnCommandBufferEndRecording);
    NVN_REG(nvnCommandBufferCallCommands);
    NVN_REG(nvnCommandBufferCopyCommands);
    NVN_REG(nvnCommandBufferSetCopyRowStride);
    NVN_REG(nvnCommandBufferSetCopyImageStride);
    NVN_REG(nvnCommandBufferGetCopyRowStride);
    NVN_REG(nvnCommandBufferGetCopyImageStride);
    NVN2_REG(nvnCommandBufferSetRecordingMode);
}

}  // namespace nxrt::nvn
