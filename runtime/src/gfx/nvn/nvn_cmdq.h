#pragma once
// Deferred Vulkan command recording ([nvn] replay_threads, nvn_cmdq.cpp).
//
// The render thread replays the game's NVN command lists: it decides what every draw needs (state, descriptors, layouts, uploads) and
// records it with vkCmd* calls. About a quarter of its time was the driver's own work inside those calls (bind pipeline, push descriptors,
// barriers, dynamic state, descriptor writes: ~1.5 us of the 6.5 us a draw cost in a 2600-draw street frame). With replay_threads >= 2 the
// calls the render thread makes while it records a submission are not run there: their arguments are copied into a queue and one worker
// thread makes the real calls, in the same order, on the same command buffers. The render thread waits for the worker only where it must:
// before a command buffer is ended, submitted, reset or allocated from, and before code outside the runtime records into it (Direct).
// Recording has no effect on anything until the command buffer is submitted, so the order of the calls is all that matters, and it is kept.
namespace nxrt::nvn::be::cmdq {

// Once the device's entry points are loaded (volk), before anything records: the volk vkCmd* entry points (and a few others) are
// wrapped. A no-op unless [nvn] replay_threads >= 2. Calls from any thread outside a Scope go straight to the driver, as before.
void install();
// The worker stops (back end shutdown). The wrappers stay and pass every call through.
void shutdown();
bool enabled();

// Render thread, around the recording of one submission: this thread's vkCmd* and vkUpdateDescriptorSets calls are queued for the worker.
// The destructor waits until the worker has made them all.
class Scope {
public:
    Scope();
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
private:
    bool on_ = false;
};

// Inside a Scope: waits for the worker, then lets this thread's calls go straight to the driver until it ends. For code that records into the
// command buffer through other entry points than the runtime's volk ones (NGX / DLSS, Streamline, FidelityFX).
class Direct {
public:
    Direct();
    ~Direct();
    Direct(const Direct&) = delete;
    Direct& operator=(const Direct&) = delete;
private:
    bool was_ = false;
};

// Waits until the worker has made every call queued so far (no-op outside a Scope).
void sync();

}  // namespace nxrt::nvn::be::cmdq
