// SPDX-License-Identifier: GPL-3.0-or-later
// Background preparation of texture upload data (S21). prepare_texture (nvn_texdata.cpp) reads the texture's storage in the game's
// memory, hashes it, reads the BC7 cache file and untiles; on the render thread, at an area load, that is a few dozen textures in one frame, and a
// few huge sparse volumes (23 and 46 MB each) that take 15 to 60 ms apiece. This module runs the same function on worker threads before the
// texture is first used and hands the result to tex_ensure_current.
//
// When a job starts (only block-compressed textures, the game's assets; render targets and video planes are not candidates):
//   * a handle of the texture is bound in a command list being recorded (nvnCommandBufferBindTexture*), which is shortly before the list is submitted;
//   * the texture is registered in a texture pool and has at least [nvn] tex_prep_early_kb of storage (the big ones are registered long before use);
//   * a sparse texture that was named by one of the above is written again (tile maps and tile writes arrive in many small steps): the job starts
//     when the writes have stopped for 40 ms.
// The game gives no signal when it has finished writing a texture (96% of the textures see no flush at all), so nothing here assumes the data is final.
//
// Correctness. A result is used only when both of these hold when the texture is first used:
//   * no CPU write was reported since the job started (Texture::prep_gen, bumped by every flush of pool memory, texel write and tile remap);
//   * the storage still hashes to what it hashed to before and after the job (so a write nobody reported cannot slip in; for a sparse texture the
//     hash covers the mapped tiles and where they sit).
// Otherwise the render thread prepares the texture itself exactly as before. A job still running for the texture is waited for, so a texture is
// never drawn with data that is not ready and never with older data than the synchronous path would have used. Jobs never run on a texture the game
// finalized, and are guarded against a page that disappears.
//
// Transcodes. An ASTC texture without a cache entry (the GPU has no ASTC) is transcoded by the job itself, on one thread at below-normal priority, and
// the result is written to the cache. A job in that state is not waited for: the render thread uploads a quick stand-in (preset 0, all threads, not
// cached; TexCaps::provisional) and calls provisional_uploaded; when the job's result is ready for the same content the texture is marked dirty, and
// its next use uploads the real data through take (final_uploaded). A stand-in is made once per content generation: if the real result is lost (budget,
// a failed job) the next upload of the same content transcodes on the render thread as before.
//
// Settings: [nvn] tex_prep (default on), tex_prep_threads, tex_prep_mb (budget of finished results held at once), tex_prep_min_kb,
// tex_prep_early_kb, tex_prep_verify (test aid: every used result is compared with a fresh preparation), tex_prep_log. Implementation:
// nvn_texprep.cpp; tests: gfx_nvn_texgpu.texprep_background_upload.
#pragma once
#include "nvn_core.h"
#include "nvn_texdata.h"
#include "nxrt/gfx.h"

namespace nxrt::nvn::be::texprep {

bool enabled();
// [nvn] tex_async = standin (default; off: the synchronous behaviour described above). The render thread neither reads the BC7 disk cache nor waits
// for a job: an ASTC texture whose background result is not ready gets a quick stand-in (TexCaps::standin_first), its job moves to the front of
// the queue and its result replaces the stand-in at the next use. A job already running is waited for at most [nvn] tex_prep_wait_ms (2 ms here).
// This relaxes the rule above that a texture is never drawn before its data is ready: for a few frames it is drawn with the stand-in.
bool async_standin();
// This thread is about to copy or clear the texture it prepares (tex_ensure_exact): no stand-in, the real data now. Returns the previous value.
bool set_exact_thread(bool on);
bool exact_thread();

// Any thread. on_flushed: the CPU wrote the storage (called by gpu::mark_texture_cpu_dirty). on_bound: a handle of the texture was bound in a command list.
void on_flushed(Texture* t);
void on_bound(Texture* t);
void on_registered(Texture* t);   // registered in a texture pool: only the big textures are prepared this early
// Render thread: the texture has been uploaded (it is not a first use any more).
void note_uploaded(Texture* t);
// Render thread, while its image is made: the format of the cache entry a background job found for t (VK_FORMAT_UNDEFINED: none, or BC7), or
// nullopt when no job exists. Saves hashing the storage a second time on the render thread.
bool peek_entry_format(Texture* t, VkFormat& entry);

// Render thread. True: out holds the data of the texture's current content (waits for a job in flight on this texture). False: nothing usable,
// prepare synchronously. The caller still checks that out.format is the format of its image.
// When the result was big it comes with an upload buffer that already holds the data (out.data is then empty): *staged is set, and the caller must
// hand it to Frame::adopt_staging. Without `staged` big results are not staged.
// src_hash, when set: the content hash the result was checked against (the storage as the upload uses it).
bool take(Texture* t, PreparedTexture& out, gfx::Buffer* staged = nullptr, uint64_t* src_hash = nullptr);
// Render thread. wants_provisional: t may get a quick stand-in now (texprep is on, t is a job candidate, and no stand-in was made for its current
// content). provisional_uploaded: a stand-in went into the image; the real data follows (a job is queued unless one exists). final_uploaded: real data went in.
bool wants_provisional(Texture* t);
void provisional_uploaded(Texture* t);
void final_uploaded(Texture* t);
// Render thread. standin_on_screen: the image holds a stand-in of the texture's current content (tex_async = standin). standin_kept: an upload of
// such a texture found no result of its job; the stand-in stays and the job is queued again at the front unless one is queued, running or done
// for this content (its end marks the texture dirty, and the next use uploads the result). Nothing is prepared on the render thread for it.
bool standin_on_screen(Texture* t);
void standin_kept(Texture* t);
// Any thread: hash of the texture's storage as take checks it (false: no storage or a page is gone).
bool hash_storage(const Texture& t, uint64_t& h);
// The GPU layer is going away (gpu::tex_shutdown): finished results give their upload buffers back and no new ones are made until reopen_staging.
void release_staging();
void reopen_staging();

// Counters for the log line and the tests.
struct Stats {
    uint64_t scheduled = 0, hits = 0, waited_hits = 0, stale = 0, stolen = 0, deferred = 0, none = 0, mismatches = 0, verified = 0, crashes = 0, failed = 0, skipped_budget = 0, evicted = 0;
};
Stats stats();
// Test aid: returns when no job is queued or running (at most 10 s).
void wait_idle_for_test();

}  // namespace nxrt::nvn::be::texprep
