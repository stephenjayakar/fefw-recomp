# perf-stutter: notes

Branch `perf-stutter` (from `v0.1.7`). The goal is to remove the drops from about 60 fps to about 2 fps that look like shader
stutter. Every change has a config toggle; the old behaviour is one ini line away.

## 1. Diagnosis

The evidence is the `fefw.log` of a 1 h 45 min session (v0.1.5, RTX 3080, Ryzen 7 3700X, 24 GB). The previous session's log and
`fefw_freeze.txt` agree with it. The tables come from `tools/perf/hitch_report.py`, which buckets every "hitch: frame N took X ms"
line by where that frame's time went.

Slow frames over 100 ms, current session:

| cause | count | total ms | worst ms |
|---|---:|---:|---:|
| render thread: texture upload / BC7 cache read / transcode (`uploads` phase) | 69 | 437 371 | 161 479 |
| render thread: texture prep outside `uploads` (image creation peeks, pool preparation) | 50 | 63 533 | 14 843 |
| game idle while textures stream (texprep busy; 22 of them in loading screens) | 55 | 60 300 | 17 709 |
| game: guest CPU, nothing tracked (mostly at loading transitions) | 64 | 26 817 | 6 361 |
| game: stall while pipelines build in the background | 14 | 4 464 | 848 |
| game: system-call waits | 11 | 3 972 | 531 |
| game: memory pool import (vkAllocateMemory) | 3 | 1 963 | 917 |
| game: GPU sync waits | 6 | 1 525 | 396 |
| present back pressure (GPU / swapchain) | 7 | 1 342 | 445 |
| render thread: pool mirror copies / write faults | 2 | 595 | 338 |
| **render thread: pipeline creation at the draw (the "shader stutter")** | **0** | **19 ms in all** | |

Previous session: 15 frames of texture prep outside `uploads` (94 s, worst 40 s) and 16 frames of texture upload (54 s, worst 19 s).
Pipeline creation there was 181 ms in all.

**It is not shader compilation.** The pipeline side already never blocks:

- Over the whole session, 0 pipelines were built whole at the draw.
- 171 draws used a GPL link, 33 ms in all.
- 193 draws were deferred while their parts compiled.
- The prewarm built all 2952 recorded pipelines ahead (`0 waited 0.0 ms`).

**Root cause: the render thread reads BC7 texture-cache files from a slow hard disk, synchronously, thousands at a time.**

- `[nvn] bc7_cache` (`shader_cache/bc7`, about 10 GB) is on drive E:. That drive is a 2011 Hitachi HUA723020ALA641, a 7200 rpm
  SATA HDD, and it is 99% full. The romfs is on the NVMe drive.
- Of 4248 "BC7 data from cache" reads, 3420 (80%) ran on `nvn-render` and only 828 on the background `nvn-texprep` workers.
  Texprep's own summary agrees: `used 2409, no job 19581`. The jobs were never started for most textures. A texture under
  `tex_prep_early_kb` (1 MB) is only scheduled when its handle is bound, but this game samples through texture pools, so the
  first sign of a texture is the pool preparation that uploads it.
- Each render-thread read takes 31 ms (median), 94 ms (mean), 184 ms (p90) and 1.16 s (p99), going by the spacing of consecutive
  render-thread read lines.
- At a first visit, pool preparation uploads every registered texture in one frame. The worst frame (t=653 s) took 161 s:
  2544 uploads, 2.1 GB, 1716 BC7 entries, 86 s in `uploads` and 69 s of other replay. The 69 s is mostly image creation, which
  hashes the storage twice and opens up to two cache files per texture (the high and the fast preset) to learn the entry format.
- With `[gfx] astc_quality = high` every texture looks for two entries (high preset, then fast) and the background upgrade writes
  new entries, which means more seeks on the same disk.
- Only 1.5 to 4.5 GB of RAM was free during the session (other programs), so the OS file cache cannot keep the cache files and
  revisits read from the disk again.
- The 60 -> 2 fps episodes are exactly these frames. The 1-second summaries show 13 windows under 20 fps and 5 under 5 fps, and
  every one of them lines up with a burst of render-thread cache reads.

### What upstream 0.1.6 / 0.1.7 already fixed (not redone here)

`git log v0.1.5..v0.1.7` on the texture and pipeline files:

- `standin_full_px` default 256 -> 64 and a fast solid BC7 block: cache *misses* get a 1 ms stand-in instead of a 50-150 ms encode
  (2252115).
- A stand-in's real result marks pool-drawn textures dirty too (2252115). Its budget-eviction use-after-free is fixed.
- The battle-map list leak (#69/#85), shadow-pass splitting, push uniforms, and per-character work in crowds.
- Video memory: staging in system memory, `VK_EXT_memory_budget`, recovery after an allocation failure, and allocation-failure logs
  limited to one a second (cbaeed9, 2252115). The BAR-heap retry churn this task listed is already handled there.

None of this touches cache *hits* read on the render thread, which is what costs the time on this machine. On a fast NVMe drive
they cost a few ms each, which is probably why upstream never saw it.

## 2. Changes (each with its toggle)

| commit | what | toggle (default) |
|---|---|---|
| `nvn texdata: stand-in path that skips the disk cache; untile only the levels a stand-in encodes` | `TexCaps::standin_first`: a stand-in without computing cache keys or opening a file. A stand-in now untiles only the small levels it encodes, not the whole mip chain. | used by `tex_async` |
| `nvn texprep: [nvn] tex_async = standin ...` | The render thread never reads the BC7 cache or transcodes for an ASTC texture whose background result is not ready: it uploads the stand-in, moves the job to the front of the queue, and the result replaces the stand-in a few frames later. It waits at most 2 ms for a job that is already running (was 5 s). A stand-in's follow-up job is never dropped for the budget. Copies and clears (`tex_ensure_exact`) still get exact content. A sparse texture only gets a stand-in at its first upload. | `[nvn] tex_async = standin` (`off` = old behaviour); `[nvn] tex_prep_wait_ms = 2` (5000 when off) |
| `nvn texdata: no cache file opened on the render thread when an image is made; hitch counters ...` | Entry formats are learned from every read, peek or write. With `tex_async`, image creation uses what was learned (BC7 unless a non-BC7 entry was seen), with no hashing and no file opens. New hitch-line counters `bc7_cache_reads_render` and `bc7_cache_read_ms_render`. | follows `tex_async` |
| `nvn texprep: start background jobs at pool registration from 256 KB` + `... default 64 KB` | More textures are prepared in the background when they are registered in a pool, long before their first draw: fewer stand-ins, less pop-in. | `[nvn] tex_prep_early_kb = 64` (upstream 1024) |
| `nvn texprep: stand-ins also for ASTC textures under tex_prep_min_kb` | Small textures also get a stand-in and a follow-up job: their cache read is a disk seek too (about 4% of the render-thread reads were BC7 files under 16 KB). | follows `tex_async` |
| `nvn pipelines: background build threads a quarter of the CPU; EcoQoS only on hybrid CPUs` | `gpl_threads` and `prewarm_threads` default to cpus/4, clamped 1..6 (4 here, was 2). EcoQoS is only on for CPUs with P and E cores. | `[nvn] gpl_threads`, `prewarm_threads`, `compile_eco = auto` (true/false force it) |
| `CONFIG.md: ...` | Documents the above. | |
| `tools/perf/hitch_report.py` | The log analyser used for the tables above. | |

What `tex_async = standin` trades: on a first visit, textures can look blurry or flat-coloured for a moment (one colour per 4x4
block, taken from a 64-px level) until the background job's result lands. Expect a few frames on an SSD, and a few seconds for a
big area on the HDD. That is the Dolphin-style async trade, applied to textures.

### Ideas not done (the diagnosis does not point at them)

- Skipping draws when no pipeline is ready: the runtime already defers draws while parts compile, and no draw built a pipeline.
- Ahead-of-time pipeline warmup at boot: it already exists (`prewarm` builds every key in `pipeline_keys.bin` from boot).
- Persisting GPL parts: the driver's `VkPipelineCache` already holds them, and there were 0 waits.
- A per-frame budget for stand-ins: a stand-in costs about 1 ms, but a 2500-texture registration still makes a 1-3 s frame.
  The real fix for that frame is a smaller first-visit batch (pool readiness) or an SSD. Left for after a measurement.

## 3. Build and deploy

The runtime is `fefw_rt.dll`. `fefw.exe` is linked on the player's PC against `tools/build/sdk/fefw_rt.lib`, and its exports
(`game-build/fefw/src/fefw_rt.def`) are unchanged on this branch. So on top of a **v0.1.7** install, a runtime-only change is a
**drop-in swap of `fefw_rt.dll`**: no Set up, no game rebuild, no shader re-translation. A v0.1.5 `fefw.exe` must not be paired with
this DLL. Build:

```
pwsh -File tools\textures\scripts\fetch_third_party.ps1      (or clone astc-encoder 5.7.0 and bc7enc_rdo b943862 into tools\textures\third_party)
pwsh -File game-build\scripts\build_fefw.ps1 -Target fefw_rt -BuildDir build\rtdll -CMakeArgs "-DFEFW_RT_DLL=ON" -Jobs 6
```

Output: `game-build\fefw\build\rtdll\fefw_rt.dll`. The release DLL uses the C++ bc7e encoder (its log: `bc7e C++ (AVX2)`), so no
ISPC build is needed to match it.

Deploy, with the game closed:

1. Back up `<install>\fefw_rt.dll` as `fefw_rt.dll.v017`.
2. Copy the new `fefw_rt.dll` over it. The launcher keeps every `[nvn]` key except the paths it owns.
3. To roll back, copy `fefw_rt.dll.v017` back. A config-only A/B needs no swap: `[nvn] tex_async = off` in `fefw.ini`.

A side-by-side install (`E:/BigData/games/fefw-recomp-perf/`) is only needed to keep v0.1.5 playable as well. For that: unzip the
v0.1.7 release, run Set up with the dump at `C:/Games/fefw-dump/updates/131072/{exefs,romfs}`, copy `save/`, `mods/`,
`upscalers/` (DLSS), `fefw.ini` (fix the paths) and `controls.ini`, then swap the DLL. `shader_cache/bc7` can be shared through a
junction, since its format (`kCacheVersion` 1) is the same in 0.1.5 and 0.1.7. `pipeline_cache.bin` and `pipeline_keys.bin` can be
copied but are rebuilt anyway, because 0.1.7 re-translates the shaders.

### Heavy steps (need the other project paused)

| step | estimate |
|---|---|
| Set up for 0.1.7: lift and compile the game (`fefw.exe`) plus shader re-translation | 30-60+ min of full CPU (the user is running it now) |
| `fefw_rt.dll` build from a cold folder (566 objects with SDL3, zstd, astcenc, bc7e and dav1d), 6 jobs | measured 76 s (16:35-16:36, 11 Nov); configure about 40 s; light enough not to need coordination |
| in-game tests below | about 20 min |

### Build status (11 Nov)

- `game-build/fefw/build/rtdll/fefw_rt.dll` built from `perf-stutter` with clang-cl 23.1.2 and VS 2022, `-DFEFW_RT_DLL=ON`, C++ bc7e (the
  same encoder as the release).
- Its export table is identical to the v0.1.7 release `fefw_rt.dll`: 16 exports, compared with `llvm-readobj --coff-exports`.
- `nxrt_gfx_tests gfx_nvn_texdata`: 19 of 19 pass, including the new `standin_first_skips_the_cache_entry`.
- `gfx_nvn_texgpu`: every functional check passes, including `texprep_background_upload`. The only failing check is
  `validation_enabled()`, because the Vulkan validation layer is not installed on this machine.
- The game itself has not been run.

## 4. Test plan

Get the BC7 cache off the HDD first, or test both ways (see quick wins). The code fix is meant to work on the HDD too, but the
speed of pop-in depends on the disk.

Logging, in `fefw.ini`:

```ini
[nvn]
frame_csv = frames.csv
[play]
log_status = true
```

The hitch lines are always on. After each run: `python -I tools\perf\hitch_report.py fefw.log`.

Scenes, each from a fresh start (first visits are the point). Use a save just before:

1. Dagsion, first visit (the worst case upstream measured). Walk the main street for 60 s.
2. A crowded town scene (crowds #70/#79/#83).
3. A battle start and the battle map (turn through 2-3 phases).
4. Revisit Dagsion in the same session (warm OS cache).

Pass criteria with `tex_async = standin`:

- `bc7_cache_reads_render` stays at 0 on hitch lines, and the BC7 "data from cache" lines are almost all on `nvn-texprep`.
- No gameplay frame over 1 s. Frames over 100 ms come in single digits per scene, and none are in the 1-second windows under 20 fps.
- The texprep summary shows "stand-in while queued or running" counting up, and "no job" far lower than before.
- No texture stays blurry after about 5 s on an SSD (about 30 s on the HDD for a big area). If one does, the follow-up job was lost:
  note the texture id from `[nvn] tex_prep_log = true`.

A/B toggles, one at a time against the default:

| A/B | expect |
|---|---|
| `[nvn] tex_async = off` | the old multi-second freezes return on first visits (HDD) |
| `[nvn] tex_prep_early_kb = 1024` | more stand-ins and pop-in, same frame times |
| `[nvn] tex_prep_wait_ms = 5000` | occasional 30-1000 ms waits on running jobs |
| `[nvn] standin_full_px = 128` | sharper stand-ins, a bit more render-thread time per stand-in |
| `[nvn] compile_eco = true`, `gpl_threads = 2`, `prewarm_threads = 2` | boot prewarm takes longer; no gameplay change expected |
| `[display] astc_quality = fast` (controls.ini) | one cache lookup per texture, no upgrade writes; slightly lower texture quality |

## 5. Quick wins without new code (v0.1.5 or v0.1.7)

1. **Put the BC7 cache on the NVMe drive** (the biggest win; C: has about 66 GB free, and the cache is about 10-15 GB). With the game
   and launcher closed:
   ```
   robocopy E:\BigData\games\fefw-recomp\shader_cache\bc7 C:\Games\fefw-cache\bc7 /E /MOVE
   mklink /J E:\BigData\games\fefw-recomp\shader_cache\bc7 C:\Games\fefw-cache\bc7
   ```
   Use a junction, not `[nvn] bc7_cache`: the launcher rewrites the `[nvn]` paths before every Play. Moving the whole install to C:
   also helps: the 217 MB `fefw.exe` pages in from the HDD when RAM is short.
2. `fefw.ini`, `[nvn]` section:
   ```ini
   tex_prep_early_kb = 64     ; start the background reads when textures are registered, not at their first draw
   standin_full_px = 64       ; v0.1.5 only (0.1.7 has it): 1 ms stand-ins instead of 10-40 ms
   gpl_threads = 4
   prewarm_threads = 4
   compile_eco = false        ; Ryzen 3700X has no efficiency cores
   ```
3. `controls.ini` `[display] astc_quality = fast`: one cache lookup per texture instead of two, and no background re-encode writing to
   the disk. Textures are slightly less sharp (bc7e preset 1 instead of 3).
4. Keep RAM free while playing (the session had 1.5 GB free): the OS file cache is what makes revisits cheap.

## Stutter track (perf/stutter, 13 Nov)

Goal: native 4K60 with no gameplay frame over 100 ms after warm-up, first visits to big areas under about 250 ms, no recurring 33 ms+ spikes.

### Diagnosis (`tools/perf/hitch_detail.py`)

U = the user's 0.1.7 session (11-12 17:05, 61 min, BC7 cache on the E: HDD); W = tester run w1a (perf-stutter, NVMe cache, 1440p);
X = tester run x2-explore (perf-stutter, NVMe, 4K). Gameplay frames >= 100 ms: count / total s / worst ms.

| cause | U | W | X | status |
|---|---|---|---|---|
| render thread reads BC7 cache files | 124 / 198.3 / 38280 | 0 | 1 / 8.0 / 7991 (1243 reads) | U: fixed by perf-stutter + NVMe. X: evicted results, fixed by 523706b |
| render: first-visit uploads without disk (stand-ins, CPU untile, staging) | 3 / 0.5 / 227 | 3 / 1.6 / 996 | 2 / 0.46 / 277 (+ the 8 s load-in frame: 4378 uploads, 3.8 GB) | 19f219b, 575c10f, 1726755 |
| render: replay outside uploads with 3-4 big ASTC transcodes (upstream waits up to 5 s for the running job) | 13 / 8.4 / 1320 | 0 | 0 | perf-stutter (2 ms wait) |
| game thread busy, render idle, nothing tracked | 34 / 7.4 / 442 (+78 at 50-100 ms) | 1 / 0.17 / 169 | 1 / 0.47 / 472 | open: profiler in stutter-1 |
| game thread waits on another guest thread (svc waits 95-225 ms) | 9 / 1.9 / 257 | loading only | 0 | open: profiler |
| write-watch arming + fault handler (~180 ms, around memory pool imports of ~300 ms) | 0 | 1 / 0.18 / 180 | 3 / 0.65 / 234 | open |
| game: GPU sync waits | 13 / 2.5 / 390 | 0 | 1 / 0.40 / 397 | next to big upload frames |
| pipelines building in the background (no draw-time compiles) | 5 / 0.8 / 233 | 1 / 0.2 / 196 | loading | not shader stutter |
| memory pool import (vkAllocateMemory over game memory) | boot | boot 3.1 s | loading 0.35-0.64 s | boot/loading |
| BAR-heap allocation failures, movie, audio, save writes, romfs IPC | - | - | - | not seen (1 allocation failure at boot; fixed upstream) |

The x2-explore load-in frame: early (registration) jobs filled the 512 MB texprep budget and evicted 3623 finished results; textures that
already showed a stand-in found no result at their next upload and, the stand-in having been made for that content, were prepared on the
render thread (1243 cache reads, 3.4 s). The same frame uploaded every registered pool texture (4378) before the first draw.

### Changes

| commit | what | toggle (default) |
|---|---|---|
| 9ce4703 | slow-frame profiler `[prof] hitch_ms` (folded stacks per slow frame, `tools/perf/prof_hitch.py`), finer hitch line (uploads split take / prepare / staging, image creation, render-thread prepared bytes, game thread CPU time, page faults) | `[prof] hitch_ms = 0` |
| ba83adb | stand-in solid blocks encoded once per source texel and copied (byte-identical) | `[nvn] standin_copy = true` |
| 19f219b | persistent staging ring for render-thread uploads | `[nvn] staging_ring_mb = 256` |
| 575c10f | compact stand-ins: solid levels as distinct block rows, copied row by row (4096^2: 2 MB in 1.7 ms instead of 22 MB in 12-15 ms) | `[nvn] standin_rows = true` |
| 523706b | stand-in kept until its job's result exists; promised results never evicted; registration jobs never evict; budget 1024 MB | `[nvn] standin_keep`, `tex_prep_keep_promised = true`, `tex_prep_mb = 1024` |
| 1726755 | first uploads of pool-only asset textures limited to 24 ms of render thread per frame (pool dummy meanwhile) | `[nvn] pool_upload_ms = 24` |
| 8f3df04 | within that budget, textures with a ready background result first | with the above |

| 21a101e | frame trace written outside its lock: traced runs had a 33-46 ms frame every 60-61 presents (the writer held the zone-name lock while fprintf'ing a second of events); a measurement artifact only | `[prof] trace_flush_locked = false` |
| a07e3e2 | held-back pool textures leave the draw's list before the pass is ended (no pass break per pool draw while streaming) | with 1726755 |

Where the 8 s load-in frame of x2-explore went (its frame trace, render thread): `tex.prepare` 6.6 s of 3226 prepares, of which the
textures read from the cache 5.7 s (1243 textures, each also had a stand-in) and stand-in-only textures 1.6 ms each; CPU untile of BCn and
uncompressed textures 50 ms; image creation under the trace's 30 us floor. So the cost was the cache reads (fixed by 523706b) and stand-ins
(575c10f), then the count (1726755).

Test request `queue/stutter-1` (DLL 72f45f52, a07e3e2).

#### stutter-1 result (tester run s1-stutter1, 7d72d56, 4K hub scenario)

| scene | stutter-1 | upstream 0.1.7 (b3) | perf-stutter (b2) |
|---|---|---|---|
| menu_to_world worst frame | 312 ms (+ one 892 ms load-in upload frame) | 419 ms (+ 3828 ms) | 624 ms (+ 17942 ms) |
| hub_cold | 0 frames > 100 ms, worst 46 ms; but 906 frames > 20 ms | 1 > 100 ms (192); 415 > 20 ms | 1 (216); 531 |
| street_cold / hub_warm / street_warm worst | 450 / 288 / 414 ms | 433 / 301 / 370 | 462 / 303 / 404 |

0 render-thread BC7 reads (2875 on texprep), texprep evictions 439 (b2: 4123), no texture faults in the stills.

What the profile (`tools/perf/prof_hitch.py`) showed and what followed (stutter-2):

- 729 ms load frame: the main thread 170 of 339 samples and a job worker 151 in one contended atomic, the global import-call counter in
  `host_import`: striped per thread (e7a2c85).
- 892 ms load-in frame: bc7e encode of the stand-ins' 64/32 px levels 97 of 254 upload samples (2389144, `standin_encode_px` 16); 3093
  images created in the frame, 151 ms (7b46ffa, lazy images for held textures).
- hub_cold's extra 20 ms frames: every pool draw re-scanned the held textures (c17ea4d) and the flat 24 ms budget stretched streaming
  frames (6dc02c8: 6 ms, 24 ms only with a backlog).
- 246 ms arrival frame: a render-thread pool chunk import only to GPU-untile a first upload (2389144, `untile_import` false).
- Fast-travel arrival frames of 300-740 ms: the game's main thread is on the CPU for 95% of them ("presenting thread on the CPU" in the
  hitch line), in game code (`f_71000cd800`, `f_71000d3ee0` under romfs reads: probably decompression). Game-side; open.

#### stutter-2 result (s2-default, 47ab015)

hub_cold 57.2 fps avg, 1% low 38.5, 482 frames over 20 ms (s1 906, upstream b3 415), 4 over 33 ms, worst 36 ms (upstream 192). The
menu_to_world load-in frame 582 ms (s1 892, upstream 3828, perf-stutter 17942); the load frame that lost 340 ms to the import counter
went from 729 to 452 ms. street_cold / hub_warm / street_warm worst frames 457 / 278 / 421 ms: game code at fast-travel arrivals.

stutter-3 (58d4987): during a burst, first uploads of textures the draws bind themselves wait too (the dummy is sampled meanwhile), and a
draw needing more than 64 first uploads starts the burst at once.

stutter-3 also has: 14be59c (the render thread's prepared data reuses one buffer: load-in frames showed 190k-540k page faults) and
10cf072 (a 64 MiB pool-copy chunk made ahead on a worker; the 277 ms street_warm arrival frame spent ~54 ms making one).

What is left after stutter-2, by the profile: the area-transition frames (street_cold 457 ms at the hub -> street change, hub_warm 278,
street_warm 421 + loading-screen frames of 436-695 ms) are the game's main thread on the CPU in game code, 85-95% of the samples in lifted
guest functions (f_71000cd800 / f_71000d3ee0 under romfs reads in several of them, which look like zstd decompression; zstd is statically
linked into `main`). A runtime-only fix would need an entry hook site at those functions, i.e. a relift of fefw.exe: out of scope for a
drop-in DLL.

#### stutter-3 result (s3-default, 10cf072)

menu_to_world load-in frame 397 ms (stutter-2 583, stutter-1 892, upstream 3828); hub_cold 56.3 / 35.3 fps, 591 frames over 20 ms, worst
37 ms; area-transition frames 414-499 ms (game code). The load-in frame still made 576 uploads: its first uploads were spread over many
draws, none above the per-draw trigger. stutter-4 (da103d3): the burst starts once the frame's draws needed more than 64 distinct first
uploads.

#### stutter-4 result (s4-default, a121890) and stutter-5

The menu_to_world load-in upload frame 295 ms (stutter-3 397), but street_cold regressed: 531 frames over 33 ms in the first minute of the
street (stutter-3: 20), render thread 14.4 ms/frame (11.5): the in-frame burst trigger at 64 distinct first uploads also fired while
walking into the cold street. stutter-5 (afc5697): that trigger at 256 (`pool_upload_burst_firsts`); the previous-frame backlog trigger
stays at 64.

## 6. GPU track (branch `perf/gpu`)

Goal: GPU time per frame comfortably under 16.6 ms at native 3840x2160 (render ratio 1.0, no upscaler) on the RTX 3080 10 GB, in
Dagsion crowds and battle maps. Test requests go to `C:/Users/aj12a/tmp/fefw-perf/queue/gpu-<n>/`; only the tester runs the game.

### What the back end does per frame (static reading, 13 Nov)

- Dynamic rendering. A pass is (re)opened whenever the targets change, a draw samples a texture that needs a layout change or upload
  (`prepare_textures`: the pass breaks), the depth attachment switches between writable and read-only (a draw samples the depth target),
  and around every dispatch, copy, clear-texture, fill, query and game barrier. Load op is LOAD (CLEAR only for a never-used image),
  store op STORE; on an immediate-mode GPU like the 3080 that costs next to nothing, so load/store ops are not a lever here.
- Barriers: with `[nvn] pass_barriers = tracked` (default) a pass that only wrote its attachments ends without a barrier and layout
  changes use narrow stage masks. Every dispatch is followed by a full barrier (all commands, all memory), every copy/fill/clear-texture
  too, and every `nvnCommandBufferBarrier` of the game becomes a full barrier that also ends the pass.
- Shadow maps: the game's own base size (2048 docked, `[fefw] shadow_size`), independent of the output size.
- Present: one full-screen draw of the 3840x2160 window texture into the swapchain image plus the (empty) overlay pass; ~0.1-0.2 ms.
- Upscaler step at render ratio 1.0: `[dlss] upscaler` decides; `taau`/`bilinear` run a 4K compute pass into an RGBA16F history image
  and blit it into the game's output texture (two 4K images read, two written).
- Pool copies (`pool_mirror`) keep vertex/uniform reads off PCIe; compute shadows and copy-backs move a few MB per second.

### First profile (tester trace 13 Nov 09:50, 4K, render x1.0, `gpu_passes` so drains are included)

`tmp/fefw-perf/gpu/gp_report.py` over the trace. Gameplay holds 60 fps; GPU 10-16 ms per frame with the profiler's drains.

| what | ms/frame | notes |
|---|---:|---|
| probe-filter compute `cs 8ab94383` (+ `eaac1801`, `13a045cc`, `7419dc86`) | 2.2-3.8 | 24-48 dispatches x ~57 us; 8x8 groups, a cube-sampling loop writing a 2D image (prefiltered reflection probes); each followed by a full barrier, spread between unrelated passes |
| full-screen 1-draw passes at 3840x2160 (RGBA16F, RGBA8, R11G11B10F) | ~4-5 | the game's post and lighting passes; one RGBA16F pass is ~1.2 ms |
| G-buffer (5 targets + D32F) and depth pre-pass | 0.6-0.9 | 100-370 draws |
| `copy texture` (the game's CopyTextureToTexture) | 1.0 | 13-18 a frame at ~70 us (full-screen sized) |
| `write memory` (UpdateUniformBuffer) | 0.5 | 60 a frame in runs of 15, two full barriers each |
| clustered light compute `cs ee541a5d` | 0.43 | one dispatch, 256-wide groups |
| feedback copies (runtime) | 0.24 | 4 a frame, every level copied |
| `dlss` step | 0.24 | |
| shadow maps 2048 / 1024 | 0.2-0.4 | not resolution-bound |

### Changes

| commit | what | toggle (default) |
|---|---|---|
| `d4e3410` | `[prof] gpu_stats`: GPU ms per frame (submission timestamps), commands and barriers per frame, pass profile in the log (`gpu_passes` without the trace) | `[prof] gpu_stats = 0` |
| `d4e3410` | tracked dispatch barriers: the full barrier after a dispatch is left open until something depends on it | `[nvn] dispatch_barriers = full` (`tracked` to try) |
| `5d3e3e9` | tracked game barriers: `nvnCommandBufferBarrier` left out (pass kept open) when only attachment writes are unbarriered | `[nvn] game_barriers = full` (`tracked` to try) |
| `5d3e3e9` | sticky read-only depth pass for draws that neither sample nor write depth/stencil | `[nvn] depth_ro_sticky = false` |
| `4f41980` | UBO update runs: one barrier before and after a run of back-to-back `UpdateUniformBuffer` instead of two per update | `[nvn] batch_ubo_updates = false` |
| `4f41980` | feedback copies hold only the sampled levels (bloom mip chains no longer copy level 0) | `[nvn] feedback_levels = false` |
| `10e0b76` | draws that share nothing with open dispatches run without the barrier: compute overlaps graphics | `[nvn] dispatch_barriers = tracked_draws` |

### Test log

| request | what | result |
|---|---|---|
| gpu-1 | baseline GPU ms/frame + pass profile at 4K; A/B of the three toggles together | queued 13 Nov (5d3e3e9) |

## GPU2 track (branch `perf/gpu2`)

Goal: GPU time at or under ~14 ms p50 and under 16 ms p99 at 2600 draws per frame (the densest Dagsion street windows), at native
3840x2160, render ratio 1.0, no upscaler. Test requests go to `C:/Users/aj12a/tmp/fefw-perf/queue/gpu2-<n>/`.

### The 17-18 ms was not GPU work (13 Nov)

With `[nvn] split_submit_ms = 2` the tester's dense windows read "GPU p50 17.0-18.0 ms", next to render-thread work p50 16.6-17.4 ms.
- gpustats adds up each submission's first-to-last timestamp (top-of-pipe at the start, bottom-of-pipe at the end).
- A split submission hands its first parts to the GPU while the render thread still records the rest. So the span includes the GPU
  waiting for the next part, and it follows the render thread.
- The same windows without split show the real GPU time:

| run | split | draws/frame | GPU p50 | p99 | render work p50 |
|---|---|---:|---:|---:|---:|
| m5-set | off | 2502-2591 | 12.25-12.44 | 14.4-14.6 | 16.0-17.3 |
| g2B-gpu2-toggles | off | 2588-2614 | 12.3-12.5 | 15.0-15.35 | - |
| m5-split / -b | 2 ms | 2352-2633 | 17.1-18.4 (span) | 21.6-24.5 (span) | 16.6-17.4 |
| w11-pgo-b | 2 ms | 2443-2636 | 17.0-18.0 (span) | 22.7-23.6 (span) | 16.6-17.4 |

So the GPU at 2600 draws is about 12.4 ms p50 and under 16 p99: the target is met. The dense street is **render-thread bound**. Split
does not change render work (it is a win for other reasons, measured by the tester in window 6).

GPU cost by draw count (gp-m3-set pass profile, 1129 vs 2525 draws, timestamps drain the GPU):
- About +2 ms from 1100 to 2500 draws, about 1.4 us per draw.
- Where it goes: depth pre-pass `c0:0 d126` +0.3, forward `c4:97 d126` +0.36, the 7 full-screen `c1:97` passes +0.4, G-buffer
  `c5:43 d126` +0.13, clustered lights `cs ee541a5d` +0.16, shadows +0.1.
- The other ~10.4 ms is fixed 4K work: post passes, copies, UBO writes, light culling.
- Per-draw state changes are not where the GPU time goes at this draw count.

### Changes

| commit | what | toggle (default) |
|---|---|---|
| `33f6c13` | gpustats: a timestamp pair at every split boundary and a second line, `GPU busy ms/frame ...`. Busy is the union of the parts' intervals across all submissions; gaps where the GPU waits for the render thread are left out, and overlaps count once. `idle inside submissions` = span - busy | with `[prof] gpu_stats` (or the frame trace) only |

### Test log

| request | what | result |
|---|---|---|
| gpu2-1 | P (best stack + split + stats) and Q (no split): busy vs span at 2600 draws, and the GPU cost of the split parts | queued 13 Nov |

## CPU track (branch `perf/cpu`)

Goal: on the Ryzen 7 3700X both the game thread and the render thread fit in 16.6 ms with headroom at `[fefw] fps60 = 1`, with steady
pacing. Test requests go to `tmp/fefw-perf/queue/cpu-<n>/`.

### What the code already does (studied, not redone)

- Frame limiter (`src/gfx/frame_limiter.cpp`): QPC schedule, high-resolution waitable timer, spin only the last 0.3-3 ms (twice the
  timer's average overshoot), carry of at most 0.1 period after a late frame, resync after 4 periods. fefw paces in the present call
  (`[latency] pace = present`) on the render thread, right before acquire and vkQueuePresent, at 60 / present interval
  (`game_rate`). At 1440p (run w1a) the presents leave the limiter at p5 16.585 / p50 16.667 / p95 16.669 / p99 16.714 ms: the limiter
  itself is not a jitter source. The user plays borderless, vsync off (`immediate`), so presentation is not quantised to the 240 Hz grid.
- Threads: the console core model gives the six guest cores ideal processors on six physical cores (core of CPU 0 left out), boost
  off, process in the high priority class. nvn-render is a normal-priority host thread (below the game's job workers, level with main and
  the 5 texprep workers). `[sched] render_place` (core5 / apart / mask) exists, no fps change measured upstream.
- Render thread: dirty bits, a layout memo, push descriptors for uniforms (`push_uniforms`), set caching, `replay_stats` per-draw cost
  timers. The game thread's NVN recording is a header check, an uncontended lock and an append per call.
- Guest sleeps: high-resolution timer, short sleeps yield-polled; 1 ms system timer resolution held while audio runs.
- Existing profilers: `[prof] sample_hz` (folded stacks; `sample_exact = true` + the DLL's linker map resolve `fefw_rt.dll` frames),
  `[prof] trace`, `[nvn] replay_stats`, `[kernel] occupancy`, `[kernel] wait_stats`.

### Changes

| commit | what | toggle (default) |
|---|---|---|
| `diag: [diag] budget ...` | Every 10 s summary also logs (tag `budget`): render thread work / CPU / GPU waits per frame (p50 p95 p99 max), game thread frame / work (gap minus present back pressure) / CPU per present, slow frames (over 1.2x the median) by cause (waiting for the game, render work, GPU/swapchain), and CPU % per thread group of the whole process (driver threads named by their module). Worker threads get Windows thread descriptions. | `[diag] budget = true` |
| same | `[sched] render_priority = normal / above / highest`: the render thread's priority. | `normal` (unchanged) |
| same | `[nvn] tex_prep_priority = normal / below`: background texture workers below the game and render threads. | `normal` (unchanged) |
| `sched: [sched] render_place = reserve ...` | One physical core with no console core on it (3700X: core 7 = CPUs 14-15, the CCX of main's core 6) is kept for nvn-render alone, and main's SMT sibling is kept free; guest threads get the smaller pool, and a sweeper moves every other host thread (runtime workers, driver threads) to that pool every 2 s. | `[sched] render_place = off` (unchanged) |
| `nvn: [nvn] split_submit_ms ...` | At a render pass boundary, once N ms of a submission's recording have passed (16+ draws, `split_min_draws`), the command buffer so far goes to the GPU (no signal; the final buffer signals the serial) and recording continues in a fresh command buffer of the slot, with all command buffer state rebound. Not with queries or stream-out active, batched submits or a pending pool copy. The GPU starts on a long submission while the render thread still records it. | `0` (off) |
| `core: import, HLE thunk and SVC call counters per thread ...` + `core: host_import resolves target and name in one lookup` + `d50e1ac` (import counter as a plain per-thread block) | Every cross-module call, HLE thunk call and SVC bumped a shared atomic; now each thread counts into its own block (`include/nxrt/tl_counts.h`), summed on read, and host_import resolves target and name in one lookup. | always on (same counts in reports) |
| `nvn: command lists reserve their last size up front ...` | A command buffer's next list reserves the bytes and held objects of its last one (decaying hint, capped), so a big list is not grown and copied a dozen times while main records it. | always on |
| `nvn: tex_sampled_view remembers the last sampled-format check ...` | The per-image `std::map<VkFormat,bool>` lookup per descriptor is skipped when the format is the last one checked. | always on |

Tools: `tmp/fefw-perf/cpu/prof_fold.py <folded> <fefw_rt.map>` resolves sampled `fefw_rt.dll` offsets and prints per-thread self and
inclusive tables.

### Measurements

- Tester's x2-explore trace (perf-stutter, 4K, gpu_passes on): 1525 of 10711 frames are 19-45 ms. In them main's work goes from about 8.5
  to 17 ms (frame minus back pressure, sync and slow svc waits), nvn-render replay from 5.3 to 15.6-19.4 ms (normal frames: 10.8 ms in
  the limiter), GPU exec from 10.7 to 16.3 ms. Slow frames come in runs (heavier views: more draws, shadow passes): CPU and GPU both
  scale with the scene. (`tmp/fefw-perf/cpu/slowframes.py`, `busy.py`.)
- Tester's baseline (BASELINE.md): main CPU 13-15 ms/frame, render replay 10 ms (hub) / 15 ms (street), GPU 15-16 ms, street 52 fps;
  the street's game thread waits on GPU fences two presents old ~300 times per 10 s: main, render and GPU are serialised.
- cpu-1 A/D (tester window 2): the game thread is the limiter at 4K: work p50 16.7 ms (hub) / 20.7 ms (street), its CPU p50
  14.5-17 ms plus 3.6-5 ms of waits on its job workers; render thread work 9.6 / 13.0 ms; GPU 15 ms. `render_place = reserve`: no
  win (render work a little lower, game work the same, p99s worse), stays off.
- stutter-1's slow-frame samples, resolved with its map (`tmp/fefw-perf/cpu/prof_fold.py`, `rt_from_guest.py`, `label_main.py`):
  79 % of main's busy samples are lifted game code; **8.9 % (23.7 % in the post-fast-travel stalls) sit on one `fetch_add`: the
  import-call counter `host_import` bumps on every cross-module call**, and the job workers guest-82..85 spend 46-52 % of their busy
  samples there. A shared counter written by ~20 threads at millions of calls a second: the cache line moves on every call.
  Fixed: per-thread counters (`tl_counts.h`) for imports, HLE thunks and SVCs, and one lookup per import (`import_resolve_named`).
- cpu-1 window 3: A2 (per-thread counters) = A within noise in steady state (main CPU unchanged: the counter only hurts under
  load-time contention); H (`[kernel] spin_us = 40`) is a loss (street_cold 43.5 vs 50.9 fps, main CPU up, waits not down).
- Run B (sampler, steady state, 4K, `tmp/fefw-perf/runs/c1-cpu1-B`): main's busy samples are **85.8 % lifted game code**, NVN
  recording 5 % (a fifth of it vector growth, fixed), the import path ~3.5 %, SVCs ~2.5 %. The render thread at 2586 draws/frame
  (street) replays 16.8 ms = 6.5 us/draw: descriptor sets 1.9, compute-out checks 1.1, texture prep 0.6, pipeline key 0.6, vertex
  buffers 0.4, the draw call 0.2 (`nvn.rstats`); the hottest single leaves: run_list, build_set, tex_or_null, out_overlaps.
- **The `-O1` in `tools/build/fefw_build.py` is not what it looks like:** the bundled Zig maps `-O1`, `-O2` and `-O3` all to clang
  `-O2 -vectorize-loops -vectorize-slp` (`zig cc -###`). An `-O2` exe built at 4 jobs (1073 s) came out with every section the same
  size as 0.1.7's. The game code is already -O2. Remaining compiler levers for the game thread: a profile-guided build (instrumented
  exe, a profiling run of the real scenes, rebuild with the profile; typical 10-20 % on branchy code; several hours of build and
  test time), or a native-ISA variant (`-mcpu=znver2`: LZCNT for the guest's CLZ, BMI2 shifts; a few % at most).
