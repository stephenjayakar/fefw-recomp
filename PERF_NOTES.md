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

## Waits track (branch `perf/waits`)

Goal: the game thread (main) spends about 12 ms of a ~17.7 ms street frame on the CPU; the rest is waiting. This track measures what it
waits on and cuts the waits that are the runtime's (guest sync, wake-ups, placement, submission cost), not the render thread's replay
speed (RENDER track) or GPU cost (GPU2 track). Test requests go to `tmp/fefw-perf/queue/waits-<n>/`; scratch and tools in
`tmp/fefw-perf/waits/`.

### What main waits on (best stack: b59bc62 + gpu set + split 2 + PGO exe; tester runs w11-pgo-b, c3-cpu3-O, w8-cg2-prof)

Per presented frame, from the budget lines (`waits` = main's own system-call waits; GPU sync = nvnSyncWait of any thread):

| street window | frame p50 | main CPU p50 | main svc waits p50 | GPU sync waits | render work p50 |
|---|---:|---:|---:|---:|---:|
| light (800-1300 draws) | 16.7 | 10.9-11.8 | 2.5-2.8 | 0-0.5 ms (p99 0.5-3) | 8.6-12 |
| dense (2400-2640 draws, 130-160 s, 290-320 s) | 18.3-19.1 | 12.0-13.5 | 3.6-4.8 | ~1.7 ms (464 waits / 914 ms in 10 s) | 16.6-17.4 |

- **GPU sync waits** are all on fences two presents old ("sync waits by fence age: 2"), and all "on the GPU only" (the render thread had
  already submitted the fenced work). In the dense windows they exist because the render thread (16.6-17.4 ms) is the limiter: the GPU
  finishes frame N-2 late because its replay finished late. Deeper game/render pipelining cannot help there (throughput-bound, not
  latency-bound: `present_ahead` 1 already lets the game record N+1 while the render thread replays N, and the game's own fence is two
  frames back). In light windows they are near 0. Not a game-side lever: they go away when the RENDER track shortens replay.
- **Main's system-call waits** (occupancy run c3-cpu3-O, hub, ~600 frames per 10 s; "who woke main"):

  | woken by | waits / frame | blocked / frame | per wait |
  |---|---:|---:|---:|
  | Basic_Worker1-4, address (job completion) | ~8 | ~1.5 ms | ~185 us |
  | Basic_Worker1-4, mutex hand-over | ~25 | ~0.5 ms | ~20 us |
  | GFX_Worker0, address | ~11 | ~0.25 ms | ~23 us |
  | GFX_Worker1-4, address | ~15 | ~0.12 ms | ~8 us |
  | SND_Update, mutex | ~1 | 0.1-0.18 ms | ~150 us |
  | PX_Executor, address | ~1 | ~0.08 ms | ~85 us |

  While main waits on an address, on average only 1.0-1.5 of the 4 Basic workers run, and none for 25-45 % of that time (the sampler
  counts a thread as blocked until it runs again, so that share includes the host's wake-up of the worker and of main): main waits on
  short chains of jobs, not on a saturated pool. The Basic workers are idle 85 % of the time (98.7 % of their waits are the job
  condition variable).
- **Sampled stacks** (w8-cg2-prof, main, whole run): of main's waiting samples 38 % are WaitForAddress (17 % from one call site of the
  frame loop, f_7100a99ae0 < f_710180b3a0; 15 % from a job-wait helper f_7100a86a90 called from many places), 37 % present back
  pressure (pacing in light frames), 12 % ArbitrateLock (most through the lock wrapper f_7101979ee0), 6 % nvnSyncWait, and 2.8 %
  *inside* SignalToAddress / ArbitrateUnlock (the waker side: the kernel lock or the wake system call).
- **Wake-up latency on this PC** (tmp/fefw-perf/waits/bench/wakebench.cpp, idle machine, two pinned threads, sleeper really asleep):
  SRW + condition variable woken under the lock (what kernel_sync.cpp does) p50 6.0 us / p99 11 us; woken after the unlock 4.0 / 8.5;
  WaitOnAddress 4.0 / 8.7; a spin 0.1. Same CCX, other CCX and SMT sibling all within 1-2 us of each other. So one host wake-up is
  ~5 us: at ~60 blocking waits of main per frame plus the job wake-ups in its chains, the host's wake-up latency is a few tenths of a
  ms per frame, not milliseconds. The rest of main's ~2.6-4.5 ms is the jobs' and lock holders' own run time.
- **Placement**: console cores 0..5 -> processors 12, 10, 8, 6, 4, 2 (physical cores 6, 5, 4 on CCX1; 3, 2, 1 on CCX0), soft affinity
  on 0xfffc. Main is core 0; the Basic job pool runs on cores 1..4 ([sched] core_number_at note), so two of the four workers sit on
  the other CCX.
- **Starvation guard**: its thread polls NtQuerySystemInformation(SystemProcessInformation) every 50 ms at THREAD_PRIORITY_HIGHEST and
  costs 1.4-1.6 ms of CPU per frame (8-9 % of a core, more than any GFX worker), all of it a snapshot of every process on the PC.
- **Arrival frames** (~320 ms): only 8-12 ms of system calls and 100-400 page faults per arrival frame (w11-pgo-b hitch lines); the
  allocator share (~14 %) is the lifted SDK allocator itself (s5-default samples under f_7100652310: 97 % lifted SDK code, ~10 % the
  cross-module call path host_import / import_resolve, 1.4 % write-watch faults; no virtual-memory or kernel memory service on the
  path). Nothing for the runtime's memory services there.

### Large pages for guest memory: feasibility (item 4)

- **What a TLB miss costs here** (tmp/fefw-perf/waits/bench/tlbbench.cpp: a random pointer chase over the same number of cache lines,
  packed 64 B apart vs one line per 4 KiB page; Zen 2's L2 DTLB holds 2048 4 KiB entries = 8 MB of reach):

  | lines | packed | one per page | per-load cost of the page walk |
  |---:|---:|---:|---:|
  | 4096 (256 KB, L2) | 2.8 ns | 14.4 ns | +11.6 ns |
  | 16384 (1 MB, L3) | 8.9 ns | 23.0 ns | +14 ns |
  | 65536 (4 MB, L3) | 10.5 ns | 32.3 ns | +22 ns |
  | 262144 (16 MB, past one CCX's L3) | 72 ns | 137 ns | +65 ns |

  With 2 MiB pages a few hundred MB of game objects need a few hundred TLB entries instead of tens of thousands, so most of that walk
  cost goes. The game thread's samples sit ~48 % on guest loads (codegen track); how many of them miss the DTLB is not known without
  hardware counters (AMD uProf's "L2 DTLB miss" events would say it in one run; not installed here). Plausible range: a few percent to
  ~15 % of main's CPU.
- **Why it is not a toggle yet:**
  - The heap is ONE pagefile-backed section (SEC_RESERVE, primary view at kHeapBase plus alias views for pool imports; memory.cpp).
    Large pages need `SEC_LARGE_PAGES | SEC_COMMIT` (committed and non-pageable up front: the game's ~11 GB heap would be locked RAM on a
    24 GB PC), 2 MiB-aligned views, and a section is either large-paged or not.
  - Write-watch (write_watch.cpp) arms single 4 KiB pages read-only with VirtualProtect to see CPU writes to GPU pools that lie inside
    that heap; a large page cannot be protected per 4 KiB page. So only heap ranges that never hold a watched pool could use large pages,
    and the game decides where its pools go after the heap is committed at boot. A workable design: the heap as many 2 MiB-granular
    sections, large ones only for ranges a recorded pool map of earlier runs says stay unwatched, with a fallback to 4 KiB sections when
    the large allocation fails (it can take seconds or fail outright on a fragmented system). That is a substantial rework of memory.cpp
    and the alias/import paths, for a gain not yet measured.
  - Allocating ~10 GB of large pages at boot needs Windows to find contiguous 2 MiB physical runs; after hours of uptime it may fail or
    take many seconds.
- **The privilege step the user would need** (only if this is ever built and wanted): run `secpol.msc` -> Local Policies -> User Rights
  Assignment -> "Lock pages in memory" -> add your own account -> sign out and back in. `whoami /priv` then lists
  SeLockMemoryPrivilege. Nothing was changed on this PC; the benchmark reports the privilege as not held.
- Recommendation: first measure DTLB misses on main in the street with AMD uProf (free) on the current build. Only if they are a large
  share of its loads is the heap rework worth it.

## Render track (branch `perf/render`)

Goal: the render thread (NVN command replay to Vulkan) at 2600 draws/frame in the dense street: work p50 <= ~11 ms, p99 <= ~15 ms (was
p50 16.6-17.4, p99 22-24 ms, ~6.5 us a draw; COORD 21:30). Test requests go to `tmp/fefw-perf/queue/render-<n>/`.

### Where a draw's time goes (w8-cg2-prof, bdd8f2f, resolved to source lines)

Tools (`tmp/fefw-perf/render/`): `tree.py` (busy-sample call tree of one thread), `drv.py`/`drv2.py` (driver share, by calling
function), `lines.py` / `incl.py` (self / inclusive samples per source line: the profiled DLL's offsets moved onto a byte-identical
function of a build linked with a PDB, `linkpdb.cmd`, then `llvm-symbolizer`). Shares of the render thread's busy samples:

| what | share | notes |
|---|---:|---|
| NVIDIA driver (nvoglv64) | 28 % | vkCmd* ~22 % (bind pipeline 5.1, push descriptors + bind sets 3.4, dynamic state 1.8, begin/end rendering 2.3, barriers 2.3, dispatch 1.3, draws), vkUpdateDescriptorSets 2.4, vkAllocateDescriptorSets 1.5, submits 1.9 |
| build_set (descriptor sets, inclusive) | 23.7 % | UBO resolve_read 3.3, tex_sampled_view 2.4, vkUpdateDescriptorSets 2.4, shadow_of 1.7, alloc_set 1.6, sampler_for 1.6, make_request 1.0, set cache 0.8, the rest spread thin |
| out_draw_bindings | 11-12 % | the full walk over ~84 buffer slots (16 vertex + 2 stages x 32 + 4 xfb), taken after every program bind; the dirty-only path is 0.1 % |
| prepare_textures | 4.5 % | |
| samples_depth_target | 3.5 % | first resolve of each texture binding of the draw + tex_or_null cache misses |
| vertex buffers (resolve_read) | 3.4 % | |
| ensure_pass | 2.8 % | |
| prog_layout | 2.0 % | its mutex and map, ~1100 program switches a frame |
| graphics_pipeline + build_key | 2.0 % | |
| do_dispatch (inclusive) | 10.7 % | 179 dispatches a frame |

### Changes

| commit | what | toggle (default) |
|---|---|---|
| 916068c | deferred Vulkan recording (`nvn_cmdq.cpp`): the vkCmd* / vkUpdateDescriptorSets calls the render thread makes while it records a submission go into a ring (arguments and the arrays they point to copied) and a worker thread `nvn-rec` makes them, in order, on the same command buffers. Begin/End/submit/reset/allocate/free wait for the worker first; DLSS / FSR / Streamline / frame-generation recorders run after a sync (`cmdq::Direct`). The volk entry points are wrapped once at bring-up; outside a recording scope every call goes straight through. | `[nvn] replay_threads = 1` (2 = on) |
| 916068c | out_draw_bindings walks only bound slots (masks kept by BindBuffers), same order and result | `[nvn] out_masks = true` |
| 916068c | 256-entry direct-mapped program layout cache for draws and dispatches | `[nvn] layout_cache = true` |

### Test log

| request | what | result |
|---|---|---|
| render-1 | out_masks + layout_cache (defaults), replay_threads = 2 A/B, battle, profile | queued |

## Pipeline-stall track (branch `perf/pipe`)

Goal: no first-use stall over ~50 ms (new areas, spells, effects, cutscenes) even with cold pipeline caches. Test requests go to
`tmp/fefw-perf/queue/pipe-<n>/`; cold-cache runs through `tmp/fefw-perf/pipe/cold_batch.py`.

### Diagnosis: the r1-tbat battle stall (3 frames of 319 / 326 / 453 ms in `dungeon_after`, t = 100-102 s)

It was not a pipeline compile on any frame's critical path:
- render thread idle 300-433 ms of each frame (replay 5.6-8.1 ms, pipeline creation 0.1-0.3 ms): it had nothing to replay.
- the game thread was off the CPU: gap 327-459 ms, but on the CPU only 18-37 ms and in guest system calls only 3-4 ms. GPU sync waits 0,
  present back pressure 0. So it was blocked in host code that is not a guest system call.
- in those 2 s the game set 45 programs (35 new) and the battle's 32 new pipelines appeared. The GPL side did its job: 116 new shader parts
  compiled on the background threads (4090 ms of worker time, ~35 ms each, so a cold driver cache), 43 draws deferred for 11 parts, 0 parts
  built or waited for at the draw, 0 whole pipelines at the draw. Those deferred draws are skipped (pop-in), not stalls.
- the 32 new keys use 52 distinct shaders; **38 of them were not in `spirv.preload.pack` and not in the order file** (`pipe/keys_vs_pack.py`).
  `nvnProgramSetShaders` -> `ShaderCache::find` -> `load` -> `read_spv` reads such a shader as a separate `.spv` file **on the game thread**,
  from `E:/.../shader_cache/spirv` (the slow, 99 % full HDD; the test install points there too). Not in the Windows file cache, each is a
  directory/MFT/data seek or three: 38 x ~30 ms = ~1.1 s, the length of the three frames. Repeat runs did not stall because the files were then
  in the file cache (and the keys in the key file), which is also why the user sees it again after a reboot or once the cache is evicted.
- why the pack lacked them: the pack only holds the order file's shaders (12,088 records, ~3,600 packed), and the order file's new lines were
  held until 77 had come or until a shutdown, which a closed window never runs. The test install's `shader_order.txt` had not grown since
  13 Nov 09:45 (1560 lines) through dozens of runs.

Other first-use paths found by reading the code (not seen in the warm logs, but hit with cold caches):
- compute pipelines: no key file, no parts, built synchronously by the render thread at the first dispatch (`compute_pipeline`).
- a draw whose key-file pipeline the prewarm is still building, and whose parts are not ready, **waits** for the whole build on the render
  thread (`prewarm_take`, "prewarm.wait"): after a driver update every recorded key compiles cold at boot.
- the draw's fast link and its vertex-input / fragment-output parts are made with the pipeline cache, whose lock the saver thread holds while
  it serialises the whole cache (`vkGetPipelineCacheData`, every 2 s while new pipelines appear, 300+ MB here).
- "Prepare shaders" (`gpl_prepare`) cannot run for Fortune's Weave: it needs `shader_cache/programs.bin` (vertex/fragment pairs), which the
  FE setup does not write. Shader parts depend on the pair (shared layout, producer outputs), so a full-corpus parts prepass needs that list
  first (follow-up: write programs.bin from the BNSH variations at setup).

### Changes

| commit | what | toggle (default) |
|---|---|---|
| 8c10ac3 | the pack holds every shader of `results.tsv` with SPIR-V (order file's first); the first start that finds it incomplete writes the full `.new` pack in the background (30 s after boot, background I/O priority); used from the next start | `[nvn] shader_pack_all = true` |
| 8c10ac3 | the pack's frames stay in memory after the boot read (~170 MB for the full pack): a shader load never touches the disk | `[nvn] shader_pack_resident = true` |
| 8c10ac3 | order file appended every 2 s by its own thread (`nvn-order-write`), never on the game thread | always |
| 8c10ac3 | instrumentation: `shader loads by the game: ...` line every 10 s (by source), a warning per load over 20 ms, hitch counters `shader_loads`, `shader_file_reads`, `shader_load_ms` | always |
| 8c10ac3 | compute pipelines built on the pipeline-library threads when the game sets a compute program; the dispatch takes it, waits for one under way, or builds one still queued; `compute pipelines ahead: ...` line | `[nvn] compute_async = prebuild` (`off`, `skip`) |
| 8c10ac3 | a draw whose prewarm build is under way (not linkable) is skipped while frames flow instead of waiting | `[nvn] prewarm_wait = false` |
| 8c10ac3 | 2 extra pipeline-library threads that only take urgent jobs (parts a deferred draw needs, compute, bound programs) | `[nvn] gpl_burst_threads = 2` |
| 8c10ac3 | fast links and vertex-input / fragment-output parts without the pipeline cache (no lock shared with the saver) | `[nvn] gpl_link_cache = false` |

### Cold-cache test (no live files touched)

`pipe/cold_batch.py <plan>` wraps `harness/batch.py`. A plan entry with `"cold": {...}` gets its own empty pipeline cache, key file and order
file (`C:/Games/fefw-test/cold/<label>/`, via `NXRT_NVN_PIPELINE_CACHE` / `_KEYS` / `NXRT_NVN_SHADER_ORDER`), a fresh `NXRT_NVN_SPIRV_SALT`
(the NVIDIA disk cache misses too), and, with `"spirv": "fresh"`, a copy of the live shader cache written with unbuffered I/O to
`E:/BigData/games/fefw-pipe-cold` (its files are not in the Windows file cache, as after a reboot). Equivalent by hand: rename
`pipeline_cache.bin` / `pipeline_keys.bin` / `shader_order.txt` in the TEST install and restore them afterwards.

### Test log

| request | what | result |
|---|---|---|
| pipe-1 | cold A/B (base 9635cb3 vs 5f195a3, DLL md5 0708780b), window 21: battle + hub with empty pipeline cache/keys/order, new spirv_salt, uncached HDD copy of the shader cache; + warm check | **battle: worst gameplay frame 142 ms -> 44 ms** (A: 142 / 99 / 92 / 80 ms frames = compute pipelines compiled at the dispatch on the render thread 98 / 86 / 27 ms, and main blocked in ShaderCache::pack_read on the HDD; B: battle_open..dungeon_after <= 19 ms, the 40-44 ms frames are texture uploads at the dungeon load-in). Max render-thread "pipeline creation" in any hitch line 815 -> 52 ms (battle), 750 -> 23 ms (hub). Hub cold: no pipeline/shader frames over 50 ms in either (A loaded these shaders from the pack at loading time); remaining >50 ms = fast-travel arrivals (game code, ~300 ms) and texture frames after the load-in. B shader loads: 1155 / 1155 from the pack in memory, 0 .spv reads, worst 7.7 ms. Compute: 69 prebuilt, 0 built at the dispatch, 2 waits (380-450 ms) only at boot (title, t=6-9 s). Full pack written in 134 s (12,009 shaders, 681 MiB in 175 MiB). Warm check (p1-B-warm): within noise / slightly better than w17-base (street_cold 58.1 / 44.0, 239 frames >20 ms vs 57.6-57.8 / 42.1-42.9, 297-318). |

#### What is left (cold caches)
- The menu_to_world load-in frame is 1.3-1.5 s cold in both (warm ~260 ms): the render thread waits for shader parts at the draw (25 parts,
  1.6 s in all) because frames are not "flowing" during loading, so gpl_async builds instead of skipping. Only on a first-ever visit with a cold
  driver cache; a loading-screen-adjacent frame.
- Compute builds still in progress at the first dispatch are waited for (2 waits of 380-450 ms, at the title screen in these runs).
  `compute_async = skip` would skip them; not default (a skipped dispatch may leave results later frames use).
- In the first cold minute the background compiles (gpl 4 + burst 2 threads, ~70 s of CPU) slightly lower hub 1% lows (B-hub hub_cold 44.4
  vs A 50.4 in one run; B-build 50.7): noise-level, no frame over 33 ms.
- "Prepare shaders" cannot run for Fortune's Weave without `shader_cache/programs.bin` (setup does not write one); with the key file and the
  full pack, first-use is covered at runtime anyway.

