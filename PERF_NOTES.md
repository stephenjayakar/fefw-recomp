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
