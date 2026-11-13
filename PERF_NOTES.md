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
| `nvn: [nvn] out_memo_dirty ...` | A draw checks only the buffer slots bound since the last draw against held compute results (every slot after a program bind, a change of the held set, or an unmemoised slot). `[nvn] rt_check = 1` verifies the skipped slots. | `[nvn] out_memo_dirty = true` |

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

- cpu-3 (tester window 5, DLL 0c04e18): A3b (all the small savings) vs cpu-1 A2: neutral (street render work 12.7 -> 12.3 ms, game
  work the same, stills fine). C (`render_priority = highest`, `tex_prep_priority = below`, clean): hub_cold 56.2 / 1 % low 36.4,
  about the clean merged baseline; neutral to a slight plus in one run (confirmation queued as cpu-4). F (`split_submit_ms = 2`):
  void (SWBG builds loaded the machine 30-72 %); it did split 2-5 times a frame and cut main's GPU sync waits p99 from 7-9 to ~3 ms;
  re-run queued (cpu-4).
- cpu-3 O (`[kernel] occupancy`, street): main is blocked 13-20 % of the time: job waits on the Basic workers (~1.5 ms/frame of
  real work), ~100 short address waits a frame on GFX_Worker0 (~40 us each), ~25 mutex waits a frame on Basic workers (~20 us),
  SND_Update's mutex (~150 us each, ~0.15 ms/frame). Wake-up latency is a few tenths of a ms a frame at most.

### Where the CPU budget stands (4K native, Ryzen 7 3700X)

| scene | game thread work p50 | of it on-core CPU | render thread work p50 | verdict |
|---|---:|---:|---:|---|
| battle (tester) | 7.6-9 ms | | | fits with headroom: 60 fps, 1 % low 58+ |
| hub (Arena Plaza) | 16.2-16.6 ms | 13.2-13.5 ms | 9.2 ms | game thread at the budget |
| street (City Streets) | 19.0-19.8 ms | 14.3-15.0 ms | 12.1-12.7 ms | game thread over it (about 51 fps) |
| warm plaza (fast-travel point) | 17 ms | 13 ms | 10 ms | game thread at the budget |

The render thread fits (9-13 ms, 16 ms in the street's worst windows). The game thread does not fit in the towns, and 86 % of its
busy time is the game's own lifted code (run B), already compiled at clang -O2. The runtime's share of it (NVN recording, imports,
SVCs, ~10 %) has been trimmed where the profile pointed; none of the trims moves fps beyond the run-to-run noise (about 3 fps).
The rest needs the game code itself to get faster: the codegen track (hot functions, PGO).

Pacing at 60: the limiter leaves presents at p50 16.667 / p95 16.669 / p99 16.714 ms when the game keeps up (w1a), and pacing is
on the render thread right before acquire and present. The user plays borderless with vsync off (immediate), so the 240 Hz grid does
not quantise presents; slow frames in the towns are the game thread's (budget "waiting for the game"), not the limiter's. No 60/30
alternation exists in this pipeline (the present interval only follows the game's target). Nothing to fix there.

### Verdicts of the knobs (all default off unless said)

- `[diag] budget` (on): keep; it is how every number above was taken.
- per-thread call counters, one-lookup imports, command list reserve, sampled-format memo, `out_memo_dirty` (on): neutral in fps,
  no regressions seen; kept (cheaper paths, less load-time contention).
- `[sched] render_place = reserve`: no win. `[kernel] spin_us = 40`: loss. `[sched] render_priority = highest` +
  `[nvn] tex_prep_priority = below`: neutral to slight plus, awaiting cpu-4. `[nvn] split_submit_ms = 2`: awaiting a clean run (cpu-4).
- An `-O2` fefw.exe: void (Zig already compiles the game code at clang -O2).
