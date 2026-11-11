# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarise the hitch lines of a fefw.log: what the slow frames were spent on.

    python -I tools/perf/hitch_report.py fefw.log [more logs] [--min-ms 100]

Streams the log (it can be large). Each "hitch: frame N took X ms" line is put in one bucket by where the frame's time went:
the render thread busy in replay (texture uploads, BC7 cache reads and transcodes, pipeline creation, pool copies), the render thread
waiting on present (GPU / swapchain), or the render thread idle while the game had not submitted the next frame (guest CPU, system
calls, GPU sync waits, memory pool imports). Frames between "pacing off: loading" and "pacing on" count as loading.
It also counts BC7 cache reads by thread (render thread vs background jobs), the quick stand-ins, and the pipeline lines.
"""
import argparse
import collections
import re

NUM = r'([0-9.]+)'


def num(rx, s, d=0.0):
    m = re.search(rx, s)
    return float(m.group(1)) if m else d


def parse(path):
    loading = False
    rows, windows = [], []
    reads = collections.Counter()
    standins = 0
    pipe_last = prewarm_last = gpl_last = texprep_last = None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'pacing off: loading' in line:
                loading = True
            elif 'nvn.replay: pacing on' in line:
                loading = False
            if 'data from cache' in line:
                m = re.search(r'\bt\d+ (nvn-[a-z]+)\]', line)
                reads[m.group(1) if m else '?'] += 1
            if 'quick stand-in' in line:
                standins += 1
            if 'nvn.pipe: pipelines, last' in line:
                pipe_last = line.strip()
            elif 'nvn.prewarm: prewarm so far' in line:
                prewarm_last = line.strip()
            elif 'nvn.gpl: libraries so far' in line:
                gpl_last = line.strip()
            elif 'nvn.texprep: scheduled' in line:
                texprep_last = line.strip()
            if 'hitch: ' not in line:
                continue
            m = re.search(r'hitch: (\d+) frames in ([0-9.]+) s: mean ([0-9.]+) fps, 1% low ([0-9.]+) fps, worst ([0-9.]+) ms', line)
            if m:
                windows.append(tuple(float(x) for x in m.groups()))
                continue
            m = re.search(r'hitch: frame (\d+) took ([0-9.]+) ms \(t=([0-9.]+) s\)', line)
            if not m:
                continue
            r = dict(frame=int(m.group(1)), took=float(m.group(2)), t=float(m.group(3)), loading=loading)
            ev = {}
            dm = re.search(r'during it: (.*?); render thread', line)
            if dm:
                for k, v in re.findall(r'(\w+) \+(\d+)', dm.group(1)):
                    ev[k] = int(v)
            r['ev'] = ev
            r['idle'] = num(r'render thread ms: idle ' + NUM, line)
            r['replay'] = num(r'replay ' + NUM, line)
            r['uploads'] = num(r'uploads ' + NUM, line)
            r['pipe'] = num(r'pipeline creation ' + NUM, line)
            r['present'] = num(r'present ' + NUM + r' \(', line)
            r['sys'] = num(r'presenting thread in system calls ' + NUM, line)
            r['ipc'] = num(r'IPC ' + NUM, line)
            r['gpusync'] = num(r'GPU sync waits \(all threads\) ' + NUM, line)
            r['poolimp'] = num(r'memory pool imports ' + NUM, line)
            rows.append(r)
    return rows, windows, reads, standins, (pipe_last, prewarm_last, gpl_last, texprep_last)


def classify(r):
    T, ev = r['took'], r['ev']
    if r['replay'] >= 0.5 * T:
        if ev.get('bc7_cache_read_ms_render', 0) >= 0.3 * r['replay']:
            return 'render: BC7 cache reads on the render thread'
        if r['uploads'] >= 0.4 * r['replay']:
            return 'render: texture upload / cache read / transcode'
        if r['pipe'] >= 0.4 * r['replay']:
            return 'render: pipeline compile at the draw'
        if ev.get('bc7_transcodes') or ev.get('texture_uploads'):
            return 'render: texture prep outside "uploads" (image creation, pool prep)'
        if ev.get('write_faults') or ev.get('pool_mirror_copied_kb'):
            return 'render: pool mirror copies / write faults'
        return 'render: replay, other'
    if r['present'] >= 0.5 * T:
        return 'present back pressure (GPU / swapchain)'
    if r['ipc'] >= 0.3 * T:
        return 'game: IPC / file I/O'
    if r['sys'] >= 0.3 * T:
        return 'game: system-call waits'
    if r['gpusync'] >= 0.3 * T:
        return 'game: GPU sync waits'
    if r['poolimp'] >= 0.3 * T:
        return 'game: memory pool import (vkAllocateMemory)'
    if ev.get('texprep_staged_kb', 0) >= 1024 or ev.get('texture_uploads'):
        return 'game: stall while textures stream (texprep busy)'
    if ev.get('pipelines_created'):
        return 'game: stall while pipelines build in the background'
    return 'game: guest CPU, nothing tracked'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--min-ms', type=float, default=100.0)
    a = ap.parse_args()
    for path in a.logs:
        rows, windows, reads, standins, last = parse(path)
        print('=' * 24, path, '-', len(rows), 'slow-frame lines')
        for phase in ('gameplay', 'loading', 'all'):
            agg = {}
            for r in rows:
                if r['took'] < a.min_ms or (phase == 'gameplay' and r['loading']) or (phase == 'loading' and not r['loading']):
                    continue
                c = classify(r)
                x = agg.setdefault(c, [0, 0.0, 0.0, None])
                x[0] += 1
                x[1] += r['took']
                if r['took'] > x[2]:
                    x[2], x[3] = r['took'], r
            print(f'--- {phase} (frames over {a.min_ms:.0f} ms)')
            print(f"{'cause':<70} {'count':>6} {'total ms':>10} {'worst ms':>9}  worst at")
            for c, x in sorted(agg.items(), key=lambda kv: -kv[1][1]):
                print(f"{c:<70} {x[0]:>6} {x[1]:>10.0f} {x[2]:>9.0f}  t={x[3]['t']:.0f} s frame {x[3]['frame']}")
        tot = collections.Counter()
        for r in rows:
            for k in ('replay', 'uploads', 'pipe', 'idle', 'present', 'poolimp'):
                tot['ms.' + k] += r[k]
            for k, v in r['ev'].items():
                tot[k] += v
        print('--- sums over the slow frames:', ', '.join(f'{k} {v:.0f}' for k, v in sorted(tot.items())))
        print('--- BC7 cache reads by thread:', dict(reads), '| quick stand-ins logged:', standins)
        print(f'--- 1-s windows: {len(windows)}, mean under 20 fps: {sum(1 for w in windows if w[2] < 20)}, under 5 fps: {sum(1 for w in windows if w[2] < 5)}')
        for l in last:
            if l:
                print('---', l[:400])


if __name__ == '__main__':
    main()
