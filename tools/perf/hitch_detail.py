# SPDX-License-Identifier: GPL-3.0-or-later
"""Finer classification of a fefw.log's slow frames than hitch_report.py: one row per cause, with the frame's main
contributor named, and the BC7 cache reads of the render thread placed inside each frame by their log time (logs from
builds without the `bc7_cache_reads_render` counter, e.g. upstream 0.1.7).

    python -I tools/perf/hitch_detail.py fefw.log [--min-ms 100] [--warmup-s 30] [--list]

A frame spans [t - took, t] (t = the hitch line's "t=" value, seconds since the monitor started, which is the log clock
within a few ms). Causes, in the order they are tested:
  render: BC7 cache reads        replay >= half the frame and render-thread cache reads inside the frame
  render: uploads (no disk)      replay >= half, uploads >= 40% of replay, no render-thread cache reads
  render: write-watch arming     write-watch arming or write-fault handler >= 30% of the frame
  render: pool import            memory pool imports >= 30% of the frame while the render thread was busy
  render: image/pool prep        replay >= half, textures uploaded or transcoded, uploads < 40% of replay
  render: pool mirror copies     replay >= half, pool mirror copies, nothing else
  present back pressure          present >= half the frame
  game: GPU sync waits           GPU sync waits >= 30%
  game: IPC / file reads         presenting thread IPC >= 30%
  game: waits on guest threads   presenting thread system-call waits >= 30%
  game: pool import              memory pool imports >= 30%
  game: texprep busy             texprep staged >= 1 MB or uploads in the frame
  game: pipelines building       pipelines created in the frame
  game: guest CPU                the rest
"""
import argparse
import bisect
import collections
import re

NUM = r'([0-9.]+)'


def num(rx, s, d=0.0):
    m = re.search(rx, s)
    return float(m.group(1)) if m else d


def parse(path):
    loading = False
    rows, reads, freezes = [], collections.defaultdict(list), []
    rx_t = re.compile(r'^\[\s*([0-9.]+) \w t\d+ ([^\]]*)\]')
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'pacing off: loading' in line:
                loading = True
            elif 'nvn.replay: pacing on' in line:
                loading = False
            m = rx_t.match(line)
            if not m:
                continue
            ts, thr = float(m.group(1)), m.group(2)
            if 'data from cache' in line:
                reads[thr].append(ts)
                continue
            if 'watchdog: FREEZE' in line:
                freezes.append(ts)
                continue
            m = re.search(r'hitch: frame (\d+) took ([0-9.]+) ms \(t=([0-9.]+) s\)', line)
            if not m:
                continue
            r = dict(frame=int(m.group(1)), took=float(m.group(2)), t=float(m.group(3)), loading=loading, line=line.strip())
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
            r['backp'] = num(r'present back pressure ' + NUM, line)
            r['sys'] = num(r'presenting thread in system calls ' + NUM, line)
            r['ipc'] = num(r'IPC ' + NUM, line)
            r['waits'] = num(r'\(?(?:IPC [0-9.]+, )?waits ' + NUM, line)
            r['gpusync'] = num(r'GPU sync waits \(all threads\) ' + NUM, line)
            r['poolimp'] = num(r'memory pool imports ' + NUM, line)
            r['wfault'] = num(r'write-fault handler ' + NUM, line)
            r['wwarm'] = num(r'write-watch arming ' + NUM, line)
            r['gap'] = num(r'longest gap between presents ' + NUM, line)
            rows.append(r)
    for v in reads.values():
        v.sort()
    return rows, reads, freezes


def render_reads(reads, t0, t1):
    v = reads.get('nvn-render', [])
    return bisect.bisect_right(v, t1) - bisect.bisect_left(v, t0)


def classify(r):
    T, ev = r['took'], r['ev']
    busy = r['replay'] >= 0.5 * T
    if busy and (r['rreads'] > 0 or ev.get('bc7_cache_reads_render', 0) > 0) and (r['uploads'] >= 0.2 * r['replay'] or r['rreads'] >= 3):
        return 'render: BC7 cache reads on the render thread'
    if busy and r['uploads'] >= 0.4 * r['replay']:
        return 'render: texture uploads, no disk (staging, standins, transcodes)'
    if (r['wwarm'] + r['wfault']) >= 0.3 * T and r['replay'] >= 0.3 * T:
        return 'render: write-watch arming / write-fault handler'
    if busy and r['poolimp'] >= 0.3 * T:
        return 'render: memory pool import (vkAllocateMemory)'
    if busy and (ev.get('bc7_transcodes') or ev.get('texture_uploads') or ev.get('texture_standins')):
        return 'render: image creation / pool prep outside uploads'
    if busy and (ev.get('pool_mirror_copied_kb') or ev.get('write_faults')):
        return 'render: pool mirror copies'
    if busy:
        return 'render: replay, other'
    if r['present'] >= 0.5 * T:
        return 'present back pressure (GPU / swapchain)'
    if r['gpusync'] >= 0.3 * T:
        return 'game: GPU sync waits'
    if r['ipc'] >= 0.3 * T:
        return 'game: IPC / file reads'
    if (r['wwarm'] + r['wfault']) >= 0.3 * T:
        return 'game: write-fault handler / arming'
    if r['sys'] >= 0.3 * T:
        return 'game: waits on other guest threads (svc waits)'
    if r['poolimp'] >= 0.3 * T:
        return 'game: memory pool import (vkAllocateMemory)'
    if ev.get('texprep_staged_kb', 0) >= 1024 or ev.get('texture_uploads'):
        return 'game: guest CPU while textures stream'
    if ev.get('pipelines_created'):
        return 'game: guest CPU while pipelines build'
    return 'game: guest CPU, nothing tracked'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--min-ms', type=float, default=100.0)
    ap.add_argument('--warmup-s', type=float, default=30.0, help='frames before this are boot / warm-up')
    ap.add_argument('--list', action='store_true', help='print every frame with its cause')
    a = ap.parse_args()
    rows, reads, freezes = parse(a.log)
    for r in rows:
        r['rreads'] = render_reads(reads, r['t'] - r['took'] / 1000.0 - 0.05, r['t'] + 0.05)
        r['cause'] = classify(r)
        r['phase'] = 'boot' if r['t'] < a.warmup_s else ('loading' if r['loading'] else 'gameplay')
    print(f'{a.log}: {len(rows)} slow-frame lines, render-thread cache reads {len(reads.get("nvn-render", []))}, '
          f'texprep cache reads {len(reads.get("nvn-texprep", []))}, watchdog freezes {len(freezes)} at {", ".join(f"{t:.0f}" for t in freezes)} s')
    for phase in ('gameplay', 'loading', 'boot'):
        for lo, hi in ((a.min_ms, 1e12), (50.0, a.min_ms)):
            agg = collections.OrderedDict()
            for r in rows:
                if r['phase'] != phase or not (lo <= r['took'] < hi):
                    continue
                x = agg.setdefault(r['cause'], [0, 0.0, 0.0, None, 0])
                x[0] += 1
                x[1] += r['took']
                x[4] += r['rreads']
                if r['took'] > x[2]:
                    x[2], x[3] = r['took'], r
            if not agg:
                continue
            print(f'--- {phase}, frames {lo:.0f}..{"" if hi > 1e11 else f"{hi:.0f}"} ms')
            print(f"| {'cause':<66} | {'count':>5} | {'total ms':>9} | {'worst ms':>8} | {'worst at':<18} | {'rt reads':>8} |")
            print('|' + '-' * 68 + '|------:|----------:|---------:|' + '-' * 20 + '|---------:|')
            for c, x in sorted(agg.items(), key=lambda kv: -kv[1][1]):
                print(f"| {c:<66} | {x[0]:>5} | {x[1]:>9.0f} | {x[2]:>8.0f} | t={x[3]['t']:<7.0f} f{x[3]['frame']:<8} | {x[4]:>8} |")
    if a.list:
        for r in rows:
            if r['took'] >= a.min_ms:
                print(f"{r['phase']:<8} t={r['t']:>7.1f} f{r['frame']:<7} {r['took']:>8.1f} ms  rtreads {r['rreads']:>4}  {r['cause']}")


if __name__ == '__main__':
    main()
