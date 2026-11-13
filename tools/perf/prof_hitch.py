# SPDX-License-Identifier: GPL-3.0-or-later
"""Per slow frame stacks from the sampling profiler in slow-frame mode (`[prof] hitch_ms`), joined with the hitch lines of the log.

    python -I tools/perf/prof_hitch.py prof_samples.folded [--log fefw.log] [--map fefw_rt.map] [--min-ms 100] [--top 6]
                                                            [--threads main,nvn-render] [--frame N] [--summary]

The folded file's lines are `<label>;<thread>;<outermost>;...;<leaf> <count>`; in slow-frame mode the label is `f<frame>`. Frames of
fefw_rt.dll are written as `fefw_rt.dll!+0x<offset>` and resolved here with the DLL's linker map (/MAP), lifted game code is
`f_<guest address>`. For each slow frame (over --min-ms in the log, or every labelled frame without a log) and each thread it prints the
share of samples spent waiting in the kernel and the functions that hold most samples (inclusive: anywhere on the stack; the leaf-most
runtime function of each sample is also counted as "self"). --summary adds the same tables over all slow frames together.
"""
import argparse
import collections
import re


def load_map(path):
    pref = 0x180000000
    syms = []
    rx = re.compile(r'^\s*([0-9a-fA-F]{4}):([0-9a-fA-F]{8})\s+(\S+)\s+([0-9a-fA-F]{16})\s')
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.search(r'Preferred load address is ([0-9a-fA-F]+)', line)
            if m:
                pref = int(m.group(1), 16)
                continue
            m = rx.match(line)
            if m and m.group(1) != '0000':
                syms.append((int(m.group(4), 16) - pref, m.group(3)))
    syms.sort()
    return [s[0] for s in syms], [s[1] for s in syms]


def demangle(n):
    if not n.startswith('?'):
        return n
    body = n[1:].split('@@', 1)[0]
    parts = [p for p in body.split('@') if p]
    if not parts:
        return n
    if parts[0].startswith('?'):          # operators and special names: keep them short
        parts[0] = parts[0].lstrip('?')
    return '::'.join(reversed(parts))


class Resolver:
    def __init__(self, mp):
        self.addrs, self.names = load_map(mp) if mp else ([], [])
        self.cache = {}

    def __call__(self, fr):
        if fr in self.cache:
            return self.cache[fr]
        out = fr
        m = re.match(r'fefw_rt\.dll!\+0x([0-9a-fA-F]+)$', fr)
        if m and self.addrs:
            import bisect
            off = int(m.group(1), 16)
            i = bisect.bisect_right(self.addrs, off) - 1
            if i >= 0 and off - self.addrs[i] < (1 << 20):
                out = demangle(self.names[i])
        self.cache[fr] = out
        return out


def is_system(fr):
    return fr.startswith(('ntdll.dll', 'KERNELBASE.dll', 'KERNEL32.DLL', 'win32u.dll', 'nvoglv64', 'nvgpucomp64', 'vulkan-1', 'unknown')) or fr == '[wait]'


def hitch_lines(log):
    out = {}
    if not log:
        return out
    rx = re.compile(r'hitch: frame (\d+) took ([0-9.]+) ms \(t=([0-9.]+) s\)')
    loading = False
    with open(log, encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'pacing off: loading' in line:
                loading = True
            elif 'nvn.replay: pacing on' in line:
                loading = False
            m = rx.search(line)
            if m:
                out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), loading)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folded')
    ap.add_argument('--log')
    ap.add_argument('--map')
    ap.add_argument('--min-ms', type=float, default=100.0)
    ap.add_argument('--top', type=int, default=6)
    ap.add_argument('--threads', default='')
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    res = Resolver(a.map)
    want_threads = set(t for t in a.threads.split(',') if t)
    hl = hitch_lines(a.log)
    # frame -> thread -> [total, waiting, Counter(incl), Counter(self)]
    data = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, collections.Counter(), collections.Counter()]))
    with open(a.folded, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            stack, _, cnt = line.rstrip('\n').rpartition(' ')
            try:
                cnt = int(cnt)
            except ValueError:
                continue
            parts = stack.split(';')
            if len(parts) < 3 or not re.match(r'f\d+$', parts[0]):
                continue
            fr_n = int(parts[0][1:])
            thr = parts[1]
            if want_threads and thr not in want_threads:
                continue
            frames = [res(x) for x in parts[2:]]
            d = data[fr_n][thr]
            d[0] += cnt
            if frames and frames[-1] == '[wait]':
                d[1] += cnt
            seen = set()
            for x in frames:
                if x not in seen and not is_system(x):
                    d[2][x] += cnt
                    seen.add(x)
            leaf = next((x for x in reversed(frames) if not is_system(x)), None)
            if leaf:
                d[3][leaf + (' [wait]' if frames[-1] == '[wait]' else '')] += cnt
    frames = sorted(data)
    if a.frame:
        frames = [a.frame] if a.frame in data else []
    elif hl:
        frames = [n for n in frames if n in hl and hl[n][0] >= a.min_ms]
    summ = collections.defaultdict(lambda: [0, 0, collections.Counter(), collections.Counter()])
    for n in frames:
        took = hl.get(n)
        hdr = f'=== frame {n}' + (f': {took[0]:.0f} ms at t={took[1]:.0f} s{" (loading)" if took[2] else ""}' if took else '')
        print(hdr)
        for thr, d in sorted(data[n].items(), key=lambda kv: -kv[1][0]):
            busy = d[0] - d[1]
            if d[0] < 3:
                continue
            print(f'  {thr}: {d[0]} samples, {100.0 * d[1] / d[0]:.0f}% waiting')
            print('    self: ' + '; '.join(f'{k} {v}' for k, v in d[3].most_common(a.top)))
            print('    incl: ' + '; '.join(f'{k} {v}' for k, v in d[2].most_common(a.top)))
            s = summ[thr]
            s[0] += d[0]
            s[1] += d[1]
            s[2].update(d[2])
            s[3].update(d[3])
    if a.summary:
        print('=== all listed frames together')
        for thr, d in sorted(summ.items(), key=lambda kv: -kv[1][0]):
            if d[0] < 3:
                continue
            print(f'  {thr}: {d[0]} samples, {100.0 * d[1] / d[0]:.0f}% waiting')
            print('    self: ' + '; '.join(f'{k} {v}' for k, v in d[3].most_common(a.top * 2)))
            print('    incl: ' + '; '.join(f'{k} {v}' for k, v in d[2].most_common(a.top * 2)))


if __name__ == '__main__':
    main()
