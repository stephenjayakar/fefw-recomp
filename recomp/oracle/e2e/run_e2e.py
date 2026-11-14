# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end oracle: C/C++ test programs -> AArch64 ELF (clang) -> lifted by nxrecomp -> x64 exe (clang-cl + Ninja via
CMake) -> run; compared with Unicorn executing the original AArch64 code. PASS only if, for every case, every register
(x0..x30, SP, v0..v31, NZCV, TPIDR*) and every writable byte (data, bss, stack, TLS blocks) is identical.

    python run_e2e.py [--cases 01_int,03_fp] [--opts O2,O0] [--build D:/path/build] [--jobs N]

Each program is built at -O2 (EXPECT-INSN lines enforced: the program must really contain the instruction classes it
claims) and at -O0 (compile + link only). Exit code 0 = all cases identical.
"""
import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import build_prog          # noqa: E402
import unicorn_side        # noqa: E402
from nxrecomp.image import Image   # noqa: E402

NX_ENV = os.path.join(ROOT, 'scripts', 'nx_env.cmd')


def sh(cmd, **kw):
    return subprocess.run(['cmd', '/c', NX_ENV] + cmd, capture_output=True, text=True, **kw)


def read_x64(prefix, img):
    regs = {}
    for line in open(prefix + '.txt'):
        parts = line.split()
        if len(parts) == 2:                       # ("hit <addr> <count>" lines belong to oracle/hooks)
            regs[parts[0]] = parts[1]
    blob = open(prefix + '.bin', 'rb').read()
    mem, p = {}, 0
    for name, addr, size in unicorn_side.rw_regions(img):
        mem[name] = hashlib.sha256(blob[p:p + size]).hexdigest()
        p += size
    return regs, mem


# Registers the AAPCS64 leaves undefined after a return may differ when the lifter is told to rely on that (NXRECOMP_OPT, nxrecomp/opt.py):
# 'tregs' keeps x8..x17 in function locals (not written back at returns), 'flags' NZCV. Memory and every other register must match.
_OPT = set((os.environ.get('NXRECOMP_OPT') or '').split(','))
ABI_DEAD_X = set(range(8, 18)) if 'tregs' in _OPT else set()
ABI_DEAD_NZCV = 'flags' in _OPT


def compare(uc, regs, mem):
    diffs = []
    for i in range(31):
        if i in ABI_DEAD_X:
            continue
        if uc['x'][i] != int(regs['x%d' % i], 16):
            diffs.append('x%d uc=%x x64=%s' % (i, uc['x'][i], regs['x%d' % i]))
    if uc['sp'] != int(regs['sp'], 16):
        diffs.append('sp uc=%x x64=%s' % (uc['sp'], regs['sp']))
    for i in range(32):
        if int(uc['v'][i], 16) != int(regs['v%d' % i], 16):
            diffs.append('v%d uc=%s x64=%s' % (i, uc['v'][i], regs['v%d' % i]))
    if not ABI_DEAD_NZCV and uc['nzcv'] != int(regs['nzcv'], 16):
        diffs.append('nzcv uc=%x x64=%s' % (uc['nzcv'], regs['nzcv']))
    for name, h in uc['mem'].items():
        if h != mem.get(name):
            diffs.append('memory region %s differs' % name)
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='')
    ap.add_argument('--opts', default='O2,O0')
    ap.add_argument('--build', default=os.path.join(ROOT, 'build', 'e2e'))
    ap.add_argument('--jobs', type=int, default=max(2, (os.cpu_count() or 8) // 2))
    ap.add_argument('--uc-model', default='max')
    a = ap.parse_args()
    t0 = time.time()
    srcs = sorted(glob.glob(os.path.join(HERE, 'progs', '[0-9][0-9]_*.c')) + glob.glob(os.path.join(HERE, 'progs', '[0-9][0-9]_*.cpp')))
    if a.cases:
        want = a.cases.split(',')
        srcs = [s for s in srcs if any(os.path.basename(s).startswith(w) for w in want)]
    opts = ['-' + o for o in a.opts.split(',')]
    elfdir = os.path.join(a.build, 'elf')
    os.makedirs(elfdir, exist_ok=True)
    cases = []
    failed = []
    for s in srcs:
        name = os.path.splitext(os.path.basename(s))[0]
        for o in opts:
            case = '%s_%s' % (name, o[1:])
            elf = os.path.join(elfdir, case + '.elf')
            cmd = build_prog.compile_cmd(s, elf, o)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                failed.append((case, 'compile failed: ' + (r.stderr or r.stdout)[-400:]))
                continue
            if o == '-O2':
                ok, msg = build_prog.build(s, elf, o)
                if not ok:
                    failed.append((case, msg))
                    continue
            cases.append((case, elf))
    cm = os.path.join(a.build, 'cases.cmake')
    with open(cm, 'w') as f:
        f.write('set(E2E_CASE_NAMES %s)\n' % ' '.join(c for c, _ in cases))
        for c, e in cases:
            f.write('set(E2E_ELF_%s "%s")\n' % (c, e.replace('\\', '/')))
    bdir = os.path.join(a.build, 'cmake')
    r = sh(['cmake', '-S', os.path.join(HERE), '-B', bdir, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_TOOLCHAIN_FILE=' + os.path.join(ROOT, 'cmake', 'clang-cl.cmake').replace('\\', '/'),
            '-DE2E_CASES=' + cm.replace('\\', '/')])
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        return 2
    for old in glob.glob(os.path.join(bdir, 'run_*.exe')):
        os.remove(old)                                   # a stale exe must never be mistaken for this build's
    r = sh(['cmake', '--build', bdir, '-j', str(a.jobs), '--', '-k', '0'])
    built = {c: os.path.exists(os.path.join(bdir, 'run_%s.exe' % c)) for c, _ in cases}
    if not all(built.values()):
        print((r.stdout + r.stderr)[-6000:])

    def one(item):
        case, elf = item
        if not built[case]:
            return case, ['x64 build failed']
        try:
            uc = unicorn_side.run(elf, a.uc_model)
        except BaseException as e:             # noqa: BLE001
            return case, ['unicorn: %s' % e]
        if not uc['finished']:
            return case, ['unicorn did not reach the return address (pc=%x)' % uc['pc']]
        prefix = os.path.join(a.build, 'out_' + case)
        r = subprocess.run([os.path.join(bdir, 'run_%s.exe' % case), elf, prefix], capture_output=True, text=True)
        if r.returncode:
            return case, ['x64 run exit %d: %s' % (r.returncode, r.stderr.strip()[-300:])]
        img = Image.from_static_elf(elf)
        regs, mem = read_x64(prefix, img)
        return case, compare(uc, regs, mem)
    results = []
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        for case, diffs in ex.map(one, cases):
            results.append((case, diffs))
    passed = sum(1 for _, d in results if not d)
    for case, diffs in results:
        print('%-4s %s%s' % ('PASS' if not diffs else 'FAIL', case, '' if not diffs else '   ' + '; '.join(diffs[:6]) + (' ...(%d more)' % (len(diffs) - 6) if len(diffs) > 6 else '')))
    for case, msg in failed:
        print('FAIL %s   %s' % (case, msg))
    total = len(results) + len(failed)
    print('e2e: %d of %d cases identical (%d programs x %d optimisation levels, %.0fs)' % (passed, total, len(srcs), len(opts), time.time() - t0))
    return 0 if passed == total and total > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
