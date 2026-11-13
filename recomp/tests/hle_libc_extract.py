# SPDX-License-Identifier: GPL-3.0-or-later
"""Writes guest_fns.inc for hle_libc_test.cpp: the guest's own C string routines (the inline_replace targets of games/fefw/config.py,
plus the memcpy forwarders memmove tail-calls) from a lift WITHOUT opt / inline_replace, renamed g_<addr>. The functions are game
code: generate the file locally from your own dump's lift, never commit it.

    NXRECOMP_OPT=none python -m nxrecomp lift <exefs>/sdk <out>/sdk --name sdk --base 0x7500000000 --shards 50 --game generic --index 1
    python recomp/tests/hle_libc_extract.py <out>/sdk recomp/tests/guest_fns.inc
    (then build hle_libc_test.cpp like nxflags_test.cpp, with -I recomp/tests)
"""
import glob
import os
import re
import sys

WANT = ['75007bbc80', '750078df20', '750078df00', '750078e1e0', '75007a6c20', '75007bc140', '75007bc240']
head = re.compile(r'^void f_([0-9a-f]+)\(Ctx \*__restrict c\) \{$')
bodies = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], 'c_*.cpp'))):
    lines = open(f, encoding='utf-8').read().split('\n')
    i = 0
    while i < len(lines):
        m = head.match(lines[i])
        if m and m.group(1) in WANT:
            j = i
            while lines[j] != '}':
                j += 1
            bodies[m.group(1)] = '\n'.join(lines[i:j + 1])
            i = j + 1
        else:
            i += 1
missing = [w for w in WANT if w not in bodies]
if missing:
    sys.exit('missing: ' + ' '.join(missing))
text = '\n\n'.join(bodies[w] for w in WANT)
names = sorted(set(re.findall(r'\bf_([0-9a-f]+)\(c\)', text)) | set(WANT))
out = ['void g_%s(Ctx *__restrict c);' % n for n in names]
out.append(re.sub(r'\bf_([0-9a-f]{10})\b', r'g_\1', text))
open(sys.argv[2], 'w', newline='\n').write('\n'.join(out) + '\n')
print('wrote', sys.argv[2])
