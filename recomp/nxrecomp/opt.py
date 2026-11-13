# SPDX-License-Identifier: GPL-3.0-or-later
"""Function-level optimisation of the lifter's C++ (program.lift_function output): guest registers and NZCV in locals.

The plain output keeps every guest register in the context (`c->x[n]`, `c->v[n]`, `c->N` ...). The host compiler can keep those
in host registers only between two calls, and not at all across an atomic access (an acquire or seq_cst access orders every
memory access, the context included, so each one is reloaded and stored around it). NZCV fares worst: every flag-setting
instruction writes all four flags to the context, and the stores (and the work behind them) survive whenever a call, an atomic
or a loop back edge follows before the next flag-setting instruction.

This pass rewrites one lifted function so that it works on locals (gx<n>, gv<n>, gf) and meets the context only where something
outside the function can look at it:
  - entry: every register the function names is loaded from the context (unused loads cost nothing after optimisation);
  - before a call (BL, BLR, tail calls, host imports), a return and an SVC: the registers that may have changed since the last such
    point (a forward data-flow over the function's control-flow graph) are stored;
  - after a call: x0..x18, x30 and v0..v31 are read back; x19..x29 and SP are kept, because the callee preserves them (AAPCS64:
    callee-saved, which every compiled guest function honours: its epilogue writes back exactly what the caller had);
  - after an SVC every X register is read back (the kernel writes outputs, the rest are its inputs unchanged); after an inline
    site callout or a mid-function hook, everything (flags included);
  - helpers that index the vector registers at run time (v_tbl, v_ldn, v_stn) see the vector registers synced around them.

NZCV ('flags' mode) follows the AAPCS64 too: the flags are not preserved across a call and carry no argument or result, so they live
in the local `gf` only and are never stored. The pass checks that per function with a backward liveness over the same graph: a
function that may read the flags before writing them on entry, or after a call returns, is reported in the result (`flag_entry`,
`flag_after_call`) and keeps the context flags (`gf` stored at every call and return, read back after calls), so its own behaviour
is unchanged even then. Without 'flags' mode, the flags are local but always written back and read back like that.

The result is the same guest behaviour: the same values reach guest memory, calls see the same registers, and on return the context
holds the same registers as before (in 'flags' mode except NZCV, which the ABI leaves undefined there). Functions the pass does
not understand (an unknown helper that takes the context, optnone functions) are left as they are.

Status (perf/codegen, 2026-11): only the flags part is used (game config opt = ('flags',); program.lift_function calls optimize()
with xregs = vregs = False). It passes the end-to-end oracle ('flagsafe': 30 of 30 identical; 'flags': only the final NZCV differs)
and recomp/tests/nxflags_test.cpp. The register modes ('xregs', 'vregs') are EXPERIMENTAL and not correct yet (e2e: 3 of 30 cases
differ, around jump tables and tail calls), and the code they give is larger: loading every register at entry and every
caller-saved one right after each call stretches live ranges, and the host compiler spills them (it cannot reload a local from the
context the way it reloads a context field). A version that loads lazily (at the first use after a sync point) would be needed.
"""
import re

CALLEE_SAVED = frozenset(list(range(19, 30)) + [31])
CALLER_X = tuple(i for i in range(31) if i not in CALLEE_SAVED)          # x0..x18, x30
FLAGS = 'NZCV'

_ASSIGN = r'\s*(?:[-+*/%|&^]|<<|>>)?=(?!=)'
TOK = re.compile(r'''
 (?P<xa>&c->x\[(?P<xan>\d+)\])
|(?P<xw>c->x\[(?P<xwn>\d+)\])(?=''' + _ASSIGN + r''')
|(?P<xr>c->x\[(?P<xrn>\d+)\])
|(?P<va>&c->v\[(?P<van>\d+)\])
|(?P<vw>c->v\[(?P<vwn>\d+)\])(?=(?:\.\w+\[[^\]]*\])?''' + _ASSIGN + r''')
|(?P<vr>c->v\[(?P<vrn>\d+)\])
|(?P<ff>c->(?P<ffn>[NZCV])\b)
|(?P<fw>\b(?:nx::)?(?:adds64|adds32|logic64|logic32|set_nzcv|subs64|subs32|fcmp)\(c,)
|(?P<fr>\b(?:nx::)?cond\(c,)
|(?P<fg>\b(?:nx::)?get_nzcv\(c\))
|(?P<call>\b(?:f_[0-9a-f]+|h_\w+)\(c\);|\bcall_indirect\(c,\ (?P<ciarg>t|jt|c->x\[\d+\])\);|\bhost_import\(c,\ NX_IMPORT_ID\(\d+,\ \d+\)\);)
|(?P<svc>\bguest_svc\(c,\ [^;]*?\);)
|(?P<site>\bnx_site_inline\(c,\ 0x[0-9a-f]+ULL\);)
|(?P<mid>\bif\ \(nx_site_midhook\(c,\ 0x[0-9a-f]+ULL\)\)\ return;)
|(?P<vsync>\bv_(?:tbl|ldn|stn)\(c,\ [^;]*?\);)
|(?P<pure>\b(?:guest_mrs|guest_msr|stx64|stx32|stx16|stx8|stxp32|stxp64|ldxp128)\(c,\ )
|(?P<goto>\bgoto\ (?P<gl>L_\w+);)
|(?P<ret>\breturn;)
|(?P<other>\b\w+\(c[,)]|>\(c[,)])
''', re.X)
GOTO = re.compile(r'\bgoto (L_\w+);')
# 'vstore': `vzero(&V); V.f[0] = t;` (a scalar result into element 0, the rest cleared) as one whole-register store (cxx.h lane0_*)
VSTORE = re.compile(r'\bvzero\(&c->v\[(\d+)\]\); c->v\[\1\]\.(f|df|w|d)\[0\] = t;')
VSTORE_REP = r'c->v[\1] = nx::lane0_\2(t);'
VSET = re.compile(r'\bvzero\(&c->v\[(\d+)\]\); vset\(&c->v\[\1\], 2, 0, t\);')
VSET_REP = r'c->v[\1] = nx::lane0_w((uint32_t)t);'
VSTORE_D = re.compile(r'\bvzero\(&c->v\[(\d+)\]\); vset\(&c->v\[\1\], 3, 0, t\);')
VSTORE_D_REP = r'c->v[\1] = nx::lane0_d((uint64_t)t);'
LABEL = re.compile(r'^(L_\w+): ')
HEAD = re.compile(r'^(?:__attribute__\(\(section\("[^"]*"\)\)\) )?((?:NX_OPTNONE |__attribute__\(\(noinline\)\) )?)void ([fu])_([0-9a-f]+)\(Ctx \*__restrict c(, uint64_t rp)?\) \{$')


class Skip(Exception):
    pass


def _xs(regs):
    return sorted(regs)


class Func:
    """One lifted function, split into nodes (one per statement line; the rp and dispatch switches one node each)."""

    def __init__(self, lines, vregs=True, xregs=True, vstore=False):
        self.head = lines[0]
        self.pre = []                 # lines before the first node: `c->fn = ...;`, the entry hook
        nodes = []                    # [label or None, text, is_block]
        i, n = 1, len(lines) - 1      # lines[-1] == '}'
        while i < n and (lines[i].startswith('  c->fn = ') or lines[i].startswith('  if (nx_hook_entry(c, ')):
            self.pre.append(lines[i])
            i += 1
        while i < n:
            ln = lines[i]
            if ln.startswith('  if (rp) switch (rp) {') or ln.startswith('L_dispatch: switch (jt) {'):
                j = i
                while not lines[j].rstrip().endswith('}') or j == i:
                    j += 1
                    if j >= n:
                        raise Skip('unterminated switch')
                label = 'L_dispatch' if ln.startswith('L_dispatch') else None
                text = ' '.join(x.strip() for x in lines[i:j + 1])
                if label:
                    text = text[len('L_dispatch: '):]
                nodes.append([label, text, True])
                i = j + 1
                continue
            m = LABEL.match(ln)
            if m:
                nodes.append([m.group(1), ln[m.end():], False])
            elif ln.startswith('  '):
                nodes.append([None, ln[2:], False])
            else:
                raise Skip('line shape: ' + ln[:40])
            i += 1
        if vstore:
            for nd in nodes:
                nd[1] = VSTORE_D.sub(VSTORE_D_REP, VSTORE.sub(VSTORE_REP, VSET.sub(VSET_REP, nd[1])))
        self.nodes = nodes
        self.index = {lab: k for k, (lab, _, _) in enumerate(nodes) if lab}
        self.toks = [[m for m in TOK.finditer(t) if (vregs or m.lastgroup not in ('va', 'vw', 'vr', 'vsync')) and (xregs or m.lastgroup not in ('xa', 'xw', 'xr'))] for _, t, _ in nodes]
        for toks in self.toks:
            for m in toks:
                if m.lastgroup == 'other':
                    raise Skip('helper ' + m.group(0))
        self.succ = []
        for k, (lab, text, block) in enumerate(nodes):
            s = [self.index[g] for g in GOTO.findall(text) if g in self.index]
            if any(g not in self.index for g in GOTO.findall(text)):
                raise Skip('goto to an unknown label')
            if not self._terminal(text, block) and k + 1 < len(nodes):
                s.append(k + 1)
            self.succ.append(sorted(set(s)))

    @staticmethod
    def _terminal(text, block):
        t = text.rstrip()
        if block:
            return not t.startswith('if (rp)')
        if t.startswith('if ('):
            return False
        if t.startswith('goto '):
            return True
        return bool(re.search(r'(?:^|; |\{ )return;(?: \})*$', t) or re.search(r'(?:^|; |\{ )goto L_\w+;(?: \})*$', t))


def _uses(func):
    xs, vs = set(), set()
    for toks in func.toks:
        for m in toks:
            g = m.lastgroup
            if g in ('xa', 'xw', 'xr'):
                xs.add(int(m.group(g + 'n')))
            elif g in ('va', 'vw', 'vr'):
                vs.add(int(m.group(g + 'n')))
            elif g == 'call' and m.group('ciarg') and m.group('ciarg').startswith('c->x'):
                xs.add(int(m.group('ciarg')[5:-1]))
    return xs, vs


CCMP = re.compile(r'^if \(cond\(c, \d+\)\) [^;]*\b(?:adds64|adds32|fcmp)\(c, [^;]*; else set_nzcv\(c, \d+\);$')


def _flag_effects(toks, text):
    """(gen, kill) of one node for the flag liveness, in token order: gen = a flag read before any write or call."""
    gen = kill = False
    partial = 'if (' in text and not CCMP.match(text.strip())
    for m in toks:
        g = m.lastgroup
        if kill:
            break
        if g in ('ff', 'fr', 'fg'):
            gen = True
        elif g == 'fw':
            # a write: adds/logic/set_nzcv/fcmp; ccmp is `if (cond(c, k)) adds(...); else set_nzcv(...)`: read first, then a write on
            # both paths. A writer inside any other `if` may be partial: not taken as a kill (conservative).
            kill = not partial
        elif g == 'call':
            kill = True
    return gen, kill


def optimize(text, flags_abi=True, vregs=True, keep_cs=True, xregs=True, vstore=False):
    """-> (new text, info dict). info: 'skipped' (reason) or 'flag_entry' / 'flag_after_call' (bool) and counts."""
    lines = text.split('\n')
    m = HEAD.match(lines[0])
    if not m or lines[-1] != '}':
        return text, {'skipped': 'shape'}
    if m.group(1).startswith('NX_OPTNONE'):
        return text, {'skipped': 'optnone'}
    try:
        func = Func(lines, vregs, xregs, vstore)
    except Skip as e:
        return text, {'skipped': str(e)}
    N = len(func.nodes)
    pre = list(func.pre)
    # the jt / lr0 header lines are ordinary nodes before the first label; keep them in place (they declare locals)
    xs, vs = _uses(func)
    if not xregs:
        xs = set()

    # ---- flag liveness (backward): any flag live at the start of each node
    eff = [_flag_effects(t, func.nodes[k][1]) for k, t in enumerate(func.toks)]
    live_in = [False] * N
    changed = True
    while changed:
        changed = False
        for k in range(N - 1, -1, -1):
            out = any(live_in[s] for s in func.succ[k])
            gen, kill = eff[k]
            v = gen or (out and not kill)
            if v != live_in[k]:
                live_in[k] = v
                changed = True
    flag_entry = live_in[0] if N else False
    flag_after_call = False
    for k, toks in enumerate(func.toks):
        out = any(live_in[s] for s in func.succ[k])
        for i, mt in enumerate(toks):
            if mt.lastgroup != 'call':
                continue
            after = out
            for mt2 in toks[i + 1:]:
                if mt2.lastgroup in ('ff', 'fr', 'fg'):
                    after = True
                    break
                if mt2.lastgroup in ('fw', 'call', 'ret'):
                    after = False
                    break
            flag_after_call = flag_after_call or after
    safe_flags = (not flags_abi) or flag_entry or flag_after_call

    # ---- registers (forward): which are held in their local (L) and which of those may differ from the context (D)
    # A register enters L at its first access after a sync point: the context value is loaded at the start of that segment (the
    # statement, or the part of it after the sync), whether the access reads or writes (a write may be partial: a lane, a compound
    # assignment; a load nobody needs is dropped by the compiler). At a join, a register in L on any incoming edge is in L, and the
    # edges where it is not get the load (at the goto, or at the end of a falling-through node). D is the union over the edges. At a
    # sync point the registers in D are stored; after a call the caller-saved ones leave L (x19..x29 and SP stay with keep_cs).
    XREG = {'xa': 'xan', 'xw': 'xwn', 'xr': 'xrn'}
    VREG = {'va': 'van', 'vw': 'vwn', 'vr': 'vrn'}
    WRITES = ('xa', 'xw', 'va', 'vw')
    after_call_keep = frozenset('x%d' % i for i in CALLEE_SAVED) if keep_cs else frozenset()
    rkey = lambda r: (r[0], int(r[1:]) if r != 'f' else 0)

    def stores(regs):
        s = []
        for r in sorted(regs, key=rkey):
            if r == 'f':
                if safe_flags:
                    s.append('c->N = gf.N; c->Z = gf.Z; c->C = gf.C; c->V = gf.V;')
            elif r[0] == 'x':
                s.append('c->x[%s] = gx%s;' % (r[1:], r[1:]))
            else:
                s.append('c->v[%s] = gv%s;' % (r[1:], r[1:]))
        return ' '.join(s)

    def loads(regs):
        s = []
        for r in sorted(regs, key=rkey):
            if r == 'f':
                s.append('gf.N = c->N; gf.Z = c->Z; gf.C = c->C; gf.V = c->V; gf.k = NXF_RAW;')
            elif r[0] == 'x':
                s.append('gx%s = c->x[%s];' % (r[1:], r[1:]))
            else:
                s.append('gv%s = c->v[%s];' % (r[1:], r[1:]))
        return ' '.join(s)

    def regname(mt):
        g = mt.lastgroup
        if g in XREG:
            return 'x' + mt.group(XREG[g])
        if g in VREG:
            return 'v' + mt.group(VREG[g])
        if g == 'call' and xregs and mt.group('ciarg') and mt.group('ciarg').startswith('c->x'):
            return 'x' + mt.group('ciarg')[5:-1]
        return None

    SYNC = ('call', 'svc', 'site', 'mid', 'vsync', 'ret')

    def walk(k, L, D, Lin, emit):
        """Run node k from (L, D). Returns (edges [(target, L, D)], L, D at the end, new text or None). Lin: the L of every node's
        start (for the loads on edges; None while the analysis runs)."""
        text = func.nodes[k][1]
        toks = func.toks[k]
        out, pos = [], 0
        L, D = set(L), set(D)
        # flags: always 'in their local'; 'f' in D means written since the last sync (stored in safe mode)
        edges = []
        # segment starts: the node start, and right after each sync token
        def segment_loads(i0):
            need = set()
            for mt in toks[i0:]:
                r = regname(mt)                  # (a call's register operand is read before the call)
                if r and r not in L:
                    need.add(r)
                if mt.lastgroup in SYNC:
                    break
            return need

        need = segment_loads(0)
        if emit and need:
            out.append(loads(need) + ' ')
        L |= need
        for i, mt in enumerate(toks):
            g = mt.lastgroup
            if emit:
                out.append(text[pos:mt.start()])
            rep = mt.group(0)
            r = regname(mt)
            if g in WRITES:
                D.add(r)
            if g == 'xa':
                rep = '&gx' + mt.group('xan')
            elif g in ('xw', 'xr'):
                rep = 'gx' + mt.group(XREG[g])
            elif g == 'va':
                rep = '&gv' + mt.group('van')
            elif g in ('vw', 'vr'):
                rep = 'gv' + mt.group(VREG[g])
            elif g == 'ff':
                rep = 'gf.' + mt.group('ffn')
            elif g == 'fw':
                D.add('f')
                rep = mt.group(0)[:-3] + '(&gf,'
            elif g == 'fr':
                rep = mt.group(0)[:-3] + '(&gf,'
            elif g == 'fg':
                rep = mt.group(0)[:-3] + '(&gf)'
            elif g == 'goto':
                t = func.index[mt.group('gl')]
                edges.append((t, frozenset(L), frozenset(D)))
                if emit:
                    fix = Lin[t] - L
                    if fix:
                        rep = '{ %s %s }' % (loads(fix), rep)
            elif g in SYNC:
                if g == 'call':
                    call = mt.group(0)
                    if r:
                        call = 'call_indirect(c, gx%s);' % r[1:]
                    keep = {x for x in L if x in after_call_keep}
                    rep = '{ %s %s %s }' % (stores(D), call, loads({'f'}) if safe_flags else '')
                    L, D = keep, set()
                elif g == 'svc':
                    rep = '{ %s %s }' % (stores(D), mt.group(0))
                    L, D = {x for x in L if x[0] != 'x'}, set()
                elif g in ('site', 'mid'):
                    fl = 'c->N = gf.N; c->Z = gf.Z; c->C = gf.C; c->V = gf.V;'
                    body = '%s %s %s %s' % (stores(D - {'f'}), fl, mt.group(0), loads({'f'}))
                    rep = ('{ %s }' % body) if g == 'site' else body
                    L, D = set(), set()
                elif g == 'vsync':
                    rep = '{ %s %s }' % (stores({x for x in D if x[0] == 'v'}), mt.group(0))
                    L, D = {x for x in L if x[0] != 'v'}, {x for x in D if x[0] != 'v'}
                elif g == 'ret':
                    rep = '{ %s return; }' % stores(D)
                if g != 'ret':
                    nd = segment_loads(i + 1)
                    if nd:
                        rep = rep + ' ' + loads(nd)
                        L |= nd
            if emit:
                out.append(rep)
            pos = mt.end()
        if emit:
            out.append(text[pos:])
        return edges, L, D, (''.join(out) if emit else None)

    N_ = N
    Lin = [None] * N_
    Din = [None] * N_
    Lin[0], Din[0] = frozenset(), frozenset()
    work = [0]
    while work:
        k = work.pop()
        edges, L, D, _ = walk(k, Lin[k], Din[k], None, False)
        text, block = func.nodes[k][1], func.nodes[k][2]
        if k + 1 < N_ and not Func._terminal(text, block):
            edges.append((k + 1, frozenset(L), frozenset(D)))
        for t, l2, d2 in edges:
            nl = l2 if Lin[t] is None else Lin[t] | l2
            nd = d2 if Din[t] is None else Din[t] | d2
            if nl != Lin[t] or nd != Din[t]:
                Lin[t], Din[t] = frozenset(nl), frozenset(nd)
                work.append(t)
    for k in range(N_):
        if Lin[k] is None:               # unreachable in this graph (dead code after a terminator): nothing held, nothing pending
            Lin[k], Din[k] = frozenset(), frozenset()

    # ---- emit
    decl = []
    if xs:
        decl.append('  uint64_t ' + ', '.join('gx%d = c->x[%d]' % (i, i) for i in _xs(xs)) + ';')
    for i in _xs(vs):
        decl.append('  V128 gv%d = c->v[%d];' % (i, i))
    decl.append('  NxFlags gf = {c->N, c->Z, c->C, c->V};')
    out = [lines[0]] + pre + decl
    for k, (lab, text, block) in enumerate(func.nodes):
        _, L, D, new = walk(k, Lin[k], Din[k], Lin, True)
        if k + 1 < N_ and (k + 1) in func.succ[k] and not Func._terminal(text, block):
            fix = Lin[k + 1] - L
            if fix:
                new = new + ' ' + loads(fix)
        if lab == 'L_dispatch':
            out.append('L_dispatch: ' + new)
        elif lab:
            out.append('%s: %s' % (lab, new))
        else:
            out.append('  ' + new)
    out.append('}')
    info = {'flag_entry': flag_entry, 'flag_after_call': flag_after_call, 'safe_flags': safe_flags, 'x': len(xs), 'v': len(vs)}
    return '\n'.join(out), info


def optimize_shard(src, flags_abi=True, stats=None, vregs=True, keep_cs=True, xregs=True, vstore=False):
    """The same on a whole generated c_NNN.cpp (for experiments on existing lifter output): every multi-line function."""
    out, i = [], 0
    lines = src.split('\n')
    while i < len(lines):
        ln = lines[i]
        if HEAD.match(ln):
            j = i
            while lines[j] != '}':
                j += 1
            new, info = optimize('\n'.join(lines[i:j + 1]), flags_abi, vregs, keep_cs, xregs, vstore)
            if stats is not None:
                key = info.get('skipped') or ('safe' if info.get('safe_flags') else 'ok')
                stats[key] = stats.get(key, 0) + 1
                if info.get('flag_entry'):
                    stats.setdefault('flag_entry_fns', []).append(ln.split('(')[0].split()[-1])
                if info.get('flag_after_call'):
                    stats.setdefault('flag_after_call_fns', []).append(ln.split('(')[0].split()[-1])
            out.append(new)
            i = j + 1
        else:
            out.append(ln)
            i += 1
    return '\n'.join(out)
