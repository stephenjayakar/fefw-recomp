/* SPDX-License-Identifier: GPL-3.0-or-later
 * Advanced SIMD helpers for lifted code. Included by ctx.h. Adapted from smo-recomp runtime/core/neon.h
 * (GPL-3.0-or-later). Written for correctness first: every helper works element by element. */
#ifndef NXRECOMP_NEON_H
#define NXRECOMP_NEON_H

/* x86-64 hosts: the single-precision vector forms (v_f3v, v_f3e, v_f1v) run the whole register through the host's packed SSE
 * instructions, which give the IEEE result of the scalar ones in every lane, and hand the instruction to the element-by-element
 * code (the *_ref functions, out of line) when a result lane is a NaN, so the Arm NaN rules stay exact. The fused forms need FMA3
 * (-mfma); without it they stay element by element. NX_NO_SIMD turns all of this off. */
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__SSE2__) && !defined(NX_NO_SIMD)
#include <immintrin.h>
#define NX_X86_SIMD 1
#endif
#define NX_HOT static inline __attribute__((always_inline))
#define NX_SLOW static __attribute__((noinline, cold))

/* Lane access through whole-register (16-byte) loads and stores: a lane written in place would be a 1..8-byte store that a later
 * 16-byte load of the same register can neither take from the store buffer (a store-forwarding stall on the host) nor see through
 * in the compiler (the register goes through memory). As one 16-byte load + insert + 16-byte store, the compiler keeps the register
 * in an XMM register across lane writes, and whole-register reads after them forward. */
typedef uint8_t nx_vu8 __attribute__((vector_size(16)));
typedef uint16_t nx_vu16 __attribute__((vector_size(16)));
typedef uint32_t nx_vu32 __attribute__((vector_size(16)));
typedef uint64_t nx_vu64 __attribute__((vector_size(16)));
static inline uint64_t vget(const V128 *v, int sz, int i) {
    switch (sz) {
    case 0: { nx_vu8 t; memcpy(&t, v, 16); return t[i]; }
    case 1: { nx_vu16 t; memcpy(&t, v, 16); return t[i]; }
    case 2: { nx_vu32 t; memcpy(&t, v, 16); return t[i]; }
    default: { nx_vu64 t; memcpy(&t, v, 16); return t[i]; }
    }
}
static inline void vset(V128 *v, int sz, int i, uint64_t x) {
    switch (sz) {
    case 0: { nx_vu8 t; memcpy(&t, v, 16); t[i] = (uint8_t)x; memcpy(v, &t, 16); break; }
    case 1: { nx_vu16 t; memcpy(&t, v, 16); t[i] = (uint16_t)x; memcpy(v, &t, 16); break; }
    case 2: { nx_vu32 t; memcpy(&t, v, 16); t[i] = (uint32_t)x; memcpy(v, &t, 16); break; }
    default: { nx_vu64 t; memcpy(&t, v, 16); t[i] = x; memcpy(v, &t, 16); break; }
    }
}
/* A whole register as one 16-byte store. */
static inline void nx_vst(V128 *d, V128 r) { nx_vu64 t; memcpy(&t, &r, 16); memcpy(d, &t, 16); }
static inline int64_t vsext(uint64_t x, int sz) {
    switch (sz) {
    case 0: return (int8_t)x;
    case 1: return (int16_t)x;
    case 2: return (int32_t)x;
    default: return (int64_t)x;
    }
}

/* AArch64 FMAX/FMIN propagate NaN; the NM variants return the number: exact versions in fpexact.h (fp_maxmin32 / 64). */

enum { F_ADD, F_SUB, F_MUL, F_DIV, F_MLA, F_MLS, F_MAX, F_MIN, F_MAXNM, F_MINNM, F_ABD,
       F_CMEQ, F_CMGE, F_CMGT, F_RECPS, F_RSQRTS, F_ADDP };

static inline float f3s(int op, float acc, float a, float b, uint32_t *mask, int *is_mask) {
    switch (op) {
    case F_ADD: return fp_add32(a, b);
    case F_SUB: return fp_sub32(a, b);
    case F_MUL: return fp_mul32(a, b);
    case F_DIV: return fp_div32(a, b);
    case F_MLA: return fp_fma32(a, b, acc);
    case F_MLS: return fp_fma32(fp_neg32(a), b, acc);
    case F_MAX: return fp_maxmin32(a, b, 0, 0);
    case F_MIN: return fp_maxmin32(a, b, 1, 0);
    case F_MAXNM: return fp_maxmin32(a, b, 0, 1);
    case F_MINNM: return fp_maxmin32(a, b, 1, 1);
    case F_ABD: return fp_abs32(fp_sub32(a, b));
    case F_RECPS: return fp_recps32(a, b);
    case F_RSQRTS: return fp_rsqrts32(a, b);
    case F_CMEQ: *is_mask = 1; *mask = a == b ? ~0u : 0; return 0;
    case F_CMGE: *is_mask = 1; *mask = a >= b ? ~0u : 0; return 0;
    case F_CMGT: *is_mask = 1; *mask = a > b ? ~0u : 0; return 0;
    }
    return 0;
}
static inline double f3d(int op, double acc, double a, double b, uint64_t *mask, int *is_mask) {
    switch (op) {
    case F_ADD: return fp_add64(a, b);
    case F_SUB: return fp_sub64(a, b);
    case F_MUL: return fp_mul64(a, b);
    case F_DIV: return fp_div64(a, b);
    case F_MLA: return fp_fma64(a, b, acc);
    case F_MLS: return fp_fma64(fp_neg64(a), b, acc);
    case F_MAX: return fp_maxmin64(a, b, 0, 0);
    case F_MIN: return fp_maxmin64(a, b, 1, 0);
    case F_MAXNM: return fp_maxmin64(a, b, 0, 1);
    case F_MINNM: return fp_maxmin64(a, b, 1, 1);
    case F_ABD: return fp_abs64(fp_sub64(a, b));
    case F_RECPS: return fp_recps64(a, b);
    case F_RSQRTS: return fp_rsqrts64(a, b);
    case F_CMEQ: *is_mask = 1; *mask = a == b ? ~0ull : 0; return 0;
    case F_CMGE: *is_mask = 1; *mask = a >= b ? ~0ull : 0; return 0;
    case F_CMGT: *is_mask = 1; *mask = a > b ? ~0ull : 0; return 0;
    }
    return 0;
}

/* Floating-point "three same": n elements, upper part of the destination cleared. */
NX_SLOW void v_f3v_ref(V128 *d, const V128 *pn, const V128 *pm, int op, int dbl, int n) {
    V128 a = *pn, b = *pm, acc = *d, r;
    r.d[0] = 0; r.d[1] = 0;
    if (op == F_ADDP) {
        if (dbl) { r.df[0] = fp_add64(a.df[0], a.df[1]); if (n == 2) r.df[1] = fp_add64(b.df[0], b.df[1]); }
        else {
            for (int i = 0; i < n / 2; i++) r.f[i] = fp_add32(a.f[2 * i], a.f[2 * i + 1]);
            for (int i = 0; i < n / 2; i++) r.f[n / 2 + i] = fp_add32(b.f[2 * i], b.f[2 * i + 1]);
        }
        nx_vst(d, r);
        return;
    }
    for (int i = 0; i < n; i++) {
        int is_mask = 0;
        if (dbl) {
            uint64_t m = 0;
            double v = f3d(op, acc.df[i], a.df[i], b.df[i], &m, &is_mask);
            if (is_mask) r.d[i] = m; else r.df[i] = v;      /* every arithmetic result is already exact (fpexact.h) */
        } else {
            uint32_t m = 0;
            float v = f3s(op, acc.f[i], a.f[i], b.f[i], &m, &is_mask);
            if (is_mask) r.w[i] = m; else r.f[i] = v;
        }
    }
    nx_vst(d, r);
}
/* Same, with the second operand taken from one element of vm. */
NX_SLOW void v_f3e_ref(V128 *d, const V128 *pn, const V128 *pm, int idx, int op, int dbl, int n) {
    V128 b;
    for (int i = 0; i < 4; i++) { if (dbl) b.df[i & 1] = pm->df[idx]; else b.f[i] = pm->f[idx]; }
    v_f3v_ref(d, pn, &b, op, dbl, n);
}

#ifdef NX_X86_SIMD
/* Elements 0..n-1 (n = 1, 2, 4) of a single-precision register; the other lanes read as zero. */
NX_HOT __m128 nx_ldps(const V128 *p, int n) {
    if (n == 4) return _mm_loadu_ps(p->f);
    if (n == 2) return _mm_castpd_ps(_mm_load_sd(p->df));
    return _mm_load_ss(p->f);
}
/* Writes lanes 0..n-1 of r to d and clears the rest, in one store. */
NX_HOT void nx_stps(V128 *d, __m128 r, int n) {
    if (n == 2) r = _mm_castpd_ps(_mm_move_sd(_mm_setzero_pd(), _mm_castps_pd(r)));
    else if (n == 1) r = _mm_move_ss(_mm_setzero_ps(), r);
    _mm_storeu_ps(d->f, r);
}
NX_HOT int nx_lanes(int n) { return (1 << n) - 1; }
NX_HOT int nx_nanps(__m128 r, int n) { return _mm_movemask_ps(_mm_cmpunord_ps(r, r)) & nx_lanes(n); }
NX_HOT __m128 nx_signps(void) { return _mm_castsi128_ps(_mm_set1_epi32((int)0x80000000u)); }

/* The packed form of f3s on lanes 0..n-1 of a and b (acc: the destination's old value, for the accumulating forms). Returns 1 when it
 * wrote d, 0 when the element-by-element code must run instead: a NaN result (or, for the max/min family, a NaN operand), a form
 * without a packed equivalent. */
NX_HOT int v_f3s_fast(V128 *d, __m128 a, __m128 b, const V128 *acc, int op, int n) {
    const __m128 sign = nx_signps();
    __m128 r;
    switch (op) {
    case F_ADD: r = _mm_add_ps(a, b); break;
    case F_SUB: r = _mm_sub_ps(a, b); break;
    case F_MUL: r = _mm_mul_ps(a, b); break;
    case F_DIV: r = _mm_div_ps(a, b); break;
    case F_ABD:
        r = _mm_sub_ps(a, b);
        if (__builtin_expect(nx_nanps(r, n), 0)) return 0;
        nx_stps(d, _mm_andnot_ps(sign, r), n);
        return 1;
#ifdef __FMA__
    case F_MLA: r = _mm_fmadd_ps(a, b, nx_ldps(acc, n)); break;
    case F_MLS: r = _mm_fnmadd_ps(a, b, nx_ldps(acc, n)); break;       /* fma(-a, b, acc): the negation is exact */
    case F_RECPS: r = _mm_fnmadd_ps(a, b, _mm_set1_ps(2.0f)); break;   /* inf * 0 gives a NaN here: the element code returns 2.0 */
    case F_RSQRTS:                                                     /* fp_rsqrts32's main case: |a| >= 2 * FLT_MIN, a halved */
        if ((_mm_movemask_ps(_mm_cmpge_ps(_mm_andnot_ps(sign, a), _mm_set1_ps(2.0f * 1.17549435e-38f))) & nx_lanes(n)) != nx_lanes(n)) return 0;
        r = _mm_fmadd_ps(_mm_mul_ps(_mm_xor_ps(a, sign), _mm_set1_ps(0.5f)), b, _mm_set1_ps(1.5f));
        break;
#endif
    case F_MAX: case F_MIN: case F_MAXNM: case F_MINNM: {             /* without NaN operands: the larger / smaller, +0 > -0 */
        if (_mm_movemask_ps(_mm_cmpunord_ps(a, b)) & nx_lanes(n)) return 0;
        const __m128 z = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(_mm_andnot_ps(sign, _mm_or_ps(a, b))), _mm_setzero_si128()));
        const int mx = op == F_MAX || op == F_MAXNM;
        const __m128 zr = mx ? _mm_and_ps(a, b) : _mm_or_ps(a, b), nr = mx ? _mm_max_ps(a, b) : _mm_min_ps(a, b);
        nx_stps(d, _mm_or_ps(_mm_and_ps(z, zr), _mm_andnot_ps(z, nr)), n);
        return 1;
    }
    case F_CMEQ: nx_stps(d, _mm_cmpeq_ps(a, b), n); return 1;
    case F_CMGE: nx_stps(d, _mm_cmpge_ps(a, b), n); return 1;
    case F_CMGT: nx_stps(d, _mm_cmpgt_ps(a, b), n); return 1;
    case F_ADDP:                                                       /* pairs of a, then pairs of b */
        if (n == 2) { a = _mm_movelh_ps(a, b); b = a; }
        else if (n != 4) return 0;
        r = _mm_add_ps(_mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1)));
        break;
    default: return 0;
    }
    if (__builtin_expect(nx_nanps(r, n), 0)) return 0;
    nx_stps(d, r, n);
    return 1;
}
#endif

NX_HOT void v_f3v(V128 *d, const V128 *pn, const V128 *pm, int op, int dbl, int n) {
#ifdef NX_X86_SIMD
    if (!dbl && v_f3s_fast(d, nx_ldps(pn, n), nx_ldps(pm, n), d, op, n)) return;
#endif
    v_f3v_ref(d, pn, pm, op, dbl, n);
}
NX_HOT void v_f3e(V128 *d, const V128 *pn, const V128 *pm, int idx, int op, int dbl, int n) {
#ifdef NX_X86_SIMD
    if (!dbl && v_f3s_fast(d, nx_ldps(pn, n), _mm_set1_ps(pm->f[idx]), d, op, n)) return;
#endif
    v_f3e_ref(d, pn, pm, idx, op, dbl, n);
}

enum { F1_ABS, F1_NEG, F1_SQRT, F1_RSQRTE, F1_RECPE, F1_SCVTF, F1_UCVTF, F1_CVTZS, F1_CVTZU,
       F1_CMEQ0, F1_CMGT0, F1_CMGE0, F1_CMLT0, F1_CMLE0 };

NX_SLOW void v_f1v_ref(V128 *d, const V128 *pn, int op, int dbl, int n) {
    V128 a = *pn, r;
    r.d[0] = 0; r.d[1] = 0;
    for (int i = 0; i < n; i++) {
        if (dbl) {
            double x = a.df[i];
            switch (op) {
            case F1_ABS: r.df[i] = fp_abs64(x); break;
            case F1_NEG: r.df[i] = fp_neg64(x); break;
            case F1_SQRT: r.df[i] = fp_sqrt64(x); break;
            case F1_RSQRTE: r.df[i] = rsqrte_d(x); break;
            case F1_RECPE: r.df[i] = recpe_d(x); break;
            case F1_SCVTF: r.df[i] = (double)(int64_t)a.d[i]; break;
            case F1_UCVTF: r.df[i] = (double)a.d[i]; break;
            case F1_CVTZS: r.d[i] = f2s64(trunc(x)); break;
            case F1_CVTZU: r.d[i] = f2u64(trunc(x)); break;
            case F1_CMEQ0: r.d[i] = x == 0 ? ~0ull : 0; break;
            case F1_CMGT0: r.d[i] = x > 0 ? ~0ull : 0; break;
            case F1_CMGE0: r.d[i] = x >= 0 ? ~0ull : 0; break;
            case F1_CMLT0: r.d[i] = x < 0 ? ~0ull : 0; break;
            case F1_CMLE0: r.d[i] = x <= 0 ? ~0ull : 0; break;
            }
        } else {
            float x = a.f[i];
            switch (op) {
            case F1_ABS: r.f[i] = fp_abs32(x); break;
            case F1_NEG: r.f[i] = fp_neg32(x); break;
            case F1_SQRT: r.f[i] = fp_sqrt32(x); break;
            case F1_RSQRTE: r.f[i] = rsqrte_f(x); break;
            case F1_RECPE: r.f[i] = recpe_f(x); break;
            case F1_SCVTF: r.f[i] = (float)(int32_t)a.w[i]; break;
            case F1_UCVTF: r.f[i] = (float)a.w[i]; break;
            case F1_CVTZS: r.w[i] = f2s32(trunc((double)x)); break;
            case F1_CVTZU: r.w[i] = f2u32(trunc((double)x)); break;
            case F1_CMEQ0: r.w[i] = x == 0 ? ~0u : 0; break;
            case F1_CMGT0: r.w[i] = x > 0 ? ~0u : 0; break;
            case F1_CMGE0: r.w[i] = x >= 0 ? ~0u : 0; break;
            case F1_CMLT0: r.w[i] = x < 0 ? ~0u : 0; break;
            case F1_CMLE0: r.w[i] = x <= 0 ? ~0u : 0; break;
            }
        }
    }
    nx_vst(d, r);
}

NX_HOT void v_f1v(V128 *d, const V128 *pn, int op, int dbl, int n) {
#ifdef NX_X86_SIMD
    if (!dbl) {
        const __m128 a = nx_ldps(pn, n), zero = _mm_setzero_ps();
        __m128 r = a;
        int ok = 1;
        switch (op) {
        case F1_ABS: r = _mm_andnot_ps(nx_signps(), a); break;
        case F1_NEG: r = _mm_xor_ps(a, nx_signps()); break;
        case F1_SQRT: r = _mm_sqrt_ps(a); ok = !nx_nanps(r, n); break;
        case F1_SCVTF: r = _mm_cvtepi32_ps(_mm_castps_si128(a)); break;
        case F1_CMEQ0: r = _mm_cmpeq_ps(a, zero); break;
        case F1_CMGT0: r = _mm_cmpgt_ps(a, zero); break;
        case F1_CMGE0: r = _mm_cmpge_ps(a, zero); break;
        case F1_CMLT0: r = _mm_cmplt_ps(a, zero); break;
        case F1_CMLE0: r = _mm_cmple_ps(a, zero); break;
        default: ok = 0; break;
        }
        if (__builtin_expect(ok, 1)) { nx_stps(d, r, n); return; }
    }
#endif
    v_f1v_ref(d, pn, op, dbl, n);
}

/* FRINTA / FRINTM / FRINTP / FRINTZ / FRINTN / FRINTX / FRINTI on n elements; modes as in fp_frint32 (fpexact.h). */
static inline void v_frint(V128 *d, const V128 *pn, int mode, int dbl, int n) {
    V128 a = *pn, r;
    r.d[0] = 0; r.d[1] = 0;
    for (int i = 0; i < n; i++) {
        if (dbl) r.df[i] = fp_frint64(a.df[i], mode);
        else r.f[i] = fp_frint32(a.f[i], mode);
    }
    nx_vst(d, r);
}

/* FMAXV / FMINV / FMAXNMV / FMINNMV across the four single lanes. Arm's Reduce is a tree, (lane0 op lane1) op
 * (lane2 op lane3), with the operand order of fpexact.h. The result goes to element 0; the rest is cleared. */
static inline void v_fred32(V128 *d, const V128 *pn, int is_min, int num) {
    uint32_t lo = fp_maxmin32_bits(pn->w[0], pn->w[1], is_min, num);
    uint32_t hi = fp_maxmin32_bits(pn->w[2], pn->w[3], is_min, num);
    uint32_t r = fp_maxmin32_bits(lo, hi, is_min, num);
    V128 t; t.d[0] = r; t.d[1] = 0;
    nx_vst(d, t);
}

enum { I_ADD, I_SUB, I_MUL, I_CMEQ, I_CMGT, I_CMHI, I_CMGE, I_CMHS, I_CMTST, I_SQADD, I_UQADD, I_SQSUB, I_UQSUB, I_MLA };

/* saturation of an exact result to a signed / unsigned element of size sz (FPSR.QC is not modelled; the lifter
 * reads FPSR as 0) */
static inline uint64_t v_sat_s(__int128 v, int sz) {
    __int128 hi = sz == 3 ? (__int128)INT64_MAX : (((__int128)1 << ((8 << sz) - 1)) - 1), lo = -hi - 1;
    if (v > hi) v = hi;
    if (v < lo) v = lo;
    return (uint64_t)v & (sz == 3 ? ~0ull : ((1ull << (8 << sz)) - 1));
}
static inline uint64_t v_sat_u(__int128 v, int sz) {
    __int128 hi = sz == 3 ? (__int128)UINT64_MAX : (((__int128)1 << (8 << sz)) - 1);
    if (v > hi) v = hi;
    if (v < 0) v = 0;
    return (uint64_t)v;
}

static inline void v_i3v(V128 *d, const V128 *pn, const V128 *pm, int op, int sz, int n) {
    V128 a = *pn, b = *pm, acc = *d, r;
    r.d[0] = 0; r.d[1] = 0;
    uint64_t ones = sz == 3 ? ~0ull : ((1ull << (8 << sz)) - 1);
    for (int i = 0; i < n; i++) {
        uint64_t x = vget(&a, sz, i), y = vget(&b, sz, i), z = 0;
        switch (op) {
        case I_ADD: z = x + y; break;
        case I_SUB: z = x - y; break;
        case I_MUL: z = x * y; break;
        case I_CMEQ: z = x == y ? ones : 0; break;
        case I_CMGT: z = vsext(x, sz) > vsext(y, sz) ? ones : 0; break;
        case I_CMGE: z = vsext(x, sz) >= vsext(y, sz) ? ones : 0; break;
        case I_CMHI: z = x > y ? ones : 0; break;
        case I_CMHS: z = x >= y ? ones : 0; break;
        case I_CMTST: z = (x & y) ? ones : 0; break;
        case I_SQADD: z = v_sat_s((__int128)vsext(x, sz) + vsext(y, sz), sz); break;
        case I_UQADD: z = v_sat_u((__int128)x + y, sz); break;
        case I_SQSUB: z = v_sat_s((__int128)vsext(x, sz) - vsext(y, sz), sz); break;
        case I_UQSUB: z = v_sat_u((__int128)x - y, sz); break;
        case I_MLA: z = vget(&acc, sz, i) + x * y; break;            /* d += n * m, modulo the size */
        }
        vset(&r, sz, i, z);
    }
    nx_vst(d, r);
}

/* ADDV: the sum of the n elements (size sz) of n, modulo the size, alone in d (the rest zeroed). */
static inline void v_addv(V128 *d, const V128 *pn, int sz, int n) {
    uint64_t s = 0;
    for (int i = 0; i < n; i++) s += vget(pn, sz, i);
    V128 r;
    r.d[0] = 0; r.d[1] = 0;
    vset(&r, sz, 0, s);
    nx_vst(d, r);
}

/* SMULL/UMULL, SMLAL/UMLAL, SMLSL/UMLSL (vector and by element). Source elements of size sz from the lower
 * (upper = 0) or upper (SMULL2 ..., upper = 1) half of n and m; results of size sz + 1 fill d. acc: 0 = d := product,
 * 1 = d += product, 2 = d -= product (modulo the result size, as the instructions do). elem >= 0: every product uses
 * element `elem` of m (the indexed form; m is then read whole, not its upper half). */
static inline void v_mull(V128 *d, const V128 *pn, const V128 *pm, int sz, int upper, int is_unsigned, int acc, int elem) {
    V128 a = *pn, b = *pm, r = *d;
    int n = 8 >> sz, base = upper ? n : 0, rs = sz + 1;
    for (int i = 0; i < n; i++) {
        uint64_t x = vget(&a, sz, base + i), y = vget(&b, sz, elem >= 0 ? elem : base + i);
        uint64_t p = is_unsigned ? x * y : (uint64_t)(vsext(x, sz) * vsext(y, sz));
        uint64_t old = acc ? vget(&r, rs, i) : 0;
        uint64_t z = acc == 1 ? old + p : acc == 2 ? old - p : p;
        vset(&r, rs, i, z);
    }
    if (!acc) { V128 t; t.d[0] = 0; t.d[1] = 0; for (int i = 0; i < n; i++) vset(&t, rs, i, vget(&r, rs, i)); r = t; }
    nx_vst(d, r);
}

/* MUL / MLA / MLS (vector, by element): every element of n times element `elem` of m, modulo the size. */
static inline void v_mul_elem(V128 *d, const V128 *pn, const V128 *pm, int sz, int elem, int acc, int n) {
    V128 a = *pn, r = *d, out;
    uint64_t y = vget(pm, sz, elem);
    out.d[0] = 0; out.d[1] = 0;
    for (int i = 0; i < n; i++) {
        uint64_t p = vget(&a, sz, i) * y, old = vget(&r, sz, i);
        vset(&out, sz, i, acc == 1 ? old + p : acc == 2 ? old - p : p);
    }
    nx_vst(d, out);
}

/* saturating narrowing shift right: SQSHRN / SQRSHRN (signed -> signed), UQSHRN / UQRSHRN (unsigned ->
 * unsigned), SQSHRUN / SQRSHRUN (signed -> unsigned). Elements of size dsz + 1 of n are shifted right by `shift`
 * (with rounding: + 1 << (shift - 1) first), saturated to size dsz, and written to the lower half of d (upper half
 * zeroed) or, for the "2" forms (upper = 1), to the upper half (lower half kept). */
static inline void v_qshrn(V128 *d, const V128 *pn, int shift, int dsz, int round, int src_signed, int dst_signed, int upper) {
    V128 a = *pn, r = *d;
    int n = 8 >> dsz, ssz = dsz + 1;
    uint64_t out[16];
    for (int i = 0; i < n; i++) {
        uint64_t x = vget(&a, ssz, i);
        __int128 v = src_signed ? (__int128)vsext(x, ssz) : (__int128)x;
        if (round) v += (__int128)1 << (shift - 1);
        v >>= shift;
        out[i] = dst_signed ? v_sat_s(v, dsz) : v_sat_u(v, dsz);
    }
    if (!upper) { r.d[0] = 0; r.d[1] = 0; }
    for (int i = 0; i < n; i++) vset(&r, dsz, (upper ? n : 0) + i, out[i]);
    nx_vst(d, r);
}
/* the scalar forms of the same (SQRSHRN h, s, #n ...): element 0 only, the rest of d zeroed. */
static inline void v_qshrn_s(V128 *d, const V128 *pn, int shift, int dsz, int round, int src_signed, int dst_signed) {
    uint64_t x = vget(pn, dsz + 1, 0);
    __int128 v = src_signed ? (__int128)vsext(x, dsz + 1) : (__int128)x;
    if (round) v += (__int128)1 << (shift - 1);
    v >>= shift;
    V128 r;
    r.d[0] = 0; r.d[1] = 0;
    vset(&r, dsz, 0, dst_signed ? v_sat_s(v, dsz) : v_sat_u(v, dsz));
    nx_vst(d, r);
}

enum { I1_NOT, I1_NEG, I1_ABS, I1_CMEQ0, I1_REV64, I1_CNT };

static inline void v_i1v(V128 *d, const V128 *pn, int op, int sz, int n) {
    V128 a = *pn, r;
    r.d[0] = 0; r.d[1] = 0;
    uint64_t ones = sz == 3 ? ~0ull : ((1ull << (8 << sz)) - 1);
    int per64 = 8 >> sz;
    for (int i = 0; i < n; i++) {
        uint64_t x = vget(&a, sz, i), z = 0;
        switch (op) {
        case I1_NOT: z = ~x; break;
        case I1_NEG: z = 0 - x; break;
        case I1_ABS: z = vsext(x, sz) < 0 ? 0 - x : x; break;
        case I1_CMEQ0: z = x == 0 ? ones : 0; break;
        case I1_CNT: z = (uint64_t)__builtin_popcountll(x); break;
        case I1_REV64: vset(&r, sz, (i / per64) * per64 + (per64 - 1 - i % per64), x); continue;
        }
        vset(&r, sz, i, z);
    }
    nx_vst(d, r);
}

/* Narrow each element of vn to half size: XTN writes the low half of the destination and zeroes the upper half; XTN2
 * (upper = 1) writes the upper half and keeps the lower. */
static inline void v_xtn(V128 *d, const V128 *pn, int sz, int upper) {
    V128 a = *pn, r;
    if (upper) r = *d; else { r.d[0] = 0; r.d[1] = 0; }
    for (int i = 0; i < (8 >> sz); i++) vset(&r, sz, (upper ? (8 >> sz) : 0) + i, vget(&a, sz + 1, i));
    nx_vst(d, r);
}

/* BSL / BIT / BIF */
static inline void v_bitsel(V128 *d, const V128 *pn, const V128 *pm, int kind, int q) {
    V128 r;
    for (int i = 0; i < 2; i++) {
        uint64_t x = d->d[i], n = pn->d[i], m = pm->d[i];
        r.d[i] = kind == 1 ? (x & n) | (~x & m)              /* bsl: mask in destination */
               : kind == 2 ? (x & ~m) | (n & m)              /* bit: insert if true      */
               :             (x & m) | (n & ~m);             /* bif: insert if false     */
    }
    if (!q) r.d[1] = 0;
    nx_vst(d, r);
}

static inline void v_dup_ref(V128 *d, uint64_t value, int sz, int n) {
    V128 r;
    r.d[0] = 0; r.d[1] = 0;
    for (int i = 0; i < n; i++) vset(&r, sz, i, value);
    nx_vst(d, r);
}
NX_HOT void v_dup(V128 *d, uint64_t value, int sz, int n) {
#ifdef NX_X86_SIMD
    const uint64_t v = sz == 0 ? (uint8_t)value * 0x0101010101010101ull : sz == 1 ? (uint16_t)value * 0x0001000100010001ull
                     : sz == 2 ? (uint32_t)value * 0x0000000100000001ull : value;
    const int bytes = n << sz;
    if (bytes == 16 || bytes == 8) {
        _mm_storeu_si128((__m128i *)d, _mm_set_epi64x(bytes == 16 ? (long long)v : 0, (long long)v));
        return;
    }
#endif
    v_dup_ref(d, value, sz, n);
}

static inline void v_ext_ref(V128 *d, const V128 *pn, const V128 *pm, int imm, int q) {
    V128 a = *pn, b = *pm, r;
    int n = q ? 16 : 8;
    r.d[0] = 0; r.d[1] = 0;
    for (int i = 0; i < n; i++) r.b[i] = (imm + i < n) ? a.b[imm + i] : b.b[imm + i - n];
    nx_vst(d, r);
}
#ifdef NX_X86_SIMD
#define NX_EXT_CASE(k) case k: r = _mm_or_si128(_mm_srli_si128(a, k), _mm_slli_si128(b, 16 - k)); break;
/* Bytes imm.. of a:b (q = 1), or of the low halves of a and b taken as one 16-byte value (q = 0). The byte shifts need constant
 * counts: the lifter's imm is one, and the switch folds once this is inlined. */
NX_HOT void v_ext(V128 *d, const V128 *pn, const V128 *pm, int imm, int q) {
    __m128i a = _mm_loadu_si128((const __m128i *)pn), b = _mm_loadu_si128((const __m128i *)pm), r;
    if (!q) { a = _mm_unpacklo_epi64(a, b); b = _mm_setzero_si128(); }
    switch (imm) {
    case 0: r = a; break;
    NX_EXT_CASE(1) NX_EXT_CASE(2) NX_EXT_CASE(3) NX_EXT_CASE(4) NX_EXT_CASE(5) NX_EXT_CASE(6) NX_EXT_CASE(7) NX_EXT_CASE(8)
    NX_EXT_CASE(9) NX_EXT_CASE(10) NX_EXT_CASE(11) NX_EXT_CASE(12) NX_EXT_CASE(13) NX_EXT_CASE(14) NX_EXT_CASE(15)
    default: v_ext_ref(d, pn, pm, imm, q); return;
    }
    if (!q) r = _mm_move_epi64(r);
    _mm_storeu_si128((__m128i *)d, r);
}
#undef NX_EXT_CASE
#else
#define v_ext v_ext_ref
#endif

/* TBL / TBX with `len` consecutive table registers starting at index rn. */
static inline void v_tbl(Ctx *c, int rd, int rn, int len, int rm, int q, int tbx) {
    uint8_t table[64];
    for (int i = 0; i < len; i++) memcpy(table + 16 * i, c->v[(rn + i) & 31].b, 16);
    V128 idx = c->v[rm], r = c->v[rd];
    int n = q ? 16 : 8;
    for (int i = 0; i < n; i++) {
        if (idx.b[i] < 16 * len) r.b[i] = table[idx.b[i]];
        else if (!tbx) r.b[i] = 0;
    }
    if (!q) r.d[1] = 0;
    c->v[rd] = r;
}

/* LDn / STn multiple structures (n = 2..4): elements are interleaved in memory,
 * element i of register k lives at memory element (i * n + k). */
static inline void v_ldn(Ctx *c, uint64_t a, int rt, int n, int sz, int q) {
    int count = (q ? 16 : 8) >> sz, es = 1 << sz;
    V128 r[4];
    for (int k = 0; k < n; k++) { r[k].d[0] = 0; r[k].d[1] = 0; }
    for (int i = 0; i < count; i++)
        for (int k = 0; k < n; k++) {
            uint64_t v = 0;
            memcpy(&v, (void *)(uintptr_t)(a + (uint64_t)((i * n + k) * es)), (size_t)es);
            vset(&r[k], sz, i, v);
        }
    for (int k = 0; k < n; k++) c->v[(rt + k) & 31] = r[k];
}
static inline void v_stn(Ctx *c, uint64_t a, int rt, int n, int sz, int q) {
    int count = (q ? 16 : 8) >> sz, es = 1 << sz;
    for (int i = 0; i < count; i++)
        for (int k = 0; k < n; k++) {
            uint64_t v = vget(&c->v[(rt + k) & 31], sz, i);
            memcpy((void *)(uintptr_t)(a + (uint64_t)((i * n + k) * es)), &v, (size_t)es);
        }
}

/* UZP1/TRN1/ZIP1/UZP2/TRN2/ZIP2: opcode is bits 14:12 of the instruction. */
static inline void v_perm(V128 *d, const V128 *pn, const V128 *pm, int opcode, int sz, int n) {
    V128 a = *pn, b = *pm, r;
    int part = (opcode >> 2) & 1, kind = opcode & 3, half = n / 2;
    r.d[0] = 0; r.d[1] = 0;
    for (int i = 0; i < half; i++) {
        uint64_t lo, hi;
        if (kind == 1) {                                     /* uzp */
            lo = vget(&a, sz, 2 * i + part);
            hi = vget(&b, sz, 2 * i + part);
            vset(&r, sz, i, lo);
            vset(&r, sz, half + i, hi);
        } else if (kind == 2) {                              /* trn */
            vset(&r, sz, 2 * i, vget(&a, sz, 2 * i + part));
            vset(&r, sz, 2 * i + 1, vget(&b, sz, 2 * i + part));
        } else {                                             /* zip */
            vset(&r, sz, 2 * i, vget(&a, sz, part * half + i));
            vset(&r, sz, 2 * i + 1, vget(&b, sz, part * half + i));
        }
    }
    nx_vst(d, r);
}

/* SSHLL/USHLL (also SXTL/UXTL): widen the low (or high) half and shift left. */
static inline void v_shll(V128 *d, const V128 *pn, int shift, int sz, int is_unsigned, int upper) {
    V128 a = *pn, r;
    int n = 8 >> sz;
    r.d[0] = 0; r.d[1] = 0;
    for (int i = 0; i < n; i++) {
        uint64_t x = vget(&a, sz, i + (upper ? n : 0));
        vset(&r, sz + 1, i, (is_unsigned ? x : (uint64_t)vsext(x, sz)) << shift);
    }
    nx_vst(d, r);
}
/* SHL / SSHR / USHR by immediate. kind: 0 shl, 1 sshr, 2 ushr. */
static inline void v_shift(V128 *d, const V128 *pn, int shift, int sz, int kind, int n) {
    V128 a = *pn, r;
    int bits = 8 << sz;
    r.d[0] = 0; r.d[1] = 0;
    for (int i = 0; i < n; i++) {
        uint64_t x = vget(&a, sz, i), z;
        if (kind == 0) z = x << shift;
        else if (kind == 1) z = (uint64_t)(vsext(x, sz) >> (shift >= bits ? bits - 1 : shift));
        else z = shift >= bits ? 0 : x >> shift;
        vset(&r, sz, i, z);
    }
    nx_vst(d, r);
}

#endif
