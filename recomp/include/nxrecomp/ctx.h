/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * nxrecomp: the generated-code ABI.
 *
 * Everything the lifted code (the output of the nxrecomp lifter) needs from the outside, in one header:
 *   - the guest CPU context `Ctx` and the 128-bit vector type `V128`;
 *   - the small inline helpers generated code calls (flags, memory, bit ops, exclusives);
 *   - the DECLARATIONS of the functions the runtime must provide (section "runtime-provided").
 * The runtime owns the implementations of everything in that section; the lifter owns this header.
 *
 * Valid C11 and C++17. Include path: <repo>/recomp/include, then `#include <nxrecomp/ctx.h>`.
 *
 * Provenance: adapted from smo-recomp runtime/core/runtime.h (GPL-3.0-or-later), which in turn follows ssbu-recomp.
 * V128 member aliases and the TPIDRRO / FPCR / FPSR / 128-bit exclusive fields come from SuperMarioWonderRecomp
 * (MIT, Copyright (c) 2026 BelmanteGu) and ScarletNX (GPL-3.0-or-later). See recomp/README.md.
 *
 * Bump NXRECOMP_ABI_VERSION for any change of a layout or of a helper's result.
 */
#ifndef NXRECOMP_CTX_H
#define NXRECOMP_CTX_H

#define NXRECOMP_ABI_VERSION 3   /* 2: host_import takes a globally unique import id; NxModule.index
                                  * 3: DynSym.value is always a guest address; NxEh.types are final addresses and NxEh.type_syms /
                                  *    NxEh.specs are new; every module also defines `const uint32_t nxmod_<name>_abi` */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

/* NX_NOESC marks the context parameter of every function the lifted code calls (the lifted functions themselves, call_indirect, host_import,
 * the SVC and site entries): the callee uses the context only for the duration of the call and keeps no pointer to it, and guest memory never
 * lies inside a Ctx. With it the host compiler (clang: `noescape` = LLVM `captures(none)`) knows that no guest load or store can touch the
 * context, so a register value read from or written to `c->x[n]` before a guest memory access is still valid after it (otherwise every guest
 * store forces the context registers in use to be reloaded). Only the lifted code defines NX_NOESCAPE_CTX (fefw_build.py, NxRecomp.cmake); the
 * runtime's definitions of the same functions are built without it (the attribute is part of the function type in clang, and the lifted code
 * and the runtime are separate units, so the two never meet). Needs `-mllvm -capture-tracking-max-uses-to-explore=<large>` to pay off in big
 * functions: LLVM gives up proving a pointer uncaptured after 100 uses, and the context has thousands. */
#if defined(NX_NOESCAPE_CTX) && defined(__clang__)
#define NX_NOESC __attribute__((noescape))
#else
#define NX_NOESC
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* One 128-bit vector register. The same storage under several names: the smo-style (b h w d f df) and the
 * Wonder-style (u8 s8 u16 s16 u32 s32 u64 s64 f32 f64) views. Lane 0 is the lowest-addressed. */
typedef union __attribute__((aligned(16))) V128 {
    uint8_t b[16];   uint8_t u8[16];   int8_t s8[16];
    uint16_t h[8];   uint16_t u16[8];  int16_t s16[8];
    uint32_t w[4];   uint32_t u32[4];  int32_t s32[4];
    uint64_t d[2];   uint64_t u64[2];  int64_t s64[2];
    float f[4];      float f32[4];
    double df[2];    double f64[2];
} V128;

typedef struct Ctx {
    uint64_t x[32];              /* x0..x30; x[31] is SP. XZR is handled by the lifter, never stored */
    V128 v[32];
    uint32_t N, Z, C, V;         /* PSTATE.NZCV, each 0 or 1 */
    uint64_t fpcr, fpsr;         /* stored and read back by MSR/MRS, NOT honoured by the FP code (docs/GAPS.md) */
    uint64_t tpidr;              /* TPIDR_EL0   (read/write at EL0) */
    uint64_t tpidrro;            /* TPIDRRO_EL0 (read-only at EL0; the console keeps the thread's TLS block here) */
    uint64_t ex_addr;            /* exclusive monitor: address of the last LDXR / LDXP, or ~0 when open */
    uint64_t ex_val, ex_val_hi;  /* ... and the value(s) it read (hi: second half of an LDXP of two X registers) */
    uint64_t ex_fail;            /* consecutive failed STXR, for diagnostics */
    uint64_t fn;                 /* entry address of the lifted function entered most recently (crash reports) */
    const char *in_import;       /* name of the host import currently executing on this context, or NULL */
    uint64_t import_from;        /* guest return address of that import call */
} Ctx;

/* ======================================================================================================== */
/* runtime-provided: declared here, implemented by the runtime (or, for tests, by oracle/e2e/rt)             */
/* ======================================================================================================== */

/* Indirect call or tail call to a guest address (BLR, BR out of a function, an unresolved jump table).
 * Looks the address up in the function tables of every loaded module. */
void call_indirect(Ctx *NX_NOESC c, uint64_t addr);
/* A call to a symbol another module provides (a PLT stub whose target is undefined in this module). `import_id` is globally
 * unique: NX_IMPORT_ID(module index, symbol index). The module index is NxModule.index of the calling module (the order of
 * the game config's module list, or --index of `nxrecomp lift`); the symbol index is the position in that module's
 * dyn_syms (NxModule.syms), so the runtime finds the name with nx_module_by_index(NX_IMPORT_MODULE(id))->syms[NX_IMPORT_SYM(id)].
 * (ABI 1 passed only the symbol index.) */
#define NX_IMPORT_ID(module_index, sym) (((uint32_t)(module_index) << 20) | (uint32_t)(sym))
#define NX_IMPORT_MODULE(id) ((uint32_t)(id) >> 20)
#define NX_IMPORT_SYM(id) ((uint32_t)(id) & 0xfffffu)
void host_import(Ctx *NX_NOESC c, uint32_t import_id);
/* Raised by the generated code for: an undefined / reserved instruction word, BRK, falling off the end of a function. */
void guest_trap(const char *what, uint64_t addr);
/* An instruction the lifter could not translate (a gap in coverage, listed by `cover`). */
void guest_unimpl(uint64_t pc, uint32_t insn, const char *what);
void guest_svc(Ctx *NX_NOESC c, uint32_t number);
/* System registers the lifter does not model inline (anything but TPIDR_EL0, TPIDRRO_EL0, FPCR, FPSR, NZCV, DCZID, CTR,
 * CNTVCT/CNTFRQ). `sysreg` is bits 20:5 of the MRS/MSR word (o0:op1:CRn:CRm:op2). */
uint64_t guest_mrs(Ctx *NX_NOESC c, uint32_t sysreg);
void guest_msr(Ctx *NX_NOESC c, uint32_t sysreg, uint64_t value);
/* Sites compiled in by the lifter from a site list (nxrecomp/program.py load_sites): a callout before an instruction, a
 * function hook at a function's start, a hook at a mid-function address. A hook returns nonzero when it has done the
 * function's work (the lifted function then returns at once). */
void nx_site_inline(Ctx *NX_NOESC c, uint64_t addr);
int nx_site_midhook(Ctx *NX_NOESC c, uint64_t addr);
int nx_hook_entry(Ctx *NX_NOESC c, uint64_t addr);
/* Diagnostics of a failed STXR (bits < 0: the monitor was not open on that address). May be a no-op. */
void stx_diag(Ctx *NX_NOESC c, uint64_t a, uint64_t expected, uint64_t seen, uint64_t v, int bits);

/* One lifted function of a module and its guest address. */
typedef struct { uint64_t addr; void (*fn)(Ctx *); } FnEntry;
/* One dynamic symbol of a module, by index: its name, value and whether the module defines it.
 * value (ABI 3): for a defined symbol its GUEST ADDRESS (module base included, for every input kind: an NSO's st_value is
 * module-relative and the lifter adds NxModule.base); 0 for an undefined one. Two exceptions keep the raw st_value: a
 * thread-local symbol (STT_TLS: a TLS offset) and an SHN_ABS symbol (an absolute number). */
typedef struct { const char *name; uint64_t value; int defined; } DynSym;
/* A C++ landing pad: resuming the lifted function `fn` at guest address `pad` is fn(c, pad). */
typedef struct { uint64_t pad; void (*fn)(Ctx *, uint64_t); } NxResume;
/* C++ exception support. A lifted function that contains a catch clause whose handler can return is entered as
 * nx_catch_call(c, u_<entry>): the runtime runs body(c, 0) under a host frame it can unwind back to, and on an exception
 * that this function catches, calls body(c, landing pad) there instead. (ssbu-recomp's nro_catch_call.) */
void nx_catch_call(Ctx *NX_NOESC c, void (*body)(Ctx *, uint64_t));

/* Unwind tables (from .eh_frame and the C++ LSDAs, see nxrecomp/eh.py and ehtables.py). All arrays are sorted by address.
 * Row: from `pc` (inclusive) to the next row of the same FDE, the CFA is x[cfa_reg] + cfa_off (cfa_reg 29 = FP, 31 = SP)
 * and each NxCfiSaved of the row says where a callee-saved register was stored: at CFA + off. reg 0..30 = x0..x30,
 * 64..95 = v0..v31 (low 64 bits). The return address is the x30 entry. */
typedef struct { uint64_t pc; uint8_t cfa_reg; uint8_t nsaved; int32_t cfa_off; uint32_t saved_first; } NxCfiRow;
typedef struct { uint8_t reg; int32_t off; } NxCfiSaved;
/* One FDE: the function [start, end), its CFI rows, its LSDA call sites, and where its type table starts. */
typedef struct { uint64_t start, end; uint32_t row_first, row_count; uint32_t site_first, site_count; uint32_t type_first; } NxFde;
/* Call site: [start, start + len) may throw; `pad` is the landing pad (0 = none: the exception continues to the caller);
 * `action` is 1 + an index into NxEh.actions, 0 = cleanup only (run the pad, then _Unwind_Resume). */
typedef struct { uint64_t start, len, pad; uint32_t action; } NxCallSite;
/* Action record: filter > 0 catches the type types[fde.type_first + filter - 1] (typeinfo address; 0 = catch(...)),
 * filter == 0 is a cleanup. filter < 0 is an exception specification (ABI 3): -filter - 1 is the index into NxEh.specs of its
 * list, a sequence of type filters (each k selects types[fde.type_first + k - 1] like a positive filter) ended by 0; the
 * exception is allowed when it matches one of them (an empty list, `throw()`, allows nothing), otherwise
 * __cxa_call_unexpected / std::terminate applies.
 * next: 1 + the index of the next record of the chain, 0 = end. */
typedef struct { int32_t filter; uint32_t next; } NxAction;
/* types (ABI 3): FINAL guest addresses of the typeinfo objects, after the module's relocations (R_AARCH64_RELATIVE added to
 * NxModule.base; a typeinfo slot bound to a symbol this module defines resolves to that symbol), 0 = catch(...).
 * type_syms is parallel to types: -1 when types[i] is final; >= 0 when the typeinfo is a symbol this module only imports:
 * types[i] is then 0 and the typeinfo is the address of the symbol NxModule.syms[type_syms[i]].name in whichever module
 * defines it (the runtime resolves it by name, like an import). specs: see NxAction (concatenated 0-terminated lists). */
typedef struct {
    const NxFde *fdes; int nfdes;
    const NxCfiRow *rows; const NxCfiSaved *saved;
    const NxCallSite *sites; const NxAction *actions; const uint64_t *types;
    const int32_t *type_syms; const uint32_t *specs;
} NxEh;

/* Everything the lifter writes about one module, in `nxmod_<name>` (module_<name>.cpp). The same file defines
 * `const uint32_t nxmod_<name>_abi = NXRECOMP_ABI_VERSION;` (ABI 3): the version of the ctx.h the module was lifted and built
 * with, so a stale prebuilt library is refused by the runtime's registration (compare it with the runtime's own version). */
typedef struct NxModule {
    const char *name;
    uint32_t index;              /* position in the program's module list (see NX_IMPORT_ID); unique per program */
    uint64_t base;               /* guest address the image is mapped at */
    uint64_t text_end;           /* base + size of .text */
    const FnEntry *fns; int nfns;            /* sorted by address */
    const DynSym *syms; int nsyms;
    const NxResume *resume; int nresume;     /* sorted by pad */
    const NxEh *eh;              /* NULL when the module has no unwind data lifted */
} NxModule;

/* ======================================================================================================== */
/* guest memory: guest pointers are host pointers (the image is mapped at its guest address)                  */
/* ======================================================================================================== */

#define NX_DEF_MEM(bits) \
    static inline uint##bits##_t rd##bits(uint64_t a) { uint##bits##_t v; memcpy(&v, (void *)(uintptr_t)a, sizeof v); return v; } \
    static inline void wr##bits(uint64_t a, uint##bits##_t v) { memcpy((void *)(uintptr_t)a, &v, sizeof v); }
NX_DEF_MEM(8) NX_DEF_MEM(16) NX_DEF_MEM(32) NX_DEF_MEM(64)

/* STXR: store only if the location still holds the value the matching LDXR read (compare-and-swap). Returns 0 on
 * success. Known gap: an ABA change between LDXR and STXR is not noticed (docs/GAPS.md). */
#define NX_DEF_STX(bits) \
    static inline uint32_t stx##bits(Ctx *c, uint64_t a, uint##bits##_t v) { \
        if (c->ex_addr != a) { stx_diag(c, a, 0, 0, v, -bits); return 1; } \
        c->ex_addr = ~0ull; \
        { uint##bits##_t exp_ = (uint##bits##_t)c->ex_val; \
          if (__atomic_compare_exchange_n((uint##bits##_t *)(uintptr_t)a, &exp_, v, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) { c->ex_fail = 0; return 0; } \
          stx_diag(c, a, c->ex_val, exp_, v, bits); } \
        return 1; \
    }
NX_DEF_STX(8) NX_DEF_STX(16) NX_DEF_STX(32) NX_DEF_STX(64)

/* Exclusive pairs. 32-bit pair: one 64-bit access through ex_addr / ex_val and stx64. 64-bit pair: a 128-bit access,
 * compared and swapped as a whole (the monitor keeps the high half in ex_val_hi). Needs -mcx16 on x64 (cmpxchg16b). */
static inline void ldxp128(Ctx *c, uint64_t a, uint64_t *lo, uint64_t *hi) {
    unsigned __int128 v = __atomic_load_n((unsigned __int128 *)(uintptr_t)a, __ATOMIC_ACQUIRE);
    c->ex_addr = a; c->ex_val = (uint64_t)v; c->ex_val_hi = (uint64_t)(v >> 64);
    *lo = (uint64_t)v; *hi = (uint64_t)(v >> 64);
}
static inline uint32_t stxp32(Ctx *c, uint64_t a, uint32_t lo, uint32_t hi) { return stx64(c, a, (uint64_t)lo | ((uint64_t)hi << 32)); }
static inline uint32_t stxp64(Ctx *c, uint64_t a, uint64_t lo, uint64_t hi) {
    if (c->ex_addr != a) { stx_diag(c, a, 0, 0, lo, -128); return 1; }
    c->ex_addr = ~0ull;
    unsigned __int128 e = ((unsigned __int128)c->ex_val_hi << 64) | c->ex_val, n = ((unsigned __int128)hi << 64) | lo;
    return __atomic_compare_exchange_n((unsigned __int128 *)(uintptr_t)a, &e, n, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) ? 0 : 1;
}

static inline V128 rdv(uint64_t a, int n) { V128 v; v.d[0] = 0; v.d[1] = 0; memcpy(&v, (void *)(uintptr_t)a, (size_t)n); return v; }
static inline void wrv(uint64_t a, const V128 *v, int n) { memcpy((void *)(uintptr_t)a, v, (size_t)n); }
static inline void vzero(V128 *v) { v->d[0] = 0; v->d[1] = 0; }

/* ======================================================================================================== */
/* flags and integer helpers                                                                                  */
/* ======================================================================================================== */

static inline uint64_t adds64(Ctx *c, uint64_t a, uint64_t b, uint32_t cin) {
    uint64_t r = a + b + cin;
    c->N = (uint32_t)(r >> 63);
    c->Z = r == 0;
    c->C = (r < a) || (cin && r == a);
    c->V = (uint32_t)(((a ^ r) & (b ^ r)) >> 63);
    return r;
}
static inline uint32_t adds32(Ctx *c, uint32_t a, uint32_t b, uint32_t cin) {
    uint64_t wide = (uint64_t)a + b + cin;
    uint32_t r = (uint32_t)wide;
    c->N = r >> 31;
    c->Z = r == 0;
    c->C = (uint32_t)(wide >> 32);
    c->V = ((a ^ r) & (b ^ r)) >> 31;
    return r;
}
static inline uint64_t logic64(Ctx *c, uint64_t r) { c->N = (uint32_t)(r >> 63); c->Z = r == 0; c->C = 0; c->V = 0; return r; }
static inline uint32_t logic32(Ctx *c, uint32_t r) { c->N = r >> 31; c->Z = r == 0; c->C = 0; c->V = 0; return r; }
static inline void set_nzcv(Ctx *c, int f) { c->N = (f >> 3) & 1; c->Z = (f >> 2) & 1; c->C = (f >> 1) & 1; c->V = f & 1; }
static inline uint64_t get_nzcv(const Ctx *c) { return ((uint64_t)c->N << 31) | ((uint64_t)c->Z << 30) | ((uint64_t)c->C << 29) | ((uint64_t)c->V << 28); }

static inline int cond(const Ctx *c, int cc) {
    int r;
    switch (cc >> 1) {
    case 0: r = c->Z; break;
    case 1: r = c->C; break;
    case 2: r = c->N; break;
    case 3: r = c->V; break;
    case 4: r = c->C && !c->Z; break;
    case 5: r = c->N == c->V; break;
    case 6: r = c->N == c->V && !c->Z; break;
    default: return 1;
    }
    return (cc & 1) ? !r : r;
}

static inline void fcmp(Ctx *c, double a, double b) {
    if (a != a || b != b) set_nzcv(c, 3);
    else if (a == b) set_nzcv(c, 6);
    else if (a < b) set_nzcv(c, 8);
    else set_nzcv(c, 2);
}

static inline uint32_t ror32(uint32_t v, unsigned n) { n &= 31; return n ? (v >> n) | (v << (32 - n)) : v; }
static inline uint64_t ror64(uint64_t v, unsigned n) { n &= 63; return n ? (v >> n) | (v << (64 - n)) : v; }
static inline uint32_t bswap32(uint32_t v) { return __builtin_bswap32(v); }
static inline uint64_t bswap64(uint64_t v) { return __builtin_bswap64(v); }
static inline uint64_t rev32_64(uint64_t v) { return ((uint64_t)bswap32((uint32_t)(v >> 32)) << 32) | bswap32((uint32_t)v); }
static inline uint32_t rev16_32(uint32_t v) { return ((v & 0xff00ff00u) >> 8) | ((v & 0x00ff00ffu) << 8); }
static inline uint64_t rev16_64(uint64_t v) { return ((v & 0xff00ff00ff00ff00ull) >> 8) | ((v & 0x00ff00ff00ff00ffull) << 8); }
static inline uint32_t clz32(uint32_t v) { return v ? (uint32_t)__builtin_clz(v) : 32; }
static inline uint64_t clz64(uint64_t v) { return v ? (uint64_t)__builtin_clzll(v) : 64; }
static inline uint32_t cls32(uint32_t v) { return clz32(((v >> 1) ^ v) & 0x7fffffffu) - 1; }
static inline uint64_t cls64(uint64_t v) { return clz64(((v >> 1) ^ v) & 0x7fffffffffffffffull) - 1; }
/* RBIT: clang's bit-reverse builtin (the same bits; the compiler also folds clz(rbit(x)), the guest's count-trailing-zeros idiom in
 * strlen / memchr / memcmp, into one tzcnt); other compilers keep the loop. */
#if defined(__clang__)
static inline uint32_t rbit32(uint32_t v) { return __builtin_bitreverse32(v); }
static inline uint64_t rbit64(uint64_t v) { return __builtin_bitreverse64(v); }
#else
static inline uint32_t rbit32(uint32_t v) { uint32_t r = 0; for (int i = 0; i < 32; i++) r |= ((v >> i) & 1u) << (31 - i); return r; }
static inline uint64_t rbit64(uint64_t v) { uint64_t r = 0; for (int i = 0; i < 64; i++) r |= ((v >> i) & 1ull) << (63 - i); return r; }
#endif
static inline uint64_t umulh(uint64_t a, uint64_t b) { return (uint64_t)(((unsigned __int128)a * b) >> 64); }
static inline uint64_t smulh(uint64_t a, uint64_t b) { return (uint64_t)(((__int128)(int64_t)a * (int64_t)b) >> 64); }
static inline uint32_t sdiv32(uint32_t a, uint32_t b) {
    if (b == 0) return 0;
    if (a == 0x80000000u && b == 0xffffffffu) return a;
    return (uint32_t)((int32_t)a / (int32_t)b);
}
static inline uint64_t sdiv64(uint64_t a, uint64_t b) {
    if (b == 0) return 0;
    if (a == 0x8000000000000000ull && b == ~0ull) return a;
    return (uint64_t)((int64_t)a / (int64_t)b);
}

/* float -> integer with AArch64 saturation semantics */
static inline uint32_t f2s32(double v) { if (v != v) return 0; if (v >= 2147483647.0) return 0x7fffffffu; if (v <= -2147483648.0) return 0x80000000u; return (uint32_t)(int32_t)v; }
static inline uint32_t f2u32(double v) { if (v != v || v <= 0) return 0; if (v >= 4294967295.0) return 0xffffffffu; return (uint32_t)v; }
static inline uint64_t f2s64(double v) { if (v != v) return 0; if (v >= 9223372036854775807.0) return 0x7fffffffffffffffull; if (v <= -9223372036854775808.0) return 0x8000000000000000ull; return (uint64_t)(int64_t)v; }
static inline uint64_t f2u64(double v) { if (v != v || v <= 0) return 0; if (v >= 18446744073709551615.0) return ~0ull; return (uint64_t)v; }

#ifdef __cplusplus
}
#endif

/* WFE (kind 1) and YIELD (kind 0): the hints a spin-wait loop gives. AArch64 hardware pauses the core; here the runtime may set
 * nx_spin_hook to hand the host processor to another ready thread (the console runs a spinner and the lock holder on separate
 * cores, a host with fewer processors does not). Without a hook the hint is an x86 PAUSE. The hook is a function pointer the
 * compiler cannot see through, so a plain-load spin loop is not folded into an endless loop either. No guest state changes. */
#ifdef __cplusplus
inline void (*nx_spin_hook)(int kind) = nullptr;
static inline void nx_spin_hint(int kind) {
    if (void (*f)(int) = nx_spin_hook) f(kind);
#if defined(__x86_64__) || defined(_M_X64)
    else __builtin_ia32_pause();
#endif
}
#else
static inline void nx_spin_hint(int kind) { (void)kind; }
#endif

#include "fpexact.h"
#include "neon.h"

#endif /* NXRECOMP_CTX_H */
