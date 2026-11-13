// SPDX-License-Identifier: GPL-3.0-or-later
// nxrecomp/opt.py 'flags': cond() / get_nzcv() on an NxFlags (cxx.h) against ctx.h's on a Ctx, every condition code, random and
// edge operands, both widths, carry-in 0 / 1, FCMP with NaN / infinities / signed zeros, and the inlined constant-condition paths.
// Build and run (any x64 clang or zig c++; the flags of the lifted code):
//   zig c++ -target x86_64-windows-gnu -mcpu=x86_64+cx16+fma -std=c++20 -O1 -fno-strict-aliasing -ffp-contract=off -fno-math-errno \
//       -I recomp/include recomp/tests/nxflags_test.cpp -o nxflags_test.exe && nxflags_test.exe     (prints '... 0 failures', exit 0)
#include <nxrecomp/cxx.h>
#include <cstdio>
#include <random>
static uint64_t edges[] = {0, 1, 2, 0x7f, 0x80, 0xff, 0x7fffffff, 0x80000000, 0x80000001, 0xffffffff, 0x100000000ull, 0x7fffffffffffffffull,
                           0x8000000000000000ull, 0x8000000000000001ull, 0xfffffffffffffffeull, 0xffffffffffffffffull, 0xfffffffeull};
static long fails = 0, tests = 0;
template <class T> __attribute__((noinline)) void chk(uint64_t a, uint64_t b, uint32_t cin) {
    Ctx c{}; NxFlags f{};
    volatile int cc_v;
    if (sizeof(T) == 8) { adds64(&c, a, b, cin); adds64(&f, a, b, cin); }
    else { adds32(&c, (uint32_t)a, (uint32_t)b, cin); adds32(&f, (uint32_t)a, (uint32_t)b, cin); }
    for (int cc = 0; cc < 16; cc++) {
        cc_v = cc; int k = cc_v; tests++;
        if (cond(&c, k) != cond(&f, k)) { if (fails++ < 20) printf("FAIL w%d a=%llx b=%llx cin=%u cc=%d\n", (int)sizeof(T)*8, (unsigned long long)a, (unsigned long long)b, cin, cc); }
    }
    if (get_nzcv(&c) != get_nzcv(&f)) fails++;
}
__attribute__((noinline)) void chkf(double a, double b) {
    Ctx c{}; NxFlags f{}; volatile int cc_v;
    fcmp(&c, a, b); fcmp(&f, a, b);
    for (int cc = 0; cc < 16; cc++) { cc_v = cc; int k = cc_v; tests++;
        if (cond(&c, k) != cond(&f, k)) { if (fails++ < 40) printf("FAIL fcmp %g %g cc=%d\n", a, b, cc); } }
    if (get_nzcv(&c) != get_nzcv(&f)) fails++;
}
// the same with constant condition codes (the inlined, folded paths the lifted code gets)
template <int cc> __attribute__((noinline)) int k64(uint64_t a, uint64_t b, uint32_t cin) { NxFlags f; adds64(&f, a, b, cin); return cond(&f, cc); }
template <int cc> __attribute__((noinline)) int r64(uint64_t a, uint64_t b, uint32_t cin) { Ctx f{}; adds64(&f, a, b, cin); return cond(&f, cc); }
template <int cc> __attribute__((noinline)) int k32(uint32_t a, uint32_t b) { NxFlags f; adds32(&f, a, ~b, 1); return cond(&f, cc); }
template <int cc> __attribute__((noinline)) int r32(uint32_t a, uint32_t b) { Ctx f{}; adds32(&f, a, ~b, 1); return cond(&f, cc); }
template <int cc> __attribute__((noinline)) int kf(float a, float b) { NxFlags f; fcmp(&f, (double)a, (double)b); return cond(&f, cc); }
template <int cc> __attribute__((noinline)) int rf(float a, float b) { Ctx f{}; fcmp(&f, (double)a, (double)b); return cond(&f, cc); }
template <int cc> void all_const(uint64_t a, uint64_t b) {
    for (uint32_t cin = 0; cin < 2; cin++) { tests++; if (k64<cc>(a, ~b, cin) != r64<cc>(a, ~b, cin) || k64<cc>(a, b, cin) != r64<cc>(a, b, cin)) { if (fails++ < 60) printf("FAIL const64 cc=%d\n", cc); } }
    tests++; if (k32<cc>((uint32_t)a, (uint32_t)b) != r32<cc>((uint32_t)a, (uint32_t)b)) { if (fails++ < 60) printf("FAIL const32 cc=%d %llx %llx\n", cc, (unsigned long long)a, (unsigned long long)b); }
    float fa, fb; uint32_t ua = (uint32_t)a, ub = (uint32_t)b; memcpy(&fa, &ua, 4); memcpy(&fb, &ub, 4);
    tests++; if (kf<cc>(fa, fb) != rf<cc>(fa, fb)) { if (fails++ < 60) printf("FAIL constf cc=%d %g %g\n", cc, fa, fb); }
}
template <int... cc> void all_cc(uint64_t a, uint64_t b, std::integer_sequence<int, cc...>) { (all_const<cc>(a, b), ...); }
int main() {
    std::mt19937_64 rng(12345);
    const int ne = sizeof edges / sizeof edges[0];
    for (int i = 0; i < ne; i++) for (int j = 0; j < ne; j++) for (uint32_t cin = 0; cin < 2; cin++) {
        chk<uint64_t>(edges[i], edges[j], cin); chk<uint64_t>(edges[i], ~edges[j], cin);
        chk<uint32_t>(edges[i], edges[j], cin); chk<uint32_t>(edges[i], ~edges[j], cin);
        all_cc(edges[i], edges[j], std::make_integer_sequence<int, 16>{});
    }
    for (int i = 0; i < 300000; i++) {
        uint64_t a = rng(), b = rng(); int sh = rng() % 64; if (rng() & 1) b = a + (rng() % 5) - 2; if (rng() & 1) a >>= sh;
        chk<uint64_t>(a, b, i & 1); chk<uint64_t>(a, ~b, 1); chk<uint32_t>(a, b, i & 1); chk<uint32_t>(a, ~b, 1);
        if (i < 20000) all_cc(a, b, std::make_integer_sequence<int, 16>{});
    }
    double fv[] = {0.0, -0.0, 1.0, -1.0, 1e-310, -1e-310, 3.4e38, -3.4e38, __builtin_inf(), -__builtin_inf(), __builtin_nan(""), -__builtin_nan(""), 1.5, 1.5000001};
    for (double a : fv) for (double b : fv) chkf(a, b);
    for (int i = 0; i < 100000; i++) { uint64_t x = rng(), y = rng(); double a, b; memcpy(&a, &x, 8); memcpy(&b, &y, 8); if (i & 1) b = a; chkf(a, b); chkf((float)a, (float)b); }
    printf("%ld tests, %ld failures\n", tests, fails);
    return fails != 0;
}
