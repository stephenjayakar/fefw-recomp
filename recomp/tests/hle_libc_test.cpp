// SPDX-License-Identifier: GPL-3.0-or-later
// GameConfig.inline_replace (fefw): cxx.h's hle_memmove / hle_memset / hle_strlen / hle_memcmp / hle_strcmp against the guest's
// own routines as the lifter translates them (guest_fns.inc, made by hle_libc_extract.py from a local lift: nnSdk 22.2.0's __memcpy_aarch64, memmove, memset, strlen, memcmp,
// strcmp, lifted without opt, renamed g_<addr>), on random sizes, alignments, overlaps and contents in host memory (guest pointers
// are host pointers). Checks x0 and every byte of the buffers.
#include <nxrecomp/cxx.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#define NX_OPTNONE __attribute__((optnone, noinline))
extern "C" {
#include "guest_fns.inc"
void g_7500856130(Ctx *__restrict c) { g_750078df00(c); }
void g_750087bbe0(Ctx *__restrict c) { g_75007bbc80(c); }
}
// unused runtime hooks the header declares
extern "C" void call_indirect(Ctx *, uint64_t) { abort(); }
extern "C" void guest_trap(const char *, uint64_t) { abort(); }
extern "C" void stx_diag(Ctx *, uint64_t, uint64_t, uint64_t, uint64_t, int) {}

static long fails = 0, tests = 0;
static std::mt19937_64 rng(0x5eed);
static const size_t N = 1 << 16, PAD = 256;

static void fill(std::vector<uint8_t> &b, int kind) {
    for (auto &x : b) x = kind == 0 ? (uint8_t)rng() : kind == 1 ? (uint8_t)(rng() % 3) : (uint8_t)(1 + rng() % 255);
}
static void report(const char *what, uint64_t a, uint64_t b, uint64_t n, uint64_t x0g, uint64_t x0h) {
    if (fails++ < 30) printf("FAIL %s a=+%llu b=+%llu n=%llu x0 guest=%llx hle=%llx\n", what, (unsigned long long)a, (unsigned long long)b,
                             (unsigned long long)n, (unsigned long long)x0g, (unsigned long long)x0h);
}
static uint64_t rsize() {
    switch (rng() % 4) {
    case 0: return rng() % 17;
    case 1: return rng() % 200;
    case 2: return rng() % 1200;
    default: return rng() % 9000;
    }
}

int main() {
    std::vector<uint8_t> A(N + 2 * PAD), B(N + 2 * PAD), S(N + 2 * PAD);
    uint8_t *a = A.data() + PAD, *b = B.data() + PAD;
    for (int it = 0; it < 200000; it++) {
        int which = it % 6;
        fill(S, it % 3);
        A = S; B = S;
        Ctx cg{}, ch{};
        uint64_t n = rsize(), o1 = rng() % (N - 9100), o2;
        if (which == 0 || which == 1) {           // memcpy / memmove: overlapping half the time
            o2 = (rng() & 1) ? o1 + (int64_t)(rng() % 600) - 300 : rng() % (N - 9100);
            if ((int64_t)o2 < 0) o2 = 0;
            if (o2 > N - 9100) o2 = N - 9100;
            cg.x[0] = (uint64_t)(a + o1); cg.x[1] = (uint64_t)(a + o2); cg.x[2] = n;
            ch.x[0] = (uint64_t)(b + o1); ch.x[1] = (uint64_t)(b + o2); ch.x[2] = n;
            if (which == 0) g_75007bbc80(&cg); else g_750078df20(&cg);
            nx::hle_memmove(&ch);
            tests++;
            if (memcmp(A.data(), B.data(), A.size()) || cg.x[0] - (uint64_t)a != ch.x[0] - (uint64_t)b)
                report(which ? "memmove" : "memcpy", o1, o2, n, cg.x[0] - (uint64_t)a, ch.x[0] - (uint64_t)b);
        } else if (which == 2) {                  // memset, x1 with junk in the upper bits
            uint64_t v = rng();
            cg.x[0] = (uint64_t)(a + o1); cg.x[1] = v; cg.x[2] = n;
            ch.x[0] = (uint64_t)(b + o1); ch.x[1] = v; ch.x[2] = n;
            g_750078e1e0(&cg); nx::hle_memset(&ch);
            tests++;
            if (memcmp(A.data(), B.data(), A.size()) || cg.x[0] - (uint64_t)a != ch.x[0] - (uint64_t)b)
                report("memset", o1, v, n, cg.x[0] - (uint64_t)a, ch.x[0] - (uint64_t)b);
        } else if (which == 3) {                  // strlen: a terminator at o1 + n
            for (uint64_t i = 0; i < n; i++) if (!a[o1 + i]) { a[o1 + i] = b[o1 + i] = 0x41; }
            a[o1 + n] = b[o1 + n] = 0;
            cg.x[0] = (uint64_t)(a + o1); ch.x[0] = (uint64_t)(b + o1);
            g_75007a6c20(&cg); nx::hle_strlen(&ch);
            tests++;
            if (cg.x[0] != ch.x[0] || ch.x[0] != n) report("strlen", o1, 0, n, cg.x[0], ch.x[0]);
        } else if (which == 4) {                  // memcmp: equal prefix of random length, then a random difference (or none)
            o2 = rng() % (N - 9100);
            memmove(a + o2, a + o1, n);
            uint64_t d = n ? rng() % (n + n / 4 + 1) : 0;
            if (d < n) a[o2 + d] = (uint8_t)rng();
            memcpy(b, a, N);
            cg.x[0] = (uint64_t)(a + o1); cg.x[1] = (uint64_t)(a + o2); cg.x[2] = n;
            ch.x[0] = (uint64_t)(b + o1); ch.x[1] = (uint64_t)(b + o2); ch.x[2] = n;
            g_75007bc140(&cg); nx::hle_memcmp(&ch);
            tests++;
            if (cg.x[0] != ch.x[0]) report("memcmp", o1, o2, n, cg.x[0], ch.x[0]);
        } else {                                  // strcmp: two strings sharing a prefix, then a difference or an end
            o2 = rng() % (N - 9100);
            uint64_t l1 = n % 600, l2 = l1 + rng() % 3 - 1;
            if ((int64_t)l2 < 0) l2 = 0;
            for (uint64_t i = 0; i < l1; i++) if (!a[o1 + i]) a[o1 + i] = 0x5a;
            a[o1 + l1] = 0;
            memmove(a + o2, a + o1, l1 + 1);
            if (l2 != l1) { a[o2 + l2] = 0; for (uint64_t i = l1; i < l2; i++) a[o2 + i] = 1 + rng() % 255; }
            if (l1 && (rng() & 1)) a[o2 + rng() % l1] = (uint8_t)rng();
            memcpy(b, a, N);
            cg.x[0] = (uint64_t)(a + o1); cg.x[1] = (uint64_t)(a + o2);
            ch.x[0] = (uint64_t)(b + o1); ch.x[1] = (uint64_t)(b + o2);
            g_75007bc240(&cg); nx::hle_strcmp(&ch);
            tests++;
            if (cg.x[0] != ch.x[0]) report("strcmp", o1, o2, l1, cg.x[0], ch.x[0]);
        }
    }
    printf("%ld tests, %ld failures\n", tests, fails);
    return fails != 0;
}
