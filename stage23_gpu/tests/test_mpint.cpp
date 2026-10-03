/* Int<L> arithmetic against GMP: add, sub, neg, mul, cmp, floor division, exact
 * division, truncating conversion to double, and overflow reporting, on random operands
 * of random sizes (including the sizes right at the overflow boundary). */
#include <stdio.h>
#include <stdlib.h>
#include "gmp_bridge.h"

using namespace s23;

static gmp_randstate_t rs;
static long failures = 0;

#define CHECK(cond, ...)                                                                 \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            if (failures++ < 20) {                                                       \
                fprintf(stderr, "FAIL L=%d %s:%d: ", L, __FILE__, __LINE__);             \
                fprintf(stderr, __VA_ARGS__);                                            \
                fprintf(stderr, "\n");                                                   \
            }                                                                            \
        }                                                                                \
    } while (0)

/* random signed integer with up to maxbits bits, biased toward small and edge sizes */
static void rand_z(mpz_t z, int maxbits)
{
    int pick = (int)gmp_urandomm_ui(rs, 8);
    int bits = pick == 0 ? 0 : pick == 1 ? maxbits : (int)gmp_urandomm_ui(rs, maxbits + 1);
    mpz_rrandomb(z, rs, bits); /* long runs of 0s and 1s: carry edge cases */
    if (gmp_urandomm_ui(rs, 2))
        mpz_neg(z, z);
}

template <int L> static bool fits(const mpz_t z)
{
    /* -2^(32L-1) .. 2^(32L-1) - 1 */
    mpz_t lim;
    mpz_init(lim);
    mpz_ui_pow_ui(lim, 2, 32 * L - 1);
    bool ok = mpz_cmp(z, lim) < 0;
    mpz_neg(lim, lim);
    ok = ok && mpz_cmp(z, lim) >= 0;
    mpz_clear(lim);
    return ok;
}

template <int L> static void run(long iters)
{
    const int maxbits = 32 * L - 1;
    mpz_t za, zb, zr, zq, zrem, t;
    mpz_inits(za, zb, zr, zq, zrem, t, NULL);
    Int<L> a, b, r, q, rem;
    for (long it = 0; it < iters; it++) {
        rand_z(za, maxbits);
        rand_z(zb, maxbits);
        if (!fits<L>(za) || !fits<L>(zb))
            continue;
        from_mpz(a, za);
        from_mpz(b, zb);

        /* round trip */
        to_mpz(t, a);
        CHECK(mpz_cmp(t, za) == 0, "round trip");

        /* cmp */
        int c1 = cmp(a, b), c2 = mpz_cmp(za, zb);
        CHECK((c1 > 0) == (c2 > 0) && (c1 < 0) == (c2 < 0), "cmp");

        /* add / sub / neg with overflow */
        mpz_add(zr, za, zb);
        bool ok = add(r, a, b);
        CHECK(ok == fits<L>(zr), "add overflow flag");
        if (ok) { to_mpz(t, r); CHECK(mpz_cmp(t, zr) == 0, "add value"); }
        mpz_sub(zr, za, zb);
        ok = sub(r, a, b);
        CHECK(ok == fits<L>(zr), "sub overflow flag");
        if (ok) { to_mpz(t, r); CHECK(mpz_cmp(t, zr) == 0, "sub value"); }
        mpz_neg(zr, za);
        ok = neg(r, a);
        CHECK(ok == fits<L>(zr), "neg overflow flag");
        if (ok) { to_mpz(t, r); CHECK(mpz_cmp(t, zr) == 0, "neg value"); }

        /* mul: may report overflow conservatively only at the -2^(32L-1) edge */
        mpz_mul(zr, za, zb);
        ok = mul(r, a, b);
        if (ok) { to_mpz(t, r); CHECK(mpz_cmp(t, zr) == 0, "mul value"); }
        else CHECK(!fits<L>(zr) || mpz_sizeinbase(zr, 2) == (size_t)maxbits + 1, "mul spurious overflow");
        if (fits<L>(zr) && mpz_sizeinbase(zr, 2) <= (size_t)maxbits) CHECK(ok, "mul missed result");

        /* floor division */
        if (mpz_sgn(zb) != 0) {
            mpz_fdiv_qr(zq, zrem, za, zb);
            ok = divmod_floor(q, rem, a, b);
            if (fits<L>(zq)) {
                CHECK(ok, "divmod failed");
                if (ok) {
                    to_mpz(t, q); CHECK(mpz_cmp(t, zq) == 0, "fdiv q");
                    to_mpz(t, rem); CHECK(mpz_cmp(t, zrem) == 0, "fdiv r");
                }
            }
            /* exact division of a multiple */
            mpz_mul(zr, zq, zb);
            Int<L> prod;
            if (fits<L>(zr) && from_mpz(prod, zr) && fits<L>(zq)) {
                ok = div_exact(q, prod, b);
                CHECK(ok, "div_exact failed");
                if (ok) { to_mpz(t, q); CHECK(mpz_cmp(t, zq) == 0, "div_exact value"); }
            }
            /* and a non-multiple must be rejected */
            if (mpz_sgn(zrem) != 0)
                CHECK(!div_exact(q, a, b), "div_exact accepted a non-multiple");
        }

        /* truncating double conversion */
        CHECK(get_d(a) == mpz_get_d(za), "get_d");
    }
    mpz_clears(za, zb, zr, zq, zrem, t, NULL);
    printf("L=%-3d %ld iterations: %s\n", L, iters, failures ? "FAILURES" : "ok");
}

int main(int argc, char **argv)
{
    long iters = argc > 1 ? atol(argv[1]) : 200000;
    gmp_randinit_default(rs);
    gmp_randseed_ui(rs, 12345);
    run<2>(iters);
    run<3>(iters);
    run<8>(iters);
    run<17>(iters / 4);
    run<64>(iters / 20);
    run<128>(iters / 50);
    printf("%s (%ld failures)\n", failures ? "FAILED" : "all passed", failures);
    gmp_randclear(rs);
    return failures != 0;
}
