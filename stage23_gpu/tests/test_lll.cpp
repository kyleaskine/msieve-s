/* sopt_best_norm (fixed-width, host build of the device code) against CADO itself:
 * for each case "d c0..cd Y0 Y1 k", the skew is computed as CADO's sopt_get_skewness,
 * the lattice is built as LLL_set_matrix_from_polys, and CADO's exact LLL (utils/lll.c,
 * delta = 1/1) and the port must give the same reduced basis, entry for entry, the same
 * chosen row and the same polynomial. Also reports the widest intermediate the port
 * needed, which sets the limb count L for the GPU.
 *
 *   test_lll CASES [max_cases]
 */
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gmp_bridge.h"
#include "sopt_lll.h"
#include "lll.h" /* CADO utils/lll.h */
#include "sopt_skew.h"

using namespace s23;

int s23::g_max_bits = 0;

static const int LIMBS = 128; /* 4096 bits; the run reports what was actually needed */
static const int DEG = 5;
typedef Int<LIMBS> Z;


/* c(x + k) in place */
static void mpz_translate(mpz_t *c, int d, mpz_t k)
{
    for (int i = 0; i < d; i++)
        for (int j = d - 1; j >= i; j--)
            mpz_addmul(c[j], k, c[j + 1]);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: test_lll CASES [max_cases]\n");
        return 2;
    }
    long max_cases = argc > 2 ? atol(argv[2]) : -1;
    FILE *fh = fopen(argv[1], "r");
    if (!fh) {
        perror(argv[1]);
        return 2;
    }
    static LLLState<LIMBS, DEG, DEG + 1> S;
    static SoptLLLResult<LIMBS, DEG> R;
    static BestNormWork<LIMBS, DEG> W;
    mpz_t f[DEG + 1], g[2], k, skew, det, a, b, tmp, norm2, coef;
    for (int i = 0; i <= DEG; i++)
        mpz_init(f[i]);
    mpz_inits(g[0], g[1], k, skew, det, a, b, tmp, norm2, coef, NULL);
    mpz_set_ui(a, 1);
    mpz_set_ui(b, 1);

    long ncases = 0, mismatches = 0, port_fail = 0;
    char buf[8192];
    while (fgets(buf, sizeof buf, fh) && (max_cases < 0 || ncases < max_cases)) {
        char *p = buf;
        int d = (int)strtol(p, &p, 10);
        if (d != DEG)
            continue;
        char tok[2048];
        int off;
        for (int i = 0; i <= DEG; i++) { sscanf(p, "%2047s%n", tok, &off); p += off; mpz_set_str(f[i], tok, 10); }
        sscanf(p, "%2047s%n", tok, &off); p += off; mpz_set_str(g[0], tok, 10);
        sscanf(p, "%2047s%n", tok, &off); p += off; mpz_set_str(g[1], tok, 10);
        sscanf(p, "%2047s%n", tok, &off); p += off; mpz_set_str(k, tok, 10);
        ncases++;

        /* --- CADO side (GMP) --- */
        sopt_skew_mpz(skew, g[0], f[DEG], DEG);
        mpz_t ft[DEG + 1], gt[2];
        for (int i = 0; i <= DEG; i++) mpz_init_set(ft[i], f[i]);
        mpz_init_set(gt[0], g[0]);
        mpz_init_set(gt[1], g[1]);
        mpz_translate(ft, DEG, k);
        mpz_translate(gt, 1, k);
        mat_Z m;
        LLL_init(&m, DEG, DEG + 1);
        for (int r = 1; r <= m.NumRows; r++)
            for (int c = 1; c <= m.NumCols; c++)
                mpz_set_ui(m.coeff[r][c], 0);
        mpz_set_ui(tmp, 1);
        for (int c = 0; c < m.NumCols; c++) { /* LLL_set_matrix_from_polys */
            if (c > 0) mpz_mul(tmp, tmp, skew);
            if (c <= m.NumRows - 2) mpz_set(m.coeff[c + 2][c + 1], tmp);
            mpz_mul(m.coeff[1][c + 1], tmp, ft[c]);
        }
        for (int c = 0; c <= m.NumRows - 2; c++) {
            mpz_mul(m.coeff[c + 2][c + 2], m.coeff[c + 2][c + 1], skew);
            mpz_mul(m.coeff[c + 2][c + 1], m.coeff[c + 2][c + 1], gt[0]);
            mpz_mul(m.coeff[c + 2][c + 2], m.coeff[c + 2][c + 2], gt[1]);
        }
        LLL(det, m, NULL, a, b);
        double min_norm = DBL_MAX;
        int best = 0;
        for (int r = 1; r <= m.NumRows; r++)
            if (mpz_sgn(m.coeff[r][DEG + 1]) != 0) {
                mpz_set_ui(norm2, 0);
                for (int c = 1; c <= m.NumCols; c++)
                    mpz_addmul(norm2, m.coeff[r][c], m.coeff[r][c]);
                double nd = mpz_get_d(norm2);
                if (nd < min_norm) { min_norm = nd; best = r; }
            }

        /* --- port --- */
        Z fz[DEG + 1], gz[2], kz, sz;
        bool conv = true;
        for (int i = 0; i <= DEG; i++) conv &= from_mpz(fz[i], f[i]);
        conv &= from_mpz(gz[0], g[0]) && from_mpz(gz[1], g[1]) && from_mpz(kz, k) && from_mpz(sz, skew);
        if (!conv) { port_fail++; goto next; }
        sopt_best_norm<LIMBS, DEG>(R, S, W, fz, gz, kz, sz);
        if (R.status != LLL_OK) {
            if (port_fail++ < 5) fprintf(stderr, "case %ld: port status %d\n", ncases, R.status);
            goto next;
        }
        {
            bool same = R.row == best && R.norm == min_norm;
            for (int r = 1; r <= DEG && same; r++)
                for (int c = 1; c <= DEG + 1 && same; c++) {
                    to_mpz(tmp, S.B[r][c]);
                    same = mpz_cmp(tmp, m.coeff[r][c]) == 0;
                }
            for (int j = 0; j <= DEG && same; j++) {
                mpz_set_ui(norm2, 1);
                for (int e = 0; e < j; e++) mpz_mul(norm2, norm2, skew);
                mpz_divexact(coef, m.coeff[best][j + 1], norm2);
                to_mpz(tmp, R.f[j]);
                same = mpz_cmp(tmp, coef) == 0;
            }
            if (!same && mismatches++ < 5)
                fprintf(stderr, "case %ld: mismatch (row %d vs CADO %d)\n", ncases, R.row, best);
        }
    next:
        LLL_clear(&m);
        for (int i = 0; i <= DEG; i++) mpz_clear(ft[i]);
        mpz_clear(gt[0]);
        mpz_clear(gt[1]);
    }
    fclose(fh);
    printf("%ld cases: %ld identical to CADO, %ld mismatches, %ld port failures (overflow/dependent)\n",
           ncases, ncases - mismatches - port_fail, mismatches, port_fail);
    printf("widest intermediate in the port: %d bits (Int<%d> holds %d)\n", g_max_bits, LIMBS, 32 * LIMBS - 1);
    return (mismatches || port_fail) ? 1 : 0;
}
