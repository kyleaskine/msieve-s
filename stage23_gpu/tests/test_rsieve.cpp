/* Root-sieve tables (include/rsieve.h) vs brute-force root counting, on real seeds.
 *
 *   test_rsieve POLYFILE ...
 *
 * For each polynomial (CADO format, degree 5): lines u in a fixed set (0, +-1, 7, 90,
 * 120, -5, and large u), cells v = -1000 .. 3000 and 2000 random large v per line; the
 * table score of every cell must equal the brute-force score (N_e counted by evaluating
 * f_{u,v} at every x mod p^e, for every prime power <= 200) to within float rounding.
 * One count off changes a score by at least log(199)/200 = 0.026. The brute force counts
 * projective roots too (from the full u and v), so this also checks the projective
 * tables' periodicity.
 *
 * Then the projective part on its own against CADO (get_alpha_projective, primes < 200)
 * at a few rotations: ours, to level emax + 3, must be within 0.02 of CADO's full value,
 * and differences between rotations within 0.01. The c168 winner is the case that showed
 * projective alpha is not constant under rotation (u even -2.7025, u odd -2.4715).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include <algorithm>
#include "rseed.h"

using namespace s23;

int main(int argc, char **argv)
{
    long cells = 0, bad = 0;
    double worst = 0;
    unsigned long long rng = 0x9e3779b97f4a7c15ull;
    for (int a = 1; a < argc; a++) {
        FILE *fh = fopen(argv[a], "r");
        if (!fh) {
            perror(argv[a]);
            return 2;
        }
        cio_poly P;
        cio_init(&P);
        if (cio_read_next(fh, &P) != 1) {
            fprintf(stderr, "%s: no polynomial\n", argv[a]);
            return 2;
        }
        fclose(fh);
        RsSeed S;
        if (!rs_seed_from_poly(S, P)) {
            fprintf(stderr, "%s: not degree 5\n", argv[a]);
            return 2;
        }
        std::vector<float> PT; /* projective values (rsieve.h) */
        rs_proj_tables(S, PT);
        static float T[8192];
        const int64_t us[] = {0, 1, -1, 7, 90, 120, -5, 1234567, -987654321};
        long fcells = 0, fbad = 0;
        for (int64_t u : us) {
            rs_line_tables(T, S, PT.data(), u);
            for (int pass = 0; pass < 2; pass++) {
                for (int j = 0; j < (pass ? 2000 : 4001); j++) {
                    int64_t v;
                    if (pass == 0)
                        v = -1000 + j;
                    else {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        v = (int64_t)(rng >> 3) - (int64_t)(1ull << 60);
                    }
                    double sb = rs_cell_score_brute(S, u, v);
                    float st = rs_cell_score(T, S, v);
                    double d = fabs(sb - st);
                    worst = d > worst ? d : worst;
                    fcells++;
                    if (d > 1e-4) {
                        if (fbad < 5)
                            printf("  %s: u %lld v %lld: table %.6f brute %.6f\n", argv[a], (long long)u,
                                   (long long)v, st, sb);
                        fbad++;
                    }
                }
            }
        }
        printf("%s: %ld cells, %ld mismatches\n", argv[a], fcells, fbad);
        /* projective part vs CADO */
        const long rot[][2] = {{0, 0}, {1, 0}, {2, 1}, {3, 5}, {-7, 12345}, {90, -977}, {1234567, -98765}};
        double ours[7], cado[7], worst_abs = 0, worst_diff = 0;
        for (int r = 0; r < 7; r++) {
            double sum = 0;
            for (int i = 0; i < RS_NPRIMES; i++) {
                if (S.poff[i] == RS_NOPROJ)
                    continue;
                const RsPrime &Pr = S.pr[i];
                double w[RS_WMAX];
                double q = Pr.p;
                for (uint32_t e = 1; e <= Pr.emax + RS_PROJ_EXTRA; e++, q *= Pr.p)
                    w[e - 1] = log((double)Pr.p) / (q + q / Pr.p);
                const int64_t m = Pr.qproj;
                sum += rs_proj_value<double>(S, i, (uint64_t)(((rot[r][0] % m) + m) % m),
                                             (uint64_t)(((rot[r][1] % m) + m) % m), w);
            }
            ours[r] = -sum;
            cado[r] = cio_alpha_projective_rot(&P, rot[r][0], rot[r][1], RS_BOUND);
            worst_abs = std::max(worst_abs, fabs(ours[r] - cado[r]));
            worst_diff = std::max(worst_diff, fabs((ours[r] - ours[0]) - (cado[r] - cado[0])));
        }
        const bool pbad = worst_abs > 0.02 || worst_diff > 0.01;
        printf("%s: projective vs CADO at 7 rotations: worst %.4f, worst difference %.4f; (0,0) %.4f, (1,0) %.4f "
               "(CADO %.4f, %.4f)%s\n",
               argv[a], worst_abs, worst_diff, ours[0], ours[1], cado[0], cado[1], pbad ? "  MISMATCH" : "");
        bad += pbad;
        cells += fcells;
        bad += fbad;
        cio_clear(&P);
    }
    printf("all: %ld cells, %ld mismatches, worst |table - brute| %.2e\n", cells, bad, worst);
    return bad ? 1 : 0;
}
