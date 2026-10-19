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
 *
 * Then what s23_ropt -rerank ranks a cell by (rs_exact_cell: the exact lognorm and CADO's
 * alpha) against the polynomial s23_ropt writes for that cell (rs_written_poly, translated,
 * content divided out), at 56 cells a polynomial. The c168 seeds with a content lattice
 * (s0051, d = 2; s0068, d = 3) supply content cells, and at least one must be checked.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include <algorithm>
#include "rhost.h"

using namespace s23;

/* |a - b| for agreement checks: 0 when equal (also both infinite), infinite when undefined */
static double adiff(double a, double b)
{
    if (a == b)
        return 0;
    const double d = fabs(a - b);
    return std::isnan(d) ? INFINITY : d;
}

int main(int argc, char **argv)
{
    long cells = 0, bad = 0, all_content = 0;
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
        /* what -rerank ranks a cell by (rs_exact_cell) vs the polynomial s23_ropt writes for it
           (rs_written_poly, content divided out): its CADO alpha and its lognorm. Cells: the 7
           rotations and u, v in -3..3, which include content cells on seeds with a content
           lattice. Checked with !(x < tol), so a NaN fails. */
        RsizePoly<RS_LP> R;
        bool xfail = !rs_size_poly(R, P);
        double worst_a = 0, worst_l = 0, worst_screen = 0;
        long ncont = 0;
        cio_poly W;
        cio_init(&W);
        auto check_cell = [&](int64_t u, int64_t v) {
            int64_t t = 0;
            const RsExact x = rs_exact_cell(P, R, u, v, t);
            unsigned long content = 0;
            Int<RS_LP> fw[6];
            bool ok = rs_written_poly(W, P, R, u, v, t, content) && content > 0;
            for (int i = 0; i <= RS_DEG && ok; i++)
                ok = from_mpz(fw[i], W.f[i]);
            double lc = 1;
            const double da = ok ? fabs(x.A - cio_alpha_rot(&W, 0, 0, cio_alpha_bound(), &lc)) : NAN;
            const double dl = ok ? fabs(x.L - rs_lognorm_at(fw, 0)) : NAN;
            if (!(da < 1e-9) || !(dl < 1e-9) || lc != 0)
                xfail = true;
            // The intermediate screen must score the same primitive polynomial,
            // including rotations whose content is divided out on export.
            for (unsigned long bound : {200UL, cio_alpha_bound()}) {
                double screen_content = 0;
                const double alpha = ok ? cio_alpha_rot(&W, 0, 0, bound, &screen_content) : NAN;
                const double key = ok ? rs_alpha_screen_key(P, u, v, x.L + log((double)content), bound, 1.3) : NAN;
                const double ds = fabs(key - (x.L + 1.3 * alpha));
                if (!(ds < 1e-9) || screen_content != 0) xfail = true;
                worst_screen = std::isnan(ds) ? INFINITY : std::max(worst_screen, ds);
            }
            worst_a = std::isnan(da) ? INFINITY : std::max(worst_a, da);
            worst_l = std::isnan(dl) ? INFINITY : std::max(worst_l, dl);
            ncont += content > 1;
        };
        for (int r = 0; r < 7 && !xfail; r++)
            check_cell(rot[r][0], rot[r][1]);
        for (int u = -3; u <= 3 && !xfail; u++)
            for (int v = -3; v <= 3; v++)
                check_cell(u, v);
        cio_clear(&W);
        /* the size model's warm-started skewness (rs_skewness) against CADO's root isolation
           (L2_skewness_d), along t-chains as rs_size runs them (each evaluation warm-starts
           from the last) at line minima and far along the lines: the lognorms must agree to a
           few ulps, and the line minima found both ways too */
        double worst_w = 0;
        long nw = 0;
        for (int64_t u : {0, 1, -2, 5}) {
            if (xfail)
                break;
            RsLine Lw, Lc;
            rs_line_min(Lw, R, u, 0, 0, 1 << 20);
            rs_warm_skew = false;
            rs_line_min(Lc, R, u, 0, 0, 1 << 20);
            rs_warm_skew = true;
            worst_w = std::max(worst_w, adiff(Lw.L_min, Lc.L_min));
            for (int64_t dv : {0LL, 1000LL, -1000000LL, 1000000000LL}) {
                Int<RS_LP> fr[6];
                if (!rs_rotate(fr, R, u, Lw.v_min + dv))
                    continue;
                double xh = 0;
                for (int64_t step = 1 << 20; step >= 1; step /= 4)
                    for (int dir = -1; dir <= 1; dir += 2) {
                        const double w = rs_lognorm_at(fr, Lw.t_min + dir * step, &xh);
                        const double c = rs_lognorm_at(fr, Lw.t_min + dir * step);
                        worst_w = std::max(worst_w, adiff(w, c));
                        nw++;
                    }
            }
        }
        const bool wbad = !(worst_w < 1e-13);
        printf("%s: warm-started skewness vs CADO's, %ld evaluations and 4 line minima: worst lognorm "
               "difference %.2e%s\n",
               argv[a], nw, worst_w, wbad ? "  MISMATCH" : "");
        bad += wbad;
        /* the size model's fast searches (rs_fast_search: t guesses, false-position band edges,
           kept doubling points) against the old ones (-oldsearch -exactskew), on 7 lines: line
           minima; band edges at three falling budgets (the -maxcells loop's order) then two
           rising ones (as -plan's step table follows it); and 16 knots across the last band.
           Edges may move by an ulp-level tie only (within 1e-9 of the band's width); knot
           lognorms are compared where both bands are the same. */
        double worst_s = 0, worst_e = 0, worst_k = 0;
        for (int64_t u : {0, 1, -1, 3, -4, 9, 20}) {
            if (xfail)
                break;
            RsLine Ln[2];
            for (int fast = 0; fast < 2; fast++) {
                rs_fast_search = fast;
                rs_warm_skew = fast;
                rs_line_min(Ln[fast], R, u, 0, 0, 1 << 20);
            }
            worst_s = std::max(worst_s, adiff(Ln[0].L_min, Ln[1].L_min));
            for (double b : {2.0, 1.5, 1.0, 3.0, 2.5}) {
                for (int fast = 0; fast < 2; fast++) {
                    rs_fast_search = fast;
                    rs_warm_skew = fast;
                    rs_line_band(Ln[fast], R, Ln[0].L_min + b, 1, 0);
                }
                const double w = std::max(1.0, (double)(Ln[0].v_hi - Ln[0].v_lo));
                worst_e = std::max({worst_e, fabs((double)(Ln[0].v_lo - Ln[1].v_lo)) / w,
                                    fabs((double)(Ln[0].v_hi - Ln[1].v_hi)) / w});
            }
            for (int fast = 0; fast < 2; fast++) {
                rs_fast_search = fast;
                rs_warm_skew = fast;
                rs_line_knots(Ln[fast], R, 16);
            }
            if (Ln[0].v_lo == Ln[1].v_lo && Ln[0].v_hi == Ln[1].v_hi)
                for (size_t i = 0; i < Ln[0].kL.size(); i++)
                    worst_k = std::max(worst_k, adiff(Ln[0].kL[i], Ln[1].kL[i]));
        }
        rs_fast_search = true;
        rs_warm_skew = true;
        const bool sbad = !(worst_s < 1e-12) || !(worst_e < 1e-9) || !(worst_k < 1e-12);
        printf("%s: fast vs old size-model searches, 7 lines: worst line minimum difference %.2e, band edge "
               "%.2e of the band, knot lognorm %.2e%s\n",
               argv[a], worst_s, worst_e, worst_k, sbad ? "  MISMATCH" : "");
        bad += sbad;
        const bool abad = xfail;
        printf("%s: re-rank values vs the written polynomials, 56 cells (%ld with content): worst alpha %.2e, "
               "lognorm %.2e, alpha-screen key %.2e%s\n",
               argv[a], ncont, worst_a, worst_l, worst_screen, abad ? "  MISMATCH" : "");
        all_content += ncont;
        bad += abad;
        cells += fcells;
        bad += fbad;
        cio_clear(&P);
    }
    if (all_content == 0) {
        printf("no content cell was checked: give a seed with a content lattice (c168 s0051, s0068)\n");
        bad++;
    }
    printf("all: %ld cells, %ld mismatches, worst |table - brute| %.2e; %ld content cells checked\n", cells, bad,
           worst, all_content);
    return bad ? 1 : 0;
}
