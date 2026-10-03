/* Full-precision CADO size-optimization stats, using CADO's own code.
 *
 *   cado_expe [-keep-skew] [file ...]        (stdin if no file)
 *
 * Reads CADO-format polynomials (n:, Y0:, Y1:, c0: ..., optional skew:), e.g. a poly file
 * or a CADO sopt output (whose echoed raw inputs are '#'-commented and skipped), and
 * prints one tab-separated line per polynomial:
 *
 *   Y1  Y0  c0  skew  lognorm  exp_E  alpha  alpha_proj
 *
 * with 9 decimals (skew 6). Like CADO sopt, the skew is recomputed (CADO's
 * L2_combined_skewness2 of g and f) unless -keep-skew is given, so exp_E matches what sopt
 * would print, without its 2-decimal rounding. That is not quite the objective sopt
 * minimizes (lognorm at f's own L2 skew): see GPU_STAGE23_PLAN.md, M1.
 * Blank lines between polynomials are skipped (CADO's own reader stops at the first one,
 * which is the line before sopt output's first block). Exit status 1 if a block does not
 * parse or no polynomial was read.
 * This is the scorer for M1's acceptance rule (GPU_STAGE23_PLAN.md).
 *
 * Built on first use by polyfmt.cado_expe_binary() against the CADO build named in
 * nfs_config.ini (or $CADO_BUILD_DIR); it needs CADO's static libs and OpenMP.
 */
#include "cado.h" // IWYU pragma: keep
#include <stdio.h>
#include <string.h>
#include <gmp.h>
#include "auxiliary.h"
#include "cado_poly.h"
#include "mpz_poly.h"
#include "params.h"

/* CADO's cado_poly_set_plist (utils/cado_poly.c; exported, not declared in a header) */
int cado_poly_set_plist(cado_poly_ptr cpoly, param_list_ptr pl);

/* 1: a polynomial was read; 0: end of input; -1: a block that does not parse */
static int read_next(FILE *in, cado_poly cpoly)
{
    for (;;) {
        param_list pl;
        param_list_init(pl);
        int ok = param_list_read_stream(pl, in, 1);
        if (param_list_empty(pl)) {
            param_list_clear(pl);
            if (!ok || ferror(in))
                return -1;
            if (feof(in))
                return 0;
            continue; /* a blank line */
        }
        cado_poly_reset(cpoly);
        ok = ok && cado_poly_set_plist(cpoly, pl);
        param_list_clear(pl);
        return ok ? 1 : -1;
    }
}

/* number of polynomials scored, or -1 if a block did not parse */
static int score_stream(FILE *in, const char *name, int keep_skew)
{
    cado_poly cpoly;
    cado_poly_stats stats;
    int count = 0, r;

    cado_poly_init(cpoly);
    cado_poly_stats_init(stats, 2);
    while ((r = read_next(in, cpoly)) > 0) {
        if (!keep_skew)
            cpoly->skew = 0;
        cado_poly_set_skewness_if_undefined(cpoly);
        cado_poly_compute_expected_stats(stats, cpoly);
        mpz_poly_srcptr f = cpoly->pols[ALG_SIDE], g = cpoly->pols[RAT_SIDE];
        gmp_printf("%Zd\t%Zd\t%Zd\t%.6f\t%.9f\t%.9f\t%.9f\t%.9f\n",
                   mpz_poly_coeff_const(g, 1), mpz_poly_coeff_const(g, 0), mpz_poly_coeff_const(f, 0),
                   cpoly->skew,
                   stats->pols[ALG_SIDE]->lognorm, stats->pols[ALG_SIDE]->exp_E,
                   stats->pols[ALG_SIDE]->alpha, stats->pols[ALG_SIDE]->alpha_proj);
        count++;
    }
    if (r < 0)
        fprintf(stderr, "cado_expe: %s: bad polynomial block after %d polynomials\n", name, count);
    cado_poly_stats_clear(stats);
    cado_poly_clear(cpoly);
    return r < 0 ? -1 : count;
}

int main(int argc, char *argv[])
{
    int keep_skew = 0, nfiles = 0, total = 0, bad = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-keep-skew")) {
            keep_skew = 1;
            continue;
        }
        FILE *fh = fopen(argv[i], "r");
        if (!fh) {
            perror(argv[i]);
            return 1;
        }
        int c = score_stream(fh, argv[i], keep_skew);
        fclose(fh);
        bad |= c < 0;
        total += c > 0 ? c : 0;
        nfiles++;
    }
    if (nfiles == 0) {
        int c = score_stream(stdin, "stdin", keep_skew);
        bad |= c < 0;
        total += c > 0 ? c : 0;
    }
    if (!bad && total == 0)
        fprintf(stderr, "cado_expe: no polynomials read\n");
    return bad || total == 0;
}
