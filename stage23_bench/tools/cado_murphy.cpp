/* MurphyE of CADO-format polynomials at their best skew (and optionally best
 * translation), with CADO's own code, in-process.
 *
 *   cado_murphy [-Bf X] [-Bg Y] [-area Z] [-trans T] [-K N] [-Keval N] [-t THREADS]
 *               [-selftest] [FILE]
 *
 * For each polynomial of FILE (n:, Y0:, Y1:, c0: ...; '#' lines and blank lines between
 * blocks are skipped) prints one tab-separated line, in input order:
 *
 *   index  MurphyE  skew  t  Y0  c0
 *
 * MurphyE is CADO's (polyselect/murphyE.cpp, alpha to get_alpha_bound()) with alpha
 * computed once per polynomial instead of once per skew; the integration loop is CADO's,
 * so values equal MurphyE() at the same skew and K (checked by -selftest). -K N sets the
 * number of sample angles for the skew and translation search (default CADO's 1000);
 * -Keval N recomputes the printed MurphyE at the chosen skew and translation with N
 * samples (default: the same K). The skew
 * search is a fixed grid, 2^(j/32) over s0/8 .. 8 s0 (s0 = |c0/c5|^(1/5)), then
 * 2^(j/256) around the best two grid points, as score_polys.py does: at K = 1000 MurphyE
 * has many spurious peaks in skew (sampling noise; at 64,000 points the curve has one).
 * -trans T also searches the translation x -> x + t (pattern search on MurphyE from step T
 * down to 1; alpha does not change), and prints the translated pair's Y0 and c0 so it can
 * be rebuilt. Defaults are CADO's bounds (1e7, 5e6, 1e16).
 *
 * K = 1000 reads high by about 0.2% (up to 1.4%), and searching skew or translation on it
 * fits that noise: the translation gains it finds are not real (GPU_STAGE23_PLAN.md,
 * "Review checks"). For close polys use -K 4000 -Keval 16000 (16,000 is within 0.02% of
 * 256,000 on the c168 winners; K = 4000 reads a median 0.05% high).
 *
 * Built on first use by polyfmt.cado_murphy_binary() against the configured CADO build.
 */
#include "cado.h" // IWYU pragma: keep
#define PI 3.14159265358979324

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <gmp.h>
#include "auxiliary.h"
#include "cado_poly.h"
#include "rho.h"
#include "murphyE.h"
#include "polynomial.hpp"
#include "mpz_poly.h"
#include "cxx_mpz.hpp"
#include "params.h"
#include "polyselect_alpha.h"

extern "C" int cado_poly_set_plist(cado_poly_ptr cpoly, param_list_ptr pl);

/* 1: read; 0: end; -1: bad block (blank lines between blocks skipped) */
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
            continue;
        }
        cado_poly_reset(cpoly);
        ok = ok && cado_poly_set_plist(cpoly, pl);
        param_list_clear(pl);
        return ok ? 1 : -1;
    }
}

struct Ctx {
    double Bf, Bg, area;
    int K; /* sample angles */
};

/* CADO's MurphyE loop with alpha given */
static double murphy_at(cado_poly_srcptr cpoly, double skew, double alpha_f, double alpha_g, const Ctx &C)
{
    double E = 0;
    const int K = C.K;
    const double x = sqrt(C.area * skew);
    const double y = sqrt(C.area / skew);
    const polynomial<cxx_mpz> f(cpoly->pols[ALG_SIDE]);
    const polynomial<cxx_mpz> g(cpoly->pols[RAT_SIDE]);
    const double one_over_logBf = 1.0 / log(C.Bf);
    const double one_over_logBg = 1.0 / log(C.Bg);
    for (int i = 0; i < K; i++) {
        const double ti = PI / (double)K * ((double)i + 0.5);
        const double xi = x * cos(ti);
        const double yi = y * sin(ti);
        double vf = log(std::abs(f(xi, yi))) + alpha_f;
        double vg = log(std::abs(g(xi, yi))) + alpha_g;
        vf *= one_over_logBf;
        vg *= one_over_logBg;
        E += dickman_rho(vf) * dickman_rho(vg);
    }
    return E / (double)K;
}

/* best skew on the fixed grid; returns E, sets skew */
static double best_skew(cado_poly_srcptr cpoly, double alpha_f, double alpha_g, const Ctx &C, double &skew)
{
    mpz_poly_srcptr f = cpoly->pols[ALG_SIDE];
    const int d = f->deg;
    const double s0 = pow(fabs(mpz_get_d(mpz_poly_coeff_const(f, 0)) / mpz_get_d(mpz_poly_coeff_const(f, d))),
                          1.0 / d);
    const int j0 = (int)floor(32 * log2(s0));
    double e1 = -1, e2 = -1;
    int j1 = j0, j2 = j0;
    for (int j = j0 - 96; j <= j0 + 96; j++) {
        const double e = murphy_at(cpoly, exp2(j / 32.0), alpha_f, alpha_g, C);
        if (e > e1) {
            e2 = e1;
            j2 = j1;
            e1 = e;
            j1 = j;
        } else if (e > e2) {
            e2 = e;
            j2 = j;
        }
    }
    double best = e1;
    skew = exp2(j1 / 32.0);
    for (int jj : {j1, j2})
        for (int k = -7; k <= 7; k++) {
            if (!k)
                continue;
            const double s = exp2((8 * jj + k) / 256.0);
            const double e = murphy_at(cpoly, s, alpha_f, alpha_g, C);
            if (e > best) {
                best = e;
                skew = s;
            }
        }
    return best;
}

/* f(x + t), g(x + t) in place */
static void translate_pair(cado_poly cpoly, long t)
{
    cxx_mpz tt;
    mpz_set_si(tt, t);
    mpz_poly_translation(cpoly->pols[ALG_SIDE], cpoly->pols[ALG_SIDE], tt);
    mpz_poly_translation(cpoly->pols[RAT_SIDE], cpoly->pols[RAT_SIDE], tt);
}

int main(int argc, char **argv)
{
    Ctx C = {1e7, 5e6, 1e16, MURPHY_K};
    int Keval = -1; /* -1: same as K */
    long trans = 0;
    int threads = 8, selftest = 0;
    const char *path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-Bf") && i + 1 < argc)
            C.Bf = atof(argv[++i]);
        else if (!strcmp(argv[i], "-Bg") && i + 1 < argc)
            C.Bg = atof(argv[++i]);
        else if (!strcmp(argv[i], "-area") && i + 1 < argc)
            C.area = atof(argv[++i]);
        else if (!strcmp(argv[i], "-trans") && i + 1 < argc)
            trans = atol(argv[++i]);
        else if (!strcmp(argv[i], "-K") && i + 1 < argc)
            C.K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-Keval") && i + 1 < argc)
            Keval = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc)
            threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-selftest"))
            selftest = 1;
        else if (argv[i][0] != '-' && !path)
            path = argv[i];
        else {
            fprintf(stderr, "cado_murphy: bad argument '%s' (an option without its value, an unknown option, "
                            "or a second file)\nusage: cado_murphy [-Bf X] [-Bg Y] [-area Z] [-trans T] "
                            "[-K N] [-Keval N] [-t THREADS] [-selftest] [FILE]\n",
                    argv[i]);
            return 2;
        }
    }
    if (Keval == -1)
        Keval = C.K;
    if (C.Bf <= 1 || C.Bg <= 1 || C.area <= 0 || trans < 0 || threads < 1 || C.K < 1 || Keval < 1) {
        fprintf(stderr, "cado_murphy: need Bf, Bg > 1, area > 0, trans >= 0, threads >= 1, K and Keval >= 1\n");
        return 2;
    }
    FILE *in = path ? fopen(path, "r") : stdin;
    if (!in) {
        perror(path);
        return 1;
    }
    std::vector<cado_poly_ptr> polys;
    int r;
    for (;;) {
        cado_poly_ptr p = (cado_poly_ptr)malloc(sizeof(cado_poly_s));
        cado_poly_init(p);
        r = read_next(in, p);
        if (r <= 0) {
            cado_poly_clear(p);
            free(p);
            break;
        }
        polys.push_back(p);
    }
    if (r < 0) {
        fprintf(stderr, "cado_murphy: bad polynomial block after %zu polynomials\n", polys.size());
        return 1;
    }
    if (selftest) { /* our loop vs CADO's MurphyE at a few skews */
        int bad = 0;
        for (auto *p : polys) {
            const double af = get_alpha(p->pols[ALG_SIDE], get_alpha_bound());
            const double ag = get_alpha(p->pols[RAT_SIDE], get_alpha_bound());
            for (double s : {1e3, 1e5, 3.3e6, 2e8}) {
                p->skew = s;
                const double a = MurphyE(p, C.Bf, C.Bg, C.area, C.K, get_alpha_bound());
                const double b = murphy_at(p, s, af, ag, C);
                bad += a != b;
            }
        }
        printf("selftest: %zu polys x 4 skews, K = %d: %d differ from CADO's MurphyE\n", polys.size(), C.K, bad);
        return bad != 0;
    }
    std::vector<double> E(polys.size()), S(polys.size());
    std::vector<long> T(polys.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t i = 0; i < polys.size(); i++) {
        cado_poly_ptr p = polys[i];
        const double af = get_alpha(p->pols[ALG_SIDE], get_alpha_bound());
        const double ag = get_alpha(p->pols[RAT_SIDE], get_alpha_bound());
        double s, e = best_skew(p, af, ag, C, s);
        long t = 0;
        for (long step = trans; step >= 1;) {
            bool moved = false;
            for (int dir = 1; dir >= -1 && !moved; dir -= 2) {
                translate_pair(p, dir * step);
                double s2, e2 = best_skew(p, af, ag, C, s2);
                if (e2 > e) {
                    e = e2;
                    s = s2;
                    t += dir * step;
                    moved = true;
                } else
                    translate_pair(p, -dir * step);
            }
            if (moved)
                step *= 2;
            else
                step /= 2;
        }
        if (Keval != C.K) {
            Ctx CE = C;
            CE.K = Keval;
            e = murphy_at(p, s, af, ag, CE);
        }
        E[i] = e;
        S[i] = s;
        T[i] = t;
    }
    for (size_t i = 0; i < polys.size(); i++) {
        gmp_printf("%zu\t%.6e\t%.1f\t%ld\t%Zd\t%Zd\n", i, E[i], S[i], T[i], mpz_poly_coeff_const(polys[i]->pols[RAT_SIDE], 0),
                   mpz_poly_coeff_const(polys[i]->pols[ALG_SIDE], 0));
        cado_poly_clear(polys[i]);
        free(polys[i]);
    }
    return 0;
}
