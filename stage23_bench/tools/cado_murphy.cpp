/* MurphyE of CADO-format polynomials at their best skew (and optionally best
 * translation), with CADO's own code, in-process; or (-lattice) the relations a lattice
 * siever can expect over a special-q band.
 *
 *   cado_murphy [-Bf X] [-Bg Y] [-area Z] [-trans T] [-K N] [-Keval N] [-t THREADS]
 *               [-selftest] [FILE]
 *   cado_murphy [-Bf X] [-Bg Y] [-area Z] [-K N] [-Keval N] [-t THREADS]
 *               -lattice LOGI,J (-qband QMIN,QMAX,NQ | -qpoints Q1,Q2,...)
 *               [-nlat M] [-npts P] [-sqside 1|0] [-useskew] [FILE]
 *   cado_murphy -refine -K 16000 -Keval 64000 [-maxeval 384] [-trans T]
 *               [lattice options] [-eval-nlat 128] [-eval-npts 4096] [-eval-seed 1] [FILE]
 *
 * For each polynomial of FILE (n:, Y0:, Y1:, c0: ...; '#' lines and blank lines between
 * blocks are skipped) prints one tab-separated line, in input order:
 *
 *   index  MurphyE  skew  t  Y0  c0                         (default)
 *   index  band  skew  MurphyE  rel(q_1)  rel(q_2) ...       (-lattice)
 *   index start_skew train0 dt skew train1 eval0 eval1 E0 E1 calls limited Y0 c0 (-refine)
 *   index error message                                    (-refine failed input)
 * Refinement writes one success/error row per input; exit 0 means the batch was
 * processed. The wrapper reports partial failures with exit 1 after saving results.
 *
 * -lattice: rel(q) is the expected number of relations per special-q at q by MurphyE's
 * smoothness model (rho of log norm + alpha over log B, both sides) averaged over the region
 * the siever covers: for M (default 32) special-q lattices with uniform roots, reduced at
 * the polynomial's skew as the user's GPU siever reduces them, P (default 2048) points of
 * the 2^LOGI x J rectangle; the special-q side's (-sqside, default 1 = algebraic) norm is
 * divided by q. band is the trapezoid rule over the q points of rel(q) / ln q (about one
 * special-q per prime). The skew is MurphyE's best (as below), or with -useskew the file's
 * own (a poly without a skew line gets the best one, and a warning). MurphyE is printed at
 * that skew. Every polynomial gets the same random roots and sample points, so ratios
 * between polynomials have reduced sampling noise. -latseed S changes that stream (default
 * 0); independent streams check whether a gain survives different samples. On c208 the ratios match test sieves within
 * 0.04 (GPU_STAGE23_PLAN.md, known problem 7); the absolute rel(q) reads about 1.9x a test
 * sieve's.
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
 * can fit that noise: the old c168 translation gains were not real (GPU_STAGE23_PLAN.md,
 * "Review checks"). C181 has real translation sensitivity. For close polys use
 * -K 4000 -Keval 16000 (16,000 is within 0.02% of
 * 256,000 on the c168 winners; K = 4000 reads a median 0.05% high).
 *
 * -refine jointly searches t and log2(skew), retaining the starting configuration as
 * a control (the declared skew, or MurphyE's best if absent). Its search is bounded
 * to +/- 4 translation steps (default step = starting skew / 2) and skew /4 .. *4,
 * with at most -maxeval objective evaluations. It evaluates the proposal at Keval >=
 * 4*K, and in lattice mode on a separate sample stream with >= the search counts
 * (search floors: 32 lattices, 2048 points per lattice).
 * Validation never feeds back into the search. It reports both values, without
 * choosing between them; refine_polys.py exports both exact polynomials and selects.
 * -useskew also works without -lattice, to recheck a fixed exported configuration.
 *
 * Built on first use by polyfmt.cado_murphy_binary() against the configured CADO build.
 */
#include "cado.h" // IWYU pragma: keep

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <vector>
#include <map>
#include <limits>
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
#include "murphy_search.h"

using namespace s23_murphy;

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

/* ---- -lattice: expected relations per special-q over the region a lattice siever sieves ---- */
struct Lat {
    int logI = 0; /* 0: off */
    long J = 0;
    std::vector<double> q; /* evaluation points, increasing */
    int nlat = 32, npts = 2048, sqside = 1, useskew = 0;
    uint64_t seed = 0; /* independent validation stream; 0 preserves the original model */
};

static uint64_t splitmix(uint64_t &x)
{
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static double unif(uint64_t &x)
{
    return (double)(splitmix(x) >> 11) * 0x1.0p-53;
}

/* the user's GPU siever's reduction (~/code/cuda-sieve/bench/poly.c, qlat_build): Gauss
 * reduction of <(q, 0), (rho, 1)> under |(a, b)|^2 = a^2/s + b^2 s, (a0, a1) the shorter */
static void qlat(int64_t q, int64_t rho, double skew, int64_t &a0, int64_t &a1, int64_t &b0, int64_t &b1)
{
    a0 = q, a1 = 0, b0 = rho, b1 = 1;
    const double wa = 1.0 / skew, wb = skew;
    for (int guard = 0; guard < 200; guard++) {
        double na = wa * (double)a0 * a0 + wb * (double)a1 * a1;
        const double nb = wa * (double)b0 * b0 + wb * (double)b1 * b1;
        if (nb < na) {
            std::swap(a0, b0);
            std::swap(a1, b1);
            na = nb;
        }
        const double mu = (wa * (double)a0 * b0 + wb * (double)a1 * b1) / na;
        const int64_t m = (int64_t)(mu < 0 ? mu - 0.5 : mu + 0.5);
        if (m == 0)
            break;
        b0 -= m * a0;
        b1 -= m * a1;
    }
}

/* expected relations per special-q at q, by MurphyE's smoothness model (rho of the log
 * norm plus alpha over log B, both sides) averaged over the siever's region instead of
 * MurphyE's ellipse: for nlat special-q lattices (q, rho) with rho uniform, reduced at the
 * polynomial's skew, the points a = i a0 + j b0, b = i a1 + j b1 with i uniform in
 * [-I/2, I/2) and j in [0, J); the special-q side's norm is divided by q. Times
 * (6/pi^2) I J for coprimality. Every polynomial gets the same rho and (i, j) samples
 * (common random numbers), reducing noise in comparisons but not eliminating it. */
static double lattice_rel(const std::vector<double> &cf, double g0, double g1, double af, double ag, double skew,
                          double q, int qk, const Lat &L, const Ctx &C)
{
    const int d = (int)cf.size() - 1;
    const double I = ldexp(1.0, L.logI), J = (double)L.J, lq = log(q);
    const double ilf = 1.0 / log(C.Bf), ilg = 1.0 / log(C.Bg);
    double sum = 0;
    long n = 0;
    for (int m = 0; m < L.nlat; m++) {
        uint64_t st = 0x5eed000000000000ull ^ ((uint64_t)qk << 32) ^ (uint64_t)m ^
                      (L.seed * 0xd1342543de82ef95ull);
        const int64_t qi = (int64_t)q;
        const int64_t rho = 1 + (int64_t)(unif(st) * (double)(qi - 1));
        int64_t a0, a1, b0, b1;
        qlat(qi, rho, skew, a0, a1, b0, b1);
        for (int p = 0; p < L.npts; p++, n++) {
            const double i = (unif(st) - 0.5) * I, j = unif(st) * J;
            const double a = i * (double)a0 + j * (double)b0, b = i * (double)a1 + j * (double)b1;
            double F = cf[d], bp = 1;
            for (int k = d - 1; k >= 0; k--) {
                bp *= b;
                F = F * a + cf[k] * bp;
            }
            const double G = g1 * a + g0 * b;
            if (F == 0 || G == 0)
                continue;
            double vf = log(fabs(F)) + af, vg = log(fabs(G)) + ag;
            if (L.sqside == 1)
                vf -= lq;
            else
                vg -= lq;
            /* CADO's dickman_rho is 0 below 0: a norm under e^-alpha is smooth, not never */
            sum += dickman_rho(std::max(vf * ilf, 0.0)) * dickman_rho(std::max(vg * ilg, 0.0));
        }
    }
    return 6.0 / (PI * PI) * I * J * sum / (double)n;
}

static double lattice_band(cado_poly_srcptr p, double af, double ag, double s, const Lat &L, const Ctx &C,
                           std::vector<double> *samples = nullptr)
{
    mpz_poly_srcptr f = p->pols[ALG_SIDE], g = p->pols[RAT_SIDE];
    std::vector<double> cf(f->deg + 1), rel;
    for (int k = 0; k <= f->deg; k++)
        cf[k] = mpz_get_d(mpz_poly_coeff_const(f, k));
    const double g0 = mpz_get_d(mpz_poly_coeff_const(g, 0)), g1 = mpz_get_d(mpz_poly_coeff_const(g, 1));
    for (size_t k = 0; k < L.q.size(); k++)
        rel.push_back(lattice_rel(cf, g0, g1, af, ag, s, L.q[k], (int)k, L, C));
    double band = 0;
    for (size_t k = 0; k + 1 < L.q.size(); k++)
        band += 0.5 * (L.q[k + 1] - L.q[k]) * (rel[k] / log(L.q[k]) + rel[k + 1] / log(L.q[k + 1]));
    if (samples) *samples = std::move(rel);
    return band;
}

/* A bounded finalist search, separate from the legacy -trans path. Joint pattern
 * search in integer translation and log2(skew); alpha is invariant because both
 * polynomials are translated exactly. Keep the input as the validation control.
 * No validation values feed back into the search. The caller retains both polys. */
using Refined = s23_murphy::Refinement;

static int search_selftest()
{
    // The origin is the best coarse point. Its local search cannot cross the dip
    // to the better peak at (-288, 0). Adjacent starts return to the origin;
    // the separated second start must recover the other peak. That peak must be
    // wide enough to own (-256, 0), or the second start climbs back to the origin.
    auto objective = [](long t, double z) {
        const double x = t / 128.0;
        return 10 + std::max(-0.1*x*x-z*z, 0.1-2*(x+2.25)*(x+2.25)-10*z*z);
    };
    const auto full = pattern_search(128, 1000, objective);
    const auto exact = pattern_search(128, full.calls, objective);
    const auto short_run = pattern_search(128, full.calls-1, objective);
    const bool distinct = full.starts[0].t != full.starts[1].t || full.starts[0].z != full.starts[1].z;
    const bool ok = distinct && full.starts[0].t == 0 && full.starts[0].z == 0 &&
                    full.starts[1].t == -256 && full.starts[1].z == 0 &&
                    full.best.t == -288 && full.best.z == 0 && !full.limited &&
                    !exact.limited && exact.calls == full.calls && short_run.limited &&
                    short_run.calls == full.calls-1;
    printf("search selftest: distinct starts, second peak, exact budget: %s (%d evaluations)\n",
           ok ? "ok" : "FAIL", full.calls);
    return ok ? 0 : 1;
}

static Refined refine_pair(cado_poly p, const Ctx &C, const Ctx &CE, const Lat &L, const Lat &LE,
                          int maxeval, long trans)
{
    const double af = get_alpha(p->pols[ALG_SIDE], get_alpha_bound());
    const double ag = get_alpha(p->pols[RAT_SIDE], get_alpha_bound());
    return s23_murphy::refine_pair(p, af, ag, C, p->skew, maxeval, trans,
        [&](double skew) {
            return L.logI ? lattice_band(p, af, ag, skew, L, C) : murphy_at(p, skew, af, ag, C);
        }, [&](double skew) {
            const double e = murphy_at(p, skew, af, ag, CE);
            return std::array<double, 2>{L.logI ? lattice_band(p, af, ag, skew, LE, CE) : e, e};
        });
}

int main(int argc, char **argv)
{
    Ctx C = {1e7, 5e6, 1e16, MURPHY_K};
    int Keval = -1; /* -1: same as K */
    long trans = 0;
    int threads = 8, selftest = 0, searchtest = 0, refine = 0, maxeval = 384;
    int eval_nlat = 128, eval_npts = 4096;
    uint64_t eval_seed = 1;
    const char *path = NULL;
    Lat L;
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
        else if (!strcmp(argv[i], "-search-selftest"))
            searchtest = 1;
        else if (!strcmp(argv[i], "-refine"))
            refine = 1;
        else if (!strcmp(argv[i], "-maxeval") && i + 1 < argc)
            maxeval = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-eval-nlat") && i + 1 < argc)
            eval_nlat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-eval-npts") && i + 1 < argc)
            eval_npts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-latseed") && i + 1 < argc)
            L.seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-eval-seed") && i + 1 < argc)
            eval_seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-lattice") && i + 1 < argc) {
            if (sscanf(argv[++i], "%d,%ld", &L.logI, &L.J) != 2)
                L.logI = -1;
        } else if (!strcmp(argv[i], "-qband") && i + 1 < argc) {
            double q0, q1;
            int nq;
            if (sscanf(argv[++i], "%lf,%lf,%d", &q0, &q1, &nq) == 3 && nq >= 2 && q1 > q0)
                for (int k = 0; k < nq; k++)
                    L.q.push_back(floor(q0 + (q1 - q0) * k / (nq - 1)));
            else
                L.q.assign(1, -1.0);
        } else if (!strcmp(argv[i], "-qpoints") && i + 1 < argc) {
            for (char *t = strtok(argv[++i], ","); t; t = strtok(NULL, ","))
                L.q.push_back(floor(atof(t)));
        } else if (!strcmp(argv[i], "-nlat") && i + 1 < argc)
            L.nlat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-npts") && i + 1 < argc)
            L.npts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-sqside") && i + 1 < argc)
            L.sqside = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-useskew"))
            L.useskew = 1;
        else if (argv[i][0] != '-' && !path)
            path = argv[i];
        else {
            fprintf(stderr, "cado_murphy: bad argument '%s' (an option without its value, an unknown option, "
                            "or a second file)\nusage: cado_murphy [-Bf X] [-Bg Y] [-area Z] [-trans T] "
                            "[-K N] [-Keval N] [-t THREADS] [-selftest]\n"
                            "       [-lattice LOGI,J (-qband QMIN,QMAX,NQ | -qpoints Q1,Q2,...) [-nlat M] [-npts P] "
                            "[-latseed S] [-sqside 1|0]] [-useskew]\n"
                            "       [-refine [-maxeval N] [-eval-nlat M] [-eval-npts P] [-eval-seed S]] [FILE]\n",
                    argv[i]);
            return 2;
        }
    }
    if (searchtest) return search_selftest();
    if (Keval == -1)
        Keval = C.K;
    if (!std::isfinite(C.Bf) || !std::isfinite(C.Bg) || !std::isfinite(C.area) ||
        C.Bf <= 1 || C.Bg <= 1 || C.area <= 0 || trans < 0 || threads < 1 || C.K < 1 || Keval < 1) {
        fprintf(stderr, "cado_murphy: need Bf, Bg > 1, area > 0, trans >= 0, threads >= 1, K and Keval >= 1\n");
        return 2;
    }
    if (L.useskew && trans && !refine) {
        fprintf(stderr, "cado_murphy: -useskew evaluates a fixed configuration; use -refine for a joint search\n");
        return 2;
    }
    if (L.logI != 0) {
        bool ok = L.logI >= 2 && L.logI <= 30 && L.J >= 1 && !L.q.empty() && L.nlat >= 1 && L.npts >= 1 &&
                  (L.sqside == 0 || L.sqside == 1) && (trans == 0 || refine);
        for (size_t k = 0; k < L.q.size(); k++)
            ok = ok && L.q[k] >= 3 && L.q[k] <= 0x1p52 && (k == 0 || L.q[k] > L.q[k - 1]); /* exact in doubles */
        if (!ok) {
            fprintf(stderr, "cado_murphy: -lattice needs LOGI in [2, 30], J >= 1, increasing q points in [3, 2^52] "
                            "(-qband QMIN,QMAX,NQ with NQ >= 2, or -qpoints), -nlat/-npts >= 1, -sqside 0 or 1, "
                            "and no -trans unless -refine is enabled\n");
            return 2;
        }
    } else if (!L.q.empty()) {
        fprintf(stderr, "cado_murphy: -qband and -qpoints need -lattice\n");
        return 2;
    }
    if (refine && (selftest || maxeval < 32 || C.K < 16000 || Keval < 4 * (int64_t)C.K ||
                   trans > std::numeric_limits<long>::max() / 16 ||
                   (L.logI && (L.q.size() < 2 || L.nlat < 32 || L.npts < 2048 ||
                              eval_nlat < L.nlat || eval_npts < L.npts || eval_seed == L.seed)))) {
        fprintf(stderr, "cado_murphy: -refine needs K >= 16000, Keval >= 4*K, maxeval >= 32, a bounded translation "
                        "step, no -selftest; lattice refinement needs >= 2 q points, nlat >= 32, npts >= 2048, evaluation sample counts "
                        ">= search counts and an independent -eval-seed\n");
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
    if (refine) {
        Ctx CE = C;
        CE.K = Keval;
        Lat LE = L;
        LE.nlat = eval_nlat;
        LE.npts = eval_npts;
        LE.seed = eval_seed;
        std::vector<Refined> results(polys.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
        for (size_t i = 0; i < polys.size(); i++)
            results[i] = refine_pair(polys[i], C, CE, L, LE, maxeval, trans);
        for (size_t i = 0; i < polys.size(); i++) {
            const auto &v = results[i];
            if (!(v.eval0 > 0 && v.eval1 > 0 && std::isfinite(v.eval0) && std::isfinite(v.eval1))) {
                fprintf(stderr, "cado_murphy: invalid refinement score for polynomial %zu\n", i);
                printf("%zu\terror\tinvalid refinement score\n", i);
            } else
                gmp_printf("%zu\t%.17g\t%.17g\t%ld\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%d\t%d\t%Zd\t%Zd\n",
                       i, v.start_skew, v.train0, v.t, v.skew, v.train1, v.eval0, v.eval1, v.e0, v.e1,
                       v.calls, v.limited, mpz_poly_coeff_const(polys[i]->pols[RAT_SIDE], 0),
                       mpz_poly_coeff_const(polys[i]->pols[ALG_SIDE], 0));
            cado_poly_clear(polys[i]);
            free(polys[i]);
        }
        return 0;
    }
    if (L.logI > 0) { /* index, band total, skew, MurphyE at that skew, relations per special-q at each q */
        std::vector<std::vector<double>> rel(polys.size());
        std::vector<double> E(polys.size()), S(polys.size()), band(polys.size());
        std::vector<char> no_skew(polys.size(), 0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
        for (size_t i = 0; i < polys.size(); i++) {
            cado_poly_ptr p = polys[i];
            mpz_poly_srcptr f = p->pols[ALG_SIDE], g = p->pols[RAT_SIDE];
            const double af = get_alpha(f, get_alpha_bound());
            const double ag = get_alpha(g, get_alpha_bound());
            double s, e = 0;
            const bool declared = L.useskew && p->skew > 0 && std::isfinite(p->skew);
            if (declared)
                s = p->skew;
            else
                e = best_skew(p, af, ag, C, s);
            if (L.useskew && !declared)
                no_skew[i] = 1;
            if (declared || Keval != C.K) {
                Ctx CE = C;
                CE.K = Keval;
                e = murphy_at(p, s, af, ag, CE);
            }
            E[i] = e;
            S[i] = s;
            band[i] = lattice_band(p, af, ag, s, L, C, &rel[i]);
        }
        size_t nos = 0;
        for (char c : no_skew)
            nos += c;
        if (nos)
            fprintf(stderr, "cado_murphy: -useskew: %zu of %zu polynomials have no valid skew line; MurphyE's best skew "
                            "was used for them\n", nos, polys.size());
        for (size_t i = 0; i < polys.size(); i++) {
            printf("%zu\t%.6e\t%.1f\t%.6e", i, band[i], S[i], E[i]);
            for (double r : rel[i])
                printf("\t%.4f", r);
            printf("\n");
            cado_poly_clear(polys[i]);
            free(polys[i]);
        }
        return 0;
    }
    std::vector<double> E(polys.size()), S(polys.size());
    std::vector<long> T(polys.size());
    std::vector<char> no_skew(polys.size(), 0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t i = 0; i < polys.size(); i++) {
        cado_poly_ptr p = polys[i];
        const double af = get_alpha(p->pols[ALG_SIDE], get_alpha_bound());
        const double ag = get_alpha(p->pols[RAT_SIDE], get_alpha_bound());
        double s = p->skew;
        const bool declared = L.useskew && s > 0 && std::isfinite(s);
        double e = declared ? 0 : best_skew(p, af, ag, C, s);
        no_skew[i] = L.useskew && !declared;
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
        if (declared || Keval != C.K) {
            Ctx CE = C;
            CE.K = Keval;
            e = murphy_at(p, s, af, ag, CE);
        }
        E[i] = e;
        S[i] = s;
        T[i] = t;
    }
    size_t nos = 0;
    for (char c : no_skew) nos += c;
    if (nos)
        fprintf(stderr, "cado_murphy: -useskew: %zu of %zu polynomials have no valid skew line; MurphyE's best skew "
                        "was used for them\n", nos, polys.size());
    for (size_t i = 0; i < polys.size(); i++) {
        gmp_printf("%zu\t%.6e\t%.1f\t%ld\t%Zd\t%Zd\n", i, E[i], S[i], T[i], mpz_poly_coeff_const(polys[i]->pols[RAT_SIDE], 0),
                   mpz_poly_coeff_const(polys[i]->pols[ALG_SIDE], 0));
        cado_poly_clear(polys[i]);
        free(polys[i]);
    }
    return 0;
}
