/* See cado_io.h. Compiled as C against the CADO build named in nfs_config.ini. */
#include "cado.h" // IWYU pragma: keep
#include <math.h>
#include <stdio.h>
#include <gmp.h>
#include "auxiliary.h"
#include "cado_poly.h"
#include "mpz_poly.h"
#include "params.h"
#include "polyselect_alpha.h"
#include "size_optimization.h"
#include "cado_io.h"
#include "sopt_skew.h"

void cio_init(cio_poly *p)
{
    p->deg = -1;
    mpz_init(p->n);
    for (int i = 0; i <= CIO_MAXDEG; i++)
        mpz_init(p->f[i]);
    mpz_init(p->g[0]);
    mpz_init(p->g[1]);
}

void cio_clear(cio_poly *p)
{
    mpz_clear(p->n);
    for (int i = 0; i <= CIO_MAXDEG; i++)
        mpz_clear(p->f[i]);
    mpz_clear(p->g[0]);
    mpz_clear(p->g[1]);
}

void cio_set(cio_poly *dst, const cio_poly *src)
{
    dst->deg = src->deg;
    mpz_set(dst->n, src->n);
    for (int i = 0; i <= CIO_MAXDEG; i++)
        mpz_set(dst->f[i], src->f[i]);
    mpz_set(dst->g[0], src->g[0]);
    mpz_set(dst->g[1], src->g[1]);
}

static void to_cado(cado_poly cpoly, const cio_poly *p)
{
    while (cpoly->nb_polys < 2) /* side 0 = rational (g), side 1 = algebraic (f) */
        cado_poly_provision_new_poly(cpoly);
    mpz_set(cpoly->n, p->n);
    mpz_poly_set_zero(cpoly->pols[ALG_SIDE]);
    for (int i = 0; i <= p->deg; i++)
        mpz_poly_setcoeff(cpoly->pols[ALG_SIDE], i, p->f[i]);
    mpz_poly_set_zero(cpoly->pols[RAT_SIDE]);
    mpz_poly_setcoeff(cpoly->pols[RAT_SIDE], 0, p->g[0]);
    mpz_poly_setcoeff(cpoly->pols[RAT_SIDE], 1, p->g[1]);
    cpoly->skew = 0;
}

/* 0 if f does not fit (degree above CIO_MAXDEG) or g is not linear */
static int from_mpz_polys(cio_poly *p, mpz_poly_srcptr f, mpz_poly_srcptr g)
{
    if (f->deg > CIO_MAXDEG || g->deg != 1)
        return 0;
    p->deg = f->deg;
    for (int i = 0; i <= CIO_MAXDEG; i++)
        mpz_set_ui(p->f[i], 0);
    for (int i = 0; i <= f->deg; i++)
        mpz_set(p->f[i], mpz_poly_coeff_const(f, i));
    mpz_set(p->g[0], mpz_poly_coeff_const(g, 0));
    mpz_set(p->g[1], mpz_poly_coeff_const(g, 1));
    return 1;
}

/* CADO's cado_poly_set_plist (utils/cado_poly.c; exported, not declared in a header) */
int cado_poly_set_plist(cado_poly_ptr cpoly, param_list_ptr pl);

int cio_read_next(FILE *in, cio_poly *p)
{
    /* as cado_poly_read_next_poly_from_stream, except that blank lines between blocks are
     * skipped (CADO stops at the first one) and a bad block is told apart from the end */
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
        cado_poly cpoly;
        cado_poly_init(cpoly);
        ok = ok && cado_poly_set_plist(cpoly, pl) && cpoly->nb_polys == 2;
        if (ok) {
            mpz_set(p->n, cpoly->n);
            ok = from_mpz_polys(p, cpoly->pols[ALG_SIDE], cpoly->pols[RAT_SIDE]);
            if (!ok)
                fprintf(stderr, "cio_read_next: unsupported polynomial pair (degree of f above %d, "
                                "or g not linear)\n", CIO_MAXDEG);
        }
        cado_poly_clear(cpoly);
        param_list_clear(pl);
        return ok ? 1 : -1;
    }
}

void cio_sopt_skew(mpz_t skew, const cio_poly *raw)
{
    sopt_skew_mpz(skew, raw->g[0], raw->f[raw->deg], raw->deg);
}

double cio_cado_sopt(cio_poly *opt, const cio_poly *raw, unsigned effort)
{
    mpz_poly f, g, fo, go;
    mpz_poly_init(f, raw->deg);
    mpz_poly_init(g, 1);
    mpz_poly_init(fo, raw->deg);
    mpz_poly_init(go, 1);
    for (int i = 0; i <= raw->deg; i++)
        mpz_poly_setcoeff(f, i, raw->f[i]);
    mpz_poly_setcoeff(g, 0, raw->g[0]);
    mpz_poly_setcoeff(g, 1, raw->g[1]);
    double v = size_optimization(fo, go, f, g, effort, 0);
    mpz_set(opt->n, raw->n);
    from_mpz_polys(opt, fo, go); /* same degrees as raw, which fit */
    mpz_poly_clear(f);
    mpz_poly_clear(g);
    mpz_poly_clear(fo);
    mpz_poly_clear(go);
    return v;
}

double cio_alpha_projective_rot(const cio_poly *p, long u, long v, unsigned long B)
{
    mpz_poly f;
    mpz_t c, t;
    mpz_poly_init(f, p->deg);
    mpz_init(c);
    mpz_init(t);
    for (int i = 0; i <= p->deg; i++)
        mpz_poly_setcoeff(f, i, p->f[i]);
    /* + (u x + v)(g1 x + g0): x^2 u g1, x (u g0 + v g1), 1 v g0 */
    mpz_mul_si(t, p->g[1], u);
    mpz_add(c, p->f[2], t);
    mpz_poly_setcoeff(f, 2, c);
    mpz_mul_si(t, p->g[0], u);
    mpz_add(c, p->f[1], t);
    mpz_mul_si(t, p->g[1], v);
    mpz_add(c, c, t);
    mpz_poly_setcoeff(f, 1, c);
    mpz_mul_si(t, p->g[0], v);
    mpz_add(c, p->f[0], t);
    mpz_poly_setcoeff(f, 0, c);
    const double a = get_alpha_projective(f, B);
    mpz_clear(t);
    mpz_clear(c);
    mpz_poly_clear(f);
    return a;
}

void cio_print_pair(FILE *out, unsigned idx, const cio_poly *raw, const cio_poly *opt, int raw_stats)
{
    cado_poly cpoly;
    cado_poly_stats stats;
    cado_poly_init(cpoly);
    cado_poly_stats_init(stats, 2);

    fprintf(out, "\n### Input raw polynomial (%u) ###\n", idx);
    to_cado(cpoly, raw);
    cado_poly_set_skewness_if_undefined(cpoly);
    cado_poly_fprintf(out, "# ", cpoly);
    if (raw_stats) {
        cado_poly_compute_expected_stats(stats, cpoly);
        cado_poly_fprintf_stats(out, "# ", cpoly, stats);
    }

    fprintf(out, "### Size-optimized polynomial (%u) ###\n", idx);
    to_cado(cpoly, opt);
    cado_poly_set_skewness_if_undefined(cpoly);
    cado_poly_compute_expected_stats(stats, cpoly);
    cado_poly_fprintf(out, NULL, cpoly);
    cado_poly_fprintf_stats(out, NULL, cpoly, stats);

    cado_poly_stats_clear(stats);
    cado_poly_clear(cpoly);
}
