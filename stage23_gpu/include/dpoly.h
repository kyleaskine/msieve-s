/* Double-precision polynomials and real-root finding, ported from CADO-NFS
 * utils/double_poly.cpp expression for expression (the same Horner forms, bounds,
 * false-position and derivative-cascade root isolation), so the roots, and everything
 * the size optimization derives from them, are the same doubles CADO gets.
 *
 * Rounding: CADO's double_poly.cpp is C++ built with -O3 -march=native, where GCC fuses
 * multiply-adds (objdump: the Horner steps of double_poly_eval and a*pb in false
 * position), while CADO's C code (norms, alpha, size optimization) is C99 and unfused.
 * Our builds disable automatic contraction everywhere (-ffp-contract=off, --fmad=false)
 * and write exactly CADO's fused operations as explicit fma() calls here.
 */
#pragma once

#include <float.h>
#include <math.h>
#include "hd.h"

namespace s23 {

enum { DP_MAX = 8 }; /* coefficients; enough for degree <= 7 */

struct DPoly {
    int deg;
    double c[DP_MAX];
};

/* CADO's double_poly_eval, as GCC compiles it (C++, -O3 -march=native): every Horner
 * step is a fused multiply-add, acc = fma(acc, x, f[k]), for every degree. Both our
 * builds disable automatic contraction, so this is written out explicitly. */
S23_HD double dp_eval(const DPoly &p, const double x)
{
    if (p.deg < 0)
        return 0;
    double r = p.c[p.deg];
    for (int k = p.deg - 1; k >= 0; k--)
        r = fma(r, x, p.c[k]);
    return r;
}

S23_HD void dp_cleandeg(DPoly &f, int deg)
{
    for (; deg >= 0 && f.c[deg] == 0; deg--)
        ;
    f.deg = deg;
}

S23_HD void dp_derivative(DPoly &df, const DPoly &f)
{
    for (int n = 1; n <= f.deg; n++)
        df.c[n - 1] = f.c[n] * (double)n;
    df.deg = f.deg - 1;
}

S23_HD double dp_dichotomy(const DPoly &p, double a, double b, double sa)
{
    double s;
    for (;;) {
        s = (a + b) * 0.5;
        if (s == a || s == b)
            return s;
        if (dp_eval(p, s) * sa > 0)
            a = s;
        else
            b = s;
    }
}

/* weighted false position on [a, b], assuming a single sign change */
S23_HD double dp_falseposition(const DPoly &p, double a, double b, double pa)
{
    double pb;
    int side = 0;
    const double a0 = a, b0 = b, pa0 = pa;
    pb = dp_eval(p, b);
    for (;;) {
        /* as CADO's compiled code: b*pa rounded, a*pb fused (vmulsd + vfmsub231sd) */
        double s = fma(a, pb, -(b * pa)) / (pb - pa);
        double middle = (a + b) * 0.5;
        if (s < a || s > b || ((s == a || s == b) && !(middle == a || middle == b)))
            s = middle;
        if (s == a || s == b)
            return s;
        const double ps = dp_eval(p, s);
        if (ps * pa > 0) {
            a = s;
            pa = ps;
            if (side == 1)
                pb /= 2;
            side = 1;
        } else {
            b = s;
            pb = ps;
            if (side == -1)
                pa /= 2;
            side = -1;
        }
        if (isnan(b))
            return dp_dichotomy(p, a0, b0, pa0);
    }
}

S23_HD unsigned recurse_roots(const DPoly &poly, double *roots, const unsigned sign_changes, const double s)
{
    unsigned new_sign_changes = 0;
    if (poly.deg <= 0) {
    } else if (poly.deg == 1) {
        if (poly.c[0] * dp_eval(poly, s) < 0) {
            new_sign_changes = 1;
            roots[0] = -poly.c[0] / poly.c[1];
        }
    } else {
        double a = 0.0;
        double va = poly.c[0];
        for (unsigned l = 0; l <= sign_changes; l++) {
            const double b = (l < sign_changes) ? roots[l] : s;
            const double vb = dp_eval(poly, b);
            if (va * vb < 0)
                roots[new_sign_changes++] = dp_falseposition(poly, a, b, va);
            a = b;
            va = vb;
        }
    }
    return new_sign_changes;
}

/* bound on the positive roots (double_poly_bound_roots) */
S23_HD double dp_bound_roots(const DPoly &p)
{
    const int d = p.deg;
    DPoly q;
    double s = p.c[d] > 0 ? 1.0 : -1.0;
    for (int i = 0; i < d; i++)
        q.c[i] = (s * p.c[i] < 0) ? s * p.c[i] : 0.0;
    q.c[d] = fabs(p.c[d]);
    q.deg = d;
    dp_cleandeg(q, d);
    s = 1.0;
    while (dp_eval(q, s) < 0)
        s = s + s;
    return s;
}

/* all roots in (0, s] (double_poly_compute_roots) */
S23_HD unsigned dp_compute_roots(double *roots, const DPoly &poly, double s)
{
    const int d = poly.deg;
    if (d <= 0)
        return 0;
    DPoly dg[DP_MAX];
    dg[0] = poly;
    for (int k = 1; k < d; k++)
        dp_derivative(dg[k], dg[k - 1]);
    unsigned sign_changes = 0;
    for (int k = d; k > 0; k--)
        sign_changes = recurse_roots(dg[k - 1], roots, sign_changes, s);
    return sign_changes;
}

S23_HD void dp_neg_x(DPoly &r, const DPoly &s)
{
    int i;
    r.deg = s.deg;
    for (i = 0; i < s.deg; i += 2) {
        r.c[i] = s.c[i];
        r.c[i + 1] = -s.c[i + 1];
    }
    if (i == s.deg)
        r.c[i] = s.c[i];
}

/* all real roots with |root| <= B (double_poly_compute_all_roots_with_bound) */
S23_HD unsigned dp_compute_all_roots_with_bound(double *roots, const DPoly &poly, double B)
{
    double bound = dp_bound_roots(poly);
    bound = (B < bound) ? B : bound; /* std::min(bound, B) */
    const unsigned nr_roots_pos = dp_compute_roots(roots, poly, bound);
    DPoly t;
    dp_neg_x(t, poly);
    bound = dp_bound_roots(t);
    bound = (B < bound) ? B : bound;
    unsigned nr_roots_neg = dp_compute_roots(roots + nr_roots_pos, t, bound);
    for (unsigned i = 0; i < nr_roots_neg; i++)
        roots[nr_roots_pos + i] *= -1.;
    if (poly.c[0] == 0.0)
        roots[nr_roots_pos + nr_roots_neg++] = 0.0;
    return nr_roots_pos + nr_roots_neg;
}

S23_HD unsigned dp_compute_all_roots(double *roots, const DPoly &poly)
{
    return dp_compute_all_roots_with_bound(roots, poly, DBL_MAX);
}

} // namespace s23
