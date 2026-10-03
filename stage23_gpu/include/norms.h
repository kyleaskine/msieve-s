/* CADO's circular L2 lognorm and L2 skewness (polyselect/polyselect_norms.c), ported
 * expression for expression. Coefficients enter as doubles converted from the exact
 * integers with mpz_get_d semantics (truncation), as CADO's double_poly_set_mpz_poly.
 */
#pragma once

#include "dpoly.h"
#include "mpint.h"

namespace s23 {

/* coeffs_integral[d][k] from polyselect_norms.c, rows d = 0..8 */
#define S23_COEFFS_INTEGRAL                                                              \
    {                                                                                    \
        {1}, {2}, {1, 6}, {4, 20}, {3, 10, 70}, {12, 28, 252}, {10, 28, 84, 924},        \
            {40, 72, 264, 3432}, {35, 90, 198, 858, 12870},                              \
    }
#ifdef __CUDACC__
__constant__ uint32_t s23_coeffs_integral_dev[9][5] = S23_COEFFS_INTEGRAL;
#endif
static const uint32_t s23_coeffs_integral_host[9][5] = S23_COEFFS_INTEGRAL;

S23_HD uint32_t coeffs_integral(int d, int k)
{
#ifdef __CUDA_ARCH__
    return s23_coeffs_integral_dev[d][k];
#else
    return s23_coeffs_integral_host[d][k];
#endif
}

template <int L> S23_HD void dp_from_ints(DPoly &p, const Int<L> *f, int d)
{
    for (int i = 0; i <= d; i++)
        p.c[i] = get_d(f[i]);
    p.deg = d;
}

S23_HD double L2_lognorm_d(const DPoly &p, double s)
{
    const double *a = p.c;
    const int d = p.deg;
    const double invs = 1.0 / s;
    const double s2 = s * s;
    const double is2 = invs * invs;
    double tt = 0;
    double u0 = (d & 1) ? invs : 1;
    double u1 = (d & 1) ? s : 1;
    const int p0 = d / 2;
    const int p1 = d - p0;
    for (int k = 0; k <= d / 2; k++) {
        double t0 = 0;
        double t1 = 0;
        for (int i = 1; i <= d / 2 - k; i++) {
            t0 += a[p0 - k - i] * a[p0 - k + i];
            t1 += a[p1 + k - i] * a[p1 + k + i];
        }
        t0 = (2 * t0 + a[p0 - k] * a[p0 - k]) * u0;
        t1 = (2 * t1 + a[p1 + k] * a[p1 + k]) * u1;
        u0 *= is2;
        u1 *= s2;
        tt += (coeffs_integral(d, k)) * (t0 + t1);
    }
    tt = ldexp(tt * 3.14159265358979323846 / (double)(d + 1), -2 * d);
    if (isnan(tt) || isinf(tt))
        tt = DBL_MAX;
    return log(tt) / 2;
}

S23_HD void L2_skewness_derivative_numerator(DPoly &dP, const DPoly &p)
{
    const double *a = p.c;
    const int d = p.deg;
    for (int k = 0; k <= d; k++) {
        double t0 = 0;
        for (int i = 1; i <= k && i <= d - k; i++)
            t0 += a[k - i] * a[k + i];
        t0 = (2 * t0 + a[k] * a[k]);
        const int dk = d - 2 * k;
        const int j = ((dk < 0 ? -dk : dk) - (d & 1)) / 2;
        t0 = t0 * coeffs_integral(d, j) * (2 * k - d);
        dP.c[k] = t0;
    }
    dP.deg = d;
    dp_cleandeg(dP, d);
}

/* L2_skewness: the s minimizing the L2 lognorm (0 if no extremum was found) */
S23_HD double L2_skewness_d(const DPoly &P)
{
    DPoly dP;
    double roots[DP_MAX];
    L2_skewness_derivative_numerator(dP, P);
    const double B = dp_bound_roots(dP);
    const unsigned nroots = dp_compute_roots(roots, dP, B + 1);
    if (nroots < 1)
        return 0.0; /* CADO asserts here */
    double best_s = sqrt(roots[0]);
    if (nroots > 1) {
        double l = L2_lognorm_d(P, best_s);
        for (unsigned i = 1; i < nroots; i++) {
            const double s = sqrt(roots[i]);
            const double li = L2_lognorm_d(P, s);
            if (li < l) {
                best_s = s;
                l = li;
            }
        }
    }
    return best_s;
}

template <int L> S23_HD double L2_lognorm(const Int<L> *f, int d, double s)
{
    DPoly P;
    dp_from_ints(P, f, d);
    return L2_lognorm_d(P, s);
}

template <int L> S23_HD double L2_skewness(const Int<L> *f, int d)
{
    DPoly P;
    dp_from_ints(P, f, d);
    return L2_skewness_d(P);
}

template <int L> S23_HD double L2_skew_lognorm(const Int<L> *f, int d)
{
    DPoly P;
    dp_from_ints(P, f, d);
    return L2_lognorm_d(P, L2_skewness_d(P));
}

} // namespace s23
