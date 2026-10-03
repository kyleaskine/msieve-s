/* CADO-NFS size optimization (polyselect/size_optimization.c, size_optimization_aux
 * with max_rot = d - 2), ported to fixed-width integers and doubles for host and
 * device, degree 5:
 *
 *   translation candidates k (closed-form quadratics, rational approximations of the
 *   q2 roots, plus Farey fractions per unit of effort) and k = 0, sorted, deduplicated
 *   -> for each k: best_norm2 (exact LLL via best_norm, recursing on the roots of a
 *      quadratic in k) -> skip a k already optimized -> local descent on translation
 *      and rotations (up to 300 steps) -> lognorm + expected_rotation_gain
 *   -> keep the lowest (that sum is exp_E).
 *
 * Polynomials live in Int<LP> (a few hundred bits); the LLL and the discriminant use
 * Int<LL> (thousands of bits) in caller-provided scratch, since they are too large for
 * a GPU thread's stack. The skew used by the LLL is computed by the caller, exactly as
 * CADO's sopt_get_skewness (see sopt_lll.h).
 */
#pragma once

#include <chrono>
#include "alpha_proj.h"
#include "norms.h"
#include "sopt_lll.h"

namespace s23 {

/* SOPT_TIMEOUT: the deadline passed (the caller redoes the polynomial elsewhere) */
enum { SOPT_OK = 0, SOPT_FAIL = 1, SOPT_TIMEOUT = 2 };
#ifndef S23_SOPT_MAX_RECURSION
#define S23_SOPT_MAX_RECURSION 128
#endif
#ifndef S23_SOPT_MAX_EFFORT
#define S23_SOPT_MAX_EFFORT 100
#endif
/* translations_deg5 gives at most 2 roots k for each of 16 rational approximations of
 * each of the (at most 2) q2 roots and 16 Farey fractions per unit of effort; plus k = 0 */
enum {
    SOPT_MAX_EFFORT = S23_SOPT_MAX_EFFORT,
    SOPT_MAX_K = 2 * 16 * (2 + SOPT_MAX_EFFORT) + 1,
    SOPT_MAX_RECURSION = S23_SOPT_MAX_RECURSION
};

/* nanoseconds, for deadlines: the global timer on the device, steady_clock on the host */
S23_HD uint64_t now_ns()
{
#ifdef __CUDA_ARCH__
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#else
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
#endif
}

/* one level of best_norm2's recursion, kept in scratch (the GPU stack is small) */
template <int LP, int DEG>
struct Bn2Frame {
    Int<LP> ft[DEG + 1], gt[2], k, best_k;
    double min_norm;
    double roots[DP_MAX];
    unsigned nroots, l;
};

template <int LP, int LL, int DEG>
struct SoptScratch {
    Bn2Frame<LP, DEG> fr[SOPT_MAX_RECURSION + 1];
    LLLState<LL, DEG, DEG + 1> lll;
    SoptLLLResult<LL, DEG> res;
    BestNormWork<LL, DEG> bnw;
    Int<LL> fw[DEG + 1], gw[2], kw, sw; /* best_norm2's wide copies (dead before it recurses) */
    DiscScratch<LL, DEG> disc;
    int64_t list_k[SOPT_MAX_K];
    int64_t list_k_opt[SOPT_MAX_K];
};

/* fr = f + k * x^t * g (deg g = 1) */
template <int L, int DEG>
S23_HD bool rotate(Int<L> *fr, const Int<L> *f, const Int<L> *g, const Int<L> &k, int t)
{
    Int<L> x;
    for (int i = 0; i <= DEG; i++)
        fr[i] = f[i];
    for (int i = 0; i <= 1; i++)
        if (!mul(x, g[i], k) || !add(fr[i + t], fr[i + t], x))
            return false;
    return true;
}

template <int L, int D>
S23_HD bool translated(Int<L> *fr, const Int<L> *f, const Int<L> &k)
{
    for (int i = 0; i <= D; i++)
        fr[i] = f[i];
    return translate<L, D>(fr, k);
}

template <int L, int DEG>
S23_HD void copy_poly(Int<L> *dst, const Int<L> *src)
{
    for (int i = 0; i <= DEG; i++)
        dst[i] = src[i];
}

/* sopt_local_descent with use_translation = 1, deg_rotation = d - 2 */
template <int L, int DEG>
S23_HD double local_descent(Int<L> *f_opt, Int<L> *g_opt, const Int<L> *f_raw, const Int<L> *g_raw,
                            unsigned max_iter, bool &ok)
{
    const double GUARD = 0.001;
    const int DR = DEG - 2;
    Int<L> k[DEG], kt, tmp, ftmp[DEG + 1], gtmp[2];
    int changed[DEG];
    double logmu_opt = L2_skew_lognorm(f_raw, DEG), logmu;

    for (int i = 0; i <= DR; i++) {
        set_si(k[i], 1);
        for (int guard = 0;; guard++) {
            if (guard > 200 || !rotate<L, DEG>(ftmp, f_raw, g_raw, k[i], i)) {
                ok = false;
                return 0;
            }
            logmu = L2_skew_lognorm(ftmp, DEG);
            if (logmu > logmu_opt + GUARD) {
                neg(tmp, k[i]);
                if (!rotate<L, DEG>(ftmp, f_raw, g_raw, tmp, i)) {
                    ok = false;
                    return 0;
                }
                logmu = L2_skew_lognorm(ftmp, DEG);
                if (logmu > logmu_opt + GUARD)
                    break;
            }
            if (!mul2(k[i], k[i])) {
                ok = false;
                return 0;
            }
        }
    }
    set_si(kt, 1);
    for (int guard = 0;; guard++) {
        if (guard > 200 || !translated<L, DEG>(ftmp, f_raw, kt)) {
            ok = false;
            return 0;
        }
        logmu = L2_skew_lognorm(ftmp, DEG);
        if (logmu > logmu_opt + GUARD) {
            neg(tmp, kt);
            if (!translated<L, DEG>(ftmp, f_raw, tmp)) {
                ok = false;
                return 0;
            }
            logmu = L2_skew_lognorm(ftmp, DEG);
            if (logmu > logmu_opt + GUARD)
                break;
        }
        if (!mul2(kt, kt)) {
            ok = false;
            return 0;
        }
    }

    copy_poly<L, DEG>(f_opt, f_raw);
    g_opt[0] = g_raw[0];
    g_opt[1] = g_raw[1];

    unsigned iter = 0;
    while (iter < max_iter) {
        iter++;
        int changedt = 0;
        for (int i = 0; i <= DR; i++)
            changed[i] = 0;

        if (!translated<L, DEG>(ftmp, f_opt, kt)) {
            ok = false;
            return 0;
        }
        logmu = L2_skew_lognorm(ftmp, DEG);
        if (logmu < logmu_opt) {
            changedt = 1;
            logmu_opt = logmu;
            copy_poly<L, DEG>(f_opt, ftmp);
            if (!translated<L, 1>(gtmp, g_opt, kt)) {
                ok = false;
                return 0;
            }
            g_opt[0] = gtmp[0];
            g_opt[1] = gtmp[1];
        } else {
            neg(tmp, kt);
            if (!translated<L, DEG>(ftmp, f_opt, tmp)) {
                ok = false;
                return 0;
            }
            logmu = L2_skew_lognorm(ftmp, DEG);
            if (logmu < logmu_opt) {
                changedt = 1;
                logmu_opt = logmu;
                copy_poly<L, DEG>(f_opt, ftmp);
                if (!translated<L, 1>(gtmp, g_opt, tmp)) {
                    ok = false;
                    return 0;
                }
                g_opt[0] = gtmp[0];
                g_opt[1] = gtmp[1];
            }
        }

        for (int i = 0; i <= DR; i++) {
            if (!rotate<L, DEG>(ftmp, f_opt, g_opt, k[i], i)) {
                ok = false;
                return 0;
            }
            logmu = L2_skew_lognorm(ftmp, DEG);
            if (logmu < logmu_opt) {
                changed[i] = 1;
                logmu_opt = logmu;
                copy_poly<L, DEG>(f_opt, ftmp);
            } else {
                neg(tmp, k[i]);
                if (!rotate<L, DEG>(ftmp, f_opt, g_opt, tmp, i)) {
                    ok = false;
                    return 0;
                }
                logmu = L2_skew_lognorm(ftmp, DEG);
                if (logmu < logmu_opt) {
                    changed[i] = 1;
                    logmu_opt = logmu;
                    copy_poly<L, DEG>(f_opt, ftmp);
                }
            }
        }

        bool is_finished = (changedt == 0 && is_one(kt));
        for (int i = 0; i <= DR && is_finished; i++)
            is_finished = (changed[i] == 0 && is_one(k[i]));
        if (is_finished)
            break;

        Int<L> one;
        set_si(one, 1);
        if (changedt == 1) {
            if (!mul2(kt, kt)) {
                ok = false;
                return 0;
            }
        } else if (cmp(kt, one) > 0) {
            fdiv2(kt, kt);
        }
        for (int i = 0; i <= DR; i++) {
            if (changed[i] == 1) {
                if (!mul2(k[i], k[i])) {
                    ok = false;
                    return 0;
                }
            } else if (cmp(k[i], one) > 0) {
                fdiv2(k[i], k[i]);
            }
        }
    }
    return logmu_opt;
}

/* expected_growth: [kmin, kmax] of k with lognorm(f0 + k x^i g, skew s) <= maxlognorm */
template <int L, int DEG>
S23_HD void expected_growth(double &rkmin, double &rkmax, const Int<L> *f0, const Int<L> *g, int i,
                            double maxlognorm, double s, bool &ok)
{
    Int<L> kmin, kmax, k, f[DEG + 1];
    double n2;
    set_si(kmin, -1);
    for (int guard = 0;; guard++) {
        if (guard > 400 || !rotate<L, DEG>(f, f0, g, kmin, i)) {
            ok = false;
            return;
        }
        n2 = L2_lognorm(f, DEG, s);
        if (n2 > maxlognorm)
            break;
        if (!mul2(kmin, kmin)) {
            ok = false;
            return;
        }
    }
    tdiv2(kmax, kmin);
    for (;;) {
        if (!add(k, kmin, kmax)) {
            ok = false;
            return;
        }
        fdiv2(k, k);
        if (cmp(k, kmin) == 0 || cmp(k, kmax) == 0)
            break;
        if (!rotate<L, DEG>(f, f0, g, k, i)) {
            ok = false;
            return;
        }
        n2 = L2_lognorm(f, DEG, s);
        if (n2 > maxlognorm)
            kmin = k;
        else
            kmax = k;
    }
    rkmin = get_d(kmax);
    set_si(kmax, 1);
    for (int guard = 0;; guard++) {
        if (guard > 400 || !rotate<L, DEG>(f, f0, g, kmax, i)) {
            ok = false;
            return;
        }
        n2 = L2_lognorm(f, DEG, s);
        if (n2 > maxlognorm)
            break;
        if (!mul2(kmax, kmax)) {
            ok = false;
            return;
        }
    }
    tdiv2(kmin, kmax);
    for (;;) {
        if (!add(k, kmin, kmax)) {
            ok = false;
            return;
        }
        fdiv2(k, k);
        if (cmp(k, kmin) == 0 || cmp(k, kmax) == 0)
            break;
        if (!rotate<L, DEG>(f, f0, g, k, i)) {
            ok = false;
            return;
        }
        n2 = L2_lognorm(f, DEG, s);
        if (n2 > maxlognorm)
            kmax = k;
        else
            kmin = k;
    }
    rkmax = get_d(kmin);
}

template <int L, int LL, int DEG>
S23_HD double expected_rotation_gain(const Int<L> *f, const Int<L> *g, DiscScratch<LL, DEG> &D, bool &ok)
{
    const double NORM_MARGIN = 0.2;
    double S = 1.0, s, incr = 0.0, rkmin = 0, rkmax = 0;
    double proj_alpha = alpha_projective_100<L, LL, DEG>(f, D, ok);
    double skew = L2_skewness(f, DEG);
    double n = L2_lognorm(f, DEG, skew);
    for (int i = 0; 2 * i < DEG; i++) {
        expected_growth<L, DEG>(rkmin, rkmax, f, g, i, n + NORM_MARGIN, skew, ok);
        s = rkmax - rkmin + 1.0;
        S *= s;
        if (s >= 2.0)
            incr += NORM_MARGIN / 2.0;
    }
    return proj_alpha + expected_alpha(log(S)) + incr;
}

S23_HD unsigned compute_rational_approximation(double *Q, double q, unsigned nb_approx, double bound)
{
    unsigned n = 0;
    double best_e = 2.0;
    for (double den = 1.0; n < nb_approx && den <= bound; den += 1.0) {
        double num = floor(den * q + 0.5);
        double e = fabs(q - num / den);
        if (e >= best_e)
            continue;
        best_e = e;
        Q[n++] = num / den;
    }
    return n;
}

S23_HD uint64_t gcd_u64(uint64_t a, uint64_t b)
{
    while (b) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* mpz_set_d(x, e > 0 ? e + 0.5 : e - 0.5) into an int64 (list_mpz_append_from_rounded_double) */
S23_HD bool round_k(int64_t &k, double e)
{
    double r = e > 0 ? e + 0.5 : e - 0.5;
    if (!(r > -4.611686018427388e18 && r < 4.611686018427388e18))
        return false;
    k = (int64_t)r;
    return true;
}

/* sopt_find_translations_deg5: append candidate translations to list_k */
template <int L>
S23_HD bool translations_deg5(int64_t *list_k, int &len, const Int<L> *f, const Int<L> *g, int sopt_effort)
{
    double a5 = get_d(f[5]), a4 = get_d(f[4]), a3 = get_d(f[3]), a2 = get_d(f[2]);
    double g1 = get_d(g[1]), g0 = get_d(g[0]);
    DPoly res;
    double roots_q2[DP_MAX];
    res.c[2] = (5.0 * a5 * g0 - a4 * g1) * (5.0 * a5 * g0 - a4 * g1);
    res.c[1] = 8.0 * a4 * a4 * a4 * g0 - 30.0 * a3 * a4 * a5 * g0 + 50.0 * a2 * a5 * a5 * g0 -
               2.0 * a3 * a4 * a4 * g1 + 10.0 * a3 * a3 * a5 * g1 - 10.0 * a2 * a4 * a5 * g1;
    res.c[0] = -3.0 * a3 * a3 * a4 * a4 + 8.0 * a2 * a4 * a4 * a4 + 10.0 * a3 * a3 * a3 * a5 -
               30.0 * a2 * a3 * a4 * a5 + 25.0 * a2 * a2 * a5 * a5;
    res.deg = 2;
    dp_cleandeg(res, 2);
    unsigned nb_q2roots = res.deg >= 0 ? dp_compute_all_roots_with_bound(roots_q2, res, 1e10) : 0;

    DPoly C;
    int a = 1, b = 1;
    for (unsigned i = 0; i < nb_q2roots + (unsigned)sopt_effort; i++) {
        unsigned t;
        double q2_rat_approx[16];
        if (i < nb_q2roots)
            t = compute_rational_approximation(q2_rat_approx, roots_q2[i], 16, 100.0);
        else
            t = 16;
        for (unsigned j = 0; j < t; j++) {
            double q2_rat = q2_rat_approx[j];
            double double_roots_k[DP_MAX];
            if (i >= nb_q2roots) {
                q2_rat = (double)a / (double)b;
                if (a > 0)
                    a = -a;
                else {
                    a = (-a) + 1;
                    while (gcd_u64((uint64_t)a, (uint64_t)b) != 1)
                        a++;
                    if (a > b) {
                        a = 1;
                        b = b + 1;
                    }
                }
            }
            C.c[2] = 10.0 * a5;
            C.c[1] = 4.0 * a4;
            C.c[0] = g1 * q2_rat + a3;
            C.deg = 2;
            dp_cleandeg(C, 2);
            unsigned nb_k_roots = dp_compute_all_roots(double_roots_k, C);
            for (unsigned l = 0; l < nb_k_roots; l++) {
                if (len >= SOPT_MAX_K || !round_k(list_k[len], double_roots_k[l]))
                    return false;
                len++;
            }
        }
    }
    return true;
}

/* best_norm on (f_raw, g_raw) translated by k: the reduced row into ft, gt; false on failure */
template <int LP, int LL, int DEG>
S23_HD bool best_norm_lp(double &norm, Int<LP> *ft, Int<LP> *gt, const Int<LP> *f_raw, const Int<LP> *g_raw,
                         const Int<LP> &skew, const Int<LP> &k, SoptScratch<LP, LL, DEG> &S)
{
    bool ok = true;
    for (int i = 0; i <= DEG; i++)
        ok = ok && resize(S.fw[i], f_raw[i]);
    ok = ok && resize(S.gw[0], g_raw[0]) && resize(S.gw[1], g_raw[1]) && resize(S.kw, k) && resize(S.sw, skew);
    if (!ok)
        return false;
    sopt_best_norm<LL, DEG>(S.res, S.lll, S.bnw, S.fw, S.gw, S.kw, S.sw);
    if (S.res.status != LLL_OK)
        return false;
    norm = S.res.norm;
    for (int i = 0; i <= DEG; i++)
        ok = ok && resize(ft[i], S.res.f[i]);
    return ok && resize(gt[0], S.res.g[0]) && resize(gt[1], S.res.g[1]);
}

/* CADO's best_norm2, which recurses on the roots of a quadratic in k, run as a loop
 * over an explicit frame stack (S.fr) with the same control flow:
 *
 *   norm = best_norm(k); if norm >= min_norm: return min_norm (k unchanged)
 *   min_norm = norm; best_k = k; out = this result
 *   for each root of (d(d-1)/2 f_d) k^2 + (d-1) f_{d-1} k + f_{d-2}:
 *       k = round(root); norm = best_norm2(k, min_norm)   (the child writes here on success)
 *       if norm < min_norm: min_norm = norm; best_k = k; out = this frame's result
 *   k = best_k; return min_norm
 *
 * k is in/out. Returns the best norm. */
template <int LP, int LL, int DEG>
S23_HD double best_norm2(Int<LP> *fopt, Int<LP> *gopt, const Int<LP> *f_raw, const Int<LP> *g_raw,
                         const Int<LP> &skew, Int<LP> &k, double min_norm, SoptScratch<LP, LL, DEG> &S, bool &ok)
{
    int top = 0;
    S.fr[0].k = k;
    S.fr[0].min_norm = min_norm;
    bool entering = true;
    double ret_norm = 0;
    for (;;) {
        Bn2Frame<LP, DEG> &F = S.fr[top];
        Int<LP> *out_f = top == 0 ? fopt : S.fr[top - 1].ft;
        Int<LP> *out_g = top == 0 ? gopt : S.fr[top - 1].gt;
        bool returning = false;
        if (entering) {
            double norm;
            if (!best_norm_lp<LP, LL, DEG>(norm, F.ft, F.gt, f_raw, g_raw, skew, F.k, S)) {
                ok = false;
                return min_norm;
            }
            if (norm >= F.min_norm) {
                ret_norm = F.min_norm; /* no improvement; k unchanged */
                returning = true;
            } else {
                F.min_norm = norm;
                F.best_k = F.k;
                copy_poly<LP, DEG>(out_f, F.ft);
                out_g[0] = F.gt[0];
                out_g[1] = F.gt[1];
                DPoly eq;
                eq.c[2] = (double)(DEG * (DEG - 1)) / 2.0 * get_d(F.ft[DEG]);
                eq.c[1] = (double)(DEG - 1) * get_d(F.ft[DEG - 1]);
                eq.c[0] = get_d(F.ft[DEG - 2]);
                eq.deg = 2;
                dp_cleandeg(eq, 2);
                F.nroots = eq.deg >= 0 ? dp_compute_all_roots(F.roots, eq) : 0;
                F.l = 0;
            }
        } else {
            /* back from the child at top + 1, which returned ret_norm and left its k */
            F.k = S.fr[top + 1].k;
            if (ret_norm < F.min_norm) {
                F.min_norm = ret_norm;
                F.best_k = F.k;
                copy_poly<LP, DEG>(out_f, F.ft);
                out_g[0] = F.gt[0];
                out_g[1] = F.gt[1];
            }
            F.l++;
        }
        if (!returning) {
            if (F.l < F.nroots) {
                if (top + 1 > SOPT_MAX_RECURSION) {
                    ok = false;
                    return min_norm;
                }
                double r = F.roots[F.l];
                if (!set_d_trunc(F.k, r >= 0 ? r + 0.5 : r - 0.5)) {
                    ok = false;
                    return min_norm;
                }
                Bn2Frame<LP, DEG> &C = S.fr[top + 1];
                C.k = F.k;
                C.min_norm = F.min_norm;
                top++;
                entering = true;
                continue;
            }
            F.k = F.best_k;
            ret_norm = F.min_norm;
        }
        if (top == 0) {
            k = F.k;
            return ret_norm;
        }
        top--;
        entering = false;
    }
}

template <int L> S23_HD int64_t get_si(const Int<L> &a)
{
    return (int64_t)(((uint64_t)a.w[1] << 32) | a.w[0]);
}

/* size_optimization (max_rot = d - 2). Returns SOPT_OK and the best pair and its
 * exp_E (lognorm + expected_rotation_gain), SOPT_FAIL (overflow, a dependent LLL basis,
 * recursion or loop caps, non-finite values, effort above SOPT_MAX_EFFORT), or
 * SOPT_TIMEOUT if now_ns() passes deadline (0: none) before a translation candidate:
 * use the CPU path then. */
template <int LP, int LL, int DEG>
S23_HD int size_optimize(Int<LP> *f_opt, Int<LP> *g_opt, double &best_lognorm, const Int<LP> *f_raw,
                         const Int<LP> *g_raw, const Int<LP> &skew, int sopt_effort, uint64_t deadline,
                         SoptScratch<LP, LL, DEG> &S)
{
    if (sopt_effort < 0 || sopt_effort > SOPT_MAX_EFFORT)
        return SOPT_FAIL;
    bool ok = true;
    best_lognorm = L2_skew_lognorm(f_raw, DEG);
    best_lognorm += expected_rotation_gain<LP, LL, DEG>(f_raw, g_raw, S.disc, ok);
    if (!ok)
        return SOPT_FAIL;

    int len = 0;
    if (!translations_deg5<LP>(S.list_k, len, f_raw, g_raw, sopt_effort) || len >= SOPT_MAX_K)
        return SOPT_FAIL;
    S.list_k[len++] = 0;
    for (int i = 1; i < len; i++) { /* sort ascending */
        int64_t x = S.list_k[i];
        int j = i - 1;
        while (j >= 0 && S.list_k[j] > x) {
            S.list_k[j + 1] = S.list_k[j];
            j--;
        }
        S.list_k[j + 1] = x;
    }
    int n = 1;
    for (int i = 1; i < len; i++)
        if (S.list_k[i] != S.list_k[n - 1])
            S.list_k[n++] = S.list_k[i];
    len = n;

    copy_poly<LP, DEG>(f_opt, f_raw);
    g_opt[0] = g_raw[0];
    g_opt[1] = g_raw[1];
    int nopt = 0;
    Int<LP> ft[DEG + 1], gt[2], fld[DEG + 1], gld[2], ki;
    for (int i = 0; i < len; i++) {
        if (deadline && now_ns() > deadline)
            return SOPT_TIMEOUT;
        set_si(ki, S.list_k[i]);
        best_norm2<LP, LL, DEG>(ft, gt, f_raw, g_raw, skew, ki, DBL_MAX, S, ok);
        if (!ok)
            return SOPT_FAIL;
        int64_t kv = get_si(ki);
        bool is_new = true;
        for (int j = 0; j < nopt; j++)
            if (S.list_k_opt[j] == kv) {
                is_new = false;
                break;
            }
        if (!is_new)
            continue;
        S.list_k_opt[nopt++] = kv;
        double lognorm = local_descent<LP, DEG>(fld, gld, ft, gt, 300, ok);
        if (!ok)
            return SOPT_FAIL;
        lognorm += expected_rotation_gain<LP, LL, DEG>(fld, gld, S.disc, ok);
        if (!ok)
            return SOPT_FAIL;
        if (lognorm < best_lognorm) {
            best_lognorm = lognorm;
            copy_poly<LP, DEG>(f_opt, fld);
            g_opt[0] = gld[0];
            g_opt[1] = gld[1];
        }
    }
    return SOPT_OK;
}

} // namespace s23
