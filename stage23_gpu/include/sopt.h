/* CADO-NFS size optimization (polyselect/size_optimization.c, size_optimization_aux
 * with max_rot = d - 2), ported to fixed-width integers and doubles for host and
 * device, degree 5, as five phases over a batch of polynomials:
 *
 *   prepare  (per poly)      the raw pair's objective (lognorm + expected_rotation_gain),
 *                            and its translation candidates k: closed-form quadratics,
 *                            rational approximations of the q2 roots, Farey fractions
 *                            per unit of effort, and k = 0, sorted and deduplicated
 *   lll      (per candidate) best_norm2: exact LLL via best_norm, recursing on the roots
 *                            of a quadratic in k; gives a pair and the final k
 *   dedupe   (per poly)      in list order, a candidate whose final k an earlier one
 *                            already produced is dropped (CADO skips it)
 *   descent  (per remaining  local descent on translation and rotations (up to 300
 *             candidate)     steps), then lognorm + expected_rotation_gain
 *   reduce   (per poly)      in list order, the first strictly lowest objective, starting
 *                            from the raw pair's (that objective is the exp_E sopt minimizes)
 *
 * This is CADO's sequential loop with its work split up: the result, ties included, is
 * the same. The candidates of one polynomial are independent until dedupe and reduce,
 * which keep CADO's order, so the drivers (tools/s23_sopt.cu) run each phase over all
 * items of a batch in any order, on CPU threads or GPU warps.
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

/* SOPT_FAIL: overflow, a dependent LLL basis, recursion or loop caps, non-finite values:
 * the caller redoes the polynomial with CADO */
enum { SOPT_OK = 0, SOPT_FAIL = 1 };
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

/* nanoseconds, for time slices: the global timer on the device, steady_clock on the host */
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
};

/* the candidates a polynomial can have at this effort (translations_deg5's bound) */
S23_HD int sopt_max_cand(int effort)
{
    return 2 * 16 * (2 + effort) + 1;
}

/* one polynomial of a batch */
template <int LP, int DEG>
struct SoptPoly {
    Int<LP> f[DEG + 1], g[2], skew; /* input: the raw pair, and sopt_get_skewness */
    int status;                     /* input SOPT_OK (SOPT_FAIL: skip it); output */
    int ncand;                      /* prepare: candidates, in its first ncand slots */
    double obj;                     /* prepare: the raw pair's objective; reduce: the result's */
    Int<LP> f_opt[DEG + 1], g_opt[2]; /* reduce: the result */
};

/* CAND_NEW: set by prepare, so a slot whose lll never ran (slots are reused across
 * batches) cannot pass for a finished candidate */
enum { CAND_FAIL = 0, CAND_LLL = 1, CAND_DUP = 2, CAND_DONE = 3, CAND_NEW = 4 };

/* one translation candidate of a polynomial */
template <int LP, int DEG>
struct SoptCand {
    Int<LP> f[DEG + 1], g[2]; /* lll: best_norm2's pair; descent: the local optimum */
    Int<LP> kv;               /* lll: k after best_norm2 */
    double obj;               /* descent: lognorm + expected_rotation_gain */
    int64_t k;                /* prepare: the translation */
    int state;                /* CAND_* after lll, dedupe, descent */
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
S23_HD bool translations_deg5(int64_t *list_k, int &len, int cap, const Int<L> *f, const Int<L> *g, int sopt_effort)
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
                if (len >= cap || !round_k(list_k[len], double_roots_k[l]))
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

/* Phase 1, per polynomial: the raw pair's objective and its translation candidates,
 * written to C[0 .. ncand-1].k (C has sopt_max_cand(effort) slots). */
template <int LP, int LL, int DEG>
S23_HD void sopt_prepare(SoptPoly<LP, DEG> &P, SoptCand<LP, DEG> *C, int effort, SoptScratch<LP, LL, DEG> &S)
{
    P.ncand = 0;
    if (P.status != SOPT_OK)
        return;
    bool ok = effort >= 0 && effort <= SOPT_MAX_EFFORT;
    if (ok)
        P.obj = L2_skew_lognorm(P.f, DEG) + expected_rotation_gain<LP, LL, DEG>(P.f, P.g, S.disc, ok);
    int len = 0;
    const int cap = sopt_max_cand(effort);
    if (!ok || !translations_deg5<LP>(S.list_k, len, cap - 1, P.f, P.g, effort)) {
        P.status = SOPT_FAIL;
        return;
    }
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
    for (int i = 0; i < n; i++) {
        C[i].k = S.list_k[i];
        C[i].state = CAND_NEW;
    }
    P.ncand = n;
}

/* Phase 2, per candidate: best_norm2 from its k */
template <int LP, int LL, int DEG>
S23_HD void sopt_lll(const SoptPoly<LP, DEG> &P, SoptCand<LP, DEG> &c, SoptScratch<LP, LL, DEG> &S)
{
    bool ok = true;
    set_si(c.kv, c.k);
    best_norm2<LP, LL, DEG>(c.f, c.g, P.f, P.g, P.skew, c.kv, DBL_MAX, S, ok);
    c.state = ok ? CAND_LLL : CAND_FAIL;
}

/* Phase 3, per polynomial: in list order, a candidate whose final k an earlier kept one
 * already has becomes CAND_DUP (CADO compares the whole k, mpz_cmp). Any failed
 * candidate fails the polynomial, as it would stop CADO's loop. */
template <int LP, int DEG>
S23_HD void sopt_dedupe(SoptPoly<LP, DEG> &P, SoptCand<LP, DEG> *C)
{
    if (P.status != SOPT_OK)
        return;
    for (int i = 0; i < P.ncand; i++) {
        if (C[i].state == CAND_FAIL || C[i].state == CAND_NEW) { /* NEW: the driver missed its lll */
            P.status = SOPT_FAIL;
            return;
        }
        for (int j = 0; j < i; j++)
            if (C[j].state == CAND_LLL && C[j].kv.w[0] == C[i].kv.w[0] && cmp(C[j].kv, C[i].kv) == 0) {
                C[i].state = CAND_DUP;
                break;
            }
    }
}

/* Phase 4, per kept candidate: local descent, then the objective */
template <int LP, int LL, int DEG>
S23_HD void sopt_descent(SoptCand<LP, DEG> &c, SoptScratch<LP, LL, DEG> &S)
{
    if (c.state != CAND_LLL)
        return;
    bool ok = true;
    Int<LP> fld[DEG + 1], gld[2];
    double lognorm = local_descent<LP, DEG>(fld, gld, c.f, c.g, 300, ok);
    if (ok)
        lognorm += expected_rotation_gain<LP, LL, DEG>(fld, gld, S.disc, ok);
    copy_poly<LP, DEG>(c.f, fld);
    c.g[0] = gld[0];
    c.g[1] = gld[1];
    c.obj = lognorm;
    c.state = ok ? CAND_DONE : CAND_FAIL;
}

/* Phase 5, per polynomial: the first strictly lowest objective, in list order, starting
 * from the raw pair's (CADO: if (lognorm < best_lognorm)) */
template <int LP, int DEG>
S23_HD void sopt_reduce(SoptPoly<LP, DEG> &P, const SoptCand<LP, DEG> *C)
{
    if (P.status != SOPT_OK)
        return;
    int best = -1;
    for (int i = 0; i < P.ncand; i++) {
        /* a kept candidate that never reached the descent (CAND_LLL) means the driver
         * missed an item: fail, so the poly goes to CADO instead of a silent wrong answer */
        if (C[i].state == CAND_FAIL || C[i].state == CAND_LLL || C[i].state == CAND_NEW) {
            P.status = SOPT_FAIL;
            return;
        }
        if (C[i].state == CAND_DONE && C[i].obj < P.obj) {
            P.obj = C[i].obj;
            best = i;
        }
    }
    const Int<LP> *f = best < 0 ? P.f : C[best].f, *g = best < 0 ? P.g : C[best].g;
    copy_poly<LP, DEG>(P.f_opt, f);
    P.g_opt[0] = g[0];
    P.g_opt[1] = g[1];
}

} // namespace s23
