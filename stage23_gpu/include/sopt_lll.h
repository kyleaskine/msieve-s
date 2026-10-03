/* CADO's best_norm (polyselect/size_optimization.c) on fixed-width integers, host and
 * device: translate (f, g) by k, build the skew-scaled rotation lattice, LLL-reduce it
 * exactly, and return the shortest reduced row that still has a degree-d term, which is
 * a*f(x+k) + r(x)*g(x+k) for some multiplier a and rotation r.
 *
 * The lattice has M = d rows (f, then x^i*g for i = 0..d-2: rotation up to x^(d-2)*g,
 * CADO's max_rot = d-2) of N = d+1 columns; column j holds the coefficient of x^j times
 * skew^j. The skew is an integer computed once per raw polynomial on the host exactly as
 * CADO's sopt_get_skewness does (pow on doubles, then truncation), so the device never
 * depends on its own pow() agreeing with glibc's in the last bit.
 */
#pragma once

#include <float.h>
#include "lll_exact.h"

namespace s23 {

/* c[0..d] <- coefficients of c(x + k) (Taylor shift by repeated synthetic division). */
template <int L, int DEG>
S23_HD bool translate(Int<L> *c, const Int<L> &k)
{
    Int<L> t;
    for (int i = 0; i < DEG; i++)
        for (int j = DEG - 1; j >= i; j--)
            if (!mul(t, k, c[j + 1]) || !add(c[j], c[j], t))
                return false;
    return true;
}

/* big temporaries of sopt_best_norm: kept out of the caller's stack frame, since a
   GPU thread's stack is small and best_norm2 recurses */
template <int L, int DEG>
struct BestNormWork {
    Int<L> ft[DEG + 1], gt[2], tmp, sk[DEG + 1], norm2;
};

template <int L, int DEG>
struct SoptLLLResult {
    int status;        /* LLL_OK, LLL_OVERFLOW or LLL_DEPENDENT */
    int row;           /* 1-based row of the reduced basis that was chosen */
    double norm;       /* its squared norm as CADO compares it (truncated to double) */
    Int<L> f[DEG + 1]; /* the chosen polynomial: row / skew^j */
    Int<L> g[2];       /* g(x + k) */
};

/* best_norm: f_raw, g_raw are the raw polynomial; k the translation; skew > 0. */
template <int L, int DEG>
S23_HD void sopt_best_norm(SoptLLLResult<L, DEG> &out, LLLState<L, DEG, DEG + 1> &S, BestNormWork<L, DEG> &W,
                           const Int<L> *f_raw, const Int<L> *g_raw, const Int<L> &k, const Int<L> &skew)
{
    const int M = DEG, N = DEG + 1;
    Int<L> *ft = W.ft, *gt = W.gt, *sk = W.sk, &tmp = W.tmp, &norm2 = W.norm2;
    out.status = LLL_OVERFLOW;
    out.row = 0;
    for (int i = 0; i <= DEG; i++)
        ft[i] = f_raw[i];
    gt[0] = g_raw[0];
    gt[1] = g_raw[1];
    if (!translate<L, DEG>(ft, k) || !translate<L, 1>(gt, k))
        return;

    /* LLL_set_matrix_from_polys */
    for (int r = 1; r <= M; r++)
        for (int c = 1; c <= N; c++)
            set_zero(S.B[r][c]);
    set_si(sk[0], 1);
    for (int j = 1; j <= DEG; j++)
        if (!mul(sk[j], sk[j - 1], skew))
            return;
    for (int j = 0; j < N; j++) {
        if (!mul(S.B[1][j + 1], sk[j], ft[j]))
            return;
    }
    for (int i = 0; i <= M - 2; i++) { /* row i+2 = x^i * g, scaled */
        if (!mul(S.B[i + 2][i + 1], sk[i], gt[0]) || !mul(S.B[i + 2][i + 2], sk[i + 1], gt[1]))
            return;
    }

    int st = lll_exact(S);
    if (st != LLL_OK) {
        out.status = st;
        return;
    }

    /* shortest row with a nonzero degree-d column, compared as CADO does: squared norm
       converted with mpz_get_d semantics, strict <, rows in order */
    double min_norm = DBL_MAX; /* as CADO */
    for (int r = 1; r <= M; r++) {
        if (is_zero(S.B[r][N]))
            continue;
        set_zero(norm2);
        for (int c = 1; c <= N; c++)
            if (!mul(tmp, S.B[r][c], S.B[r][c]) || !add(norm2, norm2, tmp))
                return;
        double nd = get_d(norm2);
        if (nd < min_norm) {
            min_norm = nd;
            out.row = r;
        }
    }
    if (out.row == 0)
        return;
    for (int j = 0; j <= DEG; j++)
        if (!div_exact(out.f[j], S.B[out.row][j + 1], sk[j]))
            return;
    out.g[0] = gt[0];
    out.g[1] = gt[1];
    out.norm = min_norm;
    out.status = LLL_OK;
}

} // namespace s23
