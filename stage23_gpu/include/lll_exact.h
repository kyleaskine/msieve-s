/* Exact-integer LLL for small fixed-size lattices, host and device.
 *
 * A port of CADO-NFS utils/lll.c (Paul Zimmermann's GMP translation of NTL's LLL_ZZ,
 * integral LLL after de Weger / Cohen 2.6.7) for the case CADO's size optimization
 * uses: M linearly independent row vectors of length N, delta = a/b = 1/1. Every step
 * (incremental Gram-Schmidt, size reduction with BalDiv rounding, the swap test and the
 * swap update) is the same arithmetic in the same order, so the reduced basis is the
 * same as CADO's, row for row.
 *
 * NTL's handling of linearly dependent vectors (P[k] = 0 and the gcdext swap branch) is
 * not ported: the sopt lattice (f, g, x*g, ..., x^(d-2)*g) is always independent. If a
 * dependent vector shows up anyway, lll_exact returns LLL_DEPENDENT so the caller can
 * fall back to the CPU.
 *
 * Indices are 1-based like the original: B[1..M][1..N], D[0..M], lam[1..M][1..M].
 */
#pragma once

#include "mpint.h"

namespace s23 {

enum { LLL_OK = 0, LLL_OVERFLOW = 1, LLL_DEPENDENT = 2 };

template <int L, int M, int N>
struct LLLState {
    Int<L> B[M + 1][N + 1];
    Int<L> D[M + 1];
    Int<L> lam[M + 1][M + 1];
};

/* u = <a, b> over columns 1..N */
template <int L, int N>
S23_HD bool inner_product(Int<L> &u, const Int<L> *a, const Int<L> *b)
{
    Int<L> t;
    if (!mul(u, a[1], b[1]))
        return false;
    for (int i = 2; i <= N; i++)
        if (!mul(t, a[i], b[i]) || !add(u, u, t))
            return false;
    return true;
}

/* IncrementalGS for an independent row k (P[j] = j for all j). */
template <int L, int M, int N>
S23_HD int incremental_gs(LLLState<L, M, N> &S, int k)
{
    Int<L> u, t1, t2, r;
    for (int j = 1; j <= k - 1; j++) {
        if (!inner_product<L, N>(u, S.B[k], S.B[j]))
            return LLL_OVERFLOW;
        for (int i = 1; i <= j - 1; i++) {
            /* u = (D[i]*u - lam[k][i]*lam[j][i]) / D[i-1]   (mpz_div = floor) */
            if (!mul(t1, S.D[i], u) || !mul(t2, S.lam[k][i], S.lam[j][i]) || !sub(t1, t1, t2) ||
                !divmod_floor(u, r, t1, S.D[i - 1]))
                return LLL_OVERFLOW;
        }
        S.lam[k][j] = u;
    }
    if (!inner_product<L, N>(u, S.B[k], S.B[k]))
        return LLL_OVERFLOW;
    for (int i = 1; i <= k - 1; i++) {
        if (!mul(t1, S.D[i], u) || !mul(t2, S.lam[k][i], S.lam[k][i]) || !sub(t1, t1, t2) ||
            !divmod_floor(u, r, t1, S.D[i - 1]))
            return LLL_OVERFLOW;
    }
    if (is_zero(u))
        return LLL_DEPENDENT;
    S.D[k] = u;
    return LLL_OK;
}

/* Round a/d to the nearest integer, ties toward zero (CADO's BalDiv). d > 0. */
template <int L> S23_HD bool bal_div(Int<L> &q, const Int<L> &a, const Int<L> &d)
{
    Int<L> r, r2;
    if (!divmod_floor(q, r, a, d) || !add(r2, r, r))
        return false;
    int c = cmp(r2, d);
    if (c > 0 || (c == 0 && is_neg(q))) {
        Int<L> one;
        set_si(one, 1);
        if (!add(q, q, one))
            return false;
    }
    return true;
}

/* reduce(k, l): size-reduce row k against row l. */
template <int L, int M, int N>
S23_HD int reduce(LLLState<L, M, N> &S, int k, int l)
{
    Int<L> t1, r, t;
    if (!add(t1, S.lam[k][l], S.lam[k][l]))
        return LLL_OVERFLOW;
    if (is_neg(t1) && !neg(t1, t1))
        return LLL_OVERFLOW;
    if (cmp(t1, S.D[l]) <= 0)
        return LLL_OK;
    if (!bal_div(r, S.lam[k][l], S.D[l]))
        return LLL_OVERFLOW;
    for (int i = 1; i <= N; i++) /* B[k] -= r * B[l] */
        if (!mul(t, r, S.B[l][i]) || !sub(S.B[k][i], S.B[k][i], t))
            return LLL_OVERFLOW;
    for (int j = 1; j <= l - 1; j++)
        if (!mul(t, S.lam[l][j], r) || !sub(S.lam[k][j], S.lam[k][j], t))
            return LLL_OVERFLOW;
    if (!mul(t, S.D[l], r) || !sub(S.lam[k][l], S.lam[k][l], t))
        return LLL_OVERFLOW;
    return LLL_OK;
}

/* Swap test with delta = 1: D[k-1]^2 > D[k]*D[k-2] + lam[k][k-1]^2. */
template <int L, int M, int N>
S23_HD int swap_test(const LLLState<L, M, N> &S, int k, bool &do_swap)
{
    Int<L> t1, t2;
    if (!mul(t1, S.D[k], S.D[k - 2]) || !mul(t2, S.lam[k][k - 1], S.lam[k][k - 1]) || !add(t1, t1, t2) ||
        !mul(t2, S.D[k - 1], S.D[k - 1]))
        return LLL_OVERFLOW;
    do_swap = cmp(t2, t1) > 0;
    return LLL_OK;
}

/* swapLLL(k, m) for independent vectors: swap rows k-1 and k and update lam, D. */
template <int L, int M, int N>
S23_HD int swap_rows(LLLState<L, M, N> &S, int k, int m)
{
    Int<L> t1, t2, t3, x;
    for (int i = 1; i <= N; i++) {
        x = S.B[k - 1][i];
        S.B[k - 1][i] = S.B[k][i];
        S.B[k][i] = x;
    }
    for (int j = 1; j <= k - 2; j++) {
        x = S.lam[k - 1][j];
        S.lam[k - 1][j] = S.lam[k][j];
        S.lam[k][j] = x;
    }
    const Int<L> &lk = S.lam[k][k - 1];
    for (int i = k + 1; i <= m; i++) {
        /* t1 = (lam[k][k-1]*lam[i][k-1] + D[k-2]*lam[i][k]) / D[k-1] */
        if (!mul(t2, lk, S.lam[i][k - 1]) || !mul(t3, S.D[k - 2], S.lam[i][k]) || !add(t2, t2, t3) ||
            !div_exact(t1, t2, S.D[k - 1]))
            return LLL_OVERFLOW;
        /* lam[i][k] = (D[k]*lam[i][k-1] - lam[k][k-1]*lam[i][k]) / D[k-1] */
        if (!mul(t2, S.D[k], S.lam[i][k - 1]) || !mul(t3, lk, S.lam[i][k]) || !sub(t2, t2, t3) ||
            !div_exact(S.lam[i][k], t2, S.D[k - 1]))
            return LLL_OVERFLOW;
        S.lam[i][k - 1] = t1;
    }
    /* D[k-1] = (D[k-2]*D[k] + lam[k][k-1]^2) / D[k-1] */
    if (!mul(t2, S.D[k - 2], S.D[k]) || !mul(t3, lk, lk) || !add(t2, t2, t3) ||
        !div_exact(t1, t2, S.D[k - 1]))
        return LLL_OVERFLOW;
    S.D[k - 1] = t1;
    return LLL_OK;
}

/* LLL-reduce the rows of S.B in place (delta = 1), as CADO's LLL(det, B, NULL, 1, 1). */
template <int L, int M, int N>
S23_HD int lll_exact(LLLState<L, M, N> &S)
{
    for (int j = 0; j <= M; j++) {
        set_si(S.D[j], j == 0 ? 1 : 0);
        for (int k = 0; k <= M; k++)
            set_zero(S.lam[j][k]);
    }
    int k = 1, max_k = 0, st;
    while (k <= M) {
        if (k > max_k) {
            if ((st = incremental_gs(S, k)) != LLL_OK)
                return st;
            max_k = k;
        }
        if (k == 1) {
            k++;
            continue;
        }
        if ((st = reduce(S, k, k - 1)) != LLL_OK)
            return st;
        bool do_swap;
        if ((st = swap_test(S, k, do_swap)) != LLL_OK)
            return st;
        if (do_swap) {
            if ((st = swap_rows(S, k, max_k)) != LLL_OK)
                return st;
            k--;
        } else {
            for (int j = k - 2; j >= 1; j--)
                if ((st = reduce(S, k, j)) != LLL_OK)
                    return st;
            k++;
        }
    }
    return LLL_OK;
}

} // namespace s23
