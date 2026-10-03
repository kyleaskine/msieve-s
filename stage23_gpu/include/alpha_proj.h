/* CADO's projective alpha as used by expected_rotation_gain (polyselect_alpha.c:
 * get_alpha_projective with B = ALPHA_BOUND_SMALL = 100), ported exactly, including
 * that it subtracts special_valuation_affine from special_valuation even when the two
 * take different code paths (p exactly dividing the discriminant), and CADO's
 * expected_alpha. The discriminant is Res(f, f')/lc(f), computed exactly with a
 * fraction-free (Bareiss) determinant of the Sylvester matrix; only its divisibility
 * by p and p^2 is used. The p-adic recursion special_val0 is the same algorithm as CADO
 * (Hanrot's), on fixed-width integers.
 */
#pragma once

#include "mpint.h"
#include "sopt_lll.h" /* translate */

namespace s23 {

/* recursion cap for special_val0 (multiple roots that lift repeatedly); beyond it the
   caller falls back to the CPU. Each level costs ~2 KB of GPU stack. */
#ifndef S23_VAL0_MAX_DEPTH
#define S23_VAL0_MAX_DEPTH 12
#endif
enum { VAL0_MAX_DEPTH = S23_VAL0_MAX_DEPTH };

/* scratch for the discriminant: the (2d-1) x (2d-1) Sylvester matrix */
template <int LL, int DEG>
struct DiscScratch {
    Int<LL> M[2 * DEG - 1][2 * DEG - 1];
};

/* pvaluation of disc(f) at p, capped at 2 (as CADO's special_valuation computes it).
 * Returns -1 on overflow. */
template <int LL, int DEG>
S23_HD int disc_pvaluation_table(int *pv, const uint32_t *primes, int nprimes, const Int<LL> *f,
                                 DiscScratch<LL, DEG> &S)
{
    const int n = 2 * DEG - 1;
    Int<LL> df[DEG], t1, t2, prev;
    for (int i = 1; i <= DEG; i++) {
        Int<LL> m;
        set_si(m, i);
        if (!mul(df[i - 1], f[i], m))
            return -1;
    }
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            set_zero(S.M[r][c]);
    for (int r = 0; r < DEG - 1; r++)
        for (int j = 0; j <= DEG; j++)
            S.M[r][r + j] = f[DEG - j];
    for (int r = 0; r < DEG; r++)
        for (int j = 0; j <= DEG - 1; j++)
            S.M[DEG - 1 + r][r + j] = df[DEG - 1 - j];
    set_si(prev, 1);
    bool zero_det = false;
    for (int k = 0; k < n - 1 && !zero_det; k++) {
        if (is_zero(S.M[k][k])) {
            int r = k + 1;
            while (r < n && is_zero(S.M[r][k]))
                r++;
            if (r == n) {
                zero_det = true;
                break;
            }
            for (int c = 0; c < n; c++) {
                Int<LL> x = S.M[k][c];
                S.M[k][c] = S.M[r][c];
                S.M[r][c] = x;
            }
        }
        for (int i = k + 1; i < n; i++)
            for (int j = k + 1; j < n; j++) {
                if (!mul(t1, S.M[i][j], S.M[k][k]) || !mul(t2, S.M[i][k], S.M[k][j]) || !sub(t1, t1, t2) ||
                    !div_exact(S.M[i][j], t1, prev))
                    return -1;
            }
        prev = S.M[k][k];
    }
    Int<LL> disc;
    if (zero_det)
        set_zero(disc);
    else if (!div_exact(disc, S.M[n - 1][n - 1], f[DEG]))
        return -1;
    for (int i = 0; i < nprimes; i++) {
        uint32_t p = primes[i];
        if (mod_u32(disc, p) != 0)
            pv[i] = 0;
        else
            pv[i] = mod_u32(disc, p * p) != 0 ? 1 : 2;
    }
    return 0;
}

/* f(x) mod p at x, coefficients already reduced */
S23_HD uint32_t eval_mod(const uint32_t *fm, int d, uint32_t x, uint32_t p)
{
    uint64_t r = 0;
    for (int i = d; i >= 0; i--)
        r = (r * x + fm[i]) % p;
    return (uint32_t)r;
}

/* number of distinct roots of f mod p (mpz_poly_roots_ulong with NULL) */
template <int L> S23_HD unsigned nroots_mod(const Int<L> *f, int d, uint32_t p)
{
    uint32_t fm[16];
    for (int i = 0; i <= d; i++)
        fm[i] = mod_u32(f[i], p);
    unsigned n = 0;
    for (uint32_t x = 0; x < p; x++)
        if (eval_mod(fm, d, x, p) == 0)
            n++;
    return n;
}

/* CADO's special_val0: average p-valuation of f(x), x uniform in Z_p (recursive) */
template <int L, int DEG>
S23_HD_CALL double special_val0(const Int<L> *f, uint32_t p, int depth, bool &ok)
{
    double v = 0.0;
    Int<L> g[DEG + 1], pi;
    bool allzero = true;
    for (int i = 0; i <= DEG; i++) {
        g[i] = f[i];
        allzero = allzero && is_zero(g[i]);
    }
    if (allzero) {
        ok = false;
        return 0.0;
    }
    set_si(pi, p);
    for (;;) { /* divide out the p-part of the content */
        bool div = true;
        for (int i = 0; i <= DEG && div; i++)
            div = mod_u32(g[i], p) == 0;
        if (!div)
            break;
        for (int i = 0; i <= DEG; i++)
            if (!div_exact(g[i], g[i], pi)) {
                ok = false;
                return 0.0;
            }
        v++;
    }
    uint32_t gm[DEG + 1], dgm[DEG];
    for (int i = 0; i <= DEG; i++)
        gm[i] = mod_u32(g[i], p);
    for (int i = 1; i <= DEG; i++)
        dgm[i - 1] = (uint32_t)(((uint64_t)gm[i] * (uint64_t)(i % p)) % p);
    for (uint32_t r = 0; r < p; r++) {
        if (eval_mod(gm, DEG, r, p) != 0)
            continue;
        if (eval_mod(dgm, DEG - 1, r, p) != 0) {
            v += 1.0 / (double)(p - 1);
        } else {
            if (depth >= VAL0_MAX_DEPTH) {
                ok = false;
                return v;
            }
            /* H(x) = g(p*x + r) */
            Int<L> H[DEG + 1], rr, pw;
            for (int i = 0; i <= DEG; i++)
                H[i] = g[i];
            set_si(rr, r);
            if (!translate<L, DEG>(H, rr)) {
                ok = false;
                return v;
            }
            set_si(pw, 1);
            for (int i = 0; i <= DEG; i++) {
                if ((i > 0 && !mul(pw, pw, pi)) || !mul(H[i], H[i], pw)) {
                    ok = false;
                    return v;
                }
            }
            v += special_val0<L, DEG>(H, p, depth + 1, ok) / (double)p;
            if (!ok)
                return v;
        }
    }
    return v;
}

/* special_valuation(f, p) - special_valuation_affine(f, p), as CADO computes the two
 * (polyselect_alpha.c), with special_val0(f, p) evaluated once and shared: when p divides
 * the discriminant exactly once, special_valuation uses the shortcut (p*e - 1)/(p^2 - 1)
 * while the affine part takes the exact recursion, and CADO subtracts the two anyway. */
template <int L, int DEG>
S23_HD double projective_term(const Int<L> *f, uint32_t p, int pvaluation_disc, bool &ok)
{
    const double pd = (double)p;
    const bool p_divides_lc = mod_u32(f[DEG], p) == 0;
    if (pvaluation_disc == 0) {
        unsigned e = nroots_mod(f, DEG, p);
        double aff = (pd * e) / (pd * pd - 1);
        if (p_divides_lc)
            e++;
        return (pd * e) / (pd * pd - 1) - aff;
    }
    const double v0 = special_val0<L, DEG>(f, p, 0, ok);
    double aff = v0 * pd;
    aff /= pd + 1.0;
    double full;
    if (pvaluation_disc == 1) {
        unsigned e = nroots_mod(f, DEG, p);
        if (p_divides_lc)
            e++;
        full = (pd * e - 1) / (pd * pd - 1);
    } else {
        full = v0 * pd;
        if (p_divides_lc) {
            Int<L> G[DEG + 1], pw, pi;
            set_si(pw, 1);
            set_si(pi, p);
            for (int i = 0; i <= DEG; i++) {
                if ((i > 0 && !mul(pw, pw, pi)) || !mul(G[i], f[DEG - i], pw)) {
                    ok = false;
                    return 0.0;
                }
            }
            full += special_val0<L, DEG>(G, p, 0, ok);
        }
        full /= pd + 1.0;
    }
    return full - aff;
}

/* get_alpha_projective(f, 100) */
template <int L, int LL, int DEG>
S23_HD double alpha_projective_100(const Int<L> *f, DiscScratch<LL, DEG> &S, bool &ok)
{
    const uint32_t primes[25] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                                 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    int pv[25];
    Int<LL> fw[DEG + 1];
    for (int i = 0; i <= DEG; i++)
        if (!resize(fw[i], f[i])) {
            ok = false;
            return 0.0;
        }
    if (disc_pvaluation_table<LL, DEG>(pv, primes, 25, fw, S) != 0) {
        ok = false;
        return 0.0;
    }
    double e = projective_term<L, DEG>(f, 2, pv[0], ok);
    double alpha = (-e) * log(2.0);
    for (int i = 1; i < 25; i++) {
        e = projective_term<L, DEG>(f, primes[i], pv[i], ok);
        alpha += (-e) * log((double)primes[i]);
    }
    return alpha;
}

/* expected_alpha(logK) with MU = 0, SIGMA = 0.824 */
S23_HD double expected_alpha(double logK)
{
    if (logK < 0.999)
        return 0.0;
    return 0.0 - 0.824 * (sqrt(2 * logK) - (log(logK) + 1.3766) / (2 * sqrt(2 * logK)));
}

} // namespace s23
