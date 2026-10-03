/* Fixed-width signed multi-precision integers for host and device.
 *
 * Int<L> is an L-limb (32 bits each) two's-complement integer. Every operation that can
 * overflow returns false instead of wrapping, so callers can fall back to the CPU path.
 * Division is Knuth's algorithm D (after Hacker's Delight, divmnu) on magnitudes, with
 * floor or exact semantics on top, matching what CADO's GMP code computes:
 *   divmod_floor  = mpz_fdiv_qr
 *   div_exact     = mpz_divexact (fails if there is a remainder)
 *   get_d         = mpz_get_d (truncates toward zero)
 *
 * Plain loops, no carry intrinsics: correctness first; PTX fast paths can come later
 * behind the same interface.
 */
#pragma once

#include <math.h>
#include "hd.h"

namespace s23 {

#ifdef S23_TRACK_BITS
extern int g_max_bits; /* host tests only: widest magnitude ever produced */
#endif

template <int L>
struct Int {
    uint32_t w[L];
};

/* ---- basic helpers ---------------------------------------------------- */

template <int L> S23_HD void set_zero(Int<L> &r)
{
    for (int i = 0; i < L; i++)
        r.w[i] = 0;
}

template <int L> S23_HD void set_si(Int<L> &r, int64_t x)
{
    uint64_t u = (uint64_t)x;
    r.w[0] = (uint32_t)u;
    r.w[1] = (uint32_t)(u >> 32);
    uint32_t fill = x < 0 ? 0xffffffffu : 0u;
    for (int i = 2; i < L; i++)
        r.w[i] = fill;
}

template <int L> S23_HD bool is_neg(const Int<L> &a)
{
    return (a.w[L - 1] >> 31) != 0;
}

template <int L> S23_HD bool is_zero(const Int<L> &a)
{
    for (int i = 0; i < L; i++)
        if (a.w[i])
            return false;
    return true;
}

template <int L> S23_HD int sgn(const Int<L> &a)
{
    return is_neg(a) ? -1 : (is_zero(a) ? 0 : 1);
}

/* Signed comparison: -1, 0, 1. */
template <int L> S23_HD int cmp(const Int<L> &a, const Int<L> &b)
{
    bool na = is_neg(a), nb = is_neg(b);
    if (na != nb)
        return na ? -1 : 1;
    for (int i = L - 1; i >= 0; i--)
        if (a.w[i] != b.w[i])
            return a.w[i] < b.w[i] ? -1 : 1;
    return 0;
}

/* |a| into an unsigned limb array (always fits: |min| = 2^(32L-1)). */
template <int L> S23_HD void magnitude(uint32_t *m, const Int<L> &a)
{
    if (!is_neg(a)) {
        for (int i = 0; i < L; i++)
            m[i] = a.w[i];
        return;
    }
    uint64_t c = 1;
    for (int i = 0; i < L; i++) {
        c += (uint64_t)(~a.w[i]);
        m[i] = (uint32_t)c;
        c >>= 32;
    }
}

/* Compare two unsigned magnitudes of n limbs: -1, 0, 1. */
S23_HD int ucmp(const uint32_t *a, const uint32_t *b, int n)
{
    for (int i = n - 1; i >= 0; i--)
        if (a[i] != b[i])
            return a[i] < b[i] ? -1 : 1;
    return 0;
}

S23_HD int limbs_used(const uint32_t *m, int n)
{
    while (n > 0 && m[n - 1] == 0)
        n--;
    return n;
}

/* r = +/- m (m an unsigned magnitude of L limbs); false if it does not fit. */
template <int L> S23_HD bool from_magnitude(Int<L> &r, const uint32_t *m, bool negative)
{
    bool top = (m[L - 1] >> 31) != 0;
    if (top) {
        /* only -2^(32L-1) fits */
        if (!negative || m[L - 1] != 0x80000000u)
            return false;
        for (int i = 0; i < L - 1; i++)
            if (m[i])
                return false;
    }
    if (!negative) {
        for (int i = 0; i < L; i++)
            r.w[i] = m[i];
    } else {
        uint64_t c = 1;
        for (int i = 0; i < L; i++) {
            c += (uint64_t)(~m[i]);
            r.w[i] = (uint32_t)c;
            c >>= 32;
        }
    }
#ifdef S23_TRACK_BITS
    {
        int n = limbs_used(m, L);
        int bits = n ? 32 * (n - 1) + (32 - clz32(m[n - 1])) : 0;
        if (bits > g_max_bits)
            g_max_bits = bits;
    }
#endif
    return true;
}

template <int L> S23_HD int bitlen(const Int<L> &a)
{
    uint32_t m[L];
    magnitude(m, a);
    int n = limbs_used(m, L);
    return n ? 32 * (n - 1) + (32 - clz32(m[n - 1])) : 0;
}

/* ---- add, sub, neg ------------------------------------------------------ */

template <int L> S23_HD bool add(Int<L> &r, const Int<L> &a, const Int<L> &b)
{
    bool na = is_neg(a), nb = is_neg(b);
    uint64_t c = 0;
    for (int i = 0; i < L; i++) {
        c += (uint64_t)a.w[i] + b.w[i];
        r.w[i] = (uint32_t)c;
        c >>= 32;
    }
    return !(na == nb && is_neg(r) != na);
}

template <int L> S23_HD bool sub(Int<L> &r, const Int<L> &a, const Int<L> &b)
{
    bool na = is_neg(a), nb = is_neg(b);
    int64_t c = 0;
    for (int i = 0; i < L; i++) {
        c += (int64_t)a.w[i] - (int64_t)b.w[i];
        r.w[i] = (uint32_t)c;
        c >>= 32; /* arithmetic shift: 0 or -1 */
    }
    return !(na != nb && is_neg(r) != na);
}

template <int L> S23_HD bool neg(Int<L> &r, const Int<L> &a)
{
    bool na = is_neg(a);
    uint64_t c = 1;
    for (int i = 0; i < L; i++) {
        c += (uint64_t)(~a.w[i]);
        r.w[i] = (uint32_t)c;
        c >>= 32;
    }
    /* -min overflows: the result is still negative */
    return !(na && is_neg(r));
}

/* ---- multiplication ----------------------------------------------------- */

template <int L> S23_HD_CALL bool mul(Int<L> &r, const Int<L> &a, const Int<L> &b)
{
    uint32_t ma[L], mb[L], p[L + 1];
    magnitude(ma, a);
    magnitude(mb, b);
    int na = limbs_used(ma, L), nb = limbs_used(mb, L);
    bool negative = is_neg(a) != is_neg(b);
    if (na == 0 || nb == 0) {
        set_zero(r);
        return true;
    }
    if (na + nb - 1 > L)
        return false;
    for (int i = 0; i <= L; i++)
        p[i] = 0;
    for (int i = 0; i < na; i++) {
        uint64_t c = 0;
        for (int j = 0; j < nb; j++) {
            c += (uint64_t)ma[i] * mb[j] + p[i + j];
            p[i + j] = (uint32_t)c;
            c >>= 32;
        }
        p[i + nb] = (uint32_t)c; /* i + nb <= L since na + nb - 1 <= L */
    }
    if (p[L] != 0)
        return false;
    return from_magnitude(r, p, negative);
}

/* ---- division ----------------------------------------------------------- */

/* Unsigned long division: u (m limbs) / v (n limbs, v[n-1] != 0, m >= n).
 * q gets m - n + 1 limbs, rem gets n limbs. Arrays are sized for L limbs. */
template <int L>
S23_HD_CALL void udivmod(uint32_t *q, uint32_t *rem, const uint32_t *u, int m, const uint32_t *v, int n)
{
    const uint64_t b = 1ull << 32;
    if (n == 1) {
        uint64_t k = 0;
        for (int j = m - 1; j >= 0; j--) {
            uint64_t cur = (k << 32) | u[j];
            q[j] = (uint32_t)(cur / v[0]);
            k = cur % v[0];
        }
        rem[0] = (uint32_t)k;
        return;
    }
    int s = clz32(v[n - 1]);
    uint32_t vn[L], un[L + 1];
    for (int i = n - 1; i > 0; i--)
        vn[i] = (v[i] << s) | (s ? (uint32_t)((uint64_t)v[i - 1] >> (32 - s)) : 0u);
    vn[0] = v[0] << s;
    un[m] = s ? (uint32_t)((uint64_t)u[m - 1] >> (32 - s)) : 0u;
    for (int i = m - 1; i > 0; i--)
        un[i] = (u[i] << s) | (s ? (uint32_t)((uint64_t)u[i - 1] >> (32 - s)) : 0u);
    un[0] = u[0] << s;

    for (int j = m - n; j >= 0; j--) {
        uint64_t num = ((uint64_t)un[j + n] << 32) | un[j + n - 1];
        uint64_t qhat = num / vn[n - 1];
        uint64_t rhat = num - qhat * vn[n - 1];
        while (qhat >= b || qhat * vn[n - 2] > ((rhat << 32) | un[j + n - 2])) {
            qhat--;
            rhat += vn[n - 1];
            if (rhat >= b)
                break;
        }
        int64_t k = 0, t;
        for (int i = 0; i < n; i++) {
            uint64_t p = qhat * vn[i];
            t = (int64_t)un[i + j] - k - (int64_t)(p & 0xffffffffull);
            un[i + j] = (uint32_t)t;
            k = (int64_t)(p >> 32) - (t >> 32);
        }
        t = (int64_t)un[j + n] - k;
        un[j + n] = (uint32_t)t;
        q[j] = (uint32_t)qhat;
        if (t < 0) {
            q[j]--;
            uint64_t c = 0;
            for (int i = 0; i < n; i++) {
                c += (uint64_t)un[i + j] + vn[i];
                un[i + j] = (uint32_t)c;
                c >>= 32;
            }
            un[j + n] += (uint32_t)c;
        }
    }
    for (int i = 0; i < n - 1; i++)
        rem[i] = (un[i] >> s) | (s ? (uint32_t)((uint64_t)un[i + 1] << (32 - s)) : 0u);
    rem[n - 1] = un[n - 1] >> s;
}

/* q = floor(a / b), r = a - q*b (r has the sign of b), as mpz_fdiv_qr. b != 0. */
template <int L> S23_HD_CALL bool divmod_floor(Int<L> &q, Int<L> &r, const Int<L> &a, const Int<L> &b)
{
    uint32_t ma[L], mb[L], uq[L], ur[L];
    magnitude(ma, a);
    magnitude(mb, b);
    int na = limbs_used(ma, L), nb = limbs_used(mb, L);
    if (nb == 0)
        return false;
    for (int i = 0; i < L; i++)
        uq[i] = ur[i] = 0;
    if (na < nb || (na == nb && ucmp(ma, mb, na) < 0)) {
        for (int i = 0; i < L; i++)
            ur[i] = ma[i]; /* |a| < |b|: quotient 0 */
    } else {
        udivmod<L>(uq, ur, ma, na, mb, nb);
    }
    bool sa = is_neg(a), sb = is_neg(b);
    bool rem_zero = limbs_used(ur, L) == 0;
    Int<L> Q, R;
    if (!from_magnitude(Q, uq, sa != sb) || !from_magnitude(R, ur, sa))
        return false; /* R = a - trunc(a/b)*b, sign of a */
    if (sa != sb && !rem_zero) {
        /* truncation rounded toward zero; floor needs one more step down */
        Int<L> one;
        set_si(one, 1);
        if (!sub(Q, Q, one) || !add(R, R, b))
            return false;
    }
    q = Q;
    r = R;
    return true;
}

/* q = a / b, which must be exact (as mpz_divexact); false otherwise. */
template <int L> S23_HD bool div_exact(Int<L> &q, const Int<L> &a, const Int<L> &b)
{
    Int<L> r;
    if (!divmod_floor(q, r, a, b))
        return false;
    return is_zero(r);
}

/* ---- conversion ---------------------------------------------------------- */

/* Truncating conversion to double, as mpz_get_d. */
template <int L> S23_HD double get_d(const Int<L> &a)
{
    uint32_t m[L];
    magnitude(m, a);
    int n = limbs_used(m, L);
    if (n == 0)
        return 0.0;
    int bits = 32 * (n - 1) + (32 - clz32(m[n - 1]));
    int shift = bits > 53 ? bits - 53 : 0;
    /* the 53 (or fewer) bits starting at bit `shift`: exact in a double */
    uint64_t top = 0;
    for (int k = 0; k < 64; k += 32) {
        int bit = shift + k;
        int limb = bit >> 5, off = bit & 31;
        uint64_t lo = limb < L ? m[limb] : 0;
        uint64_t hi = limb + 1 < L ? m[limb + 1] : 0;
        uint64_t part = ((hi << 32) | lo) >> off;
        top |= (part & 0xffffffffull) << k;
    }
    if (bits > 53)
        top &= (1ull << 53) - 1;
    double d = ldexp((double)top, shift);
    return is_neg(a) ? -d : d;
}

/* ---- small helpers used by the size optimization ------------------------- */

/* a mod m in [0, m) for 0 < m < 2^32 */
template <int L> S23_HD uint32_t mod_u32(const Int<L> &a, uint32_t m)
{
    uint32_t mg[L];
    magnitude(mg, a);
    uint64_t r = 0;
    for (int i = L - 1; i >= 0; i--)
        r = ((r << 32) | mg[i]) % m;
    if (is_neg(a) && r)
        r = m - r;
    return (uint32_t)r;
}

/* r = trunc(x) as mpz_set_d; false for NaN/inf or |x| >= 2^62 */
template <int L> S23_HD bool set_d_trunc(Int<L> &r, double x)
{
    if (!(x > -4.611686018427388e18 && x < 4.611686018427388e18))
        return false;
    set_si(r, (int64_t)x);
    return true;
}

/* change width (sign-extending); false if the value does not fit */
template <int L2, int L1> S23_HD bool resize(Int<L2> &r, const Int<L1> &a)
{
    uint32_t fill = is_neg(a) ? 0xffffffffu : 0u;
    for (int i = 0; i < L2; i++)
        r.w[i] = i < L1 ? a.w[i] : fill;
    for (int i = L2; i < L1; i++)
        if (a.w[i] != fill)
            return false;
    return is_neg(r) == is_neg(a);
}

/* r = a * 2 (false on overflow) */
template <int L> S23_HD bool mul2(Int<L> &r, const Int<L> &a)
{
    return add(r, a, a);
}

/* floor(a / 2), as mpz_fdiv_q_2exp(., ., 1): arithmetic shift right */
template <int L> S23_HD void fdiv2(Int<L> &r, const Int<L> &a)
{
    uint32_t top = a.w[L - 1];
    for (int i = 0; i < L - 1; i++)
        r.w[i] = (a.w[i] >> 1) | (a.w[i + 1] << 31);
    r.w[L - 1] = (uint32_t)((int32_t)top >> 1);
}

/* trunc(a / 2), as mpz_tdiv_q_2exp(., ., 1) */
template <int L> S23_HD void tdiv2(Int<L> &r, const Int<L> &a)
{
    bool odd_neg = is_neg(a) && (a.w[0] & 1);
    fdiv2(r, a);
    if (odd_neg) { /* floor went one below trunc */
        Int<L> one;
        set_si(one, 1);
        add(r, r, one);
    }
}

template <int L> S23_HD bool is_one(const Int<L> &a)
{
    if (a.w[0] != 1)
        return false;
    for (int i = 1; i < L; i++)
        if (a.w[i])
            return false;
    return true;
}

} // namespace s23
