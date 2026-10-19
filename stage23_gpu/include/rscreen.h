/* Exact-arithmetic alpha/size screen, shared by host validation and CUDA.
 * Hanrot's valuation recurrence is valid for every prime, so it avoids the
 * large discriminant used only to choose CADO's shortcuts. The valuations
 * agree mathematically; the different floating-point order need not be bitwise
 * identical. Overflow/depth/iteration exhaustion requests host fallback.
 */
#pragma once
#include "alpha_proj.h"
#include "rsize.h"
#include "rprimes.h"

namespace s23 {

enum { RS_SCREEN_MAX_BOUND = 2000, RS_SCREEN_MAX_PRIMES = 303 };
// Empirical guard for divergent local minima, not a proved libm error bound.
constexpr double RS_SCREEN_CUT_GUARD = 0.125;

inline bool rs_screen_primes(int bound, std::vector<uint32_t> &primes)
{
    if (bound < 2 || bound > RS_SCREEN_MAX_BOUND) return false;
    primes = rs_primes(bound);
    return !primes.empty() && primes.size() <= RS_SCREEN_MAX_PRIMES;
}

inline double rs_screen_key(double L, double A, double aw)
{
    const double key = L + aw * A;
    return std::isfinite(key) ? key : INFINITY;
}

inline bool rs_screen_near_cut(double key, double cut)
{
    return std::isfinite(key) && std::isfinite(cut) && fabs(key - cut) <= RS_SCREEN_CUT_GUARD;
}

template<int L> S23_HD bool rs_screen_primitive(Int<L> *f, double &log_content)
{
    Int<L> content;
    set_zero(content);
    // Start at the leading coefficient: content is usually tiny.
    for (int i = 5; i >= 0; --i) {
        Int<L> a = f[i], q, r;
        if (is_neg(a) && !neg(a, a)) return false;
        while (!is_zero(a)) {
            if (!divmod_floor(q, r, content, a)) return false;
            content = a;
            a = r;
        }
        if (bitlen(content) == 1) break;
    }
    if (is_zero(content)) return false;
    log_content = log(get_d(content));
    if (bitlen(content) > 1)
        for (int i = 0; i <= 5; ++i)
            if (!div_exact(f[i], f[i], content)) return false;
    return true;
}

template<int L> S23_HD double rs_screen_valuation(const Int<L> *f, uint32_t p, bool &ok)
{
    const double pd = p;
    double v = special_val0<L, 5>(f, p, 0, ok) * pd;
    if (ok && mod_u32(f[5], p) == 0) {
        Int<L> g[6];
        if (!projective_poly<L, 5>(g, f, p)) { ok = false; return 0; }
        v += special_val0<L, 5>(g, p, 0, ok);
    }
    return v / (pd + 1.0);
}

// Same cold-skew pattern search and initial step as rs_exact_cell. Work and
// coordinates are bounded here because a device thread must always terminate.
template<int L> S23_HD double rs_screen_size(const Int<L> *f, int64_t &t, bool &ok)
{
    return rs_cold_size(f, t, 1 << 10, ok, 4096);
}

} // namespace s23
