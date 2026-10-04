/* Size model for the root sieve (M2, host code for now): the L2 lognorm (CADO's, at
 * f's own optimal skew) of a rotation f_{u,v} = f + (u x + v) g after re-translating it,
 *
 *   L(u, v) = min over integer t of L2_skew_lognorm(f_{u,v}(x + t)),
 *
 * since alpha does not depend on t (design constraint 5). L is smooth along v, so a
 * u-line is described by its minimum, the band [v_lo, v_hi] where L <= L_ref + budget,
 * and knots for interpolation inside the band. The t search is a pattern search
 * (doubling while it improves, halving otherwise) started from a nearby optimum.
 */
#pragma once

#include <vector>
#include "mpint.h"
#include "norms.h"
#include "sopt_lll.h" /* translate */

namespace s23 {

template <int L> struct RsizePoly {
    Int<L> f[6], g[2];
};

/* f + (u x + v) g, exactly; false on overflow */
template <int L> bool rs_rotate(Int<L> *fr, const RsizePoly<L> &P, int64_t u, int64_t v)
{
    Int<L> U, V, x;
    set_si(U, u);
    set_si(V, v);
    for (int i = 0; i <= 5; i++)
        fr[i] = P.f[i];
    /* c0 += v g0, c1 += v g1 + u g0, c2 += u g1 */
    return mul(x, V, P.g[0]) && add(fr[0], fr[0], x) && mul(x, V, P.g[1]) && add(fr[1], fr[1], x) &&
           mul(x, U, P.g[0]) && add(fr[1], fr[1], x) && mul(x, U, P.g[1]) && add(fr[2], fr[2], x);
}

/* lognorm of fr(x + t); +inf on overflow */
template <int L> double rs_lognorm_at(const Int<L> *fr, int64_t t)
{
    Int<L> ft[6], k;
    for (int i = 0; i <= 5; i++)
        ft[i] = fr[i];
    set_si(k, t);
    if (!translate<L, 5>(ft, k))
        return INFINITY;
    return L2_skew_lognorm(ft, 5);
}

/* L(u, v): t is the start and returns the best translation found */
template <int L> double rs_size(const RsizePoly<L> &P, int64_t u, int64_t v, int64_t &t, int64_t step0 = 1 << 16)
{
    Int<L> fr[6];
    if (!rs_rotate(fr, P, u, v))
        return INFINITY;
    double best = rs_lognorm_at(fr, t);
    int64_t step = step0 > 0 ? step0 : 1;
    while (step >= 1) {
        bool moved = false;
        for (int dir = 1; dir >= -1 && !moved; dir -= 2) {
            const double c = rs_lognorm_at(fr, t + dir * step);
            if (c < best) {
                best = c;
                t += dir * step;
                moved = true;
            }
        }
        if (moved) {
            if (step < ((int64_t)1 << 40))
                step *= 2;
        } else
            step /= 2;
    }
    return best;
}

struct RsLine {
    int64_t u;
    int64_t v_min, t_min; /* the line's minimum */
    double L_min;
    int64_t v_lo = 1, v_hi = 0; /* band: L <= L_ref + budget (empty if v_lo > v_hi) */
    std::vector<int64_t> kv, kt; /* knots inside the band: v, best t */
    std::vector<double> kL;
};

/* minimum of L along line u, by a pattern search over v from v (with t warm-started) */
template <int L> void rs_line_min(RsLine &R, const RsizePoly<L> &P, int64_t u, int64_t v, int64_t t, int64_t vstep)
{
    double best = rs_size(P, u, v, t);
    int64_t step = vstep > 0 ? vstep : 1;
    while (step >= 1) {
        bool moved = false;
        for (int dir = 1; dir >= -1 && !moved; dir -= 2) {
            int64_t tt = t;
            const double c = rs_size(P, u, v + dir * step, tt);
            if (c < best) {
                best = c;
                v += dir * step;
                t = tt;
                moved = true;
            }
        }
        if (moved) {
            if (step < ((int64_t)1 << 50))
                step *= 2;
        } else
            step /= 2;
    }
    R.u = u;
    R.v_min = v;
    R.t_min = t;
    R.L_min = best;
}

/* the band edge on one side (dir = +1 or -1) of the minimum: the last v with
 * L <= limit, to within `res` cells, by doubling then bisection */
template <int L> int64_t rs_band_edge(const RsLine &R, const RsizePoly<L> &P, double limit, int dir, int64_t res)
{
    int64_t in = R.v_min, step = res > 0 ? res : 1, t = R.t_min, t_in = R.t_min;
    int64_t out;
    for (;;) {
        const int64_t v = R.v_min + dir * step;
        int64_t tt = t_in;
        const double c = rs_size(P, R.u, v, tt);
        if (c > limit) {
            out = v;
            break;
        }
        in = v;
        t_in = tt;
        if (step > ((int64_t)1 << 55)) /* no edge in range */
            return v;
        step *= 2;
    }
    t = t_in;
    while ((out > in ? out - in : in - out) > res) {
        const int64_t mid = in + (out - in) / 2;
        int64_t tt = t;
        const double c = rs_size(P, R.u, mid, tt);
        if (c > limit)
            out = mid;
        else {
            in = mid;
            t = tt;
        }
    }
    return in;
}

/* `nknots` knots across the band [v_lo, v_hi] of line R (set beforehand), computed from
 * the knot nearest the minimum outwards, so each warm-starts from its neighbour */
template <int L> void rs_line_knots(RsLine &R, const RsizePoly<L> &P, int nknots)
{
    R.kv.clear();
    R.kt.clear();
    R.kL.clear();
    if (R.v_lo > R.v_hi || nknots <= 0)
        return;
    int64_t t = R.t_min;
    std::vector<int64_t> v(nknots);
    for (int i = 0; i < nknots; i++)
        v[i] = R.v_lo + (int64_t)((double)(R.v_hi - R.v_lo) * i / (nknots > 1 ? nknots - 1 : 1));
    std::vector<double> Lk(nknots);
    std::vector<int64_t> tk(nknots);
    int mid = 0;
    for (int i = 0; i < nknots; i++)
        if ((v[i] > R.v_min ? v[i] - R.v_min : R.v_min - v[i]) <
            (v[mid] > R.v_min ? v[mid] - R.v_min : R.v_min - v[mid]))
            mid = i;
    for (int i = mid; i < nknots; i++) {
        Lk[i] = rs_size(P, R.u, v[i], t);
        tk[i] = t;
    }
    t = tk[mid];
    for (int i = mid - 1; i >= 0; i--) {
        Lk[i] = rs_size(P, R.u, v[i], t);
        tk[i] = t;
    }
    R.kv = v;
    R.kt = tk;
    R.kL = Lk;
}

/* the band of line R (its minimum already found) and `nknots` knots across it (none if
 * nknots <= 0: the edges only) */
template <int L> void rs_line_band(RsLine &R, const RsizePoly<L> &P, double limit, int64_t res, int nknots)
{
    R.kv.clear();
    R.kt.clear();
    R.kL.clear();
    if (R.L_min > limit) {
        R.v_lo = 1;
        R.v_hi = 0;
        return;
    }
    R.v_lo = rs_band_edge(R, P, limit, -1, res);
    R.v_hi = rs_band_edge(R, P, limit, +1, res);
    rs_line_knots(R, P, nknots);
}

/* L and t by linear interpolation between the knots (an estimate) */
inline double rs_line_interp(const RsLine &R, int64_t v, int64_t &t)
{
    const size_t n = R.kv.size();
    if (n == 0) {
        t = R.t_min;
        return R.L_min;
    }
    if (v <= R.kv[0] || n == 1) {
        t = R.kt[0];
        return R.kL[0];
    }
    if (v >= R.kv[n - 1]) {
        t = R.kt[n - 1];
        return R.kL[n - 1];
    }
    size_t i = 1;
    while (R.kv[i] < v)
        i++;
    const double a = (double)(v - R.kv[i - 1]) / (double)(R.kv[i] - R.kv[i - 1]);
    t = R.kt[i - 1] + (int64_t)(a * (double)(R.kt[i] - R.kt[i - 1]));
    return R.kL[i - 1] + a * (R.kL[i] - R.kL[i - 1]);
}

} // namespace s23
