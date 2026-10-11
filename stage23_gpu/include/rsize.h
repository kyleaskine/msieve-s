/* Size model for the root sieve (M2, host code for now): the L2 lognorm (CADO's, at
 * f's own optimal skew) of a rotation f_{u,v} = f + (u x + v) g after re-translating it,
 *
 *   L(u, v) = min over integer t of L2_skew_lognorm(f_{u,v}(x + t)),
 *
 * since alpha does not depend on t (design constraint 5). L is smooth along v, so a
 * u-line is described by its minimum, the band [v_lo, v_hi] where L <= L_ref + budget,
 * and knots for interpolation inside the band. The t search is a pattern search
 * (doubling while it improves, halving otherwise) started from a nearby optimum.
 *
 * Speed (2026-10-10, about 14x on the size model of 14 seeds, c168 to c208): the rotations and
 * translations are exact multiply-adds by 64-bit scalars (addmul_si); the line scan, band edges
 * and knots find each skewness by Newton from the previous evaluation's (rs_skewness), start
 * each t search from t interpolated between nearby points with a first step of 1, find band
 * edges by false position, and keep the band edges' doubling points across the -maxcells
 * loop's budgets (rs_fast_search). Callers that pass no hint to rs_size (the exact values
 * -rerank ranks by) keep CADO's root isolation and their own first step.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "mpint.h"
#include "norms.h"

namespace s23 {

template <int L> struct RsizePoly {
    Int<L> f[6], g[2];
};

/* f + (u x + v) g, exactly; false on overflow */
template <int L> bool rs_rotate(Int<L> *fr, const RsizePoly<L> &P, int64_t u, int64_t v)
{
    for (int i = 0; i <= 5; i++)
        fr[i] = P.f[i];
    /* c0 += v g0, c1 += v g1 + u g0, c2 += u g1 */
    return addmul_si(fr[0], fr[0], P.g[0], v) && addmul_si(fr[1], fr[1], P.g[1], v) &&
           addmul_si(fr[1], fr[1], P.g[0], u) && addmul_si(fr[2], fr[2], P.g[1], u);
}

/* c(x + k) by repeated synthetic division, as translate<L, 5> with a 64-bit k */
template <int L> bool rs_translate(Int<L> *c, int64_t k)
{
    for (int i = 0; i < 5; i++)
        for (int j = 4; j >= i; j--)
            if (!addmul_si(c[j], c[j], c[j + 1], k))
                return false;
    return true;
}

/* rs_warm_skew = false: every skewness by CADO's root isolation (s23_ropt -exactskew) */
inline bool rs_warm_skew = true;
/* rs_fast_search = false: the size model's own searches (line minima, band edges, knots) as
   before 2026-10-10 (s23_ropt -oldsearch): each t search from the neighbour's t with a first
   step of 2^16, band edges by doubling and bisection, from scratch at every budget. With it (the
   default): t starts from a linear inter- or extrapolation of two nearby points' optimal t with a
   first step of 1, band edges are bracketed from kept doubling points (RsEdge) and shrunk by
   Illinois false position (rs_band_edge). The t search from a guess is a local descent: where
   L(t) has more than one local minimum it can stop at another one than the old first step of
   2^16 reached (the old search was local too, at a larger scale). Checked on 14 seeds (same line
   minima and band edges) and the C184's 320 searches (same best cell in each). */
inline bool rs_fast_search = true;
inline int64_t rs_tstep() { return rs_fast_search ? 1 : 1 << 16; }

/* a start for t at v on a line, from the optimal t at two points of it (v0, t0), (v1, t1):
   linear inter- or extrapolation (t1 if v0 == v1 or the guess is out of range) */
inline int64_t rs_t_guess(int64_t v, int64_t v0, int64_t t0, int64_t v1, int64_t t1)
{
    if (!rs_fast_search || v1 == v0)
        return t1;
    const double t = (double)t0 + (double)(v - v0) / (double)(v1 - v0) * (double)(t1 - t0);
    return fabs(t) < 0x1p62 ? (int64_t)llround(t) : t1;
}

/* true if p (degree d, 1 <= d <= 5, p(0) != 0) has exactly one distinct real root in
 * (0, inf), by Sturm's theorem in doubles; false if it has more, or if the chain is too close
 * to degenerate to tell (a remainder, its leading coefficient, or a constant term within
 * 1e-9 of the size of what it came from). Callers scale p so its root of interest is near 1. */
inline bool rs_sturm_one_positive(const double *b, int d)
{
    double S[6][6];
    int deg[6];
    for (int k = 0; k <= d; k++)
        S[0][k] = b[k];
    for (int k = 0; k < d; k++)
        S[1][k] = (k + 1) * b[k + 1];
    deg[0] = d;
    deg[1] = d - 1;
    int m = 2;
    for (; deg[m - 1] > 0; m++) {
        double a[6], amax = 0;
        const int da = deg[m - 2], dq = deg[m - 1];
        const double *q = S[m - 1];
        for (int k = 0; k <= da; k++) {
            a[k] = S[m - 2][k];
            amax = fmax(amax, fabs(a[k]));
        }
        for (int i = da; i >= dq; i--) {
            const double f = a[i] / q[dq];
            for (int j = 0; j < dq; j++)
                a[i - dq + j] -= f * q[j];
            a[i] = 0;
        }
        double rmax = 0;
        for (int k = 0; k < dq; k++)
            rmax = fmax(rmax, fabs(a[k]));
        if (!(rmax > 1e-9 * amax) || !(fabs(a[dq - 1]) > 1e-9 * rmax))
            return false;
        for (int k = 0; k < dq; k++)
            S[m][k] = -a[k];
        deg[m] = dq - 1;
    }
    int v0 = 0, vinf = 0, s0 = 0, sinf = 0;
    for (int i = 0; i < m; i++) {
        double cmax = 0;
        for (int k = 0; k <= deg[i]; k++)
            cmax = fmax(cmax, fabs(S[i][k]));
        if (!(fabs(S[i][0]) > 1e-9 * cmax))
            return false;
        const int c0 = S[i][0] > 0 ? 1 : -1, cl = S[i][deg[i]] > 0 ? 1 : -1;
        v0 += s0 && c0 != s0;
        vinf += sinf && cl != sinf;
        s0 = c0;
        sinf = cl;
    }
    return v0 - vinf == 1;
}

/* L2_skewness_d, warm-started from xh, the root x = s^2 of a nearby evaluation (0: none),
 * which gets this one's root. CADO's skewness is the root of the derivative numerator dP(x)
 * with the lowest lognorm (L2_skewness_d isolates every positive root through the derivative
 * cascade, about 70% of an evaluation). Here: Newton from xh, kept inside a bracket [lo, hi]
 * (dP(lo) < 0 < dP(hi)) and bisecting when a step leaves it; dP must change sign across the
 * result r, and Sturm's theorem on dP(r y) must show dP has only one positive root, so r is
 * CADO's. Otherwise (no hint, degree above 5, Newton not converged in 60 steps, no sign change,
 * Sturm undecided or more roots) CADO's isolation. Both find the same root to a few
 * ulps, where the lognorm is stationary: the lognorms agree to an ulp (Newton is not CADO's
 * sequence of doubles, so an ulp can differ). Descartes' rule does not do as the test: dP's
 * coefficients usually change sign three times (a complex pair near the axis) with one root. */
inline double rs_skewness(const DPoly &P, double &xh)
{
    DPoly dP;
    L2_skewness_derivative_numerator(dP, P);
    const int d = dP.deg;
    auto eval = [&](double x) {
        double f = dP.c[d];
        for (int k = d - 1; k >= 0; k--)
            f = f * x + dP.c[k];
        return f;
    };
    if (rs_warm_skew && xh > 0 && d >= 1 && d <= 5 && dP.c[0] < 0 && dP.c[d] > 0) {
        double lo = 0, hi = INFINITY, x = xh;
        for (int it = 0; it < 60; it++) {
            double f = dP.c[d], fp = 0;
            for (int k = d - 1; k >= 0; k--) {
                fp = fp * x + f;
                f = f * x + dP.c[k];
            }
            if (f < 0)
                lo = x;
            else if (f > 0)
                hi = x;
            else
                lo = hi = x;
            double xn = x - f / fp;
            if (!(xn >= lo && xn <= hi)) /* left the bracket, or NaN */
                xn = hi < INFINITY ? 0.5 * (lo + hi) : 2 * x;
            if (fabs(xn - x) <= 0x1p-50 * x || hi - lo <= 0x1p-50 * x) {
                /* a small step only says x is near where dP is small: require dP to change sign
                   across xn (2^-40 either side, far outside the rounding noise of a simple root) */
                if (!(eval(xn * (1 - 0x1p-40)) < 0 && eval(xn * (1 + 0x1p-40)) > 0))
                    break;
                double b[6], rk = 1, bmax = 0;
                for (int k = 0; k <= d; k++) {
                    b[k] = dP.c[k] * rk;
                    rk *= xn;
                    bmax = fmax(bmax, fabs(b[k]));
                }
                for (int k = 0; k <= d; k++)
                    b[k] /= bmax;
                if (!(xn > 0) || !rs_sturm_one_positive(b, d))
                    break;
                xh = xn;
                return sqrt(xn);
            }
            x = xn;
        }
    }
    const double s = L2_skewness_d(P);
    xh = s * s;
    return s;
}

/* lognorm of fr(x + t); +inf on overflow. With xh, the skewness is warm-started from it
 * (rs_skewness); without, CADO's. */
template <int L> double rs_lognorm_at(const Int<L> *fr, int64_t t, double *xh = nullptr)
{
    Int<L> ft[6];
    for (int i = 0; i <= 5; i++)
        ft[i] = fr[i];
    if (!rs_translate(ft, t))
        return INFINITY;
    DPoly P;
    dp_from_ints(P, ft, 5);
    return L2_lognorm_d(P, xh ? rs_skewness(P, *xh) : L2_skewness_d(P));
}

/* L(u, v): t is the start and returns the best translation found. xh: as rs_lognorm_at, the
 * hint carried from evaluation to evaluation */
template <int L>
double rs_size(const RsizePoly<L> &P, int64_t u, int64_t v, int64_t &t, int64_t step0 = 1 << 16,
               double *xh = nullptr)
{
    Int<L> fr[6];
    if (!rs_rotate(fr, P, u, v))
        return INFINITY;
    double best = rs_lognorm_at(fr, t, xh);
    int64_t step = step0 > 0 ? step0 : 1;
    while (step >= 1) {
        bool moved = false;
        for (int dir = 1; dir >= -1 && !moved; dir -= 2) {
            const double c = rs_lognorm_at(fr, t + dir * step, xh);
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

/* the doubling points of one side's band edge search (v_min + dir r 2^k, k = 0, 1, ...), each
 * with its optimal t, L and the skewness hint after it: they do not depend on the limit, so a
 * search at another budget reads them instead of evaluating them again (rs_fast_search) and
 * brackets the edge exactly as a fresh search would. Kept for one r and one rs_warm_skew. */
struct RsEdge {
    int64_t r = 0;
    bool warm = false;
    std::vector<int64_t> v, t;
    std::vector<double> L, x;
};

struct RsLine {
    int64_t u;
    int64_t v_min, t_min; /* the line's minimum */
    double L_min;
    double x_min = 0; /* the skewness hint (s^2) near the minimum, to warm-start the line's searches */
    int64_t v_lo = 1, v_hi = 0; /* band: L <= L_ref + budget (empty if v_lo > v_hi) */
    std::vector<int64_t> kv, kt; /* knots inside the band: v, best t */
    std::vector<double> kL;
    RsEdge edge[2]; /* the band edge searches' doubling points, below and above v_min */
};

/* rs_size from the start g (a guess), or, if that search finds no finite value (the guess so far
 * out that every translation it tries overflows), from `fallback`, a t already evaluated nearby */
template <int L>
double rs_size_from(const RsizePoly<L> &P, int64_t u, int64_t v, int64_t &t, int64_t g, int64_t fallback, double *xh)
{
    t = g;
    double c = rs_size(P, u, v, t, rs_tstep(), xh);
    if (!(c < INFINITY) && g != fallback) {
        t = fallback;
        c = rs_size(P, u, v, t, rs_tstep(), xh);
    }
    return c;
}

/* minimum of L along line u, by a pattern search over v from v (with t warm-started) */
template <int L> void rs_line_min(RsLine &R, const RsizePoly<L> &P, int64_t u, int64_t v, int64_t t, int64_t vstep)
{
    double xh = 0;
    double best = rs_size(P, u, v, t, rs_tstep(), &xh);
    int64_t vl = v, tl = t; /* the last other point evaluated: t's slope along the line */
    int64_t step = vstep > 0 ? vstep : 1;
    while (step >= 1) {
        bool moved = false;
        for (int dir = 1; dir >= -1 && !moved; dir -= 2) {
            int64_t tt;
            const double c = rs_size_from(P, u, v + dir * step, tt, rs_t_guess(v + dir * step, vl, tl, v, t), t, &xh);
            if (c < best) {
                vl = v;
                tl = t;
                best = c;
                v += dir * step;
                t = tt;
                moved = true;
            } else if (c < INFINITY) {
                vl = v + dir * step;
                tl = tt;
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
    R.x_min = xh;
    R.edge[0] = R.edge[1] = RsEdge(); /* its doubling points start from this minimum */
}

/* the band edge on one side (dir = +1 or -1) of the minimum: the last v with L <= limit, to
 * within `res` cells. A bracket (a point inside, one outside) by doubling the distance from
 * v_min, then shrunk: by bisection (rs_fast_search off), or by false position on L - limit with
 * the Illinois rule and a bisection step whenever two steps have not halved the bracket, each t
 * search starting from the t interpolated between the bracket's points. With rs_fast_search the
 * doubling points are kept (RsEdge), so the -maxcells loop's lower budgets bracket from them
 * without evaluating. Where L is monotone between the bracket's points both find the same edge
 * (but at ulp-level ties); some lines have L non-monotone there (a second dip in v), and the two
 * then may find different crossings, as either search depends on where it probes. */
template <int L> int64_t rs_band_edge(RsLine &R, const RsizePoly<L> &P, double limit, int dir, int64_t res)
{
    RsEdge &E = R.edge[dir > 0];
    const int64_t r = res > 0 ? res : 1;
    if (E.r != r || E.warm != rs_warm_skew) {
        E = RsEdge();
        E.r = r;
        E.warm = rs_warm_skew;
    }
    int64_t in = R.v_min, t_in = R.t_min, out = 0, t_out = 0, vp = R.v_min, tp = R.t_min;
    double g_in = R.L_min - limit, g_out = INFINITY, xh = R.x_min; /* g: L - limit */
    for (size_t k = 0;; k++) {
        const int64_t step = r << k;
        int64_t v, tt;
        double c;
        if (rs_fast_search && k < E.v.size()) {
            v = E.v[k];
            tt = E.t[k];
            c = E.L[k];
            xh = E.x[k];
        } else {
            v = R.v_min + dir * step;
            c = rs_size_from(P, R.u, v, tt, rs_t_guess(v, vp, tp, in, t_in), t_in, &xh);
            if (rs_fast_search) {
                E.v.push_back(v);
                E.t.push_back(tt);
                E.L.push_back(c);
                E.x.push_back(xh);
            }
        }
        if (c > limit) {
            out = v;
            t_out = tt;
            g_out = c - limit;
            break;
        }
        vp = in;
        tp = t_in;
        in = v;
        t_in = tt;
        g_in = c - limit;
        if (step > ((int64_t)1 << 55)) /* no edge in range */
            return v;
    }
    if (!rs_fast_search) {
        int64_t t = t_in;
        while ((out > in ? out - in : in - out) > r) {
            const int64_t mid = in + (out - in) / 2;
            int64_t tt = t;
            const double c = rs_size(P, R.u, mid, tt, rs_tstep(), &xh);
            if (c > limit)
                out = mid;
            else {
                in = mid;
                t = tt;
            }
        }
        return in;
    }
    /* false position on g; Illinois halves the g of an end kept twice */
    int last = 0, n = 0;
    int64_t w2 = INT64_MAX, w1 = INT64_MAX; /* bracket widths two steps and one step back */
    while ((out > in ? out - in : in - out) > r) {
        const int64_t w = out > in ? out - in : in - out;
        int64_t mid;
        if (std::isfinite(g_out) && g_out > g_in && (n < 2 || w <= w2 / 2))
            mid = in + (int64_t)(g_in / (g_in - g_out) * (double)(out - in));
        else
            mid = in + (out - in) / 2;
        /* strictly inside: (double)(out - in) rounds above 2^53, so the step can overshoot */
        mid = out > in ? std::min(std::max(mid, in + 1), out - 1) : std::max(std::min(mid, in - 1), out + 1);
        int64_t tt;
        const double c = rs_size_from(P, R.u, mid, tt, rs_t_guess(mid, in, t_in, out, t_out), t_in, &xh);
        if (c > limit) {
            out = mid;
            t_out = tt;
            g_out = c - limit;
            if (last == 1)
                g_in *= 0.5;
            last = 1;
        } else {
            in = mid;
            t_in = tt;
            g_in = c - limit;
            if (last == -1)
                g_out *= 0.5;
            last = -1;
        }
        w2 = w1;
        w1 = w;
        n++;
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
    double xh = R.x_min;
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
    /* each knot's t search starts from t extrapolated from the two knots before it (the first
       two of each direction: from the neighbour's t) */
    for (int i = mid; i < nknots; i++) {
        const int64_t g = i >= mid + 2 ? rs_t_guess(v[i], v[i - 2], tk[i - 2], v[i - 1], tk[i - 1]) : t;
        Lk[i] = rs_size_from(P, R.u, v[i], t, g, i > mid ? tk[i - 1] : R.t_min, &xh);
        tk[i] = t;
    }
    t = tk[mid];
    for (int i = mid - 1; i >= 0; i--) {
        const int64_t g = i + 2 < nknots ? rs_t_guess(v[i], v[i + 2], tk[i + 2], v[i + 1], tk[i + 1]) : t;
        Lk[i] = rs_size_from(P, R.u, v[i], t, g, tk[i + 1], &xh);
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
