/* Shared host MurphyE integral, skew grid and bounded translation/skew search.
 * Used by cado_murphy and s23_ropt's in-process final ranking. CADO headers require
 * C++20; keep this out of CUDA translation units. Include cado.h first.
 */
#pragma once
#include <cmath>
#include <algorithm>
#include <map>
#include <vector>
#include <utility>
#include <array>
#include <limits>
#include "cado_poly.h"
#include "auxiliary.h"
#include "rho.h"
#include "polynomial.hpp"
#include "mpz_poly.h"
#include "cxx_mpz.hpp"
#include "polyselect_norms.h"
#include "murphy_quadrature.h"

namespace s23_murphy {

struct Ctx {
    double Bf, Bg, area;
    int K; /* sample angles */
};

/* Only the quadrature order affects the unit directions. Keep the search and
 * validation grids per worker, rather than doing two transcendental evaluations
 * at every angle of every objective call. Two slots suffice for K/Keval and keep
 * memory bounded even if a library caller cycles through many orders. Large
 * one-off evaluations use the streaming path (at most 8 MiB cached per thread).
 */
struct AngleGrid {
    int K = 0;
    std::vector<std::array<double, 2>> xy;
};

static const AngleGrid *angle_grid(int K)
{
    if (K < 1 || K > 262144)
        return nullptr;
    static thread_local AngleGrid cache[2];
    static thread_local unsigned recent = 0;
    for (unsigned i = 0; i < 2; i++)
        if (cache[i].K == K) {
            recent = i;
            return &cache[i];
        }
    recent ^= 1;
    auto &grid = cache[recent];
    grid.xy.resize(K);
    for (int i = 0; i < K; i++) {
        const double ti = angle(i, K);
        grid.xy[i] = {cos(ti), sin(ti)};
    }
    grid.K = K;
    return &grid;
}

/* CADO's MurphyE loop with alpha given. Convert each coefficient with CADO's
 * number_context<double> once per call instead of once per angle. The evaluator,
 * explicit FMAs, angle order and scalar sum are unchanged; no fast-math or
 * parallel reduction. Translated GMP coefficients are freshly converted on every
 * call, so neither a translation nor a different polynomial can reuse stale data.
 */
template <typename F, typename G>
static double integral(const F &f, int degree, const G &g, double skew, double alpha_f, double alpha_g, const Ctx &C)
{
    double E = 0;
    const int K = C.K;
    const double x = sqrt(C.area * skew);
    const double y = sqrt(C.area / skew);
    const AngleGrid *grid = angle_grid(K);
    const double one_over_logBf = 1.0 / log(C.Bf);
    const double one_over_logBg = 1.0 / log(C.Bg);
    for (int i = 0; i < K; i++) {
        const double ti = angle(i, K);
        const double xi = x * (grid ? grid->xy[i][0] : cos(ti));
        const double yi = y * (grid ? grid->xy[i][1] : sin(ti));
        double vf = log(std::abs(homogeneous(f, degree, xi, yi))) + alpha_f;
        double vg = log(std::abs(homogeneous(g, 1, xi, yi))) + alpha_g;
        vf *= one_over_logBf;
        vg *= one_over_logBg;
        E += dickman_rho(vf) * dickman_rho(vg);
    }
    return E / (double)K;
}

static double murphy_at(cado_poly_srcptr p, double skew, double af, double ag, const Ctx &c)
{
    const polynomial<double> f(p->pols[ALG_SIDE]), g(p->pols[RAT_SIDE]);
    return integral(f, p->pols[ALG_SIDE]->deg, g, skew, af, ag, c);
}

/* best skew on the fixed grid; returns E, sets skew */
static double best_skew(cado_poly_srcptr cpoly, double alpha_f, double alpha_g, const Ctx &C, double &skew)
{
    mpz_poly_srcptr f = cpoly->pols[ALG_SIDE];
    const int d = f->deg;
    double s0 = pow(fabs(mpz_get_d(mpz_poly_coeff_const(f, 0)) / mpz_get_d(mpz_poly_coeff_const(f, d))),
                    1.0 / d);
    if (!(s0 > 0 && std::isfinite(s0)))
        s0 = L2_skewness(f);
    if (!(s0 > 0 && std::isfinite(s0))) {
        skew = NAN;
        return NAN;
    }
    const int j0 = (int)floor(32 * log2(s0));
    double e1 = -1, e2 = -1;
    int j1 = j0, j2 = j0;
    for (int j = j0 - 96; j <= j0 + 96; j++) {
        const double e = murphy_at(cpoly, exp2(j / 32.0), alpha_f, alpha_g, C);
        if (e > e1) {
            e2 = e1;
            j2 = j1;
            e1 = e;
            j1 = j;
        } else if (e > e2) {
            e2 = e;
            j2 = j;
        }
    }
    double best = e1;
    skew = exp2(j1 / 32.0);
    for (int jj : {j1, j2})
        for (int k = -7; k <= 7; k++) {
            if (!k)
                continue;
            const double s = exp2((8 * jj + k) / 256.0);
            const double e = murphy_at(cpoly, s, alpha_f, alpha_g, C);
            if (e > best) {
                best = e;
                skew = s;
            }
        }
    return best;
}

/* f(x + t), g(x + t) in place */
static void translate_pair(cado_poly cpoly, long t)
{
    cxx_mpz tt;
    mpz_set_si(tt, t);
    mpz_poly_translation(cpoly->pols[ALG_SIDE], cpoly->pols[ALG_SIDE], tt);
    mpz_poly_translation(cpoly->pols[RAT_SIDE], cpoly->pols[RAT_SIDE], tt);
}

struct SearchPoint { long t; double z, value; };
struct SearchResult {
    SearchPoint best, starts[2];
    double initial;
    int calls = 0;
    bool limited = false;
};

// The coarse starts and each fixed-center neighborhood are independent. Admit
// uncached points in scalar order before submitting them together; cached points
// remain usable even when a preceding point exhausts the budget.
template <typename BatchObjective>
static SearchResult pattern_search_batched(long scale, int maxeval, BatchObjective objective)
{
    SearchResult out{};
    const long radius = scale * 4;
    const long mint = std::max(1L, scale / 2048);
    const double mins = 1.0 / 512;
    std::map<std::pair<long, double>, double> cache;
    auto score = [&](SearchPoint *points, size_t n) {
        std::vector<SearchPoint> pending;
        pending.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const auto &p = points[i];
            if (p.t < -radius || p.t > radius || fabs(p.z) > 2) continue;
            const auto key = std::make_pair(p.t, p.z);
            if (cache.find(key) != cache.end()) continue;
            if (out.calls + int(pending.size()) >= maxeval) {
                out.limited = true;
                continue;
            }
            // Reserve the key so a duplicate in this batch uses one evaluation.
            cache[key] = NAN;
            pending.push_back(p);
        }
        if (!pending.empty()) {
            std::vector<double> values(pending.size());
            objective(pending.data(), pending.size(), values.data());
            out.calls += int(pending.size());
            for (size_t i = 0; i < pending.size(); ++i)
                cache[{pending[i].t, pending[i].z}] = std::isfinite(values[i]) ? values[i] : -1.0;
        }
        for (size_t i = 0; i < n; ++i) {
            const auto it = cache.find({points[i].t, points[i].z});
            points[i].value = it == cache.end() ? -1.0 : it->second;
        }
    };
    std::vector<SearchPoint> starts{{0, 0, 0}};
    // A small broad grid helps cross a shallow dip before local refinement.
    for (int a : {-4, -2, -1, 0, 1, 2, 4})
        for (double z : {-1.0, 0.0, 1.0})
            if (a || z != 0) // the unchanged polynomial is already present
                starts.push_back({a * scale, z, 0});
    score(starts.data(), starts.size());
    SearchPoint best = starts[0];
    out.initial = best.value;
    std::stable_sort(starts.begin(), starts.end(), [](const SearchPoint &a, const SearchPoint &b) {
        return a.value > b.value;
    });
    // Two spatially separated starts, sharing a strict budget/cache. After the
    // first local search, exclude coarse points within two grid units of either
    // its start or its endpoint. This avoids spending the second search on the
    // adjacent shoulder of the same peak; it cannot guarantee distinct basins.
    SearchPoint first_end = starts[0];
    for (size_t k = 0; k < 2; k++) {
        SearchPoint start = starts[0];
        if (k) {
            auto separated = [&](const SearchPoint &a, const SearchPoint &b) {
                const double x = (double)(a.t - b.t) / scale, z = a.z - b.z;
                return x*x + z*z >= 4;
            };
            const auto it = std::find_if(starts.begin(), starts.end(), [&](const SearchPoint &p) {
                return separated(p, starts[0]) && separated(p, first_end);
            });
            if (it == starts.end()) break;
            start = *it;
        }
        SearchPoint cur = out.starts[k] = start;
        long dt = scale;
        double dz = 0.25;
        // Cached evaluations remain usable at the budget. Mark limited only if
        // an uncached, in-bounds evaluation is actually refused.
        while (!out.limited) {
            SearchPoint next = cur;
            std::array<SearchPoint, 8> neighbors;
            size_t count = 0;
            for (int a = -1; a <= 1; a++)
                for (int b = -1; b <= 1; b++) {
                    if (!a && !b) continue;
                    neighbors[count++] = {cur.t + a * dt, cur.z + b * dz, 0};
                }
            score(neighbors.data(), count);
            for (const auto &q : neighbors)
                if (q.value > next.value) next = q;
            if (next.value > cur.value)
                cur = next;
            else {
                if (dt <= mint && dz <= mins) break;
                dt = std::max(mint, dt / 2);
                dz = std::max(mins, dz / 2);
            }
        }
        if (cur.value > best.value) best = cur;
        if (!k) first_end = cur;
    }
    out.best = best;
    return out;
}

template <typename Objective>
static SearchResult pattern_search(long scale, int maxeval, Objective objective)
{
    return pattern_search_batched(scale, maxeval, [&](const SearchPoint *points, size_t n, double *values) {
        for (size_t i = 0; i < n; ++i) values[i] = objective(points[i].t, points[i].z);
    });
}

struct Refinement {
    double start_skew = 0, skew = 0, train0 = 0, train1 = 0, eval0 = 0, eval1 = 0, e0 = 0, e1 = 0;
    long t = 0;
    int calls = 0;
    bool limited = false;
};

/* Shared search/validation orchestration. initial_skew > 0 preserves a declared
 * control; zero selects the MurphyE grid optimum. evaluate(s) returns {objective,
 * MurphyE}, allowing the lattice caller to validate both without duplicating the
 * search. objective(s0, points, n, values) evaluates translations and log2-skew
 * offsets relative to p/s0, leaving p unchanged. Leaves p at the exact proposal;
 * selection is the caller's policy.
 */
template <typename BatchObjective, typename Validation>
static Refinement refine_pair_batched(cado_poly p, double af, double ag, const Ctx &train,
                              double initial_skew, int maxeval, long trans,
                              BatchObjective objective, Validation evaluate)
{
    Refinement out;
    double s0 = initial_skew;
    if (!(s0 > 0 && std::isfinite(s0)))
        best_skew(p, af, ag, train, s0);
    if (!(s0 > 0 && std::isfinite(s0)) || s0 > (double)std::numeric_limits<long>::max() / 32)
        return out;
    out.start_skew = s0;
    const long scale = trans ? trans : std::max(1L, (long)llround(s0 / 2));
    const auto search = pattern_search_batched(scale, maxeval, [&](const SearchPoint *points, size_t n, double *values) {
        objective(s0, points, n, values);
    });
    out.train0 = search.initial;
    out.train1 = search.best.value;
    out.calls = search.calls;
    out.limited = search.limited;
    out.t = search.best.t;
    out.skew = s0 * exp2(search.best.z);
    const auto control = evaluate(s0);
    out.eval0 = control[0];
    out.e0 = control[1];
    translate_pair(p, out.t);
    const auto proposal = evaluate(out.skew);
    out.eval1 = proposal[0];
    out.e1 = proposal[1];
    return out;
}

// CPU callers retain scalar objective evaluation and exact GMP translations.
// A device caller can instead prepare the same inputs and submit one batch.
template <typename Objective, typename Validation>
static Refinement refine_pair(cado_poly p, double af, double ag, const Ctx &train,
                              double initial_skew, int maxeval, long trans,
                              Objective objective, Validation evaluate)
{
    return refine_pair_batched(p, af, ag, train, initial_skew, maxeval, trans,
        [&](double s0, const SearchPoint *points, size_t n, double *values) {
            for (size_t i = 0; i < n; ++i) {
                translate_pair(p, points[i].t);
                values[i] = objective(s0 * exp2(points[i].z));
                translate_pair(p, -points[i].t);
            }
        }, evaluate);
}

static inline bool proposal_improves(const Refinement &r, double min_gain = 0)
{
    return std::isfinite(r.train0) && r.train0 > 0 && std::isfinite(r.train1) && r.train1 > r.train0 &&
           std::isfinite(r.eval0) && r.eval0 > 0 && std::isfinite(r.eval1) && r.eval1 > r.eval0 * (1 + min_gain);
}

} // namespace s23_murphy
