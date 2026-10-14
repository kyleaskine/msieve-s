/* Optional host-side discovery of distant u basins, before band construction.
 * A coarse sample is a local minimum in v/t, not a certified lower bound.
 * Keep the legacy scan's lines and add explicitly bounded probe/neighborhood work.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <utility>
#include <vector>
#include "rsize.h"

namespace s23 {

struct RsScanConfig {
    int64_t umax = 0;
    int step = 0;       // zero disables the scan
    int basins = 4;
    int radius = 128;   // extra dense lines on either side of each refined center
};

inline bool rs_scan_valid(const RsScanConfig &c)
{
    if (!c.step)
        return true;
    if (c.umax < 0 || c.umax > 1000000000 || c.step < 1 || c.basins < 1 || c.basins > 16 ||
        c.radius < 0 || c.radius > 4096)
        return false;
    const int64_t half = c.umax / c.step + (c.umax % c.step != 0);
    return 2 * half + 1 <= 4097 && (2LL * c.radius + 1) * c.basins <= 16384;
}

inline std::vector<int64_t> rs_scan_grid(const RsScanConfig &c)
{
    std::vector<int64_t> out;
    if (!c.step || !rs_scan_valid(c))
        return out;
    out.push_back(0);
    for (int64_t u = c.step; u < c.umax; u += c.step) {
        out.push_back(-u);
        out.push_back(u);
    }
    if (c.umax) {
        out.push_back(-c.umax);
        out.push_back(c.umax);
    }
    std::sort(out.begin(), out.end());
    return out;
}

struct RsScanStats {
    size_t samples = 0, added = 0, evaluations = 0, nonfinite = 0;
    std::vector<int64_t> centers;
};

/* Evaluate(u, v_start, t_start) returns a freshly initialized line minimum.
 * This injection also permits tests with known synthetic landscapes. */
template <class Evaluate>
RsScanStats rs_scan_basins(std::map<int64_t, RsLine> &lines, const RsScanConfig &c, Evaluate evaluate)
{
    RsScanStats stats;
    const auto grid = rs_scan_grid(c);
    if (grid.empty())
        return stats;
    const size_t old_size = lines.size();
    const auto probe = [&](int64_t u) -> const RsLine & {
        auto found = lines.find(u);
        if (found != lines.end())
            return found->second;
        // Use the nearest finite adjacent cached line, plus the independent zero
        // start. Neither start guarantees the global minimum over v/translation.
        const RsLine *near = nullptr;
        auto right = lines.lower_bound(u);
        if (right != lines.end() && std::isfinite(right->second.L_min))
            near = &right->second;
        if (right != lines.begin()) {
            const auto &left = std::prev(right)->second;
            if (std::isfinite(left.L_min) && (!near || u - left.u <= near->u - u))
                near = &left;
        }
        RsLine best = evaluate(u, near ? near->v_min : 0, near ? near->t_min : 0);
        stats.evaluations++;
        if (near && (near->v_min || near->t_min)) {
            RsLine other = evaluate(u, 0, 0);
            stats.evaluations++;
            if (std::isfinite(other.L_min) && (!std::isfinite(best.L_min) || other.L_min < best.L_min))
                best = std::move(other);
        }
        best.u = u;
        if (!std::isfinite(best.L_min)) {
            best.L_min = INFINITY;
            stats.nonfinite++;
        }
        return lines.emplace(u, std::move(best)).first->second;
    };
    for (int64_t u : grid)
        probe(u);
    stats.samples = grid.size();

    // Local coarse minima include range endpoints. Deterministic order on ties.
    // A flat plateau gets a representative first, then spatially separated
    // representatives up to the explicitly requested number of basins.
    std::vector<int64_t> candidates;
    const auto value_at = [&](int64_t u) {
        const double x = lines.at(u).L_min;
        return std::isfinite(x) ? x : INFINITY;
    };
    for (size_t i = 0; i < grid.size(); i++) {
        const double value = value_at(grid[i]);
        if (std::isfinite(value) && (i == 0 || value <= value_at(grid[i - 1])) &&
            (i + 1 == grid.size() || value <= value_at(grid[i + 1])))
            candidates.push_back(grid[i]);
    }
    std::sort(candidates.begin(), candidates.end(), [&](int64_t a, int64_t b) {
        const double x = lines.at(a).L_min, y = lines.at(b).L_min;
        return x < y || (x == y && a < b);
    });
    std::vector<int64_t> chosen;
    for (int64_t u : candidates) {
        bool overlaps = false;
        for (int64_t v : chosen)
            overlaps |= std::llabs(u - v) <= std::max<int64_t>(c.step, 2LL * c.radius);
        if (!overlaps)
            chosen.push_back(u);
        if (chosen.size() == (size_t)c.basins)
            break;
    }
    for (int64_t u : chosen) {
        const int64_t lo = std::max(-c.umax, u - c.step), hi = std::min(c.umax, u + c.step);
        // Pattern search within this coarse bracket. Bound the total probes too,
        // since a narrow missed minimum must not turn into an unbounded scan.
        int evaluations = 0;
        for (int64_t step = std::max(1, c.step / 2); step >= 1 && evaluations < 128; step /= 2) {
            bool moved;
            do {
                moved = false;
                int64_t best_u = u;
                for (int dir : {-1, 1}) {
                    const int64_t next = u + dir * step;
                    if (next < lo || next > hi || evaluations >= 128)
                        continue;
                    evaluations++;
                    if (probe(next).L_min < probe(best_u).L_min)
                        best_u = next;
                }
                if (best_u != u) {
                    u = best_u;
                    moved = true;
                }
            } while (moved && evaluations < 128);
        }
        if (std::find(stats.centers.begin(), stats.centers.end(), u) == stats.centers.end())
            stats.centers.push_back(u);
        const int64_t first = std::max(-c.umax, u - c.radius), last = std::min(c.umax, u + c.radius);
        // Outwards from the center supplies a nearby evaluated warm start.
        for (int64_t delta = 0; delta <= c.radius; delta++) {
            if (u - delta >= first)
                probe(u - delta);
            if (delta && u + delta <= last)
                probe(u + delta);
        }
    }
    stats.added = lines.size() - old_size;
    return stats;
}

} // namespace s23
