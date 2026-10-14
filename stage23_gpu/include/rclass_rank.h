/* Experimental class proposals from conditional small-prime root scores.
 * A class fixes u,v mod gcd(M,q); average the remaining lifts of each full
 * prime-power table, including its projective contribution. Primes coprime to
 * M have a class-independent average and do not affect the ordering.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <vector>
#include "rclass.h"
#include "rsieve.h"

namespace s23 {

struct RsClassMean {
    uint32_t period;
    std::vector<double> roots, content;
};

inline std::vector<RsClassMean> rs_class_means(const RsSeed &s, const float *pt, uint32_t modulus)
{
    std::vector<RsClassMean> means;
    for (int i = 0; i < RS_NPRIMES; i++) {
        const auto &prime = s.pr[i];
        const uint32_t q = prime.qmax, d = std::gcd(q, modulus);
        if (d == 1)
            continue;
        RsClassMean mean{d, std::vector<double>(d * d), std::vector<double>(d * d)};
        std::vector<float> table(q);
        for (uint32_t u = 0; u < q; u++) {
            rs_prime_table(table.data(), s, pt, i, u);
            for (uint32_t v = 0; v < q; v++)
                mean.roots[(u % d) * d + v % d] += table[v];
        }
        for (double &x : mean.roots)
            x /= (q / d) * (q / d);
        // A common factor fixed by this class must be removed in the ranking
        // estimate just as it is removed from exported polynomials. This is the
        // guaranteed part (up to d); extra content remains for exact reranking.
        for (uint32_t u = 0; u < d; u++)
            for (uint32_t v = 0; v < d; v++) {
                for (uint32_t pe = prime.p; pe <= d; pe *= prime.p) {
                    uint32_t f[6];
                    for (int k = 0; k < 6; k++)
                        f[k] = s.f[i][k] % pe;
                    f[0] = (f[0] + v * s.g[i][0]) % pe;
                    f[1] = (f[1] + u * s.g[i][0] + v * s.g[i][1]) % pe;
                    f[2] = (f[2] + u * s.g[i][1]) % pe;
                    bool content = true;
                    for (auto x : f)
                        content &= x == 0;
                    if (!content)
                        break;
                    mean.content[u * d + v] += log((double)prime.p);
                }
            }
        means.push_back(std::move(mean));
    }
    return means;
}

struct RsClassProposal {
    RsClass cell;
    double roots = 0, content = 0, size = INFINITY;
    uint32_t exploration = UINT32_MAX;
    bool selected_exploration = false;
    double key(double aw) const { return size - content - aw * (roots - content); }
};

inline RsClassProposal rs_class_proposal(const std::vector<RsClassMean> &means, uint32_t m, uint32_t u, uint32_t v)
{
    RsClassProposal c{{m, u, v}};
    for (const auto &mean : means) {
        const uint32_t j = (u % mean.period) * mean.period + v % mean.period;
        c.roots += mean.roots[j];
        c.content += mean.content[j];
    }
    return c;
}

/* A bounded promising pool plus deterministic uniform proposals for exploration.
 * M <= 4096 bounds enumeration to 16M classes; memory does not grow as M^2. */
inline std::vector<RsClassProposal> rs_class_pool(const RsSeed &s, const float *pt, uint32_t m, int count)
{
    const auto means = rs_class_means(s, pt, m);
    const size_t pool_size = std::min<size_t>((uint64_t)m * m, 4 * count);
    const auto better = [](const RsClassProposal &a, const RsClassProposal &b) {
        const double x = a.roots - a.content, y = b.roots - b.content;
        return x > y || (x == y && (a.cell.u < b.cell.u || (a.cell.u == b.cell.u && a.cell.v < b.cell.v)));
    };
    std::priority_queue<RsClassProposal, std::vector<RsClassProposal>, decltype(better)> heap(better);
    for (uint32_t u = 0; u < m; u++)
        for (uint32_t v = 0; v < m; v++) {
            auto c = rs_class_proposal(means, m, u, v);
            if (heap.size() < pool_size)
                heap.push(c);
            else if (better(c, heap.top())) {
                heap.pop();
                heap.push(c);
            }
        }
    std::map<uint64_t, RsClassProposal> pool;
    while (!heap.empty()) {
        const auto c = heap.top();
        pool[(uint64_t)c.cell.u * m + c.cell.v] = c;
        heap.pop();
    }
    uint64_t state = 0x9e3779b97f4a7c15ull;
    for (int i = 0; i < RS_NPRIMES; i++)
        for (int k = 0; k < 6; k++)
            state = (state ^ s.f[i][k]) * 1099511628211ull;
    if (!state)
        state = 1;
    std::set<uint64_t> drawn;
    const size_t explore_size = std::min<size_t>((uint64_t)m * m, std::max(16, count));
    while (drawn.size() < explore_size) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        const uint64_t id = state % ((uint64_t)m * m);
        if (!drawn.insert(id).second)
            continue;
        auto it = pool.emplace(id, rs_class_proposal(means, m, id / m, id % m)).first;
        it->second.exploration = (uint32_t)drawn.size() - 1;
    }
    std::vector<RsClassProposal> out;
    for (const auto &entry : pool)
        out.push_back(entry.second);
    return out;
}

/* Size estimates are filled by the caller from the actual class's sampled u
 * lines and nearby admissible v coordinates. Preserve an explicit exploration
 * allowance outside the leading size/root ranking; skip failed size probes. */
inline std::vector<RsClassProposal> rs_choose_classes(std::vector<RsClassProposal> pool, int count, int explore, double aw)
{
    pool.erase(std::remove_if(pool.begin(), pool.end(), [](const auto &c) { return !std::isfinite(c.size); }), pool.end());
    std::sort(pool.begin(), pool.end(), [=](const auto &a, const auto &b) {
        const double x = a.key(aw), y = b.key(aw);
        return x < y || (x == y && (a.cell.u < b.cell.u || (a.cell.u == b.cell.u && a.cell.v < b.cell.v)));
    });
    const size_t leading = std::min<size_t>(pool.size(), count - explore);
    std::vector<RsClassProposal> out(pool.begin(), pool.begin() + leading);
    std::vector<RsClassProposal> tail(pool.begin() + leading, pool.end());
    std::stable_sort(tail.begin(), tail.end(), [](const auto &a, const auto &b) { return a.exploration < b.exploration; });
    for (auto c : tail)
        if (out.size() < (size_t)count && c.exploration != UINT32_MAX) {
            c.selected_exploration = true;
            out.push_back(c);
        }
    for (const auto &c : pool) {
        if (out.size() >= (size_t)count)
            break;
        if (std::none_of(out.begin(), out.end(), [&](const auto &x) { return x.cell.u == c.cell.u && x.cell.v == c.cell.v; }))
            out.push_back(c);
    }
    return out;
}

} // namespace s23
