/* Pin scalar search behavior before batching (commit 1b19a1c). Dyadic synthetic
 * objectives avoid platform libm differences. Exercise ties, two basins, invalid
 * values, boundaries, and budgets ending in the middle of a neighborhood.
 */
#include "cado.h"
#include "murphy_search.h"
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace s23_murphy;

static uint64_t bits(double x)
{
    uint64_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static std::vector<uint64_t> result_words(const SearchResult &r)
{
    std::vector<uint64_t> out{bits(r.initial), uint64_t(r.calls), uint64_t(r.limited)};
    for (auto p : {r.best, r.starts[0], r.starts[1]}) {
        out.push_back(uint64_t(p.t)); out.push_back(bits(p.z)); out.push_back(bits(p.value));
    }
    return out;
}

int main()
{
    uint64_t hash = UINT64_C(14695981039346656037);
    auto add = [&](uint64_t word) {
        for (int b = 0; b < 8; ++b) {
            hash ^= (word >> (8*b)) & 255;
            hash *= UINT64_C(1099511628211);
        }
    };
    int bad = 0, cases = 0;
    std::vector<int> budgets;
    for (int i = 0; i <= 40; ++i) budgets.push_back(i);
    for (int i : {64, 128, 384, 1000}) budgets.push_back(i);
    for (long scale : {1L, 128L}) for (int mode = 0; mode < 4; ++mode) for (int budget : budgets) {
        auto value = [&](long t, double z) {
            const double x = double(t) / scale;
            switch (mode) {
            case 0: return 10 + std::max(-0.125*x*x-z*z, 0.125-2*(x+2.25)*(x+2.25)-8*z*z);
            case 1: return 7.0; // stable ties, shared-cache overlap
            case 2: return 10 + x + z; // move to the translation/skew bounds
            default: return t < 0 ? NAN : (z > 0 ? INFINITY : 10-x*x-z*z);
            }
        };
        std::vector<std::pair<long, double>> serial, batched;
        const auto reference = pattern_search(scale, budget, [&](long t, double z) {
            serial.emplace_back(t, z);
            return value(t, z);
        });
        add(uint64_t(scale)); add(mode); add(budget);
        for (const auto &p : serial) { add(uint64_t(p.first)); add(bits(p.second)); }
        for (uint64_t w : result_words(reference)) add(w);
#ifndef S23_SEARCH_RECORD
        const auto result = pattern_search_batched(scale, budget,
            [&](const SearchPoint *p, size_t n, double *values) {
                bad += n == 0 || n > (batched.empty() ? 21U : 8U);
                for (size_t i = 0; i < n; ++i) {
                    batched.emplace_back(p[i].t, p[i].z);
                    values[i] = value(p[i].t, p[i].z);
                }
            });
        bad += serial != batched || result_words(reference) != result_words(result);
        bad += result.calls > budget || result.calls != int(batched.size());
#endif
        ++cases;
    }
    // S23_SEARCH_RECORD builds against the historical header to record the
    // oracle. Normal tests must match it, independently of the shared adapter.
#ifndef S23_SEARCH_RECORD
    bad += hash != UINT64_C(0x851ed84a01367c65);
#endif
    printf("Murphy search: %d cases, trace hash %016llx, %d failures\n",
           cases, (unsigned long long)hash, bad);
    return bad ? 1 : 0;
}
