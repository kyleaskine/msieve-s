/* Observe the class-proposal cutoff; never add the reference to the pool.
 * Compile against the run's frozen headers. Output one JSON object per seed.
 */
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <set>
#include "rclass_rank.h"
#include "rhost.h"

using namespace s23;

static bool integer(const char *s, int64_t &n)
{
    char *end;
    errno = 0;
    n = strtoll(s, &end, 10);
    return !errno && end != s && !*end;
}

int main(int argc, char **argv)
{
    int64_t modulus, u, v;
    if (argc != 5 || !integer(argv[2], modulus) || !integer(argv[3], u) || !integer(argv[4], v) ||
        modulus < 2 || modulus > 4096)
        return 2;
    FILE *file = fopen(argv[1], "r");
    if (!file)
        return 2;
    cio_poly poly;
    cio_init(&poly);
    const bool read = cio_read_next(file, &poly) == 1;
    fclose(file);
    RsSeed seed;
    RsizePoly<RS_LP> size;
    if (!read || !rs_seed_and_size(seed, size, poly)) {
        cio_clear(&poly);
        return 2;
    }
    cio_clear(&poly);
    std::vector<float> pt;
    rs_proj_tables(seed, pt);
    const uint32_t m = modulus, ru = rs_residue(u, m), rv = rs_residue(v, m);
    const auto means = rs_class_means(seed, pt.data(), m);
    const auto target = rs_class_proposal(means, m, ru, rv);
    const double value = target.roots - target.content;
    uint64_t greater = 0, equal = 0, earlier_ties = 0, row_greater = 0, row_earlier_ties = 0;
    double best = -INFINITY;
    // Same double operations and tie ordering as rs_class_pool. Counts include
    // the target itself in equal; the observed rank is one-based.
    for (uint32_t cu = 0; cu < m; cu++)
        for (uint32_t cv = 0; cv < m; cv++) {
            const auto proposal = rs_class_proposal(means, m, cu, cv);
            const double x = proposal.roots - proposal.content;
            best = std::max(best, x);
            greater += x > value;
            equal += x == value;
            earlier_ties += x == value && (cu < ru || (cu == ru && cv < rv));
            row_greater += cu == ru && x > value;
            row_earlier_ties += cu == ru && x == value && cv < rv;
        }
    printf("{\"modulus\":%u,\"u\":%u,\"v\":%u,\"target_roots_minus_content\":%.17g,"
           "\"best_roots_minus_content\":%.17g,\"strictly_better_classes\":%llu,"
           "\"equal_score_classes\":%llu,\"earlier_ties\":%llu,\"root_proposal_rank\":%llu,"
           "\"rank_within_u\":%llu,\"pools\":[",
           m, ru, rv, value, best, (unsigned long long)greater, (unsigned long long)equal,
           (unsigned long long)earlier_ties, (unsigned long long)(1 + greater + earlier_ties),
           (unsigned long long)(1 + row_greater + row_earlier_ties));
    for (int count : {16, 32, 64}) {
        const auto pool = rs_class_pool(seed, pt.data(), m, count);
        bool proposed = false;
        std::set<uint32_t> ranked_u;
        for (const auto &p : pool) {
            proposed |= p.cell.u == ru && p.cell.v == rv;
            if (p.exploration == UINT32_MAX)
                ranked_u.insert(p.cell.u);
        }
        printf("%s{\"selected_count\":%d,\"pool_size\":%zu,\"proposed\":%s,"
               "\"nonexploration_distinct_u\":%zu}", count == 16 ? "" : ",", count, pool.size(),
               proposed ? "true" : "false", ranked_u.size());
    }
    printf("]}\n");
}
