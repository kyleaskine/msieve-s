/* Conditional class means vs independently counted roots, proposal selection,
 * guaranteed content factors, and mixed-class cap geometry. */
#include <cassert>
#include <cstdio>
#include "rclass_rank.h"
#include "rhost.h"

using namespace s23;

static double prime_brute(const RsSeed &s, int i, uint32_t u, uint32_t v)
{
    const auto &p = s.pr[i];
    double score = 0;
    for (uint32_t q = p.p; q <= p.qmax; q *= p.p) {
        unsigned roots = 0;
        for (uint32_t x = 0; x < q; x++) {
            uint64_t f = s.f[i][5] % q;
            for (int k = 4; k >= 0; k--)
                f = (f * x + s.f[i][k]) % q;
            const uint64_t g = (s.g[i][0] + (uint64_t)s.g[i][1] * x) % q;
            roots += (f + ((uint64_t)u * x + v) * g) % q == 0;
        }
        score += roots * log((double)p.p) / (q + q / p.p);
    }
    if (s.poff[i] != RS_NOPROJ)
        score += rs_cell_proj_brute(s, i, u, v);
    return score;
}

static void geometry_and_choice()
{
    RsClassRows rows;
    rows.modulus = 6;
    rows.rows[2] = {1, 4, 5};
    std::vector<int64_t> want;
    for (int64_t v = -37; v <= 41; v++)
        if (rs_residue(v, 6) == 1 || rs_residue(v, 6) == 4 || rs_residue(v, 6) == 5)
            want.push_back(v);
    assert(rows.count(-4, -37, 41) == want.size());
    assert(rows.count(-3, -37, 41) == 0);
    for (size_t i = 0; i < want.size(); i++)
        assert(rows.ordinal(-4, -37, 41, i) == want[i]);
    for (uint64_t n = 0; n <= want.size() + 1; n++)
        for (int64_t minimum : {-100, -37, -1, 0, 17, 41, 100}) {
            const auto p = rows.trim(-4, -37, 41, minimum, n);
            assert(p.count == std::min<uint64_t>(n, want.size()));
            assert(rows.count(-4, p.first, p.last) == p.count);
            if (p.count)
                assert(p.first >= want.front() && p.last <= want.back());
        }
    std::vector<RsClassProposal> pool(4);
    for (int i = 0; i < 4; i++)
        pool[i].cell = {6, 1, (uint32_t)i};
    pool[0].size = 100;
    pool[1].size = 50;
    pool[2].size = 60;
    pool[2].exploration = 0;
    pool[3].size = NAN;
    auto selected = rs_choose_classes(pool, 2, 1, 1.3);
    assert(selected.size() == 2 && selected[0].cell.v == 1 && selected[1].cell.v == 2);
    assert(!selected[0].selected_exploration && selected[1].selected_exploration);
    selected = rs_choose_classes(pool, 4, 1, 1.3);
    assert(selected.size() == 3);
    pool[0].size = 53;
    pool[0].roots = 5;
    pool[0].content = 4;
    assert(fabs(pool[0].key(1.3) - (49 - 1.3)) < 1e-12);
}

int main(int argc, char **argv)
{
    geometry_and_choice();
    for (int arg = 1; arg < argc; arg++) {
        FILE *file = fopen(argv[arg], "r");
        assert(file);
        cio_poly p, written;
        cio_init(&p);
        cio_init(&written);
        assert(cio_read_next(file, &p) == 1);
        fclose(file);
        RsSeed s;
        RsizePoly<RS_LP> r;
        assert(rs_seed_and_size(s, r, p));
        for (int projective : {0, 1}) {
            s.proj = projective;
            std::vector<float> pt;
            rs_proj_tables(s, pt);
            const auto means = rs_class_means(s, pt.data(), 30);
            size_t im = 0;
            for (int i = 0; i < RS_NPRIMES; i++) {
                const uint32_t d = std::gcd(30u, s.pr[i].qmax);
                if (d == 1)
                    continue;
                for (uint32_t residue : {0u, d - 1}) {
                    double sum = 0;
                    unsigned count = 0;
                    for (uint32_t u = residue; u < s.pr[i].qmax; u += d)
                        for (uint32_t v = residue; v < s.pr[i].qmax; v += d) {
                            sum += prime_brute(s, i, u, v);
                            count++;
                        }
                    assert(fabs(sum / count - means[im].roots[residue * d + residue]) < 1e-5);
                }
                im++;
            }
            for (uint32_t u = 0; u < 6; u++)
                for (uint32_t v = 0; v < 6; v++) {
                    unsigned long content;
                    assert(rs_written_poly(written, p, r, u, v, 0, content) && content);
                    const auto c = rs_class_proposal(means, 30, u, v);
                    assert(fabs(c.content - log((double)std::gcd(content, 30ul))) < 1e-12);
                }
            const auto pool = rs_class_pool(s, pt.data(), 12, 2);
            const auto again = rs_class_pool(s, pt.data(), 12, 2);
            assert(pool.size() == again.size() && pool.size() <= 24);
            std::set<std::pair<uint32_t, uint32_t>> ids;
            size_t explored = 0;
            for (size_t i = 0; i < pool.size(); i++) {
                assert(ids.emplace(pool[i].cell.u, pool[i].cell.v).second);
                assert(pool[i].cell.u == again[i].cell.u && pool[i].cell.v == again[i].cell.v &&
                       pool[i].exploration == again[i].exploration && pool[i].roots == again[i].roots);
                explored += pool[i].exploration != UINT32_MAX;
            }
            assert(explored == 16);
            const auto all_means = rs_class_means(s, pt.data(), 12);
            std::vector<RsClassProposal> exhaustive;
            for (uint32_t u = 0; u < 12; u++)
                for (uint32_t v = 0; v < 12; v++)
                    exhaustive.push_back(rs_class_proposal(all_means, 12, u, v));
            std::sort(exhaustive.begin(), exhaustive.end(), [](const auto &a, const auto &b) {
                const double x = a.roots - a.content, y = b.roots - b.content;
                return x > y || (x == y && (a.cell.u < b.cell.u || (a.cell.u == b.cell.u && a.cell.v < b.cell.v)));
            });
            for (size_t i = 0; i < 8; i++)
                assert(ids.count({exhaustive[i].cell.u, exhaustive[i].cell.v}));
        }
        cio_clear(&written);
        cio_clear(&p);
        printf("%s: conditional means/brute roots, content, deterministic proposals and exploration passed\n", argv[arg]);
    }
}
