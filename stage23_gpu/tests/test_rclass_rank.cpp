/* Conditional class means vs independently counted roots, proposal selection,
 * guaranteed content factors, and mixed-class cap geometry. */
#include <cassert>
#include <cstdio>
#include "rclass_rank.h"
#include "rhost.h"

using namespace s23;

static void size_proposals()
{
    for (int64_t umax : {0LL, 1LL, 32LL, 20000LL, 1000000000LL}) {
        const auto grid = rs_class_size_grid(umax);
        assert(grid.size() <= 129 && grid.front() == -umax && grid.back() == umax);
        assert(std::is_sorted(grid.begin(), grid.end()));
        assert(std::set<int64_t>(grid.begin(), grid.end()).size() == grid.size());
        assert(std::find(grid.begin(), grid.end(), 0) != grid.end());
    }
    const std::vector<std::pair<int64_t, double>> samples{{-23, 9}, {-8, 2}, {0, 5}, {7, NAN}, {19, 1}};
    for (uint32_t m = 2; m <= 30; ++m) {
        const auto hints = rs_class_size_hints(m, samples);
        std::vector<double> brute(m, INFINITY);
        for (int64_t u = -23; u <= 19; ++u) {
            double size = INFINITY;
            for (const auto &sample : samples)
                if (sample.first == u && std::isfinite(sample.second))
                    size = sample.second;
            for (size_t i = 1; i < samples.size(); ++i)
                if (u >= samples[i-1].first && u <= samples[i].first &&
                    std::isfinite(samples[i-1].second) && std::isfinite(samples[i].second)) {
                    double f = (double)(u-samples[i-1].first)/(samples[i].first-samples[i-1].first);
                    size = samples[i-1].second + f*(samples[i].second-samples[i-1].second);
                }
            brute[rs_residue(u,m)] = std::min(brute[rs_residue(u,m)],size);
        }
        for (uint32_t u = 0; u < m; ++u)
            assert(hints[u] == brute[u] || fabs(hints[u]-brute[u]) < 1e-12);
    }
    std::vector<RsClassProposal> pool;
    for (uint32_t u = 0; u < 6; ++u)
        for (uint32_t v = 0; v < 6; ++v) {
            RsClassProposal c{{6,u,v}};
            c.size = 10*u + v;
            pool.push_back(c);
        }
    const auto chosen = rs_choose_classes(pool,6,0,1.3,true);
    std::set<uint32_t> rows;
    for (const auto &c : chosen) rows.insert(c.cell.u);
    assert(chosen.size()==6 && rows.size()==3);
}

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
    size_proposals();
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
            // A row with poor root-only rank remains eligible when its size
            // is much better, and unrelated/nonfinite rows cannot dominate it.
            std::vector<double> hints(12, INFINITY);
            for (uint32_t u = 0; u < 6; ++u) hints[u] = 20*u;
            const auto sized = rs_class_pool(s, pt.data(), 12, 2, hints, 1.3);
            std::set<uint32_t> ranked_rows;
            for (const auto &c : sized)
                if (std::isfinite(c.size)) ranked_rows.insert(c.cell.u);
            assert(sized.size() <= 24);
            for (uint32_t u = 0; u < 4; ++u) assert(ranked_rows.count(u));
            // Full-enumeration oracle for both independent quotas. Checking
            // only row coverage lets a broken global heap hide behind the
            // one-per-row reserve. Include all expected random exploration IDs.
            std::vector<RsClassProposal> oracle;
            for (auto c : exhaustive) {
                if (!std::isfinite(hints[c.cell.u])) continue;
                c.size=hints[c.cell.u];
                oracle.push_back(c);
            }
            std::sort(oracle.begin(),oracle.end(),[](const auto &a,const auto &b) {
                const double x=a.size-1.3*a.roots+0.3*a.content;
                const double y=b.size-1.3*b.roots+0.3*b.content;
                return x<y || (x==y && (a.cell.u<b.cell.u || (a.cell.u==b.cell.u && a.cell.v<b.cell.v)));
            });
            std::set<std::pair<uint32_t,uint32_t>> expected,actual,reserved;
            for (size_t i=0;i<4;++i) expected.emplace(oracle[i].cell.u,oracle[i].cell.v);
            std::set<uint32_t> rows;
            for (const auto &c : oracle)
                if (rows.size()<4 && rows.insert(c.cell.u).second)
                    reserved.emplace(c.cell.u,c.cell.v);
            for (const auto &c : pool)
                if (c.exploration!=UINT32_MAX) reserved.emplace(c.cell.u,c.cell.v);
            assert(std::any_of(expected.begin(),expected.end(),[&](const auto &id){return !reserved.count(id);}));
            expected.insert(reserved.begin(),reserved.end());
            for (const auto &c : sized) actual.emplace(c.cell.u,c.cell.v);
            assert(actual==expected);
        }
        cio_clear(&written);
        cio_clear(&p);
        printf("%s: conditional means/brute roots, content, deterministic proposals and exploration passed\n", argv[arg]);
    }
}
