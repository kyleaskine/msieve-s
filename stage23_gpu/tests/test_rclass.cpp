/* Residue-class geometry, exhaustive top-k, and all-class/dense equivalence.
 * Original-coordinate brute counts test p | M, conditional prime-power lifts,
 * projective roots and content cells on the same real fixtures as test_rsieve.
 */
#include <cassert>
#include <cstdio>
#include <map>
#include <numeric>
#include <set>
#include "rseed.h"
#include "rsegment.h"

using namespace s23;

static void geometry()
{
    for (uint32_t m : {1, 2, 3, 6, 16, 2520, 65536})
        for (uint32_t r : {0u, m / 2, m - 1})
            for (int64_t lo : {-35, -10, -1, 0, 1, 10})
                for (int64_t hi : {-36, -10, -1, 0, 1, 41}) {
                    std::vector<int64_t> want;
                    for (int64_t v = lo; v <= hi; v++)
                        if (rs_residue(v, m) == r)
                            want.push_back(v);
                    const auto p = rs_progression(lo, hi, m, r);
                    assert(p.count == want.size());
                    if (p.count) {
                        assert(p.first == want.front() && p.last == want.back());
                        for (uint64_t n : {0, 1, 2, 7, 100}) {
                            const auto q = rs_trim_progression(p, m, -3, n);
                            assert(q.count == std::min(n, p.count));
                            if (q.count) {
                                assert(q.first >= p.first && q.last <= p.last);
                                assert(rs_residue(q.first, m) == r && rs_residue(q.last, m) == r);
                                assert((uint64_t)(q.last - q.first) / m + 1 == q.count);
                            }
                        }
                    }
                }
    // Endpoint clipping and cap trimming must not overflow signed coordinates.
    for (int64_t lo : {INT64_MIN, INT64_MAX - 1000}) {
        const auto p = rs_progression(lo, lo + 1000, 252, 19);
        assert(p.count > 0 && p.first >= lo && p.last <= lo + 1000);
        const auto q = rs_trim_progression(p, 252, lo, 2);
        assert(q.count == 2 && q.first == p.first);
    }
    auto p = rs_progression(INT64_MIN, INT64_MAX, 2, 0);
    assert(p.count == (1ull << 63));
    p = rs_trim_progression(p, 2, INT64_MAX, 2);
    assert(p.first == INT64_MAX - 3 && p.last == INT64_MAX - 1);

    // Knots use original coordinates; clipping and block starts adjust the
    // intercept, whereas the per-selected-cell slope multiplies by the stride.
    std::vector<Seg> segs;
    class_segment(segs, {7, -23, 71, 10., .125}, {6, 1, 2});
    BlockGen gen(segs, 2);
    Blk b;
    uint64_t count = 0;
    while (gen.next(b)) {
        assert(b.step == 6 && b.dL == .75f);
        assert(b.L0 == 10. + .125 * (b.v0 + 23));
        for (uint32_t i = 0; i < b.len; i++) {
            assert(rs_residue(b.v0 + i * 6, 6) == 2);
            count++;
        }
    }
    assert(count == seg_cells(segs[0]) && gen.made == seg_blocks(segs[0], 2));
}

int main(int argc, char **argv)
{
    geometry();
    size_t all_cells = 0, all_blocks = 0, brute = 0;
    for (int a = 1; a < argc; a++) {
        FILE *file = fopen(argv[a], "r");
        assert(file);
        cio_poly p;
        cio_init(&p);
        assert(cio_read_next(file, &p) == 1);
        fclose(file);
        RsSeed s;
        assert(rs_seed_from_poly(s, p));
        for (int projective : {0, 1}) {
            s.proj = projective;
            std::vector<float> pt, table(s.ntab), scratch(s.ntab), acc(1 << 16);
            rs_proj_tables(s, pt);
            // Every class together is exactly the dense window, once per cell.
            std::map<std::pair<int64_t, int64_t>, float> dense;
            for (int64_t u = -5; u <= 7; u++) {
                rs_line_tables(table.data(), s, pt.data(), u);
                for (int64_t v = -35; v <= 41; v++)
                    dense[{u, v}] = rs_cell_score(table.data(), s, v);
            }
            for (uint32_t m : {1, 2, 3, 6}) {
                std::map<std::pair<int64_t, int64_t>, float> classes;
                for (uint32_t ru = 0; ru < m; ru++)
                    for (uint32_t rv = 0; rv < m; rv++) {
                        std::vector<Seg> segs;
                        for (int64_t u = -5; u <= 7; u++)
                            if (rs_residue(u, m) == ru)
                                class_segment(segs, {u, -35, 41, 0., 0.}, {m, ru, rv});
                        BlockGen gen(segs, 3);
                        Blk b;
                        while (gen.next(b)) {
                            Hit hits[K_MAX];
                            cpu_block(s, pt.data(), b, K_MAX, 1, hits, scratch, acc);
                            for (const auto &h : hits)
                                if (h.score > -INFINITY) {
                                    assert(rs_residue(h.u, m) == ru && rs_residue(h.v, m) == rv);
                                    assert(classes.emplace(std::make_pair(h.u, h.v), h.score).second);
                                }
                        }
                        all_blocks += gen.made;
                    }
                assert(classes == dense);
                all_cells += classes.size();
            }
            // Long blocks cross CPU chunks and GPU warp/chunk boundaries. Direct
            // scores in original coordinates supply an independent exhaustive top-k.
            for (uint32_t m : {1, 2, 8, 128, 2520, 65536})
                for (uint32_t len : {1, 7, 301, 66001}) {
                    const int64_t u = -987654321, v0 = -8765432100123;
                    const float aw = len == 7 ? 0.f : 1.3f;
                    const Blk b{u, v0, len, 55.f, len == 7 ? 0.f : -.0001f, m};
                    Hit got[K_MAX];
                    cpu_block(s, pt.data(), b, K_MAX, aw, got, scratch, acc);
                    rs_line_tables(table.data(), s, pt.data(), u);
                    std::vector<float> scores(len), keys(len);
                    std::vector<uint32_t> order(len);
                    std::iota(order.begin(), order.end(), 0u);
                    for (uint32_t j = 0; j < len; j++) {
                        scores[j] = rs_cell_score(table.data(), s, v0 + (int64_t)j * m);
                        keys[j] = aw * scores[j] - (b.L0 + b.dL * (float)j);
                    }
                    std::stable_sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) { return keys[x] > keys[y]; });
                    for (int j = 0; j < K_MAX; j++) {
                        if ((uint32_t)j >= len)
                            assert(got[j].score == -INFINITY);
                        else {
                            assert(got[j].v == v0 + (int64_t)order[j] * m && got[j].u == u);
                            assert(got[j].score == scores[order[j]] && got[j].key == keys[order[j]]);
                        }
                    }
                    for (uint32_t j : {0u, len / 2, len - 1}) {
                        assert(fabs(rs_cell_score_brute(s, u, v0 + (int64_t)j * m) - scores[j]) < 1e-4);
                        brute++;
                    }
                    all_cells += len;
                    all_blocks++;
                }
        }
        cio_clear(&p);
        printf("%s: class geometry, all-class/dense scores, exhaustive top-k, brute roots passed\n", argv[a]);
    }
    printf("residue sieve: %zu cells, %zu blocks, %zu brute root comparisons passed\n", all_cells, all_blocks, brute);
}
