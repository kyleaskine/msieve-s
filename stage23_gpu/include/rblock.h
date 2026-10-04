/* M2 root sieve: one block of work (a segment of one u-line), a kept cell, the order of
 * cells within a block, and the block's choice on the CPU (shared by tools/s23_ropt.cu,
 * whose GPU kernel makes the same choice, and tests/test_rselect.cpp). */
#pragma once

#include <stdint.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include "hd.h"
#include "rsieve.h"

namespace s23 {

struct Hit {
    float score, key;
    int64_t u, v;
};

/* one block of work: a segment of one line. Cells are chosen by
 * key = aw * score - (L0 + dL * offset), the proxy with the lognorm interpolated linearly
 * across the block (search: blocks lie between two knots; window: L0 = dL = 0) */
struct Blk {
    int64_t u, v0;
    uint32_t len;
    float L0, dL;
};

static const int K_MAX = 16; /* at most 32: the GPU keeps a warp's best k one per lane */

/* the sieve's order of cells within a block: higher key first, then lower offset */
S23_HD bool cell_before(float ka, uint32_t oa, float kb, uint32_t ob)
{
    return ka > kb || (ka == kb && oa < ob);
}

/* CPU: the GPU kernel's per-cell sums in the same prime order, and its choice: the
 * block's exact best k cells by cell_before (outputs past the block's cells: score -inf) */
inline void cpu_block(const RsSeed &S, const float *PT, const Blk &B, int k, float aw, Hit *out,
                      std::vector<float> &T, std::vector<float> &acc)
{
    rs_line_tables(T.data(), S, PT, B.u);
    float bk[K_MAX], ba[K_MAX];
    uint32_t bo[K_MAX];
    for (int m = 0; m < k; m++) {
        bk[m] = ba[m] = -INFINITY;
        bo[m] = UINT32_MAX;
    }
    for (uint32_t base = 0; base < B.len; base += (uint32_t)acc.size()) {
        const uint32_t n = std::min<uint32_t>((uint32_t)acc.size(), B.len - base);
        std::fill(acc.begin(), acc.begin() + n, 0.0f);
        for (int i = 0; i < RS_NPRIMES; i++) {
            const int64_t q = S.pr[i].qmax;
            const float *Ti = T.data() + S.pr[i].off;
            uint32_t idx = (uint32_t)((((B.v0 + (int64_t)base) % q) + q) % q);
            for (uint32_t c = 0; c < n; c++) {
                acc[c] += Ti[idx];
                if (++idx == q)
                    idx = 0;
            }
        }
        for (uint32_t c = 0; c < n; c++) {
            const uint32_t off = base + c;
            const float key = aw * acc[c] - (B.L0 + B.dL * (float)off);
            if (!(key > bk[k - 1])) /* offsets only grow, so a tie with the last loses */
                continue;
            int m = k - 1;
            for (; m > 0 && cell_before(key, off, bk[m - 1], bo[m - 1]); m--) {
                bk[m] = bk[m - 1];
                ba[m] = ba[m - 1];
                bo[m] = bo[m - 1];
            }
            bk[m] = key;
            ba[m] = acc[c];
            bo[m] = off;
        }
    }
    for (int m = 0; m < k; m++)
        out[m] = {ba[m], bk[m], B.u, bk[m] > -INFINITY ? B.v0 + (int64_t)bo[m] : 0};
}

} // namespace s23
