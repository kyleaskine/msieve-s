/* The root sieve's per-block choice (include/rblock.h, cpu_block; the GPU kernel makes
 * the same choice, checked against it by s23_ropt runs) vs an exhaustive sort.
 *
 *   test_rselect POLYFILE ...
 *
 * For each polynomial: blocks on random lines and v, long (2^18 cells, minus a little: 4
 * of cpu_block's chunks) and short (1-300 cells, fewer cells than k on some), with and
 * without a lognorm slope, aw 1 or 1.3 (or 0 on every sixth block: every key ties, so
 * the first k cells must win), k from 1 to K_MAX. The kept cells must be the exhaustive top k by (key desc, v asc), in
 * that order, with the same scores, and slots past the block's cells must be empty.
 * The old choice (each of 256 thread classes' best, then the best k) failed this on a
 * third of long blocks.
 */
#include <stdio.h>
#include <numeric>
#include "rseed.h"
#include "rblock.h"

using namespace s23;

int main(int argc, char **argv)
{
    unsigned long long rng = 0x2545f4914f6cdd1dull;
    auto next = [&rng]() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
    };
    long blocks = 0, bad = 0, ties = 0;
    for (int a = 1; a < argc; a++) {
        FILE *fh = fopen(argv[a], "r");
        if (!fh) {
            perror(argv[a]);
            return 2;
        }
        cio_poly P;
        cio_init(&P);
        if (cio_read_next(fh, &P) != 1) {
            fprintf(stderr, "%s: no polynomial\n", argv[a]);
            return 2;
        }
        fclose(fh);
        RsSeed S;
        if (!rs_seed_from_poly(S, P)) {
            fprintf(stderr, "%s: not degree 5\n", argv[a]);
            return 2;
        }
        std::vector<float> PT, T(S.ntab), acc(1 << 16), sc, key;
        rs_proj_tables(S, PT);
        std::vector<uint32_t> ord;
        long fbad = 0;
        for (int b = 0; b < 24; b++) {
            const int64_t u = (int64_t)(next() % 201) - 100;
            const int64_t v0 = (int64_t)(next() % 4000000000ull) - 2000000000;
            const uint32_t len = b % 3 == 0 ? 1 + (uint32_t)(next() % 300) : (1u << 18) - (uint32_t)(next() % 1000);
            const float aw = b % 6 == 5 ? 0.0f : b % 2 ? 1.0f : 1.3f;
            const float dL = b % 4 < 2 ? 0.0f : (float)((double)(next() % 2000) * 1e-9 - 1e-6);
            const int k = 1 + (int)(next() % K_MAX);
            const Blk B = {u, v0, len, 55.0f, dL};
            Hit got[K_MAX];
            cpu_block(S, PT.data(), B, k, aw, got, T, acc);
            /* exhaustive: every cell's score from the line's tables, keys, a full sort */
            rs_line_tables(T.data(), S, PT.data(), u);
            sc.assign(len, 0.0f);
            for (int i = 0; i < RS_NPRIMES; i++) {
                const int64_t q = S.pr[i].qmax;
                const float *Ti = T.data() + S.pr[i].off;
                uint32_t j = (uint32_t)(((v0 % q) + q) % q);
                for (uint32_t x = 0; x < len; x++) {
                    sc[x] += Ti[j];
                    if (++j == q)
                        j = 0;
                }
            }
            key.resize(len);
            for (uint32_t x = 0; x < len; x++)
                key[x] = aw * sc[x] - (B.L0 + B.dL * (float)x);
            ord.resize(len);
            std::iota(ord.begin(), ord.end(), 0u);
            std::stable_sort(ord.begin(), ord.end(), [&](uint32_t x, uint32_t y) { return key[x] > key[y]; });
            const int kk = std::min<int>(k, (int)len);
            bool ok = true;
            for (int m = 0; m < k; m++) {
                if (m < kk)
                    ok &= got[m].u == u && got[m].v == v0 + (int64_t)ord[m] && got[m].key == key[ord[m]] &&
                          got[m].score == sc[ord[m]];
                else
                    ok &= !(got[m].score > -INFINITY);
            }
            if (kk >= 1 && (uint32_t)kk < len && key[ord[kk - 1]] == key[ord[kk]])
                ties++; /* the k-th place was decided by v */
            fbad += !ok;
            blocks++;
        }
        printf("%s: 24 blocks, %ld differ from the exhaustive top k\n", argv[a], fbad);
        bad += fbad;
        cio_clear(&P);
    }
    printf("all: %ld blocks (%ld with a tie at the k-th place), %ld differ\n", blocks, ties, bad);
    return bad ? 1 : 0;
}
