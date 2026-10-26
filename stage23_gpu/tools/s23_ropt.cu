/* s23_ropt: root-sieve experiments (M2), GPU or CPU from the same tables.
 *
 *   window: s23_ropt -poly FILE -u U0 U1 -v V0 V1 [common options]
 *   search: s23_ropt -poly FILE -search [-budget B] [-band B] [-maxcells C] [-aw W] [-patience P]
 *           [-umax U] [-maxlines M] [-out FILE] [-refine N] [-rerank N] [-plan] [common options]
 *           [-u-scan STEP [-u-basins N] [-u-radius R]]
 *           [-murphy -Bf BF -Bg BG -area A [-murphy-points 16000] [-murphy-eval-points 64000] [-maxeval 384]]
 *   cell:   s23_ropt -poly FILE -cell U V
 *   common: [-top N] [-k K] [-seg LOG2] [-noproj] [-cpu] [-dev N] [-t THREADS] [-check N] [-launch LOG2]
 *           [-plan] [-exactskew] [-oldsearch]
 *           [-residue M U V]  (window/search: (u,v) = (U,V) mod M)
 *           [-classes M K [-class-explore E]]  (search: automatic classes, shared cap)
 *           [-pool-out FILE | -pool-in FILE] (save/replay cells before exact refinement)
 *           [-alpha-screen N B] (search: screen N proxy cells with alpha through prime B)
 *           [-alpha-screen-exact-size] (also compute exact size before the screen cut)
 *           [-alpha-screen-gpu [-alpha-screen-check]] (CUDA exact-size screen; optional CPU audit)
 *           [-refine-gpu [-refine-check]] (full-bound CUDA alpha/translation refinement)
 *           [-murphy-gpu [-murphy-check]] (CUDA quadrature; requires -murphy)
 * Or: s23_ropt -batch MANIFEST (serial independent searches in one CUDA session;
 *     S23_ROPT_BATCH 1 header, then one tab-separated argv row per seed).
 *           [-trace-cell U V] (report the first cutoff losing this cell; never force retention)
 *
 * For the seed in FILE (CADO format, degree 5), scores cells (u, v) of
 * f_{u,v} = f + (u x + v) g by the alpha sieve of include/rsieve.h (p < 200: affine roots
 * to p^e <= 200, projective roots to p^(e+3); a higher score is a better alpha:
 * alpha_s = sum log p/(p-1) - score, the tail excluded; -noproj leaves the projective
 * roots out, as before 2026-10-04) and keeps the exact best K cells of every block of at
 * most 2^LOG2 cells of a line (default K 8, at most 16; LOG2 20), by the key below, ties
 * to the lower v.
 *
 * window: every cell of the box; prints the best N by score.
 * search: the size model of include/rsize.h first: from u = 0 outwards, each line's
 *   minimum lognorm after re-translation, and its band, the v where that lognorm is
 *   within B (default 0.5) of the best line minimum seen; each direction stops P (default
 *   20) lines after the line minima exceed the best + B + 0.3, or after M/2 (default M
 *   1000) lines within the budget, so a seed whose size is flat in u stays bounded.
 *   -maxcells C lowers the budget in steps of 0.25 (not below 0.5) until the bands hold
 *   at most C cells; if they still hold more, only the lines nearest the best line are
 *   kept, the last one trimmed around its minimum, so at most C cells are sieved. Then the
 *   sieve over the bands, ranked by the proxy E = lognorm + W alpha_s (lower is
 *   better; W default 1, about 1.3 fits MurphyE better), with lognorm interpolated between
 *   knots; blocks are split at the knots, so each block's cells are chosen by the same
 *   proxy. The best N (default 20) are recomputed exactly (integer translation) and
 *   printed, with the best cell of each line, and -out writes the best max(N, -refine)
 *   (-refine default 200) as CADO-format polynomials, translated, for scoring with CADO's MurphyE
 *   (stage23_bench/tools/score_polys.py). A cell whose polynomial has content (possible
 *   when p | the multiplier a and f = (linear) g mod p) is written with it divided out,
 *   as CADO's ropt does, and labelled; its proxy is not right, so such lattices are
 *   searched properly as their own seeds (f + (u0 x + v0) g)/p.
 * -u-scan STEP (search, experimental; default 0 disables): also sample the complete
 *   [-umax,umax] range at this spacing, with endpoints. Refine at most -u-basins N
 *   coarse local minima (default 4), then add dense neighborhoods of -u-radius R
 *   lines on either side (default 128). Newly probed lines use nearby and independent
 *   v/t starts. These are local minima, not certified bounds. Limit: 4097 samples,
 *   16 basins, radius 4096, 16384 total neighborhood lines, and 128 refinement
 *   probes per basin; umax <= 1e9.
 *   The ordinary scan and -maxlines are unchanged. -maxcells still applies to the
 *   merged lines: a lower new minimum can change which old cells survive the cap.
 *   Report discovery and the center lines retained after the cap separately.
 * -rerank N (search, default 4096): the best N by the proxy get the exact lognorm and CADO's
 *   alpha (get_alpha, affine and projective, primes up to the bound CADO's MurphyE uses,
 *   2000), both of the polynomial as written, content divided out. They are re-ranked by
 *   Ea = lognorm + W alpha (labelled "Ea", not "E"), and -out writes the best max(-top,
 *   -refine) of them; -rerank 0 ranks the best max(-top, -refine) by E, as before
 *   2026-10-06. alpha_s stops at p < 200 and p^e <= 200, and is off from CADO's alpha by
 *   -0.24 to +0.93 nats per cell (c208), as wide as the 200 cells kept: a cell scoring within
 *   2% of the best sat at rank 57 and fell out of the 200 as the budget grew. Ranked by
 *   CADO's alpha, the accurately best cell of 16 c208 searches is at rank 7 at worst (95 by
 *   alpha_s), and the lost cell is kept. About 0.5 s a search on 4 threads.
 *   The re-rank fixes only the cut to the written cells. Each block's best K are still chosen
 *   by E, so a cell that alpha_s misjudges can be lost inside its block first.
 * -alpha-screen N B (search, experimental): retain N by the original proxy, then
 *   rank them using interpolated lognorm plus W times CADO alpha through prime B,
 *   with both terms corrected for content. Fully refine only max(-rerank, -refine,
 *   -top) of these. N must be at least that count, and 2 <= B <= the full alpha bound.
 *   This host screening stage can recover cells outside the ordinary proxy pool,
 *   but cannot recover per-block losses. It adds a heuristic cut, so validate on
 *   independent seeds before enabling it. Pools are saved BEFORE this stage in the
 *   unchanged v1 format, allowing replay with different screening policies.
 *   -alpha-screen-exact-size replaces interpolated size by the usual integer-
 *   translation size search before this cut. Its size and translation are reused
 *   in full refinement; only alpha is recomputed to the full bound. This costs
 *   more at screening but protects candidates with pessimistic size interpolation.
 *   -alpha-screen-gpu moves that exact-size screen to CUDA (B <= 2000). It uses
 *   bounded fixed-width arithmetic, with host fallback, and rechecks near-cut
 *   scores on the host within an empirical 0.125 guard. This is not a proof of
 *   CPU shortlist identity. Host checks retain their canonical translations;
 *   other finalists are refined from their saved starts. -alpha-screen-check
 *   compares every successful device score with CADO and selects by the host
 *   values on success; use it for validation, not timing.
 * -murphy (search, opt-in, requires -rerank > 0 and explicit Bf/Bg/area): optimize
 *   translation and skew for ALL max(-rerank, -refine, -top) retained proxy cells
 *   on the host, before the output cut. Search at K >= 16000, validate control and
 *   proposal at Keval >= 4*K, retain the better validation score, and sort by it.
 *   The shared bench search uses two starts, at most maxeval joint evaluations
 *   (in addition to the initial skew grid), and exact GMP translations. Output
 *   includes the chosen skew and MurphyE, control/proposal scores and search status.
 *   A scoring failure is reported and excluded; valid outputs are saved with
 *   exit 1. Cells invalid before scoring are skipped without causing exit 1.
 *   This does not recover cells outside the retained proxy pool or change the
 *   per-block proxy selection. -murphy-gpu evaluates the integral on CUDA, while
 *   retaining host search control and exact GMP translation. -murphy-check audits
 *   every integral against the scalar reference. Full-pool refinement is expensive;
 *   measure it before enabling it for breadth. Without -murphy, output is unchanged.
 * -band B (search): the bands at budget B (at most -budget) instead of -budget; the line scan
 *   still runs to -budget, so B = -budget - 0.25 k gives the same lines and cells as
 *   "# step B" of -plan with the same options. -maxcells then lowers it from B.
 * -plan stops before the sieve and prints the cell and block counts (and, for a search, the
 *   size model's lines, and the cells at each budget from -budget down in steps of 0.25,
 *   as "# step B: N cells, L lines", so a later pass can give a seed its next step with
 *   -band B).
 * Memory does not grow with the search: the blocks (one per at most 2^LOG2 cells of a
 *   line, at least one per knot interval) are generated a launch at a time from the lines'
 *   pieces. Before 2026-10-06 the whole list was built first (32 bytes a block, up to 2e8
 *   blocks on c208 seeds with a far, flat basin), and nine runs took 107 GB and crashed the
 *   machine.
 * -launch LOG2 (default 32): a GPU launch takes blocks until it holds about 2^LOG2 cells
 *   (about 0.1 s on a 5070), and at least four waves of blocks. Results do not depend on it:
 *   the kept cells are chosen by a total order. Long kernels made the Windows desktop lag
 *   under WSL2, and with -seg above about 22 a single block already takes long.
 * -exactskew: the size model's line scan, band edges and knots find every skewness by CADO's
 *   root isolation, as before 2026-10-10, instead of by Newton from the previous evaluation's
 *   (include/rsize.h, rs_skewness). The lognorms agree to about an ulp either way.
 * -oldsearch: the size model's line minima, band edges and knots search as before 2026-10-10
 *   (include/rsize.h, rs_fast_search). With both, the size model is the old one exactly.
 * -check N rescores N random cells of the sieve's output (as the GPU or CPU scored them)
 *   and every printed cell by brute-force root counting; -cell prints one cell's score
 *   both ways.
 *
 * GPU: one block per (line, segment) builds the line's 46 per-prime tables in shared
 * memory (about 19 KB), then each warp steps through 32 consecutive cells at a time,
 * 46 table lookups per cell. Each warp keeps its best K one per lane (a cell that beats
 * the warp's K-th is inserted with warp votes and shuffles; rare after the first chunks),
 * thread 0 merges the 8 warp lists, and the chosen cells' scores are summed again.
 * CPU: the same tables, the same order of additions and the same exact choice, so
 * results are bit-identical.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <atomic>
#include <map>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <thread>
#include <errno.h>
#include <limits.h>
#include <gmp.h>
#include "cado_io.h"
#include "gmp_bridge.h"
#include "rseed.h"
#include "rsize.h"
#include "rscan.h"
#include "rblock.h"
#include "rsegment.h"
#include "rclass_rank.h"
#include "rpool.h"
#include "rtrace.h"
#include "rhost.h"
#include "rscreen.h"
#include "murphy_cell.h"
#include "ropt_batch.h"
#include <memory>

using namespace s23;

static const int LP = RS_LP;

static const int CPT = 32; /* cells per thread per chunk: a warp does 32 x CPT */
static const int NT = 256; /* threads per block */

#ifdef __CUDACC__
#define CUDA_CHECK(x)                                                                     \
    do {                                                                                  \
        cudaError_t e = (x);                                                              \
        if (e != cudaSuccess) {                                                           \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
            exit(2);                                                                      \
        }                                                                                 \
    } while (0)

#include "rscreen_cuda.h"
#include "murphy_cuda.h"
#include "rsieve_cuda.h"

template <bool Strided>
__global__ void __launch_bounds__(NT) sieve_kernel(const RsSeed *Sg, const float *PT, const Blk *blks, int k, float aw,
                                                   Hit *out)
{
    extern __shared__ float T[];
    __shared__ uint32_t rmod[RS_NPRIMES];
    __shared__ float wkey[NT / 32][K_MAX], sel_key[K_MAX];
    __shared__ uint32_t woff[NT / 32][K_MAX], sel_off[K_MAX];
    const RsSeed &S = *Sg;
    const Blk B = blks[blockIdx.x];
    for (int i = threadIdx.x; i < RS_NPRIMES; i += blockDim.x) {
        const int64_t q = S.pr[i].qmax;
        rs_prime_table(T + S.pr[i].off, S, PT, i, (uint32_t)(((B.u % q) + q) % q));
        rmod[i] = (uint32_t)(((B.v0 % q) + q) % q);
    }
    __syncthreads();

    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    /* the warp's best k cells by cell_before, one per lane (lane m < k holds the m-th);
     * (tk, to) is the k-th, the bar a cell must pass to get in. Cells that pass are rare
     * after the first few chunks, so the cost is one vote per 32 cells */
    float ek = -INFINITY, tk = -INFINITY;
    uint32_t eo = UINT32_MAX, to = UINT32_MAX;
    for (uint32_t c0 = warp * 32 * CPT; c0 < B.len; c0 += nwarps * 32 * CPT) {
        float acc[CPT];
#pragma unroll
        for (int j = 0; j < CPT; j++)
            acc[j] = 0.0f;
        const uint32_t o = c0 + lane; /* offset of this lane's first cell */
        for (int i = 0; i < RS_NPRIMES; i++) {
            const uint32_t q = S.pr[i].qmax;
            const uint32_t stride = Strided ? B.step % q : 1;
            const uint32_t step = (32 * stride) % q;
            const float *Ti = T + S.pr[i].off;
            uint32_t idx = (rmod[i] + (o % q) * stride) % q;
#pragma unroll
            for (int j = 0; j < CPT; j++) {
                acc[j] += Ti[idx];
                idx += step;
                if (idx >= q)
                    idx -= q;
            }
        }
#pragma unroll
        for (int j = 0; j < CPT; j++) {
            const uint32_t off = c0 + lane + 32 * j;
            const float key = aw * acc[j] - (B.L0 + B.dL * (float)off);
            unsigned want = __ballot_sync(0xffffffffu, off < B.len && cell_before(key, off, tk, to));
            while (want) { /* warp-uniform: insert the lowest lane's cell, then re-vote */
                const int src = __ffs(want) - 1;
                const float ck = __shfl_sync(0xffffffffu, key, src);
                const uint32_t co = __shfl_sync(0xffffffffu, off, src);
                const int pos = __popc(__ballot_sync(0xffffffffu, (int)lane < k && cell_before(ek, eo, ck, co)));
                const float uk = __shfl_up_sync(0xffffffffu, ek, 1);
                const uint32_t uo = __shfl_up_sync(0xffffffffu, eo, 1);
                if ((int)lane > pos) {
                    ek = uk;
                    eo = uo;
                } else if ((int)lane == pos) {
                    ek = ck;
                    eo = co;
                }
                tk = __shfl_sync(0xffffffffu, ek, k - 1);
                to = __shfl_sync(0xffffffffu, eo, k - 1);
                want &= want - 1;
                want &= __ballot_sync(0xffffffffu, off < B.len && cell_before(key, off, tk, to));
            }
        }
    }
    if ((int)lane < k) {
        wkey[warp][lane] = ek;
        woff[warp][lane] = eo;
    }
    __syncthreads();
    /* the block's best k: merge the warps' sorted lists */
    if (threadIdx.x == 0) {
        int h[NT / 32];
        for (unsigned w = 0; w < nwarps; w++)
            h[w] = 0;
        for (int kk = 0; kk < k; kk++) {
            unsigned bw = 0;
            for (unsigned w = 1; w < nwarps; w++)
                if (cell_before(wkey[w][h[w]], woff[w][h[w]], wkey[bw][h[bw]], woff[bw][h[bw]]))
                    bw = w;
            sel_key[kk] = wkey[bw][h[bw]];
            sel_off[kk] = woff[bw][h[bw]];
            h[bw]++; /* never past k - 1: each round takes one entry in total */
        }
    }
    __syncthreads();
    /* the chosen cells' scores, summed again in the sieve's order (bit-identical) */
    if ((int)threadIdx.x < k) {
        const float key = sel_key[threadIdx.x];
        const uint32_t off = sel_off[threadIdx.x];
        float a = -INFINITY;
        int64_t v = 0;
        if (key > -INFINITY) {
            a = 0.0f;
            for (int i = 0; i < RS_NPRIMES; i++) {
                const uint32_t q = S.pr[i].qmax;
                const uint32_t stride = Strided ? B.step % q : 1;
                a += T[S.pr[i].off + (rmod[i] + (off % q) * stride) % q];
            }
            v = B.v0 + (int64_t)off * (Strided ? B.step : 1);
        }
        out[(size_t)blockIdx.x * k + threadIdx.x] = {a, key, B.u, v};
    }
}
#endif

/* calls seg(a, b, L0, dL) for each piece of a line's band between knots, where the
 * lognorm is L0 + dL (v - a) */
template <class F>
static void band_segments(const RsLine &L, F seg)
{
    const size_t n = L.kv.size();
    if (n < 2) {
        if (L.v_lo <= L.v_hi)
            seg(L.v_lo, L.v_hi, n ? L.kL[0] : L.L_min, 0.0);
        return;
    }
    for (size_t i = 0; i + 1 < n; i++) {
        const int64_t a = L.kv[i], b = i + 2 < n ? L.kv[i + 1] - 1 : L.kv[i + 1];
        if (b < a)
            continue;
        const double dL = L.kv[i + 1] > L.kv[i] ? (L.kL[i + 1] - L.kL[i]) / (double)(L.kv[i + 1] - L.kv[i]) : 0;
        seg(a, b, L.kL[i], dL);
    }
}

/* runs the sieve over the blocks of segs; `keep` gets each launch's hits. Returns the
 * number of blocks sieved. */
template <class F>
static size_t run_sieve(const RsSeed &S, const std::vector<float> &PT, const std::vector<Seg> &segs, int seg_log, int k,
                        float aw, int use_cpu, int dev, int launch_log2, F keep)
{
    BlockGen gen(segs, seg_log);
    Blk B;
    bool more = gen.next(B);
    if (use_cpu) {
        const size_t chunk = 1024;
        std::vector<Blk> batch;
        std::vector<Hit> part(chunk * k);
        while (more) {
            batch.clear();
            while (more && batch.size() < chunk) {
                batch.push_back(B);
                more = gen.next(B);
            }
            const size_t nb = batch.size();
#pragma omp parallel
            {
                std::vector<float> T(S.ntab), acc(1 << 16);
#pragma omp for schedule(dynamic, 1)
                for (size_t b = 0; b < nb; b++)
                    cpu_block(S, PT.data(), batch[b], k, aw, &part[b * k], T, acc);
            }
            keep(part.data(), nb * k);
        }
        return gen.made;
    }
#ifdef __CUDACC__
    CUDA_CHECK(cudaSetDevice(dev));
    RsSeed *Sg;
    Blk *bg;
    Hit *out;
    float *PTg;
    const size_t shmem = S.ntab * sizeof(float);
    /* a launch takes blocks until it holds about 2^launch_log2 cells (default 2^32, about
     * 0.1 s on a 5070), and at least four waves' worth of blocks so every SM has work and
     * the last wave's tail stays small, whatever the block lengths; at most max_blocks. Short launches keep the Windows desktop
     * responsive: under WSL2 it shares the GPU, and multi-second kernels (2^36 cells before
     * 2026-10-06) made input lag. A block is one CUDA block, so with -seg above about 22 a
     * single block already takes long. */
    int sms = 1, per_sm = 1;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const bool strided = std::any_of(segs.begin(), segs.end(), [](const Seg &s) { return s.step != 1; });
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, strided ? sieve_kernel<true> : sieve_kernel<false>, NT, shmem));
    // Expanded four-cell tables reduce lookup/index instructions on long blocks.
    // Preserve the scalar path for short blocks, limited devices and cache overflow.
    const size_t vector_shmem = S.ntab * sizeof(float4);
    int shared_limit = 0, thread_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&thread_limit, cudaDevAttrMaxThreadsPerBlock, dev));
    cudaFuncAttributes vector_attributes;
    CUDA_CHECK(cudaFuncGetAttributes(&vector_attributes,
        strided ? rs_vector_sieve_kernel<true> : rs_vector_sieve_kernel<false>));
    bool vector_available = thread_limit >= RS_VECTOR_NT &&
        vector_shmem + vector_attributes.sharedSizeBytes <= (size_t)shared_limit;
    if (vector_available) {
        CUDA_CHECK(cudaFuncSetAttribute(strided ? rs_vector_sieve_kernel<true> : rs_vector_sieve_kernel<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)vector_shmem));
        int vector_blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&vector_blocks,
            strided ? rs_vector_sieve_kernel<true> : rs_vector_sieve_kernel<false>, RS_VECTOR_NT, vector_shmem));
        vector_available = vector_blocks > 0;
    }
    RsVectorWorkspace vector_workspace;
    size_t vector_launches = 0, scalar_launches = 0;
    const size_t max_blocks = 1u << 16;
    const size_t min_blocks = std::min(max_blocks, (size_t)(4 * std::max(1, sms) * std::max(1, per_sm)));
    const double launch_cells = ldexp(1.0, launch_log2);
    CUDA_CHECK(cudaMalloc(&Sg, sizeof S));
    CUDA_CHECK(cudaMemcpy(Sg, &S, sizeof S, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&PTg, std::max<size_t>(1, PT.size()) * sizeof(float)));
    if (!PT.empty())
        CUDA_CHECK(cudaMemcpy(PTg, PT.data(), PT.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&bg, max_blocks * sizeof(Blk)));
    CUDA_CHECK(cudaMalloc(&out, max_blocks * k * sizeof(Hit)));
    std::vector<Blk> batch;
    batch.reserve(max_blocks);
    std::vector<Hit> part(max_blocks * k);
    while (more) {
        batch.clear();
        double cells = 0;
        while (more && batch.size() < max_blocks && (batch.size() < min_blocks || cells < launch_cells)) {
            batch.push_back(B);
            cells += B.len;
            more = gen.next(B);
        }
        const size_t nb = batch.size();
        CUDA_CHECK(cudaMemcpy(bg, batch.data(), nb * sizeof(Blk), cudaMemcpyHostToDevice));
        const bool vector = vector_available && cells >= (double)nb * 65536 &&
                            vector_workspace.prepare(S, Sg, PTg, batch);
        if (vector) {
            if (strided)
                rs_vector_sieve_kernel<true><<<(unsigned)nb, RS_VECTOR_NT, vector_shmem>>>(
                    Sg, vector_workspace.tables(), vector_workspace.row_ids(), bg, k, aw, out);
            else
                rs_vector_sieve_kernel<false><<<(unsigned)nb, RS_VECTOR_NT, vector_shmem>>>(
                    Sg, vector_workspace.tables(), vector_workspace.row_ids(), bg, k, aw, out);
            vector_launches++;
        } else if (strided)
            sieve_kernel<true><<<(unsigned)nb, NT, shmem>>>(Sg, PTg, bg, k, aw, out);
        else
            sieve_kernel<false><<<(unsigned)nb, NT, shmem>>>(Sg, PTg, bg, k, aw, out);
        if (!vector) scalar_launches++;
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(part.data(), out, nb * k * sizeof(Hit), cudaMemcpyDeviceToHost));
        keep(part.data(), nb * k);
    }
    printf("# root sieve GPU: %zu vector launches, %zu scalar launches; line-table cache at most 1024 rows\n",
           vector_launches, scalar_launches);
    cudaFree(out);
    cudaFree(bg);
    cudaFree(PTg);
    cudaFree(Sg);
#else
    (void)dev;
    (void)launch_log2;
#endif
    return gen.made;
}

static unsigned long long rng_state = 0x2545F4914F6CDD1Dull;
static unsigned long long rng()
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/* the sampled cells as the sieve (GPU or CPU) scored them vs brute force; returns how many differ */
static int check_hits(const RsSeed &S, const std::vector<Hit> &sample)
{
    int bad = 0;
    for (const Hit &h : sample)
        bad += fabs(rs_cell_score_brute(S, h.u, h.v) - h.score) > 1e-4;
    return bad;
}

static int parse_count(const char *s)
{
    char *end;
    errno = 0;
    const long n = strtol(s, &end, 10);
    return !*s || *end || errno || n < 0 || n > INT_MAX ? -1 : (int)n;
}

static int64_t parse_coordinate(const char *s, bool &valid)
{
    char *end;
    errno = 0;
    const long long n = strtoll(s, &end, 10);
    valid &= *s && !*end && !errno;
    return n;
}

static double parse_bound(const char *s)
{
    char *end;
    errno = 0;
    const double n = strtod(s, &end);
    return !*s || *end || errno ? NAN : n;
}

struct RoptSession {
#ifdef __CUDACC__
    RsScreenWorkspace screen;
    std::vector<std::unique_ptr<MurphyCudaEvaluator>> murphy_workers;
#endif
};

static int ropt_job(int argc, char **argv, RoptSession &session, RoptJobPaths *preflight = nullptr)
{
    // These are process globals for the shared host model, but options and
    // sampling must behave exactly as in a fresh process for every seed.
    rs_warm_skew = true;
    rs_fast_search = true;
    rng_state = 0x2545F4914F6CDD1Dull;
    const char *path = NULL, *out_path = NULL;
    const char *pool_in = nullptr, *pool_out = nullptr;
    RsCellTrace trace;
    bool trace_valid = true;
    int64_t u0 = 0, u1 = 0, v0 = 0, v1 = 0, cell_u = 0, cell_v = 0, umax = 5000;
    int top = 20, k = 8, seg_log = 20, use_cpu = 0, dev = 0, threads = 8, check = 0, cell = 0, search = 0;
    int patience = 20, refine = 200, rerank = 4096, window = 0, plan = 0, noproj = 0;
    int screen_count = 0, screen_bound = 0;
    bool screen_options = false, screen_exact_size = false;
    bool screen_gpu = false, screen_check = false;
    bool refine_gpu = false, refine_check = false, murphy_gpu = false, murphy_check = false;
    double budget = 0.5, aw = 1.0, maxcells = 0, band = 0;
    int maxlines = 1000, launch_log2 = 32;
    RsScanConfig scan;
    bool scan_options = false;
    int residue_mod = 0;
    int64_t residue_u = 0, residue_v = 0;
    bool residue_valid = true;
    RsClass residue;
    int class_mod = 0, class_count = 0, class_explore = -1;
    bool class_valid = true, class_options = false;
    RsClassRows class_rows;
    std::vector<RsClassProposal> selected_classes;
    bool murphy = false, murphy_options = false;
    MurphyConfig mc;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-poly") && i + 1 < argc)
            path = argv[++i];
        else if (!strcmp(argv[i], "-pool-in") && i + 1 < argc)
            pool_in = argv[++i];
        else if (!strcmp(argv[i], "-pool-out") && i + 1 < argc)
            pool_out = argv[++i];
        else if (!strcmp(argv[i], "-alpha-screen") && i + 2 < argc) {
            screen_options = true;
            screen_count = parse_count(argv[++i]);
            screen_bound = parse_count(argv[++i]);
        }
        else if (!strcmp(argv[i], "-alpha-screen-exact-size"))
            screen_exact_size = true;
        else if (!strcmp(argv[i], "-alpha-screen-gpu"))
            screen_gpu = true;
        else if (!strcmp(argv[i], "-alpha-screen-check"))
            screen_check = true;
        else if (!strcmp(argv[i], "-refine-gpu"))
            refine_gpu = true;
        else if (!strcmp(argv[i], "-refine-check"))
            refine_check = true;
        else if (!strcmp(argv[i], "-murphy-gpu"))
            murphy_gpu = true;
        else if (!strcmp(argv[i], "-murphy-check"))
            murphy_check = true;
        else if (!strcmp(argv[i], "-trace-cell") && i + 2 < argc) {
            trace.enabled = true;
            trace.u = parse_coordinate(argv[++i], trace_valid);
            trace.v = parse_coordinate(argv[++i], trace_valid);
        }
        else if (!strcmp(argv[i], "-u") && i + 2 < argc) {
            window = 1;
            u0 = atoll(argv[++i]);
            u1 = atoll(argv[++i]);
        } else if (!strcmp(argv[i], "-v") && i + 2 < argc) {
            window = 1;
            v0 = atoll(argv[++i]);
            v1 = atoll(argv[++i]);
        } else if (!strcmp(argv[i], "-cell") && i + 2 < argc) {
            cell = 1;
            cell_u = atoll(argv[++i]);
            cell_v = atoll(argv[++i]);
        } else if (!strcmp(argv[i], "-search"))
            search = 1;
        else if (!strcmp(argv[i], "-plan"))
            plan = 1;
        else if (!strcmp(argv[i], "-budget") && i + 1 < argc)
            budget = atof(argv[++i]);
        else if (!strcmp(argv[i], "-aw") && i + 1 < argc)
            aw = atof(argv[++i]);
        else if (!strcmp(argv[i], "-maxcells") && i + 1 < argc)
            maxcells = atof(argv[++i]);
        else if (!strcmp(argv[i], "-maxlines") && i + 1 < argc)
            maxlines = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-launch") && i + 1 < argc)
            launch_log2 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-patience") && i + 1 < argc)
            patience = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-umax") && i + 1 < argc)
            umax = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-u-scan") && i + 1 < argc)
            scan.step = parse_count(argv[++i]);
        else if (!strcmp(argv[i], "-u-basins") && i + 1 < argc) {
            scan.basins = parse_count(argv[++i]);
            scan_options = true;
        } else if (!strcmp(argv[i], "-u-radius") && i + 1 < argc) {
            scan.radius = parse_count(argv[++i]);
            scan_options = true;
        } else if (!strcmp(argv[i], "-residue") && i + 3 < argc) {
            residue_mod = parse_count(argv[++i]);
            residue_valid &= residue_mod >= 1 && residue_mod <= 65536;
            residue_u = parse_coordinate(argv[++i], residue_valid);
            residue_v = parse_coordinate(argv[++i], residue_valid);
        } else if (!strcmp(argv[i], "-classes") && i + 2 < argc) {
            class_mod = parse_count(argv[++i]);
            class_count = parse_count(argv[++i]);
            class_valid &= class_mod >= 2 && class_mod <= 4096 && class_count >= 1 &&
                           class_count <= 64 && class_count <= (int64_t)class_mod * class_mod;
        } else if (!strcmp(argv[i], "-class-explore") && i + 1 < argc) {
            class_explore = parse_count(argv[++i]);
            class_options = true;
        } else if (!strcmp(argv[i], "-out") && i + 1 < argc)
            out_path = argv[++i];
        else if (!strcmp(argv[i], "-refine") && i + 1 < argc)
            refine = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-murphy"))
            murphy = true;
        else if (!strcmp(argv[i], "-Bf") && i + 1 < argc) {
            mc.Bf = parse_bound(argv[++i]);
            murphy_options = true;
        } else if (!strcmp(argv[i], "-Bg") && i + 1 < argc) {
            mc.Bg = parse_bound(argv[++i]);
            murphy_options = true;
        } else if (!strcmp(argv[i], "-area") && i + 1 < argc) {
            mc.area = parse_bound(argv[++i]);
            murphy_options = true;
        } else if ((!strcmp(argv[i], "-murphy-points") || !strcmp(argv[i], "-K")) && i + 1 < argc) {
            mc.K = parse_count(argv[++i]);
            murphy_options = true;
        } else if ((!strcmp(argv[i], "-murphy-eval-points") || !strcmp(argv[i], "-Keval")) && i + 1 < argc) {
            mc.Keval = parse_count(argv[++i]);
            murphy_options = true;
        } else if (!strcmp(argv[i], "-maxeval") && i + 1 < argc) {
            mc.maxeval = parse_count(argv[++i]);
            murphy_options = true;
        } else if (!strcmp(argv[i], "-rerank") && i + 1 < argc) {
            rerank = parse_count(argv[++i]);
            if (rerank > (1 << 24)) rerank = -1;
        } else if (!strcmp(argv[i], "-band") && i + 1 < argc)
            band = atof(argv[++i]);
        else if (!strcmp(argv[i], "-top") && i + 1 < argc)
            top = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc)
            k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-seg") && i + 1 < argc)
            seg_log = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-dev") && i + 1 < argc)
            dev = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc)
            threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-check") && i + 1 < argc)
            check = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-cpu"))
            use_cpu = 1;
        else if (!strcmp(argv[i], "-noproj"))
            noproj = 1;
        else if (!strcmp(argv[i], "-exactskew"))
            rs_warm_skew = false;
        else if (!strcmp(argv[i], "-oldsearch"))
            rs_fast_search = false;
        else {
            fprintf(stderr, "usage: see the comment at the top of %s\n", __FILE__);
            return 2;
        }
    }
#ifndef __CUDACC__
    use_cpu = 1;
#endif
    omp_set_num_threads(threads > 0 ? threads : 1);
    if (!trace_valid || (trace.enabled && (cell || (!search && !window)))) {
        fprintf(stderr, "s23_ropt: -trace-cell needs integer U/V and search or window mode, not -cell\n");
        return 2;
    }
    if ((pool_in || pool_out) && (!search || window || cell || plan || !rerank ||
            (pool_in && (pool_out || class_mod || residue_mod || scan.step)))) {
        fprintf(stderr, "s23_ropt: pools require search and rerank; pool replay cannot use search selection options\n");
        return 2;
    }
    scan.umax = umax;
    if (!class_valid || (class_options && (!class_mod || class_explore < 0 || class_explore > class_count)) ||
        (class_mod && (!search || window || cell || residue_mod || scan.step || umax < 0 || umax > 1000000000 ||
                       maxlines < 1 || patience < 1))) {
        fprintf(stderr, "s23_ropt: -classes needs search mode, M in [2,4096], K in [1,min(64,M*M)], "
                        "0 <= -class-explore <= K, positive maxlines/patience, umax in [0,1e9]; "
                        "cannot combine with -residue/-u-scan/-cell/window\n");
        return 2;
    }
    if (class_mod && class_explore < 0)
        class_explore = class_count / 4;
    if (!residue_valid || (residue_mod && (cell || scan.step || (!search && !window) ||
        (!window && (umax < 0 || umax > 1000000000)) ||
        (window && ((__int128)v1 - v0 > INT64_MAX || (__int128)u1 - u0 > INT64_MAX))))) {
        fprintf(stderr, "s23_ropt: -residue needs M in [1,65536], integer U/V, window or search mode, "
                        "no -u-scan/-cell; search umax in [0,1e9], window spans <= INT64_MAX\n");
        return 2;
    }
    if (residue_mod) {
        residue = {(uint32_t)residue_mod, rs_residue(residue_u, residue_mod), rs_residue(residue_v, residue_mod)};
        class_rows.modulus = residue.modulus;
        class_rows.rows[residue.u] = {residue.v};
        if (window && rs_progression(u0, u1, residue.modulus, residue.u).count > 1000000) {
            fprintf(stderr, "s23_ropt: -residue window permits at most 1000000 selected u lines\n");
            return 2;
        }
        if (!preflight)
            printf("# residue class: u = %u, v = %u (mod %u); counts are selected cells\n",
                   residue.u, residue.v, residue.modulus);
    }
    if ((scan_options && scan.step <= 0) || scan.step < 0 ||
        (scan.step && (!search || window || cell || !rs_scan_valid(scan)))) {
        fprintf(stderr, "s23_ropt: -u-scan needs search mode, 0 <= umax <= 1e9, at most 4097 samples; "
                        "1 <= -u-basins <= 16, 0 <= -u-radius <= 4096, and at most 16384 neighborhood lines; "
                        "-u-basins/-u-radius need -u-scan > 0\n");
        return 2;
    }
    if (!path || (window && (u1 < u0 || v1 < v0)) || (!window && !search && !cell) || k < 1 || k > K_MAX ||
        seg_log < 12 || seg_log > 30 || budget <= 0 || aw <= 0 || launch_log2 < 20 || launch_log2 > 40) {
        fprintf(stderr, "s23_ropt: need -poly and one of -u/-v, -search, -cell; 1 <= -k <= %d, 12 <= LOG2 <= 30, "
                        "20 <= -launch <= 40\n",
                K_MAX);
        return 2;
    }
    if (top < 1 || refine < 0 || rerank < 0 || band < 0 || band > budget) {
        fprintf(stderr, "s23_ropt: need -top >= 1, -refine >= 0, -rerank an integer from 0 to 2^24, and "
                        "0 < -band <= -budget\n");
        return 2;
    }
    if ((refine_check && !refine_gpu) || (murphy_check && !murphy_gpu) ||
        ((refine_gpu || murphy_gpu) && (use_cpu || !search || window || cell || plan || !rerank)) ||
        (murphy_gpu && !murphy)) {
        fprintf(stderr, "s23_ropt: -refine-gpu/-murphy-gpu require a CUDA build, search with rerank, no -cpu/window/cell/plan; "
                        "-murphy-gpu requires -murphy; each -check requires its -gpu option\n");
        return 2;
    }
    if ((screen_check && !screen_gpu) ||
        (screen_gpu && (!screen_options || !screen_exact_size || use_cpu || screen_bound > RS_SCREEN_MAX_BOUND))) {
        fprintf(stderr, "s23_ropt: -alpha-screen-gpu requires a CUDA build, no -cpu, "
                        "-alpha-screen N B with B <= 2000, and -alpha-screen-exact-size; "
                        "-alpha-screen-check requires -alpha-screen-gpu\n");
        return 2;
    }
    if ((screen_exact_size && !screen_options) || (screen_options && (!search || window || cell || plan || rerank <= 0 ||
        screen_count < std::max({rerank, refine, top}) || screen_count > (1 << 24) ||
        screen_bound < 2 || (unsigned long)screen_bound > cio_alpha_bound() || !std::isfinite(aw)))) {
        fprintf(stderr, "s23_ropt: -alpha-screen needs search with rerank > 0, "
                        "max(rerank,refine,top) <= N <= 2^24, 2 <= B <= %lu; no window/cell/plan; "
                        "-alpha-screen-exact-size requires -alpha-screen\n",
                        cio_alpha_bound());
        return 2;
    }
    if ((murphy_options && !murphy) || (murphy &&
        (!search || window || cell || plan || !rerank || !murphy_config_valid(mc)))) {
        fprintf(stderr, "s23_ropt: -murphy requires -search, -rerank > 0, explicit finite -Bf/-Bg > 1 and "
                        "-area > 0, -murphy-points >= 16000, -murphy-eval-points >= 4*points, -maxeval >= 32; "
                        "Murphy options require -murphy and cannot be used with window/-cell/-plan\n");
        return 2;
    }
    if (preflight) {
        if (!search || window || cell || plan || !out_path) {
            fprintf(stderr, "s23_ropt: each batch job requires -search and -out; no window/cell/plan\n");
            return 2;
        }
        preflight->inputs = {path};
        if (pool_in) preflight->inputs.push_back(pool_in);
        preflight->outputs = {out_path};
        if (pool_out) preflight->outputs.push_back(pool_out);
        return 0;
    }
    FILE *fh = fopen(path, "r");
    if (!fh) {
        perror(path);
        return 2;
    }
    struct ScopedPoly {
        cio_poly value;
        ScopedPoly() { cio_init(&value); }
        ~ScopedPoly() { cio_clear(&value); }
    } owner;
    cio_poly &P = owner.value;
    const int read_status = cio_read_next(fh, &P);
    fclose(fh);
    if (read_status != 1) {
        fprintf(stderr, "s23_ropt: %s: no polynomial\n", path);
        return 2;
    }
    RsSeed S{};
    RsizePoly<LP> R{};
    if (!rs_seed_and_size(S, R, P)) {
        fprintf(stderr, "s23_ropt: only degree 5 (and coefficients up to 512 bits)\n");
        return 2;
    }
    /* the GPU's context is created in the background while the host runs the size model:
     * on WSL2 creating it takes 1-13 s */
    struct GpuStart {
        std::thread th;
        bool screen_ready = false;
        ~GpuStart()
        {
            if (th.joinable())
                th.join();
        }
    } gpu_start;
#ifdef __CUDACC__
    if (!use_cpu && !cell && !plan && (!pool_in || screen_gpu || refine_gpu || murphy_gpu))
        gpu_start.th = std::thread([dev, screen_gpu, refine_gpu, &gpu_start, &session] {
            if (screen_gpu || refine_gpu) gpu_start.screen_ready = session.screen.start(dev);
            else if (cudaSetDevice(dev) == cudaSuccess) cudaFree(0);
        });
#endif
    S.proj = !noproj;
    std::vector<float> PT;
    rs_proj_tables(S, PT);
    const char *alpha_what = noproj ? "affine roots, p^e <= 200" : "affine roots p^e <= 200, projective p^(e+3)";
    std::vector<float> T(S.ntab);
    if (cell) {
        rs_line_tables(T.data(), S, PT.data(), cell_u);
        const float st = rs_cell_score(T.data(), S, cell_v);
        int64_t t = 0;
        const double L = rs_size(R, cell_u, cell_v, t);
        printf("cell u %lld v %lld: score %.6f (brute force %.6f), alpha_s %.6f; lognorm %.4f at t %lld\n",
               (long long)cell_u, (long long)cell_v, st, rs_cell_score_brute(S, cell_u, cell_v), S.alpha0 - st, L,
               (long long)t);
        return 0;
    }

    /* ---- the work: a box, or the size model's bands ---- */
    std::vector<Seg> segs; /* what is sieved: pieces of lines, split into blocks as it goes */
    std::map<int64_t, RsLine> lines;
    std::vector<RsLine *> kept; /* search: the lines whose bands are sieved */
    double L_best = INFINITY, budget_used = band > 0 ? band : budget;
    if (!pool_in) {
        trace.stage("region", window ? trace.u >= u0 && trace.u <= u1 && trace.v >= v0 && trace.v <= v1
                                    : (__int128)trace.u >= -(__int128)umax && trace.u <= umax);
        if (residue_mod)
            trace.stage("explicit_class", rs_residue(trace.u, residue.modulus) == residue.u &&
                                           rs_residue(trace.v, residue.modulus) == residue.v);
    }
    /* every line's band at L_best + b (the -maxcells loop and -plan's step table): the cells, and
     * in *nl the lines with a band */
    std::vector<RsLine *> all;
    const bool sparse = class_mod || residue.modulus > 1;
    auto band_cells = [&](const RsLine *L) {
        if (sparse)
            return (double)class_rows.count(L->u, L->v_lo, L->v_hi);
        return L->v_lo <= L->v_hi ? (double)(L->v_hi - L->v_lo) + 1 : 0.0;
    };
    auto bands_at = [&](double b, size_t *nl) {
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < all.size(); i++) {
            rs_line_band(*all[i], R, L_best + b, 1, 0);
            if (sparse) {
                const auto p = class_rows.clip(all[i]->u, all[i]->v_lo, all[i]->v_hi);
                all[i]->v_lo = p.first;
                all[i]->v_hi = p.last;
            }
        }
        double c = 0;
        size_t n = 0;
        for (auto *L : all) {
            c += band_cells(L);
            n += band_cells(L) > 0;
        }
        if (nl)
            *nl = n;
        return c;
    };
    size_t lines_dropped = 0, lines_trimmed = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto secs = [](std::chrono::steady_clock::time_point a) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
    };
    double t_scan = 0, t_bands = 0, t_knots = 0; /* the size model's phases */
    int n_bands = 0;
    RsScanStats scan_stats;
    if (!window && !pool_in) {
        /* line minima from u = 0 outwards, warm-started from the neighbour; a direction
         * stops after `patience` lines beyond the budget, or after maxlines / 2 lines within
         * it (a seed whose size is flat in u would otherwise run to umax) */
        int64_t u_best = 0;
        const auto scan_u = [&](uint32_t ur, uint32_t stride) {
            double class_best = INFINITY;
            for (int dir = 1; dir >= -1; dir -= 2) {
                const auto anchor = lines.find(ur);
                int64_t v = dir > 0 || anchor == lines.end() ? 0 : anchor->second.v_min;
                int64_t t = dir > 0 || anchor == lines.end() ? 0 : anchor->second.t_min;
                int beyond = 0, within = 0;
                const int64_t start_u = dir > 0 ? (int64_t)ur : (int64_t)ur - stride;
                for (int64_t u = start_u; u <= umax && u >= -umax; u += dir * (int64_t)stride) {
                    RsLine L;
                    rs_line_min(L, R, u, v, t, 1 << 20);
                    if (stride > 1 && (v || t)) {
                        RsLine independent;
                        rs_line_min(independent, R, u, 0, 0, 1 << 20);
                        if (independent.L_min < L.L_min)
                            L = std::move(independent);
                    }
                    v = L.v_min;
                    t = L.t_min;
                    lines[u] = L;
                    if (L.L_min < L_best) {
                        L_best = L.L_min;
                        u_best = u;
                    }
                    class_best = std::min(class_best, L.L_min);
                    const double reference = class_mod ? class_best : L_best;
                    beyond = L.L_min > reference + budget + 0.3 ? beyond + 1 : 0;
                    within += L.L_min <= reference + budget;
                    if (beyond >= patience || (maxlines > 0 && 2 * within >= maxlines))
                        break;
                }
            }
        };
        if (!class_mod)
            scan_u(residue.u, residue.modulus);
        else {
            auto pool = rs_class_pool(S, PT.data(), class_mod, class_count);
            const auto is_trace_class = [&](const RsClassProposal &c) {
                return trace.enabled && c.cell.u == rs_residue(trace.u, class_mod) &&
                       c.cell.v == rs_residue(trace.v, class_mod);
            };
            trace.stage("class_proposal", std::any_of(pool.begin(), pool.end(), is_trace_class));
            std::set<uint32_t> urows;
            for (const auto &c : pool)
                urows.insert(c.cell.u);
            uint64_t possible_lines = 0;
            for (uint32_t u : urows)
                possible_lines += rs_progression(-umax, umax, class_mod, u).count;
            if (possible_lines > 65536) {
                fprintf(stderr, "s23_ropt: -classes candidate pool exceeds 65536 possible u lines; reduce -umax\n");
                return 2;
            }
            for (uint32_t u : urows)
                scan_u(u, class_mod);
            std::map<uint32_t, const RsLine *> minima;
            for (const auto &entry : lines) {
                const RsLine &line = entry.second;
                auto &best = minima[rs_residue(line.u, class_mod)];
                if (std::isfinite(line.L_min) && (!best || line.L_min < best->L_min))
                    best = &line;
            }
            for (auto &c : pool) {
                auto it = minima.find(c.cell.u);
                if (it == minima.end() || !it->second)
                    continue;
                const RsLine &line = *it->second;
                const int64_t delta = (c.cell.v + class_mod - rs_residue(line.v_min, class_mod)) % class_mod;
                for (int64_t offset : {delta, delta - class_mod}) {
                    const __int128 v = (__int128)line.v_min + offset;
                    if (v < INT64_MIN || v > INT64_MAX)
                        continue;
                    int64_t t = line.t_min;
                    c.size = std::min(c.size, rs_size(R, line.u, (int64_t)v, t, 1 << 10));
                }
            }
            selected_classes = rs_choose_classes(pool, class_count, class_explore, aw);
            if (trace.active()) {
                const auto ranked = rs_choose_classes(pool, (int)pool.size(), 0, aw);
                const auto it = std::find_if(ranked.begin(), ranked.end(), is_trace_class);
                trace.stage("class_size", it != ranked.end());
                if (it != ranked.end())
                    printf("# trace-cell class mod %d u %u v %u: root mean %.9g size %.9g key %.9g "
                           "size-root rank %zu of %zu exploration %u\n", class_mod, it->cell.u, it->cell.v,
                           it->roots, it->size, it->key(aw), (size_t)(it - ranked.begin()) + 1,
                           ranked.size(), it->exploration);
                trace.stage("class_selection", std::any_of(selected_classes.begin(), selected_classes.end(), is_trace_class));
            }
            class_rows.modulus = class_mod;
            printf("# automatic classes: mod %d, %zu proposals, %zu sampled lines, %zu selected, exploration requested %d\n",
                   class_mod, pool.size(), lines.size(), selected_classes.size(), class_explore);
            for (const auto &c : selected_classes) {
                class_rows.rows[c.cell.u].push_back(c.cell.v);
                printf("# class u %u v %u: root mean %.6f, guaranteed log content %.6f, size estimate %.6f, key %.6f, %s\n",
                       c.cell.u, c.cell.v, c.roots, c.content, c.size, c.key(aw),
                       c.selected_exploration ? "exploration" : "ranked");
            }
            for (auto it = lines.begin(); it != lines.end();)
                if (class_rows.at(it->first).empty())
                    it = lines.erase(it);
                else
                    ++it;
            L_best = INFINITY;
            for (const auto &entry : lines)
                if (entry.second.L_min < L_best) {
                    L_best = entry.second.L_min;
                    u_best = entry.first;
                }
        }
        if (scan.step) {
            scan_stats = rs_scan_basins(lines, scan, [&](int64_t u, int64_t v, int64_t t) {
                RsLine line;
                rs_line_min(line, R, u, v, t, 1 << 20);
                return line;
            });
            for (const auto &entry : lines)
                if (entry.second.L_min < L_best) {
                    L_best = entry.second.L_min;
                    u_best = entry.first;
                }
            printf("# u scan: step %d, %zu samples, %zu lines added, %zu model evaluations, %zu nonfinite; centers",
                   scan.step, scan_stats.samples, scan_stats.added, scan_stats.evaluations, scan_stats.nonfinite);
            for (int64_t u : scan_stats.centers)
                printf(" %lld", (long long)u);
            printf("\n");
        }
        if (!std::isfinite(L_best)) {
            fprintf(stderr, "s23_ropt: no finite line minimum found\n");
            return 1;
        }
        /* the band edges at -budget (or -band); with -maxcells, the budget is lowered in steps
         * of 0.25 (not below 0.5) until the bands hold at most that many cells, and if they
         * still hold more, only the lines nearest the best line are kept. Knots last, for the
         * kept lines. */
        t_scan = secs(t0);
        auto tb = std::chrono::steady_clock::now();
        for (auto &kv : lines)
            all.push_back(&kv.second);
        trace.stage("line_scan", lines.count(trace.u) != 0);
        const auto in_trace_band = [&]() {
            const auto it = lines.find(trace.u);
            return it != lines.end() && it->second.v_lo <= trace.v && trace.v <= it->second.v_hi;
        };
        for (;;) {
            const double c = bands_at(budget_used, nullptr);
            if (!n_bands)
                trace.stage("size_band", in_trace_band());
            n_bands++;
            if (maxcells <= 0 || c <= maxcells || budget_used <= 0.5)
                break;
            budget_used = std::max(0.5, budget_used - 0.25);
        }
        trace.stage("cell_cap_budget", in_trace_band());
        t_bands = secs(tb);
        for (auto *L : all)
            if (band_cells(L) > 0)
                kept.push_back(L);
        std::stable_sort(kept.begin(), kept.end(), [&](const RsLine *a, const RsLine *b) {
            return std::llabs(a->u - u_best) < std::llabs(b->u - u_best);
        });
        double c = 0;
        size_t nkeep = 0;
        for (; nkeep < kept.size() && maxcells > 0; nkeep++) {
            RsLine *L = kept[nkeep];
            const double room = maxcells - c;
            if (room < 1)
                break;
            if (band_cells(L) > room) { /* the last line, trimmed around its minimum */
                if (sparse) {
                    const auto p = class_rows.trim(L->u, L->v_lo, L->v_hi, L->v_min, (uint64_t)room);
                    L->v_lo = p.first;
                    L->v_hi = p.last;
                } else {
                    const int64_t half = (int64_t)((room - 1) / 2); /* 2 half + 1 <= room cells */
                    L->v_lo = std::max(L->v_lo, L->v_min - half);
                    L->v_hi = std::min(L->v_hi, L->v_min + half);
                }
                lines_trimmed++;
            }
            c += band_cells(L);
        }
        if (maxcells <= 0)
            nkeep = kept.size();
        for (size_t i = nkeep; i < kept.size(); i++) {
            kept[i]->v_lo = 1;
            kept[i]->v_hi = 0;
        }
        lines_dropped = kept.size() - nkeep;
        kept.resize(nkeep);
        trace.stage("cell_cap_lines", in_trace_band());
        auto tk = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < kept.size(); i++)
            rs_line_knots(*kept[i], R, 64);
        t_knots = secs(tk);
    }
    auto add_segment = [&](Seg s) {
        if (sparse)
            for (uint32_t v : class_rows.at(s.u))
                class_segment(segs, s, {class_rows.modulus, rs_residue(s.u, class_rows.modulus), v});
        else
            segs.push_back(s);
    };
    if (window) {
        const auto us = rs_progression(u0, u1, residue.modulus, residue.u);
        int64_t u = us.first;
        for (uint64_t i = 0; i < us.count; i++) {
            add_segment({u, v0, v1, 0.0, 0.0});
            if (i + 1 < us.count)
                u += residue.modulus;
        }
    } else
        for (auto *L : kept)
            band_segments(*L, [&](int64_t a, int64_t b, double L0, double dL) { add_segment({L->u, a, b, L0, dL}); });
    size_t nblk = 0;
    double ncells = 0;
    for (const Seg &G : segs) {
        nblk += seg_blocks(G, seg_log);
        ncells += (double)seg_cells(G);
    }
    if (trace.active() && !pool_in) {
        const bool admitted = std::any_of(segs.begin(), segs.end(), [&](const Seg &s) {
            return s.u == trace.u && s.a <= trace.v && trace.v <= s.b &&
                   ((uint64_t)trace.v - (uint64_t)s.a) % s.step == 0;
        });
        trace.stage("segments", admitted);
    }
    if (class_mod) {
        std::map<std::pair<uint32_t, uint32_t>, double> admitted;
        for (const auto &s : segs)
            admitted[{rs_residue(s.u, class_mod), rs_residue(s.a, class_mod)}] += seg_cells(s);
        for (const auto &c : selected_classes)
            printf("# class cells u %u v %u: %.0f selected (%s)\n", c.cell.u, c.cell.v,
                   admitted[{c.cell.u, c.cell.v}], c.selected_exploration ? "exploration" : "ranked");
    }
    if (scan.step) {
        printf("# u scan retained center lines:");
        size_t retained = 0;
        for (int64_t u : scan_stats.centers)
            if (band_cells(&lines.at(u)) > 0) {
                printf(" %lld", (long long)u);
                retained++;
            }
        printf(" (%zu/%zu after -maxcells)\n", retained, scan_stats.centers.size());
    }
    const double t_model = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (plan && window) {
        printf("# window: %.3g cells in %zu blocks\n", ncells, nblk);
        trace.finish("plan_admitted");
        return 0;
    }
    if (plan) {
        printf("# size model %.2f s: best line minimum %.4f, budget %.2f: %.3g cells in %zu blocks (%zu lines "
               "dropped, %zu trimmed by -maxcells)\n",
               t_model, L_best, budget_used, ncells, nblk, lines_dropped, lines_trimmed);
        printf("# size model phases: line scan %.2f s, bands %.2f s at %d budgets, knots %.2f s\n", t_scan, t_bands,
               n_bands, t_knots);
        printf("#      u   L_min        v_min           band v_lo .. v_hi          cells\n");
        for (auto &kv : lines) {
            const RsLine &L = kv.second;
            const double c = band_cells(&L);
            printf("%8lld %8.4f %16lld %16lld %16lld %10.3g\n", (long long)L.u, L.L_min, (long long)L.v_min,
                   (long long)L.v_lo, (long long)L.v_hi, c);
        }
        /* the cells at each budget from -budget down in steps of 0.25, which -band B sieves
           exactly (last: it rewrites the lines' bands, and is not in the size model's time) */
        for (double b = budget;; b = std::max(0.5, b - 0.25)) {
            size_t nl = 0;
            const double c = bands_at(b, &nl);
            printf("# step %.2f: %.0f cells, %zu lines\n", b, c, nl);
            if (b <= 0.5)
                break;
        }
        trace.finish("plan_admitted");
        return 0;
    }

    /* ---- sieve; keep hits by the proxy (search) or the score (window) ---- */
    using Cand = RsCandidate;
    std::vector<Cand> best;
    const auto class_survivors = [&](const char *stage, size_t n) {
        if (!class_mod)
            return;
        std::map<std::pair<uint32_t, uint32_t>, size_t> counts;
        for (size_t i = 0; i < std::min(n, best.size()); i++)
            counts[{rs_residue(best[i].h.u, class_mod), rs_residue(best[i].h.v, class_mod)}]++;
        for (const auto &c : selected_classes)
            printf("# class %s u %u v %u: %zu cells\n", stage, c.cell.u, c.cell.v, counts[{c.cell.u, c.cell.v}]);
    };
    std::map<int64_t, Cand> line_best;
    /* a total order, so the kept set does not depend on how the hits arrive in launches */
    auto order = [](const Cand &a, const Cand &b) {
        return a.key < b.key || (a.key == b.key && (a.h.u < b.h.u || (a.h.u == b.h.u && a.h.v < b.h.v)));
    };
    const size_t cap = (size_t)std::max<int64_t>({4 * (int64_t)std::max(top, refine), rerank, screen_count, 4096});
    const auto proxy_candidate = [&](const Hit &hit) {
        Cand c{};
        c.h = hit;
        c.A = NAN;
        c.L = window ? 0 : rs_line_interp(lines.at(hit.u), hit.v, c.t);
        c.key = window ? -hit.score : c.L + aw * (S.alpha0 - hit.score);
        return c;
    };
    // Count the watched cell's global proxy rank even if streaming truncation
    // removes it. Evaluate its proxy exactly as keep() does, without inserting it.
    Cand watched{};
    size_t trace_before = 0, trace_total = 0;
    bool trace_score_agrees = true;
    if (trace.active() && !pool_in) {
        rs_line_tables(T.data(), S, PT.data(), trace.u);
        watched = proxy_candidate({rs_cell_score(T.data(), S, trace.v), 0, trace.u, trace.v});
        printf("# trace-cell proxy key %.17g interpolated lognorm %.17g alpha_s %.9g\n",
               watched.key, watched.L, S.alpha0 - watched.h.score);
    }
    std::vector<Hit> sample; /* -check: a uniform sample of the sieve's hits (reservoir) */
    size_t nhits = 0;
    auto keep = [&](const Hit *h, size_t n) {
        for (size_t i = 0; i < n; i++) {
            if (!(h[i].score > -INFINITY))
                continue;
            if (trace.matches(h[i].u, h[i].v))
                trace.block_hit = true;
            if (check) {
                nhits++;
                if (sample.size() < (size_t)check)
                    sample.push_back(h[i]);
                else if (rng() % nhits < (size_t)check)
                    sample[rng() % check] = h[i];
            }
            Cand c = proxy_candidate(h[i]);
            if (!window) {
                auto it = line_best.find(h[i].u);
                if (it == line_best.end() || c.key < it->second.key)
                    line_best[h[i].u] = c;
            }
            if (trace.active()) {
                trace_total++;
                trace_before += order(c, watched);
                if (trace.matches(c.h.u, c.h.v))
                    trace_score_agrees = c.key == watched.key;
            }
            best.push_back(c);
        }
        if (best.size() > 4 * cap) {
            std::nth_element(best.begin(), best.begin() + cap, best.end(), order);
            best.resize(cap);
        }
    };
    auto t_wait = std::chrono::steady_clock::now();
    if (gpu_start.th.joinable())
        gpu_start.th.join();
    auto t1 = std::chrono::steady_clock::now();
    const double t_start = std::chrono::duration<double>(t1 - t_wait).count();
    size_t made = 0;
    if (pool_in) {
        if (!rs_pool_read(pool_in, P, aw, !noproj, best)) {
            fprintf(stderr, "s23_ropt: invalid, truncated or incompatible candidate pool %s\n", pool_in);
            return 2;
        }
        printf("# replaying %zu saved candidates; no size model or sieve\n", best.size());
    } else
        made = run_sieve(S, PT, segs, seg_log, k, (float)aw, use_cpu, dev, launch_log2, keep);
    const double t_sieve = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    if (made != nblk) { /* seg_blocks and the generator split segments the same way */
        fprintf(stderr, "s23_ropt: internal error: sieved %zu blocks, counted %zu\n", made, nblk);
        return 2;
    }
    std::sort(best.begin(), best.end(), order);
    const auto trace_rank = [&](const char *stage, size_t keep) {
        if (!trace.active()) return;
        const auto it = std::find_if(best.begin(), best.end(), [&](const Cand &c) {
            return trace.matches(c.h.u, c.h.v);
        });
        trace.rank(stage, it == best.end() ? 0 : (size_t)(it - best.begin()) + 1, keep, best.size());
    };
    if (!pool_in)
        trace.stage("block_topk", trace.block_hit);
    else
        trace_rank("replay_input", best.size());

    /* search: exact translation and lognorm for the best max(top, refine), then re-rank; with -rerank N, for
       the best N, re-ranked by CADO's alpha instead of alpha_s, and the best max(top, refine) of them kept */
    double t_refine = 0, t_screen = 0;
    std::vector<std::unique_ptr<MurphyCell>> murphy_cells;
    size_t murphy_skipped = 0, murphy_failed = 0, murphy_accepted = 0, murphy_limited = 0, murphy_count = 0;
    if (!window) {
        const auto t2 = std::chrono::steady_clock::now();
        const size_t nout = (size_t)std::max(refine, top);
        const size_t nref = std::min(best.size(), std::max(nout, (size_t)rerank));
        const size_t nproxy = screen_options ? std::min(best.size(), (size_t)screen_count) : nref;
        if (!pool_in && trace_score_agrees)
            trace.rank("proxy_pool", trace_before + 1, nproxy, trace_total);
        else {
            if (!trace_score_agrees)
                printf("# trace-cell: host/device proxy disagreement; rank uses the remaining pool only\n");
            trace_rank("proxy_pool", nproxy);
        }
        if (pool_out && !rs_pool_write(pool_out, P, aw, !noproj, best, nproxy)) {
            fprintf(stderr, "s23_ropt: cannot write candidate pool %s\n", pool_out);
            return 2;
        }
        class_survivors("proxy pool", nproxy);
        // A screen with no cut is unnecessary, except when explicitly auditing CUDA.
        if (screen_options && (nproxy > nref || screen_check)) {
            const auto ts = std::chrono::steady_clock::now();
            best.resize(nproxy);
            bool device_screened = false;
#ifdef __CUDACC__
            if (screen_gpu && gpu_start.screen_ready) {
                size_t fallbacks = 0;
                device_screened = rs_screen_gpu(best, R, P, screen_bound, aw, dev, screen_check, fallbacks, session.screen);
                if (device_screened) {
                    printf("# alpha-screen GPU: %zu cells, %zu host fallbacks; finalists canonicalized on host\n",
                           nproxy, fallbacks);
                    std::sort(best.begin(), best.end(), order);
                    const double cut = nref ? best[nref - 1].key : 0;
                    size_t rechecks = 0;
                    if (nref < nproxy) {
#pragma omp parallel for schedule(dynamic, 1) reduction(+:rechecks)
                        for (size_t i = 0; i < nproxy; ++i) {
                            auto &c = best[i];
                            if (c.screen_host || !rs_screen_near_cut(c.key, cut)) continue;
                            const RsExact x = rs_exact_cell(P, R, c.h.u, c.h.v, c.t, screen_bound);
                            c.L = x.L; c.A = x.A; c.key = rs_screen_key(c.L, c.A, aw);
                            c.screen_host = true;
                            ++rechecks;
                        }
                    }
                    printf("# alpha-screen GPU: %zu boundary rechecks (absolute key guard %.6g)\n",
                           rechecks, RS_SCREEN_CUT_GUARD);
                }
            }
#endif
            if (!device_screened) {
#pragma omp parallel for schedule(dynamic, 1)
                for (size_t i = 0; i < nproxy; i++) {
                    Cand &c = best[i];
                    if (screen_exact_size) {
                        const RsExact x = rs_exact_cell(P, R, c.h.u, c.h.v, c.t, screen_bound);
                        c.L = x.L;
                        c.A = x.A;
                        c.key = rs_screen_key(c.L, c.A, aw);
                        c.screen_host = true;
                    } else {
                        c.key = rs_alpha_screen_key(P, c.h.u, c.h.v, c.L, screen_bound, aw);
                        if (!std::isfinite(c.key)) c.key = INFINITY;
                    }
                }
            }
            size_t invalid = 0;
            for (auto &c : best) {
                if (!std::isfinite(c.key)) { c.key = INFINITY; ++invalid; }
            }
            if (invalid)
                printf("# alpha-screen: %zu nonfinite scores ranked last\n", invalid);
            std::sort(best.begin(), best.end(), order);
            trace_rank("alpha_screen_cut", nref);
            class_survivors("alpha screen", nref);
            t_screen = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
            printf("# alpha-screen: %zu proxy cells, alpha through %d, %zu to full refinement, %.6f s, size %s\n",
                   nproxy, screen_bound, nref, t_screen, screen_exact_size ? "exact" : "interpolated");
        }
        bool device_refined = false;
#ifdef __CUDACC__
        if (refine_gpu && gpu_start.screen_ready) {
            std::vector<Cand> finalists(best.begin(), best.begin() + nref);
            size_t fallbacks = 0;
            device_refined = rs_screen_gpu(finalists, R, P, cio_alpha_bound(), aw, dev,
                                           refine_check, fallbacks, session.screen, true);
            if (device_refined) {
                std::copy(finalists.begin(), finalists.end(), best.begin());
                printf("# full-refine GPU: %zu cells, alpha bound %lu, %zu host fallbacks; host size canonicalization\n",
                       nref, cio_alpha_bound(), fallbacks);
            }
        }
#endif
        if (!device_refined) {
#pragma omp parallel for schedule(dynamic, 1)
            for (size_t i = 0; i < nref; i++) {
                Cand &c = best[i];
                if (rerank) { /* of the polynomial written, content divided out */
                    if (screen_exact_size && c.screen_host) {
                        // The screen already ran the same size search from the saved
                        // start and removed content. Reuse it without a second search.
                        if (screen_bound != (int)cio_alpha_bound()) {
                            double log_content = 0;
                            c.A = cio_alpha_rot(&P, (long)c.h.u, (long)c.h.v, cio_alpha_bound(), &log_content);
                        }
                    } else {
                        const RsExact x = rs_exact_cell(P, R, c.h.u, c.h.v, c.t);
                        c.L = x.L;
                        c.A = x.A;
                    }
                    c.key = rs_screen_key(c.L, c.A, aw);
                } else {
                    c.L = rs_size(R, c.h.u, c.h.v, c.t, 1 << 10);
                    c.key = c.L + aw * (S.alpha0 - c.h.score);
                }
            }
        }
        best.resize(nref);
        // Optimize every member of the larger proxy pool BEFORE the output cut.
        // Keval validates each proposal and ranks its retained configuration.
        if (murphy) {
            fprintf(stderr, "s23_ropt: MurphyE refining %zu cells before the %zu-output cut "
                            "(K=%d, Keval=%d, maxeval=%d)\n", nref, nout, mc.K, mc.Keval, mc.maxeval);
            murphy_cells.resize(nref);
            size_t completed = 0;
            std::atomic<bool> audit_abort{false};
#ifdef __CUDACC__
            if (murphy_gpu) {
                const size_t workers = size_t(omp_get_max_threads());
                session.murphy_workers.resize(workers);
                for (auto &worker : session.murphy_workers) {
                    if (!worker || worker->device != dev || worker->check != murphy_check)
                        worker = std::make_unique<MurphyCudaEvaluator>(dev, murphy_check);
                    worker->begin_job(audit_abort);
                }
            }
#endif
#pragma omp parallel for schedule(dynamic, 1)
            for (size_t i = 0; i < nref; i++) {
                if (audit_abort.load(std::memory_order_relaxed)) continue;
                Cand &c = best[i];
                auto m = std::make_unique<MurphyCell>();
                MurphyEvaluator *evaluator = nullptr;
#ifdef __CUDACC__
                if (murphy_gpu) evaluator = session.murphy_workers[omp_get_thread_num()].get();
#endif
                m->refine(P, R, c.h.u, c.h.v, c.t, c.L, c.A, mc, evaluator);
                c.key = m->result.valid ? -m->result.score : INFINITY;
                murphy_cells[i] = std::move(m);
#pragma omp critical
                {
                    completed++;
                    if (completed % 64 == 0 || completed == nref)
                        fprintf(stderr, "s23_ropt: MurphyE %zu/%zu cells\n", completed, nref);
                }
            }
#ifdef __CUDACC__
            if (murphy_gpu) {
                size_t gpu_cells = 0, fallbacks = 0;
                double error = 0, alpha_error = 0;
                for (auto &worker : session.murphy_workers) worker->abort = nullptr;
                if (audit_abort.load()) return 2;
                for (const auto &m : murphy_cells) {
                    gpu_cells += m->result.gpu;
                    fallbacks += m->result.fallback;
                }
                for (const auto &worker : session.murphy_workers) {
                    if (worker->audit_failed) return 2;
                    error = std::max(error, worker->max_relative_error);
                    alpha_error = std::max(alpha_error, worker->max_alpha_error);
                }
                printf("# MurphyE GPU: %zu cells, %zu host fallbacks%s; max integral relative error %.12g; "
                       "max alpha error %.12g\n", gpu_cells, fallbacks,
                       murphy_check ? " (CPU integral and alpha audit)" : " (unchecked)", error, alpha_error);
            }
#endif
            std::vector<size_t> ranked;
            for (size_t i = 0; i < nref; i++) {
                const auto &m = *murphy_cells[i];
                murphy_count += m.attempted;
                murphy_skipped += !m.attempted;
                murphy_failed += m.failed();
                murphy_accepted += m.result.valid && m.result.accepted;
                murphy_limited += m.result.limited;
                if (m.failed())
                    fprintf(stderr, "s23_ropt: MurphyE failed for u %lld v %lld; valid cells retained\n",
                            (long long)best[i].h.u, (long long)best[i].h.v);
                if (m.result.valid) ranked.push_back(i);
            }
            std::sort(ranked.begin(), ranked.end(), [&](size_t a, size_t b) { return order(best[a], best[b]); });
            if (trace.active()) {
                const auto it = std::find_if(ranked.begin(), ranked.end(), [&](size_t i) {
                    return trace.matches(best[i].h.u, best[i].h.v);
                });
                trace.stage("murphy_evaluation", it != ranked.end());
                trace.rank("murphy_cut", it == ranked.end() ? 0 : (size_t)(it - ranked.begin()) + 1,
                           nout, ranked.size());
            }
            ranked.resize(std::min(ranked.size(), nout));
            std::vector<Cand> chosen;
            std::vector<std::unique_ptr<MurphyCell>> selected;
            for (size_t i : ranked) {
                chosen.push_back(best[i]);
                selected.push_back(std::move(murphy_cells[i]));
            }
            best = std::move(chosen);
            murphy_cells = std::move(selected);
        } else {
            std::sort(best.begin(), best.end(), order);
            trace_rank(rerank ? "exact_alpha_cut" : "exact_size_cut", nout);
            best.resize(std::min(best.size(), nout));
        }
        t_refine = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
        class_survivors("finalists", best.size());
    }

    if (pool_in)
        printf("# s23_ropt: %s, saved-pool replay", path);
    else
        printf("# s23_ropt: %s, %s: %.3g cells, sieve %.2f s (%.3g cells/s, %s)", path, window ? "window" : "search",
               ncells, t_sieve, ncells / t_sieve, use_cpu ? "CPU" : "GPU");
    if (!window)
        printf(", size model %.2f s, refine %.2f s", t_model, t_refine);
    if (screen_options)
        printf(" (includes alpha screen %.2f s)", t_screen);
    if (!use_cpu)
        printf(", waited %.2f s for the GPU context", t_start);
    printf("\n");
    if (!window) {
        int64_t ulo = INT64_MAX, uhi = INT64_MIN;
        size_t nl = 0;
        for (auto &kv : lines)
            if (kv.second.v_lo <= kv.second.v_hi) {
                ulo = std::min(ulo, kv.first);
                uhi = std::max(uhi, kv.first);
                nl++;
            }
        if (!nl)
            ulo = uhi = 0;
        if (!pool_in) {
            printf("# size model: best line minimum %.4f; %zu lines within %.2f (u %lld..%lld), %zu lines scanned",
                   L_best, nl, budget_used, (long long)ulo, (long long)uhi, lines.size());
            if (lines_dropped || lines_trimmed)
                printf("; -maxcells dropped %zu lines and trimmed %zu", lines_dropped, lines_trimmed);
            printf("; line scan %.2f s, bands %.2f s at %d budgets, knots %.2f s\n", t_scan, t_bands, n_bands, t_knots);
        }
        if (murphy)
            printf("# best %d by validated MurphyE: %zu optimized, %zu accepted, %zu limited, %zu failed, %zu skipped before MurphyE; "
                   "Bf %.17g Bg %.17g area %.17g K %d Keval %d maxeval %d\n", top, murphy_count,
                   murphy_accepted, murphy_limited, murphy_failed, murphy_skipped, mc.Bf, mc.Bg, mc.area, mc.K, mc.Keval, mc.maxeval);
        else if (screen_options)
            printf("# best %d by Ea = lognorm + %.2f alpha (CADO's, p <= %lu; content divided out), "
                   "after alpha-screen %d %d and full refinement of at most %d cells:\n",
                   top, aw, cio_alpha_bound(), screen_count, screen_bound, std::max({rerank, refine, top}));
        else if (rerank)
            printf("# best %d by Ea = lognorm + %.2f alpha (CADO's, p <= %lu; content divided out), of the best %d by "
                   "the proxy E = lognorm + %.2f alpha_s (%s; exact translation):\n",
                   top, aw, cio_alpha_bound(), std::max({rerank, refine, top}), aw, alpha_what);
        else
            printf("# best %d by E = lognorm + %.2f alpha_s (%s; exact translation):\n", top, aw, alpha_what);
    } else
        printf("# best %d by score (alpha_s = %.4f - score, %s):\n", top, S.alpha0, alpha_what);
    int bad = 0;
    for (int i = 0; i < top && i < (int)best.size(); i++) {
        const Cand &c = best[i];
        const bool differs = check && fabs(rs_cell_score_brute(S, c.h.u, c.h.v) - c.h.score) > 1e-4;
        bad += differs;
        if (window)
            printf("u %6lld  v %16lld  score %.6f  alpha_s %.4f%s\n", (long long)c.h.u, (long long)c.h.v,
                   c.h.score, S.alpha0 - c.h.score, differs ? "  BRUTE FORCE DIFFERS" : "");
        else if (murphy) {
            const auto &m = murphy_cells[i]->result;
            printf("u %6lld  v %16lld  t %14lld  MurphyE %.17g  skew %.17g  lognorm %.6f  lognorm_at_murphy_skew %.6f  alpha %.6f%s\n",
                   (long long)c.h.u, (long long)c.h.v, (long long)(c.t + m.translation), m.score,
                   m.skew, c.L, m.lognorm, c.A, differs ? "  BRUTE FORCE DIFFERS" : "");
        } else {
            printf("u %6lld  v %16lld  t %14lld  %s %.4f  lognorm %.4f", (long long)c.h.u, (long long)c.h.v,
                   (long long)c.t, rerank ? "Ea" : "E", c.key, c.L);
            if (rerank)
                printf("  alpha %.4f", c.A);
            printf("  alpha_s %.4f%s\n", S.alpha0 - c.h.score, differs ? "  BRUTE FORCE DIFFERS" : "");
        }
    }
    if (!window) {
        std::vector<Cand> lb;
        for (auto &kv : line_best)
            lb.push_back(kv.second);
        std::sort(lb.begin(), lb.end(), order);
        printf("# best cell of the best 10 lines (E with interpolated lognorm):\n");
        for (int i = 0; i < 10 && i < (int)lb.size(); i++)
            printf("u %6lld  v %16lld  E~ %.4f  lognorm~ %.4f  alpha_s %.4f\n", (long long)lb[i].h.u,
                   (long long)lb[i].h.v, lb[i].key, lb[i].L, S.alpha0 - lb[i].h.score);
    }
    if (check) {
        bad += check_hits(S, sample);
        printf("# check: %zu random cells of the sieve's output + the printed ones vs brute force: %d differ\n",
               sample.size(), bad);
    }
    if (out_path && !window) {
        FILE *o = fopen(out_path, "w");
        if (!o) {
            perror(out_path);
            return 2;
        }
        cio_poly q;
        cio_init(&q);
        size_t nwritten = 0, nskipped = 0;
        bool trace_written = false;
        for (size_t j = 0; j < best.size(); j++) {
            const Cand &c = best[j];
            unsigned long content;
            if (murphy) {
                cio_set(&q, &murphy_cells[j]->poly);
                content = murphy_cells[j]->content;
            } else if (!rs_written_poly(q, P, R, c.h.u, c.h.v, c.t, content)) {
                nskipped++;
                continue;
            }
            nwritten++;
            trace_written |= trace.matches(c.h.u, c.h.v);
            if (murphy) {
                const auto &m = murphy_cells[j]->result;
                fprintf(o, "# u %lld v %lld t %lld MurphyE %.17g lognorm %.6f lognorm_at_murphy_skew %.6f alpha_s %.6f "
                           "control %.17g proposal %.17g accepted %d calls %d limited %d",
                        (long long)c.h.u, (long long)c.h.v, (long long)(c.t + m.translation),
                        m.score, c.L, m.lognorm, S.alpha0 - c.h.score, m.control, m.proposal,
                        m.accepted, m.calls, m.limited);
            } else
                fprintf(o, "# u %lld v %lld t %lld %s %.6f lognorm %.6f alpha_s %.6f", (long long)c.h.u,
                        (long long)c.h.v, (long long)c.t, rerank ? "Ea" : "E", c.key, c.L, S.alpha0 - c.h.score);
            if (rerank)
                fprintf(o, " alpha %.6f", c.A);
            fprintf(o, "%s\n", content != 1 ? " content divided" : "");
            gmp_fprintf(o, "n: %Zd\nY0: %Zd\nY1: %Zd\n", q.n, q.g[0], q.g[1]);
            if (murphy)
                fprintf(o, "skew: %.17g\n", murphy_cells[j]->result.skew);
            for (int i = 0; i <= 5; i++)
                gmp_fprintf(o, "c%d: %Zd\n", i, q.f[i]);
            fprintf(o, "\n");
        }
        fclose(o);
        cio_clear(&q);
        printf("# wrote %zu polynomials to %s", nwritten, out_path);
        if (nskipped)
            printf(" (%zu skipped: translation out of range)", nskipped);
        printf("\n");
        trace.stage("export", trace_written);
    }
    if (window)
        trace_rank("window_top", (size_t)top);
    trace.finish("retained");
    // Status 3 means a completed partial export; 1 remains a failed result.
    return bad || (murphy && best.empty()) ? 1 : murphy_failed ? 3 : 0;
}

int main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "-capabilities")) {
        printf("ropt-pool-v1 partial-murphy-exit-3 alpha-screen-v1 alpha-screen-exact-size-v1 ropt-batch-v1 ropt-batch-events-v1"
#ifdef __CUDACC__
               " alpha-screen-gpu-v1 full-refine-gpu-v1 murphy-gpu-v1"
#endif
               "\n");
        return 0;
    }
    RoptSession session;
    if (argc >= 2 && !strcmp(argv[1], "-batch")) {
        int events_fd = -1;
        if (argc == 5 && !strcmp(argv[3], "-batch-events-fd")) {
            char *end = nullptr;
            errno = 0;
            const long fd = strtol(argv[4], &end, 10);
            if (errno || !*argv[4] || *end || fd <= STDERR_FILENO || fd > INT_MAX) {
                fprintf(stderr, "s23_ropt: invalid batch events descriptor\n");
                return 2;
            }
            events_fd = int(fd);
        } else if (argc != 3) {
            fprintf(stderr, "s23_ropt: use -batch FILE [-batch-events-fd FD]; each manifest row supplies all job options\n");
            return 2;
        }
        const int status = ropt_batch(argv[2], argv[0],
            [&](int n, char **a, RoptJobPaths &p) { return ropt_job(n, a, session, &p); },
            [&](int n, char **a) { return ropt_job(n, a, session); }, events_fd);
#ifdef __CUDACC__
        printf("# batch GPU resources: device setups %zu, buffer allocations %zu, prime uploads %zu\n",
               session.screen.setups, session.screen.allocations, session.screen.prime_uploads);
#endif
        return status;
    }
    return ropt_job(argc, argv, session);
}
