/* Lossless four-cell root sieve. Each table entry holds four consecutive selected
 * cells, pre-shifted for this block's v0/stride. One aligned float4 shared lookup
 * replaces four scalar lookups; each cell still sums primes in the original order.
 * The expanded tables need about 76 KiB, so use 1024 threads to keep 32 warps
 * resident. Small blocks and devices without enough shared memory use the scalar
 * kernel. Raw line tables are built once per distinct u in a bounded launch cache.
 */
#pragma once
#include "rblock.h"
#include <map>
#include <vector>

static constexpr int RS_VECTOR_NT = 1024;
static constexpr int RS_VECTOR_CPT = 8; // four cells per accumulator

/* No allocation grows with total search size. A launch with more distinct lines
 * than fit here can use the scalar path without changing any block boundaries. */
class RsVectorWorkspace {
    static constexpr size_t max_rows = 1024, max_blocks = 1u << 16;
    int64_t *linesg = nullptr;
    uint32_t *rowsg = nullptr;
    float *tablesg = nullptr;
    bool disabled = false;
    std::vector<int64_t> lines;
    std::vector<uint32_t> rows;
    std::map<int64_t, uint32_t> index;
public:
    RsVectorWorkspace() = default;
    RsVectorWorkspace(const RsVectorWorkspace &) = delete;
    RsVectorWorkspace &operator=(const RsVectorWorkspace &) = delete;
    ~RsVectorWorkspace() { release(); }
    void release()
    {
        if (tablesg) cudaFree(tablesg);
        if (rowsg) cudaFree(rowsg);
        if (linesg) cudaFree(linesg);
        tablesg = nullptr; rowsg = nullptr; linesg = nullptr;
    }
    bool prepare(const RsSeed &S, const RsSeed *Sg, const float *PTg, const std::vector<Blk> &blocks);
    const float *tables() const { return tablesg; }
    const uint32_t *row_ids() const { return rowsg; }
};

__global__ void rs_vector_tables_kernel(const RsSeed *Sg, const float *PT, const int64_t *lines, float *tables)
{
    const RsSeed &S = *Sg;
    const int i = threadIdx.x;
    if (i < RS_NPRIMES) {
        const int64_t q = S.pr[i].qmax;
        const int64_t u = lines[blockIdx.x];
        rs_prime_table(tables + (size_t)blockIdx.x * S.ntab + S.pr[i].off,
                       S, PT, i, (uint32_t)(((u % q) + q) % q));
    }
}

inline bool RsVectorWorkspace::prepare(const RsSeed &S, const RsSeed *Sg, const float *PTg,
                                      const std::vector<Blk> &blocks)
{
    if (disabled || blocks.size() > max_blocks) return false;
    lines.clear(); rows.clear(); index.clear();
    for (const Blk &b : blocks) {
        auto row = index.find(b.u);
        if (row == index.end()) {
            if (lines.size() == max_rows) return false;
            row = index.emplace(b.u, (uint32_t)lines.size()).first;
            lines.push_back(b.u);
        }
        rows.push_back(row->second);
    }
    if (!tablesg) {
        if (cudaMalloc(&linesg, max_rows * sizeof(int64_t)) != cudaSuccess ||
            cudaMalloc(&rowsg, max_blocks * sizeof(uint32_t)) != cudaSuccess ||
            cudaMalloc(&tablesg, max_rows * S.ntab * sizeof(float)) != cudaSuccess) {
            release();
            cudaGetLastError(); // failed optional allocation: scalar kernel remains available
            disabled = true;
            return false;
        }
    }
    CUDA_CHECK(cudaMemcpy(linesg, lines.data(), lines.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rowsg, rows.data(), rows.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    rs_vector_tables_kernel<<<(unsigned)lines.size(), 64>>>(Sg, PTg, linesg, tablesg);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <bool Strided>
__global__ void __launch_bounds__(RS_VECTOR_NT) rs_vector_sieve_kernel(const RsSeed *Sg, const float *tables, const uint32_t *rows,
                                                  const Blk *blks, int k, float aw,
                                                   Hit *out)
{
    extern __shared__ float4 T4[];
    __shared__ float wkey[RS_VECTOR_NT / 32][K_MAX], sel_key[K_MAX];
    __shared__ uint32_t woff[RS_VECTOR_NT / 32][K_MAX], sel_off[K_MAX];
    const RsSeed &S = *Sg;
    const Blk B = blks[blockIdx.x];
    const float *line = tables + (size_t)rows[blockIdx.x] * S.ntab;
    for (unsigned i = 0; i < RS_NPRIMES; i++) {
        const uint32_t q = S.pr[i].qmax;
        const uint32_t start = (uint32_t)(((B.v0 % (int64_t)q) + q) % q);
        const uint32_t stride = Strided ? B.step % q : 1;
        for (unsigned j = threadIdx.x; j < q; j += blockDim.x) {
            float4 v;
            v.x = line[S.pr[i].off + (start + (4*j+0)*stride) % q];
            v.y = line[S.pr[i].off + (start + (4*j+1)*stride) % q];
            v.z = line[S.pr[i].off + (start + (4*j+2)*stride) % q];
            v.w = line[S.pr[i].off + (start + (4*j+3)*stride) % q];
            T4[S.pr[i].off+j] = v;
        }
    }
    __syncthreads();

    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    /* the warp's best k cells by cell_before, one per lane (lane m < k holds the m-th);
     * (tk, to) is the k-th, the bar a cell must pass to get in. Cells that pass are rare
     * after the first few chunks, so the cost is one vote per 32 cells */
    float ek = -INFINITY, tk = -INFINITY;
    uint32_t eo = UINT32_MAX, to = UINT32_MAX;
    for (uint32_t c0 = warp * 128 * RS_VECTOR_CPT; c0 < B.len; c0 += nwarps * 128 * RS_VECTOR_CPT) {
        float4 acc[RS_VECTOR_CPT];
#pragma unroll
        for (int j = 0; j < RS_VECTOR_CPT; j++)
            acc[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const uint32_t o = c0/4 + lane;
        for (int i = 0; i < RS_NPRIMES; i++) {
            const uint32_t q = S.pr[i].qmax;
            const uint32_t step = 32 % q;
            const float4 *Ti = T4 + S.pr[i].off;
            uint32_t idx = o % q;
#pragma unroll
            for (int j = 0; j < RS_VECTOR_CPT; j++) {
                const float4 v = Ti[idx];
                acc[j].x += v.x;
                acc[j].y += v.y;
                acc[j].z += v.z;
                acc[j].w += v.w;
                idx += step;
                if (idx >= q)
                    idx -= q;
            }
        }
#pragma unroll
        for (int j = 0; j < RS_VECTOR_CPT; j++) {
#pragma unroll
            for (int component = 0; component < 4; component++) {
                const uint32_t off = c0 + lane*4 + 128*j + component;
                const float value = component == 0 ? acc[j].x : component == 1 ? acc[j].y : component == 2 ? acc[j].z : acc[j].w;
                const float key = aw * value - (B.L0 + B.dL * (float)off);
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
    }
    if ((int)lane < k) {
        wkey[warp][lane] = ek;
        woff[warp][lane] = eo;
    }
    __syncthreads();
    /* the block's best k: merge the warps' sorted lists */
    if (threadIdx.x == 0) {
        int h[RS_VECTOR_NT / 32];
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
                const float4 value = T4[S.pr[i].off + ((off / 4) % q)];
                a += off % 4 == 0 ? value.x : off % 4 == 1 ? value.y : off % 4 == 2 ? value.z : value.w;
            }
            v = B.v0 + (int64_t)off * (Strided ? B.step : 1);
        }
        out[(size_t)blockIdx.x * k + threadIdx.x] = {a, key, B.u, v};
    }
}
