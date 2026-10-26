/* Exact per-block parity against cpu_block, including vector tails, ties, signed
 * coordinates, content/projective tables, strides and bounded-cache reuse. */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "rseed.h"
#include "rblock.h"
using namespace s23;
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(2); } } while (0)
#include "rsieve_cuda.h"

int main(int argc, char **argv)
{
    if (argc < 2) return 2;
    size_t checked = 0;
    for (int f = 1; f < argc; f++) {
        FILE *fh = fopen(argv[f], "r");
        if (!fh) return 2;
        cio_poly poly;
        cio_init(&poly);
        if (cio_read_next(fh, &poly) != 1) return 2;
        fclose(fh);
        RsSeed seed;
        if (!rs_seed_from_poly(seed, poly)) return 2;
        for (int projective : {0, 1}) {
            seed.proj = projective;
            std::vector<float> pt, table(seed.ntab), acc(1 << 16);
            rs_proj_tables(seed, pt);
            RsSeed *sg;
            float *pg;
            Blk *bg;
            Hit *hg;
            CUDA_CHECK(cudaMalloc(&sg, sizeof(seed)));
            CUDA_CHECK(cudaMalloc(&pg, std::max<size_t>(1, pt.size()) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bg, 32 * sizeof(Blk)));
            CUDA_CHECK(cudaMalloc(&hg, 32 * K_MAX * sizeof(Hit)));
            CUDA_CHECK(cudaMemcpy(sg, &seed, sizeof(seed), cudaMemcpyHostToDevice));
            if (!pt.empty()) CUDA_CHECK(cudaMemcpy(pg, pt.data(), pt.size()*sizeof(float), cudaMemcpyHostToDevice));
            const size_t shmem = seed.ntab * sizeof(float4);
            CUDA_CHECK(cudaFuncSetAttribute(rs_vector_sieve_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
            CUDA_CHECK(cudaFuncSetAttribute(rs_vector_sieve_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
            RsVectorWorkspace workspace;
            std::vector<Blk> overflow(1025);
            for (size_t j = 0; j < overflow.size(); j++) overflow[j] = {(int64_t)j, 0, 1, 0, 0, 1};
            if (workspace.prepare(seed, sg, pg, overflow)) {
                fprintf(stderr, "cache accepted too many rows\n"); return 1;
            }
            for (bool strided : {false, true}) {
                std::vector<Blk> blocks;
                const uint32_t lengths[] = {1, 3, 31, 32, 63, 65, 1023, 4097, 65535, 65536, 65537, 262147};
                const uint32_t strides[] = {1, 2, 6, 8, 128, 2520, 65536};
                for (int j = 0; j < 32; j++) {
                    int64_t u = j % 2 ? INT64_MAX-j : INT64_MIN+j;
                    int64_t v = j % 3 == 0 ? INT64_MIN+1000000000000ll :
                                j % 3 == 1 ? INT64_MAX-1000000000000ll : -123456789;
                    blocks.push_back({u, v, lengths[j % 12], 55.0f,
                        j % 3 == 0 ? 0.0f : j % 3 == 1 ? 1e-7f : -1e-7f,
                        strided ? strides[j % 7] : 1});
                }
                CUDA_CHECK(cudaMemcpy(bg, blocks.data(), blocks.size()*sizeof(Blk), cudaMemcpyHostToDevice));
                for (int k : {1, 3, 8, 16}) for (float aw : {0.0f, 1.3f}) {
                    if (!workspace.prepare(seed, sg, pg, blocks)) return 1;
                    if (strided)
                        rs_vector_sieve_kernel<true><<<blocks.size(), RS_VECTOR_NT, shmem>>>(
                            sg, workspace.tables(), workspace.row_ids(), bg, k, aw, hg);
                    else
                        rs_vector_sieve_kernel<false><<<blocks.size(), RS_VECTOR_NT, shmem>>>(
                            sg, workspace.tables(), workspace.row_ids(), bg, k, aw, hg);
                    CUDA_CHECK(cudaGetLastError());
                    std::vector<Hit> got(blocks.size()*k);
                    CUDA_CHECK(cudaMemcpy(got.data(), hg, got.size()*sizeof(Hit), cudaMemcpyDeviceToHost));
                    for (size_t j = 0; j < blocks.size(); j++) {
                        Hit expected[K_MAX];
                        cpu_block(seed, pt.data(), blocks[j], k, aw, expected, table, acc);
                        for (int rank = 0; rank < k; rank++) {
                            const Hit &a = expected[rank], &b = got[j*k+rank];
                            if (memcmp(&a.score, &b.score, sizeof(float)) || memcmp(&a.key, &b.key, sizeof(float)) ||
                                a.u != b.u || a.v != b.v) {
                                fprintf(stderr, "%s: proj %d strided %d block %zu k %d aw %g rank %d differs\n",
                                    argv[f], projective, strided, j, k, aw, rank);
                                return 1;
                            }
                            checked++;
                        }
                    }
                }
            }
            workspace.release();
            CUDA_CHECK(cudaFree(hg)); CUDA_CHECK(cudaFree(bg));
            CUDA_CHECK(cudaFree(pg)); CUDA_CHECK(cudaFree(sg));
        }
        cio_clear(&poly);
        printf("%s: dense/strided, affine/projective, tails and ties match CPU bits\n", argv[f]);
    }
    printf("%zu exact retained-hit comparisons passed\n", checked);
    return 0;
}
