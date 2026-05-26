#ifndef COLLISION_COMMON_CUH
#define COLLISION_COMMON_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint32_t uint32;
typedef uint64_t uint64;

/* Bucket count chosen so per-bucket hash tables fit in shared memory with
   healthy occupancy. At N=50M:
     - 4096 buckets ⇒ ilog2≈18 ⇒ 96 KB smem ⇒ 1 block/SM (kills occupancy)
     - 16384 buckets ⇒ ilog2≈16 ⇒ 24 KB smem ⇒ 4 blocks/SM (good)
   Make compile-time tunable so the sweep can probe other values. */
#ifndef NUM_BUCKETS
#define NUM_BUCKETS 16384u
#endif
#ifndef LOG2_NUM_BUCKETS
#define LOG2_NUM_BUCKETS  14u
#endif

#define BUCKET_MASK       (NUM_BUCKETS - 1u)
#define MAX_FILTER_ITERS  20

/* Mixing constant for the optional bucket-hash mode. Matches Knuth's
   multiplicative hash (golden-ratio reciprocal). */
#define BUCKET_HASH_MIX   0x9E3779B97F4A7C15ULL

__host__ __device__ static inline uint32
compute_bucket(uint64 key, int hash_mode) {
    if (hash_mode) {
        return (uint32)((key * BUCKET_HASH_MIX) >> (64 - LOG2_NUM_BUCKETS));
    }
    return (uint32)(key & BUCKET_MASK);
}

#define CUDA_TRY(call) do {                                                   \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                  \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define CUDA_CHECK_LAST() do {                                                \
        cudaError_t _e = cudaGetLastError();                                  \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA kernel error %s:%d: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(_e));                        \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#endif
