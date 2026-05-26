#include "collision_common.cuh"

/* Phase A: bucket keys into NUM_BUCKETS storage segments.

   Single-pass design: each bucket occupies a fixed slot range
   [bucket * max_per_bucket, (bucket+1) * max_per_bucket) in bucket_storage.
   One scatter pass with warp-aggregated atomicAdd on bucket_count[] places
   items into their bucket's slot range; bucket_count[bucket] gives the
   actual end. Avoids the count + scan + scatter three-pass design, saving
   ~1.5-2ms at N=50M.

   Trade-off: bucket_storage allocation is NUM_BUCKETS * max_per_bucket
   instead of N. For N=50M with max_per_bucket=5000 → 640 MB (vs 400 MB).
   Headroom is set conservatively at init time from max_n + 6σ.

   Storage layout: each slot holds (key >> LOG2_NUM_BUCKETS), matching
   collision_ref.c:216. Phase B hashes from bit 0 of these stripped values. */

__global__ void scatter_one_pass_kernel(
        const uint64 * __restrict__ keys,
        uint32 N,
        uint32 max_per_bucket,
        int hash_mode,
        uint32 * __restrict__ bucket_count,         /* zeroed by caller */
        uint32 * __restrict__ overflow_flag,        /* zeroed by caller */
        uint64 * __restrict__ bucket_storage) {

    uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    uint64 k = keys[tid];
    uint32 bucket = compute_bucket(k, hash_mode);
    /* Store the FULL key (not key >> LOG2_NUM_BUCKETS). With --bucket-hash
       the bucket id isn't the low bits of the key, so reconstructing the
       key from (stripped, bucket) in Phase B doesn't work. Storing the
       full key costs nothing (we're already uint64) and supports both
       bucketing modes. Phase B hashes from bit LOG2_NUM_BUCKETS onward,
       so the bucket-correlated low bits (in low-bits mode) don't waste
       hash discrimination. */

    uint32 lane = threadIdx.x & 31u;
    uint32 mask = __match_any_sync(0xFFFFFFFFu, bucket);
    uint32 leader = __ffs(mask) - 1u;
    uint32 count  = __popc(mask);

    uint32 base = 0;
    if (lane == leader) {
        base = atomicAdd(&bucket_count[bucket], count);
    }
    base = __shfl_sync(0xFFFFFFFFu, base, leader);
    uint32 within = __popc(mask & ((1u << lane) - 1u));
    uint32 slot = base + within;

    if (slot >= max_per_bucket) {
        atomicOr(overflow_flag, 1u);
        return;
    }

    bucket_storage[(size_t)bucket * max_per_bucket + slot] = k;
}

extern "C" void launch_scatter_one_pass(
        const uint64 *keys, uint32 N, uint32 max_per_bucket,
        int hash_mode,
        uint32 *bucket_count, uint32 *overflow_flag,
        uint64 *bucket_storage, cudaStream_t stream) {
    CUDA_TRY(cudaMemsetAsync(bucket_count, 0,
                             NUM_BUCKETS * sizeof(uint32), stream));
    CUDA_TRY(cudaMemsetAsync(overflow_flag, 0, sizeof(uint32), stream));
    dim3 block(256);
    dim3 grid((N + block.x - 1u) / block.x);
    scatter_one_pass_kernel<<<grid, block, 0, stream>>>(
            keys, N, max_per_bucket, hash_mode,
            bucket_count, overflow_flag, bucket_storage);
    CUDA_CHECK_LAST();
}
