#include "collision_common.cuh"

/* Phase C: verify candidates against original keys.

   Mirrors collision_ref.c:384-429. After Phase B emits a candidate list with
   bucket ids re-attached, we need to:
     1. Sort the candidates (so duplicates are adjacent)
     2. Keep only the values that appear at least twice
     3. Build a secondary hash table over those unique values
     4. Walk the original key array; for each original key, look up the
        secondary hash. Each match is one near-collision.

   This counts EACH occurrence of a duplicated key, matching the metric
   collision_ref.c:494-497 reports and what qsort_collision_count returns. */

/* --- Dedup kernel: keep candidates whose value appears in a run of ≥2. ----
   Input:  sorted_keys[csize]
   Output: dedup_keys[csize_out] — unique values that had ≥2 occurrences,
           one entry per such value.
   collision_ref.c:386-389 picks the LAST element of each duplicated run; we
   pick the FIRST (logically equivalent for the downstream hash). */

__global__ void dedup_kernel(
        const uint64 * __restrict__ sorted_keys, uint32 csize,
        uint64 * __restrict__ dedup_keys, uint32 * __restrict__ dedup_cnt) {

    uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= csize) return;

    uint64 k = sorted_keys[tid];
    bool is_first_of_run = (tid == 0) || (sorted_keys[tid - 1] != k);
    bool has_neighbor    = (tid + 1 < csize) && (sorted_keys[tid + 1] == k);
    if (is_first_of_run && has_neighbor) {
        uint32 pos = atomicAdd(dedup_cnt, 1u);
        dedup_keys[pos] = k;
    }
}

/* --- Phase C secondary hash build ---
   ilog2 derived from dedup_cnt; D[] and S[] sized to it. */

__device__ static inline uint32
phase_c_ilog2(uint32 csize) {
    if (csize <= 1) return 5;
    uint32 lg = 32u - __clz(csize);   /* ceil-ish */
    uint32 il = max(5u, lg + 4u);
    if (il > 31u) il = 31u;
    return il;
}

__global__ void
count_secondary_kernel(
        const uint64 * __restrict__ dedup_keys, uint32 csize,
        uint32 hash_value,
        uint32 * __restrict__ D,    /* dsize uint32, zeroed by caller */
        uint32 * __restrict__ S) {  /* ssize uint32, zeroed by caller */

    uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= csize) return;

    uint64 k = dedup_keys[tid];
    uint32 hv = (uint32)(k & hash_value);
    /* Write count at D[hv] (not hv+1) so exclusive-scan yields D_scan[hv]
       = start of hv's slot, D_scan[hv+1] = end. scan_originals walks this
       half-open interval. */
    atomicAdd(&D[hv], 1u);
    atomicOr(&S[hv >> 5], 1u << (hv & 31u));
}

__global__ void
scatter_secondary_kernel(
        const uint64 * __restrict__ dedup_keys, uint32 csize,
        uint32 hash_value,
        uint32 * __restrict__ D_pos,  /* mutable cursor: starts == D after scan */
        uint64 * __restrict__ X) {

    uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= csize) return;

    uint64 k = dedup_keys[tid];
    uint32 hv = (uint32)(k & hash_value);
    uint32 slot = atomicAdd(&D_pos[hv], 1u);
    X[slot] = k;
}

/* --- Scan originals: count hits per original key. --- */

__global__ void
scan_originals_kernel(
        const uint64 * __restrict__ keys, uint32 N,
        uint32 hash_value,
        const uint32 * __restrict__ S,
        const uint32 * __restrict__ D,
        const uint64 * __restrict__ X,
        uint32 * __restrict__ hits_out) {

    uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    uint64 k = keys[tid];
    uint32 hv = (uint32)(k & hash_value);
    if ((S[hv >> 5] & (1u << (hv & 31u))) == 0) return;

    /* D is the exclusive-scan output of the count array. D[hv] is the start
       of hv's slot, D[hv+1] is the end. (collision_ref.c reaches the same answer
       via an inclusive-scan + post-mutation by the X-build pass; we use the
       cleaner immutable-D path here.) */
    uint32 lo = D[hv];
    uint32 hi = D[hv + 1u];
    for (uint32 j = lo; j < hi; j++) {
        if (X[j] == k) {
            atomicAdd(hits_out, 1u);
            return;
        }
    }
}

__device__ static uint32
gcd32_device(uint32 a, uint32 b) {
    while (b != 0) {
        uint32 t = a % b;
        a = b;
        b = t;
    }
    return a;
}

__device__ static inline int
find_candidate_slot(uint64 k, uint32 hash_value,
                    const uint32 * __restrict__ S,
                    const uint32 * __restrict__ D,
                    const uint64 * __restrict__ X,
                    uint32 *slot_out) {
    uint32 hv = (uint32)(k & hash_value);
    if ((S[hv >> 5] & (1u << (hv & 31u))) == 0)
        return 0;

    uint32 lo = D[hv];
    uint32 hi = D[hv + 1u];
    for (uint32 j = lo; j < hi; j++) {
        if (X[j] == k) {
            *slot_out = j;
            return 1;
        }
    }
    return 0;
}

__global__ void
count_matched_values_kernel(
        const uint64 * __restrict__ keys,
        const uint32 * __restrict__ values,
        uint32 N,
        uint32 hash_value,
        const uint32 * __restrict__ S,
        const uint32 * __restrict__ D,
        const uint64 * __restrict__ X,
        uint32 dedup_count,
        uint32 * __restrict__ value_counts,
        uint32 * __restrict__ match_total,
        uint32 * __restrict__ value_overflow,
        uint32 value_cap) {

    uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    if (keys[tid] == 0)
        return;

    uint32 slot;
    if (find_candidate_slot(keys[tid], hash_value, S, D, X, &slot)) {
        if (slot < dedup_count) {
            uint32 total_pos = atomicAdd(match_total, 1u);
            atomicAdd(&value_counts[slot], 1u);
            if (total_pos >= value_cap)
                atomicOr(value_overflow, 1u);
        }
    }
    (void)values;
}

__global__ void
scatter_matched_values_kernel(
        const uint64 * __restrict__ keys,
        const uint32 * __restrict__ values,
        uint32 N,
        uint32 hash_value,
        const uint32 * __restrict__ S,
        const uint32 * __restrict__ D,
        const uint64 * __restrict__ X,
        uint32 dedup_count,
        uint32 * __restrict__ value_cursor,
        uint32 * __restrict__ matched_values,
        uint32 value_cap) {

    uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    if (keys[tid] == 0)
        return;

    uint32 slot;
    if (find_candidate_slot(keys[tid], hash_value, S, D, X, &slot)) {
        if (slot < dedup_count) {
            uint32 pos = atomicAdd(&value_cursor[slot], 1u);
            if (pos < value_cap)
                matched_values[pos] = values[tid];
        }
    }
}

__global__ void
count_msieve_pairs_kernel(
        const uint32 * __restrict__ matched_values,
        const uint32 * __restrict__ value_offsets,
        uint32 dedup_count,
        uint32 pshift,
        uint32 * __restrict__ hits_out) {

    uint32 slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= dedup_count) return;

    uint32 lo = value_offsets[slot];
    uint32 hi = value_offsets[slot + 1u];
    uint32 mask = (pshift >= 32u) ? 0xFFFFFFFFu : ((1u << pshift) - 1u);
    uint32 local_hits = 0;

    for (uint32 i = lo; i + 1u < hi; i++) {
        uint32 v1 = matched_values[i];
        uint32 q1 = (pshift >= 32u) ? 0u : (v1 >> pshift);
        uint32 p1 = v1 & mask;

        for (uint32 j = i + 1u; j < hi; j++) {
            uint32 v2 = matched_values[j];
            uint32 q2 = (pshift >= 32u) ? 0u : (v2 >> pshift);
            uint32 p2 = v2 & mask;
            if (q1 == q2 && gcd32_device(p1, p2) == 1u)
                local_hits++;
        }
    }

    if (local_hits)
        atomicAdd(hits_out, local_hits);
}

/* C-callable launchers. The CUB sort + scan calls are driven from the host
   wrapper (bench_gpu.cu) so we keep those out of the kernel TU. */

extern "C" void launch_dedup(
        const uint64 *sorted_keys, uint32 csize,
        uint64 *dedup_keys, uint32 *dedup_cnt, cudaStream_t stream) {
    CUDA_TRY(cudaMemsetAsync(dedup_cnt, 0, sizeof(uint32), stream));
    if (csize == 0) return;
    dim3 block(256), grid((csize + block.x - 1u) / block.x);
    dedup_kernel<<<grid, block, 0, stream>>>(sorted_keys, csize,
                                              dedup_keys, dedup_cnt);
    CUDA_CHECK_LAST();
}

extern "C" void launch_count_secondary(
        const uint64 *dedup_keys, uint32 csize, uint32 hash_value,
        uint32 *D, uint32 dsize, uint32 *S, uint32 ssize,
        cudaStream_t stream) {
    CUDA_TRY(cudaMemsetAsync(D, 0, dsize * sizeof(uint32), stream));
    CUDA_TRY(cudaMemsetAsync(S, 0, ssize * sizeof(uint32), stream));
    if (csize == 0) return;
    dim3 block(256), grid((csize + block.x - 1u) / block.x);
    count_secondary_kernel<<<grid, block, 0, stream>>>(
            dedup_keys, csize, hash_value, D, S);
    CUDA_CHECK_LAST();
}

extern "C" void launch_scatter_secondary(
        const uint64 *dedup_keys, uint32 csize, uint32 hash_value,
        uint32 *D_pos, uint64 *X, cudaStream_t stream) {
    if (csize == 0) return;
    dim3 block(256), grid((csize + block.x - 1u) / block.x);
    scatter_secondary_kernel<<<grid, block, 0, stream>>>(
            dedup_keys, csize, hash_value, D_pos, X);
    CUDA_CHECK_LAST();
}

extern "C" void launch_scan_originals(
        const uint64 *keys, uint32 N, uint32 hash_value,
        const uint32 *S, const uint32 *D, const uint64 *X,
        uint32 *hits_out, cudaStream_t stream) {
    CUDA_TRY(cudaMemsetAsync(hits_out, 0, sizeof(uint32), stream));
    dim3 block(256), grid((N + block.x - 1u) / block.x);
    scan_originals_kernel<<<grid, block, 0, stream>>>(
            keys, N, hash_value, S, D, X, hits_out);
    CUDA_CHECK_LAST();
}

extern "C" void launch_count_matched_values(
        const uint64 *keys, const uint32 *values, uint32 N,
        uint32 hash_value,
        const uint32 *S, const uint32 *D, const uint64 *X,
        uint32 dedup_count,
        uint32 *value_counts, uint32 *match_total,
        uint32 *value_overflow, uint32 value_cap,
        cudaStream_t stream) {
    CUDA_TRY(cudaMemsetAsync(value_counts, 0,
                             (dedup_count + 1u) * sizeof(uint32), stream));
    CUDA_TRY(cudaMemsetAsync(match_total, 0, sizeof(uint32), stream));
    CUDA_TRY(cudaMemsetAsync(value_overflow, 0, sizeof(uint32), stream));
    if (N == 0 || dedup_count == 0) return;
    dim3 block(256), grid((N + block.x - 1u) / block.x);
    count_matched_values_kernel<<<grid, block, 0, stream>>>(
            keys, values, N, hash_value, S, D, X, dedup_count,
            value_counts, match_total, value_overflow, value_cap);
    CUDA_CHECK_LAST();
}

extern "C" void launch_scatter_matched_values(
        const uint64 *keys, const uint32 *values, uint32 N,
        uint32 hash_value,
        const uint32 *S, const uint32 *D, const uint64 *X,
        uint32 dedup_count,
        uint32 *value_cursor, uint32 *matched_values, uint32 value_cap,
        cudaStream_t stream) {
    if (N == 0 || dedup_count == 0) return;
    dim3 block(256), grid((N + block.x - 1u) / block.x);
    scatter_matched_values_kernel<<<grid, block, 0, stream>>>(
            keys, values, N, hash_value, S, D, X, dedup_count,
            value_cursor, matched_values, value_cap);
    CUDA_CHECK_LAST();
}

extern "C" void launch_count_msieve_pairs(
        const uint32 *matched_values,
        const uint32 *value_offsets,
        uint32 dedup_count, uint32 pshift,
        uint32 *hits_out, cudaStream_t stream) {
    CUDA_TRY(cudaMemsetAsync(hits_out, 0, sizeof(uint32), stream));
    if (dedup_count == 0) return;
    dim3 block(256), grid((dedup_count + block.x - 1u) / block.x);
    count_msieve_pairs_kernel<<<grid, block, 0, stream>>>(
            matched_values, value_offsets, dedup_count, pshift, hits_out);
    CUDA_CHECK_LAST();
}
