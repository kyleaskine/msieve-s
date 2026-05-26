#include "collision_common.cuh"

/* Phase B: per-bucket ping-pong hash filter.

   Mirrors collision_ref.c:247-382 (no-loss path only). One block per bucket.

   Smem holds the three bit-array hash tables T, T2, T3 (sized to the worst
   per-bucket ilog2 observed across all buckets — caller passes the right
   smem byte count). The candidate working set `arr` is kept in DRAM and
   ping-ponged between two per-block scratch slices (arr_a, arr_b), because
   for N up to 100M a bucket's items (~6500 × 8 B = 52 KB) plus 24-48 KB of
   hash tables would exceed the 99 KB dyn-smem budget on sm_120.

   Correctness vs collision_ref.c's sequential reference: the sequential pattern
       if (U2[hv2>>5] & bit) U3[hv2>>5] |= bit;
       else                  U2[hv2>>5] |= bit;
   becomes
       prev = atomicOr(&U2[hv2>>5], bit);
       if (prev & bit) atomicOr(&U3[hv2>>5], bit);
   which is order-equivalent: whichever thread wins the first atomicOr sees
   prev=0 and skips U3; every subsequent thread sees prev with the bit and
   correctly sets U3. Crucially this is NOT the same as a naive
   `if (U2[...]) ... else ...` test in CUDA, which races and silently drops
   collisions (manifests as allow_loss mode behaviour).

   Convergence: matches collision_ref.c's break condition — round count maxes out,
   cnt drops to 0, or three rounds of no progress (nsize[it-3]==nsize[it]). */

#define BLOCK_THREADS 128

/* Dynamic smem layout depends on which kernel is launched.
   For the DRAM-arr fallback (large buckets): T | T2 | T3 only.
   For the smem-arr fast path: T | T2 | T3 | s_arr_in | s_arr_out. */
extern __shared__ uint32 s_hash[];

__device__ static inline uint32
compute_ilog2(uint32 cnt, uint32 key_bits) {
    /* collision_ref.c:234 / 313-315: (int)(log2(cnt+1) + 4.5), floored. */
    if (cnt == 0) return 5;
    uint32 lg = 31u - __clz(cnt);
    uint32 il = max(5u, lg + 5u);  /* +4.5 rounded by (int) cast in C ≈ +4 here.
                                       Use +5 to err on the larger table side
                                       (matches the rounding direction of the
                                       reference more often for our N range). */
    uint32 hash_bits_available = key_bits - LOG2_NUM_BUCKETS;
    if (il > hash_bits_available) il = hash_bits_available;
    return il;
}

__device__ static void
clear_table_parallel(uint32 *tbl, uint32 nwords) {
    for (uint32 j = threadIdx.x; j < nwords; j += BLOCK_THREADS) {
        tbl[j] = 0;
    }
}

__global__ void
filter_per_bucket_kernel(
        const uint32 * __restrict__ bucket_count_in,
        uint32 max_per_bucket,
        uint64       * __restrict__ arr_a,
        uint64       * __restrict__ arr_b,
        uint64       * __restrict__ candidate_keys,
        uint32       * __restrict__ candidate_cnt,
        uint32       * __restrict__ candidate_overflow,
        uint32 candidate_cap,
        uint32 key_bits,
        uint32 max_tsize_words) {

    uint32 bucket = blockIdx.x;
    uint32 cnt    = bucket_count_in[bucket];
    if (cnt > max_per_bucket)
        cnt = max_per_bucket;
    if (cnt == 0) return;

    size_t offset = (size_t)bucket * max_per_bucket;
    uint64 *arr_in  = arr_a + offset;
    uint64 *arr_out = arr_b + offset;

    uint32 *T  = s_hash;
    uint32 *T2 = s_hash + max_tsize_words;
    uint32 *T3 = s_hash + 2u * max_tsize_words;

    /* --- Round 0: build T (seen ≥2) and T2 (seen ≥1), no filtering. */

    uint32 ilog2 = compute_ilog2(cnt, key_bits);
    uint32 tsize = 1u << (ilog2 - 5u);
    uint32 hash_value2 = (ilog2 >= 32u) ? 0xFFFFFFFFu : ((1u << ilog2) - 1u);
    /* Start hashing past the bucket-correlated low bits. For default low-bits
       bucketing those LOG2_NUM_BUCKETS bits are constant within a bucket and
       contribute nothing to discrimination. For --bucket-hash they're random
       (so shift isn't strictly needed) but starting here is harmless. */
    uint32 my_shift2   = LOG2_NUM_BUCKETS;

    clear_table_parallel(T,  tsize);
    clear_table_parallel(T2, tsize);
    __syncthreads();

    for (uint32 j = threadIdx.x; j < cnt; j += BLOCK_THREADS) {
        uint64 item = arr_in[j];
        uint32 hv = (uint32)((item >> my_shift2) & hash_value2);
        uint32 bit = 1u << (hv & 31u);
        uint32 widx = hv >> 5;
        uint32 prev = atomicOr(&T2[widx], bit);
        if (prev & bit) {
            atomicOr(&T[widx], bit);
        }
    }
    __syncthreads();

    /* --- Ping-pong filter rounds. */

    __shared__ uint32 s_nsize[MAX_FILTER_ITERS];
    __shared__ uint32 s_cnt_out;

    bool emit = false;
    for (int it = 0; it < MAX_FILTER_ITERS; it++) {

        uint32 my_shift = my_shift2;
        uint32 hash_value = hash_value2;

        ilog2 = compute_ilog2(cnt, key_bits);
        tsize = 1u << (ilog2 - 5u);

        uint32 *U, *U2, *U3;
        if ((it & 1) == 0) { U = T;  U2 = T2; U3 = T3; }
        else               { U = T3; U2 = T2; U3 = T;  }

        my_shift2 = my_shift + 6u;
        /* Available hash bits = key_bits - LOG2_NUM_BUCKETS (we start at
           LOG2_NUM_BUCKETS); wrap back to LOG2_NUM_BUCKETS when we'd overflow. */
        if (my_shift2 + ilog2 > key_bits) my_shift2 = LOG2_NUM_BUCKETS;
        hash_value2 = (ilog2 >= 32u) ? 0xFFFFFFFFu : ((1u << ilog2) - 1u);

        clear_table_parallel(U2, tsize);
        clear_table_parallel(U3, tsize);
        if (threadIdx.x == 0) s_cnt_out = 0;
        __syncthreads();

        /* Filter pass. Each thread walks a stride of arr_in and emits
           survivors into arr_out via warp-aggregated atomic slot claim. */
        for (uint32 j_base = 0; j_base < cnt; j_base += BLOCK_THREADS) {
            uint32 j = j_base + threadIdx.x;
            bool   survives = false;
            uint64 item     = 0;

            if (j < cnt) {
                item = arr_in[j];
                uint32 hv = (uint32)((item >> my_shift) & hash_value);
                uint32 bit = 1u << (hv & 31u);
                if (U[hv >> 5] & bit) {
                    uint32 hv2 = (uint32)((item >> my_shift2) & hash_value2);
                    uint32 bit2 = 1u << (hv2 & 31u);
                    uint32 widx2 = hv2 >> 5;
                    uint32 prev = atomicOr(&U2[widx2], bit2);
                    if (prev & bit2) atomicOr(&U3[widx2], bit2);
                    survives = true;
                }
            }

            uint32 vote = __ballot_sync(0xFFFFFFFFu, survives);
            uint32 lane = threadIdx.x & 31u;
            uint32 leader_slot = 0;
            if (vote) {
                uint32 wcount = __popc(vote);
                if (lane == 0) {
                    leader_slot = atomicAdd(&s_cnt_out, wcount);
                }
                leader_slot = __shfl_sync(0xFFFFFFFFu, leader_slot, 0);
                if (survives) {
                    uint32 within = __popc(vote & ((1u << lane) - 1u));
                    arr_out[leader_slot + within] = item;
                }
            }
        }

        __syncthreads();
        uint32 cnt_out = s_cnt_out;
        if (threadIdx.x == 0) s_nsize[it] = cnt_out;
        __syncthreads();

        cnt = cnt_out;

        /* Swap pointers FIRST so arr_in always points at the most recently
           written buffer — convergence check + emit then read arr_in. */
        uint64 *tmp = arr_in; arr_in = arr_out; arr_out = tmp;

        /* Convergence: empty, hit maxit, or no progress over 3 rounds. */
        bool stop = (cnt == 0) || (it == MAX_FILTER_ITERS - 1) ||
                    (it >= 3 && s_nsize[it - 3] == s_nsize[it]);

        if (stop) { emit = true; break; }
    }

    /* Emit survivors with bucket id appended (collision_ref.c:376-378). */
    if (!emit) return;
    if (cnt == 0) return;

    __shared__ uint32 s_emit_base;
    if (threadIdx.x == 0) {
        s_emit_base = atomicAdd(candidate_cnt, cnt);
    }
    __syncthreads();
    uint32 base = s_emit_base;
    if (base + cnt > candidate_cap) {
        if (threadIdx.x == 0)
            atomicOr(candidate_overflow, 1u);
        return;
    }

    for (uint32 j = threadIdx.x; j < cnt; j += BLOCK_THREADS) {
        candidate_keys[base + j] = arr_in[j];  /* full key stored as-is */
    }
    (void)bucket;
}

/* ============================================================== */
/* Smem-arr variant: items stay in shared memory across rounds.
   Eliminates the per-round DRAM ping-pong (~0.7-1 ms savings at
   N=50M msieve regime where Phase B was DRAM-bandwidth limited
   at ~500 GB/s of ~575 GB/s peak).

   Layout: hash tables T|T2|T3 first, then s_arr_in then s_arr_out.
   Caller must size dyn-smem to:
     3 * max_tsize_words * 4  +  2 * smem_arr_cap * 8
   Caller must ensure all buckets fit (bucket_count[i] <= smem_arr_cap). */
/* ============================================================== */

__global__ void
filter_per_bucket_smem_kernel(
        const uint32 * __restrict__ bucket_count_in,
        uint32 max_per_bucket,                 /* still need DRAM offset for init load */
        const uint64 * __restrict__ arr_a,
        uint64       * __restrict__ candidate_keys,
        uint32       * __restrict__ candidate_cnt,
        uint32       * __restrict__ candidate_overflow,
        uint32 candidate_cap,
        uint32 key_bits,
        uint32 max_tsize_words,
        uint32 smem_arr_cap) {

    uint32 bucket = blockIdx.x;
    uint32 cnt    = bucket_count_in[bucket];
    if (cnt > max_per_bucket)
        cnt = max_per_bucket;
    if (cnt == 0) return;

    /* Hash tables T|T2|T3, then two uint64 arr buffers (after the uint32
       hash region — explicit alignment to 8 bytes). */
    uint32 *T  = s_hash;
    uint32 *T2 = s_hash + max_tsize_words;
    uint32 *T3 = s_hash + 2u * max_tsize_words;
    uint64 *s_arr_in  = (uint64 *)(s_hash + 3u * max_tsize_words);
    uint64 *s_arr_out = s_arr_in + smem_arr_cap;

    /* Initial load: arr_a slice → s_arr_in. The single DRAM read of the
       round-0 bucket data. */
    const uint64 *arr_global = arr_a + (size_t)bucket * max_per_bucket;
    for (uint32 j = threadIdx.x; j < cnt; j += BLOCK_THREADS) {
        s_arr_in[j] = arr_global[j];
    }

    /* --- Round 0: build T (seen ≥2) and T2 (seen ≥1) from s_arr_in. */
    uint32 ilog2 = compute_ilog2(cnt, key_bits);
    uint32 tsize = 1u << (ilog2 - 5u);
    uint32 hash_value2 = (ilog2 >= 32u) ? 0xFFFFFFFFu : ((1u << ilog2) - 1u);
    /* Start hashing past the bucket-correlated low bits. For default low-bits
       bucketing those LOG2_NUM_BUCKETS bits are constant within a bucket and
       contribute nothing to discrimination. For --bucket-hash they're random
       (so shift isn't strictly needed) but starting here is harmless. */
    uint32 my_shift2   = LOG2_NUM_BUCKETS;

    clear_table_parallel(T,  tsize);
    clear_table_parallel(T2, tsize);
    __syncthreads();

    for (uint32 j = threadIdx.x; j < cnt; j += BLOCK_THREADS) {
        uint64 item = s_arr_in[j];
        uint32 hv = (uint32)((item >> my_shift2) & hash_value2);
        uint32 bit = 1u << (hv & 31u);
        uint32 widx = hv >> 5;
        uint32 prev = atomicOr(&T2[widx], bit);
        if (prev & bit) atomicOr(&T[widx], bit);
    }
    __syncthreads();

    /* --- Ping-pong filter rounds, entirely in smem. */
    __shared__ uint32 s_nsize[MAX_FILTER_ITERS];
    __shared__ uint32 s_cnt_out;

    bool emit = false;
    for (int it = 0; it < MAX_FILTER_ITERS; it++) {

        uint32 my_shift = my_shift2;
        uint32 hash_value = hash_value2;

        ilog2 = compute_ilog2(cnt, key_bits);
        tsize = 1u << (ilog2 - 5u);

        uint32 *U, *U2, *U3;
        if ((it & 1) == 0) { U = T;  U2 = T2; U3 = T3; }
        else               { U = T3; U2 = T2; U3 = T;  }

        my_shift2 = my_shift + 6u;
        /* Available hash bits = key_bits - LOG2_NUM_BUCKETS (we start at
           LOG2_NUM_BUCKETS); wrap back to LOG2_NUM_BUCKETS when we'd overflow. */
        if (my_shift2 + ilog2 > key_bits) my_shift2 = LOG2_NUM_BUCKETS;
        hash_value2 = (ilog2 >= 32u) ? 0xFFFFFFFFu : ((1u << ilog2) - 1u);

        clear_table_parallel(U2, tsize);
        clear_table_parallel(U3, tsize);
        if (threadIdx.x == 0) s_cnt_out = 0;
        __syncthreads();

        for (uint32 j_base = 0; j_base < cnt; j_base += BLOCK_THREADS) {
            uint32 j = j_base + threadIdx.x;
            bool   survives = false;
            uint64 item     = 0;

            if (j < cnt) {
                item = s_arr_in[j];
                uint32 hv = (uint32)((item >> my_shift) & hash_value);
                uint32 bit = 1u << (hv & 31u);
                if (U[hv >> 5] & bit) {
                    uint32 hv2 = (uint32)((item >> my_shift2) & hash_value2);
                    uint32 bit2 = 1u << (hv2 & 31u);
                    uint32 widx2 = hv2 >> 5;
                    uint32 prev = atomicOr(&U2[widx2], bit2);
                    if (prev & bit2) atomicOr(&U3[widx2], bit2);
                    survives = true;
                }
            }

            uint32 vote = __ballot_sync(0xFFFFFFFFu, survives);
            uint32 lane = threadIdx.x & 31u;
            uint32 leader_slot = 0;
            if (vote) {
                uint32 wcount = __popc(vote);
                if (lane == 0) {
                    leader_slot = atomicAdd(&s_cnt_out, wcount);
                }
                leader_slot = __shfl_sync(0xFFFFFFFFu, leader_slot, 0);
                if (survives) {
                    uint32 within = __popc(vote & ((1u << lane) - 1u));
                    s_arr_out[leader_slot + within] = item;
                }
            }
        }

        __syncthreads();
        uint32 cnt_out = s_cnt_out;
        if (threadIdx.x == 0) s_nsize[it] = cnt_out;
        __syncthreads();

        cnt = cnt_out;

        uint64 *tmp = s_arr_in; s_arr_in = s_arr_out; s_arr_out = tmp;

        bool stop = (cnt == 0) || (it == MAX_FILTER_ITERS - 1) ||
                    (it >= 3 && s_nsize[it - 3] == s_nsize[it]);

        if (stop) { emit = true; break; }
    }

    if (!emit || cnt == 0) return;

    __shared__ uint32 s_emit_base;
    if (threadIdx.x == 0) s_emit_base = atomicAdd(candidate_cnt, cnt);
    __syncthreads();
    uint32 base = s_emit_base;
    if (base + cnt > candidate_cap) {
        if (threadIdx.x == 0)
            atomicOr(candidate_overflow, 1u);
        return;
    }

    for (uint32 j = threadIdx.x; j < cnt; j += BLOCK_THREADS) {
        candidate_keys[base + j] = s_arr_in[j];  /* full key stored as-is */
    }
    (void)bucket;
}

extern "C" void launch_filter_per_bucket(
        const uint32 *bucket_count_in,
        uint32 max_per_bucket,
        uint64 *arr_a, uint64 *arr_b,
        uint64 *candidate_keys, uint32 *candidate_cnt,
        uint32 *candidate_overflow, uint32 candidate_cap,
        uint32 key_bits, uint32 max_tsize_words,
        cudaStream_t stream) {

    dim3 grid(NUM_BUCKETS);
    dim3 block(BLOCK_THREADS);

    CUDA_TRY(cudaMemsetAsync(candidate_cnt, 0, sizeof(uint32), stream));
    CUDA_TRY(cudaMemsetAsync(candidate_overflow, 0, sizeof(uint32), stream));

    /* Pick smem-arr path only if it preserves >=2 blocks/SM, i.e. uses no more
       than half the smem budget. Going below 2 blocks/SM tanks latency-hiding
       and made Phase B slower in measurement (1.95ms DRAM vs 3ms smem at
       N=50M, where smem-arr forced 1 block/SM). */
    size_t hash_bytes = 3u * max_tsize_words * sizeof(uint32);
    size_t smem_bytes_smem_arr =
            hash_bytes + 2u * (size_t)max_per_bucket * sizeof(uint64);
    constexpr size_t SMEM_PER_BLOCK_CAP = 49u * 1024u;   /* 2 blocks/SM */

    if (smem_bytes_smem_arr <= SMEM_PER_BLOCK_CAP) {
        CUDA_TRY(cudaFuncSetAttribute(
                filter_per_bucket_smem_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)smem_bytes_smem_arr));
        filter_per_bucket_smem_kernel<<<grid, block, smem_bytes_smem_arr, stream>>>(
                bucket_count_in, max_per_bucket, arr_a,
                candidate_keys, candidate_cnt, candidate_overflow, candidate_cap,
                key_bits, max_tsize_words, max_per_bucket);
        CUDA_CHECK_LAST();
        return;
    }

    /* Fall back to DRAM ping-pong (large-N path). */
    CUDA_TRY(cudaFuncSetAttribute(
            filter_per_bucket_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)hash_bytes));
    filter_per_bucket_kernel<<<grid, block, hash_bytes, stream>>>(
            bucket_count_in, max_per_bucket,
            arr_a, arr_b,
            candidate_keys, candidate_cnt,
            candidate_overflow, candidate_cap, key_bits, max_tsize_words);
    CUDA_CHECK_LAST();
}
