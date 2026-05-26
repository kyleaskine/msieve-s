#include "collision_common.cuh"
#include "bench_gpu.cuh"

#include <cub/cub.cuh>
#include <string.h>
#include <math.h>

/* External kernel launchers from the three Phase TUs. */
extern "C" void launch_scatter_one_pass(
        const uint64 *, uint32, uint32, int, uint32 *, uint32 *,
        uint64 *, cudaStream_t);
extern "C" void launch_filter_per_bucket(
        const uint32 *, uint32, uint64 *, uint64 *,
        uint64 *, uint32 *, uint32 *, uint32, uint32, uint32, cudaStream_t);
extern "C" void launch_dedup(
        const uint64 *, uint32, uint64 *, uint32 *, cudaStream_t);
extern "C" void launch_count_secondary(
        const uint64 *, uint32, uint32, uint32 *, uint32, uint32 *, uint32,
        cudaStream_t);
extern "C" void launch_scatter_secondary(
        const uint64 *, uint32, uint32, uint32 *, uint64 *, cudaStream_t);
extern "C" void launch_scan_originals(
        const uint64 *, uint32, uint32, const uint32 *, const uint32 *,
        const uint64 *, uint32 *, cudaStream_t);
extern "C" void launch_count_matched_values(
        const uint64 *, const uint32 *, uint32, uint32,
        const uint32 *, const uint32 *, const uint64 *, uint32,
        uint32 *, uint32 *, uint32 *, uint32, cudaStream_t);
extern "C" void launch_scatter_matched_values(
        const uint64 *, const uint32 *, uint32, uint32,
        const uint32 *, const uint32 *, const uint64 *, uint32,
        uint32 *, uint32 *, uint32, cudaStream_t);
extern "C" void launch_count_msieve_pairs(
        const uint32 *, const uint32 *, uint32, uint32,
        uint32 *, cudaStream_t);

/* Sized for the msieve regime (sparse: hundreds-to-thousands of collisions
   per batch). Very dense synthetic data (bits<30, large N) can blow these. */
#define CANDIDATE_CAP (1u << 22)            /* 4M slots                       */
#define VALUE_MATCH_CAP CANDIDATE_CAP
#define MAX_C_ILOG2   20u                   /* cap on Phase-C hash bits       */
#define MAX_DSIZE     ((1u << MAX_C_ILOG2) + 64u)
#define MAX_SSIZE     (((1u << MAX_C_ILOG2) >> 5) + 64u)

struct gpu_ctx_s {
    uint32 max_n;
    uint32 max_key_bits;
    uint32 fixed_max_tsize_words;  /* computed at init from max_n */
    uint32 max_per_bucket;         /* slot allocation per bucket */

    /* Device buffers. */
    uint64 *d_keys;            /* N uint64 — original keys */
    uint32 *d_bucket_count;    /* NUM_BUCKETS uint32 */
    uint32 *d_overflow_flag;   /* single uint32 — set if any bucket overflows */
    uint32 *d_candidate_overflow; /* single uint32 */
    uint64 *d_arr_a;           /* NUM_BUCKETS*max_per_bucket uint64 */
    uint64 *d_arr_b;           /* same — Phase B ping-pong scratch */
    uint32 *d_values;          /* N uint32 — optional packed msieve values */
    uint64 *d_candidate_keys;  /* CANDIDATE_CAP uint64 */
    uint32 *d_candidate_cnt;   /* single uint32 */
    uint64 *d_sorted_keys;     /* CANDIDATE_CAP uint64 */
    uint64 *d_dedup_keys;      /* CANDIDATE_CAP uint64 */
    uint32 *d_dedup_cnt;       /* single uint32 */
    uint32 *d_D;               /* MAX_DSIZE uint32 */
    uint32 *d_D_scan;          /* MAX_DSIZE uint32 (after scan) */
    uint32 *d_D_pos;           /* MAX_DSIZE uint32 (cursor copy) */
    uint32 *d_S;               /* MAX_SSIZE uint32 */
    uint64 *d_X;               /* CANDIDATE_CAP uint64 */
    uint32 *d_hits;            /* single uint32 */
    uint32 *d_max_bucket;      /* single uint32 — output of CUB max-reduce */
    uint32 *d_value_counts;    /* VALUE_MATCH_CAP+1 uint32 */
    uint32 *d_value_offsets;   /* VALUE_MATCH_CAP+1 uint32 */
    uint32 *d_value_cursor;    /* VALUE_MATCH_CAP+1 uint32 */
    uint32 *d_matched_values;  /* VALUE_MATCH_CAP uint32 */
    uint32 *d_value_match_total; /* single uint32 */
    uint32 *d_value_overflow;  /* single uint32 */
    uint32 *d_msieve_hits;     /* single uint32 */

    /* CUB temp storage. */
    void   *d_cub_max_buckets;
    size_t  max_buckets_bytes;
    void   *d_cub_sort_candidates;
    size_t  sort_candidates_bytes;
    void   *d_cub_scan_D;
    size_t  scan_D_bytes;
    void   *d_cub_scan_values;
    size_t  scan_values_bytes;

    /* CUDA events for per-phase timing. */
    cudaEvent_t ev_start, ev_h2d_done, ev_a_done, ev_b_done, ev_c_done, ev_d2h_done;

    /* Pinned host scratch. */
    uint32 *h_max_bucket;
    uint32 *h_bucket_overflow;
    uint32 *h_candidate_overflow;
    uint32 *h_dedup_cnt;
    uint32 *h_hits;
    uint32 *h_value_match_total;
    uint32 *h_value_overflow;
    uint32 *h_msieve_hits;
};

static void
alloc_or_die(void **p, size_t bytes) {
    CUDA_TRY(cudaMalloc(p, bytes));
}

gpu_ctx_t *gpu_ctx_init(uint32 max_n, uint32 max_key_bits) {
    gpu_ctx_t *ctx = (gpu_ctx_t *)calloc(1, sizeof(gpu_ctx_t));
    ctx->max_n = max_n;
    ctx->max_key_bits = max_key_bits;

    /* Conservative worst-case bucket size assuming uniform random keys:
       mean + 6 standard deviations. Random data has σ ≈ sqrt(mean*(1-1/B)),
       so for any plausible N this overestimates the real max. For real
       msieve keys (low-bit skew possible), enable --bucket-hash. */
    uint32 mean_per_bucket = (max_n + NUM_BUCKETS - 1u) / NUM_BUCKETS;
    uint32 sigma = (uint32)sqrt((double)mean_per_bucket + 1.0);
    uint32 max_bucket_est = mean_per_bucket + 6u * sigma + 32u;
    uint32 lg = (max_bucket_est <= 1u) ? 0u : (31u - __builtin_clz(max_bucket_est));
    uint32 ilog2 = lg + 5u;
    if (ilog2 < 5u) ilog2 = 5u;
    uint32 hash_bits_avail = max_key_bits - LOG2_NUM_BUCKETS;
    if (ilog2 > hash_bits_avail) ilog2 = hash_bits_avail;
    ctx->fixed_max_tsize_words = 1u << (ilog2 - 5u);

    /* Per-bucket slot capacity: mean + 6σ + 32 — same envelope used to size
       the smem hash tables, so all the sizing decisions stay consistent. */
    ctx->max_per_bucket = max_bucket_est;
    size_t bucket_storage_words = (size_t)NUM_BUCKETS * ctx->max_per_bucket;

    alloc_or_die((void **)&ctx->d_keys,            (size_t)max_n * sizeof(uint64));
    alloc_or_die((void **)&ctx->d_bucket_count,    NUM_BUCKETS * sizeof(uint32));
    alloc_or_die((void **)&ctx->d_overflow_flag,   sizeof(uint32));
    alloc_or_die((void **)&ctx->d_candidate_overflow, sizeof(uint32));
    alloc_or_die((void **)&ctx->d_arr_a,           bucket_storage_words * sizeof(uint64));
    alloc_or_die((void **)&ctx->d_arr_b,           bucket_storage_words * sizeof(uint64));
    alloc_or_die((void **)&ctx->d_values,          (size_t)max_n * sizeof(uint32));
    alloc_or_die((void **)&ctx->d_candidate_keys,  CANDIDATE_CAP * sizeof(uint64));
    alloc_or_die((void **)&ctx->d_candidate_cnt,   sizeof(uint32));
    alloc_or_die((void **)&ctx->d_sorted_keys,     CANDIDATE_CAP * sizeof(uint64));
    alloc_or_die((void **)&ctx->d_dedup_keys,      CANDIDATE_CAP * sizeof(uint64));
    alloc_or_die((void **)&ctx->d_dedup_cnt,       sizeof(uint32));
    alloc_or_die((void **)&ctx->d_D,               MAX_DSIZE * sizeof(uint32));
    alloc_or_die((void **)&ctx->d_D_scan,          MAX_DSIZE * sizeof(uint32));
    alloc_or_die((void **)&ctx->d_D_pos,           MAX_DSIZE * sizeof(uint32));
    alloc_or_die((void **)&ctx->d_S,               MAX_SSIZE * sizeof(uint32));
    alloc_or_die((void **)&ctx->d_X,               CANDIDATE_CAP * sizeof(uint64));
    alloc_or_die((void **)&ctx->d_hits,            sizeof(uint32));
    alloc_or_die((void **)&ctx->d_max_bucket,      sizeof(uint32));
    alloc_or_die((void **)&ctx->d_value_counts,    (VALUE_MATCH_CAP + 1u) * sizeof(uint32));
    alloc_or_die((void **)&ctx->d_value_offsets,   (VALUE_MATCH_CAP + 1u) * sizeof(uint32));
    alloc_or_die((void **)&ctx->d_value_cursor,    (VALUE_MATCH_CAP + 1u) * sizeof(uint32));
    alloc_or_die((void **)&ctx->d_matched_values,  VALUE_MATCH_CAP * sizeof(uint32));
    alloc_or_die((void **)&ctx->d_value_match_total, sizeof(uint32));
    alloc_or_die((void **)&ctx->d_value_overflow,  sizeof(uint32));
    alloc_or_die((void **)&ctx->d_msieve_hits,     sizeof(uint32));

    /* Query CUB temp sizes. */
    cub::DeviceReduce::Max(
            nullptr, ctx->max_buckets_bytes,
            ctx->d_bucket_count, ctx->d_max_bucket, NUM_BUCKETS);
    alloc_or_die(&ctx->d_cub_max_buckets, ctx->max_buckets_bytes);

    cub::DeviceRadixSort::SortKeys(
            nullptr, ctx->sort_candidates_bytes,
            ctx->d_candidate_keys, ctx->d_sorted_keys, CANDIDATE_CAP);
    alloc_or_die(&ctx->d_cub_sort_candidates, ctx->sort_candidates_bytes);

    cub::DeviceScan::ExclusiveSum(
            nullptr, ctx->scan_D_bytes,
            ctx->d_D, ctx->d_D_scan, MAX_DSIZE);
    alloc_or_die(&ctx->d_cub_scan_D, ctx->scan_D_bytes);

    cub::DeviceScan::ExclusiveSum(
            nullptr, ctx->scan_values_bytes,
            ctx->d_value_counts, ctx->d_value_offsets, VALUE_MATCH_CAP + 1u);
    alloc_or_die(&ctx->d_cub_scan_values, ctx->scan_values_bytes);

    CUDA_TRY(cudaEventCreate(&ctx->ev_start));
    CUDA_TRY(cudaEventCreate(&ctx->ev_h2d_done));
    CUDA_TRY(cudaEventCreate(&ctx->ev_a_done));
    CUDA_TRY(cudaEventCreate(&ctx->ev_b_done));
    CUDA_TRY(cudaEventCreate(&ctx->ev_c_done));
    CUDA_TRY(cudaEventCreate(&ctx->ev_d2h_done));

    CUDA_TRY(cudaMallocHost((void **)&ctx->h_max_bucket, sizeof(uint32)));
    CUDA_TRY(cudaMallocHost((void **)&ctx->h_bucket_overflow, sizeof(uint32)));
    CUDA_TRY(cudaMallocHost((void **)&ctx->h_candidate_overflow, sizeof(uint32)));
    CUDA_TRY(cudaMallocHost((void **)&ctx->h_dedup_cnt,  sizeof(uint32)));
    CUDA_TRY(cudaMallocHost((void **)&ctx->h_hits,       sizeof(uint32)));
    CUDA_TRY(cudaMallocHost((void **)&ctx->h_value_match_total, sizeof(uint32)));
    CUDA_TRY(cudaMallocHost((void **)&ctx->h_value_overflow, sizeof(uint32)));
    CUDA_TRY(cudaMallocHost((void **)&ctx->h_msieve_hits, sizeof(uint32)));

    return ctx;
}

void gpu_ctx_free(gpu_ctx_t *ctx) {
    if (!ctx) return;
    cudaFree(ctx->d_keys);
    cudaFree(ctx->d_bucket_count);
    cudaFree(ctx->d_overflow_flag);
    cudaFree(ctx->d_candidate_overflow);
    cudaFree(ctx->d_arr_a);
    cudaFree(ctx->d_arr_b);
    cudaFree(ctx->d_values);
    cudaFree(ctx->d_candidate_keys);
    cudaFree(ctx->d_candidate_cnt);
    cudaFree(ctx->d_sorted_keys);
    cudaFree(ctx->d_dedup_keys);
    cudaFree(ctx->d_dedup_cnt);
    cudaFree(ctx->d_D);
    cudaFree(ctx->d_D_scan);
    cudaFree(ctx->d_D_pos);
    cudaFree(ctx->d_S);
    cudaFree(ctx->d_X);
    cudaFree(ctx->d_hits);
    cudaFree(ctx->d_max_bucket);
    cudaFree(ctx->d_value_counts);
    cudaFree(ctx->d_value_offsets);
    cudaFree(ctx->d_value_cursor);
    cudaFree(ctx->d_matched_values);
    cudaFree(ctx->d_value_match_total);
    cudaFree(ctx->d_value_overflow);
    cudaFree(ctx->d_msieve_hits);
    cudaFree(ctx->d_cub_max_buckets);
    cudaFree(ctx->d_cub_sort_candidates);
    cudaFree(ctx->d_cub_scan_D);
    cudaFree(ctx->d_cub_scan_values);
    cudaEventDestroy(ctx->ev_start);
    cudaEventDestroy(ctx->ev_h2d_done);
    cudaEventDestroy(ctx->ev_a_done);
    cudaEventDestroy(ctx->ev_b_done);
    cudaEventDestroy(ctx->ev_c_done);
    cudaEventDestroy(ctx->ev_d2h_done);
    cudaFreeHost(ctx->h_max_bucket);
    cudaFreeHost(ctx->h_bucket_overflow);
    cudaFreeHost(ctx->h_candidate_overflow);
    cudaFreeHost(ctx->h_dedup_cnt);
    cudaFreeHost(ctx->h_hits);
    cudaFreeHost(ctx->h_value_match_total);
    cudaFreeHost(ctx->h_value_overflow);
    cudaFreeHost(ctx->h_msieve_hits);
    free(ctx);
}

static uint32 host_ilog2(uint32 cnt) {
    if (cnt <= 1) return 5;
    uint32 lg = 31u - __builtin_clz(cnt);
    uint32 il = (lg + 5u);
    if (il < 5u) il = 5u;
    return il;
}

static uint32 gpu_collision_search_internal(gpu_ctx_t *ctx,
                            const uint64_t *keys, const uint32_t *values,
                            uint32 n, uint32 key_bits, int bucket_hash,
                            uint32 pshift, gpu_timings_t *timings,
                            uint32 *msieve_hits_out) {
    cudaStream_t stream = 0;
    int verify_values = (values != NULL);
    if (msieve_hits_out)
        *msieve_hits_out = 0;

    CUDA_TRY(cudaEventRecord(ctx->ev_start, stream));

    /* H2D. */
    CUDA_TRY(cudaMemcpyAsync(ctx->d_keys, keys,
                             (size_t)n * sizeof(uint64),
                             cudaMemcpyHostToDevice, stream));
    if (verify_values) {
        CUDA_TRY(cudaMemcpyAsync(ctx->d_values, values,
                                 (size_t)n * sizeof(uint32),
                                 cudaMemcpyHostToDevice, stream));
    }
    CUDA_TRY(cudaEventRecord(ctx->ev_h2d_done, stream));

    /* --- Phase A: single-pass scatter into per-bucket slot ranges. --- */
    launch_scatter_one_pass(ctx->d_keys, n, ctx->max_per_bucket, bucket_hash,
                            ctx->d_bucket_count, ctx->d_overflow_flag,
                            ctx->d_arr_a, stream);
    cub::DeviceReduce::Max(
            ctx->d_cub_max_buckets, ctx->max_buckets_bytes,
            ctx->d_bucket_count, ctx->d_max_bucket, NUM_BUCKETS, stream);
    CUDA_TRY(cudaMemcpyAsync(ctx->h_max_bucket, ctx->d_max_bucket,
                             sizeof(uint32), cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaMemcpyAsync(ctx->h_bucket_overflow, ctx->d_overflow_flag,
                             sizeof(uint32), cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaEventRecord(ctx->ev_a_done, stream));

    /* --- Phase B: per-bucket filter, using strided per-bucket layout. ---
       Compute per-call max_tsize_words from THIS n (not max_n at init), so the
       smem-arr fast path can trigger on smaller cells inside a multi-N sweep. */
    uint32 mean_for_this_n = (n + NUM_BUCKETS - 1u) / NUM_BUCKETS;
    uint32 sigma_for_this_n = (uint32)sqrt((double)mean_for_this_n + 1.0);
    uint32 max_bucket_est_for_n = mean_for_this_n + 6u * sigma_for_this_n + 32u;
    uint32 lg_for_n = (max_bucket_est_for_n <= 1u)
                      ? 0u : (31u - __builtin_clz(max_bucket_est_for_n));
    uint32 ilog2_for_n = lg_for_n + 5u;
    if (ilog2_for_n < 5u) ilog2_for_n = 5u;
    uint32 hash_bits_avail_for_n = key_bits - LOG2_NUM_BUCKETS;
    if (ilog2_for_n > hash_bits_avail_for_n) ilog2_for_n = hash_bits_avail_for_n;
    uint32 per_call_max_tsize_words = 1u << (ilog2_for_n - 5u);
    if (per_call_max_tsize_words > ctx->fixed_max_tsize_words)
        per_call_max_tsize_words = ctx->fixed_max_tsize_words;

    launch_filter_per_bucket(
            ctx->d_bucket_count, ctx->max_per_bucket,
            ctx->d_arr_a, ctx->d_arr_b,
            ctx->d_candidate_keys, ctx->d_candidate_cnt,
            ctx->d_candidate_overflow, CANDIDATE_CAP,
            key_bits, per_call_max_tsize_words, stream);
    CUDA_TRY(cudaEventRecord(ctx->ev_b_done, stream));

    /* --- Phase C: sort, dedup, build hash, scan originals. --- */
    /* Get candidate count to host so we can size CUB sort + scan. */
    uint32 cand_cnt = 0;
    CUDA_TRY(cudaMemcpyAsync(&cand_cnt, ctx->d_candidate_cnt,
                             sizeof(uint32), cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaMemcpyAsync(ctx->h_candidate_overflow,
                             ctx->d_candidate_overflow,
                             sizeof(uint32), cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaStreamSynchronize(stream));

    if (*ctx->h_bucket_overflow || *ctx->h_max_bucket > ctx->max_per_bucket) {
        fprintf(stderr,
                "gpu_collision_search: bucket overflow max=%u cap=%u\n",
                *ctx->h_max_bucket, ctx->max_per_bucket);
        exit(1);
    }
    if (*ctx->h_candidate_overflow || cand_cnt > CANDIDATE_CAP) {
        fprintf(stderr, "gpu_collision_search: candidate overflow %u > %u\n",
                cand_cnt, CANDIDATE_CAP);
        exit(1);
    }

    if (cand_cnt > 0) {
        cub::DeviceRadixSort::SortKeys(
                ctx->d_cub_sort_candidates, ctx->sort_candidates_bytes,
                ctx->d_candidate_keys, ctx->d_sorted_keys, cand_cnt,
                0, sizeof(uint64) * 8, stream);
        launch_dedup(ctx->d_sorted_keys, cand_cnt,
                     ctx->d_dedup_keys, ctx->d_dedup_cnt, stream);
        CUDA_TRY(cudaMemcpyAsync(ctx->h_dedup_cnt, ctx->d_dedup_cnt,
                                 sizeof(uint32), cudaMemcpyDeviceToHost,
                                 stream));
        CUDA_TRY(cudaStreamSynchronize(stream));
    } else {
        *ctx->h_dedup_cnt = 0;
    }

    uint32 dedup_cnt = *ctx->h_dedup_cnt;
    uint32 hits = 0;

    if (getenv("BENCH_DEBUG")) {
        fprintf(stderr, "[debug] cand_cnt=%u dedup_cnt=%u\n",
                cand_cnt, dedup_cnt);
    }

    if (dedup_cnt > 0) {
        uint32 c_ilog2 = host_ilog2(dedup_cnt + 1u);
        if (c_ilog2 > MAX_C_ILOG2) c_ilog2 = MAX_C_ILOG2;
        uint32 hash_value = (1u << c_ilog2) - 1u;
        uint32 dsize = (1u << c_ilog2) + 1u;
        uint32 ssize = (1u << c_ilog2) / 32u + 1u;
        if (ssize == 0) ssize = 1u;

        launch_count_secondary(ctx->d_dedup_keys, dedup_cnt, hash_value,
                               ctx->d_D, dsize, ctx->d_S, ssize, stream);
        /* Inclusive scan over D[1..dsize-1] would give the C reference's D.
           Use ExclusiveSum starting from D[0]=0, output to D_scan, with the
           same dsize, then shift conceptually: D_scan[i] = sum of D[0..i-1].
           collision_ref.c uses D[0]=0 and the i+1 increment trick so that after
           prefix sum, D[hv] is the start of hv's slot and D[hv+1] is end.
           Our D_pos starts as a copy of D_scan, mutable cursor for scatter. */
        cub::DeviceScan::ExclusiveSum(
                ctx->d_cub_scan_D, ctx->scan_D_bytes,
                ctx->d_D, ctx->d_D_scan, dsize, stream);
        CUDA_TRY(cudaMemcpyAsync(ctx->d_D_pos, ctx->d_D_scan,
                                 dsize * sizeof(uint32),
                                 cudaMemcpyDeviceToDevice, stream));
        launch_scatter_secondary(ctx->d_dedup_keys, dedup_cnt, hash_value,
                                 ctx->d_D_pos, ctx->d_X, stream);
        launch_scan_originals(ctx->d_keys, n, hash_value,
                              ctx->d_S, ctx->d_D_scan, ctx->d_X,
                              ctx->d_hits, stream);
        CUDA_TRY(cudaMemcpyAsync(ctx->h_hits, ctx->d_hits, sizeof(uint32),
                                 cudaMemcpyDeviceToHost, stream));

        if (verify_values) {
            launch_count_matched_values(
                    ctx->d_keys, ctx->d_values, n, hash_value,
                    ctx->d_S, ctx->d_D_scan, ctx->d_X, dedup_cnt,
                    ctx->d_value_counts, ctx->d_value_match_total,
                    ctx->d_value_overflow, VALUE_MATCH_CAP, stream);
            CUDA_TRY(cudaMemcpyAsync(ctx->h_value_match_total,
                                     ctx->d_value_match_total,
                                     sizeof(uint32), cudaMemcpyDeviceToHost,
                                     stream));
            CUDA_TRY(cudaMemcpyAsync(ctx->h_value_overflow,
                                     ctx->d_value_overflow,
                                     sizeof(uint32), cudaMemcpyDeviceToHost,
                                     stream));
            CUDA_TRY(cudaStreamSynchronize(stream));
            if (*ctx->h_value_overflow ||
                *ctx->h_value_match_total > VALUE_MATCH_CAP) {
                fprintf(stderr,
                        "gpu_collision_search: value-match overflow %u > %u\n",
                        *ctx->h_value_match_total, VALUE_MATCH_CAP);
                exit(1);
            }
            cub::DeviceScan::ExclusiveSum(
                    ctx->d_cub_scan_values, ctx->scan_values_bytes,
                    ctx->d_value_counts, ctx->d_value_offsets,
                    dedup_cnt + 1u, stream);
            CUDA_TRY(cudaMemcpyAsync(ctx->d_value_cursor,
                                     ctx->d_value_offsets,
                                     dedup_cnt * sizeof(uint32),
                                     cudaMemcpyDeviceToDevice, stream));
            launch_scatter_matched_values(
                    ctx->d_keys, ctx->d_values, n, hash_value,
                    ctx->d_S, ctx->d_D_scan, ctx->d_X, dedup_cnt,
                    ctx->d_value_cursor, ctx->d_matched_values,
                    VALUE_MATCH_CAP, stream);
            launch_count_msieve_pairs(
                    ctx->d_matched_values, ctx->d_value_offsets,
                    dedup_cnt, pshift, ctx->d_msieve_hits, stream);
            CUDA_TRY(cudaMemcpyAsync(ctx->h_msieve_hits,
                                     ctx->d_msieve_hits, sizeof(uint32),
                                     cudaMemcpyDeviceToHost, stream));
        } else {
            *ctx->h_value_match_total = 0;
            *ctx->h_value_overflow = 0;
            *ctx->h_msieve_hits = 0;
        }
    } else {
        *ctx->h_hits = 0;
        *ctx->h_value_match_total = 0;
        *ctx->h_value_overflow = 0;
        *ctx->h_msieve_hits = 0;
    }

    CUDA_TRY(cudaEventRecord(ctx->ev_c_done, stream));
    CUDA_TRY(cudaEventRecord(ctx->ev_d2h_done, stream));
    CUDA_TRY(cudaStreamSynchronize(stream));

    hits = *ctx->h_hits;
    if (*ctx->h_value_overflow) {
        fprintf(stderr, "gpu_collision_search: value-match overflow %u > %u\n",
                *ctx->h_value_match_total, VALUE_MATCH_CAP);
        exit(1);
    }
    if (msieve_hits_out)
        *msieve_hits_out = *ctx->h_msieve_hits;

    if (timings) {
        CUDA_TRY(cudaEventElapsedTime(&timings->ms_h2d, ctx->ev_start,    ctx->ev_h2d_done));
        CUDA_TRY(cudaEventElapsedTime(&timings->ms_phase_a, ctx->ev_h2d_done, ctx->ev_a_done));
        CUDA_TRY(cudaEventElapsedTime(&timings->ms_phase_b, ctx->ev_a_done,   ctx->ev_b_done));
        CUDA_TRY(cudaEventElapsedTime(&timings->ms_phase_c, ctx->ev_b_done,   ctx->ev_c_done));
        CUDA_TRY(cudaEventElapsedTime(&timings->ms_d2h,     ctx->ev_c_done,   ctx->ev_d2h_done));
        CUDA_TRY(cudaEventElapsedTime(&timings->ms_total,   ctx->ev_start,    ctx->ev_d2h_done));
        timings->candidate_count = cand_cnt;
        /* max_bucket copy was queued at end of Phase A; safe to read after final sync. */
        timings->bucket_max = *ctx->h_max_bucket;
        timings->bucket_min = 0;   /* not yet wired */
        timings->bucket_overflow = *ctx->h_bucket_overflow;
        timings->candidate_overflow = *ctx->h_candidate_overflow;
        timings->value_overflow = *ctx->h_value_overflow;
        timings->value_match_count = *ctx->h_value_match_total;
        timings->msieve_hit_count = *ctx->h_msieve_hits;
    }

    return hits;
}

uint32 gpu_collision_search(gpu_ctx_t *ctx,
                            const uint64_t *keys, uint32 n,
                            uint32 key_bits, int bucket_hash,
                            gpu_timings_t *timings) {
    return gpu_collision_search_internal(ctx, keys, NULL, n, key_bits,
                                         bucket_hash, 0, timings, NULL);
}

uint32 gpu_collision_search_with_values(gpu_ctx_t *ctx,
                              const uint64_t *keys, const uint32_t *values,
                              uint32_t n, uint32_t key_bits, int bucket_hash,
                              uint32_t pshift, gpu_timings_t *timings,
                              uint32_t *msieve_hits) {
    return gpu_collision_search_internal(ctx, keys, values, n, key_bits,
                                         bucket_hash, pshift, timings,
                                         msieve_hits);
}
