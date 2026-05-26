#ifndef BENCH_GPU_CUH
#define BENCH_GPU_CUH

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpu_ctx_s gpu_ctx_t;

/* Per-phase milliseconds from CUDA events. */
typedef struct {
    float ms_h2d;
    float ms_phase_a;
    float ms_phase_b;
    float ms_phase_c;
    float ms_d2h;
    float ms_total;          /* H2D start -> D2H done */
    uint32_t candidate_count;
    uint32_t bucket_max;
    uint32_t bucket_min;
    uint32_t bucket_overflow;
    uint32_t candidate_overflow;
    uint32_t value_overflow;
    uint32_t value_match_count;
    uint32_t msieve_hit_count;
} gpu_timings_t;

/* Init/free are split out so allocation cost is excluded from per-call timing.
   `max_n` and `max_key_bits` bound the largest run we'll feed it. */
gpu_ctx_t *gpu_ctx_init(uint32_t max_n, uint32_t max_key_bits);
void       gpu_ctx_free(gpu_ctx_t *ctx);

/* Runs A/B/C on host-side `keys` array. Populates `timings` if non-null.
   `bucket_hash`: 0 = take low LOG2_NUM_BUCKETS bits of key as bucket id,
                  1 = mix via Knuth multiplicative hash first.
   Returns the GPU's near-collision count. */
uint32_t gpu_collision_search(gpu_ctx_t *ctx,
                              const uint64_t *keys, uint32_t n,
                              uint32_t key_bits, int bucket_hash,
                              gpu_timings_t *timings);

/* Same collision search, plus a msieve-style value verifier. Values are
   packed as (q_index << pshift) | p. `msieve_hits` may be null. */
uint32_t gpu_collision_search_with_values(gpu_ctx_t *ctx,
                              const uint64_t *keys, const uint32_t *values,
                              uint32_t n, uint32_t key_bits, int bucket_hash,
                              uint32_t pshift, gpu_timings_t *timings,
                              uint32_t *msieve_hits);

#ifdef __cplusplus
}
#endif

#endif
