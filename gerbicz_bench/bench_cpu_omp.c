/* OpenMP-parallel version of fun_collision_search.
   Algorithm is a near-copy of collision_ref.c:fun_collision_search; the per-bucket
   filter loop is parallelized with OpenMP. Bucket scatter and final verify
   stay serial — they're fast and the per-bucket filter dominates CPU time.

   To use: compile with -fopenmp. Without -fopenmp, falls back to sequential
   execution (OpenMP pragmas are ignored).

   Same key_bits limit as collision_ref.c: key_bits <= 32 + log2(num_buckets) = 40. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <assert.h>
#include "bench_cpu.h"

#ifdef _OPENMP
#include <omp.h>
#endif

typedef uint8_t  uint8;
typedef uint32_t uint32;
typedef uint64_t uint64;

#define MIN(a,b) ((a) < (b)? (a) : (b))
#define MAX(a,b) ((a) > (b)? (a) : (b))

/* Constants — match collision_ref.c exactly. */
#define BUCKET_SIZE      256u
#define BS1              255u
#define SH_              8u
#define NUM_BUCKETS_OMP  256u
#define HV_              255u
#define LOG2_NUM_BUCKETS_ 8u
#define D_EXTRA_BITS     4.5

static int cmp_u64(const void *a, const void *b) {
    uint64 x = *(const uint64 *)a, y = *(const uint64 *)b;
    return (x > y) - (x < y);
}

uint32 omp_collision_search(const uint64 *keys, uint32 n,
                            uint32 key_bits, int allow_loss,
                            int num_threads) {
    assert(key_bits >= 6u + LOG2_NUM_BUCKETS_);
    assert(key_bits <= 32u + LOG2_NUM_BUCKETS_);

#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif

    /* --- Bucket scatter (serial, fast). Mirrors collision_ref.c:213-223. */
    uint32 offset[NUM_BUCKETS_OMP];
    uint32 lenb[NUM_BUCKETS_OMP];
    uint32 ns = NUM_BUCKETS_OMP + 1u + (n / BUCKET_SIZE);
    uint32 last_bucket = NUM_BUCKETS_OMP - 1u;
    uint32 *nextblock = (uint32 *)malloc(ns * sizeof(uint32));
    uint32 *buckets   = (uint32 *)malloc((size_t)ns * BUCKET_SIZE * sizeof(uint32));

    for (uint32 i = 0; i < NUM_BUCKETS_OMP; i++) {
        offset[i] = i << SH_;
        lenb[i] = 0;
    }
    for (uint32 i = 0; i < n; i++) {
        uint64 hv = keys[i] & HV_;
        buckets[offset[hv]++] = (uint32)(keys[i] >> LOG2_NUM_BUCKETS_);
        if ((offset[hv] & BS1) == 0u) {
            nextblock[(offset[hv] - 1u) >> SH_] = (++last_bucket);
            offset[hv] = last_bucket << SH_;
            lenb[hv] += BUCKET_SIZE;
        }
    }
    nextblock[last_bucket] = 0;

    uint32 maxsize = 1;
    for (uint32 i = 0; i < NUM_BUCKETS_OMP; i++) {
        lenb[i] += (offset[i] & BS1);
        maxsize = MAX(maxsize, lenb[i]);
    }

    /* --- Parallel per-bucket filter.
       Each thread emits to its own local_C; after the parallel region we
       concatenate all local_C buffers into the global C serially. */
    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif

    int init_ilog2 = MAX(5, (int)(log((double)(maxsize + 1)) / log(2.0) + D_EXTRA_BITS));
    if ((uint32)init_ilog2 + LOG2_NUM_BUCKETS_ > key_bits)
        init_ilog2 = (int)key_bits - (int)LOG2_NUM_BUCKETS_;
    uint32 tsize_max = 1u << ((uint32)init_ilog2 - 5u);

    uint32 **all_arr        = (uint32 **)calloc(max_threads, sizeof(uint32 *));
    uint32 **all_T          = (uint32 **)calloc(max_threads, sizeof(uint32 *));
    uint32 **all_T2         = (uint32 **)calloc(max_threads, sizeof(uint32 *));
    uint32 **all_T3         = (uint32 **)calloc(max_threads, sizeof(uint32 *));
    uint64 **all_local_C    = (uint64 **)calloc(max_threads, sizeof(uint64 *));
    uint32  *all_local_csize = (uint32 *)calloc(max_threads, sizeof(uint32));
    for (int t = 0; t < max_threads; t++) {
        all_arr[t] = (uint32 *)malloc((size_t)maxsize * sizeof(uint32));
        all_T[t]   = (uint32 *)malloc((size_t)tsize_max * sizeof(uint32));
        all_T2[t]  = (uint32 *)malloc((size_t)tsize_max * sizeof(uint32));
        all_T3[t]  = (uint32 *)malloc((size_t)tsize_max * sizeof(uint32));
        all_local_C[t] = (uint64 *)malloc(1024 * sizeof(uint64));
    }

    #pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        uint32 *arr = all_arr[tid];
        uint32 *T   = all_T[tid];
        uint32 *T2  = all_T2[tid];
        uint32 *T3  = all_T3[tid];

        /* Per-thread local C buffer; growable. Stored in all_local_C[tid]
           so the main thread can concatenate after the parallel region. */
        uint32 local_cap = 1024;
        uint32 local_csize = 0;
        uint64 *local_C = all_local_C[tid];

        #pragma omp for schedule(dynamic, 4)
        for (uint32 ii = 0; ii < NUM_BUCKETS_OMP; ii++) {
            uint32 i = ii;

            if (lenb[i] == 0) continue;

            uint32 cnt = 0;
            uint32 *U, *U2, *U3;

            int ilog2 = MAX(5, (int)(log((double)(lenb[i] + 1)) / log(2.0) + D_EXTRA_BITS));
            if ((uint32)ilog2 + LOG2_NUM_BUCKETS_ > key_bits)
                ilog2 = (int)key_bits - (int)LOG2_NUM_BUCKETS_;
            uint32 my_shift, my_shift2 = 0;
            uint32 hash_value, hash_value2 = (1u << ilog2) - 1u;
            uint32 tsize_cur = 1u << ((uint32)ilog2 - 5u);

            for (uint32 j = 0; j < tsize_cur; j++) T[j] = 0;
            if (!allow_loss) {
                for (uint32 j = 0; j < tsize_cur; j++) T2[j] = 0;
            }

            uint32 block = i;
            uint32 p = BUCKET_SIZE * i;
            for (;;) {
                uint32 en = MIN(offset[i], p + BUCKET_SIZE);
                if (!allow_loss) {
                    for (; p < en; ) {
                        uint32 hv = buckets[p] & hash_value2;
                        arr[cnt++] = buckets[p++];
                        if (T2[hv >> 5] & (1u << (hv & 31))) {
                            T[hv >> 5] |= (1u << (hv & 31));
                        } else {
                            T2[hv >> 5] |= (1u << (hv & 31));
                        }
                    }
                } else {
                    for (; p < en; ) {
                        uint32 hv = buckets[p] & hash_value2;
                        arr[cnt++] = buckets[p++];
                        T[hv >> 5] ^= (1u << (hv & 31));
                    }
                }
                if (p == offset[i]) break;
                block = nextblock[block];
                if (block == 0) break;
                p = block * BUCKET_SIZE;
            }

            /* Ping-pong filter rounds (matches collision_ref.c:298-381). */
            #define MAXIT_OMP 20
            uint32 nsize[MAXIT_OMP];
            int converged = 0;
            for (int it = 0; it < MAXIT_OMP; it++) {
                my_shift = my_shift2;
                hash_value = hash_value2;

                ilog2 = MAX(5, (int)(log((double)cnt) / log(2.0) + D_EXTRA_BITS));
                if ((uint32)ilog2 > key_bits - LOG2_NUM_BUCKETS_)
                    ilog2 = (int)key_bits - (int)LOG2_NUM_BUCKETS_;

                if ((it & 1) == 0) { U = T;  U2 = T2; U3 = T3; }
                else               { U = T3; U2 = T2; U3 = T;  }

                my_shift2 = my_shift + 6u;
                if (my_shift2 + (uint32)ilog2 > key_bits - LOG2_NUM_BUCKETS_) my_shift2 = 0;
                hash_value2 = (1u << ilog2) - 1u;
                uint32 my_size = 1u << ((uint32)ilog2 - 5u);

                for (uint32 j = 0; j < my_size; j++) U2[j] = 0;
                for (uint32 j = 0; j < my_size; j++) U3[j] = 0;

                uint32 cnt2 = 0;
                if (!allow_loss || it > 0) {
                    for (uint32 j = 0; j < cnt; j++) {
                        uint32 hv = (arr[j] >> my_shift) & hash_value;
                        if (U[hv >> 5] & (1u << (hv & 31))) {
                            uint32 hv2 = (arr[j] >> my_shift2) & hash_value2;
                            if (U2[hv2 >> 5] & (1u << (hv2 & 31))) {
                                U3[hv2 >> 5] |= 1u << (hv2 & 31);
                            } else {
                                U2[hv2 >> 5] |= 1u << (hv2 & 31);
                            }
                            arr[cnt2++] = arr[j];
                        }
                    }
                } else {
                    for (uint32 j = 0; j < cnt; j++) {
                        uint32 hv = (arr[j] >> my_shift) & hash_value;
                        if ((U[hv >> 5] & (1u << (hv & 31))) == 0) {
                            uint32 hv2 = (arr[j] >> my_shift2) & hash_value2;
                            if (U2[hv2 >> 5] & (1u << (hv2 & 31))) {
                                U3[hv2 >> 5] |= 1u << (hv2 & 31);
                            } else {
                                U2[hv2 >> 5] |= 1u << (hv2 & 31);
                            }
                            arr[cnt2++] = arr[j];
                        }
                    }
                }
                cnt = cnt2;
                nsize[it] = cnt;
                if (cnt == 0 || it == MAXIT_OMP - 1 ||
                    (it >= 3 && nsize[it - 3] == nsize[it])) {
                    /* Emit survivors into thread-local C. */
                    while (local_csize + cnt > local_cap) {
                        local_cap += local_cap / 2;
                        local_C = (uint64 *)realloc(local_C, local_cap * sizeof(uint64));
                    }
                    for (uint32 j = 0; j < cnt; j++) {
                        local_C[local_csize++] = (((uint64)arr[j]) << LOG2_NUM_BUCKETS_) + i;
                    }
                    converged = 1;
                    break;
                }
            }
            (void)converged;
        } /* for buckets */

        /* Store local results for serial merge after parallel region. */
        all_local_C[tid] = local_C;
        all_local_csize[tid] = local_csize;
    } /* parallel */

    /* Serial merge of all per-thread local_C buffers into global C. */
    uint32 csize = 0;
    for (int t = 0; t < max_threads; t++) csize += all_local_csize[t];
    uint64 *C = (uint64 *)malloc((size_t)MAX(16u, csize) * sizeof(uint64));
    uint32 pos = 0;
    for (int t = 0; t < max_threads; t++) {
        memcpy(C + pos, all_local_C[t], all_local_csize[t] * sizeof(uint64));
        pos += all_local_csize[t];
    }

    for (int t = 0; t < max_threads; t++) {
        free(all_arr[t]); free(all_T[t]); free(all_T2[t]); free(all_T3[t]);
        free(all_local_C[t]);
    }
    free(all_arr); free(all_T); free(all_T2); free(all_T3);
    free(all_local_C); free(all_local_csize);

    /* --- Verify step: same as collision_ref.c:384-442, serial. */
    qsort(C, csize, sizeof(uint64), cmp_u64);
    uint32 csize2 = 0;
    for (uint32 i = 1; i < csize; i++) {
        if (C[i] == C[i-1] && (i == csize - 1 || C[i] != C[i+1])) {
            C[csize2++] = C[i];
        }
    }
    csize = csize2;
    if (csize == 0) {
        free(nextblock); free(buckets); free(C);
        return 0;
    }

    int ilog2v = MAX(5, (int)(log((double)(csize + 1)) / log(2.0) + D_EXTRA_BITS));
    int ssize  = 1u << ((uint32)ilog2v - 5u);
    uint32 hash_value_v = (1u << ilog2v) - 1u;
    int dsize  = (1u << ilog2v) + 1;
    uint32 *S = (uint32 *)calloc(ssize, sizeof(uint32));
    uint32 *D = (uint32 *)calloc(dsize, sizeof(uint32));
    uint64 *X = (uint64 *)malloc((size_t)csize * sizeof(uint64));

    for (uint32 i = 0; i < csize; i++) {
        uint32 hv = (uint32)(C[i] & hash_value_v);
        D[hv + 1u]++;
        S[hv >> 5] |= (1u << (hv & 31u));
    }
    for (int i = 1; i < dsize; i++) D[i] += D[i-1];
    for (uint32 i = 0; i < csize; i++) {
        X[D[C[i] & hash_value_v]++] = C[i];
    }

    uint32 near_collision = 0;
    for (uint32 i = 0; i < n; i++) {
        uint32 hv = (uint32)(keys[i] & hash_value_v);
        if (S[hv >> 5] & (1u << (hv & 31u))) {
            uint32 en = D[hv];
            for (uint32 j = (hv == 0 ? 0 : D[hv - 1]); j < en; j++) {
                if (keys[i] == X[j]) {
                    near_collision++;
                    break;
                }
            }
        }
    }

    free(nextblock); free(buckets); free(C);
    free(S); free(D); free(X);
    return near_collision;
}
