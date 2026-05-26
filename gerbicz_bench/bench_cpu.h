#ifndef BENCH_CPU_H
#define BENCH_CPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Runs Gerbicz's fun_collision_search on a host-side key array.
   Returns the near-collision count (multiplicity-counted repeats). */
uint32_t cpu_collision_search(const uint64_t *keys, uint32_t n,
                              uint32_t key_bits, int allow_loss);

/* OpenMP-parallel version. num_threads=0 means default (OMP_NUM_THREADS). */
uint32_t omp_collision_search(const uint64_t *keys, uint32_t n,
                              uint32_t key_bits, int allow_loss,
                              int num_threads);

#ifdef __cplusplus
}
#endif

#endif
