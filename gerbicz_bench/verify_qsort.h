#ifndef VERIFY_QSORT_H
#define VERIFY_QSORT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Ground-truth slow verifier: qsort + adjacency scan.
   Returns count of keys with multiplicity >= 2 (each repeated key counted
   once per occurrence), matching the metric collision_ref.c:494-497 reports. */
uint32_t qsort_collision_count(const uint64_t *keys, uint32_t n);

/* Ground-truth msieve final-hit verifier. Values are packed as
   (q_index << pshift) | p. Counts pairs with equal nonzero key, equal q_index,
   and gcd(p1,p2)==1, matching stage1_core.cu final kernels. */
uint32_t qsort_msieve_hit_count(const uint64_t *keys, const uint32_t *values,
                                uint32_t n, uint32_t pshift);

#ifdef __cplusplus
}
#endif

#endif
