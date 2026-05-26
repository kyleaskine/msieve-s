#ifndef DATA_GEN_H
#define DATA_GEN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Marsaglia multiply-with-carry, period ~2^63.
   Matches collision_ref.c:100-116 verbatim. */
uint32_t mwc_next(uint32_t *seed, uint32_t *carry);

/* Generate `n` keys of `key_bits` width into `out`, mirroring
   collision_ref.c:464-473. Both CPU and GPU paths use the exact same
   keystream so hit counts are bit-comparable. */
void generate_keys(uint64_t *out, uint32_t n, uint32_t key_bits,
                   uint32_t seed, uint32_t carry);

/* Generate msieve-like packed values: (q_index << pshift) | p.
   The q_index changes every 30000 entries, matching collision_ref.c's
   standalone driver shape. */
void generate_msieve_values(uint32_t *out, uint32_t n, uint32_t pshift,
                            uint32_t seed, uint32_t carry);

/* Force a small number of duplicate-key pairs whose packed values pass
   msieve's same-q and gcd(p1,p2)==1 final-hit predicate. */
void inject_msieve_duplicates(uint64_t *keys, uint32_t *values, uint32_t n,
                              uint32_t num_dups, uint32_t pshift);

#ifdef __cplusplus
}
#endif

#endif
