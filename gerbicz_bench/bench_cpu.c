/* Wrap collision_ref.c so the bench harness can call it without symbol clash.
   The reference algorithm itself is unmodified. */

#define BENCH_NO_MAIN
#include "collision_ref.c"

#include "bench_cpu.h"
#include <string.h>

uint32_t cpu_collision_search(const uint64_t *keys, uint32_t n,
                              uint32_t key_bits, int allow_loss) {
    cpu_thread_data_t *ctx = cpu_thread_data_init();
    grow_sort(ctx, n);
    memcpy(ctx->sort_key1, keys, (size_t)n * sizeof(uint64_t));
    /* sort_data1 is not read by fun_collision_search; leave uninitialised. */
    ctx->num_sort = n;
    uint32_t hits = fun_collision_search(ctx, key_bits, allow_loss ? 1 : 0);
    cpu_thread_data_free(ctx);
    return hits;
}
