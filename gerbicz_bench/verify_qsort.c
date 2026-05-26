#include "verify_qsort.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

uint32_t qsort_collision_count(const uint64_t *keys, uint32_t n) {
    uint64_t *a = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    memcpy(a, keys, (size_t)n * sizeof(uint64_t));
    qsort(a, n, sizeof(uint64_t), cmp_u64);

    uint32_t hits = 0;
    for (uint32_t i = 0; i < n; i++) {
        int left  = (i > 0)     && (a[i] == a[i - 1]);
        int right = (i + 1 < n) && (a[i] == a[i + 1]);
        if (left || right) hits++;
    }
    free(a);
    return hits;
}

typedef struct {
    uint64_t key;
    uint32_t value;
} key_value_t;

static int cmp_key_value(const void *a, const void *b) {
    const key_value_t *x = (const key_value_t *)a;
    const key_value_t *y = (const key_value_t *)b;
    if (x->key != y->key)
        return (x->key > y->key) - (x->key < y->key);
    return (x->value > y->value) - (x->value < y->value);
}

static uint32_t gcd32_host(uint32_t a, uint32_t b) {
    while (b != 0) {
        uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

uint32_t qsort_msieve_hit_count(const uint64_t *keys, const uint32_t *values,
                                uint32_t n, uint32_t pshift) {
    key_value_t *a = (key_value_t *)malloc((size_t)n * sizeof(key_value_t));
    uint32_t mask = (pshift >= 32) ? UINT32_MAX : ((1u << pshift) - 1u);
    uint32_t hits = 0;

    for (uint32_t i = 0; i < n; i++) {
        a[i].key = keys[i];
        a[i].value = values[i];
    }
    qsort(a, n, sizeof(key_value_t), cmp_key_value);

    for (uint32_t lo = 0; lo < n; ) {
        uint32_t hi = lo + 1;
        while (hi < n && a[hi].key == a[lo].key)
            hi++;

        if (a[lo].key != 0 && hi - lo >= 2) {
            for (uint32_t i = lo; i + 1 < hi; i++) {
                uint32_t v1 = a[i].value;
                uint32_t q1 = (pshift >= 32) ? 0 : (v1 >> pshift);
                uint32_t p1 = v1 & mask;

                for (uint32_t j = i + 1; j < hi; j++) {
                    uint32_t v2 = a[j].value;
                    uint32_t q2 = (pshift >= 32) ? 0 : (v2 >> pshift);
                    uint32_t p2 = v2 & mask;

                    if (q1 == q2 && gcd32_host(p1, p2) == 1)
                        hits++;
                }
            }
        }

        lo = hi;
    }

    free(a);
    return hits;
}
