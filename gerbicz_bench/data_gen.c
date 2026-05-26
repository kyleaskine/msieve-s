#include "data_gen.h"

#define RAND_MULT 2131995753u

uint32_t mwc_next(uint32_t *seed, uint32_t *carry) {
    uint64_t t = (uint64_t)(*seed) * (uint64_t)RAND_MULT + (uint64_t)(*carry);
    *seed  = (uint32_t)t;
    *carry = (uint32_t)(t >> 32);
    return (uint32_t)t;
}

void generate_keys(uint64_t *out, uint32_t n, uint32_t key_bits,
                   uint32_t seed, uint32_t carry) {
    uint32_t s = seed, c = carry;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t hi = (uint64_t)mwc_next(&s, &c);
        uint64_t lo = (uint64_t)mwc_next(&s, &c);
        uint64_t k  = (hi << 32) | lo;
        out[i] = k >> (64 - key_bits);
    }
}

void generate_msieve_values(uint32_t *out, uint32_t n, uint32_t pshift,
                            uint32_t seed, uint32_t carry) {
    uint32_t s = seed, c = carry;
    uint32_t mask = (1u << pshift) - 1u;
    uint32_t num_roots = 30000;
    uint32_t q_index = 0;
    uint32_t j = 0;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t p = mwc_next(&s, &c) & mask;
        if (p == 0)
            p = 1;
        out[i] = (q_index << pshift) | p;
        if (++j == num_roots) {
            j = 0;
            q_index++;
        }
    }
}

static uint32_t gcd32_host(uint32_t a, uint32_t b) {
    while (b != 0) {
        uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

void inject_msieve_duplicates(uint64_t *keys, uint32_t *values, uint32_t n,
                              uint32_t num_dups, uint32_t pshift) {
    if (num_dups == 0 || n < 2)
        return;

    uint32_t mask = (1u << pshift) - 1u;
    uint32_t q_mod = 1u << (32u - pshift);

    for (uint32_t i = 0; i < num_dups; i++) {
        uint32_t a = (uint32_t)(((uint64_t)(i + 1) * 104729u) % n);
        uint32_t b = (uint32_t)(((uint64_t)a + n / 2u + (uint64_t)i * 7919u) % n);
        if (a == b)
            b = (b + 1u) % n;

        if (keys[a] == 0)
            keys[a] = 1;
        keys[b] = keys[a];

        uint32_t q = i % q_mod;
        uint32_t p1 = (1009u + 2u * i) & mask;
        uint32_t p2 = (2003u + 4u * i) & mask;
        if (p1 == 0) p1 = 1;
        if (p2 == 0) p2 = 1;
        while (gcd32_host(p1, p2) != 1) {
            p2 = (p2 + 2u) & mask;
            if (p2 == 0) p2 = 1;
        }

        values[a] = (q << pshift) | p1;
        values[b] = (q << pshift) | p2;
    }
}
