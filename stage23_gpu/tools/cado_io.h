/* Thin C interface to CADO-NFS for the stage23_gpu tools (CADO's headers are not all
 * C++-safe): read CADO-format polynomials, print results in CADO sopt's format, compute
 * sopt's LLL skew exactly as CADO, and run CADO's own size_optimization (the CPU
 * fallback and reference). */
#pragma once

#include <stdio.h>
#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CIO_MAXDEG 10 /* CADO's MAX_DEGREE */

typedef struct {
    int deg;
    mpz_t n;
    mpz_t f[CIO_MAXDEG + 1];
    mpz_t g[2];
} cio_poly;

void cio_init(cio_poly *p);
void cio_clear(cio_poly *p);
void cio_set(cio_poly *dst, const cio_poly *src);

/* next polynomial from a CADO-format stream (# lines and blank lines between blocks
 * skipped): 1 if one was read, 0 at the end of the input, -1 for a malformed block, a
 * read error, or a pair this interface cannot hold (f of degree > CIO_MAXDEG, g not
 * linear), with CADO's message or ours on stderr */
int cio_read_next(FILE *in, cio_poly *p);

/* sopt_get_skewness: trunc((|g0 / f_d|)^(1/d)) */
void cio_sopt_skew(mpz_t skew, const cio_poly *raw);

/* CADO's size_optimization(f_opt, g_opt, f_raw, g_raw, effort, 0); returns its value */
double cio_cado_sopt(cio_poly *opt, const cio_poly *raw, unsigned effort);

/* "### Input raw polynomial (idx) ###" (commented, with stats if raw_stats) followed by
 * "### Size-optimized polynomial (idx) ###" with CADO's expected stats, as sopt prints */
void cio_print_pair(FILE *out, unsigned idx, const cio_poly *raw, const cio_poly *opt, int raw_stats);

/* CADO's projective alpha (get_alpha_projective, primes < B) of the rotation
 * f + (u x + v) g of p: a reference for the root sieve's projective tables */
double cio_alpha_projective_rot(const cio_poly *p, long u, long v, unsigned long B);

#ifdef __cplusplus
}
#endif
