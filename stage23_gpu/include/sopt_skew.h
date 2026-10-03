/* CADO's sopt_get_skewness (polyselect/size_optimization.c): the integer skew used by
 * size optimization's LLL, trunc((|g0 / f_d|)^(1/d)) in doubles. Host only (C and C++);
 * the GPU receives the result, so it never depends on its own pow() matching glibc's. */
#pragma once

#include <math.h>
#include <gmp.h>

static inline void sopt_skew_mpz(mpz_t skew, const mpz_t g0, const mpz_t fd, int d)
{
    double s = pow(fabs(mpz_get_d(g0) / mpz_get_d(fd)), 1.0 / (double)d);
    mpz_set_d(skew, s);
}
