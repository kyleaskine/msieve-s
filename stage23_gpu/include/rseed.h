/* Host helper for the root sieve, shared by tools/s23_ropt.cu and tests/test_rsieve.cpp:
 * the seed's residues of f and g modulo each prime's table period and projective modulus
 * (rsieve.h) from a
 * CADO-format polynomial read by tools/cado_io. */
#pragma once

#include <gmp.h>
#include "cado_io.h"
#include "rsieve.h"

namespace s23 {

/* false if the polynomial is not of degree RS_DEG */
inline bool rs_seed_from_poly(RsSeed &S, const cio_poly &P)
{
    if (P.deg != RS_DEG)
        return false;
    rs_init_primes(S);
    for (int i = 0; i < RS_NPRIMES; i++) {
        for (int k = 0; k <= RS_DEG; k++)
            S.f[i][k] = mpz_fdiv_ui(P.f[k], S.pr[i].qmax);
        S.g[i][0] = mpz_fdiv_ui(P.g[0], S.pr[i].qmax);
        S.g[i][1] = mpz_fdiv_ui(P.g[1], S.pr[i].qmax);
        for (int k = 0; k <= RS_DEG; k++)
            S.fp[i][k] = mpz_fdiv_ui(P.f[k], S.pr[i].qproj);
        S.gp[i][0] = mpz_fdiv_ui(P.g[0], S.pr[i].qproj);
        S.gp[i][1] = mpz_fdiv_ui(P.g[1], S.pr[i].qproj);
    }
    return true;
}

} // namespace s23
