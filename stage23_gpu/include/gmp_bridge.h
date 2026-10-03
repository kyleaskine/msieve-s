/* Host-only conversions between s23::Int<L> and GMP, for the unit tests. */
#pragma once

#include <gmp.h>
#include "mpint.h"

namespace s23 {

template <int L> void to_mpz(mpz_t z, const Int<L> &a)
{
    uint32_t m[L];
    magnitude(m, a);
    mpz_import(z, L, -1, sizeof(uint32_t), 0, 0, m);
    if (is_neg(a))
        mpz_neg(z, z);
}

/* false if z does not fit in Int<L> */
template <int L> bool from_mpz(Int<L> &a, const mpz_t z)
{
    if (mpz_sizeinbase(z, 2) > (size_t)(32 * L))
        return false;
    uint32_t m[L] = {0};
    size_t count = 0;
    mpz_export(m, &count, -1, sizeof(uint32_t), 0, 0, z);
    return from_magnitude(a, m, mpz_sgn(z) < 0);
}

} // namespace s23
