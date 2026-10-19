#pragma once
#include <vector>
#include "hd.h"

namespace s23 {
S23_HD bool rs_is_prime(uint32_t p)
{
    if (p < 2) return false;
    for (uint32_t d = 2; d <= p / d; ++d)
        if (p % d == 0) return false;
    return true;
}

inline std::vector<uint32_t> rs_primes(uint32_t bound)
{
    std::vector<uint32_t> primes;
    for (uint32_t p = 2; p <= bound; ++p)
        if (rs_is_prime(p)) primes.push_back(p);
    return primes;
}
} // namespace s23
