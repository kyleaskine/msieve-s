/* CADO-compatible homogeneous evaluation and quadrature directions. Shared by
 * the CPU scorer, its audit oracle and CUDA; contraction stays disabled apart
 * from the explicit fma. Independent tests compare against CADO's own loop. */
#pragma once
#include <cmath>

#ifdef __CUDACC__
#define S23_MURPHY_HD __host__ __device__
#else
#define S23_MURPHY_HD
#endif

namespace s23_murphy {
static constexpr double PI = 3.14159265358979324;
S23_MURPHY_HD inline double angle(int i, int K)
{
    return PI / double(K) * (double(i) + .5);
}
template <typename Coefficients>
S23_MURPHY_HD inline double homogeneous(const Coefficients &f, int d, double x, double y)
{
    double value = f[d], py = y;
    for (int k = d; k--;) {
        value *= x;
        value = fma(py, f[k], value);
        if (k) py *= y;
    }
    return value;
}
} // namespace s23_murphy
#undef S23_MURPHY_HD
