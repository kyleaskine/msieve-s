/* Host/device portability. Everything in stage23_gpu/include is written once and
 * compiles both with nvcc (device code) and with a plain C++ compiler (host), so the
 * arithmetic can be unit-tested on the CPU against GMP and CADO without a GPU. */
#pragma once

#ifdef __CUDACC__
#define S23_HD __host__ __device__ __forceinline__
/* Large multi-limb routines (mul, division): real calls on the device. Inlining them
   everywhere multiplies code size, compile time and register pressure. */
#define S23_HD_CALL __host__ __device__ __noinline__
#else
#define S23_HD inline
#define S23_HD_CALL inline
#endif

#include <stdint.h>

namespace s23 {

/* Number of leading zero bits of a nonzero 32-bit word. */
S23_HD int clz32(uint32_t x)
{
#if defined(__CUDA_ARCH__)
    return __clz((int)x);
#else
    return __builtin_clz(x);
#endif
}

} // namespace s23
