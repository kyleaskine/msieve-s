/* Logarithms used by the size-optimization objective. Only the production sopt
 * translation unit opts into the host-compatible device implementation. */
#pragma once
#include "mpint.h"
#ifdef S23_SOPT_HOST_LOG
#include "host_log.h"
#endif

namespace s23 {
S23_HD double sopt_log(double x)
{
#ifdef S23_SOPT_HOST_LOG
    return matched_log::log(x);
#else
    return log(x);
#endif
}
} // namespace s23
