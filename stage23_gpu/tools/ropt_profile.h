/* Optional host ranges for Nsight Systems. Normal builds have no NVTX dependency
 * or runtime overhead. Use a separate BUILD directory with ROPT_PROFILE=1. */
#pragma once

#ifdef S23_ROPT_PROFILE
#include <nvtx3/nvToolsExt.h>
struct RoptProfileRange {
    explicit RoptProfileRange(const char *name) { nvtxRangePushA(name); }
    ~RoptProfileRange() { nvtxRangePop(); }
    RoptProfileRange(const RoptProfileRange &) = delete;
    RoptProfileRange &operator=(const RoptProfileRange &) = delete;
};
#define S23_ROPT_JOIN_(a, b) a##b
#define S23_ROPT_JOIN(a, b) S23_ROPT_JOIN_(a, b)
#define S23_ROPT_RANGE(name) RoptProfileRange S23_ROPT_JOIN(ropt_profile_, __LINE__)(name)
#else
#define S23_ROPT_RANGE(name) ((void)0)
#endif
