/* Host-only final-ranking storage. Allocate after the proxy pool is cut, never
 * in the sieve's hot candidate path. Retain the exact selected polynomial. */
#pragma once
#include <cmath>
#include "murphy_rank.h"
#include "rhost.h"

struct MurphyCell {
    cio_poly poly;
    MurphyResult result;
    unsigned long content = 1;
    bool attempted = false;

    MurphyCell() { cio_init(&poly); }
    ~MurphyCell() { cio_clear(&poly); }
    MurphyCell(const MurphyCell &) = delete;
    MurphyCell &operator=(const MurphyCell &) = delete;

    template <int L>
    void refine(const cio_poly &seed, const s23::RsizePoly<L> &size,
                int64_t u, int64_t v, int64_t t, double lognorm, double alpha,
                const MurphyConfig &config)
    {
        // Nonfinite proxy metrics and fixed-width overflows predate MurphyE.
        // Exclude them as before, without turning a good seed into a failed run.
        if (!std::isfinite(lognorm) || !std::isfinite(alpha) ||
            !s23::rs_written_poly(poly, seed, size, u, v, t, content))
            return;
        attempted = true;
        result = murphy_refine(poly, config);
        long total;
        if (__builtin_add_overflow((long)t, result.translation, &total))
            result.valid = false;
    }

    bool failed() const { return attempted && !result.valid; }
};
