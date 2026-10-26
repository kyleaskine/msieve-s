/* Final ranking interface, shared by the CPU and CUDA ropt builds. */
#pragma once
#include <cstddef>
#include "cado_io.h"

// Plain-data boundary: CADO's C++20 types never enter the CUDA translation unit.
// Coefficients are converted from exact integers by CADO's number_context.
struct MurphyIntegralInput {
    double f[CIO_MAXDEG + 1]{}, g[2]{};
    double skew = 0, af = 0, ag = 0, Bf = 0, Bg = 0, area = 0;
    int degree = 0, K = 0;
};

struct MurphyEvaluator {
    virtual ~MurphyEvaluator() = default;
    // All entries in a batch have the same K. Recoverable failure retries the
    // refinement from its unchanged input with the CPU oracle. cancelled()
    // instead makes failure fatal, without a CPU retry.
    virtual bool evaluate(const MurphyIntegralInput *input, size_t n, double *output) = 0;
    // Numerical audit failures cancel the job, unlike recoverable device errors.
    virtual bool cancelled() const { return false; }
    virtual bool check_alpha(double, double) { return true; }
};

// Scalar oracle, with CADO's rho and angle order, for optional CUDA audits.
double murphy_integral_reference(const MurphyIntegralInput &in);

struct MurphyConfig {
    double Bf = 0, Bg = 0, area = 0; // explicit job bounds required
    int K = 16000, Keval = 64000, maxeval = 384;
};

struct MurphyResult {
    double score = 0, skew = 0, control = 0, proposal = 0, lognorm = 0;
    long translation = 0;
    int calls = 0;
    bool valid = false, accepted = false, limited = false;
    bool gpu = false, fallback = false;
};

bool murphy_config_valid(const MurphyConfig &c);
// Input must be primitive. Search starts at MurphyE's best skew of the supplied
// lognorm-translated cell. Validate control and proposal at Keval, retain the
// control unless both search and validation improve. A bad proposal cannot
// discard a valid control. The CLI refiner additionally supports a minimum gain
// and preserving a supplied control skew; these are explicit caller policies.
// Thread-safe with a private evaluator per worker. known_alpha, if supplied,
// is audited against full-bound CADO alpha; all paths use CADO alpha for ranking.
// It must describe this primitive polynomial. The chosen
// exact polynomial is returned in p on success.
MurphyResult murphy_refine(cio_poly &p, const MurphyConfig &c, MurphyEvaluator *evaluator = nullptr,
                           const double *known_alpha = nullptr);
