/* Host final ranking, shared by the CPU and CUDA ropt builds. */
#pragma once
#include "cado_io.h"

struct MurphyConfig {
    double Bf = 0, Bg = 0, area = 0; // explicit job bounds required
    int K = 16000, Keval = 64000, maxeval = 384;
};

struct MurphyResult {
    double score = 0, skew = 0, control = 0, proposal = 0, lognorm = 0;
    long translation = 0;
    int calls = 0;
    bool valid = false, accepted = false, limited = false;
};

bool murphy_config_valid(const MurphyConfig &c);
// Input must be primitive. Search starts at MurphyE's best skew of the supplied
// lognorm-translated cell. Validate control and proposal at Keval, retain the
// control unless both search and validation improve. A bad proposal cannot
// discard a valid control. The CLI refiner additionally supports a minimum gain
// and preserving a supplied control skew; these are explicit caller policies.
// Thread-safe; the chosen exact polynomial is returned in p on success.
MurphyResult murphy_refine(cio_poly &p, const MurphyConfig &c);
