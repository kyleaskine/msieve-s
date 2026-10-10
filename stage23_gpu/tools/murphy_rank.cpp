#include "cado.h" // IWYU pragma: keep
#include <cmath>
#include <limits>
#include "murphy_rank.h"
#include "murphy_search.h"
#include "polyselect_alpha.h"
#include "polyselect_norms.h"

using namespace s23_murphy;

bool murphy_config_valid(const MurphyConfig &c)
{
    return std::isfinite(c.Bf) && c.Bf > 1 && std::isfinite(c.Bg) && c.Bg > 1 &&
           std::isfinite(c.area) && c.area > 0 && c.K >= 16000 &&
           (long long)c.Keval >= 4LL * c.K && c.maxeval >= 32;
}

MurphyResult murphy_refine(cio_poly &q, const MurphyConfig &c)
{
    MurphyResult out;
    if (!murphy_config_valid(c) || q.deg < 1 || q.deg > CIO_MAXDEG || !mpz_sgn(q.g[1]))
        return out;
    cado_poly p;
    cado_poly_init(p);
    cio_to_cado(p, &q);
    const Ctx train{c.Bf, c.Bg, c.area, c.K}, eval{c.Bf, c.Bg, c.area, c.Keval};
    const double af = get_alpha(p->pols[ALG_SIDE], get_alpha_bound());
    const double ag = get_alpha(p->pols[RAT_SIDE], get_alpha_bound());
    const auto r = s23_murphy::refine_pair(p, af, ag, train, 0, c.maxeval, 0,
        [&](double skew) { return murphy_at(p, skew, af, ag, train); },
        [&](double skew) {
            const double e = murphy_at(p, skew, af, ag, eval);
            return std::array<double, 2>{e, e};
        });
    out.calls = r.calls;
    out.limited = r.limited;
    out.control = r.eval0;
    out.proposal = r.eval1;
    out.valid = out.control > 0 && std::isfinite(out.control);
    out.accepted = out.valid && proposal_improves(r);
    out.score = out.accepted ? out.proposal : out.control;
    out.skew = out.accepted ? r.skew : r.start_skew;
    out.translation = out.accepted ? r.t : 0;
    if (!out.accepted)
        translate_pair(p, -r.t);
    if (out.valid) {
        out.lognorm = L2_lognorm(p->pols[ALG_SIDE], out.skew);
        out.valid = std::isfinite(out.lognorm) && cio_from_cado(&q, p);
    }
    cado_poly_clear(p);
    return out;
}
