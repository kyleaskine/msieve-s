#include "cado.h" // IWYU pragma: keep
#include <cmath>
#include <limits>
#include "murphy_rank.h"
#include "murphy_search.h"
#include "polyselect_alpha.h"
#include "polyselect_norms.h"

using namespace s23_murphy;

double murphy_integral_reference(const MurphyIntegralInput &in)
{
    return integral(in.f, in.degree, in.g, in.skew, in.af, in.ag,
                    Ctx{in.Bf, in.Bg, in.area, in.K});
}

bool murphy_config_valid(const MurphyConfig &c)
{
    return std::isfinite(c.Bf) && c.Bf > 1 && std::isfinite(c.Bg) && c.Bg > 1 &&
           std::isfinite(c.area) && c.area > 0 && c.K >= 16000 &&
           (long long)c.Keval >= 4LL * c.K && c.maxeval >= 32;
}

static MurphyIntegralInput integral_input(cado_poly_srcptr p, double skew,
                                         double af, double ag, const Ctx &c)
{
    MurphyIntegralInput in;
    in.degree = p->pols[ALG_SIDE]->deg;
    const polynomial<double> f(p->pols[ALG_SIDE]), g(p->pols[RAT_SIDE]);
    for (int i = 0; i <= in.degree; ++i) in.f[i] = f[i];
    for (int i = 0; i < 2; ++i) in.g[i] = g[i];
    in.skew = skew; in.af = af; in.ag = ag;
    in.Bf = c.Bf; in.Bg = c.Bg; in.area = c.area; in.K = c.K;
    return in;
}

// Same grid, top-two ordering and refinement order as best_skew. Evaluating
// the independent grid points in batches amortizes launches and transfers.
static double device_best_skew(cado_poly_srcptr p, double af, double ag,
                               const Ctx &c, MurphyEvaluator &e, bool &ok)
{
    const auto f = p->pols[ALG_SIDE];
    double s0 = pow(fabs(mpz_get_d(mpz_poly_coeff_const(f, 0)) /
                         mpz_get_d(mpz_poly_coeff_const(f, f->deg))), 1.0 / f->deg);
    if (!(s0 > 0 && std::isfinite(s0))) s0 = L2_skewness(f);
    if (!(s0 > 0 && std::isfinite(s0))) { ok = false; return NAN; }
    const int j0 = (int)floor(32 * log2(s0));
    const auto base = integral_input(p, s0, af, ag, c);
    std::vector<MurphyIntegralInput> inputs;
    auto append = [&](double skew) {
        inputs.push_back(base);
        inputs.back().skew = skew;
    };
    for (int j = j0 - 96; j <= j0 + 96; ++j) append(exp2(j / 32.0));
    std::vector<double> values(inputs.size());
    if (!e.evaluate(inputs.data(), inputs.size(), values.data())) { ok = false; return NAN; }
    double best = -1, second = -1;
    int j1 = j0, j2 = j0;
    for (size_t i = 0; i < values.size(); ++i) {
        const int j = j0 - 96 + int(i);
        if (values[i] > best) { second = best; j2 = j1; best = values[i]; j1 = j; }
        else if (values[i] > second) { second = values[i]; j2 = j; }
    }
    double skew = exp2(j1 / 32.0);
    inputs.clear();
    for (int j : {j1, j2})
        for (int k = -7; k <= 7; ++k)
            if (k) append(exp2((8 * j + k) / 256.0));
    values.resize(inputs.size());
    if (!e.evaluate(inputs.data(), inputs.size(), values.data())) { ok = false; return NAN; }
    for (size_t i = 0; i < values.size(); ++i)
        if (values[i] > best) { best = values[i]; skew = inputs[i].skew; }
    return skew;
}

MurphyResult murphy_refine(cio_poly &q, const MurphyConfig &c, MurphyEvaluator *evaluator,
                           const double *known_alpha)
{
    MurphyResult out;
    if (!murphy_config_valid(c) || q.deg < 1 || q.deg > CIO_MAXDEG || !mpz_sgn(q.g[1]))
        return out;
    if (evaluator && evaluator->cancelled()) return out;
    cado_poly p;
    cado_poly_init(p);
    cio_to_cado(p, &q);
    const Ctx train{c.Bf, c.Bg, c.area, c.K}, eval{c.Bf, c.Bg, c.area, c.Keval};
    // One canonical alpha source for CPU, CUDA and recoverable fallbacks. A
    // checked device refinement additionally audits its supplied full alpha.
    const double af = get_alpha(p->pols[ALG_SIDE], get_alpha_bound());
    const double ag = get_alpha(p->pols[RAT_SIDE], get_alpha_bound());
    struct EvaluationStopped {};
    try {
        if (evaluator && (evaluator->cancelled() ||
            (known_alpha && !evaluator->check_alpha(*known_alpha, af)))) throw EvaluationStopped{};
        auto score = [&](double skew, const Ctx &ctx) -> double {
            if (!evaluator) return murphy_at(p, skew, af, ag, ctx);
            double value = NAN;
            const auto in = integral_input(p, skew, af, ag, ctx);
            if (evaluator->cancelled() || !evaluator->evaluate(&in, 1, &value)) throw EvaluationStopped{};
            return value;
        };
        bool ok = true;
        const double initial = evaluator ? device_best_skew(p, af, ag, train, *evaluator, ok) : 0;
        if (!ok) throw EvaluationStopped{};
        const auto r = s23_murphy::refine_pair(p, af, ag, train, initial, c.maxeval, 0,
            [&](double skew) { return score(skew, train); },
            [&](double skew) {
                const double e = score(skew, eval);
                return std::array<double, 2>{e, e};
            });
        out.gpu = evaluator != nullptr;
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
    } catch (const EvaluationStopped &) {
        cado_poly_clear(p);
        // An audit mismatch (including another worker's) is fatal. Never hide
        // it by completing this cell or the remaining job on the CPU.
        if (evaluator->cancelled()) return out;
        out = murphy_refine(q, c);
        out.fallback = true;
        return out;
    }
    cado_poly_clear(p);
    return out;
}
