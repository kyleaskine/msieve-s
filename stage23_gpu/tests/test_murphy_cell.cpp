/* Pre-existing invalid cells are skipped, actual scorer failures still fail. */
#include <cstdio>
#include <climits>
#include "murphy_cell.h"

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    FILE *in = fopen(argv[1], "r");
    if (!in) return 2;
    cio_poly p;
    cio_init(&p);
    const int status = cio_read_next(in, &p);
    fclose(in);
    if (status != 1) return 2;
    s23::RsizePoly<16> size;
    if (!s23::rs_size_poly(size, p)) return 2;
    MurphyConfig config;
    config.Bf = 17179869184.0;
    config.Bg = 8589934592.0;
    config.area = 3.4359738368e17;
    config.maxeval = 32;
    int bad = 0;
    for (int i = 0; i < 3; i++) {
        MurphyCell c;
        c.refine(p, size, 0, 0, 0, i == 0 ? INFINITY : 1, i == 1 ? NAN : 1,
                 i == 2 ? MurphyConfig{} : config);
        bad += i == 2 ? (!c.attempted || !c.failed()) : (c.attempted || c.failed());
    }
    // Cover failure in a search batch AND in held-out proposal validation,
    // where refine_pair_batched has left its working polynomial translated.
    // Both must restart from the untouched caller input.
    struct FailingEvaluator : MurphyEvaluator {
        int calls = 0, validations = 0;
        bool proposal = false, translated_failure = false, fatal = false, stopped = false;
        double original_g0 = 0;
        MurphyIntegralInput control{};
        bool cancelled() const override { return stopped; }
        bool evaluate(const MurphyIntegralInput *input, size_t n, double *output) override {
            ++calls;
            if (calls == 1) original_g0 = input[0].g[0];
            if (n == 1 && input[0].K == 64000) {
                if (++validations == 1) control = input[0];
                if (proposal && validations == 2) {
                    for (int i = 0; i <= input[0].degree; ++i)
                        translated_failure |= input[0].f[i] != control.f[i];
                    for (int i = 0; i < 2; ++i)
                        translated_failure |= input[0].g[i] != control.g[i];
                    stopped = fatal;
                    return false;
                }
            }
            if (!proposal && calls == 4) return false;
            for (size_t i = 0; i < n; ++i) {
                // Steer the proposal case to a nonzero translation regardless
                // of this fixture's true local optimum or host libm rounding.
                // These training scores are discarded on failure; recovery
                // must still match a complete, real CPU search of the input.
                output[i] = proposal && input[i].K != 64000
                    ? (input[i].g[0] == original_g0 ? 1.0 : 2.0)
                    : murphy_integral_reference(input[i]);
            }
            return true;
        }
    };
    cio_poly expected, recovered;
    cio_init(&expected); cio_init(&recovered);
    cio_set(&expected, &p); cio_set(&recovered, &p);
    const auto reference = murphy_refine(expected, config);
    for (int mode = 0; mode < 3; ++mode) {
        FailingEvaluator evaluator;
        evaluator.proposal = mode != 0;
        evaluator.fatal = mode == 2;
        cio_set(&recovered, &p);
        const auto fallback = murphy_refine(recovered, config, &evaluator);
        if (evaluator.proposal)
            bad += evaluator.validations != 2 || !evaluator.translated_failure;
        if (evaluator.fatal) {
            bad += fallback.valid || fallback.fallback || !evaluator.stopped;
        } else {
            bad += !reference.valid || !fallback.valid || !fallback.fallback || fallback.gpu ||
                   fallback.score != reference.score || fallback.skew != reference.skew ||
                   fallback.translation != reference.translation || evaluator.calls < 4;
        }
        const cio_poly &want = evaluator.fatal ? p : expected;
        bad += mpz_cmp(want.n, recovered.n) != 0 || want.deg != recovered.deg;
        for (int i = 0; i <= p.deg; ++i) bad += mpz_cmp(want.f[i], recovered.f[i]) != 0;
        for (int i = 0; i < 2; ++i) bad += mpz_cmp(want.g[i], recovered.g[i]) != 0;
    }
    struct Oracle : MurphyEvaluator {
        bool audit = false, stopped = false, fail_integral = false;
        int calls = 0;
        bool cancelled() const override { return stopped; }
        bool check_alpha(double supplied, double canonical) override {
            stopped = audit && fabs(supplied - canonical) > 1e-10;
            return !stopped;
        }
        bool evaluate(const MurphyIntegralInput *input, size_t n, double *output) override {
            ++calls;
            if (fail_integral && calls == 4) { stopped = true; return false; }
            for (size_t i = 0; i < n; ++i) output[i] = murphy_integral_reference(input[i]);
            return true;
        }
    } oracle;
    // A stale supplied alpha must never change unchecked device ranking. The
    // audit must nevertheless catch it before any device integral is evaluated.
    const double wrong_alpha = 123;
    cio_set(&recovered, &p);
    const auto canonical = murphy_refine(recovered, config, &oracle, &wrong_alpha);
    bad += !canonical.valid || !canonical.gpu || canonical.score != reference.score ||
           canonical.translation != reference.translation || canonical.skew != reference.skew;
    oracle.audit = true; oracle.calls = 0;
    cio_set(&recovered, &p);
    const auto rejected_alpha = murphy_refine(recovered, config, &oracle, &wrong_alpha);
    bad += rejected_alpha.valid || rejected_alpha.fallback || !oracle.stopped || oracle.calls;
    for (int i = 0; i <= p.deg; ++i) bad += mpz_cmp(p.f[i], recovered.f[i]) != 0;
    oracle.stopped = false; oracle.audit = false; oracle.fail_integral = true;
    cio_set(&recovered, &p);
    const auto rejected_integral = murphy_refine(recovered, config, &oracle);
    bad += rejected_integral.valid || rejected_integral.fallback || oracle.calls != 4 || !oracle.stopped;
    for (int i = 0; i <= p.deg; ++i) bad += mpz_cmp(p.f[i], recovered.f[i]) != 0;
    for (int i = 0; i < 2; ++i) bad += mpz_cmp(p.g[i], recovered.g[i]) != 0;
    const auto already_cancelled = murphy_refine(recovered, config, &oracle);
    bad += already_cancelled.valid || already_cancelled.fallback || oracle.calls != 4;
    cio_clear(&expected); cio_clear(&recovered);
    // The seed fits Int<16>, but translating its top coefficient does not.
    mpz_set_ui(p.f[5], 1);
    mpz_mul_2exp(p.f[5], p.f[5], 500);
    if (!s23::rs_size_poly(size, p)) return 2;
    MurphyCell overflow;
    overflow.refine(p, size, 0, 0, LONG_MAX, 1, 1, config);
    bad += overflow.attempted || overflow.failed();
    cio_clear(&p);
    printf("MurphyE invalid-cell/fallback/cancellation/alpha checks: 11 cases, %d failures\n", bad);
    return bad ? 1 : 0;
}
