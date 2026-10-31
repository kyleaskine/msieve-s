#include "host_log_check.h"
#include <cstdio>

// These checks must execute under -DNDEBUG, including rounding-mode changes.
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (false)

int main()
{
    using namespace s23::matched_log;
    auto inputs = probe_inputs();
    CHECK(inputs.size() == 2006680);
    auto fma = identify(inputs, [](double x) { return eval<true>(x); });
    auto scalar = identify(inputs, [](double x) { return eval<false>(x); });
    CHECK(fma.mode == fused && fma.fused_mismatches == 0 && fma.unfused_mismatches > 0);
    CHECK(scalar.mode == unfused && scalar.unfused_mismatches == 0 && scalar.fused_mismatches > 0);
    auto unknown = identify(probe_inputs(16), [](double x) { return std::nextafter(eval<true>(x), INFINITY); });
    CHECK(unknown.mode == unsupported);
    CHECK(identify(std::vector<double>{}, [](double x) { return x; }).mode == unsupported);
    const int original_rounding = std::fegetround();
    CHECK(original_rounding != -1);
    CHECK(std::fesetround(FE_DOWNWARD) == 0);
    CHECK(identify_host(inputs).mode == unsupported);
    CHECK(std::fesetround(FE_TONEAREST) == 0);
    for (double x : std::vector<double>{0., -0., INFINITY, -1., -INFINITY, NAN}) {
        double ref = ::log(x), a = eval<true>(x), b = eval<false>(x);
        if (std::isnan(ref)) CHECK(std::isnan(a) && std::isnan(b));
        else CHECK(bits(a) == bits(ref) && bits(b) == bits(ref));
    }
    auto host = identify_host(inputs);
    printf("host log: %s, %zu samples; FMA mismatches %zu, scalar mismatches %zu\n",
           name(host.mode), host.samples, host.fused_mismatches, host.unfused_mismatches);
    // Unknown libm is a supported fallback case, not a portable test failure.
    if (host.mode != unsupported) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            double result = host.mode == fused ? eval<true>(inputs[i]) : eval<false>(inputs[i]);
            CHECK(bits(result) == bits(host.reference[i]));
        }
    }
    CHECK(std::fesetround(original_rounding) == 0);
    puts("host log variant selection, rejection, boundaries and special values passed");
}
