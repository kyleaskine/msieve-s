#include "host_log_check.h"
#include <cstdio>

int main()
{
    using namespace s23::matched_log;
    auto inputs = probe_inputs();
    auto host = identify_host(inputs);
    if (host.mode == unsupported) {
        fprintf(stderr, "unsupported host log; production would use CADO CPU fallback\n");
        return 77;
    }
    if (!check_device(inputs, host)) return 1;
    auto scalar = identify(inputs, [](double x) { return eval<false>(x); });
    auto fused_ref = identify(inputs, [](double x) { return eval<true>(x); });
    if (!check_device(inputs, scalar) || !check_device(inputs, fused_ref)) return 1;
    host.reference[0] = std::nextafter(host.reference[0], INFINITY);
    if (check_device(inputs, host)) return 1;
    printf("device log: both variants verified on %zu inputs; mismatch rejected\n", inputs.size());
}
