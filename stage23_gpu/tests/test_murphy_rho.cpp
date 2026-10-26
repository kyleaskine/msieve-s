// Compare every approximation interval and its boundaries to linked CADO.
#include <cmath>
#include <cstdio>
#include "murphy_rho.h"
extern "C" double dickman_rho(double);

int main()
{
    int checked = 0, bad = 0;
    auto check = [&](double x) {
        const double cpu = dickman_rho(x), port = s23::murphy_rho(x);
        ++checked;
        if (cpu != port) {
            if (bad++ < 4) fprintf(stderr, "rho(%a): CADO %a port %a\n", x, cpu, port);
        }
    };
    for (int i = -8192; i <= 20 * 8192; ++i) check(double(i) / 8192);
    for (int i = 0; i <= 16; ++i) {
        check(std::nextafter(double(i), -INFINITY));
        check(std::nextafter(double(i), INFINITY));
    }
    printf("Dickman rho: %d CADO comparisons, %d differ\n", checked, bad);
    return bad ? 1 : 0;
}
