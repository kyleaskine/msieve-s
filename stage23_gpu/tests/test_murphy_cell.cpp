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
    // The seed fits Int<16>, but translating its top coefficient does not.
    mpz_set_ui(p.f[5], 1);
    mpz_mul_2exp(p.f[5], p.f[5], 500);
    if (!s23::rs_size_poly(size, p)) return 2;
    MurphyCell overflow;
    overflow.refine(p, size, 0, 0, LONG_MAX, 1, 1, config);
    bad += overflow.attempted || overflow.failed();
    cio_clear(&p);
    printf("MurphyE invalid-cell checks: 4 cases, %d failures\n", bad);
    return bad ? 1 : 0;
}
