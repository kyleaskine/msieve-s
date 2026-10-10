/* Exercise the same host ranking library ropt links, without a root-sieve run.
 * The Python test supplies compact real candidates and independently checks the
 * exact polynomial exports and high-precision MurphyE readback.
 */
#include <cstdio>
#include <cstdlib>
#include "murphy_rank.h"

int main(int argc, char **argv)
{
    if (argc != 5) return 2;
    MurphyConfig c;
    c.Bf = strtod(argv[2], nullptr);
    c.Bg = strtod(argv[3], nullptr);
    c.area = strtod(argv[4], nullptr);
    if (!murphy_config_valid(c)) return 2;
    FILE *in = fopen(argv[1], "r");
    if (!in) return 2;
    cio_poly p;
    cio_init(&p);
    int status, index = 0;
    while ((status = cio_read_next(in, &p)) == 1) {
        const auto r = murphy_refine(p, c);
        if (!r.valid) return 1;
        printf("# index %d score %.17g control %.17g proposal %.17g dt %ld accepted %d calls %d limited %d\n",
               index++, r.score, r.control, r.proposal, r.translation, r.accepted, r.calls, r.limited);
        gmp_printf("n: %Zd\nY0: %Zd\nY1: %Zd\nskew: %.17g\n", p.n, p.g[0], p.g[1], r.skew);
        for (int i = 0; i <= p.deg; i++) gmp_printf("c%d: %Zd\n", i, p.f[i]);
        printf("\n");
    }
    fclose(in);
    cio_clear(&p);
    return status < 0 || !index ? 1 : 0;
}
