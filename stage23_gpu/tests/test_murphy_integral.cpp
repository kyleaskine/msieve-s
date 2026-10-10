/* Cached integral vs CADO's original scalar loop, bit for bit. Exercise fresh
 * coefficients after translation, varying orders/cache eviction, the uncached
 * large-order path and independent OpenMP workers. No timing assertion.
 */
#include "cado.h" // IWYU pragma: keep
#include <cstdio>
#include <cmath>
#include <vector>
#include "murphy_search.h"
#include "murphyE.h"
#include "polyselect_alpha.h"
#include "cado_io.h"

using namespace s23_murphy;

int main(int argc, char **argv)
{
    if (argc < 2) return 2;
    std::vector<cxx_cado_poly> seeds;
    for (int i = 1; i < argc; i++) {
        FILE *in = fopen(argv[i], "r");
        if (!in) return 2;
        cio_poly q;
        cio_init(&q);
        int status;
        const size_t first = seeds.size();
        while ((status = cio_read_next(in, &q)) == 1) {
            cxx_cado_poly p;
            cio_to_cado(p, &q);
            seeds.push_back(p);
        }
        cio_clear(&q);
        fclose(in);
        if (status < 0 || seeds.size() == first) return 2;
        printf("%s: %zu polynomial blocks\n", argv[i], seeds.size() - first);
    }
    int bad = 0, checked = 0;
#pragma omp parallel for num_threads(4) reduction(+:bad,checked) schedule(dynamic,1)
    for (size_t j = 0; j < seeds.size() * 3; j++) {
        cxx_cado_poly p(seeds[j % seeds.size()]);
        // Force both generic degrees and the production degree-5 case. These
        // synthetic variants test evaluation only, not polynomial selection.
        const int mode = j / seeds.size();
        if (mode == 1)
            mpz_poly_setcoeff_si(p->pols[ALG_SIDE], 6, 1);
        if (mode == 2)
            mpz_poly_setcoeff_si(p->pols[ALG_SIDE], 0, 0);
        const double af = get_alpha(p->pols[ALG_SIDE], get_alpha_bound());
        const double ag = get_alpha(p->pols[RAT_SIDE], get_alpha_bound());
        Ctx c{17179869184.0, 8589934592.0, 3.4359738368e17, 0};
        // Alternate K/Keval, then evict both cached grids; >262144 streams.
        for (int k : {1000, 16000, 64000, 16000, 64000, 4000, 256000, 262145, 16000}) {
            c.K = k;
            translate_pair(p, k % 2 ? -1234567 : 89123);
            for (double skew : {1e3, 3.3e6, 2e8}) {
                p->skew = skew;
                const double want = MurphyE(p, c.Bf, c.Bg, c.area, c.K, get_alpha_bound());
                const double got = murphy_at(p, skew, af, ag, c);
                checked++;
                if (!(want == got && std::isfinite(got))) {
                    bad++;
#pragma omp critical
                    fprintf(stderr, "case %zu K %d skew %g: CADO %a cached %a\n", j, k, skew, want, got);
                }
            }
        }
        // Also exercise the initial-skew caller on the same shared implementation.
        double skew;
        const double got = best_skew(p, af, ag, c, skew);
        p->skew = skew;
        checked++;
        bad += got != MurphyE(p, c.Bf, c.Bg, c.area, c.K, get_alpha_bound());
    }
    printf("cached MurphyE: %zu polynomials, %d scalar CADO comparisons on 4 workers, %d differ\n",
           seeds.size(), checked, bad);
    return bad ? 1 : 0;
}
