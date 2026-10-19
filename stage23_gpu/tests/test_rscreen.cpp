/* Compare the GPU screen's shared arithmetic with the independent GMP/CADO
 * reference: negative rotations, content, projective roots and all alpha bounds. */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "rhost.h"
#include "rscreen.h"
using namespace s23;

int main(int argc, char **argv)
{
    size_t checked = 0, content_cells = 0;
    double worst = 0;
    for (int a = 1; a < argc; ++a) {
        FILE *in = fopen(argv[a], "r");
        if (!in) return 2;
        cio_poly p;
        cio_init(&p);
        if (cio_read_next(in, &p) != 1) return 2;
        fclose(in);
        RsizePoly<RS_LP> r;
        if (!rs_size_poly(r, p)) return 2;
        for (int64_t u : {int64_t(-987654321), int64_t(-1), int64_t(0), int64_t(1), int64_t(7)})
            for (int64_t v : {int64_t(-174864538), int64_t(-3), int64_t(0), int64_t(2), int64_t(293874)}) {
                Int<RS_LP> f[6];
                bool ok = rs_rotate(f, r, u, v);
                int64_t t = 0, reference_t = 0;
                double L = rs_screen_size(f, t, ok);
                double reference_L = rs_size(r, u, v, reference_t, 1 << 10);
                if (!ok || L != reference_L || t != reference_t) return 1;
                double lc = 0;
                if (!rs_screen_primitive(f, lc)) return 1;
                content_cells += lc != 0;
                double alpha = 0;
                for (unsigned q = 2; q <= RS_SCREEN_MAX_BOUND; ++q) {
                    if (rs_is_prime(q)) {
                        double val = rs_screen_valuation(f, q, ok);
                        alpha += (1.0 / (q - 1) - val) * log(double(q));
                    }
                    if (q != 2 && q != 3 && q != 31 && q != 200 && q != 500 && q != 2000) continue;
                    double ref_lc = 0;
                    double ref = cio_alpha_rot(&p, u, v, q, &ref_lc);
                    worst = fmax(worst, fabs(ref - alpha));
                    if (!ok || ref_lc != lc || fabs(ref - alpha) > 1e-11) {
                        fprintf(stderr, "%s (%lld,%lld) B=%u: %.17g vs %.17g; content %.17g vs %.17g\n",
                                argv[a], (long long)u, (long long)v, q, alpha, ref, lc, ref_lc);
                        return 1;
                    }
                    ++checked;
                }
            }
        cio_clear(&p);
    }
    Int<RS_LP> f[6];
    for (auto &x : f) set_zero(x);
    bool ok = true;
    set_si(f[5], 1); // x^5: infinite lift, must request fallback rather than hang
    rs_screen_valuation(f, 2, ok);
    if (ok) return 1;
    set_zero(f[5]);
    double lc = 0;
    if (rs_screen_primitive(f, lc)) return 1;
    set_si(f[0], 1); set_si(f[5], 1);
    int64_t t = INT64_MAX;
    ok = true;
    rs_screen_size(f, t, ok);
    if (ok) return 1;
    std::vector<uint32_t> primes;
    if (!rs_screen_primes(RS_SCREEN_MAX_BOUND, primes) || primes.size() != RS_SCREEN_MAX_PRIMES ||
        primes.front() != 2 || primes.back() != 1999 ||
        rs_screen_primes(RS_SCREEN_MAX_BOUND + 1, primes) || rs_screen_primes(1, primes)) return 1;
    // Local-minimum shifts are much larger than one ulp. Recheck both sides.
    if (!rs_screen_near_cut(50.001, 50) || !rs_screen_near_cut(49.9, 50) ||
        rs_screen_near_cut(50.5, 50) || rs_screen_near_cut(INFINITY, INFINITY)) return 1;
    if (rs_screen_key(NAN, 0, 1.3) != INFINITY || rs_screen_key(1, -INFINITY, 1.3) != INFINITY ||
        rs_screen_key(1, 2, 1.5) != 4) return 1;
    printf("screen arithmetic: %zu alpha checks, %zu content cells, worst error %.3g; fallback guards OK\n",
           checked, content_cells, worst);
    return checked && content_cells ? 0 : 1;
}
