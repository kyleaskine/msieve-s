/* Host helpers shared by tools/s23_ropt.cu and tests/test_rsieve.cpp: the seed's size-model
 * polynomial, the polynomial s23_ropt writes for a cell, and the exact values -rerank ranks a
 * cell by. Host only (GMP, and CADO through tools/cado_io).
 */
#pragma once

#include <math.h>
#include <gmp.h>
#include "cado_io.h"
#include "gmp_bridge.h"
#include "rseed.h"
#include "rsize.h"

namespace s23 {

/* the size model's integer width in s23_ropt (512 bits) */
static const int RS_LP = 16;

/* the size model's f, g exactly; false unless every coefficient fits Int<L> */
template <int L> bool rs_size_poly(RsizePoly<L> &R, const cio_poly &P)
{
    bool ok = P.deg == RS_DEG;
    for (int i = 0; i <= RS_DEG && ok; i++)
        ok = from_mpz(R.f[i], P.f[i]);
    return ok && from_mpz(R.g[0], P.g[0]) && from_mpz(R.g[1], P.g[1]);
}

/* the root sieve's residues (rs_seed_from_poly) and the size model's polynomial; false unless
 * f has degree 5 and every coefficient fits Int<L> */
template <int L> bool rs_seed_and_size(RsSeed &S, RsizePoly<L> &R, const cio_poly &P)
{
    return rs_seed_from_poly(S, P) && rs_size_poly(R, P);
}

/* the polynomial written for cell (u, v) at translation t: f_{u,v}(x + t) and g(x + t), with
 * f's content divided out as CADO's ropt does (content gets it, 0 if it does not fit an
 * unsigned long; content seeds search those lattices properly). false on overflow. */
template <int L>
bool rs_written_poly(cio_poly &out, const cio_poly &seed, const RsizePoly<L> &R, int64_t u, int64_t v, int64_t t,
                     unsigned long &content)
{
    Int<L> fr[6], g[2], k;
    if (!rs_rotate(fr, R, u, v))
        return false;
    set_si(k, t);
    g[0] = R.g[0];
    g[1] = R.g[1];
    if (!translate<L, 5>(fr, k) || !translate<L, 1>(g, k))
        return false;
    mpz_set(out.n, seed.n);
    out.deg = 5;
    for (int i = 0; i <= 5; i++)
        to_mpz(out.f[i], fr[i]);
    to_mpz(out.g[0], g[0]);
    to_mpz(out.g[1], g[1]);
    mpz_t c;
    mpz_init_set(c, out.f[5]);
    for (int i = 0; i < 5; i++)
        mpz_gcd(c, c, out.f[i]);
    content = mpz_fits_ulong_p(c) ? mpz_get_ui(c) : 0;
    if (mpz_cmp_ui(c, 1) > 0)
        for (int i = 0; i <= 5; i++)
            mpz_divexact(out.f[i], out.f[i], c);
    mpz_clear(c);
    return true;
}

/* what -rerank ranks cell (u, v) by: the exact lognorm over integer translations (t: the
 * search's start in, the optimum out) and CADO's alpha to the bound its MurphyE uses, both of
 * the polynomial rs_written_poly writes, content divided out. With the content left in, the
 * lognorm is log d higher and alpha log d lower, which at aw > 1 ranks such a cell
 * (aw - 1) log d too well. */
struct RsExact {
    double L, A;
};
template <int L> RsExact rs_exact_cell(const cio_poly &seed, const RsizePoly<L> &R, int64_t u, int64_t v, int64_t &t)
{
    double log_content = 0;
    const double A = cio_alpha_rot(&seed, (long)u, (long)v, cio_alpha_bound(), &log_content);
    return {rs_size(R, u, v, t, 1 << 10) - log_content, A};
}

} // namespace s23
