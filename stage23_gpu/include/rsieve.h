/* Root sieve (M2): the alpha of every rotation f_{u,v} = f + (u x + v) g of one seed
 * (degree 5, linear rotation, as CADO's ropt for degree 5), for primes p < RS_BOUND (200,
 * as CADO's ropt sieve).
 *
 * alpha = sum_p (1/(p-1) - E_p) log p, E_p = sum_e N_e / (p^e + p^(e-1)), where N_e
 * counts the roots mod p^e, affine and projective (stage23_bench/tools/alpha_proto.py).
 * Affine roots count to levels p^e <= RS_BOUND. Projective roots (the roots y = 0 mod p
 * of rev f(y) = c5 + c4 y + ... + c0 y^5, present when p | c5) count to levels
 * e <= emax + 3 (rs_proj_value); the tail goes into the exact alpha of the survivors.
 *
 * The projective part is not constant under rotation, although c5, c4, c3 are: with
 * y = p t, c2, c1, c0 enter rev f at p^3, p^4, p^5, so deep lifts depend on them (up to
 * 0.8 nats on c168 seeds, also within a u-line; GPU_STAGE23_PLAN.md, "Review checks").
 * The same powers keep the table periodic: a level e <= emax + 3 needs c2 mod p^emax,
 * c1 mod p^(emax-1) and c0 mod p^(emax-2), so it depends on u, v only mod q_max (and on
 * v only mod p^(emax-1)). emax + 3 captures about 98% of the variation on c168 seeds.
 * So the projective values are tabulated once per seed on the host (rs_proj_tables: per
 * prime, q_max rows of u mod q_max by p^(emax-1) columns of v), and a line's table only
 * adds its row.
 *
 * For a fixed u, h = f + u x g, and f_{u,v}(x) = h(x) + v g(x). One pass over x mod p^e
 * gives N_e(v) for every v mod p^e:
 *   p does not divide g(x): x is a root for the one class v = -h(x)/g(x) mod p^e;
 *   k = v_p(g(x)) >= 1 (x at g's root mod p): if e <= k, x is a root for every v when
 *   p^e | h(x), for none otherwise; else x is a root iff p^k | h(x) and
 *   v = -(h(x)/p^k) (g(x)/p^k)^(-1) mod p^(e-k).
 * (Checked against brute force in alpha_proto.py sieve-check, and in tests/test_rsieve.)
 *
 * Per prime, the levels are folded into one table of period q_max = p^e_max:
 *   T_p[v mod q_max] = sum_e N_e(v mod p^e) log p / (p^e + p^(e-1)),
 * so a cell's sieve score is S(v) = sum_p T_p[v mod q_max(p)] (46 lookups), and
 * alpha_affine(v) ~ sum_p log p / (p-1) - S(v): a higher score is a better alpha.
 */
#pragma once

#include <math.h>
#include <vector>
#include "hd.h"

namespace s23 {

enum { RS_BOUND = 200, RS_NPRIMES = 46, RS_DEG = 5 };

enum { RS_PROJ_EXTRA = 3, RS_WMAX = 7 + RS_PROJ_EXTRA }; /* emax <= 7 (2^7 = 128) */
static const uint32_t RS_NOPROJ = 0xffffffffu;

struct RsPrime {
    uint32_t p, emax, qmax; /* qmax = p^emax <= RS_BOUND < p^(emax+1) */
    uint32_t qproj;         /* p^(emax + RS_PROJ_EXTRA) < 2^32: the projective levels' modulus */
    uint32_t off;           /* this prime's table in the line's concatenated tables */
};

/* one seed, reduced: f and g mod q_max of each prime, and mod qproj for the projective
 * levels */
struct RsSeed {
    RsPrime pr[RS_NPRIMES];
    uint32_t f[RS_NPRIMES][RS_DEG + 1];
    uint32_t g[RS_NPRIMES][2];
    uint32_t fp[RS_NPRIMES][RS_DEG + 1];
    uint32_t gp[RS_NPRIMES][2];
    float w[RS_NPRIMES][RS_WMAX]; /* level weights log p / (p^e + p^(e-1)), e = 1..emax + 3,
                                     computed once on the host so every build uses the same
                                     floats */
    uint32_t ntab;   /* total table entries (sum of q_max) */
    int proj;        /* 1: projective roots in the tables (default); 0: affine only */
    uint32_t poff[RS_NPRIMES]; /* the prime's projective values in rs_proj_tables' array
                                  (row u mod q_max, column v mod p^(emax-1)), or RS_NOPROJ */
    double alpha0;   /* sum_p log p / (p - 1) */
};

S23_HD uint32_t rs_inv(uint32_t a, uint32_t m) /* a^-1 mod m, gcd(a, m) = 1, m <= 2^16 */
{
    int32_t t = 0, nt = 1, r = (int32_t)m, nr = (int32_t)(a % m);
    while (nr) {
        int32_t q = r / nr, x;
        x = t - q * nt;
        t = nt;
        nt = x;
        x = r - q * nr;
        r = nr;
        nr = x;
    }
    return (uint32_t)(t < 0 ? t + (int32_t)m : t);
}

/* the primes below RS_BOUND, their q_max and table offsets (no residues yet) */
S23_HD void rs_init_primes(RsSeed &S)
{
    int n = 0;
    uint32_t off = 0;
    S.alpha0 = 0;
    S.proj = 1;
    for (int i = 0; i < RS_NPRIMES; i++)
        S.poff[i] = RS_NOPROJ;
    for (uint32_t p = 2; p < RS_BOUND && n < RS_NPRIMES; p++) {
        bool prime = true;
        for (uint32_t d = 2; d * d <= p; d++)
            if (p % d == 0) {
                prime = false;
                break;
            }
        if (!prime)
            continue;
        RsPrime &P = S.pr[n++];
        P.p = p;
        P.emax = 1;
        P.qmax = p;
        while (P.qmax * p <= RS_BOUND) {
            P.qmax *= p;
            P.emax++;
        }
        P.qproj = P.qmax * p * p * p;
        P.off = off;
        off += P.qmax;
        S.alpha0 += log((double)p) / (double)(p - 1);
        double q = p;
        for (uint32_t e = 1; e <= P.emax + RS_PROJ_EXTRA; e++, q *= p)
            S.w[n - 1][e - 1] = (float)(log((double)p) / (q + q / p));
    }
    S.ntab = off;
}

/* The projective roots' weighted count for prime i at f + (u x + v) g, levels 1 ..
 * emax + 3, by lifting the roots y = 0 mod p of rev f (depth-first, in a fixed order, so
 * every build adds the same floats in the same order). Needs p | c5. Only u, v mod q_max
 * matter (see the top of this file), and the coefficients are taken mod qproj. weights:
 * level e adds w[e - 1]. */
template <class W>
S23_HD W rs_proj_value(const RsSeed &S, int i, uint64_t u, uint64_t v, const W *w)
{
    const RsPrime &P = S.pr[i];
    const uint64_t m = P.qproj, p = P.p;
    const int kmax = (int)P.emax + RS_PROJ_EXTRA;
    uint64_t c[RS_DEG + 1]; /* f + (u x + v) g mod qproj: c0 += v g0, c1 += u g0 + v g1, c2 += u g1 */
    for (int k = 0; k <= RS_DEG; k++)
        c[k] = S.fp[i][k];
    const uint64_t g0 = S.gp[i][0], g1 = S.gp[i][1];
    u %= m;
    v %= m;
    c[0] = (c[0] + v * g0 % m) % m;
    c[1] = (c[1] + u * g0 % m + v * g1 % m) % m;
    c[2] = (c[2] + u * g1 % m) % m;
    uint64_t pw[RS_WMAX + 2], ys[RS_WMAX + 1];
    uint32_t js[RS_WMAX + 1];
    pw[0] = 1;
    for (int k = 1; k <= kmax + 1 && k <= RS_WMAX + 1; k++)
        pw[k] = pw[k - 1] * p;
    W s = w[0]; /* y = 0 is a root mod p (p | c5) */
    int k = 1;  /* ys[k]: a root mod p^k; js[k]: the next digit to try for level k + 1 */
    ys[1] = 0;
    js[1] = 0;
    while (k >= 1) {
        if (k >= kmax || js[k] >= p) {
            k--;
            continue;
        }
        const uint64_t y = ys[k] + js[k]++ * pw[k], mk = pw[k + 1];
        uint64_t r = c[0] % mk; /* rev f(y) = sum c_{5-j} y^j, Horner from c0 */
        for (int j = 1; j <= RS_DEG; j++)
            r = (r * y + c[j]) % mk;
        if (r == 0) {
            s += w[k];
            k++;
            ys[k] = y;
            js[k] = 0;
        }
    }
    return s;
}

/* The projective values of every line and v class, for each prime dividing c5, into PT
 * (host; the GPU gets a copy); sets S.poff. Nothing when S.proj is 0. */
inline void rs_proj_tables(RsSeed &S, std::vector<float> &PT)
{
    PT.clear();
    for (int i = 0; i < RS_NPRIMES; i++) {
        const RsPrime &P = S.pr[i];
        S.poff[i] = RS_NOPROJ;
        if (!S.proj || S.fp[i][RS_DEG] % P.p != 0)
            continue;
        const uint32_t vper = P.qmax / P.p;
        S.poff[i] = (uint32_t)PT.size();
        for (uint32_t uq = 0; uq < P.qmax; uq++)
            for (uint32_t v0 = 0; v0 < vper; v0++)
                PT.push_back(rs_proj_value<float>(S, i, uq, v0, S.w[i]));
    }
}

/* T[0 .. q_max-1] for prime i on line u (uq = u mod q_max, in [0, q_max)); PT from
 * rs_proj_tables */
S23_HD void rs_prime_table(float *T, const RsSeed &S, const float *PT, int i, uint32_t uq)
{
    const RsPrime &P = S.pr[i];
    const uint32_t p = P.p, qmax = P.qmax;
    for (uint32_t v = 0; v < qmax; v++)
        T[v] = 0.0f;
    uint32_t q = 1;
    for (uint32_t e = 1; e <= P.emax; e++) {
        q *= p;
        const float w = S.w[i][e - 1];
        uint32_t fq[RS_DEG + 1];
        for (int k = 0; k <= RS_DEG; k++)
            fq[k] = S.f[i][k] % q;
        const uint32_t g0 = S.g[i][0] % q, g1 = S.g[i][1] % q, u = uq % q;
        for (uint32_t x = 0; x < q; x++) {
            const uint32_t gx = (g0 + g1 * x) % q;
            uint32_t fx = fq[RS_DEG];
            for (int k = RS_DEG - 1; k >= 0; k--)
                fx = (fx * x + fq[k]) % q;
            const uint32_t hx = (fx + (u * x % q) * gx) % q;
            /* k = v_p(gx), capped at e */
            uint32_t kk = 0, pk = 1;
            while (kk < e && gx % (pk * p) == 0) {
                pk *= p;
                kk++;
            }
            uint32_t v0, step; /* the hit classes: v = v0 mod step */
            if (kk == 0) {
                v0 = (q - hx) % q * rs_inv(gx, q) % q;
                step = q;
            } else if (kk >= e) {
                if (hx != 0)
                    continue;
                v0 = 0;
                step = 1;
            } else {
                if (hx % pk != 0)
                    continue;
                const uint32_t mk = q / pk;
                v0 = (mk - (hx / pk) % mk) % mk * rs_inv((gx / pk) % mk, mk) % mk;
                step = mk;
            }
            for (uint32_t v = v0; v < qmax; v += step)
                T[v] += w;
        }
    }
    if (S.poff[i] != RS_NOPROJ) { /* projective roots: period p^(emax-1) in v */
        const uint32_t vper = qmax / p;
        const float *row = PT + S.poff[i] + uq * vper;
        for (uint32_t v0 = 0; v0 < vper; v0++)
            for (uint32_t v = v0; v < qmax; v += vper)
                T[v] += row[v0];
    }
}

/* all tables of line u into T[0 .. S.ntab-1] */
S23_HD void rs_line_tables(float *T, const RsSeed &S, const float *PT, int64_t u)
{
    for (int i = 0; i < RS_NPRIMES; i++) {
        const int64_t q = S.pr[i].qmax;
        rs_prime_table(T + S.pr[i].off, S, PT, i, (uint32_t)(((u % q) + q) % q));
    }
}

/* a cell's score from its line's tables (reference; the kernels step indices instead) */
S23_HD float rs_cell_score(const float *T, const RsSeed &S, int64_t v)
{
    float s = 0;
    for (int i = 0; i < RS_NPRIMES; i++) {
        const int64_t q = S.pr[i].qmax;
        s += T[S.pr[i].off + (uint32_t)(((v % q) + q) % q)];
    }
    return s;
}

/* Brute force for one cell (host only): affine N_e by evaluating f_{u,v} at every x mod
 * p^e (no special rule), weighted and summed in double; projective N_e from the full u
 * and v (no periodicity assumed), counting every y = p t mod p^e with rev f_{u,v}(y) = 0
 * while there are at most 2^10 of them, then by lifting the roots found (exact: a root
 * mod p^e reduces to one mod p^(e-1)). The independent check of the tables. */
inline double rs_cell_proj_brute(const RsSeed &S, int i, int64_t u, int64_t v)
{
    const RsPrime &P = S.pr[i];
    const uint64_t m = P.qproj, p = P.p;
    const uint64_t um = (uint64_t)(((u % (int64_t)m) + (int64_t)m) % (int64_t)m);
    const uint64_t vm = (uint64_t)(((v % (int64_t)m) + (int64_t)m) % (int64_t)m);
    uint64_t c[RS_DEG + 1];
    for (int k = 0; k <= RS_DEG; k++)
        c[k] = S.fp[i][k];
    c[0] = (c[0] + vm * S.gp[i][0]) % m;
    c[1] = (c[1] + um * S.gp[i][0] % m + vm * S.gp[i][1]) % m;
    c[2] = (c[2] + um * S.gp[i][1]) % m;
    auto is_root = [&](uint64_t y, uint64_t mk) {
        uint64_t r = 0;
        for (int j = 0; j <= RS_DEG; j++)
            r = (r * y + c[j] % mk) % mk;
        return r == 0;
    };
    std::vector<uint64_t> roots, next;
    double s = 0;
    uint64_t mk = 1;
    for (uint32_t e = 1; e <= P.emax + RS_PROJ_EXTRA; e++) {
        const uint64_t prev = mk;
        mk *= p;
        next.clear();
        if (mk / p <= 1024) {
            for (uint64_t t = 0; t < mk / p; t++)
                if (is_root(p * t, mk))
                    next.push_back(p * t);
        } else {
            for (uint64_t y : roots)
                for (uint64_t j = 0; j < p; j++)
                    if (is_root(y + j * prev, mk))
                        next.push_back(y + j * prev);
        }
        if (next.empty())
            break;
        s += (double)next.size() * log((double)p) / ((double)mk + (double)(mk / p));
        roots.swap(next);
    }
    return s;
}

inline double rs_cell_score_brute(const RsSeed &S, int64_t u, int64_t v)
{
    double s = 0;
    for (int i = 0; i < RS_NPRIMES; i++) {
        const uint32_t p = S.pr[i].p;
        if (S.poff[i] != RS_NOPROJ)
            s += rs_cell_proj_brute(S, i, u, v);
        uint32_t q = 1;
        for (uint32_t e = 1; e <= S.pr[i].emax; e++) {
            q *= p;
            const int64_t qq = q;
            const uint32_t uq = (uint32_t)(((u % qq) + qq) % qq), vq = (uint32_t)(((v % qq) + qq) % qq);
            uint32_t c[RS_DEG + 1];
            for (int k = 0; k <= RS_DEG; k++)
                c[k] = S.f[i][k] % q;
            const uint32_t g0 = S.g[i][0] % q, g1 = S.g[i][1] % q;
            /* f + (u x + v) g: c0 += v g0, c1 += v g1 + u g0, c2 += u g1 */
            c[0] = (c[0] + vq * g0) % q;
            c[1] = (c[1] + vq * g1 % q + uq * g0) % q;
            c[2] = (c[2] + uq * g1) % q;
            uint32_t n = 0;
            for (uint32_t x = 0; x < q; x++) {
                uint32_t fx = c[RS_DEG];
                for (int k = RS_DEG - 1; k >= 0; k--)
                    fx = (fx * x + c[k]) % q;
                n += fx == 0;
            }
            s += n * log((double)p) / (double)(q + q / p);
        }
    }
    return s;
}

} // namespace s23
