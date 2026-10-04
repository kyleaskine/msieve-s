#!/usr/bin/env python3
"""Derived seeds for the rotations whose polynomial has content (s23_ropt searches them).

  content_seeds.py SEED.poly [SEED2.poly ...] [--out DIR] [--n N]

Each file is CADO format (n:, Y0:, Y1:, c0: ...); only the first polynomial is read.

CADO sopt returns a*f + r*g with Res(f, g) = a*N (GPU_STAGE23_PLAN.md, "The LLL lattice
contains f itself"). For a divisor d > 1 of a such that f + (u0 x + v0) g = 0 (mod d) for
some (u0, v0) (solved per prime power of d, with whichever of Y1, Y0 is a unit there,
then CRT), every rotation f + (u x + v) g with (u, v) = (u0, v0) (mod d) has all
coefficients divisible by d, and CADO's ropt divides that content out: a valid polynomial with multiplier a/d,
lognorm lower by about log d and a different alpha at d. s23_ropt's proxy is wrong for
such cells (it scores f + (u x + v) g, content included), so they are searched as seeds
of their own: f' = (f + (u0 x + v0) g)/d, with (u0, v0) the centred representatives.
The rotations f' + (u' x + v') g are exactly that lattice divided by d, so an ordinary
search on f' covers it with the right lognorm and alpha.

On the c168 job, 31 such seeds came from 30 of the 1000 seeds; searched like any other
seed they recovered CADO's content cells (seed 75) or beat them (seed 68, +5.5%).

Writes DIR/<stem>_d<d>.poly for each seed and each such d (DIR: the seed's directory),
and prints one line per derived seed.
"""

import argparse
import os
import sys

from polyfmt import cado_poly_text, parse_cado_poly, resultant_multiple


def divisors(a):
    """divisors d > 1 of |a|, ascending"""
    a, small, large = abs(a), [], []
    d = 1
    while d * d <= a:
        if a % d == 0:
            small.append(d)
            if d * d != a:
                large.append(a // d)
        d += 1
    return [d for d in small + large[::-1] if d > 1]


def factor(n):
    """{p: e} for |n| > 0"""
    n, out, d = abs(n), {}, 2
    while d * d <= n:
        while n % d == 0:
            out[d] = out.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def solve_prime_power(c, g0, g1, q, p):
    """(u, v) mod q = p^e with f + (u x + v) g = 0 (mod q), or None. c5, c4, c3 must
    vanish; g is primitive, so g1 or g0 is a unit mod p, and either fixes u and v: with
    g1 a unit, u from c2 and v from c1; with only g0 a unit, v from c0 and u from c1."""
    if any(c[k] % q for k in (3, 4, 5)):
        return None
    if g1 % p:
        inv = pow(g1, -1, q)
        u = -c[2] * inv % q
        v = -(c[1] + u * g0) * inv % q
    elif g0 % p:
        inv = pow(g0, -1, q)
        v = -c[0] * inv % q
        u = -(c[1] + v * g1) * inv % q
    else:
        return None
    if (c[2] + u * g1) % q or (c[1] + u * g0 + v * g1) % q or (c[0] + v * g0) % q:
        return None
    return u, v


def crt(residues):
    """x mod prod(m) from [(r, m)] with pairwise coprime m"""
    x, m = 0, 1
    for r, mi in residues:
        t = (r - x) * pow(m, -1, mi) % mi
        x, m = x + m * t, m * mi
    return x % m


def centred(x, d):
    x %= d
    return x - d if x > d // 2 else x


def content_seeds(p, n):
    """[(d, u0, v0, derived poly)] for every d > 1 dividing the multiplier with
    f + (u0 x + v0) g = 0 (mod d)"""
    c = [p['c%d' % k] for k in range(6)]
    g0, g1 = p['Y0'], p['Y1']
    a = resultant_multiple(p, n)
    if a is None:
        raise ValueError('N does not divide Res(f, g)')
    out = []
    for d in divisors(a):
        sols = []
        for p_, e in factor(d).items():
            sol = solve_prime_power(c, g0, g1, p_ ** e, p_)
            if sol is None:
                break
            sols.append((sol, p_ ** e))
        else:
            u0 = centred(crt([(uv[0], q) for uv, q in sols]), d)
            v0 = centred(crt([(uv[1], q) for uv, q in sols]), d)
            f = list(c)
            f[0] += v0 * g0
            f[1] += v0 * g1 + u0 * g0
            f[2] += u0 * g1
            assert all(x % d == 0 for x in f)
            dp = {'n': n, 'Y0': g0, 'Y1': g1, **{'c%d' % k: x // d for k, x in enumerate(f)}}
            assert resultant_multiple(dp, n) * d == a
            out.append((d, u0, v0, dp))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('seeds', nargs='+')
    ap.add_argument('--out', help='directory for the derived seeds (default: each seed\'s own)')
    ap.add_argument('--n', type=int, help='N, if the files have no n: line')
    args = ap.parse_args()
    total = with_content = 0
    for path in args.seeds:
        p = parse_cado_poly(path)
        n = p.get('n', args.n)
        if n is None:
            sys.exit(f'{path}: no n: line and no --n')
        derived = content_seeds(p, n)
        if derived:
            with_content += 1
        stem = os.path.splitext(os.path.basename(path))[0]
        out_dir = args.out or os.path.dirname(os.path.abspath(path))
        for d, u0, v0, dp in derived:
            name = os.path.join(out_dir, f'{stem}_d{d}.poly')
            with open(name, 'w') as fh:
                fh.write(cado_poly_text(dp, n) + '\n')
            print(f'{path}: d = {d}, lattice u = {u0}, v = {v0} (mod {d}), multiplier a/{d}: {name}')
            total += 1
    print(f'{total} derived seeds from {with_content} of {len(args.seeds)} seeds')


if __name__ == '__main__':
    main()
