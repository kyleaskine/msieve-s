#!/usr/bin/env python3
"""Exact relation between root-optimized polynomials of the same seed.

  rotation_diff.py A.poly X.poly [X2.poly ...] [--n N]

Each file is CADO format (n:, Y0:, Y1:, c0: ...); only the first polynomial is read.
For every X with the same Y1 as A, prints t and r(y) such that
    f_X(y) = f_A(y + t) + r(y) * g_X(y),   g_X(y) = g_A(y + t)
(an inverted-pass X is negated first). Also prints Res(f, g)/N for each poly, which
shows the sopt multiplier (CADO sopt returns a*f + r*g, so Res = a*N).
See Appendix A of GPU_STAGE23_PLAN.md.
"""

import argparse
import sys

from polyfmt import parse_cado_poly, rotation_between, resultant_multiple


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a')
    ap.add_argument('x', nargs='+')
    ap.add_argument('--n', type=int, help='N, if the files have no n: line')
    args = ap.parse_args()

    A = parse_cado_poly(args.a)
    n = args.n or A.get('n')
    for path, p in [(args.a, A)] + [(x, parse_cado_poly(x)) for x in args.x]:
        if n:
            k = resultant_multiple(p, n)
            print(f"{path}: Res/N = {k if k is not None else 'NOT A MULTIPLE OF N'}")
    for path in args.x:
        X = parse_cado_poly(path)
        r = rotation_between(A, X)
        if r is None:
            print(f"{path}: not a translation + rotation of {args.a} "
                  "(different Y1, different multiplier, or not exact)", file=sys.stderr)
            continue
        terms = ' + '.join(f"{c}*y^{i}" if i else str(c) for i, c in enumerate(r['rotation']) if c) or '0'
        print(f"{path}: t = {r['t']}, r(y) = {terms}" + ('  (f negated: inverted pass)' if r['negated'] else ''))


if __name__ == '__main__':
    main()
