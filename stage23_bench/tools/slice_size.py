#!/usr/bin/env python3
"""Size cost of a fixed quadratic rotation: the best L2 lognorm of f + (w x^2 + u x + v) g
for each w, over integer u, v, t (translation) and real skew.

  slice_size.py SEED.poly [SEED2.poly ...] [--w -2,-1,0,1,2]

Each file is CADO format; only the first polynomial is read. Prints, per seed and w, the
minimum found and its excess over w = 0 (the cost of holding w fixed), with the
translation and skew where it was found.

Method: at fixed translation t and skew s the squared L2 norm is an exact quadratic form
in (u, v), so the best integer (u, v) is a 2-D closest-vector problem (continuous minimum,
then the nearest points of the Gauss-reduced lattice). Skew is a grid then pattern
search; t is a grid out to |t| = 1e11 (6 points per decade) then pattern search. The
lognorm is the moment form of CADO's L2 lognorm; it reproduces s23_ropt's best line
minimum to 1e-3 on c168 seeds. The minimum is the best point found, so the cost it
reports is an upper bound: a basin between grid points could be missed.

On c168 (GPU_STAGE23_PLAN.md, "Third review"), w = +-1 costs 5.8 to 8.3 nats on 8 seeds.
"""

import argparse
import math
import sys
from math import gamma

from polyfmt import parse_cado_poly, translate

D = 5
# int over the unit disc of X^p Y^q, p + q = 2D, up to a constant (zero unless p, q even)
M = [[gamma((i + j + 1) / 2) * gamma((2 * D - i - j + 1) / 2) / gamma(D + 2) if (i + j) % 2 == 0 else 0.0
      for j in range(D + 1)] for i in range(D + 1)]


def qf(x, y):
    return sum(x[i] * M[i][j] * y[j] for i in range(D + 1) for j in range(D + 1) if M[i][j])


def best_uv(ft, gt, ls):
    """(lognorm, u, v): the best integer rotation ft + (u x + v) gt at skew e^ls"""
    s = math.exp(ls)
    W = [s ** (i - D / 2) for i in range(D + 1)]
    h = [float(ft[i]) * W[i] for i in range(D + 1)]
    a = [0.0] * (D + 1)  # x g
    a[1], a[2] = float(gt[0]) * W[1], float(gt[1]) * W[2]
    b = [0.0] * (D + 1)  # g
    b[0], b[1] = float(gt[0]) * W[0], float(gt[1]) * W[1]
    A11, A12, A22 = qf(a, a), qf(a, b), qf(b, b)
    r1, r2 = qf(a, h), qf(b, h)
    det = A11 * A22 - A12 * A12
    uc, vc = (-r1 * A22 + r2 * A12) / det, (-r2 * A11 + r1 * A12) / det

    def gram(x, y):
        return A11 * x[0] * y[0] + A12 * (x[0] * y[1] + x[1] * y[0]) + A22 * x[1] * y[1]

    B = [[1, 0], [0, 1]]  # Gauss reduction of Z^2 under A
    for _ in range(200):
        if gram(B[0], B[0]) > gram(B[1], B[1]):
            B[0], B[1] = B[1], B[0]
        mu = round(gram(B[0], B[1]) / gram(B[0], B[0]))
        if mu == 0:
            break
        B[1] = [B[1][0] - mu * B[0][0], B[1][1] - mu * B[0][1]]
    det_b = B[0][0] * B[1][1] - B[0][1] * B[1][0]
    x0 = (uc * B[1][1] - vc * B[1][0]) / det_b
    y0 = (vc * B[0][0] - uc * B[0][1]) / det_b
    best = (math.inf, 0, 0)
    for i in range(math.floor(x0) - 1, math.floor(x0) + 3):
        for j in range(math.floor(y0) - 1, math.floor(y0) + 3):
            u, v = i * B[0][0] + j * B[1][0], i * B[0][1] + j * B[1][1]
            c = list(ft)  # exact coefficients, then weights
            c[0] += v * gt[0]
            c[1] += v * gt[1] + u * gt[0]
            c[2] += u * gt[1]
            vec = [float(c[k]) * W[k] for k in range(D + 1)]
            L = 0.5 * math.log(qf(vec, vec))
            if L < best[0]:
                best = (L, u, v)
    return best


def best_at_t(f, g, t):
    """(lognorm, log skew, u, v) at translation t"""
    ft, gt = translate(f, t), translate(g, t)
    lo, hi, n = math.log(1e3), math.log(1e10), 140
    grid = [lo + k * (hi - lo) / n for k in range(n + 1)]
    L, ls = min((best_uv(ft, gt, x)[0], x) for x in grid)
    h = (hi - lo) / n
    while h > 1e-4:
        moved = False
        for c in (ls + h, ls - h):
            Lc = best_uv(ft, gt, c)[0]
            if Lc < L:
                L, ls, moved = Lc, c, True
        if not moved:
            h /= 2
    _, u, v = best_uv(ft, gt, ls)
    return L, ls, u, v


def slice_min(f, g):
    """(lognorm, t, skew, u, v): the best point found over the slice"""
    grid = [0] + [sg * int(10 ** (e / 6)) for e in range(67) for sg in (1, -1)]
    L, t = min((best_at_t(f, g, t)[0], t) for t in grid)
    step = max(1, abs(t) // 4)
    while step >= 1:
        moved = False
        for c in (t + step, t - step):
            Lc = best_at_t(f, g, c)[0]
            if Lc < L:
                L, t, moved = Lc, c, True
        if not moved:
            step //= 2
    L, ls, u, v = best_at_t(f, g, t)
    return L, t, math.exp(ls), u, v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('seeds', nargs='+')
    ap.add_argument('--w', default='-2,-1,0,1,2', help='comma-separated w values (0 is always added)')
    args = ap.parse_args()
    ws = sorted(set(int(x) for x in args.w.split(',')) | {0})
    for path in args.seeds:
        p = parse_cado_poly(path)
        if any(p.get('c%d' % k) is None for k in range(D + 1)) or p.get('c6'):
            sys.exit(f'{path}: degree 5 only')
        g = [p['Y0'], p['Y1']]
        rows = {}
        for w in ws:
            f = [p['c%d' % k] for k in range(D + 1)]
            f[3] += w * p['Y1']
            f[2] += w * p['Y0']
            rows[w] = slice_min(f, g)
        for w in ws:
            L, t, s, u, v = rows[w]
            print(f'{path}\tw={w:+d}\tlognorm {L:.3f}\tcost {L - rows[0][0]:+.3f}\tt {t}\tskew {s:.3g}\tu {u}\tv {v}',
                  flush=True)


if __name__ == '__main__':
    main()
