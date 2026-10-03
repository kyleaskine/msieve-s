#!/usr/bin/env python3
"""Write LLL test cases: one line per (polynomial, translation k):

    d c0 ... cd Y0 Y1 k

from the c146 fixture (each raw poly with k = 0 and with k = the translation of CADO's
effort-0 result) and the c204 benchmark winners (k = 0, +-1e6), so the cases cover
realistic coefficient sizes for both jobs.

  gen_cases.py OUT [--limit N]
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'stage23_bench', 'tools'))
from polyfmt import load_fixture, parse_cado_poly, coeffs  # noqa: E402

DATA = os.path.join(HERE, '..', '..', 'stage23_bench', 'data')


def line(c, y0, y1, k):
    return ' '.join(str(x) for x in [len(c) - 1] + list(c) + [y0, y1, k])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--limit', type=int, default=0, help='use only the first N fixture rows (0 = all)')
    args = ap.parse_args()
    rows = load_fixture(os.path.join(DATA, 'c146', 'sopt_sample.tsv.gz'))
    if args.limit:
        rows = rows[:args.limit]
    out = []
    for raw, opt, _, _ in rows:
        c = coeffs(raw)
        out.append(line(c, raw['Y0'], raw['Y1'], 0))
        t, rem = divmod(opt['Y0'] - raw['Y0'], raw['Y1'])
        if rem == 0 and t != 0:
            out.append(line(c, raw['Y0'], raw['Y1'], t))
    for name in ('A_msieve', 'B_cado_orig', 'C_cado_inv'):
        p = parse_cado_poly(os.path.join(DATA, 'c204', name + '.poly'))
        for k in (0, 1000000, -1000000):
            out.append(line(coeffs(p), p['Y0'], p['Y1'], k))
    with open(args.out, 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    print(f"wrote {len(out)} cases to {args.out}")


if __name__ == '__main__':
    main()
