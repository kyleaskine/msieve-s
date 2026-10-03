#!/usr/bin/env python3
"""Compare size-optimization results, joined on the raw polynomial.

  sopt_compare.py BASE [OTHER ...] [--top 150,300,1000] [--tol 0.005]

BASE is a full sopt output, e.g. cado_sopt_unsorted.txt (CADO, effort 0). Each OTHER
covers a subset of the same raw polys, e.g. pipeline_work/resopt_output.txt (effort 50)
or a GPU sopt's output; several OTHER files are merged. Reports:

  - multiplier statistics (|a| where opt = a*raw(x+k) + r(x)*g) by rank band
  - for OTHER: identical outputs, exp_E change (positive = OTHER better), how many are
    worse than BASE by more than --tol (the M1 acceptance check), multiplier changes
  - how the top-K sets change when OTHER's results replace BASE's

Every OTHER polynomial is checked for validity first: same Y1 as its raw poly, a nonzero
leading coefficient that is an integer multiple of the raw one, and Res(f, g) a nonzero
multiple of N. Any failure is listed and the exit status is 1, whatever exp_E says.

exp_E here is the 2-decimal value CADO prints, so differences below 0.01 are rounding;
--tol defaults to 0.01. M1's acceptance rule (identical poly, or no worse when both are
rescored by one full-precision exp_E) needs a full-precision exp_E implementation,
which is part of M1 and not here yet.

Used for M0 checks 1-2 in GPU_STAGE23_PLAN.md; the same join is M1's acceptance test.
"""

import argparse
import collections
import statistics
import sys

from polyfmt import iter_sopt, raw_key, multiplier, degree, resultant_multiple


def quantile(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))]


def multiplier_report(rows, bands, label):
    print(f"Multiplier |a| ({label}):")
    for n in bands + [len(rows)]:
        a = [abs(r['a']) for r in rows[:n]]
        if not a:
            continue
        ones = sum(x == 1 for x in a) / len(a)
        common = collections.Counter(a).most_common(8)
        print(f"  top {n:>7}: |a|=1 {ones:6.1%}  median {quantile(a, .5)}  95% {quantile(a, .95)}"
              f"  max {max(a)}  most common {common}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('base')
    ap.add_argument('other', nargs='*')
    ap.add_argument('--top', default='150,300,1000', help='comma-separated top-K sizes')
    ap.add_argument('--tol', type=float, default=0.01, help='exp_E tolerance for "worse" (printed values have 2 decimals)')
    args = ap.parse_args()
    bands = [int(x) for x in args.top.split(',')]

    base = []
    for raw, opt in iter_sopt(args.base):
        base.append(dict(key=raw_key(raw), e=opt['exp_E'], a=multiplier(raw, opt), c0=opt['c0']))
    base.sort(key=lambda r: r['e'])
    for i, r in enumerate(base, 1):
        r['rank'] = i
    by_key = {r['key']: r for r in base}
    print(f"BASE {args.base}: {len(base)} polys, exp_E best {base[0]['e']:.2f}")
    print()
    multiplier_report(base, bands, 'BASE')

    if not args.other:
        return

    other, invalid = {}, []
    for path in args.other:
        for raw, opt in iter_sopt(path):
            why = invalid_reason(raw, opt)
            if why:
                invalid.append((path, raw_key(raw), why))
                continue
            other[raw_key(raw)] = dict(e=opt['exp_E'], a=multiplier(raw, opt), c0=opt['c0'])
    missing = [k for k in other if k not in by_key]
    if missing:
        print(f"WARNING: {len(missing)} OTHER polys are not in BASE (different raw set?)", file=sys.stderr)
    joined = [(by_key[k], o) for k, o in other.items() if k in by_key]
    ranks = [b['rank'] for b, o in joined]
    print(f"OTHER ({len(args.other)} file(s)): {len(joined)} polys joined, BASE ranks {min(ranks)}-{max(ranks)}")

    gain = [b['e'] - o['e'] for b, o in joined]
    same = sum(b['c0'] == o['c0'] for b, o in joined)
    worse = [g for g in gain if g < -args.tol]
    print(f"  identical output: {same}/{len(joined)}")
    print(f"  exp_E gain (BASE - OTHER): mean {statistics.mean(gain):.3f}  median {statistics.median(gain):.3f}"
          f"  max {max(gain):.3f}  min {min(gain):.3f}")
    print(f"  better by > {args.tol}: {sum(g > args.tol for g in gain)}   "
          f"worse by > {args.tol}: {len(worse)}" + (f" (worst {min(worse):.3f})" if worse else ""))
    print(f"  multiplier changed: {sum(b['a'] != o['a'] for b, o in joined)}")
    print()

    # Rank with OTHER's results replacing BASE's
    merged = sorted(((other[r['key']]['e'] if r['key'] in other else r['e']), r['rank']) for r in base)
    for k in bands:
        if k > len(merged):
            continue
        cut = merged[k - 1][0]
        newcomers = sorted(rk for e, rk in merged[:k] if rk > k)
        print(f"  merged top {k} (cutoff {cut:.2f}): {len(newcomers)} from BASE rank > {k}"
              + (f", deepest {newcomers[-1]}" if newcomers else ""))

    if invalid:
        print(f"\nINVALID: {len(invalid)} OTHER polynomials (excluded above):")
        for path, key, why in invalid[:10]:
            print(f"  {path}: raw Y1 {key[0]}: {why}")
        sys.exit(1)


def invalid_reason(raw, opt):
    """Why opt is not a valid size-optimized version of raw, or None if it is."""
    if opt.get('Y1') != raw['Y1']:
        return 'Y1 changed'
    if degree(opt) != degree(raw):
        return 'degree changed'
    d = degree(raw)
    if opt['c%d' % d] == 0:
        return 'leading coefficient is zero (multiplier 0: a pure rotation, not a polynomial for N)'
    if opt['c%d' % d] % raw['c%d' % d]:
        return 'leading coefficient is not a multiple of the raw one'
    n = raw.get('n') or opt.get('n')
    if n is None:
        return 'no n: line to check the resultant against'
    k = resultant_multiple(opt, n)
    if k is None:
        return 'N does not divide Res(f, g)'
    if k == 0:
        return 'Res(f, g) = 0'
    return None


if __name__ == '__main__':
    main()
