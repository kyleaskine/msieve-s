#!/usr/bin/env python3
"""Compare size-optimization results, joined on the raw polynomial.

  sopt_compare.py BASE [OTHER ...] [--top 150,300,1000] [--tol 0.01] [--rescore]

BASE is a full sopt output, e.g. cado_sopt_unsorted.txt (CADO, effort 0), or a fixture
from make_fixture.py (sopt_sample.tsv.gz; --effort e0|e50 picks which CADO result). Each
OTHER covers a subset of the same raw polys, e.g. pipeline_work/resopt_output.txt
(effort 50) or a GPU sopt's output; several OTHER files are merged. Reports:

  - multiplier statistics (|a| where opt = a*raw(x+k) + r(x)*g) by rank band
  - for OTHER: identical outputs, exp_E change (positive = OTHER better), how many are
    worse than BASE by more than --tol (the M1 acceptance check), multiplier changes
  - how the top-K sets change when OTHER's results replace BASE's

Every OTHER polynomial is checked for validity first: same Y1 as its raw poly, a nonzero
leading coefficient that is an integer multiple of the raw one, and Res(f, g) a nonzero
multiple of N. Any failure is listed and the exit status is 1, whatever exp_E says.

exp_E here is the 2-decimal value CADO prints, so differences below 0.01 are rounding;
--tol defaults to 0.01. --rescore applies M1's acceptance rule instead: a row passes if
OTHER's polynomial is identical to BASE's, or if its exp_E recomputed at full precision
by CADO's own code (cado_expe.c, at the skew sopt prints) is no worse than BASE's recomputed
the same way (within 1e-6). With --rescore the exit status is 1 if any row is worse, or
if a fixture BASE has rows missing from OTHER in a set (top / random) OTHER covers.

M1 acceptance:
  sopt_compare.py stage23_bench/data/c146/sopt_sample.tsv.gz GPU_OUTPUT --rescore
Used for M0 checks 1-2 in GPU_STAGE23_PLAN.md.
"""

import argparse
import collections
import statistics
import sys

from polyfmt import iter_sopt, load_fixture, raw_key, multiplier, degree, resultant_multiple, coeffs, cado_expe


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
    ap.add_argument('--effort', choices=('e0', 'e50'), default='e0', help='which CADO result a fixture BASE supplies')
    ap.add_argument('--rescore', action='store_true',
                    help='M1 acceptance: identical, or no worse at full precision (CADO code)')
    args = ap.parse_args()
    bands = [int(x) for x in args.top.split(',')]

    fixture = args.base.endswith('.tsv.gz')
    if fixture:
        pairs = ((raw, opt, rank, which) for raw, opt, rank, which in load_fixture(args.base, args.effort) if opt)
    else:
        pairs = ((raw, opt, None, None) for raw, opt in iter_sopt(args.base))   # streamed
    # Rows keep a hash of the polynomial; --rescore also needs the polynomial itself, kept
    # here for a (small) fixture, and fetched for the joined rows only by a second pass over
    # a full BASE, so a multi-million-poly BASE is never held in memory.
    base, n = [], None
    for raw, opt, rank, which in pairs:
        n = n or opt.get('n') or raw.get('n')
        c = compact(opt)
        base.append(dict(key=raw_key(raw), e=opt['exp_E'], a=multiplier(raw, opt), poly=hash(c),
                         rank=rank, set=which, full=c if args.rescore and fixture else None))
    if not fixture:
        base.sort(key=lambda r: r['e'])
        for i, r in enumerate(base, 1):
            r['rank'] = i
    else:
        base.sort(key=lambda r: r['rank'])
    by_key = {r['key']: r for r in base}
    print(f"BASE {args.base}: {len(base)} polys" + (f" ({args.effort} results of a fixture)" if fixture else "")
          + f", exp_E best {min(r['e'] for r in base):.2f}")
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
            c = compact(opt)
            other[raw_key(raw)] = dict(e=opt['exp_E'], a=multiplier(raw, opt), poly=hash(c),
                                       full=c if args.rescore else None)
    missing = [k for k in other if k not in by_key]
    if missing:
        print(f"WARNING: {len(missing)} OTHER polys are not in BASE (different raw set?)", file=sys.stderr)
    joined = [(by_key[k], o) for k, o in other.items() if k in by_key]
    if args.rescore and not fixture:
        need = {b['key'] for b, o in joined}
        for raw, opt in iter_sopt(args.base):   # second pass: the joined rows' polys
            if raw_key(raw) in need:
                by_key[raw_key(raw)]['full'] = compact(opt)
    ranks = [b['rank'] for b, o in joined]
    print(f"OTHER ({len(args.other)} file(s)): {len(joined)} polys joined, BASE ranks {min(ranks)}-{max(ranks)}")

    gain = [b['e'] - o['e'] for b, o in joined]
    same = sum(identical(b, o) for b, o in joined)
    worse = [g for g in gain if g < -args.tol]
    print(f"  identical output: {same}/{len(joined)}")
    print(f"  exp_E gain (BASE - OTHER): mean {statistics.mean(gain):.3f}  median {statistics.median(gain):.3f}"
          f"  max {max(gain):.3f}  min {min(gain):.3f}")
    print(f"  better by > {args.tol}: {sum(g > args.tol for g in gain)}   "
          f"worse by > {args.tol}: {len(worse)}" + (f" (worst {min(worse):.3f})" if worse else ""))
    print(f"  multiplier changed: {sum(b['a'] != o['a'] for b, o in joined)}")
    print()

    failed = False
    if args.rescore:
        failed = acceptance(joined, n) > 0
        if fixture:
            covered = {b['set'] for b, o in joined}
            missing = [b for b in base if b['set'] in covered and b['key'] not in other]
            if missing:
                print(f"  MISSING: {len(missing)} BASE rows of the covered set(s) {sorted(covered)} are not in OTHER")
                failed = True

    # Rank with OTHER's results replacing BASE's (only meaningful for a full BASE)
    merged = sorted(((other[r['key']]['e'] if r['key'] in other else r['e']), r['rank']) for r in base)
    for k in ([] if fixture else bands):
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
    if failed:
        print("\nACCEPTANCE FAILED")
        sys.exit(1)


def compact(opt):
    """The polynomial pair itself: (Y0, Y1, (c0, ..., cd))."""
    return (opt['Y0'], opt['Y1'], tuple(coeffs(opt)))


def expand(c):
    """compact() back to a poly dict."""
    y0, y1, cs = c
    return dict(Y0=y0, Y1=y1, **{'c%d' % i: x for i, x in enumerate(cs)})


def identical(b, o):
    """Exact when both polynomials are kept (--rescore), else by hash (a report only)."""
    if b['full'] is not None and o['full'] is not None:
        return b['full'] == o['full']
    return b['poly'] == o['poly']


def acceptance(joined, n, eps=1e-6):
    """M1 rule: identical, or full-precision exp_E (CADO's code) no worse than BASE's.
    Returns the number of rows that are worse."""
    diff = [(b, o) for b, o in joined if not identical(b, o)]
    full_b = cado_expe([expand(b['full']) for b, o in diff], n) if diff else []
    full_o = cado_expe([expand(o['full']) for b, o in diff], n) if diff else []
    verdict = {id(b): 'identical' for b, o in joined if identical(b, o)}
    worst = []
    for (b, o), fb, fo in zip(diff, full_b, full_o):
        d = fo['exp_E'] - fb['exp_E']
        verdict[id(b)] = 'no worse' if d <= eps else 'worse'
        if d > eps:
            worst.append((d, b['rank'], b['set']))
    print("  M1 acceptance (identical, or full-precision exp_E no worse than BASE):")
    groups = collections.defaultdict(list)
    for b, o in joined:
        groups[b['set'] or 'all'].append(verdict[id(b)])
    for name, v in sorted(groups.items()):
        c = collections.Counter(v)
        print(f"    {name:>7}: {len(v)} rows: identical {c['identical']}, no worse {c['no worse']}, "
              f"worse {c['worse']}  -> pass {(c['identical'] + c['no worse']) / len(v):.2%}")
    for d, rank, which in sorted(worst, reverse=True)[:10]:
        print(f"      worse by {d:.6f} (BASE rank {rank}{', ' + which if which else ''})")
    print()
    return len(worst)


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
