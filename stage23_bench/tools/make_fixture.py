#!/usr/bin/env python3
"""Build a compact, checked-in test fixture from a pipeline run.

  make_fixture.py --out stage23_bench/data/c146 \\
      --base cado_sopt_unsorted.txt \\
      --e50 pipeline_work/resopt_output.txt [more effort-50 outputs ...] \\
      [--top 3000] [--sample 10000] [--seed 1]

Writes:
  n.txt              N
  sopt_sample.tsv.gz the top --top raw polys by effort-0 exp_E plus a random --sample of
                     the rest: raw coefficients, CADO's effort-0 result and, where
                     available, the effort-50 result. M1's acceptance set.
  winners/           with --ropt-dir: for the seed of the best msieve result, its raw
                     poly, its effort-50 sopt poly, and the best msieve, CADO-orig and
                     CADO-inv results for that seed (CADO format). M2's benchmark seed.

Integers are written in full; Y1 is shared by the raw poly and both results.
Columns: rank_e0, set (top|random), Y1, raw_Y0, raw_c0..raw_cd, then for e0 and e50:
Y0, c0..cd, skew, lognorm, exp_E, alpha, proj, rroots, multiplier.
"""

import argparse
import gzip
import os
import random

from polyfmt import (iter_sopt, parse_sopt, parse_msieve_p, parse_cado_ropt, raw_key, multiplier,
                     degree, cado_poly_text, resultant_multiple)

STATS = ('skew', 'lognorm', 'exp_E', 'alpha', 'proj', 'rroots')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', default='cado_sopt_unsorted.txt')
    ap.add_argument('--e50', nargs='*', default=['pipeline_work/resopt_output.txt'])
    ap.add_argument('--top', type=int, default=3000)
    ap.add_argument('--sample', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--ropt-dir', help='pipeline_results/ of the same run, for winners/')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Two passes over the (possibly multi-GB) base file: rank on exp_E alone, then fetch
    # only the selected pairs.
    order = sorted((opt['exp_E'], i) for i, (_, opt) in enumerate(iter_sopt(args.base)))
    rest = list(range(args.top, len(order)))
    picked = [(i, 'top') for i in range(min(args.top, len(order)))]
    picked += [(i, 'random') for i in sorted(random.Random(args.seed).sample(rest, min(args.sample, len(rest))))]
    want_y1 = winner_y1(args.ropt_dir) if args.ropt_dir else None
    file_idx = {order[rank][1]: rank for rank, _ in picked}
    chosen, seed_pairs = {}, []
    for i, (raw, opt) in enumerate(iter_sopt(args.base)):
        if i in file_idx:
            chosen[file_idx[i]] = (raw, opt)
        if raw['Y1'] == want_y1:
            seed_pairs.append((raw, opt))
    base = chosen
    e50 = {raw_key(r): o for path in args.e50 for r, o in parse_sopt(path)}
    first = base[picked[0][0]][0]
    d = degree(first)
    with open(os.path.join(args.out, 'n.txt'), 'w') as fh:
        fh.write('%d\n' % first['n'])

    def result_cols(raw, opt):
        if opt is None:
            return [''] * (d + 3 + len(STATS))
        return ([str(opt['Y0'])] + [str(opt['c%d' % i]) for i in range(d + 1)]
                + [str(opt[s]) for s in STATS] + [str(multiplier(raw, opt))])

    res_hdr = ['Y0'] + ['c%d' % i for i in range(d + 1)] + list(STATS) + ['multiplier']
    header = (['rank_e0', 'set', 'Y1', 'raw_Y0'] + ['raw_c%d' % i for i in range(d + 1)]
              + ['e0_' + h for h in res_hdr] + ['e50_' + h for h in res_hdr])
    path = os.path.join(args.out, 'sopt_sample.tsv.gz')
    with open(path, 'wb') as raw_fh, gzip.GzipFile(fileobj=raw_fh, mode='wb', mtime=0) as gz:
        gz.write(('\t'.join(header) + '\n').encode())
        n50 = 0
        for i, which in picked:
            raw, opt = base[i]
            o50 = e50.get(raw_key(raw))
            n50 += o50 is not None
            row = ([str(i + 1), which, str(raw['Y1']), str(raw['Y0'])]
                   + [str(raw['c%d' % k]) for k in range(d + 1)]
                   + result_cols(raw, opt) + result_cols(raw, o50))
            gz.write(('\t'.join(row) + '\n').encode())
    print(f"wrote {path}: {len(picked)} rows ({n50} with effort-50 results), "
          f"{os.path.getsize(path) / 1e6:.1f} MB")

    if args.ropt_dir:
        write_winners(args, first['n'], seed_pairs, e50)


def best_msieve(ropt_dir):
    ms = [p for f in ('msieve_ropt_orig.p', 'msieve_ropt_inv.p')
          for p in parse_msieve_p(os.path.join(ropt_dir, f))]
    return max(ms, key=lambda p: p['e'])


def winner_y1(ropt_dir):
    return best_msieve(ropt_dir)['Y1']


def write_winners(args, n, seed_pairs, e50):
    """seed_pairs: the base (raw, opt) pairs of the winning seed (same Y1)."""
    best = best_msieve(args.ropt_dir)
    y1 = best['Y1']
    out = {'msieve_best': (best, f"msieve -npr best overall, e {best['e']:.4e}")}
    for f, name in (('cado_ropt_orig.txt', 'cado_orig_best'), ('cado_ropt_inv.txt', 'cado_inv_best')):
        same = [p for p in parse_cado_ropt(os.path.join(args.ropt_dir, f)) if p['Y1'] == y1]
        if same:
            p = max(same, key=lambda p: p['MurphyE'])
            out[name] = (p, f"CADO polyselect_ropt best for this seed, MurphyE {p['MurphyE']:.4e} (default params)")
    for raw, opt in sorted(seed_pairs, key=lambda p: p[1]['exp_E'])[:1]:
        out['raw'] = (raw, 'raw stage-1 polynomial (msieve -nps)')
        o = e50.get(raw_key(raw), opt)
        out['sopt'] = (o, f"CADO sopt result (exp_E {o['exp_E']}, multiplier {multiplier(raw, o)})")
    wdir = os.path.join(args.out, 'winners')
    os.makedirs(wdir, exist_ok=True)
    for name, (p, note) in out.items():
        q = {k: v for k, v in p.items() if k in ('Y0', 'Y1', 'skew') or (k[0] == 'c' and k[1:].isdigit())}
        with open(os.path.join(wdir, name + '.poly'), 'w') as fh:
            fh.write(f"# {note}\n# Res/N = {resultant_multiple(q, n)}\n" + cado_poly_text(q, n))
    print(f"wrote {wdir}/: {', '.join(sorted(out))} (seed Y1 {y1})")


if __name__ == '__main__':
    main()
