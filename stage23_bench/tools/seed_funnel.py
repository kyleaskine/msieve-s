#!/usr/bin/env python3
"""How well does the sopt ranking predict root-optimization results, per seed?

  seed_funnel.py --sopt pipeline_work/resopt_output.txt \\
      --msieve-orig pipeline_results/msieve_ropt_orig.p [more.p ...] \\
      --msieve-inv  pipeline_results/msieve_ropt_inv.p \\
      --cado-orig   pipeline_results/cado_ropt_orig.txt \\
      --cado-inv    pipeline_results/cado_ropt_inv.txt \\
      [--bands 150,300,500,750,1000] [--tsv per_seed.tsv]

Seeds are joined on Y1 (shared by every translation/rotation of a seed). The sopt
file sets each seed's exp_E rank. Several files per source are merged (e.g. the
pipeline's top 300 plus a later run on ranks 301-1000). Scores are each tool's own:
msieve 'e' and CADO MurphyE at CADO's default parameters, which are close but not
identical; use rescore.py for one scale.

Ranking: by default seeds are ranked by exp_E from the --sopt files. exp_E is printed
with 2 decimals, so there are many ties, and the pipeline breaks them differently
(`sort -kN,Nn` compares whole lines). Pass --rank-file pipeline_work/resopt_msieve_sorted.ms
to use the pipeline's own order (line order of that file).

msieve only prints results above its min_e, so a seed it ran on can have no result,
and such seeds cluster at worse ranks. Spearman over the seeds with results alone is
biased low. Say which seeds each tool ran on, with --ran TOOL=FILE (the selection file
it was given: .ms lines or CADO format; repeatable) or --through TOOL=K (ranks 1..K).
Spearman is then also computed over all of them, a missing result counting as the
worst score (tied values get their average rank).

Reports the Spearman figures, the top seeds with their ranks, per-band statistics, and
optionally a per-seed TSV: one row per seed that a tool ran on or that has a result,
with msieve_ran / cado_ran columns, so an empty score with ran = 1 means "ran, printed
nothing" and with ran = 0 means "not run".
"""

import argparse

from polyfmt import parse_sopt, parse_msieve_p, parse_cado_ropt, multiplier, spearman

SOURCES = ('msieve_orig', 'msieve_inv', 'cado_orig', 'cado_inv')


def y1s_in(path):
    """Seeds (Y1) in a selection file, in file order: CADO format or .ms lines."""
    out = []
    with open(path) as fh:
        lines = fh.read().splitlines()
    if any(l.startswith('Y1:') for l in lines):
        out = [int(l.split()[1]) for l in lines if l.startswith('Y1:')]
    else:
        out = [int(l.split()[-5]) for l in lines if len(l.split()) >= 5]
    return out


def load_seeds(sopt_paths, rank_file=None):
    pairs = [p for path in sopt_paths for p in parse_sopt(path)]
    pairs.sort(key=lambda p: p[1]['exp_E'])
    seeds = {}
    for raw, opt in pairs:
        if opt['Y1'] not in seeds:
            seeds[opt['Y1']] = dict(exp_E=opt['exp_E'], a=multiplier(raw, opt))
    order = list(seeds)
    if rank_file:
        listed = list(dict.fromkeys(y for y in y1s_in(rank_file) if y in seeds))
        order = listed + [y for y in order if y not in set(listed)]
    for i, y in enumerate(order, 1):
        seeds[y]['rank'] = i
    return seeds


def best_scores(args):
    best = {s: {} for s in SOURCES}
    for s in SOURCES:
        for path in getattr(args, s) or []:
            if s.startswith('msieve'):
                items = [(p['Y1'], p['e']) for p in parse_msieve_p(path)]
            else:
                items = [(p['Y1'], p['MurphyE']) for p in parse_cado_ropt(path)]
            for y, e in items:
                best[s][y] = max(best[s].get(y, 0.0), e)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sopt', nargs='+', required=True, help='sopt output(s) that define the ranking')
    for s in SOURCES:
        ap.add_argument('--' + s.replace('_', '-'), dest=s, nargs='*')
    ap.add_argument('--bands', default='150,300,500,750,1000')
    ap.add_argument('--show', type=int, default=20)
    ap.add_argument('--rank-file', help="rank seeds by this file's line order (the pipeline's sorted .ms)")
    ap.add_argument('--ran', action='append', default=[], metavar='TOOL=FILE',
                    help='TOOL (msieve or cado) ran on the seeds in FILE')
    ap.add_argument('--through', action='append', default=[], metavar='TOOL=K',
                    help='TOOL (msieve or cado) ran on every seed of ranks 1..K')
    ap.add_argument('--tsv', help='write one row per seed')
    args = ap.parse_args()

    seeds = load_seeds(args.sopt, args.rank_file)
    ran = {'msieve': set(), 'cado': set()}
    for spec in args.ran + args.through:
        tool, val = spec.split('=')
        if tool not in ran:
            ap.error(f'unknown tool {tool!r}')
        if spec in args.ran:
            ran[tool] |= set(y1s_in(val))
        else:
            ran[tool] |= {y for y, s in seeds.items() if s['rank'] <= int(val)}
    best = best_scores(args)
    for tool in ('msieve', 'cado'):
        got = {}
        for s in SOURCES:
            if s.startswith(tool):
                for y, e in best[s].items():
                    got[y] = max(got.get(y, 0.0), e)
        rows = sorted(((e, seeds[y]['rank'], y) for y, e in got.items() if y in seeds), reverse=True)
        unknown = [y for y in got if y not in seeds]
        if not rows:
            continue
        print(f"== {tool}: {len(rows)} seeds with results"
              + (f" ({len(unknown)} not in the sopt ranking, ignored)" if unknown else ""))
        print(f"   Spearman(best score, -exp_E rank), seeds with results: "
              f"{spearman([e for e, r, y in rows], [-r for e, r, y in rows]):.3f}")
        if ran[tool]:
            ys = [y for y in ran[tool] if y in seeds]
            print(f"   ... all {len(ys)} seeds it ran on, missing = worst ({sum(y in got for y in ys)} have "
                  f"results): {spearman([got.get(y, 0.0) for y in ys], [-seeds[y]['rank'] for y in ys]):.3f}")
        print(f"   top {args.show} seeds: score  exp_E-rank  exp_E  multiplier")
        for e, r, y in rows[:args.show]:
            print(f"     {e:.4e}  {r:5d}  {seeds[y]['exp_E']:.2f}  {seeds[y]['a']:5d}")
        lo = 1
        for hi in [int(x) for x in args.bands.split(',')]:
            v = sorted((e for e, r, y in rows if lo <= r <= hi), reverse=True)
            if v:
                print(f"   ranks {lo:5d}-{hi:<5d}: n={len(v):4d}  best {v[0]:.4e}  "
                      f"5th {v[min(4, len(v) - 1)]:.4e}  mean {sum(v) / len(v):.4e}")
            lo = hi + 1
        for k in (10, 20, 50):
            print(f"   top {k} seeds: from ranks > 150: {sum(r > 150 for e, r, y in rows[:k])}, "
                  f"> 300: {sum(r > 300 for e, r, y in rows[:k])}")
        print()

    if args.tsv:
        with open(args.tsv, 'w') as fh:
            fh.write('rank\tY1\texp_E\tmultiplier\tmsieve_ran\tcado_ran\t' + '\t'.join(SOURCES) + '\n')
            for y, s in sorted(seeds.items(), key=lambda kv: kv[1]['rank']):
                vals = ['%.4e' % best[src][y] if y in best[src] else '' for src in SOURCES]
                flags = [int(y in ran['msieve'] or any(vals[:2])), int(y in ran['cado'] or any(vals[2:]))]
                if any(vals) or any(flags):
                    fh.write(f"{s['rank']}\t{y}\t{s['exp_E']:.2f}\t{s['a']}\t{flags[0]}\t{flags[1]}\t"
                             + '\t'.join(vals) + '\n')
        print(f"wrote {args.tsv}")


if __name__ == '__main__':
    main()
