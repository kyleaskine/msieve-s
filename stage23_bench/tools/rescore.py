#!/usr/bin/env python3
"""Rescore root-optimized polynomials with CADO's MurphyE under several parameter sets.

  rescore.py --n-from pipeline_results/best150_cado.txt \\
      --msieve pipeline_results/msieve_ropt_orig.p pipeline_results/msieve_ropt_inv.p \\
      --cado pipeline_results/cado_ropt_orig.txt pipeline_results/cado_ropt_inv.txt \\
      [--params ~/cado-nfs/parameters/factor/params.c145] \\
      [--job c145_q10M=29,30,14,10e6] [--set lims=23e6,15e6,2.7e14] \\
      [--top 30] [--per-seed] [--procs 8] [--tsv scores.tsv]

MurphyE takes only Bf (algebraic bound), Bg (rational bound) and area. Sets:
  default         Bf=1e7, Bg=5e6, area=1e16: CADO ropt's built-in values, and the same
                  bounds msieve's `e` uses; always included as the reference
  --params FILE   'job': Bf=2^lpb1, Bg=2^lpb0, area=2^(2I-1)*qmin, as cado-nfs.py derives
                  them from a CADO params file
  --job N=lpbr,lpba,I,qmin   the same derivation from GGNFS-style values
  --set N=Bf,Bg,area         explicit values
Takes the top --top polys per input file by the tool's own score (one per seed with
--per-seed), optimizes skew separately under each set, and prints each set's top 10,
its Spearman correlation with the default ranking, and how many of the default's top
10 / 20 it keeps: how much the choice of parameters changes a shortlist.
"""

import argparse
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

from polyfmt import parse_msieve_p, parse_cado_ropt, parse_cado_poly, cado_poly_text, spearman, cado_binary

DEFAULT = {'Bf': 1e7, 'Bg': 5e6, 'area': 1e16}


def job_params(path):
    vals = {}
    with open(os.path.expanduser(path)) as fh:
        for line in fh:
            m = re.match(r'\s*tasks\.(lpb0|lpb1|I|qmin)\s*=\s*(\d+)', line)
            if m:
                vals[m.group(1)] = int(m.group(2))
    return derived(vals['lpb0'], vals['lpb1'], vals['I'], vals['qmin'])


def derived(lpbr, lpba, logI, qmin):
    """cado-nfs.py's MurphyE inputs: Bf = 2^lpba, Bg = 2^lpbr, area = 2^(2I-1) * qmin."""
    return {'Bf': 2.0 ** lpba, 'Bg': 2.0 ** lpbr, 'area': 2.0 ** (2 * logI - 1) * qmin}


def murphy(score_bin, p, n, skew, params, tmpdir):
    q = dict(p, skew=skew)
    fd, fn = tempfile.mkstemp(suffix='.poly', dir=tmpdir)
    with os.fdopen(fd, 'w') as fh:
        fh.write(cado_poly_text(q, n))
    argv = [score_bin] + [x for k, v in params.items() for x in ('-' + k, repr(v))] + [fn]
    out = subprocess.run(argv, capture_output=True, text=True).stdout.strip()
    os.remove(fn)
    return float(out)


def best_skew(score_bin, p, n, params, tmpdir):
    """Coarse scan of skew over [s/4, 4s] in 2^(1/8) steps, then a 2^(1/64) refinement."""
    s0 = p['skew']
    cand = [(murphy(score_bin, p, n, s0 * 2 ** (k / 8), params, tmpdir), s0 * 2 ** (k / 8)) for k in range(-16, 17)]
    e, s = max(cand)
    cand += [(murphy(score_bin, p, n, s * 2 ** (k / 64), params, tmpdir), s * 2 ** (k / 64)) for k in range(-8, 9)]
    return max(cand)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--msieve', nargs='*', default=[])
    ap.add_argument('--cado', nargs='*', default=[])
    ap.add_argument('--n', type=int)
    ap.add_argument('--n-from', help='any CADO-format file with an n: line')
    ap.add_argument('--params', help="cado-nfs params.cNNN file: adds set 'job'")
    ap.add_argument('--job', action='append', default=[], metavar='NAME=lpbr,lpba,I,qmin')
    ap.add_argument('--set', action='append', default=[], metavar='NAME=Bf,Bg,area')
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--per-seed', action='store_true', help='keep only the best poly of each seed (Y1)')
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--score', help='CADO polyselect/score binary (default: from nfs_config.ini)')
    ap.add_argument('--tsv', help='write every poly with its score and skew under each set')
    args = ap.parse_args()
    args.score = args.score or cado_binary('score')
    n = args.n or parse_cado_poly(args.n_from)['n']

    psets = {'default': DEFAULT}
    if args.params:
        psets['job'] = job_params(args.params)
    for spec in args.job:
        name, vals = spec.split('=')
        lpbr, lpba, logI, qmin = (float(x) for x in vals.split(','))
        psets[name] = derived(lpbr, lpba, logI, qmin)
    for spec in args.set:
        name, vals = spec.split('=')
        bf, bg, area = (float(x) for x in vals.split(','))
        psets[name] = {'Bf': bf, 'Bg': bg, 'area': area}

    polys = []
    for path in args.msieve:
        polys += [dict(p, src=os.path.basename(path), own=p['e'])
                  for p in sorted(parse_msieve_p(path), key=lambda p: -p['e'])[:args.top]]
    for path in args.cado:
        polys += [dict(p, src=os.path.basename(path), own=p['MurphyE'])
                  for p in sorted(parse_cado_ropt(path), key=lambda p: -p['MurphyE'])[:args.top]]
    # one entry per distinct polynomial (orig and inv passes often agree), or per seed
    uniq = {}
    for p in polys:
        k = p['Y1'] if args.per_seed else (p['Y1'], abs(p['c0']), p['Y0'])
        if k not in uniq or p['own'] > uniq[k]['own']:
            uniq[k] = p
    polys = list(uniq.values())

    for name, ps in psets.items():
        print(f"{name:>12}: Bf {ps['Bf']:.4g}  Bg {ps['Bg']:.4g}  area {ps['area']:.4g}")
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(args.procs) as ex:
        for name, ps in psets.items():
            res = list(ex.map(lambda p: best_skew(args.score, p, n, ps, tmp), polys))
            for p, (e, s) in zip(polys, res):
                p[name], p[name + '_skew'] = e, s

    print(f"\n{len(polys)} distinct polys" + (" (one per seed)" if args.per_seed else ""))
    ref = sorted(polys, key=lambda p: -p['default'])
    rank_d = {id(p): i + 1 for i, p in enumerate(ref)}
    for name in psets:
        order = sorted(polys, key=lambda p: -p[name])
        keep10 = len({id(p) for p in order[:10]} & {id(p) for p in ref[:10]})
        keep20 = len({id(p) for p in order[:20]} & {id(p) for p in ref[:20]})
        rho = spearman([p['default'] for p in polys], [p[name] for p in polys])
        print(f"\n{name}: Spearman vs default {rho:.3f}; keeps {keep10}/10 of default's top 10, "
              f"{keep20}/20 of its top 20")
        for i, p in enumerate(order[:10], 1):
            print(f"  {i:2d}. default #{rank_d[id(p)]:<3d} {p['src']:20s} Y1 ..{str(p['Y1'])[-6:]}  "
                  f"E {p[name]:.4e}  skew {p[name + '_skew']:.3g}")
    if args.tsv:
        with open(args.tsv, 'w') as fh:
            fh.write('src\tY1\tY0\tc0\t' + '\t'.join(f"{k}\t{k}_skew" for k in psets) + '\n')
            for p in ref:
                fh.write(f"{p['src']}\t{p['Y1']}\t{p['Y0']}\t{p['c0']}\t"
                         + '\t'.join(f"{p[k]:.6e}\t{p[k + '_skew']:.6g}" for k in psets) + '\n')
        print(f"\nwrote {args.tsv}")


if __name__ == '__main__':
    main()
