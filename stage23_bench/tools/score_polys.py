#!/usr/bin/env python3
"""MurphyE (CADO's own code) of CADO-format polynomials, e.g. s23_ropt -out.

  score_polys.py FILE [--top N] [--trans T] [--params default|job:lpbr,lpba,I,qmin]
                 [--points K] [--eval-points K | --cado] [--known POLY ...] [--show N]
                 [--threads 8]

Reads the blocks of FILE (n:, Y0:, Y1:, c0: ...; a '# ...' line before a block is kept as
its label) and scores each with cado_murphy (CADO's MurphyE loop in-process, alpha
computed once per poly; bit-identical to CADO's MurphyE at the same skew) at the best
skew on a fixed grid: 2^(j/32) over s0/8 .. 8 s0 (s0 = |c0/c5|^(1/5)), then 2^(j/256)
around the best two grid points. At CADO's 1,000 sample points MurphyE has many spurious
peaks in skew (sampling noise; at 64,000 points the curve has one), and a grid relative
to each poly's own start (as in rescore.py) can give one polynomial different scores.
The 1,000-point value reads high by about 0.2% (up to 1.4%). --trans T also searches the
translation (pattern search on MurphyE from step T); its apparent 1-2% gains were that
noise (worth at most 0.1% when integrated accurately; GPU_STAGE23_PLAN.md, "Review
checks"). By default the skew (and translation) search uses 4,000 sample angles and
the printed value 16,000 (--points, --eval-points): on the c168 winners 16,000 is within
0.02% of 256,000, measured evidence rather than a guarantee. The printed value is
re-evaluated at the configuration the search chose; it does not search again at 16,000.
--cado scores exactly as CADO does (1,000 points for both), for comparison with CADO's
printed MurphyE. --known scores reference polys (e.g. the benchmark
winners) the same way and says where each would rank. Default bounds are CADO's
(Bf 1e7, Bg 5e6, area 1e16), the scale of the plan's targets.
"""

import argparse
import os
import subprocess
import tempfile

from polyfmt import cado_murphy_binary, cado_poly_text, parse_cado_poly, read_cado_blocks
from rescore import derived


read_blocks = read_cado_blocks  # the old name, still imported by scripts


def murphy(polys, n, args):
    """[(MurphyE, skew, t)] for each poly, in order"""
    with tempfile.NamedTemporaryFile('w', suffix='.poly', delete=False) as fh:
        for p in polys:
            q = {k: v for k, v in p.items() if k not in ('skew', 'n')}
            fh.write(cado_poly_text(q, n) + '\n')
        path = fh.name
    try:
        argv = [cado_murphy_binary(), '-t', str(args.threads), '-trans', str(args.trans)]
        points, eval_points = getattr(args, 'points', 0), getattr(args, 'eval_points', 0)
        if points:
            argv += ['-K', str(points)]
        if eval_points:
            argv += ['-Keval', str(eval_points)]
        if args.params != 'default':
            lpbr, lpba, logI, qmin = (float(x) for x in args.params.split(':', 1)[1].split(','))
            d = derived(lpbr, lpba, logI, qmin)
            argv += ['-Bf', repr(d['Bf']), '-Bg', repr(d['Bg']), '-area', repr(d['area'])]
        out = subprocess.run(argv + [path], capture_output=True, text=True, check=True).stdout.split('\n')
    finally:
        os.remove(path)
    res = [line.split('\t') for line in out if line]
    if len(res) != len(polys):
        raise RuntimeError(f"cado_murphy scored {len(res)} of {len(polys)} polynomials")
    return [(float(r[1]), float(r[2]), int(r[3])) for r in res]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file')
    ap.add_argument('--top', type=int, default=0, help='score only the first N blocks (0: all)')
    ap.add_argument('--trans', type=int, default=0, help='also search the translation from this step')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--params', default='default')
    ap.add_argument('--points', type=int, default=4000,
                    help='MurphyE sample angles for the skew/translation search (default 4000)')
    ap.add_argument('--eval-points', type=int, default=16000,
                    help='sample angles for the printed MurphyE at the chosen skew (default 16000)')
    ap.add_argument('--cado', action='store_true', help="CADO's MurphyE: 1000 points for both")
    ap.add_argument('--known', nargs='*', default=[])
    ap.add_argument('--show', type=int, default=20)
    args = ap.parse_args()
    if args.cado:
        args.points = args.eval_points = 1000
    blocks = read_blocks(args.file)
    if args.top:
        blocks = blocks[:args.top]
    known = [(os.path.basename(p), parse_cado_poly(p)) for p in args.known]
    n = blocks[0][1]['n'] if blocks else known[0][1]['n']
    res = murphy([p for _, p in blocks], n, args) if blocks else []
    kres = murphy([p for _, p in known], n, args) if known else []
    ranked = sorted(zip(res, range(len(blocks))), key=lambda r: -r[0][0])
    what = 'CADO defaults' if args.params == 'default' else args.params
    pts = f", K {args.points}/{args.eval_points}"
    print(f"{len(blocks)} polys scored ({what}{pts}{', translation searched' if args.trans else ''}), best first:")
    for i, ((e, s, t), j) in enumerate(ranked[:args.show]):
        print(f"{i + 1:4d}  MurphyE {e:.4e}  skew {s:12.1f}  dt {t:8d}  (input #{j + 1})  {blocks[j][0]}")
    for (name, _), (e, s, t) in zip(known, kres):
        better = sum(1 for (x, _, _), _ in ranked if x > e)
        print(f"known {name}: MurphyE {e:.4e} at skew {s:.1f}, dt {t}; {better} of the scored polys beat it")


if __name__ == '__main__':
    main()
