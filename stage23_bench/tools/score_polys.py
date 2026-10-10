#!/usr/bin/env python3
"""MurphyE (CADO's own code) of CADO-format polynomials, e.g. s23_ropt -out.

  score_polys.py FILE [--top N] [--trans T] [--params default|job:lpbr,lpba,I,qmin]
                 [--points K] [--eval-points K | --cado] [--known POLY ...] [--show N]
                 [--threads 8] [--lattice LOGI,J --qband QMIN,QMAX,NQ [--useskew]]

Reads the blocks of FILE (n:, Y0:, Y1:, c0: ...; a '# ...' line before a block is kept as
its label) and scores each with cado_murphy (CADO's MurphyE loop in-process, alpha
computed once per poly; bit-identical to CADO's MurphyE at the same skew) at the best
skew on a fixed grid: 2^(j/32) over s0/8 .. 8 s0 (s0 = |c0/c5|^(1/5)), then 2^(j/256)
around the best two grid points. At CADO's 1,000 sample points MurphyE has many spurious
peaks in skew (sampling noise; at 64,000 points the curve has one), and a grid relative
to each poly's own start (as in rescore.py) can give one polynomial different scores.
The 1,000-point value reads high by about 0.2% (up to 1.4%). --trans T also searches the
translation (pattern search on MurphyE from step T); its apparent 1-2% gains on c168
were that noise (GPU_STAGE23_PLAN.md, "Review checks"). C181 does show translation
sensitivity. For finalist refinement with controls, independent validation and exported
polynomials, use refine_polys.py. By default the skew (and translation) search uses 4,000 sample angles and
the printed value 16,000 (--points, --eval-points): on the c168 winners 16,000 is within
0.02% of 256,000, measured evidence rather than a guarantee. The printed value is
re-evaluated at the configuration the search chose; it does not search again at 16,000.
--cado scores exactly as CADO does (1,000 points for both), for comparison with CADO's
printed MurphyE. --known scores reference polys (e.g. the benchmark
winners) the same way and says where each would rank. Default bounds are CADO's
(Bf 1e7, Bg 5e6, area 1e16), the scale of the plan's targets.

--lattice LOGI,J ranks instead by the relations a lattice siever with a 2^LOGI x J region
can expect over the special-q band --qband QMIN,QMAX,NQ (NQ evenly spaced points,
trapezoid rule, about one special-q per prime): MurphyE's smoothness model (rho of log
norm + alpha over log B, with --params' bounds) averaged over the region the siever
actually covers. Each special-q lattice is reduced at the poly's skew as the user's GPU
siever does (Gauss reduction under the skewed norm, the shorter vector multiplying i), so
a skew far above (I/J) q is not realized, which MurphyE's ellipse ignores. The skew is
MurphyE's best (as skewopt would declare it), or the file's own with --useskew (a poly
without a skew line gets the best one; cado_murphy warns). On c208 it reproduces the yield
ratios of five test-sieved polys (each against poly A: 24 ratios at three q and two region
shapes) within 0.04, worst 0.038, where MurphyE missed a 5-12% loss (GPU_STAGE23_PLAN.md,
known problem 7).
Its absolute relations per special-q read about 1.9x the test sieve's: compare ratios.
--nlat, --npts and --latseed change the lattice sample counts and random stream for
independent checks. Common samples reduce comparison noise; they do not eliminate it.
--useskew also evaluates a fixed declared skew in MurphyE mode, without re-optimizing it.
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile

from polyfmt import cado_murphy_binary, cado_poly_text, parse_cado_poly, read_cado_blocks
from rescore import derived


read_blocks = read_cado_blocks  # the old name, still imported by scripts


def score_bounds(params):
    """Parse the shared --params syntax, rejecting invalid or overflowing bounds."""
    if params == 'default':
        return None
    try:
        kind, values = params.split(':', 1)
        lpbr, lpba, logi, qmin = (float(x) for x in values.split(','))
        if kind != 'job' or not all(math.isfinite(x) for x in (lpbr, lpba, logi, qmin)):
            raise ValueError()
        d = derived(lpbr, lpba, logi, qmin)
        if not all(math.isfinite(v) for v in d.values()) or min(d['Bf'], d['Bg']) <= 1 or d['area'] <= 0:
            raise ValueError()
        return d
    except (ValueError, OverflowError):
        raise ValueError('--params needs default or job:lpbr,lpba,I,qmin with finite Bf/Bg > 1 and area > 0') from None


def _run_cado_murphy(polys, n, args, extra, keep_skew=False):
    """cado_murphy's output rows (tab-split) for polys, one per poly in order, with the
    sample counts and --params bounds of args; its stderr (warnings) is passed on"""
    d = score_bounds(args.params)
    with tempfile.NamedTemporaryFile('w', suffix='.poly', delete=False) as fh:
        for p in polys:
            q = {k: v for k, v in p.items() if k != 'n' and (k != 'skew' or keep_skew)}
            fh.write(cado_poly_text(q, n) + '\n')
        path = fh.name
    try:
        argv = [cado_murphy_binary(), '-t', str(args.threads)] + extra
        points, eval_points = getattr(args, 'points', 0), getattr(args, 'eval_points', 0)
        if points:
            argv += ['-K', str(points)]
        if eval_points:
            argv += ['-Keval', str(eval_points)]
        if d is not None:
            argv += ['-Bf', repr(d['Bf']), '-Bg', repr(d['Bg']), '-area', repr(d['area'])]
        r = subprocess.run(argv + [path], capture_output=True, text=True)
    finally:
        os.remove(path)
    if r.stderr:
        sys.stderr.write(r.stderr)
    if r.returncode:
        raise RuntimeError(f"cado_murphy failed (status {r.returncode})")
    res = [line.split('\t') for line in r.stdout.split('\n') if line]
    if len(res) != len(polys):
        raise RuntimeError(f"cado_murphy scored {len(res)} of {len(polys)} polynomials")
    return res


def murphy(polys, n, args):
    """[(MurphyE, skew, t)] for each poly, in order"""
    useskew = getattr(args, 'useskew', False)
    res = _run_cado_murphy(polys, n, args, ['-trans', str(args.trans)] + (['-useskew'] if useskew else []),
                           keep_skew=useskew)
    return [(float(r[1]), float(r[2]), int(r[3])) for r in res]


def lattice_scores(polys, n, args):
    """[(band relations, skew, MurphyE at that skew, [relations per special-q at each q])]"""
    extra = ['-lattice', args.lattice, '-qband', args.qband] + (['-useskew'] if args.useskew else [])
    for name in ('nlat', 'npts', 'latseed'):
        if getattr(args, name, None) is not None:
            extra += ['-' + name, str(getattr(args, name))]
    res = _run_cado_murphy(polys, n, args, extra, keep_skew=args.useskew)
    return [(float(r[1]), float(r[2]), float(r[3]), [float(x) for x in r[4:]]) for r in res]


def main_lattice(args, blocks, known, n):
    res = lattice_scores([p for _, p in blocks], n, args) if blocks else []
    kres = lattice_scores([p for _, p in known], n, args) if known else []
    best = max([r[0] for r in res + kres] or [1])
    ranked = sorted(zip(res, range(len(blocks))), key=lambda r: -r[0][0])
    print(f"{len(blocks)} polys, lattice-aware ({args.lattice}, q band {args.qband}, {args.params}"
          f"{', declared skew where the file gives one' if args.useskew else ''}), best first; ratio to the best; relations per "
          f"special-q at each q (about 1.9x a test sieve's)")
    for i, ((b, s, e, rel), j) in enumerate(ranked[:args.show]):
        print(f"{i + 1:4d}  {b / best:.4f}  MurphyE {e:.4e}  skew {s:12.1f}  rel/q {' '.join(f'{x:.1f}' for x in rel)}"
              f"  (input #{j + 1})  {blocks[j][0]}")
    for (name, _), (b, s, e, rel) in zip(known, kres):
        better = sum(1 for (x, *_), _ in ranked if x > b)
        print(f"known {name}: {b / best:.4f}, MurphyE {e:.4e} at skew {s:.1f}; {better} of the scored polys beat it")


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
    ap.add_argument('--lattice', help='LOGI,J: rank by the lattice-aware relations over --qband')
    ap.add_argument('--qband', help='QMIN,QMAX,NQ for --lattice')
    ap.add_argument('--useskew', action='store_true', help="evaluate the file's skew instead of searching it")
    ap.add_argument('--nlat', type=int, help='lattice samples per q (default 32)')
    ap.add_argument('--npts', type=int, help='region samples per lattice (default 2048)')
    ap.add_argument('--latseed', type=int, help='independent lattice sample stream (default 0)')
    ap.add_argument('--show', type=int, default=20)
    args = ap.parse_args()
    try:
        score_bounds(args.params)
    except ValueError as e:
        ap.error(str(e))
    if args.useskew and args.trans:
        ap.error('--useskew evaluates a fixed configuration; it cannot be combined with --trans')
    if not args.lattice and any(x is not None for x in (args.nlat, args.npts, args.latseed)):
        ap.error('--nlat, --npts and --latseed need --lattice')
    if args.cado:
        args.points = args.eval_points = 1000
    blocks = read_blocks(args.file)
    if args.top:
        blocks = blocks[:args.top]
    known = [(os.path.basename(p), parse_cado_poly(p)) for p in args.known]
    n = blocks[0][1]['n'] if blocks else known[0][1]['n']
    if args.lattice or args.qband:
        if not (args.lattice and args.qband) or args.trans:
            ap.error('--lattice and --qband go together (and without --trans)')
        return main_lattice(args, blocks, known, n)
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
