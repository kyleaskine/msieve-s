#!/usr/bin/env python3
"""Refine a preselected finalist set in translation and skew, keeping its controls.

  refine_polys.py finalists.poly --out results/refined --params job:31,32,15,45e6
  refine_polys.py finalists.poly --out results/refined --params job:31,32,15,45e6 \
      --lattice 15,16384 --qband 45e6,2e8,7

Searches MurphyE, or the lattice band score when --lattice/--qband are supplied.
The input skew is the starting control; if absent, use the best MurphyE skew at
--points. The bounded search covers translations within +/- 4 initial steps and
skews within a factor of 4 of the control. The default initial translation step is
half the control skew. It is a local finalist refinement, not a new root search.

The search uses at least 16,000 MurphyE angles. A proposal is evaluated at >= 4x
that count; lattice validation uses a separate random stream and at least as many samples.
Validation scores never feed back into the search. These are model checks, not a
test-sieve result or an uncertainty bound. Close gains still need test sieving.

Writes PREFIX.poly (every control and proposal), PREFIX.selected.poly (one per
input, accepting a proposal only if both search and validation scores improve),
and PREFIX.json (settings, scores, exact translations and evaluation counts).
Existing outputs are refused unless --force is supplied. Inputs are never changed.
Common coefficient content is removed on both sides, as in CADO; the original and
divisors are retained in JSON. Failed inputs are reported and excluded from the
polynomial exports; valid inputs still finish (exit status 1 for a partial batch).
Default: one CPU thread.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time
import uuid

from polyfmt import cado_poly_text, coeffs, read_cado_blocks, resultant, translate
from score_polys import _run_cado_murphy, score_bounds


def translated_pair(p, t, skew):
    q = dict(p)
    for i, c in enumerate(translate(coeffs(p), t)):
        q[f'c{i}'] = c
    q['Y0'] += t * q['Y1']
    q['skew'] = skew
    if resultant(p) != resultant(q):
        raise RuntimeError('translation changed the resultant')
    return q


def primitive_pair(p):
    """Match CADO's positive-content division on both sides, before translation."""
    if p.get('n', 0) <= 1:
        raise ValueError('input needs n > 1')
    cs = coeffs(p)
    if len(cs) > 11:
        raise ValueError('CADO supports algebraic degree at most 10')
    if len(cs) < 2 or not cs[-1] or not p.get('Y1'):
        raise ValueError('need nonzero Y1 and algebraic leading coefficient, degree >= 1')
    if 'skew' in p and (not math.isfinite(p['skew']) or p['skew'] <= 0):
        raise ValueError('declared skew must be finite and positive')
    fcontent, gcontent = math.gcd(*cs), math.gcd(p['Y0'], p['Y1'])
    q = dict(p, Y0=p['Y0'] // gcontent, Y1=p['Y1'] // gcontent)
    q.update({f'c{i}': c // fcontent for i, c in enumerate(cs)})
    r = resultant(q)
    if not r or r % p['n']:
        raise ValueError('primitive resultant must be a nonzero multiple of n')
    return q, dict(algebraic=fcontent, rational=gcontent)


def refinement_result(p, row, args):
    if len(row) == 3 and row[1] == 'error':
        raise ValueError(row[2])
    if len(row) != 14:
        raise RuntimeError('unexpected refinement output')
    s0, train0 = float(row[1]), float(row[2])
    t, s1, train1 = int(row[3]), float(row[4]), float(row[5])
    eval0, eval1, e0, e1 = (float(x) for x in row[6:10])
    if not all(math.isfinite(x) and x > 0 for x in (s0, s1, train0, train1, eval0, eval1, e0, e1)):
        raise ValueError('nonfinite or nonpositive refinement score')
    control = dict(p, skew=s0)
    proposal = translated_pair(p, t, s1)
    if proposal['Y0'] != int(row[12]) or proposal['c0'] != int(row[13]):
        raise ValueError('Python/CADO exact translation mismatch')
    accepted = train1 > train0 and eval1 > eval0 * (1 + args.min_gain)
    return dict(status='ok', control=control, proposal=proposal, translation=t,
                train_control=train0, train_proposal=train1,
                validation_control=eval0, validation_proposal=eval1,
                murphy_control=e0, murphy_proposal=e1,
                validation_gain=eval1 / eval0 - 1, accepted=accepted,
                evaluations=int(row[10]), evaluation_limit=bool(int(row[11])))


def refine(polys, args):
    """Return success/error records in input order; one bad poly cannot drop the batch."""
    if not polys:
        raise ValueError('no polynomials to refine')
    results, groups = [], {}
    for i, p in enumerate(polys):
        # Keep invalid nonfinite input values readable in strict JSON as well.
        original = {k: repr(v) if isinstance(v, float) and not math.isfinite(v) else v for k, v in p.items()}
        results.append(dict(status='error', original=original, accepted=False))
        try:
            q, content = primitive_pair(p)
        except (ValueError, KeyError) as e:
            results[i]['error'] = str(e)
            continue
        results[i]['content'] = content
        groups.setdefault(q['n'], []).append((i, q))
    extra = ['-refine', '-maxeval', str(args.max_evals), '-trans', str(args.trans)]
    if args.lattice:
        extra += ['-lattice', args.lattice, '-qband', args.qband,
                  '-nlat', str(args.nlat), '-npts', str(args.npts), '-latseed', str(args.seed),
                  '-eval-nlat', str(args.eval_nlat), '-eval-npts', str(args.eval_npts),
                  '-eval-seed', str(args.eval_seed)]
    def run_group(n, group):
        try:
            raw = _run_cado_murphy([p for _, p in group], n, args, extra, keep_skew=True)
            if len(raw) != len(group):
                raise RuntimeError('incomplete refinement output')
        except (RuntimeError, OSError, subprocess.SubprocessError) as e:
            # A parser error or killed worker must not discard completed groups.
            # Bisect failed batches, reducing their memory footprint and isolating
            # bad inputs. Each singleton is attempted at most once after splitting.
            if len(group) > 1:
                middle = len(group) // 2
                run_group(n, group[:middle])
                run_group(n, group[middle:])
            else:
                results[group[0][0]]['error'] = str(e)
            return
        for j, ((i, p), row) in enumerate(zip(group, raw)):
            try:
                if not row or int(row[0]) != j:
                    raise ValueError('unexpected refinement output order')
                results[i].update(refinement_result(p, row, args))
            except (ValueError, RuntimeError, OverflowError) as e:
                results[i]['error'] = str(e)
    for n, group in groups.items():
        run_group(n, group)
    return results


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file')
    ap.add_argument('--out', required=True, help='output prefix; input is never overwritten')
    ap.add_argument('--force', action='store_true', help='replace existing outputs at this prefix')
    ap.add_argument('--top', type=int, default=0, help='first N inputs (0: all); supply an already diverse shortlist')
    ap.add_argument('--params', default='default', help='default or job:lpbr,lpba,I,qmin, as in score_polys.py')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--points', type=int, default=16000)
    ap.add_argument('--eval-points', type=int, default=64000)
    ap.add_argument('--max-evals', type=int, default=384, help='maximum joint objective evaluations per poly')
    ap.add_argument('--trans', type=int, default=0, help='initial translation step (0: half the starting skew)')
    ap.add_argument('--min-gain', type=float, default=0, help='minimum fractional validation gain for selection')
    ap.add_argument('--lattice', help='LOGI,J: search the lattice band score instead of MurphyE')
    ap.add_argument('--qband', help='QMIN,QMAX,NQ; required with --lattice')
    ap.add_argument('--nlat', type=int, default=32)
    ap.add_argument('--npts', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-nlat', type=int, default=128)
    ap.add_argument('--eval-npts', type=int, default=4096)
    ap.add_argument('--eval-seed', type=int, default=1)
    return ap


def write_outputs(paths, contents, force=False):
    """Stage complete files, then publish; default publication cannot clobber a file."""
    staged, created = [], []
    try:
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        for path, content in zip(paths, contents):
            # O_EXCL retains temporary-file safety; mode 0666 lets the caller's
            # umask decide readability, just like a normal output file.
            tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            staged.append(tmp)
            with os.fdopen(fd, 'w') as fh:
                fh.write(content)
        for path, tmp in zip(paths, staged):
            if force:
                os.replace(tmp, path)
            else:
                os.link(tmp, path)  # fails atomically if another run created the path
                created.append(path)
    except OSError:
        for path in created:
            path.unlink()
        raise
    finally:
        for tmp in staged:
            tmp.unlink(missing_ok=True)


def main():
    ap = parser()
    args = ap.parse_args()
    try:
        score_bounds(args.params)
    except ValueError as e:
        ap.error(str(e))
    if bool(args.lattice) != bool(args.qband):
        ap.error('--lattice and --qband go together')
    if args.points < 16000 or args.eval_points < 4 * args.points:
        ap.error('need --points >= 16000 and --eval-points >= 4 * --points')
    if args.top < 0 or args.threads < 1 or args.max_evals < 32 or args.trans < 0:
        ap.error('need top/trans >= 0, threads >= 1 and max-evals >= 32')
    if not math.isfinite(args.min_gain) or args.min_gain < 0:
        ap.error('--min-gain must be finite and nonnegative')
    if not (0 <= args.seed < 2**64 and 0 <= args.eval_seed < 2**64):
        ap.error('seeds must be unsigned 64-bit integers')
    if args.lattice and (args.nlat < 32 or args.npts < 2048 or args.eval_nlat < args.nlat or
                         args.eval_npts < args.npts or args.seed == args.eval_seed):
        ap.error('lattice search needs --nlat >= 32 and --npts >= 2048; validation needs an independent seed '
                 'and at least as many samples as the search')
    paths = [Path(args.out + s) for s in ('.poly', '.selected.poly', '.json')]
    if Path(args.file).resolve() in [p.resolve() for p in paths]:
        ap.error('output would overwrite the input')
    existing = [str(p) for p in paths if p.exists() or p.is_symlink()]
    if existing and not args.force:
        ap.error('outputs already exist (use a new --out or --force): ' + ', '.join(existing))
    blocks = read_cado_blocks(args.file)
    if args.top:
        blocks = blocks[:args.top]
    print(f'Refining {len(blocks)} finalists on {args.threads} CPU thread(s); '
          f'objective {"lattice band" if args.lattice else "MurphyE"}', flush=True)
    started = time.time_ns()
    try:
        results = refine([p for _, p in blocks], args)
    except (ValueError, RuntimeError) as e:
        ap.error(str(e))
    text, selected = [], []
    for i, ((label, _), r) in enumerate(zip(blocks, results)):
        r['label'] = label
        r['input_index'] = i + 1
        if r['status'] != 'ok':
            print(f'{i + 1:3d} FAILED: {r["error"]}: {label}', file=sys.stderr)
            continue
        for kind in ('control', 'proposal'):
            text.append(f'# input {i + 1} {kind}: {label}\n' + cado_poly_text(r[kind]) + '\n')
        kind = 'proposal' if r['accepted'] else 'control'
        selected.append(f'# input {i + 1} {kind}: {label}\n' + cado_poly_text(r[kind]) + '\n')
        print(f'{i + 1:3d} validation {100*r["validation_gain"]:+.4f}%  dt {r["translation"]}  '
              f'{r["evaluations"]} evaluations{" (limit)" if r["evaluation_limit"] else ""}; keep {kind}: {label}')
    failed = sum(r['status'] != 'ok' for r in results)
    report = dict(settings=vars(args), elapsed_seconds=(time.time_ns() - started) / 1e9,
                  elapsed_clock='CLOCK_REALTIME (time.time_ns)', failed=failed, results=results)
    try:
        write_outputs(paths, [''.join(text), ''.join(selected), json.dumps(report, indent=2, allow_nan=False) + '\n'],
                      force=args.force)
    except OSError as e:
        ap.error(str(e))
    print(f'Wrote controls/proposals: {paths[0]}; selected: {paths[1]}; measurements: {paths[2]}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
