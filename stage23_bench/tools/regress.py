#!/usr/bin/env python3
"""Regression checks for the bench tools; run by stage23_gpu's `make test`.

  regress.py

Each check pins a property or a failure seen before (GPU_STAGE23_PLAN.md, "Review
checks" and "Fixes"):

  scorer      cado_murphy equals CADO's MurphyE() at K = 1000 and K = 4000 (-selftest)
              on the c168 winners, and its K = 1000 values stay where they were.
  quadrature  the c168 trap: the translation-tuned winner reads +0.8% over msieve's at
              CADO's 1,000 sample points and is no better at 4,000/16,000.
  content     content_seeds.py: GPT's counterexample (Y1 = 2 is not invertible mod 2,
              but (f + g)/2 is a valid seed), and five c168 seeds whose derived seeds,
              written by the original script, must come out exactly; for each derived
              seed Res(f', g) * d = Res(f, g).

Exits 1 if any check fails.
"""

import os
import subprocess
import sys
import tempfile

from content_seeds import content_seeds
from polyfmt import cado_murphy_binary, cado_poly_text, parse_cado_poly, resultant

HERE = os.path.dirname(os.path.abspath(__file__))
C168 = os.path.join(HERE, '..', 'data', 'c168')
JOB = ['-Bf', '2147483648', '-Bg', '1073741824', '-area', '3355443200000000']  # lpb 31/30, I 14, qmin 25M
failures = []


def check(name, ok, detail):
    print(f'{"ok  " if ok else "FAIL"} {name}: {detail}')
    if not ok:
        failures.append(name)


def murphy(paths, *opts):
    with tempfile.NamedTemporaryFile('w', suffix='.poly', delete=False) as fh:
        for p in paths:
            fh.write(open(p).read().strip() + '\n\n')
        tmp = fh.name
    try:
        out = subprocess.run([cado_murphy_binary(), *JOB, '-t', '2', *opts, tmp], capture_output=True, text=True)
    finally:
        os.remove(tmp)
    return out


def main():
    winners = [os.path.join(C168, 'winners', f) for f in ('msieve_best.poly', 'gpu_tuned_best.poly', 'cado_best.poly')]

    # scorer: CADO's own MurphyE at two sample counts, and the K = 1000 values pinned
    for k in ('1000', '4000'):
        r = murphy(winners, '-K', k, '-selftest')
        check(f'scorer selftest K={k}', r.returncode == 0 and ' 0 differ' in r.stdout, r.stdout.strip())
    e1000 = [float(l.split('\t')[1]) for l in murphy(winners).stdout.split('\n') if l]
    pinned = [4.560404e-08, 4.597916e-08, 4.550356e-08]
    check('scorer K=1000 values', len(e1000) == 3 and all(abs(a / b - 1) < 1e-6 for a, b in zip(e1000, pinned)),
          f'{e1000} (pinned {pinned})')

    # quadrature: the apparent +0.8% of the tuned translation is sampling noise
    e_acc = [float(l.split('\t')[1]) for l in murphy(winners[:2], '-K', '4000', '-Keval', '16000').stdout.split('\n') if l]
    check('quadrature trap at K=1000', e1000[1] / e1000[0] > 1.005,
          f'tuned / msieve {e1000[1] / e1000[0]:.4f} at 1,000 points')
    check('quadrature trap at K=4000/16000', len(e_acc) == 2 and e_acc[1] / e_acc[0] < 1.0002,
          f'tuned / msieve {e_acc[1] / e_acc[0]:.5f} at 4,000/16,000 points' if len(e_acc) == 2 else 'no output')

    # content: the counterexample (Res(f, g) = 2 N with N = 77)
    toy = {'n': 77, 'Y0': 1, 'Y1': 2, 'c0': 5, 'c1': 0, 'c2': 0, 'c3': 2, 'c4': 2, 'c5': 2}
    got = [(d, {k: q[k] for k in ('c0', 'c1', 'c2', 'c3', 'c4', 'c5')}) for d, _, _, q in content_seeds(toy, 77)]
    want = [(2, {'c0': 3, 'c1': 1, 'c2': 0, 'c3': 1, 'c4': 1, 'c5': 1})]
    check('content: Y1 not invertible mod d', got == want, f'{got}')

    # content: c168 seeds against the original script's output, and Res identities
    cdir = os.path.join(C168, 'content')
    for seed in sorted(f for f in os.listdir(cdir) if f.endswith('.poly')):
        p = parse_cado_poly(os.path.join(cdir, seed))
        stem = seed[:-5]
        derived = content_seeds(p, p['n'])
        expected = sorted(f for f in os.listdir(os.path.join(cdir, 'expected')) if f.startswith(stem + '_'))
        ok = [f'{stem}_d{d}.poly' for d, _, _, _ in derived] == expected
        for d, _, _, q in derived:
            e = parse_cado_poly(os.path.join(cdir, 'expected', f'{stem}_d{d}.poly'))
            ok &= cado_poly_text(q) == cado_poly_text(e) and resultant(q) * d == resultant(p)
        check(f'content: {stem}', ok, f'd = {[d for d, _, _, _ in derived] or "none"}')

    print(f'{"all checks passed" if not failures else "FAILED: " + ", ".join(failures)}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
