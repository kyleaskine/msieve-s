#!/usr/bin/env python3
"""verify_sopt.py RUN CPU_E0_OUTPUT: streamed full-output comparison, matched in raw-input order.

Compare exact coefficients; independently score differing outputs with CADO's
full-precision exp_E. A mismatched input/order/count aborts instead of misjoining.
"""
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import cado_expe, coeffs, iter_sopt, resultant


def key(p):
    return p['n'], p['Y0'], p['Y1'], *coeffs(p)


def main():
    run, cpu = Path(sys.argv[1]), Path(sys.argv[2])
    settings = json.loads((run / 'settings.json').read_text())
    differences = []
    start = time.time()
    count = 0
    for count, pair in enumerate(itertools.zip_longest(iter_sopt(cpu), iter_sopt(run / 'sopt/e0_unsorted.txt')), 1):
        a, b = pair
        if a is None or b is None or key(a[0]) != key(b[0]) or a[0]['n'] != settings['n']:
            raise RuntimeError(f'input/order/count mismatch at polynomial {count}')
        if key(a[1]) != key(b[1]):
            if not resultant(b[1]) or resultant(b[1]) % settings['n']:
                raise RuntimeError(f'invalid GPU polynomial {count}')
            differences.append(dict(index=count, cpu=a[1], gpu=b[1]))
        if count % 100000 == 0:
            print(count, 'checked;', len(differences), 'differ', flush=True)
    if count != settings['raw_polynomials']:
        raise RuntimeError(f'expected {settings["raw_polynomials"]} outputs; got {count}')
    scores = cado_expe([p for row in differences for p in (row['cpu'], row['gpu'])], settings['n'])
    worse = 0
    for i, row in enumerate(differences):
        a, b = scores[2*i:2*i+2]
        row.update(cpu_exp_E=a['exp_E'], gpu_exp_E=b['exp_E'], gain=a['exp_E'] - b['exp_E'])
        worse += row['gain'] < -1e-6
    report = dict(polynomials=count, identical=count-len(differences), different=len(differences),
                  worse_at_full_precision=worse, tolerance=1e-6, differences=differences,
                  elapsed_seconds=time.time()-start, cpu_source=str(cpu.resolve()))
    (run / 'sopt/verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print({k:v for k,v in report.items() if k != 'differences'}, flush=True)
    return bool(worse)


if __name__ == '__main__':
    sys.exit(main())
