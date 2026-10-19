#!/usr/bin/env python3
"""CPU-only replay audit of the opt-in intermediate alpha screen.

Requires the saved C208 seed-10 65k pool and C171 depth pools. Freeze the
current CPU binary, compare three policies with alternating timing order,
then audit every C171 depth pool without tuning the bound on those seeds.
All exports are checked by exact resultant. Replays cannot test block losses
or measure end-to-end GPU throughput. Use a fresh --run-dir for a new build.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import read_cado_blocks, resultant


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cells(path):
    rows = read_cado_blocks(path)
    for _, p in rows:
        r = resultant(p)
        if not r or r % p['n']:
            raise ValueError(f'invalid resultant: {path}')
    return [tuple(map(int, re.search(r'u\s+(-?\d+)\s+v\s+(-?\d+)', label).groups()))
            for label, _ in rows]


def comparison(control, trial):
    present = set(trial)
    return dict(best_preserved=bool(control and control[0] in present),
                retained={str(n): sum(c in present for c in control[:n])
                          for n in (8, 32, 200)}, control_count=len(control), trial_count=len(trial))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, default=ROOT / 'stage23_gpu/runs/20261016_alpha_screen/replay')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--exact-size', action='store_true', help='also audit the opt-in exact-size screen')
    ap.add_argument('--control-only', action='store_true', help='time only the C208 control policies')
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error('--repeats must be positive')
    dest = args.run_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    binary = dest / 's23_ropt_cpu'
    if not binary.exists():
        shutil.copy2(ROOT / 'stage23_gpu/build/s23_ropt_cpu', binary)
    binary_hash = digest(binary)
    records = []

    def execute(tag, source, pool, opts):
        out = dest / (tag + '.poly')
        cmd = [str(binary), '-poly', str(source), '-search', '-pool-in', str(pool),
               '-aw', '1.3', '-refine', '200', '-top', '8', '-t', '2',
               *map(str, opts), '-out', str(out)]
        start = time.monotonic()
        with (dest / (tag + '.log')).open('w') as log:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record = dict(tag=tag, command=cmd, seconds=time.monotonic()-start,
                      returncode=proc.returncode, binary_sha256=binary_hash,
                      input_sha256=digest(source), pool_sha256=digest(pool))
        if proc.returncode:
            raise RuntimeError(f'{tag} failed; see log')
        selected = cells(out)
        log = (dest / (tag + '.log')).read_text()
        record['refine_seconds'] = float(re.search(r', refine ([0-9.]+) s', log)[1])
        screen = re.search(r'# alpha-screen: .*?, ([0-9.]+) s', log)
        record['screen_seconds'] = float(screen[1]) if screen else 0
        record['verified_exports'] = len(selected)
        record['trace'] = [line for line in log.splitlines() if 'trace-cell' in line]
        record['output_sha256'] = digest(out)
        (dest / (tag + '.json')).write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        return selected

    source = ROOT / 'stage23_bench/data/c208/ropt_coverage/s0010.poly'
    pool = ROOT / 'stage23_gpu/runs/20261016_loss_trace/wide_pool/seed10_pool65536.pool'
    policies = [('narrow', ['-rerank', 4096]), ('wide', ['-rerank', 32768]),
                ('screen', ['-rerank', 4096, '-alpha-screen', 32768, 500])]
    if args.exact_size:
        policies.append(('screen_exact', ['-rerank', 4096, '-alpha-screen', 32768, 500,
                                         '-alpha-screen-exact-size']))
    selected = {}
    for repeat in range(args.repeats):
        offset = repeat % len(policies)
        for name, opts in policies[offset:] + policies[:offset]:
            tag = f'c208_{name}_{repeat}'
            result = execute(tag, source, pool, opts + ['-trace-cell', 7, -174864538])
            if name in selected and result != selected[name]:
                raise ValueError(f'nondeterministic output: {tag}')
            selected[name] = result
            print(tag, records[-1]['refine_seconds'], flush=True)
    control = dict(screen=comparison(selected['wide'], selected['screen']),
                   narrow=comparison(selected['wide'], selected['narrow']),
                   reference_retained={name: (7, -174864538) in value for name, value in selected.items()},
                   median_refine_seconds={name: statistics.median(r['refine_seconds'] for r in records
                                                                  if r['tag'].rsplit('_', 1)[0] == f'c208_{name}')
                                          for name, _ in policies})
    if args.exact_size:
        control['screen_exact'] = comparison(selected['wide'], selected['screen_exact'])
    independent = []
    c171 = ROOT / 'stage23_gpu/runs/20261016_c171'
    for pool in ([] if args.control_only else sorted((c171 / 'depth').glob('*.pool'))):
        source = c171 / 'seeds' / (pool.stem + '.poly')
        baseline = execute(f'c171_{pool.stem}_wide', source, pool, ['-rerank', 4096])
        screened = execute(f'c171_{pool.stem}_screen', source, pool,
                           ['-rerank', 512, '-alpha-screen', 4096, 500])
        record = dict(seed=pool.stem, **comparison(baseline, screened))
        if args.exact_size:
            exact = execute(f'c171_{pool.stem}_screen_exact', source, pool,
                            ['-rerank', 512, '-alpha-screen', 4096, 500, '-alpha-screen-exact-size'])
            record['exact_size'] = comparison(baseline, exact)
        independent.append(record)
        print(pool.stem, independent[-1], flush=True)
    summary = dict(c208=control, independent_c171=independent, records=records,
                   limitations='CPU pool replay only. C171 uses 4096->512 because saved pools contain 4096; '
                               'this is a retention stress test, not a new wide search. No GPU sieve timing '
                               'or MurphyE/test-sieve comparison. The alpha-screen bound was chosen on C208 seed 10.')
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(control, indent=2))


if __name__ == '__main__':
    main()
