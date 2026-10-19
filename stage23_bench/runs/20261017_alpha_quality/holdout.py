#!/usr/bin/env python3
"""Replay a fixed exact-size screening policy on eight unused C147/C166 pools.

Choose the first four depth-pool filenames per job before inspecting outcomes.
Use their archived model bounds for the subsequent quality audit. These jobs'
sieve settings are inherited model settings, not confirmed test-sieve settings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import read_cado_blocks, resultant


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, default=ROOT / 'stage23_gpu/runs/20261017_alpha_quality/holdout_replay')
    args = ap.parse_args()
    dest = args.run_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    binary = dest / 's23_ropt_cpu'
    if not binary.exists():
        shutil.copy2(ROOT / 'stage23_gpu/build/s23_ropt_cpu', binary)
    cases = []
    for family in ('c147', 'c166'):
        root = ROOT / 'stage23_gpu/runs/20261015_local' / family
        pools = sorted((root / 'depth').glob('*.pool'))[:4]
        if len(pools) != 4:
            raise ValueError(f'not enough pools: {root}')
        for pool in pools:
            source = root / 'seeds' / (pool.stem + '.poly')
            cases.append(dict(case=f'{family}_{pool.stem}', source=str(source), pool=str(pool),
                              input_sha256=digest(source), pool_sha256=digest(pool),
                              config=json.loads((root / 'scoring.json').read_text())))
    (dest / 'selection.json').write_text(json.dumps(dict(cases=cases,
        policy='screen 4096 through prime 500 with exact size, fully refine 512, export 200; no retuning',
        binary_sha256=digest(binary)), indent=2) + '\n')
    records = []
    for case in cases:
        for policy, opts in [('wide', ['-rerank', '4096']),
                             ('screen', ['-rerank', '512', '-alpha-screen', '4096', '500', '-alpha-screen-exact-size'])]:
            tag = case['case'] + '_' + policy
            out = dest / (tag + '.poly')
            cmd = [str(binary), '-poly', case['source'], '-search', '-pool-in', case['pool'],
                   '-aw', '1.3', '-refine', '200', '-top', '8', '-t', '2', *opts, '-out', str(out)]
            start = time.monotonic()
            with (dest / (tag + '.log')).open('w') as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = read_cado_blocks(out)
            if len(rows) != 200 or any(not resultant(p) or resultant(p) % p['n'] for _, p in rows):
                raise ValueError(f'invalid exports: {tag}')
            record = dict(tag=tag, command=cmd, seconds=time.monotonic()-start,
                          output_sha256=digest(out), verified_exports=len(rows))
            (dest / (tag + '.json')).write_text(json.dumps(record, indent=2) + '\n')
            records.append(record)
            print(tag, round(record['seconds'], 3), flush=True)
    (dest / 'summary.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
