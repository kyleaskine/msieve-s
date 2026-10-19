#!/usr/bin/env python3
"""Reproduce the C208 cutoff audit; --gpu runs four bounded, sequential searches.

All commands, binaries, hashes, logs, pools and exports stay in --run-dir.
The GPU may be shared: these measurements diagnose retention, not throughput.
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, default=ROOT / 'stage23_gpu/runs/20261016_loss_trace')
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--wide-pool', action='store_true', help='follow-up: seed 10 with smaller blocks and larger proxy pools')
    args = ap.parse_args()
    dest = args.run_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    bindir = dest / 'trace_bin'
    bindir.mkdir(exist_ok=True)
    name = 's23_ropt' if args.gpu else 's23_ropt_cpu'
    binary = bindir / name
    if not binary.exists():
        shutil.copy2(ROOT / 'stage23_gpu/build' / name, binary)
    fixtures = ROOT / 'stage23_bench/data/c208/ropt_coverage'
    manifest = json.loads((fixtures / 'manifest.json').read_text())
    cases = [('seed10_k16', 10, 16, 20, 16, 4096), ('seed10_seg18', 10, 16, 18, 8, 4096),
             ('seed10_seg16', 10, 16, 16, 8, 4096), ('seed103_classes20', 103, 20, 20, 8, 4096)]
    if args.wide_pool:
        cases = [(f'seed10_pool{pool}', 10, 16, 16, 8, pool) for pool in (16384, 65536)]
    records = []
    for tag, seed, count, seg, k, pool in cases:
        entry = next(x for x in manifest if x['seed'] == seed)
        source = fixtures / entry['file']
        if digest(source) != entry['sha256']:
            raise ValueError(f'fixture hash mismatch: {source}')
        ref = entry['reference_rotation']
        argv = [str(binary), '-poly', str(source), '-search', '-budget', '3', '-maxcells', '2e11',
                '-maxlines', '1000', '-umax', '20000', '-classes', '2520', str(count),
                '-class-explore', '4', '-aw', '1.3', '-rerank', str(pool), '-refine', '200',
                '-top', '8', '-seg', str(seg), '-k', str(k), '-t', '2', '-launch', '30',
                '-trace-cell', str(ref['u']), str(ref['v'])]
        if args.gpu:
            argv += ['-check', '128', '-pool-out', str(dest / (tag + '.pool')),
                     '-out', str(dest / (tag + '.poly'))]
        else:
            tag += '_plan'
            argv += ['-plan']
        start = time.monotonic()
        with (dest / (tag + '.log')).open('w') as log:
            proc = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        log = (dest / (tag + '.log')).read_text()
        record = dict(tag=tag, command=argv, returncode=proc.returncode,
                      seconds=time.monotonic() - start, binary_sha256=digest(binary),
                      input_sha256=digest(source), trace=[s for s in log.splitlines() if 'trace-cell' in s])
        (dest / (tag + '.json')).write_text(json.dumps(record, indent=2) + '\n')
        if proc.returncode:
            raise RuntimeError(f'{tag}: exit {proc.returncode}; see saved log')
        if args.gpu:
            polys = read_cado_blocks(dest / (tag + '.poly'))
            for _, poly in polys:
                r = resultant(poly)
                if not r or r % poly['n']:
                    raise ValueError(f'{tag}: invalid resultant')
            if 'vs brute force: 0 differ' not in log:
                raise ValueError(f'{tag}: missing successful brute-root check')
            record['verified_exports'] = len(polys)
        records.append(record)
        print(tag, record['trace'][-1], flush=True)
    filename = 'gpu_experiments.json' if args.gpu else 'plan_experiments.json'
    (dest / filename).write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
