#!/usr/bin/env python3
"""Audit the production vector sieve, then alternate full C197 depth timings."""
import argparse
import hashlib
import json
import os
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


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def execute(command, log, env=None, timeout=1800):
    start = time.monotonic()
    with log.open('wb') as fh:
        p = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                           timeout=timeout, env=env)
    record = dict(command=list(map(str, command)), seconds=time.monotonic()-start,
                  returncode=p.returncode, log_sha256=digest(log))
    save(log.with_suffix('.json'), record)
    if p.returncode: raise RuntimeError(f'{log}: exit {p.returncode}')
    return record, log.read_text()


def checks(run):
    dest = run / 'validation'
    dest.mkdir(exist_ok=False)
    for name in ('s23_ropt', 's23_ropt_cpu', 'test_rsieve_gpu'):
        target = run / 'bin' / ('optimized' if name == 's23_ropt' else 'new_' + name)
        if target.exists(): raise ValueError(f'already frozen: {target}')
        shutil.copy2(ROOT / 'stage23_gpu/build' / name, target)
    provenance = {str(p.relative_to(run)): digest(p) for p in (run/'bin').iterdir() if p.is_file()}
    provenance.update({str(p.relative_to(ROOT)): digest(p) for p in
        [ROOT/'stage23_gpu/tools/s23_ropt.cu', ROOT/'stage23_gpu/tools/rsieve_cuda.h',
         ROOT/'stage23_gpu/tests/test_rsieve_gpu.cu']})
    save(run / 'optimized_provenance.json', provenance)
    fixtures = [ROOT / 'stage23_gpu/runs/20261024_c197_gpu/seeds/s0005.poly',
                ROOT / 'stage23_bench/data/c168/content/s0051.poly',
                ROOT / 'stage23_bench/data/c168/content/s0068.poly',
                ROOT / 'stage23_bench/data/c208/ropt_coverage/s0222.poly']
    record, text = execute([run/'bin/new_test_rsieve_gpu', *fixtures], dest/'exact_hits.log')
    count = int(re.search(r'(\d+) exact retained-hit comparisons passed', text)[1])
    print('Exact GPU/CPU retained hits:', count, flush=True)
    records = [record]
    env = dict(os.environ, S23_ROPT_BIN=str(run/'bin/optimized'))
    for test in ('test_ropt_class.py', 'test_ropt_classes.py', 'test_ropt_scan.py',
                 'test_ropt_trace.py', 'test_ropt_batch.py', 'test_refinement_gpu.py'):
        record, _ = execute([sys.executable, ROOT/'stage23_gpu/tests'/test], dest/(test+'.log'), env=env)
        records.append(record)
        print(test, 'passed', flush=True)
    save(dest/'summary.json', dict(exact_hit_comparisons=count, records=records))


def timings(run, repeats):
    if not (run/'validation/summary.json').exists(): raise ValueError('run checks first')
    dest = run / 'full_depth'
    dest.mkdir(exist_ok=False)
    source = ROOT / 'stage23_gpu/runs/20261024_c197_gpu'
    records = []
    # Two distinct production depth seeds, including the overall C197 winner.
    seeds = ('s0005', 's0001')
    controls = {seed: {ext: digest(source/'depth'/(seed+ext)) for ext in ('.poly', '.pool')} for seed in seeds}
    for repetition in range(repeats):
        for seed in seeds:
            modes = ('baseline', 'optimized') if (repetition + seeds.index(seed)) % 2 == 0 else ('optimized', 'baseline')
            for mode in modes:
                prefix = f'{repetition}_{seed}_{mode}'
                command = json.loads((source/'logs/depth'/(seed+'.json')).read_text())['command']
                command[0] = str(run/'bin'/('s23_ropt' if mode=='baseline' else 'optimized'))
                for option, extension in (('-out', '.poly'), ('-pool-out', '.pool')):
                    command[command.index(option)+1] = str(dest/(prefix+extension))
                record, text = execute(command, dest/(prefix+'.log'))
                for extension, expected in controls[seed].items():
                    if digest(dest/(prefix+extension)) != expected:
                        raise ValueError(f'{prefix}: differs from original production {extension}')
                for _, poly in read_cado_blocks(dest/(prefix+'.poly')):
                    res = resultant(poly)
                    if not res or res % poly['n']: raise ValueError('invalid resultant')
                match = re.search(r'search: ([\d.e+]+) cells, sieve ([\d.]+) s .*?size model ([\d.]+) s, refine ([\d.]+) s', text)
                if not match: raise ValueError('missing phase timing')
                cells, sieve, model, refine = map(float, match.groups())
                paths = re.search(r'root sieve GPU: (\d+) vector launches, (\d+) scalar launches', text)
                if mode == 'optimized' and (not paths or int(paths[1]) == 0): raise ValueError('vector kernel not exercised')
                record.update(seed=seed, mode=mode, repetition=repetition, cells=cells,
                    sieve_seconds=sieve, model_seconds=model, refine_seconds=refine,
                    vector_launches=int(paths[1]) if paths else 0, scalar_launches=int(paths[2]) if paths else None,
                    identical_production_pool=True, identical_production_exports=True)
                records.append(record)
                save(dest/'records.json', records)
                print(prefix, f'sieve {sieve:.2f}s, process {record["seconds"]:.3f}s; pool/exports identical', flush=True)
    summary = {}
    for seed in seeds:
        values = {mode: {key: statistics.median(r[key] for r in records if r['seed']==seed and r['mode']==mode)
                         for key in ('seconds', 'sieve_seconds')} for mode in ('baseline', 'optimized')}
        summary[seed] = dict(medians=values,
            sieve_speedup=values['baseline']['sieve_seconds']/values['optimized']['sieve_seconds'],
            process_speedup=values['baseline']['seconds']/values['optimized']['seconds'])
    save(dest/'summary.json', dict(repeats=repeats, seeds=summary, production_hashes=controls,
         note='Alternating full depth searches, one GPU process and two CPU threads. Both modes match '
              'the original production pools and exports exactly. Does not measure an entire pipeline.'))
    print(json.dumps(summary, indent=2), flush=True)


def sanitize(run):
    dest = run / 'sanitizer'
    dest.mkdir(exist_ok=False)
    fixture = ROOT / 'stage23_bench/data/c168/content/s0051.poly'
    for mode in ('memcheck', 'racecheck'):
        record, _ = execute(['compute-sanitizer', '--tool', mode, '--error-exitcode', '99',
            run/'bin/new_test_rsieve_gpu', fixture], dest/(mode+'.log'))
        print(mode, 'passed', flush=True)


def block_sizes(run):
    dest = run / 'block_sizes'
    dest.mkdir(exist_ok=False)
    fixture = ROOT / 'stage23_gpu/runs/20261024_c197_gpu/seeds/s0005.poly'
    rows = []
    for repetition in range(3):
        for seg in (16, 18, 20):
            controls = None
            modes = ('baseline', 'optimized') if repetition % 2 == 0 else ('optimized', 'baseline')
            for mode in modes:
                binary = run/'bin'/('s23_ropt' if mode=='baseline' else 'optimized')
                record, text = execute([binary, '-poly', fixture, '-u', 19, 19,
                    '-v', -8589934592, 8589934591, '-seg', seg, '-launch', 30, '-top', 128, '-t', 2],
                    dest/f'{repetition}_{seg}_{mode}.log')
                hits = '\n'.join(line for line in text.splitlines() if line.startswith('u '))
                if controls is not None and hits != controls: raise ValueError('block-size hit mismatch')
                controls = hits
                seconds = float(re.search(r', sieve ([\d.]+) s', text)[1])
                record.update(repetition=repetition, segment_log=seg, mode=mode, sieve_seconds=seconds)
                rows.append(record)
                save(dest/'records.json', rows)
                print('segment', seg, mode, seconds, flush=True)
    summary = {}
    for seg in (16, 18, 20):
        medians = {mode: statistics.median(r['sieve_seconds'] for r in rows if r['segment_log']==seg and r['mode']==mode)
                   for mode in ('baseline', 'optimized')}
        summary[seg] = dict(medians=medians, speedup=medians['baseline']/medians['optimized'])
    save(dest/'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('checks', 'timings', 'sanitize', 'block-sizes'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1: parser.error('positive repeats required')
    if args.phase == 'checks': checks(args.run_dir.resolve())
    elif args.phase == 'timings': timings(args.run_dir.resolve(), args.repeats)
    elif args.phase == 'sanitize': sanitize(args.run_dir.resolve())
    else: block_sizes(args.run_dir.resolve())


if __name__ == '__main__':
    main()
