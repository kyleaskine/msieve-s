#!/usr/bin/env python3
"""Validate the reviewed GPU batch driver against C192 in a fresh directory."""
import argparse
from pathlib import Path
import shutil
import subprocess

from run import ROOT, Run, digest, driver_check, driver_screen_check, write_json
from gpu_pipeline import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=ROOT/'stage23_gpu/runs/20261019_c192')
    parser.add_argument('--run-dir', type=Path, required=True, help='new directory; existing paths are rejected')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    source, run = args.source_run.resolve(), args.run_dir.resolve()
    pools = sorted((source/'depth').glob('*.pool'))
    if len(pools) != 16:
        raise ValueError('need the completed 16-pool C192 depth run')
    files = [Path('fixtures')/name for name in ('seed.poly', 'seed_inv.poly', 'cado_winner.poly')]
    for pool in pools:
        files += [Path('depth')/pool.name, Path('depth')/(pool.stem+'.poly'), Path('seeds')/(pool.stem+'.poly')]
    hashes = {str(path): digest(source/path) for path in files}
    meta = read_json(source/'prepared.json')
    run.mkdir(parents=True, exist_ok=False)
    for path in files:
        (run/path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/path, run/path)
        if digest(run/path) != hashes[str(path)]:
            raise ValueError(f'source changed while copying {path}')
    write_json(run/'prepared.json', dict(n=meta['n'], source_run=str(source), inputs_sha256=hashes))
    workflow = Run(run)
    binaries = {}
    for name in ('s23_ropt', 's23_ropt_cpu'):
        binary = workflow.freeze(name, ROOT/'stage23_gpu/build'/name)
        capabilities = subprocess.check_output([str(binary), '-capabilities'], text=True).split()
        required = {'ropt-batch-events-v1'} | ({'alpha-screen-gpu-v1'} if name == 's23_ropt' else set())
        if required - set(capabilities):
            raise ValueError(f'{name} missing {sorted(required - set(capabilities))}')
        binaries[name] = dict(sha256=digest(binary), capabilities=capabilities)

    driver_check(run)
    driver_screen_check(run, args.workers)
    groups = {}
    for mode in ('cpu', 'gpu_checked', 'gpu_unchecked'):
        dest = run/('driver_screen_'+mode)
        sizes = [len(p.read_bytes().splitlines())-1 for p in (dest/'batches/screen').glob('*.tsv')]
        if sorted(sizes) != [16//args.workers]*args.workers:
            raise ValueError(f'{mode}: unexpected batch sizes {sizes}')
        groups[mode] = sorted(sizes)
        for pool in pools:
            text = (dest/'logs/screen'/(pool.stem+'.log')).read_text(encoding='utf-8', errors='replace')
            if mode != 'cpu' and ('# alpha-screen GPU:' not in text or 'using CPU screen' in text):
                raise ValueError(f'{mode}/{pool.stem}: GPU screen did not execute')
            if mode == 'gpu_checked' and '# alpha-screen GPU check:' not in text:
                raise ValueError(f'{pool.stem}: GPU oracle check missing')
    if hashes != {str(path): digest(source/path) for path in files}:
        raise ValueError('source artifacts changed during validation')
    write_json(run/'review_gpu.json', dict(source_run=str(source), run_dir=str(run), binaries=binaries,
        inputs_sha256=hashes, workers=args.workers, batch_sizes=groups,
        live_search=read_json(run/'driver_check.json'), screen=read_json(run/'driver_screen_check.json'),
        source_artifacts_unchanged=True, gpu_screen_executed=True,
        note='Correctness validation; timings include resume/repair/oracle checks and are not throughput benchmarks.'))
    print('PASS: GPU batching, resume/repair, 16-pool CPU/GPU screen parity, and worker distribution', flush=True)


if __name__ == '__main__':
    main()
