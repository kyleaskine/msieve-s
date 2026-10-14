#!/usr/bin/env python3
"""Freeze and process the unprocessed C187/C165 backups, sequentially.

Usage: batch.py --prepare-only | batch.py
All inputs and outputs live in stage23_gpu/runs/, outside disposable backups/build.
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
BASE = ROOT / 'stage23_gpu/runs/20261013_backups'
JOBS = [('c187', 'backup_20261013_052537', 'job:32,33,15,50e6', '15,16384', '50e6,230e6,7'),
        ('c165', 'backup_20261012_213413', 'job:30,31,14,30e6', '14,8192', '30e6,90e6,7')]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import cado_poly_text, read_cado_blocks, resultant


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare():
    for name, backup, params, lattice, qband in JOBS:
        run = BASE / name
        if (run / 'prepared.json').exists():
            continue
        src = ROOT / backup
        for folder in ('bin', 'cpu', 'cpu_reference', 'sopt', 'seeds', 'breadth/out'):
            (run / folder).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / 'msieve.dat.deduped.ms', run / 'input.ms')
        n, count = None, 0
        with (run / 'input.ms').open() as stream:
            for line in stream:
                if line.startswith('n:'):
                    value = int(line.split(':', 1)[1])
                    if n is not None and value != n:
                        raise RuntimeError('mixed input n')
                    n = value
                    count += 1
        for binary in ('s23_sopt', 's23_ropt'):
            shutil.copy2(ROOT / 'stage23_gpu/build' / binary, run / 'bin' / binary)
        for path in (src / 'pipeline_results').iterdir():
            if path.is_file():
                shutil.copy2(path, run / 'cpu_reference' / path.name)
        shutil.copy2(run / 'cpu_reference/best300_msieve.ms', run / 'cpu/best300_msieve.ms')
        config = (run / 'cpu_reference/pipeline_settings.txt').read_text()
        if 'STATUS=complete\n' not in config:
            raise RuntimeError('CPU reference incomplete')
        top_n = int(next(line.split('=')[1] for line in config.splitlines()
                         if line.startswith('TOP_N_EXTRACT=')))
        settings = dict(n=n, raw_polynomials=count, top_n=top_n, effort=50,
                        input_source=str(src / 'msieve.dat.deduped.ms'), input_sha256=sha(run / 'input.ms'),
                        cpu_source=str(run / 'cpu_reference'), backup=backup,
                        breadth=dict(workers=2, threads=8, launch=30, budget=3, maxcells=2e11, rerank=4096))
        (run / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
        (run / 'scoring.json').write_text(json.dumps(dict(params=params, lattice=lattice, qband=qband,
            source='User-provided sieve parameters, 2026-10-13',
            note='Model comparison; no test sieving performed.'), indent=2) + '\n')
        provenance = dict(backup=backup, n=n, raw_polynomials=count, sha256={
            str(path.relative_to(run)): sha(path) for path in
            [run / 'input.ms', *sorted((run / 'bin').iterdir()), *sorted((run / 'cpu_reference').iterdir())]})
        (run / 'prepared.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print('prepared', name, count, flush=True)


def archive(run, name):
    dest = ROOT / 'stage23_bench/data' / name
    (dest / 'gpu_ropt').mkdir(parents=True, exist_ok=True)
    (dest / 'winners').mkdir(exist_ok=True)
    settings = json.loads((run / 'settings.json').read_text())
    (dest / 'n.txt').write_text(str(settings['n']) + '\n')
    for filename in ('comparison.json', 'finalists.poly', 'manifest.json', 'settings.json',
                     'scoring.json', 'timings.json', 'prepared.json', 'comparison_commands.json',
                     'breadth_analysis.log', 'status.json', 'comparison_status.json'):
        shutil.copy2(run / filename, dest / 'gpu_ropt' / filename)
    rows = json.loads((run / 'comparison.json').read_text())['rows']
    polys = dict(read_cado_blocks(run / 'finalists.poly'))
    best = {}
    for row in rows:
        best.setdefault(row['label'].split()[0], row)
    for method, row in best.items():
        p = polys[row['label']]
        if p['n'] != settings['n'] or not resultant(p) or resultant(p) % settings['n']:
            raise RuntimeError('invalid archived winner')
        (dest / 'winners' / (method + '_best.poly')).write_text(
            '# ' + row['label'] + '\n' + cado_poly_text(p))
    summary = dict(run=str(run), best=best, scoring=json.loads((run / 'scoring.json').read_text()),
                   note='Model comparison with user-provided parameters; no test sieving performed.')
    (dest / 'gpu_ropt/summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    prepare()
    if args.prepare_only:
        return
    for name, *_ in JOBS:
        run = BASE / name
        print('starting', name, flush=True)
        status = run / 'status.json'
        if not status.exists() or not json.loads(status.read_text())['stage'].startswith('GPU search complete'):
            with (run / 'run.log').open('w') as log:
                subprocess.run([sys.executable, ROOT / 'stage23_bench/runs/20261012_c196/run.py', run],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        comparison = run / 'comparison_status.json'
        if not comparison.exists() or json.loads(comparison.read_text())['stage'] != 'comparison complete':
            with (run / 'comparison.log').open('w') as log:
                subprocess.run([sys.executable, ROOT / 'stage23_bench/runs/20261012_c196/compare.py', run],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        archive(run, name)
        print('complete and archived', name, flush=True)
    (BASE / 'complete.json').write_text(json.dumps(dict(completed=time.time(), jobs=[j[0] for j in JOBS])) + '\n')


if __name__ == '__main__':
    main()
