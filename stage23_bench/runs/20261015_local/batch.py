#!/usr/bin/env python3
"""Freeze C147/C166, run GPU preprocessing, then optimize jobs with scoring.json.

Usage: python3 stage23_bench/runs/20261015_local/batch.py [--prepare-only]
Scoring files in each run contain params and optionally lattice and qband.
Rerunning resumes verified checkpoints. No scoring bounds are assumed.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'stage23_gpu/runs/20261015_local'
sys.path.insert(0, str(ROOT / 'scripts'))
from gpu_pipeline import Run, digest, write_json

JOBS = [('c147', ROOT), ('c166', ROOT / 'backup_20261015_143047')]


def prepare():
    for name, source in JOBS:
        run = BASE / name
        marker = run / 'prepared.json'
        if marker.exists():
            continue
        run.mkdir(parents=True, exist_ok=True)
        cpu = run / 'cpu_reference'
        cpu.mkdir(exist_ok=True)
        config = (source / 'pipeline_results/pipeline_settings.txt').read_text()
        if 'STATUS=complete\n' not in config:
            raise RuntimeError(f'{name}: CPU pipeline incomplete')
        files = []
        for src in sorted((source / 'pipeline_results').iterdir()):
            if src.is_file():
                dst = cpu / src.name
                shutil.copy2(src, dst)
                files.append(dst)
        # Preserve CPU effort-0 pairs as well as final root-optimization outputs.
        cpu_e0 = cpu / 'cado_sopt_unsorted.txt'
        shutil.copy2(source / 'cado_sopt_unsorted.txt', cpu_e0)
        files.append(cpu_e0)
        workflow = Run(run)
        (run / 'preprocess').mkdir(exist_ok=True)
        identity, frozen = workflow.snapshot(source / 'msieve.dat.deduped.ms', 'raw')
        for binary in ('s23_sopt', 's23_ropt'):
            workflow.freeze(binary, ROOT / 'stage23_gpu/build' / binary)
        n, count = None, 0
        with frozen.open() as stream:
            for line in stream:
                if line.startswith('n:'):
                    value = int(line.split(':', 1)[1])
                    if n is not None and value != n:
                        raise RuntimeError(f'{name}: mixed input n')
                    n, count = value, count + 1
        if len(str(n)) != int(name[1:]) or count == 0:
            raise RuntimeError(f'{name}: unexpected input')
        write_json(marker, dict(n=n, raw_polynomials=count, source=str(source),
            input=identity, cpu_sha256={str(p.relative_to(run)): digest(p) for p in files},
            top_n=int(next(line.split('=')[1] for line in config.splitlines()
                          if line.startswith('TOP_N_EXTRACT=')))))
        print(f'Prepared {name}: {count} raw polynomials', flush=True)


def command(name, stage, options):
    run = BASE / name
    write_json(BASE / 'status.json', dict(stage=stage, job=name, updated=time.time()))
    argv = [sys.executable, str(ROOT / 'scripts/gpu_pipeline.py'), stage,
            '--run-dir', str(run), '--threads', '8', '--workers', '2', '--launch', '30',
            *options]
    with (run / f'{stage}_driver.log').open('a') as log:
        result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name}: {stage} exited {result.returncode}; see {run}/{stage}_driver.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    prepare()
    if args.prepare_only:
        return
    pending = []
    for name, _ in JOBS:
        run = BASE / name
        command(name, 'preprocess', [])
        scoring = run / 'scoring.json'
        if not scoring.exists():
            pending.append(name)
            print(f'{name}: preprocessing complete; awaiting sieve settings in {scoring}', flush=True)
            continue
        values = json.loads(scoring.read_text())
        settings = json.loads((run / 'prepared.json').read_text())
        options = ['--params', values['params'], '--extract', str(settings['top_n'])]
        for flag in ('lattice', 'qband'):
            if values.get(flag):
                options += ['--' + flag, values[flag]]
        command(name, 'pipeline', options)
        print(f'{name}: GPU pipeline complete', flush=True)
    write_json(BASE / 'status.json', dict(stage='awaiting sieve settings' if pending else 'complete',
                                         pending=pending, updated=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write_json(BASE / 'status.json', dict(stage='failed', error=str(error), updated=time.time()))
        raise
