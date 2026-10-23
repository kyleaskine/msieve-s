#!/usr/bin/env python3
"""Freeze October 22 C179/C171/C196, optimize on CUDA, compare CPU controls.

Run --prepare-only to snapshot all jobs, then run without flags to resume.
Full artifacts live outside rotating pipeline output and disposable build trees.
Scoring inherits the closest historical composite size, as requested by the user.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'stage23_gpu/runs/20261022_three_jobs'
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from gpu_pipeline import Run, digest, write_json
from polyfmt import parse_cado_ropt

JOBS = [
    ('c179', Path('/home/kylea/msieve-s-backup/backup_20261022_162248'),
     ROOT / 'stage23_gpu/runs/20261017_c181/scoring.json'),
    ('c171', Path('/home/kylea/msieve-s-backup/backup_20261022_180539'),
     ROOT / 'stage23_bench/data/c171/gpu_ropt/testsieve_user_20261016.json'),
    ('c196', ROOT, ROOT / 'stage23_gpu/build/runs_20261012/c196/scoring.json'),
]


def prepare():
    BASE.mkdir(parents=True, exist_ok=True)
    for name, source, historical in JOBS:
        run = BASE / name
        marker = run / 'prepared.json'
        if marker.exists():
            continue
        run.mkdir(exist_ok=True)
        (run / 'preprocess').mkdir(exist_ok=True)
        workflow = Run(run)
        identity, snapshot = workflow.snapshot(source / 'msieve.dat.deduped.ms', 'raw')
        with snapshot.open() as stream:
            n = int(next(line for line in stream if line.startswith('n:')).split(':')[1])
        if len(str(n)) != int(name[1:]):
            raise ValueError(f'{name}: wrong input composite')
        cpu = run / 'cpu_reference'
        cpu.mkdir(exist_ok=True)
        for src in sorted((source / 'pipeline_results').iterdir()):
            if src.is_file():
                shutil.copy2(src, cpu / src.name)
        # Freeze full effort-0 pairs for a streamed exact-output comparison.
        shutil.copy2(source / 'cado_sopt_unsorted.txt', cpu / 'cado_sopt_unsorted.txt')
        for side in ('orig', 'inv'):
            rows = parse_cado_ropt(cpu / f'cado_ropt_{side}.txt')
            if len(rows) != 150 or any(p['n'] != n for p in rows):
                raise ValueError(f'{name}: incomplete or mismatched CPU reference')
        report = (cpu / 'pipeline_report.txt').read_text()
        if 'Extracted top 1000' not in report or 'Re-ran sopt with effort 50' not in report:
            raise ValueError(f'{name}: unexpected CPU selection settings')
        for binary in ('s23_sopt', 's23_ropt'):
            workflow.freeze(binary, ROOT / 'stage23_gpu/build' / binary)
        workflow.freeze('cado_murphy')
        settings = json.loads(historical.read_text())
        if name == 'c171':
            # Latest actual user test settings supersede the earlier C168 proxy.
            old = settings['settings']
            settings = dict(params=f"job:{old['lpbr']},{old['lpba']},14,40e6",
                lattice='14,8192', qband='40e6,150e6,7', **old)
        settings.update(inherited_from=str(historical), inherited_sha256=digest(historical),
            selection_reason='Closest historical composite size; latest C171 test record supersedes its earlier proxy.',
            note='Inherited settings for model comparison; no new test sieving performed.')
        write_json(run / 'scoring.json', settings)
        write_json(marker, dict(n=n, source=str(source), input=identity, top_n=1000,
            cpu_sha256={p.name: digest(p) for p in cpu.iterdir() if p.is_file()}))
        print('Prepared', name, 'with scoring', settings['params'], flush=True)


def verify_sopt(run):
    """Reuse the exact ordered-pair audit with its expected historical layout."""
    audit = run / 'sopt_audit'
    (audit / 'sopt').mkdir(parents=True, exist_ok=True)
    info = json.loads((run / 'preprocess/info.json').read_text())
    write_json(audit / 'settings.json', dict(n=info['n'], raw_polynomials=info['count']))
    link = audit / 'sopt/e0_unsorted.txt'
    if not link.exists():
        link.symlink_to(run / 'preprocess/cado_sopt_unsorted.txt')
    if not (audit / 'sopt/verification.json').exists():
        subprocess.run([sys.executable, str(ROOT / 'stage23_bench/runs/20261012_c196/verify_sopt.py'),
            str(audit), str(run / 'cpu_reference/cado_sopt_unsorted.txt')], cwd=ROOT, check=True)
    result = json.loads((audit / 'sopt/verification.json').read_text())
    if result['worse_at_full_precision']:
        raise ValueError('GPU size optimization regressed against CPU')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--background', action='store_true',
                        help='resume in a detached process with a persistent batch log')
    args = parser.parse_args()
    if args.background:
        BASE.mkdir(parents=True, exist_ok=True)
        with (BASE / 'batch.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve())],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        write_json(BASE / 'background.json', dict(pid=process.pid, started=time.time()))
        print(f'Resuming in background: PID {process.pid}; log {BASE / "batch.log"}', flush=True)
        return
    prepare()
    if args.prepare_only:
        return
    spec = importlib.util.spec_from_file_location('cpu_compare',
        ROOT / 'stage23_bench/runs/20261015_local/compare.py')
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    for name, source, _ in JOBS:
        run = BASE / name
        scoring = json.loads((run / 'scoring.json').read_text())
        write_json(BASE / 'status.json', dict(stage='GPU pipeline', job=name, updated=time.time()))
        argv = [sys.executable, str(ROOT / 'scripts/gpu_pipeline.py'), 'all',
            '--input', str(source / 'msieve.dat.deduped.ms'), '--run-dir', str(run),
            '--extract', '1000', '--seeds', '300', '--threads', '8', '--workers', '2',
            '--launch', '30', '--ropt-batch-size', '16']
        for key in ('params', 'lattice', 'qband'):
            argv += ['--' + key, scoring[key]]
        write_json(run / 'command.json', dict(argv=argv))
        with (run / 'driver.log').open('a') as log:
            subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(BASE / 'status.json', dict(stage='size optimization audit', job=name, updated=time.time()))
        verify_sopt(run)
        write_json(BASE / 'status.json', dict(stage='CPU/GPU scoring', job=name, updated=time.time()))
        comparison.compare(run)
        print(name, 'complete', flush=True)
    write_json(BASE / 'status.json', dict(stage='complete', jobs=[j[0] for j in JOBS], updated=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write_json(BASE / 'status.json', dict(stage='failed', error=str(error), updated=time.time()))
        raise
