#!/usr/bin/env python3
"""Freeze and run the October 17 C181, then compare its completed CPU results."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'stage23_gpu/runs/20261017_c181'
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from gpu_pipeline import Run, digest, write_json
from polyfmt import parse_cado_ropt


def prepare():
    marker = RUN / 'prepared.json'
    if marker.exists():
        return
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / 'preprocess').mkdir(exist_ok=True)
    workflow = Run(RUN)
    identity, snapshot = workflow.snapshot(ROOT / 'msieve.dat.ms', 'raw')
    with snapshot.open() as stream:
        n = int(stream.readline().split(':', 1)[1])
    if len(str(n)) != 181:
        raise RuntimeError('expected C181 input')
    cpu = ROOT / 'pipeline_results'
    config = (cpu / 'pipeline_settings.txt').read_text()
    if 'STATUS=complete\n' not in config:
        raise RuntimeError('CPU pipeline incomplete')
    for filename in ('cado_ropt_orig.txt', 'cado_ropt_inv.txt'):
        polys = parse_cado_ropt(cpu / filename)
        if not polys or any(p['n'] != n for p in polys):
            raise RuntimeError('CPU results do not match this C181')
    frozen_cpu = RUN / 'cpu_reference'
    frozen_cpu.mkdir(exist_ok=True)
    files = []
    for source in sorted(cpu.iterdir()):
        if source.is_file():
            target = frozen_cpu / source.name
            shutil.copy2(source, target)
            files.append(target)
    work = frozen_cpu / 'pipeline_work'
    work.mkdir(exist_ok=True)
    for filename in ('resopt_output.txt', 'resopt_msieve_sorted.ms', 'top1000_input.ms'):
        target = work / filename
        shutil.copy2(ROOT / 'pipeline_work' / filename, target)
        files.append(target)
    write_json(frozen_cpu / 'provenance.json', dict(n=n, source=str(cpu),
        sha256={str(p.relative_to(frozen_cpu)): digest(p) for p in files}))
    for binary in ('s23_sopt', 's23_ropt'):
        workflow.freeze(binary, ROOT / 'stage23_gpu/build' / binary)
    write_json(RUN / 'scoring.json', dict(params='job:31,32,15,40e6', lattice='15,16384',
        qband='40e6,140e6,7', lpbr=31, lpba=32, rlim=120000000, alim=180000000,
        region='15e', source='User-provided bounds, q range and 15e region, 2026-10-17.',
        note='Factor-base bounds retained for future test sieving; model uses LPB bounds, region and q range. No test sieving.'))
    write_json(marker, dict(n=n, input=identity, cpu_source=str(cpu),
        top_n=int(next(line.split('=')[1] for line in config.splitlines() if line.startswith('TOP_N_EXTRACT=')))))
    print('Frozen C181 input, CPU references and executables', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    prepare()
    if args.prepare_only:
        return
    settings = json.loads((RUN / 'prepared.json').read_text())
    write_json(RUN / 'batch_status.json', dict(stage='GPU optimization', updated=time.time()))
    argv = [sys.executable, ROOT / 'scripts/gpu_pipeline.py', 'all', '--input', ROOT / 'msieve.dat.ms',
        '--run-dir', RUN, '--extract', settings['top_n'], '--seeds', 300, '--threads', 8,
        '--workers', 2, '--launch', 30, '--params', 'job:31,32,15,40e6',
        '--lattice', '15,16384', '--qband', '40e6,140e6,7']
    with (RUN / 'driver.log').open('a') as log:
        subprocess.run(list(map(str, argv)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(RUN / 'batch_status.json', dict(stage='CPU/GPU comparison', updated=time.time()))
    spec = importlib.util.spec_from_file_location('local_compare', ROOT / 'stage23_bench/runs/20261015_local/compare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.compare(RUN)
    write_json(RUN / 'batch_status.json', dict(stage='complete', updated=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        RUN.mkdir(parents=True, exist_ok=True)
        write_json(RUN / 'batch_status.json', dict(stage='failed', error=str(error), updated=time.time()))
        raise
