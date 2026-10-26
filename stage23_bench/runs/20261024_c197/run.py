#!/usr/bin/env python3
"""C197 production comparison of opt-in full GPU refinement and Murphy quadrature.

Initial CPU size optimization is imported; the subsequent GPU seed optimization,
breadth/depth searches and full-pool refinement are independent of CPU winners.
All outputs stay under a named run directory. Scoring settings live in scoring.json.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
import gpu_pipeline as driver
from polyfmt import parse_cado_ropt, read_cado_blocks, cado_poly_text


def prepare(run):
    marker = run / 'prepared.json'
    if marker.exists(): return json.loads(marker.read_text())
    if run.exists(): raise ValueError('choose a fresh run directory for preparation')
    run.mkdir(parents=True)
    cpu = run / 'cpu_reference'
    shutil.copytree(ROOT / 'pipeline_results', cpu)
    shutil.copytree(ROOT / 'pipeline_work', cpu / 'pipeline_work')
    shutil.copy2(ROOT / 'nfs_config.ini', cpu / 'nfs_config.ini')
    shutil.copy2(ROOT / 'worktodo.ini', cpu / 'worktodo.ini')
    n = int((cpu / 'worktodo.ini').read_text().strip())
    if len(str(n)) != 197: raise ValueError('expected new C197')
    if 'STATUS=complete' not in (cpu / 'pipeline_settings.txt').read_text(): raise ValueError('CPU run incomplete')
    refs = []
    for side in ('orig', 'inv'):
        for i, p in enumerate(parse_cado_ropt(cpu / f'cado_ropt_{side}.txt')):
            driver.validate_poly(p, n)
            refs.append(dict(method=f'cado_{side}', index=i, poly=p))
    driver.write_json(cpu / 'cado_references.json', refs)
    workflow = driver.Run(run)
    for binary in ('s23_sopt', 's23_ropt', 's23_ropt_cpu'):
        workflow.freeze(binary, ROOT / 'stage23_gpu/build' / binary)
    workflow.freeze('cado_murphy')
    meta = dict(n=n, digits=197, initial_mode='import completed CPU initial sopt',
                cpu_sha256={str(p.relative_to(run)): driver.digest(p) for p in cpu.rglob('*') if p.is_file()},
                initial_sopt_source=str(ROOT / 'cado_sopt_unsorted.txt'),
                scope='GPU resopt/breadth/depth/final refinement after shared initial CPU sopt; no test sieving')
    driver.write_json(marker, meta)
    driver.write_json(run / 'scoring.json', dict(params='job:32,33,15.5,75e6',
        lattice='16,16384', qband='75e6,400e6,7', lpbr=32, lpba=33,
        rlim=160000000, alim=240000000, logI=16, J=16384, sieve_points=2**30,
        sq_side='algebraic', provisional=True, source='User, 2026-10-24',
        note='cuda-sieve logI=16, J=16384; Murphy equivalent I=15.5. Limits recorded for future test sieving.'))
    print('Prepared C197; frozen CPU reference and executables', flush=True)
    return meta


def base_options(run):
    return ['--run-dir', str(run), '--extract', '1000', '--resopt-effort', '50', '--seeds', '300',
            '--threads', '2', '--workers', '2', '--launch', '30', '--ropt-batch-size', '16',
            '--refine-gpu', '--murphy-gpu']


def execute(run, command, label):
    driver.write_json(run / (label + '_command.json'), dict(argv=list(map(str, command)), started=time.time()))
    start = time.monotonic()
    with (run / (label + '.log')).open('ab') as log:
        p = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    driver.write_json(run / (label + '_execution.json'), dict(returncode=p.returncode, wall_seconds=time.monotonic()-start))
    if p.returncode: raise RuntimeError(f'{label} failed: {p.returncode}; see {run / (label + ".log")}')


def preprocess(run):
    execute(run, [sys.executable, ROOT / 'scripts/gpu_pipeline.py', 'preprocess', *base_options(run),
                  '--initial-unsorted', ROOT / 'cado_sopt_unsorted.txt'], 'preprocess')


def breadth(run):
    # This phase is independent of sieve settings and can run while those are
    # being confirmed. The regular pipeline later reuses the same checkpoints.
    args = driver.parser(['preprocess', *base_options(run)])
    with (run / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        workflow = driver.Run(run)
        info = driver.existing_preprocess(workflow)
        try:
            seeds = driver.select_seeds(workflow, args, info, workflow.freeze('s23_sopt', args.sopt_binary))
            driver.search(workflow, args, seeds, workflow.freeze('s23_ropt', args.ropt_binary),
                          'breadth', args.maxcells, info['n'], save_pool=True)
            workflow.state('breadth complete; ranking pending', searches=len(seeds))
        except BaseException:
            workflow.stop.set()
            raise


def pipeline(run):
    scoring = json.loads((run / 'scoring.json').read_text())
    execute(run, [sys.executable, ROOT / 'scripts/gpu_pipeline.py', 'pipeline', *base_options(run),
                  '--params', scoring['params'], '--lattice', scoring['lattice'], '--qband', scoring['qband']], 'pipeline')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=('prepare', 'preprocess', 'breadth', 'pipeline'))
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--wait-preprocess', action='store_true', help='pipeline: wait for the already-running import')
    args = ap.parse_args()
    run = args.run_dir.resolve()
    if args.phase == 'prepare': prepare(run)
    else:
        if not (run / 'prepared.json').exists(): raise ValueError('prepare first')
        if args.wait_preprocess:
            if args.phase != 'pipeline': ap.error('--wait-preprocess requires pipeline')
            status = run / 'preprocess_execution.json'
            deadline = time.monotonic() + 1800
            while not status.exists():
                if time.monotonic() > deadline: raise TimeoutError('initial import did not complete')
                time.sleep(2)
            if json.loads(status.read_text())['returncode']:
                raise RuntimeError('initial import failed; inspect preprocess.log')
        globals()[args.phase](run)


if __name__ == '__main__': main()
