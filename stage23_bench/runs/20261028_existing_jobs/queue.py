#!/usr/bin/env python3
"""Snapshot and process missed jobs; preserve status and continue after failures."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'stage23_gpu/runs/20261028_existing_jobs'
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'stage23_bench/runs/20261027_jobs'))
import gpu_pipeline as driver
import sieve_profiles
import jobs

SPECS = [
    ('c190', None, 'c192_provisional_20261019'),
    ('c181', 'backup_20261025_125205', 'c181_user_20261017'),
    ('c148', 'backup_20261025_160649', 'c146_user_20261003'),
    ('c179_576203', 'backup_20261026_115327', 'c181_user_20261017'),
    ('c179_358030', 'backup_20261028_114513', 'c181_user_20261017'),
    ('c164', 'stage23_gpu/runs/20261029_queue_sources/c164', 'c165_user_20261013'),
    ('c170_418691', 'backup_20261029_150315', 'c171_testsieve_20261016'),
    ('c152', 'stage23_gpu/runs/20261030_queue_sources/c152', 'c146_user_20261003'),
    ('c150', 'backup_20261030_121121', 'c146_user_20261003'),
]
C190_SOURCES = [ROOT / 'backup_20261028_160830/msieve.dat.deduped.ms',
                ROOT.parent / 'msieve-s-backup/backup_20261028_160837/msieve.dat.deduped.ms']


def combine_inputs(sources, combined, workflow):
    identities, n = [], None
    # The preprocess pass validates every polynomial/resultant. Here preserve
    # the frozen source bytes and reject mixed composites without doing those
    # multi-million expensive integer checks twice.
    with combined.open('wb') as out:
        for i, path in enumerate(sources):
            child = driver.Run(workflow.path / ('source_' + str(i)))
            identity, frozen = child.snapshot(path, 'raw')
            identities.append(identity)
            count = 0
            with frozen.open('rb') as source:
                for line in source:
                    match = re.fullmatch(rb'\s*n\s*:\s*(-?\d+)\s*', line)
                    if match:
                        value = int(match[1])
                        if n is None:
                            n = value
                        if value != n:
                            raise ValueError('C190 sources contain different composites')
                        count += 1
                    out.write(line)
            if not count:
                raise ValueError('empty C190 source: ' + str(path))
            out.write(b'\n')  # A missing terminal newline must not join two fields.
    return identities, n


def read_results():
    # Per-job results are authoritative even for jobs excluded by --only.
    return {name: driver.read_json(BASE / name / 'queue_result.json')
            for name, _, _ in SPECS if (BASE / name / 'queue_result.json').exists()}


def aggregate_stage(results):
    if any(r['stage'] != 'complete' for r in results.values()):
        return 'finished with errors'
    return 'complete' if len(results) == len(SPECS) else 'incomplete'


def background(only=None, wait_lock=False):
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / 'queue.log').open('ab') as log:
        command = [sys.executable, str(Path(__file__).resolve())]
        if only:
            command += ['--only', only]
        if wait_lock:
            command += ['--wait-lock']
        proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(f'background/{proc.pid}.json',
         dict(pid=proc.pid, started=time.time(), only=only, wait_lock=wait_lock))
    return proc.pid


def save(name, data):
    driver.write_json(BASE / name, data)


def prepare(name, source, profile):
    run = BASE / name
    if (run / 'prepared.json').exists():
        return run
    if source:
        jobs.prepare(run, ROOT / source, profile)
    else:
        run.mkdir(parents=True, exist_ok=True)
        workflow = driver.Run(run)
        combined = run / 'combined.ms'
        identities, n = combine_inputs(C190_SOURCES, combined, workflow)
        identity, _ = workflow.snapshot(combined, 'raw')
        for binary in ('s23_sopt', 's23_ropt', 's23_ropt_cpu'):
            workflow.freeze(binary, ROOT / 'stage23_gpu/build' / binary)
        workflow.freeze('cado_murphy')
        selected = next(p for p in sieve_profiles.load() if p['id'] == profile)
        driver.write_json(run / 'scoring.json', sieve_profiles.scoring(selected))
        driver.write_json(run / 'prepared.json', dict(n=n, digits=len(str(n)), input=identity,
            sources=identities, scope='Union of the two dated C190 backups; deduplicate before GPU sopt; no test sieve.'))
    return run


def main(only=None, wait_lock=False):
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / '.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait_lock else fcntl.LOCK_NB))
        save('owner.json', dict(pid=os.getpid(), acquired=time.time(), only=only))
        results = read_results()
        for name, source, profile in SPECS:
            if only and name not in only.split(','):
                continue
            previous = BASE / name / 'queue_result.json'
            if previous.exists() and driver.read_json(previous).get('stage') == 'complete':
                results[name] = driver.read_json(previous)
                continue
            save('status.json', dict(stage='running', active=name, results=results, updated=time.time()))
            print('Starting', name, flush=True)
            try:
                run = prepare(name, source, profile)
                status = run / 'status.json'
                if not status.exists() or driver.read_json(status).get('stage') != 'complete':
                    jobs.pipeline(run)
                errors = {}
                if source:
                    # An audit discrepancy must remain visible without suppressing the comparison.
                    for phase in ('audit', 'compare'):
                        try:
                            jobs.execute(run, 'queue_' + phase, [sys.executable,
                                ROOT / 'stage23_bench/runs/20261027_jobs/jobs.py', phase, '--run-dir', run])
                        except Exception as error:
                            errors[phase] = str(error)
                result = dict(stage='complete' if not errors else 'complete with validation errors',
                              errors=errors, winner=str(run / 'results/best.poly'), updated=time.time())
            except Exception as error:
                traceback.print_exc()
                result = dict(stage='failed', error=str(error), updated=time.time())
            driver.write_json(BASE / name / 'queue_result.json', result)
            results[name] = result
            print(name, result, flush=True)
        results = read_results()
        save('status.json', dict(stage=aggregate_stage(results), results=results, updated=time.time()))
        save('owner.json', dict(pid=os.getpid(), finished=time.time(), only=only))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--background', action='store_true')
    ap.add_argument('--only', help='comma-separated job names to process')
    ap.add_argument('--wait-lock', action='store_true', help='wait for the active queue to finish')
    args = ap.parse_args()
    if args.only and not set(args.only.split(',')) <= {row[0] for row in SPECS}:
        ap.error('--only contains an unknown job name')
    BASE.mkdir(parents=True, exist_ok=True)
    if args.background:
        print('Started queue', background(args.only, args.wait_lock))
    else:
        main(args.only, args.wait_lock)
