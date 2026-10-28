#!/usr/bin/env python3
"""Snapshot and process missed jobs; preserve status and continue after failures."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
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
]


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
        sources = [ROOT / 'msieve.dat.deduped.ms',
                   ROOT.parent / 'msieve-s-backup/msieve.dat.deduped.ms']
        identities = []
        # Bound each copy to its initial size, preserving both trees before work starts.
        with combined.open('wb') as out:
            for i, path in enumerate(sources):
                child = driver.Run(run / ('source_' + str(i)))
                identity, frozen = child.snapshot(path, 'raw')
                identities.append(identity)
                with frozen.open('rb') as stream:
                    shutil.copyfileobj(stream, out)
                out.write(b'\n')
        identity, _ = workflow.snapshot(combined, 'raw')
        with combined.open() as stream:
            n = next(int(line.split(':', 1)[1]) for line in stream if line.startswith('n:'))
        for binary in ('s23_sopt', 's23_ropt', 's23_ropt_cpu'):
            workflow.freeze(binary, ROOT / 'stage23_gpu/build' / binary)
        workflow.freeze('cado_murphy')
        selected = next(p for p in sieve_profiles.load() if p['id'] == profile)
        driver.write_json(run / 'scoring.json', sieve_profiles.scoring(selected))
        driver.write_json(run / 'prepared.json', dict(n=n, digits=len(str(n)), input=identity,
            sources=identities, scope='Union of both current input trees; deduplicate before GPU sopt; no test sieve.'))
    return run


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / '.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = {}
        for name, source, profile in SPECS:
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
        save('status.json', dict(stage='complete' if all(r['stage'] == 'complete' for r in results.values())
            else 'finished with errors', results=results, updated=time.time()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--background', action='store_true')
    args = ap.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    if args.background:
        with (BASE / 'queue.log').open('ab') as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve())], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save('background.json', dict(pid=proc.pid, started=time.time()))
        print('Started queue', proc.pid)
    else:
        main()
