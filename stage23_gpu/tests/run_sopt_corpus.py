#!/usr/bin/env python3
"""Replay all eight saved initial-sopt jobs against their audited CPU outputs.

Local corpus data stays in ignored stage23_gpu/runs. Reference rows are the
verified original GPU output with every audited coefficient difference replaced
by its saved CPU result. Inputs and original outputs must match their checkpoint
hashes. New output is compared as a stream, avoiding another ~20 GB of logs.
Run --prepare-only while the GPU is busy; resume the same directory to execute.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import itertools
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'stage23_gpu/runs/20261028_existing_jobs'
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
sys.path.insert(0, str(ROOT / 'scripts'))
from gpu_pipeline import digest as sha, write_json as save
from polyfmt import coeffs, iter_sopt, cado_expe
from sopt_compare import invalid_reason

NAMES = ('c152', 'c170_418691', 'c148', 'c164', 'c179_358030', 'c179_576203', 'c181', 'c190')


def read(path):
    return json.loads(path.read_text())


def key(poly):
    return poly['n'], poly['Y0'], poly['Y1'], *coeffs(poly)


def audit_path(run):
    if run.name == 'c190':
        return ROOT / 'stage23_gpu/runs/20261028_results_review/c190_initial_sopt_audit.json'
    return run / 'sopt_audit/sopt/verification.json'


def prepare(dest, binary):
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='prepared',
                  reference='checkpoint-verified saved outputs plus all audited CPU corrections', jobs=[])
    for name in NAMES:
        run = BASE / name
        audit = audit_path(run)
        data = read(audit)
        checks = {}
        for checkpoint, filename in [('raw', 'msieve.dat.deduped.ms'), ('sopt_e0', 'cado_sopt_unsorted.txt')]:
            relative = 'preprocess/' + filename
            expected = read(run / 'checkpoints' / (checkpoint + '.json'))['outputs'][relative]
            path = run / relative
            if sha(path) != expected:
                raise ValueError('source checkpoint mismatch: ' + str(path))
            checks[checkpoint] = dict(path=str(path), sha256=expected)
        if data['polynomials'] != read(run / 'results/summary.json')['input_polynomials']:
            raise ValueError('reference count mismatch: ' + name)
        if len(data['differences']) != data['different']:
            raise ValueError('incomplete CPU correction list: ' + name)
        corrections = {r['index']: r for r in data['differences']}
        if len(corrections) != data['different'] or any(i < 1 or i > data['polynomials'] for i in corrections):
            raise ValueError('invalid CPU correction indices: ' + name)
        report['jobs'].append(dict(name=name, count=data['polynomials'], sources=checks,
                                  audit=dict(path=str(audit), sha256=sha(audit))))
        print('Prepared', name, data['polynomials'], flush=True)
    frozen = dest / 's23_sopt'
    shutil.copy2(binary, frozen)
    report.update(binary=str(frozen), binary_sha256=sha(frozen), script_sha256=sha(Path(__file__)),
                  total=sum(j['count'] for j in report['jobs']))
    save(dest / 'manifest.json', report)
    return report


def available():
    data = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    usage, free = map(int, data.splitlines()[0].split(','))
    return usage <= 5 and free >= 5000


def wait_available(wait=False):
    if not wait:
        if not available():
            raise RuntimeError('GPU busy; resume when idle or use --wait-gpu')
        return
    # Observe a full minute of low usage, not a pause between kernel launches.
    quiet_since = None
    while True:
        if available():
            if quiet_since is None:
                quiet_since = time.monotonic()
            if time.monotonic() - quiet_since >= 60:
                return
        else:
            quiet_since = None
        time.sleep(20)


def execute(job, plan, dest, wait=False):
    for source in [*job['sources'].values(), job['audit']]:
        if sha(Path(source['path'])) != source['sha256']:
            raise ValueError('source changed: ' + source['path'])
    corrections = {r['index']: r for r in read(Path(job['audit']['path']))['differences']}
    wait_available(wait)
    command = [plan['binary'], '-t', '2', '-batch', '4096', '-slice', '10',
               '-inputpolys', job['sources']['raw']['path']]
    record = dict(name=job['name'], status='running', command=command, count=0, differences=[],
                  script_sha256=sha(Path(__file__)), binary_sha256=plan['binary_sha256'])
    path = dest / (job['name'] + '.json')
    save(path, record)
    start = time.monotonic()
    with (dest / (job['name'] + '.stderr')).open('wb') as err:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err, bufsize=0, start_new_session=True)
        try:
            # Read only the header, without a buffered reader consuming pair data.
            header = bytearray()
            while len(header) < 4096:
                b = proc.stdout.read(1)
                if not b or b == b'\n': break
                header.extend(b)
            if b'GPU, batches of 4096' not in header:
                raise RuntimeError('GPU/batch mode unavailable: ' + header.decode(errors='replace'))
            gpu = iter_sopt(f'/proc/self/fd/{proc.stdout.fileno()}')
            reference = iter_sopt(job['sources']['sopt_e0']['path'])
            for index, pair in enumerate(itertools.zip_longest(reference, gpu), 1):
                old, new = pair
                if old is None or new is None or key(old[0]) != key(new[0]):
                    raise RuntimeError(f'input/order/count mismatch at {index}')
                raw, expected = old
                if index in corrections:
                    correction = corrections[index]
                    if key(expected) != key(correction['gpu']):
                        raise RuntimeError(f'audit does not match baseline at {index}')
                    expected = correction['cpu']
                actual = new[1]
                if key(expected) != key(actual):
                    reason = invalid_reason(raw, actual)
                    scores = cado_expe([expected, actual], raw['n']) if not reason else None
                    record['differences'].append(dict(index=index, raw=raw, cpu=expected, gpu=actual,
                                                      invalid=reason, scores=scores))
                record['count'] = index
                if index % 50000 == 0:
                    record['seconds'] = time.monotonic() - start
                    save(path, record)
                    print(job['name'], index, '/', job['count'], 'differences', len(record['differences']), flush=True)
            code = proc.wait(timeout=60)
            if code or record['count'] != job['count']:
                raise RuntimeError(f'process/count failure: exit={code}, count={record["count"]}')
            record.update(status='passed' if not record['differences'] else 'differences',
                          returncode=code, seconds=time.monotonic()-start)
        except BaseException as error:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            record.update(status='failed', error=str(error), seconds=time.monotonic()-start)
            raise
        finally:
            proc.stdout.close()
            save(path, record)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--binary', type=Path, default=ROOT / 'stage23_gpu/build/s23_sopt')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--wait-gpu', action='store_true', help='wait for one minute of idle GPU usage before each job')
    args = ap.parse_args()
    dest = args.run_dir.resolve()
    if not dest.is_relative_to(ROOT / 'stage23_gpu/runs'):
        ap.error('--run-dir must be under stage23_gpu/runs')
    dest.mkdir(parents=True, exist_ok=True)
    lock = (dest / 'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(dest / 'manifest.json') if (dest / 'manifest.json').exists() else prepare(dest, args.binary)
    if sha(Path(plan['binary'])) != plan['binary_sha256']:
        raise ValueError('frozen regression executable changed')
    if args.prepare_only:
        print('Prepared', plan['total'], 'inputs; no GPU work performed')
        return
    for job in plan['jobs']:
        previous = dest / (job['name'] + '.json')
        if previous.exists() and read(previous)['status'] == 'passed': continue
        result = execute(job, plan, dest, args.wait_gpu)
        if result['status'] != 'passed':
            raise RuntimeError('coefficient differences recorded: ' + job['name'])
    save(dest / 'summary.json', dict(status='passed', total=plan['total'], exact=plan['total'],
                                    binary_sha256=plan['binary_sha256']))


if __name__ == '__main__':
    main()
