#!/usr/bin/env python3
"""Paired root-search pilot on an idle GPU, using the audit's frozen binaries.

The configurations have equal candidate/refinement quotas, not equal cell caps
or elapsed time. Measure their actual costs before designing equal-time searches.
Each repetition must reproduce the first pool/export and pass resultant checks.
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

from audit import ROOT, VARIANTS, digest, prepare, save
from polyfmt import read_cado_blocks, resultant

CASES = ('c192_winner', 'c197_s0005', 'c197_s0001')
MODES = ('dense_breadth', 'classes2520_16', 'classes2520_32')


def idle():
    # Fresh processes only; stop between searches if another compute job starts.
    for _ in range(20):
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader'], text=True).strip()
        if apps:
            raise RuntimeError(f'GPU compute job already active: {apps}')
        usage = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu',
                                         '--format=csv,noheader,nounits'], text=True).strip()
        if len(usage.splitlines()) != 1:
            raise RuntimeError('pilot assumes exactly one GPU')
        if int(usage) <= 10:
            return dict(utilization=int(usage), active_compute_pids=[])
        time.sleep(0.5)
    raise RuntimeError('GPU remains busy; resume when idle')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    dest = args.run_dir.resolve()
    meta = prepare(dest)
    cases = {case['name']:case for case in meta['cases']}
    folder = dest / 'pilot'
    folder.mkdir(exist_ok=True)
    inventory = folder / 'gpu.txt'
    if not inventory.exists():
        inventory.write_bytes(subprocess.check_output(['nvidia-smi']))
    records = []
    for rep in range(args.repeats):
        order = MODES[rep % len(MODES):] + MODES[:rep % len(MODES)]
        for name in CASES:
            case = cases[name]
            for mode in order:
                tag = f'{name}_{mode}_r{rep}'
                log, output, pool, record = [folder / (tag + ext) for ext in ('.log', '.poly', '.pool', '.json')]
                planned = json.loads((dest / 'plans' / name / (mode + '.json')).read_text())
                if planned['returncode']:
                    raise ValueError('need a successful plan before GPU execution')
                command = [str(dest / 'bin/s23_ropt'), '-poly', str(dest / case['file']),
                    '-search', '-budget', '3', '-maxlines', '1000', '-umax', '20000', '-aw', '1.3',
                    '-rerank', '4096', '-refine', '200', '-top', '8', '-t', '2', '-launch', '30',
                    '-trace-cell', str(case['target']['u']), str(case['target']['v']), *VARIANTS[mode],
                    '-refine-gpu', '-check', '128', '-pool-out', str(pool), '-out', str(output)]
                if record.exists():
                    row = json.loads(record.read_text())
                    if row['command'] != command or row['returncode']:
                        raise ValueError(f'changed/failed run: {record}')
                    for path in (log, output, pool):
                        if digest(path) != row['sha256'][path.name]:
                            raise ValueError(f'changed artifact: {path}')
                else:
                    if any(p.exists() for p in (log, output, pool)):
                        raise ValueError(f'incomplete prior run: {tag}; inspect before retrying')
                    preflight = idle()
                    print('Running', tag, flush=True)
                    start = time.monotonic()
                    with log.open('wb') as fh:
                        proc = subprocess.run(command, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
                    row = dict(case=name, mode=mode, repetition=rep, command=command,
                        returncode=proc.returncode, seconds=time.monotonic()-start, preflight=preflight,
                        binary_sha256=digest(dest/'bin/s23_ropt'), input_sha256=digest(dest/case['file']),
                        script_sha256=digest(__file__))
                    if proc.returncode:
                        save(record, row)
                        raise RuntimeError(f'{tag}: exit {proc.returncode}')
                    text = log.read_text(encoding='utf-8', errors='replace')
                    if 'vs brute force: 0 differ' not in text:
                        raise ValueError(f'missing brute-root agreement: {tag}')
                    exports = read_cado_blocks(output)
                    if len(exports) != 200:
                        raise ValueError(f'expected 200 exports: {tag}')
                    for _, poly in exports:
                        r = resultant(poly)
                        if poly['n'] != case['n'] or not r or r % poly['n']:
                            raise ValueError(f'invalid export: {tag}')
                    match = re.search(r'sieve ([\d.]+) s .*?size model ([\d.]+) s, refine ([\d.]+) s', text)
                    if not match:
                        raise ValueError(f'missing timing: {tag}')
                    row.update(sieve_seconds=float(match[1]), model_seconds=float(match[2]),
                        refine_seconds=float(match[3]), verified_exports=len(exports),
                        trace=[s for s in text.splitlines() if s.startswith('# trace-cell')],
                        sha256={p.name:digest(p) for p in (log, output, pool)})
                    save(record, row)
                if rep:
                    for ext in ('.poly', '.pool'):
                        if digest(folder/(f'{name}_{mode}_r0'+ext)) != digest(folder/(tag+ext)):
                            raise ValueError(f'nondeterministic result: {tag}')
                records.append(row)
                print(tag, f"{row['seconds']:.3f}s", row['trace'][-1], flush=True)
                save(folder/'records.json', records)
    medians = []
    for name in CASES:
        for mode in MODES:
            rows = [r for r in records if r['case']==name and r['mode']==mode]
            medians.append(dict(case=name, mode=mode, repetitions=len(rows),
                seconds=statistics.median(r['seconds'] for r in rows),
                sieve_seconds=statistics.median(r['sieve_seconds'] for r in rows),
                model_seconds=statistics.median(r['model_seconds'] for r in rows),
                refine_seconds=statistics.median(r['refine_seconds'] for r in rows),
                trace=rows[0]['trace']))
    save(folder/'summary.json', dict(medians=medians, repeat_outputs_identical=True,
        verified_exports=sum(r['verified_exports'] for r in records), note=__doc__,
        gpu_concurrency=1, host_threads=2,
        caveat='Idle checks are between runs; no claim of exclusive hardware reservation. '
               'Process timings exclude the separate independent scoring study.'))


if __name__ == '__main__':
    main()
