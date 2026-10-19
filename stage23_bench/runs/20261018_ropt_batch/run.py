#!/usr/bin/env python3
"""Audit 25 seeds in one session, or time all 16 C171 seeds in fresh/batch processes.

Freezes binaries and records argv manifests, exact output identity, resultants,
process wall time, phase time, context wait and workspace reuse. No concurrent
workers: every mode uses two CPU threads and the same seed order/policy.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('screen_corpus', Path(__file__).parents[1] / '20261018_gpu_screen/run.py')
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import read_cado_blocks, resultant


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--timings', action='store_true')
    ap.add_argument('--check', action='store_true', help='compare all device scores with CADO; excluded from timings')
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    if args.repeats < 1 or (args.timings and args.check): ap.error('positive repeats; --check is not a timing mode')
    dest = args.run_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    binaries = {}
    for mode, name in [('gpu', 's23_ropt'), ('cpu', 's23_ropt_cpu')]:
        binary = dest / name
        if not binary.exists(): shutil.copy2(ROOT / 'stage23_gpu/build' / name, binary)
        binaries[mode] = binary
    cases = corpus.cases()
    if args.timings: cases = [c for c in cases if c['tag'].startswith('c171_')]
    assert len(cases) == (16 if args.timings else 25)
    records = []
    for repeat in range(args.repeats if args.timings else 1):
        modes = ['gpu_single', 'gpu_batch', 'cpu_single', 'cpu_batch'] if args.timings else ['gpu_batch']
        if repeat % 2: modes.reverse()
        for mode in modes:
            run = dest / f'{mode}_{repeat}'
            run.mkdir(exist_ok=True)
            binary = binaries[mode.split('_')[0]]
            jobs = []
            for case in cases:
                argv = list(map(str, case['args']))
                if mode.startswith('gpu'):
                    argv.append('-alpha-screen-gpu')
                    if args.check: argv.append('-alpha-screen-check')
                argv += ['-out', str(run / (case['tag'] + '.poly'))]
                if any('\t' in a or '\n' in a or '\r' in a for a in argv):
                    raise ValueError('manifest arguments cannot contain tabs/newlines')
                jobs.append(argv)
            batch = run / 'jobs.tsv'
            batch.write_text('S23_ROPT_BATCH 1\n' + ''.join('\t'.join(j) + '\n' for j in jobs))
            commands = [[str(binary), '-batch', str(batch)]] if mode.endswith('batch') else [[str(binary), *j] for j in jobs]
            logs = []
            start = time.monotonic()
            for i, command in enumerate(commands):
                p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
                text = p.stdout + p.stderr
                (run / f'process_{i}.log').write_text(text)
                if p.returncode: raise RuntimeError(f'{command}: {p.returncode}\n{text}')
                logs.append(text)
            wall = time.monotonic() - start
            text = '\n'.join(logs)
            if mode.startswith('gpu') and len(re.findall(r'# alpha-screen GPU: \d+ cells,', text)) != len(cases):
                raise ValueError('some screens fell back entirely to CPU; not a GPU measurement')
            outputs = []
            for case, argv in zip(cases, jobs):
                out = Path(argv[-1])
                rows = read_cado_blocks(out)
                assert len(rows) == 200 and all(resultant(p) and resultant(p) % p['n'] == 0 for _, p in rows)
                if out.read_bytes() != case['expected'].read_bytes(): raise ValueError(f'changed export: {out}')
                outputs.append(dict(case=case['tag'], input_sha256=digest(argv[argv.index('-poly')+1]),
                    pool_sha256=digest(argv[argv.index('-pool-in')+1]), output_sha256=digest(out),
                    reference_sha256=digest(case['expected']), verified_exports=len(rows)))
            rec = dict(mode=mode, repeat=repeat, commands=commands, manifest=str(batch),
                       manifest_sha256=digest(batch), binary_sha256=digest(binary), wall_seconds=wall,
                       refine_seconds=sum(map(float, re.findall(r', refine ([\d.]+) s', text))),
                       screen_seconds=sum(map(float, re.findall(r'to full refinement, ([\d.]+) s', text))),
                       context_seconds=sum(map(float, re.findall(r'waited ([\d.]+) s', text))),
                       outputs=outputs)
            resource = re.search(r'device setups (\d+), buffer allocations (\d+), prime uploads (\d+)', text)
            if resource:
                rec['workspace'] = dict(zip(['device_setups', 'buffer_allocations', 'prime_uploads'], map(int, resource.groups())))
            if args.check:
                checks = re.findall(r'max \|dL\| ([\d.e+-]+), \|dA\| ([\d.e+-]+), \|dkey\| ([\d.e+-]+); (\d+) translations', text)
                assert len(checks) == len(cases)
                rec['max_errors'] = dict(zip(['L', 'alpha', 'key'], [max(float(c[i]) for c in checks) for i in range(3)]))
                rec['translation_differences'] = sum(int(c[3]) for c in checks)
            (run / 'summary.json').write_text(json.dumps(rec, indent=2) + '\n')
            records.append(rec)
            print(mode, repeat, f'{wall:.3f}s', f'{len(outputs)*200} identical exports', rec.get('workspace', {}), flush=True)
    summary = dict(checked=args.check, records=records)
    if args.timings:
        summary['medians'] = {mode: {k: statistics.median(r[k] for r in records if r['mode']==mode)
            for k in ['wall_seconds', 'refine_seconds', 'screen_seconds', 'context_seconds']}
            for mode in ['cpu_single', 'cpu_batch', 'gpu_single', 'gpu_batch']}
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
