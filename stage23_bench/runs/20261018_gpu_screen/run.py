#!/usr/bin/env python3
"""Audit the fixed 25-case screen corpus, then alternate CPU/GPU phase timings.

GPU audits compare every screened score with CADO and every exported byte with
the previously validated exact-size screen. Timings disable that expensive
oracle and report screen/refinement, context wait, and process wall separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import statistics

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import read_cado_blocks, resultant


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases():
    old = ROOT / 'stage23_gpu/runs/20261017_alpha_quality'
    paths = [old / 'replay_exact/c208_screen_exact_0.json']
    paths += sorted((old / 'replay_exact').glob('c171_*_screen_exact.json'))
    paths += sorted((old / 'holdout_replay').glob('*_screen.json'))
    assert len(paths) == 25
    result = []
    for path in paths:
        record = json.loads(path.read_text())
        cmd = record['command']
        tag = 'c208' if path.name.startswith('c208') else '_'.join(path.stem.split('_')[:2])
        result.append(dict(tag=tag, args=cmd[1:cmd.index('-out')],
                           expected=Path(cmd[cmd.index('-out')+1])))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, default=ROOT / 'stage23_gpu/runs/20261018_gpu_screen/audit')
    ap.add_argument('--timings', action='store_true')
    ap.add_argument('--unchecked', action='store_true', help='audit exports without the per-cell CPU oracle')
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    dest = args.run_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    binaries = {}
    for mode, name in [('gpu', 's23_ropt'), ('cpu', 's23_ropt_cpu')]:
        binaries[mode] = dest / name
        if not binaries[mode].exists():
            shutil.copy2(ROOT / 'stage23_gpu/build' / name, binaries[mode])
    records = []
    selected = cases()
    tasks = [(c, 'gpu', 0) for c in selected]
    if args.timings:
        selected = [selected[0], selected[1]] # C208 seed10 and C171 seed1
        tasks = [(c, mode, repeat) for repeat in range(args.repeats) for c in selected
                 for mode in (['cpu', 'gpu', 'wide'] if repeat % 2 == 0 else ['wide', 'gpu', 'cpu'])]
    for case, mode, repeat in tasks:
        tag = f"{case['tag']}_{mode}_{repeat}"
        out = dest / (tag + '.poly')
        opts = list(case['args'])
        if mode == 'wide':
            index = opts.index('-alpha-screen')
            count = opts[index+1]
            del opts[index:index+3]
            opts.remove('-alpha-screen-exact-size')
            opts[opts.index('-rerank')+1] = count
        if mode == 'gpu':
            opts += ['-alpha-screen-gpu']
            if not args.timings and not args.unchecked: opts += ['-alpha-screen-check']
        cmd = [str(binaries['cpu' if mode == 'wide' else mode]), *opts, '-out', str(out)]
        start = time.monotonic()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
        wall = time.monotonic() - start
        text = proc.stdout + proc.stderr
        (dest / (tag + '.log')).write_text(text)
        if proc.returncode: raise RuntimeError(f'{tag}: {proc.returncode}\n{text}')
        rows = read_cado_blocks(out)
        assert len(rows) == 200 and all(resultant(p) and resultant(p) % p['n'] == 0 for _, p in rows)
        match = out.read_bytes() == case['expected'].read_bytes()
        if mode != 'wide' and not match: raise ValueError(f'changed exports: {tag}')
        rec = dict(tag=tag, case=case['tag'], mode=mode, repeat=repeat, command=cmd,
                   binary_sha256=digest(Path(cmd[0])), output_sha256=digest(out),
                   input_sha256=digest(Path(opts[opts.index('-poly')+1])),
                   pool_sha256=digest(Path(opts[opts.index('-pool-in')+1])),
                   reference_sha256=digest(case['expected']), byte_identical=match,
                   verified_exports=len(rows), wall_seconds=wall,
                   refine_seconds=float(re.search(r', refine ([\d.]+) s', text)[1]))
        if mode != 'wide':
            rec['screen_seconds'] = float(re.search(r'to full refinement, ([\d.]+) s', text)[1])
        if mode == 'gpu':
            rec['fallbacks'] = int(re.search(r'(\d+) host fallbacks', text)[1])
            rec['boundary_rechecks'] = int(re.search(r'(\d+) boundary rechecks', text)[1])
            rec['context_seconds'] = float(re.search(r'waited ([\d.]+) s', text)[1])
            if not args.timings and not args.unchecked:
                m = re.search(r'max \|dL\| ([\d.e+-]+), \|dA\| ([\d.e+-]+), \|dkey\| ([\d.e+-]+); (\d+) translations', text)
                rec.update(max_dL=float(m[1]), max_dA=float(m[2]), max_dkey=float(m[3]), translations_differ=int(m[4]))
        records.append(rec)
        (dest / (tag + '.json')).write_text(json.dumps(rec, indent=2) + '\n')
        print(tag, f'{wall:.3f}s', 'identical' if match else 'broad control', flush=True)
    summary = dict(records=records, checked=not args.timings and not args.unchecked)
    if args.timings:
        summary['medians'] = {c['tag']: {mode: {key: statistics.median(r[key] for r in records
            if r['case'] == c['tag'] and r['mode'] == mode) for key in ('refine_seconds', 'wall_seconds')}
            for mode in ('cpu', 'gpu', 'wide')} for c in selected}
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
