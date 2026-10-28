#!/usr/bin/env python3
"""Live C179 depth retention, saved-pool CPU parity, and independent MurphyE.

The known rotation is traced but is never inserted into the candidate pool.
Requires a fresh destination; all commands, binaries and outputs are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import read_cado_blocks, resultant, cado_poly_text, rotation_between
import score_polys as scores


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--score-only', action='store_true', help='rescore already verified search/replay outputs')
    args = ap.parse_args()
    run = args.run_dir.resolve()
    if not args.score_only:
        run.mkdir(parents=True, exist_ok=False)
        (run / 'bin').mkdir()
    fixture = ROOT / 'stage23_bench/data/c179/ropt_coverage'
    if not args.score_only:
        for name in ('seed.poly', 'msieve_winner.poly', 'summary.json'):
            shutil.copy2(fixture / name, run / name)
        for name in ('s23_ropt', 's23_ropt_cpu'):
            shutil.copy2(ROOT / 'stage23_gpu/build' / name, run / 'bin' / name)
        shutil.copy2(ROOT / 'stage23_gpu/runs/20261026_review/corpus/bin/cado_murphy', run / 'bin/cado_murphy')
        save(run / 'provenance.json', {str(p.relative_to(run)): digest(p)
             for p in [*sorted((run / 'bin').iterdir()), run / 'seed.poly', run / 'msieve_winner.poly', run / 'summary.json']})
    else:
        for name, expected in json.loads((run / 'provenance.json').read_text()).items():
            if digest(run / name) != expected: raise ValueError('changed input: ' + name)
    records = json.loads((run / 'commands.json').read_text()) if args.score_only else []

    def execute(tag, argv):
        argv = list(map(str, argv))
        print(tag, 'starting', flush=True)
        save(run / (tag + '_command.json'), argv)
        start = time.monotonic()
        with (run / (tag + '.log')).open('wb') as log:
            proc = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        text = (run / (tag + '.log')).read_text(errors='replace')
        record = dict(tag=tag, command=argv, seconds=time.monotonic()-start, returncode=proc.returncode,
                      log_sha256=digest(run / (tag + '.log')),
                      trace=[l for l in text.splitlines() if l.startswith('# trace-cell')])
        records.append(record)
        save(run / 'commands.json', records)
        if proc.returncode: raise RuntimeError(f'{tag}: exit {proc.returncode}; see saved log')
        print(tag, f"{record['seconds']:.2f}s", record['trace'][-1:] or '', flush=True)
        return text

    common = ['-poly', run / 'seed.poly', '-search', '-budget', '3', '-maxcells', '6e12',
              '-maxlines', '1000', '-umax', '20000', '-aw', '1.3', '-rerank', '4096',
              '-refine', '200', '-top', '8', '-t', '2', '-trace-cell', '-55', '34188178']
    if not args.score_only:
        execute('depth_gpu', [run / 'bin/s23_ropt', *common, '-launch', '30', '-check', '128',
                     '-refine-gpu', '-refine-check', '-pool-out', run / 'depth.pool', '-out', run / 'depth_gpu.poly'])
        execute('depth_cpu_replay', [run / 'bin/s23_ropt_cpu', *common, '-pool-in', run / 'depth.pool',
                                   '-out', run / 'depth_cpu.poly'])
    if len(records) != 2 or any(r['returncode'] for r in records): raise ValueError('incomplete search/replay')
    text = (run / 'depth_gpu.log').read_text()
    if 'vs brute force: 0 differ' not in text: raise ValueError('missing root-score audit')
    if digest(run / 'depth_gpu.poly') != digest(run / 'depth_cpu.poly'):
        raise ValueError('CPU/GPU final refinement exports differ')
    seed = read_cado_blocks(run / 'seed.poly')[0][1]
    n = seed['n']
    rows = []
    for method, path in [('gpu_depth', run / 'depth_gpu.poly'), ('msieve_orig', run / 'msieve_winner.poly')]:
        for i, (_, poly) in enumerate(read_cado_blocks(path)):
            r = resultant(poly)
            if not r or r % n: raise ValueError('invalid resultant')
            tr = rotation_between(seed, poly)
            uv = None if tr is None else dict(u=tr['rotation'][1],
                v=tr['rotation'][0]-tr['rotation'][1]*tr['t'], t=tr['t'])
            rows.append(dict(method=method,index=i,poly=poly,rotation=uv,
                target=uv is not None and (uv['u'], uv['v']) == (-55,34188178)))
    scores.cado_murphy_binary = lambda: str(run / 'bin/cado_murphy')
    params = SimpleNamespace(params='default', threads=2, points=16000, eval_points=64000,
                             trans=0, useskew=False)
    print('Independent scoring', len(rows), 'exports', flush=True)
    values = scores.murphy([r['poly'] for r in rows], n, params)
    for row, (e, skew, _) in zip(rows, values): row.update(murphy=e, optimized_skew=skew)
    save(run / 'screen.json', rows)
    selected = sorted([r for r in rows if r['method']=='gpu_depth'], key=lambda r:-r['murphy'])[:10]
    selected += [r for r in rows if r['method']=='msieve_orig' or r['target'] and r not in selected]
    params.useskew = True
    for angles in (256000, 1024000):
        params.eval_points = angles
        values = scores.murphy([dict(r['poly'],skew=r['optimized_skew']) for r in selected],n,params)
        for row,(e,_,_) in zip(selected,values): row[f'murphy_{angles}']=e
    winners={m:max((r for r in selected if r['method']==m),key=lambda r:r['murphy_1024000'])
             for m in ('gpu_depth','msieve_orig')}
    summary=dict(n=n,commands=records,exports_verified=len(rows)-1,cpu_gpu_bytes_equal=True,
        target_exports=sum(r['target'] and r['method']=='gpu_depth' for r in rows),
        winners=winners,finalists=selected,
        gpu_over_msieve=winners['gpu_depth']['murphy_1024000']/winners['msieve_orig']['murphy_1024000'],
        scope='Focused seed-9 depth check, not a complete C179 pipeline or a test sieve.')
    save(run / 'result.json',summary)
    for method,row in winners.items():
        (run / (method+'_best.poly')).write_text(cado_poly_text(dict(row['poly'],skew=row['optimized_skew'])))
    print('Verified',summary['exports_verified'],'exports; target copies',summary['target_exports'],
          'GPU/MS Murphy ratio',summary['gpu_over_msieve'],flush=True)


if __name__ == '__main__': main()
