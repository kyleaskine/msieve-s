#!/usr/bin/env python3
"""compare.py RUN: wait for both C196 paths, snapshot the CPU results, then compare.

Reads scoring.json; saves breadth scores, refines the two leading GPU seed searches
with in-process MurphyE, and independently scores a diverse finalist set at 256k.
This is a model comparison, not a test-sieve result. Progress: comparison_status.json.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import cado_poly_text, read_cado_blocks, resultant
from score_polys import lattice_scores, score_bounds


def main():
    run = Path(sys.argv[1]).resolve()
    settings = json.loads((run / 'settings.json').read_text())
    scoring = json.loads((run / 'scoring.json').read_text())
    n = settings['n']
    deadline = time.time() + 8 * 3600
    records = []

    def state(stage, **extra):
        data = dict(stage=stage, updated=time.time(), **extra)
        tmp = run / 'comparison_status.tmp'
        tmp.write_text(json.dumps(data, indent=2) + '\n')
        tmp.replace(run / 'comparison_status.json')
        print(stage, extra, flush=True)

    def wait():
        if time.time() > deadline:
            raise TimeoutError('GPU or CPU pipeline did not finish within eight hours')
        time.sleep(15)

    def command(name, args):
        start = time.time()
        with (run / f'{name}.log').open('w') as log:
            subprocess.run(list(map(str, args)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        records.append(dict(name=name, command=list(map(str, args)), elapsed_seconds=time.time() - start))
        (run / 'comparison_commands.json').write_text(json.dumps(records, indent=2) + '\n')

    def completed_cpu():
        # The next pipeline can archive this run. Match n, never just its pathname.
        candidates = [run / 'cpu_reference', ROOT / 'pipeline_results',
                      *sorted(ROOT.glob('backup_*/pipeline_results'), reverse=True)]
        for src in candidates:
            config = src / 'pipeline_settings.txt'
            if not config.exists() or 'STATUS=complete\n' not in config.read_text():
                continue
            poly = src / 'cado_ropt_orig.txt'
            if not poly.exists():
                continue
            with poly.open() as stream:
                found = next((int(line.split(':', 1)[1]) for line in stream if line.startswith('n:')), None)
            if found == n:
                return src
        return None

    try:
        state('waiting for CPU pipeline to finish')
        while (src := completed_cpu()) is None:
            wait()
        # Freeze comparison inputs immediately, even while GPU sopt is still running.
        hashes = {}
        for name in ('msieve_ropt_orig.p', 'msieve_ropt_inv.p', 'cado_ropt_orig.txt',
                     'cado_ropt_inv.txt', 'pipeline_report.txt', 'pipeline_settings.txt'):
            dst = run / 'cpu' / name
            shutil.copy2(src / name, dst)
            with dst.open('rb') as stream:
                hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        (run / 'cpu/provenance.json').write_text(json.dumps(dict(source=str(src), sha256=hashes), indent=2) + '\n')
        state('waiting for GPU search to finish')
        while True:
            progress = json.loads((run / 'status.json').read_text())
            if progress['stage'] == 'failed':
                raise RuntimeError(progress['error'])
            if progress['stage'].startswith('GPU search complete'):
                break
            wait()
        state('scoring breadth and CPU baseline')
        old = ROOT / 'stage23_bench/runs/20261011_c184'
        command('breadth_comparison', [sys.executable, old / 'score.py', run,
            scoring['params'], scoring['lattice'], scoring['qband'], 8])
        command('breadth_analysis', [sys.executable, old / 'analyze.py', run])

        rows = []
        for source in ('baseline', 'gpu'):
            polys = read_cado_blocks(run / f'scored/{source}.poly')
            lines = (run / f'scored/{source}.tsv').read_text().splitlines()
            if len(lines) != len(polys):
                raise RuntimeError('score/polynomial count mismatch')
            for line, (_, p) in zip(lines, polys):
                band, skew, murphy, y1, label = line.split('\t')
                rows.append(dict(band=float(band), label=label, poly=dict(p, skew=float(skew)), source=source))
        # Keep one winner per CPU method plus twelve GPU finalists; preserve diversity.
        finalists, seen_methods = [], set()
        for row in sorted(rows, key=lambda r: -r['band']):
            method = row['label'].split()[0]
            if row['source'] == 'baseline' and method not in seen_methods:
                finalists.append(row)
                seen_methods.add(method)
        gpu = sorted((r for r in rows if r['source'] == 'gpu'), key=lambda r: -r['band'])
        finalists.extend(gpu[:12])
        seeds = []
        for row in gpu:
            seed = row['label'].split()[2]
            if seed not in seeds:
                seeds.append(seed)
            if len(seeds) == 2:
                break
        (run / 'murphy').mkdir(exist_ok=True)
        bounds = score_bounds(scoring['params'])
        for seed in seeds:
            state('MurphyE refinement before output cut', seed=seed)
            dst = run / f'murphy/{seed}.poly'
            command(f'murphy/{seed}', [run / 'bin/s23_ropt', '-poly', run / f'seeds/{seed}.poly',
                '-search', '-budget', 3, '-aw', 1.3, '-maxcells', '2e11', '-rerank', 4096,
                '-top', 3, '-t', 8, '-launch', 30, '-murphy', '-Bf', bounds['Bf'],
                '-Bg', bounds['Bg'], '-area', bounds['area'], '-murphy-points', 16000,
                '-murphy-eval-points', 64000, '-maxeval', 384, '-out', dst])
            finalists.extend(dict(label=f'gpu_murphy seed {seed} rank {i} {label}', poly=p, source='gpu_murphy')
                              for i, (label, p) in enumerate(read_cado_blocks(dst)[:4]))
        if any(r['poly']['n'] != n or not resultant(r['poly']) or resultant(r['poly']) % n for r in finalists):
            raise RuntimeError('invalid finalist polynomial')
        state('independent 256k finalist scoring')
        args = SimpleNamespace(threads=8, params=scoring['params'], lattice=scoring['lattice'],
            qband=scoring['qband'], points=16000, eval_points=256000, useskew=False,
            nlat=256, npts=4096, latseed=2)
        # Give every polynomial the same independent best-skew procedure.
        values = lattice_scores([r['poly'] for r in finalists], n, args)
        best = max(v[0] for v in values)
        report = []
        with (run / 'finalists.poly').open('w') as out:
            for row, (band, skew, murphy, per_q) in zip(finalists, values):
                p = dict(row['poly'], skew=skew)
                out.write('# ' + row['label'] + '\n' + cado_poly_text(p) + '\n')
                report.append(dict(label=row['label'], band=band, ratio=band/best, murphy=murphy,
                                   skew=skew, relations_per_q=per_q))
        summary = dict(scoring=scoring, validation=dict(points=16000, eval_points=256000,
            nlat=256, npts=4096, latseed=2), note='Model comparison; no test sieving performed.',
            rows=sorted(report, key=lambda r: -r['band']))
        (run / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
        state('comparison complete', winner=summary['rows'][0]['label'])
    except BaseException as error:
        state('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
