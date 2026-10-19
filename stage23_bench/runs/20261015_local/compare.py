#!/usr/bin/env python3
"""Wait for the local GPU jobs and independently compare their CPU controls."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'stage23_gpu/runs/20261015_local'
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
sys.path.insert(0, str(ROOT / 'scripts'))
from polyfmt import cado_poly_text, parse_cado_ropt, parse_msieve_p, resultant
from score_polys import lattice_scores
from gpu_pipeline import write_json


def compare(run):
    dest = run / 'comparison'
    dest.mkdir(exist_ok=True)
    if (dest / 'summary.json').exists():
        return
    settings = json.loads((run / 'scoring.json').read_text())
    gpu = json.loads((run / 'results/summary.json').read_text())
    n = gpu['n']
    rows = []
    for method, filename, reader, score in [
        ('msieve_orig', 'msieve_ropt_orig.p', parse_msieve_p, 'e'),
        ('msieve_inv', 'msieve_ropt_inv.p', parse_msieve_p, 'e'),
        ('cado_orig', 'cado_ropt_orig.txt', parse_cado_ropt, 'MurphyE'),
        ('cado_inv', 'cado_ropt_inv.txt', parse_cado_ropt, 'MurphyE')]:
        counts = {}
        for p in sorted(reader(run / 'cpu_reference' / filename), key=lambda p: -p[score]):
            # Broaden beyond the originally printed winner: job bounds can reorder seeds.
            y1 = p['Y1']
            if counts.get(y1, 0) >= 3:
                continue
            counts[y1] = counts.get(y1, 0) + 1
            rows.append(dict(method=method, label=f'{method} Y1={y1} cell={counts[y1]}', poly=dict(p, n=n)))
    for p in rows:
        r = resultant(p['poly'])
        if not r or r % n:
            raise RuntimeError('invalid CPU control')
    args = SimpleNamespace(params=settings['params'], lattice=settings['lattice'], qband=settings['qband'],
        threads=4, points=4000, eval_points=16000, useskew=False, nlat=32, npts=2048, latseed=0)
    print(run.name, 'screening', len(rows), 'CPU candidates', flush=True)
    values = lattice_scores([r['poly'] for r in rows], n, args)
    screened = [dict(row, score=v[0], poly=dict(row['poly'], skew=v[1])) for row, v in zip(rows, values)]
    screened.sort(key=lambda r: -r['score'])
    write_json(dest / 'cpu_screen.json', screened)
    finalists = screened[:12]
    for method in ('msieve_orig', 'msieve_inv', 'cado_orig', 'cado_inv'):
        best = next(row for row in screened if row['method'] == method)
        if best not in finalists:
            finalists.append(best)
    for index, row in enumerate(gpu['finalists']):
        finalists.append(dict(method='gpu', label=f"gpu {row['phase']} {row['search']} #{index+1}", poly=row['poly']))
    for row in finalists:
        r = resultant(row['poly'])
        if row['poly']['n'] != n or not r or r % n:
            raise RuntimeError('invalid finalist')
    # Keep declared GPU configurations and the skews selected in the CPU screen.
    args.points, args.eval_points = 16000, 256000
    args.nlat, args.npts, args.latseed, args.useskew = 256, 4096, 2, True
    values = lattice_scores([r['poly'] for r in finalists], n, args)
    report = [dict(row, score=v[0], skew=v[1], murphy=v[2], per_q=v[3])
              for row, v in zip(finalists, values)]
    report.sort(key=lambda row: -row['score'])
    best = report[0]['score']
    for row in report:
        row['ratio'] = row['score'] / best
    (dest / 'finalists.poly').write_text('\n'.join('# ' + r['label'] + '\n' + cado_poly_text(r['poly']) for r in report))
    write_json(dest / 'summary.json', dict(n=n, settings=settings, cpu_screened=len(screened), rows=report,
        validation=dict(angles=256000, lattices=256, samples=4096, stream=2),
        note='Model comparison with inherited job settings; no test sieving. CPU screen retains three cells per Y1 and pass.'))
    print(run.name, 'comparison complete:', report[0]['label'], flush=True)


def main():
    deadline = time.monotonic() + 24 * 3600
    for name in ('c147', 'c166'):
        run = BASE / name
        print('Waiting for', name, flush=True)
        while not (run / 'results/summary.json').exists():
            status = json.loads((BASE / 'status.json').read_text())
            if status['stage'] == 'failed':
                raise RuntimeError(status)
            if time.monotonic() > deadline:
                raise TimeoutError('GPU results unavailable after 24 hours')
            time.sleep(30)
        compare(run)
    write_json(BASE / 'comparisons_complete.json', dict(completed=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write_json(BASE / 'comparison_failure.json', dict(error=str(error), updated=time.time()))
        raise
