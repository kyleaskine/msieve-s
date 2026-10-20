#!/usr/bin/env python3
"""Compare frozen C192 controls and GPU exports under both scoring objectives.

--target scores the focused seed experiments; otherwise require a completed
full GPU pipeline. Preserve both standard-Murphy and lattice winners per method.
"""
import argparse
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from gpu_pipeline import digest, write_json, validate_poly
from polyfmt import cado_poly_text, read_cado_blocks, parse_cado_ropt, rotation_between
import score_polys as scores


def msieve_leaders(path, n):
    """Bound memory while retaining three cells per rational family."""
    best, current = {}, None
    head = re.compile(r'# norm (\S+) alpha (\S+) e (\S+) rroots (\d+)')
    with path.open() as stream:
        for line in stream:
            m = head.match(line)
            if m:
                current = dict(n=n, e=float(m[3]))
            elif current is not None:
                m = re.fullmatch(r'(Y[01]|c[0-5]): (-?\d+)\s*', line)
                if m:
                    current[m[1]] = int(m[2])
                    if m[1] == 'Y1':
                        group = best.setdefault(current['Y1'], [])
                        group.append(current)
                        group.sort(key=lambda p: -p['e'])
                        del group[3:]
                        current = None
                elif line.startswith('skew:'):
                    current['skew'] = float(line.split()[1])
    return [p for group in best.values() for p in group]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'stage23_gpu/runs/20261019_c192')
    parser.add_argument('--target', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    dest = run / ('target_comparison' if args.target else 'comparison')
    dest.mkdir(exist_ok=True)
    n = json.loads((run / 'prepared.json').read_text())['n']
    seed = read_cado_blocks(run / 'fixtures/seed.poly')[0][1]
    # Use the frozen scorer, including in subprocesses launched by score_polys.
    scores.cado_murphy_binary = lambda: str(run / 'bin/cado_murphy')
    sources = [run / 'fixtures/references.json', run / 'scoring.json']
    rows = json.loads(sources[0].read_text())
    for row in rows:
        row['label'] = f"{row['method']} #{row['index']}"
    for path in sorted((run / 'experiments').glob('cado_e*.log')):
        for p in parse_cado_ropt(path):
            rows.append(dict(method=path.stem, label=path.stem, poly=p))
        sources.append(path)
    if args.target:
        rows = [r for r in rows if r['poly']['Y1'] == seed['Y1']]
        for path in sorted((run / 'experiments').glob('*.poly')):
            for label, p in read_cado_blocks(path):
                rows.append(dict(method=path.stem, label=label, poly=p))
            sources.append(path)
    else:
        summary_path = run / 'results/summary.json'
        gpu = json.loads(summary_path.read_text())
        sources.append(summary_path)
        for row in gpu['finalists']:
            rows.append(dict(method='gpu_pipeline', label=f"{row['phase']} {row['search']} {row['rank']}", poly=row['poly']))
        for method, name in [('msieve_orig', 'msieve_ropt_orig.p'), ('msieve_inv', 'msieve_ropt_inv.p'),
                ('msieve_deep_orig', 'msieve_deep_ropt_top32/msieve_deep_ropt_orig.p'),
                ('msieve_deep_inv', 'msieve_deep_ropt_top32/msieve_deep_ropt_inv.p')]:
            path = run / 'cpu_reference' / name
            for i, p in enumerate(msieve_leaders(path, n)):
                rows.append(dict(method=method, label=f'{method} #{i}', poly=p))
            sources.append(path)
        # Targeted experiments are separate methods, never attributed to the blind pipeline.
        for path in sorted((run / 'experiments').glob('*.poly')):
            for label, p in read_cado_blocks(path):
                rows.append(dict(method=path.stem, label=label, poly=p))
            sources.append(path)
    for row in rows:
        validate_poly(row['poly'], n)
        rel = rotation_between(seed, row['poly'])
        if rel and len(rel['rotation']) == 2:
            u = rel['rotation'][1]
            row['rotation'] = dict(u=u, v=rel['rotation'][0]-u*rel['t'], t=rel['t'])
    provenance = dict(sha256={str(p.relative_to(run)): digest(p) for p in sources},
                      scorer_sha256=digest(run / 'bin/cado_murphy'))
    cache = dest / 'screen.json'
    params = SimpleNamespace(params='default', threads=4, points=4000, eval_points=16000,
            trans=0, useskew=False, lattice='15,16384', qband='80e6,500e6,7', nlat=32, npts=2048, latseed=0)
    if cache.exists() and json.loads(cache.read_text())['provenance'] == provenance:
        rows = json.loads(cache.read_text())['rows']
    else:
        print('Screening', len(rows), 'polynomials under both objectives', flush=True)
        standard = scores.murphy([r['poly'] for r in rows], n, params)
        params.params = 'job:32,33,15,80e6'
        lattice = scores.lattice_scores([r['poly'] for r in rows], n, params)
        for row, m, l in zip(rows, standard, lattice):
            row.update(standard_murphy=m[0], standard_skew=m[1], lattice_score=l[0], lattice_skew=l[1])
        write_json(cache, dict(provenance=provenance, rows=rows))
    indices = set()
    for metric in ('standard_murphy', 'lattice_score'):
        order = sorted(range(len(rows)), key=lambda i: -rows[i][metric])
        indices.update(order[:12])
        for method in {r['method'] for r in rows}:
            indices.add(next(i for i in order if rows[i]['method'] == method))
    finals = [dict(rows[i]) for i in sorted(indices)]
    print('Validating', len(finals), 'finalists', flush=True)
    params.params, params.points, params.eval_points = 'default', 16000, 256000
    params.useskew = True
    values = scores.murphy([dict(r['poly'], skew=r['standard_skew']) for r in finals], n, params)
    params.params = 'job:32,33,15,80e6'
    params.nlat, params.npts, params.latseed = 256, 4096, 2
    lattice = scores.lattice_scores([dict(r['poly'], skew=r['lattice_skew']) for r in finals], n, params)
    for row, m, l in zip(finals, values, lattice):
        row.update(standard_murphy=m[0], lattice_score=l[0], job_murphy=l[2], per_q=l[3])
    finals.sort(key=lambda r: -r['lattice_score'])
    write_json(dest / 'summary.json', dict(n=n, provenance=provenance, screened=len(rows), rows=finals,
        settings=json.loads((run/'scoring.json').read_text()),
        validation=dict(murphy_angles=256000, lattices=256, samples=4096, stream=2),
        note='Models only, no test sieving. Each objective uses its separately selected, then frozen skew.'))
    (dest / 'finalists.poly').write_text('\n'.join('# '+r['method']+'\n'+cado_poly_text(dict(r['poly'], skew=r['lattice_skew'])) for r in finals))
    for method in sorted({r['method'] for r in finals}):
        subset = [r for r in finals if r['method']==method]
        print(method, 'standard MurphyE', max(r['standard_murphy'] for r in subset),
              'lattice', max(r['lattice_score'] for r in subset), flush=True)


if __name__ == '__main__':
    main()
