#!/usr/bin/env python3
"""Archive compact C192 evidence; full comparison is added when available."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'stage23_bench/tools'))
from gpu_pipeline import write_json, digest
from polyfmt import read_cado_blocks, rotation_between, cado_poly_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'stage23_gpu/runs/20261019_c192')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    dest = ROOT/'stage23_bench/data/c192/ropt_coverage'
    dest.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run/'prepared.json').read_text())
    winner = read_cado_blocks(run/'fixtures/cado_winner.poly')[0][1]
    recovered = []
    for rank, (label, p) in enumerate(read_cado_blocks(run/'experiments/dense_depth.poly'), 1):
        relation = rotation_between(winner, p)
        if relation and relation['rotation'] == [0]:
            recovered.append(dict(rank=rank, relation=relation, label=label, poly=p))
    if len(recovered) != 1:
        raise ValueError('expected one recovered rotation')
    for name in ('seed.poly', 'seed_inv.poly', 'cado_winner.poly'):
        shutil.copy2(run/'fixtures'/name, dest/name)
    (dest/'gpu_recovered.poly').write_text(cado_poly_text(recovered[0]['poly']))
    measurements = [json.loads(p.read_text()) for p in sorted((run/'experiments').glob('*.json'))]
    report = dict(n=meta['n'], cpu_seed_index=meta['cpu_seed_index'],
        winner_rotation=meta['winner_rotation'], identical_deep_input_indices=meta['identical_deep_input_indices'],
        recovered=recovered, experiments=measurements,
        target_comparison=json.loads((run/'target_comparison/summary.json').read_text()),
        driver_check=json.loads((run/'driver_check.json').read_text()) if (run/'driver_check.json').exists() else None,
        caveats=['The known seed was deliberately traced; the winning cell was not injected.',
                 'Full-pipeline evidence is separate and exists only after comparison.json is written.',
                 'Shared machine timings; no test sieving. Sieve settings are provisional.'])
    if (run/'comparison/summary.json').exists():
        shutil.copy2(run/'comparison/summary.json', dest/'comparison.json')
        shutil.copy2(run/'comparison/finalists.poly', dest/'finalists.poly')
        rows = json.loads((dest/'comparison.json').read_text())['rows']
        for method, filename in [('gpu_pipeline', 'gpu_best.poly'), ('cado_orig', 'cado_best.poly')]:
            best = max((r for r in rows if r['method']==method), key=lambda r:r['lattice_score'])
            (dest/filename).write_text(cado_poly_text(dict(best['poly'], skew=best['lattice_skew'])))
    for name in ('winner_validation.json', 'driver_screen_check.json'):
        if (run/name).exists():
            shutil.copy2(run/name, dest/name)
    report['fixtures_sha256'] = {p.name:digest(p) for p in dest.glob('*.poly')}
    write_json(dest/'measurements.json', report)
    print('Saved compact C192 evidence to', dest, flush=True)


if __name__ == '__main__':
    main()
