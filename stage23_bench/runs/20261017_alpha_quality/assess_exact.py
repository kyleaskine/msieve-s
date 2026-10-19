#!/usr/bin/env python3
"""Check whether exact-size screening preserves the quality audit's winners.

Reuse already validated scores only when the retained integer polynomial is
identical. This establishes preservation of the broad control's winners; it
does not claim a score for newly exported configurations.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import cado_poly_text, read_cado_blocks, resultant


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, default=ROOT / 'stage23_gpu/runs/20261017_alpha_quality')
    args = ap.parse_args()
    base = args.run_dir.resolve()
    audit = json.loads((base / 'summary.json').read_text())
    cases = []
    count = 0
    for record in audit['cases']:
        case = record['case']
        name = 'c208_screen_exact_0.poly' if case.startswith('c208_') else case + '_screen_exact.poly'
        manifest = json.loads((base / case / 'manifest.json').read_text())
        old = {tuple(e['cell']): e['poly'] for e in manifest['entries']}
        new = {}
        for label, poly in read_cado_blocks(base / 'replay_exact' / name):
            cell = tuple(map(int, re.search(r'u\s+(-?\d+)\s+v\s+(-?\d+)', label).groups()))
            r = resultant(poly)
            if not r or r % poly['n']:
                raise ValueError(f'invalid resultant: {case} {cell}')
            new[cell] = poly
            count += 1
            if cell in old and cado_poly_text(poly) != cado_poly_text(old[cell]):
                raise ValueError(f'changed polynomial configuration: {case} {cell}')
        cases.append(dict(case=case, retained_quality_winners={
            metric: tuple(data['winner_cells']['wide']) in new for metric, data in record['metrics'].items()}))
    result = dict(cases=cases, verified_exports=count,
                  preserved={metric: sum(c['retained_quality_winners'][metric] for c in cases)
                             for metric in ('murphy', 'lattice')},
                  note='Exact integer-polynomial membership check against previously validated winners. '
                       'Newly exported candidates are not rescored here. Known cases now inform the exact-size '
                       'policy, so these are regression controls, not a fresh independent acceptance set.')
    (base / 'exact_size_quality.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
