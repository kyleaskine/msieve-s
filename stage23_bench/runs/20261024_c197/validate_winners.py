#!/usr/bin/env python3
"""Recheck the frozen C197 winners with denser, independent model samples."""
import argparse
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from gpu_pipeline import digest, validate_poly, write_json
import score_polys as scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    source = run / 'comparison/summary.json'
    comparison = json.loads(source.read_text())
    settings = comparison['settings']
    binary = run / 'bin/cado_murphy'
    if digest(binary) != comparison['scorer_sha256']:
        raise ValueError('comparison scorer has changed')
    scores.cado_murphy_binary = lambda: str(binary)
    winners = comparison['winners']
    for pair in winners.values():
        for method in ('cpu', 'gpu'):
            validate_poly(pair[method]['poly'], comparison['n'])
    params = SimpleNamespace(params='default', points=16000, eval_points=1024000,
        trans=0, threads=4, useskew=True, lattice=settings['lattice'],
        qband=settings['qband'], nlat=512, npts=8192, latseed=0)

    def polys(metric, skew_key):
        return [dict(winners[metric][method]['poly'], skew=winners[metric][method][skew_key])
                for method in ('cpu', 'gpu')]

    standard = scores.murphy(polys('standard_murphy', 'standard_skew'), comparison['n'], params)
    print('standard MurphyE GPU/CPU', standard[1][0] / standard[0][0], flush=True)
    params.params = settings['params']
    records = []
    # Stream 2 selected the finalists. Hold their coefficients/skews fixed here.
    for stream in (0, 1, 3):
        params.latseed = stream
        values = scores.lattice_scores(polys('lattice_score', 'lattice_skew'), comparison['n'], params)
        records.append(dict(stream=stream, cpu=values[0], gpu=values[1],
                            ratio=values[1][0] / values[0][0]))
        print('lattice stream', stream, 'GPU/CPU', records[-1]['ratio'], flush=True)
    ratios = [r['ratio'] for r in records]
    write_json(run / 'winner_validation.json', dict(comparison_sha256=digest(source),
        scorer_sha256=digest(binary), settings=settings,
        standard_murphy=dict(angles=params.eval_points, cpu=standard[0], gpu=standard[1],
                             ratio=standard[1][0] / standard[0][0]),
        lattice=dict(lattices=params.nlat, samples=params.npts, records=records,
                     min_ratio=min(ratios), max_ratio=max(ratios), mean_ratio=statistics.mean(ratios)),
        note='Previously selected winners and skews held fixed. Stream 0 overlaps the initial screen; '
             'streams 1 and 3 are additional streams. These are model checks, not test sieving.'))


if __name__ == '__main__':
    main()
