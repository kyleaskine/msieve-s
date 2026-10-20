#!/usr/bin/env python3
"""Check the small C192 GPU/CADO model gap on additional lattice streams."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'stage23_bench/tools'))
from gpu_pipeline import write_json, digest
import score_polys as scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'stage23_gpu/runs/20261019_c192')
    run = parser.parse_args(argv).run_dir.resolve()
    source = run/'comparison/summary.json'
    comparison = json.loads(source.read_text())
    rows = comparison['rows']
    methods = ['cado_orig', 'gpu_pipeline']
    winners = {metric: [max((r for r in rows if r['method']==method), key=lambda r:r[metric])
                      for method in methods] for metric in ('standard_murphy', 'lattice_score')}
    scores.cado_murphy_binary = lambda: str(run/'bin/cado_murphy')
    params = SimpleNamespace(params='default', points=16000, eval_points=1024000, trans=0,
        threads=4, useskew=True, lattice='15,16384', qband='80e6,500e6,7', nlat=512, npts=8192, latseed=0)
    standard = scores.murphy([dict(r['poly'], skew=r['standard_skew']) for r in winners['standard_murphy']],
                             comparison['n'], params)
    records = []
    params.params = 'job:32,33,15,80e6'
    for stream in (0, 1, 3):
        params.latseed = stream
        values = scores.lattice_scores([dict(r['poly'], skew=r['lattice_skew']) for r in winners['lattice_score']],
                                       comparison['n'], params)
        records.append(dict(stream=stream, cado=values[0], gpu=values[1], ratio=values[1][0]/values[0][0]))
        print('stream', stream, 'GPU/CADO', records[-1]['ratio'], flush=True)
    write_json(run/'winner_validation.json', dict(comparison_sha256=digest(source),
        scorer_sha256=digest(run/'bin/cado_murphy'), winners=winners,
        standard_murphy=dict(angles=1024000, cado=standard[0], gpu=standard[1], ratio=standard[1][0]/standard[0][0]),
        lattice=dict(lattices=512, samples=8192, records=records),
        note='Skews frozen before these additional samples. Model checks, no test sieving.'))


if __name__ == '__main__':
    main()
