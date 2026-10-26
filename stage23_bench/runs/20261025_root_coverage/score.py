#!/usr/bin/env python3
"""Independently score all 200 exports per pilot mode, then validate leaders.

Standard MurphyE uses 16k search/64k evaluation angles, then 256k at fixed skew.
The provisional job lattice uses stream 0 for screening and streams 2/3 for
validation. These model comparisons are not measured relation yields.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from audit import ROOT, digest, prepare, save
from pilot import CASES, MODES
from polyfmt import read_cado_blocks, resultant
import score_polys as scores


def key(poly):
    sign = 1 if poly['c5'] > 0 else -1
    return (poly['Y0'], poly['Y1'], *(sign*poly[f'c{i}'] for i in range(6)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    meta = prepare(run)
    # Prevent concurrent CPU ranking from contaminating paired process timings.
    timing = json.loads((run/'pilot/summary.json').read_text())
    if not timing['repeat_outputs_identical']:
        raise ValueError('complete the pilot first')
    folder = run/'scores'
    folder.mkdir(exist_ok=True)
    binary = folder/'cado_murphy'
    if not binary.exists():
        shutil.copy2(ROOT/'stage23_gpu/runs/20261024_c197_gpu/bin/cado_murphy', binary)
    scores.cado_murphy_binary = lambda: str(binary)
    all_results = []
    for name in CASES:
        case = next(c for c in meta['cases'] if c['name']==name)
        origin = '20261019_c192' if name.startswith('c192') else '20261024_c197_gpu'
        settings_file = folder/(name+'_scoring.json')
        if not settings_file.exists():
            shutil.copy2(ROOT/'stage23_gpu/runs'/origin/'scoring.json', settings_file)
        settings = json.loads(settings_file.read_text())
        files = {mode:run/'pilot'/f'{name}_{mode}_r0.poly' for mode in MODES}
        provenance = {str(p.relative_to(run)):digest(p) for p in [binary, settings_file, *files.values()]}
        provenance['script_sha256'] = digest(__file__)
        for script in ('score_polys.py', 'polyfmt.py', 'rescore.py'):
            provenance[script] = digest(ROOT/'stage23_bench/tools'/script)
        cache = folder/(name+'.json')
        if cache.exists():
            result = json.loads(cache.read_text())
            if result['provenance'] != provenance:
                raise ValueError(f'scoring inputs changed: {cache}')
        else:
            rows = []
            for mode, file in files.items():
                blocks = read_cado_blocks(file)
                if len(blocks) != 200:
                    raise ValueError(f'need every export: {file}')
                for index, (label, poly) in enumerate(blocks):
                    r = resultant(poly)
                    if poly['n'] != case['n'] or not r or r % case['n']:
                        raise ValueError(f'invalid scoring input: {file}')
                    rows.append(dict(mode=mode, index=index, label=label, poly=poly))
            unique = {key(r['poly']):r['poly'] for r in rows}
            order = list(unique)
            polys = list(unique.values())
            params = SimpleNamespace(params='default', threads=2, points=16000, eval_points=64000,
                trans=0, useskew=False, lattice=settings['lattice'], qband=settings['qband'],
                nlat=32, npts=2048, latseed=0)
            print(name, 'scoring', len(polys), 'unique exports', flush=True)
            start = time.monotonic()
            standard = scores.murphy(polys, case['n'], params)
            params.params = settings['params']
            lattice = scores.lattice_scores(polys, case['n'], params)
            values = {k:dict(standard_murphy=m[0], standard_skew=m[1], lattice_score=l[0], lattice_skew=l[1])
                      for k,m,l in zip(order, standard, lattice)}
            for row in rows:
                row.update(values[key(row['poly'])])
            screen_seconds = time.monotonic()-start
            save(folder/(name+'_screen.json'), dict(provenance=provenance, rows=rows, seconds=screen_seconds))
            chosen = set()
            for mode in MODES:
                for metric in ('standard_murphy', 'lattice_score'):
                    chosen.update(sorted((i for i,r in enumerate(rows) if r['mode']==mode),
                                         key=lambda i:-rows[i][metric])[:4])
            leaders = [dict(rows[i]) for i in sorted(chosen)]
            print(name, 'validating', len(leaders), 'leaders', flush=True)
            start = time.monotonic()
            params.params='default'; params.eval_points=256000; params.useskew=True
            standard = scores.murphy([dict(r['poly'],skew=r['standard_skew']) for r in leaders],case['n'],params)
            for row, value in zip(leaders,standard):
                row['standard_murphy']=value[0]
            params.params=settings['params']; params.nlat=256; params.npts=4096
            for stream in (2,3):
                params.latseed=stream
                lattice=scores.lattice_scores([dict(r['poly'],skew=r['lattice_skew']) for r in leaders],case['n'],params)
                for row,value in zip(leaders,lattice):
                    row.setdefault('lattice_validation',{})[str(stream)]=dict(score=value[0],per_q=value[3])
            winners={}
            for mode in MODES:
                part=[r for r in leaders if r['mode']==mode]
                standard_best=max(part,key=lambda r:r['standard_murphy'])
                # Stream 2 chooses the lattice leader; stream 3 checks that fixed choice.
                lattice_best=max(part,key=lambda r:r['lattice_validation']['2']['score'])
                winners[mode]=dict(standard=standard_best, lattice=lattice_best)
            result=dict(case=name,provenance=provenance,settings=settings,unique_screened=len(polys),
                finalists=leaders,winners=winners,screen_seconds=screen_seconds,
                validation_seconds=time.monotonic()-start,note=__doc__,
                caveat='Equal 4096-candidate refinement and 200-export scoring quotas. '
                       'Timing/quality pilot, not an equal-elapsed-time or holdout study. '
                       'Lattice skew is optimized for job-bounds MurphyE, not directly for lattice yield.')
            save(cache,result)
        all_results.append(result)
        base=result['winners']['dense_breadth']
        for mode,winners in result['winners'].items():
            print(name,mode,'Murphy/dense',winners['standard']['standard_murphy']/base['standard']['standard_murphy'],
                  'lattice/dense',[winners['lattice']['lattice_validation'][s]['score']/
                                   base['lattice']['lattice_validation'][s]['score'] for s in ('2','3')],flush=True)
    save(folder/'summary.json',dict(results=all_results,note=__doc__))


if __name__ == '__main__':
    main()
