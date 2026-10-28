#!/usr/bin/env python3
"""Preserve completed coverage measurements, winner coefficients and provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--archive-dir', type=Path, required=True)
    args = parser.parse_args()
    run, dest = args.run_dir.resolve(), args.archive_dir.resolve()
    timing = read(run/'pilot/summary.json')
    scoring = read(run/'scores/summary.json')
    plans = read(run/'summary.json')
    replay = read(run/'validation/replay_summary.json')
    losses = read(run/'validation/quality_trace.json')
    if plans['completed'] != plans['expected'] or not timing['repeat_outputs_identical']:
        raise ValueError('incomplete study')
    if not replay['all_bytes_equal']:
        raise ValueError('CPU replay verification failed')
    if not losses['complete']:
        raise ValueError('finish tracing the scored quality losses')
    times = {(r['case'],r['mode']):r for r in timing['medians']}
    rows = []
    for result in scoring['results']:
        base = result['winners']['dense_breadth']
        for mode, winner in result['winners'].items():
            row = dict(times[result['case'],mode])
            row['murphy_over_dense'] = winner['standard']['standard_murphy']/base['standard']['standard_murphy']
            row['lattice_over_dense'] = {
                s:winner['lattice']['lattice_validation'][s]['score']/base['lattice']['lattice_validation'][s]['score']
                for s in ('2','3')}
            rows.append(row)
    summary = dict(run=str(run), results=rows, cases=read(run/'prepared.json')['cases'],
        plans_completed=plans['completed'], verified_exports=timing['verified_exports'],
        cpu_replay_exports=replay['verified_exports'],
        quality_losses=[{k:r[k] for k in ('case','mode','murphy_over_dense','target','first_loss','pool_rank')}
                        for r in losses['records']],
        repeat_outputs_identical=True, gpu_concurrency=1, host_threads=2,
        caveats=['Timing/quality comparison, not equal elapsed budgets or a full pipeline.',
                 'Four new-input seeds were selected before evaluating the policy.',
                 'Lattice scores use provisional settings; no test sieving.',
                 'Reference coordinates only trace losses; they never force selection.'])
    files = {run/p for p in ('prepared.json','summary.json','pilot/summary.json','pilot/gpu.txt',
        'pilot/records.json','scores/summary.json')}
    for pattern in ('seeds/*.poly','plans/*/*.json','pilot/*_r0.poly','pilot/*_r0.log',
                    'scores/*_scoring.json','validation/*.json','validation/*.log',
                    'source/include/*','source/tools/*'):
        files.update(run.glob(pattern))
    for row in scoring['results']:
        files.add(run/'scores'/(row['case']+'.json'))
    for path in files:
        if not path.is_file():
            raise ValueError('missing artifact: '+str(path))
    dest.mkdir(parents=True,exist_ok=False)
    hashes = {}
    for path in sorted(files):
        rel = path.relative_to(run)
        target = dest/'evidence'/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        expected = digest(path)
        shutil.copy2(path,target)
        if digest(target)!=expected:
            raise ValueError('copy mismatch: '+str(path))
        hashes[str(target.relative_to(dest))]=expected
    harness = read(run/'prepared.json')['harness_sha256']
    for name in ('archive.py','validate.py','trace_quality.py'):
        path = Path(__file__).with_name(name)
        harness[str(path)] = digest(path)
    for name, expected in harness.items():
        path = Path(name)
        if digest(path)!=expected:
            raise ValueError('harness changed: '+name)
        target = dest/'harness'/path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path,target)
        if digest(target)!=expected:
            raise ValueError('harness copy mismatch: '+name)
        hashes[str(target.relative_to(dest))]=expected
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    hashes['summary.json']=digest(dest/'summary.json')
    (dest/'manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(dest,len(hashes),'verified files')


if __name__=='__main__':
    main()
