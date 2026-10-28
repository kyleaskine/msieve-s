#!/usr/bin/env python3
"""Archive compact evidence from a completed new-job comparison to a fresh directory."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(run, dest):
    if read(run / 'status.json')['stage'] != 'complete':
        raise ValueError('pipeline is incomplete')
    execution = read(run / 'pipeline_execution.json')
    audit = read(run / 'migration_audit.json')
    if (execution['returncode'] or audit['initial_sopt']['worse_at_full_precision']
            or not audit['depth_cpu_gpu_bytes_equal'] or not audit['depth_exports_verified']):
        raise ValueError('pipeline or migration audit failed')
    comparisons = ['comparison']
    if (run / 'comparison_profiles.json').exists():
        comparisons += ['comparison_' + p for p in read(run / 'comparison_profiles.json')['profiles']]
    scores = {name: read(run / name / 'summary.json') for name in comparisons}
    results = read(run / 'results/summary.json')
    summary = dict(
        run=str(run), n=results['n'], input_polynomials=results['input_polynomials'],
        searches=results['searches'], pipeline_seconds=execution['seconds'],
        partial_murphy_searches=results['partial_murphy_searches'],
        initial_sopt={k: v for k, v in audit['initial_sopt'].items() if k != 'differences'},
        initial_sopt_max_abs_objective_difference=max(
            (abs(row['gain']) for row in audit['initial_sopt']['differences']), default=0),
        initial_sopt_cpu_fallbacks=len(re.findall(
            r'poly \d+ redone by CADO', (run / 'logs/sopt_e0.log').read_text())),
        depth_exports_verified=audit['depth_exports_verified'],
        depth_cpu_gpu_bytes_equal=audit['depth_cpu_gpu_bytes_equal'],
        comparisons={name: dict(profile_id=s['settings']['profile_id'],
            screened=s['screened'], validated=len(s['finalists']), validation=s['validation'],
            gpu_over_cpu={metric: w['ratio'] for metric, w in s['winners'].items()})
            for name, s in scores.items()},
        limitations=[
            'Model comparison of shortlisted CPU/GPU exports; no test sieving.',
            'Wall time includes GPU initial sopt; no matched full CPU timing.',
            'Alternative profiles rescore the same finalists without repeating seed selection.',
            'Compact evidence is not a standalone replay bundle; full inputs and pools remain in the run.'])
    files = {run / name for name in ('prepared.json', 'settings.json', 'execution.json',
        'scoring.json', 'status.json', 'migration_audit.json', 'pipeline.log',
        'inputs/sieve_profiles.json', 'logs/sopt_e0.log', 'logs/resopt.log')}
    for pattern in ('*_command.json', '*_execution.json', 'bin/*.json',
                    'results/*', 'depth_audit/*.json', 'depth_audit/*.log',
                    'comparison_profiles.json', 'settings_update_20261027.json'):
        files.update(run.glob(pattern))
    for name in comparisons:
        files.update(run / name / p for p in ('summary.json', 'scoring.json'))
        files.update((run / name).glob('*.poly'))
    for path in files:
        if not path.is_file():
            raise ValueError('missing evidence: ' + str(path))
    dest.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for path in sorted(files):
        rel = path.relative_to(run)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = digest(path)
        shutil.copy2(path, target)
        if digest(target) != expected:
            raise ValueError('archive copy changed: ' + str(rel))
        manifest[str(rel)] = expected
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest['summary.json'] = digest(dest / 'summary.json')
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(dest, len(manifest), 'verified files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    save(args.run_dir.resolve(), args.destination.resolve())
