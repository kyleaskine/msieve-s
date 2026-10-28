#!/usr/bin/env python3
"""Bounded size-aware proposals: frozen loss corpus, independent seeds, GPU quality/timing.

Reuse the October 25 planning, timing and independent scoring harnesses with
explicit study configurations. Old run directories and their defaults stay intact.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage23_bench/runs/20261025_root_coverage'))
import audit

VARIANTS = {name: audit.VARIANTS[name] for name in ('dense_breadth', 'classes2520_32')}
for count in (16, 32):
    VARIANTS[f'size2520_{count}'] = [
        '-maxcells', '2e11', '-classes', '2520', str(count), '-class-size-aware']
audit.VARIANTS = VARIANTS
import pilot
pilot.MODES = tuple(VARIANTS)
pilot.CASES = ('c192_winner', 'c197_s0005', 'c197_s0001', 'c179_winner',
               'c170_s0009', 'c170_s0010', 'c184_s0001', 'c184_s0003')
import score
from polyfmt import read_cado_blocks, resultant

original_prepare = audit.prepare


def prepare(dest):
    fresh = not (dest / 'prepared.json').exists()
    meta = original_prepare(dest)
    if fresh:
        extra = [dict(name='c179_winner', source=ROOT / 'stage23_bench/data/c179/ropt_coverage/seed.poly',
            target=dict(u=-55, v=34188178), provenance='Previously diagnosed msieve-original-only winner')]
        for name in ('c170', 'c184'):
            run = ROOT / ('stage23_gpu/runs/20261027_' + name)
            # Declare the first two retained depth seeds before measuring this
            # policy; never select holdouts based on new-policy performance.
            for path in sorted((run / 'depth').glob('*.poly'))[:2]:
                label, _ = read_cado_blocks(path)[0]
                match = re.match(r'u (-?\d+) v (-?\d+)', label)
                if match is None:
                    raise ValueError('missing depth reference coordinates')
                extra.append(dict(name=name+'_'+path.stem, source=run/'seeds'/path.name,
                    target=dict(u=int(match[1]), v=int(match[2])),
                    provenance='First two depth seeds from the new input, unused for policy development'))
        for case in extra:
            source = case.pop('source')
            target = dest / 'seeds' / (case['name'] + '.poly')
            shutil.copy2(source, target)
            poly = read_cado_blocks(target)[0][1]
            value = resultant(poly)
            if not value or value % poly['n']:
                raise ValueError('invalid seed: '+str(source))
            case.update(source=str(source.relative_to(ROOT)), file=str(target.relative_to(dest)), n=poly['n'])
            meta['cases'].append(case)
            meta['sha256'][case['file']] = audit.digest(target)
        (dest / 'scores').mkdir()
        for name in pilot.CASES:
            if name.startswith('c179'):
                # No confirmed test-sieve settings for this input: standard
                # MurphyE is primary; the nearest historical lattice is provisional.
                sys.path.insert(0, str(ROOT / 'scripts'))
                import sieve_profiles
                profile = next(p for p in sieve_profiles.load() if p['id']=='c181_user_20261017')
                settings = sieve_profiles.scoring(profile)
            else:
                origin = ('20261019_c192' if name.startswith('c192') else
                          '20261024_c197_gpu' if name.startswith('c197') else '20261027_'+name[:4])
                settings = json.loads((ROOT/'stage23_gpu/runs'/origin/'scoring.json').read_text())
            path = dest/'scores'/(name+'_scoring.json')
            audit.save(path, settings)
            meta['sha256'][str(path.relative_to(dest))] = audit.digest(path)
        helper_files = [Path(__file__), *(Path(audit.__file__).parent / name for name in
                        ('audit.py', 'pilot.py', 'score.py'))]
        meta['harness_sha256'] = {str(p):audit.digest(p) for p in helper_files}
        meta['pilot_cases'] = list(pilot.CASES)
        meta['note'] = __doc__
        audit.save(dest/'prepared.json', meta)
    if meta.get('pilot_cases') != list(pilot.CASES):
        raise ValueError('different or incomplete study; use a fresh directory')
    for path, expected in meta['harness_sha256'].items():
        if audit.digest(path) != expected:
            raise ValueError('study harness changed: '+path)
    return meta


audit.prepare = pilot.prepare = score.prepare = prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'plan', 'pilot', 'score'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    dest = args.run_dir.resolve()
    meta = prepare(dest)
    if args.cases and set(args.cases) - {c['name'] for c in meta['cases']}:
        parser.error('unknown case')
    if args.phase == 'plan':
        audit.plans(dest, meta, args.cases, None)
        records = [json.loads(p.read_text()) for p in sorted((dest/'plans').glob('*/*.json'))]
        audit.save(dest/'summary.json',dict(cases=meta['cases'], records=records,
            completed=len(records), expected=len(meta['cases'])*len(VARIANTS), note=__doc__))
    elif args.phase in ('pilot', 'score'):
        if args.cases:
            parser.error('--cases only applies to planning')
        sys.argv = [sys.argv[0], '--run-dir', str(dest)]
        if args.phase == 'pilot':
            sys.argv += ['--repeats', str(args.repeats)]
            pilot.main()
        else:
            score.main()


if __name__ == '__main__':
    main()
