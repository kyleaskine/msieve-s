#!/usr/bin/env python3
"""Audit MurphyE/lattice quality of the intermediate alpha screen's lost finalists.

CPU-only. Freeze the scorer, score the union of the prior replay exports, then
validate leading retained/lost configurations with fixed skews and fresh samples.
The C171 parameters are the user's actual bounds; C208 uses its archived model.
No translation refinement or actual test sieving is performed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import cado_murphy_binary, cado_poly_text, read_cado_blocks, resultant

REPLAY = ROOT / 'stage23_gpu/runs/20261016_alpha_screen/replay'
CONFIG = {
    'c147': dict(Bf=2**30, Bg=2**29, area=2**27 * 10000000, lattice='14,8192',
                 q=[10000000 + i * 20000000 / 6 for i in range(7)]),
    'c166': dict(Bf=2**31, Bg=2**30, area=2**27 * 30000000, lattice='14,8192',
                 q=[30000000 + i * 10000000 for i in range(7)]),
    'c171': dict(Bf=2**32, Bg=2**31, area=2**27 * 40000000, lattice='14,8192',
                 q=[40000000, 62000000, 84000000, 106000000, 128000000, 150000000]),
    'c208': dict(Bf=2**34, Bg=2**33, area=2**32 * 80000000, lattice='17,32768',
                 q=[80000000 + i * 115000000 for i in range(9)]),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def coordinates(label):
    return tuple(map(int, re.search(r'\bu\s+(-?\d+)\s+v\s+(-?\d+)', label).groups()))


def run_score(binary, folder, name, source, config, threads, validation_seed=None):
    extra = (['-K', '16000', '-Keval', '64000', '-nlat', '128', '-npts', '2048', '-latseed', '11']
             if validation_seed is None else
             ['-K', '16000', '-Keval', '256000', '-nlat', '512', '-npts', '4096',
              '-latseed', str(validation_seed), '-useskew'])
    cmd = [str(binary), '-t', str(threads), '-Bf', str(config['Bf']), '-Bg', str(config['Bg']),
           '-area', str(config['area']), '-lattice', config['lattice'],
           '-qpoints', ','.join(map(str, config['q'])), *extra, str(source)]
    path = folder / (name + '.json')
    identity = dict(command=cmd, binary_sha256=digest(binary), input_sha256=digest(source))
    if path.exists():
        data = json.loads(path.read_text())
        if any(data.get(k) != v for k, v in identity.items()):
            raise ValueError(f'incompatible cached score: {path}')
        return data['rows']
    start = time.monotonic()
    with (folder / (name + '.tsv')).open('w') as out, (folder / (name + '.stderr')).open('w') as err:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=err)
    if proc.returncode:
        raise RuntimeError(f'{name}: scorer exit {proc.returncode}; see {folder}')
    rows = [list(map(float, line.split())) for line in (folder / (name + '.tsv')).read_text().splitlines()]
    expected = len(read_cado_blocks(source))
    if len(rows) != expected or any(int(row[0]) != i or len(row) != 4 + len(config['q']) or
            not all(math.isfinite(x) and x > 0 for x in row[1:]) for i, row in enumerate(rows)):
        raise ValueError(f'{name}: invalid scores')
    dump(path, dict(**identity, seconds=time.monotonic()-start, rows=rows))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, default=ROOT / 'stage23_gpu/runs/20261017_alpha_quality')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--replay-dir', type=Path, default=REPLAY)
    ap.add_argument('--exact-size', action='store_true', help='also score screen_exact exports')
    ap.add_argument('--cases', nargs='*', help='default: c208_s0010 and all 16 C171 depth cases')
    args = ap.parse_args()
    if args.threads < 1:
        ap.error('--threads must be positive')
    dest = args.run_dir.resolve()
    replay = args.replay_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    binary = dest / 'cado_murphy'
    if not binary.exists():
        shutil.copy2(cado_murphy_binary(), binary)
    cases = (['c208_s0010'] if (replay / 'c208_wide_0.poly').exists() else [])
    cases += sorted(p.name.removesuffix('_wide.poly') for p in replay.glob('*_wide.poly'))
    if not cases:
        ap.error('no replay exports found')
    if args.cases:
        if set(args.cases) - set(cases):
            ap.error('unknown case')
        cases = args.cases
    summaries = []
    for case in cases:
        folder = dest / case
        folder.mkdir(exist_ok=True)
        family = case.split('_')[0]
        policies = ['wide', 'screen'] + (['narrow'] if family == 'c208' else [])
        if args.exact_size:
            policies.append('screen_exact')
        paths = ({name: replay / f'c208_{name}_0.poly' for name in policies}
                 if family == 'c208' else {name: replay / f'{case}_{name}.poly' for name in policies})
        entries, indices, members = [], {}, {}
        for policy, path in paths.items():
            members[policy] = []
            for label, poly in read_cado_blocks(path):
                cell = coordinates(label)
                if cell not in indices:
                    if not resultant(poly) or resultant(poly) % poly['n']:
                        raise ValueError(f'invalid resultant: {case} {cell}')
                    indices[cell] = len(entries)
                    entries.append(dict(cell=cell, poly=poly))
                elif cado_poly_text(poly) != cado_poly_text(entries[indices[cell]]['poly']):
                    raise ValueError(f'configuration differs between policies: {case} {cell}')
                members[policy].append(indices[cell])
        manifest = dict(case=case, config=CONFIG[family], members=members, entries=entries,
                        sources={key: dict(path=str(p), sha256=digest(p)) for key, p in paths.items()})
        dump(folder / 'manifest.json', manifest)
        source = folder / 'union.poly'
        source.write_text('\n'.join(cado_poly_text(e['poly']) for e in entries))
        print(case, 'scoring', len(entries), 'unique finalists', flush=True)
        rows = run_score(binary, folder, 'screen', source, CONFIG[family], args.threads)
        lost = set(members['wide']) - set(members['screen'])
        groups = dict(members, lost=sorted(lost))
        selected = set()
        for members_ in groups.values():
            for column in (1, 3):
                selected.update(sorted(members_, key=lambda i: (-rows[i][column], i))[:4])
        selected = sorted(selected)
        frozen = [dict(entries[i]['poly'], skew=rows[i][2]) for i in selected]
        val_source = folder / 'validation.poly'
        val_source.write_text('\n'.join(cado_poly_text(p) for p in frozen))
        validation = {}
        for seed in (21, 22, 23):
            raw = run_score(binary, folder, f'validation_{seed}', val_source, CONFIG[family], args.threads, seed)
            validation[seed] = {idx: row for idx, row in zip(selected, raw)}
        summary = dict(case=case, unique_finalists=len(entries), lost_finalists=len(lost),
                       validated_configurations=len(selected), metrics={})
        for name, column in [('lattice', 1), ('murphy', 3)]:
            def score(i):
                return sum(validation[s][i][column] for s in validation) / len(validation)
            winners = {policy: max(set(ids) & set(selected), key=score) for policy, ids in groups.items() if ids}
            wide, trial = winners['wide'], winners['screen']
            summary['metrics'][name] = dict(
                winner_cells={policy: entries[i]['cell'] for policy, i in winners.items()},
                best_scores={policy: score(i) for policy, i in winners.items()},
                wide_winner_retained=wide in members['screen'],
                screen_vs_wide_percent=100 * (score(trial) / score(wide) - 1),
                fixed_winner_ratios_by_stream=[100 * (validation[s][trial][column] /
                                                      validation[s][wide][column] - 1) for s in validation],
                lost_vs_wide_percent=(100 * (score(winners['lost']) / score(wide) - 1) if lost else None))
            if 'screen_exact' in winners:
                exact = winners['screen_exact']
                summary['metrics'][name]['exact_size'] = dict(
                    wide_winner_retained=wide in members['screen_exact'],
                    versus_wide_percent=100 * (score(exact) / score(wide) - 1),
                    fixed_winner_ratios_by_stream=[100 * (validation[s][exact][column] /
                                                         validation[s][wide][column] - 1) for s in validation])
        dump(folder / 'summary.json', summary)
        summaries.append(summary)
        dump(dest / 'summary.json', dict(cases=summaries, config=CONFIG, scorer_sha256=digest(binary),
             limitations='Audit of saved top-200 exports, not entire search spaces or all pre-refinement pools. '
                         'Skew searched at 16k angles, then frozen for validation at 256k and three fresh lattice streams. '
                         'No translation search. Uniform-root lattice model omits actual lim/mfb/lambda acceptance and '
                         'execution cost. C171 q band 40M-150M remains provisional. No test sieving.'))
        print(case, {m: round(v['screen_vs_wide_percent'], 5) for m, v in summary['metrics'].items()}, flush=True)


if __name__ == '__main__':
    main()
