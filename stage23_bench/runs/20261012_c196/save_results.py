#!/usr/bin/env python3
"""save_results.py RUN: preserve the completed C196 comparison outside build/."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import cado_poly_text, read_cado_blocks, rotation_between

run = Path(sys.argv[1]).resolve()
assert json.loads((run / 'comparison_status.json').read_text())['stage'] == 'comparison complete'
comparison = json.loads((run / 'comparison.json').read_text())
manifest = json.loads((run / 'manifest.json').read_text())
timings = json.loads((run / 'timings.json').read_text())
commands = json.loads((run / 'comparison_commands.json').read_text())
polys = dict(read_cado_blocks(run / 'finalists.poly'))
best = {}
for row in comparison['rows']:
    best.setdefault(row['label'].split()[0], row)

dest = ROOT / 'stage23_bench/data/c196'
dest.mkdir(exist_ok=False)
(dest / 'gpu_ropt').mkdir()
(dest / 'winners').mkdir()
(dest / 'n.txt').write_text(str(manifest['n']) + '\n')
for method, name in [('gpu_murphy', 'gpu_murphy_best'), ('gpu', 'gpu_breadth_best'),
                     ('msieve_orig', 'msieve_best'), ('cado_orig', 'cado_best')]:
    row = best[method]
    p = polys[row['label']]
    (dest / f'winners/{name}.poly').write_text('# ' + row['label'] + '\n' +
        '# Skew from the independent 16k search / 256k evaluation; see gpu_ropt/summary.json.\n' + cado_poly_text(p))
for source, target in [('finalists.poly', 'finalists.poly'), ('comparison.json', 'finalist_scores.json'),
                       ('sopt/verification.json', 'sopt_verification.json'),
                       ('sopt/e50_verification.log', 'e50_verification.txt'),
                       ('export_validation.json', 'export_validation.json'),
                       ('breadth_analysis.log', 'breadth_analysis.txt')]:
    shutil.copy2(run / source, dest / 'gpu_ropt' / target)

families = {}
for kind in ('baseline', 'gpu'):
    for line in (run / f'scored/{kind}.tsv').read_text().splitlines():
        band, skew, e, y1, label = line.split('\t')
        seed = int(re.search(r'\bseed s?(\d+)', label)[1])
        row = families.setdefault(seed, {})
        if float(band) > row.get(kind, (0,))[0]:
            row[kind] = float(band), label
with (dest / 'gpu_ropt/families.tsv').open('w') as out:
    out.write('seed\tcpu_band\tgpu_band\tgpu_over_cpu\tcpu_label\tgpu_label\n')
    for seed, row in sorted(families.items()):
        if 'baseline' in row and 'gpu' in row:
            a, b = row['baseline'], row['gpu']
            out.write(f'{seed}\t{a[0]}\t{b[0]}\t{b[0]/a[0]}\t{a[1]}\t{b[1]}\n')

murphy = []
for path in sorted((run / 'murphy').glob('*.log')):
    text = path.read_text()
    murphy.append(dict(seed=path.stem,
        result=re.search(r'^# best 3 by validated MurphyE: (.*)$', text, re.M)[1],
        timing=re.search(r'^# s23_ropt: (.*)$', text, re.M)[1]))
summary = dict(date='2026-10-12', run=str(run), manifest=manifest, scoring=comparison['scoring'],
    full_raw_count=manifest['settings']['raw_polynomials'], parent_seeds=300, content_seeds=manifest['searches']-300,
    same_top2000_as_cpu=(run/'cpu/top_input.ms').read_bytes()==(run/'sopt/top_input.ms').read_bytes(),
    effort50_validation='2000/2000 coefficient-identical to CADO; sopt_compare.py --rescore passed.',
    sopt_seconds=sum(c['elapsed_seconds'] for c in timings['commands'] if c['name'].startswith('sopt/')),
    breadth_seconds=timings['breadth_seconds'], breadth_searches=manifest['searches'],
    breadth_failures=sum(c['returncode'] != 0 for c in timings['commands'] if c['name'].startswith('breadth/')),
    comparison_commands=commands, murphy= murphy, finalists=best,
    gpu_over_msieve_band=best['gpu_murphy']['band']/best['msieve_orig']['band'],
    gpu_over_msieve_murphy=best['gpu_murphy']['murphy']/best['msieve_orig']['murphy'],
    transform_msieve_to_gpu=rotation_between(polys[best['msieve_orig']['label']], polys[best['gpu_murphy']['label']]),
    validation='make test passed: 25 polynomials, 2100 exact scalar CADO comparisons; GPU/CPU C196 smoke outputs identical.',
    interpretation='Essential tie in the models; no test-sieve gain established. qmax 450M is provisional.',
    timing_note='Wall times under shared-host load; WSL clocks can step. No matched CPU speedup claim.',
    cpu_provenance=json.loads((run/'cpu/provenance.json').read_text()))
summary['sha256'] = {name: hashlib.sha256((run/name).read_bytes()).hexdigest() for name in
    ('preflight_test.log','comparison.json','finalists.poly','timings.json','export_validation.json')}
(dest / 'gpu_ropt/summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(dest)
print('sopt+breadth seconds:', summary['sopt_seconds'] + summary['breadth_seconds'])
print('comparison seconds:', sum(c['elapsed_seconds'] for c in commands))
