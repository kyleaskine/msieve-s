#!/usr/bin/env python3
"""score_pass2.py DIR PARAMS LATTICE QBAND [THREADS]: score pass 2's outputs (first TOP=32 of each
search in DIR/pass2/out) as score.py scores the breadth pass, into DIR/scored/pass2.{poly,tsv}, and
print the best per method over baseline, breadth and pass 2 (band ratio to the overall best)."""
import os, sys, types
from pathlib import Path
sys.path.insert(0, str(Path.home() / 'msieve-s/stage23_bench/tools'))
from score_polys import read_blocks, lattice_scores

G = Path(sys.argv[1]).resolve()
A = types.SimpleNamespace(threads=int(sys.argv[5]) if len(sys.argv) > 5 else 14, params=sys.argv[2],
                          points=4000, eval_points=16000, lattice=sys.argv[3], qband=sys.argv[4], useskew=False)
TOP = int(os.environ.get('TOP', 32))
items = []
for s in (G / 'pass2/todo.txt').read_text().split():
    blocks = read_blocks(str(G / 'pass2/out' / (s + '.poly')))
    items += [(f'pass2 seed {s} rank {j} {lab}', p) for j, (lab, p) in enumerate(blocks[:TOP])]
N = str(items[0][1]['n'])
with open(G / 'scored/pass2.poly', 'w') as fh:
    for lab, d in items:
        fh.write(f'# {lab}\nn: {N}\nY0: {d["Y0"]}\nY1: {d["Y1"]}\n' + ''.join(f'c{i}: {d[f"c{i}"]}\n' for i in range(6)) + '\n')
res = lattice_scores([p for _, p in items], items[0][1]['n'], A)
with open(G / 'scored/pass2.tsv', 'w') as fh:
    for (lab, p), (b, sk, e, rel) in zip(items, res):
        fh.write(f"{b:.6e}\t{sk:.1f}\t{e:.6e}\t{p['Y1']}\t{lab}\n")
rows = []
for f in ('baseline', 'gpu', 'pass2'):
    for line in open(G / f'scored/{f}.tsv'):
        b, sk, e, y1, lab = line.rstrip('\n').split('\t')
        rows.append((float(b), float(e), lab))
top, etop = max(r[0] for r in rows), max(r[1] for r in rows)
best = {}
for b, e, lab in rows:
    m = lab.split()[0]
    if b > best.get(m, (0,))[0]:
        best[m] = (b, e, lab)
for m, (b, e, lab) in sorted(best.items(), key=lambda kv: -kv[1][0]):
    print(f'{m:12s} band {b / top:.5f}  MurphyE {e / etop:.5f}  {lab[:110]}')
print('pass 2 top 10:')
for b, e, lab in sorted([r for r in rows if r[2].startswith('pass2')], key=lambda r: -r[0])[:10]:
    print(f'  {b / top:.5f}  E {e / etop:.5f}  {lab[:110]}')
