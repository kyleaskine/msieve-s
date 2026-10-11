#!/usr/bin/env python3
"""compare_ab.py DIR A B: the breadth outputs of two binaries on the same searches (DIR/A/out,
DIR/B/out): byte-identical .poly files, the best Ea and its cell per search, the written cell sets,
and the time split summed over searches (from each log)."""
import re, sys
from pathlib import Path
G, A, B = Path(sys.argv[1]), sys.argv[2], sys.argv[3]

def cells(p):
    """the written cells in order, from the .poly labels: (u, v, Ea)"""
    out = []
    for line in p.read_text().splitlines():
        m = re.match(r'# u (-?\d+) v (-?\d+) t (-?\d+) Ea ([-0-9.]+)', line)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), float(m.group(4))))
    return out

def split(log):
    tot = dict(wall=0.0, sieve=0.0, size=0.0, refine=0.0, cells=0.0, n=0)
    for line in (G / log).read_text().splitlines():
        if ' rc=' not in line:
            continue
        tot['n'] += 1
        tot['wall'] += float(re.search(r'wall=(-?[0-9.]+)', line).group(1))
        tot['sieve'] += float(re.search(r'sieve ([0-9.]+) s', line).group(1))
        tot['size'] += float(re.search(r'size model ([0-9.]+) s', line).group(1))
        tot['refine'] += float(re.search(r'refine ([0-9.]+) s', line).group(1))
        tot['cells'] += float(re.search(r'([0-9.e+]+) cells', line).group(1))
    total = re.search(r'TOTAL ([0-9.]+)', (G / log).read_text())
    tot['TOTAL'] = float(total.group(1)) if total else None
    return tot

same = diff = 0
best_same = best_diff = 0
worst = 0.0
rows = []
for pa in sorted((G / A / 'out').glob('*.poly')):
    pb = G / B / 'out' / pa.name
    if not pb.exists():
        continue
    if pa.read_bytes() == pb.read_bytes():
        same += 1
        continue
    diff += 1
    ca, cb = cells(pa), cells(pb)
    if ca and cb and ca[0][:2] == cb[0][:2]:
        best_same += 1
    else:
        best_diff += 1
    if ca and cb:
        worst = max(worst, cb[0][2] - ca[0][2])
        rows.append((pa.stem, ca[0], cb[0], len(set(c[:2] for c in ca) & set(c[:2] for c in cb)), len(ca)))
print(f'{same} searches byte-identical, {diff} differ; of those, best cell same {best_same}, different {best_diff}; '
      f'worst best-Ea change (B - A, + = B worse) {worst:+.6f}')
for r in rows[:20]:
    print('  ', r)
for tag in (A, B):
    t = split(f'{tag}/log.txt')
    print(tag, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in t.items()})
