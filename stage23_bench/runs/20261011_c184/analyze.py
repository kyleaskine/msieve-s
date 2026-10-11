#!/usr/bin/env python3
"""analyze.py DIR: GPU breadth vs the CPU pipeline from DIR/scored/{baseline,gpu}.tsv
(band = lattice-aware relations total, the primary score; MurphyE at job params)."""
import collections, re, sys
from pathlib import Path

G = Path(sys.argv[1])


def rows(name):
    out = []
    for line in (G / 'scored' / name).read_text().splitlines():
        band, skew, e, y1, lab = line.split('\t')
        if lab.startswith('gpu'):
            m = re.match(r'gpu seed s(\d+)(_d\d+)? rank (\d+)', lab)
            method, seed, rank = 'gpu' + ('_content' if m.group(2) else ''), int(m.group(1)), int(m.group(3))
        else:
            method, seed, rank = lab.split()[0], int(lab.split()[2]), None
        out.append(dict(band=float(band), skew=float(skew), e=float(e), y1=y1, lab=lab,
                        method=method, seed=seed, rank=rank))
    return out


base, gpu = rows('baseline.tsv'), rows('gpu.tsv')
allr = base + gpu
best = max(allr, key=lambda r: r['band'])
print(f'{G.name}: {len(base)} CPU-pipeline polys, {len(gpu)} GPU polys; ratios to the best band {best["band"]:.6g}')
print('\nbest per method (band ratio, MurphyE ratio to the best MurphyE):')
beste = max(r['e'] for r in allr)
by = collections.defaultdict(list)
for r in allr:
    by[r['method'].replace('_content', '')].append(r)
for m, L in sorted(by.items(), key=lambda kv: -max(r['band'] for r in kv[1])):
    b = max(L, key=lambda r: r['band'])
    e = max(L, key=lambda r: r['e'])
    print(f'  {m:12s} band {b["band"] / best["band"]:.4f} (seed {b["seed"]})   MurphyE {e["e"]:.4e} = {e["e"] / beste:.4f} (seed {e["seed"]})')
print('\ntop 12 overall by band:')
for r in sorted(allr, key=lambda r: -r['band'])[:12]:
    print(f'  {r["band"] / best["band"]:.4f}  E {r["e"]:.4e}  skew {r["skew"]:12.0f}  {r["lab"][:110]}')
# seed families: best GPU vs best CPU (either tool, either pass) per parent seed
cpu_seed, gpu_seed = {}, {}
for r in base:
    if r['seed'] > 0 and r['band'] > cpu_seed.get(r['seed'], (0,))[0]:
        cpu_seed[r['seed']] = (r['band'], r['method'])
for r in gpu:
    if r['band'] > gpu_seed.get(r['seed'], (0,))[0]:
        gpu_seed[r['seed']] = (r['band'], r['rank'])
common = sorted(set(cpu_seed) & set(gpu_seed))
ratio = {s: gpu_seed[s][0] / cpu_seed[s][0] for s in common}
print(f'\nper seed family, GPU best / CPU best (band), {len(common)} seeds with both:')
print(f'  >1% better {sum(x > 1.01 for x in ratio.values())}, within 1% {sum(0.99 <= x <= 1.01 for x in ratio.values())}, '
      f'>1% worse {sum(x < 0.99 for x in ratio.values())}')
top = sorted(common, key=lambda s: -max(cpu_seed[s][0], gpu_seed[s][0]))[:10]
print('  the 10 best families (seed: CPU best ratio [method], GPU best ratio [proxy rank of its best], GPU/CPU):')
for s in top:
    print(f'    seed {s:3d}: CPU {cpu_seed[s][0] / best["band"]:.4f} [{cpu_seed[s][1]}]  GPU {gpu_seed[s][0] / best["band"]:.4f} '
          f'[rank {gpu_seed[s][1]}]  {ratio[s]:.4f}')
worst = sorted(common, key=lambda s: ratio[s])[:5]
print('  largest GPU losses: ' + ', '.join(f'seed {s} {ratio[s]:.3f}' for s in worst))
print(f'  GPU winning rank: max proxy rank of a family best = {max(r for _, r in gpu_seed.values())}')
