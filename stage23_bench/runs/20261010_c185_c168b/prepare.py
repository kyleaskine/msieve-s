"""prepare.py SRC_PIPELINE_RESULTS DIR: seeds from the CPU pipeline's best300_msieve.ms
(one-based re-sopt rank, plus content seeds), a frozen s23_ropt copy, and a manifest."""
import hashlib, json, shutil, sys
from pathlib import Path
ROOT = Path.home() / 'msieve-s'
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import cado_poly_text, resultant
from content_seeds import content_seeds

src, here = Path(sys.argv[1]), Path(sys.argv[2])
n = None
for line in (src / 'pipeline_report.txt').read_text().splitlines():
    if line.startswith('n: '):
        n = int(line.split()[1]); break
(here / 'seeds').mkdir(parents=True)
(here / 'bin').mkdir()
seeds, todo = [], []
for rank, line in enumerate((src / 'best300_msieve.ms').read_text().splitlines(), 1):
    f = line.split()
    assert len(f) == 11
    p = dict(n=n, Y1=int(f[6]), Y0=int(f[7]), **{f'c{i}': int(f[5-i]) for i in range(6)})
    assert resultant(p) != 0 and resultant(p) % n == 0
    stem = f's{rank:04d}'
    (here / 'seeds' / (stem + '.poly')).write_text(cado_poly_text(p))
    seeds.append(dict(search=stem, seed_rank=rank, content=1, exp_E=float(f[9])))
    for content, u, v, q in content_seeds(p, n):
        name = f'{stem}_d{content}'
        (here / 'seeds' / (name + '.poly')).write_text(cado_poly_text(q))
        seeds.append(dict(search=name, seed_rank=rank, content=content, u=u, v=v, exp_E=float(f[9])))
(here / 'todo_breadth.txt').write_text(''.join(s['search'] + '\n' for s in seeds))
(here / 'seeds.json').write_text(json.dumps(seeds, indent=2) + '\n')
shutil.copy2(ROOT / 'stage23_gpu/build/s23_ropt', here / 'bin/s23_ropt')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = dict(n=n, source=str(src), seed_numbering='one-based CPU re-sopt rank',
                searches=len(seeds), source_sha256={f.name: sha(f) for f in sorted(src.iterdir()) if f.is_file()},
                s23_ropt_sha256=sha(here / 'bin/s23_ropt'))
(here / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'{here.name}: n has {len(str(n))} digits, 300 parent seeds, {len(seeds)-300} content seeds')
