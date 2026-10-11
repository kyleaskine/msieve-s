#!/usr/bin/env python3
"""score.py DIR PARAMS LATTICE QBAND [THREADS]: score the CPU pipeline's root-optimized polys
(msieve best 3 per seed and pass, all CADO) and the GPU breadth outputs (first TOP=32 of each
search) with one cado_murphy -lattice pass (job MurphyE at its best skew, lattice band total).
Writes DIR/scored/{baseline,gpu}.{poly,tsv}; TSV: band, skew, MurphyE, Y1, label."""
import collections, json, os, re, sys, types
from pathlib import Path
sys.path.insert(0, str(Path.home() / 'msieve-s/stage23_bench/tools'))
from score_polys import read_blocks, lattice_scores

G = Path(sys.argv[1]).resolve()
A = types.SimpleNamespace(threads=int(sys.argv[5]) if len(sys.argv) > 5 else 14, params=sys.argv[2],
                          points=4000, eval_points=16000, lattice=sys.argv[3], qband=sys.argv[4], useskew=False)
TOP = int(os.environ.get('TOP', 32))
man = json.loads((G / 'manifest.json').read_text())
R, N = Path(man['source']), str(man['n'])
rank = {}
for s in json.loads((G / 'seeds.json').read_text()):
    if s['content'] == 1:
        d = dict(l.split(': ') for l in (G / 'seeds' / (s['search'] + '.poly')).read_text().split('\n') if ': ' in l)
        rank.setdefault(d['Y1'], s['seed_rank'])


def msieve(path, tag, per=3):
    by, cur = collections.defaultdict(list), None
    for line in open(path):
        line = line.strip()
        if line.startswith('# norm'):
            cur = {'e': float(re.search(r' e ([-+0-9.e]+)', line).group(1))}
        elif cur is not None and ':' in line:
            k, v = line.split(':', 1); cur[k.strip()] = v.strip()
            if k.strip() == 'Y1':
                by[cur['Y1']].append(cur); cur = None
    out = []
    for y1, L in by.items():
        L.sort(key=lambda d: -d['e'])
        out += [(f"{tag} seed {rank.get(y1, -1)} e {d['e']} j {j}", d) for j, d in enumerate(L[:per])]
    return out


def cado(path, tag):
    out, cur = [], None
    for line in open(path):
        line = line.rstrip()
        if line.startswith('### root-optimized polynomial'):
            cur = {}
        elif cur is not None and re.match(r'^(n|Y0|Y1|c\d): ', line):
            k, v = line.split(': ', 1); cur[k] = v
        elif cur is not None and 'MurphyE' in line:
            out.append((f"{tag} seed {rank.get(cur['Y1'], -1)} E {line.split('=')[-1]}", cur)); cur = None
    return out


def write(path, items):
    with open(path, 'w') as fh:
        for lab, d in items:
            fh.write(f'# {lab}\nn: {N}\nY0: {d["Y0"]}\nY1: {d["Y1"]}\n' +
                     ''.join(f'c{i}: {d[f"c{i}"]}\n' for i in range(6)) + '\n')


def score(path):
    blocks = read_blocks(str(path))
    res = lattice_scores([p for _, p in blocks], blocks[0][1]['n'], A)
    with open(path.with_suffix('.tsv'), 'w') as fh:
        for (lab, p), (b, s, e, rel) in zip(blocks, res):
            fh.write(f"{b:.6e}\t{s:.1f}\t{e:.6e}\t{p['Y1']}\t{lab}\n")
    return len(blocks)


(G / 'scored').mkdir(exist_ok=True)
base = (msieve(R / 'msieve_ropt_orig.p', 'msieve_orig') + msieve(R / 'msieve_ropt_inv.p', 'msieve_inv') +
        cado(R / 'cado_ropt_orig.txt', 'cado_orig') + cado(R / 'cado_ropt_inv.txt', 'cado_inv'))
write(G / 'scored/baseline.poly', base)
print('baseline', score(G / 'scored/baseline.poly'), dict(collections.Counter(l.split()[0] for l, _ in base)), flush=True)
gpu = []
for s in (G / 'todo_breadth.txt').read_text().split():
    blocks = read_blocks(str(G / 'breadth/out' / (s + '.poly')))
    gpu += [(f'gpu seed {s} rank {j} {lab}', p) for j, (lab, p) in enumerate(blocks[:TOP])]
write(G / 'scored/gpu.poly', gpu)
print('gpu', score(G / 'scored/gpu.poly'), flush=True)
