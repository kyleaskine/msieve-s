#!/usr/bin/env python3
"""save_fixtures.py: compact records of the 2026-10-10 C185 and second-C168 GPU runs in
stage23_bench/data/{c185,c168b}: winners (with a skew line), per-family comparison, and
summary.json (settings, timings, results, the independent 256k rescoring, hashes)."""
import hashlib, json, re, sys, types
from pathlib import Path
ROOT = Path.home() / 'msieve-s'
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from score_polys import read_blocks, lattice_scores
from polyfmt import cado_poly_text, resultant

# the run directory (c185/, c168/ inside it): argument 1, else where the runs were made
HERE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'stage23_gpu/build/runs_20261010'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
JOBS = {
    'c185': dict(out='c185', params='job:31,32,15,50e6', lattice='15,16384', qband='50e6,2.5e8,7',
                 bounds=dict(Bf=4294967296, Bg=2147483648, area=2.68435456e16), murphy_seed='s0004',
                 winners=[('cado_orig_best', 'baseline', 'cado_orig seed 4 E'),
                          ('msieve_best', 'baseline', 'msieve_orig seed 1 e 5.498e-14 j 0'),
                          ('gpu_seed1_best', 'gpu', 'gpu seed s0001 rank 2 ')]),
    'c168': dict(out='c168b', params='job:30,31,14,25e6', lattice='14,8192', qband='25e6,1e8,7',
                 bounds=dict(Bf=2147483648, Bg=1073741824, area=3.3554432e15), murphy_seed='s0001',
                 winners=[('msieve_best', 'baseline', 'msieve_orig seed 1 e 7.466e-13 j 0'),
                          ('cado_best', 'baseline', 'cado_orig seed 1 E')]),
}


def tsv(path):
    out = []
    for line in Path(path).read_text().splitlines():
        band, skew, e, y1, lab = line.split('\t')
        out.append(dict(band=float(band), skew=float(skew), e=float(e), lab=lab))
    return out


for d, J in JOBS.items():
    G = HERE / d
    dest = ROOT / 'stage23_bench/data' / J['out']
    (dest / 'winners').mkdir(parents=True, exist_ok=False)
    (dest / 'gpu_ropt').mkdir()
    man = json.loads((G / 'manifest.json').read_text())
    n = man['n']
    (dest / 'n.txt').write_text(f'{n}\n')
    # winners: (name, label, poly)
    blocks = {k: read_blocks(str(G / 'scored' / f'{k}.poly')) for k in ('baseline', 'gpu')}
    picks = []
    for name, src, prefix in J['winners']:
        lab, p = next((l, p) for l, p in blocks[src] if l.lstrip('# ').startswith(prefix))
        picks.append((name, lab.lstrip('# '), p))
    mlab, mp = read_blocks(str(G / 'murphy' / f"{J['murphy_seed']}_2e11.poly"))[0]
    picks.append(('gpu_murphy_best', 's23_ropt -murphy #1: ' + mlab.lstrip('# '), mp))
    for _, _, p in picks:
        assert p['n'] == n and resultant(p) % n == 0
    A = types.SimpleNamespace(threads=8, params=J['params'], points=16000, eval_points=256000,
                              lattice=J['lattice'], qband=J['qband'], useskew=False, nlat=256, npts=4096, latseed=2)
    res = lattice_scores([p for _, _, p in picks], n, A)
    b0, e0 = max(r[0] for r in res), max(r[2] for r in res)
    rescore = []
    for (name, lab, p), (b, s, e, rel) in zip(picks, res):
        q = {k: v for k, v in p.items() if k != 'skew'}
        text = cado_poly_text(q).rstrip('\n') + f'\nskew: {s:.1f}\n'
        (dest / 'winners' / f'{name}.poly').write_text(
            f'# {lab}\n# skew: the 256k rescoring\'s best (16k-point grid search); band {b / b0:.5f}, '
            f'MurphyE {e:.6e} ({J["params"]})\n' + text)
        rescore.append(dict(poly=f'winners/{name}.poly', label=lab, band_ratio=round(b / b0, 6),
                            murphy=e, murphy_ratio=round(e / e0, 6), skew=s))
    # per-family comparison from the breadth scoring (K 4000/16000, nlat default)
    base, gpu = tsv(G / 'scored/baseline.tsv'), tsv(G / 'scored/gpu.tsv')
    best = max(r['band'] for r in base + gpu)
    cpu_f, gpu_f = {}, {}
    for r in base:
        s = int(r['lab'].split()[2])
        if s > 0 and r['band'] > cpu_f.get(s, (0,))[0]:
            cpu_f[s] = (r['band'], r['lab'].split()[0], r['e'])
    for r in gpu:
        m = re.match(r'gpu seed s(\d+)(?:_d\d+)? rank (\d+)', r['lab'])
        s, k = int(m.group(1)), int(m.group(2))
        if r['band'] > gpu_f.get(s, (0,))[0]:
            gpu_f[s] = (r['band'], k, r['e'])
    rows = ['seed\tcpu_band\tcpu_method\tcpu_murphy\tgpu_band\tgpu_proxy_rank\tgpu_murphy\tgpu_over_cpu']
    ratios = []
    for s in sorted(set(cpu_f) & set(gpu_f)):
        c, g = cpu_f[s], gpu_f[s]
        ratios.append(g[0] / c[0])
        rows.append(f'{s}\t{c[0] / best:.5f}\t{c[1]}\t{c[2]:.5e}\t{g[0] / best:.5f}\t{g[1]}\t{g[2]:.5e}\t{g[0] / c[0]:.5f}')
    (dest / 'gpu_ropt/families.tsv').write_text('\n'.join(rows) + '\n')
    log = (G / 'breadth/log.txt').read_text()
    murphy_txt = (G / 'murphy' / f"{J['murphy_seed']}_2e11.txt").read_text()
    summary = dict(
        date='2026-10-10', source=man['source'], n_digits=len(str(n)),
        seeds=dict(parents=300, content=man['searches'] - 300, numbering='one-based CPU re-sopt rank (best300_msieve.ms)'),
        breadth=dict(command='s23_ropt -poly SEED -search -budget 3.0 -aw 1.3 -maxcells 2e11 -rerank 4096 -top 3 -t 4',
                     workers=4, searches=man['searches'], failed=sum(1 for l in log.splitlines() if ' rc=' in l and ' rc=0 ' not in l),
                     total_seconds=float(re.search(r'TOTAL ([0-9.]+)', log).group(1)), clock='date +%s.%N',
                     note='about nine ECM processes shared the CPU: elapsed record, not a speed measurement'),
        scoring=dict(params=J['params'], lattice=J['lattice'], qband=J['qband'], points=4000, eval_points=16000,
                     scored='CPU pipeline: msieve best 3 per seed and pass + all CADO; GPU: first 32 of each search',
                     note='job parameters assumed (user: reasonable); siever region and q range not test-sieve validated'),
        families=dict(compared=len(ratios), better_1pct=sum(x > 1.01 for x in ratios),
                      within_1pct=sum(0.99 <= x <= 1.01 for x in ratios), worse_1pct=sum(x < 0.99 for x in ratios),
                      worst=round(min(ratios), 4)),
        murphy_run=dict(seed=J['murphy_seed'], bounds=J['bounds'], K=16000, Keval=64000, maxeval=384, device='GPU',
                        summary=re.search(r'^# best 3 by validated MurphyE: ([^;]*)', murphy_txt, re.M).group(1),
                        timing=re.search(r'^# s23_ropt: [^,]*, search: (.*)$', murphy_txt, re.M).group(1)),
        rescore=dict(settings=dict(points=16000, eval_points=256000, nlat=256, npts=4096, latseed=2), rows=rescore),
        sha256={'s23_ropt': man['s23_ropt_sha256'], **{f'source/{k}': v for k, v in man['source_sha256'].items()},
                'breadth_log': sha(G / 'breadth/log.txt'), 'baseline.tsv': sha(G / 'scored/baseline.tsv'),
                'gpu.tsv': sha(G / 'scored/gpu.tsv'), 'murphy_output': sha(G / 'murphy' / f"{J['murphy_seed']}_2e11.poly")},
        full_outputs=str(G))
    (dest / 'gpu_ropt/summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(dest)
    for r in rescore:
        print(f"  {r['poly']:28s} band {r['band_ratio']:.5f}  MurphyE {r['murphy']:.6e} ({r['murphy_ratio']:.5f})")
    print('  families', summary['families'], 'breadth', summary['breadth']['total_seconds'], 's')
