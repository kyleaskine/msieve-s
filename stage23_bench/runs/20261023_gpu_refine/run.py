#!/usr/bin/env python3
"""Full-alpha corpus audit and bounded CPU/CUDA MurphyE comparison.

Uses copied saved pools, never searches new rotation space. The 25-case screen
corpus is refined at full bound without screening, plus C192's winning pool.
Murphy timing uses the top 16 exact-alpha cells from C208 and C192, two CPU
threads, and persistent processes. Timings are component measurements only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import read_cado_blocks, resultant, cado_poly_text
from score_polys import score_bounds


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def coord(label):
    return tuple(int(re.search(rf'\b{k} (-?\d+)', label)[1]) for k in ('u', 'v'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    if args.repeats < 1: ap.error('--repeats must be positive')
    dest = args.run_dir.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'inputs').mkdir()
    (dest / 'bin').mkdir()
    binaries = {}
    for mode, src in [('cpu', ROOT / 'stage23_gpu/build/s23_ropt_cpu'),
                      ('gpu', ROOT / 'stage23_gpu/build/s23_ropt'),
                      ('scorer', ROOT / 'stage23_bench/tools/cado_murphy')]:
        target = dest / 'bin' / src.name
        shutil.copy2(src, target)
        binaries[mode] = target
    spec = importlib.util.spec_from_file_location('screen_corpus', Path(__file__).parents[1] / '20261018_gpu_screen/run.py')
    corpus = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(corpus)
    sources = []
    for case in corpus.cases():
        a = case['args']
        sources.append((case['tag'], Path(a[a.index('-poly')+1]), Path(a[a.index('-pool-in')+1])))
    sources.append(('c192', ROOT / 'stage23_gpu/runs/20261019_c192/seeds/s0012.poly',
                    ROOT / 'stage23_gpu/runs/20261019_c192/depth/s0012.pool'))
    cases = []
    for tag, seed, pool in sources:
        dst_seed, dst_pool = dest / 'inputs' / (tag + '.poly'), dest / 'inputs' / (tag + '.pool')
        shutil.copy2(seed, dst_seed)
        shutil.copy2(pool, dst_pool)
        lines = dst_pool.read_text().splitlines()
        cases.append(dict(tag=tag, seed=dst_seed, pool=dst_pool, count=int(lines[11])))
    provenance = {str(p): digest(p) for _, seed, pool in sources for p in (seed, pool)}
    summary = dict(binaries={k: dict(path=str(p), sha256=digest(p)) for k, p in binaries.items()},
                   source_sha256=provenance, corpus=[], murphy=[], note=__doc__)
    save(dest / 'provenance.json', summary)

    def batch(label, mode, jobs):
        folder = dest / label
        folder.mkdir()
        manifest = folder / 'jobs.tsv'
        normalized = []
        for tag, options in jobs:
            argv = [*map(str, options), '-out', str(folder / (tag + '.poly'))]
            if any(any(c in x for c in '\t\r\n') for x in argv): raise ValueError('manifest delimiter')
            normalized.append(argv)
        manifest.write_text('S23_ROPT_BATCH 1\n' + ''.join('\t'.join(a) + '\n' for a in normalized))
        command = [str(binaries[mode]), '-batch', str(manifest)]
        start = time.monotonic()
        with (folder / 'process.log').open('wb') as log:
            p = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        wall = time.monotonic() - start
        if p.returncode: raise RuntimeError(f'{label}: exit {p.returncode}, see {folder / "process.log"}')
        text = (folder / 'process.log').read_text()
        record = dict(label=label, mode=mode, command=command, wall_seconds=wall,
                      phase_seconds=sum(map(float, re.findall(r', refine ([\d.]+) s', text))),
                      manifest_sha256=digest(manifest), log_sha256=digest(folder / 'process.log'))
        print(label, f'{wall:.3f}s', flush=True)
        return folder, text, record

    def options(c):
        return ['-poly', c['seed'], '-search', '-pool-in', c['pool'], '-aw', 1.3,
                '-rerank', c['count'], '-refine', 200, '-top', 3, '-t', 2]

    controls, _, control_time = batch('alpha_cpu', 'cpu', [(c['tag'], options(c)) for c in cases])
    checked, text, _ = batch('alpha_gpu_checked', 'gpu',
                            [(c['tag'], options(c) + ['-refine-gpu', '-refine-check']) for c in cases])
    unchecked, _, gpu_time = batch('alpha_gpu', 'gpu', [(c['tag'], options(c) + ['-refine-gpu']) for c in cases])
    audits = re.findall(r'full-refine GPU check: max \|dL\| ([\d.e+-]+), \|dA\| ([\d.e+-]+), \|dkey\| ([\d.e+-]+); (\d+) translations', text)
    if len(audits) != len(cases): raise ValueError('missing full GPU audit')
    for c, audit in zip(cases, audits):
        tag = c['tag']
        cpu, gpu = controls / (tag + '.poly'), unchecked / (tag + '.poly')
        if gpu.read_bytes() != (checked / (tag + '.poly')).read_bytes(): raise ValueError(f'{tag}: audit changed exports')
        cr, gr = read_cado_blocks(cpu), read_cado_blocks(gpu)
        if len(cr) != 200 or len(gr) != 200: raise ValueError('missing exports')
        retained = len({coord(l) for l, _ in cr} & {coord(l) for l, _ in gr})
        if retained != 200: raise ValueError(f'{tag}: CPU finalist rotation lost ({retained}/200)')
        for _, p in gr:
            if not resultant(p) or resultant(p) % p['n']: raise ValueError('bad resultant')
        summary['corpus'].append(dict(tag=tag, candidates=c['count'], retained=retained,
            byte_identical=cpu.read_bytes() == gpu.read_bytes(), cpu_sha256=digest(cpu), gpu_sha256=digest(gpu),
            max_dL=float(audit[0]), max_dA=float(audit[1]), max_dkey=float(audit[2]), different_translations=int(audit[3])))
    summary['alpha_timings'] = [control_time, gpu_time]
    save(dest / 'summary.json', summary)

    selected = [c for c in cases if c['tag'] in ('c208', 'c192')]
    jobs = []
    for c in selected:
        tag = c['tag']
        wanted = {coord(label) for label, _ in read_cado_blocks(controls / (tag + '.poly'))[:16]}
        lines = c['pool'].read_text().splitlines()
        rows = [l for l in lines[12:-1] if tuple(map(int, l.split()[:2])) in wanted]
        if len(rows) != 16: raise ValueError('short Murphy pool')
        pool = dest / 'inputs' / (tag + '_murphy.pool')
        pool.write_text('\n'.join(lines[:11] + ['16'] + rows + ['END', '']))
        params = 'job:32,33,15,80e6' if tag == 'c192' else 'job:34,35,16,80e6'
        bounds = score_bounds(params)
        opts = ['-poly', c['seed'], '-search', '-pool-in', pool, '-aw', 1.3, '-rerank', 16,
                '-refine', 16, '-top', 3, '-t', 2, '-murphy', '-Bf', bounds['Bf'], '-Bg', bounds['Bg'], '-area', bounds['area']]
        jobs.append((tag, opts))
    audits_dir, _, _ = batch('murphy_checked', 'gpu',
                            [(t, a + ['-refine-gpu', '-refine-check', '-murphy-gpu', '-murphy-check']) for t, a in jobs])
    # Warm both modes before alternating measured repetitions.
    batch('warm_cpu', 'cpu', jobs)
    batch('warm_gpu', 'gpu', [(t, a + ['-refine-gpu', '-murphy-gpu']) for t, a in jobs])
    for repeat in range(args.repeats):
        for mode in (['cpu', 'gpu'] if repeat % 2 == 0 else ['gpu', 'cpu']):
            extra = ['-refine-gpu', '-murphy-gpu'] if mode == 'gpu' else []
            folder, text, record = batch(f'murphy_{mode}_{repeat}', mode, [(t, a + extra) for t, a in jobs])
            if mode == 'gpu':
                if text.count('MurphyE GPU: 16 cells, 0 host fallbacks') != 2: raise ValueError('GPU fallback in timing')
                for tag, _ in jobs:
                    if (folder / (tag + '.poly')).read_bytes() != (audits_dir / (tag + '.poly')).read_bytes():
                        raise ValueError('checked/unchecked Murphy exports differ')
            summary['murphy'].append(record)
    # Evaluate every retained configuration independently at 256k angles, using
    # the frozen scorer and each declared skew; this CLI prints six decimals.
    quality = []
    for tag, opts in jobs:
        bounds = {key: opts[opts.index('-' + key) + 1] for key in ('Bf', 'Bg', 'area')}
        scores = {}
        for mode in ('cpu', 'gpu'):
            path = dest / f'murphy_{mode}_0' / (tag + '.poly')
            rows = read_cado_blocks(path)
            if len(rows) != 16 or any(not resultant(p) or resultant(p) % p['n'] for _, p in rows): raise ValueError('bad Murphy exports')
            command = [str(binaries['scorer']), str(path), '-useskew', '-K', '256000', '-Keval', '256000', '-t', '2']
            for key, value in bounds.items(): command += ['-' + key, str(value)]
            p = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
            (dest / f'{tag}_{mode}_256k.log').write_text(p.stdout + p.stderr)
            scores[mode] = max(float(l.split()[1]) for l in p.stdout.splitlines() if l and l[0].isdigit())
        quality.append(dict(tag=tag, **scores, ratio=scores['gpu'] / scores['cpu']))
        if scores['gpu'] / scores['cpu'] < 1 - 1e-6: raise ValueError('quality regression')
    summary['independent_256k'] = quality
    summary['murphy_median_seconds'] = {m: statistics.median(r['wall_seconds'] for r in summary['murphy'] if r['mode'] == m) for m in ('cpu', 'gpu')}
    summary['sources_unchanged'] = all(digest(p) == h for p, h in provenance.items())
    if not summary['sources_unchanged']: raise ValueError('source changed during audit')
    save(dest / 'summary.json', summary)
    print(json.dumps(dict(cases=len(cases), exports=200 * len(cases), quality=quality,
                          medians=summary['murphy_median_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    main()
