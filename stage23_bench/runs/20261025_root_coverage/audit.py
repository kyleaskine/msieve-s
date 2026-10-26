#!/usr/bin/env python3
"""CPU planning audit before equal-time root-search experiments.

Freezes inputs and binaries, then observes known rotations without injecting
them into search. No sieving, GPU work, or full-pipeline preprocessing is needed.
Plans describe admission only, not final quality or GPU throughput.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import read_cado_blocks, resultant, rotation_between

VARIANTS = {
    'dense_breadth': ['-maxcells', '2e11'],
    'dense_depth': ['-maxcells', '6e12'],
    'basins_breadth': ['-maxcells', '2e11', '-u-scan', '256', '-u-basins', '4', '-u-radius', '128'],
    'basins_depth': ['-maxcells', '6e12', '-u-scan', '256', '-u-basins', '4', '-u-radius', '128'],
    'classes2520_16': ['-maxcells', '2e11', '-classes', '2520', '16', '-class-explore', '4'],
    'classes2520_32': ['-maxcells', '2e11', '-classes', '2520', '32', '-class-explore', '8'],
    'classes2520_64': ['-maxcells', '2e11', '-classes', '2520', '64', '-class-explore', '16'],
    'classes420_16': ['-maxcells', '2e11', '-classes', '420', '16', '-class-explore', '4'],
    'classes210_16': ['-maxcells', '2e11', '-classes', '210', '16', '-class-explore', '4'],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def prepare(dest):
    marker = dest / 'prepared.json'
    if marker.exists():
        meta = json.loads(marker.read_text())
        for name, expected in meta['sha256'].items():
            if digest(dest / name) != expected:
                raise ValueError(f'frozen file changed: {name}')
        if meta['variants'] != VARIANTS:
            raise ValueError('variant definitions changed; use a new run directory')
        return meta
    # A partially prepared directory must be inspected, not silently overwritten.
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'seeds').mkdir()
    (dest / 'bin').mkdir()
    cases = []
    base = ROOT / 'stage23_bench/data/c208/ropt_coverage'
    for row in json.loads((base / 'manifest.json').read_text()):
        source = base / row['file']
        if digest(source) != row['sha256']:
            raise ValueError(f'C208 fixture hash mismatch: {source}')
        cases.append(dict(name=f"c208_s{row['seed']:04}", source=source,
                          target=row['reference_rotation'], provenance=row['historical_loss']))
    base = ROOT / 'stage23_bench/data/c192/ropt_coverage'
    seed = read_cado_blocks(base / 'seed.poly')[0][1]
    winner = read_cado_blocks(base / 'cado_winner.poly')[0][1]
    relation = rotation_between(seed, winner)
    if relation is None or len(relation['rotation']) != 2:
        raise ValueError('C192 target is not a linear rotation of its seed')
    u = relation['rotation'][1]
    cases.append(dict(name='c192_winner', source=base / 'seed.poly',
        target=dict(u=u, v=relation['rotation'][0] - u * relation['t']),
        provenance='CADO effort-5 winner missed by effort 10'))
    base = ROOT / 'stage23_gpu/runs/20261024_c197_gpu'
    for name in ('s0005', 's0001'):
        # Coordinates are recorded before export translation in the saved label.
        label, _ = read_cado_blocks(base / 'depth' / (name + '.poly'))[0]
        match = re.match(r'u (-?\d+) v (-?\d+) ', label)
        if not match:
            raise ValueError(f'no rotation label for C197 {name}')
        cases.append(dict(name='c197_' + name, source=base / 'seeds' / (name + '.poly'),
            target=dict(u=int(match[1]), v=int(match[2])),
            provenance='Saved dense-depth first exact-alpha finalist; s0005 also supplies overall winner'))
    hashes = {}
    for case in cases:
        source = case['source']
        target = dest / 'seeds' / (case['name'] + '.poly')
        shutil.copy2(source, target)
        poly = read_cado_blocks(target)[0][1]
        r = resultant(poly)
        if not r or r % poly['n']:
            raise ValueError(f'invalid seed: {source}')
        case.update(source=str(source.relative_to(ROOT)), file=str(target.relative_to(dest)), n=poly['n'])
        hashes[str(target.relative_to(dest))] = digest(target)
    for binary in ('s23_ropt', 's23_ropt_cpu'):
        target = dest / 'bin' / binary
        shutil.copy2(ROOT / 'stage23_gpu/build' / binary, target)
        hashes[str(target.relative_to(dest))] = digest(target)
    for subdir in ('include', 'tools'):
        source = ROOT / 'stage23_gpu' / subdir
        for file in source.iterdir():
            if file.suffix not in ('.h', '.c', '.cu', '.cpp'):
                continue
            target = dest / 'source' / subdir / file.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
            hashes[str(target.relative_to(dest))] = digest(target)
    meta = dict(cases=cases, variants=VARIANTS, sha256=hashes,
        script_sha256=digest(__file__), note=__doc__)
    save(marker, meta)
    return meta


def parse_plan(text, variant):
    found = re.search(r'# size model ([\d.]+) s: best line minimum ([\d.]+), budget ([\d.]+): '
                      r'([\de+.-]+) cells in (\d+) blocks \((\d+) lines dropped, (\d+) trimmed', text)
    if not found:
        raise ValueError('missing plan summary')
    choices = [dict(u=int(u), v=int(v), kind=kind) for u, v, kind in re.findall(
        r'^# class u (\d+) v (\d+): .*?, (ranked|exploration)$', text, re.M)]
    modulus = int(variant[variant.index('-classes') + 1]) if '-classes' in variant else 1
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        u, norm, center, lo, hi, _ = line.split()
        u, lo, hi = int(u), int(lo), int(hi)
        cells = max(0, hi - lo + 1) if modulus == 1 else sum(
            max(0, (hi - (lo + (c['v'] - lo) % modulus)) // modulus + 1)
            for c in choices if u % modulus == c['u'])
        rows.append(dict(u=u, cells=cells, norm=float(norm), lo=lo, hi=hi))
    trace = [line for line in text.splitlines() if line.startswith('# trace-cell')]
    finish = re.findall(r': result (\S+)', '\n'.join(trace))
    if len(finish) != 1:
        raise ValueError('missing or ambiguous final trace status')
    admitted = [row for row in rows if row['cells']]
    cells = sum(row['cells'] for row in rows)
    cap = float(variant[variant.index('-maxcells') + 1])
    if cells > cap or not cells:
        raise ValueError(f'invalid selected-cell count: {cells}, cap {cap}')
    if choices:
        logged = sum(map(int, re.findall(r'^# class cells u \d+ v \d+: (\d+) selected', text, re.M)))
        if cells != logged:
            raise ValueError('reconstructed cells disagree with per-class counters')
    return dict(model_seconds=float(found[1]), minimum=float(found[2]), budget=float(found[3]),
        cells=cells, blocks=int(found[5]), dropped_lines=int(found[6]), trimmed_lines=int(found[7]),
        scanned_lines=len(rows), admitted_lines=len(admitted),
        admitted_u_range=[min(r['u'] for r in admitted), max(r['u'] for r in admitted)],
        classes=choices, ranked_distinct_u=len({c['u'] for c in choices if c['kind'] == 'ranked'}),
        trace=trace, target_result=finish[0],
        budget_steps=[dict(budget=float(b), cells=int(c), lines=int(n)) for b,c,n in re.findall(
            r'^# step ([\d.]+): (\d+) cells, (\d+) lines', text, re.M)])


def plans(dest, meta, case_names, variant_names):
    records = []
    for case in meta['cases']:
        if case_names and case['name'] not in case_names:
            continue
        for name, extra in VARIANTS.items():
            if variant_names and name not in variant_names:
                continue
            folder = dest / 'plans' / case['name']
            folder.mkdir(parents=True, exist_ok=True)
            log, record = folder / (name + '.log'), folder / (name + '.json')
            command = [str(dest / 'bin/s23_ropt_cpu'), '-poly', str(dest / case['file']),
                '-search', '-budget', '3', '-maxlines', '1000', '-umax', '20000', '-aw', '1.3',
                '-rerank', '4096', '-refine', '200', '-top', '8', '-t', '2',
                '-trace-cell', str(case['target']['u']), str(case['target']['v']), *extra, '-plan']
            if record.exists():
                row = json.loads(record.read_text())
                if row['command'] != command or row['returncode'] or row['log_sha256'] != digest(log):
                    raise ValueError(f'changed/failed plan; choose a fresh run: {record}')
            else:
                if log.exists():
                    raise ValueError(f'unfinished plan; inspect before retrying: {log}')
                start = time.monotonic()
                with log.open('wb') as fh:
                    result = subprocess.run(command, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
                row = dict(case=case['name'], variant=name, command=command, returncode=result.returncode,
                           wall_seconds=time.monotonic()-start, log_sha256=digest(log))
                if result.returncode:
                    save(record, row)
                    raise RuntimeError(f'plan failed: {log}')
                row.update(parse_plan(log.read_text(encoding='utf-8'), extra))
                save(record, row)
            records.append(row)
            print(case['name'], name, row['target_result'], f"{row['cells']:.3g} cells",
                  f"{row['ranked_distinct_u']} ranked u classes", flush=True)
    return records


def summarize(dest):
    records = [json.loads(p.read_text()) for p in sorted((dest / 'plans').glob('*/*.json'))]
    successful = [r for r in records if not r['returncode'] and 'target_result' in r]
    result = dict(records=successful, completed=len(successful), expected=10*len(VARIANTS),
        gpu_timings_complete=False,
        note='Diagnostic corpus includes known losses and winners; not a holdout or equal-time quality comparison. '
             'plan_admitted means eligible for sieving only; all later retention/scoring cuts remain unchecked.')
    save(dest / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'plan', 'summary'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS)
    args = parser.parse_args()
    dest = args.run_dir.resolve()
    meta = prepare(dest)
    if args.cases and set(args.cases) - {c['name'] for c in meta['cases']}:
        parser.error('unknown case')
    if args.phase == 'plan':
        plans(dest, meta, args.cases, args.variants)
    if args.phase in ('plan', 'summary'):
        result = summarize(dest)
        print(f"Saved {result['completed']}/{result['expected']} plans to {dest / 'summary.json'}")


if __name__ == '__main__':
    main()
