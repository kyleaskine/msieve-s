#!/usr/bin/env python3
"""Isolated root-sieve layout experiments; no search-policy or score changes."""
import argparse
import hashlib
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
from polyfmt import cado_compile_args

LAYOUTS = [('baseline', 256, 32), ('nt384_cpt16', 384, 16),
           ('nt512_cpt16', 512, 16), ('nt512_cpt8', 512, 8)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def prepare(run, binary, source):
    run.mkdir(parents=True, exist_ok=False)
    (run/'bin').mkdir()
    (run/'source').mkdir()
    shutil.copy2(binary, run/'bin/s23_ropt')
    shutil.copy2(source, run/'source/s23_ropt.cu')
    save(run/'baseline.json', dict(binary_sha256={'s23_ropt': digest(run/'bin/s23_ropt')},
        source_sha256=digest(run/'source/s23_ropt.cu'), source_binary=str(binary), source_file=str(source)))


def build(run):
    source = (run / 'source/s23_ropt.cu').read_text()
    _, libs, flags = cado_compile_args()
    records = []
    for name, nt, cpt in LAYOUTS[1:]:
        path = run / 'source' / (name + '.cu')
        path.write_text(source.replace('static const int CPT = 32;', f'static const int CPT = {cpt};')
                              .replace('static const int NT = 256;', f'static const int NT = {nt};'))
        binary = run / 'bin' / name
        cmd = ['nvcc', '-O3', '-std=c++17', '-arch=sm_120', '--fmad=false',
               '-Iinclude', '-Itools', '-Xcompiler', '-fopenmp,-ffp-contract=off,-mfma',
               '-o', str(binary), str(path), 'build/cado_io.o', 'build/murphy_rank.o', *libs, *flags, '-lgomp']
        start = time.monotonic()
        with (run / (name + '_build.log')).open('wb') as log:
            subprocess.run(cmd, cwd=ROOT / 'stage23_gpu', stdout=log, stderr=subprocess.STDOUT, check=True)
        records.append(dict(name=name, threads=nt, cells_per_thread=cpt, command=cmd,
                            seconds=time.monotonic()-start, source_sha256=digest(path), binary_sha256=digest(binary)))
        print('built', name, flush=True)
    save(run / 'layout_builds.json', records)


def benchmark(run, repeats, layouts, name):
    dest = run / name
    dest.mkdir(exist_ok=False)
    cases = [
        ('c197_dense', ROOT / 'stage23_gpu/runs/20261024_c197_gpu/seeds/s0005.poly',
         ['-u', 12, 27, '-v', -1073741824, 1073741823, '-seg', 20]),
        ('c208_strided', ROOT / 'stage23_bench/data/c208/ropt_coverage/s0222.poly',
         ['-u', -20160, 17640, '-v', -2705829396480, 2705829396479, '-seg', 20,
          '-residue', 2520, 0, 0]),
        ('c168_short_blocks', ROOT / 'stage23_bench/data/c168/content/s0051.poly',
         ['-u', -2048, 2047, '-v', -2048, 2047, '-seg', 12]),
    ]
    for tag, source, _ in cases:
        shutil.copy2(source, dest / (tag + '.poly'))
    controls, records = {}, []
    # Warm each mode, then rotate the mode order between timed repetitions.
    for repetition in range(-1, repeats):
        order = layouts if repetition < 0 else layouts[repetition % len(layouts):] + layouts[:repetition % len(layouts)]
        for name, _, _ in order:
            binary = run / 'bin' / ('s23_ropt' if name == 'baseline' else name)
            times = []
            for tag, _, options in cases:
                cmd = [str(x) for x in [binary, '-poly', dest / (tag + '.poly'), *options,
                                       '-launch', 30, '-k', 8, '-top', 128, '-t', 2]]
                start = time.monotonic()
                log_path = dest / f'{repetition}_{name}_{tag}.log'
                with log_path.open('wb') as log:
                    subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
                elapsed = time.monotonic()-start
                output = log_path.read_text()
                cells, seconds = re.search(r'window: ([\d.e+]+) cells, sieve ([\d.]+) s', output).groups()
                times.append(seconds)
                hits = '\n'.join(line for line in output.splitlines() if line.startswith('u '))
                if not hits: raise ValueError('no hits')
                if tag in controls and controls[tag] != hits: raise ValueError(f'{name}/{tag}: selected hits changed')
                controls.setdefault(tag, hits)
                if repetition >= 0:
                    records.append(dict(repetition=repetition, layout=name, case=tag,
                        command=cmd, cells=float(cells), sieve_seconds=float(seconds), process_seconds=elapsed,
                        log_sha256=digest(log_path), hits_sha256=hashlib.sha256(hits.encode()).hexdigest()))
            print('repetition', repetition, name, times, flush=True)
            save(dest / 'records.json', records)
    summary = {}
    for tag, _, _ in cases:
        medians = {name: statistics.median(r['sieve_seconds'] for r in records if r['case']==tag and r['layout']==name)
                   for name, _, _ in layouts}
        summary[tag] = dict(median_seconds=medians,
                            speedup={name: medians['baseline']/t for name, t in medians.items()})
    save(dest / 'summary.json', dict(repeats=repeats, cases=summary,
        note='Isolated windows, rotated mode order, one warmup per mode. All 128 printed hits agree. '
             'Sieve timers are rounded to 0.01 s. Each process time includes CUDA startup.'))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'build', 'benchmark'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--variants', nargs='+', help='previously built binary names, including baseline')
    parser.add_argument('--name', default='layout_bench', help='fresh result subdirectory')
    parser.add_argument('--baseline-binary', type=Path)
    parser.add_argument('--baseline-source', type=Path)
    args = parser.parse_args()
    if args.repeats < 1: parser.error('positive repeats required')
    if Path(args.name).name != args.name or args.name in ('.', '..'): parser.error('simple result name required')
    if args.variants and ('baseline' not in args.variants or any(Path(n).name != n for n in args.variants)):
        parser.error('variant names must include baseline and have no directory components')
    run = args.run_dir.resolve()
    if args.phase == 'prepare':
        if not args.baseline_binary or not args.baseline_source:
            parser.error('prepare requires --baseline-binary and --baseline-source')
        prepare(run, args.baseline_binary.resolve(), args.baseline_source.resolve())
    elif args.phase == 'build': build(run)
    else: benchmark(run, args.repeats, [(n, 0, 0) for n in args.variants] if args.variants else LAYOUTS, args.name)


if __name__ == '__main__':
    main()
