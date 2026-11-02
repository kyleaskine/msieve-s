#!/usr/bin/env python3
"""Bounded saved-pool MurphyE timing; inputs are embedded in a JSON case file.

Example: --cases report.json --run-dir /tmp/murphy --binary .../s23_ropt
         --label gpu --gpu --repeats 3
Each process warms all cases before measured passes, retaining CUDA workspaces.
Checked runs measure audit cost and must not be used as production timings.
"""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from gpu_pipeline import digest, write_json
from polyfmt import read_cado_blocks, resultant
from score_polys import score_bounds


def replay_options(case):
    """Match the saved pool's scoring convention; make scorer defaults explicit."""
    lines = case['pool'].splitlines()
    try:
        if lines[0] != 'S23_ROPT_POOL 1':
            raise ValueError()
        weight, projective = lines[1].split()
        weight, projective = float(weight), int(projective)
        if not math.isfinite(weight) or weight <= 0 or projective not in (0, 1):
            raise ValueError()
        if int(lines[11]) != case['count'] or case['count'] < 1:
            raise ValueError()
    except (IndexError, ValueError):
        raise ValueError('case needs a valid v1 pool header and matching positive count') from None
    # Same standard bounds as cado_murphy.cpp; ropt requires explicit bounds.
    bounds = score_bounds(case['params']) or dict(Bf=1e7, Bg=5e6, area=1e16)
    options = ['-aw', weight, '-Bf', bounds['Bf'], '-Bg', bounds['Bg'], '-area', bounds['area']]
    if not projective:
        options.append('-noproj')
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--profile', action='store_true', help='Nsight CUDA trace (timings are instrumented)')
    parser.add_argument('--cpu-sample', action='store_true',
                        help='include process-tree CPU sampling in the Nsight trace; needs --profile')
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1 or (args.check and not args.gpu):
        parser.error('positive repeats/threads required; --check needs --gpu')
    if args.cpu_sample and not args.profile:
        parser.error('--cpu-sample needs --profile')
    binary = args.binary.resolve(strict=True)
    cases = json.loads(args.cases.read_text())['cases']
    if not cases or len({c['tag'] for c in cases}) != len(cases):
        parser.error('cases must be nonempty with distinct tags')
    options = [replay_options(case) for case in cases]
    folder = (args.run_dir / args.label).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    commands, outputs = [], []
    for rep in range(-1, args.repeats):
        for case, replay in zip(cases, options):
            tag = case['tag']
            seed, pool = folder / f'{tag}.seed', folder / f'{tag}.pool'
            seed.write_text(case['seed'])
            pool.write_text(case['pool'])
            output = folder / f'{tag}_{rep}.poly'
            n = case['count']
            cmd = ['-poly', seed, '-search', '-pool-in', pool, *replay,
                   '-rerank', n, '-refine', n, '-top', 1, '-t', args.threads,
                   '-murphy', '-out', output]
            if args.gpu:
                cmd += ['-refine-gpu', '-murphy-gpu']
            if args.check:
                cmd += ['-refine-check', '-murphy-check']
            commands.append(list(map(str, cmd)))
            outputs.append((tag, rep, output, n))
    manifest = folder / 'jobs.tsv'
    if any(any(c in a for c in '\t\r\n') for cmd in commands for a in cmd):
        raise ValueError('manifest arguments contain delimiters')
    manifest.write_text('S23_ROPT_BATCH 1\n' + ''.join('\t'.join(c) + '\n' for c in commands))
    command = [str(binary), '-batch', str(manifest)]
    if args.profile:
        sampling = (['--sample=process-tree', '--backtrace=dwarf'] if args.cpu_sample
                    else ['--sample=none', '--cpuctxsw=none'])
        command = ['nsys', 'profile', '--trace=cuda,nvtx', *sampling,
                   '--force-overwrite=false', '-o', str(folder / 'cuda'), *command]
    record = dict(schema=2, command=command, binary_sha256=digest(binary),
                  cases_sha256=digest(args.cases), threads=args.threads,
                  checked=args.check, profiled=args.profile, cpu_sample=args.cpu_sample,
                  cases_per_pass=len(cases), warmup_jobs=len(cases), repeats=args.repeats,
                  case_tags=[c['tag'] for c in cases], process_timing_clock='time.time (wall)')
    write_json(folder / 'result.json', record)
    start = time.time()
    with (folder / 'process.log').open('wb') as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    record.update(seconds=time.time() - start, returncode=completed.returncode,
                  log_sha256=digest(folder / 'process.log'))
    write_json(folder / 'result.json', record)
    if completed.returncode:
        raise RuntimeError(f'benchmark failed; see {folder / "process.log"}')
    if record['seconds'] <= 0:
        raise ValueError('nonpositive process wall time; possible wall-clock step')
    text = (folder / 'process.log').read_text(errors='replace')
    phases = list(map(float, re.findall(r', refine ([\d.]+) s', text)))
    steady_times = list(map(float, re.findall(r'# batch job \d+ status 0, ([\d.]+) s', text)))
    wall_times = list(map(float, re.findall(r'# batch job \d+ wall (-?[\d.]+) s', text)))
    if len(phases) != len(outputs) or len(steady_times) != len(outputs):
        raise ValueError('missing refinement timing')
    if wall_times and (len(wall_times) != len(outputs) or min(wall_times) <= 0):
        raise ValueError('missing/nonpositive wall timing; possible wall-clock step')
    record['job_timing_clock'] = 'system_clock (wall)' if wall_times else 'legacy steady_clock only'
    record['refine_timing_clock'] = 'steady_clock (diagnostic, not wall time)'
    if not wall_times:
        print('Legacy binary has no per-job wall timings; wall medians will be null.', file=sys.stderr)
    if args.gpu:
        statuses = re.findall(r'MurphyE GPU: (\d+) cells, (\d+) host fallbacks', text)
        if [(int(n), int(f)) for n, f in statuses] != [(n, 0) for _, _, _, n in outputs]:
            raise ValueError('missing GPU work or unexpected fallback')
    records = []
    composites = {c['tag']: int(c['pool'].splitlines()[2]) for c in cases}
    for index, ((tag, rep, output, n), seconds, steady_seconds) in enumerate(zip(outputs, phases, steady_times)):
        polys = read_cado_blocks(output)
        if len(polys) != n or any(p['n'] != composites[tag] or not resultant(p) or
                                  resultant(p) % p['n'] for _, p in polys):
            raise ValueError('missing or invalid exports')
        records.append(dict(tag=tag, repetition=rep, refine_seconds=seconds,
                            job_seconds=wall_times[index] if wall_times else None,
                            job_steady_seconds=steady_seconds,
                            output=str(output), sha256=digest(output)))
    for case in cases:
        hashes = {r['sha256'] for r in records if r['tag'] == case['tag']}
        if len(hashes) != 1:
            raise ValueError('persistent session changed exports')
    record['jobs'] = records
    record['median_refine_seconds'] = {
        c['tag']: statistics.median(r['refine_seconds'] for r in records
                                   if r['tag'] == c['tag'] and r['repetition'] >= 0)
        for c in cases}
    record['median_pass_seconds'] = statistics.median(
        sum(r['refine_seconds'] for r in records if r['repetition'] == rep)
        for rep in range(args.repeats))
    record['median_steady_job_pass_seconds'] = statistics.median(
        sum(r['job_steady_seconds'] for r in records if r['repetition'] == rep)
        for rep in range(args.repeats))
    record['median_job_pass_seconds'] = statistics.median(
        sum(r['job_seconds'] for r in records if r['repetition'] == rep)
        for rep in range(args.repeats)) if wall_times else None
    write_json(folder / 'result.json', record)
    print(json.dumps({key: record[key] for key in
                      ('job_timing_clock', 'median_job_pass_seconds', 'median_steady_job_pass_seconds')}), flush=True)


if __name__ == '__main__':
    main()
