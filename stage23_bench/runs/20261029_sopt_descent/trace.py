#!/usr/bin/env python3
"""Reproduce the two initial-sopt losses without the original multi-million inputs.

cpu: capture phase traces from the unmodified CPU path.
perturb: CPU-only sensitivity experiment in copied headers; NOT CUDA emulation.
gpu: explicit device execution, only when the GPU is available.

All freshly built binaries, frozen sources, inputs and detailed traces go in a NEW --run-dir
under stage23_gpu/runs (ignored). Only summary.json needs to be retained in Git.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'stage23_bench/data/gpu_pipeline/20261028.json'
GPU_DIR = ROOT / 'stage23_gpu'
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
sys.path.insert(0, str(ROOT / 'scripts'))
from gpu_pipeline import digest as sha
from polyfmt import cado_poly_text, coeffs

# A deterministic +/- one-ULP perturbation of the *returned* lognorm, keyed by
# the integral's bits. Seed 0 is the unmodified baseline. The integer arithmetic,
# candidate generation and LLL source are unchanged. Never edits production.
PERTURB = '''    double answer = sopt_log(tt) / 2;
    const char *setting = std::getenv("S23_DEBUG_LOG_SEED");
    if (setting && std::atoi(setting)) {
        uint64_t bits;
        std::memcpy(&bits, &tt, sizeof bits);
        bits ^= (uint64_t)std::atoi(setting) * UINT64_C(0x9e3779b97f4a7c15);
        bits ^= bits >> 33;
        bits *= UINT64_C(0xff51afd7ed558ccd);
        bits ^= bits >> 33;
        answer = nextafter(answer, (bits & 1) ? INFINITY : -INFINITY);
    }
    return answer;'''


def matches(record, poly):
    return (record.get('f') == list(map(str, coeffs(poly)))
            and record.get('g') == [str(poly[f'Y{i}']) for i in range(2)])


def integers(value):
    """Ignore backend/float metadata when locating the first integer-path change."""
    if isinstance(value, list):
        return [integers(v) for v in value]
    if isinstance(value, dict):
        return {k: integers(v) for k, v in value.items() if k not in ('backend', 'objective')}
    return value


def capture(binary, source, prefix, index, cpu=True, seed=0, execution=None, timeout=3600):
    execution = execution or {'batch': 1, 'threads': 1, 'slice': 10, 'device': 0}
    cmd = [str(binary), '-inputpolys', str(source), '-batch', str(execution['batch']),
           '-t', str(execution['threads']), '-slice', str(execution['slice']),
           '-dev', str(execution['device']),
           '-trace', str(prefix.with_suffix('.jsonl')), '-trace-index', str(index)]
    if cpu:
        cmd += ['-cpu', '-compare']
    prefix.with_suffix('.command.json').write_text(json.dumps(cmd, indent=2) + '\n')
    with prefix.with_suffix('.out').open('w') as out, prefix.with_suffix('.err').open('w') as err:
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=out, stderr=err, timeout=timeout,
                       env=dict(os.environ, S23_DEBUG_LOG_SEED=str(seed)))
    return [json.loads(line) for line in prefix.with_suffix('.jsonl').read_text().splitlines()]


def raw_blocks(source):
    """Keep source bytes/order, including blank lines; don't load a million polys."""
    block = []
    with source.open('rb') as f:
        for line in f:
            if line.startswith(b'n:') and block:
                yield b''.join(block)
                block = []
            if block or line.startswith(b'n:'):
                block.append(line)
        if block:
            yield b''.join(block)


def context_spec(run):
    command = json.loads((run / 'logs/sopt_e0.json').read_text())['command']
    def option(name, default):
        return command[command.index(name) + 1] if name in command else default
    source = run / 'preprocess' / Path(option('-inputpolys', '')).name
    with (run / 'preprocess/cado_sopt_unsorted.txt').open() as f:
        header = f.readline()
    batch = re.search(r'batches of (\d+)', header)
    if not batch or not source.is_file():
        raise ValueError('cannot recover actual initial-sopt batch size/input')
    log = (run / 'logs/sopt_e0.log').read_text()
    workers = re.search(r'(\d+) workers', log)
    return source, {'batch': int(batch[1]), 'threads': int(option('-t', 8)),
                    'slice': float(option('-slice', 10)), 'device': int(option('-dev', 0)),
                    'original_workers': int(workers[1]) if workers else None}


def extract_context(source, case, execution, out, warmup=1, full_prefix=False):
    target, batch = case['index'] - 1, execution['batch']
    batch_start = target // batch * batch
    first = 0 if full_prefix else max(0, batch_start - warmup * batch)
    stop = batch_start + batch
    before = sha(source)
    count, found = 0, False
    with out.open('wb') as f:
        for index, block in enumerate(raw_blocks(source)):
            if index >= stop:
                break
            if index >= first:
                f.write(block)
                count += 1
            if index == target:
                values = {k.decode(): int(v) for k, v in re.findall(
                    rb'^(n|Y[01]|c\d+):\s*(-?\d+)\s*$', block, re.M)}
                if values != case['raw']:
                    raise ValueError('original input index does not match the regression fixture')
                found = True
    if not found or sha(source) != before:
        raise ValueError('context source changed or target index absent')
    return {'source': str(source), 'source_sha256': before, 'input_sha256': sha(out),
            'original_index_zero_based': target, 'first_index_zero_based': first,
            'trace_index': target - first, 'polynomials': count, 'execution': execution,
            'full_prefix': full_prefix, 'warmup_batches': (batch_start - first) // batch,
            'limitation': 'Preserves input order and batch boundaries, not nondeterministic warp scheduling or allocator bytes.'}


def build(out, mode, inc_flags, libs, nvcc):
    """Compile from frozen local sources; hash actual compiler dependencies and libraries."""
    src = out / 'source'
    shutil.copy2(Path(__file__), out / 'reproducer.py')
    shutil.copytree(GPU_DIR / 'include', src / 'include')
    (src / 'tools').mkdir()
    for name in ('s23_sopt.cu', 'sopt_trace.h', 'host_log_check.h', 'cado_io.c', 'cado_io.h'):
        shutil.copy2(GPU_DIR / 'tools' / name, src / 'tools' / name)
    includes = ['-I' + str(src / 'include'), '-I' + str(src / 'tools')]
    cc = ['gcc', '-O2', '-std=c99', '-fopenmp', *inc_flags, *includes]
    cxx = ['g++', '-O3', '-march=native', '-ffp-contract=off', '-std=c++17', '-fopenmp', *includes]
    dependencies = set(src.rglob('*.h')) | set(src.rglob('*.c')) | set(src.rglob('*.cu'))
    dependency_commands = []
    for compiler, source in ((cc, src / 'tools/cado_io.c'), (cxx, src / 'tools/s23_sopt.cu')):
        args = compiler + (['-x', 'c++'] if source.suffix == '.cu' else [])
        dependency_commands.append([*args, '-MM', str(source)])
    if mode == 'gpu':
        dependency_commands.append([nvcc, '-std=c++17', '-arch=sm_120', *includes,
                                    '-M', str(src / 'tools/s23_sopt.cu')])
        dependencies.update((Path(nvcc).resolve().parent.parent / 'nvvm/libdevice').glob('*.bc'))
    for command in dependency_commands:
        dep = subprocess.check_output(command, text=True)
        dependencies.update(Path(p).resolve() for p in shlex.split(dep.replace('\\\n', ' ').split(':', 1)[1]))
    dependencies.update(Path(p).resolve() for p in libs if Path(p).is_file())
    before = {str(p): sha(p) for p in sorted(dependencies)}
    commands = [[*cc, '-c', str(src / 'tools/cado_io.c'), '-o', str(out / 'cado_io.o')]]
    cpu = out / 's23_sopt_cpu'
    link = ['-x', 'c++', str(src / 'tools/s23_sopt.cu'), '-x', 'none', str(out / 'cado_io.o')]
    commands.append([*cxx, *link, '-o', str(cpu), *libs])
    experimental = None
    if mode == 'perturb':
        inc = out / 'perturbed_include'
        inc.mkdir()
        shutil.copy2(src / 'include/sopt.h', inc / 'sopt.h')
        norms = (src / 'include/norms.h').read_text()
        marker = '    return sopt_log(tt) / 2;'
        if norms.count(marker) != 1:
            raise RuntimeError('norms.h changed; review the perturbation point')
        (inc / 'norms.h').write_text('#include <cstdlib>\n#include <cstring>\n' + norms.replace(marker, PERTURB))
        experimental = out / 's23_sopt_perturbed_cpu'
        commands.append([cxx[0], '-I' + str(inc), *cxx[1:], *link, '-o', str(experimental), *libs])
        before.update({str(p): sha(p) for p in inc.iterdir()})
    elif mode == 'gpu':
        experimental = out / 's23_sopt'
        commands.append([nvcc, '-O2', '-std=c++17', '-arch=sm_120', '--fmad=false', *includes,
                         '-Xcompiler', '-fopenmp,-ffp-contract=off', str(src / 'tools/s23_sopt.cu'),
                         str(out / 'cado_io.o'), '-o', str(experimental), *libs, '-lgomp'])
    with (out / 'build.log').open('w') as log:
        for cmd in commands:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    if any(sha(Path(p)) != digest for p, digest in before.items()):
        raise RuntimeError('build dependencies changed during compilation')
    manifest = {'commands': commands, 'dependency_commands': dependency_commands, 'dependencies_sha256': before,
                'compilers': {name: subprocess.check_output([name, '--version'], text=True)
                              for name in dict.fromkeys(c[0] for c in commands)},
                'libc': os.confstr('CS_GNU_LIBC_VERSION'),
                'binaries_sha256': {str(p): sha(p) for p in (cpu, experimental) if p}}
    # Keep dynamic libm/libc identity as well as static CADO library hashes.
    manifest['runtime_libraries_sha256'] = {}
    for line in subprocess.check_output(['ldd', str(cpu)], text=True).splitlines():
        for token in line.split():
            if token.startswith('/') and Path(token).is_file():
                manifest['runtime_libraries_sha256'][token] = sha(Path(token))
    (out / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return cpu, experimental


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['cpu', 'perturb', 'gpu'])
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--case', choices=['c181', 'c179_576203'])
    parser.add_argument('--context-run', type=Path, help='original pipeline run; preserve its actual batch layout')
    context = parser.add_mutually_exclusive_group()
    context.add_argument('--warmup-batches', type=int, default=1)
    context.add_argument('--full-prefix', action='store_true', help='replay every preceding batch')
    parser.add_argument('--prepare-only', action='store_true', help='extract/validate inputs without building or executing')
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    if not out.is_relative_to((GPU_DIR / 'runs').resolve()):
        parser.error('--run-dir must be under stage23_gpu/runs')
    report_bytes = SOURCE.read_bytes()
    cases = json.loads(report_bytes)['size_optimization_regressions']
    if args.case:
        cases = [c for c in cases if c['run'] == args.case]
    if args.context_run and (not args.case or args.mode == 'perturb'):
        parser.error('--context-run requires --case and cpu/gpu mode')
    if args.warmup_batches < 0 or args.timeout < 1:
        parser.error('warmup must be nonnegative and timeout positive')
    # Validate inputs, toolchains and libraries BEFORE reserving the output directory.
    context_input = context_spec(args.context_run.resolve()) if args.context_run else None
    nvcc = shutil.which('nvcc')
    inc_flags = libs = []
    if not args.prepare_only:
        for compiler in ('gcc', 'g++', *(['nvcc'] if args.mode == 'gpu' else [])):
            if not shutil.which(compiler):
                parser.error('missing compiler: ' + compiler)
        fmt = ROOT / 'stage23_bench/tools/polyfmt.py'
        inc_flags = shlex.split(subprocess.check_output([sys.executable, str(fmt), '--cado-inc', ''], text=True))
        libs = shlex.split(subprocess.check_output([sys.executable, str(fmt), '--cado-libs', ''], text=True))
        if not inc_flags or not libs or any(not Path(p).is_file() for p in libs if p.endswith('.a')):
            parser.error('missing CADO build dependencies')
    if out.exists():
        parser.error('--run-dir must be new')
    out.mkdir(parents=True)
    (out / 'source_report.json').write_bytes(report_bytes)
    replay = None
    if context_input:
        replay = extract_context(context_input[0], cases[0], context_input[1], out / 'context.poly',
                                 args.warmup_batches, args.full_prefix)
        (out / 'context.json').write_text(json.dumps(replay, indent=2) + '\n')
    for case in cases:
        (out / (case['run'] + '.poly')).write_text(cado_poly_text(case['raw']) + '\n')
    if args.prepare_only:
        print(json.dumps(replay or {'cases': [c['run'] for c in cases]}, indent=2))
        return
    binary, experimental = build(out, args.mode, inc_flags, libs, nvcc)
    summary = {'mode': args.mode, 'source_sha256': sha(out / 'source_report.json'), 'cpu_binary_sha256': sha(binary),
               'experimental_binary_sha256': sha(experimental) if experimental else None,
               'build_manifest_sha256': sha(out / 'build.json'),
               'reproducer_sha256': sha(Path(__file__)), 'context': replay,
               'device_executed': args.mode == 'gpu',
               'interpretation': ('Synthetic perturbation is not a CUDA math implementation or a device check.'
                                  if args.mode == 'perturb' else 'Unmodified numerical implementation.'),
               'cases': []}
    for case in cases:
        source = out / (case['run'] + '.poly')
        prefix = out / (case['run'] + '_cpu')
        baseline = capture(binary, source, prefix, 0)
        if 'identical to CADO: 1 of 1' not in prefix.with_suffix('.err').read_text():
            raise RuntimeError('CPU port does not match current-host CADO')
        result = {'run': case['run'], 'cpu_matches_current_cado': True,
                  'cpu_matches_historical_reference': matches(baseline[-1], case['cpu'])}
        if replay and args.mode == 'cpu':
            experimental = binary
        if experimental:
            seed = {'c181': 10, 'c179_576203': 7}[case['run']] if args.mode == 'perturb' else 0
            if args.mode == 'perturb':
                zero = capture(experimental, source, out / (case['run'] + '_seed0'), 0)
                if baseline != zero:
                    raise RuntimeError('perturbed build with seed 0 must match every baseline phase')
            trial = capture(experimental, out / 'context.poly' if replay else source,
                            out / (case['run'] + '_trial'), replay['trace_index'] if replay else 0,
                            cpu=args.mode != 'gpu', seed=seed,
                            execution=replay['execution'] if replay else None, timeout=args.timeout)
            if replay:
                with (out / (case['run'] + '_trial.out')).open() as f:
                    header = f.readline()
                actual_batch = re.search(r'batches of (\d+)', header)
                if not actual_batch or int(actual_batch[1]) != replay['execution']['batch']:
                    raise RuntimeError('replay batch size was changed by the executable; context not preserved')
                log = (out / (case['run'] + '_trial.err')).read_text()
                workers = re.search(r'(\d+) workers', log)
                result['replay_workers'] = int(workers[1]) if workers else None
            changed = [a['phase'] for a, b in zip(baseline, trial) if integers(a) != integers(dict(b, index=0))]
            result.update(seed=seed, integer_difference_phases=changed,
                          matches_cpu_coefficients=(trial[-1].get('f'), trial[-1].get('g')) ==
                                                   (baseline[-1].get('f'), baseline[-1].get('g')),
                          matches_saved_gpu_coefficients=matches(trial[-1], case['gpu']),
                          cpu_objective=baseline[-1]['objective'], trial_objective=trial[-1]['objective'],
                          changed_descent_candidates=[a['index'] for a, b in zip(
                              baseline[3]['candidates'], trial[3]['candidates']) if integers(a) != integers(b)])
        summary['cases'].append(result)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
