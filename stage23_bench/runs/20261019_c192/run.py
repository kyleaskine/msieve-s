#!/usr/bin/env python3
"""Freeze C192, trace its effort-5 winner, and run the independent GPU pipeline.

All full logs, inputs, binaries, pools and checkpoints stay in --run-dir.
Use prepare, trace, cado, pipeline, driver-check or finish. The finish phase
waits for the full pipeline, compares controls, and archives compact evidence.
The user's provisional sieve settings are models, not measured sieve yields.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from gpu_pipeline import Run, digest, write_json, validate_poly, search
from polyfmt import (cado_poly_text, parse_cado_ropt, read_cado_blocks,
                     rotation_between)


def prepare(run):
    marker = run / 'prepared.json'
    if marker.exists():
        return json.loads(marker.read_text())
    run.mkdir(parents=True, exist_ok=True)
    cpu = run / 'cpu_reference'
    shutil.copytree(ROOT / 'pipeline_results', cpu, dirs_exist_ok=True)
    work = cpu / 'pipeline_work'
    work.mkdir(exist_ok=True)
    for name in ('resopt_output.txt', 'resopt_sorted.txt', 'resopt_msieve_sorted.ms', 'top1000_input.ms'):
        shutil.copy2(ROOT / 'pipeline_work' / name, work / name)
    originals = parse_cado_ropt(cpu / 'cado_ropt_orig.txt')
    winner = max(originals, key=lambda p: p['MurphyE'])
    n = winner['n']
    if len(str(n)) != 192:
        raise ValueError('expected the October 19 C192')
    seeds = read_cado_blocks(cpu / 'best150_cado.txt')
    index = originals.index(winner)
    seed = seeds[index][1]
    relation = rotation_between(seed, winner)
    if relation is None or len(relation['rotation']) != 2:
        raise ValueError('winner is not a linear rotation of the corresponding seed')
    u = relation['rotation'][1]
    v = relation['rotation'][0] - u * relation['t']
    fixtures = run / 'fixtures'
    fixtures.mkdir(exist_ok=True)
    references = []
    for label, filename in [('cado_orig', 'cado_ropt_orig.txt'), ('cado_inv', 'cado_ropt_inv.txt'),
            ('deep_orig', 'cado_deep_ropt_effort10/cado_deep_ropt_orig.txt'),
            ('deep_inv', 'cado_deep_ropt_effort10/cado_deep_ropt_inv.txt')]:
        rows = parse_cado_ropt(cpu / filename)
        for i, p in enumerate(rows):
            validate_poly(p, n)
            references.append(dict(method=label, index=i, poly=p))
    deep = read_cado_blocks(cpu / 'cado_deep_ropt_effort10/deep_cado_orig_input.txt')
    matches = [i for i, (_, p) in enumerate(deep) if cado_poly_text(p) == cado_poly_text(seed)]
    (fixtures / 'seed.poly').write_text(cado_poly_text(seed))
    inverse = dict(seed, **{f'c{i}': -seed[f'c{i}'] for i in range(6)})
    (fixtures / 'seed_inv.poly').write_text(cado_poly_text(inverse))
    (fixtures / 'cado_winner.poly').write_text(cado_poly_text(winner))
    write_json(fixtures / 'references.json', references)
    workflow = Run(run)
    for binary in ('s23_sopt', 's23_ropt', 's23_ropt_cpu'):
        workflow.freeze(binary, ROOT / 'stage23_gpu/build' / binary)
    workflow.freeze('cado_murphy')
    workflow.freeze('polyselect_ropt', Path('/home/kylea/cado-nfs/build/DESKTOP-3J4UC68/polyselect/polyselect_ropt'))
    write_json(run / 'scoring.json', dict(params='job:32,33,15,80e6', lattice='15,16384',
        qband='80e6,500e6,7', lpbr=32, lpba=33, rlim=160000000, alim=240000000,
        region='15e', provisional=True, source='User, 2026-10-19',
        note='Factor-base limits recorded for future test sieving; model uses LPB, region and q range.'))
    meta = dict(n=n, cpu_seed_index=index, winner_rotation=dict(u=u, v=v, **relation),
        identical_deep_input_indices=matches,
        sha256={str(p.relative_to(run)): digest(p) for p in cpu.rglob('*') if p.is_file()})
    write_json(marker, meta)
    print('Prepared C192:', 'seed', index, 'rotation', (u, v), 'identical deep input', matches, flush=True)
    return meta


def command(run, tag, argv, accepted=(0,)):
    dest = run / 'experiments'
    dest.mkdir(exist_ok=True)
    record = dest / (tag + '.json')
    argv = list(map(str, argv))
    if record.exists():
        prior = json.loads(record.read_text())
        if prior['command'] == argv and prior['returncode'] in accepted and digest(argv[0]) == prior['binary_sha256']:
            print('Reusing', tag, flush=True)
            return prior
        raise ValueError(f'changed or failed experiment: {tag}; use a new tag/run')
    log = dest / (tag + '.log')
    print('Running', tag, flush=True)
    start = time.monotonic()
    with log.open('w') as out:
        proc = subprocess.run(argv, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    text = log.read_text()
    result = dict(command=argv, returncode=proc.returncode, seconds=time.monotonic()-start,
                  binary_sha256=digest(argv[0]), log_sha256=digest(log),
                  trace=[line for line in text.splitlines() if 'trace-cell' in line])
    for flag in ('-poly', '-pool-in', '-inputpolys'):
        if flag in argv:
            result[flag[1:] + '_sha256'] = digest(argv[argv.index(flag)+1])
    if proc.returncode in accepted and '-out' in argv:
        output = Path(argv[argv.index('-out')+1])
        rows = read_cado_blocks(output)
        if not rows:
            raise ValueError(f'{tag}: empty export')
        for _, p in rows:
            validate_poly(p, json.loads((run / 'prepared.json').read_text())['n'])
        result.update(verified_exports=len(rows), output_sha256=digest(output))
    write_json(record, result)
    if proc.returncode not in accepted:
        raise RuntimeError(f'{tag}: exit {proc.returncode}; see {log}')
    print(tag, f"{result['seconds']:.2f}s", result['trace'][-1:] or '', flush=True)
    return result


def trace(run, meta, gpu=False):
    ref = meta['winner_rotation']
    common = ['-poly', run / 'fixtures/seed.poly', '-search', '-budget', 3,
              '-maxlines', 1000, '-umax', 20000, '-aw', 1.3, '-rerank', 4096,
              '-refine', 200, '-top', 8, '-t', 2, '-launch', 30,
              '-trace-cell', ref['u'], ref['v']]
    cases = [('dense_breadth', ['-maxcells', '2e11']),
             ('dense_depth', ['-maxcells', '6e12']),
             ('classes16', ['-maxcells', '2e11', '-classes', 2520, 16, '-class-explore', 4])]
    for tag, extra in cases:
        argv = [run / 'bin' / ('s23_ropt' if gpu else 's23_ropt_cpu'), *common, *extra]
        if gpu:
            argv += ['-check', 128, '-pool-out', run / 'experiments' / (tag+'.pool'),
                     '-out', run / 'experiments' / (tag+'.poly')]
        else:
            argv += ['-plan']
            tag += '_plan'
        command(run, tag, argv)


def cado(run):
    for effort in (5, 10):
        for side in ('orig', 'inv'):
            seed = run / 'fixtures' / ('seed.poly' if side == 'orig' else 'seed_inv.poly')
            command(run, f'cado_e{effort}_{side}', [run / 'bin/polyselect_ropt',
                    '-ropteffort', effort, '-inputpolys', seed])


def screen(run, meta):
    common = ['-poly', run/'fixtures/seed.poly', '-search', '-pool-in', run/'experiments/dense_depth.pool',
              '-aw', 1.3, '-alpha-screen', 4096, 500, '-alpha-screen-exact-size',
              '-rerank', 512, '-refine', 200, '-top', 8, '-t', 2,
              '-trace-cell', meta['winner_rotation']['u'], meta['winner_rotation']['v']]
    for kind, binary, extra in [('cpu', 's23_ropt_cpu', []),
            ('gpu', 's23_ropt', ['-alpha-screen-gpu', '-alpha-screen-check'])]:
        command(run, 'screen_'+kind, [run/'bin'/binary, *common, *extra,
                '-out', run/'experiments'/('screen_'+kind+'.poly')])
    cpu = run/'experiments/screen_cpu.poly'
    gpu = run/'experiments/screen_gpu.poly'
    if cpu.read_bytes() != gpu.read_bytes():
        raise ValueError('checked GPU screen differs from CPU exact-size screen')
    if ': result retained' not in (run/'experiments/screen_gpu.log').read_text():
        raise ValueError('GPU screen lost the C192 target rotation')
    print('C192 GPU screen: 4096 to 512; winner retained, all 200 exports identical to CPU', flush=True)


def pipeline(run):
    # Keep a frozen raw snapshot and use the existing resumable full GPU path.
    workflow = Run(run)
    (run / 'preprocess').mkdir(exist_ok=True)
    workflow.snapshot(ROOT / 'msieve.dat.ms', 'raw')
    argv = [sys.executable, ROOT / 'scripts/gpu_pipeline.py', 'all', '--input', ROOT / 'msieve.dat.ms',
            '--run-dir', run, '--extract', 1000, '--seeds', 300, '--threads', 8,
            '--workers', 2, '--launch', 30, '--params', 'job:32,33,15,80e6',
            '--lattice', '15,16384', '--qband', '80e6,500e6,7']
    write_json(run / 'pipeline_command.json', dict(argv=list(map(str, argv))))
    with (run / 'driver.log').open('a') as log:
        subprocess.run(list(map(str, argv)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def driver_check(run):
    """Real CUDA checks of batching, checkpoint reuse and corrupted-output repair."""
    results = []
    for batch_size in (1, 16):
        dest = run / f'driver_check_{batch_size}'
        (dest / 'seeds').mkdir(parents=True, exist_ok=True)
        seeds = []
        for i, fixture in enumerate(('seed.poly', 'seed_inv.poly', 'cado_winner.poly'), 1):
            name = f's{i:04d}'
            shutil.copy2(run / 'fixtures' / fixture, dest / 'seeds' / (name+'.poly'))
            seeds.append(dict(search=name, parent=i))
        args = SimpleNamespace(budget=3, rerank=128, keep=32, score_top=32, threads=2,
                device=0, launch=30, maxlines=16, umax=8, u_scan=0, residue='', classes='',
                workers=2, ropt_batch_size=batch_size)
        workflow = Run(dest)
        n = json.loads((run/'prepared.json').read_text())['n']
        start = time.monotonic()
        search(workflow, args, seeds, run/'bin/s23_ropt', 'breadth', 1000000, n, save_pool=True)
        before = {str(p.relative_to(dest)): digest(p) for p in (dest/'checkpoints').rglob('*.json')}
        search(workflow, args, seeds, run/'bin/s23_ropt', 'breadth', 1000000, n, save_pool=True)
        assert before == {str(p.relative_to(dest)): digest(p) for p in (dest/'checkpoints').rglob('*.json')}
        if batch_size > 1:
            victim = dest/'breadth/s0002.poly'
            expected = victim.read_bytes()
            victim.write_text('corrupted export')
            search(workflow, args, seeds, run/'bin/s23_ropt', 'breadth', 1000000, n, save_pool=True)
            assert victim.read_bytes() == expected
        hashes = {p.name: digest(p) for p in (dest/'breadth').iterdir()}
        results.append(dict(batch_size=batch_size, sha256=hashes, seconds=time.monotonic()-start))
    assert results[0]['sha256'] == results[1]['sha256']
    write_json(run/'driver_check.json', dict(records=results, identical_exports_and_pools=True,
        note='Correctness check on a shared GPU, not a performance benchmark. Resume and corruption repair checked.'))
    print('Three live GPU searches: separate and batched outputs/pools identical; resume and repair passed', flush=True)


def driver_screen_check(run, workers=1):
    """Audit actual driver forwarding on all sixteen independent C192 depth pools."""
    pools = sorted((run/'depth').glob('*.pool'))
    if len(pools) != 16:
        raise ValueError('need the completed 16-pool C192 depth run')
    records = []
    for mode in ('cpu', 'gpu_checked', 'gpu_unchecked'):
        dest = run / ('driver_screen_' + mode)
        for directory in ('seeds', 'source'):
            (dest/directory).mkdir(parents=True, exist_ok=True)
        seeds = []
        for i, pool in enumerate(pools):
            name = pool.stem
            shutil.copy2(run/'seeds'/(name+'.poly'), dest/'seeds'/(name+'.poly'))
            shutil.copy2(pool, dest/'source'/pool.name)
            seeds.append(dict(search=name, parent=i))
        args = SimpleNamespace(budget=3, rerank=512, keep=200, score_top=200, threads=2,
                device=0, launch=30, maxlines=1000, umax=20000, u_scan=0, residue='', classes='',
                workers=workers, ropt_batch_size=16, alpha_screen='4096,500', alpha_screen_exact_size=True,
                alpha_screen_gpu=mode!='cpu', alpha_screen_check=mode=='gpu_checked')
        workflow = Run(dest)
        n = json.loads((run/'prepared.json').read_text())['n']
        binary = run/'bin'/('s23_ropt_cpu' if mode=='cpu' else 's23_ropt')
        rows = search(workflow, args, seeds, binary, 'screen', 6e12, n, replay_phase='source')
        outputs = []
        for seed in seeds:
            name = seed['search']
            path = dest/'screen'/(name+'.poly')
            if mode != 'cpu' and path.read_bytes() != (run/'driver_screen_cpu/screen'/path.name).read_bytes():
                raise ValueError(f'{mode}: changed exports for {name}')
            baseline = [cado_poly_text(p) for _,p in read_cado_blocks(run/'depth'/path.name)]
            screened = {cado_poly_text(p) for _,p in read_cado_blocks(path)}
            outputs.append(dict(seed=name, input_sha256=digest(dest/'seeds'/path.name),
                pool_sha256=digest(dest/'source'/(name+'.pool')), output_sha256=digest(path),
                baseline_exports_retained=sum(p in screened for p in baseline),
                baseline_top32_retained=sum(p in screened for p in baseline[:32])))
        records.append(dict(mode=mode, exports=len(rows), outputs=outputs))
        print(mode, len(rows), 'validated exports, baseline retained',
              sum(p['baseline_exports_retained'] for p in outputs), flush=True)
    write_json(run/'driver_screen_check.json', dict(records=records, workers=workers,
        cpu_gpu_exports_identical=True,
        note='Saved-depth-pool screen audit, not an end-to-end search or timing comparison. Final Murphy refinement remains unscreened.'))


def finish(run):
    status = run/'finish_status.json'
    try:
        write_json(status, dict(stage='waiting for full GPU pipeline', updated=time.time()))
        deadline = time.monotonic()+24*3600
        while True:
            state = json.loads((run/'status.json').read_text())
            if state['stage'] == 'complete':
                break
            if state['stage'] in ('failed', 'interrupted'):
                raise RuntimeError(f'GPU pipeline did not complete: {state}')
            if time.monotonic() > deadline:
                raise TimeoutError('full GPU pipeline incomplete after 24 hours')
            time.sleep(15)
        write_json(status, dict(stage='comparing CPU and GPU results', updated=time.time()))
        with (run/'compare.log').open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name('compare.py')), '--run-dir', str(run)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name('save.py')), '--run-dir', str(run)],
                       cwd=ROOT, check=True)
        write_json(status, dict(stage='complete', updated=time.time(), summary=str(run/'comparison/summary.json')))
    except BaseException as error:
        write_json(status, dict(stage='failed', error=str(error), updated=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'trace', 'cado', 'screen', 'pipeline', 'driver-check', 'driver-screen-check', 'finish'])
    parser.add_argument('--gpu', action='store_true', help='execute CUDA traces instead of CPU plans')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1,
                        help='worker count for driver-screen-check (default: 1)')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'stage23_gpu/runs/20261019_c192')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    meta = prepare(run)
    if args.phase == 'trace': trace(run, meta, args.gpu)
    elif args.phase == 'cado': cado(run)
    elif args.phase == 'screen': screen(run, meta)
    elif args.phase == 'pipeline': pipeline(run)
    elif args.phase == 'driver-check': driver_check(run)
    elif args.phase == 'driver-screen-check': driver_screen_check(run, args.workers)
    elif args.phase == 'finish': finish(run)


if __name__ == '__main__':
    main()
