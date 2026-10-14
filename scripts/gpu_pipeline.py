#!/usr/bin/env python3
"""Resumable stage-2/3 GPU pipeline; invoked by nfs_optimize.sh --gpu.

Run directories own their inputs, frozen executables, checkpoints and results.
Preprocessing needs only raw degree-5 polynomials. Pipeline scoring additionally
needs explicit job bounds. No stage-1 search is started by this driver.
"""
import argparse
import concurrent.futures
import configparser
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_SETTINGS = {'device', 'threads', 'workers', 'launch', 'slice'}
EXECUTION_FLAGS = {'-dev', '-t', '-launch', '-slice'}
# Bump only the affected format/algorithm when saved results become incompatible.
# Source hashes are provenance, not an instruction to repeat expensive work.
STEP_SCHEMAS = {'index_e0': 1, 'index_resopt': 1, 'seeds': 2}
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from content_seeds import content_seeds
from polyfmt import (cado_murphy_binary, cado_poly_text, iter_sopt,
                     msieve_ms_line, read_cado_blocks, resultant)
from score_polys import score_bounds


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def semantic_parameters(parameters):
    parameters = {k: v for k, v in parameters.items() if k not in EXECUTION_SETTINGS}
    if 'argv' in parameters:
        argv, out = iter(parameters['argv']), []
        for arg in argv:
            if arg in EXECUTION_FLAGS:
                next(argv)
            else:
                out.append(arg)
        parameters['argv'] = out
    return parameters


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_poly(p, n=None):
    required = {'n', 'Y0', 'Y1', *(f'c{i}' for i in range(6))}
    if not required <= p.keys() or any(re.fullmatch(r'c\d+', k) and int(k[1:]) > 5 for k in p):
        raise ValueError('GPU pipeline requires complete degree-5 polynomial blocks')
    if p['n'] <= 1 or not p['Y1'] or not p['c5'] or (n is not None and p['n'] != n):
        raise ValueError('invalid polynomial or mixed input n')
    r = resultant(p)
    if not r or r % p['n']:
        raise ValueError('polynomial resultant is zero or not divisible by n')


def raw_polys(path):
    """Stream raw CADO/msieve multiline blocks, including adjacent n: blocks."""
    p = {}
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = re.fullmatch(r'(n|Y[01]|c\d+)\s*:\s*(-?\d+)', line)
            if match:
                key, value = match.groups()
                if key == 'n' and p:
                    validate_poly(p)
                    yield p
                    p = {}
                if key in p:
                    raise ValueError(f'{path}:{number}: duplicate field {key}')
                p[key] = int(value)
            elif not re.match(r'(skew|type|m)\s*:', line):
                raise ValueError(f'{path}:{number}: unrecognized polynomial line')
    if p:
        validate_poly(p)
        yield p


def deduplicate(src, dst, metadata):
    # Disk-backed identity table keeps memory bounded on multi-million-poly jobs.
    with tempfile.TemporaryDirectory(dir=dst.parent) as tmp:
        with sqlite3.connect(str(Path(tmp) / 'seen.sqlite')) as db, dst.open('w') as out:
            db.execute('PRAGMA cache_size=-16384')
            db.execute('CREATE TABLE seen (poly TEXT PRIMARY KEY) WITHOUT ROWID')
            count = kept = 0
            n = None
            for p in raw_polys(src):
                n = p['n'] if n is None else n
                if p['n'] != n:
                    raise ValueError('raw input contains multiple n values')
                count += 1
                text = cado_poly_text(p)
                if db.execute('INSERT OR IGNORE INTO seen VALUES (?)', (text,)).rowcount:
                    out.write(text + '\n')
                    kept += 1
            if not kept:
                raise ValueError('raw input contains no polynomials')
    write_json(metadata, dict(n=n, input_count=count, unique_count=kept))


def sopt_text(p, index):
    return (f'### Size-optimized polynomial ({index}) ###\n' + cado_poly_text(p) +
            f"# side 1 lognorm {p['lognorm']:.2f}, exp_E {p['exp_E']:.2f}, "
            f"alpha {p['alpha']:.2f} (proj {p['proj']:.2f}), {p['rroots']} real roots\n\n")


def index_sopt(src, index, sorted_path, ms_path, metadata, expected=None):
    """Preserve each raw/optimized pair by row, never join just on Y0/Y1."""
    tmp = index.with_suffix('.partial.sqlite')
    tmp.unlink(missing_ok=True)
    with sqlite3.connect(str(tmp)) as db:
        db.execute('PRAGMA cache_size=-16384')
        db.execute('PRAGMA temp_store=FILE')
        db.execute('CREATE TABLE polys (id INTEGER PRIMARY KEY, expe REAL, raw TEXT, opt TEXT, ms TEXT)')
        count, n = 0, None
        headers = {}
        for count, (raw, opt) in enumerate(iter_sopt(src, header_counts=headers), 1):
            n = opt.get('n') if n is None else n
            validate_poly(raw, n)
            validate_poly(opt, n)
            if not all(math.isfinite(opt[k]) for k in ('exp_E', 'lognorm', 'alpha', 'proj', 'skew')):
                raise ValueError('nonfinite size-optimization statistics')
            db.execute('INSERT INTO polys VALUES (?, ?, ?, ?, ?)',
                       (count, opt['exp_E'], cado_poly_text(raw), sopt_text(opt, count - 1), msieve_ms_line(opt)))
        # iter_sopt ignores incomplete pairs: reject them rather than losing inputs.
        inputs, optimized = headers['raw'], headers['opt']
        if not count or count != optimized or count != inputs or (expected is not None and count != expected):
            raise ValueError(f'incomplete sopt output: {count} pairs, {inputs}/{optimized} headers, expected {expected}')
        db.execute('CREATE INDEX ranked ON polys(expe, id)')
        with sorted_path.open('w') as out:
            for (text,) in db.execute('SELECT opt FROM polys ORDER BY expe, id'):
                out.write(text)
        with ms_path.open('w') as out:
            for (text,) in db.execute('SELECT ms FROM polys ORDER BY expe, ms COLLATE BINARY'):
                out.write(text + '\n')
    tmp.replace(index)
    write_json(metadata, dict(n=n, count=count))


class Run:
    def __init__(self, directory):
        self.path = directory
        self.stop = threading.Event()
        sources = [Path(__file__), ROOT / 'stage23_bench/tools/polyfmt.py',
                   ROOT / 'stage23_bench/tools/content_seeds.py', ROOT / 'stage23_bench/tools/score_polys.py']
        self.version = {str(p.relative_to(ROOT)): digest(p) for p in sources}

    def state(self, stage, **extra):
        write_json(self.path / 'status.json', dict(stage=stage, updated=time.time(), **extra))
        print(stage, ' '.join(f'{k}={v}' for k, v in extra.items()), flush=True)

    def step(self, name, signature, outputs, action):
        marker = self.path / 'checkpoints' / (name + '.json')
        signature = dict(schema=STEP_SCHEMAS.get(name, 1), parameters=semantic_parameters(signature))
        if marker.exists():
            old = read_json(marker)
            prior = old['signature']
            # Legacy checkpoints used one global source hash. Keep their verified
            # artifacts when the stage's parameters still agree.
            same = (semantic_parameters(prior['parameters']) == signature['parameters'] and
                    prior.get('schema', signature['schema']) == signature['schema'])
            if not same and name == 'pipeline_settings':
                raise ValueError(f'{name}: search or scoring settings changed; use a new --run-dir')
            if same and all(p.is_file() and digest(p) == old['outputs'].get(str(p.relative_to(self.path))) for p in outputs):
                print(f'Reusing {name}', flush=True)
                return
            print(f'Rebuilding {name}: changed stage inputs, schema or output', flush=True)
            marker.unlink()
        for p in outputs:
            p.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        action()
        hashes = {str(p.relative_to(self.path)): digest(p) for p in outputs}
        write_json(marker, dict(signature=signature, source_hashes=self.version,
                                outputs=hashes, seconds=time.monotonic() - start))

    def freeze(self, name, source=None):
        dst = self.path / 'bin' / name
        marker = self.path / 'bin' / (name + '.json')
        if marker.exists():
            old = read_json(marker)
            if source is not None and str(source) != old['source']:
                raise ValueError(f'{name}: executable path changed; use a new --run-dir')
            if not dst.is_file() or digest(dst) != old['sha256']:
                raise ValueError(f'frozen {name} changed or missing; use a new --run-dir')
            return dst
        if source is None:
            source = Path(cado_murphy_binary())
        if not source.is_file() or not os.access(source, os.X_OK):
            raise ValueError(f'missing executable {source}; build stage23_gpu with make s23_sopt s23_ropt')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dst)
        write_json(marker, dict(source=str(source), sha256=digest(dst)))
        return dst

    def command(self, name, argv, output=None, accepted=(0,)):
        argv = list(map(str, argv))
        log = self.path / 'logs' / (name + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        print(f'Running {name} (log: {log})', flush=True)
        with log.open('w') as err:
            out = output.open('w') if output else err
            try:
                with subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=err, start_new_session=True) as proc:
                    try:
                        while True:
                            if self.stop.is_set():
                                raise InterruptedError('pipeline interrupted')
                            try:
                                rc = proc.wait(timeout=5)
                                break
                            except subprocess.TimeoutExpired:
                                pass
                    except BaseException:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        raise
            finally:
                if out is not err:
                    out.close()
        write_json(log.with_suffix('.json'), dict(command=argv, returncode=rc, seconds=time.monotonic() - start))
        if rc not in accepted:
            raise RuntimeError(f'{name} exited {rc}; see {log}')
        return rc

    def pool_capable(self, binary):
        marker = self.path / 'bin/ropt_capabilities.json'
        if marker.exists():
            result = read_json(marker)
            if result['sha256'] == digest(binary):
                return result['pool_v1']
        p = subprocess.run([str(binary), '-capabilities'], text=True, capture_output=True, timeout=30)
        supported = p.returncode == 0 and 'ropt-pool-v1' in p.stdout.split()
        write_json(marker, dict(sha256=digest(binary), pool_v1=supported))
        return supported

    def snapshot(self, source, mode):
        """Own exactly the bytes hashed, including a bounded prefix of a live input."""
        provenance = self.path / 'preprocess/source.json'
        frozen = self.path / 'inputs' / ('source.' + mode)
        if provenance.exists():
            identity = read_json(provenance)
            if identity['mode'] != mode or identity['source'] != str(source):
                raise ValueError('input source changed; use a new --run-dir')
            if frozen.exists():
                if digest(frozen) != identity['sha256']:
                    raise ValueError('frozen input changed; use a new --run-dir')
                return identity, frozen
            if identity.get('snapshot'):
                raise ValueError('frozen input missing; restore it before resuming')
            # Upgrade an old run when its original bytes are still available.
            # Otherwise its verified preprocessing checkpoints can still resume.
            if not source.exists() or digest(source) != identity['sha256']:
                return identity, frozen
        frozen.parent.mkdir(parents=True, exist_ok=True)
        partial = frozen.with_suffix(frozen.suffix + '.partial')
        h = hashlib.sha256()
        with source.open('rb') as src, partial.open('wb') as dst:
            remaining = os.fstat(src.fileno()).st_size
            while remaining:
                block = src.read(min(remaining, 1024 * 1024))
                if not block:
                    raise ValueError('input truncated while snapshotting; retry with a stable input')
                dst.write(block)
                h.update(block)
                remaining -= len(block)
        identity = dict(mode=mode, source=str(source), sha256=h.hexdigest())
        if provenance.exists() and identity != read_json(provenance):
            raise ValueError('original input changed while restoring its snapshot')
        if not provenance.exists():
            identity['snapshot'] = str(frozen.relative_to(self.path))
        partial.replace(frozen)
        write_json(provenance, identity)
        return identity, frozen


def preprocess(run, args):
    base = run.path / 'preprocess'
    base.mkdir(exist_ok=True)
    unsorted = base / 'cado_sopt_unsorted.txt'
    provenance = base / 'source.json'
    source = args.initial_unsorted or args.input or args.default_input
    mode = 'sopt' if args.initial_unsorted else 'raw'
    if not args.initial_unsorted and not args.input and provenance.exists():
        prior = read_json(provenance)
        source, mode = Path(prior['source']), prior['mode']
    identity, source = run.snapshot(source, mode)
    if mode == 'sopt':
        run.step('import_sopt', identity, [unsorted],
                 lambda: shutil.copyfile(source, unsorted))
        expected = None
    else:
        deduped, raw_info = base / 'msieve.dat.deduped.ms', base / 'raw.json'
        run.step('raw', identity, [deduped, raw_info], lambda: deduplicate(source, deduped, raw_info))
        expected = read_json(raw_info)['unique_count']
        binary = run.freeze('s23_sopt', args.sopt_binary)
        run.step('sopt_e0', dict(input=digest(deduped), binary=digest(binary)), [unsorted],
                 lambda: run.command('sopt_e0', [binary, '-sopteffort', 0, '-t', args.threads,
                                                '-dev', args.device, '-slice', args.slice,
                                                '-inputpolys', deduped], unsorted))
    index_outputs = [base / 'index.sqlite', base / 'cado_sopt_output.txt',
                     base / 'cado_results_sorted.ms', base / 'info.json']
    run.step('index_e0', dict(input=digest(unsorted), expected=expected), index_outputs,
             lambda: index_sopt(unsorted, *index_outputs, expected=expected))
    return read_json(base / 'info.json')


def existing_preprocess(run):
    marker = run.path / 'checkpoints/index_e0.json'
    if not marker.exists():
        raise ValueError('run preprocess first, use pipeline --input RAW, or --initial-unsorted CADO_OUTPUT')
    old = read_json(marker)
    if old['signature'].get('schema', 1) != STEP_SCHEMAS['index_e0']:
        raise ValueError('preprocessing index format changed; rerun preprocess to update it')
    for name, expected in old['outputs'].items():
        p = run.path / name
        if not p.is_file() or digest(p) != expected:
            raise ValueError('preprocessing is incomplete or changed; rerun preprocess to repair it')
    return read_json(run.path / 'preprocess/info.json')


def select_seeds(run, args, info, sopt):
    work = run.path / 'pipeline'
    src = run.path / 'preprocess/index.sqlite'
    raw = work / 'top_input.ms'
    count = min(args.extract, info['count'])

    def extract():
        with sqlite3.connect(src.as_uri() + '?mode=ro', uri=True) as db, raw.open('w') as out:
            for (text,) in db.execute('SELECT raw FROM polys ORDER BY expe, id LIMIT ?', (count,)):
                out.write(text + '\n')

    run.step('select_inputs', dict(index=digest(src), count=count), [raw], extract)
    resopt = work / 'resopt_output.txt'
    run.step('resopt', dict(input=digest(raw), binary=digest(sopt), effort=args.resopt_effort), [resopt],
             lambda: run.command('resopt', [sopt, '-sopteffort', args.resopt_effort, '-t', args.threads,
                                           '-dev', args.device, '-slice', args.slice, '-inputpolys', raw], resopt))
    outputs = [work / 'index.sqlite', work / 'resopt_sorted.txt', work / 'resopt_msieve_sorted.ms', work / 'info.json']
    run.step('index_resopt', dict(input=digest(resopt), count=count), outputs,
             lambda: index_sopt(resopt, *outputs, expected=count))
    # Tie order matches LC_ALL=C sort -k10,10n used by the reference runs.
    pairs = list(iter_sopt(resopt))
    pairs.sort(key=lambda pair: (pair[1]['exp_E'], msieve_ms_line(pair[1])))
    seeds = []
    for rank, (_, p) in enumerate(pairs[:args.seeds], 1):
        name = f's{rank:04d}'
        seeds.append(dict(search=name, parent=rank, content=1, poly=p))
        for d, u, v, q in content_seeds(p, info['n']):
            seeds.append(dict(search=f'{name}_d{d}', parent=rank, content=d, u=u, v=v, poly=q))
    if args.residue:
        # Parent coordinates are (u0 + d*u', v0 + d*v'). Intersect the
        # parent's requested class with this content lattice before searching.
        seeds = [s for s in seeds if seed_residue(args.residue, s) is not None]
    paths = [run.path / 'seeds' / (s['search'] + '.poly') for s in seeds]

    def export():
        for seed, path in zip(seeds, paths):
            validate_poly(seed['poly'], info['n'])
            path.write_text(cado_poly_text(seed['poly']))
        write_json(run.path / 'seeds.json', seeds)

    signature = dict(input=digest(resopt), count=args.seeds)
    if args.residue:
        signature['residue'] = args.residue
    run.step('seeds', signature, [*paths, run.path / 'seeds.json'], export)
    return seeds


def seed_residue(residue, seed):
    m, u, v = map(int, residue.split(','))
    d = seed['content']
    g = math.gcd(m, d)
    u, v = u - seed.get('u', 0), v - seed.get('v', 0)
    if u % g or v % g:
        return None
    m //= g
    inverse = pow(d // g, -1, m) if m > 1 else 0
    return [m, (u // g * inverse) % m, (v // g * inverse) % m]


def search(run, args, seeds, binary, phase, cap, n, accurate=False, save_pool=False, replay_phase=None):
    def one(seed):
        name = seed['search']
        src = run.path / 'seeds' / (name + '.poly')
        dst = run.path / phase / (name + '.poly')
        partial = dst.with_suffix('.partial.poly')
        pool_path = dst.with_suffix('.pool')
        pool_partial = pool_path.with_suffix('.partial.pool')
        replay = run.path / replay_phase / (name + '.pool') if replay_phase else None
        argv = [binary, '-poly', src, '-search', '-budget', args.budget, '-maxcells', cap,
                '-aw', 1.3, '-rerank', args.rerank, '-top', 3, '-refine', args.keep,
                '-t', args.threads, '-dev', args.device, '-launch', args.launch,
                '-maxlines', args.maxlines, '-umax', args.umax, '-out', partial]
        if replay:
            argv += ['-pool-in', replay]
        elif save_pool:
            argv += ['-pool-out', pool_partial]
        if args.u_scan and not replay:
            argv += ['-u-scan', args.u_scan, '-u-basins', args.u_basins, '-u-radius', args.u_radius]
        if args.residue and not replay:
            argv += ['-residue', *seed_residue(args.residue, seed)]
        if args.classes and not replay:
            argv += ['-classes', *args.classes.split(','), '-class-explore', args.class_explore]
        if accurate:
            bounds = score_bounds(args.params)
            argv += ['-murphy', '-Bf', bounds['Bf'], '-Bg', bounds['Bg'], '-area', bounds['area'],
                     '-murphy-points', 16000, '-murphy-eval-points', 64000, '-maxeval', 384]

        def action():
            # An old frozen ropt uses 1 for partial Murphy failure; newer builds
            # use 3. Accept it only with a completed, validated partial export.
            rc = run.command(f'{phase}/{name}', argv, accepted=(0, 1, 3) if accurate else (0,))
            blocks = read_cado_blocks(partial)
            if not blocks:
                raise ValueError(f'{phase}/{name}: empty polynomial export')
            for _, p in blocks:
                validate_poly(p, n)
            if rc:
                log = (run.path / 'logs' / phase / (name + '.log')).read_text()
                if (not re.search(r'\b[1-9]\d* failed, \d+ skipped before MurphyE', log) or
                        f'# wrote {len(blocks)} polynomials to ' not in log or
                        re.search(r'BRUTE FORCE DIFFERS|vs brute force: [1-9]', log)):
                    raise RuntimeError(f'{phase}/{name} exited {rc} without a verified partial MurphyE export')
                print(f'{phase}/{name}: retaining {len(blocks)} valid cells after partial MurphyE failure; see log', flush=True)
            partial.replace(dst)
            if save_pool:
                if not pool_partial.is_file() or not pool_partial.stat().st_size:
                    raise ValueError(f'{phase}/{name}: missing candidate pool')
                pool_partial.replace(pool_path)

        signature = dict(argv=list(map(str, argv)), input=digest(src), binary=digest(binary))
        if replay:
            signature['pool'] = digest(replay)
        run.step(f'{phase}/{name}', signature, [dst, pool_path] if save_pool else [dst], action)
        return name

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = [pool.submit(one, seed) for seed in seeds]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            run.state(phase, completed=i, searches=len(seeds))
    except BaseException:
        run.stop.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    rows = []
    for seed in seeds:
        blocks = read_cado_blocks(run.path / phase / (seed['search'] + '.poly'))
        for rank, (label, p) in enumerate(blocks[:args.score_top]):
            rows.append(dict(search=seed['search'], parent=seed['parent'], phase=phase,
                             rank=rank, label=label, poly=p))
    return rows


def score(run, args, rows, binary, name, final=False):
    dst = run.path / 'scores' / (name + '.json')
    pool = dst.with_suffix('.poly')
    tsv = dst.with_suffix('.tsv')
    bounds = score_bounds(args.params)
    argv = [binary, '-t', args.threads, '-K', 16000 if final else 4000,
            '-Keval', 256000 if final else 16000, '-Bf', bounds['Bf'], '-Bg', bounds['Bg'], '-area', bounds['area']]
    if args.lattice:
        argv += ['-lattice', args.lattice, '-qband', args.qband, '-nlat', 256 if final else 32,
                 '-npts', 4096 if final else 2048, '-latseed', 2 if final else 0]
    if final:
        argv += ['-useskew']
    argv.append(pool)

    def action():
        pool.write_text('\n'.join(cado_poly_text(r['poly']) for r in rows))
        run.command(f'score_{name}', argv, tsv)
        lines = tsv.read_text().splitlines()
        if len(lines) != len(rows):
            raise ValueError(f'{name}: scorer returned {len(lines)} rows for {len(rows)} polynomials')
        scored = []
        for i, (row, line) in enumerate(zip(rows, lines)):
            fields = line.split('\t')
            values = list(map(float, fields[1:]))
            if int(fields[0]) != i or not all(math.isfinite(x) for x in values) or min(values[:2]) <= 0:
                raise ValueError(f'{name}: invalid score row {i}')
            value, skew = values[:2]
            murphy = values[2] if args.lattice else value
            scored.append(dict(row, score=value, murphy=murphy, poly=dict(row['poly'], skew=skew)))
        write_json(dst, sorted(scored, key=lambda row: -row['score']))

    run.step(f'score_{name}', dict(rows=rows, argv=list(map(str, argv)), binary=digest(binary)), [pool, tsv, dst], action)
    return read_json(dst)


def leading_searches(rows, seeds, count):
    """Spend depth on distinct parent families, keeping each family's best search."""
    chosen, seen = [], set()
    lookup = {s['search']: s for s in seeds}
    for row in sorted(rows, key=lambda r: -r['score']):
        if row['parent'] not in seen:
            seen.add(row['parent'])
            chosen.append(lookup[row['search']])
            if len(chosen) == count:
                break
    return chosen if count else []


def unique_rows(rows):
    unique = {}
    for row in rows:
        unique.setdefault(cado_poly_text(row['poly']), row)
    return list(unique.values())


def pipeline(run, args, info):
    sopt = run.freeze('s23_sopt', args.sopt_binary)
    ropt = run.freeze('s23_ropt', args.ropt_binary)
    pools = run.pool_capable(ropt)
    scorer = run.freeze('cado_murphy')
    settings = {k: v for k, v in vars(args).items() if k not in
                ('command', 'input', 'initial_unsorted', 'default_input', 'resume', 'config')}
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in settings.items()}
    run.step('pipeline_settings', settings, [run.path / 'settings.json'],
             lambda: write_json(run.path / 'settings.json', settings))
    seeds = select_seeds(run, args, info, sopt)
    rows = search(run, args, seeds, ropt, 'breadth', args.maxcells, info['n'], save_pool=pools)
    ranked = score(run, args, rows, scorer, 'breadth')
    deeper = leading_searches(ranked, seeds, args.depth_seeds)
    if deeper:
        rows = search(run, args, deeper, ropt, 'depth', args.depth_maxcells, info['n'], save_pool=pools)
        ranked += score(run, args, rows, scorer, 'depth')
    finalists = unique_rows(sorted(ranked, key=lambda r: -r['score']))[:args.finalists]
    accurate = leading_searches(ranked, seeds, args.murphy_seeds)
    # Refine the full saved proxy pool from the largest completed search, before
    # its output cut. Old frozen binaries lack replay and retain the legacy path.
    deep_names = {s['search'] for s in deeper}
    for source_phase, cap, selected in (
            ('breadth', args.maxcells, [s for s in accurate if s['search'] not in deep_names]),
            ('depth', args.depth_maxcells, [s for s in accurate if s['search'] in deep_names])):
        if selected:
            rows = search(run, args, selected, ropt, 'murphy', cap, info['n'], accurate=True,
                          replay_phase=source_phase if pools else None)
            # Preserve their selected skew and translation during independent validation.
            finalists.extend(rows)
    # Keep controls from the coarse passes and remove only exact duplicate configurations.
    validated = score(run, args, unique_rows(finalists), scorer, 'final', final=True)
    results = run.path / 'results'
    outputs = [results / 'best.poly', results / 'finalists.poly', results / 'summary.json', results / 'report.txt']

    def export():
        for row in validated:
            validate_poly(row['poly'], info['n'])
        winner = validated[0]
        outputs[0].write_text(cado_poly_text(winner['poly']))
        outputs[1].write_text('\n'.join(f"# {r['phase']} {r['search']} score {r['score']:.12g}\n" +
                                       cado_poly_text(r['poly']) for r in validated))
        objective = 'lattice band model' if args.lattice else 'MurphyE'
        partial = []
        for log in sorted((run.path / 'logs/murphy').glob('*.json')):
            result = read_json(log)
            if result['returncode']:
                partial.append(dict(search=log.stem, returncode=result['returncode'], log=str(log.with_suffix('.log'))))
        summary = dict(n=info['n'], input_polynomials=info['count'], searches=len(seeds),
                       objective=objective, settings=settings, winner=winner, finalists=validated,
                       partial_murphy_searches=partial,
                       note='Model ranking; no test sieving performed. Host scoring and size model still use CPU.')
        write_json(outputs[2], summary)
        outputs[3].write_text(f"GPU polynomial optimization complete\nRun: {run.path}\n"
                              f"Inputs: {info['count']}; breadth searches: {len(seeds)}\n"
                              f"Ranking: {objective}; {args.params}\n"
                              f"Winner: {winner['phase']} {winner['search']}\n"
                              f"Score: {winner['score']:.9g}; MurphyE: {winner['murphy']:.9g}\n"
                              f"Polynomial: {outputs[0]}\nFinalists: {outputs[1]}\n"
                              "Final validation: 256,000 MurphyE angles; exact resultants checked.\n"
                              f"Partial MurphyE searches: {len(partial)} (valid outputs retained; details in summary.json).\n"
                              "Model ranking; no test sieving performed.\n")

    run.step('export', dict(finals=digest(run.path / 'scores/final.json')), outputs, export)
    run.state('complete', winner=str(outputs[0]))
    print(outputs[3].read_text(), end='')


def parser(argv=None):
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument('--config', type=Path, default=ROOT / 'nfs_config.ini')
    first.add_argument('--run-dir', type=Path)
    known, _ = first.parse_known_args(argv)
    cfg = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
    cfg.read(known.config)

    def setting(section, key, default):
        return os.path.expandvars(os.path.expanduser(cfg.get(section, key, fallback=str(default)).strip()))

    ap = argparse.ArgumentParser(description=__doc__, parents=[first], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('preprocess', 'pipeline', 'all', 'report'))
    run_dir = (ROOT / (known.run_dir or Path(setting('gpu', 'run_dir', 'stage23_gpu/runs/current')))).resolve()
    ap.set_defaults(run_dir=run_dir)
    inputs = ap.add_mutually_exclusive_group()
    inputs.add_argument('-i', '--input', type=Path, help='raw multiline polynomials; pipeline runs preprocess first')
    inputs.add_argument('--initial-unsorted', type=Path, help='import existing CPU/GPU CADO sopt raw/output pairs')
    ap.add_argument('--resume', action='store_true', help='reuse verified checkpoints (also the default)')
    ap.add_argument('-n', '--extract', type=int, default=setting('pipeline', 'extract_top_n', 1000))
    ap.add_argument('--resopt-effort', type=int, default=setting('pipeline', 'resopt_effort', 50))
    ap.add_argument('--seeds', type=int, default=setting('gpu', 'seeds', setting('pipeline', 'msieve_ropt_count', 300)))
    ap.add_argument('-t', '--threads', type=int, default=setting('system', 'threads', 8))
    ap.add_argument('--device', type=int, default=setting('gpu', 'device', 0))
    ap.add_argument('--workers', type=int, default=setting('gpu', 'workers', 2))
    ap.add_argument('--launch', type=int, default=setting('gpu', 'launch', 30))
    ap.add_argument('--slice', type=float, default=setting('gpu', 'slice', 10))
    ap.add_argument('--budget', type=float, default=setting('gpu', 'budget', 3))
    ap.add_argument('--maxcells', type=float, default=setting('gpu', 'maxcells', 2e11))
    ap.add_argument('--depth-maxcells', type=float, default=setting('gpu', 'depth_maxcells', 6e12))
    ap.add_argument('--depth-seeds', type=int, default=setting('gpu', 'depth_seeds', 16))
    ap.add_argument('--murphy-seeds', type=int, default=setting('gpu', 'murphy_seeds', 4))
    ap.add_argument('--rerank', type=int, default=setting('gpu', 'rerank', 4096))
    ap.add_argument('--keep', type=int, default=setting('gpu', 'keep', 200))
    ap.add_argument('--score-top', type=int, default=setting('gpu', 'score_top', 32))
    ap.add_argument('--finalists', type=int, default=setting('gpu', 'finalists', 32))
    ap.add_argument('--maxlines', type=int, default=setting('gpu', 'maxlines', 1000))
    ap.add_argument('--umax', type=int, default=setting('gpu', 'umax', 20000))
    ap.add_argument('--residue', default=setting('gpu', 'residue', ''),
                    help='experimental single class M,U,V in original coordinates; requires --u-scan 0')
    ap.add_argument('--classes', default=setting('gpu', 'classes', ''),
                    help='experimental automatic classes M,K; one shared cell cap and candidate pool')
    ap.add_argument('--class-explore', type=int, default=setting('gpu', 'class_explore', -1),
                    help='reserve E of K class slots for exploration; -1 selects K/4')
    ap.add_argument('--u-scan', type=int, default=setting('gpu', 'u_scan', 0),
                    help='experimental coarse u spacing across +/-umax; 0 disables it')
    ap.add_argument('--u-basins', type=int, default=setting('gpu', 'u_basins', 4),
                    help='maximum coarse basins to expand (1..16)')
    ap.add_argument('--u-radius', type=int, default=setting('gpu', 'u_radius', 128),
                    help='dense expansion radius around each refined basin (0..4096)')
    ap.add_argument('--params', default=setting('gpu', 'params', ''), help='job:lpbr,lpba,I,qmin (required for pipeline/all)')
    ap.add_argument('--lattice', default=setting('gpu', 'lattice', ''), help='LOGI,J; enables lattice ranking with --qband')
    ap.add_argument('--qband', default=setting('gpu', 'qband', ''), help='QMIN,QMAX,NQ')
    ap.add_argument('--sopt-binary', type=Path, default=ROOT / 'stage23_gpu/build/s23_sopt')
    ap.add_argument('--ropt-binary', type=Path, default=ROOT / 'stage23_gpu/build/s23_ropt')
    # A continuation inherits the run's settings; explicit options still win.
    saved = run_dir / 'settings.json'
    if saved.exists():
        options = {a.dest for a in ap._actions} - {'command', 'config', 'run_dir', 'input', 'initial_unsorted', 'resume'}
        defaults = {k: v for k, v in read_json(saved).items() if k in options}
        for key in ('sopt_binary', 'ropt_binary'):
            if key in defaults:
                defaults[key] = Path(defaults[key])
        ap.set_defaults(**defaults)
    execution = run_dir / 'execution.json'
    if execution.exists():
        ap.set_defaults(**{k: v for k, v in read_json(execution).items() if k in EXECUTION_SETTINGS})
    args = ap.parse_args(argv)
    args.default_input = Path(setting('preprocessing', 'input_file', 'msieve.dat.ms'))
    for k, v in vars(args).items():
        if isinstance(v, Path):
            setattr(args, k, (ROOT / v).resolve())
    if args.command == 'report':
        return args
    if args.threads == 0:
        args.threads = os.cpu_count() or 1
    for key in ('extract', 'seeds', 'threads', 'workers', 'rerank', 'keep', 'score_top', 'finalists', 'maxlines', 'umax'):
        if getattr(args, key) < 1:
            ap.error(f'--{key.replace("_", "-")} must be positive')
    if args.workers > 2:
        ap.error('--workers must be 1 or 2 (shared desktop GPU limit)')
    if not 0 <= args.u_scan <= 2147483647 or not 1 <= args.u_basins <= 16 or not 0 <= args.u_radius <= 4096:
        ap.error('need u-scan in [0,2147483647], u-basins in [1,16], u-radius in [0,4096]')
    if args.u_scan and (args.umax > 1000000000 or
                        2 * ((args.umax + args.u_scan - 1) // args.u_scan) + 1 > 4097 or
                        (2 * args.u_radius + 1) * args.u_basins > 16384):
        ap.error('coarse u scan exceeds limit: umax <= 1e9, <=4097 samples, <=16384 neighborhood lines')
    if args.residue:
        try:
            modulus, u, v = map(int, args.residue.split(','))
            if not (1 <= modulus <= 65536 and -2**63 <= u < 2**63 and -2**63 <= v < 2**63):
                raise ValueError()
        except ValueError:
            ap.error('--residue needs M,U,V with M in [1,65536] and signed 64-bit U/V')
        if args.u_scan or args.umax > 1000000000:
            ap.error('--residue needs --u-scan 0 and --umax <= 1e9')
        args.residue = f'{modulus},{u % modulus},{v % modulus}'
    if args.classes:
        try:
            modulus, count = map(int, args.classes.split(','))
            if not (2 <= modulus <= 4096 and 1 <= count <= min(64, modulus * modulus)):
                raise ValueError()
        except ValueError:
            ap.error('--classes needs M,K with M in [2,4096] and K in [1,min(64,M*M)]')
        if args.residue or args.u_scan or args.umax > 1000000000:
            ap.error('--classes requires an empty --residue, --u-scan 0 and --umax <= 1e9')
        if args.class_explore == -1:
            args.class_explore = count // 4
        if not 0 <= args.class_explore <= count:
            ap.error('--class-explore must be in [0,K], or -1 for K/4')
        args.classes = f'{modulus},{count}'
        # The exact proposed rows depend on each optimized seed. Bound every
        # possible pool here so a configuration cannot fail after hours of sopt.
        rows = min(modulus, 4 * count + max(16, count))
        quotient, remainder = divmod(2 * args.umax + 1, modulus)
        if rows * quotient + min(rows, remainder) > 65536:
            ap.error('--classes may exceed 65536 candidate u lines; reduce --umax or K')
    elif args.class_explore != -1:
        ap.error('--class-explore requires --classes')
    if args.device < 0 or not 20 <= args.launch <= 36 or not 0 <= args.resopt_effort <= 100:
        ap.error('need device >= 0, launch in [20,36], resopt effort in [0,100]')
    if args.depth_seeds < 0 or args.murphy_seeds < 1:
        ap.error('need depth-seeds >= 0 and murphy-seeds >= 1')
    for key in ('slice', 'budget', 'maxcells', 'depth_maxcells'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            ap.error(f'--{key.replace("_", "-")} must be finite and positive')
    if args.budget < .5 or args.score_top > args.keep or args.rerank < max(args.keep, 3):
        ap.error('need budget >= 0.5, score-top <= keep, rerank >= max(keep,3)')
    if args.depth_seeds and args.depth_maxcells <= args.maxcells:
        ap.error('--depth-maxcells must exceed --maxcells when depth is enabled')
    if args.command in ('pipeline', 'all'):
        try:
            if not args.params.startswith('job:') or score_bounds(args.params) is None:
                raise ValueError('supply --params job:lpbr,lpba,I,qmin or [gpu] params in nfs_config.ini')
            if bool(args.lattice) != bool(args.qband):
                raise ValueError('--lattice and --qband must be supplied together')
            if args.lattice:
                logi, j = map(int, args.lattice.split(','))
                lo, hi, points = args.qband.split(',')
                lo, hi, points = float(lo), float(hi), int(points)
                if not (2 <= logi <= 30 and j >= 1 and 3 <= lo < hi <= 2**52 and 2 <= points <= 10000
                        and (hi - lo) / (points - 1) >= 1):
                    raise ValueError('invalid lattice or q band')
        except (ValueError, OverflowError) as error:
            ap.error(str(error))
    return args


def main(argv=None):
    args = parser(argv)
    if args.command == 'report':
        report = args.run_dir / 'results/report.txt'
        if not report.exists():
            raise ValueError(f'no completed report at {report}')
        print(report.read_text(), end='')
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'another pipeline is using {args.run_dir}') from None
        run = Run(args.run_dir)
        write_json(run.path / 'execution.json', {k: getattr(args, k) for k in EXECUTION_SETTINGS})
        try:
            run.state('starting', command=args.command)
            if args.command in ('preprocess', 'all') or args.input or args.initial_unsorted:
                info = preprocess(run, args)
            else:
                info = existing_preprocess(run)
            if args.command == 'preprocess':
                run.state('preprocess complete', polynomials=info['count'], n=info['n'])
            else:
                pipeline(run, args, info)
        except BaseException as error:
            run.stop.set()
            run.state('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
            raise


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyboardInterrupt) as error:
        print(f'GPU pipeline: {error or "interrupted"}', file=sys.stderr)
        sys.exit(130 if isinstance(error, KeyboardInterrupt) else 1)
