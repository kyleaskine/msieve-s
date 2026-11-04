#!/usr/bin/env python3
"""Test-sieve two or more polynomials over matched q windows with the user's GPU siever.

  pair_testsieve.py RUN --poly NAME=FILE --poly NAME=FILE --lim R,A --lpb R,A --mfb R,A
                    --logI L --J J --band QMIN,QMAX [--side a|r] [--points 9] [--width 20000]
                    [--params job:lpbr,lpba,I,qmin] [--siever DIR] [--gpu-idle 30] [--gpu-wait 1800]
  pair_testsieve.py RUN --resume [--params P]   continue a stopped run (settings come from RUN)
  pair_testsieve.py RUN --report [--params P]   recompute results.json from the finished windows
  pair_testsieve.py RUN --save DIR              copy results, manifest, jobs and sources to DIR

Keep RUN under the ignored stage23_gpu/runs/, not stage23_gpu/build/ (make clean deletes
that): it holds the relation files whose hashes the results record, and the logs.

Inputs are checked before RUN is created. Each polynomial is the first block of its FILE
and needs n: and skew: (the siever reduces every special-q lattice at that skew). Its
resultant must be divisible by n, and all of them must factor the same n. The siever
directory must hold bench, fbgen_gpu and normscan, --params must parse (and cado_murphy
must build), and every window must hold at least MIN_PRIMES special-q primes with roots
for every polynomial; those root counts are computed here and frozen in RUN/roots.json.
The binaries, the jobs and copies of the source files are frozen into RUN. To sieve a later
case with the same engine as an earlier one, pass --siever EARLIER_RUN/bin.

Each factor base is built once. Then every polynomial sieves the same windows [q0, q0+W),
the order rotating at each window so drift in clocks or host load does not favour one of
them. Before each window the GPU has to be idle: median utilization of three one-second
samples at most --gpu-idle percent (negative disables the check), waiting up to
--gpu-wait seconds; a window that starts busy anyway is flagged. nvidia-smi under WSL2
lists no processes, so utilization (and memory in use, recorded before and after) is the
only signal. Every window is sieved from scratch (bench --restart) and counts only once its
relation file is final: bench also exits 0 after a clean signal stop, leaving NAME.part. A
stopped window is sieved again by --resume. Every relation file is checked with bench
--check-relations. The first polynomial is the reference for ratios.

Yield is unique relations per (q, rho) pair. Each relation is assigned to its special-q
(the sieve-side factor inside the window), so the 95% intervals come from the spread over
special-q primes (a ratio estimator clustered by prime), not from a Poisson assumption.
The frozen root counts must sum to the siever's own pair count in every window. The band
projection integrates yield x 1/ln q (on average one pair per prime) with the trapezoid
rule over the window midpoints, flat out to the band edges; ratios use the same weights for
every polynomial. The projected relations do not remove duplicates between special-q of a
full run. Timing is the siever's complete wall clock per pair from one run per window:
throughput ratios are unreplicated and carry host and GPU noise that the yield intervals
do not include.

With --params, the lattice model (score_polys --lattice at the file's skew and the sieved
special-q side) is evaluated at the window midpoints for comparison, and the scorer's hash
is recorded. Its absolute relations read about 1.9x a test sieve's, so compare ratios.
--params given to --report or --resume replaces the frozen value for that report.

--save writes results.json (every parsed count, interval, timing and relation-file hash,
plus raw_artifacts naming RUN), manifest.json, the jobs and the source polynomials to a
new DIR: six files for a pair. As stage23_bench/data/README.md asks, the logs and command
records stay in RUN.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polyfmt import cado_poly_text, coeffs, read_cado_blocks, resultant

BINARIES = ('bench', 'fbgen_gpu', 'normscan')
MIN_PRIMES = 30
Z95 = 1.959963984540054


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def is_prime(n):
    """Deterministic Miller-Rabin for n < 3.3e24."""
    if n < 2:
        return False
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in bases:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _trim(a):
    while a and a[-1] == 0:
        a.pop()
    return a


def _polymod(a, m, q):
    """a mod m over GF(q), m monic; coefficient lists, lowest degree first."""
    a = _trim([x % q for x in a])
    d = len(m) - 1
    while len(a) - 1 >= d:
        lead, shift = a[-1], len(a) - 1 - d
        for i, c in enumerate(m):
            a[shift + i] = (a[shift + i] - lead * c) % q
        _trim(a)
    return a


def _polymulmod(a, b, m, q):
    out = [0] * (len(a) + len(b) - 1) if a and b else []
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return _polymod(out, m, q)


def _monic(a, q):
    inv = pow(a[-1], -1, q)
    return [x * inv % q for x in a]


def distinct_roots(c, q):
    """Number of distinct affine roots of f = sum c_i x^i modulo the prime q."""
    f = _trim([x % q for x in c])
    if len(f) < 2:
        raise ValueError(f'f vanishes or is constant modulo {q}')
    f = _monic(f, q)
    result, base, e = [1], [0, 1], q
    while e:
        if e & 1:
            result = _polymulmod(result, base, f, q)
        base = _polymulmod(base, base, f, q)
        e >>= 1
    g = result + [0] * max(0, 2 - len(result))
    g[1] = (g[1] - 1) % q
    a, b = f, _trim(g)
    while b:
        a, b = b, _polymod(a, _monic(b, q), q)
    return len(a) - 1


def special_q_roots(poly, side, q0, width):
    """{q: pairs} for every prime q in [q0, q0 + width) on the special-q side."""
    out = {}
    for q in range(q0 | 1 if q0 > 2 else 2, q0 + width, 1 if q0 <= 2 else 2):
        if not is_prime(q):
            continue
        if side == 1:
            c = coeffs(poly)
            if c[-1] % q == 0:
                raise ValueError(f'projective special-q root at {q}; not modelled')
            out[q] = distinct_roots(c, q)
        else:
            out[q] = 0 if poly['Y1'] % q == 0 else 1
    return out


def relation_clusters(dat, side, q0, width):
    """(raw, {q: unique relations}, ambiguous): assign each unique (a, b) to its special-q."""
    seen, per_q, raw, ambiguous = set(), {}, 0, 0
    for line in dat.read_text().splitlines():
        if not line.strip():
            continue
        raw += 1
        ab, rational, algebraic = line.split(':')
        if ab in seen:
            continue
        seen.add(ab)
        field = algebraic if side == 1 else rational
        hits = sorted({v for v in (int(x, 16) for x in field.split(',') if x) if q0 <= v < q0 + width})
        if not hits:
            raise ValueError(f'{dat}: relation {ab} has no special-q in [{q0}, {q0 + width})')
        ambiguous += len(hits) > 1
        per_q[hits[0]] = per_q.get(hits[0], 0) + 1
    return raw, per_q, ambiguous


def ratio_estimate(roots, per_q):
    """(relations per pair, standard error) clustered by special-q prime."""
    units = [(per_q.get(q, 0), m) for q, m in roots.items() if m]
    extra = set(per_q) - {q for q, m in roots.items() if m}
    if extra:
        raise ValueError(f'relations at special-q without roots: {sorted(extra)[:5]}')
    k = len(units)
    if k < 2:
        raise ValueError(f'{k} special-q prime(s) with roots; an interval needs at least 2')
    y, m = sum(u[0] for u in units), sum(u[1] for u in units)
    r = y / m
    var = sum((yi - r * mi) ** 2 for yi, mi in units) / m ** 2 * k / (k - 1)
    return r, math.sqrt(var)


def settings_from(args):
    def pair(text, kind=int):
        try:
            a, b = (kind(float(x)) for x in text.split(','))
        except ValueError:
            raise SystemExit(f'expected two comma-separated numbers, got {text!r}') from None
        return a, b
    rlim, alim = pair(args.lim)
    lpbr, lpba = pair(args.lpb)
    mfbr, mfba = pair(args.mfb)
    qmin, qmax = pair(args.band)
    if args.points < 2 or args.width < 1 or qmax - qmin < args.width * args.points:
        raise SystemExit('need at least two windows that fit inside the band without overlapping')
    q0s = [qmin + round(i * (qmax - qmin - args.width) / (args.points - 1)) for i in range(args.points)]
    polys = {}
    for item in args.poly:
        name, _, path = item.partition('=')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or not path or name in polys:
            raise SystemExit(f'--poly NAME=FILE with a distinct simple NAME, got {item!r}')
        polys[name] = str(Path(path).resolve())
    if len(polys) < 2:
        raise SystemExit('give at least two --poly NAME=FILE')
    return dict(polys=polys, rlim=rlim, alim=alim, lpbr=lpbr, lpba=lpba, mfbr=mfbr, mfba=mfba,
                logI=args.logI, J=args.J, sq_side=1 if args.side == 'a' else 0,
                band=[qmin, qmax], points=args.points, width=args.width, q0s=q0s,
                params=args.params, siever=str(Path(args.siever).expanduser().resolve()),
                gpu_idle=args.gpu_idle, gpu_wait=args.gpu_wait)


def load_polys(paths):
    """{name: poly} for the first block of each file, checked as a GNFS pair for one n."""
    polys = {}
    for name, path in paths.items():
        try:
            blocks = read_cado_blocks(path)
        except OSError as error:
            raise SystemExit(f'{name}: cannot read {path}: {error}') from None
        p = blocks[0][1] if blocks else {}
        if not all(k in p for k in ('n', 'Y0', 'Y1', 'skew')):
            raise SystemExit(f'{path}: the first block needs n:, Y0:, Y1:, c0: ... and skew:')
        if resultant(p) % p['n']:
            raise SystemExit(f'{path}: the resultant is not a multiple of n')
        polys[name] = p
    if len({p['n'] for p in polys.values()}) != 1:
        raise SystemExit('the polynomials factor different numbers')
    return polys


def check_params(params):
    """Fail fast on a --params value the lattice model would reject, and build its scorer."""
    if not params:
        return
    import score_polys
    try:
        score_polys.score_bounds(params)
    except ValueError as error:
        raise SystemExit(f'--params: {error}') from None
    score_polys.cado_murphy_binary()


def window_roots(polys, settings):
    """{name: {q0: {q: pairs}}} over the primes with roots; each window needs MIN_PRIMES."""
    out = {}
    for name, p in polys.items():
        out[name] = {}
        for q0 in settings['q0s']:
            try:
                found = special_q_roots(p, settings['sq_side'], q0, settings['width'])
            except ValueError as error:
                raise SystemExit(f'{name}, window {q0}: {error}') from None
            roots = {q: m for q, m in found.items() if m}
            if len(roots) < MIN_PRIMES:
                raise SystemExit(f"{name}: window [{q0}, {q0 + settings['width']}) holds {len(roots)} "
                                 f'special-q primes with roots; widen --width (need {MIN_PRIMES})')
            out[name][q0] = roots
    return out


def load_roots(run, settings, polys):
    path = run / 'roots.json'
    if not path.exists():           # runs prepared before roots were frozen
        return window_roots(polys, settings)
    data = json.loads(path.read_text())
    return {name: {int(q0): {q: m for q, m in pairs} for q0, pairs in windows.items()}
            for name, windows in data.items()}


def prepare(run, settings):
    polys = load_polys(settings['polys'])
    source = Path(settings['siever'])
    missing = [name for name in BINARIES if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"{source}: missing {', '.join(missing)}")
    check_params(settings['params'])
    roots = window_roots(polys, settings)
    run.mkdir(parents=True, exist_ok=False)
    for folder in ('bin', 'jobs', 'fb', 'raw', 'sources'):
        (run / folder).mkdir()
    for name in BINARIES:
        shutil.copy2(source / name, run / 'bin' / name)
    for name, path in settings['polys'].items():
        shutil.copy2(path, run / f'sources/{name}.poly')
        job = cado_poly_text(polys[name]) + ''.join(f'{k}: {settings[k]}\n' for k in
                                                    ('rlim', 'alim', 'lpbr', 'lpba', 'mfbr', 'mfba'))
        (run / f'jobs/{name}.job').write_text(job)
    write_json(run / 'roots.json', {name: {str(q0): sorted(r.items()) for q0, r in windows.items()}
                                    for name, windows in roots.items()})
    frozen = [p for d in ('jobs', 'bin', 'sources') for p in sorted((run / d).iterdir())] + [run / 'roots.json']
    manifest = dict(settings=settings, created=time.time(),
                    poly_sources={name: sha(run / f'sources/{name}.poly') for name in settings['polys']},
                    sha256={str(p.relative_to(run)): sha(p) for p in frozen})
    write_json(run / 'manifest.json', manifest)


def check_frozen(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    for path, digest in manifest['sha256'].items():
        if sha(run / path) != digest:
            raise SystemExit(f'frozen input changed: {path}')
    return manifest['settings']


def gpu_utilization():
    """Utilization of the first GPU in percent, or None when nvidia-smi cannot say."""
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30)
        return float(r.stdout.split()[0]) if r.returncode == 0 and r.stdout.split() else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def wait_for_idle_gpu(limit, wait, poll=30.0):
    """Wait until the median of three 1 s utilization samples is at most limit percent, for up
    to wait seconds. busy is None when nvidia-smi gives no reading."""
    if limit is None or limit < 0:
        return dict(disabled=True)
    start = time.time()
    while True:
        samples = []
        for i in range(3):
            if i:
                time.sleep(1)
            samples.append(gpu_utilization())
        known = [s for s in samples if s is not None]
        busy = statistics.median(known) > limit if known else None
        if not busy or time.time() - start >= wait:
            return dict(samples=samples, busy=busy, limit=limit, waited_seconds=round(time.time() - start, 1))
        print(f'GPU busy {samples}; waiting', flush=True)
        time.sleep(poll)


def gpu_snapshot():
    try:
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                               '--format=csv,noheader'], capture_output=True, text=True, timeout=30)
        util = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                               '--format=csv,noheader'], capture_output=True, text=True, timeout=30)
        return dict(apps=[x.strip() for x in apps.stdout.splitlines() if x.strip()],
                    gpu=util.stdout.strip(), returncodes=[apps.returncode, util.returncode])
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(error=repr(error))


def set_state(run, stage, **extra):
    write_json(run / 'status.json', dict(stage=stage, updated=time.time(), **extra))
    print(stage, extra, flush=True)


def command(run, tag, argv, resume, outputs=()):
    """Run argv with its output in RUN/TAG.out. It is done only when it exits 0 AND every file
    in outputs exists (removed first, so none is stale)."""
    record = run / f'{tag}.command.json'
    argv = [str(x) for x in argv]
    outputs = [Path(p) for p in outputs]
    if resume and record.exists():
        previous = json.loads(record.read_text())
        if (previous['returncode'] == 0 and previous.get('complete', True) and previous['argv'] == argv
                and (run / f'{tag}.out').exists() and all(p.exists() for p in outputs)):
            return
    for p in outputs:
        p.unlink(missing_ok=True)
    start = time.time()
    with (run / f'{tag}.out').open('w') as log:
        result = subprocess.run(argv, cwd=run, stdout=log, stderr=subprocess.STDOUT)
    missing = [p.name for p in outputs if not p.exists()]
    write_json(record, dict(argv=argv, returncode=result.returncode, complete=not result.returncode and not missing,
                            elapsed_seconds=time.time() - start))
    if result.returncode:
        raise RuntimeError(f'{tag} failed ({result.returncode}); see {tag}.out')
    if missing:
        raise RuntimeError(f"{tag} stopped before finishing (no {', '.join(missing)}); --resume sieves it again")


def sieve(run, settings, resume, params=None):
    names = list(settings['polys'])
    logI, J, side, width = settings['logI'], settings['J'], settings['sq_side'], settings['width']
    polys = {name: read_cado_blocks(run / f'jobs/{name}.job')[0][1] for name in names}
    roots = load_roots(run, settings, polys)
    rows_path = run / 'rows.json'
    rows = json.loads(rows_path.read_text()) if resume and rows_path.exists() else []
    done = {(r['poly'], r['q0']) for r in rows}
    try:
        for name in names:
            set_state(run, 'preparing factor base', poly=name)
            command(run, f'fb/{name}_normscan', [run / 'bin/normscan', '--poly', run / f'jobs/{name}.job',
                    '--sq-side', side, '--logI', logI, '--J', J, '--qmin', settings['band'][0],
                    '--qmax', settings['band'][1], '--samples', 400000], resume)
            fb = run / f'fb/{name}.m{logI}'
            command(run, f'fb/{name}_fbgen', [run / 'bin/fbgen_gpu', '--poly', run / f'jobs/{name}.job',
                    '--lim', settings['alim'], '--maxbits', logI, '--out', fb], resume, outputs=[fb])
        total = len(names) * len(settings['q0s'])
        for k, q0 in enumerate(settings['q0s']):
            order = names[k % len(names):] + names[:k % len(names)]
            for name in order:
                if (name, q0) in done:
                    continue
                tag = f'raw/{name}_{q0}'
                set_state(run, 'sieving', poly=name, q0=q0, completed=len(rows), total=total)
                gate = wait_for_idle_gpu(settings.get('gpu_idle', 30), settings.get('gpu_wait', 1800))
                if gate.get('busy'):
                    print(f'warning: GPU still busy after {gate["waited_seconds"]} s; sieving {tag} anyway', flush=True)
                before = gpu_snapshot()
                dat = run / f'{tag}.dat'
                command(run, tag, [run / 'bin/bench', '--pipeline', '--cofactor', '--poly', run / f'jobs/{name}.job',
                        '--fb1', run / f'fb/{name}.m{logI}', '--sq-side', side, '--logI', logI, '--J', J,
                        '--qrange', f'{q0}:{q0 + width - 1}', '--restart', '--relations', dat,
                        '--log', run / f'{tag}.log', '--log-every', 1], resume, outputs=[dat])
                after = gpu_snapshot()
                text = (run / f'{tag}.out').read_text()
                band_line = re.search(r'--- band of (\d+)', text)
                rel_lines = re.findall(r'total relations\s+(\d+)', text)
                ms_lines = re.findall(r'wall clock per q, COMPLETE\s+([\d.]+)', text)
                if not (band_line and rel_lines and ms_lines):
                    raise RuntimeError(f'{tag}: band summary incomplete in {tag}.out')
                pairs, rels, ms = int(band_line[1]), int(rel_lines[-1]), float(ms_lines[-1])
                if len(re.findall(r'factors x cofactor == norm.*PASS', text)) != 2:
                    raise RuntimeError(f'{tag}: missing norm verification')
                command(run, tag + '_verify', [run / 'bin/bench', '--poly', run / f'jobs/{name}.job',
                        '--check-relations', dat], resume)
                raw, per_q, ambiguous = relation_clusters(dat, side, q0, width)
                window = roots[name][q0]
                if raw != rels or pairs <= 0 or ms <= 0:
                    raise RuntimeError(f'{tag}: {raw} relation lines, siever reported {rels}; {pairs} pairs, {ms} ms')
                if sum(window.values()) != pairs:
                    raise RuntimeError(f'{tag}: {sum(window.values())} special-q roots here, siever sieved {pairs}')
                rate, se = ratio_estimate(window, per_q)
                watts = [float(x) for x in re.findall(r'board=([\d.]+)W', (run / f'{tag}.log').read_text())]
                unique = sum(per_q.values())
                rows.append(dict(poly=name, q0=q0, width=width, order=order.index(name), pairs=pairs,
                                 primes=len(window), relations=rels,
                                 unique_relations=unique, duplicates=rels - unique, ambiguous_special_q=ambiguous,
                                 rel_per_pair=rate, se_rel_per_pair=se, ms_per_pair=ms,
                                 rel_per_second=rate * 1000 / ms,
                                 board_watts=sum(watts) / len(watts) if watts else None,
                                 gpu_gate=gate, gpu_before=before, gpu_after=after, verified=True,
                                 relation_sha256=sha(dat)))
                write_json(rows_path, rows)
                print(f"{name} q0={q0}: {unique} rels / {pairs} pairs = {rate:.3f} +- {Z95 * se:.3f}, "
                      f"{ms:.1f} ms/pair", flush=True)
        report(run, settings, rows, params)
    except BaseException as error:
        set_state(run, 'failed', error=repr(error))
        raise


def band_weights(settings):
    """Trapezoid weights over window midpoints, flat to the band edges, times 1/ln q."""
    qmin, qmax = settings['band']
    xs = [q0 + settings['width'] / 2 for q0 in settings['q0s']]
    w = []
    for i, x in enumerate(xs):
        left = (x - xs[i - 1]) / 2 if i else x - qmin
        right = (xs[i + 1] - x) / 2 if i + 1 < len(xs) else qmax - x
        w.append((left + right) / math.log(x))
    return xs, w


def model_scores(run, settings, xs, params) -> tuple[dict[str, dict], str]:
    """({name: model scores at the window midpoints}, sha256 of the scorer binary)."""
    import score_polys
    names = list(settings['polys'])
    polys = [read_cado_blocks(run / f'jobs/{name}.job')[0][1] for name in names]
    args = SimpleNamespace(params=params, threads=8, lattice=f"{settings['logI']},{settings['J']}",
                           qband=f'{xs[0]!r},{xs[-1]!r},{len(xs)}', useskew=True, nlat=256, npts=4096,
                           latseed=2, sqside=settings['sq_side'])
    res = score_polys.lattice_scores(polys, polys[0]['n'], args)
    model: dict[str, dict] = {name: dict(skew=r[1], murphy_e=r[2], relations_per_q=r[3])
                              for name, r in zip(names, res)}
    return model, sha(score_polys.cado_murphy_binary())


def report(run, settings, rows, params=None):
    params = settings.get('params') if params is None else params
    names = list(settings['polys'])
    xs, w = band_weights(settings)
    by = {}
    for name in names:
        got = {r['q0']: r for r in rows if r['poly'] == name}
        missing = [q0 for q0 in settings['q0s'] if q0 not in got]
        if missing:
            raise RuntimeError(f'{name}: windows not sieved: {missing}')
        by[name] = [got[q0] for q0 in settings['q0s']]
    band = {}
    for name in names:
        rs = by[name]
        relations = sum(wi * r['rel_per_pair'] for wi, r in zip(w, rs))
        se = math.sqrt(sum((wi * r['se_rel_per_pair']) ** 2 for wi, r in zip(w, rs)))
        seconds = sum(wi * r['ms_per_pair'] / 1000 for wi, r in zip(w, rs))
        band[name] = dict(relations=relations, se_relations=se, gpu_seconds=seconds,
                          gpu_days=seconds / 86400, relations_per_second=relations / seconds)

    def ratio(a, sa, b, sb):
        r = a / b
        s = math.sqrt((sa / a) ** 2 + (sb / b) ** 2)
        return dict(ratio=r, low=r * math.exp(-Z95 * s), high=r * math.exp(Z95 * s))

    ref = names[0]
    ratios, per_window, throughput = {}, {}, {}
    for name in names[1:]:
        key = f'{name}/{ref}'
        ratios[key] = ratio(band[name]['relations'], band[name]['se_relations'],
                            band[ref]['relations'], band[ref]['se_relations'])
        per_window[key] = [dict(q0=a['q0'], **ratio(a['rel_per_pair'], a['se_rel_per_pair'],
                                                   b['rel_per_pair'], b['se_rel_per_pair']),
                                time=a['ms_per_pair'] / b['ms_per_pair'])
                           for a, b in zip(by[name], by[ref])]
        throughput[key] = band[name]['relations_per_second'] / band[ref]['relations_per_second']
    result = dict(settings=settings, midpoints=xs, weights=w, band=band, ratios=ratios,
                  throughput_ratios=throughput, per_window=per_window, rows=rows,
                  verified_relations=sum(r['relations'] for r in rows),
                  duplicates_removed=sum(r['duplicates'] for r in rows),
                  busy_windows=sum(1 for r in rows if r.get('gpu_gate', {}).get('busy')),
                  engine_commits=sorted({m[1] for p in (run / 'raw').glob('*.log')
                                         if (m := re.search(r'^# commit\s+(.*)$', p.read_text(), re.M))}),
                  caveat='Yield intervals are 95%, clustered by special-q prime within each window and '
                         'independent between windows. Timing is one unreplicated run per window. Projected '
                         'relations do not remove duplicates between special-q of a full run.')
    if params:
        model, scorer = model_scores(run, settings, xs, params)
        model_ratios: dict[str, dict] = {}
        for name in names[1:]:
            mine, theirs = model[name]['relations_per_q'], model[ref]['relations_per_q']
            model_ratios[f'{name}/{ref}'] = dict(
                band=sum(wi * v for wi, v in zip(w, mine)) / sum(wi * v for wi, v in zip(w, theirs)),
                per_window=[a / b for a, b in zip(mine, theirs)])
        result.update(model=model, model_ratios=model_ratios, model_params=params, model_scorer_sha256=scorer)
    write_json(run / 'results.json', result)
    print_summary(result)
    set_state(run, 'complete', verified_relations=result['verified_relations'],
              band_ratio=ratios, throughput_ratio=throughput)
    return result


def print_summary(result):
    for name in result['settings']['polys']:
        b = result['band'][name]
        print(f"{name}: {b['relations']:.4g} +- {Z95 * b['se_relations']:.2g} relations over the band, "
              f"{b['gpu_days']:.2f} GPU days")
    for key, r in result['ratios'].items():
        model = result.get('model_ratios', {}).get(key)
        print(f"{key} band yield {r['ratio']:.4f} [{r['low']:.4f}, {r['high']:.4f}]"
              + (f", model {model['band']:.4f}" if model else '')
              + f"; throughput {result['throughput_ratios'][key]:.4f} (unreplicated)")
        for i, wdw in enumerate(result['per_window'][key]):
            m = f"  model {model['per_window'][i]:.4f}" if model else ''
            print(f"  q0 {wdw['q0']:>11}  yield {wdw['ratio']:.4f} [{wdw['low']:.4f}, {wdw['high']:.4f}]"
                  f"  time {wdw['time']:.4f}{m}")
    if result.get('busy_windows'):
        print(f"warning: {result['busy_windows']} window(s) started with the GPU busy; their timing is suspect")


def save(run, dest):
    status = json.loads((run / 'status.json').read_text()) if (run / 'status.json').exists() else {}
    if status.get('stage') != 'complete' or not (run / 'results.json').exists():
        raise SystemExit('the test sieve has no complete report; run --resume or --report first')
    dest.mkdir(parents=True, exist_ok=False)
    for folder in ('jobs', 'sources'):
        if (run / folder).is_dir():
            (dest / folder).mkdir()
            for p in sorted((run / folder).iterdir()):
                shutil.copy2(p, dest / folder / p.name)
    shutil.copy2(run / 'manifest.json', dest / 'manifest.json')
    result = json.loads((run / 'results.json').read_text())
    result['raw_artifacts'] = str(run)
    write_json(dest / 'results.json', result)
    print(dest)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=Path)
    ap.add_argument('--poly', action='append', default=[], metavar='NAME=FILE')
    ap.add_argument('--lim', help='rlim,alim')
    ap.add_argument('--lpb', help='lpbr,lpba')
    ap.add_argument('--mfb', help='mfbr,mfba')
    ap.add_argument('--logI', type=int)
    ap.add_argument('--J', type=int)
    ap.add_argument('--side', choices=('a', 'r'), default='a')
    ap.add_argument('--band', help='QMIN,QMAX (absolute)')
    ap.add_argument('--points', type=int, default=9)
    ap.add_argument('--width', type=int, default=20000)
    ap.add_argument('--params', help='job:lpbr,lpba,I,qmin for the lattice model comparison')
    ap.add_argument('--siever', default='~/code/cuda-sieve/bench')
    ap.add_argument('--gpu-idle', type=float, default=30,
                    help='largest GPU utilization (%%) to start a window at; negative disables the check')
    ap.add_argument('--gpu-wait', type=float, default=1800, help='seconds to wait for an idle GPU')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--report', action='store_true')
    mode.add_argument('--save', type=Path, metavar='DIR')
    args = ap.parse_args(argv)
    run = args.run.resolve()
    if args.resume or args.report or args.save:
        if any((args.poly, args.lim, args.lpb, args.mfb, args.logI, args.J, args.band)):
            ap.error('--resume, --report and --save take their settings from RUN')
        if args.save and args.params:
            ap.error('--params goes with a new run, --resume or --report')
        settings = check_frozen(run)
        if args.save:
            save(run, args.save.resolve())
            return
        check_params(args.params)
        if args.report:
            report(run, settings, json.loads((run / 'rows.json').read_text()), args.params)
        else:
            sieve(run, settings, resume=True, params=args.params)
        return
    if not all((args.poly, args.lim, args.lpb, args.mfb, args.logI, args.J, args.band)):
        ap.error('a new run needs --poly (twice or more), --lim, --lpb, --mfb, --logI, --J and --band')
    settings = settings_from(args)
    prepare(run, settings)
    sieve(run, check_frozen(run), resume=False)


if __name__ == '__main__':
    main()
