#!/usr/bin/env python3
"""Regression checks for the bench tools; run by stage23_gpu's `make test`.

  regress.py

Each check pins a property or a failure seen before (GPU_STAGE23_PLAN.md, "Review
checks" and "Fixes"):

  scorer      cado_murphy equals CADO's MurphyE() at K = 1000 and K = 4000 (-selftest)
              on the c168 winners, and its K = 1000 values stay where they were.
  quadrature  the c168 trap: the translation-tuned winner reads +0.8% over msieve's at
              CADO's 1,000 sample points and is no better at 4,000/16,000.
  content     content_seeds.py: GPT's counterexample (Y1 = 2 is not invertible mod 2,
              but (f + g)/2 is a valid seed), and five c168 seeds whose derived seeds,
              written by the original script, must come out exactly; for each derived
              seed Res(f', g) * d = Res(f, g).
  lattice     cado_murphy -lattice against the c208 test sieves (data/c208/testsieve: five
              polys, three q, two region shapes, measured with the user's GPU siever):
              each poly's yield relative to poly A (24 ratios: four polys x three q x two
              shapes) within 0.05 of the measured ratio (worst 0.038 on 2026-10-06), and
              one value pinned (the model is deterministic); a missing fixture fails the
              check rather than crashing.
  refinement  C181 translation pairs, exact exports with retained controls, the seed-6
              gain at 64k and an independent 256k points; fixed-skew evaluation,
              rejection of K=1000, lattice validation/export agreement and independent
              lattice sample streams.

Exits 1 if any check fails.
"""

import os
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from content_seeds import content_seeds
from polyfmt import cado_murphy_binary, cado_poly_text, parse_cado_poly, resultant
from rescore import derived
from refine_polys import parser as refine_parser, refine, translated_pair, write_outputs
from score_polys import murphy as score_murphy, lattice_scores
from polyfmt import read_cado_blocks, rotation_between

HERE = os.path.dirname(os.path.abspath(__file__))
C168 = os.path.join(HERE, '..', 'data', 'c168')
C208 = os.path.join(HERE, '..', 'data', 'c208')
C181 = os.path.join(HERE, '..', 'data', 'c181', 'translation')
JOB = ['-Bf', '2147483648', '-Bg', '1073741824', '-area', '3355443200000000']  # lpb 31/30, I 14, qmin 25M
failures = []


def check(name, ok, detail):
    print(f'{"ok  " if ok else "FAIL"} {name}: {detail}')
    if not ok:
        failures.append(name)


def run_check(name, function, fixtures=()):
    missing = [f for f in fixtures if not os.path.isfile(f)]
    if missing:
        check(name, False, 'fixture missing: ' + ', '.join(missing))
        return
    try:
        function()
    except Exception as e:
        # A test error is a failed check, not a reason to skip all later checks.
        detail = e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) and e.stderr else str(e)
        check(name, False, f'{type(e).__name__}: {detail}')


def murphy(paths, *opts):
    with tempfile.NamedTemporaryFile('w', suffix='.poly', delete=False) as fh:
        for p in paths:
            fh.write(open(p).read().strip() + '\n\n')
        tmp = fh.name
    try:
        out = subprocess.run([cado_murphy_binary(), *JOB, '-t', '2', *opts, tmp], capture_output=True, text=True)
    finally:
        os.remove(tmp)
    return out


def lattice_check(ts):
    """each poly's yield relative to poly A (24 ratios: four polys x three q x two shapes)
    within 0.05 of the test sieve's, and one model value pinned"""
    meas = {}
    for l in open(os.path.join(ts, 'measured.tsv')):
        if not l.startswith('#'):
            name, geo, q0, pairs, nyield, relpair = l.split()
            meas.setdefault(geo, {}).setdefault(name, {})[int(q0)] = float(nyield)
    names = [l[2:].strip() for l in open(os.path.join(ts, 'five.poly')) if l.startswith('# ')]
    d = derived(33, 34, 16.5, 80e6)  # the c208 job: lpb 33/34, area 2^32 x 80M
    bounds = ['-Bf', repr(d['Bf']), '-Bg', repr(d['Bg']), '-area', repr(d['area'])]
    worst, n, pin, errors = 0.0, 0, None, []
    for geo in sorted(meas):
        qs = sorted(next(iter(meas[geo].values())))
        r = subprocess.run([cado_murphy_binary(), *bounds, '-K', '4000', '-Keval', '16000', '-t', '2', '-lattice',
                            geo, '-qpoints', ','.join(str(q + 1000) for q in qs), '-useskew',
                            os.path.join(ts, 'five.poly')], capture_output=True, text=True)
        if r.returncode or r.stderr:
            errors.append(f'{geo}: status {r.returncode}: {r.stderr.strip()}')
        model = {names[int(l.split()[0])]: [float(x) for x in l.split()[4:]] for l in r.stdout.split('\n') if l}
        ref = names[0]
        for name in names[1:]:
            for k, q in enumerate(qs):
                if name in model and ref in model:
                    worst = max(worst, abs(model[name][k] / model[ref][k] - meas[geo][name][q] / meas[geo][ref][q]))
                    n += 1
        if geo == '17,32768' and ref in model:
            pin = model[ref][0]
    check('lattice vs c208 test sieves', not errors and n == 24 and worst <= 0.05,
          f'{n} ratios, worst |model - measured| {worst:.3f} (limit 0.05)' + (f'; {"; ".join(errors)}' if errors else ''))
    check('lattice value pinned', pin is not None and abs(pin / 164.4257 - 1) < 1e-5,
          f'poly A, 17,32768, q 80M: {pin} relations per special-q (pinned 164.4257)')


def refinement_check():
    """C181 seed 6: close the real translation gap; recheck at 256k angles without
    changing the exported skew. Also check the exact fixture pairing and export CLI."""
    gpu = read_cado_blocks(os.path.join(C181, 'gpu.poly'))
    refs = read_cado_blocks(os.path.join(C181, 'msieve.poly'))
    pairs = [rotation_between(p, r) for (_, p), (_, r) in zip(gpu, refs)]
    check('refinement: C181 pairs are translations', len(pairs) == 5 and
          all(p and not any(p['rotation']) for p in pairs), f'{len(pairs)} cells')
    p, ref = gpu[2][1], refs[2][1]  # seed 6, the largest measured translation gap
    args = refine_parser().parse_args(['unused', '--out', 'unused', '--params', 'job:31,32,15,45e6'])
    with tempfile.TemporaryDirectory() as tmp:
        src, prefix = os.path.join(tmp, 'input.poly'), os.path.join(tmp, 'result')
        original = cado_poly_text(p)
        with open(src, 'w') as fh:
            fh.write(original)
        process = subprocess.run([sys.executable, os.path.join(HERE, 'refine_polys.py'), src, '--out', prefix,
                                  '--params', args.params], capture_output=True, text=True)
        if process.returncode:
            check('refinement: export CLI', False, f'status {process.returncode}: {process.stderr.strip()}')
            return
        with open(prefix + '.json') as fh:
            r = json.load(fh)['results'][0]
        exported = [x for _, x in read_cado_blocks(prefix + '.poly')]
        selected = read_cado_blocks(prefix + '.selected.poly')
        check('refinement: preserves controls and input', exported == [r['control'], r['proposal']] and
              open(src).read() == original and len(selected) == 1 and
              selected[0][1] == r['proposal' if r['accepted'] else 'control'], 'input, both alternatives and selection')
    relation = rotation_between(p, r['proposal'])
    check('refinement: exact exported pair', resultant(p) == resultant(r['proposal']) and
          relation is not None and relation['rotation'] == [0], f'dt = {r["translation"]}')
    check('refinement: C181 held-out gain', r['accepted'] and r['validation_gain'] > 0.005,
          f'{100*r["validation_gain"]:+.4f}% at 64k points')
    # A separate 256k check, never used by the optimizer or acceptance decision.
    # Choose the reference skew explicitly before evaluating every configuration
    # at fixed skew; -useskew must never hide a reference-only optimization.
    args.useskew = False
    ref = dict(ref, skew=score_murphy([ref], p['n'], args)[0][1])
    args.useskew = True
    args.eval_points = 256000
    values = score_murphy([r['control'], r['proposal'], ref], p['n'], args)
    gain = values[1][0] / values[0][0] - 1
    ratio = values[1][0] / values[2][0]
    check('refinement: independent 256k check', gain > 0.005 and ratio >= 0.9995 and
          abs(gain - r['validation_gain']) < 0.0005,
          f'gain {100*gain:+.4f}%, refined / msieve {ratio:.6f}')
    # Direct fixed-skew evaluation must not silently optimize the declared skew.
    args.eval_points = args.points = 1000
    odd = translated_pair(p, -1234567, 1e5)
    fixed = score_murphy([odd], p['n'], args)[0]
    check('refinement: fixed-skew evaluation', fixed[1] == odd['skew'] and fixed[2] == 0,
          f'skew {fixed[1]}, dt {fixed[2]}')
    # Reject the old noisy search accuracy before running an expensive search.
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run([sys.executable, os.path.join(HERE, 'refine_polys.py'),
                              os.path.join(C181, 'gpu.poly'), '--out', os.path.join(tmp, 'result'),
                              '--points', '1000'], capture_output=True, text=True)
        check('refinement: rejects noisy search', out.returncode == 2 and not os.listdir(tmp) and
              'need --points >= 16000 and --eval-points >= 4 * --points' in out.stderr,
              'K=1000 rejected without writing outputs')


def refinement_lattice_check():
    """Cheap lattice refinement: independent validation agrees with rescoring the
    exact export via the regular lattice path, and the control survives rejection."""
    p = read_cado_blocks(os.path.join(C181, 'gpu.poly'))[2][1]
    p['skew'] = 78307994.0
    args = refine_parser().parse_args(['unused', '--out', 'unused', '--params', 'job:31,32,15,45e6',
                                      '--lattice', '15,16384', '--qband', '45e6,2e8,7', '--max-evals', '32',
                                      '--min-gain', '100'])  # force the control to be retained
    r = refine([p], args)[0]
    args.useskew, args.nlat, args.npts, args.latseed = True, args.eval_nlat, args.eval_npts, args.eval_seed
    scores = lattice_scores([r['control'], r['proposal']], p['n'], args)
    expected = [r['validation_control'], r['validation_proposal']]
    check('refinement: lattice export validation', all(abs(s[0]/v-1) < 1e-6 for s, v in zip(scores, expected))
          and not r['accepted'] and r['evaluations'] <= 32, 'fixed exports agree, rejected proposal keeps control')
    args.latseed = 2
    independent = lattice_scores([r['control']], p['n'], args)[0][0]
    check('refinement: lattice sample streams', abs(independent/expected[0]-1) > 1e-6,
          'changing the random stream changes the sampled estimate')


def refinement_failure_check():
    """Exercise the actual CLI/protocol failure paths and safe output handling."""
    binary = cado_murphy_binary()
    r = subprocess.run([binary, '-search-selftest'], capture_output=True, text=True)
    check('refinement: distinct starts and exact budget', r.returncode == 0 and ': ok ' in r.stdout,
          r.stdout.strip() or r.stderr.strip())
    p = dict(read_cado_blocks(os.path.join(C181, 'gpu.poly'))[2][1], skew=78307994.0)
    content = {k: v * (2 if k.startswith('c') else 3 if k in ('Y0', 'Y1') else 1) for k, v in p.items()}
    bad = dict(p, c0=p['c0'] + 1)
    failed_score = dict(p, skew=1e100)  # finite, but exceeds the C++ search's integer bounds
    with tempfile.TemporaryDirectory() as tmp:
        src, prefix = Path(tmp) / 'input.poly', Path(tmp) / 'result'
        src.write_text('\n'.join(cado_poly_text(q) for q in (p, content, bad, failed_score)))
        cmd = [sys.executable, os.path.join(HERE, 'refine_polys.py'), str(src), '--out', str(prefix),
               '--params', 'job:31,32,15,45e6', '--max-evals', '32']
        r = subprocess.run(cmd, capture_output=True, text=True)
        report = json.loads(prefix.with_suffix('.json').read_text())
        rows = report['results']
        check('refinement: partial batch survives', r.returncode == 1 and report['failed'] == 2 and
              [x['status'] for x in rows] == ['ok', 'ok', 'error', 'error'] and
              len(read_cado_blocks(str(prefix) + '.poly')) == 4 and
              len(read_cado_blocks(str(prefix) + '.selected.poly')) == 2 and
              'primitive resultant' in rows[2]['error'] and 'invalid refinement score' in rows[3]['error'],
              'valid results exported; bad resultant and failed C++ score reported in input order')
        check('refinement: primitive content on both sides', rows[1]['original'] == content and
              rows[1]['content'] == {'algebraic': 2, 'rational': 3} and
              rows[0]['proposal'] == rows[1]['proposal'] and rows[0]['control'] == rows[1]['control'],
              'content 2/3 normalized exactly; original retained')
        outputs = [Path(str(prefix) + suffix) for suffix in ('.poly', '.selected.poly', '.json')]
        before = [f.read_bytes() for f in outputs]
        r = subprocess.run(cmd, capture_output=True, text=True)
        check('refinement: refuses existing outputs', r.returncode == 2 and 'outputs already exist' in r.stderr and
              before == [f.read_bytes() for f in outputs], 'all previous files preserved')
        # Test publication's race guard independently of the preflight check.
        race = [Path(tmp) / name for name in ('a', 'b', 'c')]
        race[1].write_text('existing')
        try:
            write_outputs(race, ['new'] * 3)
            refused = False
        except FileExistsError:
            refused = True
        check('refinement: publication cannot clobber', refused and not race[0].exists() and
              race[1].read_text() == 'existing' and not race[2].exists(), 'race refused; own partial files removed')
        write_outputs(race, ['replacement'] * 3, force=True)
        check('refinement: explicit replacement', all(f.read_text() == 'replacement' for f in race),
              'force replaces only the requested output files')
        # Reject tiny lattice searches for their sample counts, through both interfaces.
        cli = cmd + ['--out', str(Path(tmp) / 'tiny'), '--lattice', '15,16384', '--qband', '45e6,2e8,7',
                     '--nlat', '1', '--npts', '1']
        direct = [binary, '-refine', '-K', '16000', '-Keval', '64000', '-lattice', '15,16384',
                  '-qband', '45e6,2e8,7', '-nlat', '1', '-npts', '1', str(src)]
        for label, command, reason in [('CLI', cli, '--nlat >= 32 and --npts >= 2048'),
                                       ('C++', direct, 'nlat >= 32, npts >= 2048')]:
            r = subprocess.run(command, capture_output=True, text=True)
            check(f'refinement: lattice sample guard {label}', r.returncode == 2 and reason in r.stderr,
                  '1 x 1 search rejected for insufficient samples')
        for script in ('refine_polys.py', 'score_polys.py'):
            command = [sys.executable, os.path.join(HERE, script), str(src), '--params', 'wrong:31,32,15,45e6']
            if script == 'refine_polys.py':
                command += ['--out', str(Path(tmp) / 'badparams')]
            r = subprocess.run(command, capture_output=True, text=True)
            check(f'params validation: {script}', r.returncode == 2 and '--params needs default or job:' in r.stderr,
                  'unknown bounds prefix rejected')
        missing = dict(p)
        del missing['skew']
        src.write_text(cado_poly_text(missing) + '\n' + cado_poly_text(p))
        for mode in ([], ['-lattice', '15,16384', '-qband', '45e6,2e8,7', '-nlat', '1', '-npts', '1']):
            r = subprocess.run([binary, '-t', '1', '-K', '1000', '-Keval', '1000', '-useskew', *mode, str(src)],
                               capture_output=True, text=True)
            check('fixed skew fallback warning: ' + ('lattice' if mode else 'MurphyE'),
                  r.returncode == 0 and '1 of 2 polynomials have no' in r.stderr and 'best skew' in r.stderr,
                  'missing skew warned; explicitly declared skew still used')
        local_failures = []
        with patch.object(sys.modules[__name__], 'failures', local_failures), redirect_stdout(StringIO()) as captured:
            run_check('missing fixture', lambda: None, [str(Path(tmp) / 'absent.poly')])
            def failed_process():
                raise subprocess.CalledProcessError(1, 'refine', stderr='refinement failed')
            run_check('failed subprocess', failed_process)
        check('regression runner: failures do not crash', len(local_failures) == 2 and
              'fixture missing' in captured.getvalue() and 'refinement failed' in captured.getvalue(),
              'missing fixture and subprocess failure recorded as FAIL')


def main():
    winners = [os.path.join(C168, 'winners', f) for f in ('msieve_best.poly', 'gpu_tuned_best.poly', 'cado_best.poly')]

    # scorer: CADO's own MurphyE at two sample counts, and the K = 1000 values pinned
    for k in ('1000', '4000'):
        r = murphy(winners, '-K', k, '-selftest')
        check(f'scorer selftest K={k}', r.returncode == 0 and ' 0 differ' in r.stdout, r.stdout.strip())
    e1000 = [float(l.split('\t')[1]) for l in murphy(winners).stdout.split('\n') if l]
    pinned = [4.560404e-08, 4.597916e-08, 4.550356e-08]
    check('scorer K=1000 values', len(e1000) == 3 and all(abs(a / b - 1) < 1e-6 for a, b in zip(e1000, pinned)),
          f'{e1000} (pinned {pinned})')

    # quadrature: the apparent +0.8% of the tuned translation is sampling noise
    e_acc = [float(l.split('\t')[1]) for l in murphy(winners[:2], '-K', '4000', '-Keval', '16000').stdout.split('\n') if l]
    check('quadrature trap at K=1000', e1000[1] / e1000[0] > 1.005,
          f'tuned / msieve {e1000[1] / e1000[0]:.4f} at 1,000 points')
    check('quadrature trap at K=4000/16000', len(e_acc) == 2 and e_acc[1] / e_acc[0] < 1.0002,
          f'tuned / msieve {e_acc[1] / e_acc[0]:.5f} at 4,000/16,000 points' if len(e_acc) == 2 else 'no output')

    # content: the counterexample (Res(f, g) = 2 N with N = 77)
    toy = {'n': 77, 'Y0': 1, 'Y1': 2, 'c0': 5, 'c1': 0, 'c2': 0, 'c3': 2, 'c4': 2, 'c5': 2}
    got = [(d, {k: q[k] for k in ('c0', 'c1', 'c2', 'c3', 'c4', 'c5')}) for d, _, _, q in content_seeds(toy, 77)]
    want = [(2, {'c0': 3, 'c1': 1, 'c2': 0, 'c3': 1, 'c4': 1, 'c5': 1})]
    check('content: Y1 not invertible mod d', got == want, f'{got}')

    # content: c168 seeds against the original script's output, and Res identities
    cdir = os.path.join(C168, 'content')
    for seed in sorted(f for f in os.listdir(cdir) if f.endswith('.poly')):
        p = parse_cado_poly(os.path.join(cdir, seed))
        stem = seed[:-5]
        derived = content_seeds(p, p['n'])
        expected = sorted(f for f in os.listdir(os.path.join(cdir, 'expected')) if f.startswith(stem + '_'))
        ok = [f'{stem}_d{d}.poly' for d, _, _, _ in derived] == expected
        for d, _, _, q in derived:
            e = parse_cado_poly(os.path.join(cdir, 'expected', f'{stem}_d{d}.poly'))
            ok &= cado_poly_text(q) == cado_poly_text(e) and resultant(q) * d == resultant(p)
        check(f'content: {stem}', ok, f'd = {[d for d, _, _, _ in derived] or "none"}')

    # lattice: the lattice-aware yield model against the c208 test sieves
    ts = os.path.join(C208, 'testsieve')
    fixture = [os.path.join(ts, f) for f in ('measured.tsv', 'five.poly')]
    if not all(os.path.exists(f) for f in fixture):
        check('lattice vs c208 test sieves', False, f'fixture missing: {", ".join(f for f in fixture if not os.path.exists(f))}')
    else:
        lattice_check(ts)

    fixtures = [os.path.join(C181, f) for f in ('gpu.poly', 'msieve.poly')]
    run_check('refinement', refinement_check, fixtures)
    run_check('refinement lattice', refinement_lattice_check, fixtures[:1])
    run_check('refinement failures', refinement_failure_check, fixtures[:1])

    print(f'{"all checks passed" if not failures else "FAILED: " + ", ".join(failures)}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
