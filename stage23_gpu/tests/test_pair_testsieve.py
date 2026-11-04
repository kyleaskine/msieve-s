"""Special-q accounting, band statistics and run handling of the pair test-sieve tool."""
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
import pair_testsieve as ts

# A stand-in for the GPU siever's bench. Each special-q root yields 3 relations for a poly
# with Y0 > 0 and 4 otherwise. With FAKE_STOP pointing at an existing file it removes that
# file, stages half the window in NAME.part and exits 0, like a clean signal stop. Rerun
# without --restart it resumes from the .part and reports only the resumed session's counts,
# as the real bench does.
FAKE_BENCH = '''#!/usr/bin/env python3
import os, sys
sys.path.insert(0, {tools!r})
from pair_testsieve import special_q_roots
from polyfmt import read_cado_blocks
argv = sys.argv[1:]
opt = lambda name: argv[argv.index(name) + 1]
if '--check-relations' in argv:
    sys.exit(0)
poly = read_cado_blocks(opt('--poly'))[0][1]
q0, q1 = (int(x) for x in opt('--qrange').split(':'))
dat, log = opt('--relations'), opt('--log')
part = dat + '.part'
per = 3 if poly['Y0'] > 0 else 4
lines, pairs = [], 0
for q, m in sorted(special_q_roots(poly, 1, q0, q1 - q0 + 1).items()):
    pairs += m
    lines += [f'{{i}},{{q}}:2:{{q:x}}' for i in range(per * m)]
if '--restart' in argv and os.path.exists(part):
    os.remove(part)
stop = os.environ.get('FAKE_STOP')
if stop and os.path.exists(stop):
    os.remove(stop)
    open(part, 'w').write(''.join(x + chr(10) for x in lines[:len(lines) // 2]))
    print(f'--- band of {{pairs // 2}} (q, rho)')
    print(f'total relations {{len(lines) // 2}}')
    sys.exit(0)
done = 0
if os.path.exists(part):
    done = len(open(part).read().splitlines())
    os.remove(part)
open(dat, 'w').write(''.join(x + chr(10) for x in lines))
open(log, 'w').write('# commit fake' + chr(10) + 'q=1 board=100.0W' + chr(10))
print(f'--- band of {{pairs - pairs * done // max(1, len(lines))}} (q, rho)')
print('factors x cofactor == norm, side 0: PASS')
print('factors x cofactor == norm, side 1: PASS')
print(f'total relations {{len(lines) - done}}')
print(f'wall clock per q, COMPLETE {{2.0 if per == 3 else 2.5}} ms')
'''
FAKE_FBGEN = '''#!/usr/bin/env python3
import sys
open(sys.argv[sys.argv.index('--out') + 1], 'w').write('fb' + chr(10))
'''
FAKE_NORMSCAN = '''#!/usr/bin/env python3
print('OK')
'''


def brute_roots(c, q):
    return sum(1 for x in range(q) if sum(ci * pow(x, i, q) for i, ci in enumerate(c)) % q == 0)


def poly_file(path, y0):
    """f = x^2 - 2, g = x + y0: the resultant is y0^2 - 2, which is n."""
    path.write_text(f'n: {y0 * y0 - 2}\nY0: {y0}\nY1: 1\nc0: -2\nc1: 0\nc2: 1\nskew: 1.0\n')
    return path


class FakeSieve(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.siever = base / 'siever'
        self.siever.mkdir()
        for name, text in (('bench', FAKE_BENCH.format(tools=str(ROOT / 'stage23_bench/tools'))),
                           ('fbgen_gpu', FAKE_FBGEN), ('normscan', FAKE_NORMSCAN)):
            (self.siever / name).write_text(text)
            (self.siever / name).chmod(0o755)
        self.a = poly_file(base / 'a.poly', 1000)
        self.b = poly_file(base / 'b.poly', -1000)
        self.run_dir = base / 'run'
        self.stop = base / 'stop'
        quiet = patch('builtins.print')
        idle = patch.object(ts, 'wait_for_idle_gpu', return_value=dict(samples=[0, 0, 0], busy=False))
        snapshot = patch.object(ts, 'gpu_snapshot', return_value=dict(gpu='0 %'))
        env = patch.dict(os.environ, FAKE_STOP=str(self.stop))
        for p in (quiet, idle, snapshot, env):
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def new_run(self, *extra, width=2000):
        ts.main([str(self.run_dir), '--poly', f'a={self.a}', '--poly', f'b={self.b}',
                 '--lim', '100000,100000', '--lpb', '20,20', '--mfb', '40,40', '--logI', '11', '--J', '1024',
                 '--band', '10000,40000', '--points', '3', '--width', str(width),
                 '--siever', str(self.siever), *extra])

    def expected_pairs(self, q0):
        p = ts.read_cado_blocks(self.a)[0][1]
        return sum(ts.special_q_roots(p, 1, q0, 2000).values())

    def test_run_report_and_save(self):
        self.new_run()
        status = json.loads((self.run_dir / 'status.json').read_text())
        self.assertEqual(status['stage'], 'complete')
        result = json.loads((self.run_dir / 'results.json').read_text())
        self.assertAlmostEqual(result['ratios']['b/a']['ratio'], 4 / 3)
        for row in result['rows']:
            self.assertEqual(row['pairs'], self.expected_pairs(row['q0']))
            self.assertEqual(row['se_rel_per_pair'], 0.0)
        argv = json.loads((self.run_dir / 'raw/a_10000.command.json').read_text())['argv']
        self.assertIn('--restart', argv)
        dest = Path(self.tmp.name) / 'saved'
        ts.main([str(self.run_dir), '--save', str(dest)])
        files = sorted(str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file())
        self.assertEqual(files, ['jobs/a.job', 'jobs/b.job', 'manifest.json', 'results.json',
                                 'sources/a.poly', 'sources/b.poly'])
        saved = json.loads((dest / 'results.json').read_text())
        self.assertEqual(saved['raw_artifacts'], str(self.run_dir.resolve()))
        self.assertEqual(saved['rows'], result['rows'])

    def test_a_stopped_window_is_sieved_again(self):
        self.stop.touch()
        with self.assertRaisesRegex(RuntimeError, 'stopped before finishing'):
            self.new_run()
        record = json.loads((self.run_dir / 'raw/a_10000.command.json').read_text())
        self.assertEqual((record['returncode'], record['complete']), (0, False))
        self.assertEqual(json.loads((self.run_dir / 'status.json').read_text())['stage'], 'failed')
        ts.main([str(self.run_dir), '--resume'])
        result = json.loads((self.run_dir / 'results.json').read_text())
        row = next(r for r in result['rows'] if r['poly'] == 'a' and r['q0'] == 10000)
        self.assertEqual(row['pairs'], self.expected_pairs(10000))
        self.assertAlmostEqual(result['ratios']['b/a']['ratio'], 4 / 3)

    def test_inputs_are_checked_before_the_run_exists(self):
        self.a.write_text(self.a.read_text().replace('Y0: 1000', 'Y0: 999'))
        with self.assertRaisesRegex(SystemExit, 'resultant'):
            self.new_run()
        self.assertFalse(self.run_dir.exists())
        poly_file(self.a, 1000)
        with self.assertRaisesRegex(SystemExit, 'widen --width'):
            self.new_run(width=40)
        with self.assertRaisesRegex(SystemExit, '--params'):
            self.new_run('--params', 'job:30,31')
        self.assertFalse(self.run_dir.exists())

    def test_report_with_new_params_completes_a_failed_run(self):
        self.new_run()
        ts.set_state(self.run_dir, 'failed', error='scorer did not build')
        with self.assertRaisesRegex(SystemExit, 'no complete report'):
            ts.main([str(self.run_dir), '--save', str(Path(self.tmp.name) / 'early')])
        model = {name: dict(skew=1.0, murphy_e=1.0, relations_per_q=[v] * 3) for name, v in (('a', 10.), ('b', 11.))}
        with patch.object(ts, 'check_params') as check, \
                patch.object(ts, 'model_scores', return_value=(model, 'abc')) as scores:
            ts.main([str(self.run_dir), '--report', '--params', 'job:30,31,13.5,1e4'])
        check.assert_called_once_with('job:30,31,13.5,1e4')
        self.assertEqual(scores.call_args.args[3], 'job:30,31,13.5,1e4')
        result = json.loads((self.run_dir / 'results.json').read_text())
        self.assertEqual((result['model_params'], result['model_scorer_sha256']), ('job:30,31,13.5,1e4', 'abc'))
        self.assertAlmostEqual(result['model_ratios']['b/a']['band'], 1.1)
        ts.main([str(self.run_dir), '--save', str(Path(self.tmp.name) / 'saved')])


class ModelAndGate(unittest.TestCase):
    def test_model_uses_the_sieved_special_q_side(self):
        import score_polys
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'jobs').mkdir()
            for name, y0 in (('a', 1000), ('b', -1000)):
                poly_file(run / f'jobs/{name}.job', y0)
            scorer = poly_file(run / 'scorer', 7)
            settings = dict(polys=dict(a='', b=''), logI=11, J=1024, sq_side=0)
            calls = []
            fake = lambda polys, n, args: calls.append(args) or [(1., 1., 1., [1., 1.])] * len(polys)
            with patch.object(score_polys, 'lattice_scores', fake), \
                    patch.object(score_polys, 'cado_murphy_binary', return_value=str(scorer)):
                _, digest = ts.model_scores(run, settings, [1e4, 2e4], 'job:30,31,13.5,1e4')
        self.assertEqual(calls[0].sqside, 0)
        self.assertEqual(len(digest), 64)

    def test_the_gpu_gate_waits_for_idle(self):
        readings = iter([90, 95, 90, 5, 0, 10])
        with patch.object(ts, 'gpu_utilization', side_effect=lambda: next(readings)), \
                patch.object(ts.time, 'sleep'), patch('builtins.print'):
            gate = ts.wait_for_idle_gpu(30, 1800)
        self.assertEqual((gate['samples'], gate['busy']), ([5, 0, 10], False))
        with patch.object(ts, 'gpu_utilization', return_value=None), patch.object(ts.time, 'sleep'):
            self.assertIsNone(ts.wait_for_idle_gpu(30, 1800)['busy'])
        self.assertTrue(ts.wait_for_idle_gpu(-1, 0)['disabled'])
        with patch.object(ts, 'gpu_utilization', return_value=99), patch.object(ts.time, 'sleep'):
            self.assertTrue(ts.wait_for_idle_gpu(30, 0)['busy'])


class PairTestSieve(unittest.TestCase):
    def test_root_counts_match_brute_force(self):
        rng = random.Random(7)
        for _ in range(200):
            c = [rng.randrange(-10**9, 10**9) for _ in range(5)] + [rng.randrange(1, 10**4)]
            for q in (2, 3, 101, 211, 997):
                if c[-1] % q:
                    self.assertEqual(ts.distinct_roots(c, q), brute_roots(c, q), (c, q))
        # A repeated root counts once: (x - 3)^2 (x - 5) (x^2 + 1) mod 101.
        c = [-45, 39, -56, 34, -11, 1]
        self.assertEqual(ts.distinct_roots(c, 101), brute_roots(c, 101))

    def test_special_q_roots_by_side(self):
        poly = dict(n=1, Y0=-7, Y1=103, c0=-2, c1=0, c2=1)       # f = x^2 - 2
        alg = ts.special_q_roots(poly, 1, 90, 30)
        self.assertEqual(sorted(alg), [97, 101, 103, 107, 109, 113])
        for q, m in alg.items():
            self.assertEqual(m, brute_roots([-2, 0, 1], q))
        rat = ts.special_q_roots(poly, 0, 90, 30)
        self.assertEqual(rat[103], 0)
        self.assertEqual(rat[97], 1)

    def test_relations_are_assigned_to_their_special_q_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            dat = Path(tmp) / 'r.dat'
            lines = ['1,2:3,5:61,2,7', '3,4:2:2,65', '1,2:3,5:61,2,7', '5,6:65:3,61,67']
            dat.write_text('\n'.join(lines) + '\n')
            raw, per_q, ambiguous = ts.relation_clusters(dat, 1, 0x60, 16)
            self.assertEqual((raw, per_q, ambiguous), (4, {0x61: 2, 0x65: 1}, 1))
            dat.write_text('1,2:3:5,7\n')
            with self.assertRaisesRegex(ValueError, 'no special-q'):
                ts.relation_clusters(dat, 1, 0x60, 16)

    def test_ratio_estimate_is_clustered_by_prime(self):
        roots = {11: 1, 13: 2, 17: 0, 19: 1}
        rate, se = ts.ratio_estimate(roots, {11: 10, 13: 30, 19: 0})
        self.assertAlmostEqual(rate, 10.0)
        units = [(10, 1), (30, 2), (0, 1)]
        var = sum((y - 10 * m) ** 2 for y, m in units) / 16 * 3 / 2
        self.assertAlmostEqual(se, math.sqrt(var))
        with self.assertRaisesRegex(ValueError, 'without roots'):
            ts.ratio_estimate(roots, {17: 1})
        with self.assertRaisesRegex(ValueError, 'at least 2'):
            ts.ratio_estimate({11: 2, 13: 0}, {11: 5})

    def test_band_weights_integrate_the_pair_density(self):
        settings = dict(band=[10**8, 5 * 10**8], width=20000,
                        q0s=[10**8 + round(i * (4 * 10**8 - 20000) / 8) for i in range(9)])
        xs, w = ts.band_weights(settings)
        self.assertEqual(len(xs), 9)
        self.assertAlmostEqual(sum(wi * math.log(x) for wi, x in zip(w, xs)), 4 * 10**8)
        exact = sum(1 / math.log(10**8 + k * 1000 + 500) for k in range(400000)) * 1000
        self.assertLess(abs(sum(w) / exact - 1), 2e-3)

    def test_report_ratios_and_intervals(self):
        settings = dict(polys=dict(a='a', b='b'), band=[1000, 3000], width=100, q0s=[1000, 2900], params=None)
        rows = [dict(poly=p, q0=q0, rel_per_pair=y, se_rel_per_pair=s, ms_per_pair=t, relations=10, duplicates=0)
                for p, q0, y, s, t in [('a', 1000, 10., .1, 100.), ('a', 2900, 8., .1, 100.),
                                       ('b', 1000, 9., .1, 100.), ('b', 2900, 9., .1, 50.)]]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'raw').mkdir()
            result = ts.report(run, settings, rows)
            saved = json.loads((run / 'results.json').read_text())
        _, w = ts.band_weights(settings)
        expected = (w[0] * 9 + w[1] * 9) / (w[0] * 10 + w[1] * 8)
        r = result['ratios']['b/a']
        self.assertAlmostEqual(r['ratio'], expected)
        self.assertLess(r['low'], expected)
        self.assertGreater(r['high'], expected)
        windows = result['per_window']['b/a']
        self.assertAlmostEqual(windows[0]['ratio'], 0.9)
        self.assertAlmostEqual(windows[1]['time'], 0.5)
        seconds = [sum(wi * t / 1000 for wi, t in zip(w, ts_)) for ts_ in ((100, 100), (100, 50))]
        self.assertAlmostEqual(result['throughput_ratios']['b/a'], expected * seconds[0] / seconds[1])
        self.assertEqual(saved['ratios'], json.loads(json.dumps(result['ratios'])))


if __name__ == '__main__':
    unittest.main()
