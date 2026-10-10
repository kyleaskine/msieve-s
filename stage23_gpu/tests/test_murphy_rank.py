#!/usr/bin/env python3
"""Real-case checks of ropt's in-process MurphyE ranking; no GPU required."""
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import cado_poly_text, read_cado_blocks, resultant, rotation_between
from score_polys import murphy, score_bounds

BUILD = ROOT / 'stage23_gpu/build'
C181 = ROOT / 'stage23_bench/data/c181/translation/gpu.poly'
C205 = ROOT / 'stage23_bench/data/c205/gpu_ropt/verified.poly'


def fields(label):
    words = label.split()
    return dict(zip(words[::2], words[1::2]))


def bounds(job):
    d = score_bounds(job)
    return [str(d[k]) for k in ('Bf', 'Bg', 'area')]


def score(polys, job, k=64000):
    args = SimpleNamespace(params=job, threads=2, points=16000, eval_points=k,
                           trans=0, useskew=True)
    return [v[0] for v in murphy(polys, polys[0]['n'], args)]


class MurphyRanking(unittest.TestCase):
    def probe(self, polys, job, tmp):
        src, dst = tmp / 'input.poly', tmp / 'output.poly'
        src.write_text('\n'.join(cado_poly_text(p) for p in polys))
        r = subprocess.run([str(BUILD / 'murphy_rank_probe'), str(src), *bounds(job)],
                           capture_output=True, text=True, check=True)
        dst.write_text(r.stdout)
        rows = [(fields(label), p) for label, p in read_cado_blocks(dst)]
        self.assertEqual(len(polys), len(rows))
        for original, (stats, selected) in zip(polys, rows):
            self.assertGreater(float(stats['score']), 0)
            self.assertTrue(math.isfinite(float(stats['score'])))
            self.assertGreaterEqual(float(stats['score']), float(stats['control']))
            self.assertEqual(resultant(original), resultant(selected))
            transform = rotation_between(original, selected)
            self.assertIsNotNone(transform)
            self.assertFalse(any(transform['rotation']))
            self.assertEqual(selected['Y0'], original['Y0'] + int(stats['dt']) * original['Y1'])
        values = score([p for _, p in rows], job)
        for value, (stats, _) in zip(values, rows):
            self.assertAlmostEqual(value / float(stats['score']), 1, delta=2e-6)
        return rows

    def test_c205_ranking_reversal(self):
        blocks = read_cado_blocks(C205)
        # The proxy puts the inferior cell first. Keep that order here.
        polys = [blocks[4][1], blocks[0][1]]
        self.assertIn('rank 1 ', blocks[4][0])
        self.assertIn('rank 2 ', blocks[0][0])
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.probe(polys, 'job:33,34,16.5,80e6', Path(tmp))
        ratio = float(rows[1][0]['score']) / float(rows[0][0]['score'])
        self.assertGreater(ratio, 1.009)
        independent = score([p for _, p in rows], 'job:33,34,16.5,80e6', 256000)
        self.assertGreater(independent[1] / independent[0], 1.009)

    def test_c181_translation_gain(self):
        p = read_cado_blocks(C181)[2][1]
        with tempfile.TemporaryDirectory() as tmp:
            [(stats, selected)] = self.probe([p], 'job:31,32,15,45e6', Path(tmp))
        self.assertEqual(stats['accepted'], '1')
        self.assertNotEqual(int(stats['dt']), 0)
        self.assertGreater(float(stats['score']) / float(stats['control']), 1.005)
        # Reference skew chosen at K=16000; 256k only rechecks fixed configurations.
        args = SimpleNamespace(params='job:31,32,15,45e6', threads=2, points=16000,
                               eval_points=64000, trans=0, useskew=False)
        control = dict(p, skew=murphy([p], p['n'], args)[0][1])
        values = score([control, selected], args.params, 256000)
        self.assertGreater(values[1] / values[0], 1.005)

    def test_search_before_output_cut_and_exact_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seed = read_cado_blocks(C205)[4][1]
            src, out = tmp / 'seed.poly', tmp / 'selected.poly'
            src.write_text(cado_poly_text(seed))
            cmd = [str(BUILD / 's23_ropt_cpu'), '-poly', str(src), '-search', '-budget', '0.5',
                   '-umax', '0', '-maxcells', '10000', '-rerank', '8', '-refine', '1', '-top', '1',
                   '-t', '2', '-check', '16', '-out', str(out)]
            bf, bg, area = bounds('job:33,34,16.5,80e6')
            opts = ['-murphy', '-Bf', bf, '-Bg', bg, '-area', area]
            r = subprocess.run(cmd + opts + ['-murphy-points', '16000', '-murphy-eval-points', '64000'],
                               capture_output=True, text=True, check=True)
            self.assertIn('8 cells before the 1-output cut', r.stderr)
            self.assertIn('8 optimized', r.stdout)
            self.assertIn('0 failed', r.stdout)
            self.assertIn('0 skipped before MurphyE', r.stdout)
            self.assertIn('0 differ', r.stdout)
            rows = read_cado_blocks(out)
            self.assertEqual(len(rows), 1)
            label, selected = rows[0]
            self.assertIn('lognorm_at_murphy_skew ', label)
            u, v, t = (int(re.search(rf'\b{k} (-?\d+)', label)[1]) for k in ('u', 'v', 't'))
            # Reconstruct the rotation and integer translation independently in Python.
            from polyfmt import translate, coeffs
            f = coeffs(seed)
            f[0] += v * seed['Y0']
            f[1] += u * seed['Y0'] + v * seed['Y1']
            f[2] += u * seed['Y1']
            content = math.gcd(*f)
            expected = translate([x // content for x in f], t)
            self.assertEqual(coeffs(selected), expected)
            self.assertEqual(selected['Y0'], seed['Y0'] + t * seed['Y1'])
            self.assertEqual(resultant(selected) * content, resultant(seed))
            written_score = float(re.search(r'\bMurphyE (\S+)', label)[1])
            self.assertAlmostEqual(score([selected], 'job:33,34,16.5,80e6')[0] / written_score, 1, delta=2e-6)
            # Invalid or inapplicable scoring options fail before any output opens.
            original = out.read_bytes()
            invalid = [opts + ['-K', '1000'], opts + ['-Keval', '16000'], opts + ['-K', '16k'],
                       opts + ['-maxeval', '31'], opts + ['-Bf', 'nan'], opts + ['-area', 'inf'],
                       opts + ['-Bg', '1'], opts + ['-rerank', '0'], ['-murphy'],
                       ['-Bf', bf], opts + ['-u', '0', '1'], opts + ['-cell', '0', '0'], opts + ['-plan'],
                       opts + ['-murphy-points', '16k'], opts + ['-murphy-eval-points', '16000']]
            for option in invalid:
                with self.subTest(option=option):
                    bad = subprocess.run(cmd + option, capture_output=True, text=True)
                    self.assertEqual(bad.returncode, 2)
                    self.assertIn('-murphy requires', bad.stderr)
                    self.assertEqual(out.read_bytes(), original)
            # -k names the sieve count, not MurphyE's former uppercase alias.
            bad = subprocess.run(cmd + ['-k', '17'], capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)
            self.assertIn('1 <= -k <=', bad.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
