#!/usr/bin/env python3
"""CUDA final refinement, exact exports and independently evaluated MurphyE.

Use S23_ROPT_BIN=.../s23_ropt for device coverage. Small saved pools keep these
tests independent of expensive search coverage or disposable run directories.
"""
import math
import re
import tempfile
import unittest
from pathlib import Path
from test_ropt_class import BIN, ROOT, run, read_cado_blocks, resultant
from test_murphy_rank import bounds, score
from polyfmt import cado_poly_text, coeffs, translate
import subprocess

GPU = 'full-refine-gpu-v1' in subprocess.check_output([str(BIN), '-capabilities'], text=True).split()


def pool_for(path, seed, cells):
    header = ['S23_ROPT_POOL 1', '1.3 1', str(seed['n']), str(seed['Y0']), str(seed['Y1'])]
    header += [str(seed[f'c{i}']) for i in range(6)]
    header += [str(len(cells))]
    # All scores are deliberately tied. Replay tests numerical refinement, not
    # proxy selection. The coordinates, seed identity and exact starts are saved.
    header += [f'{u} {v} 0 0 0 {t}' for u, v, t in cells]
    path.write_text('\n'.join(header + ['END', '']))


@unittest.skipUnless(GPU, 'set S23_ROPT_BIN to the CUDA build')
class GpuRefinement(unittest.TestCase):
    def checked(self, src, *args):
        p = run(src, *args)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p.stdout

    def verify_exports(self, seed, out):
        rows = read_cado_blocks(out)
        self.assertTrue(rows)
        for label, p in rows:
            u, v, t = (int(re.search(rf'\b{k} (-?\d+)', label)[1]) for k in ('u', 'v', 't'))
            f = coeffs(seed)
            f[0] += v * seed['Y0']
            f[1] += u * seed['Y0'] + v * seed['Y1']
            f[2] += u * seed['Y1']
            content = math.gcd(*f)
            self.assertEqual(coeffs(p), translate([x // content for x in f], t))
            self.assertEqual(p['Y0'], seed['Y0'] + t * seed['Y1'])
            self.assertEqual(resultant(p) * content, resultant(seed))
        return rows

    def test_full_alpha_translation_content_checked_and_unchecked(self):
        for fixture in ('c161/winners/sopt.poly', 'c168/content/s0051.poly',
                        'c192/ropt_coverage/seed.poly', 'c208/ropt_coverage/s0010.poly'):
            with self.subTest(fixture=fixture), tempfile.TemporaryDirectory() as tmp:
                src = ROOT / 'stage23_bench/data' / fixture
                seed = read_cado_blocks(src)[0][1]
                pool, out = Path(tmp) / 'pool', Path(tmp) / 'out.poly'
                cells = [(0, 0, 0), (1, -20, 0), (-1, 35, 0), (7, -174864538, 0)]
                if fixture.startswith('c192'): cells += [(-34, -5754737141, 0)]
                pool_for(pool, seed, cells)
                common = ['-search', '-pool-in', pool, '-aw', 1.3, '-rerank', len(cells),
                          '-refine', len(cells), '-top', 1, '-out', out]
                text = self.checked(src, *common, '-refine-gpu', '-refine-check')
                self.assertIn('full-refine GPU check:', text)
                checked = out.read_bytes()
                self.verify_exports(seed, out)
                self.checked(src, *common, '-refine-gpu')
                self.assertEqual(checked, out.read_bytes())
                gpu_rows = read_cado_blocks(out)
                self.checked(src, *common, '-cpu')
                cpu_rows = read_cado_blocks(out)
                # Final translations use the same CPU canonicalization even
                # when unchecked; a device local-minimum tie cannot leak out.
                self.assertEqual(checked, out.read_bytes())
                rotation = lambda label: tuple(int(re.search(rf'\b{k} (-?\d+)', label)[1]) for k in ('u', 'v'))
                self.assertEqual({rotation(l) for l, _ in gpu_rows}, {rotation(l) for l, _ in cpu_rows})

    def test_murphy_translation_gain_and_independent_score(self):
        seed = read_cado_blocks(ROOT / 'stage23_bench/data/c181/translation/gpu.poly')[2][1]
        with tempfile.TemporaryDirectory() as tmp:
            src, pool, out = [Path(tmp) / name for name in ('seed.poly', 'pool', 'out.poly')]
            src.write_text(cado_poly_text(seed))
            pool_for(pool, seed, [(0, 0, 0), (1, 10, 0)])
            bf, bg, area = bounds('job:31,32,15,45e6')
            common = ['-search', '-pool-in', pool, '-aw', 1.3, '-rerank', 2, '-refine', 2,
                      '-top', 1, '-murphy', '-Bf', bf, '-Bg', bg, '-area', area, '-out', out]
            text = self.checked(src, *common, '-refine-gpu', '-refine-check', '-murphy-gpu', '-murphy-check')
            self.assertIn('MurphyE GPU: 2 cells, 0 host fallbacks', text)
            checked = out.read_bytes()
            rows = self.verify_exports(seed, out)
            independent = score([p for _, p in rows], 'job:31,32,15,45e6')
            for (label, _), value in zip(rows, independent):
                reported = float(re.search(r'\bMurphyE (\S+)', label)[1])
                # The independent CLI prints %.6e; the in-process integral
                # audit above checks full doubles to 2e-10 relative.
                self.assertAlmostEqual(value / reported, 1, delta=1e-6)
            self.checked(src, *common, '-refine-gpu', '-murphy-gpu')
            self.assertEqual(checked, out.read_bytes())
            self.checked(src, *common, '-cpu')
            control = score([p for _, p in read_cado_blocks(out)], 'job:31,32,15,45e6', 256000)
            selected = score([p for _, p in rows], 'job:31,32,15,45e6', 256000)
            self.assertGreaterEqual(max(selected) / max(control), 1 - 1e-6)
            expected = out.read_bytes()
            text = self.checked(src, *common, '-refine-gpu', '-murphy-gpu', '-dev', 2147483647)
            self.assertIn('MurphyE GPU: 0 cells, 2 host fallbacks', text)
            self.assertEqual(expected, out.read_bytes())

    def test_screen_then_full_refinement(self):
        src = ROOT / 'stage23_bench/data/c168/content/s0051.poly'
        seed = read_cado_blocks(src)[0][1]
        with tempfile.TemporaryDirectory() as tmp:
            pool, out = Path(tmp) / 'pool', Path(tmp) / 'out.poly'
            pool_for(pool, seed, [(u, v, 0) for u in range(-2, 3) for v in range(-2, 3)])
            common = ['-search', '-pool-in', pool, '-aw', 1.3, '-rerank', 8, '-refine', 8,
                      '-top', 1, '-alpha-screen', 25, 500, '-alpha-screen-exact-size',
                      '-alpha-screen-gpu', '-refine-gpu', '-out', out]
            self.checked(src, *common, '-refine-check')
            checked = out.read_bytes()
            self.checked(src, *common)
            self.assertEqual(checked, out.read_bytes())
            self.verify_exports(seed, out)

    def test_bounded_alpha_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, pool, out = [Path(tmp) / p for p in ('seed.poly', 'pool', 'out.poly')]
            seed = dict(n=2**80 + 1, Y0=-1, Y1=1, c0=2**80, c1=0, c2=0, c3=0, c4=0, c5=1)
            src.write_text(cado_poly_text(seed))
            pool_for(pool, seed, [(0, 0, 0), (1, 2, 0)])
            common = ['-search', '-pool-in', pool, '-aw', 1.3, '-rerank', 2,
                      '-refine', 2, '-top', 1, '-out', out]
            text = self.checked(src, *common, '-refine-gpu', '-refine-check')
            self.assertGreaterEqual(int(re.search(r'full-refine GPU: .*?, (\d+) host fallbacks', text)[1]), 1)
            self.verify_exports(seed, out)

    def test_persistent_session_changes_orders_devices_and_recovers(self):
        src = ROOT / 'stage23_bench/data/c192/ropt_coverage/seed.poly'
        seed = read_cado_blocks(src)[0][1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pool = tmp / 'pool'
            pool_for(pool, seed, [(-34, -5754737141, 0)])
            bf, bg, area = bounds('job:32,33,15,80e6')
            common = ['-poly', src, '-search', '-pool-in', pool, '-aw', 1.3, '-rerank', 1,
                      '-refine', 1, '-top', 1, '-t', 2, '-refine-gpu', '-murphy', '-murphy-gpu',
                      '-Bf', bf, '-Bg', bg, '-area', area]
            jobs, expected = [], []
            for i, extra in enumerate(([], ['-K', 32000, '-Keval', 128000, '-murphy-check'],
                                       ['-dev', 2147483647], [])):
                out = tmp / f'individual{i}.poly'
                argv = list(map(str, common + extra + ['-out', out]))
                p = subprocess.run([str(BIN), *argv], capture_output=True, text=True, timeout=120)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                expected.append(out.read_bytes())
                argv[-1] = str(tmp / f'batch{i}.poly')
                jobs.append(argv)
            manifest = tmp / 'jobs.tsv'
            manifest.write_text('S23_ROPT_BATCH 1\n' + ''.join('\t'.join(a) + '\n' for a in jobs))
            p = subprocess.run([str(BIN), '-batch', str(manifest)], capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            for i, data in enumerate(expected):
                self.assertEqual((tmp / f'batch{i}.poly').read_bytes(), data)


class RefinementOptions(unittest.TestCase):
    def test_invalid_options_preserve_output(self):
        src = ROOT / 'stage23_bench/data/c161/winners/sopt.poly'
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out.poly'
            out.write_bytes(b'original')
            for flags in (['-refine-check'], ['-murphy-check'], ['-murphy-gpu'],
                          ['-refine-gpu', '-cpu'], ['-refine-gpu', '-rerank', 0],
                          ['-refine-gpu', '-plan']):
                p = run(src, '-search', '-out', out, *flags)
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                self.assertEqual(out.read_bytes(), b'original')


if __name__ == '__main__':
    unittest.main(verbosity=2)
