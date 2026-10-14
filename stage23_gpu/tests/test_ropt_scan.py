#!/usr/bin/env python3
"""Real-polynomial basin coverage, bounded exports and CLI regressions.

S23_ROPT_BIN can select the CUDA binary for the same acceptance checks.
"""
import os
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import cado_poly_text, read_cado_blocks, resultant, rotation_between

BIN = Path(os.environ.get('S23_ROPT_BIN', ROOT / 'stage23_gpu/build/s23_ropt_cpu')).resolve()
CONTROL = ROOT / 'stage23_bench/data/c161/winners/sopt.poly'
C208 = ROOT / 'stage23_bench/data/c208/ropt_coverage'


def run(poly, *args):
    return subprocess.run([str(BIN), '-poly', str(poly), '-t', '2', *map(str, args)],
                          capture_output=True, text=True, timeout=120)


def plan(poly, *args):
    p = run(poly, '-search', '-plan', '-budget', '.5', '-maxcells', 100000,
            '-maxlines', 32, '-umax', 8192, *args)
    if p.returncode:
        raise AssertionError(p.stderr)
    minimum = float(re.search(r'best line minimum ([\d.]+)', p.stdout)[1])
    rows = [line.split() for line in p.stdout.splitlines() if line and not line.startswith('#')]
    kept = {int(r[0]) for r in rows if int(r[3]) <= int(r[4])}
    cells = sum(max(0, int(r[4]) - int(r[3]) + 1) for r in rows)
    return minimum, kept, cells, p.stdout


class BasinScan(unittest.TestCase):
    def test_loss_fixture_checksums_and_exact_reference_rotations(self):
        references = dict(read_cado_blocks(C208 / 'references.poly'))
        for entry in json.loads((C208 / 'manifest.json').read_text()):
            src = C208 / entry['file']
            self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(), entry['sha256'])
            seed = read_cado_blocks(src)[0][1]
            self.assertEqual(resultant(seed) % seed['n'], 0)
            if 'reference_label' in entry:
                ref = references[entry['reference_label']]
                delta = rotation_between(seed, ref)
                self.assertIsNotNone(delta)
                self.assertFalse(delta['negated'])
                v, u = delta['rotation']
                self.assertEqual(dict(u=u, v=v-u*delta['t'], t=delta['t']), entry['reference_rotation'])
                self.assertEqual(resultant(ref) % ref['n'], 0)

    def test_real_shift_recovers_distant_basin_and_respects_cap(self):
        p = read_cado_blocks(CONTROL)[0][1]
        original = resultant(p)
        p['c1'] += 4096 * p['Y0']
        p['c2'] += 4096 * p['Y1']
        self.assertEqual(resultant(p), original)
        with tempfile.TemporaryDirectory() as tmp:
            shifted = Path(tmp) / 'shifted.poly'
            shifted.write_text(cado_poly_text(p))
            control, _, _, _ = plan(CONTROL)
            local, local_lines, _, _ = plan(shifted)
            distant, kept, cells, diagnostic = plan(shifted, '-u-scan', 256, '-u-basins', 2, '-u-radius', 8)
        self.assertGreater(local - control, 4.)
        self.assertAlmostEqual(distant, control, places=3)
        self.assertNotIn(-4096, local_lines)
        self.assertIn(-4096, kept)
        self.assertGreater(cells, 0)
        self.assertLessEqual(cells, 100000)
        self.assertIn('u scan retained center lines: -4096', diagnostic)

    def test_archived_c208_losses_have_deeper_distant_minima(self):
        for seed, bound in (('s0222', 65.4), ('s0032', 63.0)):
            with self.subTest(seed=seed):
                src = C208 / (seed + '.poly')
                self.assertEqual(resultant(read_cado_blocks(src)[0][1]) % read_cado_blocks(src)[0][1]['n'], 0)
                local, _, _, _ = plan(src, '-umax', 20000, '-maxlines', 1000, '-budget', 3)
                distant, kept, cells, _ = plan(src, '-umax', 20000, '-maxlines', 1000, '-budget', 3,
                                              '-u-scan', 256, '-u-basins', 4, '-u-radius', 128)
                self.assertLess(distant, bound)
                self.assertLess(distant, local - 1.)
                self.assertTrue(any(abs(u) > 1000 for u in kept))
                self.assertLessEqual(cells, 100000)

    def test_zero_option_preserves_exports_and_enabled_exports_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = []
            for i, options in enumerate(([], ['-u-scan', 0], ['-u-scan', 256, '-u-radius', 8])):
                dest = Path(tmp) / f'{i}.poly'
                p = run(CONTROL, '-search', '-budget', '.5', '-maxcells', 100000, '-maxlines', 32,
                        '-umax', 8192, '-rerank', 32, '-refine', 3, '-top', 3, '-check', 8,
                        '-launch', 30, '-out', dest, *options)
                self.assertEqual(p.returncode, 0, p.stderr)
                blocks = read_cado_blocks(dest)
                self.assertTrue(blocks)
                for _, poly in blocks:
                    self.assertNotEqual(resultant(poly), 0)
                    self.assertEqual(resultant(poly) % poly['n'], 0)
                outputs.append(dest.read_bytes())
            self.assertEqual(outputs[0], outputs[1])

    def test_invalid_options_fail_before_opening_input(self):
        for args in (['-search', '-u-scan', -1], ['-search', '-u-scan', '2.5'],
                     ['-search', '-u-scan', 1, '-umax', 20000],
                     ['-search', '-u-basins', 2], ['-search', '-u-radius', 1],
                     ['-search', '-u-scan', 256, '-u-basins', 17],
                     ['-search', '-u-scan', 256, '-u-radius', 4097],
                     ['-search', '-u-scan', 256, '-u-basins', 16, '-u-radius', 1024],
                     ['-cell', 0, 0, '-u-scan', 256],
                     ['-u', 0, 1, '-v', 0, 1, '-u-scan', 256]):
            with self.subTest(args=args):
                p = run('/no/such/input.poly', *args)
                self.assertEqual(p.returncode, 2)
                self.assertIn('-u-scan', p.stderr)


if __name__ == '__main__':
    unittest.main()
