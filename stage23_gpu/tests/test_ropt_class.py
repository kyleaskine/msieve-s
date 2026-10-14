#!/usr/bin/env python3
"""Explicit-class CLI, cap accounting, archive coverage and exact exports.

Set S23_ROPT_BIN to the CUDA binary to repeat on the GPU. Automatic class
selection is deliberately outside this first execution-path acceptance.
"""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import read_cado_blocks, resultant

BIN = Path(os.environ.get('S23_ROPT_BIN', ROOT / 'stage23_gpu/build/s23_ropt_cpu')).resolve()
CONTROL = ROOT / 'stage23_bench/data/c161/winners/sopt.poly'
C208 = ROOT / 'stage23_bench/data/c208/ropt_coverage'


def run(poly, *args):
    return subprocess.run([str(BIN), '-poly', str(poly), '-t', '2', *map(str, args)],
                          capture_output=True, text=True, timeout=120)


class ResidueClass(unittest.TestCase):
    def checked(self, poly, *args):
        p = run(poly, *args)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        if '-check' in args:
            self.assertIn('vs brute force: 0 differ', p.stdout)
        return p.stdout

    def window(self, *args):
        output = self.checked(CONTROL, '-u', -3, 3, '-v', -4, 4, '-k', 16, '-top', 100,
                              '-check', 64, '-launch', 30, *args)
        cells = {}
        for line in output.splitlines():
            match = re.match(r'u\s+(-?\d+)\s+v\s+(-?\d+)\s+score\s+([\d.]+)', line)
            if match:
                u, v, score = match.groups()
                cells[int(u), int(v)] = float(score)
        return cells

    def test_union_of_classes_equals_dense_window(self):
        dense = self.window()
        self.assertEqual(len(dense), 63)
        self.assertEqual(dense, self.window('-residue', 1, -7, 93))
        for modulus in (2, 3):
            combined = {}
            for u in range(modulus):
                for v in range(modulus):
                    cells = self.window('-residue', modulus, u, v)
                    self.assertTrue(all(x % modulus == u and y % modulus == v for x, y in cells))
                    self.assertFalse(combined.keys() & cells.keys())
                    combined.update(cells)
            self.assertEqual(dense, combined)
        self.assertEqual({}, self.window('-residue', 2520, 1000, 1000))
        self.assertEqual(self.window('-residue', 6, -1, -2), self.window('-residue', 6, 5, 4))

    def test_plan_counts_selected_cells_and_trims_to_cap(self):
        for cap in (1, 2, 17, 100000):
            text = self.checked(CONTROL, '-search', '-plan', '-budget', .5, '-maxcells', cap,
                                '-umax', 12, '-maxlines', 10, '-residue', 6, 1, -2)
            count = 0
            for line in text.splitlines():
                if line.startswith('#') or not line:
                    continue
                u, _, _, lo, hi, _ = line.split()
                u, lo, hi = int(u), int(lo), int(hi)
                self.assertEqual(u % 6, 1)
                if lo <= hi:
                    self.assertEqual(lo % 6, 4)
                    self.assertEqual(hi % 6, 4)
                    count += (hi - lo) // 6 + 1
            self.assertGreater(count, 0)
            self.assertLessEqual(count, cap)

    def test_known_reference_classes_admit_archived_winners(self):
        for record in json.loads((C208 / 'manifest.json').read_text()):
            if 'reference_label' not in record:
                continue
            with self.subTest(seed=record['seed']):
                r = record['reference_rotation']
                text = self.checked(C208 / record['file'], '-search', '-plan', '-budget', 3,
                                    '-maxcells', '2e11', '-umax', 20000, '-maxlines', 1000,
                                    '-residue', 2520, r['u'], r['v'])
                rows = [line.split() for line in text.splitlines() if line and not line.startswith('#')]
                matching = [row for row in rows if int(row[0]) == r['u'] and int(row[3]) <= r['v'] <= int(row[4])]
                self.assertEqual(len(matching), 1)
                count = sum((int(row[4]) - int(row[3])) // 2520 + 1 for row in rows if int(row[3]) <= int(row[4]))
                self.assertLessEqual(count, 200000000000)

    def test_identity_and_content_class_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            common = ['-search', '-budget', .5, '-maxcells', 100000, '-umax', 16, '-maxlines', 16,
                      '-rerank', 32, '-refine', 3, '-top', 3, '-launch', 30, '-check', 8]
            dense, identity = Path(tmp) / 'dense.poly', Path(tmp) / 'identity.poly'
            self.checked(CONTROL, *common, '-out', dense)
            self.checked(CONTROL, *common, '-residue', 1, 10, -17, '-out', identity)
            self.assertEqual(dense.read_bytes(), identity.read_bytes())
            for name, modulus in (('s0051', 2), ('s0068', 3)):
                src = ROOT / f'stage23_bench/data/c168/content/{name}.poly'
                seed = read_cado_blocks(src)[0][1]
                content_classes = []
                for u in range(modulus):
                    for v in range(modulus):
                        f = [seed[f'c{i}'] for i in range(6)]
                        f[0] += v * seed['Y0']
                        f[1] += v * seed['Y1'] + u * seed['Y0']
                        f[2] += u * seed['Y1']
                        if math.gcd(*f) % modulus == 0:
                            content_classes.append((u, v))
                self.assertTrue(content_classes)
                u, v = content_classes[0]
                dest = Path(tmp) / (name + '.poly')
                self.checked(src, *common, '-residue', modulus, u, v, '-out', dest)
                self.assertIn('content divided', dest.read_text())
                blocks = read_cado_blocks(dest)
                self.assertTrue(blocks)
                for label, poly in blocks:
                    self.assertEqual(poly['n'], seed['n'])
                    self.assertNotEqual(resultant(poly), 0)
                    self.assertEqual(resultant(poly) % seed['n'], 0)
                    coords = re.match(r'u (-?\d+) v (-?\d+)', label)
                    self.assertEqual(int(coords[1]) % modulus, u)
                    self.assertEqual(int(coords[2]) % modulus, v)

    def test_invalid_class_options(self):
        for args in (['-residue', 0, 1, 2], ['-residue', -1, 0, 0], ['-residue', 65537, 0, 0],
                     ['-residue', 6, '1.5', 0], ['-residue', 6, 2**63, 0],
                     ['-residue', 6, 0, 0, '-u-scan', 256], ['-residue', 6, 0, 0, '-cell', 0, 0],
                     ['-residue', 6, 0, 0, '-umax', 1000000001]):
            with self.subTest(args=args):
                p = run('/no/such/file.poly', '-search', *args)
                self.assertEqual(p.returncode, 2)
                self.assertIn('-residue', p.stderr)


if __name__ == '__main__':
    unittest.main()
