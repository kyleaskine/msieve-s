#!/usr/bin/env python3
"""Automatic proposals, mixed-class cell caps, deterministic exports and CLI limits."""
import re
import tempfile
from pathlib import Path
import unittest

from test_ropt_class import CONTROL, ROOT, run, read_cado_blocks, resultant


def selections(text):
    return [(int(u), int(v), kind) for u, v, kind in re.findall(
        r'^# class u (\d+) v (\d+): .*?, (ranked|exploration)$', text, re.M)]


class AutomaticClasses(unittest.TestCase):
    def checked(self, source, *args):
        p = run(source, *args)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        if '-check' in args:
            self.assertIn('vs brute force: 0 differ', p.stdout)
        return p.stdout

    def test_shared_caps_and_determinism(self):
        for modulus, count, explore in ((2, 4, 1), (6, 8, 2), (2520, 16, 4)):
            for cap in (1, 2, 17, 100000):
                with self.subTest(modulus=modulus, cap=cap):
                    args = ['-search', '-plan', '-budget', .5, '-maxcells', cap, '-umax', 6000,
                            '-maxlines', 10, '-classes', modulus, count, '-class-explore', explore]
                    text = self.checked(CONTROL, *args)
                    chosen = selections(text)
                    self.assertEqual(len(chosen), count)
                    self.assertEqual(len({(u, v) for u, v, _ in chosen}), count)
                    self.assertEqual(sum(k == 'exploration' for _, _, k in chosen), explore)
                    if cap == 17:
                        self.assertEqual(chosen, selections(self.checked(CONTROL, *args)))
                    counts = {(u, v): 0 for u, v, _ in chosen}
                    for line in text.splitlines():
                        if not line or line.startswith('#'):
                            continue
                        u, _, _, lo, hi, _ = line.split()
                        u, lo, hi = int(u), int(lo), int(hi)
                        for ru, rv in counts:
                            if u % modulus == ru and lo <= hi:
                                first = lo + (rv - lo) % modulus
                                counts[ru, rv] += max(0, (hi - first) // modulus + 1)
                    logged = {(int(u), int(v)): int(n) for u, v, n in re.findall(
                        r'^# class cells u (\d+) v (\d+): (\d+) selected', text, re.M)}
                    self.assertEqual(counts, logged)
                    self.assertGreater(sum(counts.values()), 0)
                    self.assertLessEqual(sum(counts.values()), cap)

    def test_exports_and_content_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for source, modulus, count in ((CONTROL, 6, 8),
                    (ROOT / 'stage23_bench/data/c168/content/s0051.poly', 2, 4),
                    (ROOT / 'stage23_bench/data/c168/content/s0068.poly', 3, 9)):
                dest = Path(tmp) / 'out.poly'
                text = self.checked(source, '-search', '-budget', .5, '-maxcells', 100000,
                                    '-umax', 16, '-maxlines', 16, '-classes', modulus, count,
                                    '-rerank', 32, '-refine', 3, '-top', 3, '-check', 16,
                                    '-launch', 30, '-out', dest)
                chosen = {(u, v) for u, v, _ in selections(text)}
                blocks = read_cado_blocks(dest)
                self.assertTrue(blocks)
                seed = read_cado_blocks(source)[0][1]
                for label, poly in blocks:
                    coords = re.match(r'u (-?\d+) v (-?\d+)', label)
                    self.assertIn((int(coords[1]) % modulus, int(coords[2]) % modulus), chosen)
                    self.assertEqual(poly['n'], seed['n'])
                    self.assertNotEqual(resultant(poly), 0)
                    self.assertEqual(resultant(poly) % seed['n'], 0)
                finalists = re.findall(r'^# class finalists u \d+ v \d+: (\d+) cells', text, re.M)
                self.assertEqual(sum(map(int, finalists)), len(blocks))
                if modulus in (2, 3):
                    # Content classes remain eligible; the global output cut is
                    # allowed to prefer a different class on this tiny search.
                    contents = re.findall(r'guaranteed log content ([\d.]+)', text)
                    self.assertGreater(max(map(float, contents)), 0)

    def test_invalid_options(self):
        for args in (['-classes', 1, 1], ['-classes', 4097, 16], ['-classes', 2, 5],
                     ['-classes', 6, 65], ['-classes', 6, '1.5'], ['-class-explore', 0],
                     ['-classes', 6, 4, '-class-explore', -1], ['-classes', 6, 4, '-class-explore', 5],
                     ['-classes', 6, 4, '-u-scan', 256], ['-classes', 6, 4, '-residue', 6, 0, 0],
                     ['-classes', 6, 4, '-cell', 0, 0], ['-classes', 6, 4, '-u', -2, 2],
                     ['-classes', 6, 4, '-umax', 1000000001], ['-classes', 6, 4, '-maxlines', 0]):
            with self.subTest(args=args):
                p = run('/no/such/file.poly', '-search', *args)
                self.assertEqual(p.returncode, 2)
                self.assertIn('-classes', p.stderr)


if __name__ == '__main__':
    unittest.main()
