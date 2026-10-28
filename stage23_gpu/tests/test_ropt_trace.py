#!/usr/bin/env python3
"""Trace cutoffs without changing the numerical search, pools or exports.

Set S23_ROPT_BIN to repeat the bounded checks with the CUDA binary.
"""
import re
import tempfile
from pathlib import Path
import unittest

from test_ropt_class import CONTROL, C208, run, read_cado_blocks


def coords(label):
    m = re.search(r'u\s+(-?\d+)\s+v\s+(-?\d+)', label)
    return tuple(map(int, m.groups()))


class CellTrace(unittest.TestCase):
    def checked(self, source, *args):
        p = run(source, *args)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p.stdout

    def result(self, text):
        return re.search(r'^# trace-cell .*: result (\S+)$', text, re.M).group(1)

    def test_window_cutoffs(self):
        args = ['-u', 0, 0, '-v', -4, 4, '-k', 16, '-top', 9, '-launch', 30]
        text = self.checked(CONTROL, *args)
        cells = [coords(line) for line in text.splitlines() if line.startswith('u ')]
        self.assertEqual(len(cells), 9)
        for target, extra, expected in (
                (cells[0], [], 'retained'),
                (cells[-1], ['-k', 1], 'block_topk'),
                (cells[-1], ['-top', 1], 'window_top'),
                ((1, 0), [], 'region'),
                ((0, 1), ['-residue', 2, 0, 0], 'explicit_class')):
            with self.subTest(expected=expected):
                text = self.checked(CONTROL, *args, *extra, '-trace-cell', *target)
                self.assertEqual(self.result(text), expected)

    def test_search_noninterference_and_cuts(self):
        args = ['-search', '-budget', .5, '-maxcells', 50000, '-maxlines', 16, '-umax', 16,
                '-seg', 12, '-rerank', 32, '-refine', 4, '-top', 3, '-launch', 30]
        with tempfile.TemporaryDirectory() as tmp:
            out, pool = Path(tmp) / 'out.poly', Path(tmp) / 'out.pool'
            for mode in ([], ['-residue', 6, 1, 2], ['-classes', 6, 8]):
                with self.subTest(mode=mode):
                    self.checked(CONTROL, *args, *mode, '-out', out, '-pool-out', pool)
                    original, saved_pool = out.read_bytes(), pool.read_bytes()
                    finalists = {coords(label) for label, _ in read_cado_blocks(out)}
                    target = coords(read_cado_blocks(out)[0][0])
                    text = self.checked(CONTROL, *args, *mode, '-out', out, '-pool-out', pool,
                                        '-trace-cell', *target)
                    self.assertEqual(self.result(text), 'retained')
                    self.assertEqual(out.read_bytes(), original)
                    self.assertEqual(pool.read_bytes(), saved_pool)
                    cells = [tuple(map(int, line.split()[:2]))
                             for line in saved_pool.decode().splitlines()[12:-1]]
                    # The last saved proxy cell is outside a smaller refinement pool.
                    self.assertGreater(len(cells), 8)
                    text = self.checked(CONTROL, *args, *mode, '-rerank', 8,
                                        '-trace-cell', *cells[-1])
                    self.assertEqual(self.result(text), 'proxy_pool')
                    rejected = next(c for c in cells if c not in finalists)
                    text = self.checked(CONTROL, *args, *mode, '-trace-cell', *rejected)
                    self.assertEqual(self.result(text), 'exact_alpha_cut')
                    text = self.checked(CONTROL, '-search', '-pool-in', pool, '-rerank', 32,
                                        '-refine', 4, '-top', 3, '-trace-cell', *target)
                    self.assertEqual(self.result(text), 'retained')
                    self.assertNotIn('stage block_topk', text)

    def test_plan_does_not_claim_sieving(self):
        text = self.checked(CONTROL, '-u', 0, 0, '-v', -4, 4, '-plan', '-trace-cell', 0, 0)
        self.assertEqual(self.result(text), 'plan_admitted')
        self.assertNotIn('block_topk', text)

    def test_historical_class_selection_loss(self):
        text = self.checked(C208 / 's0103.poly', '-search', '-plan', '-budget', 3,
                            '-maxcells', '2e11', '-umax', 20000, '-maxlines', 1000,
                            '-classes', 2520, 16, '-class-explore', 4, '-aw', 1.3,
                            '-trace-cell', 2014, 569611649768)
        self.assertIn('stage class_proposal pass', text)
        self.assertIn('score-only rank 15 of 80', text)
        self.assertEqual(self.result(text), 'class_selection')

    def test_invalid_options(self):
        for args in (['-search', '-trace-cell', 'oops', 0],
                     ['-search', '-trace-cell', 0, 2**63],
                     ['-cell', 0, 0, '-trace-cell', 0, 0],
                     ['-trace-cell', 0, 0]):
            with self.subTest(args=args):
                self.assertEqual(run(CONTROL, *args).returncode, 2)


if __name__ == '__main__':
    unittest.main()
