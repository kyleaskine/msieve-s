#!/usr/bin/env python3
"""Fault injection for batch isolation, protocol handling, and output permissions."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
import refine_polys as refiner
from polyfmt import cado_poly_text, read_cado_blocks


class RefinementFailures(unittest.TestCase):
    def setUp(self):
        self.args = refiner.parser().parse_args(['unused', '--out', 'unused'])
        self.p = dict(read_cado_blocks(ROOT / 'stage23_bench/data/c181/translation/gpu.poly')[2][1],
                      skew=78307994.0)

    @staticmethod
    def rows(polys):
        # Fixed control/proposal: no search is needed to exercise the process boundary.
        return [[str(i), str(p['skew']), '1', '0', str(p['skew']), '1', '1', '1', '1', '1',
                 '21', '0', str(p['Y0']), str(p['c0'])] for i, p in enumerate(polys)]

    def test_killed_group_preserves_completed_groups_and_isolates_one_bad_input(self):
        other = dict(self.p, n=2 * self.p['n'], c0=2 * self.p['c0'] + self.p['Y0'],
                     c1=2 * self.p['c1'] + self.p['Y1'])
        for i in range(2, 6):
            other[f'c{i}'] *= 2
        poison = dict(self.p, skew=1e100)
        calls = []

        def scorer(polys, n, args, extra, keep_skew):
            calls.append(len(polys))
            if len(polys) > 1 or polys[0]['skew'] == 1e100:
                raise RuntimeError('cado_murphy failed (status -9)')
            return self.rows(polys)

        with patch.object(refiner, '_run_cado_murphy', side_effect=scorer):
            results = refiner.refine([other, self.p, poison, self.p], self.args)
        self.assertEqual([r['status'] for r in results], ['ok', 'ok', 'error', 'ok'])
        self.assertIn('status -9', results[2]['error'])
        self.assertIn(3, calls)
        self.assertEqual(calls.count(1), 4)

    def test_bad_protocol_row_does_not_drop_valid_rows(self):
        for row in (['1', 'short'], ['wrong-index'], ['1', 'error', 'worker failed']):
            with self.subTest(row=row):
                rows = self.rows([self.p, self.p, self.p])
                rows[1] = row
                with patch.object(refiner, '_run_cado_murphy', return_value=rows):
                    results = refiner.refine([self.p] * 3, self.args)
                self.assertEqual([r['status'] for r in results], ['ok', 'error', 'ok'])

    def test_degree_11_cli_keeps_valid_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / 'in.poly', Path(tmp) / 'out'
            # Still has a valid resultant modulo n, so the degree guard is necessary.
            # cado_poly_text needs every coefficient up to the degree.
            bad = dict(self.p, **{f'c{i}': 0 for i in range(6, 11)}, c11=self.p['n'])
            src.write_text('\n'.join(cado_poly_text(p) for p in (self.p, bad, self.p)))
            r = subprocess.run([sys.executable, str(ROOT / 'stage23_bench/tools/refine_polys.py'),
                                str(src), '--out', str(out), '--max-evals', '32'],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 1, r.stderr)
            self.assertIn('degree at most 10', r.stderr)
            self.assertEqual(len(read_cado_blocks(str(out) + '.selected.poly')), 2)

    def test_output_modes_follow_umask(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mask in (0o022, 0o027, 0o077):
                paths = [Path(tmp) / f'{mask}.{i}' for i in range(3)]
                old = os.umask(mask)
                try:
                    refiner.write_outputs(paths, ['data'] * 3)
                finally:
                    os.umask(old)
                self.assertEqual([p.stat().st_mode & 0o777 for p in paths], [0o666 & ~mask] * 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
