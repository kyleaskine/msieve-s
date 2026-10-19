#!/usr/bin/env python3
"""Intermediate alpha screening: replay, cutoff diagnostics, and exact exports.

S23_ROPT_BIN can select the CUDA build for the bounded live-search checks.
"""
import tempfile
import re
from pathlib import Path
import unittest

from test_ropt_class import CONTROL, ROOT, run, read_cado_blocks, resultant
from test_ropt_trace import coords


class AlphaScreen(unittest.TestCase):
    common = ['-search', '-budget', .5, '-maxcells', 100000, '-maxlines', 16,
              '-umax', 16, '-seg', 12, '-k', 16, '-aw', 1.3, '-top', 4,
              '-refine', 4, '-launch', 30]

    def checked(self, poly, *args):
        p = run(poly, *args)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p.stdout

    def test_no_screen_cut_preserves_exports_and_original_pool(self):
        sources = [CONTROL, ROOT / 'stage23_bench/data/c168/content/s0051.poly']
        with tempfile.TemporaryDirectory() as tmp:
            out, pool = Path(tmp) / 'out.poly', Path(tmp) / 'out.pool'
            for src in sources:
                with self.subTest(src=src):
                    self.checked(src, *self.common, '-rerank', 128, '-pool-out', pool, '-out', out)
                    expected, original_pool = out.read_bytes(), pool.read_bytes()
                    self.checked(src, *self.common, '-rerank', 128, '-alpha-screen', 128, 200,
                                 '-pool-out', pool, '-out', out)
                    self.assertEqual(out.read_bytes(), expected)
                    # Exact-size screening must reuse the same translation/size,
                    # including content division, when the screen drops no cells.
                    self.checked(src, '-search', '-aw', 1.3, '-pool-in', pool, '-rerank', 128,
                                 '-refine', 4, '-top', 4, '-alpha-screen', 128, 500,
                                 '-alpha-screen-exact-size', '-out', out)
                    self.assertEqual(out.read_bytes(), expected)
                    self.assertEqual(pool.read_bytes(), original_pool)
                    self.checked(src, '-search', '-aw', 1.3, '-pool-in', pool, '-rerank', 128,
                                 '-refine', 4, '-top', 4, '-alpha-screen', 128, 500, '-out', out)
                    self.assertEqual(out.read_bytes(), expected)

    def test_screen_cut_trace_replay_and_exact_resultants(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, replay, pool = [Path(tmp) / p for p in ('out.poly', 'replay.poly', 'pool')]
            text = self.checked(CONTROL, *self.common, '-rerank', 8, '-refine', 8,
                                '-alpha-screen', 128, 500, '-pool-out', pool, '-out', out)
            self.assertIn('128 proxy cells, alpha through 500, 8 to full refinement', text)
            finalists = {coords(label) for label, _ in read_cado_blocks(out)}
            self.assertEqual(len(finalists), 8)
            for _, poly in read_cado_blocks(out):
                r = resultant(poly)
                self.assertTrue(r and r % poly['n'] == 0)
            self.checked(CONTROL, '-search', '-aw', 1.3, '-pool-in', pool,
                         '-rerank', 8, '-refine', 8, '-top', 4,
                         '-alpha-screen', 128, 500, '-out', replay)
            self.assertEqual(out.read_bytes(), replay.read_bytes())
            cells = [tuple(map(int, line.split()[:2])) for line in pool.read_text().splitlines()[12:-1]]
            rejected = next(c for c in cells if c not in finalists)
            for target, expected in [(rejected, 'alpha_screen_cut'), (next(iter(finalists)), 'retained')]:
                text = self.checked(CONTROL, '-search', '-aw', 1.3, '-pool-in', pool,
                                    '-rerank', 8, '-refine', 8, '-top', 4,
                                    '-alpha-screen', 128, 500, '-trace-cell', *target)
                self.assertIn('stage proxy_pool pass', text)
                self.assertIn(f'result {expected}', text)

    def test_pool_larger_than_old_streaming_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp) / 'pool'
            self.checked(CONTROL, *self.common, '-maxcells', 1500000,
                         '-rerank', 8, '-alpha-screen', 5000, 200, '-pool-out', pool)
            self.assertEqual(int(pool.read_text().splitlines()[11]), 5000)

    def test_full_alpha_exact_size_screen_matches_full_refinement(self):
        # With the full alpha bound, the screen becomes the full ranking itself.
        # Cutting 128 -> 8 must give the same eight as refining all 128 normally.
        with tempfile.TemporaryDirectory() as tmp:
            pool, out = Path(tmp) / 'pool', Path(tmp) / 'out.poly'
            source = ROOT / 'stage23_bench/data/c168/content/s0051.poly'
            text = self.checked(source, *self.common, '-rerank', 128, '-refine', 8,
                                '-pool-out', pool, '-out', out)
            bound = int(re.search(r"CADO's, p <= (\d+)", text)[1])
            expected = out.read_bytes()
            self.checked(source, '-search', '-aw', 1.3, '-pool-in', pool, '-rerank', 8,
                         '-refine', 8, '-top', 4, '-alpha-screen', 128, bound,
                         '-alpha-screen-exact-size', '-out', out)
            self.assertEqual(out.read_bytes(), expected)

    def test_invalid_options(self):
        for extra in (['-alpha-screen-exact-size'], ['-alpha-screen', 0, 500], ['-alpha-screen', 31, 500],
                      ['-alpha-screen', 32, 1], ['-alpha-screen', 32, 100000],
                      ['-alpha-screen', 'oops', 500], ['-alpha-screen', 2**24 + 1, 500],
                      ['-alpha-screen', 32, 500, '-rerank', 0],
                      ['-alpha-screen', 32, 500, '-refine', 33],
                      ['-alpha-screen', 32, 500, '-plan'],
                      ['-alpha-screen', 32, 500, '-u', 0, 0, '-v', 0, 1],
                      ['-alpha-screen', 32, 500, '-cell', 0, 0]):
            with self.subTest(extra=extra):
                p = run(CONTROL, *self.common, '-rerank', 32, *extra)
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                self.assertIn('-alpha-screen needs', p.stderr)

    def test_nonfinite_candidate_does_not_discard_valid_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, pool, out = [Path(tmp) / p for p in ('seed.poly', 'pool', 'out.poly')]
            c = 2**500
            src.write_text(f'n: {c+1}\nY0: -1\nY1: 1\nc0: 1\nc1: 0\nc2: 0\nc3: 0\nc4: 0\nc5: {c}\n')
            # Both cells have a valid rotation; the second saved translation
            # overflows the fixed-width size model. Keep the first result.
            pool.write_text(f'S23_ROPT_POOL 1\n1.3 1\n{c+1}\n-1\n1\n1\n0\n0\n0\n0\n{c}\n'
                            '2\n0 0 0 0 1 0\n1 2 0 1 1 9223372036854775807\nEND\n')
            text = self.checked(src, '-cpu', '-search', '-pool-in', pool, '-aw', 1.3,
                                '-rerank', 1, '-refine', 1, '-top', 1, '-alpha-screen', 2, 2,
                                '-alpha-screen-exact-size', '-out', out)
            self.assertIn('1 nonfinite scores ranked last', text)
            polys = read_cado_blocks(out)
            self.assertEqual(len(polys), 1)
            self.assertTrue(resultant(polys[0][1]) % (c+1) == 0)


if __name__ == '__main__':
    unittest.main()
