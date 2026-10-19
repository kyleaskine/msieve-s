#!/usr/bin/env python3
"""Real CUDA screening checks. Select the CUDA executable with S23_ROPT_BIN."""
import re
import subprocess
import tempfile
from pathlib import Path
import unittest
from concurrent.futures import ThreadPoolExecutor
from test_ropt_class import BIN, ROOT, run, read_cado_blocks, resultant

GPU = 'alpha-screen-gpu-v1' in subprocess.check_output([str(BIN), '-capabilities'], text=True).split()


@unittest.skipUnless(GPU, 'set S23_ROPT_BIN to the CUDA build')
class GpuAlphaScreen(unittest.TestCase):
    def checked(self, source, *args):
        p = run(source, *args)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p.stdout

    def test_live_gpu_sieve_screen_and_replay(self):
        source = ROOT / 'stage23_bench/data/c161/winners/sopt.poly'
        with tempfile.TemporaryDirectory() as tmp:
            out, pool = Path(tmp) / 'out.poly', Path(tmp) / 'pool'
            common = ['-search', '-aw', 1.3, '-rerank', 64, '-refine', 32, '-top', 8,
                      '-alpha-screen', 512, 500, '-alpha-screen-exact-size', '-out', out]
            self.checked(source, *common, '-budget', .5, '-maxcells', 300000,
                         '-maxlines', 16, '-umax', 16, '-seg', 12, '-k', 16,
                         '-pool-out', pool, '-alpha-screen-gpu', '-alpha-screen-check')
            expected = out.read_bytes()
            self.checked(source, *common, '-cpu', '-pool-in', pool)
            self.assertEqual(out.read_bytes(), expected)
            self.checked(source, *common, '-pool-in', pool, '-alpha-screen-gpu')
            self.assertEqual(out.read_bytes(), expected)

    def test_full_bound_content_and_small_bound(self):
        source = ROOT / 'stage23_bench/data/c168/content/s0051.poly'
        with tempfile.TemporaryDirectory() as tmp:
            out, pool = Path(tmp) / 'out.poly', Path(tmp) / 'pool'
            self.checked(source, '-cpu', '-search', '-budget', .5, '-maxcells', 100000,
                         '-maxlines', 16, '-umax', 16, '-seg', 12, '-k', 16, '-aw', 1.3,
                         '-rerank', 128, '-refine', 8, '-top', 4, '-pool-out', pool, '-out', out)
            wide = out.read_bytes()
            for bound in (2, 500, 2000):
                common = ['-search', '-pool-in', pool, '-aw', 1.3, '-rerank', 8,
                          '-refine', 8, '-top', 4, '-alpha-screen', 128, bound,
                          '-alpha-screen-exact-size', '-out', out]
                self.checked(source, '-cpu', *common)
                expected = out.read_bytes()
                text = self.checked(source, *common, '-alpha-screen-gpu', '-alpha-screen-check')
                self.assertIn('GPU check:', text)
                self.assertEqual(out.read_bytes(), expected)
                if bound == 2000:
                    self.assertEqual(out.read_bytes(), wide)
                for _, p in read_cado_blocks(out):
                    self.assertTrue(resultant(p) and resultant(p) % p['n'] == 0)

    def test_depth_fallback(self):
        # Square-free x^5 + 2^80 has a root lifting beyond the GPU recursion cap.
        # Its resultant with x-1 is n. CADO's unbounded reference terminates.
        with tempfile.TemporaryDirectory() as tmp:
            src, pool, out = [Path(tmp) / p for p in ('seed.poly', 'pool', 'out.poly')]
            c = 2**80
            src.write_text(f'n: {c+1}\nY0: -1\nY1: 1\nc0: {c}\nc1: 0\nc2: 0\nc3: 0\nc4: 0\nc5: 1\n')
            pool.write_text(f'S23_ROPT_POOL 1\n1.3 1\n{c+1}\n-1\n1\n{c}\n0\n0\n0\n0\n1\n'
                            '2\n0 0 0 0 1 0\n1 2 0 1 1 0\nEND\n')
            common = ['-search', '-pool-in', pool, '-aw', 1.3, '-rerank', 2,
                      '-refine', 2, '-top', 2, '-alpha-screen', 2, 500,
                      '-alpha-screen-exact-size', '-out', out]
            self.checked(src, '-cpu', *common)
            expected = out.read_bytes()
            text = self.checked(src, *common, '-alpha-screen-gpu', '-alpha-screen-check')
            self.assertGreaterEqual(int(re.search(r'(\d+) host fallbacks', text)[1]), 1)
            self.assertEqual(out.read_bytes(), expected)

    def test_invalid_gpu_options(self):
        source = ROOT / 'stage23_bench/data/c168/content/s0051.poly'
        for args in (['-alpha-screen-gpu'], ['-alpha-screen-check'],
                     ['-alpha-screen', 4096, 500, '-alpha-screen-gpu'],
                     ['-alpha-screen', 4096, 500, '-alpha-screen-exact-size', '-alpha-screen-gpu', '-cpu']):
            p = run(source, '-search', *args)
            self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
            self.assertIn('-alpha-screen-gpu requires', p.stderr)

    def test_device_setup_failure_uses_cpu_screen(self):
        source = ROOT / 'stage23_bench/data/c168/content/s0051.poly'
        with tempfile.TemporaryDirectory() as tmp:
            out, pool = Path(tmp) / 'out.poly', Path(tmp) / 'pool'
            self.checked(source, '-cpu', '-search', '-budget', .5, '-maxcells', 100000,
                         '-maxlines', 16, '-umax', 16, '-seg', 12, '-k', 16, '-aw', 1.3,
                         '-rerank', 128, '-refine', 8, '-top', 4, '-pool-out', pool, '-out', out)
            common = ['-search', '-pool-in', pool, '-aw', 1.3, '-rerank', 8,
                      '-refine', 8, '-top', 4, '-alpha-screen', 128, 500,
                      '-alpha-screen-exact-size', '-out', out]
            self.checked(source, '-cpu', *common)
            expected = out.read_bytes()
            p = run(source, *common, '-alpha-screen-gpu', '-dev', 2147483647)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn('using CPU screen', p.stderr)
            self.assertEqual(out.read_bytes(), expected)
            # Exercise the normal two-worker policy on a shared GPU. Either
            # worker may fall back if stack/scratch reservation cannot fit.
            def worker(index):
                target = Path(tmp) / f'worker{index}.poly'
                self.checked(source, *common, '-out', target, '-alpha-screen-gpu')
                return target.read_bytes()
            with ThreadPoolExecutor(max_workers=2) as workers:
                self.assertEqual(list(workers.map(worker, range(2))), [expected, expected])

    def test_no_cut_does_not_run_redundant_screen(self):
        source = ROOT / 'stage23_bench/data/c168/content/s0051.poly'
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp) / 'pool'
            self.checked(source, '-cpu', '-search', '-budget', .5, '-maxcells', 100000,
                         '-maxlines', 16, '-umax', 16, '-seg', 12, '-k', 16, '-aw', 1.3,
                         '-rerank', 128, '-refine', 8, '-top', 4, '-pool-out', pool)
            text = self.checked(source, '-search', '-pool-in', pool, '-aw', 1.3,
                                 '-rerank', 128, '-refine', 8, '-top', 4,
                                 '-alpha-screen', 128, 500, '-alpha-screen-exact-size', '-alpha-screen-gpu')
            self.assertNotIn('boundary rechecks', text)


if __name__ == '__main__':
    unittest.main()
