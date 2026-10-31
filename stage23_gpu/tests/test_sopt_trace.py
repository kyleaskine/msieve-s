#!/usr/bin/env python3
"""CPU regression checks for the two October 28 initial-sopt losses and phase traces.

S23_SOPT_BIN may select the CUDA build; these tests always pass -cpu so they
remain safe while a GPU is occupied. Actual device comparison is a separate check.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import cado_poly_text, coeffs, iter_sopt
BINARY = Path(os.environ.get('S23_SOPT_BIN', ROOT / 'stage23_gpu/build/s23_sopt_cpu'))
REPORT = ROOT / 'stage23_bench/data/gpu_pipeline/20261028.json'
CASES = json.loads(REPORT.read_text())['size_optimization_regressions']
PHASES = ['prepare', 'lll', 'dedupe', 'descent', 'reduce']


def pair(p):
    return list(map(str, coeffs(p))), [str(p[f'Y{i}']) for i in range(2)]


def run(source, *args):
    return subprocess.run([str(BINARY), '-cpu', '-t', '2', '-inputpolys', str(source),
                           *map(str, args)], text=True, capture_output=True, timeout=60)


class SoptTrace(unittest.TestCase):
    def checked(self, source, *args):
        p = run(source, *args)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p

    def check_trace(self, path, case, index, expected):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([r['phase'] for r in records], PHASES)
        for r in records:
            self.assertEqual(r['schema'], 1)
            self.assertEqual(r['backend'], 'CPU')
            self.assertEqual(r['index'], index)
            self.assertEqual(r['effort'], 0)
            self.assertEqual(r['status'], 0)
            self.assertGreater(r['ncand'], 0)
            self.assertEqual(r['ncand'], records[0]['ncand'])
            self.assertEqual(len(r['candidates']), r['ncand'])
        prepare, lll, dedupe, descent, reduce = records
        self.assertEqual((prepare['f'], prepare['g']), pair(case['raw']))
        ks = [c['k'] for c in prepare['candidates']]
        self.assertEqual(ks, sorted(set(ks)))
        self.assertTrue(all(c['state'] == 4 and 'f' not in c for c in prepare['candidates']))
        self.assertTrue(all(c['state'] == 1 for c in lll['candidates']))
        self.assertTrue(all(c['state'] in (1, 2) for c in dedupe['candidates']))
        self.assertTrue(all(c['state'] in (2, 3) for c in descent['candidates']))
        best = min([prepare, *[c for c in descent['candidates'] if c['state'] == 3]],
                   key=lambda c: c['objective'])
        self.assertEqual((reduce['f'], reduce['g']), (best['f'], best['g']))
        self.assertEqual((reduce['f'], reduce['g']), pair(expected))
        self.assertEqual(reduce['objective'], best['objective'])
        # Historical winners are observations on one libm/CPU, not portable assertions.
        print(f"{case['run']}: matches historical CPU winner: "
              f"{(reduce['f'], reduce['g']) == pair(case['cpu'])}")
        return records

    def test_phase_trace_and_cado_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'raw.poly'
            source.write_text(''.join(cado_poly_text(c['raw']) + '\n' for c in CASES))
            control = self.checked(source, '-compare')
            self.assertIn('identical to CADO: 2 of 2', control.stderr)
            control_path = Path(tmp) / 'control.out'
            control_path.write_text(control.stdout)
            expected = [p for _, p in iter_sopt(control_path)]
            for i, case in enumerate(CASES):
                with self.subTest(case=case['run']):
                    trace = Path(tmp) / f'{i}.jsonl'
                    p = self.checked(source, '-compare', '-trace', trace, '-trace-index', i)
                    self.assertEqual(p.stdout, control.stdout)
                    self.assertIn('identical to CADO: 2 of 2', p.stderr)
                    self.check_trace(trace, case, i, expected[i])

    def test_trace_index_across_batches_and_reused_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, trace = Path(tmp) / 'raw.poly', Path(tmp) / 'trace.jsonl'
            source.write_text(''.join(cado_poly_text(c['raw']) + '\n' for c in [*CASES, CASES[0]]))
            control = self.checked(source, '-batch', 1, '-t', 1, '-compare')
            traced = self.checked(source, '-batch', 1, '-t', 1, '-compare', '-trace', trace, '-trace-index', 2)
            self.assertEqual(traced.stdout, control.stdout)
            self.assertIn('identical to CADO: 3 of 3', traced.stderr)
            control_path = Path(tmp) / 'control.out'
            control_path.write_text(control.stdout)
            self.check_trace(trace, CASES[0], 2, list(iter_sopt(control_path))[2][1])

    def test_invalid_options_and_preserved_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'raw.poly'
            original = cado_poly_text(CASES[0]['raw']) + '\n'
            source.write_text(original)
            for args in (['-trace-index', 0], ['-trace-index', -1],
                         ['-trace-index', 'oops'], ['-trace-index', 2**100],
                         ['-trace', source]):
                with self.subTest(args=args):
                    self.assertEqual(run(source, *args).returncode, 2)
                    self.assertEqual(source.read_text(), original)
            trace = Path(tmp) / 'missing.jsonl'
            p = run(source, '-trace', trace, '-trace-index', 1)
            self.assertEqual(p.returncode, 2)
            self.assertIn('trace index 1 not found', p.stderr)
            self.assertFalse(trace.exists())
            self.checked(source, '-trace', trace)
            self.assertEqual(run(source, '-trace', trace).returncode, 2)
            fresh = Path(tmp) / 'retry.jsonl'
            self.assertEqual(run(Path(tmp) / 'absent.poly', '-trace', fresh).returncode, 2)
            self.assertFalse(fresh.exists())
            self.checked(source, '-trace', fresh)

    def test_bad_trace_parent_fails_before_first_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / 'raw.poly'
            source.write_text((cado_poly_text(CASES[0]['raw']) + '\n') * 3)
            paths = [base / 'absent/trace.jsonl', source / 'trace.jsonl', '']
            locked = base / 'locked'
            locked.mkdir(mode=0o500)
            if not os.access(locked, os.W_OK, effective_ids=True):
                paths.append(locked / 'trace.jsonl')
            for path in paths:
                with self.subTest(path=path):
                    p = run(source, '-batch', 1, '-trace', path, '-trace-index', 2)
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertEqual(p.stdout, '')

    def test_fallback_has_no_uninitialized_candidate_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, trace = Path(tmp) / 'raw.poly', Path(tmp) / 'trace.jsonl'
            # Valid degree-four pair, outside the GPU size port's supported degree.
            source.write_text('n: 17\nY0: -2\nY1: 1\nc0: 1\nc1: 0\nc2: 0\nc3: 0\nc4: 1\n')
            control = self.checked(source)
            traced = self.checked(source, '-trace', trace)
            self.assertEqual(traced.stdout, control.stdout)
            self.assertIn('1 redone by CADO', traced.stderr)
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual([r['phase'] for r in records], PHASES)
            for r in records:
                self.assertEqual(r['status'], 1)
                self.assertEqual(r['ncand'], 0)
                self.assertEqual(r['candidates'], [])
                self.assertIsNone(r['objective'])
                self.assertNotIn('f', r)


if __name__ == '__main__':
    unittest.main()
