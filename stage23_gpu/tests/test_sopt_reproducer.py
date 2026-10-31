#!/usr/bin/env python3
"""Batch extraction, preflight and queue bookkeeping; no GPU/process launches."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trace = load('sopt_study', 'stage23_bench/runs/20261029_sopt_descent/trace.py')
queue = load('sopt_queue', 'stage23_bench/runs/20261028_existing_jobs/queue.py')
CASES = json.loads(trace.SOURCE.read_text())['size_optimization_regressions']


class Reproducer(unittest.TestCase):
    def test_batch_layout_and_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            polys = [dict(CASES[0]['raw'], c0=CASES[0]['raw']['c0'] + i) for i in range(13)]
            blocks = [(trace.cado_poly_text(p) + '\n').encode() for p in polys]
            source = root / 'source.poly'
            source.write_bytes(b''.join(blocks))
            case = {'index': 10, 'raw': polys[9]}
            for warmup, full, first in ((0, False, 8), (1, False, 4), (1, True, 0)):
                with self.subTest(warmup=warmup, full=full):
                    out = root / f'{first}.poly'
                    result = trace.extract_context(source, case, {'batch': 4}, out, warmup, full)
                    self.assertEqual(out.read_bytes(), b''.join(blocks[first:12]))
                    self.assertEqual(result['trace_index'], 9 - first)
                    self.assertEqual(result['polynomials'], 12 - first)
            with self.assertRaisesRegex(ValueError, 'index does not match'):
                trace.extract_context(source, dict(case, index=9), {'batch': 4}, root / 'bad.poly')

    def test_missing_compiler_does_not_reserve_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            out = base / 'runs/new'
            with patch.object(trace, 'GPU_DIR', base), patch.object(trace.shutil, 'which', return_value=None), \
                    patch.object(sys, 'argv', ['trace.py', 'cpu', '--run-dir', str(out)]):
                with self.assertRaises(SystemExit) as stopped:
                    trace.main()
            self.assertEqual(stopped.exception.code, 2)
            self.assertFalse(out.exists())

    def test_saved_command_and_effective_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'logs').mkdir()
            (root / 'preprocess').mkdir()
            (root / 'logs/sopt_e0.json').write_text(json.dumps({'command': [
                'old/s23_sopt', '-inputpolys', '/old/raw.poly', '-t', '2', '-slice', '7.5']}))
            (root / 'logs/sopt_e0.log').write_text('# card, 123 workers\n')
            (root / 'preprocess/raw.poly').write_text('')
            (root / 'preprocess/cado_sopt_unsorted.txt').write_text('# s23_sopt: effort 0, GPU, batches of 4096\n')
            source, execution = trace.context_spec(root)
            self.assertEqual(source, root / 'preprocess/raw.poly')
            self.assertEqual(execution, dict(batch=4096, threads=2, slice=7.5, device=0, original_workers=123))


class Queue(unittest.TestCase):
    def test_only_refreshes_every_job_result(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(queue, 'BASE', Path(tmp)), \
                patch.object(queue, 'SPECS', [('a', 'fixed_a', 'p'), ('b', 'fixed_b', 'p')]), \
                patch.object(queue.jobs, 'pipeline') as pipeline:
            queue.save('status.json', {'results': {'a': {'stage': 'failed'}, 'b': {'stage': 'complete'}}})
            queue.save('a/queue_result.json', {'stage': 'complete'})
            queue.save('b/queue_result.json', {'stage': 'complete with validation errors'})
            queue.main(only='a')
            state = queue.driver.read_json(queue.BASE / 'status.json')
            self.assertEqual(state['stage'], 'finished with errors')
            self.assertEqual(state['results']['b']['stage'], 'complete with validation errors')
            pipeline.assert_not_called()
            self.assertEqual(queue.aggregate_stage({'a': {'stage': 'complete'}}), 'incomplete')

    def test_waiting_background_keeps_owner_pid(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(queue, 'BASE', Path(tmp)), \
                patch.object(queue.subprocess, 'Popen', side_effect=[SimpleNamespace(pid=22), SimpleNamespace(pid=33)]):
            queue.save('owner.json', {'pid': 11})
            queue.save('background.json', {'pid': 11})
            queue.background(wait_lock=True)
            queue.background(wait_lock=True)
            self.assertEqual(queue.driver.read_json(queue.BASE / 'owner.json')['pid'], 11)
            self.assertEqual(queue.driver.read_json(queue.BASE / 'background.json')['pid'], 11)
            self.assertEqual(sorted(p.name for p in (queue.BASE / 'background').iterdir()), ['22.json', '33.json'])

    def test_mixed_composites_rejected_before_combined_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sources = []
            for i, case in enumerate(CASES):
                p = base / f'input{i}.poly'
                p.write_text(trace.cado_poly_text(case['raw']) + '\n')
                sources.append(p)
            run = queue.driver.Run(base / 'run')
            with self.assertRaisesRegex(ValueError, 'different composites'):
                queue.combine_inputs(sources, base / 'combined.poly', run)
            self.assertFalse((run.path / 'inputs/source.raw').exists())


if __name__ == '__main__':
    unittest.main()
