#!/usr/bin/env python3
"""Benchmark protocol/layout/clock regressions; no GPU required."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
import bench_murphy as bench
import profile_murphy as profile


class Benchmark(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.case = copy.deepcopy(json.loads(
            (ROOT / 'stage23_bench/data/gpu_refinement/20261031_batching.json').read_text())['cases'][0])

    def test_default_bounds_and_pool_conventions(self):
        self.case['params'] = 'default'
        self.case['pool'] = self.case['pool'].replace('1.3 1\n', '2.5 0\n', 1)
        args = bench.replay_options(self.case)
        for flag, value in [('-aw', 2.5), ('-Bf', 1e7), ('-Bg', 5e6), ('-area', 1e16)]:
            self.assertEqual(args[args.index(flag)+1], value)
        self.assertIn('-noproj', args)

    def test_invalid_pool_headers_and_counts(self):
        for header in ['nan 1', '1.3 2', '1.3', '']:
            with self.subTest(header=header), self.assertRaises(ValueError):
                case = dict(self.case, pool=self.case['pool'].replace('1.3 1\n', header+'\n', 1))
                bench.replay_options(case)
        self.case['count'] += 1
        with self.assertRaises(ValueError):
            bench.replay_options(self.case)

    def run_benchmark(self, wall):
        self.case['params'] = 'default'
        self.case['count'] = 1
        rows = self.case['pool'].splitlines()
        self.case['pool'] = '\n'.join(rows[:11]+['1', rows[12], 'END'])+'\n'
        cases, binary = self.base/'cases.json', self.base/'binary'
        cases.write_text(json.dumps(dict(cases=[self.case])))
        binary.write_text('fake executable')

        def execute(command, **kwargs):
            log = kwargs['stdout']
            for index, line in enumerate(Path(command[-1]).read_text().splitlines()[1:], 1):
                args = line.split('\t')
                self.assertEqual(float(args[args.index('-Bf')+1]), 1e7)
                Path(args[args.index('-out')+1]).write_text(self.case['seed'])
                log.write(f', refine 3.00 s\n# batch job {index} status 0, 5.000000 s\n'.encode())
                if wall:
                    log.write(f'# batch job {index} wall {wall} s\n'.encode())
            return subprocess.CompletedProcess(command, 0)

        argv = ['bench_murphy.py', '--cases', str(cases), '--binary', str(binary),
                '--run-dir', str(self.base), '--label', 'run', '--repeats', '1']
        with patch.object(sys, 'argv', argv), patch.object(bench.subprocess, 'run', execute), \
                patch.object(bench.time, 'time', side_effect=[100.0, 107.0]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            bench.main()
        return json.loads((self.base/'run/result.json').read_text())

    def test_wall_and_steady_timings_are_separate(self):
        result = self.run_benchmark('2.000000')
        self.assertEqual(result['seconds'], 7)
        self.assertEqual(result['median_job_pass_seconds'], 2)
        self.assertEqual(result['median_steady_job_pass_seconds'], 5)
        self.assertEqual(result['median_pass_seconds'], 3)
        self.assertEqual(result['cases_per_pass'], 1)
        self.assertEqual(result['warmup_jobs'], 1)

    def test_legacy_clock_is_not_reported_as_wall(self):
        result = self.run_benchmark(None)
        self.assertIsNone(result['median_job_pass_seconds'])
        self.assertEqual(result['median_steady_job_pass_seconds'], 5)

    def test_backwards_wall_clock_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'wall timing'):
            self.run_benchmark('-1.000000')


class Profiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.db = self.base/'trace.sqlite'

    def trace(self, width=1, repeats=5, gpu=False):
        jobs = []
        with sqlite3.connect(self.db) as c:
            c.execute('CREATE TABLE NVTX_EVENTS (start INTEGER,end INTEGER,globalTid INTEGER,text TEXT)')
            if gpu:
                c.execute('CREATE TABLE StringIds (id INTEGER,value TEXT)')
                c.execute("INSERT INTO StringIds VALUES (1,'integral')")
                c.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL '
                          '(start INTEGER,end INTEGER,shortName INTEGER,gridY INTEGER)')
            for rep in range(-1, repeats):
                for tag in range(width):
                    start = len(jobs)*20_000_000
                    jobs.append(dict(tag=f'c{tag}', repetition=rep))
                    c.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,1,'ropt_job')", (start, start+10_000_000))
                    c.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,1,'host_canonical_alpha')",
                              (start+1_000_000, start+3_000_000))
                    if gpu:
                        for a, b in [(0, 6), (4, 8)]:
                            c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,1,2)',
                                      (start+a*1_000_000, start+b*1_000_000))
        (self.base/'result.json').write_text(json.dumps(dict(jobs=jobs, returncode=0)))

    def test_one_case_inferred_even_when_wrong_three_case_count_divides(self):
        self.trace()  # Six jobs: the former default wrongly reported one measured pass.
        result = profile.summarize(self.db)
        self.assertEqual(result['passes'], 5)
        self.assertEqual(result['warmup_jobs'], 1)
        self.assertEqual(result['cases_per_pass'], 1)
        self.assertEqual(result['mean_pass']['job_ms'], 10)

    def test_conflicting_explicit_layout_is_rejected(self):
        self.trace()
        with self.assertRaisesRegex(ValueError, 'disagree'):
            profile.summarize(self.db, 3, 3)

    def test_cpu_profile_without_cuda_tables(self):
        self.trace(width=3, repeats=2)
        result = profile.summarize(self.db)
        self.assertEqual(result['mean_pass']['job_ms'], 30)
        self.assertEqual(result['host_ranges']['host_canonical_alpha']['total_ms'], 6)
        self.assertIsNone(result['mean_pass']['gpu_busy_ms'])
        for key in ['kernels', 'copies', 'cuda_apis']:
            self.assertEqual(result[key], {})

    def test_kernel_only_trace_handles_missing_copy_and_runtime_tables(self):
        self.trace(gpu=True)
        result = profile.summarize(self.db)
        self.assertEqual(result['mean_pass']['gpu_busy_ms'], 8)  # overlap counted once
        self.assertEqual(result['kernels']['integral']['total_ms'], 10)
        self.assertEqual(result['copies'], {})
        self.assertEqual(result['cuda_apis'], {})

    def test_standalone_trace_requires_explicit_layout(self):
        self.trace()
        (self.base/'result.json').unlink()
        with self.assertRaisesRegex(ValueError, 'provide both'):
            profile.summarize(self.db)
        self.assertEqual(profile.summarize(self.db, 1, 1)['passes'], 5)

    def test_incomplete_trace_rejected_against_metadata(self):
        self.trace()
        with sqlite3.connect(self.db) as c:
            c.execute('DELETE FROM NVTX_EVENTS WHERE start>=100000000')
        with self.assertRaisesRegex(ValueError, 'trace job count'):
            profile.summarize(self.db)

    def test_unclosed_job_range_is_rejected(self):
        self.trace()
        with sqlite3.connect(self.db) as c:
            c.execute("UPDATE NVTX_EVENTS SET end=NULL WHERE start=100000000 AND text='ropt_job'")
        with self.assertRaisesRegex(ValueError, 'incomplete or invalid'):
            profile.summarize(self.db)


class BuildFlags(unittest.TestCase):
    def test_profile_toggle_rebuilds_both_ropt_binaries_and_shared_object(self):
        # Exercise the real Makefile dependency graph with a recording compiler;
        # no CUDA compiler or GPU execution is needed for this regression.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = base/'build'
            build.mkdir()
            (build/'cado_io.o').touch()
            calls = base/'calls.jsonl'
            compiler = base/'compiler'
            compiler.write_text(f'''#!{sys.executable}
import json, sys
from pathlib import Path
args = sys.argv[1:]
if '-dumpmachine' in args:
    print('x86_64-linux-gnu')
else:
    output = Path(args[args.index('-o')+1])
    with Path({str(calls)!r}).open('a') as f:
        f.write(json.dumps([output.name, '-DS23_ROPT_PROFILE' in args])+'\\n')
    output.write_text('compiled')
''')
            compiler.chmod(0o755)
            for flag, count in [(0, 3), (1, 6), (1, 6), (0, 9)]:
                subprocess.run(['make', '-s', '-C', str(ROOT/'stage23_gpu'), f'BUILD={build}',
                                f'CXX={compiler}', f'NVCC={compiler}', f'ROPT_PROFILE={flag}',
                                's23_ropt_cpu', 's23_ropt'], check=True, capture_output=True, timeout=30)
                rows = [json.loads(line) for line in calls.read_text().splitlines()]
                self.assertEqual(len(rows), count)
                self.assertEqual({name for name, _ in rows[-3:]},
                                 {'s23_ropt', 's23_ropt_cpu', 'murphy_rank.o'})
                self.assertTrue(all(profiled == bool(flag) for _, profiled in rows[-3:]))


if __name__ == '__main__':
    unittest.main()
