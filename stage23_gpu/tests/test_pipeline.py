#!/usr/bin/env python3
"""Driver regressions: format handoffs, resumability, corruption and failures.

Small executable fixtures isolate orchestration from numerical kernel tests.
Real GPU/CPU smoke runs use the same CLI with bounded search settings.
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('gpu_pipeline', ROOT / 'scripts/gpu_pipeline.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.config = self.base / 'config.ini'
        self.config.write_text('[system]\nthreads=1\n[pipeline]\nextract_top_n=2\nresopt_effort=0\nmsieve_ropt_count=2\n')
        self.p = driver.read_cado_blocks(ROOT / 'stage23_bench/data/c146/winners/sopt.poly')[0][1]
        # Different raw polynomials can share all of n, Y0 and Y1. Extraction must
        # retain the actual pair selected by exp_E, not the last matching rational.
        self.q = copy.deepcopy(self.p)
        self.q['c0'] += self.q['Y0']
        self.q['c1'] += self.q['Y1']
        self.raw = self.base / 'raw.ms'
        self.raw.write_text(driver.cado_poly_text(self.p) + '\n' + driver.cado_poly_text(self.q) + '\n' +
                            driver.cado_poly_text(self.p))
        self.executable = self.base / 'fake-tool'
        self.executable.write_text(f'''#!{sys.executable}
import sys
import os
import subprocess
import time
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'scripts')!r})
import gpu_pipeline as g
args = sys.argv[1:]
base = Path({str(self.base)!r})
with (base / 'calls').open('a') as out:
    out.write(Path(sys.argv[0]).name + ' ' + ' '.join(args) + '\\n')
def opt(flag): return args[args.index(flag) + 1]
if '-capabilities' in args:
    if (base / 'probe-fail').exists(): sys.exit(7)
    if not (base / 'legacy-tool').exists():
        print('ropt-pool-v1 partial-murphy-exit-3 ropt-batch-v1')
        if not (base / 'old-batch').exists(): print('ropt-batch-events-v1')
        if not (base / 'no-screen').exists():
            print('alpha-screen-v1 alpha-screen-exact-size-v1 alpha-screen-gpu-v1')
elif '-batch' in args:
    jobs = Path(opt('-batch')).read_text().splitlines()
    assert jobs.pop(0) == 'S23_ROPT_BATCH 1'
    event_fd = int(opt('-batch-events-fd'))
    def event(text):
        sys.stdout.flush()
        sys.stderr.flush()
        os.write(event_fd, (text + ' ' + str(os.lseek(1, 0, os.SEEK_CUR)) + '\\n').encode('ascii'))
    if (base / 'batch-barrier').exists() and '/breadth/' in opt('-batch'):
        Path(opt('-batch')).with_suffix('.started').touch()
        until = time.monotonic() + 10
        while len(list(Path(opt('-batch')).parent.glob('*.started'))) < 2:
            if time.monotonic() > until: sys.exit(8)
            time.sleep(.01)
    partial = False
    for i, job in enumerate(jobs, 1):
        print(f'# batch job {{i}}/{{len(jobs)}} begin', flush=True)
        event(f'begin {{i}} {{len(jobs)}}')
        if i == 2 and (base / 'batch-wait').exists(): time.sleep(60)
        if i == 2 and (base / 'batch-fail').exists():
            (base / 'batch-fail').unlink()
            event(f'end {{i}} 7 0.010000')
            print(f'# batch job {{i}} status 7, 0.010000 s', flush=True)
            sys.exit(7)
        p = subprocess.run([sys.argv[0], *job.split('\\t')])
        if not (base / 'batch-no-end').exists():
            event(f'end {{i}} {{p.returncode}} 0.010000')
            print(f'# batch job {{i}} status {{p.returncode}}, 0.010000 s', flush=True)
        if p.returncode not in (0, 3): sys.exit(p.returncode)
        partial |= p.returncode == 3
    sys.exit(3 if partial else 0)
elif '-inputpolys' in args:
    polys = list(g.raw_polys(opt('-inputpolys')))
    for i, p in enumerate(polys):
        print(f'### Input raw polynomial ({{i}}) ###')
        print('\\n'.join('# ' + line for line in g.cado_poly_text(p).splitlines()))
        p.update(lognorm=50., exp_E=10. + i, alpha=-2., proj=-1., rroots=1, skew=1000.)
        print(g.sopt_text(p, i))
elif '-search' in args:
    if (base / 'log-bytes').exists(): os.write(1, (base / 'log-bytes').read_bytes())
    if (base / 'fail-once').exists():
        (base / 'fail-once').unlink()
        Path(opt('-out')).write_text('incomplete output')
        sys.exit(7)
    p = g.read_cado_blocks(opt('-poly'))[0][1]
    p['skew'] = 2000. if '-murphy' in args else 1000.
    Path(opt('-out')).write_text('# fixture output\\n' + g.cado_poly_text(p))
    if '-pool-out' in args: Path(opt('-pool-out')).write_text('fixture candidate pool')
    if '-pool-in' in args: assert Path(opt('-pool-in')).read_text() == 'fixture candidate pool'
    if '-murphy' in args and (base / 'murphy-failure').exists():
        kind = (base / 'murphy-failure').read_text()
        if kind == 'empty': Path(opt('-out')).write_text('')
        elif kind == 'invalid': Path(opt('-out')).write_text('n: 7\\nc0: 1\\n')
        if kind != 'fatal':
            print('# best 3 by validated MurphyE: 1 optimized, 0 accepted, 0 limited, 1 failed, 0 skipped before MurphyE;')
            print('# wrote 1 polynomials to ' + opt('-out'))
        sys.exit(1 if kind in ('legacy', 'fatal') else 3)
else:
    for i, (_, p) in enumerate(g.read_cado_blocks(args[-1])):
        skew = p.get('skew', 1000.) if '-useskew' in args else 1500.
        value = skew / 1e10
        if '-lattice' in args:
            print(f'{{i}}\\t{{value}}\\t{{skew}}\\t{{value / 2}}\\t3\\t4')
        else:
            print(f'{{i}}\\t{{value}}\\t{{skew}}\\t0')
''')
        self.executable.chmod(0o755)
        self.patcher = patch.object(driver, 'cado_murphy_binary', return_value=str(self.executable))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def invoke(self, command, run, *extra):
        argv = [command, '--config', str(self.config), '--run-dir', str(run),
                '--sopt-binary', str(self.executable), '--ropt-binary', str(self.executable), *map(str, extra)]
        with contextlib.redirect_stdout(io.StringIO()):
            driver.main(argv)

    def params(self):
        return ['--params', 'job:29,30,14,10e6', '--lattice', '14,8192', '--qband', '10e6,20e6,2',
                '--workers', '1', '--depth-seeds', '1', '--murphy-seeds', '1']

    def calls(self):
        return (self.base / 'calls').read_text().splitlines()

    def test_split_equals_all_and_preserves_raw_pair(self):
        split, whole = self.base / 'split', self.base / 'all'
        self.invoke('preprocess', split, '--input', self.raw)
        self.assertEqual(driver.read_json(split / 'preprocess/raw.json')['unique_count'], 2)
        self.invoke('pipeline', split, *self.params())
        extracted = list(driver.raw_polys(split / 'pipeline/top_input.ms'))
        self.assertEqual(extracted[0]['c0'], self.p['c0'])
        self.assertEqual(extracted[1]['c0'], self.q['c0'])
        self.invoke('all', whole, '--input', self.raw, *self.params())
        self.assertEqual((split / 'results/best.poly').read_text(), (whole / 'results/best.poly').read_text())
        result = driver.read_json(split / 'results/summary.json')
        self.assertEqual(result['winner']['phase'], 'murphy')
        self.assertEqual(result['winner']['poly']['skew'], 2000.)
        self.assertTrue(any(row['phase'] == 'breadth' for row in result['finalists']))

    def test_resume_without_raw_or_binaries_and_report_without_scoring(self):
        run = self.base / 'run'
        self.invoke('all', run, '--input', self.raw, *self.params())
        before = self.calls()
        self.raw.unlink()
        self.executable.unlink()
        self.invoke('all', run, '--resume')
        self.invoke('pipeline', run, '--resume')
        self.invoke('report', run)
        self.assertEqual(before, self.calls())

    def test_failed_search_is_retried_and_corrupted_output_repaired(self):
        run = self.base / 'run'
        (self.base / 'fail-once').touch()
        with self.assertRaisesRegex(RuntimeError, 'exited 7'):
            self.invoke('pipeline', run, '--input', self.raw, *self.params())
        self.assertEqual(driver.read_json(run / 'status.json')['stage'], 'failed')
        self.invoke('pipeline', run)
        sopt_calls = [c for c in self.calls() if '-inputpolys' in c]
        victim = sorted((run / 'breadth').glob('*.poly'))[0]
        victim.write_text('truncated')
        self.invoke('pipeline', run)
        self.assertEqual(sopt_calls, [c for c in self.calls() if '-inputpolys' in c])
        self.assertTrue(driver.read_cado_blocks(victim))

    def test_snapshot_survives_live_input_changes_and_rejects_changed_settings(self):
        run = self.base / 'run'
        self.invoke('all', run, '--input', self.raw, *self.params())
        original = (run / 'preprocess/cado_sopt_unsorted.txt').read_bytes()
        self.raw.write_text(driver.cado_poly_text(self.q))
        before = self.calls()
        self.invoke('all', run, '--input', self.raw)
        self.assertEqual(before, self.calls())
        self.assertEqual(driver.digest(run / 'inputs/source.raw'),
                         driver.read_json(run / 'preprocess/source.json')['sha256'])
        with self.assertRaisesRegex(ValueError, 'new --run-dir'):
            self.invoke('all', run, '--input', self.base / 'different.ms')
        with self.assertRaisesRegex(ValueError, 'new --run-dir'):
            self.invoke('pipeline', run, '--budget', '4')
        with self.assertRaisesRegex(ValueError, 'new --run-dir'):
            self.invoke('preprocess', run, '--initial-unsorted', run / 'pipeline/resopt_output.txt')
        self.assertEqual(original, (run / 'preprocess/cado_sopt_unsorted.txt').read_bytes())

    def test_import_existing_sopt(self):
        source, dest = self.base / 'source', self.base / 'dest'
        self.invoke('preprocess', source, '--input', self.raw)
        before = len([c for c in self.calls() if '-inputpolys' in c])
        self.invoke('pipeline', dest, '--initial-unsorted', source / 'preprocess/cado_sopt_unsorted.txt', *self.params())
        after = len([c for c in self.calls() if '-inputpolys' in c])
        self.assertEqual(after - before, 1)  # only the higher-effort pass

    def test_invalid_raw_and_incomplete_sopt(self):
        bad = self.base / 'bad.ms'
        for text in ('n: 7\nc0: 1\n', driver.cado_poly_text(dict(self.p, c6=1)),
                     driver.cado_poly_text(dict(self.p, c0=self.p['c0'] + 1))):
            bad.write_text(text)
            with self.assertRaises(ValueError):
                list(driver.raw_polys(bad))
        src = self.base / 'source'
        self.invoke('preprocess', src, '--input', self.raw)
        bad.write_text((src / 'preprocess/cado_sopt_unsorted.txt').read_text() +
                       '\n### Input raw polynomial (99) ###\n# n: 7\n### Size-optimized polynomial (99) ###\n')
        with self.assertRaisesRegex(ValueError, 'incomplete sopt'):
            self.invoke('preprocess', self.base / 'dest', '--initial-unsorted', bad)

    def test_missing_bounds_fail_before_preprocessing_and_lock(self):
        run = self.base / 'run'
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke('all', run, '--input', self.raw)
        self.assertFalse(run.exists())
        run.mkdir()
        with (run / '.lock').open('a') as lock:
            driver.fcntl.flock(lock, driver.fcntl.LOCK_EX | driver.fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, 'another pipeline'):
                self.invoke('preprocess', run, '--input', self.raw)

    def test_wrapper_dispatches_gpu_help_and_cpu_help(self):
        for options in (['--gpu', 'pipeline', '--help'], ['pipeline', '--help']):
            result = subprocess.run([str(ROOT / 'nfs_optimize.sh'), *options], capture_output=True, text=True,
                                    env=dict(os.environ, NFS_CONFIG_FILE=str(self.base / 'missing.ini')))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--run-dir' if '--gpu' in options else 'Unified Interface', result.stdout)

    def test_basin_options_reach_all_search_phases_and_resume(self):
        run = self.base / 'run'
        self.invoke('all', run, '--input', self.raw, *self.params(),
                    '--u-scan', 256, '--u-basins', 3, '--u-radius', 64)
        before = self.calls()
        calls = [c for c in before if '-search' in c]
        self.assertGreaterEqual(len(calls), 4)  # breadth, depth and accurate MurphyE
        self.assertTrue(all('-u-scan 256 -u-basins 3 -u-radius 64' in c for c in calls if '-pool-in' not in c))
        self.invoke('pipeline', run)
        self.assertEqual(self.calls(), before)
        self.assertEqual(driver.read_json(run / 'settings.json')['u_scan'], 256)
        with self.assertRaisesRegex(ValueError, 'new --run-dir'):
            self.invoke('pipeline', run, '--u-scan', 0)

    def test_basin_work_limits_fail_before_preprocessing(self):
        run = self.base / 'run'
        for options in (['--u-scan', -1], ['--u-scan', 1], ['--u-basins', 17],
                        ['--u-scan', 256, '--u-basins', 16, '--u-radius', 1024]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke('all', run, '--input', self.raw, *self.params(), *options)
            self.assertFalse(run.exists())

    def test_residue_class_reaches_search_phases_and_resume(self):
        run = self.base / 'run'
        self.invoke('all', run, '--input', self.raw, *self.params(), '--residue', '6,-1,8')
        before = self.calls()
        calls = [c for c in before if '-search' in c]
        self.assertGreaterEqual(len(calls), 4)
        self.assertTrue(all('-residue 6 5 2' in c for c in calls if '-pool-in' not in c))
        self.invoke('pipeline', run)
        self.assertEqual(before, self.calls())
        self.assertEqual(driver.read_json(run / 'settings.json')['residue'], '6,5,2')
        with self.assertRaisesRegex(ValueError, 'new --run-dir'):
            self.invoke('pipeline', run, '--residue', '6,0,0')

    def test_invalid_residue_fails_before_preprocessing(self):
        run = self.base / 'run'
        for options in (['--residue', '0,1,2'], ['--residue', '65537,1,2'],
                        ['--residue', '6,1'], ['--residue', '6,1.5,0'],
                        ['--residue', f'6,{2**63},0'], ['--residue', '6,1,2', '--u-scan', 256]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke('all', run, '--input', self.raw, *self.params(), *options)
            self.assertFalse(run.exists())

    def test_automatic_classes_reach_split_and_all_searches_and_resume(self):
        for split in (False, True):
            with self.subTest(split=split):
                run = self.base / str(split)
                if split:
                    self.invoke('preprocess', run, '--input', self.raw)
                    self.invoke('pipeline', run, *self.params(), '--classes', ' 2520,16 ')
                else:
                    self.invoke('all', run, '--input', self.raw, *self.params(), '--classes', '2520,16')
                calls = [c for c in self.calls() if '-search' in c]
                self.assertTrue(calls)
                self.assertTrue(all('-classes 2520 16 -class-explore 4' in c for c in calls if '-pool-in' not in c))
                settings = driver.read_json(run / 'settings.json')
                self.assertEqual(settings['classes'], '2520,16')
                self.assertEqual(settings['class_explore'], 4)
                before = self.calls()
                self.invoke('pipeline', run)
                self.assertEqual(before, self.calls())
                with self.assertRaisesRegex(ValueError, 'new --run-dir'):
                    self.invoke('pipeline', run, '--class-explore', 2)

    def test_invalid_automatic_classes_fail_before_preprocessing(self):
        run = self.base / 'run'
        for options in (['--classes', '1,2'], ['--classes', '4097,16'], ['--classes', '2,5'],
                        ['--classes', '2520,65'], ['--classes', '6,1.5'], ['--classes', '6'],
                        ['--classes', '6,4', '--residue', '6,1,2'],
                        ['--classes', '6,4', '--u-scan', 256],
                        ['--classes', '6,4', '--class-explore', 5],
                        ['--classes', '2520,16', '--umax', 2000000],
                        ['--classes', '6,4', '--class-explore', -2], ['--class-explore', 0]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke('all', run, '--input', self.raw, *self.params(), *options)
            self.assertFalse(run.exists())

    def test_murphy_partial_results_checkpoint_and_do_not_resieve(self):
        for kind in ('partial', 'legacy'):
            with self.subTest(kind=kind):
                (self.base / 'murphy-failure').write_text(kind)
                run = self.base / kind
                self.invoke('all', run, '--input', self.raw, *self.params())
                before = self.calls()
                self.invoke('all', run)
                self.assertEqual(before, self.calls())
                calls = [c for c in before if '-murphy ' in c]
                self.assertTrue(calls)
                self.assertTrue(all('-pool-in ' in c and '/depth/' in c for c in calls))
                self.assertTrue((run / 'checkpoints/murphy/s0001.json').is_file())
                self.assertEqual(len(driver.read_json(run / 'results/summary.json')['partial_murphy_searches']), 1)

    def test_murphy_rejects_empty_invalid_and_unrelated_failure_outputs(self):
        for kind in ('empty', 'invalid', 'fatal'):
            with self.subTest(kind=kind):
                (self.base / 'murphy-failure').write_text(kind)
                run = self.base / kind
                with self.assertRaises((ValueError, RuntimeError)):
                    self.invoke('all', run, '--input', self.raw, *self.params())
                self.assertFalse((run / 'checkpoints/murphy/s0001.json').exists())

    def test_execution_settings_and_source_hash_changes_reuse_checkpoints(self):
        run = self.base / 'run'
        self.invoke('all', run, '--input', self.raw, *self.params())
        # Simulate legacy global-hash checkpoints, including a different source revision.
        for marker in (run / 'checkpoints').rglob('*.json'):
            data = driver.read_json(marker)
            data['signature'].pop('schema', None)
            data['signature']['version'] = {'older/driver.py': 'old hash'}
            driver.write_json(marker, data)
        before = self.calls()
        self.invoke('all', run, '--device', 1, '--threads', 3, '--workers', 2, '--launch', 31, '--slice', 7)
        self.invoke('pipeline', run)
        self.assertEqual(before, self.calls())
        self.assertEqual(driver.read_json(run / 'execution.json')['threads'], 3)
        # A missing completed output must be repaired using the current execution options.
        (run / 'breadth/s0002.poly').write_text('broken')
        self.invoke('pipeline', run)
        added = self.calls()[len(before):]
        self.assertEqual(len(added), 1)
        self.assertIn('-t 3 -dev 1 -launch 31', added[0])

    def test_unique_controls_are_selected_before_finalist_limit(self):
        run = self.base / 'run'
        self.invoke('all', run, '--input', self.raw, *self.params(), '--finalists', 2)
        rows = driver.read_json(run / 'scores/final.json')
        self.assertEqual(sum(r['phase'] != 'murphy' for r in rows), 2)
        self.assertEqual(len(rows), 3)

    def test_batch_matches_single_and_execution_switch_reuses_checkpoints(self):
        single, batch = self.base / 'single', self.base / 'batch with spaces'
        self.invoke('all', single, '--input', self.raw, *self.params())
        self.invoke('all', batch, '--input', self.raw, *self.params(), '--ropt-batch-size', 16)
        self.assertEqual((single / 'results/best.poly').read_bytes(), (batch / 'results/best.poly').read_bytes())
        self.assertEqual(len(list((batch / 'batches/breadth').glob('*.tsv'))), 1)
        before = self.calls()
        self.invoke('pipeline', batch, '--ropt-batch-size', 1)
        self.assertEqual(before, self.calls())

    def test_batch_failure_preserves_completed_seed_and_repairs_only_pending(self):
        run = self.base / 'batch'
        (self.base / 'batch-fail').touch()
        with self.assertRaisesRegex(RuntimeError, 'exited 7'):
            self.invoke('all', run, '--input', self.raw, *self.params(), '--ropt-batch-size', 16)
        first = run / 'checkpoints/breadth/s0001.json'
        self.assertTrue(first.exists())
        self.assertFalse((run / 'checkpoints/breadth/s0002.json').exists())
        first_bytes = first.read_bytes()
        self.invoke('pipeline', run)
        self.assertEqual(first_bytes, first.read_bytes())
        manifests = sorted((run / 'batches/breadth').glob('*.tsv'))
        self.assertEqual(len(manifests), 2)
        self.assertIn('s0001', manifests[0].read_text())
        manifest = manifests[-1].read_text()
        self.assertNotIn('s0001', manifest)
        self.assertIn('s0002', manifest)
        (run / 'breadth/s0002.poly').write_text('corrupt')
        self.invoke('pipeline', run)
        self.assertEqual(first_bytes, first.read_bytes())
        self.assertTrue(driver.read_cado_blocks(run / 'breadth/s0002.poly'))

    def test_batch_interruption_commits_completed_seed_while_next_job_runs(self):
        run = self.base / 'batch'
        (self.base / 'batch-wait').touch()
        record = driver.Run.record_step

        def stop_after_first(workflow, name, *args):
            record(workflow, name, *args)
            if name == 'breadth/s0001':
                workflow.stop.set()

        with patch.object(driver.Run, 'record_step', stop_after_first):
            with self.assertRaises(InterruptedError):
                self.invoke('all', run, '--input', self.raw, *self.params(), '--ropt-batch-size', 16)
        self.assertTrue((run / 'checkpoints/breadth/s0001.json').exists())
        self.assertFalse((run / 'checkpoints/breadth/s0002.json').exists())
        (self.base / 'batch-wait').unlink()
        self.invoke('pipeline', run)
        self.assertEqual(driver.read_json(run / 'status.json')['stage'], 'complete')

    def test_batch_partial_murphy_and_missing_status(self):
        run = self.base / 'partial'
        (self.base / 'murphy-failure').write_text('partial')
        self.invoke('all', run, '--input', self.raw, *self.params(), '--ropt-batch-size', 16)
        report = driver.read_json(run / 'results/summary.json')
        self.assertEqual(len(report['partial_murphy_searches']), 1)
        self.assertTrue((run / 'checkpoints/murphy/s0001.json').exists())
        (self.base / 'murphy-failure').write_text('invalid')
        invalid = self.base / 'invalid-partial'
        with self.assertRaisesRegex(ValueError, 'complete degree-5'):
            self.invoke('all', invalid, '--input', self.raw, *self.params(), '--ropt-batch-size', 16)
        self.assertFalse((invalid / 'checkpoints/murphy/s0001.json').exists())
        (self.base / 'batch-no-end').touch()
        broken = self.base / 'missing-status'
        with self.assertRaisesRegex(RuntimeError, 'batch job ordering'):
            self.invoke('all', broken, '--input', self.raw, *self.params(), '--ropt-batch-size', 16)
        self.assertFalse((broken / 'checkpoints/breadth/s0001.json').exists())

    def test_batch_rejects_unsupported_binary(self):
        (self.base / 'legacy-tool').touch()
        with self.assertRaisesRegex(ValueError, 'ropt-batch-v1'):
            self.invoke('all', self.base / 'batch', '--input', self.raw, *self.params(), '--ropt-batch-size', 16)
        self.assertFalse(any('-inputpolys' in c for c in self.calls()))
        self.assertFalse((self.base / 'batch/execution.json').exists())

    def test_rejected_batch_setting_does_not_poison_resume(self):
        run = self.base / 'old-batch-run'
        (self.base / 'old-batch').touch()
        self.invoke('all', run, '--input', self.raw, *self.params())
        before, execution = self.calls(), (run / 'execution.json').read_bytes()
        with self.assertRaisesRegex(ValueError, '--ropt-batch-size 1'):
            self.invoke('pipeline', run, '--ropt-batch-size', 16)
        self.assertEqual((run / 'execution.json').read_bytes(), execution)
        self.invoke('pipeline', run)
        self.assertEqual(before, self.calls())
        # Also allow recovery from execution.json poisoned by the older driver.
        saved = driver.read_json(run / 'execution.json')
        saved['ropt_batch_size'] = 16
        driver.write_json(run / 'execution.json', saved)
        self.invoke('pipeline', run, '--ropt-batch-size', 1)
        self.assertEqual(driver.read_json(run / 'execution.json')['ropt_batch_size'], 1)
        with self.assertRaisesRegex(ValueError, 'new --run-dir'):
            self.invoke('pipeline', run, '--budget', 4, '--workers', 2)
        self.assertEqual((run / 'execution.json').read_bytes(), execution)

    def test_batch_small_phase_uses_both_workers(self):
        (self.base / 'batch-barrier').touch()
        run = self.base / 'two-workers'
        self.invoke('all', run, '--input', self.raw, *self.params(), '--workers', 2, '--ropt-batch-size', 16)
        manifests = list((run / 'batches/breadth').glob('*.tsv'))
        self.assertEqual(len(manifests), 2)
        self.assertTrue(all(len(path.read_bytes().splitlines()) == 2 for path in manifests))

    def test_batch_logs_preserve_bytes_without_parsing_diagnostics(self):
        payload = b'non-UTF8: \xff\r\n# batch job 1/2 begin\n# batch job 1 status 0, 0.000001 s\nno newline'
        (self.base / 'log-bytes').write_bytes(payload)
        for size in (1, 16):
            run = self.base / f'bytes-{size}'
            self.invoke('all', run, '--input', self.raw, *self.params(), '--ropt-batch-size', size)
            for phase in ('breadth', 'depth', 'murphy'):
                for log in (run / 'logs' / phase).glob('*.log'):
                    self.assertEqual(log.read_bytes(), payload)

    def test_progress_reports_reused_seeds_and_completed_jobs_inside_batch(self):
        run = self.base / 'progress'
        states = []
        state, record = driver.Run.state, driver.Run.record_step

        def capture(workflow, phase, **extra):
            state(workflow, phase, **extra)
            states.append(driver.read_json(workflow.path / 'status.json'))

        def stop_after_first(workflow, name, *args):
            record(workflow, name, *args)
            if name == 'breadth/s0001':
                workflow.stop.set()

        (self.base / 'batch-wait').touch()
        with patch.object(driver.Run, 'state', capture), patch.object(driver.Run, 'record_step', stop_after_first):
            with self.assertRaises(InterruptedError):
                self.invoke('all', run, '--input', self.raw, *self.params(), '--ropt-batch-size', 16)
        self.assertEqual([s['completed'] for s in states if s['stage'] == 'breadth'], [0, 1])
        (self.base / 'batch-wait').unlink()
        self.invoke('pipeline', run)
        before = self.calls()
        states.clear()
        with patch.object(driver.Run, 'state', capture):
            self.invoke('pipeline', run)
        self.assertEqual(before, self.calls())
        self.assertEqual([s['completed'] for s in states if s['stage'] == 'breadth'], [0, 1, 2])

    def test_capability_probe_failure_preserves_legacy_pool_cache_and_retries(self):
        run = driver.Run(self.base / 'probe')
        marker = run.path / 'bin/ropt_capabilities.json'
        original = dict(sha256=driver.digest(self.executable), pool_v1=True)
        driver.write_json(marker, original)
        (self.base / 'probe-fail').touch()
        self.assertTrue(run.pool_capable(self.executable))
        with self.assertRaisesRegex(RuntimeError, 'cache unchanged'):
            run.ropt_capabilities(self.executable)
        self.assertEqual(driver.read_json(marker), original)
        self.assertTrue(run.pool_capable(self.executable))
        (self.base / 'probe-fail').unlink()
        self.assertIn('ropt-batch-events-v1', run.ropt_capabilities(self.executable))
        self.assertTrue(driver.read_json(marker)['pool_v1'])
        # An ambiguous empty cache written by the old driver must be re-probed.
        driver.write_json(marker, dict(original, pool_v1=False, capabilities=[]))
        self.assertTrue(run.pool_capable(self.executable))

    def test_failed_uncached_capability_probe_is_not_cached(self):
        run = driver.Run(self.base / 'probe')
        (self.base / 'probe-fail').touch()
        with self.assertRaisesRegex(RuntimeError, 'probe exited 7'):
            run.ropt_capabilities(self.executable)
        self.assertFalse((run.path / 'bin/ropt_capabilities.json').exists())
        with patch.object(driver.subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 30)):
            with self.assertRaisesRegex(RuntimeError, 'cache unchanged'):
                run.ropt_capabilities(self.executable)
        (self.base / 'probe-fail').unlink()
        self.assertTrue(run.pool_capable(self.executable))

    def test_legacy_pool_cache_resumes_without_reprobing_or_replaying(self):
        run = self.base / 'legacy-cache'
        self.invoke('all', run, '--input', self.raw, *self.params())
        driver.write_json(run / 'bin/ropt_capabilities.json',
                          dict(sha256=driver.digest(run / 'bin/s23_ropt'), pool_v1=True))
        before = self.calls()
        checkpoints = {str(p): p.read_bytes() for p in (run / 'checkpoints').rglob('*.json')}
        (self.base / 'probe-fail').touch()
        self.invoke('all', run)
        self.assertEqual(before, self.calls())
        self.assertEqual(checkpoints, {str(p): p.read_bytes() for p in (run / 'checkpoints').rglob('*.json')})

    def test_gpu_screen_forwarding_preserves_full_pool_murphy_and_resume(self):
        run = self.base / 'screen'
        self.invoke('all', run, '--input', self.raw, *self.params(), '--ropt-batch-size', 16,
                    '--rerank', 512, '--alpha-screen', '4096,500', '--alpha-screen-exact-size',
                    '--alpha-screen-gpu', '--alpha-screen-check')
        calls = [c for c in self.calls() if '-search' in c]
        ordinary = [c for c in calls if '-murphy' not in c]
        final = [c for c in calls if '-murphy' in c]
        self.assertTrue(ordinary and final)
        self.assertTrue(all('-alpha-screen 4096 500 -alpha-screen-exact-size -alpha-screen-gpu -alpha-screen-check' in c
                            and '-rerank 512 ' in c for c in ordinary))
        self.assertTrue(all('-alpha-screen' not in c and '-rerank 4096 ' in c and '-pool-in' in c for c in final))
        before = self.calls()
        self.invoke('pipeline', run)
        self.assertEqual(before, self.calls())
        with self.assertRaisesRegex(ValueError, 'new --run-dir'):
            self.invoke('pipeline', run, '--no-alpha-screen-check')

    def test_screen_validation_and_unsupported_binary(self):
        for flags in (['--alpha-screen', '4,500'], ['--alpha-screen', '4096,2001'],
                      ['--alpha-screen', 'bad'], ['--alpha-screen', '16777217,500'],
                      ['--alpha-screen-exact-size'], ['--alpha-screen', '4096,500', '--alpha-screen-gpu'],
                      ['--alpha-screen', '4096,500', '--alpha-screen-check']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke('all', self.base / 'invalid', '--input', self.raw, *self.params(), *flags)
            self.assertFalse((self.base / 'invalid').exists())
        (self.base / 'no-screen').touch()
        with self.assertRaisesRegex(ValueError, 'lacks requested screening'):
            self.invoke('all', self.base / 'unsupported', '--input', self.raw, *self.params(),
                        '--alpha-screen', '4096,500')
        self.assertFalse(any('-inputpolys' in c for c in self.calls()))

    def test_old_checkpoint_ignores_absent_neutral_screen_options(self):
        run = self.base / 'old'
        self.invoke('all', run, '--input', self.raw, *self.params())
        settings = driver.read_json(run / 'settings.json')
        for name in driver.SCREEN_OPTIONS:
            settings.pop(name)
        driver.write_json(run / 'settings.json', settings)
        marker = run / 'checkpoints/pipeline_settings.json'
        record = driver.read_json(marker)
        record['outputs']['settings.json'] = driver.digest(run / 'settings.json')
        driver.write_json(marker, record)
        self.config.write_text(self.config.read_text() + '[gpu]\nalpha_screen=4096,500\n'
                               'alpha_screen_exact_size=true\nalpha_screen_gpu=true\nalpha_screen_check=true\n')
        before = self.calls()
        self.invoke('pipeline', run)
        self.assertEqual(before, self.calls())
        with self.assertRaisesRegex(ValueError, 'new --run-dir'):
            self.invoke('pipeline', run, '--alpha-screen', '4096,500')

    def test_blank_screen_config_is_false_and_help_and_report_work(self):
        self.config.write_text(self.config.read_text() + '[gpu]\n' +
                               ''.join(key + '=\n' for key in driver.SCREEN_OPTIONS))
        run = self.base / 'blanks'
        self.invoke('all', run, '--input', self.raw, *self.params())
        self.invoke('report', run)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
            driver.parser(['--config', str(self.config), '--help'])
        self.assertEqual(result.exception.code, 0)
        settings = driver.read_json(run / 'settings.json')
        for key, option in driver.SCREEN_OPTIONS.items():
            self.assertEqual(settings[key], option['default'])

    def test_sqlite_uri_reserved_characters_and_frozen_input_repair(self):
        run = self.base / 'run?#% with spaces'
        self.invoke('all', run, '--input', self.raw, *self.params())
        before = self.calls()
        self.raw.unlink()
        (run / 'preprocess/msieve.dat.deduped.ms').write_text('broken')
        self.invoke('all', run)
        self.assertEqual(before, self.calls())

    def test_snapshot_hash_describes_bytes_consumed_after_source_mutation(self):
        run = self.base / 'run'
        real_deduplicate = driver.deduplicate
        original = self.raw.read_bytes()
        def mutate_then_read(source, *args):
            self.assertNotEqual(source, self.raw)
            self.raw.write_text(driver.cado_poly_text(self.q))
            return real_deduplicate(source, *args)
        with patch.object(driver, 'deduplicate', side_effect=mutate_then_read):
            self.invoke('preprocess', run, '--input', self.raw)
        self.assertEqual((run / 'inputs/source.raw').read_bytes(), original)
        self.assertEqual(driver.read_json(run / 'preprocess/raw.json')['unique_count'], 2)

    def test_content_residue_intersections_match_parent_coordinates(self):
        for m in (1, 2, 6, 12):
            for d in (1, 2, 3, 4, 6):
                seed = dict(content=d, u=-2, v=3)
                for ru in range(m):
                    residue = driver.seed_residue(f'{m},{ru},1', seed)
                    expected = {(u, v) for u in range(-12, 13) for v in range(-12, 13)
                                if (-2 + d * u) % m == ru and (3 + d * v) % m == 1 % m}
                    actual = set() if residue is None else {
                        (u, v) for u in range(-12, 13) for v in range(-12, 13)
                        if u % residue[0] == residue[1] and v % residue[0] == residue[2]}
                    self.assertEqual(expected, actual)

    def test_content_seed_classes_are_mapped_or_skipped_in_searches(self):
        parent = dict(self.p, **{f'c{i}': 2 * self.p[f'c{i}'] for i in range(6)})
        self.raw.write_text(driver.cado_poly_text(parent))
        for residue, intersects in (('6,2,4', True), ('6,1,4', False)):
            run = self.base / residue
            before = len(self.calls()) if (self.base / 'calls').exists() else 0
            with patch.object(driver, 'content_seeds', return_value=[(2, 0, 0, self.p)]):
                self.invoke('all', run, '--input', self.raw, *self.params(), '--residue', residue)
            derived = [c for c in self.calls()[before:] if '-search' in c and '_d2.poly' in c]
            self.assertEqual(bool(derived), intersects)
            if intersects:
                self.assertTrue(all('-residue 3 1 2' in c for c in derived if '-pool-in' not in c))

    def test_legacy_binary_fallback(self):
        (self.base / 'legacy-tool').touch()
        run = self.base / 'run'
        self.invoke('all', run, '--input', self.raw, *self.params())
        calls = self.calls()
        self.assertFalse(any('-pool-in' in c or '-pool-out' in c for c in calls))
        self.invoke('all', run)
        self.assertEqual(calls, self.calls())

    def test_cpu_wrapper_all_routes_options_before_running(self):
        root = self.base / 'wrapper'
        (root / 'scripts').mkdir(parents=True)
        shutil.copy2(ROOT / 'nfs_optimize.sh', root / 'nfs_optimize.sh')
        (root / 'polyselect').mkdir()
        (root / 'polyselect/sopt').touch()
        config = root / 'nfs_config.ini'
        config.write_text(f'[paths]\ncado_build_dir={root}\nmsieve_binary={self.executable}\nutils_dir={root}\n')
        for name in ('dedupe_and_sopt.sh', 'full_optimization_pipeline.sh'):
            stub = root / 'scripts' / name
            stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "${0}.args"\n')
            stub.chmod(0o755)
        cmd = [str(root / 'nfs_optimize.sh'), 'all']
        for invalid in (['--unknown'], ['--initial-unsorted', 'import.txt'], ['--input']):
            p = subprocess.run(cmd + invalid, capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertFalse(list((root / 'scripts').glob('*.args')))
        p = subprocess.run(cmd + ['--input', 'chosen raw.ms', '--threads', '3', '--extract', '2'],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        pre = (root / 'scripts/dedupe_and_sopt.sh.args').read_text().splitlines()
        pipeline = (root / 'scripts/full_optimization_pipeline.sh.args').read_text().splitlines()
        self.assertEqual(pre[-4:], ['--input', 'chosen raw.ms', '--threads', '3'])
        self.assertNotIn('--input', pipeline)
        self.assertEqual(pipeline[-4:], ['--threads', '3', '--extract', '2'])


if __name__ == '__main__':
    unittest.main()
