#!/usr/bin/env python3
"""Multi-seed isolation, replay, preflight and persistent CUDA scratch."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_ropt_class import BIN, ROOT, CONTROL, read_cado_blocks, resultant

GPU = 'alpha-screen-gpu-v1' in subprocess.check_output([str(BIN), '-capabilities'], text=True).split()
CONTENT = ROOT / 'stage23_bench/data/c168/content/s0051.poly'


def manifest(path, jobs):
    path.write_text('S23_ROPT_BATCH 1\n' + ''.join('\t'.join(map(str, row)) + '\n' for row in jobs))


class RoptBatch(unittest.TestCase):
    common = ['-search', '-budget', .5, '-maxcells', 100000, '-maxlines', 16,
              '-umax', 16, '-seg', 12, '-k', 16, '-aw', 1.3, '-rerank', 128,
              '-refine', 8, '-top', 4, '-t', 2]

    def invoke(self, args, status=0):
        p = subprocess.run([str(BIN), *map(str, args)], capture_output=True, text=True, timeout=180)
        self.assertEqual(p.returncode, status, p.stdout + p.stderr)
        return p

    def test_live_jobs_reset_options_and_match_independent_processes(self):
        with tempfile.TemporaryDirectory(prefix='ropt batch ') as tmp:
            base = Path(tmp)
            jobs, expected = [], []
            for j, (source, extra) in enumerate([(CONTROL, ['-exactskew', '-oldsearch', '-noproj']),
                                                (CONTENT, []), (CONTROL, ['-check', 16]), (CONTROL, [])]):
                out, pool = base / f'job {j}.poly', base / f'job {j}.pool'
                common = ['-poly', source, '-cpu', *self.common, *extra]
                self.invoke([*common, '-out', out, '-pool-out', pool])
                expected.append((out.read_bytes(), pool.read_bytes()))
                jobs.append([*common, '-out', base / f'batch {j}.poly', '-pool-out', base / f'batch {j}.pool'])
            batch = base / 'jobs.tsv'
            manifest(batch, jobs)
            text = self.invoke(['-batch', batch]).stdout
            self.assertIn('4 jobs complete, 0 partial', text)
            self.assertIn('vs brute force: 0 differ', text)
            for j, (out, pool) in enumerate(expected):
                self.assertEqual((base / f'batch {j}.poly').read_bytes(), out)
                self.assertEqual((base / f'batch {j}.pool').read_bytes(), pool)

    def test_preflight_rejects_bad_options_and_aliases_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            seed = base / 'seed.poly'
            seed.write_bytes(CONTROL.read_bytes())
            alias = base / 'hardlink.poly'
            os.link(seed, alias)
            sym = base / 'symlink.poly'
            sym.symlink_to(seed)
            out = base / 'out.poly'
            first = ['-poly', seed, '-cpu', *self.common, '-out', out]
            batch = base / 'jobs.tsv'
            for bad in ([*first, '-not-an-option'],
                        first,
                        ['-poly', seed, '-cpu', *self.common, '-out', alias],
                        ['-poly', seed, '-cpu', *self.common, '-out', sym],
                        ['-poly', seed, '-cpu', *self.common, '-out', batch],
                        ['-batch', batch]):
                manifest(batch, [first, bad])
                p = self.invoke(['-batch', batch], 2)
                self.assertIn('no jobs started', p.stderr)
                self.assertFalse(out.exists())
                self.assertEqual(seed.read_bytes(), CONTROL.read_bytes())

    def test_runtime_failure_keeps_completed_jobs_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            jobs = [['-poly', source, '-cpu', *self.common, '-out', base / f'{j}.poly']
                    for j, source in enumerate([CONTROL, base / 'missing.poly', CONTENT])]
            batch = base / 'jobs.tsv'
            manifest(batch, jobs)
            p = self.invoke(['-batch', batch], 2)
            self.assertIn('batch stopped at job 2', p.stderr)
            self.assertTrue((base / '0.poly').is_file())
            self.assertFalse((base / '1.poly').exists())
            self.assertFalse((base / '2.poly').exists())

    def test_manifest_format_and_exclusive_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            batch = base / 'jobs.tsv'
            for text in ('', 'S23_ROPT_BATCH 2\n', 'S23_ROPT_BATCH 1\n',
                         'S23_ROPT_BATCH 1\n-poly\t\t-search\n',
                         'S23_ROPT_BATCH 1\n-poly\tbad\0name\n'):
                batch.write_text(text)
                self.invoke(['-batch', batch], 2)
            manifest(batch, [['-poly', CONTROL, '-cpu', *self.common, '-out', base / 'out.poly']])
            batch.write_bytes(batch.read_bytes().replace(b'\n', b'\r\n'))
            self.invoke(['-batch', batch])
            self.invoke(['-batch', batch, '-t', 2], 2)

    def test_event_offsets_preserve_opaque_output_and_failure_boundaries(self):
        # Exercise the actual C++ batch transport with adversarial diagnostics,
        # independent of which logging conventions today's numerical job uses.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source, binary = base / 'events.cpp', base / 'events'
            source.write_text(r'''
#include "ropt_batch.h"
int main(int argc, char **argv) {
    return ropt_batch(argv[1], argv[0],
        [](int, char **, RoptJobPaths &) { return 0; },
        [](int, char **args) {
            const char body[] = "\xff\r\n# batch job 2/3 begin\n# batch job 1 status 0, 0.100000 s\nunterminated";
            fwrite(body, 1, sizeof(body)-1, stdout);
            fflush(stdout);
            fputs("stderr fragment", stderr);
            return std::stoi(args[1]);
        }, std::stoi(argv[2]));
}
''')
            subprocess.run(['g++', '-std=c++17', '-I', str(ROOT / 'stage23_gpu/tools'),
                            str(source), '-o', str(binary)], check=True, capture_output=True)
            payload = (b'\xff\r\n# batch job 2/3 begin\n# batch job 1 status 0, 0.100000 s\n'
                       b'unterminatedstderr fragment')
            batch = base / 'jobs.tsv'
            for codes, expected_rc, expected_jobs in (([0, 3, 0], 3, 3), ([0, 7, 0], 7, 2)):
                manifest(batch, [[code] for code in codes])
                log_path, events_path = base / 'batch.log', base / 'batch.events'
                with log_path.open('wb') as log, events_path.open('wb') as events:
                    result = subprocess.run([str(binary), str(batch), str(events.fileno())],
                                            stdout=log, stderr=log, pass_fds=(events.fileno(),), timeout=10)
                self.assertEqual(result.returncode, expected_rc)
                records = [row.split() for row in events_path.read_text().splitlines()]
                self.assertEqual(len(records), 2 * expected_jobs)
                raw = log_path.read_bytes()
                for j in range(expected_jobs):
                    begin, end = records[2*j:2*j+2]
                    self.assertEqual(begin[:3], ['begin', str(j+1), '3'])
                    self.assertEqual(end[:3], ['end', str(j+1), str(codes[j])])
                    self.assertEqual(raw[int(begin[3]):int(end[4])], payload)

    def test_events_require_valid_descriptor_and_shared_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            batch = base / 'jobs.tsv'
            out = base / 'out.poly'
            manifest(batch, [['-poly', CONTROL, '-cpu', *self.common, '-out', out]])
            for fd in ('2', '-1', 'abc', '2147483648', '999999'):
                self.invoke(['-batch', batch, '-batch-events-fd', fd], 2)
                self.assertFalse(out.exists())
            # A valid event descriptor still requires a common regular log.
            with (base / 'events').open('wb') as events:
                result = subprocess.run([str(BIN), '-batch', str(batch), '-batch-events-fd', str(events.fileno())],
                                        capture_output=True, pass_fds=(events.fileno(),), timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(out.exists())

    @unittest.skipUnless(GPU, 'select CUDA executable with S23_ROPT_BIN')
    def test_live_gpu_batch_matches_separate_gpu_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            jobs, expected = [], []
            for j, source in enumerate([CONTENT, CONTROL]):
                common = ['-poly', source, *self.common, '-check', 16,
                          '-alpha-screen', 512, 500, '-alpha-screen-exact-size', '-alpha-screen-gpu']
                out = base / f'fresh{j}.poly'
                self.invoke([*common, '-out', out])
                expected.append(out.read_bytes())
                jobs.append([*common, '-out', base / f'batch{j}.poly'])
            batch = base / 'jobs.tsv'
            manifest(batch, jobs)
            p = self.invoke(['-batch', batch])
            self.assertIn('device setups 1, buffer allocations 1, prime uploads 1', p.stdout)
            self.assertEqual(p.stdout.count('vs brute force: 0 differ'), 2)
            for j, content in enumerate(expected):
                self.assertEqual((base / f'batch{j}.poly').read_bytes(), content)

    @unittest.skipUnless(GPU, 'select CUDA executable with S23_ROPT_BIN')
    def test_gpu_reuse_bound_changes_and_invalid_device_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pools = []
            for j, (source, count) in enumerate([(CONTROL, 128), (CONTENT, 512)]):
                pool = base / f'{j}.pool'
                self.invoke(['-poly', source, '-cpu', *self.common, '-rerank', count, '-pool-out', pool])
                pools.append((source, pool))
            jobs, expected = [], []
            # Grow scratch, reuse it across seeds, update prime tables, fail one
            # device setup, and recover the valid device in the same process.
            configs = [(0, 128, 500, 0), (1, 512, 500, 0), (0, 128, 500, 0),
                       (1, 512, 2000, 0), (1, 512, 500, 2147483647), (0, 128, 500, 0)]
            for j, (index, count, bound, device) in enumerate(configs):
                source, pool = pools[index]
                common = ['-poly', source, '-search', '-pool-in', pool, '-aw', 1.3,
                          '-rerank', 8, '-refine', 8, '-top', 4, '-t', 2,
                          '-alpha-screen', count, bound, '-alpha-screen-exact-size']
                out = base / f'cpu{j}.poly'
                self.invoke([*common, '-cpu', '-out', out])
                expected.append(out.read_bytes())
                jobs.append([*common, '-alpha-screen-gpu', '-dev', device, '-out', base / f'gpu{j}.poly'])
            batch = base / 'jobs.tsv'
            manifest(batch, jobs)
            p = self.invoke(['-batch', batch])
            self.assertIn('using CPU screen', p.stderr)
            self.assertIn('6 jobs complete', p.stdout)
            self.assertIn('device setups 2, buffer allocations 3, prime uploads 4', p.stdout)
            for j, content in enumerate(expected):
                out = base / f'gpu{j}.poly'
                self.assertEqual(out.read_bytes(), content)
                for _, poly in read_cado_blocks(out):
                    self.assertTrue(resultant(poly) and resultant(poly) % poly['n'] == 0)


if __name__ == '__main__':
    unittest.main()
