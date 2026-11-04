#!/usr/bin/env python3
"""Corpus-runner thread and timing bookkeeping, using a stub binary; no GPU required."""
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sopt_corpus as corpus

PAIR = """
### Input raw polynomial (0) ###
# n: 1000003
# Y0: -7
# Y1: 3
# c0: 1
# c1: 2
# c2: 3
# c3: 4
# c4: 5
# c5: 6
# skew: 2.000
### Size-optimized polynomial (0) ###
n: 1000003
Y0: -7
Y1: 3
c0: 1
c1: 2
c2: 3
c3: 4
c4: 5
c5: 6
skew: 2.000
# side 1 lognorm 1.00, exp_E 1.00, alpha 0.00 (proj 0.00), 1 real roots
"""


class Threads(unittest.TestCase):
    def test_new_run_uses_the_request_or_the_historical_default(self):
        self.assertEqual(corpus.resolve_threads(None, 8), 8)
        self.assertEqual(corpus.resolve_threads(None, None), corpus.DEFAULT_THREADS)

    def test_resume_keeps_the_saved_count(self):
        self.assertEqual(corpus.resolve_threads(dict(threads=8), None), 8)
        self.assertEqual(corpus.resolve_threads(dict(threads=8), 8), 8)

    def test_manifest_without_a_count_is_a_two_thread_replay(self):
        self.assertEqual(corpus.resolve_threads(dict(total=1), None), 2)
        self.assertEqual(corpus.resolve_threads(dict(total=1), 2), 2)

    def test_changing_the_count_on_resume_is_rejected(self):
        for saved in (dict(threads=8), dict(total=1)):
            with self.assertRaisesRegex(ValueError, 'new --run-dir'):
                corpus.resolve_threads(saved, 4)


class Execute(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.args_file = self.base / 'args.txt'
        output = self.base / 'output.txt'
        output.write_text('# s23_sopt: effort 0, GPU, batches of 4096\n' + PAIR)
        self.stub = self.base / 'stub.sh'
        self.stub.write_text(f'#!/bin/sh\necho "$@" > {self.args_file}\ncat {output}\n')
        self.stub.chmod(self.stub.stat().st_mode | stat.S_IEXEC)
        self.reference = self.base / 'reference.txt'
        self.reference.write_text(PAIR)
        self.audit = self.base / 'audit.json'
        self.audit.write_text(json.dumps(dict(differences=[])))

    def run_job(self, plan):
        job = dict(name='stub', count=1, audit=dict(path=str(self.audit), sha256=corpus.sha(self.audit)),
                   sources=dict(raw=dict(path=str(self.reference), sha256=corpus.sha(self.reference)),
                                sopt_e0=dict(path=str(self.reference), sha256=corpus.sha(self.reference))))
        plan = dict(plan, binary=str(self.stub), binary_sha256=corpus.sha(self.stub))
        with patch.object(corpus, 'wait_available'):
            return corpus.execute(job, plan, self.base)

    def test_thread_count_reaches_the_command_and_both_clocks_are_saved(self):
        record = self.run_job(dict(threads=8))
        self.assertEqual(self.args_file.read_text().split()[:2], ['-t', '8'])
        self.assertEqual((record['status'], record['threads'], record['count']), ('passed', 8, 1))
        saved = json.loads((self.base / 'stub.json').read_text())
        self.assertGreaterEqual(saved['seconds'], 0)
        self.assertIn('wall_seconds', saved)

    def test_old_manifest_replays_at_two_threads(self):
        record = self.run_job(dict())
        self.assertEqual(self.args_file.read_text().split()[:2], ['-t', '2'])
        self.assertEqual(record['threads'], 2)


if __name__ == '__main__':
    unittest.main()
