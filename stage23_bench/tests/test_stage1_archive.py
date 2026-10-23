"""Removal must fail closed if a verified artifact changed."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('archive_cleanup',
    ROOT/'stage23_bench/runs/20261023_stage1_archive/archive.py')
archive=importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class Cleanup(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.here=self.root/'stage23_bench/run';self.here.mkdir(parents=True)
        old_root,old_here=archive.ROOT,archive.HERE
        archive.ROOT,archive.HERE=self.root,self.here
        self.addCleanup(lambda:setattr(archive,'ROOT',old_root))
        self.addCleanup(lambda:setattr(archive,'HERE',old_here))
        self.source=self.root/'source.jsonl';self.source.write_bytes(b'original')
        self.packed=self.root/'source.jsonl.zst';self.packed.write_bytes(b'verified archive stand-in')
        manifest=dict(all_verified=True,removed=[],extracts=[],archives=[dict(
            source=self.source.name,archive=self.packed.name,
            source_identity=archive.identity(self.source),archive_identity=archive.identity(self.packed),
            archive_sha256=archive.digest(self.packed))],original_bytes=8,
            redundant_extract_bytes=0,archive_bytes=self.packed.stat().st_size)
        (self.here/'manifest.json').write_text(json.dumps(manifest))
        (self.here/'prepared.json').write_text(json.dumps(dict(candidates=1687829)))
        for i in range(8):
            (self.here/f'{i}.complete.json').write_text(json.dumps(dict(artifact=dict(
                path=self.packed.name,sha256=archive.digest(self.packed),
                candidates=1687829 if i==0 else 0))))
        (self.root/'stage23_bench/sidecar_archives.json').write_text('{"extracts":{}}')

    def test_changed_source_is_preserved(self):
        self.source.write_bytes(b'changed source')
        with self.assertRaises(AssertionError):archive.remove_verified()
        self.assertTrue(self.source.exists());self.assertTrue(self.packed.exists())

    def test_changed_archive_preserves_source(self):
        import os
        before=self.packed.stat()
        self.packed.write_bytes(b'X'*before.st_size)
        os.utime(self.packed,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaises(AssertionError):archive.remove_verified()
        self.assertTrue(self.source.exists())

    def test_verified_cleanup_and_idempotent_resume(self):
        archive.remove_verified()
        self.assertFalse(self.source.exists());self.assertTrue(self.packed.exists())
        archive.remove_verified()
        self.assertEqual(json.loads((self.here/'manifest.json').read_text())['removed'],[self.source.name])


if __name__=='__main__':unittest.main()
