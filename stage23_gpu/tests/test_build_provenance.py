#!/usr/bin/env python3
"""Build manifests: record, import and verify against a tiny real gcc project.

The project mirrors the stage23_gpu Makefile's structure: an intermediate
object, a static library and a printf/cmp flag stamp that always looks stale
to `make -n`. Needs make, gcc and ar; no GPU.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('build_provenance', ROOT / 'stage23_bench/tools/build_provenance.py')
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)

MAKEFILE = r'''BUILD := build
FLAGS := -O1
CUDA ?= 120
PARTCC := gcc
.PHONY: FORCE
FORCE:
$(BUILD):
	mkdir -p $(BUILD)
$(BUILD)/flags: FORCE | $(BUILD)
	@printf '%s\n' 'gcc' '$(FLAGS)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi
$(BUILD)/part.o: part.c hello.h second/shadow.h $(BUILD)/flags | $(BUILD)
	$(PARTCC) $(FLAGS) -Ifirst -Isecond -c -o $@ part.c
$(BUILD)/libextra.a: extra.c | $(BUILD)
	gcc -O1 -c -o $(BUILD)/extra.o extra.c && ar rcs $@ $(BUILD)/extra.o
$(BUILD)/hello: hello.c hello.h $(BUILD)/part.o $(BUILD)/libextra.a $(BUILD)/flags
	gcc $(FLAGS) -o $@ hello.c $(BUILD)/part.o $(BUILD)/libextra.a -lm
# Like s23_sopt: no flag stamp and no Makefile dependency.
$(BUILD)/plain: plain.c | $(BUILD)
	gcc -O1 -DARCH=$(CUDA) -o $@ plain.c
'''
FILES = {
    'hello.h': 'int part(void);\nint extra(void);\n',
    'hello.c': '#include <stdio.h>\n#include "hello.h"\nint main(void) { printf("%d\\n", part() + extra()); return 0; }\n',
    'part.c': '#include "hello.h"\n#include <shadow.h>\nint part(void) { return SHADOW; }\n',
    'second/shadow.h': '#define SHADOW 1\n',
    'extra.c': 'int extra(void) { return 2; }\n',
    'plain.c': 'int main(void) { return ARCH == 0; }\n',
}


class Manifest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name).resolve()
        (self.dir / 'Makefile').write_text(MAKEFILE)
        for name, text in FILES.items():
            (self.dir / name).parent.mkdir(exist_ok=True)
            (self.dir / name).write_text(text)
        self.binary = self.dir / 'build/hello'
        self.make()

    def make(self, *args, target='build/hello'):
        subprocess.run(['make', '-s', *args, target], cwd=self.dir, check=True, capture_output=True)

    def verify(self, binary=None):
        return bp.verify(binary or self.binary, self.dir)

    def later(self, path):
        """Give a file a modification time after every build output."""
        future = self.binary.stat().st_mtime + 60
        os.utime(path, (future, future))

    def test_record_and_verify_follow_content_not_timestamps(self):
        bp.record(self.binary, self.dir, {})
        manifest = json.loads(bp.manifest_path(self.binary).read_text())
        deps = manifest['dependencies']
        for name in ('hello.c', 'hello.h', 'part.c', 'second/shadow.h', 'build/libextra.a', 'build/flags'):
            self.assertIn(str(self.dir / name), deps)
        self.assertIn('/usr/include/stdio.h', deps)
        self.assertNotIn(str(self.dir / 'build/part.o'), deps)  # intermediate: covered by the binary hash
        self.assertTrue(self.verify()['verified'])
        header = self.dir / 'hello.h'
        header.write_text(FILES['hello.h'] + '/* changed */\n')
        result = self.verify()
        self.assertFalse(result['verified'])
        self.assertIn(str(header), result['problems'][0])
        header.write_text(FILES['hello.h'])  # same bytes, new timestamp
        self.assertTrue(self.verify()['verified'])

    def test_record_refuses_inputs_newer_than_the_binary(self):
        future = self.binary.stat().st_mtime + 60
        os.utime(self.dir / 'part.c', (future, future))
        with self.assertRaisesRegex(bp.ProvenanceError, 'older than 1 input'):
            bp.record(self.binary, self.dir)

    def test_record_refuses_a_build_with_other_flags(self):
        self.make('FLAGS=-O2')
        with self.assertRaisesRegex(bp.ProvenanceError, 'build/flags does not match the current build flags'):
            bp.record(self.binary, self.dir)

    def test_record_refuses_an_unstamped_binary_make_did_not_rebuild(self):
        plain = self.dir / 'build/plain'
        self.make(target='build/plain')
        bp.record(plain, self.dir, {})
        # A variable change make cannot see: same bytes, other commands.
        with self.assertRaisesRegex(bp.ProvenanceError, 'same bytes with other build commands'):
            bp.record(plain, self.dir, {'CUDA': '89'})
        # A recipe edit: the rule has no Makefile dependency, so make leaves it.
        makefile = self.dir / 'Makefile'
        makefile.write_text(MAKEFILE.replace('-O1 -DARCH', '-O2 -DARCH'))
        self.later(makefile)
        with self.assertRaisesRegex(bp.ProvenanceError, r'older than 1 input\(s\), e.g. .*Makefile; .*make -B build/plain'):
            bp.record(plain, self.dir, {})
        self.assertFalse(self.verify(plain)['verified'])
        past = self.binary.stat().st_mtime - 60
        os.utime(makefile, (past, past))  # undo the test's future timestamp
        plain.unlink()
        self.make(target='build/plain')
        bp.record(plain, self.dir, {})
        self.assertTrue(self.verify(plain)['verified'])

    def test_record_refuses_untracked_newer_inputs_with_a_forced_rebuild_hint(self):
        # Make does not track system headers; a toolchain update needs make -B.
        stdio = self.dir / 'stdio_copy.h'
        stdio.write_text('/* stand-in for an updated system header */\n')
        (self.dir / 'hello.c').write_text('#include "stdio_copy.h"\n' + FILES['hello.c'])
        self.make()
        self.later(stdio)
        with self.assertRaisesRegex(bp.ProvenanceError, r'make -B build/hello if make reports it up to date'):
            bp.record(self.binary, self.dir)

    def test_make_variables_are_recorded_and_verify_ignores_the_caller_environment(self):
        plain = self.dir / 'build/plain'
        self.make('CUDA=89', target='build/plain')
        bp.record(plain, self.dir, {'CUDA': '89'})
        self.assertEqual(json.loads(bp.manifest_path(plain).read_text())['make_variables'], {'CUDA': '89'})
        with patch.dict(os.environ, {'CUDA': '120', 'MAKEFLAGS': 'CUDA=120'}):
            self.assertTrue(self.verify(plain)['verified'])
            # record reads the variables from the environment, as make does.
            self.assertEqual(bp.environment_variables(), {'CUDA': '120'})
        bp.record(self.binary, self.dir, {})
        with patch.dict(os.environ, {'CUDA': '75'}):
            self.assertTrue(self.verify()['verified'])

    def test_versioned_and_launched_compilers_are_tracked_and_unknown_ones_rejected(self):
        tools, gcc = self.dir / 'bin', shutil.which('gcc')
        assert gcc
        tools.mkdir()
        (tools / 'gcc-99').symlink_to(gcc)
        (tools / 'ccache').write_text('#!/bin/sh\nexec "$@"\n')
        (tools / 'ccache').chmod(0o755)
        (tools / 'mycc').symlink_to(gcc)
        makefile = self.dir / 'Makefile'
        makefile.write_text(MAKEFILE.replace('PARTCC := gcc', 'PARTCC := ./bin/ccache ./bin/gcc-99'))
        self.make('-B')
        bp.record(self.binary, self.dir, {})
        self.assertIn(str(self.dir / 'part.c'), json.loads(bp.manifest_path(self.binary).read_text())['dependencies'])
        (self.dir / 'part.c').write_text(FILES['part.c'] + '/* changed */\n')
        self.assertIn('part.c', self.verify()['problems'][0])
        makefile.write_text(MAKEFILE.replace('PARTCC := gcc', 'PARTCC := ./bin/mycc'))
        with self.assertRaisesRegex(bp.ProvenanceError, "build command './bin/mycc'.*not a recognized compiler"):
            bp.collect(self.binary, self.dir, {})

    def test_verify_notices_an_include_that_now_resolves_elsewhere(self):
        bp.record(self.binary, self.dir, {})
        # Identical bytes earlier on the -I path: every recorded file is unchanged.
        (self.dir / 'first').mkdir()
        (self.dir / 'first/shadow.h').write_text(FILES['second/shadow.h'])
        problems = self.verify()['problems']
        self.assertEqual(len(problems), 1)
        self.assertRegex(problems[0], 'reads different inputs: added .*first/shadow.h; dropped .*second/shadow.h')

    def test_verify_reports_unreadable_manifests_and_missing_make(self):
        bp.record(self.binary, self.dir, {})
        manifest = bp.manifest_path(self.binary)
        text = manifest.read_text()
        for broken in ('{', '[]', json.dumps(dict(json.loads(text), dependencies=None))):
            manifest.write_text(broken)
            result = self.verify()
            self.assertFalse(result['verified'])
            self.assertIn('unreadable manifest', result['problems'][0])
        manifest.write_text(text)
        with patch.dict(os.environ, {'PATH': str(self.dir / 'no-tools')}):
            result = self.verify()
        self.assertFalse(result['verified'])
        self.assertIn('cannot run make', result['problems'][0])

    def test_verify_reports_changed_commands_binary_and_missing_manifest(self):
        self.assertIn('no provenance manifest', self.verify()['problems'][0])
        bp.record(self.binary, self.dir)
        makefile = self.dir / 'Makefile'
        makefile.write_text(MAKEFILE + '# a comment changes no command\n')
        self.assertTrue(self.verify()['verified'])
        makefile.write_text(MAKEFILE.replace('-lm', '-lm -s'))
        self.assertEqual(self.verify()['problems'], ['make now builds it with different commands'])
        makefile.write_text(MAKEFILE)
        self.binary.write_bytes(self.binary.read_bytes() + b'\0')
        self.assertIn('differs from the binary', self.verify()['problems'][0])

    def test_import_checks_validated_hashes_and_toolchain(self):
        snapshot = self.dir / 'snapshot'
        validated = {str(snapshot / name): bp.sha256(self.dir / name) for name in FILES}
        validated[str(self.dir / 'build/libextra.a')] = bp.sha256(self.dir / 'build/libextra.a')
        version = subprocess.run(['gcc', '--version'], capture_output=True, text=True).stdout
        record = self.dir / 'validation.json'
        record.write_text(json.dumps(dict(dependencies_sha256=validated, compilers={'gcc': version})))
        path_map = [(str(snapshot) + '/', str(self.dir) + '/')]
        digest = bp.sha256(self.binary)
        with self.assertRaisesRegex(bp.ProvenanceError, 'not the validated'):
            bp.import_validation(self.binary, record, '0' * 64, path_map, self.dir)
        # Validated with -MM: system headers are accepted from the recorded toolchain.
        bp.import_validation(self.binary, record, digest, path_map, self.dir)
        origin = json.loads(bp.manifest_path(self.binary).read_text())['origin']
        self.assertIn('/usr/include/stdio.h', origin['unvalidated_toolchain_inputs'])
        self.assertEqual(origin['toolchain_versions_matched'], ['gcc'])
        self.assertTrue(self.verify()['verified'])
        record.write_text(json.dumps(dict(dependencies_sha256=validated, compilers={'gcc': 'gcc 0.1'})))
        with self.assertRaisesRegex(bp.ProvenanceError, 'gcc differs'):
            bp.import_validation(self.binary, record, digest, path_map, self.dir)
        changed = dict(validated, **{str(snapshot / 'part.c'): '0' * 64})
        record.write_text(json.dumps(dict(dependencies_sha256=changed, compilers={'gcc': version})))
        with self.assertRaisesRegex(bp.ProvenanceError, 'changed since validation'):
            bp.import_validation(self.binary, record, digest, path_map, self.dir)
        del validated[str(snapshot / 'hello.h')]
        record.write_text(json.dumps(dict(dependencies_sha256=validated, compilers={'gcc': version})))
        with self.assertRaisesRegex(bp.ProvenanceError, 'absent from the validation, e.g. .*hello.h'):
            bp.import_validation(self.binary, record, digest, path_map, self.dir)

    def test_cli_verify_exit_status(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(bp.main(['verify', str(ROOT / 'stage23_gpu/build/does-not-exist')]), 1)
        self.assertIn('NOT verified', out.getvalue())


if __name__ == '__main__':
    unittest.main()
