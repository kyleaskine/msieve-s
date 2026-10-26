#!/usr/bin/env python3
"""Measure target ranks in the unchanged, frozen class-proposal algorithm."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from audit import ROOT, digest, prepare, save
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import cado_compile_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    dest = args.run_dir.resolve()
    meta = prepare(dest)
    folder = dest / 'proposal_audit'
    folder.mkdir(exist_ok=True)
    binary = folder / 'probe'
    build = folder / 'build.json'
    if not build.exists():
        if binary.exists():
            raise ValueError('incomplete previous build; inspect before retrying')
        source = folder / 'proposal_probe.cpp'
        shutil.copy2(Path(__file__).with_name(source.name), source)
        inc, libs, ld = cado_compile_args()
        cio = dest / 'source/tools/cado_io.c'
        commands = [
            ['gcc', '-O2', '-std=c99', '-fopenmp', *inc, '-I'+str(dest/'source/include'),
             '-c', str(cio), '-o', str(folder/'cado_io.o')],
            ['g++', '-O2', '-std=c++17', '-Wall', '-Wextra', '-ffp-contract=off', '-fopenmp',
             '-I'+str(dest/'source/include'), '-I'+str(dest/'source/tools'), str(source),
             str(folder/'cado_io.o'), *libs, *ld, '-o', str(binary)],
        ]
        with (folder / 'build.log').open('wb') as fh:
            for command in commands:
                subprocess.run(command, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, check=True)
        save(build, dict(commands=commands, sha256={str(p):digest(p) for p in [source, binary, *map(Path,libs)]},
                         frozen_sources=meta['sha256']))
    provenance = json.loads(build.read_text())
    for name, expected in provenance['sha256'].items():
        if digest(name) != expected:
            raise ValueError(f'probe build input/output changed: {name}')
    records = []
    for case in meta['cases']:
        for modulus in (2520, 420, 210):
            record = folder / f"{case['name']}_{modulus}.json"
            command = [str(binary), str(dest/case['file']), str(modulus),
                       str(case['target']['u']), str(case['target']['v'])]
            if record.exists():
                row = json.loads(record.read_text())
                if row['command'] != command or row['binary_sha256'] != digest(binary):
                    raise ValueError(f'changed probe run: {record}')
            else:
                start = time.monotonic()
                proc = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
                row = json.loads(proc.stdout)
                row.update(case=case['name'], command=command, seconds=time.monotonic()-start,
                    binary_sha256=digest(binary), input_sha256=digest(dest/case['file']))
                save(record, row)
            records.append(row)
            print(case['name'], modulus, 'root rank', row['root_proposal_rank'],
                  'within u', row['rank_within_u'], 'pool admits',
                  [p['selected_count'] for p in row['pools'] if p['proposed']], flush=True)
    save(folder/'summary.json', dict(records=records, note=__doc__, script_sha256=digest(__file__)))


if __name__ == '__main__':
    main()
