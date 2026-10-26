#!/usr/bin/env python3
"""Check final rebuilt CUDA artifacts against the completed review corpus."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    dest = run/'final_validation'
    dest.mkdir(exist_ok=False)
    binary = ROOT/'stage23_gpu/build/s23_ropt'
    records = []
    env = dict(os.environ, S23_ROPT_BIN=str(binary))

    def command(tag, argv):
        start = time.monotonic()
        with (dest/(tag+'.log')).open('wb') as log:
            subprocess.run(list(map(str, argv)), cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        records.append(dict(tag=tag,command=list(map(str,argv)),seconds=time.monotonic()-start,
                            log_sha256=digest(dest/(tag+'.log'))))

    command('fault_state', [ROOT/'stage23_gpu/build/test_murphy_cuda'])
    command('cli', ['python3', ROOT/'stage23_gpu/tests/test_refinement_gpu.py'])
    checked = 0
    for mode in ('alpha_gpu', 'murphy_checked'):
        original = run/'corpus'/mode
        target = dest/mode
        target.mkdir()
        lines = (original/'jobs.tsv').read_text().splitlines()
        jobs = []
        for line in lines[1:]:
            args = line.split('\t')
            pos = args.index('-out')+1
            args[pos] = str(target/Path(args[pos]).name)
            jobs.append('\t'.join(args))
        manifest = target/'jobs.tsv'
        manifest.write_text(lines[0]+'\n'+'\n'.join(jobs)+'\n')
        command(mode, [binary, '-batch', manifest])
        for output in original.glob('*.poly'):
            if digest(output) != digest(target/output.name):
                raise ValueError(f'final rebuild differs: {mode}/{output.name}')
            checked += 1
    summary = dict(records=records, matching_exports_files=checked,
        binary_sha256=digest(binary), test_binary_sha256=digest(ROOT/'stage23_gpu/build/test_murphy_cuda'))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Final CUDA rebuild: fault/state tests, six CLI tests and',checked,'golden export files pass')


if __name__=='__main__':
    main()
