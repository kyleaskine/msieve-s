#!/usr/bin/env python3
"""Bounded frozen-reference C196 timing/profiling; never starts production."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PILOT = HERE.parent/'20261020_as660_stage1_pilot'
sys.path.insert(0, str(ROOT/'stage23_bench/tools'))
from polyfmt import read_cado_blocks


def run(name, logging, profiler=None):
    manifest = json.loads((PILOT/'manifest.json').read_text())
    for path, digest in manifest['files'].items():
        if hashlib.sha256((PILOT/path).read_bytes()).hexdigest() != digest:
            raise RuntimeError('frozen input changed: '+path)
    wd = HERE/name
    wd.mkdir()
    (wd/'coeff_list.txt').write_text('138600\n')
    (wd/'stage1_core.ptx').symlink_to(PILOT/'bin/stage1_core.ptx')
    (wd/'cub').symlink_to(PILOT/'bin/cub', target_is_directory=True)
    hi = 2140100000 if profiler else 2150000000
    args = ('coeff_list=1 polydegree=5 collengine=gerbicz '
            f'stage1_qmin=2140000000 stage1_qmax={hi} stage1_aprog=1')
    if logging:
        args += ' stage1_provenance=provenance'
    cmd = ['stdbuf', '-oL', str(PILOT/'bin/msieve'), '-v', '-g', '0',
           '-t', '1', '-np1', '-nps', args, manifest['N']]
    if profiler == 'nsys':
        cmd = ['/usr/local/cuda/bin/nsys', 'profile', '--trace=cuda',
               '--sample=none', '--cpuctxsw=none', '--export=sqlite',
               '-o', str(wd/'trace')] + cmd
    elif profiler == 'ncu':
        cmd = ['/usr/local/cuda/bin/ncu', '--kernel-name',
               'sieve_kernel_trans_pp64_r64', '--launch-skip', '30',
               '--launch-count', '3', '--clock-control', 'none',
               '--cache-control', 'none', '--section', 'SpeedOfLight',
               '--section', 'MemoryWorkloadAnalysis', '--section',
               'WarpStateStats', '--section', 'LaunchStats',
               '--section', 'Occupancy', '--section', 'SourceCounters',
               '-o', str(wd/'trans')] + cmd
    (wd/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    with (wd/'gpu-before.csv').open('w') as output:
        subprocess.run(['nvidia-smi', '--query-gpu=timestamp,name,clocks.sm,'
                        'clocks.mem,power.draw,temperature.gpu,utilization.gpu,'
                        'memory.used', '--format=csv'], stdout=output, check=True)
    start = time.monotonic()
    coeff_start = coeff_end = None
    with (wd/'process.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=wd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            if 'coeff 138600 norm ' in line:
                coeff_start = time.monotonic()
            if 'coeff 138600: found ' in line:
                coeff_end = time.monotonic()
        rc = proc.wait()
    result = dict(name=name, logging=logging, profiler=profiler, returncode=rc,
                  process_seconds=time.monotonic()-start,
                  coefficient_seconds=(coeff_end-coeff_start
                      if coeff_start is not None and coeff_end is not None else None))
    raw = wd/'msieve.dat.ms'
    if raw.exists():
        polys = sorted(json.dumps(p,sort_keys=True) for _,p in read_cado_blocks(raw))
        result.update(raw_count=len(polys),
                      raw_set_sha256=hashlib.sha256('\n'.join(polys).encode()).hexdigest())
    (wd/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if rc or coeff_end is None:
        raise SystemExit('run/profile failed; see '+str(wd/'process.log'))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'pair':
        i = int(sys.argv[2])
        for logging in ([False, True] if i % 2 == 0 else [True, False]):
            run(f'pair{i}-{"on" if logging else "off"}',logging)
    elif mode == 'pairs':
        for i in range(5):
            for logging in ([False, True] if i % 2 == 0 else [True, False]):
                run(f'pair{i}-{"on" if logging else "off"}',logging)
    elif mode == 'warmup':
        run('warmup',False)
    elif mode in ('nsys','ncu'):
        run(mode,False,mode)
    else:
        raise SystemExit('warmup | pair INDEX | nsys | ncu')
