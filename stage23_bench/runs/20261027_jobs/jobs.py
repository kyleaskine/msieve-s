#!/usr/bin/env python3
"""Freeze and run a new composite through GPU stage 2/3 with CPU references.

Preparation snapshots raw inputs and CPU initial sopt before pipeline rotation.
Profile inheritance is explicit and recorded. All phases are independently resumable.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
import gpu_pipeline as driver
import sieve_profiles
from polyfmt import parse_cado_ropt, read_cado_blocks


def prepare(run, source, profile):
    if run.exists(): raise ValueError('preparation requires a fresh directory')
    cpu_source = source / 'pipeline_results'
    if 'STATUS=complete\n' not in (cpu_source / 'pipeline_settings.txt').read_text():
        raise ValueError('CPU reference incomplete')
    n = read_cado_blocks(cpu_source / 'best150_cado.txt')[0][1]['n']
    profiles = sieve_profiles.load()
    selected = next(p for p in profiles if p['id'] == profile)
    run.mkdir(parents=True)
    cpu = run / 'cpu_reference'
    cpu.mkdir()
    sources = [p for p in cpu_source.iterdir() if p.is_file()]
    sources.append(source / 'cado_sopt_unsorted.txt')
    before = {str(p):driver.digest(p) for p in sources}
    for p in sources:
        shutil.copy2(p, cpu / p.name)
        if driver.digest(cpu / p.name) != before[str(p)]: raise ValueError('source changed: ' + str(p))
    refs=[]
    for side in ('orig', 'inv'):
        for i,p in enumerate(parse_cado_ropt(cpu / f'cado_ropt_{side}.txt')):
            driver.validate_poly(p,n)
            refs.append(dict(method='cado_'+side,index=i,poly=p))
    if not refs: raise ValueError('no CPU references')
    driver.write_json(cpu / 'cado_references.json',refs)
    workflow = driver.Run(run)
    for name in ('s23_sopt', 's23_ropt', 's23_ropt_cpu'):
        workflow.freeze(name, ROOT / 'stage23_gpu/build' / name)
    workflow.freeze('cado_murphy')
    raw = source / 'msieve.dat.deduped.ms'
    if not raw.exists(): raw = source / 'msieve.dat.ms'
    identity,_ = workflow.snapshot(raw, 'raw')
    shutil.copy2(sieve_profiles.CATALOG, run / 'inputs/sieve_profiles.json')
    with (run / 'inputs/source.raw').open() as f:
        raw_n = next(int(l.split(':',1)[1]) for l in f if l.startswith('n:'))
    if raw_n != n: raise ValueError('raw/CPU n mismatch')
    driver.write_json(run / 'scoring.json',sieve_profiles.scoring(selected))
    driver.write_json(run / 'prepared.json',dict(n=n,digits=len(str(n)),source=str(source),input=identity,
        scope='GPU initial sopt, effort-50 resopt, breadth/depth/final refinement; independent CPU references; no test sieve.',
        sources_sha256=before,cpu_sha256={p.name:driver.digest(p) for p in cpu.iterdir()}))
    print('Prepared C'+str(len(str(n))),profile,flush=True)


def execute(run, tag, argv):
    argv=list(map(str,argv))
    driver.write_json(run / (tag+'_command.json'),dict(argv=argv,started=time.time()))
    start=time.monotonic()
    with (run / (tag+'.log')).open('ab') as log:
        proc=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    driver.write_json(run / (tag+'_execution.json'),dict(returncode=proc.returncode,seconds=time.monotonic()-start))
    if proc.returncode: raise RuntimeError(f'{tag}: exit {proc.returncode}; see {run / (tag+".log")}')
    print(tag,'complete',round(time.monotonic()-start,2),'seconds',flush=True)


def pipeline(run):
    scoring=driver.read_json(run / 'scoring.json')
    meta=driver.read_json(run / 'prepared.json')
    argv=[sys.executable,ROOT / 'scripts/gpu_pipeline.py','all','--run-dir',run,
        '--config',ROOT / 'nfs_config.ini.template','--input',meta['input']['source'],
        '--extract','1000','--resopt-effort','50','--seeds','300','--threads','2','--workers','2',
        '--launch','30','--ropt-batch-size','16','--refine-gpu','--murphy-gpu']
    for k in ('params','lattice','qband'): argv += ['--'+k,scoring[k]]
    execute(run,'pipeline',argv)


def audit(run):
    info=driver.read_json(run / 'preprocess/info.json')
    dest=run / 'sopt_audit'
    (dest / 'sopt').mkdir(parents=True,exist_ok=True)
    driver.write_json(dest / 'settings.json',dict(n=info['n'],raw_polynomials=info['count']))
    link=dest / 'sopt/e0_unsorted.txt'
    if not link.exists(): link.symlink_to(run / 'preprocess/cado_sopt_unsorted.txt')
    if not (dest / 'sopt/verification.json').exists():
        execute(run,'initial_sopt_audit',[sys.executable,ROOT / 'stage23_bench/runs/20261012_c196/verify_sopt.py',
            dest,run / 'cpu_reference/cado_sopt_unsorted.txt'])
    report=driver.read_json(dest / 'sopt/verification.json')
    if report['worse_at_full_precision']: raise ValueError('initial sopt regression')
    dest=run / 'depth_audit'
    dest.mkdir(exist_ok=True)
    verified=0
    for pool in sorted((run / 'depth').glob('*.pool')):
        out=dest / (pool.stem+'.poly')
        if not out.exists():
            execute(run,'depth_audit/'+pool.stem,[run / 'bin/s23_ropt_cpu', '-poly',run / 'seeds' / out.name,
                '-search','-aw','1.3','-pool-in',pool,'-rerank','4096','-refine','200','-top','3','-t','2','-out',out])
        if driver.digest(out) != driver.digest(run / 'depth' / out.name):
            raise ValueError('CPU/GPU depth exports differ: '+pool.stem)
        for _,p in read_cado_blocks(out): driver.validate_poly(p,info['n']);verified+=1
    if not verified: raise ValueError('no depth exports audited')
    driver.write_json(run / 'migration_audit.json',dict(initial_sopt=report,depth_exports_verified=verified,
        depth_cpu_gpu_bytes_equal=True))
    print('Audited',info['count'],'initial inputs and',verified,'depth exports',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=('prepare','pipeline','audit','compare','finish'))
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--source-dir',type=Path)
    ap.add_argument('--profile')
    args=ap.parse_args()
    run=args.run_dir.resolve()
    if args.phase=='prepare':
        if args.source_dir is None or args.profile is None: ap.error('prepare needs --source-dir and --profile')
        prepare(run,args.source_dir.resolve(),args.profile)
    else:
        if args.phase=='pipeline': pipeline(run)
        if args.phase in ('audit','finish'): audit(run)
        if args.phase in ('compare','finish'):
            execute(run,'comparison',[sys.executable,Path(__file__).with_name('compare.py'),'--run-dir',run])


if __name__=='__main__': main()
