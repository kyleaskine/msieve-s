#!/usr/bin/env python3
"""Validate the reviewed class planner without modifying the original study."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage23_bench/runs/20261025_root_coverage'))
from audit import digest, save, parse_plan
from pilot import idle
from polyfmt import read_cado_blocks, resultant

SOURCE = ROOT/'stage23_gpu/runs/20261027_root_coverage'


def prepare(run):
    marker=run/'prepared.json'
    if marker.exists():
        meta=json.loads(marker.read_text())
        for path,expected in meta['sha256'].items():
            if digest(run/path)!=expected: raise ValueError('changed snapshot: '+path)
        return meta
    run.mkdir(parents=True,exist_ok=False)
    files={p:Path('seeds')/p.name for p in (SOURCE/'seeds').glob('*.poly')}
    for name in ('s23_ropt','s23_ropt_cpu'):
        files[ROOT/'stage23_gpu/build'/name]=Path('bin')/name
    for folder in ('include','tools'):
        for p in (ROOT/'stage23_gpu'/folder).iterdir():
            if p.suffix in ('.h','.cu','.cpp','.c'): files[p]=Path('source')/folder/p.name
    hashes={}
    for source,target in files.items():
        dest=run/target;dest.parent.mkdir(parents=True,exist_ok=True)
        expected=digest(source);shutil.copy2(source,dest)
        if digest(dest)!=expected: raise ValueError('copy changed')
        hashes[str(target)]=expected
    meta=dict(sha256=hashes,source_study=str(SOURCE),script_sha256=digest(__file__))
    save(marker,meta)
    return meta


def execute(run,tag,command):
    record=run/(tag+'.json');log=run/(tag+'.log')
    record.parent.mkdir(parents=True,exist_ok=True)
    if record.exists():
        row=json.loads(record.read_text())
        if row['command']!=command or row['returncode'] or digest(log)!=row['log_sha256']:
            raise ValueError('changed or failed command: '+tag)
        return row,log.read_text()
    if log.exists(): raise ValueError('unfinished command: '+tag)
    start=time.monotonic()
    with log.open('wb') as stream:
        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    row=dict(command=command,returncode=result.returncode,seconds=time.monotonic()-start,log_sha256=digest(log))
    save(record,row)
    if result.returncode: raise RuntimeError('failed command: '+tag)
    return row,log.read_text()


def host(run):
    records=[]
    for path in sorted((SOURCE/'plans').glob('*/*.json')):
        old=json.loads(path.read_text());command=list(old['command'])
        command[0]=str(run/'bin/s23_ropt_cpu')
        command[command.index('-poly')+1]=str(run/'seeds'/(old['case']+'.poly'))
        _,log=execute(run,'plans/'+old['case']+'/'+old['variant'],command)
        extra=json.loads((SOURCE/'prepared.json').read_text())['variants'][old['variant']]
        plan=parse_plan(log,extra)
        legacy=not old['variant'].startswith('size')
        if legacy:
            for key in ('classes','cells','target_result','budget_steps'):
                if plan[key]!=old[key]: raise ValueError('legacy plan changed: '+str(path)+' '+key)
        records.append(dict(case=old['case'],variant=old['variant'],legacy=legacy,
            previous_target=old['target_result'],target=plan['target_result'],cells=plan['cells'],
            classes=plan['classes'],coarse_and_scan_counts=re.findall(r'^# class probes: .*$',log,re.M),
            basins=re.findall(r'^# class basin .*$',log,re.M)))
        print(old['case'],old['variant'],plan['target_result'],flush=True)
    # Reproduce the review's large-range example with default maxlines too.
    command=[str(run/'bin/s23_ropt_cpu'),'-poly',str(run/'seeds/c197_s0001.poly'),
        '-search','-classes','30','16','-class-size-aware','-umax','100000',
        '-maxcells','100000','-t','2','-plan']
    execute(run,'large_umax',command)
    save(run/'host_summary.json',dict(records=records,legacy_plans_equal=sum(r['legacy'] for r in records),
        completed=len(records),large_umax_passed=True,note='Planning/parity check, not a new quality or timing study.'))


def gpu(run):
    records=[]
    for name in ('c208_s0222','c192_winner','c197_s0001'):
        for mode in ('size2520_16','size2520_32'):
            tag='gpu/'+name+'_'+mode
            row=json.loads((run/'plans'/name/(mode+'.json')).read_text())
            command=[x for x in row['command'] if x!='-plan']
            command[0]=str(run/'bin/s23_ropt')
            output=run/(tag+'.poly');pool=run/(tag+'.pool')
            command+=['-launch','30','-check','128','-refine-gpu','-refine-check',
                      '-pool-out',str(pool),'-out',str(output)]
            idle()
            _,log=execute(run,tag,command)
            if 'vs brute force: 0 differ' not in log: raise ValueError('missing root audit')
            hostlog=(run/'plans'/name/(mode+'.log')).read_text()
            for pattern in (r'^# class u .*$',r'^# class basin .*$',r'^# class probes: .*$',r'^# class cells .*$'):
                if re.findall(pattern,log,re.M)!=re.findall(pattern,hostlog,re.M):
                    raise ValueError('CPU/GPU planning disagreement')
            cpu=run/('replay/'+name+'_'+mode+'.poly')
            command=[str(run/'bin/s23_ropt_cpu'),'-poly',str(run/'seeds'/(name+'.poly')),
                '-search','-aw','1.3','-pool-in',str(pool),'-rerank','4096','-refine','200',
                '-top','8','-t','2','-out',str(cpu)]
            execute(run,'replay/'+name+'_'+mode,command)
            if output.read_bytes()!=cpu.read_bytes(): raise ValueError('CPU/GPU exports differ')
            polys=read_cado_blocks(output);seed=read_cado_blocks(run/'seeds'/(name+'.poly'))[0][1]
            if len(polys)!=200: raise ValueError('wrong export count')
            for _,p in polys:
                value=resultant(p)
                if p['n']!=seed['n'] or not value or value%p['n']: raise ValueError('invalid export')
            records.append(dict(case=name,mode=mode,exports=len(polys),cpu_gpu_equal=True,
                output_sha256=digest(output),pool_sha256=digest(pool),trace=[l for l in log.splitlines() if l.startswith('# trace-cell')]))
            print(name,mode,'200 identical CPU/GPU exports',flush=True)
    save(run/'gpu_summary.json',dict(records=records,verified_exports=sum(r['exports'] for r in records)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','host','gpu'))
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();run=args.run_dir.resolve();prepare(run)
    if args.phase=='host':host(run)
    if args.phase=='gpu':gpu(run)
