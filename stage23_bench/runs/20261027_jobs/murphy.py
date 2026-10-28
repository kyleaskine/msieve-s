#!/usr/bin/env python3
"""Bounded, audited C179 Murphy refinement from the unchanged saved depth pool."""
import argparse
import json
import re
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from retention import ROOT, digest, save
from polyfmt import read_cado_blocks, cado_poly_text, resultant
import score_polys as scores


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--score-only',action='store_true')
    args=ap.parse_args()
    run=args.run_dir.resolve()
    dest=run / 'murphy_audit'
    if not args.score_only: dest.mkdir(exist_ok=False)
    common=['-poly',run / 'seed.poly','-search','-aw','1.3','-pool-in',run / 'depth.pool',
        '-rerank','32','-refine','32','-top','8','-t','2','-murphy',
        '-Bf','1e7','-Bg','5e6','-area','1e16','-murphy-points','16000',
        '-murphy-eval-points','64000','-maxeval','384','-trace-cell','-55','34188178']
    records=json.loads((dest/'commands.json').read_text()) if args.score_only else []
    modes=[] if args.score_only else [('gpu','s23_ropt',['-refine-gpu','-refine-check','-murphy-gpu','-murphy-check']),
                              ('cpu','s23_ropt_cpu',[])]
    for mode,binary,flags in modes:
        argv=list(map(str,[run / 'bin' / binary,*common,*flags,'-out',dest / (mode+'.poly')]))
        start=time.monotonic()
        with (dest / (mode+'.log')).open('wb') as log:
            p=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(mode=mode,argv=argv,returncode=p.returncode,seconds=time.monotonic()-start))
        save(dest / 'commands.json',records)
        if p.returncode: raise RuntimeError(f'{mode}: exit {p.returncode}; see log')
    if len(records)!=2 or any(r['returncode'] for r in records): raise ValueError('incomplete audit')
    # FP score comments can differ in their last bits. Require identical integer
    # coefficients, skews, ranking and discrete search decisions separately.
    cpu=read_cado_blocks(dest/'cpu.poly');gpu=read_cado_blocks(dest/'gpu.poly')
    if len(cpu)!=len(gpu) or any(p!=q for (_,p),(_,q) in zip(cpu,gpu)):
        raise ValueError('Murphy polynomial configurations differ')
    error=0.0
    for (a,_),(b,_) in zip(cpu,gpu):
        for key in ('u','v','t','accepted','calls','limited'):
            if re.search(r'\b'+key+r' (\S+)',a)[1]!=re.search(r'\b'+key+r' (\S+)',b)[1]:
                raise ValueError('Murphy search decisions differ')
        for key in ('MurphyE','control','proposal'):
            x=float(re.search(r'\b'+key+r' (\S+)',a)[1]); y=float(re.search(r'\b'+key+r' (\S+)',b)[1])
            error=max(error,abs(x-y)/max(abs(x),abs(y)))
    if error>2e-10: raise ValueError('Murphy scores differ beyond audit tolerance')
    n=read_cado_blocks(run/'seed.poly')[0][1]['n']
    rows=[]
    for method,path in [('gpu_murphy',dest/'gpu.poly'),('msieve_orig',run/'msieve_winner.poly')]:
        for i,(_,p) in enumerate(read_cado_blocks(path)):
            r=resultant(p)
            if not r or r%n: raise ValueError('invalid resultant')
            rows.append(dict(method=method,index=i,poly=p))
    scores.cado_murphy_binary=lambda:str(run/'bin/cado_murphy')
    params=SimpleNamespace(params='default',threads=2,points=16000,eval_points=64000,trans=0,useskew=False)
    values=scores.murphy([r['poly'] for r in rows],n,params)
    for r,(e,s,_) in zip(rows,values): r.update(murphy=e,optimized_skew=s)
    params.useskew=True
    params.eval_points=1024000
    values=scores.murphy([dict(r['poly'],skew=r['optimized_skew']) for r in rows],n,params)
    for r,(e,_,_) in zip(rows,values): r['murphy_1024000']=e
    winners={m:max((r for r in rows if r['method']==m),key=lambda r:r['murphy_1024000'])
             for m in ('gpu_murphy','msieve_orig')}
    result=dict(n=n,commands=records,cpu_gpu_bytes_equal=digest(dest/'cpu.poly')==digest(dest/'gpu.poly'),
        cpu_gpu_configurations_equal=True,max_score_comment_relative_error=error,verified_exports=len(rows)-1,
        pool_sha256=digest(run/'depth.pool'),winners=winners,rows=rows,
        gpu_over_msieve=winners['gpu_murphy']['murphy_1024000']/winners['msieve_orig']['murphy_1024000'],
        scope='First 32 saved proxy candidates; bounded Murphy audit, not full-pool or complete-pipeline comparison.')
    save(dest/'result.json',result)
    w=winners['gpu_murphy']
    (dest/'best.poly').write_text(cado_poly_text(dict(w['poly'],skew=w['optimized_skew'])))
    print('Murphy check: CPU/GPU configurations identical;',len(rows)-1,'verified; GPU/MS',result['gpu_over_msieve'],flush=True)


if __name__=='__main__': main()
