#!/usr/bin/env python3
"""Inspect the C179 msieve-original-only winner without launching a GPU search.

Rescore all CADO outputs, the top 20 msieve exports per orientation, the top
10 in the winning seed family, and every exported copy of the target rotation.
This is a focused shortlist, not an exhaustive rescore of all msieve outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'stage23_bench/tools'))
from polyfmt import (cado_poly_text, parse_cado_ropt, parse_msieve_p,
                     read_cado_blocks, resultant, rotation_between, cado_murphy_binary)
import score_polys as scores


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')


def key(poly):
    sign = 1 if poly['c5'] > 0 else -1
    return (poly['Y0'], poly['Y1'], *(sign*poly[f'c{i}'] for i in range(6)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT/'pipeline_results')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    source, run = args.source_dir.resolve(), args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    files = dict(msieve_orig='msieve_ropt_orig.p', msieve_inv='msieve_ropt_inv.p',
                 cado_orig='cado_ropt_orig.txt', cado_inv='cado_ropt_inv.txt')
    sources = [source/name for name in [*files.values(), 'best150_cado.txt',
               'best150_cado_inv.txt', 'best300_msieve.ms', 'best300_msieve_inv.ms',
               'pipeline_settings.txt', 'pipeline_report.txt']]
    provenance = {str(p):digest(p) for p in sources}
    seeds = read_cado_blocks(source/'best150_cado.txt')
    n = seeds[0][1]['n']
    if len(str(n)) != 179:
        raise ValueError('expected C179')
    data = {m:(parse_msieve_p if m.startswith('msieve') else parse_cado_ropt)(source/f)
            for m,f in files.items()}
    winner = dict(max(data['msieve_orig'], key=lambda p:p['e']), n=n)
    matches = [(i,p) for i,(_,p) in enumerate(seeds) if p['Y1']==winner['Y1']]
    if len(matches)!=1:
        raise ValueError('ambiguous seed family')
    index, seed = matches[0]
    inverse_seed = read_cado_blocks(source/'best150_cado_inv.txt')[index][1]
    if any(inverse_seed[f'c{i}'] != -seed[f'c{i}'] for i in range(6)) or any(
            inverse_seed[k]!=seed[k] for k in ('n','Y0','Y1')):
        raise ValueError('inverted seed is not exactly the sign-flipped original')
    for name in ('best300_msieve.ms','best300_msieve_inv.ms'):
        values = list(map(int,(source/name).read_text().splitlines()[index].split()[:8]))
        p = seed if name=='best300_msieve.ms' else inverse_seed
        if values != [p[f'c{i}'] for i in range(5,-1,-1)]+[p['Y1'],p['Y0']]:
            raise ValueError('msieve and CADO did not receive the same seed')

    def rotation(poly):
        relation = rotation_between(seed, poly)
        if relation and len(relation['rotation'])==2:
            u=relation['rotation'][1]
            return dict(u=u,v=relation['rotation'][0]-u*relation['t'],t=relation['t'])

    target = rotation(winner)
    if target is None:
        raise ValueError('winner is not a linear rotation')

    def same(poly):
        r=rotation(poly)
        return r is not None and (r['u'],r['v'])==(target['u'],target['v'])

    rows, counts = [], {}
    for method, polys in data.items():
        metric='e' if method.startswith('msieve') else 'MurphyE'
        family=[p for p in polys if p['Y1']==seed['Y1']]
        copies=[p for p in family if same(p)]
        best_family=max(family,key=lambda p:p[metric])
        counts[method]=dict(total=len(polys),family_exports=len(family),
            target_copies=len(copies),target_translations=[rotation(p)['t'] for p in copies],
            family_leading_coefficients=sorted({p['c5'] for p in family}),
            printed_best=max(p[metric] for p in polys),printed_family_best=best_family[metric],
            best_family_rotation=rotation(best_family))
        if method.startswith('msieve'):
            selected=sorted(polys,key=lambda p:-p[metric])[:20]+sorted(family,key=lambda p:-p[metric])[:10]+copies
        else:
            selected=polys
        seen=set()
        for poly in selected:
            if key(poly) in seen:
                continue
            seen.add(key(poly))
            poly=dict(poly,n=n)
            value=resultant(poly)
            if not value or value % n:
                raise ValueError('invalid selected polynomial')
            rows.append(dict(method=method,poly=poly,printed=poly[metric],rotation=rotation(poly),
                             target_rotation=same(poly)))
    for path, expected in provenance.items():
        if digest(path)!=expected:
            raise ValueError('pipeline results changed during inspection')
    save(run/'prepared.json',dict(n=n,seed_index_zero_based=index,target=target,counts=counts,
        sources_sha256=provenance,script_sha256=digest(__file__),shortlist=__doc__))
    save(run/'shortlist.json',rows)
    for name,p in [('seed',seed),('seed_inv',inverse_seed),('msieve_winner',winner)]:
        (run/(name+'.poly')).write_text(cado_poly_text(p))
    print('C179 seed',index+1,'target',target,'shortlist',len(rows),flush=True)
    (run/'bin').mkdir()
    for name,path in [('s23_ropt_cpu',ROOT/'stage23_gpu/build/s23_ropt_cpu'),
                      ('cado_murphy',Path(cado_murphy_binary()))]:
        shutil.copy2(path,run/'bin'/name)
    save(run/'binaries.json',{p.name:digest(p) for p in (run/'bin').iterdir()})
    plans=[]
    for name,extra in [('dense_breadth',['-maxcells','2e11']),('dense_depth',['-maxcells','6e12']),
                      ('classes16',['-maxcells','2e11','-classes','2520','16','-class-explore','4']),
                      ('classes32',['-maxcells','2e11','-classes','2520','32','-class-explore','8'])]:
        command=[str(run/'bin/s23_ropt_cpu'),'-poly',str(run/'seed.poly'),'-search','-plan',
            '-budget','3','-maxlines','1000','-umax','20000','-aw','1.3','-rerank','4096',
            '-refine','200','-top','8','-t','2','-trace-cell',str(target['u']),str(target['v']),*extra]
        start=time.monotonic()
        with (run/(name+'.log')).open('wb') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(run/(name+'.log')).read_text()
        row=dict(name=name,command=command,seconds=time.monotonic()-start,
            log_sha256=digest(run/(name+'.log')),
            trace=[s for s in text.splitlines() if s.startswith('# trace-cell')],
            model=[s for s in text.splitlines() if s.startswith('# size model')])
        plans.append(row)
        print(name,row['trace'][-1],flush=True)
    save(run/'plans.json',plans)
    scores.cado_murphy_binary=lambda:str(run/'bin/cado_murphy')
    params=SimpleNamespace(params='default',threads=2,points=16000,eval_points=64000,
                           trans=0,useskew=False)
    unique={key(r['poly']):r['poly'] for r in rows}
    print('Uniform MurphyE scoring:',len(unique),'unique polynomials',flush=True)
    values=scores.murphy(list(unique.values()),n,params)
    by_key=dict(zip(unique,values))
    for row in rows:
        score=by_key[key(row['poly'])]
        row.update(murphy=score[0],optimized_skew=score[1])
    save(run/'screen.json',rows)
    chosen=set()
    for method in files:
        chosen.update(sorted((i for i,r in enumerate(rows) if r['method']==method),
                             key=lambda i:-rows[i]['murphy'])[:5])
    chosen.update(i for i,r in enumerate(rows) if r['target_rotation'])
    final=[dict(rows[i]) for i in sorted(chosen)]
    params.useskew=True
    for angles in (256000,1024000):
        params.eval_points=angles
        values=scores.murphy([dict(r['poly'],skew=r['optimized_skew']) for r in final],n,params)
        for row,value in zip(final,values):
            row[f'murphy_{angles}']=value[0]
    winners={m:max((r for r in final if r['method']==m),key=lambda r:r['murphy_1024000']) for m in files}
    save(run/'summary.json',dict(n=n,seed_rank=index+1,target=target,counts=counts,winners=winners,
        finalists=final,plans=plans,screened_records=len(rows),screened_unique=len(unique),
        validation=dict(search_angles=16000,screen_angles=64000,final_angles=[256000,1024000]),
        caveats=[__doc__,'Standard MurphyE only; no job lattice, test sieve or GPU search.',
                 'Plan admission does not establish retention through the sieve or final scoring.']))
    for method,row in winners.items():
        (run/(method+'_best.poly')).write_text(cado_poly_text(dict(row['poly'],skew=row['optimized_skew'])))
        print(method,row['murphy_1024000'],'target',row['target_rotation'],flush=True)


if __name__=='__main__':
    main()
