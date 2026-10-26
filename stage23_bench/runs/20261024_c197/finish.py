#!/usr/bin/env python3
"""Wait for C197, audit new CUDA paths on its pools, compare and archive results."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'stage23_bench/tools'))
from gpu_pipeline import write_json,digest,validate_poly
from polyfmt import read_cado_blocks, iter_sopt, raw_key, cado_poly_text
import score_polys as scores


def coords(label):
    return tuple(int(re.search(rf'\b{k} (-?\d+)',label)[1]) for k in ('u','v'))


def timing_summary(run):
    phases = {}
    for phase in ('breadth', 'depth', 'murphy'):
        jobs = [json.loads(p.read_text()) for p in (run/'logs'/phase).glob('*.json')]
        batches = list((run/'logs/batches'/phase).glob('*.json'))
        summaries = [line for p in (run/'logs'/phase).glob('*.log')
                     for line in p.read_text().splitlines() if line.startswith('# s23_ropt:')]
        parts = {name: sum(float(v) for line in summaries for v in re.findall(pattern, line))
                 for name, pattern in [('size_model', r'size model ([0-9.]+) s'),
                                       ('sieve', r'sieve ([0-9.]+) s'),
                                       ('refine', r'refine ([0-9.]+) s')]}
        phases[phase] = dict(seeds=len(jobs), batches=len(batches),
            seed_process_seconds_sum=sum(j['seconds'] for j in jobs),
            logged_component_seconds_sum=parts)
        if batches:
            starts = [p.stat().st_mtime-json.loads(p.read_text())['seconds'] for p in batches]
            stops = [p.stat().st_mtime for p in batches]
            phases[phase]['wall_span_estimate_seconds'] = max(stops)-min(starts)
    scoring = {name: json.loads((run/'logs'/f'score_{name}.json').read_text())['seconds']
               for name in ('breadth', 'depth', 'final')}
    result = dict(pipeline=json.loads((run/'pipeline_execution.json').read_text()),
        preprocessing=json.loads((run/'preprocess_execution.json').read_text()),
        resopt_seconds=json.loads((run/'logs/resopt.json').read_text())['seconds'],
        phases=phases, cpu_ranking_seconds=scoring,
        note='Pipeline includes GPU resopt, searches, scoring and bookkeeping; excludes initial CPU sopt, '
             'the separate import, and later audits. Phase wall spans are estimates from batch completion '
             'log mtimes minus recorded process durations. Seed/component sums overlap across workers '
             'and must not be added as wall time. This is not a matched CPU/GPU speedup.')
    write_json(run/'timing_summary.json', result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--archive-dir',type=Path,required=True)
    args=ap.parse_args()
    run=args.run_dir.resolve();archive=args.archive_dir.resolve()
    state=run/'study_status.json'
    write_json(state,dict(stage='waiting for pipeline',updated=time.time()))
    deadline=time.monotonic()+6*3600
    marker=run/'pipeline_execution.json'
    while not marker.exists():
        if time.monotonic()>deadline: raise TimeoutError('pipeline not complete after six hours')
        time.sleep(5)
    if json.loads(marker.read_text())['returncode']: raise RuntimeError('GPU pipeline failed')
    metadata=json.loads((run/'prepared.json').read_text())
    settings=json.loads((run/'scoring.json').read_text())
    cpu_resopt={raw_key(a):b for a,b in iter_sopt(run/'cpu_reference/pipeline_work/resopt_output.txt')}
    gpu_resopt={raw_key(a):b for a,b in iter_sopt(run/'pipeline/resopt_output.txt')}
    shared=cpu_resopt.keys() & gpu_resopt.keys()
    write_json(run/'resopt_comparison.json',dict(cpu_count=len(cpu_resopt),gpu_count=len(gpu_resopt),
        identical_raw_inputs=len(shared),identical_optimized=sum(
            cado_poly_text(cpu_resopt[k])==cado_poly_text(gpu_resopt[k]) for k in shared)))
    output=run/'migration_audit';output.mkdir(exist_ok=True)
    def batch(label,mode,jobs):
        dest=output/label;dest.mkdir(exist_ok=True)
        argv=[]
        for name,options in jobs:
            row=list(map(str,[*options,'-out',dest/(name+'.poly')]))
            if any(any(c in x for c in '\t\r\n') for x in row):raise ValueError('batch delimiter')
            argv.append(row)
        manifest=dest/'jobs.tsv';manifest.write_text('S23_ROPT_BATCH 1\n'+''.join('\t'.join(a)+'\n' for a in argv))
        command=[str(run/'bin'/('s23_ropt_cpu' if mode=='cpu' else 's23_ropt')),'-batch',str(manifest)]
        start=time.monotonic()
        with (dest/'process.log').open('wb') as log:
            p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,timeout=1800)
        if p.returncode:raise RuntimeError(f'{label} failed; see {dest}/process.log')
        write_json(dest/'execution.json',dict(command=command,wall_seconds=time.monotonic()-start))
        print(label,'complete',flush=True)
        return dest,(dest/'process.log').read_text()
    def options(seed,pool,count=4096,keep=200):
        return ['-poly',run/'seeds'/(seed+'.poly'),'-search','-pool-in',pool,'-aw',1.3,
                '-rerank',count,'-refine',keep,'-top',3,'-t',2]
    write_json(state,dict(stage='depth full-alpha audit',updated=time.time()))
    jobs=[(p.stem,options(p.stem,p)) for p in sorted((run/'depth').glob('*.pool'))]
    cpu,_=batch('depth_cpu','cpu',jobs)
    gpu,text=batch('depth_gpu_checked','gpu',[(n,a+['-refine-gpu','-refine-check']) for n,a in jobs])
    depth=[]
    for seed,_ in jobs:
        baseline=run/'depth'/(seed+'.poly');checked=gpu/(seed+'.poly');reference=cpu/(seed+'.poly')
        if baseline.read_bytes()!=checked.read_bytes():raise ValueError('checked full-alpha changes GPU exports')
        cr,gr=read_cado_blocks(reference),read_cado_blocks(checked)
        if len(gr)!=200:raise ValueError('short depth exports')
        for _,p in gr:validate_poly(p,metadata['n'])
        retained=len({coords(l) for l,_ in cr}&{coords(l) for l,_ in gr})
        if retained!=len(cr):raise ValueError(f'{seed}: CPU rotation lost in GPU refinement')
        depth.append(dict(seed=seed,exports=len(gr),same_rotations=retained,
                          byte_identical=reference.read_bytes()==checked.read_bytes(),
                          cpu_sha256=digest(reference),gpu_sha256=digest(checked)))
    audits=re.findall(r'full-refine GPU check: max \|dL\| ([\d.e+-]+), \|dA\| ([\d.e+-]+), \|dkey\| ([\d.e+-]+); (\d+) translations',text)
    if len(audits)!=len(jobs):raise ValueError('missing full-alpha audit')
    record=dict(depth=depth,depth_max_errors={k:max(float(x[i]) for x in audits) for i,k in enumerate(('L','alpha','key'))},
                depth_translation_differences=sum(int(x[3]) for x in audits))
    write_json(output/'summary.json',record)
    write_json(state,dict(stage='MurphyE sampled-pool audit',updated=time.time()))
    bounds=scores.score_bounds(settings['params']);jobs=[]
    for path in sorted((run/'murphy').glob('*.poly')):
        seed=path.stem;source=run/'depth'/(seed+'.pool')
        if not source.exists():source=run/'breadth'/(seed+'.pool')
        wanted={coords(label) for label,_ in read_cado_blocks(path)[:16]}
        lines=source.read_text().splitlines();rows=[l for l in lines[12:-1] if tuple(map(int,l.split()[:2])) in wanted]
        if len(rows)!=16:raise ValueError('missing sampled Murphy starts')
        pool=output/(seed+'.pool');pool.write_text('\n'.join(lines[:11]+['16']+rows+['END','']))
        a=options(seed,pool,16,16)+['-murphy','-Bf',bounds['Bf'],'-Bg',bounds['Bg'],'-area',bounds['area']]
        jobs.append((seed,a))
    cpu,_=batch('murphy_cpu','cpu',jobs)
    gpu,text=batch('murphy_gpu_checked','gpu',[(n,a+['-refine-gpu','-refine-check','-murphy-gpu','-murphy-check']) for n,a in jobs])
    scores.cado_murphy_binary=lambda:str(run/'bin/cado_murphy')
    params=SimpleNamespace(params=settings['params'],threads=4,points=16000,eval_points=256000,useskew=True,trans=0)
    quality=[]
    for seed,_ in jobs:
        values={}
        for mode,folder in [('cpu',cpu),('gpu',gpu)]:
            rows=read_cado_blocks(folder/(seed+'.poly'))
            for _,p in rows:validate_poly(p,metadata['n'])
            evaluated=scores.murphy([p for _,p in rows],metadata['n'],params)
            values[mode]={coords(l):m[0] for (l,_),m in zip(rows,evaluated)}
        if values['cpu'].keys()!=values['gpu'].keys():raise ValueError('sampled Murphy cells differ')
        ratios=[values['gpu'][k]/v for k,v in values['cpu'].items()]
        if min(ratios)<1-2e-6:raise ValueError(f'{seed}: sampled Murphy quality regression')
        quality.append(dict(seed=seed,cells=len(ratios),min_ratio=min(ratios),max_ratio=max(ratios),
                            best_ratio=max(values['gpu'].values())/max(values['cpu'].values())))
    record.update(murphy=quality,murphy_max_integral_relative_error=max(map(float,re.findall(r'max integral relative error ([\d.e+-]+)',text))))
    write_json(output/'summary.json',record)
    write_json(state,dict(stage='independent CPU/GPU comparison',updated=time.time()))
    with (run/'comparison.log').open('wb') as log:
        subprocess.run([sys.executable,str(Path(__file__).with_name('compare.py')),'--run-dir',str(run)],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (run/'winner_validation.log').open('wb') as log:
        subprocess.run([sys.executable,str(Path(__file__).with_name('validate_winners.py')),'--run-dir',str(run)],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    timing_summary(run)
    cpu_unchanged=all(digest(run/p)==h for p,h in metadata['cpu_sha256'].items())
    if not cpu_unchanged:raise ValueError('frozen CPU reference changed')
    if archive.exists():raise ValueError('archive destination exists; choose a new one')
    archive.mkdir(parents=True)
    for src,name in [(run/'comparison/summary.json','comparison.json'),(output/'summary.json','migration_audit.json'),
                     (run/'winner_validation.json','winner_validation.json'),
                     (run/'timing_summary.json','timing_summary.json'),
                     (run/'scoring.json','scoring.json'),(run/'resopt_comparison.json','resopt_comparison.json'),
                     (run/'results/best.poly','pipeline_best.poly')]:shutil.copy2(src,archive/name)
    for src in (run/'comparison').glob('*.poly'):shutil.copy2(src,archive/src.name)
    logs={}
    for phase in ('breadth','depth','murphy'):
        files=sorted((run/'logs'/phase).glob('*.log'))
        logs[phase]=dict(seeds=len(files),full_refine_gpu=sum('full-refine GPU:' in p.read_text() for p in files),
                        full_refine_fallbacks=sum(sum(map(int,re.findall(r'full-refine GPU: .*?, (\d+) host fallbacks',p.read_text()))) for p in files))
        if phase=='murphy':
            logs[phase]['murphy_gpu']=sum('MurphyE GPU:' in p.read_text() for p in files)
            logs[phase]['murphy_cpu_fallbacks']=sum(sum(map(int,re.findall(r'MurphyE GPU: .*?, (\d+) host fallbacks',p.read_text()))) for p in files)
    timing=json.loads((run/'pipeline_execution.json').read_text())
    write_json(archive/'execution.json',dict(n=metadata['n'],initial_mode=metadata['initial_mode'],pipeline=timing,phases=logs,
        initial_count=json.loads((run/'preprocess/info.json').read_text())['count'],cpu_reference_unchanged=cpu_unchanged,
        binary_sha256={p.name:digest(p) for p in (run/'bin').iterdir() if p.is_file() and not p.name.endswith('.json')},
        run_dir=str(run),note='Initial CPU sopt is shared. Pipeline timing excludes import and later audits; no matched full CPU timing or test sieve.'))
    write_json(state,dict(stage='complete',archive=str(archive),updated=time.time()))
    print('C197 study complete:',archive,flush=True)


if __name__=='__main__':main()
