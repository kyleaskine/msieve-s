#!/usr/bin/env python3
"""Independent C197 CPU/GPU ranking under standard MurphyE and the job lattice.

Screen three msieve exports per rational family and every CADO export; validate
leaders at 256k angles and an independent lattice stream. This is a measured
shortlist comparison, not exhaustive rescoring of all ~600k msieve outputs.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from gpu_pipeline import digest, write_json, validate_poly
from polyfmt import cado_poly_text, rotation_between
import score_polys as scores
spec = importlib.util.spec_from_file_location('prior_compare', Path(__file__).parents[1] / '20261019_c192/compare.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def key(p):
    sign = 1 if p['c5'] > 0 else -1
    return (p['Y0'], p['Y1'], *(sign * p[f'c{i}'] for i in range(6)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--cpu-only', action='store_true')
    args = ap.parse_args()
    run = args.run_dir.resolve()
    dest = run / 'comparison'
    dest.mkdir(exist_ok=True)
    n = json.loads((run / 'prepared.json').read_text())['n']
    settings = json.loads((run / 'scoring.json').read_text())
    scores.cado_murphy_binary = lambda: str(run / 'bin/cado_murphy')
    params = SimpleNamespace(params='default', threads=4, points=4000, eval_points=16000,
        trans=0, useskew=False, lattice=settings['lattice'], qband=settings['qband'], nlat=32,npts=2048,latseed=0)
    rows = []
    for group in (['cpu'] if args.cpu_only else ['cpu', 'gpu']):
        sources = [run / 'scoring.json', run / 'bin/cado_murphy']
        if group == 'cpu':
            sources += [run / 'cpu_reference/cado_references.json',
                        run / 'cpu_reference/msieve_ropt_orig.p', run / 'cpu_reference/msieve_ropt_inv.p']
        else:
            sources += [run / 'results/summary.json']
        provenance = {str(p.relative_to(run)):digest(p) for p in sources}
        cache = dest / (group + '_screen.json')
        if cache.exists() and json.loads(cache.read_text())['provenance'] == provenance:
            part = json.loads(cache.read_text())['rows']
        else:
            if group == 'cpu':
                part = json.loads((run / 'cpu_reference/cado_references.json').read_text())
                for side in ('orig','inv'):
                    for i,p in enumerate(prior.msieve_leaders(run / 'cpu_reference' / f'msieve_ropt_{side}.p',n)):
                        part.append(dict(method='msieve_'+side,index=i,poly=p))
            else:
                part = [dict(method='gpu_pipeline', index=i, label=f"{r['phase']}/{r['search']}/{r['rank']}",poly=r['poly'])
                        for i,r in enumerate(json.loads((run / 'results/summary.json').read_text())['finalists'])]
            unique = {}
            for row in part:
                validate_poly(row['poly'],n)
                unique.setdefault(key(row['poly']),row['poly'])
            order = list(unique)
            polys = [unique[k] for k in order]
            print(f'{group}: screening {len(part)} records, {len(polys)} unique configurations',flush=True)
            params.params = 'default'
            standard = scores.murphy(polys,n,params)
            params.params = settings['params']
            lattice = scores.lattice_scores(polys,n,params)
            values = {k:dict(standard_murphy=m[0],standard_skew=m[1],lattice_score=l[0],lattice_skew=l[1])
                      for k,m,l in zip(order,standard,lattice)}
            for row in part: row.update(values[key(row['poly'])])
            write_json(cache,dict(provenance=provenance,rows=part))
        rows += part
    if args.cpu_only:
        print('CPU comparison screen saved',flush=True)
        return
    indices=set()
    for metric in ('standard_murphy','lattice_score'):
        order=sorted(range(len(rows)),key=lambda i:-rows[i][metric])
        # Deduplicate the top configurations before applying the quota.
        seen=set()
        for i in order:
            k=key(rows[i]['poly'])
            if k in seen: continue
            seen.add(k); indices.add(i)
            if len(seen)==16: break
        for method in {r['method'] for r in rows}:
            indices.add(next(i for i in order if rows[i]['method']==method))
    final=[dict(rows[i]) for i in sorted(indices)]
    print('Validating',len(final),'finalists at 256k and independent lattice stream',flush=True)
    params.params='default';params.points=16000;params.eval_points=256000;params.useskew=True
    standard=scores.murphy([dict(r['poly'],skew=r['standard_skew']) for r in final],n,params)
    params.params=settings['params'];params.nlat=256;params.npts=4096;params.latseed=2
    lattice=scores.lattice_scores([dict(r['poly'],skew=r['lattice_skew']) for r in final],n,params)
    for r,m,l in zip(final,standard,lattice):
        r.update(standard_murphy=m[0],lattice_score=l[0],job_murphy=l[2],per_q=l[3])
    winners={}
    for metric in ('standard_murphy','lattice_score'):
        cpu=max((r for r in final if r['method']!='gpu_pipeline'),key=lambda r:r[metric])
        gpu=max((r for r in final if r['method']=='gpu_pipeline'),key=lambda r:r[metric])
        winners[metric]=dict(cpu=cpu,gpu=gpu,ratio=gpu[metric]/cpu[metric],transform=rotation_between(cpu['poly'],gpu['poly']))
        print(metric,'GPU/CPU',winners[metric]['ratio'],'CPU',cpu['method'],'GPU',gpu.get('label'),flush=True)
    final.sort(key=lambda r:-r['lattice_score'])
    result=dict(n=n,settings=settings,screened=len(rows),finalists=final,winners=winners,
                scorer_sha256=digest(run / 'bin/cado_murphy'),
                validation=dict(angles=256000,lattices=256,samples=4096,stream=2),
                note=__doc__+' Initial CPU size optimization was shared; no test sieving performed.')
    write_json(dest / 'summary.json',result)
    for metric,value in winners.items():
        for method in ('cpu','gpu'):
            r=value[method];skew=r['standard_skew' if metric=='standard_murphy' else 'lattice_skew']
            (dest / f'{metric}_{method}.poly').write_text(cado_poly_text(dict(r['poly'],skew=skew)))


if __name__=='__main__':main()
