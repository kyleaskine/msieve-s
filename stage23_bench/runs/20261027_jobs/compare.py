#!/usr/bin/env python3
"""Independent new-job CPU/GPU ranking under standard MurphyE and the job lattice.

Screen three distinct msieve leaders per rational family, every content-reduced
msieve export, and every CADO export; validate
leaders at 256k angles and an independent lattice stream. This is a measured
shortlist comparison, not exhaustive rescoring of all msieve outputs.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from gpu_pipeline import digest, write_json, validate_poly
import sieve_profiles
from polyfmt import cado_poly_text, rotation_between
import score_polys as scores
from final_scoring import (score_candidates, winner_configuration, msieve_shortlist,
                           polynomial_key, normalize_row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--cpu-only', action='store_true')
    ap.add_argument('--profile', help='rescore the same finalists under another historical profile')
    ap.add_argument('--output-dir', type=Path, help='alternate comparison directory inside the run directory')
    args = ap.parse_args()
    run = args.run_dir.resolve()
    dest = (run / (args.output_dir or
                   Path('comparison_' + args.profile if args.profile else 'comparison'))).resolve()
    if not dest.is_relative_to(run) or dest == run:
        ap.error('--output-dir must be a subdirectory of --run-dir')
    dest.mkdir(parents=True, exist_ok=True)
    n = json.loads((run / 'prepared.json').read_text())['n']
    settings = json.loads((run / 'scoring.json').read_text())
    if args.profile:
        catalog = run / 'inputs/sieve_profiles.json'
        if not catalog.exists(): catalog = sieve_profiles.CATALOG
        settings = sieve_profiles.scoring(next(p for p in sieve_profiles.load(catalog) if p['id'] == args.profile), catalog)
    write_json(dest / 'scoring.json', settings)
    scores.cado_murphy_binary = lambda: str(run / 'bin/cado_murphy')
    params = SimpleNamespace(params='default', threads=4, points=4000, eval_points=16000,
        trans=0, useskew=False, lattice=settings['lattice'], qband=settings['qband'], nlat=32,npts=2048,latseed=0)
    rows = []
    for group in (['cpu'] if args.cpu_only else ['cpu', 'gpu']):
        sources = [dest / 'scoring.json', run / 'bin/cado_murphy']
        if group == 'cpu':
            sources += [run / 'cpu_reference/cado_references.json',
                        run / 'cpu_reference/msieve_ropt_orig.p', run / 'cpu_reference/msieve_ropt_inv.p']
        else:
            sources += [run / 'results/summary.json']
        provenance = {str(p.relative_to(run)):digest(p) for p in sources}
        provenance['comparison_policy'] = 'primitive-declared-and-proposed-skews-content-reserve-v3'
        provenance['comparison_script'] = digest(Path(__file__).resolve())
        provenance['scoring_script'] = digest(ROOT / 'stage23_bench/tools/final_scoring.py')
        provenance['scorer_interface'] = digest(ROOT / 'stage23_bench/tools/score_polys.py')
        cache = dest / (group + '_screen.json')
        if cache.exists() and json.loads(cache.read_text())['provenance'] == provenance:
            part = json.loads(cache.read_text())['rows']
        else:
            if group == 'cpu':
                part = json.loads((run / 'cpu_reference/cado_references.json').read_text())
                for side in ('orig','inv'):
                    part.extend(dict(method='msieve_'+side, **row) for row in
                                msieve_shortlist(run / 'cpu_reference' / f'msieve_ropt_{side}.p', n))
            else:
                part = [dict(method='gpu_pipeline', index=i,
                             label=f"{r['phase']}/{r['search']}/{r['rank']}/{r.get('configuration', 'declared')}",
                             configuration=r.get('configuration', 'declared'),
                             source_content=r.get('source_content', 1), poly=r['poly'])
                        for i,r in enumerate(json.loads((run / 'results/summary.json').read_text())['finalists'])]
            part = [normalize_row(row, n) for row in part]
            for row in part:
                validate_poly(row['poly'],n)
            configurations = {(polynomial_key(r['poly']), r['poly'].get('skew')) for r in part}
            print(f'{group}: screening {len(part)} records, {len(configurations)} distinct input configurations',flush=True)
            params.params = settings['params']
            part = score_candidates(part,n,params,propose=True)
            write_json(cache,dict(provenance=provenance,rows=part))
        rows += part
    if args.cpu_only:
        print('CPU comparison screen saved',flush=True)
        return
    indices=set()
    for metric in ('standard_murphy','lattice_score'):
        order=sorted(range(len(rows)),key=lambda i:-rows[i][metric])
        # Reserve distinct coefficient sets; retained rows carry their candidate skews.
        seen=set()
        for i in order:
            k=polynomial_key(rows[i]['poly'])
            if k in seen: continue
            seen.add(k); indices.add(i)
            if len(seen)==16: break
        for method in {r['method'] for r in rows}:
            indices.add(next(i for i in order if rows[i]['method']==method))
    final=[dict(rows[i]) for i in sorted(indices)]
    print('Validating',len(final),'finalists at 256k and independent lattice stream',flush=True)
    params.points=16000;params.eval_points=256000;params.useskew=True
    params.params=settings['params'];params.nlat=256;params.npts=4096;params.latseed=2
    final=score_candidates(final,n,params)
    winners={}
    for metric in ('standard_murphy','lattice_score'):
        cpu=max((r for r in final if r['method']!='gpu_pipeline'),key=lambda r:r[metric])
        gpu=max((r for r in final if r['method']=='gpu_pipeline'),key=lambda r:r[metric])
        winners[metric]=dict(cpu=winner_configuration(cpu,metric),gpu=winner_configuration(gpu,metric),
                            ratio=gpu[metric]/cpu[metric],transform=rotation_between(cpu['poly'],gpu['poly']))
        print(metric,'GPU/CPU',winners[metric]['ratio'],'CPU',cpu['method'],'GPU',gpu.get('label'),flush=True)
    final.sort(key=lambda r:-r['lattice_score'])
    result=dict(n=n,settings=settings,screened=len(rows),finalists=final,winners=winners,
                configuration_policy='Normalize every producer to primitive coefficients; retain declared and proposed skews through fixed-configuration independent validation; report declared_scores separately.',
                msieve_selection='Three distinct printed-score leaders per rational family plus all distinct content-reduced exports; normalize before common scoring.',
                scorer_sha256=digest(run / 'bin/cado_murphy'),
                validation=dict(angles=256000,lattices=256,samples=4096,stream=2),
                note=__doc__+' Initial size optimization ran independently on GPU; no test sieving performed.')
    write_json(dest / 'summary.json',result)
    for metric,value in winners.items():
        for method in ('cpu','gpu'):
            r=value[method];skew=r['standard_skew' if metric=='standard_murphy' else 'lattice_skew']
            (dest / f'{metric}_{method}.poly').write_text(cado_poly_text(dict(r['poly'],skew=skew)))


if __name__=='__main__':main()
