#!/usr/bin/env python3
"""Archive compact evidence only after every coverage and scoring check passes."""
import argparse
import json
from pathlib import Path
import re
import shutil

from audit import ROOT, digest, prepare, save
from pilot import CASES, MODES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--archive-dir',type=Path,required=True)
    args=parser.parse_args()
    run=args.run_dir.resolve(); dest=args.archive_dir.resolve()
    meta=prepare(run)
    plans=json.loads((run/'summary.json').read_text())
    probes=json.loads((run/'proposal_audit/summary.json').read_text())['records']
    timing=json.loads((run/'pilot/summary.json').read_text())
    executions=json.loads((run/'pilot/records.json').read_text())
    quality=json.loads((run/'scores/summary.json').read_text())['results']
    if plans['completed']!=90 or len(probes)!=30 or len(quality)!=len(CASES):
        raise ValueError('study is incomplete')
    crosschecks=0
    for row in plans['records']:
        if not row['variant'].startswith('classes'):
            continue
        modulus,count=map(int,row['variant'][7:].split('_'))
        probe=next(r for r in probes if r['case']==row['case'] and r['modulus']==modulus)
        state=next(r for r in probe['pools'] if r['selected_count']==count)
        if (row['target_result']=='class_proposal') == state['proposed']:
            raise ValueError('proposal instrumentation disagrees with production CLI')
        crosschecks+=1
    for probe in probes:
        for state in probe['pools']:
            if probe['root_proposal_rank']<=4*state['selected_count'] and not state['proposed']:
                raise ValueError('enumeration rank disagrees with top pool')
    refined=0
    for row in executions:
        for name,expected in row['sha256'].items():
            if digest(run/'pilot'/name)!=expected:
                raise ValueError(f'changed GPU artifact: {name}')
        log=(run/'pilot'/f"{row['case']}_{row['mode']}_r{row['repetition']}.log").read_text()
        found=re.search(r'# full-refine GPU: (\d+) cells, alpha bound 2000, (\d+) host fallbacks',log)
        if not found or int(found[1])!=4096 or int(found[2]):
            raise ValueError('unequal refinement effort or unexpected host fallback')
        refined+=int(found[1])
    comparisons=[]
    for result in quality:
        base=result['winners']['dense_breadth']
        base_time=next(r['seconds'] for r in timing['medians'] if r['case']==result['case'] and r['mode']=='dense_breadth')
        for mode in MODES:
            wins=result['winners'][mode]
            elapsed=next(r for r in timing['medians'] if r['case']==result['case'] and r['mode']==mode)
            comparisons.append(dict(case=result['case'],mode=mode,median_seconds=elapsed['seconds'],
                time_ratio_to_dense=elapsed['seconds']/base_time,
                standard_murphy=wins['standard']['standard_murphy'],
                standard_ratio_to_dense=wins['standard']['standard_murphy']/base['standard']['standard_murphy'],
                lattice_ratios_to_dense={s:wins['lattice']['lattice_validation'][s]['score']/
                    base['lattice']['lattice_validation'][s]['score'] for s in ('2','3')},
                model_seconds=elapsed['model_seconds'],sieve_seconds=elapsed['sieve_seconds']))
    sources=[run/name for name in ('prepared.json','summary.json','proposal_audit/summary.json',
                                  'proposal_audit/build.json','pilot/summary.json','pilot/records.json','scores/summary.json')]
    compact=dict(cases=meta['cases'],variants=meta['variants'],
        planning=[{k:r[k] for k in ('case','variant','cells','blocks','budget','minimum','model_seconds',
            'wall_seconds','scanned_lines','admitted_lines','admitted_u_range','ranked_distinct_u',
            'target_result','trace','budget_steps')} for r in plans['records']],
        proposals=[{k:v for k,v in r.items() if k not in ('command',)} for r in probes],
        timings=timing,comparisons=comparisons,
        quality=[{k:r[k] for k in ('case','settings','unique_screened','winners','screen_seconds',
                                 'validation_seconds','note','caveat','provenance')} for r in quality],
        validation=dict(plan_count=90,proposal_rank_audits=30,independent_cli_crosschecks=crosschecks,
            verified_exports=timing['verified_exports'],repeat_outputs_identical=timing['repeat_outputs_identical']),
        provenance={str(p.relative_to(run)):digest(p) for p in sources},
        binary_sha256={k:v for k,v in meta['sha256'].items() if k.startswith('bin/')},
        scripts_sha256={p.name:digest(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.cpp')},
        caveats=['Diagnostic corpus selected from known losses/winners, not an independent holdout.',
            'Three rotated-order repetitions, one GPU process and two host threads per search.',
            'Equal refinement/scoring quotas, measured unequal elapsed times; no equal-time general win claimed.',
            'Independent scoring is separate from reported process times.',
            'Plan admission does not guarantee later retention or quality.',
            'No full-pipeline rerun or measured sieve-yield comparison; defaults unchanged.'])
    compact['validation'].update(gpu_searches=len(executions),gpu_refined_candidates=refined,host_fallbacks=0,
        independent_unique_exports_scored=sum(r['unique_screened'] for r in quality),
        scoring=dict(search_angles=16000,evaluation_angles=64000,validation_angles=256000,
                     lattice_screen=[32,2048,0],lattice_validation=[256,4096,[2,3]]))
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'fixtures').mkdir()
    for case in meta['cases']:
        shutil.copy2(run/case['file'],dest/'fixtures'/(case['name']+'.poly'))
    compact['fixture_sha256']={p.name:digest(p) for p in (dest/'fixtures').iterdir()}
    save(dest/'summary.json',compact)
    print('Archived',dest/'summary.json')
    for r in comparisons:
        print(r['case'],r['mode'],f"{r['median_seconds']:.3f}s",f"Murphy/dense {r['standard_ratio_to_dense']:.6f}",
              'lattice/dense',r['lattice_ratios_to_dense'])


if __name__=='__main__':
    main()
