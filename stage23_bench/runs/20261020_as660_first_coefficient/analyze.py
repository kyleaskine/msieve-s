#!/usr/bin/env python3
"""Summarize the audited first coefficient without rereading its 6.4 GB sidecar."""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'stage23_bench/tools'))
from polyfmt import cado_poly_text


def main():
    ratios=[]; multiplicity=Counter(); hits=Counter(); multiple_q=0
    # Save one exact raw polynomial per candidate for the next CPU sopt pass.
    with (HERE/'unique.ms').open('w') as out:
        for line in (HERE/'ledger.jsonl').open():
            e=json.loads(line); discoveries=e['discoveries']
            ratios.extend(d['ratio'] for d in discoveries)
            multiplicity[len(discoveries)]+=1
            multiple_q+=len({d['q'] for d in discoveries})>1
            hits.update(d['q']>>28 for d in discoveries)
            if e['raw']:
                raw=e['raw'][0]['polynomial']
                if any(r['polynomial']!=raw for r in e['raw']):
                    raise ValueError('multiple raw expansions for one identity')
                out.write(cado_poly_text(raw)+'\n')
    ratios.sort()
    audit=json.loads((HERE/'audit.json').read_text())
    exposure=Counter(); progression_exposure=Counter()
    for e in audit['summary']['exposure']:
        b=e['q_min']>>28
        exposure[b]+=e['qroots']
        progression_exposure[b]+=e['qroots']*e['aprog']
    result=dict(
        ratio_quantiles={str(f):ratios[int(f*(len(ratios)-1))] for f in (0,.5,.9,.99,1)},
        discovery_multiplicity=dict(multiplicity), identities_with_multiple_q=multiple_q,
        q_bins=[dict(q_min=max(1,b<<28),q_max=((b+1)<<28)-1,
            qroots=exposure[b],qroot_progressions=progression_exposure[b],hits=hits[b],
            hits_per_million_qroots=hits[b]/exposure[b]*1e6) for b in sorted(exposure)],
        caveats=['Hits include repeat discoveries. Counts are not polynomial quality.',
                 'Mixed-batch timings are not allocated to hit q values.',
                 'The final 10 q roots had five progressions rather than one.'])
    pairs=[]
    for i in range(5):
        on,off=(HERE/f'pair{i}-{mode}/result.json' for mode in ('on','off'))
        if not on.exists() or not off.exists():
            continue
        on,off=json.loads(on.read_text()),json.loads(off.read_text())
        if on['raw_set_sha256']!=off['raw_set_sha256']:
            raise ValueError('logging changed raw results')
        pairs.append(dict(pair=i,on=on['coefficient_seconds'],off=off['coefficient_seconds'],
            fractional_overhead=on['coefficient_seconds']/off['coefficient_seconds']-1,
            raw_count=on['raw_count'],raw_set_sha256=on['raw_set_sha256']))
    result['logging_pairs']=pairs
    if pairs:
        result['logging_summary']=dict(
            median_fractional_overhead=statistics.median(p['fractional_overhead'] for p in pairs),
            pooled_fractional_overhead=sum(p['on'] for p in pairs)/sum(p['off'] for p in pairs)-1,
            min_fractional_overhead=min(p['fractional_overhead'] for p in pairs),
            max_fractional_overhead=max(p['fractional_overhead'] for p in pairs))
    (HERE/'first-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='q_bins'},indent=2))


if __name__=='__main__':
    main()
