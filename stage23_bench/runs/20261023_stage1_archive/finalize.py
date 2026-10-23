#!/usr/bin/env python3
"""Build a small index of all eight coefficient populations and q exposure."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'stage23_bench/tools'))
from sidecar_io import open_sidecar
from stage1_provenance import candidate_id


def main():
    archive=json.loads((HERE/'manifest.json').read_text())
    assert archive['all_verified']
    prepared=json.loads((HERE/'prepared.json').read_text())
    assert prepared['complete']
    numbers={r['run']['N'] for r in archive['archives']}
    assert len(numbers)==1
    n=next(iter(numbers))
    coefficients={}
    for path in HERE.glob('*.exposure.json'):
        data=json.loads(path.read_text())
        by_id={}
        for row in data['metadata']:
            if row['type']=='coefficient':
                by_id[row['coefficient']]=row['ad']
                coefficients[row['ad']]=dict(parameters=row,source=data['source'])
            elif row['type'] in ('setup','coefficient_end'):
                coefficients[by_id[row['coefficient']]][row['type']]=row
        for row in data['log2_exposure']:
            coefficients[row['ad']].setdefault('log2_exposure',[]).append(row)
    for ad in ('180180','360360'):
        data=json.loads((HERE/f'{ad}.audit.json').read_text())
        c=data['summary']['coefficients'][0]
        coefficients[ad]=dict(parameters=c['parameters'],setup=c['setup'],
            coefficient_end=c['ending'],source=Path(c['sidecar']).name,
            audit_counts=data['counts'],log2_exposure=data['summary']['log2_exposure'])
    assert len(coefficients)==8
    for ad,c in coefficients.items():
        artifact=json.loads((HERE/f'{ad}.complete.json').read_text())['artifact']
        c['candidates']=artifact
        assert sum(r['qroots'] for r in c['log2_exposure'])==c['coefficient_end']['qroots_done']
        assert artifact['discoveries']==artifact['raw_links']==c['coefficient_end']['found']
        count=discoveries=raw=0
        with open_sidecar(ROOT/artifact['path']) as stream:
            for line in stream:
                row=json.loads(line)
                assert row['ad']==ad
                assert row['candidate_id']==candidate_id((n,ad,row['P'],row['m']))
                assert all(1<=d['q']<=4294967295 for d in row['discoveries'])
                count+=1;discoveries+=len(row['discoveries']);raw+=row['raw_count']
        assert (count,discoveries,raw)==(artifact['candidates'],artifact['discoveries'],artifact['raw_links'])
    assert sum(c['candidates']['candidates'] for c in coefficients.values())==1687829
    result=dict(schema=1,N=numbers.pop(),coefficients=dict(sorted(coefficients.items(),key=lambda v:int(v[0]))),
        interpretation=[
            'Bin 0 contains q=1 only; bin b>=1 contains 2^b <= q < 2^(b+1).',
            'qroots is searched root occurrences, including unsuccessful work; it is not distinct q or unique root coverage.',
            'progression_qroots weights root occurrences by recorded progression count.',
            'Candidate records preserve every discovery; a candidate may occupy several q bins. Deduplicate or retain ambiguity explicitly.',
            'Do not assign a whole mixed-q batch time to successful hits or infer per-bin cost from root counts.',
            'Identity is SHA-256 of N:ad:P:m. Compact rows omit repeated N; use this index N.',
            'All eight collision populations have independent reconstruction and raw-output joins; first six reuse prior audited ledgers.',
            'The first elapsed time used CLOCK_MONOTONIC under contention; subsequent runs use CLOCK_MONOTONIC_RAW.'
        ])
    with (HERE/'research-index.json').open('x') as out:
        json.dump(result,out,indent=2);out.write('\n')
    print('Verified compact index for eight coefficients, 1,687,829 identities, and',
          sum(c['coefficient_end']['qroots_done'] for c in coefficients.values()),'searched q roots')


if __name__=='__main__':main()
