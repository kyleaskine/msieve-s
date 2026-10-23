#!/usr/bin/env python3
"""Exploratory, candidate-deduplicated CPU sopt comparison; lower exp_E is better."""
from collections import Counter,defaultdict
import json,math,random,statistics,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FIRST=HERE.parent/'20261020_as660_first_coefficient'
sys.path.insert(0,str(ROOT/'stage23_bench/tools'))
from stage1_provenance import join_sopt
from polyfmt import resultant

def quantile(values,q):
    pos=(len(values)-1)*q; i=int(pos); f=pos-i
    return values[i]*(1-f)+values[min(i+1,len(values)-1)]*f

def stats(values):
    values=sorted(values)
    return dict(n=len(values),mean=statistics.mean(values),best=values[0],
                q01=quantile(values,.01),q10=quantile(values,.1),median=quantile(values,.5))

def main():
    result={}; score_arrays=[]
    for ad,folder in ((138600,FIRST),(277200,HERE)):
        audit=json.loads((folder/'audit.json').read_text())
        counts=audit['counts']; batches=counts['batches']
        ratios=[]; raw_hits_by_qbin=Counter(); unique=0; repeated=0
        with (folder/'ledger.jsonl').open() as f:
            for line in f:
                e=json.loads(line);unique+=1;repeated+=len(e['discoveries'])-1
                for d in e['discoveries']:
                    ratios.append(d['ratio']);raw_hits_by_qbin[d['q']>>30]+=1
        exposure=Counter()
        for e in audit['summary']['exposure']:
            exposure[e['q_min']>>30]+=e['qroots']
        candidates={}
        for line in (HERE/f'sample-{ad}.jsonl').open():
            e=json.loads(line);candidates[e['candidate_id']]=e
        assert join_sopt(candidates,[HERE/f'sopt-{ad}.out'])==0
        qvalues=defaultdict(list); timevalues=defaultdict(list); scores=[]
        ambiguous_q=ambiguous_time=0
        with (HERE/f'scores-{ad}.jsonl').open('w') as out:
            for e in candidates.values():
                assert len(e['sopt'])==1
                opt=e['sopt'][0]['polynomial']
                assert resultant(opt)%int(e['N'])==0
                score=opt['exp_E'];scores.append(score)
                qb={d['q']>>30 for d in e['discoveries']}
                tb={min(3,(d['batch']-1)*4//batches) for d in e['discoveries']}
                if len(qb)==1:qvalues[next(iter(qb))].append(score)
                else:ambiguous_q+=1
                if len(tb)==1:timevalues[next(iter(tb))].append(score)
                else:ambiguous_time+=1
                out.write(json.dumps(dict(candidate_id=e['candidate_id'],exp_E=score,
                    q_bins=sorted(qb),chronology_bins=sorted(tb),lognorm=opt['lognorm'],
                    skew=opt['skew']))+'\n')
        assert len(scores)==10000
        score_arrays.append(scores)
        result[str(ad)]=dict(returned_hits=counts['returned_hits'],unique_candidates=unique,
            repeat_discoveries=repeated,repeat_fraction=repeated/counts['returned_hits'],
            admitted_hits=counts['admitted_hits'],qroots=counts['qroots'],
            hits_per_million_qroots=counts['returned_hits']/counts['qroots']*1e6,
            unique_per_million_qroots=unique/counts['qroots']*1e6,
            gate_ratio_max=max(ratios),coefficient=audit['summary']['coefficients'][0],
            sopt=stats(scores),q_bins=[dict(bin=b,qroots=exposure[b],raw_hits=raw_hits_by_qbin[b],
                hits_per_million_qroots=raw_hits_by_qbin[b]/exposure[b]*1e6,
                sample=stats(qvalues[b])) for b in range(4)],
            chronology_bins=[dict(bin=b,sample=stats(timevalues[b])) for b in range(4)],
            sampled_identities_spanning_q_bins=ambiguous_q,
            sampled_identities_spanning_chronology_bins=ambiguous_time)
    # Descriptive bootstrap intervals; no correction for exploratory comparisons.
    rng=random.Random(660); differences=defaultdict(list)
    for _ in range(400):
        a,b=(sorted(rng.choices(s,k=len(s))) for s in score_arrays)
        differences['mean'].append(statistics.mean(b)-statistics.mean(a))
        for name,q in (('median',.5),('q01',.01),('q10',.1)):
            differences[name].append(quantile(b,q)-quantile(a,q))
    result['bootstrap_second_minus_first']={k:dict(
        estimate=result['277200']['sopt'][k]-result['138600']['sopt'][k],
        percentile95=[quantile(sorted(v),.025),quantile(sorted(v),.975)])
        for k,v in differences.items()}
    result['interpretation']=[
        'Lower sopt exp_E is better; it is not final root-optimized Murphy E.',
        'Uniform SHA-256 samples of 10,000 distinct identities per coefficient, same CPU sopt effort 0.',
        'Q bins are equal-width quarters of the uint32 q range; chronology bins are quarters of batch traversal.',
        'Q/chronology tables exclude identities discovered across multiple bins; all remain in the overall comparison.',
        'Raw hit yields include repeat discoveries and measure yield, not polynomial quality.',
        'Bootstrap intervals are descriptive; exploratory bins and best observed scores do not establish an optimum.',
        'Runtime is not a controlled comparison: first run had contention, and host duration clocks differ.']
    (HERE/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{n:v[n] for n in ('unique_candidates','repeat_fraction','hits_per_million_qroots','gate_ratio_max','sopt')}
                     for k,v in result.items() if k.isdigit()},indent=2))
    print(json.dumps(result['bootstrap_second_minus_first'],indent=2))

if __name__=='__main__': main()
