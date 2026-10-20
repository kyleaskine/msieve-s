"""Reproduce the saved-fixture raw-norm audit; CPU only, no new polyselect runs.

Run: python3 stage23_bench/runs/20261019_stage1_audit/audit.py
Outputs norm_audit.json beside this script. Requires g++ and the repository headers.
"""
import csv,gzip,json,math,subprocess,bisect,collections,sys,tempfile
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[2]
sys.path.insert(0,str(root/'stage23_bench/tools'))
from polyfmt import spearman
scratch=tempfile.TemporaryDirectory(prefix='stage1-norm-audit-')
probe=Path(scratch.name)/'norm_probe'
subprocess.run(['g++','-O2','-std=c++17','-I',str(root/'stage23_gpu/include'),str(here/'norm_probe.cpp'),'-o',str(probe)],check=True)
result={}
for job in ['c146','c161']:
 d=root/'stage23_bench/data'/job
 with gzip.open(d/'sopt_sample.tsv.gz','rt') as f: rows=list(csv.DictReader(f,delimiter='\t'))
 with gzip.open(d/'ropt_seeds.tsv.gz','rt') as f: ropt=list(csv.DictReader(f,delimiter='\t'))
 N=int((d/'n.txt').read_text().strip())
 lines=[]
 for r in rows:
  lines.append('0 '+' '.join(r['raw_c'+str(i)] for i in range(6)))
  lines.append(r['e0_skew']+' '+' '.join(r['e0_c'+str(i)] for i in range(6)))
 p=subprocess.run([str(probe)],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True)
 vals=[tuple(map(float,l.split())) for l in p.stdout.splitlines()]
 assert len(vals)==2*len(rows)
 errors=[]
 for i,r in enumerate(rows):
  r['raw_lognorm']=vals[2*i][1]
  errors.append(abs(vals[2*i+1][1]-float(r['e0_lognorm'])))
  assert math.isfinite(r['raw_lognorm'])
  logm=(math.log(N)-math.log(abs(int(r['raw_c5']))))/5
  r['log_B_proxy']=(logm+3*math.log(max(1,abs(int(r['raw_c3'])))))/4
 assert max(errors)<.006,(job,max(errors))
 groups=collections.defaultdict(list)
 for r in rows:groups[r['Y1']].append(r)
 dup={k:v for k,v in groups.items() if len(v)>1}
 report={'n_sample':len(rows),'random_n':sum(r['set']=='random' for r in rows),'duplicate_Y1_groups':len(dup),'max_norm_validation_error':max(errors),'correlations_random':{},'retention':{},'finalists':{}}
 random=[r for r in rows if r['set']=='random']
 for metric in ['raw_lognorm','log_B_proxy']:
  xs=sorted(r[metric] for r in random)
  report['correlations_random'][metric]={m:spearman([r[metric] for r in random],[float(r[m]) for r in random]) for m in ['e0_lognorm','e0_exp_E']}
  retain=[]
  for frac in [.25,.5,.75,.9,.95]:
   threshold=xs[min(len(xs)-1,int(len(xs)*frac))]
   rr={'retained_random_fraction':frac,'threshold':threshold}
   for k in [100,1000]:
    top=[r for r in rows if int(r['rank_e0'])<=k]
    rr['top'+str(k)+'_kept']=sum(r[metric]<=threshold for r in top)
    rr['top'+str(k)+'_total']=len(top)
   retain.append(rr)
  report['retention'][metric]=retain
 for tool in ['msieve','cado']:
  finalists=[]
  for r in ropt:
   scores=[float(r[k]) for k in [tool+'_orig',tool+'_inv'] if r[k]]
   if scores:finalists.append((max(scores),r))
  finalists.sort(key=lambda x:-x[0]); out=[]
  for score,r in finalists[:10]:
   matches=groups.get(r['Y1'],[])
   item={'score':score,'Y1':r['Y1'],'ropt_input_rank':int(r['rank']),'raw_matches':len(matches)}
   if len(matches)==1:
    raw=matches[0]
    for metric in ['raw_lognorm','log_B_proxy']:
     xs=sorted(x[metric] for x in random)
     item[metric]=raw[metric]
     item[metric+'_random_percentile']=100*bisect.bisect_right(xs,raw[metric])/len(xs)
    item['raw_c5']=raw['raw_c5'];item['raw_c3']=raw['raw_c3'];item['e0_lognorm']=float(raw['e0_lognorm'])
   out.append(item)
  report['finalists'][tool]=out
 result[job]=report
 print(job,json.dumps({k:v for k,v in report.items() if k!='finalists'},indent=2))
 for tool,items in report['finalists'].items():print(tool,'TOP3',json.dumps(items[:3],indent=2))
(here/'norm_audit.json').write_text(json.dumps(result,indent=2)+'\n')
