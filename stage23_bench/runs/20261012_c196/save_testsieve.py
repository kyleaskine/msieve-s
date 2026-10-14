#!/usr/bin/env python3
"""save_testsieve.py RUN: preserve the C196 CADO/msieve test-sieve measurements."""
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
run=Path(sys.argv[1]).resolve()
status=json.loads((run/'status.json').read_text())
if status['stage']!='complete':
    raise SystemExit('test sieves have not completed')
result=json.loads((run/'results.json').read_text())
manifest=json.loads((run/'manifest.json').read_text())
model=json.loads((ROOT/'stage23_bench/data/c196/gpu_ropt/finalist_scores.json').read_text())
reference={name:next(r for r in model['rows'] if r['label'].startswith(prefix)) for name,prefix in
           [('cado','cado_orig '),('msieve','msieve_orig ')]}
result['model_cado_over_msieve_band']=reference['cado']['band']/reference['msieve']['band']
result['model_cado_over_msieve_per_q']=[reference['cado']['relations_per_q'][i]/
                                     reference['msieve']['relations_per_q'][i] for i in (0,3,6)]
result['model_note']='Prior lattice model at 80M, 265M, 450M; test-window starts differ by up to 4000.'
result['verified_relations']=sum(r['relations'] for r in result['rows'])
result['duplicates_removed']=sum(r['duplicates'] for r in result['rows'])
result['unique_relations_within_windows']=sum(r['unique_relations'] for r in result['rows'])
result['skews']={k:reference[k]['skew'] for k in reference}
result['engine_compiled_commits']=sorted({re.search(r'^# commit\s+(.*)$',p.read_text(),re.M)[1]
                                         for p in (run/'raw').glob('*.log')})
result['manifest']=manifest
result['raw_artifacts']=str(run)
dest=ROOT/'stage23_bench/data/c196/testsieve'
dest.mkdir(exist_ok=False)
for directory in ('jobs','logs'):
    (dest/directory).mkdir()
for p in (run/'jobs').glob('*.job'):
    shutil.copy2(p,dest/'jobs'/p.name)
for p in (run/'raw').iterdir():
    if p.suffix in ('.out','.log','.json'):
        shutil.copy2(p,dest/'logs'/p.name)
for p in (run/'fb').glob('*normscan.out'):
    shutil.copy2(p,dest/'logs'/p.name)
(dest/'measurements.json').write_text(json.dumps(result,indent=2)+'\n')
with (dest/'measured.tsv').open('w') as out:
    out.write('poly\tq0\tpairs\traw_relations\tunique_relations\tunique_rel_per_pair\tms_per_pair\tunique_rel_per_second\n')
    for r in result['rows']:
        out.write('\t'.join(str(r[k]) for k in ('poly','q0','pairs','relations','unique_relations',
                                                 'rel_per_pair','ms_per_pair','rel_per_second'))+'\n')
print(dest)
print('CADO/msieve yield ratios by q:',result['cado_over_msieve_per_q'])
print('CADO/msieve band yield:',result['cado_over_msieve_band'])
print('CADO/msieve throughput:',result['cado_over_msieve_throughput'])
print('Verified:',result['verified_relations'],'duplicates removed:',result['duplicates_removed'])
