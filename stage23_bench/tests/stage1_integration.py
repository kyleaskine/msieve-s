#!/usr/bin/env python3
"""Bounded GPU correctness checks; run only when small GPU tests are acceptable.
Creates an isolated run directory; never treats timings as performance evidence.
Usage: python3 stage23_bench/tests/stage1_integration.py /tmp/new-directory
"""
from pathlib import Path
from collections import Counter
import hashlib,json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'stage23_bench/tools'))
from polyfmt import read_cado_blocks
from stage1_provenance import audit,join_raw

def main():
    r=Path(sys.argv[1]).resolve(); r.mkdir(exist_ok=False)
    (r/'cub').mkdir()
    for name in ('msieve','stage1_core.ptx','cub/sort_engine.so','cub/collision_engine.so'):
        shutil.copy2(ROOT/name,r/name)
    n=(ROOT/'profiling/msieve.gerbicz.dat.ms').read_text().splitlines()[0].split(': ')[1]
    manifest={'N':n,'purpose':'correctness only; GPU may be shared',
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff'],cwd=ROOT)).hexdigest(),
        'binaries':{name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in
            ('msieve','stage1_core.ptx','cub/sort_engine.so','cub/collision_engine.so')}}
    (r/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def run(tag,lo,hi,aprog=1,logging=True,engine='gerbicz',threads=1,admax=420):
        args=(f'polydegree=5 min_coeff=420 max_coeff={admax} high_coeff_mult=420 '
            f'collengine={engine} gpu_mem_mb=128 stage1_qmin={lo} stage1_qmax={hi}')
        if aprog: args+=f' stage1_aprog={aprog}'
        if logging: args+=f' stage1_provenance={tag}'
        cmd=[str(r/'msieve'),'-v','-t',str(threads),'-s',tag,'-np1','-nps',args,n]
        (r/f'{tag}.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        with (r/f'{tag}.log').open('w') as log:
            subprocess.run(cmd,cwd=r,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
        raw=Counter(tuple(sorted(p.items())) for _,p in read_cado_blocks(r/f'{tag}.ms'))
        if logging:
            candidates,counts,work,incomplete=audit(sorted(r.glob(f'{tag}.w*.jsonl')),streaming=False)
            assert not incomplete,(tag,incomplete)
            assert join_raw(candidates,[r/f'{tag}.ms'])==0
            result={'counts':dict(counts),'raw_count':sum(raw.values()),'incomplete':incomplete}
            (r/f'{tag}.summary.json').write_text(json.dumps(result,indent=2)+'\n')
        else: work=Counter()
        print(tag,'raw',sum(raw.values()),flush=True)
        return raw,work
    full,work=run('full',100000000,100200000)
    assert sum(full.values())>0,'test needs actual hits'
    off,_=run('off',100000000,100200000,logging=False)
    assert full==off,'logging changed candidates'
    left,lwork=run('left',100000000,100099999)
    right,rwork=run('right',100100000,100200000)
    assert not (lwork & rwork),'overlapping q/root work'
    assert work==lwork+rwork,'split windows omit or add q/root work'
    assert full==left+right,'split windows changed fixed-offset candidates'
    sort,_=run('sort',100000000,100200000,engine='sort')
    assert full==sort,'sort/Gerbicz candidate mismatch'
    multi,_=run('multi',100000000,100020000,aprog=3)
    multi_off,_=run('multi_off',100000000,100020000,aprog=3,logging=False)
    assert multi==multi_off,'multi-progression logging parity'
    auto,_=run('auto',1,5000,aprog=0)
    auto_off,_=run('auto_off',1,5000,aprog=0,logging=False)
    assert auto==auto_off,'adaptive-progression logging parity'
    run('workers',1,5000,threads=2,admax=840)
    run('empty',4294967295,4294967295)
    (r/'PASS').write_text('Correctness checks passed; no timing conclusions.\n')
    print('all stage1 integration checks passed',flush=True)

if __name__=='__main__': main()
