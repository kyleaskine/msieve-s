#!/usr/bin/env python3
"""Bounded review regressions and C196 timing. Never resumes production."""
from pathlib import Path
import hashlib,json,re,resource,shutil,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'stage23_bench/tools'))
from stage1_provenance import audit
from polyfmt import read_cado_blocks


def main():
    out=Path(sys.argv[1]).resolve(); out.mkdir()
    (out/'bin/cub').mkdir(parents=True)
    files=('msieve','stage1_core.ptx','cub/sort_engine.so','cub/collision_engine.so')
    for name in files: shutil.copy2(ROOT/name,out/'bin'/name)
    hashes={name:hashlib.sha256((out/'bin'/name).read_bytes()).hexdigest() for name in files}
    (out/'manifest.json').write_text(json.dumps(dict(files=hashes),indent=2)+'\n')
    n157=(ROOT/'profiling/msieve.gerbicz.dat.ms').read_text().splitlines()[0].split(': ')[1]
    n196=json.loads((ROOT/'stage23_bench/runs/20261020_as660_stage1_pilot/manifest.json').read_text())['N']
    results=[]

    def run(tag,n,ad,options='',failure=False):
        wd=out/tag; wd.mkdir()
        (wd/'cub').symlink_to(out/'bin/cub',target_is_directory=True)
        (wd/'stage1_core.ptx').symlink_to(out/'bin/stage1_core.ptx')
        (wd/'coeff_list.txt').write_text(str(ad)+'\n')
        args=f'coeff_list=1 polydegree=5 collengine=gerbicz {options}'
        cmd=['stdbuf','-oL',str(out/'bin/msieve'),'-v','-t','1','-g','0','-np1','-nps',args,n]
        (wd/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        def limited():
            signal.signal(signal.SIGXFSZ,signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE,(65536,65536))
        start=end=None
        with (wd/'stdout.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=wd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,preexec_fn=limited if failure else None)
            for line in proc.stdout:
                log.write(line)
                if f'coeff {ad} norm ' in line: start=(time.monotonic(),time.time())
                if f'coeff {ad}: found ' in line: end=(time.monotonic(),time.time())
            rc=proc.wait(timeout=120)
        rawpath=wd/'msieve.dat.ms'
        raw=[p for _,p in read_cado_blocks(rawpath)] if rawpath.exists() else []
        result=dict(tag=tag,returncode=rc,raw_count=len(raw),
            raw_sha256=hashlib.sha256('\n'.join(sorted(json.dumps(p,sort_keys=True) for p in raw)).encode()).hexdigest())
        if start and end:
            result.update(monotonic_seconds=end[0]-start[0],wall_seconds=end[1]-start[1])
        sidecar=wd/'provenance.w0.jsonl'
        if sidecar.exists():
            summary={}; _,counts,_,incomplete=audit([sidecar],summary=summary)
            result.update(counts=dict(counts),incomplete=incomplete,sidecar_bytes=sidecar.stat().st_size)
            (wd/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
            with sidecar.open() as f: header=json.loads(next(f))
            assert header['ptx_sha256']==hashes['stage1_core.ptx']
            assert Path(header['ptx_path']).resolve()==(out/'bin/stage1_core.ptx').resolve()
            if not failure:
                assert not incomplete
                for t in summary['timing']:
                    total=sum(t[k] for k in ('generation_seconds','audit_prepare_seconds',
                                            'search_seconds','readback_submit_seconds'))
                    assert abs(total-t['seconds'])<t['batches']*3e-9
                assert counts['batches']==sum(x['calls'] for x in summary['launches'] if x['group']==0)
        if failure:
            log=(wd/'stdout.log').read_text()
            assert rc==1 and 'queued polynomial work drained' in log,log
            found=int(re.search(rf'coeff {ad}: found (\d+) polys',log)[1])
            assert found>0 and found==len(raw),(found,len(raw))
            assert result['incomplete']
        elif 'stage1_provenance=bad' in args:
            assert rc==1 and 'invalid stage1 audit options' in (wd/'stdout.log').read_text()
            assert start is None
        else: assert rc==0 and end is not None,(rc,tag)
        results.append(result)
        (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return wd,result

    run('comma',n157,420,'stage1_provenance=bad/c196,ad138600')
    run('equals',n157,420,'stage1_provenance=bad/min_coeff=99')
    run('write_failure',n157,420,'gpu_mem_mb=128 stage1_qmin=100000000 '
        'stage1_qmax=100200000 stage1_aprog=1 stage1_provenance=provenance',failure=True)
    n100=n157[:99]+'9'  # synthetic input for the small-p arithmetic path
    width=[]
    for mode in ('auto','fixed'):
        wd,_=run('width32_'+mode,n100,420,'stage1_norm=1e15 stage2_norm=1e15 '
            'poly_deadline=3600 gpu_mem_mb=128 stage1_qmin=1 '
            'stage1_qmax=5000000 stage1_provenance=provenance'+
            (' stage1_aprog=1' if mode=='fixed' else ''))
        with (wd/'provenance.w0.jsonl').open() as f:
            first=next(json.loads(l) for l in f if '"type":"batch"' in l)
        assert first['key_bits']<=32
        width.append(first)
    assert width[0]['qroots']==width[1]['qroots']
    assert width[0]['roots']==width[1]['roots']
    common='stage1_qmin=2140000000 stage1_qmax=2142000000 stage1_aprog=1'
    run('warmup',n196,138600,common)
    for i in range(3):
        for on in ([False,True] if i%2==0 else [True,False]):
            run(f'pair{i}_'+('on' if on else 'off'),n196,138600,common+
                (' stage1_provenance=provenance' if on else ''))
    pairs=[]
    for i in range(3):
        on=next(r for r in results if r['tag']==f'pair{i}_on')
        off=next(r for r in results if r['tag']==f'pair{i}_off')
        assert on['raw_sha256']==off['raw_sha256']
        pairs.append(on['monotonic_seconds']/off['monotonic_seconds']-1)
    (out/'PASS').write_text(json.dumps(dict(logging_overhead_pairs=pairs),indent=2)+'\n')
    print('review regressions passed',pairs,flush=True)


if __name__=='__main__': main()
