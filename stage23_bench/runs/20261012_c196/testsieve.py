#!/usr/bin/env python3
"""testsieve.py RUN [--prepare-only | --resume]: C196 CADO/msieve comparison on the GPU.

Freeze the existing siever binaries, alternate polynomial order across identical
q intervals, verify every relation, and save exact counts and complete timings.
Uses the user's lpbr/lpba=32/33, mfbr/mfba=62/95, limr/lima=200M/300M,
and I=17, J=32768. A three-window screen, not a full-band measurement.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from polyfmt import cado_poly_text, read_cado_blocks, resultant


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    cases = ('cado', 'msieve')
    width, q0s = 4000, (80000000, 264998000, 449996000)
    settings = dict(lpbr=32, lpba=33, mfbr=62, mfba=95, rlim=200000000,
                    alim=300000000, logI=17, J=32768, sq_side=1, width=width,
                    q0s=q0s, q_band=[80000000,450000000])
    if not (run / 'manifest.json').exists():
        run.mkdir(parents=True, exist_ok=False)
        for folder in ('bin','jobs','fb','raw'):
            (run / folder).mkdir()
        source = Path('/home/kylea/code/cuda-sieve/bench')
        for name in ('bench','fbgen_gpu','normscan'):
            shutil.copy2(source / name, run / 'bin' / name)
        for name in cases:
            p = read_cado_blocks(ROOT / f'stage23_bench/data/c196/winners/{name}_best.poly')[0][1]
            if not resultant(p) or resultant(p) % p['n']:
                raise ValueError(f'invalid {name} polynomial')
            job = cado_poly_text(p) + '\n' + ''.join(f'{k}: {settings[k]}\n' for k in
                                                    ('rlim','alim','lpbr','lpba','mfbr','mfba'))
            (run / f'jobs/{name}.job').write_text(job)
        manifest = dict(settings=settings, created=time.time(), source=str(source),
                        sha256={str(p.relative_to(run)):sha(p) for d in ('jobs','bin') for p in (run/d).iterdir()})
        (run / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['settings'] != json.loads(json.dumps(settings)):
        raise ValueError('prepared run has different settings')
    for path, digest in manifest['sha256'].items():
        if sha(run / path) != digest:
            raise ValueError(f'frozen input changed: {path}')
    if '--prepare-only' in sys.argv:
        print(run)
        return
    if (run / 'status.json').exists() and '--resume' not in sys.argv:
        raise ValueError('run already started; choose a fresh directory')

    def state(stage, **extra):
        data = dict(stage=stage,updated=time.time(),**extra)
        (run / 'status.json').write_text(json.dumps(data,indent=2)+'\n')
        print(stage,extra,flush=True)

    def command(tag,args):
        record=run/f'{tag}.command.json'
        if '--resume' in sys.argv and record.exists():
            previous=json.loads(record.read_text())
            if previous['returncode']==0 and previous['argv']==list(map(str,args)) and (run/f'{tag}.out').exists():
                return
        start=time.time()
        with (run / f'{tag}.out').open('w') as log:
            result=subprocess.run(list(map(str,args)),cwd=run,stdout=log,stderr=subprocess.STDOUT)
        (run/f'{tag}.command.json').write_text(json.dumps(dict(argv=list(map(str,args)),
            returncode=result.returncode,elapsed_seconds=time.time()-start),indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'{tag} failed ({result.returncode}); see {tag}.out')

    try:
        for name in cases:
            state('preparing factor base',poly=name)
            command(f'fb/{name}_normscan',[run/'bin/normscan','--poly',run/f'jobs/{name}.job',
                '--sq-side',1,'--logI',17,'--J',32768,'--qmin',80000000,'--qmax',450000000,'--samples',400000])
            command(f'fb/{name}_fbgen',[run/'bin/fbgen_gpu','--poly',run/f'jobs/{name}.job',
                '--lim',300000000,'--maxbits',17,'--out',run/f'fb/{name}.m17'])
        rows=[]
        for k,q0 in enumerate(q0s):
            order=cases if k%2==0 else cases[::-1]
            for name in order:
                tag=f'raw/{name}_{q0}'
                state('sieving',poly=name,q0=q0,completed=len(rows),total=len(q0s)*len(cases))
                command(tag,[run/'bin/bench','--pipeline','--cofactor','--poly',run/f'jobs/{name}.job',
                    '--fb1',run/f'fb/{name}.m17','--sq-side',1,'--logI',17,'--J',32768,
                    '--qrange',f'{q0}:{q0+width-1}','--relations',run/f'{tag}.dat',
                    '--log',run/f'{tag}.log','--log-every',1])
                text=(run/f'{tag}.out').read_text()
                pairs=int(re.search(r'--- band of (\d+)',text)[1])
                rels=int(re.findall(r'total relations\s+(\d+)',text)[-1])
                ms=float(re.findall(r'wall clock per q, COMPLETE\s+([\d.]+)',text)[-1])
                if len(re.findall(r'factors x cofactor == norm.*PASS',text))!=2:
                    raise RuntimeError(f'{tag}: missing norm verification')
                command(tag+'_verify',[run/'bin/bench','--poly',run/f'jobs/{name}.job',
                    '--check-relations',run/f'{tag}.dat'])
                coords=[line.split(':',1)[0] for line in (run/f'{tag}.dat').read_text().splitlines()]
                if len(coords)!=rels or pairs<=0 or ms<=0:
                    raise RuntimeError(f'{tag}: invalid relation/pair counts')
                unique=len(set(coords))
                row=dict(poly=name,q0=q0,width=width,pairs=pairs,relations=rels,
                    unique_relations=unique,duplicates=rels-unique,
                    rel_per_pair=unique/pairs,ms_per_pair=ms,rel_per_second=unique/pairs*1000/ms,
                    verified=True,relation_sha256=sha(run/f'{tag}.dat'))
                rows.append(row)
                (run/'measurements.json').write_text(json.dumps(dict(settings=settings,rows=rows),indent=2)+'\n')
                print(row,flush=True)
        by={name:sorted((r for r in rows if r['poly']==name),key=lambda r:r['q0']) for name in cases}
        def integral(name,key):
            values=[(r['q0'],r[key]/math.log(r['q0'])) for r in by[name]]
            return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(values,values[1:]))+width*values[-1][1]
        projected={name:dict(relations=integral(name,'rel_per_pair'),seconds=integral(name,'ms_per_pair')/1000) for name in cases}
        result=dict(settings=settings,rows=rows,projections=projected,
            cado_over_msieve_per_q=[a['rel_per_pair']/b['rel_per_pair'] for a,b in zip(by['cado'],by['msieve'])],
            cado_over_msieve_band=projected['cado']['relations']/projected['msieve']['relations'],
            cado_over_msieve_throughput=(projected['cado']['relations']/projected['cado']['seconds']) /
                                      (projected['msieve']['relations']/projected['msieve']['seconds']),
            caveat='Three short q windows; normalized by observed (q,rho) pairs. Whole-band estimates use a trapezoid. '
                   'Alternating order, but unreplicated timing under shared-host load.')
        (run/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        state('complete',verified_relations=sum(r['relations'] for r in rows),
              cado_over_msieve_band=result['cado_over_msieve_band'],
              cado_over_msieve_throughput=result['cado_over_msieve_throughput'])
    except BaseException as error:
        state('failed',error=str(error))
        raise


if __name__=='__main__':
    main()
