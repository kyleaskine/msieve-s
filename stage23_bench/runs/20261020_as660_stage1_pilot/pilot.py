#!/usr/bin/env python3
"""Frozen, bounded AS660 C196 stage-1 pilot. prepare, run WINDOW, or summarize."""
from pathlib import Path
import collections,hashlib,json,os,shutil,subprocess,sys,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
WINDOWS={'low':(1,100000),'middle':(2147433648,2147533647),'high':(4294867296,4294967295)}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)

def prepare():
    if (HERE/'manifest.json').exists(): raise SystemExit('already prepared')
    n=(ROOT/'worktodo.ini').read_text().strip()
    assert n.isdigit() and len(n)==196
    (HERE/'bin/cub').mkdir(parents=True)
    (HERE/'source').mkdir();(HERE/'tools').mkdir()
    for name in ('msieve','stage1_core.ptx','cub/collision_engine.so','cub/sort_engine.so'):
        shutil.copy2(ROOT/name,HERE/'bin'/name)
    for name in ('stage1_provenance.py','polyfmt.py'):
        shutil.copy2(ROOT/'stage23_bench/tools'/name,HERE/'tools'/name)
    sources=['gnfs/poly/stage1/stage1_core_gpu/stage1_core.cu',
        'gnfs/poly/stage1/stage1_core_gpu/stage1_core.h',
        'gnfs/poly/stage1/stage1_core_gpu/cuda_intrinsics.h',
        'gnfs/poly/stage1/stage1_sieve_gpu.c','gnfs/poly/stage1/stage1_audit.h',
        'gnfs/poly/stage1/stage1.c','gnfs/poly/stage1/stage1_roots.c',
        'gnfs/poly/poly_param.c','cub/collision_engine.cu','cub/collision_engine.h','Makefile']
    for name in sources:
        dest=HERE/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    (HERE/'source/working-tree.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=ROOT))
    (HERE/'worktodo.ini').write_text(n+'\n')
    files={str(p.relative_to(HERE)):sha(p) for folder in ('bin','source','tools') for p in (HERE/folder).rglob('*') if p.is_file()}
    write(HERE/'manifest.json',dict(N=n,digits=196,start_number=660,
        input_source=str(ROOT/'worktodo.ini'),input_sha256=sha(ROOT/'worktodo.ini'),
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        reference=dict(degree=5,min_coeff=138600,high_coeff_mult=138600,num_polys=1600000),
        pilot=dict(coefficient=138600,windows=WINDOWS,aprog=1,gpu_mem_mb=1024,workers=1),
        purpose='Geometry/provenance pilot; every valid candidate retained. Not a quality ranking or controlled speed benchmark.',
        files=files,created=time.time()))
    print('prepared',HERE)

def run(name):
    m=json.loads((HERE/'manifest.json').read_text())
    for p,h in m['files'].items():
        if sha(HERE/p)!=h: raise RuntimeError('frozen file changed: '+p)
    lo,hi=m['pilot']['windows'][name];wd=HERE/name
    wd.mkdir(exist_ok=False)
    (wd/'worktodo.ini').write_text(m['N']+'\n');(wd/'coeff_list.txt').write_text('138600\n')
    (wd/'stage1_core.ptx').symlink_to(HERE/'bin/stage1_core.ptx')
    (wd/'cub').symlink_to(HERE/'bin/cub',target_is_directory=True)
    args=f'coeff_list=1 polydegree=5 collengine=gerbicz gpu_mem_mb=1024 stage1_qmin={lo} stage1_qmax={hi} stage1_aprog=1 stage1_provenance=provenance'
    cmd=[str(HERE/'bin/msieve'),'-v','-g','0','-t','1','-np1','-nps',args,m['N']]
    write(wd/'command.json',cmd)
    with (wd/'gpu-before.csv').open('w') as f:
        subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.used,memory.free,utilization.gpu','--format=csv'],stdout=f,stderr=subprocess.STDOUT)
    start=time.monotonic()
    with (wd/'process.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=wd,stdout=log,stderr=subprocess.STDOUT)
        write(wd/'status.json',dict(state='running',pid=p.pid,started=time.time()))
        rc=p.wait()
    write(wd/'status.json',dict(state='exited',returncode=rc,process_seconds=time.monotonic()-start))
    if rc: raise SystemExit(rc)
    cmd=[sys.executable,str(HERE/'tools/stage1_provenance.py'),str(wd/'provenance.w0.jsonl'),
         '--raw',str(wd/'msieve.dat.ms'),'--ledger',str(wd/'ledger.jsonl'),'--require-complete']
    with (wd/'audit.json').open('w') as f:
        subprocess.run(cmd,stdout=f,check=True)
    status=json.loads((wd/'status.json').read_text());status['state']='validated';write(wd/'status.json',status)
    print(name,(wd/'audit.json').read_text(),flush=True)

def summarize():
    sys.path.insert(0,str(HERE/'tools'))
    from polyfmt import read_cado_blocks,cado_poly_text
    combined={};results={}
    for name in WINDOWS:
        wd=HERE/name
        if not (wd/'audit.json').exists(): continue
        status=json.loads((wd/'status.json').read_text())
        if status['state']!='validated': continue
        rows=[json.loads(l) for l in (wd/'provenance.w0.jsonl').read_text().splitlines()]
        coeff=next(r for r in rows if r['type']=='coefficient')
        setup=next(r for r in rows if r['type']=='setup')
        shapes=collections.Counter((r['group'],r['num_p'],r['roots_per_p'],r['blocks_x'],r['blocks_y'],r['threads'],r['q_tile']) for r in rows if r['type']=='launch')
        batches=[r for r in rows if r['type']=='batch']
        hits=[r for r in rows if r['type']=='hit']
        counts=json.loads((wd/'audit.json').read_text())['counts']
        results[name]=dict(status=status,coefficient=coeff,setup=setup,counts=counts,
            q_batch_sizes=dict(collections.Counter(r['qroots'] for r in batches)),
            shapes=[dict(group=k[0],num_p=k[1],roots_per_p=k[2],grid=[k[3],k[4]],threads=k[5],q_tile=k[6],calls=v) for k,v in shapes.items()],
            gate_ratio_range=[min(r['ratio'] for r in hits),max(r['ratio'] for r in hits)] if hits else None)
        for _,p in read_cado_blocks(wd/'msieve.dat.ms'):
            key=tuple(sorted(p.items()));combined[key]=p
    with (HERE/'pilot_candidates.ms').open('w') as f:
        for p in combined.values(): f.write(cado_poly_text(p)+'\n')
    write(HERE/'summary.json',dict(windows=results,unique_polynomials=len(combined),
        corpus='pilot_candidates.ms',caution='Small stratified pilot; shared-device state not controlled; no q-quality conclusion.'))
    print(json.dumps({k:v['counts'] for k,v in results.items()},indent=2));print('unique polynomials',len(combined))

if __name__=='__main__':
    if sys.argv[1]=='prepare': prepare()
    elif sys.argv[1]=='run': run(sys.argv[2])
    elif sys.argv[1]=='summarize': summarize()
    else: raise SystemExit('prepare | run low/middle/high | summarize')
