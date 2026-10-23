#!/usr/bin/env python3
"""CPU-only snapshots and matched-effort quality pilot; does not touch the search."""
import hashlib,heapq,json,os,shutil,subprocess,sys,time
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FIRST=HERE.parent/'20261020_as660_first_coefficient'
sys.path.insert(0,str(ROOT/'stage23_bench/tools'))
from stage1_provenance import audit,join_raw,join_sopt
from polyfmt import cado_poly_text,iter_sopt

def save(name,data):
    (HERE/name).write_text(json.dumps(data,indent=2)+'\n')

def snapshot():
    source=ROOT/'as660-stage1-b.w0.jsonl'
    digest=hashlib.sha256(); size=0; marker=b'{"type":"coefficient_end"'
    with source.open('rb') as f,(HERE/'second.w0.jsonl').open('xb') as out:
        pending=b''
        while True:
            chunk=f.read(1024*1024)
            if not chunk: raise RuntimeError('completed coefficient not found')
            pending+=chunk
            pos=pending.find(marker)
            end=pending.find(b'\n',pos) if pos>=0 else -1
            if end>=0:
                ending=json.loads(pending[pos:end]); assert ending['complete']
                piece=pending[:end+1]; out.write(piece);digest.update(piece);size+=len(piece)
                break
            if pos<0:
                piece=pending[:-len(marker)];out.write(piece);digest.update(piece);size+=len(piece)
                pending=pending[-len(marker):]
    # The sole worker's previous coefficient is already complete and drained.
    # Still verify the extracted raw rows against exact candidate identities.
    skip=185201; take=ending['found']; number=0; block=[]
    rawhash=hashlib.sha256()
    with (ROOT/'as660.ms').open('rb') as f,(HERE/'second.ms').open('xb') as out:
        for line in f:
            block.append(line)
            if line.strip(): continue
            if not any(x.startswith(b'n:') for x in block): block=[];continue
            if skip<=number<skip+take:
                data=b''.join(block);out.write(data);rawhash.update(data)
            number+=1;block=[]
            if number==skip+take:break
    assert number==skip+take
    save('snapshot.json',dict(source=str(source),prefix_bytes=size,sha256=digest.hexdigest(),
        boundary=ending,worker_active_at_snapshot=True,synthetic_records_added=False,
        raw_source='as660.ms',raw_start_index=skip,raw_rows=take,raw_sha256=rawhash.hexdigest()))
    print('saved completed coefficient prefix',size,'bytes',flush=True)
    summary={}
    candidates,counts,_,incomplete=audit([HERE/'second.w0.jsonl'],summary=summary)
    assert incomplete==[f'{HERE / "second.w0.jsonl"}: missing worker_end'],incomplete
    counts['unmatched_raw']=join_raw(candidates,[HERE/'second.ms'])
    counts['raw_links']=sum(len(e['raw']) for e in candidates.values())
    assert counts['unmatched_raw']==0 and counts['raw_links']==take
    assert counts['valid_hits']==take
    with (HERE/'ledger.jsonl').open('x') as f:
        for e in candidates.values():f.write(json.dumps(e)+'\n')
    save('audit.json',dict(counts=counts,summary=summary,incomplete=incomplete,
        completed_coefficient_validated=True,
        scope='Exact source prefix through coefficient_end; worker continues in source.'))
    print(json.dumps(dict(counts=counts,incomplete=incomplete)),flush=True)

def prepare():
    (HERE/'bin').mkdir()
    binary=Path('/home/kylea/cado-nfs/build/DESKTOP-3J4UC68/polyselect/sopt')
    shutil.copy2(binary,HERE/'bin/sopt')
    metadata={}
    for ad,source in ((138600,FIRST/'ledger.jsonl'),(277200,HERE/'ledger.jsonl')):
        entries=heapq.nsmallest(10000,(json.loads(l) for l in source.open()),key=lambda e:e['candidate_id'])
        assert len(entries)==10000
        with (HERE/f'sample-{ad}.ms').open('x') as raw,(HERE/f'sample-{ad}.jsonl').open('x') as ledger:
            for e in entries:
                p=e['raw'][0]['polynomial']
                assert all(r['polynomial']==p for r in e['raw'])
                raw.write(cado_poly_text(p)+'\n');ledger.write(json.dumps(e)+'\n')
        metadata[str(ad)]=dict(source_ledger=str(source),sample_count=len(entries),
            selection='10,000 lexicographically smallest SHA-256 candidate IDs; one raw per identity',
            sample_sha256=hashlib.file_digest(open(HERE/f'sample-{ad}.ms','rb'),'sha256').hexdigest())
    save('quality-manifest.json',dict(sopt=str(binary),effort=0,samples=metadata,
        binary_sha256=hashlib.file_digest(open(HERE/'bin/sopt','rb'),'sha256').hexdigest(),
        caution='Exploratory uniform candidate samples. Sopt exp_E is not final Murphy E.'))

def quality():
    statuses=[]
    for ad in (138600,277200):
        cmd=[str(HERE/'bin/sopt'),'-sopteffort','0','-inputpolys',str(HERE/f'sample-{ad}.ms')]
        start=time.monotonic()
        with (HERE/f'sopt-{ad}.out').open('x') as out,(HERE/f'sopt-{ad}.err').open('x') as err:
            result=subprocess.run(cmd,stdout=out,stderr=err,env=dict(os.environ,OMP_NUM_THREADS='1'))
        statuses.append(dict(ad=ad,command=cmd,returncode=result.returncode,seconds=time.monotonic()-start))
        save('quality-status.json',statuses)
        if result.returncode:raise RuntimeError('CPU sopt failed')
        print('CPU sopt finished',ad,flush=True)

if __name__=='__main__':
    {'snapshot':snapshot,'prepare':prepare,'quality':quality}[sys.argv[1]]()
