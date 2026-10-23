#!/usr/bin/env python3
"""Audit the last two AS660 coefficients and preserve compact research data."""
from collections import Counter
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUNS=HERE.parent
sys.path.insert(0,str(ROOT/'stage23_bench/tools'))
from stage1_provenance import audit
from polyfmt import resultant


def save(path,data):
    with path.open('x') as f:
        json.dump(data,f,indent=2); f.write('\n')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


@contextmanager
def packed(path):
    temp=Path(str(path)+'.partial')
    assert not path.exists()
    with temp.open('xb') as out:
        process=subprocess.Popen(['zstd','-q','-3','-T1','-c'],stdin=subprocess.PIPE,stdout=out)
        try:
            yield process.stdin
            process.stdin.close()
            if process.wait(): raise RuntimeError('candidate compression failed')
        except BaseException:
            process.terminate(); process.wait(); raise
    temp.rename(path)


def compact(ad, entries):
    path=HERE/f'{ad}.candidates.jsonl.zst'
    count=hits=raw=0
    with packed(path) as out:
        for e in entries:
            discoveries=[{k:d[k] for k in ('run','worker','coefficient','batch','hit','q','ratio','admitted')}
                         for d in e['discoveries']]
            nraw=e.get('raw_count',len(e['raw']))
            row=dict(candidate_id=e['candidate_id'],ad=str(ad),P=e['P'],m=e['m'],
                     raw_count=nraw,discoveries=discoveries)
            out.write((json.dumps(row,separators=(',',':'))+'\n').encode())
            count+=1;hits+=len(discoveries);raw+=nraw
    # Independently read back the compressed artifact and verify totals.
    from sidecar_io import open_sidecar
    check_count=check_hits=check_raw=0
    with open_sidecar(path) as stream:
        for line in stream:
            row=json.loads(line)
            check_count+=1;check_hits+=len(row['discoveries']);check_raw+=row['raw_count']
    assert (count,hits,raw)==(check_count,check_hits,check_raw)
    return dict(path=str(path.relative_to(ROOT)),sha256=digest(path),bytes=path.stat().st_size,
                candidates=count,discoveries=hits,raw_links=raw)


def iter_raw(path):
    row={}
    pattern=re.compile(r'^(n|Y0|Y1|c\d+):\s*(-?\d+)\s*$')
    with path.open() as f:
        for line in f:
            if not line.strip():
                if row:
                    yield row;row={}
            else:
                m=pattern.match(line)
                if m: row[m[1]]=int(m[2])
    if row: yield row


def new_audit(ad, suffix, expected_hits, expected_unique):
    marker=HERE/f'{ad}.complete.json'
    if marker.exists():return
    source=ROOT/f'as660-stage1-{suffix}.w0.jsonl'
    before=source.stat()
    summary={}
    print(f'{ad}: independent full collision audit started',flush=True)
    entries,counts,_,incomplete=audit([source],summary=summary)
    assert not incomplete,incomplete
    assert counts['returned_hits']==counts['valid_hits']==counts['admitted_hits']==expected_hits
    assert len(entries)==expected_unique
    index={(int(e['P']),-int(e['m'])):e for e in entries.values()}
    assert len(index)==len(entries)
    raw_source=ROOT/'msieve.dat.ms'
    raw_before=raw_source.stat()
    joined=0
    for row in iter_raw(raw_source):
        if row['c5']!=ad:continue
        assert resultant(row)%row['n']==0
        e=index[(row['Y1'],row['Y0'])]
        assert int(e['N'])==row['n']
        e['raw_count']=e.get('raw_count',0)+1
        joined+=1
    assert joined==expected_hits
    assert all(e.get('raw_count')==len(e['discoveries']) for e in entries.values())
    assert (source.stat().st_size,source.stat().st_mtime_ns)==(before.st_size,before.st_mtime_ns)
    assert (raw_source.stat().st_size,raw_source.stat().st_mtime_ns)==(raw_before.st_size,raw_before.st_mtime_ns)
    counts.update(raw_links=joined,unmatched_raw=0)
    save(HERE/f'{ad}.audit.json',dict(counts=counts,summary=summary,incomplete=incomplete))
    info=compact(ad,entries.values())
    save(marker,dict(artifact=info,source=str(source.relative_to(ROOT)),
         source_bytes=before.st_size,source_mtime_ns=before.st_mtime_ns,
         raw_source=str(raw_source.relative_to(ROOT)),raw_size=raw_before.st_size,
         raw_mtime_ns=raw_before.st_mtime_ns))
    print(f'{ad}: audit and compact readback passed: {info}',flush=True)
    del entries,index
    gc.collect()


def old_compact():
    ledgers={138600:RUNS/'20261020_as660_first_coefficient/ledger.jsonl',
             277200:RUNS/'20261021_as660_two_coefficients/ledger.jsonl',
             415800:RUNS/'20261021_as660_third_coefficient/ledger.jsonl'}
    ledgers.update({ad:RUNS/f'20261022_as660_six_coefficients/{ad}/ledger.jsonl'
                    for ad in (554400,693000,831600)})
    for ad,path in ledgers.items():
        marker=HERE/f'{ad}.complete.json'
        if marker.exists():continue
        with path.open() as f:info=compact(ad,map(json.loads,f))
        save(marker,dict(artifact=info,previously_audited_ledger=str(path.relative_to(ROOT)),
                        ledger_sha256=digest(path)))
        print(f'{ad}: compact provenance verified',flush=True)


def old_exposure():
    for name in ('as660-stage1.w0.jsonl','as660-stage1-b.w0.jsonl','as660-stage1-c.w0.jsonl'):
        source=ROOT/name
        marker=HERE/(name+'.exposure.json')
        if marker.exists():continue
        before=source.stat()
        exposure=Counter();counts=Counter();coefficients={};metadata=[];endings={}
        print(f'{name}: log2 q exposure started',flush=True)
        with source.open('rb') as f:
            for lineno,line in enumerate(f,1):
                if line.startswith(b'{"type":"batch",'):
                    row=json.loads(line)
                    ad=coefficients[row['coefficient']]
                    assert len(row['roots'])==row['qroots']
                    for bin_id,n in Counter(q.bit_length()-1 for q,_ in row['roots']).items():
                        exposure[(ad,bin_id,row['aprog'])]+=n
                    counts[ad]+=row['qroots']
                elif any(line.startswith(b'{"type":"'+k+b'",') for k in
                         (b'run',b'coefficient',b'setup',b'coefficient_end')):
                    row=json.loads(line);metadata.append(row)
                    if row['type']=='coefficient':coefficients[row['coefficient']]=row['ad']
                    if row['type']=='coefficient_end':
                        assert row['complete'];endings[coefficients[row['coefficient']]]=row
                if lineno%2000000==0:print(f'{name}: {lineno:,} records',flush=True)
        assert json.loads(line)['type']=='worker_end'
        assert all(counts[ad]==e['qroots_done'] for ad,e in endings.items())
        assert set(endings)==set(counts)
        assert (source.stat().st_size,source.stat().st_mtime_ns)==(before.st_size,before.st_mtime_ns)
        rows=[dict(ad=ad,log2_bin=b,q_min=1<<b,q_max=(1<<(b+1))-1,
                   aprog=a,qroots=n,progression_qroots=n*a) for (ad,b,a),n in sorted(exposure.items())]
        save(marker,dict(source=name,bytes=before.st_size,mtime_ns=before.st_mtime_ns,
                         metadata=metadata,log2_exposure=rows))
        print(f'{name}: exposure totals verified',flush=True)


if __name__=='__main__':
    new_audit(180180,'d',167046,161826)
    new_audit(360360,'e',350615,335886)
    old_compact()
    old_exposure()
    save(HERE/'prepared.json',dict(complete=True,
        note='q=1 has bin 0; each bin b>=1 covers [2^b,2^(b+1)-1]. Exposure counts searched roots, not only hits. No per-q time is inferred.',
        candidates=sum(json.loads(p.read_text())['artifact']['candidates'] for p in HERE.glob('*.complete.json'))))
    print('Preparation complete',flush=True)
