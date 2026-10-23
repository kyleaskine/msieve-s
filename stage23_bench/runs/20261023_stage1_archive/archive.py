#!/usr/bin/env python3
"""Verify AS660 zstd archives and register byte-exact virtual extracts.

Normal mode never removes sources. --remove-verified removes only manifest-listed
originals/extracts after preparation, compression, and all integrity checks pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'stage23_bench/tools'))
from sidecar_io import binary_stream, chunks


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,data):
    temp=Path(str(path)+'.tmp')
    with temp.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
    temp.replace(path)


def identity(path):
    s=path.stat()
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,device=s.st_dev,inode=s.st_ino)


def extracts():
    runs=HERE.parent
    specs=[(runs/'20261021_as660_two_coefficients','second.w0.jsonl'),
           (runs/'20261021_as660_third_coefficient','third.w0.jsonl')]
    specs.extend((runs/f'20261022_as660_six_coefficients/{ad}','coefficient.w0.jsonl')
                 for ad in (554400,693000,831600))
    result=[]
    for folder,name in specs:
        metadata=json.loads((folder/'snapshot.json').read_text())
        path=folder/name
        ranges=metadata.get('source_byte_ranges',[[0,metadata.get('prefix_bytes',0)]])
        assert all(0<=a<b for a,b in ranges)
        assert digest(path)==metadata['sha256']
        assert sum(b-a for a,b in ranges)==path.stat().st_size
        result.append(dict(path=str(path.relative_to(ROOT)),source=Path(metadata['source']).name,
                           ranges=ranges,sha256=metadata['sha256'],identity=identity(path),
                           snapshot=str((folder/'snapshot.json').relative_to(ROOT))))
    return result


def pack():
    # Compression is independent of the CPU audit; deletion below requires it.
    sources=[ROOT/name for name in ('as660-stage1.w0.jsonl','as660-stage1-b.w0.jsonl',
        'as660-stage1-c.w0.jsonl','as660-stage1-d.w0.jsonl','as660-stage1-e.w0.jsonl')]
    derived=extracts()
    manifest=dict(schema=1,archives=[],extracts=derived,all_verified=False,removed=[])
    for source in sources:
        marker=HERE/(source.name+'.verified.json')
        if marker.exists():
            info=json.loads(marker.read_text())
            assert identity(source)==info['source_identity']
            assert identity(ROOT/info['archive'])==info['archive_identity']
            manifest['archives'].append(info)
            continue
        before=identity(source)
        with source.open('rb') as f:
            header=json.loads(f.readline());f.seek(-128,2)
            assert json.loads(f.read().splitlines()[-1])['type']=='worker_end'
        original_hash=digest(source)
        target=Path(str(source)+'.zst')
        partial=Path(str(target)+'.partial')
        assert not target.exists() and not partial.exists()
        print(f'{source.name}: compressing {before["bytes"]:,} bytes',flush=True)
        subprocess.run(['zstd','-q','-3','-T1','--keep',str(source),'-o',str(partial)],check=True)
        related=[row for row in derived if row['source']==source.name]
        hashes={r['path']:hashlib.sha256() for r in related}
        full=hashlib.sha256();position=0
        # .partial is not the archive reader's recognized suffix.
        with subprocess.Popen(['zstd','-q','-d','-c',str(partial)],stdout=subprocess.PIPE) as process:
            while data:=process.stdout.read(4*1024*1024):
                full.update(data);end=position+len(data)
                for row in related:
                    for start,stop in row['ranges']:
                        a,b=max(position,start),min(end,stop)
                        if a<b:hashes[row['path']].update(data[a-position:b-position])
                position=end
            assert process.wait()==0
        assert position==before['bytes'] and full.hexdigest()==original_hash
        for row in related:assert hashes[row['path']].hexdigest()==row['sha256']
        assert identity(source)==before
        partial.rename(target)
        info=dict(source=source.name,archive=target.name,source_identity=before,
                  archive_identity=identity(target),source_sha256=original_hash,
                  archive_sha256=digest(target),run=header,
                  verified_extracts=[r['path'] for r in related])
        write(marker,info);manifest['archives'].append(info)
        print(f'{source.name}: decompressed SHA-256 and {len(related)} extract hashes verified',flush=True)
    manifest['all_verified']=True
    manifest['original_bytes']=sum(r['source_identity']['bytes'] for r in manifest['archives'])
    manifest['archive_bytes']=sum(r['archive_identity']['bytes'] for r in manifest['archives'])
    manifest['redundant_extract_bytes']=sum(r['identity']['bytes'] for r in derived)
    write(HERE/'manifest.json',manifest)
    registry=ROOT/'stage23_bench/sidecar_archives.json'
    data=json.loads(registry.read_text()) if registry.exists() else dict(schema=1,extracts={})
    for row in derived:
        data['extracts'][row['path']]=dict(archive=row['source']+'.zst',ranges=row['ranges'],
                                         sha256=row['sha256'],bytes=row['identity']['bytes'])
    write(registry,data)
    print('All archives verified; registry installed. Originals/extracts still present.',flush=True)


def remove_verified():
    manifest=json.loads((HERE/'manifest.json').read_text())
    assert manifest['all_verified']
    assert json.loads((HERE/'prepared.json').read_text())['candidates']==1687829
    compact_records=[json.loads(p.read_text())['artifact'] for p in HERE.glob('*.complete.json')]
    assert len(compact_records)==8 and sum(r['candidates'] for r in compact_records)==1687829
    for row in compact_records:
        assert digest(ROOT/row['path'])==row['sha256']
    registry=json.loads((ROOT/'stage23_bench/sidecar_archives.json').read_text())
    # Recheck compressed bytes before committing to source removal.
    for row in manifest['archives']:
        path=ROOT/row['archive']
        assert identity(path)==row['archive_identity']
        assert digest(path)==row['archive_sha256']
    victims=[(r['source'],r['source_identity']) for r in manifest['archives']]
    victims.extend((r['path'],r['identity']) for r in manifest['extracts'])
    for row in manifest['extracts']:
        mapped=registry['extracts'][row['path']]
        assert mapped['sha256']==row['sha256'] and mapped['ranges']==row['ranges']
        assert mapped['archive']==row['source']+'.zst'
    for name,expected in victims:
        path=ROOT/name
        if path.exists():assert identity(path)==expected
        else:assert name in manifest['removed']
    for name,_ in victims:
        if name in manifest['removed']:continue
        (ROOT/name).unlink()
        manifest['removed'].append(name)
        write(HERE/'manifest.json',manifest)
        print('Removed verified redundant source:',name,flush=True)
    manifest['reclaimed_bytes']=manifest['original_bytes']+manifest['redundant_extract_bytes']-manifest['archive_bytes']
    write(HERE/'manifest.json',manifest)
    print('Net bytes reclaimed:',manifest['reclaimed_bytes'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remove-verified',action='store_true')
    args=parser.parse_args()
    remove_verified() if args.remove_verified else pack()
