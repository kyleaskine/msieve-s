#!/usr/bin/env python3
"""Check the actual archive reader on a whole archive and a virtual extract."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'stage23_bench/tools'))
import sidecar_io


def verify(path,expected,size):
    state=hashlib.sha256();total=0
    for block in sidecar_io.chunks(path):state.update(block);total+=len(block)
    assert state.hexdigest()==expected and total==size
    return dict(bytes=total,sha256=state.hexdigest())


manifest=json.loads((HERE/'manifest.json').read_text())
first=manifest['archives'][0]
results={'whole_archive':verify(ROOT/first['archive'],first['source_sha256'],first['source_identity']['bytes'])}
print('Whole archive reader verified',flush=True)
extract=manifest['extracts'][1]  # header + noncontiguous suffix, not just a prefix
with tempfile.TemporaryDirectory() as tmp:
    alias=Path(tmp)/'missing-extract.jsonl'
    registry=Path(tmp)/'registry.json'
    registry.write_text(json.dumps(dict(extracts={str(alias):dict(
        archive=extract['source']+'.zst',ranges=extract['ranges'])})))
    with patch.object(sidecar_io,'REGISTRY',registry):
        results['virtual_extract']=verify(alias,extract['sha256'],extract['identity']['bytes'])
results['virtual_extract']['original_path']=extract['path']
with (HERE/'reader-verification.json').open('x') as out:
    json.dump(results,out,indent=2);out.write('\n')
print('Noncontiguous virtual extract reader verified',flush=True)
