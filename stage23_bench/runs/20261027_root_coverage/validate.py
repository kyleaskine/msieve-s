#!/usr/bin/env python3
"""Check GPU planning against CPU plans and replay each saved pool on the CPU."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

from study import ROOT, audit, pilot, prepare
from polyfmt import read_cado_blocks, resultant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    meta = prepare(run)
    timing = json.loads((run/'pilot/summary.json').read_text())
    if not timing['repeat_outputs_identical']:
        raise ValueError('complete paired timings first')
    folder = run/'validation/replay'
    folder.mkdir(parents=True,exist_ok=True)
    records = []
    for name in pilot.CASES:
        case = next(c for c in meta['cases'] if c['name']==name)
        for mode in pilot.MODES:
            tag = name+'_'+mode+'_r0'
            plan = json.loads((run/'plans'/name/(mode+'.json')).read_text())
            log = (run/'pilot'/(tag+'.log')).read_text()
            classes = [dict(u=int(u),v=int(v),kind=kind) for u,v,kind in re.findall(
                r'^# class u (\d+) v (\d+): .*?, (ranked|exploration)$',log,re.M)]
            if classes != plan['classes']:
                raise ValueError('CPU/GPU selected classes differ: '+tag)
            if classes:
                cells = sum(map(int,re.findall(r'^# class cells u \d+ v \d+: (\d+) selected',log,re.M)))
                if cells != plan['cells']:
                    raise ValueError('CPU/GPU admitted geometry differs: '+tag)
            pool = run/'pilot'/(tag+'.pool')
            expected = run/'pilot'/(tag+'.poly')
            output, logfile, record = [folder/(tag+ext) for ext in ('.poly','.log','.json')]
            command = [str(run/'bin/s23_ropt_cpu'),'-poly',str(run/case['file']),'-search',
                '-aw','1.3','-pool-in',str(pool),'-rerank','4096','-refine','200',
                '-top','8','-t','2','-out',str(output)]
            provenance = {str(p):audit.digest(p) for p in (pool,expected,run/case['file'],run/'bin/s23_ropt_cpu')}
            if record.exists():
                row = json.loads(record.read_text())
                if row['command']!=command or row['provenance']!=provenance or row['returncode']:
                    raise ValueError('changed or failed replay: '+tag)
            else:
                if output.exists() or logfile.exists():
                    raise ValueError('unfinished replay: '+tag)
                start = time.monotonic()
                with logfile.open('wb') as stream:
                    process = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                row = dict(case=name,mode=mode,command=command,returncode=process.returncode,
                    seconds=time.monotonic()-start,provenance=provenance)
                if process.returncode:
                    audit.save(record,row)
                    raise ValueError('CPU replay failed: '+tag)
            if output.read_bytes()!=expected.read_bytes():
                raise ValueError('CPU/GPU refinement differs: '+tag)
            exports = read_cado_blocks(output)
            for _,p in exports:
                value = resultant(p)
                if p['n']!=case['n'] or not value or value%case['n']:
                    raise ValueError('invalid export: '+tag)
            row.update(exports=len(exports),cpu_gpu_bytes_equal=True,classes_equal=True,
                output_sha256=audit.digest(output),log_sha256=audit.digest(logfile))
            audit.save(record,row)
            records.append(row)
            print(tag,len(exports),'identical exports',flush=True)
    audit.save(run/'validation/replay_summary.json',dict(records=records,
        verified_exports=sum(r['exports'] for r in records),all_bytes_equal=True))


if __name__=='__main__':
    main()
