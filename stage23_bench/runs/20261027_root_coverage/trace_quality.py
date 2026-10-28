#!/usr/bin/env python3
"""Trace the dense MurphyE leader through losing class policies without forcing it."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from study import ROOT, audit, pilot, prepare
from polyfmt import read_cado_blocks


def coordinates(label):
    match = re.match(r'u (-?\d+) v (-?\d+)', label)
    if match is None:
        raise ValueError('missing rotation coordinates: '+label)
    return int(match[1]),int(match[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    prepare(run)
    if args.cases and set(args.cases)-set(pilot.CASES):
        parser.error('unknown case')
    folder = run/'validation/quality_trace'
    folder.mkdir(parents=True,exist_ok=True)
    for name in args.cases or pilot.CASES:
        source = run/'scores'/(name+'.json')
        scores = json.loads(source.read_text())
        baseline = scores['winners']['dense_breadth']['standard']
        target = coordinates(baseline['label'])
        for mode in pilot.MODES:
            ratio = scores['winners'][mode]['standard']['standard_murphy']/baseline['standard_murphy']
            if mode=='dense_breadth' or ratio>=1-1e-6:
                continue
            plan = json.loads((run/'plans'/name/(mode+'.json')).read_text())
            command = list(plan['command'])
            index = command.index('-trace-cell')
            command[index+1:index+3] = list(map(str,target))
            log,record = [folder/(name+'_'+mode+ext) for ext in ('.log','.json')]
            if record.exists():
                row = json.loads(record.read_text())
                if row['command']!=command or row['scores_sha256']!=audit.digest(source) or row['returncode']:
                    raise ValueError('changed or failed quality trace')
                if row['log_sha256']!=audit.digest(log):
                    raise ValueError('changed trace log')
            else:
                if log.exists():
                    raise ValueError('unfinished quality trace')
                with log.open('wb') as stream:
                    process = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                if process.returncode:
                    raise ValueError('quality trace failed: '+str(log))
                text = log.read_text()
                result = re.findall(r'^# trace-cell .*: result (\S+)',text,re.M)
                if len(result)!=1:
                    raise ValueError('missing trace result')
                pool = run/'pilot'/(name+'_'+mode+'_r0.pool')
                lines = pool.read_text().splitlines()
                if lines[0]!='S23_ROPT_POOL 1' or lines[-1]!='END' or len(lines)!=13+int(lines[11]):
                    raise ValueError('unrecognized pool format')
                cells = [tuple(map(int,line.split()[:2])) for line in lines[12:-1]]
                exports = run/'pilot'/(name+'_'+mode+'_r0.poly')
                retained = any(coordinates(label)==target for label,_ in read_cado_blocks(exports))
                loss = result[0]
                if loss=='plan_admitted':
                    loss = 'block_or_proxy_cut' if target not in cells else 'retained' if retained else 'final_refinement_cut'
                row = dict(case=name,mode=mode,target=dict(u=target[0],v=target[1]),
                    murphy_over_dense=ratio,command=command,returncode=0,
                    scores_sha256=audit.digest(source),pool_sha256=audit.digest(pool),
                    exports_sha256=audit.digest(exports),log_sha256=audit.digest(log),
                    trace=[line for line in text.splitlines() if line.startswith('# trace-cell')],
                    first_loss=loss,pool_rank=cells.index(target)+1 if target in cells else None,
                    exported=retained)
                audit.save(record,row)
            print(name,mode,row['first_loss'],'Murphy/dense',ratio,flush=True)
    records = [json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))]
    audit.save(run/'validation/quality_trace.json',dict(records=records,note=__doc__,
        considered_cases=args.cases or list(pilot.CASES),complete=not args.cases))


if __name__=='__main__':
    main()
