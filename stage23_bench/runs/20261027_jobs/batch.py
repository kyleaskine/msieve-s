#!/usr/bin/env python3
"""Finish the running C170 pipeline, then C184, audits and common comparisons.

GPU pipelines run sequentially. CPU validation of C170 overlaps the C184 GPU
pipeline. Results and failures are persisted, so a detached tool session cannot
silently strand the second prepared input.
"""
import concurrent.futures
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from jobs import ROOT, driver, execute


def main():
    first=ROOT / 'stage23_gpu/runs/20261027_c170'
    second=ROOT / 'stage23_gpu/runs/20261027_c184'
    state=ROOT / 'stage23_gpu/runs/20261027_jobs_status.json'
    script=Path(__file__).with_name('jobs.py')

    def status(stage):
        driver.write_json(state,dict(stage=stage,updated=time.time()))
        print(stage,flush=True)

    try:
        status('waiting for active C170 GPU pipeline')
        marker=first / 'pipeline_execution.json'
        deadline=time.monotonic()+4*3600
        while not marker.exists():
            if time.monotonic()>deadline: raise TimeoutError('C170 pipeline has not finished')
            time.sleep(5)
        if driver.read_json(marker)['returncode']: raise RuntimeError('C170 pipeline failed')
        status('C184 GPU pipeline; C170 CPU audit and comparison')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            tasks=[pool.submit(execute,second,'batch_pipeline',[sys.executable,script,'pipeline','--run-dir',second]),
                   pool.submit(execute,first,'batch_finish',[sys.executable,script,'finish','--run-dir',first])]
            for task in tasks: task.result()
        status('C184 CPU audit and comparison')
        execute(second,'batch_finish',[sys.executable,script,'finish','--run-dir',second])
        for profile in driver.read_json(second / 'comparison_profiles.json')['profiles']:
            status('C184 alternative profile comparison: ' + profile)
            execute(second,'alternative_comparison_'+profile,[sys.executable,Path(__file__).with_name('compare.py'),
                '--run-dir',second,'--profile',profile])
        status('complete')
    except BaseException as error:
        driver.write_json(state,dict(stage='failed',error=str(error),updated=time.time()))
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--background',action='store_true',help='detach the queue and persist its output')
    args=ap.parse_args()
    if args.background:
        log=ROOT / 'stage23_gpu/runs/20261027_jobs_batch.log'
        with log.open('ab') as stream:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve())],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        driver.write_json(ROOT / 'stage23_gpu/runs/20261027_jobs_background.json',
                          dict(pid=proc.pid,started=time.time(),log=str(log)))
        print('Started queue',proc.pid,'log',log)
    else:
        main()
