#!/usr/bin/env python3
"""Start/monitor the ordinary AS660 reference search using the frozen pilot build.
The target is the distributed runner's 1.6M admitted hits; in-flight work finishes.
The monitor survives this terminal, records progress, and never submits externally.
"""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
WD=HERE/'reference_search'
def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def start():
    m=json.loads((HERE/'manifest.json').read_text())
    for p,h in m['files'].items():
        if hashlib.sha256((HERE/p).read_bytes()).hexdigest()!=h:
            raise RuntimeError('frozen file changed: '+p)
    WD.mkdir(exist_ok=False)
    (WD/'worktodo.ini').write_text(m['N']+'\n')
    (WD/'stage1_core.ptx').symlink_to(HERE/'bin/stage1_core.ptx')
    (WD/'cub').symlink_to(HERE/'bin/cub',target_is_directory=True)
    args='min_coeff=138600 high_coeff_mult=138600 num_polys=1600000 collengine=gerbicz gpu_mem_mb=1024 stage1_provenance=provenance'
    cmd=[str(HERE/'bin/msieve'),'-v','-g','0','-t','1','-np1','-nps',args,m['N']]
    write(WD/'command.json',cmd)
    write(WD/'manifest.json',dict(parent_manifest='../manifest.json',N=m['N'],start_number=660,
        reference_target=1600000,worker_count=1,gpu_mem_mb=1024,
        rationale='Same max entry caps as default GPU memory on this device; preserve standard adaptive offsets and range enumeration.',
        pilot_corpus='../pilot_candidates.ms',pilot_overlap='Deduplicate pilot/raw corpus at final pool assembly. Pilot q windows are re-searched.',
        created=time.time()))
    with (WD/'monitor.log').open('w') as f:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'monitor'],
            cwd=HERE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    write(WD/'monitor.json',dict(pid=p.pid))
    print('started monitor',p.pid,flush=True)
def monitor():
    cmd=json.loads((WD/'command.json').read_text());start_time=time.time();reader=None;records=0
    with (WD/'process.log').open('w') as f:
        p=subprocess.Popen(cmd,cwd=WD,stdout=f,stderr=subprocess.STDOUT)
        while True:
            rc=p.poll()
            raw=WD/'msieve.dat.ms'
            if reader is None and raw.exists(): reader=raw.open()
            if reader:
                while True:
                    pos=reader.tell();line=reader.readline()
                    if not line: break
                    if not line.endswith('\n'): reader.seek(pos);break
                    if line.startswith('Y0:'): records+=1
            state=dict(state='running' if rc is None else 'exited',pid=p.pid,
                monitor_pid=os.getpid(),started=start_time,updated=time.time(),
                raw_records=records,raw_bytes=raw.stat().st_size if raw.exists() else 0,
                provenance_bytes=(WD/'provenance.w0.jsonl').stat().st_size if (WD/'provenance.w0.jsonl').exists() else 0,
                returncode=rc)
            write(WD/'status.json',state)
            if rc is not None: break
            time.sleep(5)
    if reader: reader.close()
if __name__=='__main__':
    if sys.argv[1]=='start': start()
    elif sys.argv[1]=='monitor': monitor()
    else: print((WD/'status.json').read_text())
