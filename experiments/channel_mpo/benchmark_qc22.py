"""QC-22 isolated-process verifier benchmark with process-tree memory accounting.

Linux /proc smaps_rollup samples are estimates. They measure PSS, USS and
summed RSS across a parent checker and its worker descendants. Timings are
single-run observations and include seal copying and validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
COMPARE=('verified_bound','events_checked','segments_checked',
         'witness_complex_entries','checkpoints_checked')


def _descendants(pid:int):
    """Use Linux parent/child lists to avoid psutil dependencies."""
    active={pid}
    frontier=[pid]
    while frontier:
        current=frontier.pop()
        try:
            children=(Path(f'/proc/{current}/task/{current}/children')
                      .read_text().split())
        except OSError:
            continue
        for token in children:
            child=int(token)
            if child not in active:
                active.add(child)
                frontier.append(child)
    return active


def _mem(pid):
    try:
        values={}
        for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
            if ':' in line:
                k,v=line.split(':',1)
                if k in ('Rss','Pss','Private_Clean','Private_Dirty'):
                    values[k]=int(v.split()[0])
        return (values.get('Rss',0),values.get('Pss',0),
                values.get('Private_Clean',0)+values.get('Private_Dirty',0))
    except (OSError,ValueError,IndexError):
        return (0,0,0)


def run(manifest, policy:str, workers=4):
    executable='qc22_checker.py' if policy=='sealed' else 'qc21_checker.py'
    cmd=[sys.executable,'-I','-S',str(HERE/executable),str(manifest),
         '--workers',str(workers)]
    if policy=='sealed':
        cmd+=['--start-method','fork']
    elif policy=='baseline':
        cmd+=['--start-method','fork']
    else:
        raise ValueError(policy)
    start=time.perf_counter()
    process=subprocess.Popen(cmd,cwd=HERE,stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,text=True)
    peak=[0,0,0]
    while process.poll() is None:
        mem=[0,0,0]
        for pid in _descendants(process.pid):
            reading=_mem(pid)
            mem=[a+b for a,b in zip(mem,reading)]
        peak=[max(a,b) for a,b in zip(peak,mem)]
        time.sleep(0.05)
    out,err=process.communicate()
    elapsed=time.perf_counter()-start
    if process.returncode!=0:
        raise RuntimeError(f'{policy} failed with {process.returncode}: {err[-1200:]}')
    record=json.loads(out)
    return {'policy':policy,'workers':workers,'elapsed_seconds':elapsed,
            'peak_sum_rss_mib':peak[0]/1024,
            'peak_sum_pss_mib':peak[1]/1024,
            'peak_sum_uss_mib':peak[2]/1024,
            **{k:record[k] for k in COMPARE},
            'sealed_bytes':record.get('sealed_bytes',0),
            'seal_seconds':record.get('seal_seconds',0.0)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('qc22_benchmark.csv'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--repeats',type=int,default=2)
    args=p.parse_args()
    rows=[]
    for _ in range(args.repeats):
        baseline=run(args.manifest,'baseline',args.workers)
        sealed=run(args.manifest,'sealed',args.workers)
        for key in COMPARE:
            if baseline[key]!=sealed[key]:
                raise AssertionError(f'proof mismatch {key}: {baseline[key]} != {sealed[key]}')
        rows.extend((baseline,sealed))
        print(f"baseline {baseline['elapsed_seconds']:.3f}s, {baseline['peak_sum_uss_mib']:.1f} MiB USS; "
              f"sealed {sealed['elapsed_seconds']:.3f}s, {sealed['peak_sum_uss_mib']:.1f} MiB USS",flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(args.output)

if __name__=='__main__':main()
