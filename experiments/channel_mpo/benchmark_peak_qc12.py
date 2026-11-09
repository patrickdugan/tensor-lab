"""Isolated process RSS measurements for deep QC-12 backward-MPO observables.

Linux ru_maxrss reports KiB; each sample is measured in a fresh interpreter.
Single-trial resource observations are not statistical performance intervals.
"""
from __future__ import annotations
import argparse,csv,json,os,platform,resource,subprocess,sys
from pathlib import Path
from channel_mpo import gate_layers,kraus
from qc12_heisenberg import operator_cone_expectation


def one(n:int,depth:int,target:int,channel:str,p:float,cap:int,seed:int):
    layers=gate_layers(n,depth,seed)
    result=operator_cone_expectation(n,layers,kraus(channel,p),(target,),'Z',bond_cap=cap)
    return dict(n=n,depth=depth,target=target,seed=seed,channel=channel,p=p,
                rank_cap=cap,cone_width=result.cone_width,
                expectation=result.expectation,error_bound=result.error_bound,
                seconds=result.seconds,max_mpo_entries=result.max_mpo_entries,
                max_gate_matrix_entries=result.max_working_entries,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)


def run(*,depths=(16,24),channels=('dephasing','depolarizing'),
        strengths=(.2,.5),caps=(8,16),seed=3,n=48,target=22):
    rows=[]
    for depth in depths:
      for ch in channels:
       for p in strengths:
        for cap in caps:
         cmd=[sys.executable,__file__,'--one',str(n),str(depth),str(target),ch,
                str(p),str(cap),str(seed)]
         env=os.environ.copy()
         for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
             env[key]='1'
         raw=subprocess.run(cmd,check=True,capture_output=True,text=True,env=env)
         rows.append(json.loads(raw.stdout))
         print(depth,ch,p,cap,'RSS',round(rows[-1]['peak_rss_mib'],2),flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--one',nargs=7)
    ap.add_argument('--depths',default='16,24')
    ap.add_argument('--channels',default='dephasing,depolarizing')
    ap.add_argument('--strengths',default='.2,.5')
    ap.add_argument('--caps',default='8,16')
    ap.add_argument('--seed',type=int,default=3)
    ap.add_argument('--output',type=Path,default=Path('qc12_peak_rss.csv'))
    args=ap.parse_args()
    if args.one:
        n,dep,target,ch,p,cap,seed=args.one
        print(json.dumps(one(int(n),int(dep),int(target),ch,float(p),int(cap),int(seed))))
        return
    toint=lambda v:tuple(map(int,v.split(',')))
    rows=run(depths=toint(args.depths),channels=tuple(args.channels.split(',')),
             strengths=tuple(map(float,args.strengths.split(','))),caps=toint(args.caps),seed=args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('saved',len(rows),'rows',args.output)

if __name__=='__main__':main()
