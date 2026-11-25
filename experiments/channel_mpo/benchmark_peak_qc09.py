"""Isolated-process QC-09 peak RSS against the QC-08 sequential-gate baseline.

One policy and one noise parameter per process. Peak RSS from getrusage is
process-wide and includes Python and LAPACK allocations. Values are single
trials, not statistically independent timing estimates.
"""
from __future__ import annotations
import argparse,csv,json,os,platform,resource,subprocess,sys
from pathlib import Path
from time import perf_counter
import numpy as np


def run_child(n:int,p:float,seed:int,policy:str,peak_cap:int|None):
    from channel_mpo import kraus,gate_layers
    from qc08_streaming import StreamingJointChain
    from qc09_certified import CertifiedPeakChain
    from benchmark_qc09 import frozen_budgets
    budgets=frozen_budgets(n)
    ks=kraus('depolarizing',p)
    layers=gate_layers(n,len(budgets),seed)
    if policy=='qc08':
        chain=StreamingJointChain(n,max_bond=12,max_env=12,work_cap_entries=750_000)
    else:
        chain=CertifiedPeakChain(n,max_bond=12,max_env=12,work_cap_entries=750_000,
             gate_method='sketch' if policy in ('qc09_sketch','qc09_peak') else 'full',
             oversample=4,power_iters=2,peak_gate_cap_entries=peak_cap if policy=='qc09_peak' else None)
    start=perf_counter();retentions=[];blocked=0
    for t,layer in enumerate(layers):
        if policy=='qc08':
            for site,u in layer:chain.gate(site,u)
        else:chain.gate_layer(layer)
        for i in range(n):chain.channel(i,ks)
        if policy=='qc08':
            r=chain.compress_layer_stream(budgets[t],ks,order='classic')
            retentions.append(chain.normalize())
        else:
            r=chain.compress_layer_peak(budgets[t],ks)
            retentions.append(r.normalized_trace_before)
            blocked+=r.plan.blocked_increments
    elapsed=perf_counter()-start
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(n=n,p=p,seed=seed,policy=policy,
                peak_cap_entries=peak_cap if policy=='qc09_peak' else None,
                seconds=elapsed,peak_rss_mib=rss/1024. if sys.platform.startswith('linux') else rss/(1024**2),
                modeled_live_peak_entries=getattr(chain,'modeled_live_peak_entries',
                      chain.predicted_working_peak_entries),
                max_retained_entries=chain.dimensions()['purified_entries'],
                min_layer_retention=min(retentions),
                blocked_increments=blocked,
                python=platform.python_version(),numpy=np.__version__)


def run_parent(ns=(8,24,48),ps=(.2,.5),seed=3,policies=('qc08','qc09_full','qc09_sketch','qc09_peak'),peak_caps=None):
    peak_caps={8:280000,24:350000,48:550000} if peak_caps is None else peak_caps
    out=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for n in ns:
      for p in ps:
        for policy in policies:
            args=[sys.executable,str(Path(__file__).resolve()),'--child','--qubits',str(n),
                  '--strengths',str(p),'--seed',str(seed),'--policy',policy,
                  '--peak-cap',str(peak_caps[n])]
            proc=subprocess.run(args,env=env,capture_output=True,text=True,check=True,timeout=100)
            result=json.loads(proc.stdout.strip().splitlines()[-1])
            print(n,p,policy,'time',round(result['seconds'],3),
                  'rss',round(result['peak_rss_mib'],2),'modeled',result['modeled_live_peak_entries'],flush=True)
            out.append(result)
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--child',action='store_true')
    ap.add_argument('--qubits',default='8,24,48')
    ap.add_argument('--strengths',default='.2,.5')
    ap.add_argument('--seed',type=int,default=3)
    ap.add_argument('--policy',default='qc09_peak')
    ap.add_argument('--peak-cap',type=int,default=550000)
    ap.add_argument('--output',type=Path,default=Path('qc09_peak_rss.csv'))
    a=ap.parse_args()
    if a.child:
        print(json.dumps(run_child(int(a.qubits),float(a.strengths),a.seed,a.policy,a.peak_cap)))
    else:
        rows=run_parent(tuple(map(int,a.qubits.split(','))),
                        tuple(map(float,a.strengths.split(','))),a.seed)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print('saved',len(rows),a.output)

if __name__=='__main__':main()
