"""Measure peak process RSS for QC-06 policies in fresh Python subprocesses.

Run python benchmark_peak_qc06.py --qubits 8,24,48. Each policy is
launched as a separate process and reads the same pinned entry budget.
Linux ru_maxrss is KiB; macOS uses bytes. This includes NumPy/interpreter
startup and process allocations, and excludes dense oracle calculations.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import resource
import numpy as np

from channel_mpo import gate_layers,kraus
from purified_mpo import PurifiedChain
from qc06_adaptive import AdaptivePurifiedChain


def child_run(n: int, depth: int, seed: int, p: float, bond: int,
              env: int, cap: int, mode: str, budgets: list[int]):
    gates=gate_layers(n,depth,seed)
    ks=kraus('depolarizing',p)
    if mode=='uniform': sim=PurifiedChain(n,bond,env)
    elif mode=='adaptive': sim=AdaptivePurifiedChain(n,bond,cap)
    else: raise ValueError(mode)
    retained=[]; retentions=[]
    t=perf_counter()
    for layer_index, layer in enumerate(gates):
        for i,u in layer:sim.gate(i,u)
        for i in range(n):sim.channel(i,ks)
        if mode=='uniform':
            for i in range(n):sim.compress_env(i)
        else:
            sim.compress_layer_budgeted(budgets[layer_index],max_site_env=cap)
        retentions.append(sim.normalize())
        retained.append(sim.dimensions()['purified_entries'])
        if mode=='uniform' and retained[-1]!=budgets[layer_index]:
            raise AssertionError('pinned schedule not matched to baseline')
        if mode=='adaptive' and retained[-1]>budgets[layer_index]:
            raise AssertionError('adaptive retained-entry budget exceeded')
    elapsed=perf_counter()-t
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib=peak/(1024 if sys.platform!='darwin' else 1024**2)
    return dict(mode=mode,n=n,seed=seed,p=p,depth=depth,
                elapsed_seconds=elapsed,peak_rss_mib=rss_mib,
                boundary_max_entries=max(retained),last_entries=retained[-1],
                min_layer_retention=min(retentions),
                tracked_largest_working_entries=sim.largest_working_entries)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--child',action='store_true')
    ap.add_argument('--mode',choices=['uniform','adaptive'])
    ap.add_argument('--n',type=int)
    ap.add_argument('--p',type=float)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--depth',type=int,default=6)
    ap.add_argument('--bond',type=int,default=12)
    ap.add_argument('--uniform-env',type=int,default=6)
    ap.add_argument('--adaptive-cap',type=int,default=12)
    ap.add_argument('--qubits',default='8,24,48')
    ap.add_argument('--strengths',default='0.2,0.5')
    ap.add_argument('--frozen-budgets',type=Path,default=Path('qc06_budget_schedule.json'))
    ap.add_argument('--output',type=Path,default=Path('qc06_peak_rss.csv'))
    a=ap.parse_args()
    config=json.loads(a.frozen_budgets.read_text())
    if (config['circuit_depth'],config['bond_cap'],config['uniform_env_cap']) != (a.depth,a.bond,a.uniform_env):
        ap.error('frozen schedule protocol mismatch')
    by_n=config['budgets_by_n']
    if a.child:
        if a.n is None or a.mode is None or a.p is None: ap.error('child requires n,mode,p')
        print(json.dumps(child_run(a.n,a.depth,a.seed,a.p,a.bond,a.uniform_env,
                                   a.adaptive_cap,a.mode,by_n[str(a.n)])))
        return
    rows=[]
    for n in map(int,a.qubits.split(',')):
        for p in map(float,a.strengths.split(',')):
            res={}
            for mode in ('uniform','adaptive'):
                args=[sys.executable,str(Path(__file__).resolve()),'--child',
                      '--mode',mode,'--n',str(n),'--p',str(p),'--seed',str(a.seed),
                      '--depth',str(a.depth),'--bond',str(a.bond),
                      '--uniform-env',str(a.uniform_env),'--adaptive-cap',str(a.adaptive_cap),
                      '--frozen-budgets',str(a.frozen_budgets.resolve())]
                env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
                finished=subprocess.run(args,env=env,capture_output=True,text=True,check=True)
                res[mode]=json.loads(finished.stdout)
            row=dict(n=n,p=p,seed=a.seed,
                     seconds_uniform=res['uniform']['elapsed_seconds'],
                     seconds_adaptive=res['adaptive']['elapsed_seconds'],
                     peak_rss_mib_uniform=res['uniform']['peak_rss_mib'],
                     peak_rss_mib_adaptive=res['adaptive']['peak_rss_mib'],
                     working_entries_uniform=res['uniform']['tracked_largest_working_entries'],
                     working_entries_adaptive=res['adaptive']['tracked_largest_working_entries'],
                     min_retention_uniform=res['uniform']['min_layer_retention'],
                     min_retention_adaptive=res['adaptive']['min_layer_retention'],
                     last_entries_uniform=res['uniform']['last_entries'],
                     last_entries_adaptive=res['adaptive']['last_entries'])
            rows.append(row)
            print(f'n={n:2} p={p} peakRSS MiB {row["peak_rss_mib_uniform"]:.1f}/'
                  f'{row["peak_rss_mib_adaptive"]:.1f} '
                  f'elapsed {row["seconds_uniform"]:.3f}/{row["seconds_adaptive"]:.3f}s',flush=True)
    with a.output.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print('saved',a.output)

if __name__=='__main__':main()
