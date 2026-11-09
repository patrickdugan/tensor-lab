"""Paired seeds QC-02 vs QC-03 full-SVD vs QC-04 screened Gram gauge search.

Measure evolution-layer time separately from dense-reference validation. Run
with BLAS threads 1 to avoid tiny-matrix multithread noise. Records all
seeds, including QC-03 fidelity regressions.
"""
from __future__ import annotations
import argparse
import csv
import os
import platform
from pathlib import Path
import numpy as np

from purified_mpo import run_one
from gauge_pair import run_pair


def run_grid(ns=(6,8), seeds=range(3), strengths=(.2,.5),depth=6,
             bond=12, env=4):
    rows=[]
    for n in ns:
        if n>10:
            raise ValueError('dense reference only supported for n <=10')
        for p in strengths:
            for seed in seeds:
                # All three policies receive identical gate ensembles and
                # density oracle; timing excludes density oracle evaluation.
                base=run_one(n,depth,seed,'depolarizing',p,bond,env,0,validate=True)
                legacy=run_pair(n,depth,seed,'depolarizing',p,bond,env,0,factor_swaps=True)
                fast=run_pair(n,depth,seed,'depolarizing',p,bond,env,0,
                              factor_swaps=True,fast_factor_search=True)
                for name,seq in [('qc02',base),('qc03',legacy),('qc04',fast)]:
                    final=seq[-1]
                    rows.append({
                        'n':n,'depth':depth,'seed':seed,'channel':'depolarizing',
                        'p':p,'bond_cap':bond,'env_cap':env,'policy':name,
                        'trace_distance':final['trace_distance'],
                        'total_evolution_seconds':sum(s['runtime_layer_seconds'] for s in seq),
                        'min_layer_retention':min(s.get('layer_retention', s.get('trace_before_normalize')) for s in seq),
                        'gauge_seconds':final.get('gauge_seconds',0),
                        'gauge_evaluations':final.get('gauge_evaluations',0),
                        'pruned_candidates':final.get('gauge_pruned',0),
                        'exact_candidates_scored':final.get('gauge_exact_scores',0),
                        'gauge_accepted':final.get('gauge_acceptances',0),
                        'max_bond':final['max_bond'],'max_env':final['max_env'],
                        'purified_entries':final['purified_entries'],
                        'numpy':np.__version__, 'python':platform.python_version(),
                        'blas_threads':os.environ.get('OPENBLAS_NUM_THREADS','unspecified')
                    })
                a,b,c=rows[-3:]
                speed=b['total_evolution_seconds']/c['total_evolution_seconds']
                print(f"{n}q p={p} seed={seed} trace QC02/QC03/QC04="
                      f"{a['trace_distance']:.6f}/{b['trace_distance']:.6f}/{c['trace_distance']:.6f}"
                      f" QC03/QC04 time={b['total_evolution_seconds']:.3f}/"
                      f"{c['total_evolution_seconds']:.3f}s speedup={speed:.2f}x "
                      f"pruned={c['pruned_candidates']}",flush=True)
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qubits',default='6,8')
    parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--strengths',default='0.2,0.5')
    parser.add_argument('--depth',type=int,default=6)
    parser.add_argument('--bond',type=int,default=12)
    parser.add_argument('--env',type=int,default=4)
    parser.add_argument('--output',type=Path,default=Path('qc04_paired.csv'))
    args=parser.parse_args()
    rows=run_grid(tuple(int(x) for x in args.qubits.split(',')),range(args.seeds),
                  tuple(float(x) for x in args.strengths.split(',')),args.depth,args.bond,args.env)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print('saved',args.output,'rows',len(rows))
