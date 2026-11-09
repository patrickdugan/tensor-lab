"""QC-03 comparison: QC-02 sequential compression versus ancillary factor re-routing.

Every reported trace distance uses the independent exact density simulator.
Fidelity is measured after the complete circuit, not at the local objective.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from purified_mpo import run_one
from gauge_pair import run_pair


def run_grid(ns=(6,8), seeds=range(3), strengths=(.2,.5),
             channels=('depolarizing',), depth=6, bond=12, env=4):
    rows=[]
    for n in ns:
        if n>10:raise ValueError('requires dense oracle with <=10 qubits')
        for seed in seeds:
            for channel in channels:
                for p in strengths:
                    baseline=run_one(n,depth,seed,channel,p,bond,env,0,validate=True)
                    variants=[('qc02_sequential',baseline)]
                    variants.append(('factor_swaps',run_pair(n,depth,seed,channel,p,bond,env,0,factor_swaps=True)))
                    for policy,run in variants:
                        final=run[-1]
                        row={'n':n,'depth':depth,'seed':seed,'channel':channel,'p':p,
                             'bond_cap':bond,'env_cap':env,'policy':policy,
                             'trace_distance':final['trace_distance'],
                             'purified_entries':final['purified_entries'],
                             'max_bond':final['max_bond'],'max_env':final['max_env'],
                             'total_runtime_seconds':sum(r['runtime_layer_seconds'] for r in run),
                             'min_layer_retention':min(r['layer_retention'] if policy=='factor_swaps'
                                                       else r['trace_before_normalize'] for r in run),
                             'z0_abs_error':final['z0_abs_error'] if policy=='qc02_sequential' else '',
                             'gauge_evaluations':final['gauge_evaluations'] if policy=='factor_swaps' else 0,
                             'gauge_surrogate_gain':final['gauge_surrogate_gain'] if policy=='factor_swaps' else 0,
                             'min_eigenvalue':final['min_eigenvalue'],
                             }
                        rows.append(row)
                    a,b=rows[-2:]
                    print(f"{n}q seed={seed} {channel} p={p:.2f} "
                          f"base={a['trace_distance']:.6f} "
                          f"swaps={b['trace_distance']:.6f} "
                          f"Δ={a['trace_distance']-b['trace_distance']:+.6f} "
                          f"ret={a['min_layer_retention']:.3f}/{b['min_layer_retention']:.3f}",flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qubits',default='6,8')
    ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--channels',default='depolarizing')
    ap.add_argument('--strengths',default='0.2,0.5')
    ap.add_argument('--depth',type=int,default=6)
    ap.add_argument('--bond',type=int,default=12)
    ap.add_argument('--env',type=int,default=4)
    ap.add_argument('--output',type=Path,default=Path('qc03_gauge_comparison.csv'))
    args=ap.parse_args()
    rows=run_grid(tuple(map(int,args.qubits.split(','))),range(args.seeds),
                  tuple(map(float,args.strengths.split(','))),
                  tuple(args.channels.split(',')),args.depth,args.bond,args.env)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('wrote',args.output,'rows',len(rows))

if __name__=='__main__':main()
