"""QC-10: inspect artificial long-range correlations under global truncation.

For n>>depth the backward cones of left and right boundary measurements are
disjoint. The exact two-site density factorizes for the initial product state.
This tool detects spurious connected correlations from the global MPDO approximation.
"""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
from channel_mpo import gate_layers,kraus,Z
from benchmark_qc09 import frozen_budgets
from qc09_certified import CertifiedPeakChain
from qc10_cone import tensor_marginal,trace_distance,independent_cones_density


def run_remote(n:int,p:float,seed:int,*,channel:str='depolarizing',mode:str='peak'):
    if mode not in ('peak','sketch'):raise ValueError('unknown mode')
    layers=gate_layers(n,6,seed);ops=kraus(channel,p)
    exact=independent_cones_density(n,layers,ops,(0,),(n-1,))
    chain=CertifiedPeakChain(n,max_bond=12,max_env=12,work_cap_entries=750000,
        gate_method='sketch',oversample=4,power_iters=2,
        peak_gate_cap_entries=({24:350000,48:550000}.get(n) if mode=='peak' else None))
    for t,layer in enumerate(layers):
        chain.gate_layer(layer)
        for i in range(n):chain.channel(i,ops)
        chain.compress_layer_peak(frozen_budgets(n)[t],ops)
    approx=tensor_marginal(chain,(0,n-1))
    a=tensor_marginal(chain,(0,));b=tensor_marginal(chain,(n-1,))
    corr=float((np.trace(np.kron(Z,Z)@approx)-
          np.trace(Z@a)*np.trace(Z@b)).real)
    return {'n':n,'p':p,'seed':seed,'channel':channel,'mode':mode,
            'remote_pair_trace_distance':trace_distance(exact,approx),
            'spurious_independence_distance':trace_distance(approx,np.kron(a,b)),
            'spurious_connected_ZZ':corr}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qubits',type=int,default=48)
    parser.add_argument('--strengths',default='.2,.5')
    parser.add_argument('--seeds',default='3,4,5,6,7')
    parser.add_argument('--modes',default='sketch,peak')
    parser.add_argument('--output',type=Path,default=Path('qc10_remote_edges.csv'))
    args=parser.parse_args()
    rows=[]
    for p in map(float,args.strengths.split(',')):
      for seed in map(int,args.seeds.split(',')):
       for mode in args.modes.split(','):
        row=run_remote(args.qubits,p,seed,mode=mode)
        rows.append(row)
        print(p,seed,mode,round(row['remote_pair_trace_distance'],5),
              round(row['spurious_independence_distance'],8),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('saved',len(rows),'records to',args.output)

if __name__=='__main__':main()
