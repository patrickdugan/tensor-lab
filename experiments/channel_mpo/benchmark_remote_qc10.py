"""QC-10: test spurious long-range correlations of distant edge regions."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
from channel_mpo import gate_layers,kraus,Z
from benchmark_qc09 import frozen_budgets
from qc09_certified import CertifiedPeakChain
from qc10_cone import tensor_marginal,trace_distance,independent_cones_density

def run_remote(n:int,p:float,seed:int,*,channel:str="depolarizing",mode:str="peak"):
    if mode not in ("peak","sketch"):raise ValueError("unknown mode")
    layers=gate_layers(n,6,seed)
    operators=kraus(channel,p)
    exact=independent_cones_density(n,layers,operators,(0,),(n-1,))
    chain=CertifiedPeakChain(n,max_bond=12,max_env=12,work_cap_entries=750_000,
        gate_method="sketch",oversample=4,power_iters=2,
        peak_gate_cap_entries=({24:350000,48:550000}.get(n) if mode=="peak" else None))
    for t,layer in enumerate(layers):
        chain.gate_layer(layer)
        for i in range(n):chain.channel(i,operators)
        chain.compress_layer_peak(frozen_budgets(n)[t],operators)
    approximated=tensor_marginal(chain,(0,n-1))
    left=tensor_marginal(chain,(0,))
    right=tensor_marginal(chain,(n-1,))
    connected=float((np.trace(np.kron(Z,Z)@approximated)-
                     np.trace(Z@left)*np.trace(Z@right)).real)
    return {"n":n,"p":p,"seed":seed,"channel":channel,"mode":mode,
            "remote_pair_trace_distance":trace_distance(exact,approximated),
            "spurious_independence_distance":trace_distance(approximated,np.kron(left,right)),
            "spurious_connected_ZZ":connected}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits",type=int,default=48)
    parser.add_argument("--strengths",default=".2,.5")
    parser.add_argument("--seeds",default="3,4,5,6,7")
    parser.add_argument("--modes",default="sketch,peak")
    parser.add_argument("--output",type=Path,default=Path("qc10_remote_edges.csv"))
    args=parser.parse_args()
    rows=[]
    for p in map(float,args.strengths.split(",")):
        for seed in map(int,args.seeds.split(",")):
            for mode in args.modes.split(","):
                row=run_remote(args.qubits,p,seed,mode=mode)
                rows.append(row)
                print(p,seed,mode,round(row["remote_pair_trace_distance"],5),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="") as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved",len(rows),"rows to",args.output)
if __name__=="__main__":main()
