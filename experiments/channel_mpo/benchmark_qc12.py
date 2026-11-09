"""QC-12: fixed-cap Heisenberg-MPO benchmarks and independent exact windows.

Deep runs do not use a dense oracle. An `accepted` flag means only that the
numerical residual ledger is below a predeclared ideal-arithmetic expectation
bound. Exact windows compare 1/2-site tomography to QC-11's independent dense
causal-cone simulator (width <= 10). No oracle feeds back into rank choice.
"""
from __future__ import annotations
import argparse,csv,platform
from pathlib import Path
from time import perf_counter
import numpy as np
from channel_mpo import gate_layers,kraus
from qc12_heisenberg import operator_cone_expectation,pauli_density
from qc11_streaming_cone import streaming_cone_density
from qc10_cone import trace_distance


def deep_grid(*,n=48,depths=(10,16,20,24),targets=(0,22),
              caps=(4,8,16),channels=('dephasing','depolarizing','amplitude_damping'),
              strengths=(.2,.5),seeds=(3,4,5,6,7),target_error=.05):
    rows=[]
    for depth in depths:
      for seed in seeds:
        layers=gate_layers(n,depth,seed)
        for channel in channels:
          for p in strengths:
            ops=kraus(channel,p)
            for target in targets:
              for cap in caps:
                r=operator_cone_expectation(n,layers,ops,(target,),'Z',bond_cap=cap,
                                            max_cone=n,max_entries=1_000_000)
                rows.append(dict(n=n,depth=depth,seed=seed,channel=channel,p=p,
                    target=target,observable='Z',cone_width=r.cone_width,rank_cap=cap,
                    estimated_expectation=r.expectation,error_bound=r.error_bound,
                    accepts_005=bool(r.error_bound<=target_error),
                    truncations=sum(x.frobenius_residual>1e-12 for x in r.local_residuals),
                    max_bond=r.max_bond,peak_mpo_entries=r.max_mpo_entries,
                    largest_gate_matrix_entries=r.max_working_entries,seconds=r.seconds,
                    numpy_version=np.__version__,python_version=platform.python_version()))
      print('deep depth',depth,'rows',len(rows),flush=True)
    return rows


def exact_grid(*,n=48,depth=8,seeds=(3,5),
               channels=('dephasing','depolarizing','amplitude_damping'),
               strengths=(.2,.5),caps=(4,8,16)):
    rows=[]
    for seed in seeds:
      layers=gate_layers(n,depth,seed)
      for channel in channels:
        for p in strengths:
          ops=kraus(channel,p)
          for targets in ((0,),(0,1)):
            ref=streaming_cone_density(n,layers,ops,targets,max_sites=10)
            for cap in caps:
              t=perf_counter()
              rho,bound,paulis=pauli_density(n,layers,ops,targets,bond_cap=cap,max_cone=n)
              duration=perf_counter()-t
              rows.append(dict(n=n,depth=depth,seed=seed,channel=channel,p=p,
                    targets='-'.join(map(str,targets)),cone_width=len(ref.cone.sites),
                    rank_cap=cap,trace_distance=trace_distance(rho,ref.reduced_density),
                    density_bound=bound,accepts_005=bool(bound<=.05),
                    certificate_holds=bool(trace_distance(rho,ref.reduced_density)<=bound+1e-9),
                    seconds=duration,reference_seconds=ref.seconds,
                    numpy_version=np.__version__))
      print('exact seed',seed,'rows',len(rows),flush=True)
    return rows


def save(rows,path:Path):
    if not rows:raise ValueError('no results')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('deep','exact'),default='deep')
    p.add_argument('--n',type=int,default=48)
    p.add_argument('--depths',default='10,16,20,24')
    p.add_argument('--targets',default='0,22')
    p.add_argument('--caps',default='4,8,16')
    p.add_argument('--channels',default='dephasing,depolarizing,amplitude_damping')
    p.add_argument('--strengths',default='.2,.5')
    p.add_argument('--seeds',default='3,4,5,6,7')
    p.add_argument('--output',type=Path,default=Path('qc12_deep.csv'))
    args=p.parse_args()
    ints=lambda s:tuple(int(x) for x in s.split(','))
    floats=lambda s:tuple(float(x) for x in s.split(','))
    channels=tuple(args.channels.split(','))
    if args.mode=='deep':
        rows=deep_grid(n=args.n,depths=ints(args.depths),targets=ints(args.targets),
            caps=ints(args.caps),channels=channels,strengths=floats(args.strengths),
            seeds=ints(args.seeds))
    else:
        dep=ints(args.depths)
        rows=exact_grid(n=args.n,depth=dep[0],seeds=ints(args.seeds),channels=channels,
            strengths=floats(args.strengths),caps=ints(args.caps))
    save(rows,args.output)
    print('wrote',len(rows),'rows to',args.output,flush=True)


if __name__=='__main__':main()
