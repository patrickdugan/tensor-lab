"""QC-10: 48-qubit local-observable validation via exact backward light cones.

Unlike reconstructing a global density matrix, the oracle builds only the
finite input dependency cone of a requested output region.  For six layers
at a boundary, the 1-site and adjacent 2-site cones have 6 and 8 qubits.
Report trace distances and operational measurement bounds, along with the
QC-09 low-retention warning, under the same frozen QC-06 entry schedule.
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
from channel_mpo import gate_layers, kraus, Z
from benchmark_qc09 import frozen_budgets
from qc09_certified import CertifiedPeakChain
from qc10_cone import exact_cone_density, tensor_marginal, trace_distance

CAPS={6:260_000,8:280_000,24:350_000,48:550_000}
MODES=('sketch','peak')


def target_regions(n:int):
    return {'left_1':(0,), 'left_2':(0,1),
            'right_1':(n-1,), 'right_2':(n-2,n-1)}


def run_case(n:int,p:float,seed:int,*,channel:str='depolarizing',depth:int=6,
             modes:tuple[str,...]=MODES,max_sites:int=8)->list[dict]:
    if n<4 or not 0<=p<=1 or depth<1:
        raise ValueError('invalid circuit configuration')
    if not modes or any(mode not in MODES for mode in modes):
        raise ValueError('supported modes are sketch and peak')
    layers=gate_layers(n,depth,seed)
    ops=kraus(channel,p)
    budgets=frozen_budgets(n)
    if depth!=len(budgets):
        raise ValueError('requires the original frozen six-layer schedule')
    regions=target_regions(n)
    reference={label:exact_cone_density(n,layers,ops,target,max_sites=max_sites)
               for label,target in regions.items()}
    rows=[]
    ordered=modes[seed%len(modes):]+modes[:seed%len(modes)]
    for mode in ordered:
        chain=CertifiedPeakChain(n,max_bond=12,max_env=12,work_cap_entries=750_000,
               gate_method='sketch',oversample=4,power_iters=2,
               peak_gate_cap_entries=CAPS[n] if mode=='peak' else None)
        started=perf_counter();ret=1.;blocked=0
        for t,layer in enumerate(layers):
            chain.gate_layer(layer)
            for site in range(n):chain.channel(site,ops)
            r=chain.compress_layer_peak(budgets[t],ops)
            ret=min(ret,r.normalized_trace_before)
            blocked+=r.plan.blocked_increments
        duration=perf_counter()-started
        entries=chain.dimensions()['purified_entries']
        for label,target in regions.items():
            oracle=reference[label]
            reduced=tensor_marginal(chain,target)
            dist=trace_distance(oracle.reduced_density,reduced)
            oz=np.kron(Z,Z) if len(target)==2 else Z
            obs_abs_error=abs(float(np.trace(oz@oracle.reduced_density).real-
                                    np.trace(oz@reduced).real))
            assert obs_abs_error<=2*dist+1e-9
            rows.append({'n':n,'depth':depth,'channel':channel,'p':p,'seed':seed,
               'split':'heldout' if seed>=3 else 'calibration',
               'mode':mode,'region':label,'sites':','.join(map(str,target)),
               'cone_width':len(oracle.cone.sites),'oracle_seconds':oracle.seconds,
               'sim_seconds':duration,'local_trace_distance':dist,
               'worst_event_probability_error':dist,
               'worst_norm1_observable_error_bound':2*dist,
               'observable':('ZZ' if len(target)==2 else 'Z'),
               'observable_abs_error':obs_abs_error,
               'reference_observable':float(np.trace(oz@oracle.reduced_density).real),
               'approximate_observable':float(np.trace(oz@reduced).real),
               'retained_entries':entries,'min_layer_retention':ret,
               'gate_bound_sum':sum(c.trace_distance_bound for c in chain.gate_certificates),
               'total_truncation_bound_estimate':chain.error_bound_sum,
               'modeled_live_peak_entries':chain.modeled_live_peak_entries,
               'blocked_rank_increments':blocked,
               'oracle_rounding':'float64',
               'numpy':np.__version__,'python':platform.python_version()})
    return rows


def run_grid(ns=(24,48),strengths=(.2,.5),seeds=(3,4,5,6,7),*,
             channels=('depolarizing',),modes=MODES):
    rows=[]
    for name in channels:
      for n in ns:
       for p in strengths:
        for seed in seeds:
            result=run_case(n,p,seed,channel=name,modes=modes)
            rows.extend(result)
            subset=[r for r in result if r['region']=='left_1']
            print(name,n,p,seed,[(r['mode'],round(r['local_trace_distance'],5),
                   round(r['sim_seconds'],3)) for r in subset],flush=True)
    return rows


def save(rows:list[dict],path:Path):
    if not rows:raise ValueError('empty results')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qubits',default='24,48')
    parser.add_argument('--strengths',default='.2,.5')
    parser.add_argument('--seeds',default='3,4,5,6,7')
    parser.add_argument('--channels',default='depolarizing')
    parser.add_argument('--modes',default='sketch,peak')
    parser.add_argument('--output',type=Path,default=Path('qc10_local_validation.csv'))
    args=parser.parse_args()
    rows=run_grid(tuple(int(x) for x in args.qubits.split(',')),
                  tuple(float(x) for x in args.strengths.split(',')),
                  tuple(int(x) for x in args.seeds.split(',')),
                  channels=tuple(args.channels.split(',')),modes=tuple(args.modes.split(',')))
    save(rows,args.output)
    print(f'wrote {len(rows)} records to {args.output}')


if __name__=='__main__':main()
