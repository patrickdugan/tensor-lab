"""QC-11 paired operational evaluation with target schedules frozen before rollout.

Exact finite causal-cone references are computed only after candidate evolution
and never passed into the rank allocator. All modes share the same circuit and
QC-06 retained-entry schedule; QC-09 peak-live model and cap are shared too.
"""
from __future__ import annotations
import argparse
import csv
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
from channel_mpo import gate_layers, kraus
from benchmark_qc09 import frozen_budgets
from benchmark_qc10 import CAPS, target_regions
from qc09_certified import CertifiedPeakChain
from qc11_focus import ObservablePeakChain,target_schedule
from qc10_cone import exact_cone_density,tensor_marginal,trace_distance

MODES=('baseline','left','right','edges','left_halo','right_halo','edges_halo')


def run_case(n:int,p:float,seed:int,*,channel:str='dephasing',depth:int=6,
             modes:tuple[str,...]=MODES,off_cone:float=.02,
             regions:tuple[str,...]=('left_1','left_2','right_1','right_2'),
             validate:bool=True)->list[dict]:
    if n<4 or not 0<=p<=1 or depth!=6 or not modes or any(x not in MODES for x in modes):
        raise ValueError('invalid benchmark case')
    layers=gate_layers(n,depth,seed)
    operators=kraus(channel,p)
    budgets=frozen_budgets(n)
    targets=target_regions(n)
    if any(r not in targets for r in regions):raise ValueError('unknown test region')
    plan_left=target_schedule(n,layers,(targets['left_2'],))
    plan_right=target_schedule(n,layers,(targets['right_2'],))
    plan_edges=target_schedule(n,layers,(targets['left_2'],targets['right_2']))
    rows=[]
    for mode in modes[seed%len(modes):]+modes[:seed%len(modes)]:
        cls=CertifiedPeakChain if mode=='baseline' else ObservablePeakChain
        chain=cls(n,max_bond=12,max_env=12,work_cap_entries=750_000,
                  gate_method='sketch',oversample=4,power_iters=2,
                  peak_gate_cap_entries=CAPS[n])
        start=perf_counter();minret=1.;blocked=0;max_bound=0;layer_entries=[];rank_plans=[]
        schedule=(plan_right if mode.startswith('right') else
                  plan_left if mode.startswith('left') else plan_edges)
        halo=int(mode.endswith('_halo'))
        for t,gates in enumerate(layers):
            chain.gate_layer(gates)
            for site in range(n):chain.channel(site,operators)
            if mode=='baseline':
                result=chain.compress_layer_peak(budgets[t],operators)
                plan=result.plan
                retained=result.normalized_trace_before
            else:
                result=chain.compress_focused(budgets[t],operators,schedule,t,
                                             off_cone=off_cone,halo=halo)
                plan=result.plan
                retained=result.min_retained_fraction
            layer_entries.append(chain.dimensions()['purified_entries'])
            if layer_entries[-1]>budgets[t]:raise AssertionError('budget exceeded')
            minret=min(minret,retained)
            blocked+=plan.blocked_increments
            rank_plans.append((plan.joint.bond_ranks,plan.joint.environment_ranks))
        seconds=perf_counter()-start
        for name in regions:
            target=targets[name]
            approx=tensor_marginal(chain,target)
            reference=(exact_cone_density(n,layers,operators,target,max_sites=8).reduced_density
                       if validate else None)
            distance=trace_distance(approx,reference) if validate else float('nan')
            rows.append({'n':n,'depth':depth,'channel':channel,'p':p,'seed':seed,
                'split':'calibration' if seed<3 else 'heldout',
                'mode':mode,'off_cone':off_cone,'halo':halo,'region':name,
                'local_trace_distance':distance,'event_gate_05':distance<=.05,
                'sim_seconds':seconds,'retained_entries':layer_entries[-1],
                'max_layer_entries':max(layer_entries),'min_layer_retention':minret,
                'modeled_live_peak_entries':chain.modeled_live_peak_entries,
                'blocked_rank_increments':blocked,'rank_plans':repr(rank_plans),
                'numpy_version':np.__version__,'python':platform.python_version()})
    return rows


def run_grid(ns=(48,),strengths=(.2,.5),seeds=(3,4,5,6,7),*,
             channel='dephasing',modes=MODES,off_cone=.02):
    result=[]
    for n in ns:
      for p in strengths:
       for seed in seeds:
        rows=run_case(n,p,seed,channel=channel,modes=modes,off_cone=off_cone)
        result.extend(rows)
        subset=[r for r in rows if r['region']=='right_2']
        print(n,channel,p,seed,[(r['mode'],round(r['local_trace_distance'],5),
                               round(r['sim_seconds'],3),r['retained_entries']) for r in subset],flush=True)
    return result


def save(rows,path):
    if not rows:raise ValueError('empty grid')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qubits',default='48')
    ap.add_argument('--strengths',default='.2,.5')
    ap.add_argument('--seeds',default='3,4,5,6,7')
    ap.add_argument('--channel',default='dephasing')
    ap.add_argument('--modes',default=','.join(MODES))
    ap.add_argument('--off-cone',type=float,default=.02)
    ap.add_argument('--output',type=Path,default=Path('qc11_heldout.csv'))
    args=ap.parse_args()
    rows=run_grid(tuple(map(int,args.qubits.split(','))),
                 tuple(map(float,args.strengths.split(','))),
                 tuple(map(int,args.seeds.split(','))),channel=args.channel,
                 modes=tuple(args.modes.split(',')),off_cone=args.off_cone)
    save(rows,args.output)
    print('saved',len(rows),'rows',args.output)

if __name__=='__main__':main()
