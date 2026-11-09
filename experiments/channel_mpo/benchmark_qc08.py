"""QC-08: paired exact-oracle benchmark of the streaming allocator.

All policies use QC-06's frozen budgets and the same noise/gate sequence.
Measurement excludes dense oracle evaluation.  'classic' retains QC-07's
bond-then-environment truncation ordering, while 'fused' changes the order.
The policy never invokes the dense oracle and does not select modes by oracle.
"""
from __future__ import annotations
import argparse, csv, json, platform
from pathlib import Path
from time import perf_counter
import numpy as np
from channel_mpo import gate_layers,kraus,apply_operator,apply_channel
from qc07_joint import JointPurifiedChain
from qc08_streaming import StreamingJointChain
from benchmark_qc07 import trace_distance


def run_case(n:int,seed:int,p:float,*,channel:str='depolarizing',
             depth:int=6,budgets:list[int]|None=None,
             bond_cap:int=12, env_cap:int=12,
             work_cap_entries:int=750_000,
             validate:bool=True, track_marginals:bool=False,
             policies:tuple[str,...]=('qc07','classic','classic_mem','fused')) -> dict:
    if budgets is None or len(budgets)!=depth:
        raise ValueError('required frozen per-layer budgets')
    if validate and n>8:
        raise ValueError('dense oracle limited to n<=8')
    ops=kraus(channel,p)
    layers=gate_layers(n,depth,seed)
    results={}
    sims={}
    # Rotate execution order by seed to reduce systematic warm-cache bias.
    rotation=seed%len(policies)
    ordered=list(policies[rotation:]+policies[:rotation])
    for mode in ordered:
        if mode=='qc07':
            sim=JointPurifiedChain(n,max_bond=bond_cap,max_env=env_cap,
                                   work_cap_entries=work_cap_entries)
        else:
            sim=StreamingJointChain(n,max_bond=bond_cap,max_env=env_cap,
                                    work_cap_entries=work_cap_entries)
        start=perf_counter()
        plans=[];layer_counts=[];retentions=[];svds=0;scans=[]
        for t,layer in enumerate(layers):
            if mode=='qc07':
                for site,u in layer: sim.gate(site,u)
            else:
                sim.gate_layer(layer)
            for i in range(n):sim.channel(i,ops)
            if mode=='qc07':
                res=sim.compress_layer_joint(budgets[t],ops,bond_cap=bond_cap,env_cap=env_cap)
            else:
                res=sim.compress_layer_stream(budgets[t],ops,bond_cap=bond_cap,
                                              env_cap=env_cap,order='classic' if mode=='classic_mem' else mode,
                                              track_marginals=track_marginals)
                svds+=res.compression_svds
                scans.append(res.scan)
            plans.append((res.plan.bond_ranks,res.plan.environment_ranks))
            entries=sim.dimensions()['purified_entries']
            if entries>budgets[t]:
                raise AssertionError('retained-entry budget overflow')
            layer_counts.append(entries)
            retentions.append(sim.normalize())
        elapsed=perf_counter()-start
        sims[mode]=sim
        results[mode]={
            'seconds':elapsed,
            'entries':layer_counts[-1],
            'max_entries':max(layer_counts),
            'error_bound_sum':sim.error_bound_sum,
            'min_layer_retention':min(retentions),
            'modeled_peak_entries':sim.predicted_working_peak_entries,
            'max_bond':sim.dimensions()['max_bond'],
            'max_env':sim.dimensions()['max_env'],
            'plans':plans,
            'compression_svds':svds if mode!='qc07' else None,
            'scoring_svds':sum(s.bond_score_svds+s.environment_score_svd_fallbacks for s in scans),
            'scoring_gram_eighs':sum(s.environment_gram_scores for s in scans),
            'scan_right_qr_sweeps':sum(s.right_canonical_sweeps for s in scans),
        }
    exact=None
    oracle_seconds=None
    if validate:
        start=perf_counter()
        exact=np.zeros((2**n,2**n),complex)
        exact[0,0]=1.
        for layer in layers:
            for i,u in layer:exact=apply_operator(exact,u,(i,i+1),n)
            for i in range(n):exact=apply_channel(exact,ops,i,n)
        for mode,sim in sims.items():
            results[mode]['trace_distance']=trace_distance(exact,sim.dense_density())
        oracle_seconds=perf_counter()-start
    reference=results.get('qc07')
    for mode in ('classic','classic_mem','fused'):
        if mode in results and reference:
            results[mode]['plan_equal_qc07']=results[mode]['plans']==reference['plans']
            results[mode]['density_distance_qc07']=trace_distance(
                sims[mode].dense_density(),sims['qc07'].dense_density()) if validate else None
    row={
        'n':n,'seed':seed,'split':'calibration' if seed<3 else 'heldout',
        'p':p,'channel':channel,'depth':depth,
        'bond_cap':bond_cap,'env_cap':env_cap,
        'peak_layer_entry_budget':max(budgets),
        'final_layer_entry_budget':budgets[-1],
        'oracle_seconds_excluded':oracle_seconds,
        'python':platform.python_version(),'numpy':np.__version__,
    }
    for mode,result in results.items():
        for k,v in result.items():
            row[f'{mode}_{k}']=json.dumps(v) if k=='plans' else v
    return row


def run_grid(ns=(6,8),ps=(.2,.5),seeds=(3,4,5,6,7),
             *,channel='depolarizing',depth=6,
             validate=True,track_marginals=False,work_cap_entries=750_000):
    budget_file=Path(__file__).with_name('qc06_budget_schedule.json')
    schedule=json.loads(budget_file.read_text())
    if schedule['circuit_depth']!=depth:
        raise ValueError('frozen-budget depth mismatch')
    records=[]
    for n in ns:
        for p in ps:
            for seed in seeds:
                row=run_case(n,seed,p,channel=channel,depth=depth,
                             budgets=schedule['budgets_by_n'][str(n)],
                             validate=validate,track_marginals=track_marginals,
                             work_cap_entries=work_cap_entries)
                records.append(row)
                print(f"{n}q p={p} seed={seed} "
                      f"runtime={row['qc07_seconds']:.3f}/"
                      f"{row['classic_seconds']:.3f}/{row['classic_mem_seconds']:.3f}/"
                      f"{row['fused_seconds']:.3f} "
                      f"D={row.get('qc07_trace_distance')}/{row.get('fused_trace_distance')}",
                      flush=True)
    return records


def save_rows(records, path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qubits',default='6,8')
    p.add_argument('--strengths',default='.2,.5')
    p.add_argument('--seeds',default='3,4,5,6,7')
    p.add_argument('--channel',default='depolarizing')
    p.add_argument('--depth',type=int,default=6)
    p.add_argument('--work-cap-entries',type=int,default=750000)
    p.add_argument('--marginal-diagnostics',action='store_true')
    p.add_argument('--no-validate',action='store_true')
    p.add_argument('--output',type=Path,default=Path('qc08_heldout.csv'))
    a=p.parse_args()
    records=run_grid(tuple(int(x) for x in a.qubits.split(',')),
                     tuple(float(x) for x in a.strengths.split(',')),
                     tuple(int(x) for x in a.seeds.split(',')),
                     channel=a.channel,depth=a.depth,
                     validate=not a.no_validate,
                     track_marginals=a.marginal_diagnostics,
                     work_cap_entries=a.work_cap_entries)
    save_rows(records,a.output)
    print('saved',len(records),'rows to',a.output)

if __name__=='__main__': main()
