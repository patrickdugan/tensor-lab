"""QC-09 benchmark: matched budgets, numerical residual certificates and peak model.

The exact density oracle is called after evolution only, for <=8 qubits.
Mode order is rotated per seed to reduce systematic warm-cache effects.
"""
from __future__ import annotations
import argparse,csv,json,platform
from pathlib import Path
from time import perf_counter
import numpy as np
from channel_mpo import gate_layers,kraus,apply_operator,apply_channel
from benchmark_qc07 import trace_distance
from qc08_streaming import StreamingJointChain
from qc09_certified import CertifiedPeakChain

MODES=('qc08','qc09_full','qc09_sketch','qc09_peak')
CAPS={6:260_000,8:280_000}


def run_case(n:int,p:float,seed:int,*,channel:str='depolarizing',depth:int=6,
             budgets:list[int]|None=None,validate:bool=True,
             modes:tuple[str,...]=MODES,peak_caps:dict[int,int]|None=None)->list[dict]:
    if budgets is None or len(budgets)!=depth:raise ValueError('frozen layer budget required')
    if validate and n>8:raise ValueError('oracle limited to <=8 qubits')
    layers=gate_layers(n,depth,seed)
    ops=kraus(channel,p)
    caps=CAPS if peak_caps is None else peak_caps
    ordered=list(modes[seed%len(modes):]+modes[:seed%len(modes)])
    rows=[]
    states={}
    for mode in ordered:
        if mode=='qc08':
            sim=StreamingJointChain(n,max_bond=12,max_env=12,work_cap_entries=750_000)
        else:
            kind='full' if mode=='qc09_full' else 'sketch'
            sim=CertifiedPeakChain(n,max_bond=12,max_env=12,work_cap_entries=750_000,
                                   gate_method=kind,oversample=4,power_iters=2,
                                   peak_gate_cap_entries=caps.get(n) if mode=='qc09_peak' else None)
        time_start=perf_counter()
        counts=[];min_ret=1.;blocked=0;rank_plan=[];future_peaks=[]
        for t,layer in enumerate(layers):
            if mode=='qc08':
                for site,u in layer:sim.gate(site,u)
            else:
                sim.gate_layer(layer)
            for site in range(n):sim.channel(site,ops)
            if mode=='qc08':
                r=sim.compress_layer_stream(budgets[t],ops,order='classic',
                                             bond_cap=12,env_cap=12)
                retained_before=sim.normalize()
                rank_plan.append((r.plan.bond_ranks,r.plan.environment_ranks))
            else:
                r=sim.compress_layer_peak(budgets[t],ops,bond_cap=12,env_cap=12)
                retained_before=r.normalized_trace_before
                blocked+=r.plan.blocked_increments
                rank_plan.append((r.plan.joint.bond_ranks,r.plan.joint.environment_ranks))
                future_peaks.append(r.plan.modeled_next_gate_peak_entries)
            min_ret=min(min_ret,retained_before)
            entries=sim.dimensions()['purified_entries']
            counts.append(entries)
            if entries>budgets[t]:raise AssertionError(f'{mode} violates boundary budget')
        seconds=perf_counter()-time_start
        states[mode]=sim
        row={'n':n,'p':p,'seed':seed,'split':'calibration' if seed<3 else 'heldout',
             'channel':channel,'depth':depth,'mode':mode,
             'gate_algorithm':'full' if mode in ('qc08','qc09_full') else 'sketch',
             'peak_cap_entries':caps.get(n) if mode=='qc09_peak' else '',
             'seconds':seconds,'entries':counts[-1],'max_layer_entries':max(counts),
             'min_layer_retention':min_ret,'max_bond':sim.dimensions()['max_bond'],
             'max_env':sim.dimensions()['max_env'],
             'predicted_live_peak_entries':getattr(sim,'modeled_live_peak_entries',
                                        sim.predicted_working_peak_entries),
             'future_gate_peak_max':max(future_peaks) if future_peaks else '',
             'peak_blocked_increments':blocked,
             'num_certified_gates':len(sim.gate_certificates) if mode!='qc08' else '',
             'gate_bound_sum':sum(c.trace_distance_bound for c in sim.gate_certificates) if mode!='qc08' else '',
             'max_projection_defect':max((c.projection_defect for c in sim.gate_certificates),default=0.) if mode!='qc08' else '',
             'sum_error_estimates':sim.error_bound_sum,
             'plans_json':json.dumps(rank_plan),
             'python':platform.python_version(),'numpy':np.__version__,
             'trace_distance':'','density_distance_qc08':'',
             'physical_trace_eigenvalue_min':'',
             'validation_seconds_excluded':'',
            }
        rows.append(row)
    if validate:
        start=perf_counter()
        exact=np.zeros((2**n,2**n),complex);exact[0,0]=1.
        for layer in layers:
            for site,u in layer:exact=apply_operator(exact,u,(site,site+1),n)
            for site in range(n):exact=apply_channel(exact,ops,site,n)
        baseline=states['qc08'].dense_density()
        for row in rows:
            approx=states[row['mode']].dense_density()
            row['trace_distance']=trace_distance(exact,approx)
            row['density_distance_qc08']=trace_distance(baseline,approx)
            row['physical_trace_eigenvalue_min']=np.linalg.eigvalsh((approx+approx.conj().T)/2)[0]
        for row in rows:row['validation_seconds_excluded']=perf_counter()-start
    return rows


def frozen_budgets(n:int)->list[int]:
    schedule=json.loads(Path(__file__).with_name('qc06_budget_schedule.json').read_text())
    return schedule['budgets_by_n'][str(n)]


def grid(ns=(6,8),strengths=(.2,.5),seeds=(3,4,5,6,7),*,channel='depolarizing'):
    out=[]
    for n in ns:
        for p in strengths:
            for seed in seeds:
                rows=run_case(n,p,seed,channel=channel,budgets=frozen_budgets(n))
                out.extend(rows)
                stats={r['mode']:(round(r['trace_distance'],5),round(r['seconds'],3),r['predicted_live_peak_entries']) for r in rows}
                print(n,p,seed,stats,flush=True)
    return out


def write_csv(rows,path:Path):
    if not rows:raise ValueError('empty benchmark')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qubits',default='6,8')
    ap.add_argument('--strengths',default='.2,.5')
    ap.add_argument('--seeds',default='3,4,5,6,7')
    ap.add_argument('--channel',default='depolarizing')
    ap.add_argument('--output',type=Path,default=Path('qc09_heldout.csv'))
    args=ap.parse_args()
    rows=grid(tuple(map(int,args.qubits.split(','))),
              tuple(map(float,args.strengths.split(','))),
              tuple(map(int,args.seeds.split(','))),channel=args.channel)
    write_csv(rows,args.output)
    print('saved',len(rows),'rows to',args.output)

if __name__=='__main__':main()
