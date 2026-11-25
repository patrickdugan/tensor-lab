"""QC-07: paired three-policy benchmark under QC-06 frozen entry schedules.

Compute the same layer-by-layer evolution for uniform env6, QC-06 adaptive
ancilla, and QC-07 joint ancilla/bond ranks.  All decisions use current
network spectra and the predeclared budgets.  The exact density is computed
only afterward, outside the reported algorithm times, and only for n<=8.

The workspace cap is a preflight model of explicit live-array entries, NOT an
OS or LAPACK peak-memory guarantee.  Separate subprocess RSS is measured in
benchmark_peak_qc07.py.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import platform
import numpy as np

from channel_mpo import gate_layers, kraus, apply_operator, apply_channel
from purified_mpo import PurifiedChain
from qc06_adaptive import AdaptivePurifiedChain
from qc07_joint import JointPurifiedChain


def trace_distance(x: np.ndarray, y: np.ndarray) -> float:
    d = (x-y + (x-y).conj().T)/2
    return float(0.5*np.sum(abs(np.linalg.eigvalsh(d))))


def run_case(n:int, seed:int, p:float, *, depth:int=6,
             channel:str='depolarizing',bond_cap:int=12,env_cap:int=6,
             joint_bond_cap:int=12,joint_env_cap:int=12,
             work_cap_entries:int|None=500000,
             bond_weight_override:float|None=None,
             budget_by_layer:list[int]|None=None,
             validate:bool=True)->dict:
    if n<2 or depth<1 or bond_cap<1 or env_cap<1 or joint_bond_cap<1 or joint_env_cap<1:
        raise ValueError('invalid benchmark configuration')
    if validate and n>8:raise ValueError('dense oracle limited to n<=8')
    if budget_by_layer is None or len(budget_by_layer)!=depth:
        raise ValueError('pass a predeclared layer budget')
    ops=kraus(channel,p)
    layers=gate_layers(n,depth,seed)
    uniform=PurifiedChain(n,bond_cap,env_cap)
    adaptive=AdaptivePurifiedChain(n,bond_cap,joint_env_cap)
    joint=JointPurifiedChain(n,max_bond=joint_bond_cap,
                             max_env=joint_env_cap,work_cap_entries=work_cap_entries)
    runtime=dict(uniform=0.,qc06=0.,qc07=0.)
    retained=[];target_bonds=[];target_envs=[]
    measured=[];losses=[];marginal=[]
    layer_ret={'uniform':[],'qc06':[],'qc07':[]}
    for t,layer in enumerate(layers):
        tic=perf_counter()
        for i,u in layer:uniform.gate(i,u)
        for i in range(n):uniform.channel(i,ops)
        for i in range(n):uniform.compress_env(i)
        observed=uniform.dimensions()['purified_entries']
        if observed!=budget_by_layer[t]:
            raise ValueError(f'QC-06 budget mismatch at layer {t}: {observed}!={budget_by_layer[t]}')
        layer_ret['uniform'].append(uniform.normalize())
        runtime['uniform']+=perf_counter()-tic
        tic=perf_counter()
        for i,u in layer:adaptive.gate(i,u)
        for i in range(n):adaptive.channel(i,ops)
        allocation=adaptive.compress_layer_budgeted(observed,max_site_env=joint_env_cap)
        layer_ret['qc06'].append(adaptive.normalize())
        runtime['qc06']+=perf_counter()-tic
        tic=perf_counter()
        for i,u in layer:joint.gate(i,u)
        for i in range(n):joint.channel(i,ops)
        jr=joint.compress_layer_joint(observed,ops,bond_cap=joint_bond_cap,env_cap=joint_env_cap,
                                      bond_weight_override=bond_weight_override)
        layer_ret['qc07'].append(joint.normalize())
        runtime['qc07']+=perf_counter()-tic
        retained.append(jr.retained_entries)
        measured.append(observed)
        target_bonds.append(jr.plan.bond_ranks)
        target_envs.append(jr.plan.environment_ranks)
        losses.append(jr.layer_error_bound_increment)
        marginal.append(max(jr.marginal_changes))
        if jr.retained_entries>observed or allocation.retained_entries>observed:
            raise AssertionError('layer budget exceeded')
    exact_seconds=0.
    distances={k:float('nan') for k in ('uniform','qc06','qc07')}
    z0errors={k:float('nan') for k in distances}
    if validate:
        tic=perf_counter()
        exact=np.zeros((2**n,2**n),complex);exact[0,0]=1
        for layer in layers:
            for i,u in layer:exact=apply_operator(exact,u,(i,i+1),n)
            for i in range(n):exact=apply_channel(exact,ops,i,n)
        rho={'uniform':uniform.dense_density(),
             'qc06':adaptive.dense_density(),
             'qc07':joint.dense_density()}
        ref_z=2*float(np.sum(np.diag(exact).real[:2**(n-1)]))-1
        distances={k:trace_distance(exact,x) for k,x in rho.items()}
        z0errors={k:abs(ref_z - c.z_expectations()[0])
                  for k,c in [('uniform',uniform),('qc06',adaptive),('qc07',joint)]}
        exact_seconds=perf_counter()-tic
    return {
        'n':n,'seed':seed,'split':'calibration' if seed<=2 else 'heldout',
        'depth':depth,'channel':channel,'p':p,
        'bond_cap_baseline':bond_cap,'env_cap_baseline':env_cap,
        'joint_bond_cap':joint_bond_cap,'joint_env_cap':joint_env_cap,
        'channel_bond_weight_override':bond_weight_override,
        'workspace_admission_cap_entries':work_cap_entries,
        'distance_uniform':distances['uniform'],
        'distance_qc06':distances['qc06'],
        'distance_qc07':distances['qc07'],
        'delta_qc07_minus_qc06':distances['qc07']-distances['qc06'],
        'z0_error_uniform':z0errors['uniform'],
        'z0_error_qc06':z0errors['qc06'],
        'z0_error_qc07':z0errors['qc07'],
        'seconds_uniform':runtime['uniform'],
        'seconds_qc06':runtime['qc06'],
        'seconds_qc07':runtime['qc07'],
        'seconds_dense_oracle_excluded':exact_seconds,
        'final_boundary_budget':measured[-1],
        'final_qc07_entries':retained[-1],
        'peak_boundary_budget':max(measured),
        'peak_qc07_boundary':max(retained),
        'matched_boundary_budget':all(v<=b for v,b in zip(retained,measured)),
        'qc07_modeled_working_peak_entries':joint.predicted_working_peak_entries,
        'qc07_largest_svd_matrix_entries':joint.dimensions()['largest_working_entries'],
        'qc06_largest_svd_matrix_entries':adaptive.dimensions()['largest_working_entries'],
        'qc07_max_bond':joint.dimensions()['max_bond'],
        'qc07_max_env':joint.dimensions()['max_env'],
        'qc07_local_trace_bound_clipped':min(1.,float(joint.error_bound_sum)),
        'qc07_trace_bound_sum':float(joint.error_bound_sum),
        'max_single_site_marginal_shift_by_layer':json.dumps(marginal),
        'min_layer_retention_uniform':min(layer_ret['uniform']),
        'min_layer_retention_qc06':min(layer_ret['qc06']),
        'min_layer_retention_qc07':min(layer_ret['qc07']),
        'qc07_bonds_by_layer':json.dumps(target_bonds),
        'qc07_envs_by_layer':json.dumps(target_envs),
        'qc07_error_bound_increment_by_layer':json.dumps(losses),
        'qc06_strict_matched_budget':True,
        'python':platform.python_version(),'numpy':np.__version__,
    }


def run_grid(ns=(6,8),strengths=(.2,.5),seeds=(0,1,2),
             channel='depolarizing',bond_cap=12,env_cap=6,
             joint_bond_cap=12,joint_env_cap=12,depth=6,
             work_cap_entries=500000,validate=True,
             bond_weight_override=None,
             budget_file:Path|None=None):
    if budget_file is None:
        budget_file=Path(__file__).with_name('qc06_budget_schedule.json')
    data=json.loads(budget_file.read_text())
    if data['circuit_depth']!=depth or data['bond_cap']!=bond_cap or data['uniform_env_cap']!=env_cap:
        raise ValueError('frozen QC-06 protocol mismatch')
    rows=[]
    for n in ns:
        budgets=data['budgets_by_n'][str(n)]
        for p in strengths:
            for seed in seeds:
                r=run_case(n,seed,p,depth=depth,channel=channel,bond_cap=bond_cap,
                           env_cap=env_cap,joint_bond_cap=joint_bond_cap,
                           joint_env_cap=joint_env_cap,work_cap_entries=work_cap_entries,
                           budget_by_layer=budgets,validate=validate,
                           bond_weight_override=bond_weight_override)
                rows.append(r)
                print(f"{n}q {channel} p={p} seed={seed} "
                      f"D(QC06/QC07)={r['distance_qc06']:.5f}/{r['distance_qc07']:.5f} "
                      f"entries={r['final_qc07_entries']}/{r['final_boundary_budget']} "
                      f"time={r['seconds_qc06']:.3f}/{r['seconds_qc07']:.3f}s",flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qubits',default='6,8')
    ap.add_argument('--strengths',default='.2,.5')
    ap.add_argument('--seeds',default='0,1,2')
    ap.add_argument('--channel',default='depolarizing')
    ap.add_argument('--bond-cap',type=int,default=12)
    ap.add_argument('--env-cap',type=int,default=6)
    ap.add_argument('--joint-bond-cap',type=int,default=12)
    ap.add_argument('--joint-env-cap',type=int,default=12)
    ap.add_argument('--depth',type=int,default=6)
    ap.add_argument('--work-cap-entries',type=int,default=500000)
    ap.add_argument('--bond-weight-override',type=float,default=None)
    ap.add_argument('--no-validate',action='store_true')
    ap.add_argument('--output',type=Path,default=Path('qc07_calibration.csv'))
    a=ap.parse_args()
    rows=run_grid(tuple(int(x) for x in a.qubits.split(',')),
                  tuple(float(x) for x in a.strengths.split(',')),
                  tuple(int(x) for x in a.seeds.split(',')),
                  a.channel,a.bond_cap,a.env_cap,a.joint_bond_cap,
                  a.joint_env_cap,a.depth,a.work_cap_entries,not a.no_validate,
                  bond_weight_override=a.bond_weight_override)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print('WROTE',len(rows),'records',a.output)

if __name__=='__main__':main()
