"""QC-06: matched *layer-boundary* storage budgets, exact small-n scoring.

The cap-6 baseline is first evolved for each layer. Its *current* retained
entry count supplies the adaptive policy's numerical budget. This is an
experimentally useful online budget schedule, but generating it requires a
separate baseline rollout. Report both adaptive-only and end-to-end paired
runtime; the selected budget schedule is not free in an actual deployment.

Exact densities are constructed only after BOTH approximate algorithms finish,
never used in allocation. Max working entries omit temporary LAPACK allocations
and are not equivalent to process RSS.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from channel_mpo import apply_channel, apply_operator, gate_layers, kraus
from purified_mpo import PurifiedChain
from qc06_adaptive import AdaptivePurifiedChain


def trace_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = (a - b + (a - b).conj().T) / 2
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(diff))))


def run_case(n: int, depth: int, seed: int, p: float,
             bond: int = 12, uniform_env: int = 6,
             adaptive_site_cap: int = 12, validate: bool = True,
             channel: str = 'depolarizing',
             frozen_budgets: list[int] | None = None) -> dict:
    if n < 2 or depth < 1 or bond < 1 or uniform_env < 1 or adaptive_site_cap < 1:
        raise ValueError('invalid benchmark settings')
    if validate and n > 8:
        raise ValueError('dense validation limited to n<=8')
    if frozen_budgets is not None and (len(frozen_budgets) != depth or any(int(x) < 1 for x in frozen_budgets)):
        raise ValueError('frozen budget schedule must contain one positive entry per layer')
    ops = kraus(channel, p)
    layers = gate_layers(n, depth, seed)
    uniform = PurifiedChain(n, max_bond=bond, max_env=uniform_env)
    adaptive = AdaptivePurifiedChain(n, max_bond=bond, max_env=adaptive_site_cap)
    runtimes = {'uniform': 0., 'adaptive': 0.}
    budgets, used_entries, ranks = [], [], []
    min_ret = {'uniform': 1., 'adaptive': 1.}
    min_per_layer_ret = {'uniform': [], 'adaptive': []}
    alloc_probe_svds = 0; alloc_svds = 0
    pred_losses=[]; actual_losses=[]

    for layer_index, layer in enumerate(layers):
        tic=perf_counter()
        for i, u in layer: uniform.gate(i, u)
        for i in range(n): uniform.channel(i, ops)
        for i in range(n): uniform.compress_env(i)
        retention_uniform = uniform.normalize()
        runtimes['uniform'] += perf_counter() - tic
        measured_uniform_entries = uniform.dimensions()['purified_entries']
        budget = (measured_uniform_entries if frozen_budgets is None
                  else int(frozen_budgets[layer_index]))
        if measured_uniform_entries != budget:
            raise ValueError(f'frozen budget {budget} differs from baseline entries {measured_uniform_entries} at layer {layer_index+1}')
        budgets.append(budget)
        min_ret['uniform'] = min(min_ret['uniform'], retention_uniform)
        min_per_layer_ret['uniform'].append(retention_uniform)

        tic=perf_counter()
        for i, u in layer: adaptive.gate(i, u)
        for i in range(n): adaptive.channel(i, ops)
        allocation = adaptive.compress_layer_budgeted(budget, max_site_env=adaptive_site_cap)
        retention_adapt = adaptive.normalize()
        runtimes['adaptive'] += perf_counter() - tic
        min_ret['adaptive'] = min(min_ret['adaptive'], retention_adapt)
        min_per_layer_ret['adaptive'].append(retention_adapt)
        used_entries.append(allocation.retained_entries)
        ranks.append(allocation.target_ranks)
        alloc_probe_svds += allocation.probe_svds
        alloc_svds += allocation.compression_svds
        pred_losses.append(allocation.predicted_discarded_weight)
        actual_losses.append(allocation.post_compression_loss)

    distance_uniform = float('nan'); distance_adapt=float('nan')
    z_error_uniform=float('nan'); z_error_adapt=float('nan')
    oracle_seconds=0.
    if validate:
        tic=perf_counter()
        exact=np.zeros((2**n,2**n), complex)
        exact[0,0]=1.
        for layer in layers:
            for i, u in layer: exact=apply_operator(exact,u,(i,i+1),n)
            for i in range(n): exact=apply_channel(exact,ops,i,n)
        distance_uniform=trace_distance(exact,uniform.dense_density())
        distance_adapt=trace_distance(exact,adaptive.dense_density())
        ref_z=float((np.diag(exact).real[:2**(n-1)]).sum()-(np.diag(exact).real[2**(n-1):]).sum())
        z_error_uniform=abs(ref_z-uniform.z_expectations()[0])
        z_error_adapt=abs(ref_z-adaptive.z_expectations()[0])
        oracle_seconds=perf_counter()-tic
    return dict(n=n,depth=depth,seed=seed,p=p,channel=channel,
                split='calibration' if seed<3 else 'heldout',
                bond_cap=bond,uniform_env_cap=uniform_env,
                adaptive_site_env_cap=adaptive_site_cap,
                distance_uniform=distance_uniform,distance_adaptive=distance_adapt,
                distance_delta=distance_adapt-distance_uniform,
                z0_error_uniform=z_error_uniform,z0_error_adaptive=z_error_adapt,
                seconds_uniform=runtimes['uniform'],seconds_adaptive=runtimes['adaptive'],
                seconds_adaptive_including_budget_generation=sum(runtimes.values()),
                seconds_dense_oracle_excluded=oracle_seconds,
                final_budget_entries=budgets[-1],final_adaptive_entries=used_entries[-1],
                peak_boundary_budget_entries=max(budgets),
                peak_boundary_adaptive_entries=max(used_entries),
                strict_matched_budget=all(x<=b for b,x in zip(budgets,used_entries)),
                frozen_schedule=frozen_budgets is not None,
                max_working_entries_uniform=uniform.dimensions()['largest_working_entries'],
                max_working_entries_adaptive=adaptive.dimensions()['largest_working_entries'],
                max_site_environment_uniform=uniform.dimensions()['max_env'],
                max_site_environment_adaptive=adaptive.dimensions()['max_env'],
                min_retention_uniform=min_ret['uniform'],
                min_retention_adaptive=min_ret['adaptive'],
                allocated_ranks_by_layer=json.dumps(ranks),
                baseline_boundary_budget_by_layer=json.dumps(budgets),
                adaptive_boundary_entries_by_layer=json.dumps(used_entries),
                retention_uniform_by_layer=json.dumps(min_per_layer_ret['uniform']),
                retention_adaptive_by_layer=json.dumps(min_per_layer_ret['adaptive']),
                predicted_discarded_weight_sum=sum(pred_losses),
                actual_discarded_weight_sum=sum(actual_losses),
                adaptive_probe_svds=alloc_probe_svds,
                adaptive_compress_svds=alloc_svds,
                python=platform.python_version(),numpy=np.__version__)


def run_grid(ns=(6,8),strengths=(.2,.5),seeds=range(8),depth=6,
             bond=12,uniform_env=6,adaptive_site_cap=12,validate=True,
             frozen_schedules: dict[int,list[int]] | None = None):
    rows=[]
    for n in ns:
        for p in strengths:
            for seed in seeds:
                try:
                    schedule = frozen_schedules.get(n) if frozen_schedules is not None else None
                    if frozen_schedules is not None and schedule is None:
                        raise ValueError(f'missing frozen budget schedule for n={n}')
                    x=run_case(n,depth,seed,p,bond,uniform_env,adaptive_site_cap,validate,
                               frozen_budgets=schedule)
                except ValueError as exc:
                    print(f'INFEASIBLE n={n} p={p} seed={seed}: {exc}',flush=True)
                    raise
                rows.append(x)
                print(f'{n}q p={p} seed={seed} '
                      f'dist(uniform/adaptive)={x["distance_uniform"]:.5f}/{x["distance_adaptive"]:.5f} '
                      f'entries={x["final_adaptive_entries"]}/{x["final_budget_entries"]} '
                      f'sec={x["seconds_uniform"]:.3f}/{x["seconds_adaptive"]:.3f}',flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qubits',default='6,8')
    ap.add_argument('--strengths',default='0.2,0.5')
    ap.add_argument('--seeds',default='0,1,2,3,4,5,6,7')
    ap.add_argument('--depth',type=int,default=6)
    ap.add_argument('--bond',type=int,default=12)
    ap.add_argument('--uniform-env',type=int,default=6)
    ap.add_argument('--adaptive-site-cap',type=int,default=12)
    ap.add_argument('--no-validate',action='store_true')
    ap.add_argument('--frozen-budgets',type=Path,default=None,
                    help='pinned calibration budget JSON to replay on held-out seeds')
    ap.add_argument('--output',type=Path,default=Path('qc06_results.csv'))
    a=ap.parse_args()
    schedules = None
    if a.frozen_budgets is not None:
        data=json.loads(a.frozen_budgets.read_text())
        if data['circuit_depth']!=a.depth or data['bond_cap']!=a.bond or data['uniform_env_cap']!=a.uniform_env:
            ap.error('frozen schedule protocol mismatch')
        schedules={int(k):v for k,v in data['budgets_by_n'].items()}
    rows=run_grid(tuple(int(v) for v in a.qubits.split(',')),
                  tuple(float(v) for v in a.strengths.split(',')),
                  tuple(int(v) for v in a.seeds.split(',')),a.depth,
                  a.bond,a.uniform_env,a.adaptive_site_cap,not a.no_validate,
                  frozen_schedules=schedules)
    a.output.parent.mkdir(exist_ok=True,parents=True)
    with a.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(f'SAVED {len(rows)} rows -> {a.output}')

if __name__=='__main__':main()
