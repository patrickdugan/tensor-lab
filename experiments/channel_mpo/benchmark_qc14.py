"""QC-14 reproducible end-to-end dyadic certification benchmark.

Validation references are floating-point independent density/cone simulators;
the `audited_error_upper` is the certified quantity against an explicitly
polar-corrected physical circuit. Oracle comparisons are diagnostics only.

Reproduction (single-threaded BLAS recommended):
  python benchmark_qc14.py --grid quick --output qc14_quick.csv
  python benchmark_qc14.py --grid deep --output qc14_deep.csv
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from channel_mpo import Z, gate_layers, kraus
from qc10_cone import exact_cone_density, backward_cone
from qc11_streaming_cone import streaming_cone_density
from qc12_heisenberg import operator_cone_expectation
from qc14_certified import certify_observable


def comparison(n:int,depth:int,seed:int,channel:str,p:float,bond_cap:int,target:int):
    layers=gate_layers(n,depth,seed)
    operators=kraus(channel,p)
    r=certify_observable(n,layers,operators,(target,),'Z',bond_cap=bond_cap)
    control=operator_cone_expectation(n,layers,operators,(target,),'Z',bond_cap=bond_cap)
    assert abs(control.expectation-r.reported_expectation)<1e-11
    cone=backward_cone(n,layers,(target,))
    exact=None
    if len(cone.sites)<=8:
        if len(cone.sites)<=6:
            rho=exact_cone_density(n,layers,operators,(target,),max_sites=8).reduced_density
        else:
            rho=streaming_cone_density(n,layers,operators,(target,),max_sites=8).reduced_density
        exact=float(np.trace(rho@Z).real)
    by={kind:sum(e.global_residual_upper for e in r.events if e.kind==kind)
        for kind in ('channel','left_qr','right_qr','gate_svd')}
    return dict(n=n,depth=depth,seed=seed,channel=channel,p=p,target=target,
                bond_cap=bond_cap,cone_width=r.cone_width,
                expectation=r.reported_expectation,
                qc12_numerical_ledger=control.error_bound,
                qc14_certified_error_upper=r.reported_error_upper,
                certified_within_005=r.reported_error_upper<=.05,
                literal_dense_reference=exact,
                literal_reference_abs_error=(abs(exact-r.reported_expectation) if exact is not None else ''),
                numerical_rounding_and_qr_upper=by['left_qr']+by['right_qr']+by['channel'],
                gate_and_truncation_upper=by['gate_svd'],
                polar_corrections_upper=r.polar_correction_sum_upper,
                exact_readout_rounding_upper=r.readout_rounding_upper,
                number_exact_events=len(r.events),max_mpo_entries=r.max_mpo_entries,
                audit_seconds=r.seconds,
                certificate_target='polar-corrected exact-binary64 channel/gates')


def grid(name:str):
    if name=='quick':
        for n,depth,cap,target,seeds in [(10,6,4,0,(3,4,5)),(12,8,8,0,(3,4,5))]:
            for seed in seeds:
                for channel in ('dephasing','depolarizing','amplitude_damping'):
                    yield n,depth,seed,channel,.5,cap,target
    elif name=='scaling':
        for seed in (3,5):
            for channel in ('dephasing','depolarizing','amplitude_damping'):
                yield 48,10,seed,channel,.5,4,22
    elif name=='deep':
        yield 48,16,3,'dephasing',.5,8,22
    else:raise ValueError('unknown grid')


def run_grid(name:str):
    rows=[]
    for x in grid(name):
        row=comparison(*x)
        rows.append(row)
        print(f"{row['n']}q d{row['depth']} {row['channel']} seed {row['seed']} "
              f"QC12={row['qc12_numerical_ledger']:.5g} "
              f"QC14={row['qc14_certified_error_upper']:.5g} "
              f"seconds={row['audit_seconds']:.2f}",flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--grid',choices=('quick','scaling','deep'),required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    rows=run_grid(args.grid)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as handle:
        out=csv.DictWriter(handle,fieldnames=list(rows[0]))
        out.writeheader();out.writerows(rows)
    print('Wrote',len(rows),'rows to',args.output,flush=True)

if __name__=='__main__':main()
