"""QC-13: exact-symbolic 48q probes, tiny rational, and dyadic local audits.

Frozen CSVs separate rigorous (exact oracle/dyadic components) from numerical
(local-ledger approximations). The Clifford oracle is valid only for this
synthetic exact Clifford ensemble. The dyadic local audit is NOT an end-to-end
formal certificate for generic floating Haar circuits.
"""
from __future__ import annotations
import argparse,csv
from fractions import Fraction as F
from pathlib import Path
from time import perf_counter
from channel_mpo import gate_layers,kraus
from qc12_heisenberg import operator_cone_expectation
from qc13_audit import audited_observable
from qc13_clifford import clifford_layers,exact_clifford_expectation
from qc13_rational import exact_circuit_expectation
from qc13_acceptance import physical_range_bound


def clifford_grid(*,n=48,depth=16,seeds=(3,4,5,6,7),
                  channels=('dephasing','depolarizing','amplitude_damping'),
                  strengths=(F(1,5),F(1,2)),caps=(1,2,4,8),target=22,
                  max_terms=20_000):
    rows=[]
    for seed in seeds:
        layerlist=clifford_layers(n,depth,seed)
        gates=[[(site,u) for site,u,_ in layer] for layer in layerlist]
        for channel in channels:
            for p in strengths:
                t=perf_counter()
                try:
                    ref=exact_clifford_expectation(n,layerlist,channel,p,target,
                                                   max_terms=max_terms)
                except MemoryError:
                    rows.append(dict(n=n,depth=depth,seed=seed,channel=channel,
                        p=str(p),target=target,bond_cap='oracle_cap',
                        oracle_supported=False,oracle_max_terms=max_terms))
                    continue
                oracle_seconds=perf_counter()-t
                for cap in caps:
                    sim=operator_cone_expectation(n,gates,kraus(channel,float(p)),
                        (target,),'Z',bond_cap=cap)
                    true_error_upper=ref.absolute_difference(sim.expectation)
                    rows.append(dict(n=n,depth=depth,seed=seed,channel=channel,
                        p=str(p),target=target,bond_cap=cap,oracle_supported=True,
                        oracle_max_terms=ref.max_terms,oracle_final_terms=ref.final_terms,
                        reference_lo=ref.expectation_lo,reference_hi=ref.expectation_hi,
                        approximate=sim.expectation,exact_family_error_upper=true_error_upper,
                        accepts_true_error_005=bool(true_error_upper<=.05),
                        accepts_true_error_001=bool(true_error_upper<=.01),
                        numerical_QC12_ledger=sim.error_bound,
                        qc12_ledger_accepts_005=bool(sim.error_bound<=.05),
                        max_mpo_entries=sim.max_mpo_entries,sim_seconds=sim.seconds,
                        oracle_seconds=oracle_seconds,
                        end_to_end_certification='exact_Clifford_ideal_reference_only'))
        print('clifford seed',seed,'records',len(rows),flush=True)
    return rows


def tiny_rational_grid(*,n=3,depth=3,seeds=(0,1,2,3,4),
                       channels=('dephasing','depolarizing','amplitude_damping'),
                       strengths=(.2,.5)):
    rows=[]
    for seed in seeds:
        gates=gate_layers(n,depth,seed)
        for channel in channels:
            for p in strengths:
                ops=kraus(channel,p)
                t=perf_counter()
                ref=exact_circuit_expectation(n,gates,ops,(0,),'Z')
                oracle_seconds=perf_counter()-t
                for cap in (1,2,8):
                    result=operator_cone_expectation(n,gates,ops,(0,),'Z',bond_cap=cap)
                    rows.append(dict(n=n,depth=depth,seed=seed,channel=channel,p=p,
                        rank_cap=cap,reference_lo=ref.expectation_real_lo,
                        reference_hi=ref.expectation_real_hi,
                        exact_float_circuit_error_upper=ref.absolute_difference(result.expectation),
                        numerical_qc12_bound=result.error_bound,
                        exact_float_trace_defect=float(ref.trace_real_error),
                        sim_seconds=result.seconds,reference_seconds=oracle_seconds,
                        end_to_end_certification='exact_supplied_binary64_model'))
    return rows


def dyadic_audit_grid(*,n=48,depth=16,seeds=(3,),
                      channels=('dephasing','depolarizing','amplitude_damping'),
                      strengths=(.2,.5),cap=16,target=22):
    rows=[]
    for seed in seeds:
        layers=gate_layers(n,depth,seed)
        for channel in channels:
            for p in strengths:
                result=audited_observable(n,layers,kraus(channel,p),(target,),'Z',bond_cap=cap)
                rows.append(dict(n=n,depth=depth,seed=seed,channel=channel,p=p,
                    target=target,rank_cap=cap,expectation=result.expectation,
                    original_numerical_ledger=result.ideal_ledger,
                    exact_dyadic_local_ledger=result.local_residual_upper,
                    local_residuals_audited=result.local_certified_count,
                    exact_dyadic_readout_error=result.arithmetic_readout_gap(),
                    max_gate_unitary_defect=result.gate_coeff_defect,
                    kraus_completeness_defect=result.kraus_completeness_defect,
                    max_mpo_entries=result.max_mpo_entries,seconds=result.seconds,
                    physical_range_error_upper=physical_range_bound(channel,F(str(p)),'Z',result.readout),
                    end_to_end_certified=False,
                    remaining_unchecked='QR and channel/gate contractions; physical idealization'))
                print('audited',seed,channel,p,'ledger',result.local_residual_upper,flush=True)
    return rows


def save(rows,path:Path):
    if not rows:raise ValueError('empty benchmark')
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print('wrote',len(rows),'rows to',path,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('clifford','tiny','audit'),required=True)
    p.add_argument('--n',type=int,default=None)
    p.add_argument('--depth',type=int,default=None)
    p.add_argument('--seeds',default='3,4,5,6,7')
    p.add_argument('--channels',default='dephasing,depolarizing,amplitude_damping')
    p.add_argument('--strengths',default='.2,.5')
    p.add_argument('--caps',default='1,2,4,8')
    p.add_argument('--target',type=int,default=22)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    ints=lambda s:tuple(int(x) for x in s.split(','))
    strings=lambda s:tuple(s.split(','))
    if a.mode=='clifford':
        rows=clifford_grid(n=a.n or 48,depth=a.depth or 16,seeds=ints(a.seeds),
            channels=strings(a.channels),strengths=tuple(F(p) for p in a.strengths.split(',')),
            caps=ints(a.caps),target=a.target)
    elif a.mode=='tiny':
        rows=tiny_rational_grid(n=a.n or 3,depth=a.depth or 3,seeds=ints(a.seeds),
            channels=strings(a.channels),strengths=tuple(float(p) for p in a.strengths.split(',')))
    else:
        rows=dyadic_audit_grid(n=a.n or 48,depth=a.depth or 16,seeds=ints(a.seeds),
            channels=strings(a.channels),strengths=tuple(float(p) for p in a.strengths.split(',')),
            cap=ints(a.caps)[0],target=a.target)
    save(rows,a.output)

if __name__=='__main__':main()
