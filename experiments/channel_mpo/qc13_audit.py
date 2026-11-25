"""QC-13: independent exact-dyadic audits for QC-12's numerical MPOs.

Local QR / channel / gate contractions remain BLAS/LAPACK computations. Thus
`local_residual_upper` certifies ONLY the supplied local matrix versus stored
SVD factors, and `readout` certifies ONLY the stored final MPO contraction.
`ideal_ledger` is *not* a full end-to-end verified error bound for a generic
48-qubit circuit until those intermediate transformations are checked too.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from unittest.mock import patch
from typing import Sequence
import numpy as np
from qc13_dyadic import (svd_stored_certificate,residual_certificate,
                         mpo_zero_readout_certificate,_upper_float,DyadicReadout)
import qc12_heisenberg as qc12


@dataclass(frozen=True)
class AuditedObservable:
    expectation:float
    ideal_ledger:float
    local_residual_upper:float
    readout:DyadicReadout
    local_certified_count:int
    local_fraction_exact_sum:F
    gate_coeff_defect:float
    kraus_completeness_defect:float
    seconds:float
    cone_width:int
    max_mpo_entries:int
    max_bond:int
    def arithmetic_readout_gap(self):return self.readout.maximum_real_error(self.expectation)


def audited_observable(n:int,layers:Sequence,operators:tuple[np.ndarray,...],
                       targets:Sequence[int],label:str,*,bond_cap:int=16,
                       max_entries:int=1_000_000,max_cone:int=64)->AuditedObservable:
    """Single-threaded, independently checked a posteriori *local* audit.

    The QC-12 simulator is called as-is with its exact local-SVD residual
    function replaced by our integer one. There is no control feedback from
    an exact density oracle. `patch` makes this function single-thread only.
    """
    records=[]
    saved=[]
    original=qc12.OperatorMPO
    class AuditedMPO(original):
        def input_zero_expectation(self):
            saved.append(mpo_zero_readout_certificate(self.cores))
            return super().input_zero_expectation()

    def checked_residual(m,u,s,vh,keep,block=256):
        r=svd_stored_certificate(m,u,s,vh,keep)
        records.append(r)
        return r.upper

    # Validate inputs before running, through QC-12's standard checks.
    with patch.object(qc12,'OperatorMPO',AuditedMPO),patch.object(qc12,'_independent_residual',checked_residual):
        result=qc12.operator_cone_expectation(n,layers,operators,targets,label,
                   bond_cap=bond_cap,max_entries=max_entries,max_cone=max_cone)
    if len(saved)!=1 or len(records)!=len(result.local_residuals):
        raise AssertionError('dyadic audit was not applied to every gate')
    upper_sum=_upper_float(sum((F(r.upper) for r in records),F(0)))
    defects=[residual_certificate(np.eye(4,dtype=complex),u.conj().T,u).upper
             for layer in layers for _,u in layer]
    # For the positive global CPTP completeness defect, bounding each local
    # Kraus term separately would be too weak. The independent exact sum is
    # computed via qc13_kraus_completeness_certificate below.
    from qc13_dyadic import kraus_completeness_certificate
    defect=kraus_completeness_certificate(operators)
    return AuditedObservable(result.expectation,result.error_bound,upper_sum,saved[0],
                             len(records),sum((F(r.upper) for r in records),F(0)),
                             max(defects,default=0.),defect,result.seconds,
                             result.cone_width,result.max_mpo_entries,result.max_bond)
