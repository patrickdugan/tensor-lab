"""QC-13 rigorously scoped acceptance and universal physical-range fallback.

`clifford_accept` uses an exact oracle and is valid ONLY for the explicitly
specified synthetic Clifford circuits and ideal analytic noise parameter p.
`physical_range_bound` certifies a much looser bound for any physical circuit
whose final local channel is the declared CPTP map; it is independent of the
QC-12 truncation ledger and the trustworthiness of floating-point simulation.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from math import isfinite
from typing import Sequence
from channel_mpo import kraus
from qc12_heisenberg import operator_cone_expectation
from qc13_clifford import exact_clifford_expectation
from qc13_dyadic import DyadicReadout,_upper_float


def _sqrt_bounds(q:F,bits=160)->tuple[F,F]:
    from qc13_clifford import _sqrt_bracket
    return _sqrt_bracket(q,bits)


def physical_pauli_range(channel:str,p:F,label:str)->tuple[F,F]:
    """Rigorous spectrum enclosure of E^dag(P) for P=X,Y,Z.

    Since earlier gates and channels map a density matrix to a density matrix,
    this is a universal expectation range for the final measured Pauli.
    """
    p=F(p)
    if not 0<=p<=1 or label not in 'XYZ':raise ValueError('invalid Pauli/strength')
    if channel=='depolarizing':width=1-p;return -width,width
    if channel=='dephasing':
        width=F(1) if label=='Z' else abs(1-2*p)
        return -width,width
    if channel=='amplitude_damping':
        if label=='Z':return 2*p-1,F(1)
        lo,hi=_sqrt_bounds(1-p)
        return -hi,hi
    raise ValueError('invalid channel')


def physical_range_bound(channel:str,p:F,label:str,readout:DyadicReadout)->float:
    """End-to-end but often weak |true observable - readout| upper.

    Explicit assumptions: ideal physical channel coefficients (parameter p),
    normalized input state and valid CPTP evolution, with the final channel
    applied to the measured site. The approximate MPO output can be arbitrary.
    """
    lo,hi=physical_pauli_range(channel,p,label)
    approx_lo=F(readout.real_lo);approx_hi=F(readout.real_hi)
    error=max(abs(x-y) for x in (lo,hi) for y in (approx_lo,approx_hi))
    return _upper_float(error)


@dataclass(frozen=True)
class CertifiedCliffordDecision:
    accepted:bool
    chosen_cap:int|None
    error_upper:float
    tried_caps:tuple[int,...]
    expectation:float|None
    exact_reference_interval:tuple[float,float]
    certification_scope:str='exact_Clifford_model_only'


def threshold_clifford(n:int,clifford_layers,channel:str,p:F,target:int,
                       *,caps:Sequence[int]=(1,2,4,8),max_error:float=.05,
                       max_terms:int=30_000)->CertifiedCliffordDecision:
    """Select rank using the exact symbolic oracle (special-case only).

    The oracle is queried in selection. Therefore this result is not a claim
    that a generic circuit admits data-independent error-certified ranks.
    """
    caps=tuple(caps)
    if not caps or caps!=tuple(sorted(set(caps))) or any(x<1 for x in caps):
        raise ValueError('caps must be strictly increasing positive integers')
    if not isfinite(max_error) or max_error<0:raise ValueError('bad threshold')
    ref=exact_clifford_expectation(n,clifford_layers,channel,F(p),target,max_terms=max_terms)
    gates=[[(i,u) for i,u,_ in layer] for layer in clifford_layers]
    last=float('inf')
    for j,cap in enumerate(caps):
        r=operator_cone_expectation(n,gates,kraus(channel,float(p)),(target,),'Z',bond_cap=cap)
        last=ref.absolute_difference(r.expectation)
        if last<=max_error:
            return CertifiedCliffordDecision(True,cap,last,caps[:j+1],r.expectation,
                                              (ref.expectation_lo,ref.expectation_hi))
    return CertifiedCliffordDecision(False,None,last,caps,None,
                                     (ref.expectation_lo,ref.expectation_hi))
