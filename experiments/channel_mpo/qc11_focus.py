"""QC-11: observable-aware Kraus/bond allocation using backward light cones.

A forward compression at layer t weights sites according to the *known* backward
cone of the requested final observable, at the post-layer-t time slice. The
weights are fixed before simulation and use only circuit topology, not exact
reference densities, measured target errors, or hidden validation results.

Target-site and bond weights rescale singular-value *scores* in QC-09's greedy
rank scheduler. Actual SVD compression still uses the original tensor values.
Positivity follows from the purified representation. A nominal retained-entry
cap is exact. The prospective live-buffer cap is QC-09's heuristic model, not
an OS RSS guarantee. Fidelity is assessed independently after simulation.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Sequence
import numpy as np
from qc10_cone import backward_cone, Cone
from qc09_certified import CertifiedPeakChain, allocate_peak_safe, PeakPlan
from qc07_joint import pauli_transfer_norm


@dataclass(frozen=True)
class TargetSchedule:
    n:int
    depth:int
    targets:tuple[tuple[int,...],...]
    active_after:tuple[tuple[int,...],...]


def target_schedule(n:int, layers:Sequence, targets:Sequence[Sequence[int]]) -> TargetSchedule:
    """Exact-in-topology backwards support for multiple nominated observables."""
    if not targets: raise ValueError('must nominate an observable region')
    canonical=[]
    for region in targets:
        sites=tuple(region)
        if not sites or tuple(sorted(set(sites)))!=sites or any(
                not isinstance(site,int) or site<0 or site>=n for site in sites):
            raise ValueError('target sites must be sorted, distinct, in range')
        if sites in canonical: raise ValueError('duplicate region')
        canonical.append(sites)
    cones=[backward_cone(n,layers,region) for region in canonical]
    return TargetSchedule(n,len(layers),tuple(canonical),
        tuple(tuple(sorted(set().union(*(set(c.active_before[t+1]) for c in cones))))
              for t in range(len(layers))))


def cone_weights(n:int, active:Sequence[int], *,off_cone:float=.02,
                 halo:int=0) -> tuple[np.ndarray,np.ndarray]:
    """Importance of each spatial bond and each local environment rank.

    The bond spanning the cone boundary is weighted because its Schmidt space
    can influence the reduced state after tracing the exterior. A bond entirely
    outside the support receives off_cone. Halo is an optional conservative
    spatial enlargement; it never narrows the true backward cone.
    """
    if n<2 or not active or halo<0 or not isinstance(halo,int):
        raise ValueError('invalid chain/focus/halo')
    if not np.isfinite(off_cone) or not 0 < off_cone <= 1:
        raise ValueError('off_cone must be in (0,1]')
    sites=tuple(active)
    if sites!=tuple(sorted(set(sites))) or any(not isinstance(i,int) or i<0 or i>=n for i in sites):
        raise ValueError('active sites must be sorted, distinct, in range')
    extended=set()
    for site in sites:
        extended.update(range(max(0,site-halo),min(n,site+halo+1)))
    env=np.array([1. if site in extended else off_cone for site in range(n)],dtype=float)
    bond=np.array([1. if (i in extended or i+1 in extended) else off_cone
                   for i in range(n-1)],dtype=float)
    return bond,env


@dataclass(frozen=True)
class FocusResult:
    plan:PeakPlan
    retained_entries:int
    min_retained_fraction:float
    seconds:float
    focused_sites:int
    focused_bonds:int


class ObservablePeakChain(CertifiedPeakChain):
    """QC-09 gate method with QC-11 observable-weighted rank allocation."""
    def compress_focused(self,budget_entries:int,operators:tuple[np.ndarray,...],
                         schedule:TargetSchedule,layer:int,*,off_cone:float=.02,
                         halo:int=0)->FocusResult:
        if schedule.n!=self.n or not 0<=layer<schedule.depth:
            raise ValueError('mismatched time slice or system')
        t0=perf_counter()
        weights_b,weights_e=cone_weights(self.n,schedule.active_after[layer],
                                        off_cone=off_cone,halo=halo)
        chosen:list[PeakPlan]=[]
        def allocator(bond_s,env_s,budget_entries,bond_cap,env_cap,*,channel_bond_weight):
            sb=[np.asarray(s)*np.sqrt(w) for s,w in zip(bond_s,weights_b)]
            se=[np.asarray(s)*np.sqrt(w) for s,w in zip(env_s,weights_e)]
            peak=allocate_peak_safe(sb,se,budget_entries,bond_cap,env_cap,
                      channel_bond_weight=channel_bond_weight,
                      peak_cap_entries=self.peak_gate_cap_entries,
                      gate_method=self.gate_method,oversample=self.oversample,
                      safety_factor=self.planner_safety_factor,
                      kraus_count=len(operators),account_channel_peak=True)
            chosen.append(peak)
            return peak.joint
        result=self.compress_layer_stream(budget_entries,operators,
                    bond_cap=self.max_bond,env_cap=self.max_env,
                    order='classic',rank_allocator=allocator)
        norm=self.normalize()
        if len(chosen)!=1 or result.retained_entries>budget_entries:
            raise AssertionError('focused allocator invalid')
        return FocusResult(chosen[0],result.retained_entries,norm,
                           perf_counter()-t0,int(np.count_nonzero(weights_e==1.)),
                           int(np.count_nonzero(weights_b==1.)))

