"""QC-11: exact-in-model streaming causal-cone contraction.

For a product initial state, local nearest-neighbor gates followed by local
trace-preserving channels, evolve only the backward-dependent degrees of
freedom. At each forward layer, gates that can influence the *future* target
are applied; all sites leaving the backward cone are traced out immediately.
Noise on discarded sites commutes with this partial trace and is omitted.

This computes the physical reduced density directly, with no global MPO, and
uses full local Kraus sums on the active sites. The result is mathematically
exact in ideal arithmetic; NumPy floating-point operations are not formally
verified. The conservative cap checks dense matrix entries before allocation.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from collections.abc import Sequence
import numpy as np
from channel_mpo import apply_channel, apply_operator
from qc10_cone import backward_cone, reduced_density, Cone


@dataclass(frozen=True)
class StreamingConeResult:
    reduced_density:np.ndarray
    cone:Cone
    seconds:float
    relevant_gates:int
    kraus_site_updates:int
    discarded_sites:int
    max_density_entries:int
    peak_density_entries:int
    per_layer_widths:tuple[int,...]


def streaming_cone_density(n:int,layers:Sequence,operators:tuple[np.ndarray,...],
                           targets:Sequence[int],*,max_sites:int=8,
                           max_density_entries:int|None=None)->StreamingConeResult:
    """Return a target density using progressively shrinking causal support.

    Exactness assumptions: product |0>^n initial state; disjoint nearest-
    neighbor gates per layer; identical single-site CPTP channel after every
    gate layer. The gate layout is checked by backward_cone. Permutations,
    long-range gates and non-product inputs require an extended construction.
    """
    c=backward_cone(n,layers,targets)
    # The backward-cone pruning argument assumes physical, unitary gates.
    # Shape/disjointness have already been checked by backward_cone.
    for layer in layers:
        for _,u in layer:
            if not np.isfinite(u).all() or not np.allclose(
                    u.conj().T@u,np.eye(4),atol=1e-11,rtol=0):
                raise ValueError('two-qubit gate must be unitary')
    if not operators or any(np.shape(k)!=(2,2) or not np.isfinite(k).all() for k in operators):
        raise ValueError('finite qubit Kraus operators required')
    completeness=sum((k.conj().T@k for k in operators),np.zeros((2,2),complex))
    if not np.allclose(completeness,np.eye(2),atol=1e-11,rtol=0):
        raise ValueError('channel must be trace preserving')
    if max_sites<1 or len(c.sites)>max_sites:
        raise ValueError('causal cone exceeds maximum exact width')
    predicted=4**len(c.sites)
    if max_density_entries is not None and (max_density_entries<1 or predicted>max_density_entries):
        raise MemoryError('cone matrix exceeds declared entry budget')
    started=perf_counter()
    active=c.active_before[0]
    rho=np.zeros((2**len(active),2**len(active)),dtype=complex)
    rho[0,0]=1
    peak=rho.size
    count_gates=0;count_noise=0;discarded=0
    widths=[len(active)]
    for t,layer in enumerate(layers):
        after=c.active_before[t+1]
        assert set(after)<=set(active)
        lookup={site:i for i,site in enumerate(active)}
        after_set=set(after)
        for j,u in layer:
            if j in after_set or j+1 in after_set:
                # The backward cone guarantees both endpoints in the current
                # active set. A gate outside the new support has no causal
                # effect and may be discarded.
                if j not in lookup or j+1 not in lookup:
                    raise AssertionError('inconsistent causal cone')
                rho=apply_operator(rho,u,(lookup[j],lookup[j+1]),len(active))
                count_gates+=1
        if after!=active:
            rho=reduced_density(rho,len(active),tuple(lookup[j] for j in after))
            discarded+=len(active)-len(after)
        active=after
        # Partial trace commutes with channels on disjoint discarded sites.
        # Applying only the remaining local Kraus maps is exactly equivalent.
        for i in range(len(active)):
            rho=apply_channel(rho,operators,i,len(active))
            count_noise+=1
        peak=max(peak,rho.size)
        widths.append(len(active))
    if tuple(active)!=c.targets:
        raise AssertionError('final causal support must be the nominated sites')
    result=(rho+rho.conj().T)/2
    tr=float(np.trace(result).real)
    if not np.isfinite(tr) or tr<=0:raise FloatingPointError('invalid trace')
    # Avoid concealing large numerical errors behind normalization.
    if abs(tr-1)>1e-7:raise ArithmeticError(f'non-trace-preserving evolution: {tr}')
    return StreamingConeResult(result,c,perf_counter()-started,count_gates,
                               count_noise,discarded,predicted,peak,tuple(widths))


def streaming_region_bundle(n:int,layers:Sequence,operators:tuple[np.ndarray,...],
                            regions:dict[str,Sequence[int]],*,max_sites:int=8,
                            max_density_entries:int|None=None
                            )->tuple[dict[str,np.ndarray],dict[str,StreamingConeResult]]:
    """Compute maximal nominated regions once; obtain nested marginals by trace.

    This is a deterministic geometry-only reuse policy. In particular,
    [0,1] supplies [0], and [n-2,n-1] supplies [n-1]. It performs no oracle-
    based method selection and never uses an approximate global state.
    """
    if not regions: raise ValueError('at least one region required')
    normalized={}
    for label,targets in regions.items():
        selected=tuple(targets)
        if not isinstance(label,str) or not label or not selected or tuple(sorted(set(selected)))!=selected:
            raise ValueError('invalid region definition')
        if any(not isinstance(x,int) or x<0 or x>=n for x in selected):
            raise ValueError('invalid region qubit')
        normalized[label]=selected
    maximal=tuple(sorted((target for target in set(normalized.values())
                          if not any(set(target)<set(other) for other in normalized.values())),
                         key=lambda x:(len(x),x)))
    memo={}
    for target in maximal:
        memo[target]=streaming_cone_density(n,layers,operators,target,
                        max_sites=max_sites,max_density_entries=max_density_entries)
    reduced={}
    for label,target in normalized.items():
        parent=min((candidate for candidate in maximal if set(target)<=set(candidate)),
                   key=lambda x:(len(x),x))
        raw=memo[parent].reduced_density
        if target==parent:
            reduced[label]=raw
        else:
            reduced[label]=reduced_density(raw,len(parent),
                        tuple(parent.index(site) for site in target))
    return reduced,{','.join(map(str,k)):v for k,v in memo.items()}
