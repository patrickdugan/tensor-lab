"""QC-10: exact causal-cone density oracle for shallow local noisy circuits.

For an initially product state, a local observable depends only on the qubits
in its backward causal cone.  On each reverse gate layer, support expands
across gates touching it.  Trace-preserving single-site channels never expand
the support of the Heisenberg observable (their adjoints are unital).

We simulate the restricted subcircuit on those cone qubits, preserving all
its local channels.  Gates crossing the boundary can be discarded because
none intersects the support at its time in the backward computation.  The
returned reduced state is exact in ideal arithmetic; NumPy floating-point
rounding is outside the guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from collections.abc import Sequence
import numpy as np
from channel_mpo import apply_operator, apply_channel


@dataclass(frozen=True)
class Cone:
    targets: tuple[int, ...]
    sites: tuple[int, ...]
    active_before: tuple[tuple[int, ...], ...]
    layers: int


@dataclass(frozen=True)
class ConeDensity:
    reduced_density: np.ndarray
    cone: Cone
    seconds: float
    relevant_gates: int


def backward_cone(n: int, layers: Sequence[Sequence[tuple[int, np.ndarray]]],
                  targets: Sequence[int]) -> Cone:
    """Find the precise causal cone, validating the nearest-neighbor layout."""
    if not isinstance(n, int) or n < 2:
        raise ValueError('n must be >= 2')
    selected=tuple(int(i) for i in targets)
    if not selected or tuple(sorted(set(selected)))!=selected or any(i<0 or i>=n for i in selected):
        raise ValueError('targets must be sorted, distinct and within the circuit')
    active=set(selected)
    slices=[None]*(len(layers)+1)
    slices[-1]=selected
    for t in range(len(layers)-1,-1,-1):
        occupied=set()
        for j,u in layers[t]:
            if not isinstance(j,int) or j<0 or j>=n-1 or np.shape(u)!=(4,4):
                raise ValueError('invalid nearest-neighbor gate')
            if j in occupied or j+1 in occupied:
                raise ValueError('same-layer gates must be disjoint')
            occupied.update((j,j+1))
        for j,u in layers[t]:
            if j in active or j+1 in active:
                active.update((j,j+1))
        slices[t]=tuple(sorted(active))
    return Cone(selected,tuple(sorted(active)),tuple(slices),len(layers))


def reduced_density(rho: np.ndarray,n:int,targets:Sequence[int])->np.ndarray:
    """Partial trace over all but selected sites in computational ordering."""
    targets=tuple(targets)
    if len(set(targets))!=len(targets) or tuple(sorted(targets))!=targets or any(x<0 or x>=n for x in targets):
        raise ValueError('targets must be sorted and distinct')
    if rho.shape!=(2**n,2**n):
        raise ValueError('density dimension mismatch')
    tensor=rho.reshape((2,)*(2*n))
    current=n
    for j in range(n-1,-1,-1):
        if j not in targets:
            tensor=np.trace(tensor,axis1=j,axis2=j+current)
            current-=1
    return tensor.reshape(2**len(targets),2**len(targets))


def exact_cone_density(n:int,layers:Sequence[Sequence[tuple[int,np.ndarray]]],
                       operators:tuple[np.ndarray,...],targets:Sequence[int],
                       *,max_sites:int=9)->ConeDensity:
    """Return exact small-cone local density without allocating full 4**n."""
    c=backward_cone(n,layers,targets)
    if max_sites<1 or len(c.sites)>max_sites:
        raise ValueError(f'causal cone has {len(c.sites)} qubits; max_sites={max_sites}')
    if not operators or any(np.shape(k)!=(2,2) for k in operators):
        raise ValueError('qubit Kraus operators required')
    complete=sum((k.conj().T@k for k in operators),np.zeros((2,2),complex))
    if not np.allclose(complete,np.eye(2),atol=1e-11):
        raise ValueError('noise must be CPTP')
    start=perf_counter()
    sites=c.sites
    lookup={global_idx:local_idx for local_idx,global_idx in enumerate(sites)}
    m=len(sites)
    rho=np.zeros((2**m,2**m),dtype=complex)
    rho[0,0]=1
    gates=0
    for layer in layers:
        for j,u in layer:
            if j in lookup and j+1 in lookup:
                rho=apply_operator(rho,u,(lookup[j],lookup[j+1]),m)
                gates+=1
        for i in range(m):
            rho=apply_channel(rho,operators,i,m)
    reduced=reduced_density(rho,m,tuple(lookup[i] for i in c.targets))
    return ConeDensity((reduced+reduced.conj().T)/2,c,perf_counter()-start,gates)


def trace_distance(a:np.ndarray,b:np.ndarray)->float:
    if a.shape!=b.shape or a.ndim!=2 or a.shape[0]!=a.shape[1]:
        raise ValueError('matching square density matrices required')
    difference=(a-b+(a-b).conj().T)/2
    return float(.5*np.sum(np.abs(np.linalg.eigvalsh(difference))))


def local_comparison(oracle:ConeDensity,marginal:np.ndarray)->dict[str,float]:
    """D bounds all single-site POVM event errors; 2D bounds ||O||<=1."""
    ref=oracle.reduced_density
    if ref.shape!=marginal.shape:raise ValueError('marginal has wrong dimension')
    td=trace_distance(ref,marginal)
    return {'trace_distance':td,'worst_povm_event_error':td,
            'worst_observable_abs_error_bound':2*td,
            'z_abs_error':abs(float(np.trace(ref@np.array([[1,0],[0,-1]],complex)).real-
                            np.trace(marginal@np.array([[1,0],[0,-1]],complex)).real))
                           if ref.shape==(2,2) else float('nan')}


def tensor_marginal(chain,targets:Sequence[int])->np.ndarray:
    """Get any small k-site reduced density from a purified chain in O(n)."""
    targets=tuple(targets)
    if not targets or targets!=tuple(sorted(set(targets))) or any(i<0 or i>=chain.n for i in targets):
        raise ValueError('targets must be sorted, distinct and in range')
    if len(targets)>5:
        raise ValueError('observable region exceeds small-marginal limit')
    selected=set(targets)
    transfer=np.ones((1,1,1,1),complex)
    for j,a in enumerate(chain.cores):
        if j in selected:
            output=np.einsum('lmxy,liar,mjas->rsxiyj',transfer,a,a.conj(),optimize=True)
            transfer=output.reshape(a.shape[-1],a.shape[-1],transfer.shape[2]*2,transfer.shape[3]*2)
        else:
            transfer=np.einsum('lmxy,liar,mias->rsxy',transfer,a,a.conj(),optimize=True)
    result=transfer[0,0]/chain.trace()
    return (result+result.conj().T)/2


def independent_cones_density(n:int,layers:Sequence[Sequence[tuple[int,np.ndarray]]],
                              operators:tuple[np.ndarray,...],
                              first:Sequence[int],second:Sequence[int],
                              *,max_sites:int=8)->np.ndarray:
    """Exact joint density of separate regions with disjoint input cones.

    Requires a product initial state, local CPTP maps, and disjoint backward
    causal cones. The complete joint marginal factorizes even if the regions
    are distant in the global circuit. Returned ordering is first, second.
    """
    a=backward_cone(n,layers,first)
    b=backward_cone(n,layers,second)
    if set(a.sites).intersection(b.sites):
        raise ValueError('backward cones overlap: joint factorization unavailable')
    aa=exact_cone_density(n,layers,operators,first,max_sites=max_sites)
    bb=exact_cone_density(n,layers,operators,second,max_sites=max_sites)
    return np.kron(aa.reduced_density,bb.reduced_density)
