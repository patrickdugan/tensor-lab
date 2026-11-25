"""QC-12: Backward Heisenberg operator-MPO with local-observable error ledger.

Supports product |0>^n input, disjoint nearest-neighbor unitary layers, and
independent single-qubit CPTP channels after each layer. Evolves one nominated
Pauli string backwards over its exact causal cone and approximates the operator
by a bond-limited MPO. Each truncated two-site SVD is evaluated in a mixed
canonical gauge, and its independently recomputed absolute Frobenius residual
is logged.

In exact arithmetic, CPTP adjoints are unital and operator-norm contractions.
Consequently |<O> - <O_approx>| <= sum(local Frobenius residuals). This is an
operational, observable-specific a posteriori bound for a Hermitian local O.
NumPy residual measurements are not certified interval bounds on FP rounding.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Sequence
import numpy as np
from channel_mpo import I2, X, Y, Z
from qc10_cone import backward_cone

PAULIS={'I':I2,'X':X,'Y':Y,'Z':Z}


@dataclass(frozen=True)
class LocalResidual:
    layer:int
    cut:int
    original_rank:int
    retained_rank:int
    frobenius_residual:float
    raw_entries:int


@dataclass(frozen=True)
class ObservableResult:
    expectation:float
    error_bound:float
    local_residuals:tuple[LocalResidual,...]
    target_sites:tuple[int,...]
    label:str
    cone_width:int
    max_bond:int
    max_mpo_entries:int
    max_working_entries:int
    seconds:float
    per_layer_widths:tuple[int,...]


def _independent_residual(m:np.ndarray,u:np.ndarray,s:np.ndarray,vh:np.ndarray,
                          keep:int,block:int=256)->float:
    """Recompute local matrix error in row blocks, including SVD roundoff."""
    ss=0.0
    for start in range(0,m.shape[0],block):
        stop=min(start+block,m.shape[0])
        r=m[start:stop]-(u[start:stop,:keep]*s[:keep])@vh[:keep]
        ss+=float(np.vdot(r,r).real)
    return float(np.sqrt(ss))


class OperatorMPO:
    """Matrix product operator with physical axes (bra-output, ket-input)."""
    def __init__(self,targets:tuple[int,...],label:str,bond_cap:int,
                 *,local_tol:float=0.):
        if not targets or targets!=tuple(sorted(set(targets))):
            raise ValueError('targets must be sorted and nonempty')
        if len(label)!=len(targets) or any(x not in PAULIS for x in label):
            raise ValueError('one I/X/Y/Z symbol required per target')
        if bond_cap<1 or not np.isfinite(local_tol) or local_tol<0:
            raise ValueError('invalid rank cap/tolerance')
        self.first=targets[0]
        self.last=targets[-1]
        self.cores=[]
        lookup=dict(zip(targets,label))
        for i in range(self.first,self.last+1):
            op=PAULIS[lookup.get(i,'I')]
            self.cores.append(op.reshape(1,2,2,1).copy())
        self.cap=bond_cap
        self.tol=local_tol
        self.residuals=[]
        self.max_working_entries=0
        self.max_entries=sum(a.size for a in self.cores)
        self.max_bond=1

    @property
    def width(self):return len(self.cores)

    def _note(self):
        self.max_entries=max(self.max_entries,sum(a.size for a in self.cores))
        self.max_bond=max(self.max_bond,max(max(a.shape[0],a.shape[-1]) for a in self.cores))

    def grow(self,first:int,last:int):
        if first>self.first or last<self.last:
            raise ValueError('backward cone must grow monotonically')
        if first<0 or last<first:raise ValueError('invalid support')
        for _ in range(self.first-first):
            self.cores.insert(0,I2.reshape(1,2,2,1).copy())
        for _ in range(last-self.last):
            self.cores.append(I2.reshape(1,2,2,1).copy())
        self.first,self.last=first,last
        self._note()

    def channel_adjoint(self,site:int,operators:tuple[np.ndarray,...]):
        idx=site-self.first
        a=self.cores[idx]
        # L a b R, sum K^dag[a,x] O[x,y] K[y,b]
        op=np.zeros_like(a)
        for k in operators:
            op+=np.einsum('ax,lxyr,yb->labr',k.conj().T,a,k,optimize=True)
        self.cores[idx]=op
        self._note()

    def _left_qr(self,end:int):
        for j in range(end):
            a=self.cores[j];dl,_,_,dr=a.shape
            q,r=np.linalg.qr(a.reshape(dl*4,dr),mode='reduced')
            self.cores[j]=q.reshape(dl,2,2,q.shape[1])
            self.cores[j+1]=np.tensordot(r,self.cores[j+1],axes=(1,0))

    def _right_qr(self,start:int):
        for j in range(self.width-1,start-1,-1):
            a=self.cores[j];dl,_,_,dr=a.shape
            q,r=np.linalg.qr(a.reshape(dl,4*dr).T,mode='reduced')
            self.cores[j]=q.T.reshape(q.shape[1],2,2,dr)
            self.cores[j-1]=np.tensordot(self.cores[j-1],r.T,axes=(3,0))

    def gate_adjoint(self,site:int,unitary:np.ndarray,layer:int)->float:
        j=site-self.first
        if j<0 or j+1>=self.width:
            raise ValueError('gate endpoints outside active support')
        self._left_qr(j)
        self._right_qr(j+2)
        a,b=self.cores[j:j+2]
        dl,dr=a.shape[0],b.shape[-1]
        combined=np.tensordot(a,b,axes=(3,0))
        theta=combined.transpose(0,1,3,2,4,5).reshape(dl,4,4,dr)
        updated=np.einsum('xa,labr,by->lxyr',unitary.conj().T,theta,unitary,optimize=True)
        matrix=updated.reshape(dl,2,2,2,2,dr).transpose(0,1,3,2,4,5).reshape(dl*4,4*dr)
        self.max_working_entries=max(self.max_working_entries,matrix.size,combined.size)
        u,s,vh=np.linalg.svd(matrix,full_matrices=False)
        # Use the hard rank, plus optional relative-tail budget. Numerical zeros
        # remain part of the reported cut rank only if essential to residual.
        keep=min(len(s),self.cap)
        if self.tol>0:
            tail=np.r_[np.cumsum((s[::-1]**2))[::-1],0.]
            possible=np.flatnonzero(tail<=self.tol**2*float(np.dot(s,s)))
            if len(possible):keep=max(1,min(keep,int(possible[0])))
        residual=_independent_residual(matrix,u,s,vh,keep)
        self.residuals.append(LocalResidual(layer,site,len(s),keep,residual,int(matrix.size)))
        self.cores[j]=u[:,:keep].reshape(dl,2,2,keep)
        self.cores[j+1]=(s[:keep,None]*vh[:keep]).reshape(keep,2,2,dr)
        self._note()
        return residual

    def input_zero_expectation(self)->float:
        state=np.ones((1,),dtype=complex)
        for core in self.cores:
            state=state@core[:,0,0,:]
        value=complex(state[0])
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise FloatingPointError('nonfinite contracted observable')
        if abs(value.imag)>1e-7:
            raise ArithmeticError('Hermitian observable expectation is non-real')
        return float(value.real)


def operator_cone_expectation(n:int,layers:Sequence,operators:tuple[np.ndarray,...],
                              targets:Sequence[int],label:str,*,bond_cap:int=16,
                              local_tol:float=0.,max_cone:int=64,
                              max_entries:int=1_000_000)->ObservableResult:
    """Return expectation and an absolute local-observable error ledger.

    The bound concerns ideal-arithmetic truncations of the forward circuit's
    exactly specified observable, for a product |0> initial state. It is not a
    rigorous floating-point interval certificate, and never substitutes an
    exact oracle to make rank decisions.
    """
    start=perf_counter()
    targets=tuple(targets)
    if not targets or targets!=tuple(range(targets[0],targets[-1]+1)):
        raise ValueError('QC-12 requires a contiguous target block')
    if not isinstance(max_cone,int) or max_cone<1 or max_entries<1:
        raise ValueError('invalid resource limits')
    cone=backward_cone(n,layers,targets)
    if len(cone.sites)>max_cone:raise MemoryError('backward cone exceeds width cap')
    if not operators or any(np.shape(k)!=(2,2) or not np.isfinite(k).all() for k in operators):
        raise ValueError('finite 2x2 Kraus operators required')
    if not np.allclose(sum((k.conj().T@k for k in operators),np.zeros((2,2),complex)),
                       I2,rtol=0,atol=1e-11):
        raise ValueError('Kraus map is not trace preserving')
    for layer in layers:
        for _,u in layer:
            if not np.isfinite(u).all() or not np.allclose(u.conj().T@u,np.eye(4),atol=1e-11,rtol=0):
                raise ValueError('gate must be unitary')
    chain=OperatorMPO(targets,label,bond_cap,local_tol=local_tol)
    if chain.max_entries>max_entries:raise MemoryError('initial operator MPO exceeds entry limit')
    widths=[chain.width]
    for t in range(len(layers)-1,-1,-1):
        active_after=cone.active_before[t+1]
        # Apply channel dual on the current causal support, before each gate's dual.
        for site in range(chain.first,chain.last+1):
            chain.channel_adjoint(site,operators)
        active_before=cone.active_before[t]
        if tuple(range(active_before[0],active_before[-1]+1))!=active_before:
            raise ValueError('QC-12 requires contiguous causal support')
        chain.grow(active_before[0],active_before[-1])
        if chain.max_entries>max_entries:
            raise MemoryError('retained operator MPO exceeds entry limit')
        if chain.width!=len(active_before):raise AssertionError('cone width mismatch')
        for site,unitary in layers[t]:
            if site in active_after or site+1 in active_after:
                chain.gate_adjoint(site,unitary,t)
                if chain.max_entries>max_entries:
                    raise MemoryError('retained operator MPO exceeds entry limit')
        widths.append(chain.width)
    return ObservableResult(chain.input_zero_expectation(),
              float(sum(r.frobenius_residual for r in chain.residuals)),
              tuple(chain.residuals),targets,label,len(cone.sites),chain.max_bond,
              chain.max_entries,chain.max_working_entries,perf_counter()-start,
              tuple(widths))


def pauli_density(n:int,layers:Sequence,operators:tuple[np.ndarray,...],
                  targets:Sequence[int],*,bond_cap:int=16,local_tol:float=0.,
                  max_cone:int=64,max_entries:int=1_000_000):
    """Reconstruct small local density, with a trace-distance error bound.

    Pauli basis orthogonality yields D(rho,true_rho) <=
    (1/2)*sqrt(sum_{P != I} epsilon_P**2), for epsilon_P bounds on the
    errors of each Pauli expectation. Candidate matrix can be non-PSD.
    """
    from itertools import product
    ts=tuple(targets)
    if not 1<=len(ts)<=2:raise ValueError('1 or 2 target sites required')
    if ts!=tuple(range(ts[0],ts[-1]+1)):raise ValueError('contiguous targets required')
    size=2**len(ts)
    result=np.zeros((size,size),complex)
    eps2=0.
    pieces={}
    for letters in product('IXYZ',repeat=len(ts)):
        label=''.join(letters)
        if all(x=='I' for x in label):
            expectation=1.;delta=0.
        else:
            obs=operator_cone_expectation(n,layers,operators,ts,label,
                 bond_cap=bond_cap,local_tol=local_tol,max_cone=max_cone,
                 max_entries=max_entries)
            expectation=obs.expectation;delta=obs.error_bound
        p=PAULIS[letters[0]]
        for c in letters[1:]:p=np.kron(p,PAULIS[c])
        result+=expectation*p/size
        pieces[label]={'expectation':expectation,'bound':delta}
        eps2+=delta*delta
    return (result+result.conj().T)/2,float(.5*np.sqrt(eps2)),pieces


@dataclass(frozen=True)
class ThresholdDecision:
    accepted:bool
    chosen_cap:int | None
    estimate:float | np.ndarray | None
    bound:float
    tried_caps:tuple[int,...]
    target:str


def threshold_observable(n:int,layers:Sequence,operators:tuple[np.ndarray,...],
                         targets:Sequence[int],label:str,*,caps:Sequence[int]=(4,8,16),
                         max_error:float=.05,max_cone:int=64,
                         max_entries:int=1_000_000)->ThresholdDecision:
    """Select smallest predeclared cap whose residual ledger meets the target.

    The procedure sees neither a dense oracle nor actual physical output error.
    If every cap fails, it returns accepted=False and no expectation estimate.
    The bound is ideal-arithmetic; NumPy rounding remains uncertified.
    """
    caps=tuple(caps)
    if not caps or any(c<1 for c in caps) or len(set(caps))!=len(caps) or tuple(sorted(caps))!=caps:
        raise ValueError('rank caps must be strictly increasing positive integers')
    if not np.isfinite(max_error) or max_error<0:
        raise ValueError('invalid observable error target')
    last=float('inf')
    for cap in caps:
        result=operator_cone_expectation(n,layers,operators,targets,label,bond_cap=cap,
                                         max_cone=max_cone,max_entries=max_entries)
        last=result.error_bound
        if last<=max_error:
            return ThresholdDecision(True,cap,result.expectation,last,caps[:caps.index(cap)+1],label)
    return ThresholdDecision(False,None,None,last,caps,label)


def threshold_density(n:int,layers:Sequence,operators:tuple[np.ndarray,...],
                      targets:Sequence[int],*,caps:Sequence[int]=(4,8,16),
                      max_error:float=.05,max_cone:int=64,
                      max_entries:int=1_000_000)->ThresholdDecision:
    """Gate a 1/2-site tomography result on its Pauli-derived trace bound."""
    caps=tuple(caps)
    if not caps or any(c<1 for c in caps) or len(set(caps))!=len(caps) or tuple(sorted(caps))!=caps:
        raise ValueError('rank caps must be strictly increasing positive integers')
    if not np.isfinite(max_error) or max_error<0:
        raise ValueError('invalid trace-distance error target')
    last=float('inf')
    for cap in caps:
        rho,bound,_=pauli_density(n,layers,operators,targets,bond_cap=cap,
                                  max_cone=max_cone,max_entries=max_entries)
        last=bound
        if bound<=max_error:
            return ThresholdDecision(True,cap,rho,bound,caps[:caps.index(cap)+1],
                                     'local_density')
    return ThresholdDecision(False,None,None,last,caps,'local_density')
