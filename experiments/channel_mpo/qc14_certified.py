"""QC-14: end-to-end *a posteriori* audit of a backward Heisenberg MPO.

All locally computed residuals below are evaluated with arbitrary-precision
integer arithmetic on the exact dyadic values of stored complex128 arrays.
This captures rounded QR, contractions, channels and SVD reconstruction without
assuming BLAS or LAPACK backward stability.  A conservative product of external
core Frobenius norms bounds a local change in global operator norm.

The physical target is explicitly defined: every stored gate U is replaced by
its exact polar unitary U (U†U)^(-1/2); the stacked stored Kraus isometry W is
replaced by W (W†W)^(-1/2).  The correction from literal dyadic coefficients
is included in the error ledger.  The target is defined mathematically, not
constructed by an approximate polar decomposition.

This is a mathematically rigorous enclosure under the assumption that Python
integer arithmetic behaves exactly and binary64 values are read faithfully.
The auditor is a separate implementation of the QC-12 updates and has a
fail-closed resource limit. It is *not* a machine-checked proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from math import isqrt, isfinite
from time import perf_counter
from typing import Sequence
import numpy as np

from qc13_dyadic import (_exponent, _scale, _parts, _upper_float,
    residual_certificate, kraus_completeness_certificate,
    mpo_zero_readout_certificate, DyadicReadout)
from qc12_heisenberg import OperatorMPO, LocalResidual, _independent_residual
from qc10_cone import backward_cone


def _scaled(a: np.ndarray, power: int):
    """Flatten an array of binary64 complex numbers into exact integer pairs."""
    return [(_scale(re, power), _scale(im, power)) for re, im
            in (_parts(z) for z in np.asarray(a).flat)]


def _mul(a, b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def _add(a, b):
    return a[0]+b[0], a[1]+b[1]


def _up_sqrt_integer(square: int, power: int) -> float:
    """Rigorous upper bound on sqrt(square)/2**power."""
    if square < 0 or power < 0:
        raise ValueError('invalid exact square/scale')
    root = isqrt(square)
    if root * root < square:
        root += 1
    return _upper_float(F(root, 1 << power))


def _tensor_norm_bound(a: np.ndarray) -> float:
    a = np.asarray(a)
    if a.dtype not in (np.dtype('float64'), np.dtype('complex128')):
        raise TypeError('QC-14 requires exact binary64 components')
    if not np.isfinite(a).all():
        raise ValueError('nonfinite input cannot be certified')
    power = max(_exponent(x) for z in a.flat for x in _parts(z))
    q = _scaled(a, power)
    return _up_sqrt_integer(sum(x*x+y*y for x,y in q), power)


def _pair_integer(a: np.ndarray, b: np.ndarray, power: int):
    """Exact dyadic contraction A[l,a,b,m] B[m,c,d,r]."""
    dl,_,_,dm = a.shape
    if b.shape[0] != dm:
        raise ValueError('two-site bond mismatch')
    dr = b.shape[-1]
    left = _scaled(a.reshape(dl*4, dm), power)
    right = _scaled(b.reshape(dm, 4*dr), power)
    out = []
    for row in range(dl*4):
        for col in range(4*dr):
            re=im=0
            for mid in range(dm):
                x,y = _mul(left[row*dm+mid], right[mid*(4*dr)+col])
                re+=x;im+=y
            out.append((re,im))
    return out,dl,dr


def _gate_integer(two_site, dl: int, dr: int, unitary: np.ndarray, power: int):
    """Exact 2-qubit adjoint U† O U on an integer-scaled two-site tensor."""
    # two_site is indexed by (left-bond, left out, left in, right out,
    # right in, right-bond); existing denominator is 2**(2*power).
    u=_scaled(unitary,power)
    out=[None]*(dl*4*4*dr)
    stride=4*dr
    for l in range(dl):
        for r in range(dr):
            old=[]
            for i in range(4):
                row=[]
                ao,bo=divmod(i,2)
                for j in range(4):
                    ai,bi=divmod(j,2)
                    idx=((l*4+ao*2+ai)*stride)+(bo*2+bi)*dr+r
                    row.append(two_site[idx])
                old.append(row)
            for i in range(4):
                ao,bo=divmod(i,2)
                for j in range(4):
                    ai,bi=divmod(j,2)
                    re=im=0
                    for x in range(4):
                        # U†[i,x] = conj(U[x,i])
                        uxi=u[x*4+i]
                        left=(uxi[0],-uxi[1])
                        for y in range(4):
                            xy=_mul(_mul(left,old[x][y]),u[y*4+j])
                            re+=xy[0];im+=xy[1]
                    idx=((l*4+ao*2+ai)*stride)+(bo*2+bi)*dr+r
                    out[idx]=(re,im)
    return out


def _valid_pair(a,b,c,d):
    for x in (a,b,c,d):
        if x.ndim!=4 or x.shape[1:3]!=(2,2):
            raise ValueError('four cores required with 2x2 physical axes')
        if x.dtype not in (np.dtype('float64'),np.dtype('complex128')) or not np.isfinite(x).all():
            raise ValueError('only finite binary64 cores are accepted')
    if a.shape[-1]!=b.shape[0] or c.shape[-1]!=d.shape[0]:
        raise ValueError('broken two-site link')
    if a.shape[0]!=c.shape[0] or b.shape[-1]!=d.shape[-1]:
        raise ValueError('outer dimensions changed')


def two_site_residual(a,b,c,d,unitary=None):
    """Upper on exact dyadic two-site update residual, including all rounding.

    With `unitary=None`, returns ||contract(a,b)-contract(c,d)||F.
    Otherwise returns ||U† contract(a,b) U - contract(c,d)||F.
    The supplied stored floating values are interpreted exactly.
    """
    a,b,c,d=(np.asarray(x) for x in (a,b,c,d))
    _valid_pair(a,b,c,d)
    arrs=[a,b,c,d]
    if unitary is not None:
        unitary=np.asarray(unitary)
        if unitary.shape!=(4,4) or unitary.dtype not in (np.dtype('float64'),np.dtype('complex128')) or not np.isfinite(unitary).all():
            raise ValueError('finite binary64 4x4 gate required')
        arrs.append(unitary)
    p=max(_exponent(x) for ar in arrs for z in ar.flat for x in _parts(z))
    left,dl,dr=_pair_integer(a,b,p)
    right,dl2,dr2=_pair_integer(c,d,p)
    assert (dl,dr)==(dl2,dr2)
    if unitary is not None:
        left=_gate_integer(left,dl,dr,unitary,p)
        right=[(re<<(2*p),im<<(2*p)) for re,im in right]
        scale=4*p
    else:
        scale=2*p
    sq=sum((x-u)**2+(y-v)**2 for (x,y),(u,v) in zip(left,right))
    return _up_sqrt_integer(sq,scale)


def channel_local_residual(before, after, operators):
    """Upper for exact sum K† A K versus the stored rounded channel core."""
    a=np.asarray(before);b=np.asarray(after)
    ks=tuple(np.asarray(k) for k in operators)
    if a.shape != b.shape or a.ndim!=4 or a.shape[1:3]!=(2,2):
        raise ValueError('channel core shapes must match')
    arrs=[a,b,*ks]
    if not ks or any(k.shape!=(2,2) for k in ks):
        raise ValueError('finite 2x2 Kraus operators required')
    if any(x.dtype not in (np.dtype('complex128'),np.dtype('float64')) or not np.isfinite(x).all() for x in arrs):
        raise ValueError('finite binary64 arrays required')
    p=max(_exponent(x) for ar in arrs for z in ar.flat for x in _parts(z))
    before_i=_scaled(a,p)
    after_i=_scaled(b,p)
    k_values=[_scaled(k,p) for k in ks]
    dl,_,_,dr=a.shape
    sq=0
    for l in range(dl):
        for ao in range(2):
            for ai in range(2):
                for r in range(dr):
                    re=im=0
                    for k in k_values:
                        for x in range(2):
                            # stored K†[ao,x] = conj(K[x,ao])
                            z=k[x*2+ao]
                            kz=(z[0],-z[1])
                            for y in range(2):
                                old=before_i[((l*2+x)*2+y)*dr+r]
                                v=_mul(_mul(kz,old),k[y*2+ai])
                                re+=v[0];im+=v[1]
                    ax=after_i[((l*2+ao)*2+ai)*dr+r]
                    re-=ax[0]<<(2*p)
                    im-=ax[1]<<(2*p)
                    sq+=re*re+im*im
    return _up_sqrt_integer(sq,3*p)


def _polar_dual_defect(defect:float)->float:
    """Upper for ||U†OU - V†OV||/||O|| using polar unitary/isometry V.

    W†W−I has operator norm at most its certified Frobenius defect d < 1.
    ||W−polar(W)|| <= d/(1+sqrt(1−d)) <= d and ||W|| <= 1+d.
    Thus map difference <= d(2+d). Fractions make rounding outward.
    """
    if not isfinite(defect) or not 0<=defect<F(1,2):
        raise ValueError('polar correction requires a small certified defect')
    d=F(defect)
    return _upper_float(d*(2+d))


@dataclass(frozen=True)
class AuditEvent:
    kind:str
    layer:int
    site:int
    local_residual_upper:float
    external_norm_upper:float
    global_residual_upper:float
    polar_correction_upper:float


@dataclass(frozen=True)
class CertifiedExpectation:
    reported_expectation:float
    exact_readout:DyadicReadout
    physical_error_upper:float
    readout_rounding_upper:float
    reported_error_upper:float
    truncation_and_rounding_upper:float
    polar_correction_sum_upper:float
    event_counts:dict[str,int]
    events:tuple[AuditEvent,...]
    cone_width:int
    max_mpo_entries:int
    seconds:float
    target_definition:str = 'polar-corrected stored gate/Kraus coefficients'


class VerifiedOperatorMPO(OperatorMPO):
    """QC-12 execution with exact-dyadic, per-operation residual hooks."""
    def __init__(self, *args, operators, max_events=4000, **kw):
        super().__init__(*args,**kw)
        self.events=[]
        self.e=0.
        self.polar_total=0.
        self.max_events=max_events
        self._operators=operators
        self._channel_alpha=_polar_dual_defect(kraus_completeness_certificate(operators))
        self._gate_alphas={}
        self._layer=-1
        self.readout_cert=None
        self._error_without_polar=0.
        self._frob_cache={}
        self._canon={}

    def _replace(self, index:int):
        self._frob_cache.pop(index,None)
        self._canon.pop(index,None)

    def grow(self,first,last):
        old_first=self.first
        super().grow(first,last)
        shift=old_first-first
        if shift:
            self._frob_cache={i+shift:bound for i,bound in self._frob_cache.items()}
            self._canon={i+shift:value for i,value in self._canon.items()}

    def _orthogonal(self,index:int,orientation:str):
        a=self.cores[index]
        dl,_,_,dr=a.shape
        if orientation=='L':
            q=a.reshape(dl*4,dr)
            defect=residual_certificate(np.eye(dr,dtype=np.complex128),q.conj().T,q).upper
        elif orientation=='R':
            q=a.reshape(dl,4*dr)
            defect=residual_certificate(np.eye(dl,dtype=np.complex128),q,q.conj().T).upper
        else:
            raise ValueError('unknown canonical orientation')
        # ||Q||_2 <= sqrt(1+defect) <= 1+defect, with outward rounding.
        self._canon[index]=(orientation,_upper_float(F(1)+F(defect)))

    def _other_bound(self, excluded, *, isometric_gate=False):
        q=F(1)
        low=min(excluded)
        high=max(excluded)
        for i,core in enumerate(self.cores):
            if i in excluded:continue
            desired='L' if i<low else 'R' if i>high else None
            if isometric_gate and desired is not None:
                if i not in self._canon or self._canon[i][0]!=desired:
                    raise ArithmeticError('missing certified mixed-canonical gate context')
                bound=self._canon[i][1]
            else:
                if i not in self._frob_cache:
                    self._frob_cache[i]=_tensor_norm_bound(core)
                bound=self._frob_cache[i]
            q*=F(bound)
        return _upper_float(q)

    def _record(self,kind,layer,site,local,excluded,alpha=0.):
        if len(self.events)>=self.max_events:
            raise MemoryError('exact-dyadic event audit exceeded configured budget')
        external=self._other_bound(excluded,isometric_gate=(kind=="gate_svd"))
        global_residual=_upper_float(F(local)*F(external))
        e_before=F(self.e)
        # Before applying an *ideal physical* map, the earlier error is at
        # most e. The polar correction changes that map by alpha*(1+e).
        self.e=_upper_float(e_before + F(alpha)*(1+e_before) + F(global_residual))
        self._error_without_polar=_upper_float(F(self._error_without_polar)+F(global_residual))
        self.polar_total=_upper_float(F(self.polar_total)+F(alpha)*(1+e_before))
        self.events.append(AuditEvent(kind,layer,site,local,external,global_residual,alpha))

    def channel_adjoint(self,site:int,operators):
        idx=site-self.first
        before=self.cores[idx].copy()
        super().channel_adjoint(site,operators)
        self._replace(idx)
        local=channel_local_residual(before,self.cores[idx],operators)
        self._record('channel',self._layer,site,local,{idx},self._channel_alpha)

    def _left_qr(self,end:int):
        for j in range(end):
            a,b=self.cores[j:j+2]
            dl,_,_,dr=a.shape
            q,r=np.linalg.qr(a.reshape(dl*4,dr),mode='reduced')
            self.cores[j]=q.reshape(dl,2,2,q.shape[1])
            self.cores[j+1]=np.tensordot(r,b,axes=(1,0))
            self._replace(j);self._replace(j+1)
            self._orthogonal(j,'L')
            residual=two_site_residual(a,b,self.cores[j],self.cores[j+1])
            self._record('left_qr',self._layer,j+self.first,residual,{j,j+1})

    def _right_qr(self,start:int):
        for j in range(self.width-1,start-1,-1):
            a=self.cores[j];b=self.cores[j-1]
            dl,_,_,dr=a.shape
            q,r=np.linalg.qr(a.reshape(dl,4*dr).T,mode='reduced')
            self.cores[j]=q.T.reshape(q.shape[1],2,2,dr)
            self.cores[j-1]=np.tensordot(b,r.T,axes=(3,0))
            self._replace(j);self._replace(j-1)
            self._orthogonal(j,'R')
            residual=two_site_residual(b,a,self.cores[j-1],self.cores[j])
            self._record('right_qr',self._layer,j+self.first,residual,{j-1,j})

    def gate_adjoint(self,site:int,unitary:np.ndarray,layer:int)->float:
        j=site-self.first
        if j<0 or j+1>=self.width:
            raise ValueError('gate endpoints outside active support')
        self._layer=layer
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
        keep=min(len(s),self.cap)
        if self.tol>0:
            tail=np.r_[np.cumsum((s[::-1]**2))[::-1],0.]
            possible=np.flatnonzero(tail<=self.tol**2*float(np.dot(s,s)))
            if len(possible):keep=max(1,min(keep,int(possible[0])))
        residual=_independent_residual(matrix,u,s,vh,keep)
        self.residuals.append(LocalResidual(layer,site,len(s),keep,residual,int(matrix.size)))
        self.cores[j]=u[:,:keep].reshape(dl,2,2,keep)
        self.cores[j+1]=(s[:keep,None]*vh[:keep]).reshape(keep,2,2,dr)
        self._replace(j);self._replace(j+1)
        self._orthogonal(j,'L')
        self._note()
        key=unitary.tobytes()
        if key not in self._gate_alphas:
            d=residual_certificate(np.eye(4,dtype=np.complex128),unitary.conj().T,unitary).upper
            self._gate_alphas[key]=_polar_dual_defect(d)
        local=two_site_residual(a,b,self.cores[j],self.cores[j+1],unitary=unitary)
        self._record('gate_svd',layer,site,local,{j,j+1},self._gate_alphas[key])
        return residual

    def input_zero_expectation(self):
        self.readout_cert=mpo_zero_readout_certificate(self.cores)
        return super().input_zero_expectation()


def certify_observable(n:int,layers:Sequence,operators:tuple[np.ndarray,...],
                       targets:Sequence[int],label:str,*,bond_cap:int=16,
                       max_cone:int=64,max_entries:int=1_000_000,
                       max_events:int=4000)->CertifiedExpectation:
    """Return validated bound for a *polar-corrected* generic physical circuit.

    The input coefficients define the polar-corrected physical circuit. The
    bound includes every QR, gate, channel and readout rounding discrepancy,
    plus every MPO truncation. Integer residual checks are executed for each
    primitive; the bound is conservative and may exceed a useful tolerance.
    """
    start=perf_counter()
    targets=tuple(targets)
    if not targets or targets!=tuple(range(targets[0],targets[-1]+1)):
        raise ValueError('contiguous nonempty targets required')
    if bond_cap<1 or max_entries<1 or max_events<1:
        raise ValueError('invalid resource limits')
    cone=backward_cone(n,layers,targets)
    if len(cone.sites)>max_cone:raise MemoryError('causal cone exceeds cap')
    ks=tuple(np.asarray(k) for k in operators)
    if not ks or any(k.shape!=(2,2) for k in ks):
        raise ValueError('invalid Kraus map')
    # Validate once before beginning an expensive audit.
    # Mirror the QC-12 physical/shape checks, without actually running QC-12.
    if any(not np.isfinite(k).all() for k in ks):raise ValueError('nonfinite Kraus input')
    for layer in layers:
        for _,u in layer:
            if np.shape(u)!=(4,4) or not np.isfinite(u).all():
                raise ValueError('nonfinite or malformed gate')
    chain=VerifiedOperatorMPO(targets,label,bond_cap,operators=ks,max_events=max_events)
    if chain.max_entries>max_entries:raise MemoryError('initial MPO too large')
    for t in range(len(layers)-1,-1,-1):
        chain._layer=t
        active_after=cone.active_before[t+1]
        for site in range(chain.first,chain.last+1):
            chain.channel_adjoint(site,ks)
        active_before=cone.active_before[t]
        if tuple(range(active_before[0],active_before[-1]+1))!=active_before:
            raise ValueError('QC-14 requires contiguous causal cone')
        chain.grow(active_before[0],active_before[-1])
        if chain.max_entries>max_entries:raise MemoryError('retained MPO entry cap exceeded')
        for site,u in layers[t]:
            if site in active_after or site+1 in active_after:
                chain.gate_adjoint(site,np.asarray(u),t)
                if chain.max_entries>max_entries:raise MemoryError('retained MPO entry cap exceeded')
    expectation=chain.input_zero_expectation()
    readout=chain.readout_cert
    rounding=readout.maximum_real_error(expectation)
    total=_upper_float(F(chain.e)+F(rounding))
    counts={kind:sum(e.kind==kind for e in chain.events)
            for kind in ('channel','left_qr','right_qr','gate_svd')}
    return CertifiedExpectation(expectation,readout,chain.e,rounding,total,
           chain._error_without_polar,chain.polar_total,counts,tuple(chain.events),
           len(cone.sites),chain.max_entries,perf_counter()-start)
