"""Heterogeneous nearest-neighbor CPTP channels; normalized Pauli Heisenberg MPO.
The ledger bounds truncation in exact arithmetic, not floating-point roundoff.
Site 0 is the most significant tensor factor. T[a,b] = Tr(P_a E(P_b))/d.
"""
from dataclasses import dataclass
from itertools import product
import math
import numpy as np

P = np.array([[[1,0],[0,1]],[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)
PAULIS = {1:P, 2:np.array([np.kron(a,b) for a,b in product(P,P)])}

@dataclass
class Channel:
    site: int
    ks: tuple
    label: str = ''
    def __post_init__(self):
        self.ks = tuple(np.asarray(k,complex) for k in self.ks)
        if not self.ks or self.ks[0].shape not in ((2,2),(4,4)):
            raise ValueError('only one/two adjacent sites supported')
        d = self.ks[0].shape[0]
        if any(k.shape != (d,d) or not np.isfinite(k).all() for k in self.ks):
            raise ValueError('invalid Kraus shape/value')
        if not np.allclose(sum(k.conj().T @ k for k in self.ks),np.eye(d),atol=1e-12,rtol=0):
            raise ValueError('Kraus map is not trace preserving')
        self.width = 1 if d==2 else 2
        p = PAULIS[self.width]
        e = np.array([sum(k @ b @ k.conj().T for k in self.ks) for b in p])
        t = np.einsum('aij,bji->ab',p,e)/d
        if np.max(np.abs(t.imag))>1e-12: raise ArithmeticError('nonreal PTM')
        self.t = t.real

def rotation(pauli, angle):
    return np.cos(angle/2)*np.eye(len(pauli))-1j*np.sin(angle/2)*pauli

def stochastic(site, width, probabilities):
    if any(not np.isfinite(v) or v<0 for v in probabilities.values()) or sum(probabilities.values())>1:
        raise ValueError('invalid Pauli probabilities')
    if any(len(k)!=width or any(c not in 'IXYZ' for c in k) or k=='I'*width for k in probabilities):
        raise ValueError('invalid Pauli fault label')
    ks=[math.sqrt(1-sum(probabilities.values()))*np.eye(2**width)]
    for lab,pr in probabilities.items():
        idx=0
        for c in lab: idx=4*idx+'IXYZ'.index(c)
        ks.append(math.sqrt(pr)*PAULIS[width][idx])
    return Channel(site,tuple(ks),'Pauli')

def thermal(site, duration, t1, t2):
    if not all(np.isfinite(v) for v in (duration,t1,t2)) or duration<0 or min(t1,t2)<=0: raise ValueError('invalid thermal parameters')
    g=-np.expm1(-duration/t1)
    tphi=max(0.,1/t2-1/(2*t1))
    p=-np.expm1(-duration*tphi)/2
    a=(np.diag([1,math.sqrt(1-g)]),np.array([[0,math.sqrt(g)],[0,0]]))
    b=(math.sqrt(1-p)*P[0],math.sqrt(p)*P[3])
    return Channel(site,tuple(y@x for x,y in product(a,b)),'thermal')

def readout_vector(e0=0.,e1=0.):
    if not (0<=e0<=1 and 0<=e1<=1): raise ValueError('readout probabilities')
    return np.array([e1-e0,0.,0.,1-e0-e1])

def check(n,channels,target):
    if n<1 or not 0<=target<n or any(c.site<0 or c.site+c.width>n for c in channels): raise ValueError('site outside register')

def center(a,i):
    # Make the environments of the updated pair isometric.
    for j in range(i):
        l,d,r=a[j].shape
        q,s=np.linalg.qr(a[j].reshape(l*d,r),mode='reduced')
        a[j]=q.reshape(l,d,-1); a[j+1]=np.einsum('ab,bic->aic',s,a[j+1])
    for j in range(len(a)-1,i+1,-1):
        l,d,r=a[j].shape
        q,s=np.linalg.qr(a[j].reshape(l,d*r).T,mode='reduced')
        a[j]=q.T.reshape(-1,d,r); a[j-1]=np.einsum('aib,bc->aic',a[j-1],s.T)

def mpo_z(n,channels,target,cap=16,readout=(0.,0.)):
    channels=list(channels); check(n,channels,target)
    if cap<1: raise ValueError('cap must be positive')
    a=[np.array([1.,0.,0.,0.]).reshape(1,4,1) for _ in range(n)]
    a[target]=readout_vector(*readout).reshape(1,4,1)
    ledger=[]; peak=1
    for c in reversed(channels):
        i=c.site
        if c.width==1:
            a[i]=np.einsum('xy,ayb->axb',c.t.T,a[i]); continue
        center(a,i)
        z=np.einsum('aib,bjc->aijc',a[i],a[i+1]); l,_,_,r=z.shape
        z=np.einsum('xy,ayb->axb',c.t.T,z.reshape(l,16,r)).reshape(l*4,4*r)
        u,s,v=np.linalg.svd(z,full_matrices=False)
        k=min(cap,max(1,int(np.count_nonzero(s>1e-14)))); err=float(np.linalg.norm(s[k:]))
        ledger.append(err); peak=max(peak,k)
        a[i]=u[:,:k].reshape(l,4,k); a[i+1]=(s[:k,None]*v[:k]).reshape(k,4,r)
    v=np.ones(1)
    for t in a: v=v@(t[:,0,:]+t[:,3,:])
    # Normalized HS norm => operator norm via sqrt(2^n). CPTP adjoints
    # contract operator norm on Hermitian operators, including nonunital maps.
    bound=2**(n/2)*math.fsum(ledger)
    return {'z':float(v[0]),'truncation_bound_exact_arithmetic':bound,'discarded_norm_sum':math.fsum(ledger),'peak_bond':peak,'svds':len(ledger)}

def dense_z(n,channels,target,readout=(0.,0.)):
    channels=list(channels); check(n,channels,target)
    if n>8: raise ValueError('dense reference limited to 8 sites')
    rho=np.zeros((2**n,2**n),complex); rho[0,0]=1
    for c in channels:
        new=np.zeros_like(rho)
        for k in c.ks:
            u=np.kron(np.kron(np.eye(2**c.site),k),np.eye(2**(n-c.site-c.width)))
            new+=u@rho@u.conj().T
        rho=new
    v=readout_vector(*readout)
    o=np.kron(np.kron(np.eye(2**target),v[0]*P[0]+v[3]*P[3]),np.eye(2**(n-target-1)))
    return float(np.trace(o@rho).real)

def native_channels(ops,snapshot,sx_offsets=None):
    """Preserve QC23 Model A fault locations, plus optional native SX offsets.
    QC23's thermal faults are Pauli-twirled; use thermal() for nonunital maps.
    Offsets need independent native calibration; logical QC24 offsets are not SX offsets.
    """
    from qc23_noise import model_a_faults
    from qc23_mpo import _matrix,two_qubit_big_endian
    faults,readout=model_a_faults(ops,snapshot)
    out=[]; sx_offsets=sx_offsets or {}
    def append_faults(k):
        for f in faults:
            if f.after_op!=k: continue
            sites=f.qubits; width=len(sites)
            if width==2 and abs(sites[0]-sites[1])!=1: raise ValueError('nonadjacent fault')
            probs=f.paulis if tuple(sites)==tuple(sorted(sites)) else {lab[::-1]:v for lab,v in f.paulis.items()}
            out.append(stochastic(min(sites),width,probs))
    append_faults(-1)
    for k,o in enumerate(ops):
        if o.name not in ('measure','barrier','delay','id'):
            if len(o.logical)==1:
                u=_matrix(o.name,o.params)
                if o.name=='sx': u=rotation(P[1],sx_offsets.get(o.physical[0],0.))@u
            elif len(o.logical)==2 and abs(o.logical[0]-o.logical[1])==1:
                u=two_qubit_big_endian(o.name,o.params,*o.logical)
            else: raise ValueError('nonlocal or unsupported operation')
            out.append(Channel(min(o.logical),(u,),o.name))
        append_faults(k)
    return out,[(r,r) for r in readout]
