"""QC-13: exact finite-binary64 circuit oracle on at most three qubits.

Uses fractions.Fraction for every real and imaginary component, so there is
zero arithmetic roundoff. Its reference circuit consists of the *literal*
complex128 coefficients supplied, interpreted as exact dyadic rationals.
Floating QR unitaries and Kraus sets may not be exactly CPTP as dyadics; the
result is an exact audit of the supplied coefficients, not an unspecified
ideal unitary/CPTP completion. Intended only for tiny independent checks.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from typing import Sequence
import numpy as np
from qc13_dyadic import _lower_float,_upper_float

C=tuple[F,F]
ZERO=(F(0),F(0))
ONE=(F(1),F(0))

def cv(z)->C:
    z=complex(z)
    if not np.isfinite([z.real,z.imag]).all():
        raise ValueError('nonfinite exact-circuit input')
    return F(z.real),F(z.imag)

def add(a:C,b:C)->C:return a[0]+b[0],a[1]+b[1]
def conj(a:C)->C:return a[0],-a[1]
def mul(a:C,b:C)->C:return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]

def qmatrix(a:np.ndarray):
    a=np.asarray(a)
    if a.ndim!=2:raise ValueError('matrix required')
    return [[cv(z) for z in row] for row in a]

def eye(n:int):return [[ONE if i==j else ZERO for j in range(n)] for i in range(n)]

def dagger(a):return [[conj(a[i][j]) for i in range(len(a))] for j in range(len(a[0]))]

def dot(a,b):
    nr,nc,k=len(a),len(b[0]),len(b)
    if len(a[0])!=k:raise ValueError('matrix dimensions')
    out=[]
    for i in range(nr):
        row=[]
        for j in range(nc):
            re=im=F(0)
            for t in range(k):
                ar,ai=a[i][t];br,bi=b[t][j]
                if (ar or ai) and (br or bi):
                    re+=ar*br-ai*bi;im+=ar*bi+ai*br
            row.append((re,im))
        out.append(row)
    return out

def kron(a,b):
    return [[mul(x,y) for x in a[i] for y in b[s]]
            for i in range(len(a)) for s in range(len(b))]

def embed(n:int,site:int,mat):
    width=1 if len(mat)==2 else 2
    if not 0<=site<=n-width:raise ValueError('bad gate site')
    return kron(kron(eye(2**site),mat),eye(2**(n-site-width)))

def sandwich(q,rho):return dot(dot(q,rho),dagger(q))

def sum_matrices(a,b):
    return [[add(x,y) for x,y in zip(ra,rb)] for ra,rb in zip(a,b)]

def trace(a)->C:
    re=im=F(0)
    for i in range(len(a)):
        a0,b0=a[i][i];re+=a0;im+=b0
    return re,im


@dataclass(frozen=True)
class RationalOracle:
    expectation_real_lo:float
    expectation_real_hi:float
    expectation_imag_lo:float
    expectation_imag_hi:float
    exact_real:F
    exact_imag:F
    trace_real_error:F
    trace_imag_error:F
    qubits:int
    depth:int
    def absolute_difference(self,computed:float)->float:
        if not np.isfinite(computed):raise ValueError('invalid comparison')
        return _upper_float(abs(self.exact_real-F(computed)))


def exact_circuit_expectation(n:int,layers:Sequence,operators:tuple[np.ndarray,...],
                              targets:Sequence[int],label:str)->RationalOracle:
    """Exact dyadic forward evolution and Pauli expectation, n<=3.

    The density matrix need not remain exactly trace one because the literal
    finite-precision gate/Kraus coefficients are not exact isometries.
    Returns that observed trace defect explicitly.
    """
    if not 1<=n<=3:raise ValueError('exact oracle restricted to <=3 qubits')
    targets=tuple(targets)
    if not targets or len(targets)!=len(label) or len(set(targets))!=len(targets) or any(t<0 or t>=n for t in targets):
        raise ValueError('invalid targets')
    if any(ch not in 'IXYZ' for ch in label):raise ValueError('invalid Pauli labels')
    if not operators:raise ValueError('Kraus operators required')
    z=eye(2**n)
    state=[[ZERO for _ in range(2**n)] for _ in range(2**n)]
    state[0][0]=ONE
    channels=[[embed(n,site,qmatrix(k)) for k in operators] for site in range(n)]
    for layer in layers:
        for site,u in layer:
            state=sandwich(embed(n,site,qmatrix(u)),state)
        for site in range(n):
            new=[[ZERO for _ in range(2**n)] for _ in range(2**n)]
            for k in channels[site]:
                new=sum_matrices(new,sandwich(k,state))
            state=new
    from channel_mpo import I2,X,Y,Z
    ops=dict(zip(targets,label)); pauli=qmatrix({'I':I2,'X':X,'Y':Y,'Z':Z}[ops.get(0,'I')])
    for i in range(1,n):
        pauli=kron(pauli,qmatrix({'I':I2,'X':X,'Y':Y,'Z':Z}[ops.get(i,'I')]))
    ex=trace(dot(state,pauli))
    norm=trace(state)
    return RationalOracle(_lower_float(ex[0]),_upper_float(ex[0]),
                          _lower_float(ex[1]),_upper_float(ex[1]),
                          ex[0],ex[1],norm[0]-1,norm[1],n,len(layers))
