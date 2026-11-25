"""QC-13: exact Pauli-transfer oracle for deep Clifford + Pauli/AD circuits.

This independent oracle uses Fraction coefficients, exact Clifford-conjugation
tables derived with rational Gaussian matrix arithmetic, and the quadratic
number field Q(sqrt(1-p)) for amplitude damping. A dyadic square-root bracket
then encloses the final expectation. Works beyond dense-reference cone widths,
provided the number of expanded Pauli strings stays within max_terms.

It certifies the *specified ideal Clifford/channel circuit*. A NumPy Kraus
representation at an inexact p is a nearby, numerically defined circuit;
coefficient and QR/SVD roundoff still prevent certifying the generic QC-12
Heisenberg output with this oracle alone.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from math import isqrt
import numpy as np
from qc13_rational import qmatrix,dot,dagger,kron
from qc13_dyadic import _lower_float,_upper_float
from channel_mpo import I2,X,Y,Z

PAULI={'I':I2,'X':X,'Y':Y,'Z':Z}
HH=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],complex)/2
CNOT=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],complex)
CZ=np.diag([1,1,1,-1]).astype(complex)
SWAP=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],complex)
S_I=np.diag([1,1,1j,1j]).astype(complex)
GATES={'HH':HH,'CNOT':CNOT,'CZ':CZ,'SWAP':SWAP,'S_I':S_I}


def _exact_equal(a,b):
    return all(x==y for ra,rb in zip(a,b) for x,y in zip(ra,rb))


def clifford_tables():
    """Compute sign, Pauli mapping exactly using Gaussian rational arithmetic."""
    table={}
    products={(a,b):kron(qmatrix(PAULI[a]),qmatrix(PAULI[b]))
              for a in 'IXYZ' for b in 'IXYZ'}
    for name,unitary in GATES.items():
        u=qmatrix(unitary);v=dagger(u)
        t={}
        for key,old in products.items():
            after=dot(dot(v,old),u)
            matches=[]
            for dest,target in products.items():
                if _exact_equal(after,target):matches.append((dest,1))
                neg=[[(-x[0],-x[1]) for x in row] for row in target]
                if _exact_equal(after,neg):matches.append((dest,-1))
            if len(matches)!=1:
                raise ArithmeticError(f'{name}: Clifford identity cannot be certified for {key}')
            t[key]=matches[0]
        table[name]=t
    return table

TABLES=clifford_tables()


def clifford_layers(n:int,depth:int,seed:int):
    if not 2<=n<=256 or depth<0:raise ValueError('invalid circuit size')
    rng=np.random.default_rng(seed)
    names=tuple(GATES)
    out=[]
    for t in range(depth):
        layer=[]
        for i in range(t%2,n-1,2):
            name=names[int(rng.integers(len(names)))]
            layer.append((i,GATES[name],name))
        out.append(layer)
    return out


@dataclass(frozen=True)
class Algebraic:
    a:F
    b:F
    # Represents a + b * sqrt(q), where q=1-p, supplied separately.
    def __add__(self,o:'Algebraic')->'Algebraic':return Algebraic(self.a+o.a,self.b+o.b)
    def scale(self,x:F)->'Algebraic':return Algebraic(self.a*x,self.b*x)
    def root(self,q:F)->'Algebraic':return Algebraic(self.b*q,self.a)
    def iszero(self):return not(self.a or self.b)


@dataclass(frozen=True)
class CliffordOracle:
    expectation_lo:float
    expectation_hi:float
    exact_a:F
    exact_b:F
    root_argument:F
    max_terms:int
    final_terms:int
    qubits:int
    depth:int
    channel:str
    p:F
    def absolute_difference(self,x:float)->float:
        if not np.isfinite(x):raise ValueError('invalid approximate expectation')
        err_lo=self.expectation_lo-x
        err_hi=self.expectation_hi-x
        # Extra nextafter happens in _upper_float on the *exact* coefficient
        # enclosure below; the float subtraction here may be rounded.
        from math import nextafter,inf
        return nextafter(max(abs(err_lo),abs(err_hi)),inf)


def _sqrt_bracket(q:F,bits:int=160):
    if not 0<=q<=1:raise ValueError('q out of range')
    m=(q.numerator<<(2*bits))//q.denominator
    lo=isqrt(m)
    return F(lo,1<<bits),F(lo+1,1<<bits)


def _algebraic_interval(x:Algebraic,q:F)->tuple[float,float]:
    lo,hi=_sqrt_bracket(q)
    if x.b<0:lo,hi=hi,lo
    return (_lower_float(x.a+x.b*lo),_upper_float(x.a+x.b*hi))


def exact_clifford_expectation(n:int,layers,channel:str,p:F,
                               target:int,label:str='Z',*,max_terms:int=100_000)->CliffordOracle:
    p=F(p);q=1-p
    if channel not in ('depolarizing','dephasing','amplitude_damping') or not 0<=p<=1:
        raise ValueError('invalid noise channel or strength')
    if not 0<=target<n or label not in 'XYZ' or max_terms<1:
        raise ValueError('invalid target/limits')
    word='I'*target+label+'I'*(n-target-1)
    states={word:Algebraic(F(1),F(0))}
    maximum=1
    for layer in reversed(layers):
        # Channel adjoint on all active sites. Gates after channels in forward
        # time means the channel adjoint precedes the gate adjoint here.
        if channel=='amplitude_damping':
            for site in range(n):
                out={}
                for word,co in states.items():
                    ch=word[site]
                    if ch=='Z':
                        options=(('Z',co.scale(q)),('I',co.scale(p)))
                    elif ch in 'XY':options=((ch,co.root(q)),)
                    else:options=((ch,co),)
                    for next_ch,next_co in options:
                        if next_co.iszero():continue
                        w=word[:site]+next_ch+word[site+1:]
                        out[w]=out.get(w,Algebraic(F(0),F(0)))+next_co
                states={k:v for k,v in out.items() if not v.iszero()}
                if len(states)>max_terms:raise MemoryError('Pauli expansion exceeds max_terms')
                maximum=max(maximum,len(states))
        else:
            out={}
            for word,co in states.items():
                if channel=='depolarizing':
                    factor=(1-p)**sum(c!='I' for c in word)
                else:
                    factor=(1-2*p)**sum(c in 'XY' for c in word)
                new=co.scale(factor)
                if not new.iszero():out[word]=new
            states=out
        for site,_,gate_name in layer:
            new={}
            table=TABLES[gate_name]
            for word,co in states.items():
                dest,sign=table[(word[site],word[site+1])]
                w=word[:site]+''.join(dest)+word[site+2:]
                term=co.scale(F(sign))
                new[w]=new.get(w,Algebraic(F(0),F(0)))+term
            states={k:v for k,v in new.items() if not v.iszero()}
        maximum=max(maximum,len(states))
    answer=Algebraic(F(0),F(0))
    for word,co in states.items():
        if all(ch in 'IZ' for ch in word):answer=answer+co
    lo,hi=_algebraic_interval(answer,q)
    return CliffordOracle(lo,hi,answer.a,answer.b,q,maximum,len(states),n,len(layers),channel,p)
