"""Real Householder QFT compiler to explicit H, RY, RZ, CX and global phase.
Big-endian qubit indices. No Qiskit, external service, or hardware used.
"""
from itertools import combinations
from math import atan2, pi
import numpy as np


def negative_vectors(u, tol=1e-10):
    assert np.max(np.abs(u.imag)) < tol
    u = u.real
    assert np.max(np.abs(u-u.T)) < tol and np.max(np.abs(u@u-np.eye(len(u)))) < tol
    p = (np.eye(len(u))-u)/2
    basis = []
    for v in p.T:
        w = v.copy()
        for _ in range(2):
            for a in basis:
                w -= (a@w)*a
        n = np.linalg.norm(w)
        if n > tol:
            basis.append(w/n)
    assert len(basis)==round(np.trace(p))
    return basis


def mux_ry(angles, controls, target):
    """Gray-cycle uniformly controlled RY; angles indexed by binary controls."""
    k = len(controls)
    assert len(angles)==1<<k
    if not k:
        return [('ry',target,float(angles[0]))]
    gray = [i^(i>>1) for i in range(1<<k)]
    out = []
    for i,g in enumerate(gray):
        a = sum((-1)**((b&g).bit_count())*x for b,x in enumerate(angles))/(1<<k)
        out.append(('ry',target,float(a)))
        bit = (g^gray[(i+1)%len(gray)]).bit_length()-1
        out.append(('cx',controls[k-1-bit],target))
    return out


def prepare(v, wires):
    n = 1<<len(wires)
    a = np.pad(v,(0,n-len(v)))
    assert np.isclose(np.linalg.norm(a),1)
    gates = []
    for k,t in enumerate(wires):
        block = n>>k
        angles = []
        for j in range(1<<k):
            x = a[j*block:(j+1)*block]
            if block==2:
                l,r = x
            else:
                l,r = np.linalg.norm(x[:block//2]),np.linalg.norm(x[block//2:])
            angles.append(2*atan2(r,l) if l or r else 0.)
        gates += mux_ry(angles,wires[:k],t)
    return gates


def inverse(gates):
    return [(g[0],g[1],-g[2]) if g[0] in ('ry','rz') else ('phase',-g[1]) if g[0]=='phase' else g
            for g in reversed(gates)]


def zero_reflection(wires):
    """I-2|0><0| = exp(i*pi/2^b * sum of all Z-subset products)."""
    theta = pi/(1<<len(wires))
    gates = [('phase',theta)]
    for k in range(1,len(wires)+1):
        for s in combinations(wires,k):
            p = s[-1]
            a = [('cx',c,p) for c in s[:-1]]
            gates += a+[('rz',p,-2*theta)]+a[::-1]
    return gates


def compile_involution(u,wires):
    gates = []
    for v in negative_vectors(u):
        p = prepare(v,wires)
        gates += inverse(p)+zero_reflection(wires)+p
    return gates


def unitary(gates,n):
    """Independent direct simulation of the emitted one- and two-qubit gates."""
    a = np.eye(1<<n,dtype=complex)
    ids = np.arange(1<<n)
    for g in gates:
        kind = g[0]
        if kind=='phase':
            a *= np.exp(1j*g[1]);continue
        bit = 1<<(n-1-g[1])
        lo = ids[(ids&bit)==0];hi=lo|bit
        if kind=='cx':
            t=1<<(n-1-g[2]);pick=ids[((ids&bit)!=0)&((ids&t)==0)];a[pick],a[pick|t]=a[pick|t].copy(),a[pick].copy()
        elif kind=='x':
            a[lo],a[hi]=a[hi].copy(),a[lo].copy()
        elif kind=='h':
            a[lo],a[hi]=(a[lo]+a[hi])/np.sqrt(2),(a[lo]-a[hi])/np.sqrt(2)
        elif kind=='ry':
            c,s=np.cos(g[2]/2),np.sin(g[2]/2)
            a[lo],a[hi]=c*a[lo]-s*a[hi],s*a[lo]+c*a[hi]
        elif kind=='rz':
            a[lo]*=np.exp(-.5j*g[2]);a[hi]*=np.exp(.5j*g[2])
        else:
            raise ValueError(g)
    return a


def qasm(gates,n):
    lines=['OPENQASM 3.0;','include "stdgates.inc";',f'qubit[{n}] q;']
    for g in gates:
        k=g[0]
        if k=='phase':lines.append(f'gphase({g[1]:.17g});')
        elif k=='cx':lines.append(f'cx q[{g[1]}], q[{g[2]}];')
        elif k in ('h','x'):lines.append(f'{k} q[{g[1]}];')
        else:lines.append(f'{k}({g[2]:.17g}) q[{g[1]}];')
    return '\n'.join(lines)+'\n'
