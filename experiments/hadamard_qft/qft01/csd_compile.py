"""Recursive real cosine-sine synthesis, with all sign diagonals commuted to the end.
Requires scipy; numerical compiler. All emitted gates use the same exact
controlled-rotation decomposition as circuit.py (up to angle roundoff).
"""
from math import pi
import numpy as np
from scipy.linalg import cossin
from circuit import mux_ry


def layers(mats, n, prefix=0):
    """Chronological real CS layers for a block diagonal of matrices."""
    d = 1<<(n-prefix)
    assert len(mats)==1<<prefix and all(x.shape==(d,d) for x in mats)
    if d==1:
        signs = np.array([round(float(x[0,0])) for x in mats])
        assert np.all(np.abs(signs)==1)
        return [('sign',signs)]
    left,right,angles=[],[],[]
    for a in mats:
        u,t,v = cossin(a,p=d//2,q=d//2,separate=True)
        left.extend(u);right.extend(v);angles.extend(2*t)
    return layers(right,n,prefix+1)+[('mux',prefix,np.array(angles))]+layers(left,n,prefix+1)


def diagonal_phase(phases,wires):
    """Exact phase table decomposed by averaging and differences."""
    if not wires:
        return [('phase',float(phases[0]))]
    p = np.asarray(phases).reshape(-1,2)
    angles = p[:,1]-p[:,0]
    gates = []
    if np.max(abs(angles))>1e-14:
        # X RZ(theta) X = RZ(-theta), same Gray-cycle construction.
        gates = [('rz',*g[1:]) if g[0]=='ry' else g for g in mux_ry(angles,wires[:-1],wires[-1])]
    return gates+diagonal_phase(p.mean(axis=1),wires[:-1])


def compile_csd(u,wires):
    n = len(wires);d=1<<n
    assert np.max(abs(u.imag))<1e-10
    a=np.eye(d);a[:len(u),:len(u)]=u.real
    assert np.max(abs(a@a.T-np.eye(d)))<1e-10
    signs=np.ones(d,dtype=np.int64);gates=[]
    for l in layers([a],n):
        if l[0]=='sign':
            signs*=l[1];continue
        t,theta=l[1:];bit=1<<(n-1-t)
        lo=np.array([i for i in range(d) if not i&bit]);hi=lo|bit
        # R D = D (D R D); diagonal signs alter the conditional angles only.
        angles=theta*signs[lo]*signs[hi]
        if np.max(abs(angles))>1e-14:
            gates+=mux_ry(angles,[q for j,q in enumerate(wires) if j!=t],wires[t])
    gates+=diagonal_phase(np.where(signs==1,0.,pi),wires)
    return gates
