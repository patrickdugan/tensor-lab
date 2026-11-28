"""Recursive shell Fourier synthesis with and without clean ancillas.
All gate angles are binary64 approximations to the stated analytic angles.
Binary shell encoding, wire 0 is MSB. Circuit list is chronological.
"""
from math import asin, atan, atan2, sqrt, log
from collections import Counter
import numpy as np


def mux(angles, controls, target):
    """Fast Walsh-Hadamard transform of the rotation table, Gray-cycle ordering."""
    k = len(controls)
    a = np.array(angles, dtype=float)
    assert len(a) == 1 << k
    if not k:
        return [('ry', target, float(a[0]))]
    for r in range(k):
        step = 1 << r
        z = a.reshape(-1, 2*step)
        x, y = z[:, :step].copy(), z[:, step:].copy()
        z[:, :step], z[:, step:] = x+y, x-y
    a /= len(a)
    out = []
    for i in range(1 << k):
        g = i ^ (i >> 1)
        j = (i+1) % (1 << k)
        gnext = j ^ (j >> 1)
        bit = (g ^ gnext).bit_length()-1
        out += [('ry',target,float(a[g])), ('cx',controls[k-1-bit],target)]
    return out


def inverse(gates):
    return [(*g[:-1], -g[-1]) if g[0] in ('ry','rz','cry','p','phase') else g
            for g in reversed(gates)]


def angles(p, r, k):
    # ratio of adjacent binary branches of an exponentially decaying vector
    t = float(p)**(-(1 << (r-1-k))/2)
    return 2*atan(t), 2*asin(t)


def preparation(wires, p, anc=None):
    """P maps |0,0> to |0,L-1>, and |1,0> to |1,W_L^T 0>.
    Ancilla mode uses a reversible prefix-AND chain, uncomputed at exit.
    """
    h, *low = wires
    r = len(low)
    assert r >= 1
    out = [('x',h)] + [('cx',h,t) for t in low] + [('x',h)]
    if anc is None:
        for k,t in enumerate(low):
            ordinary,special = angles(p,r,k)
            a = [0.]*(1<<k) + [ordinary]*(1<<k)
            a[-1] = special
            out += mux(a,[h]+low[:k],t)
    else:
        assert len(anc) >= r-1
        chain=[]
        for k,t in enumerate(low):
            ordinary,special = angles(p,r,k)
            if k==0:
                out.append(('cry',h,t,special))
            else:
                g=('ccx',h if k==1 else anc[k-2],low[k-1],anc[k-1])
                out.append(g);chain.append(g)
                out += [('cry',h,t,ordinary), ('cry',anc[k-1],t,special-ordinary)]
        out += chain[::-1]
    return out


def boundary(wires,p,anc=None):
    h,*low=wires
    theta=-2*atan2(1,sqrt(p-1))
    if anc is None:
        return mux([theta]+[0.]*((1<<len(low))-1),low,h)
    if len(low)==1:
        return [('x',low[0]),('cry',low[0],h,theta),('x',low[0])]
    chain=[('ccx',low[0],low[1],anc[0])]
    for j in range(2,len(low)):
        chain.append(('ccx',anc[j-2],low[j],anc[j-1]))
    flips=[('x',x) for x in low]
    return flips+chain+[('cry',anc[len(low)-2],h,theta)]+chain[::-1]+flips


def w_product(wires,p,mode,anc):
    if len(wires)==1:
        return [('ry',wires[0],-2*atan2(1,sqrt(p-1)))]
    r=len(wires)-1
    # Choose the smaller CNOT count per recursive level, deterministically.
    use_clean=mode=='clean' or (mode=='hybrid' and 46*r-38 < 5*(1<<r)+2*r-4)
    a=anc if use_clean else None
    prep=preparation(wires,p,a)
    return inverse(prep)+boundary(wires,p,a)+prep+w_product(wires[1:],p,mode,anc)


def compile_shell(b,p=2,mode='noanc'):
    """Q=W R, R is bitwise complement of the binary shell label."""
    assert b>=1 and p>=2 and mode in ('noanc','clean','hybrid')
    # Prime condition needed for the mathematical application; gate formula also
    # makes sense for real p>=2. Restrict this public interface to primes.
    assert all(p%d for d in range(2,int(sqrt(p))+1))
    a=max(0,b-2) if mode=='clean' or (mode=='hybrid' and b>=7) else 0
    wires=list(range(b)); anc=list(range(b,b+a))
    return [('x',i) for i in wires]+w_product(wires,p,mode,anc), b+a


def expand_cry(gates):
    out=[]
    for g in gates:
        if g[0]=='cry':
            _,c,t,a=g
            out += [('ry',t,a/2),('cx',c,t),('ry',t,-a/2),('cx',c,t)]
        else: out.append(g)
    return out


def expand_ccx(gates):
    """Standard exact 6-CX, 7-T/Tdag Toffoli decomposition (no phase ambiguity)."""
    out=[]
    for g in gates:
        if g[0]=='ccx':
            _,a,b,t=g
            out += [('h',t),('cx',b,t),('tdg',t),('cx',a,t),('t',t),('cx',b,t),('tdg',t),('cx',a,t),('t',b),('t',t),('h',t),('cx',a,b),('t',a),('tdg',b),('cx',a,b)]
        else: out.append(g)
    return out


def counts(gates):
    c=Counter(g[0] for g in gates)
    return dict(abstract=dict(sorted(c.items())),cx=c['cx']+2*c['cry']+6*c['ccx'],
                ry=c['ry']+2*c['cry'],t_from_ccx=7*c['ccx'],h_from_ccx=2*c['ccx'])


def simulate(gates,n,states):
    """Independent direct state evolution. Supports arbitrary complex columns."""
    a=np.array(states,dtype=complex,copy=True); assert a.shape[0]==1<<n
    ids=np.arange(1<<n,dtype=np.int64)
    for g in gates:
        k=g[0]
        if k=='phase': a*=np.exp(1j*g[1]);continue
        if k in ('cx','ccx','cry'):
            target=g[2] if k!='ccx' else g[3]
            controls=g[1:2] if k!='ccx' else g[1:3]
        else:
            target=g[1]; controls=()
        bit=1<<(n-1-target); mask=sum(1<<(n-1-i) for i in controls)
        lo=ids[((ids&bit)==0)&((ids&mask)==mask)];hi=lo|bit
        if k in ('cx','ccx','x'):
            a[lo],a[hi]=a[hi].copy(),a[lo].copy()
        elif k in ('ry','cry'):
            c,s=np.cos(g[-1]/2),np.sin(g[-1]/2);x,y=a[lo].copy(),a[hi].copy()
            a[lo],a[hi]=c*x-s*y,s*x+c*y
        elif k=='h':
            x,y=a[lo].copy(),a[hi].copy();a[lo],a[hi]=(x+y)/sqrt(2),(x-y)/sqrt(2)
        elif k=='p': a[hi]*=np.exp(1j*g[-1])
        elif k=='rz': a[lo]*=np.exp(-.5j*g[-1]);a[hi]*=np.exp(.5j*g[-1])
        elif k in ('t','tdg'): a[hi]*=np.exp((1 if k=='t' else -1)*1j*np.pi/4)
        else: raise ValueError(g)
    return a


def qasm(gates,n):
    lines=['OPENQASM 3.0;', 'include "stdgates.inc";',f'qubit[{n}] q;']
    for g in expand_cry(gates):
        k=g[0]
        if k=='phase': lines.append(f'gphase({g[1]:.17g});')
        elif k in ('ry','rz','p'): lines.append(f'{k}({g[2]:.17g}) q[{g[1]}];')
        else: lines.append(k+' '+', '.join(f'q[{i}]' for i in g[1:])+';')
    return '\n'.join(lines)+'\n'
