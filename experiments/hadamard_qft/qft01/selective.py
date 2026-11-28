"""Ancilla-free selective rotations via a relative-phase flip commutator.
MSB-first wires, chronological x/ry/rz/p/h/cx gates. Exact algebra, FP64 angles.
"""
from math import pi, atan, asin, sqrt
from functools import lru_cache
import numpy as np
from recursive import mux, inverse, simulate
from radial_qft import radial_matrix
from compact import level_bound, simplify


def count(g): return sum(x[0]=='cx' for x in g)


def onehot(controls,target,theta,axis='ry'):
    vals=[theta]+[0.] *((1<<len(controls))-1)
    return [(axis,*g[1:]) if g[0]=='ry' else g for g in mux(vals,controls,target)]


def relative_flip(controls,target):
    """A monomial unitary flipping t iff all controls are zero; phases arbitrary."""
    k=len(controls);vals=[pi]+[0.] *((1<<k)-1)
    if not k:return [('x',target)]
    if k==3:
        a,b,c=controls;t=target
        # Six-CX relative-phase triple-controlled flip. The middle diagonal
        # loop is I unless a=b=1, when it is iZ. The outer conjugation is I
        # for c=0, and maps Z to an equatorial Pauli for c=1. Hence the
        # complete gate is monomial with precisely the desired permutation.
        pre=[('h',t),('p',t,pi/4),('cx',c,t),('p',t,-pi/4),('h',t)]
        mid=[('cx',a,t),('p',t,pi/4),('cx',b,t),('p',t,-pi/4),('cx',a,t),('p',t,pi/4),('cx',b,t),('p',t,-pi/4)]
        flips=[('x',i) for i in controls]
        return flips+pre+mid+inverse(pre)+flips
    vals=[pi-v if i>>(k-1) else v for i,v in enumerate(vals)]
    g=mux(vals,controls,target)
    assert g[-1]==('cx',controls[0],target)
    return g[:-1]


@lru_cache(None)
def cost(r):
    # Controlled Rz uses 2^r cyclic mux or a relative flip of a controls
    # and a recursively synthesized Rz with r-a controls, twice each.
    if r==0:return (0,0)
    return min([(1<<r,0)]+[(2*((6 if a==3 else (1<<a)-1)+cost(r-a)[0]),a) for a in range(1,r)])


def selective_z(controls,target,theta,force_direct=False):
    r=len(controls);a=0 if force_direct else cost(r)[1]
    if not a:return onehot(controls,target,theta,'rz')
    A=relative_flip(controls[:a],target)
    B=selective_z(controls[a:],target,-theta/2)
    # B^-1 A^-1 B A = Rz(theta) when both predicates hold, I otherwise.
    return A+B+inverse(A)+inverse(B)


def selective_y(controls,target,theta,force_direct=False):
    if force_direct or not cost(len(controls))[1]:return onehot(controls,target,theta)
    return [('p',target,-pi/2),('h',target)]+selective_z(controls,target,theta)+[('h',target),('p',target,pi/2)]


def prep(h,low,approx_from=4,prune=(),order='rev'):
    out=[];r=len(low)
    for k,t in enumerate(low):
        z=2**(-(1<<(r-1-k))/2);a=2*atan(z)
        if r>=approx_from:
            vals=[pi,0. if (r,k) in prune else a];cs=[h]
        else:
            vals=[pi]*(1<<k)+[a]*(1<<k);vals[-1]=2*asin(z);cs=[h]+low[:k]
        order_cs=cs[::-1] if order=='rev' else cs
        vals2=[];l=len(cs)
        for x in range(1<<l):
            old=0
            for q in cs:old=2*old+((x>>(l-1-order_cs.index(q)))&1)
            v=vals[old]; vals2.append(pi-v if x>>(l-1) else v)
        g=mux(vals2,order_cs,t);assert g[-1]==('cx',order_cs[0],t)
        out+=g[:-1]
    return out


def core(b,style='commutator',order='rev',approx_from=4,prune=(),level_orders=None):
    """Use same compact preparations as QPU03 and replace boundary rotations."""
    assert b>=1 and order in ('rev','forward') and style in ('commutator','direct')
    level_orders=level_orders or {}
    out=[('x',i) for i in range(b)]
    for h in range(b-1):
        low=list(range(h+1,b));r=len(low)
        P=prep(h,low,approx_from,prune,order)
        cs=level_orders.get(r,low[::-1] if order=='rev' else low)
        assert sorted(cs)==low
        C=selective_y(cs,h,-pi/2,force_direct=style=='direct')
        out+=inverse(P)+C+P
    out+=[('ry',b-1,-pi/2)]
    # Existing simplify handles x,ry,cx and treats other 1q gates as blockers.
    return simplify(out)


def bound(b,approx_from=4,prune=()):
    return sum(level_bound(r) for r in range(approx_from,b))+sum(2*np.sin(atan(2**(-(1<<(r-1-k))/2))/2) for r,k in prune)


def audit(g,b,eps):
    d=1<<b;u=simulate(g,b,np.eye(d,dtype=complex));q=radial_matrix(d-1)
    err=float(np.linalg.norm(u-q,ord=2));orth=float(np.max(abs(u.conj().T@u-np.eye(d))))
    assert err<=eps+1e-10 and orth<1e-10,(err,eps,orth)
    rng=np.random.default_rng(1414+b);x=rng.normal(size=(d,12))+1j*rng.normal(size=(d,12));x/=np.linalg.norm(x,axis=0)
    ran=float(np.max(np.linalg.norm(simulate(g,b,x)-q@x,axis=0)))
    assert ran<=err+1e-10
    return dict(b=b,cx=count(g),operator_error=err,analytic_bound=float(eps),unitarity=orth,random_error=ran)


if __name__=='__main__':
    for r in range(1,9):
        g=selective_y(list(range(1,r+1)),0,-pi/2);n=1<<(r+1);v=np.eye(n,dtype=complex);L=1<<r;v[0,0]=v[L,L]=sqrt(.5);v[0,L]=sqrt(.5);v[L,0]=-sqrt(.5)
        e=np.max(abs(simulate(g,r+1,np.eye(n))-v));assert e<1e-11
        print('boundary',r,count(g),e,cost(r),flush=True)
    for b in (4,5,6,7,8):
        for arm in ('direct','commutator'):
            print(arm,audit(core(b,arm),b,bound(b)),flush=True)
