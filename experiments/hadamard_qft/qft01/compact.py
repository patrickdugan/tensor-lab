"""Input-independent dyadic shell circuit compaction, MSB-first wire labels."""
from itertools import permutations
from math import atan, asin, sqrt, pi
from functools import lru_cache
import numpy as np
from recursive import mux, inverse, simulate
from radial_qft import radial_matrix, gray
from qpu01 import route, count_cx


def mapped(g, pos):
    return [(x[0],pos[x[1]],pos[x[2]]) if x[0]=='cx' else (x[0],pos[x[1]],*x[2:]) for x in g]


def smart_mux(a, cs, t, pos, prep=False):
    # Gray-cycle control order places the most frequent flips nearest target.
    # For state preparation only, input target is |0>. Omit final controlled X,
    # replacing theta by pi-theta when its control is one. This changes the
    # unitary extension but preserves the selected preparation vectors exactly.
    new=sorted(cs,key=lambda q:(-abs(pos[q]-pos[t]),q)); k=len(cs)
    vals=[]
    for x in range(1<<k):
        old=0
        for q in cs:old=2*old+((x>>(k-1-new.index(q)))&1)
        v=a[old]
        vals.append(pi-v if prep and k and x>>(k-1) else v)
    g=mux(vals,new,t)
    if prep and k:
        assert g[-1]==('cx',new[0],t)
        g=g[:-1]
    return g


def gray_core(b,pos):
    enc=[('cx',i,i+1) for i in reversed(range(b-1))];out=enc+[('x',0)]
    for j in reversed(range((1<<b)-1)):
        x,y=gray(j),gray(j+1);t=b-(x^y).bit_length()
        cs=[i for i in range(b) if i!=t];k=0
        for c in cs:k=2*k+((x>>(b-1-c))&1)
        a=[0.]*(1<<len(cs));a[k]=pi/2 if x&(1<<(b-1-t)) else -pi/2
        out+=smart_mux(a,cs,t,pos)
    return out+enc[::-1]


def level_bound(r):
    L=1<<r
    return (1-1/sqrt(2))*2**(-(L-1)/2)+2.**(-L)


def recursive_core(b,pos,arm):
    def wprod(w):
        if len(w)==1:return [('ry',w[0],-pi/2)]
        h,*low=w;r=len(low);p=[]
        if arm=='recursive':p=[('x',h)]+[('cx',h,t) for t in low]+[('x',h)]
        for k,t in enumerate(low):
            z=2**(-(1<<(r-1-k))/2);a=2*atan(z)
            if arm=='compact_approx' and r>=4:
                p+=smart_mux([pi,a],[h],t,pos,True)
            else:
                v=[0. if arm=='recursive' else pi]*(1<<k)+[a]*(1<<k);v[-1]=2*asin(z)
                p+=smart_mux(v,[h]+low[:k],t,pos,arm!='recursive')
        c=smart_mux([-pi/2]+[0.] *((1<<r)-1),low,h,pos)
        return inverse(p)+c+p+wprod(low)
    return [('x',i) for i in range(b)]+wprod(list(range(b)))


def simplify(gates):
    # Exact commuting cancellation and RY fusion, aside from floating roundoff.
    def wires(g):return set(g[1:3] if g[0]=='cx' else g[1:2])
    def commute(a,b):
        if not wires(a)&wires(b):return True
        if a[0]==b[0]=='cx':return a[1]!=b[2] and a[2]!=b[1]
        return False
    out=[]
    for g in gates:
        if g[0]=='ry' and g[2]==0:continue
        for k in range(len(out)-1,-1,-1):
            h=out[k]
            if h[:2]==g[:2] and h[0]==g[0]=='ry':
                v=h[2]+g[2];out.pop(k)
                if v:out.insert(k,('ry',g[1],v))
                break
            if h==g and g[0] in ('cx','x'):
                out.pop(k);break
            if not commute(h,g):out.append(g);break
        else:out.append(g)
    return out


def physical_index(b,pos):
    return np.array([sum(((j>>(b-1-i))&1)<<(b-1-pos[i]) for i in range(b)) for j in range(1<<b)])


@lru_cache(None)
def synthesize(b,arm):
    """Enumerate layouts b<=5; b6 uses 5-bit best layout insertions.
    Minimize routed CNOT count, total gates, then layout. No hardware data.
    """
    assert 3<=b<=6 and arm in ('gray','recursive','compact_exact','compact_approx')
    if b<=5:layouts=permutations(range(b))
    else:
        small=synthesize(5,arm)['pos'];layouts=sorted(set([tuple(range(b))]+[tuple([j]+[x+(x>=j) for x in small]) for j in range(b)]+[tuple([x+(x>=j) for x in small]+[j]) for j in range(b)]))
    best=None;tested=0
    for pos in layouts:
        g=gray_core(b,pos) if arm=='gray' else recursive_core(b,pos,arm)
        logical=simplify(g);routed=simplify(route(mapped(logical,pos)));tested+=1
        key=(count_cx(routed),len(routed),pos)
        if best is None or key<best[0]:best=(key,logical,routed)
    (cx,size,pos),logical,routed=best
    ix=physical_index(b,pos);I=np.eye(1<<b);Q=radial_matrix((1<<b)-1)
    u=simulate(routed,b,I[:,ix])[ix,:]
    bound=sum(level_bound(r) for r in range(4,b)) if arm=='compact_approx' else 0.
    err=float(np.linalg.norm(u-Q,ord=2));maxerr=float(np.max(abs(u-Q)))
    assert err<=bound+1e-10,(arm,b,err,bound)
    assert np.max(abs(u.conj().T@u-I))<1e-10
    return dict(b=b,arm=arm,pos=list(pos),logical_cx=count_cx(logical),line_cx=cx,
                gates=[list(x) for x in routed],size=size,analytic_bound=bound,
                operator_error=err,max_entry_error=maxerr,layouts_tested=tested)


if __name__=='__main__':
    import json,time
    from pathlib import Path
    results=[];t=time.monotonic()
    for b in (3,4,5,6):
        for arm in ('gray','recursive','compact_exact','compact_approx'):
            r=synthesize(b,arm);results.append(r);print({k:v for k,v in r.items() if k!='gates'},flush=True)
    Path(__file__).with_name('COMPACTION_AUDIT.json').write_text(json.dumps(results,indent=2)+'\n')
    print('seconds',time.monotonic()-t)
