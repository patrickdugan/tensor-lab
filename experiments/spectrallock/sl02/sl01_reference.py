"""Experimental shell preimages, synthetic inputs only."""
from itertools import groupby
from math import gcd


def spectrum(s, ell=2):
    P, t = s[0], 1
    f = []
    for x in s[1:]:
        f.append(P-t*x)
        P += (ell-1)*t*x
        t *= ell
    return (*f, P)


def rounded(s, a, q, bins, ell=2):
    return tuple(bins*(x*y%q)//q for x,y in zip(spectrum(s,ell),a))


def invert_rounded(y, a, q, bins, ell=2, cap=100000):
    n=len(y)-1
    assert n>=1 and len(a)==len(y) and q>4*ell**n and 2<=bins<q
    states={-1:(-1,),1:(1,)}
    work, peak, t = 0,2,1
    for j in range(n):
        nxt={}
        for P, path in states.items():
            for x in (-1,1):
                if work>=cap: return 'CAP',None,work,peak
                work+=1
                if bins*(a[j]*(P-t*x)%q)//q==y[j]:
                    nxt.setdefault(P+(ell-1)*t*x,path+(x,))
        states=nxt;peak=max(peak,len(states));t*=ell
        if not states: return 'EMPTY',None,work,peak
    for P,path in states.items():
        if bins*(a[-1]*P%q)//q==y[-1]:
            assert rounded(path,a,q,bins,ell)==tuple(y)
            return 'HIT',path,work,peak
    return 'EMPTY',None,work,peak


def prime_certificate(ell,n):
    """q=k*ell**n+1 >4N; Pocklington with F=N>sqrt(q)."""
    assert ell in (2,3,5,7) and n>=7
    N=ell**n
    for k in range(5,min(N,100000)):
        q=k*N+1
        if q%2==0 or any(q%p==0 for p in (3,5,7,11,13,17,19,23,29,31)):
            continue
        for a in range(2,32):
            if pow(a,q-1,q)!=1: break
            if gcd(pow(a,(q-1)//ell,q)-1,q)==1:
                assert N*N>q
                return q,k,a
    raise RuntimeError('prime search exhausted')


def power(rows, ell=2):
    return tuple(sum(x*x for x in col) for col in zip(*(spectrum(r,ell) for r in rows)))


def sign_options(P, need):
    groups=[(v,len(list(g))) for v,g in groupby(P)]
    tail=[0]*(len(groups)+1)
    for i in reversed(range(len(groups))):
        v,k=groups[i];tail[i]=tail[i+1]+v*k
    def go(i,target,signs):
        if abs(target)>tail[i]:return
        if i==len(groups):
            if target==0:yield signs
            return
        v,k=groups[i]
        for neg in (range(k+1) if v else (0,)):
            yield from go(i+1,target-v*(k-2*neg),signs+(-1,)*neg+(1,)*(k-neg))
    yield from go(0,need,())


def invert_power(y,k,ell=2,cap=100000):
    """Exact prefix multiset DP. Cap counts accepted transitions, not all nodes."""
    n=len(y)-1;assert n>=1 and k>=1
    states={(1,)*k:((1,),)*k};t=1;work=0;peak=1
    for j in range(n):
        nxt={}
        for P,rows in states.items():
            z=sum(p*p for p in P)+k*t*t-y[j]
            if z%(2*t):continue
            for signs in sign_options(P,z//(2*t)):
                if work>=cap:return 'CAP',None,work,peak
                work+=1
                pp=[]
                for p,r,x in zip(P,rows,signs):
                    v=p+(ell-1)*t*x;rr=r+(x,)
                    if v<0:v=-v;rr=tuple(-a for a in rr)
                    pp.append((v,rr))
                pp.sort();key=tuple(x[0] for x in pp)
                nxt.setdefault(key,tuple(x[1] for x in pp))
        states=nxt;peak=max(peak,len(states));t*=ell
        if not states:return 'EMPTY',None,work,peak
    for P,r in states.items():
        if sum(p*p for p in P)==y[-1]:
            assert power(r,ell)==tuple(y)
            return 'HIT',r,work,peak
    return 'EMPTY',None,work,peak
