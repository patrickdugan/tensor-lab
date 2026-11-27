"""Check the conductor-resolved nontrivial multiplicative-character QFT blocks.
Exact character exponents are used to classify conductors; complex values and
FFT are numerical. Intended for small, fully enumerated prime powers.
"""
from math import sqrt
import numpy as np


def unit_characters(p,n):
    N=p**n;units=[a for a in range(N) if a%p]
    if p==2 and n>=3:
        order=2**(n-2);logs={}
        for s in range(2):
            for k in range(order): logs[(-1 if s else 1)*pow(5,k,N)%N]=(s,k)
        assert len(logs)==len(units)
        for a in range(2):
            for b in range(order):
                yield order,{u:(a*s*(order//2)+b*k)%order for u,(s,k) in logs.items()}
    else:
        order=len(units)
        g=next(a for a in units if len({pow(a,k,N) for k in range(order)})==order)
        logs={pow(g,k,N):k for k in range(order)}
        for a in range(order):yield order,{u:(a*k)%order for u,k in logs.items()}


def conductor(p,n,exponents):
    for c in range(n+1):
        table={};ok=True
        for a,x in exponents.items():
            k=a%(p**c)
            if k in table and table[k]!=x:ok=False;break
            table[k]=x
        if ok:return c,table
    raise AssertionError('no conductor')


def audit(p,n):
    N=p**n;cols=[];checks=0;maxerr=0.;phaseerr=0.;sectors=0
    for order,exps in unit_characters(p,n):
        c,table=conductor(p,n,exps)
        def chi(a):return np.exp(2j*np.pi*table[a%(p**c)]/order)
        if not c:
            for j in range(n+1):
                v=np.zeros(N,complex)
                if j==0:v[0]=1
                else:
                    u=np.array([a for a in range(1,p**j) if a%p]);v[p**(n-j)*u]=1/sqrt(len(u))
                cols.append(v)
            continue
        sectors+=1
        tau=sum(chi(a)*np.exp(2j*np.pi*a/(p**c)) for a in range(1,p**c) if a%p)/sqrt(p**c)
        phaseerr=max(phaseerr,abs(abs(tau)-1))
        for j in range(c,n+1):
            u=np.array([a for a in range(1,p**j) if a%p]);v=np.zeros(N,complex)
            v[p**(n-j)*u]=[chi(a)/sqrt(len(u)) for a in u];cols.append(v)
            l=n+c-j;b=np.array([a for a in range(1,p**l) if a%p]);want=np.zeros(N,complex)
            want[p**(n-l)*b]=[tau*np.conjugate(chi(a))/sqrt(len(b)) for a in b]
            got=np.fft.ifft(v)*sqrt(N);err=float(np.max(abs(got-want)));maxerr=max(maxerr,err);checks+=1
            assert err<1e-10
    B=np.stack(cols,axis=1);assert B.shape==(N,N)
    orth=float(np.max(abs(B.conj().T@B-np.eye(N))));assert orth<1e-10
    return dict(p=p,n=n,N=N,nontrivial_sectors=sectors,columns=checks,max_error=maxerr,phase_error=phaseerr,orthogonality=orth)
