"""Exact toy coefficient-domain module LWR, no deployed cryptography."""
from functools import lru_cache
import numpy as np

Q = 12289


def mul(a,b,q=Q):
    a,b=np.asarray(a,dtype=np.int64),np.asarray(b,dtype=np.int64)
    n=len(a); assert len(b)==n and n*(q-1)**2 < 2**62
    c=np.convolve(a%q,b%q); out=c[:n].copy(); out[:-1]-=c[n:]
    return out%q


@lru_cache(None)
def tables(n,q):
    assert n>=1 and n&(n-1)==0 and (q-1)%(2*n)==0
    assert q*q<2**62
    psi=next(z for z in range(1,q) if pow(z,n,q)==q-1 and pow(z,2*n,q)==1)
    bits=n.bit_length()-1
    rev=np.array([int(f'{i:0{bits}b}'[::-1],2) if bits else 0 for i in range(n)])
    tw=[]
    for direction in (1,-1):
        stages=[]
        for width in (1<<j for j in range(1,bits+1)):
            stages.append(np.array([pow(psi,direction*2*n//width*j,q) for j in range(width//2)]))
        tw.append(stages)
    return rev,tw,np.array([pow(psi,i,q) for i in range(n)]),np.array([pow(psi,-i,q) for i in range(n)])


def ntt(a,q=Q,inverse=False):
    a=np.asarray(a,dtype=np.int64)%q; n=a.shape[-1]
    rev,tw,t,ti=tables(n,q)
    if not inverse:a=a*t%q
    z=a[...,rev].copy()
    for stage,w in enumerate(tw[int(inverse)]):
        width=2<<stage; v=z.reshape(*z.shape[:-1],n//width,width)
        lo=v[...,:width//2].copy(); hi=v[...,width//2:]*w%q
        v[...,:width//2]=(lo+hi)%q; v[...,width//2:]=(lo-hi)%q
    return z*pow(n,-1,q)%q*ti%q if inverse else z


def module_direct(A,s,q=Q):
    A,s=np.asarray(A),np.asarray(s); m,k,n=A.shape;assert s.shape==(k,n)
    out=np.zeros((m,n),dtype=np.int64)
    for i in range(m):
        for j in range(k):out[i]=(out[i]+mul(A[i,j],s[j],q))%q
    return out


def module_ntt(A,s,q=Q,transformed=False):
    A,s=np.asarray(A),np.asarray(s);m,k,n=A.shape;assert s.shape==(k,n)
    assert k*(q-1)**2<2**62
    Ah=A if transformed else ntt(A,q)
    return ntt((Ah*ntt(s,q)[None,:,:]).sum(axis=1)%q,q,True)


def rounded(v,q,B):
    assert 2<=B<q and B*q<2**62
    return B*(np.asarray(v,dtype=np.int64)%q)//q


def dense_matrix(A,q=Q):
    """Independent signed-circulant expansion, row order (sample,coefficient)."""
    m,k,n=A.shape; rows=[]
    for i in range(m):
        for u in range(n):
            rows.append([int(A[i,j,(u-v)%n])*(1 if u>=v else -1)%q
                         for j in range(k) for v in range(n)])
    return rows


def embed_subring(s,r):
    k,d=s.shape;out=np.zeros((k,r*d),dtype=np.int64);out[:,::r]=s;return out


def split_public(A,r):
    """A_j(X)=sum_h X^h A_{j,h}(X^r); flatten sample,coset."""
    m,k,n=A.shape;assert n%r==0
    return A.reshape(m,k,n//r,r).transpose(0,3,1,2).reshape(m*r,k,n//r)


def split_outputs(y,r):
    m,n=y.shape;return y.reshape(m,n//r,r).transpose(0,2,1).reshape(m*r,n//r)
