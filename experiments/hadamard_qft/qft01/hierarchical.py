"""Coherent nested-subgroup hopping, both compressed and full Walsh circuits."""
import numpy as np
from recursive import compile_shell


def valuation_shells(n,p=2):
    N=p**n
    labels=np.zeros(N,dtype=int)
    for j in range(1,n+1):
        a=np.arange(1,p**j);a=a[a%p!=0]
        labels[p**(n-j)*a]=j
    return labels


def embedding(n,p=2):
    labels=valuation_shells(n,p);w=np.bincount(labels,minlength=n+1)
    B=np.zeros((len(labels),n+1));B[np.arange(len(labels)),labels]=1/np.sqrt(w[labels])
    return B


def phase_shell(b,t):
    return [('p',i,-t*(1<<(b-1-i))) for i in range(b)]


def compressed_evolution(b,t,p=2,mode='noanc'):
    g,w=compile_shell(b,p,mode)
    return g+phase_shell(b,t)+g,w


def full_walsh_evolution(n,t):
    """Full n-qubit register plus n-1 clean ancillas for prefix predicates.
    Diagonalizes nested subgroup averages by tensor-product Hadamards.
    Counts include uncomputation; global phase is kept for coherent use.
    """
    assert n>=1
    a=list(range(n,2*n-1));g=[('h',i) for i in range(n)]
    g += [('phase',-t*n)]+[('x',i) for i in range(n)]+[('p',0,t)]
    chain=[]
    for i in range(1,n):
        gate=('ccx',0 if i==1 else a[i-2],i,a[i-1])
        chain.append(gate);g += [gate,('p',a[i-1],t)]
    g+=chain[::-1]+[('x',i) for i in range(n)]+[('h',i) for i in range(n)]
    return g,2*n-1


def full_fft_evolution(a,n,t,p=2):
    k=valuation_shells(n,p)
    a=np.asarray(a);ph=np.exp(-1j*t*k).reshape((len(k),)+(1,)*(a.ndim-1))
    return np.fft.ifft(ph*np.fft.fft(a,axis=0),axis=0)


def direct_generator(n,p=2):
    """Independent dense matrix H=sum_{j=1}^n(I-A_j), small controls only."""
    N=p**n;x=np.arange(N);d=x[:,None]-x[None,:]
    H=n*np.eye(N)
    for j in range(1,n+1): H-=(d%(p**(n-j))==0)/p**j
    return H
