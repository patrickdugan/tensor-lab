"""Coherent QFT on the subspace invariant under units congruent to 1 mod 4.
For N=2^n, n=2^b-1, the character flag has two sectors: trivial and chi_4.
Flag=1 uses absolute shell j>=2; flag=1,j=0,1 are unused states.
"""
import numpy as np
from recursive import compile_shell, counts
from hierarchical import embedding


def add_control(gates,flag,scratch):
    """Add one positive control to the abstract shell compiler gate set."""
    out=[]
    for g in gates:
        k=g[0]
        if k=='x':out.append(('cx',flag,g[1]))
        elif k=='cx':out.append(('ccx',flag,g[1],g[2]))
        elif k=='ry':out.append(('cry',flag,g[1],g[2]))
        elif k in ('cry','ccx'):
            c=g[1];compute=('ccx',flag,c,scratch)
            middle=('cry',scratch,g[2],g[3]) if k=='cry' else ('ccx',scratch,g[2],g[3])
            out += [compute,middle,compute]
        else:raise ValueError(g)
    return out


def shift(g):
    if g[0] in ('ry','cry'):
        return (g[0],*(i+1 for i in g[1:-1]),g[-1])
    return (g[0],*(i+1 for i in g[1:]))


def controlled_increment(wires,flag,anc):
    """Increment a binary register mod 2^h, conditioned on flag.
    Prefix carry chain is uncomputed before its source bits are changed.
    """
    h=len(wires);assert len(anc)>=max(0,h-1)
    if not h:return []
    chain=[]
    for k in range(h-1):chain.append(('ccx',flag if k==0 else anc[k-1],wires[h-1-k],anc[k]))
    out=chain.copy()
    for k,t in enumerate(wires):
        j=h-2-k
        out.append(('cx',anc[j] if j>=0 else flag,t))
        if j>=0:out.append(chain[j])
    return out


def compile_mod4(b,mode='clean'):
    assert b>=2
    g,w=compile_shell(b,2,mode);width=2*b;temp=width-1
    shell=list(range(1,b+1));anc=list(range(b+1,2*b-1))
    assert len(anc)==b-2 and w-b<=len(anc)
    out=[('x',0)]+add_control([shift(a) for a in g],0,temp)+[('x',0)]
    # chi_4 has conductor 4 and normalized Gauss phase i.
    # The valid j=2,...,n reverse by n+2-j=1-j mod 2^b.
    out += [('cx',0,t) for t in shell]
    out += controlled_increment(shell[:-1],0,anc)
    out += [('p',0,np.pi/2)]
    return out,width


def reference(b):
    D=1<<b;n=D-1
    from radial_qft import radial_matrix
    Q=np.zeros((2*D,2*D),complex);Q[:D,:D]=radial_matrix(n,2)
    for j in range(D):Q[D+(1-j)%D,D+j]=1j
    return Q


def basis(b):
    D=1<<b;n=D-1;N=1<<n
    B=np.zeros((N,2*D),complex);B[:,:D]=embedding(n,2)
    for j in range(2,n+1):
        a=np.arange(1,1<<j,2);chi=np.where(a%4==1,1,-1)
        B[(1<<(n-j))*a,D+j]=chi/np.sqrt(len(a))
    valid=list(range(D))+list(range(D+2,2*D))
    return B[:,valid],valid
