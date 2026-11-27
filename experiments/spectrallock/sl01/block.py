"""Block-causal rounded shell experiment. Dense mixing within each shell only."""
from itertools import product


def observe(v, a, q, bins):
    return tuple(bins*(sum(x*y for x,y in zip(v,r))%q)//q for r in a)


def block_forward(s, matrices, q, bins, ell=2):
    P=tuple(s[0]);t=1;y=[]
    for x,A in zip(s[1:],matrices[:-1]):
        y.append(observe(tuple(p-t*z for p,z in zip(P,x)),A,q,bins))
        P=tuple(p+(ell-1)*t*z for p,z in zip(P,x));t*=ell
    return (*y,observe(P,matrices[-1],q,bins))


def block_inverse(y, matrices, k, q, bins, ell=2, cap=200000):
    """Search any preimage, merging equal prefix vectors. CAP is unresolved."""
    choices=tuple(product((-1,1),repeat=k));n=len(y)-1
    assert q>4*ell**n and len(matrices)==len(y)
    states={s:(s,) for s in choices};peak=len(states);work=0;t=1
    for target,A in zip(y[:-1],matrices[:-1]):
        nxt={}
        for P,path in states.items():
            for x in choices:
                if work>=cap:return 'CAP',None,work,peak
                work+=1
                v=tuple(p-t*z for p,z in zip(P,x))
                if observe(v,A,q,bins)==target:
                    nxt.setdefault(tuple(p+(ell-1)*t*z for p,z in zip(P,x)),path+(x,))
        states=nxt;peak=max(peak,len(states));t*=ell
        if not states:return 'EMPTY',None,work,peak
    for P,s in states.items():
        if observe(P,matrices[-1],q,bins)==y[-1]:
            assert block_forward(s,matrices,q,bins,ell)==y
            return 'HIT',s,work,peak
    return 'EMPTY',None,work,peak
