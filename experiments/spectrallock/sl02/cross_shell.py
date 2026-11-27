"""Synthetic-only shell algebra and causal rounding experiments."""
from collections import defaultdict
from math import gcd


def spectrum(s, ell=2, q=None):
    P,t=s[0],1; out=[]
    for x in s[1:]:
        out.append(P-t*x)
        P+=(ell-1)*t*x;t*=ell
        if q is not None: P%=q;t%=q;out[-1]%=q
    return tuple(out+[P if q is None else P%q])


def bins(x,q,B):return B*(x%q)//q


def forward(s, rows, q, B, ell=2):
    f=spectrum(s,ell,q)
    return tuple(bins(sum(f[i]*a for i,a in row),q,B) for row in rows)


def invert(y, rows, q, B, ell=2, cap=200000):
    """Complete prefix enumeration, retaining full histories. CAP is unresolved."""
    n=len(rows)-1; assert n>=1 and len(y)==n+1
    due=defaultdict(list)
    for row,target in zip(rows,y):
        due[max(min(i+1,n) for i,_ in row)].append((row,target))
    states=[(-1,(-1,),()),(1,(1,),())];work=0;peak=2;t=1
    for j in range(1,n+1):
        nxt=[]
        for P,s,f in states:
            for x in (-1,1):
                if work>=cap:return 'CAP',None,work,peak
                work+=1
                ff=f+((P-t*x)%q,);pp=(P+(ell-1)*t*x)%q
                vals=ff+(pp,) if j==n else ff
                if all(bins(sum(vals[i]*a for i,a in row),q,B)==target for row,target in due[j]):
                    nxt.append((pp,s+(x,),ff))
        states=nxt;peak=max(peak,len(states));t=t*ell%q
        if not states:return 'EMPTY',None,work,peak
    s=states[0][1];assert forward(s,rows,q,B,ell)==tuple(y)
    return 'HIT',s,work,peak


def modular_diagonal(y,a,q,B,ell=2,cap=200000):
    """Finite-state quotient by prefix residue, for arbitrary q>=2."""
    n=len(y)-1;states={-1%q:(-1,),1%q:(1,)};work=0;peak=len(states);t=1%q
    for j in range(n):
        nxt={}
        for P,s in states.items():
            for x in (-1,1):
                if work>=cap:return 'CAP',None,work,peak
                work+=1
                if bins(a[j]*(P-t*x),q,B)==y[j]:
                    nxt.setdefault((P+(ell-1)*t*x)%q,s+(x,))
        states=nxt;peak=max(peak,len(states));t=t*ell%q
        if not states:return 'EMPTY',None,work,peak
    for P,s in states.items():
        if bins(a[-1]*P,q,B)==y[-1]:
            assert tuple(bins(x*z,q,B) for x,z in zip(spectrum(s,ell,q),a))==tuple(y)
            return 'HIT',s,work,peak
    return 'EMPTY',None,work,peak


def natural(s,ell,q):return spectrum(s,ell,q)[::-1]


def product_shell(a,b,ell,q):
    n=len(a)-1;N=pow(ell,n,q);assert gcd(N,q)==1 and len(a)==len(b)
    c=[x*y%q for x,y in zip(natural(a,ell,q),natural(b,ell,q))]
    return tuple(x*pow(N,-1,q)%q for x in natural(c,ell,q))
