"""Deterministic synthetic experiments; no credentials or network."""
import json,random,time
from itertools import product
from pathlib import Path
from collections import Counter
from statistics import median
from cross_shell import *
from sl01_reference import prime_certificate
P=Path(__file__).parent


def make_rows(n,rng,q,mode,w=0):
    rows=[]
    for i in range(n+1):
        j=min(n,i+w)
        if mode=='dense':ix=list(range(j+1))
        elif mode=='global':ix=sorted(rng.sample(range(n+1),min(4,n+1)))
        else:ix=sorted(set([j]+rng.sample(range(j),min(3,j))))
        rows.append(tuple((k,rng.randrange(q)) for k in ix))
    return rows


def controls():
    rng=random.Random(91801);cc=0;mm=0;ex=0;fib=0
    for ell in (2,3,5):
        for n in range(1,5):
            N=ell**n;labels=[]
            for x in range(N):
                j=n
                while j and x%ell==0:x//=ell;j-=1
                labels.append(j)
            for q in (8,11,25,49):
                if gcd(N,q)!=1:continue
                a=[rng.randrange(q) for _ in range(n+1)];b=[rng.randrange(q) for _ in a]
                c=product_shell(a,b,ell,q)
                for x in range(N):
                    assert sum(a[labels[i]]*b[labels[(x-i)%N]] for i in range(N))%q==c[labels[x]];cc+=1
                assert natural(natural(a,ell,q),ell,q)==tuple(N*x%q for x in a);mm+=n+1
            for j in range(1,n+1):
                count=Counter(spectrum(s,ell)[j-1] for s in product((-1,1),repeat=j+1))
                assert max(count.values())<=2;fib+=len(count)
    for ell in (2,3):
        n=3;q=97;B=3
        for mode,w in [('dense',0),('sparse',0),('sparse',2),('global',0)]:
            rows=make_rows(n,rng,q,mode,w)
            targets={forward(s,rows,q,B,ell) for s in product((-1,1),repeat=n+1)}
            for y in product(range(B),repeat=n+1):
                st,*_=invert(y,rows,q,B,ell);assert (st=='HIT')==(y in targets) and st!='CAP';ex+=1
        for q in (3,4,5,7):
            a=[rng.randrange(q) for _ in range(n+1)];B=2
            targets={tuple(bins(x*z,q,B) for x,z in zip(spectrum(s,ell,q),a)) for s in product((-1,1),repeat=n+1)}
            for y in product(range(B),repeat=n+1):
                st,*_=modular_diagonal(y,a,q,B,ell);assert (st=='HIT')==(y in targets) and st!='CAP';ex+=1
    return dict(convolution_coordinates=cc,involution_coordinates=mm,exhaustive_targets=ex,fourier_fibres=fib)


def main():
    start=time.process_time();c=controls();print('controls',c,flush=True);records=[];small=[]
    for ell in (2,3):
        for n in (15,63,255):
            q,*_=prime_certificate(ell,n)
            for mode,w in [('dense',0),('sparse',0),('sparse',4),('sparse',8),('sparse',12),('global',0)]:
                for root in range(8):
                    rng=random.Random(91900+root+100*n+100000*ell+1000000*(['dense','sparse','global'].index(mode))+10000*w)
                    s=tuple(rng.choice((-1,1)) for _ in range(n+1));rows=make_rows(n,rng,q,mode,w);y=forward(s,rows,q,8,ell)
                    st,v,work,peak=invert(y,rows,q,8,ell,cap=60000)
                    assert st!='EMPTY';records.append(dict(ell=ell,n=n,mode=mode,w=w,root=root,status=st,work=work,peak=peak,original=v==s))
            print('cross',ell,n,flush=True)
    for ell in (2,3):
        for q in (257,12289):
            for n in (63,255,1023):
                for root in range(8):
                    rng=random.Random(92000+root+100*n+100000*ell+q);s=tuple(rng.choice((-1,1)) for _ in range(n+1));a=[rng.randrange(1,q) for _ in s];B=8
                    y=tuple(bins(x*z,q,B) for x,z in zip(spectrum(s,ell,q),a))
                    st,v,work,peak=modular_diagonal(y,a,q,B,ell)
                    assert st=='HIT' and peak<=q;small.append(dict(ell=ell,q=q,n=n,root=root,status=st,work=work,peak=peak,original=v==s))
    result=dict(scope='SYNTHETIC_CAUSAL_SHELL_RESEARCH',controls=c,records=records,fixed_modulus=small)
    (P/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    for mode,w in [('dense',0),('sparse',0),('sparse',4),('sparse',8),('sparse',12),('global',0)]:
        a=[r for r in records if (r['mode'],r['w'])==(mode,w)];b=[r for r in a if r['n']==255]
        print(mode,w,Counter(r['status'] for r in a),'n255 median/max',median(r['work'] for r in b),max(r['work'] for r in b),'peak',max(r['peak'] for r in b))
    print('fixed',len(small),'max1024',max(r['work'] for r in small if r['n']==1023),'originals',sum(r['original'] for r in small),'cpu',time.process_time()-start)
if __name__=='__main__':main()
