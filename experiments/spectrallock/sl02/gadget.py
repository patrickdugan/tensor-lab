"""Educational R1CS for compressed convolution. No commitment, prover or setup."""
from math import gcd,isqrt
from cross_shell import product_shell


def system(n,ell,q):
    assert n>=1 and ell>=2 and all(ell%i for i in range(2,isqrt(ell)+1))
    assert q>=2 and all(q%i for i in range(2,isqrt(q)+1)) and gcd(ell,q)==1
    names=['one'];constraints=[]
    def var(s):names.append(s);return len(names)-1
    def lc(*terms):return tuple((i,a%q) for i,a in terms if a%q)
    def linear(*terms):constraints.append((lc(*terms),((0,1),),()))
    inp={z:[var(f'{z}{j}') for j in range(n+1)] for z in 'abc'}
    f={}
    for z in 'abc':
        prev=inp[z][0];t=1;freq=[]
        for j in range(1,n+1):
            P=var(f'{z}.P{j}');F=var(f'{z}.F{j}');x=inp[z][j]
            linear((P,1),(prev,-1),(x,-(ell-1)*t))
            linear((F,1),(prev,-1),(x,t))
            freq.append(F);prev=P;t=t*ell%q
        f[z]=freq+[prev]
    for x,y,z in zip(f['a'],f['b'],f['c']):constraints.append((((x,1),),((y,1),),((z,1),)))
    return names,constraints


def witness(a,b,c,ell,q):
    w=[1]+[v%q for x in (a,b,c) for v in x]
    for s in (a,b,c):
        p=s[0]%q;t=1
        for x in s[1:]:
            f=(p-t*x)%q;p=(p+(ell-1)*t*x)%q;w.extend((p,f));t=t*ell%q
    return w


def verify(constraints,w,q):
    if not w or w[0]!=1:return False
    def v(l):return sum(w[i]*a for i,a in l)%q
    return all((v(a)*v(b)-v(c))%q==0 for a,b,c in constraints)


def audit():
    import random,json
    from pathlib import Path
    r=random.Random(92101);total=0;bad=0;direct=0;res=[]
    for ell,n,q in [(2,3,257),(3,3,257),(5,3,257),(2,7,12289),(2,31,12289),(3,255,12289),(2,1023,12289)]:
        names,cs=system(n,ell,q);assert len(cs)==7*n+1 and len(names)==1+3*(n+1)+6*n
        for _ in range(16):
            a=[r.randrange(q) for _ in range(n+1)];b=[r.randrange(q) for _ in a];c=product_shell(a,b,ell,q)
            w=witness(a,b,c,ell,q);assert verify(cs,w,q);total+=1
            N=ell**n
            if N<=128:
                labels=[]
                for x in range(N):
                    j=n
                    while j and x%ell==0:x//=ell;j-=1
                    labels.append(j)
                for i in range(N):
                    assert c[labels[i]]==sum(a[labels[x]]*b[labels[(i-x)%N]] for x in range(N))%q;direct+=1
            # All coordinate mutations of one witness for small gadgets; 16 sampled otherwise.
            ids=range(len(w)) if n<=7 else r.sample(range(len(w)),16)
            for i in ids:
                v=w.copy();v[i]=(v[i]+1)%q
                # A/B edits can be valid in special cases, so test only constant,
                # output and aux wires. Other inputs are not fixed by this relation.
                if 1<=i<=2*(n+1):continue
                assert not verify(cs,v,q);bad+=1
        res.append(dict(ell=ell,n=n,q=q,linear_constraints=6*n,multiplication_constraints=n+1,total_constraints=len(cs),variables_including_one=len(names),schoolbook_products=(n+1)**2))
    out=dict(scope='R1CS_RELATION_ONLY',positive=total,mutations_rejected=bad,direct_convolution_coordinates=direct,settings=res)
    (Path(__file__).parent/'GADGET_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':audit()
