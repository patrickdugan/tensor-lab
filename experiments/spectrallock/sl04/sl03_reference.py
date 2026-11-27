"""Synthetic, planted binary LWR only. Exact verification of every recovered preimage."""
import argparse, hashlib, json, random, time
from itertools import product
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from sympy import Matrix, Rational, isprime
HERE=Path(__file__).parent


def spectrum(s,q):
    p,t=s[0],1;f=[]
    for x in s[1:]:
        f.append((p-t*x)%q);p=(p+t*x)%q;t=2*t%q
    return f+[p%q]


def shell_matrix(d,q):
    return [list(r) for r in zip(*(spectrum([int(i==j) for i in range(d)],q) for j in range(d)))]


def rounded(A,s,q,B):
    return tuple(B*(sum(a*b for a,b in zip(row,s))%q)//q for row in A)


def matrices(d,m,q,seed,mode):
    r=random.Random(seed)
    if mode=='dense':return [[r.randrange(q) for _ in range(d)] for _ in range(m)],None
    T=shell_matrix(d,q);S=[]
    if mode=='regular':
        assert m==d and d>=4
        # Four disjoint permutation supports, every coordinate appears four times.
        used=[set() for _ in range(d)]
        for _ in range(4):
            while True:
                perm=list(range(d));r.shuffle(perm)
                if all(j not in used[i] for i,j in enumerate(perm)):break
            for i,j in enumerate(perm):used[i].add(j)
        S=[[r.randrange(1,q) if j in used[i] else 0 for j in range(d)] for i in range(d)]
        assert all(sum(bool(row[j]) for row in S)==4 for j in range(d))
    else:
        assert mode=='global'
        for _ in range(m):
            row=[0]*d
            for k in r.sample(range(d),4):row[k]=r.randrange(q)
            S.append(row)
    A=[[sum(a*T[k][j] for k,a in enumerate(row) if a)%q for j in range(d)] for row in S]
    return A,S


def rank_mod(A,q):
    a=[r.copy() for r in A];m,d=len(a),len(a[0]);k=0
    for j in range(d):
        pivot=next((i for i in range(k,m) if a[i][j]%q),None)
        if pivot is None:continue
        a[k],a[pivot]=a[pivot],a[k]
        inv=pow(a[k][j]%q,-1,q);a[k]=[v*inv%q for v in a[k]]
        for i in range(k+1,m):
            c=a[i][j]%q
            if c:a[i]=[(u-c*v)%q for u,v in zip(a[i],a[k])]
        k+=1
        if k==m:break
    return k


def center(y,q,B):
    lo=(q*y+B-1)//B;hi=(q*(y+1)+B-1)//B-1
    assert lo<=hi
    return (lo+hi)//2


def embedding(A,y,q,B):
    """Rows [2w I,2 A^T,0], [0,q I,0], [w 1,t+A1,w]."""
    m,d=len(A),len(A[0]);w=max(1,q//(3*B));mid=[center(v,q,B) for v in y]
    rows=[[2*w*int(i==j) for j in range(d)]+[2*a[i]%q for a in A]+[0] for i in range(d)]
    rows += [[0]*d+[q*int(i==j) for j in range(m)]+[0] for i in range(m)]
    rows += [[w]*d+[(t+sum(a))%q for t,a in zip(mid,A)]+[w]]
    return rows,w,mid


def candidate(v,w,d,A,y,q,B):
    if abs(v[-1])!=w or any(abs(x)!=w for x in v[:d]):return None
    s=tuple(-int(v[-1]//w)*int(x//w) for x in v[:d])
    return s if rounded(A,s,q,B)==tuple(y) else None


def embed_check(A,y,s,q,B):
    rows,w,mid=embedding(A,y,q,B);d=len(s);m=len(y);x=[(v+1)//2 for v in s]
    vec=[sum(k*row[j] for k,row in zip(x+[0]*m+[-1],rows)) for j in range(d+m+1)]
    e=[sum(a*b for a,b in zip(row,s))%q-t for row,t in zip(A,mid)]
    assert vec[:d]==[w*v for v in s] and vec[-1]==-w
    assert all((v-t)%q==0 for v,t in zip(vec[d:-1],e))
    assert all(abs(t)<=((q+B-1)//B)//2 for t in e)
    assert candidate([w*v for v in s]+e+[-w],w,d,A,y,q,B)==tuple(s)
    return sum(t*t for t in e),w


def job(spec):
    d,q,B,root,mode,regime,delta=spec;m=d
    seed=92300+1000*d+100*root+(0 if regime=='fixed' else 1000000)
    r=random.Random(seed);s=tuple(r.choice((-1,1)) for _ in range(d))
    A,S=matrices(d,m,q,seed+1,mode);y=rounded(A,s,q,B);e2,w=embed_check(A,y,s,q,B)
    rank=rank_mod(A,q);coverage=None if S is None else sum(any(row[i] for row in S) for i in range(d))
    if S is not None:
        assert rank==rank_mod(S,q) and rank<=coverage and y==rounded(S,spectrum(s,q),q,B)
    basis,_,_=embedding(A,y,q,B);t=time.process_time();status='MISS';rec=None;checks=0;stage=None;err=None
    try:
        R=[tuple(map(int,row)) for row in Matrix(basis).lll(delta=Rational(delta,100)).tolist()]
        for v in R:
            checks+=1;rec=candidate(v,w,d,A,y,q,B)
            if rec is not None:stage='basis';break
        if rec is None:
            for i,a in enumerate(R):
                for b in R[i+1:]:
                    for sign in (-1,1):
                        if abs(a[-1]+sign*b[-1])!=w:continue
                        checks+=1;rec=candidate(tuple(x+sign*z for x,z in zip(a,b)),w,d,A,y,q,B)
                        if rec is not None:stage='pair';break
                    if rec is not None:break
                if rec is not None:break
        if rec is not None:status='HIT'
    except Exception as exc:
        status='ERROR';err=f'{type(exc).__name__}: {exc}'
    return dict(d=d,m=m,q=q,B=B,root=root,mode=mode,regime=regime,delta=delta/100,rank=rank,covered_frequencies=coverage,status=status,original=rec==s,stage=stage,candidate_checks=checks,cpu_seconds=time.process_time()-t,error=err,planted_error_norm2=e2,secret_scale=w,instance_sha256=hashlib.sha256(json.dumps([A,y],separators=(',',':')).encode()).hexdigest(),recovered=list(rec) if rec is not None else None)


def controls():
    r=random.Random(92301);checks=0
    for d in (3,4,6):
        for q in (17,97,257):
            assert rank_mod(shell_matrix(d,q),q)==d
            for _ in range(16):
                A=[[r.randrange(q) for _ in range(d)] for _ in range(d+1)];s=tuple(r.choice((-1,1)) for _ in range(d))
                embed_check(A,rounded(A,s,q,4),s,q,4);checks+=1
            A=[[r.randrange(q) for _ in range(d)] for _ in range(d)]
            for s in product((-1,1),repeat=d):
                embed_check(A,rounded(A,s,q,4),s,q,4);checks+=1
    return checks


def prime_large(d):
    N=2**(d-1)
    return next(k*N+1 for k in range(5,10000) if isprime(k*N+1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--d',type=int,required=True);p.add_argument('--delta',type=int,default=75);p.add_argument('--roots',type=int,default=3);p.add_argument('--regime',choices=['fixed','large','both'],default='both');a=p.parse_args()
    nchecks=controls();print('embedding controls',nchecks,flush=True)
    specs=[(a.d,12289 if regime=='fixed' else prime_large(a.d),B,root,mode,regime,a.delta) for regime in ('fixed','large') if a.regime in ('both',regime) for B in (8,32) for root in range(a.roots) for mode in ('dense','global')]
    with ProcessPoolExecutor(max_workers=4) as pool:records=list(pool.map(job,specs))
    out=HERE/f'AUDIT_d{a.d}_{a.regime}_delta{a.delta}.json'
    out.write_text(json.dumps(dict(scope='PLANTED_TOY_LWR_LLL_ONLY',controls=nchecks,records=records),indent=2)+'\n')
    for r in records:print(r['d'],r['regime'],r['B'],r['root'],r['mode'],r['status'],round(r['cpu_seconds'],3),flush=True)

if __name__=='__main__':main()
