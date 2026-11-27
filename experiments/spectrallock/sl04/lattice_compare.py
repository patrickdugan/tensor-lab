"""Matched toy binary-secret module and dense LWR; no external inputs."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib,json,random,time
import numpy as np
from sympy import Matrix,Rational
import sl03_reference as lp
from module_lwr import Q,module_direct,module_ntt,dense_matrix,embed_subring,split_public,split_outputs,rounded
P=Path(__file__).parent


def instance(spec):
    n,k,B,root,mode=spec;d=n*k;seed=92900+10000*d+100*root+B
    rng=random.Random(seed);s=tuple(rng.choice((-1,1)) for _ in range(d));rng=random.Random(seed+1)
    subring=False
    if mode=='dense':
        A=[[rng.randrange(Q) for _ in range(d)] for _ in range(d)]
    else:
        ss=np.array(s).reshape(k,n)
        if k==2:
            big=np.array([rng.randrange(Q) for _ in range(k*2*n)]).reshape(1,k,2*n)
            a=split_public(big,2);wide=module_direct(big,embed_subring(ss,2))
            assert np.array_equal(split_outputs(wide,2),module_ntt(a,ss));subring=True
        else:a=np.array([rng.randrange(Q) for _ in range(k*k*n)]).reshape(k,k,n)
        A=dense_matrix(a);want=rounded(module_ntt(a,ss),Q,B).ravel()
        assert tuple(map(int,want))==lp.rounded(A,s,Q,B)
    return A,s,lp.rounded(A,s,Q,B),subring


def job(spec):
    n,k,B,root,mode=spec;d=n*k;A,s,y,subring=instance(spec)
    e2,w=lp.embed_check(A,y,s,Q,B);basis,_,_=lp.embedding(A,y,Q,B)
    status='MISS';found=None;checks=0;err=None;stage=None;t=time.process_time()
    try:
        R=[tuple(map(int,row)) for row in Matrix(basis).lll(delta=Rational(99,100)).tolist()]
        for v in R:
            checks+=1;found=lp.candidate(v,w,d,A,y,Q,B)
            if found is not None:stage='basis';break
        if found is None:
            for i,a in enumerate(R):
                for b in R[i+1:]:
                    for sign in (-1,1):
                        if abs(a[-1]+sign*b[-1])!=w:continue
                        checks+=1;found=lp.candidate(tuple(x+sign*z for x,z in zip(a,b)),w,d,A,y,Q,B)
                        if found is not None:stage='pair';break
                    if found is not None:break
                if found is not None:break
        if found is not None:status='HIT'
    except Exception as exc:status='ERROR';err=f'{type(exc).__name__}: {exc}'
    row=dict(n=n,k=k,d=d,m=d,B=B,q=Q,root=root,mode=mode,subring_equivalence=subring,status=status,original=found==s,stage=stage,checks=checks,error=err,cpu_seconds=time.process_time()-t,rank=lp.rank_mod(A,Q),error_norm2=e2,scale=w,instance_sha256=hashlib.sha256(json.dumps([A,y],separators=(',',':')).encode()).hexdigest(),recovered=found)
    print(n,k,B,root,mode,status,round(row['cpu_seconds'],2),flush=True);return row


def roster():
    return [(n,k,B,root,mode) for n,k,roots in ((8,2,3),(8,3,3),(16,2,2)) for B in (8,32) for root in range(roots) for mode in ('dense','module')]


def verify(rows):
    seen=set();hits=0;det=0
    assert len(rows)==len(roster())
    for row,spec in zip(rows,roster()):
        assert [row[x] for x in ('n','k','B','root','mode')]==list(spec)
        A,s,y,subring=instance(spec);assert subring==row['subring_equivalence']
        sha=hashlib.sha256(json.dumps([A,y],separators=(',',':')).encode()).hexdigest();assert sha==row['instance_sha256'] and sha not in seen;seen.add(sha)
        assert (int(Matrix(A).det())%Q!=0)==(row['rank']==row['d']);det+=1
        if row['status']=='HIT':
            v=tuple(row['recovered']);assert set(v)<= {-1,1} and len(v)==row['d'] and lp.rounded(A,v,Q,row['B'])==y and (v==s)==row['original'];hits+=1
        else:assert row['status'] in ('MISS','ERROR') and row['recovered'] is None
    return dict(instances=len(seen),verified_hits=hits,determinant_checks=det)


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4) as p:rows=list(p.map(job,roster()))
    result=dict(scope='TOY_BINARY_MODULE_LWR_LLL_ONLY',rows=rows,verification=verify(rows))
    (P/'LATTICE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['verification'],flush=True)
