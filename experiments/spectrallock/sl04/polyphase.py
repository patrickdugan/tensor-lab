"""Full-sector reindexing of module-LWR, without changing its distribution."""
import json
from pathlib import Path
import numpy as np
from module_lwr import Q,module_direct,module_ntt,dense_matrix,split_public,split_outputs,rounded
P=Path(__file__).parent


def matrix(A,r,q=Q):
    """Restrict scalars from R_N to R_d for N=r*d, retaining all r sectors."""
    m,k,n=A.shape;assert n%r==0;d=n//r
    blocks=A.reshape(m,k,d,r).transpose(0,1,3,2)
    out=np.empty((m,r,k,r,d),dtype=np.int64)
    for h in range(r):
        for g in range(r):
            a=blocks[:,:,(h-g)%r,:]
            if h<g:
                b=np.roll(a,1,axis=-1);b[...,0]=-b[...,0];a=b
            out[:,h,:,g,:]=a%q
    return out.reshape(m*r,k*r,d)


def split_secret(s,r):
    k,n=s.shape;return s.reshape(k,n//r,r).transpose(0,2,1).reshape(k*r,n//r)


def audit():
    rng=np.random.default_rng(93001);cases=0;coef=0;scalar=0;norm=0
    for n in (8,16,32,64,128):
        for r in (2,4,8):
            for k,m in ((1,1),(2,2),(3,2)):
                for _ in range(4):
                    A=rng.integers(0,Q,(m,k,n));s=rng.choice([-1,1],(k,n));b=matrix(A,r);t=split_secret(s,r)
                    expected=split_outputs(module_direct(A,s),r);got=module_ntt(b,t)
                    assert np.array_equal(got,expected);coef+=m*n;cases+=1
                    assert np.sum(s*s)==np.sum(t*t) and np.count_nonzero(s)==np.count_nonzero(t);norm+=1
                    for B in (8,32):assert np.array_equal(split_outputs(rounded(module_direct(A,s),Q,B),r),rounded(got,Q,B))
                    if n<=32:
                        M=np.array(dense_matrix(A));cols=[j*n+h*r+g for j in range(k) for g in range(r) for h in range(n//r)];rows=[i*n+h*r+g for i in range(m) for g in range(r) for h in range(n//r)]
                        M2=np.array(dense_matrix(b));assert np.array_equal(M[np.ix_(rows,cols)],M2);scalar+=M2.size
    out=dict(scope='FULL_SECRET_SECTOR_REINDEXING',cases=cases,coefficient_equalities=coef,scalar_matrix_equalities=scalar,norm_checks=norm)
    (P/'POLYPHASE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':audit()
