"""CPU exact controls and descriptive evaluation timing, no secret or network input."""
import json,time,statistics,platform,sys
from pathlib import Path
import numpy as np
from module_lwr import *
P=Path(__file__).parent


def audit():
    rng=np.random.default_rng(92801);ncheck=0;cases=0;sub=0;subcases=0;dense=0;repeat=0;timings=[]
    for n in (1,2,4,8,16,32,64,128,256,512,1024,2048):
        for k,m in ((1,1),(2,2),(3,2),(4,3)):
            for _ in range(4):
                A=rng.integers(0,Q,(m,k,n));s=rng.choice([-1,1],(k,n));u=module_direct(A,s);v=module_ntt(A,s);w=module_ntt(ntt(A),s,transformed=True)
                assert np.array_equal(u,v) and np.array_equal(v,w);ncheck+=2*m*n;cases+=1
                assert np.array_equal(ntt(ntt(A),inverse=True),A);repeat+=m*k*n
                if n<=16:
                    M=np.array(dense_matrix(A),dtype=np.int64)
                    assert np.array_equal((M@s.ravel()%Q).reshape(m,n),u);dense+=m*n
                for B in (8,32):assert np.array_equal(rounded(u,Q,B),rounded(w,Q,B))
        if n in (8,16,32,64,128,256,512,1024):
            for r in (2,4,8):
                for _ in range(8):
                    k,m=2,2;A=rng.integers(0,Q,(m,k,n));s=rng.choice([-1,1],(k,n//r))
                    c=module_direct(A,embed_subring(s,r));b=split_public(A,r);d=module_ntt(b,s)
                    assert np.array_equal(split_outputs(c,r),d);sub+=m*n;subcases+=1
                    for B in (8,32):assert np.array_equal(split_outputs(rounded(c,Q,B),r),rounded(d,Q,B))
                    if n<=32:
                        # Compare the actual scalar coefficient matrices, after column restriction and row permutation.
                        M=np.array(dense_matrix(A));cols=[j*n+h*r for j in range(k) for h in range(n//r)]
                        perm=[i*n+h*r+t for i in range(m) for t in range(r) for h in range(n//r)]
                        assert np.array_equal(M[np.ix_(perm,cols)],np.array(dense_matrix(b)))
                        # If public A also lies in the subring, the omitted cosets are zero.
                        Az=np.zeros_like(A);Az[...,::r]=A[...,::r];cz=module_direct(Az,embed_subring(s,r))
                        assert all(not np.any(cz[:,t::r]) for t in range(1,r))
    for n in (32,64,128,256,512,1024,2048):
        k=m=3;A=rng.integers(0,Q,(m,k,n));s=rng.choice([-1,1],(k,n));Ah=ntt(A)
        funs={'direct':lambda:module_direct(A,s),'ntt_cold':lambda:module_ntt(A,s),'ntt_prepared':lambda:module_ntt(Ah,s,transformed=True)}
        row={'n':n,'k':k,'m':m,'repetitions':21}
        for name,f in funs.items():
            f();vals=[]
            for _ in range(21):t=time.perf_counter_ns();f();vals.append((time.perf_counter_ns()-t)/1e6)
            row[name+'_median_ms']=statistics.median(vals)
        timings.append(row)
    out=dict(scope='EXACT_ARITHMETIC_ONLY',settings=cases,ntt_coefficient_equalities=ncheck,ntt_roundtrip_coordinates=repeat,dense_matrix_coordinates=dense,subring_cases=subcases,subring_coordinate_equalities=sub,python=sys.version,numpy=np.__version__,machine=platform.machine(),timings=timings)
    (P/'ARITHMETIC_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':audit()
