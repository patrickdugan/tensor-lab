"""Deterministic synthetic tests and scaling receipts; no QPU, no hardware claims."""
import json, time, random
import numpy as np
from qc25_channels import P, Channel, rotation, stochastic, thermal, mpo_z, dense_z

def circuit(n,depth,seed):
    rng=random.Random(seed); out=[]
    for step in range(depth):
        for q in range(n):
            ang=.55+.3*(rng.random()-.5)+(.08 if q%7==1 else -.09 if q%7==5 else 0.)
            out.extend([Channel(q,(rotation(P[1],ang),),'RX'),Channel(q,(rotation(P[3],.15*(rng.random()-.5)),),'RZ'),thermal(q,1e-6,80e-6+10e-6*q,60e-6+5e-6*q)])
        for q in range(step%2,n-1,2):
            out.append(Channel(q,(rotation(np.kron(P[3],P[3]),.7+.3*(rng.random()-.5)),),'RZZ'))
            out.append(stochastic(q,2,{'XI':.001,'IZ':.002,'ZZ':.0005}))
    return out

def run():
    # Analytical nonunital test: X|0> then amplitude damping gives 1-2 exp(-t/T1).
    c=[Channel(0,(P[1],)),thermal(0,1.,2.,3.)]; z=1-2*np.exp(-.5)
    assert abs(mpo_z(1,c,0)['z']-z)<1e-12
    assert abs(dense_z(1,c,0)-z)<1e-12
    checks=[]
    for n in (2,4,6):
        for seed in (41,43):
            c=circuit(n,4,seed)
            for target in (0,n//2):
                truth=dense_z(n,c,target,(.01,.025))
                for cap in (1,2,4,8,16,64):
                    r=mpo_z(n,c,target,cap,(.01,.025)); err=abs(r['z']-truth)
                    assert err<=r['truncation_bound_exact_arithmetic']+1e-10
                    if cap==64: assert err<1e-10
                    checks.append(dict(n=n,seed=seed,target=target,cap=cap,dense=truth,error=err,**r))
    scale=[]
    for n,d in ((12,4),(16,6),(24,8)):
        c=circuit(n,d,41); prev=None
        for cap in ((4,8,16,32,48,64) if n==24 else (4,8,16,32)):
            start=time.perf_counter(); r=mpo_z(n,c,n//2,cap,(.01,.025))
            scale.append(dict(n=n,depth=d,cap=cap,seconds=time.perf_counter()-start,gap_previous=None if prev is None else abs(r['z']-prev),**r));prev=r['z']
    return dict(label='SYNTHETIC NOT HARDWARE',checks=checks,scale=scale)

if __name__=='__main__':
    import sys
    r=run(); print(json.dumps(r,indent=2))
    if len(sys.argv)>1:
        with open(sys.argv[1],'w') as f: json.dump(r,f,indent=2)
