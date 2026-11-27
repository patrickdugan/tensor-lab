"""Deterministic CPU checks; no network, QPU, or training. Run with BLAS threads=1."""
import json,time
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from recursive import compile_shell,counts,simulate,expand_ccx,expand_cry,qasm
from radial_qft import compile_radial,radial_matrix
from hierarchical import *
from character_blocks import audit as character_audit
P=Path(__file__).parent
rng=np.random.default_rng(87751)

def record(g,w,b,mode,p,all_cols):
    d=1<<b;ids=np.arange(d)<<(w-b)
    if all_cols: x=np.eye(d,dtype=complex)
    else:
        x=rng.normal(size=(d,4))+1j*rng.normal(size=(d,4));x/=np.linalg.norm(x,axis=0)
    z=np.zeros((1<<w,x.shape[1]),complex);z[ids]=x
    y=simulate(g,w,z);want=radial_matrix(d-1,p)@x
    err=float(np.max(abs(y[ids]-want)));y[ids]=0;leak=float(np.linalg.norm(y))
    assert err<1e-11 and leak<1e-11
    return dict(b=b,p=p,mode=mode,wires=w,columns=x.shape[1],max_error=err,leakage_norm=leak,**counts(g))

def main():
    results=[]
    # Direct comparison with the old Gray-cycle compiler for every p in the sweep.
    for p in (2,3,5,7):
        for b in range(1,7):
            for mode in ('noanc','clean'):
                g,w=compile_shell(b,p,mode);results.append(record(g,w,b,mode,p,True))
    for b in (7,8,9):
        for mode in ('noanc','clean','hybrid'):
            g,w=compile_shell(b,2,mode);results.append(record(g,w,b,mode,2,b==7 and mode=='noanc'))
    # Elementary gate decompositions are independently checked on every basis direction.
    ccx=[('ccx',0,1,2)];cxerr=float(np.max(abs(simulate(ccx,3,np.eye(8))-simulate(expand_ccx(ccx),3,np.eye(8)))))
    cry=[('cry',0,1,.731)];cryerr=float(np.max(abs(simulate(cry,2,np.eye(4))-simulate(expand_cry(cry),2,np.eye(4)))))
    assert max(cxerr,cryerr)<1e-14
    g,w=compile_shell(4,3,'clean');results.append(record(expand_ccx(expand_cry(g)),w,4,'fully_decomposed',3,True))
    # Resource formulas and emitted circuits, without simulating 2^(2b-2) at large b.
    resources=[]
    for b in range(1,13):
        n=(1<<b)-1;row=dict(b=b,exponent=n,old_gray_cx=0 if b==1 else n*(1<<(b-1)),full_qft_cx=n*(n-1),full_walsh_step_cx=12*(n-1))
        for mode in ('noanc','clean','hybrid'):
            g,w=compile_shell(b,2,mode);c=counts(g);row[mode]=dict(wires=w,**c)
            if mode=='noanc':assert c['cx']==5*(1<<b)+b*b-5*b-6
            if mode=='clean':assert c['cx']==(b-1)*(23*b-38)
            if b in (3,4,5,7): (P/f'shell_b{b}_{mode}.qasm').write_text(qasm(g,w))
        resources.append(row)
    # Workload: all shell columns on small prime powers, dense generator for small N,
    # and six alternating potential/hopping steps from a complex radial state.
    apps=[]
    for p,b in ((2,1),(2,2),(2,3),(2,4),(3,1),(3,2),(5,1),(5,2)):
        n=(1<<b)-1;B=embedding(n,p);N=len(B);d=n+1;t=.417
        g,w=compressed_evolution(b,t,p,'noanc');assert w==b
        U=simulate(g,b,np.eye(d));ref=full_fft_evolution(B,n,t,p)
        err=float(np.max(abs(B@U-ref)));assert err<1e-11
        denseerr=None
        if N<=128:
            dense=expm(-1j*t*direct_generator(n,p))@B;denseerr=float(np.max(abs(dense-ref)));assert denseerr<1e-11
        z=rng.normal(size=d)+1j*rng.normal(size=d);z/=np.linalg.norm(z);f=B@z
        lab=valuation_shells(n,p)
        for j in range(6):
            v=.103*(j+1);z=U@(np.exp(-1j*v*np.arange(d))*z);f=full_fft_evolution(np.exp(-1j*v*lab)*f,n,t,p)
        repeated=float(np.max(abs(B@z-f)));assert repeated<1e-10
        directerr=None;leak=None
        if p==2 and n<=7:
            wg,ww=full_walsh_evolution(n,t);ids=np.arange(N)<<(ww-n);x=np.zeros((1<<ww,d),complex);x[ids]=B;y=simulate(wg,ww,x);directerr=float(np.max(abs(y[ids]-ref)));y[ids]=0;leak=float(np.linalg.norm(y));assert max(directerr,leak)<1e-11
        apps.append(dict(p=p,b=b,n=n,N=N,columns=d,fft_error=err,dense_generator_error=denseerr,repeated_error=repeated,full_walsh_error=directerr,full_walsh_leakage=leak))
        if p==2 and b==3:(P/'hierarchical_128.qasm').write_text(qasm(g,w))
    chars=[character_audit(p,n) for p,n in [(2,k) for k in range(1,9)]+[(3,k) for k in range(1,5)]+[(5,k) for k in range(1,4)]+[(7,k) for k in range(1,3)]]
    phase_tests=[]
    for p,n in ((2,3),(2,7),(3,3),(5,2),(7,2)):
        B=embedding(n,p);N=len(B);labels=valuation_shells(n,p);mass=np.bincount(labels);F=np.fft.ifft(np.eye(N),axis=0)*np.sqrt(N)
        retained=np.sum(abs(B.T@F)**2,axis=0);want=1/mass[labels];err=float(np.max(abs(retained-want)));assert err<1e-12
        phase_tests.append(dict(p=p,n=n,columns=N,max_error=err))
    from partial_symmetry import compile_mod4, reference, basis
    partial=[]
    for b in range(2,7):
        g,w=compile_mod4(b,'clean');D=1<<b;ids=np.arange(2*D)<<(w-b-1)
        x=np.zeros((1<<w,2*D),complex);x[ids,np.arange(2*D)]=1
        y=simulate(g,w,x);Q=reference(b);err=float(np.max(abs(y[ids]-Q)));y[ids]=0;leak=float(np.linalg.norm(y));assert max(err,leak)<1e-11
        ferr=None
        if b<=4:
            B,valid=basis(b);ferr=float(np.max(abs(np.fft.ifft(B,axis=0)*np.sqrt(len(B))-B@Q[np.ix_(valid,valid)])));assert ferr<1e-11
        partial.append(dict(b=b,n=D-1,valid_dimension=2*D-2,encoded_data=b+1,total_wires=w,columns=2*D,compiled_error=err,leakage=leak,fft_error=ferr,**counts(g)))
        if b in (3,4):(P/f'mod4_qft_b{b}.qasm').write_text(qasm(g,w))
    out=dict(partial_symmetry=partial,phase_estimation_limit=phase_tests,scope='IDEAL_CPU_CIRCUITS_AND_MATHEMATICAL_IDENTITIES',compiler=results,resources=resources,applications=apps,characters=chars,ccx_error=cxerr,cry_error=cryerr)
    (P/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print('compiled cases',len(results),'columns',sum(r['columns'] for r in results),'error',max(r['max_error'] for r in results),'leakage',max(r['leakage_norm'] for r in results),flush=True)
    print('applications',len(apps),'max fft',max(r['fft_error'] for r in apps),'repeated',max(r['repeated_error'] for r in apps),flush=True)
    print('character blocks',sum(r['nontrivial_sectors'] for r in chars),'columns',sum(r['columns'] for r in chars),'error',max(r['max_error'] for r in chars),flush=True)
    for r in resources:
        print('n',r['exponent'],'old',r['old_gray_cx'],'new',[r[m]['cx'] for m in ('noanc','clean','hybrid')], 'wires',[r[m]['wires'] for m in ('noanc','clean','hybrid')],flush=True)
if __name__=='__main__':main()
