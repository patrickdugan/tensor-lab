"""Seeded toy campaigns and independent small exhaustive checks."""
import json, random, time
from itertools import product, combinations_with_replacement
from pathlib import Path
import numpy as np
from spectrallock import *
P=Path(__file__).parent

def audit():
    fft=0;max_error=0.
    for ell in (2,3,5):
        for n in (1,2,3,4):
            N=ell**n; labels=[]
            for x in range(N):
                j=n
                while x%ell==0 and j>0: x//=ell;j-=1
                labels.append(j)
            for s in product((-1,1),repeat=n+1):
                a=np.asarray(s)[labels]; f=np.fft.fft(a)
                ref=spectrum(s,ell)
                freq=[f[ell**i] for i in range(n)]+[f[0]]
                err=float(np.max(abs(np.array(freq)-ref)));assert err<1e-9
                max_error=max(max_error,err);fft+=len(ref)
    rng=random.Random(91200); rounded_checks=0; power_checks=0
    for ell,n,q in ((2,3,97),(3,2,73),(5,2,151)):
        ss=list(product((-1,1),repeat=n+1))
        for _ in range(5):
            a=[rng.randrange(1,q) for _ in range(n+1)];bins=4
            targets={rounded(s,a,q,bins,ell) for s in ss}
            for y in product(range(bins),repeat=n+1):
                status,w,*_=invert_rounded(y,a,q,bins,ell)
                assert (status=='HIT')==(y in targets) and status!='CAP'
                rounded_checks+=1
        for k in (1,2,3):
            targets={power(rows,ell) for rows in combinations_with_replacement(ss,k)}
            for y in targets:
                assert invert_power(y,k,ell)[0]=='HIT';power_checks+=1
            for _ in range(40):
                y=list(rng.choice(sorted(targets)));j=rng.randrange(n+1);y[j]+=rng.randrange(1,10)
                status,*_=invert_power(y,k,ell)
                assert (status=='HIT')==(tuple(y) in targets);power_checks+=1
    return dict(fft_coefficients=fft,max_fft_error=max_error,rounded_exhaustive_targets=rounded_checks,power_exhaustive_or_corrupted_targets=power_checks)

def main():
    controls=audit(); print('controls',controls,flush=True)
    out=[];primes=[];powers=[]
    for ell in (2,3):
        for n in (15,63,127,255):
            q,k,a=prime_certificate(ell,n);primes.append(dict(ell=ell,n=n,q=str(q),k=k,witness=a))
            for bins in (4,8,16):
                for root in range(64):
                    rng=random.Random(91300+root+1000*n+1000000*ell+100*bins)
                    s=tuple(rng.choice((-1,1)) for _ in range(n+1));a=[rng.randrange(1,q) for _ in s]
                    y=rounded(s,a,q,bins,ell);status,w,work,peak=invert_rounded(y,a,q,bins,ell)
                    assert status!='EMPTY'
                    out.append(dict(ell=ell,n=n,bins=bins,root=root,status=status,original=w==s,work=work,peak=peak))
        print('rounded done',ell,len(out),flush=True)
    for ell in (2,3):
        for k in (1,4,8,12):
            for n in (15,31,63):
                for root in range(8):
                    rng=random.Random(91400+root+100*n+10000*k+1000000*ell)
                    s=tuple(tuple(rng.choice((-1,1)) for _ in range(n+1)) for _ in range(k));y=power(s,ell)
                    status,w,work,peak=invert_power(y,k,ell,cap=5000)
                    assert status!='EMPTY'
                    powers.append(dict(ell=ell,n=n,k=k,root=root,status=status,work=work,peak=peak))
                print('power done',ell,k,n,flush=True)
    report=dict(scope='SYNTHETIC_PREIMAGE_TESTS_NOT_SECURITY_PARAMETERS',controls=controls,primes=primes,rounded=out,power=powers)
    (P/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
