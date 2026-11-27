"""Block-causal pilot, with small exhaustive existence controls."""
import json,random
from pathlib import Path
from itertools import product
from block import *
from spectrallock import prime_certificate
P=Path(__file__).parent

def main():
    rng=random.Random(91500);checks=0
    for k,n,ell,q in ((2,2,2,97),(2,2,3,73)):
        B=2;m=2
        for _ in range(5):
            A=[[[rng.randrange(q) for _ in range(k)] for _ in range(m)] for _ in range(n+1)]
            targets={block_forward(tuple(tuple(s[j*k:(j+1)*k]) for j in range(n+1)),A,q,B,ell) for s in product((-1,1),repeat=k*(n+1))}
            for z in product(range(B),repeat=m*(n+1)):
                y=tuple(tuple(z[j*m:(j+1)*m]) for j in range(n+1));status,*_=block_inverse(y,A,k,q,B,ell)
                assert (status=='HIT')==(y in targets) and status!='CAP';checks+=1
    rows=[]
    for ell in (2,3):
        for n in (63,127,255):
            q,*_=prime_certificate(ell,n)
            for k in (2,4):
                for root in range(16):
                    rng=random.Random(91600+root+n*100+k*100000+ell*1000000);m=k;B=8
                    s=tuple(tuple(rng.choice((-1,1)) for _ in range(k)) for _ in range(n+1))
                    A=[[[rng.randrange(q) for _ in range(k)] for _ in range(m)] for _ in range(n+1)]
                    y=block_forward(s,A,q,B,ell);st,w,work,peak=block_inverse(y,A,k,q,B,ell)
                    assert st!='EMPTY';rows.append(dict(ell=ell,n=n,k=k,m=m,root=root,bins=B,status=st,original=w==s,work=work,peak=peak))
    (P/'BLOCK_AUDIT.json').write_text(json.dumps(dict(scope='SYNTHETIC_BLOCK_CAUSAL_ROUNDING',exhaustive_checks=checks,rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
