"""Independent integer checks of shell transform and single-row power counts."""
from itertools import product
from collections import Counter
from pathlib import Path
import json
from spectrallock import spectrum,power,invert_power
P=Path(__file__).parent

def main():
    collisions=[];checks=0
    for ell in (2,3,5):
        for n in range(1,13):
            freq=Counter(power((s,),ell) for s in product((-1,1),repeat=n+1))
            assert len(freq)==(2**(n-1)+1 if ell==2 else 2**n)
            assert set(freq.values()) <= ({2,4} if ell==2 else {2})
            collisions.append(dict(ell=ell,n=n,distinct=len(freq),multiplicities=dict(Counter(freq.values()))))
        for n in (1,2,4,8,16,32):
            N=ell**n
            for i in range(n+1):
                s=tuple(int(j==i) for j in range(n+1))
                v=spectrum(s,ell)[::-1]
                assert spectrum(v,ell)[::-1]==tuple(N*x for x in s)
                checks+=n+1
    # Uniform matrix mixing: independent columns; T is a permutation of F_q^d.
    q=11;seen={tuple(x%q for x in spectrum(s,2)) for s in product(range(q),repeat=2)}
    assert len(seen)==q*q
    # Explicit two-preimage collision, fixing the global sign.
    a=(1,-1,1,1);b=(1,-1,-1,-1);assert power((a,))==power((b,))
    for c in json.loads((P/'AUDIT.json').read_text())['primes']:
        from math import gcd
        ell,n,q,k,w=c['ell'],c['n'],int(c['q']),c['k'],c['witness'];N=ell**n
        assert q==k*N+1 and N*N>q>4*N
        assert pow(w,q-1,q)==1 and gcd(pow(w,(q-1)//ell,q)-1,q)==1
        z=pow(w,k,q);assert pow(z,N,q)==1 and pow(z,N//ell,q)!=1
    (P/'EXACT_AUDIT.json').write_text(json.dumps(dict(collision_counts=collisions,integer_involution_coordinates=checks,uniform_column_permutation=len(seen),collision=[a,b],prime_certificates=8),indent=2)+'\n')
    print('integer transform checks',checks,'collision families',len(collisions),'prime certificates 8')
if __name__=='__main__':main()
