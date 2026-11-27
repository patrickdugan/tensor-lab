"""Exact dyadic-shell collisions from an omitted spectral coordinate.
All instances are synthetic; collision resistance and planted preimage hardness differ.
"""
import json, random
from math import comb
from fractions import Fraction
from itertools import product
from pathlib import Path
import lattice_pilot as lp
HERE=Path(__file__).parent


def missing_probability(d,k,m):
    den=comb(d,k)
    # Inclusion-exclusion counts all ordered m-tuples of k-element supports.
    covered=sum((-1)**j*comb(d,j)*comb(d-j,k)**m for j in range(d-k+1))
    p=1-Fraction(covered,den**m)
    mu=Fraction(d)*Fraction(d-k,d)**m
    assert 0<=p<=1
    return p,mu


def pair(d,i,suffix=()):
    """Opposite sign prefixes, with only spectral coordinate i differing."""
    assert 0<=i<d
    if i==d-1:return (1,)*d,(-1,)*d
    a=(1,)*(i+1)+(-1,)+(tuple(suffix) if suffix else (1,)*(d-i-2))
    b=tuple(-x for x in a[:i+2])+a[i+2:]
    assert len(a)==len(b)==d
    return a,b


def audit():
    r=random.Random(92400);checks=0
    for d in (2,3,4,8,12,16,32,64,128,256):
        q=2**d*5+1 # Ring identity check, primality not needed.
        for i in range(d):
            a,b=pair(d,i,tuple(r.choice((-1,1)) for _ in range(max(0,d-i-2))))
            f,g=lp.spectrum(a,q),lp.spectrum(b,q)
            assert [j for j in range(d) if f[j]!=g[j]]==[i];checks+=d
    # Exact independent small support controls.
    control=0
    for d,k,m in ((3,1,3),(4,2,3),(5,2,3)):
        from itertools import combinations
        supp=list(combinations(range(d),k))
        n=sum(len(set().union(*map(set,x)))<d for x in product(supp,repeat=m))
        p,_=missing_probability(d,k,m);assert p==Fraction(n,len(supp)**m);control+=len(supp)**m
    probs=[]
    for d in (16,32,64,128,256,512,1024):
        for m in (d,2*d,3*d):
            p,mu=missing_probability(d,4,m)
            probs.append(dict(d=d,m=m,k=4,missing_probability=float(p),expected_missing=float(mu)))
    out=dict(scope='EXACT_COLLISION_CONSTRUCTION_AND_SUPPORT_PROBABILITY',spectral_coordinates_checked=checks,support_tuple_controls=control,probabilities=probs)
    (HERE/'COLLISION_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print('collision coordinates',checks,'support controls',control)
    for x in probs:print(x)

if __name__=='__main__':audit()
