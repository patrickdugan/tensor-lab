"""Recover SL02 frozen synthetic inputs and verify applicable collision witnesses."""
import argparse, json, random, sys
from pathlib import Path
from collision_audit import pair
HERE=Path(__file__).parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.prior.resolve()))
    from run import make_rows
    from cross_shell import forward, spectrum, invert
    from sl01_reference import prime_certificate
    old=json.loads((a.prior/'AUDIT.json').read_text());out=[]
    for row in old['records']:
        if row['mode']!='global':continue
        ell,n,root=row['ell'],row['n'],row['root'];d=n+1;q,*_=prime_certificate(ell,n)
        r=random.Random(91900+root+100*n+100000*ell+2000000)
        s=tuple(r.choice((-1,1)) for _ in range(d));A=make_rows(n,r,q,'global',0)
        y=forward(s,A,q,8,ell)
        # Verify frozen case identity via the exact prior algorithm, not merely seed recreation.
        st,w,work,peak=invert(y,A,q,8,ell,cap=60000)
        assert (st,work,peak,w==s)==(row['status'],row['work'],row['peak'],row['original'])
        missing=sorted(set(range(d))-{j for rr in A for j,v in rr if v})
        chosen=next(iter(missing),None) if ell==2 else (d-1 if d-1 in missing else None)
        collision=None;planted_alternative=None
        if chosen is not None:
            u,v=pair(d,chosen);assert u!=v and forward(u,A,q,8,ell)==forward(v,A,q,8,ell)
            assert [j for j,(x,z) in enumerate(zip(spectrum(u,ell,q),spectrum(v,ell,q))) if x!=z]==[chosen]
            collision=chosen
        if ell==2:
            j=next((j for j in range(1,d) if s[j]!=s[0]),d)
            if j-1 in missing:
                v=tuple(-x for x in s[:j+1])+s[j+1:] if j<d else tuple(-x for x in s)
                assert v!=s and forward(v,A,q,8,ell)==y;planted_alternative=list(v)
        out.append(dict(ell=ell,d=d,root=root,prior_status=st,missing=missing,collision_coordinate=collision,alternative_planted_preimage=planted_alternative))
    (HERE/'PRIOR_ROSTER.json').write_text(json.dumps(dict(scope='FROZEN_SL02_GLOBAL_SYNTHETIC_CASES',rows=out),indent=2)+'\n')
    for ell in (2,3):
        for d in (16,64,256):
            z=[r for r in out if r['ell']==ell and r['d']==d]
            print(ell,d,'n',len(z),'omitted',sum(bool(r['missing']) for r in z),'collision',sum(r['collision_coordinate'] is not None for r in z),'planted alternate',sum(r['alternative_planted_preimage'] is not None for r in z),'prior caps',sum(r['prior_status']=='CAP' for r in z))
if __name__=='__main__':main()
