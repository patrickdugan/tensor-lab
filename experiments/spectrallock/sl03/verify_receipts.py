"""Recreate every frozen input; validate candidate and rank/coverage receipts."""
import hashlib,json,random
from pathlib import Path
import lattice_pilot as lp
HERE=Path(__file__).parent

def main():
    rows=[];seen={};replayed=0;hits=0;rank_checks=0
    for f in sorted(HERE.glob('AUDIT_*.json')):
        for r in json.loads(f.read_text())['records']:
            d,q,B,root,mode,regime=[r[k] for k in ('d','q','B','root','mode','regime')]
            seed=92300+1000*d+100*root+(0 if regime=='fixed' else 1000000)
            rng=random.Random(seed);s=tuple(rng.choice((-1,1)) for _ in range(d))
            A,S=lp.matrices(d,d,q,seed+1,mode);y=lp.rounded(A,s,q,B)
            sha=hashlib.sha256(json.dumps([A,y],separators=(',',':')).encode()).hexdigest()
            assert sha==r['instance_sha256']
            if sha in seen:replayed+=1
            seen[sha]=r['status']
            # Independent rank via integer determinant for full-rank square cases,
            # and modular elimination for rank-deficient cases.
            rank=lp.rank_mod(A,q);assert rank==r['rank']
            if S is not None:
                assert rank==lp.rank_mod(S,q)
                assert sum(any(row[i] for row in S) for i in range(d))==r['covered_frequencies']
            assert (int(lp.Matrix(A).det())%q!=0)==(rank==d);rank_checks+=1
            if r['status']=='HIT':
                v=tuple(r['recovered']);assert len(v)==d and set(v)<= {-1,1}
                assert lp.rounded(A,v,q,B)==y and (v==s)==r['original'];hits+=1
            else:assert r['status']=='MISS' and r['recovered'] is None
            rows.append(r)
    assert len(rows)==102 and len(seen)==78 and replayed==24
    result=dict(receipts=len(rows),unique_inputs=len(seen),repeated_inputs=replayed,verified_hits=hits,exact_determinant_checks=rank_checks,all_hits_original=all(x['original'] for x in rows if x['status']=='HIT'))
    (HERE/'RECEIPT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
