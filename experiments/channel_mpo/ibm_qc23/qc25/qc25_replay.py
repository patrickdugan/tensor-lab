"""QC-25B: offline replay on committed IBM counts. No IBM service calls."""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
PARENT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PARENT))
from qc23_analysis import load_set
from qc23_compile import qasm3_text, audit_mapping
from qc23_ibm import z_estimators
from qc23_stabilizer import extract_native_ops, pauli_flip_expectations
from qc23_noise import restrict_snapshot, model_a_faults
from qc23_mpo import ops_to_layers, effective_depolarizing_p, mpo_z as uniform_z
from qc23_nonclifford import ideal_z, make_kicked_ising
from qc25_channels import native_channels, mpo_z

ROOT=PARENT/'results'/'hw_root'
SOURCE='959e73ad60dd51954e7cd5f9900ba10e208555f6'
CAPS=(16,32,64)
TOL=1e-6

def sha(b): return hashlib.sha256(b).hexdigest()

def audited_set(set_name):
    manifest,items=load_set(ROOT,set_name)
    assert sha((ROOT/'compiled'/f'{set_name}.qpy').read_bytes())==manifest['qpy_sha256'], 'QPY hash mismatch'
    counts=json.loads((ROOT/'ibm'/f'{set_name}_counts.json').read_text())
    assert set(counts)==set(items), 'missing or extra counts'
    snap_path=ROOT/'compiled'/('calibration_snapshot_nc_eval.json' if set_name=='nc_eval' else 'calibration_snapshot.json')
    snap=json.loads(snap_path.read_text())
    provenance=dict(qpy_sha256=manifest['qpy_sha256'],calibration_sha256=sha(snap_path.read_bytes()),
                    counts_sha256=sha((ROOT/'ibm'/f'{set_name}_counts.json').read_bytes()),calibration_timestamp=snap['timestamp'],
                    stale_shared_qasm=[],count_job_ids=sorted(set(str(v.get('job_id','not recorded')) for v in counts.values())))
    for cid,(rec,isa) in items.items():
        expected=rec['qasm3_sha256']
        # nc_eval was submitted from its own directory. Its shared QASM text is stale.
        submitted=(PARENT/'results'/'ibm_run_nc'/'circuits'/f'{cid}.qasm') if set_name=='nc_eval' else ROOT/'compiled'/'circuits'/f'{cid}.qasm'
        assert sha(submitted.read_bytes())==expected, (cid,'submitted QASM hash mismatch')
        assert sha(qasm3_text(isa).encode())==expected,(cid,'QPY re-export hash mismatch')
        shared=ROOT/'compiled'/'circuits'/f'{cid}.qasm'
        if shared.exists() and sha(shared.read_bytes())!=expected: provenance['stale_shared_qasm'].append(cid)
        audit_mapping(isa,rec['physical_path'],rec['n'])
        raw=counts[cid]
        assert raw['physical_path']==rec['physical_path'], (cid,'raw count path mismatch')
        assert sum(raw['counts'].values())==rec['shots'],(cid,'shot count mismatch')
    return items,counts,snap,provenance

def predict(n,channels,b,ro):
    for cap in CAPS:
        result=mpo_z(n,channels,b,cap=cap,readout=ro)
        if result['truncation_bound_exact_arithmetic']<=TOL: break
    return result,cap

def uniform(n,layers,p,b,readout):
    # Exact QC23 comparator uses cap 16. Keep original cap for full-register replay.
    r=uniform_z(n,layers,p,b,bond_cap=16)
    return float(r.expectation)*(1-2*readout[b]),float(r.error_bound)

def metrics(rows,keys):
    return {k:{'mae':float(np.mean([abs(r[k]-r['measured']) for r in rows])),
               'within_005':sum(abs(r[k]-r['measured'])<=.05 for r in rows),
               'n':len(rows)} for k in keys}

def paired(rows,candidate,baseline,seed=58241,boot=30000):
    by={}
    for r in rows: by.setdefault(r['id'],[]).append(r)
    # Equal weight per circuit; explicitly distinguish from pooled per-site MAE.
    a=[float(np.mean([abs(r[baseline]-r['measured'])-abs(r[candidate]-r['measured']) for r in g])) for g in by.values()]
    rng=random.Random(seed); b=sorted(sum(rng.choices(a,k=len(a)))/len(a) for _ in range(boot))
    return dict(circuits=len(a),mean_difference=sum(a)/len(a),ci95=[b[int(.025*boot)],b[int(.975*boot)-1]],wins=sum(x>0 for x in a))

def run(set_name,width):
    start=time.perf_counter()
    items,counts,snap,provenance=audited_set(set_name)
    prior=json.loads((ROOT/'analysis_ibm.json').read_text())
    lookup={(r['id'],r['bit']):r for r in prior['rows']}
    rows=[]; max_echo_gap=0.; max_published_ideal_gap=0.; max_published_uniform_gap=0.
    for cid,(rec,isa) in sorted(items.items()):
        n=rec['n']
        if n!=width: continue
        path=rec['physical_path']; local=restrict_snapshot(snap,path)
        ops=extract_native_ops(isa,path)
        channels,ro=native_channels(ops,local)
        faults,readout=model_a_faults(ops,local)
        layers,info=ops_to_layers(ops,n)
        p=effective_depolarizing_p(faults,n,info['mpo_layers'])
        bits=tuple(range(n))
        est=z_estimators(counts[cid]['counts'],n,bits)
        is_echo=set_name=='eval_echo'
        ideal={b:1. for b in bits} if is_echo else ideal_z(make_kicked_ising(n,rec['depth'],rec['seed']),bits)
        exact=pauli_flip_expectations(ops,n,faults,readout,bits) if is_echo else {}
        for b in bits:
            r,cap=predict(n,channels,b,ro[b])
            u,ub=uniform(n,layers,p,b,readout)
            m=est[str(b)]['expectation_z']
            old=lookup.get((cid,b))
            if old is not None:
                assert abs(m-old['measured'])<1e-12,(cid,b,'count replay')
                max_published_ideal_gap=max(max_published_ideal_gap,abs(ideal[b]-old['ideal']))
                max_published_uniform_gap=max(max_published_uniform_gap,abs(u-old['mpo_modelA']))
                assert abs(ideal[b]-old['ideal'])<1e-10,(cid,b,'ideal replay')
                assert abs(u-old['mpo_modelA'])<1e-8,(cid,b,'uniform baseline replay')
            row=dict(id=cid,n=n,depth=rec['depth'],seed=rec['seed'],bit=b,physical=path[b],measured=m,
                     ideal=ideal[b],uniform_A=u,uniform_ledger=ub,native_A=r['z'],native_ledger=r['truncation_bound_exact_arithmetic'],
                     cap=cap,peak_bond=r['peak_bond'],published=old is not None)
            if is_echo:
                gap=abs(r['z']-exact[b]); max_echo_gap=max(max_echo_gap,gap)
                assert gap<=row['native_ledger']+1e-9,(cid,b,gap)
                assert gap<1e-8,(cid,b,'Clifford oracle mismatch')
                row['pauli_A_exact']=exact[b]
            if old is not None: row['legacy_B']=old['mpo_modelB']
            rows.append(row)
    assert rows, 'empty width shard'
    sparse_oracle_gap=None
    if set_name=='nc_eval' and width==12:
        oracle=json.loads((Path(__file__).with_name('qc25b_sparse_spotcheck.json')).read_text())
        index={(r['id'],r['bit']):r for r in rows}
        errors=[]
        for o in oracle['rows']:
            r=index[(o['id'],o['bit'])]
            gap=abs(r['native_A']-o['native_A'])
            assert gap<=r['native_ledger']+o['drop_bound']+1e-9,(o['id'],o['bit'],gap)
            errors.append(gap)
        sparse_oracle_gap=max(errors)
    keys=['ideal','uniform_A','native_A']
    pub=[r for r in rows if r['published']]
    summary=dict(label='retrospective IBM count replay; no new QPU',source_commit=SOURCE,set=set_name,width=width,
                 circuits=len({r['id'] for r in rows}),rows=len(rows),published_rows=len(pub),
                 metrics_all=metrics(rows,keys),metrics_published=metrics(pub,keys+['legacy_B']),
                 paired_all=paired(rows,'native_A','uniform_A'),paired_published=paired(pub,'native_A','uniform_A'),
                 max_native_ledger=max(r['native_ledger'] for r in rows),max_uniform_ledger=max(r['uniform_ledger'] for r in rows),
                 native_bound_exceeds_1e6=sum(r['native_ledger']>TOL for r in rows),
                 sparse_oracle_gap=sparse_oracle_gap,max_echo_oracle_gap=max_echo_gap,max_published_ideal_gap=max_published_ideal_gap,
                 max_published_uniform_gap=max_published_uniform_gap,
                 cap_hist={str(c):sum(r['cap']==c for r in rows) for c in CAPS},provenance=provenance,
                 seconds=time.perf_counter()-start)
    return dict(summary=summary,rows=rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--set',choices=['eval_echo','nc_eval'],required=True)
    ap.add_argument('--width',type=int,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); result=run(a.set,a.width)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('QC25B_SUMMARY '+json.dumps(result['summary'],sort_keys=True))
