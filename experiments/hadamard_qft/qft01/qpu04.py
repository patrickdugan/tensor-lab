"""HAD-QFT-QPU04: matched seven-data-qubit, ancilla-free benchmark.
Reuses QPU03 receipt, authorization and collection pipeline without editing it.
No hardware calls for logical audit. Native and hardware commands need Qiskit.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import qpu01 as old
from qpu04_design import regenerate,REGIONS,EDGES
from route_dp import index
from selective import bound,count
from recursive import simulate
from radial_qft import radial_matrix

HERE=Path(__file__).resolve().parent
ALPHA=.05; SHOTS=1024
PROBES={'edge':0,'middle':64,'tail':126,'quarter':32,'threequarter':96}
ROSTER={'edge':('plus','minus','imag','basis0','basis1'),'middle':('plus','minus'),'tail':('plus','minus'),'quarter':('plus','minus','imag'),'threequarter':('plus','minus','imag')}
PHASE={'plus':0.,'minus':math.pi,'imag':math.pi/2}


def logical_rows(design):
    routes=regenerate(design);b=7;d=1<<b;Q=radial_matrix(d-1);I=np.eye(d,dtype=complex);rows=[]
    for arm in design['arms']:
        name=arm['arm'];g,z,v=routes[name]
        for probe,a in PROBES.items():
            for state in ROSTER[probe]:
                prep=old.prepare(b,a+(state=='basis1'),a+1 if state in PHASE else None,PHASE.get(state,0.))
                x=simulate(prep,b,I[:,0]);y=simulate(g,b,x);q=Q@x;target=d-1-a
                rows.append(dict(id=f'{name}_{probe}_{state}',arm=name,kind='core',probe=probe,state=state,basis='Z',target=target,ideal=(abs(q)**2).tolist(),compiled_ideal=(abs(y)**2).tolist(),gates=[list(t) for t in prep],n_wires=b,n_data=b))
        for state in ('plus','minus'):
            prep=old.prepare(b,0,1,PHASE[state]);read=[('h',b-1)];y=simulate(prep+read,b,I[:,0]);target=int(state=='minus')
            assert abs(y[target])**2>1-1e-10
            rows.append(dict(id=f'{name}_prep_edge_{state}',arm=name,kind='prep',probe='edge',state=state,basis='Z',target=target,ideal=(abs(y)**2).tolist(),compiled_ideal=(abs(y)**2).tolist(),gates=[list(t) for t in prep+read],n_wires=b,n_data=b))
        for state,target in (('zero',0),('top',d-1)):
            prep=old.prepare(b,target);y=I[:,target]
            rows.append(dict(id=f'{name}_spam_{state}',arm=name,kind='spam',probe='basis',state=state,basis='Z',target=target,ideal=(abs(y)**2).tolist(),compiled_ideal=(abs(y)**2).tolist(),gates=[list(t) for t in prep],n_wires=b,n_data=b))
    assert len(rows)==38 and len({r['id'] for r in rows})==38
    by={r['id']:r for r in rows}
    for arm in ('A','D'):
        for probe,a in PROBES.items():
            mi=by[f'{arm}_{probe}_minus'];pl=by[f'{arm}_{probe}_plus'];k=mi['target']
            assert abs(mi['ideal'][k]-pl['ideal'][k]-(1 if a==0 else 1/math.sqrt(2)))<1e-12
            assert abs((mi['ideal'][k]-pl['ideal'][k])-(mi['compiled_ideal'][k]-pl['compiled_ideal'][k]))<=2*bound(b)+1e-12
    return routes,rows


def local_native(gates,b):
    from qiskit import transpile
    # Input already routed; basis translation only, no layout optimization.
    return transpile(old.circuit({'id':'routed','gates':gates},b,measured=False),basis_gates=['cz','sx','rz','x'],optimization_level=0,seed_transpiler=0)


def native_verify(qc,z,b):
    from qiskit.quantum_info import Operator
    n=1<<b;rev=np.array([int(f'{i:0{b}b}'[::-1],2) for i in range(n)])
    expected=simulate(z['gates'],b,np.eye(n,dtype=complex));got=Operator(qc).data[np.ix_(rev,rev)]
    phase=np.vdot(expected.ravel(),got.ravel());old.must(abs(phase)>0,'native phase');got/=phase/abs(phase)
    e=float(np.max(abs(expected-got)));old.must(e<1e-9,'native operator mismatch')
    err=float(np.linalg.norm(got[np.ix_(index(b,z['final']),index(b,z['initial']))]-radial_matrix(n-1),ord=2))
    old.must(err<=bound(b)+1e-8,'native operator bound')
    return e,err


def build_rows(design):
    import qpu03 as base
    from qiskit import QuantumCircuit
    routes,rows=logical_rows(design);items=[];info={};b=7
    for arm in design['arms']:
        name=arm['arm'];g,z,v=routes[name];il=z['initial'];fl=z['final'];phys=arm['phys']
        tc=local_native(z['gates'],b);ne,op=native_verify(tc,z,b)
        edges={tuple(sorted(x)) for x in EDGES}
        for inst in tc.data:
            if len(inst.qubits)==2:old.must(inst.operation.name=='cz' and tuple(sorted(tc.find_bit(q).index for q in inst.qubits)) in edges,'routing changed')
        nc=int(tc.count_ops().get('cz',0));old.must(nc<=arm['routed_cx'],'native gate count increased')
        if name=='D':old.must(nc<200,'sub-200 native target failed')
        info[name]=dict(phys=phys,nw=b,il=il,fl=fl,logical_cx=count(g),core_metrics=base.T.metrics(tc),ver={**v,'native_vs_routed_max_entry_error':ne,'native_operator_error':op})
        for row in rows:
            if row['arm']!=name:continue
            qc=QuantumCircuit(b,b);pre=[(x[0],il[x[1]],*x[2:]) for x in row['gates']]
            qc.compose(local_native(pre,b),inplace=True);qc.barrier()
            if row['kind']=='core':qc.compose(tc,inplace=True);qc.barrier();read=fl
            else:read=il
            for logical in range(b):qc.measure(read[logical],logical)
            base.native_check(row,qc);items.append((row,qc,phys))
    old.must(len(items)==38,'native roster');return items,info


def install_pipeline():
    import qpu03 as base
    original=base.sources
    base.build_rows=build_rows
    base.sources=lambda: {**original(),**{x:old.sha((HERE/x).read_bytes()) for x in ('selective.py','route_dp.py','qpu04_design.py','qpu04.py')}}
    return base


def joint_audit(paths):
    # 95% family over both primary D-A differences, without assuming
    # independence across the two chip regions. Fixed shot-only bounds.
    out=[];seen=set()
    for path in paths:
        m=old.read(path/'compiled.json');a=old.read(path/'analysis.json');d=m['design'];r=d['replicate'];old.must(r in (0,1) and r not in seen and m['shots']==SHOTS,'invalid replicate');seen.add(r)
        old.must(d['scope']=='FROZEN_QPU04_PHASE_CONTRAST' and a['design_sha256']==m['design_sha256'],'wrong design')
        vals={}
        for arm in ('A','D'):
            vals[arm]={p:a['rows'][f'{arm}_{p}_minus']['target_prob']-a['rows'][f'{arm}_{p}_plus']['target_prob'] for p in PROBES}
        delta=vals['D']['edge']-vals['A']['edge'];radius=math.sqrt(2*math.log(4/ALPHA)/SHOTS)
        out.append(dict(replicate=r,edge_difference=delta,interval=[delta-radius,delta+radius],contrast=vals,job_id=old.read(path/'job.json')['job_id']))
    old.must(seen=={0,1},'both frozen replicates required');return dict(scope='QPU04_TWO_REGION_PRIMARY_SHOT_ONLY',alpha=ALPHA,primary_both_positive=all(x['interval'][0]>0 for x in out),regions=out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['logical','native','build','submit','collect','analyze','joint']);p.add_argument('--design',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--authorization');p.add_argument('--stage',type=Path,action='append');a=p.parse_args()
    if a.command in ('logical','native','build'):
        d=old.read(a.design);routes,rows=logical_rows(d)
        if a.command=='logical':
            a.out.mkdir(parents=True,exist_ok=True);old.write(a.out/'logical_summary.json',{'design':d,'rows':len(rows),'checks':{k:v[2] for k,v in routes.items()}});print('Logical verification passed, 38 rows; no job submitted.')
        elif a.command=='native':
            _,info=build_rows(d);a.out.mkdir(parents=True,exist_ok=True);old.write(a.out/'native_summary.json',info);print('Native operator verification passed; no job submitted.')
        else:install_pipeline().build(a.out,a.design,d['backend'],d['replicate'])
    elif a.command=='submit':install_pipeline().submit(a.out,a.authorization)
    elif a.command=='collect':install_pipeline().collect(a.out)
    elif a.command=='analyze':install_pipeline().analyze(a.out)
    else:
        report=joint_audit(a.stage or []);old.write(a.out,report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
