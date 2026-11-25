"""QC-26A: compile and optionally submit a frozen, owner-authorized plan.

No IBM access in tests. Build uses the account already configured on the local
machine. The estimate is not a provider-enforced time limit. Never print tokens.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, io, json, math, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from qc23_access import connect,check_authorization,plan_hash
from qc23_compile import transpile_on_path,audit_mapping,qasm3_text,qpy_bytes
from qc23_noise import calibration_snapshot

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def load_plan():
    plan=json.loads((HERE/'qc26a_plan.json').read_text())
    assert plan['backend']=='ibm_kingston' and plan['cap_seconds']==20
    assert len(plan['rows'])==82 and len({r['id'] for r in plan['rows']})==82
    assert all(r['shots']==384 for r in plan['rows'])
    return plan

def h_native(qc,q):
    qc.rz(math.pi/2,q);qc.sx(q);qc.rz(math.pi/2,q)

def logical_circuit(row):
    from qiskit import QuantumCircuit
    n=len(row['path'])
    qc=QuantumCircuit(n,name=row['id'])
    family=row['family']
    if family=='spam':
        if row['state']==1:
            for q in range(n): qc.x(q)
    elif family=='sx':
        for _ in range(row['m']):
            for q in range(n): qc.sx(q)
            qc.barrier()
        if row['axis']=='Y':
            for q in range(n): qc.sx(q)
        elif row['axis']!='Z': raise ValueError('bad SX axis')
    elif family=='cz':
        if row['state']==1: qc.x(0)
        h_native(qc,1)
        qc.barrier()
        for _ in range(row['m']):
            qc.cz(0,1);qc.barrier()
        if row['axis']=='Y': qc.rz(-math.pi/2,1)
        elif row['axis']!='X': raise ValueError('bad CZ axis')
        h_native(qc,1)
    else: raise ValueError('unknown circuit family')
    qc.measure_all()
    return qc

def assert_counts(row,isa):
    c=isa.count_ops()
    f=row['family']
    want_cz=row['m'] if f=='cz' else 0
    want_sx=(len(row['path'])*(row['m']+(row['axis']=='Y')) if f=='sx' else 2 if f=='cz' else 0)
    want_x=(len(row['path'])*row['state'] if f=='spam' else row['state'] if f=='cz' else 0)
    assert c.get('cz',0)==want_cz,(row['id'],'CZ count',c)
    assert c.get('sx',0)==want_sx,(row['id'],'SX count',c)
    assert c.get('x',0)==want_x,(row['id'],'X count',c)
    assert c.get('measure',0)==len(row['path'])
    assert set(c)<=set(['rz','sx','x','cz','barrier','measure']),c

def duration(isa,target):
    # Simple ASAP critical path, honoring barriers and calibrated measurement.
    # Unknown nonvirtual gate duration is a build failure.
    clock={}
    for inst in isa.data:
        op=inst.operation.name
        qs=tuple(isa.find_bit(q).index for q in inst.qubits)
        start=max((clock.get(q,0.) for q in qs),default=0.)
        if op in ('barrier','rz'): dur=0.
        else:
            props=target[op].get(qs)
            if props is None or props.duration is None: raise ValueError(f'missing duration {op} {qs}')
            dur=float(props.duration)
            if not math.isfinite(dur) or dur<0: raise ValueError('invalid duration')
        for q in qs: clock[q]=start+dur
    return max(clock.values(),default=0.)

def estimate(plan,compiled):
    e=plan['estimate']
    return sum(r['shots']*(e['shot_repetition_delay_s']+max(e['per_shot_circuit_allowance_s'],r['duration_s']))+e['per_circuit_overhead_allowance_s'] for r in compiled)

def backend_connection():
    service,err=connect(None)
    if service is None: raise RuntimeError('IBM account unavailable on this machine')
    backend=service.backend('ibm_kingston')
    if backend.name!='ibm_kingston': raise RuntimeError('wrong backend')
    return service,backend

def build(out):
    plan=load_plan()
    _,backend=backend_connection()
    snap=calibration_snapshot(backend,source='qc26a_build')
    if snap['two_qubit_gate']!='cz': raise ValueError('QC26A requires a CZ-native backend')
    out.mkdir(parents=True,exist_ok=False)
    (out/'circuits').mkdir()
    write_json(out/'calibration_build.json',snap)
    rows=[];isas=[]
    for row in plan['rows']:
        qc=logical_circuit(row)
        isa=transpile_on_path(backend,qc,row['path'],optimization_level=0)
        mapping=audit_mapping(isa,row['path'],len(row['path']))
        assert_counts(row,isa)
        qasm=qasm3_text(isa).encode()
        (out/'circuits'/f"{row['id']}.qasm").write_bytes(qasm)
        rows.append({**row,'mapping':mapping,'duration_s':duration(isa,backend.target),
                     'gate_counts':{k:int(v) for k,v in isa.count_ops().items()},'qasm_sha256':hash_bytes(qasm)})
        isas.append(isa)
    qpy=qpy_bytes(isas)
    (out/'compiled.qpy').write_bytes(qpy)
    est=estimate(plan,rows)
    meta={'backend':plan['backend'],'version':plan['version'],'compiled_at':now(),
          'calibration_sha256':hash_bytes((out/'calibration_build.json').read_bytes()),
          'proposal_sha256':hash_bytes((HERE/'qc26a_plan.json').read_bytes()),
          'qpy_sha256':hash_bytes(qpy),'estimated_qpu_seconds':est,'cap_seconds':20,'shots':384,'circuits':rows}
    meta['plan_sha256']=plan_hash(meta)
    write_json(out/'compiled.json',meta)
    if est>20: raise RuntimeError(f'compiled estimate {est:.3f}s exceeds proposed cap; do not submit')
    print(json.dumps({'plan_sha256':meta['plan_sha256'],'circuits':len(rows),'estimate_s':est,'cap_s':20},indent=2))
    print('Build only; no QPU job submitted.')

def validate_frozen(out):
    from qiskit import qpy
    plan=load_plan()
    meta=json.loads((out/'compiled.json').read_text())
    sha=meta.pop('plan_sha256')
    if plan_hash(meta)!=sha: raise ValueError('manifest hash mismatch')
    meta['plan_sha256']=sha
    assert meta['backend']=='ibm_kingston' and meta['cap_seconds']==20 and meta['shots']==384
    assert meta['proposal_sha256']==hash_bytes((HERE/'qc26a_plan.json').read_bytes())
    assert meta['calibration_sha256']==hash_bytes((out/'calibration_build.json').read_bytes())
    data=(out/'compiled.qpy').read_bytes()
    assert meta['qpy_sha256']==hash_bytes(data)
    isas=qpy.load(io.BytesIO(data))
    assert len(isas)==len(plan['rows'])==len(meta['circuits'])
    for spec,row,isa in zip(plan['rows'],meta['circuits'],isas):
        assert all(row[k]==v for k,v in spec.items()),'plan row mismatch'
        assert_counts(row,isa)
        assert audit_mapping(isa,row['path'],len(row['path']))==row['mapping']
        assert hash_bytes((out/'circuits'/f"{row['id']}.qasm").read_bytes())==row['qasm_sha256']
        assert hash_bytes(qasm3_text(isa).encode())==row['qasm_sha256']
    assert abs(estimate(plan,meta['circuits'])-meta['estimated_qpu_seconds'])<1e-9
    return meta,isas

def submit(out,authorization):
    meta,isas=validate_frozen(out)
    if not authorization: raise ValueError('owner authorization file is required')
    ok,why=check_authorization(str(authorization),backend=meta['backend'],sha=meta['plan_sha256'],
                               estimated_seconds=meta['estimated_qpu_seconds'])
    if not ok: raise ValueError(why)
    auth=json.loads(Path(authorization).read_text())
    if auth['max_qpu_seconds']>20 or meta['estimated_qpu_seconds']>20:
        raise ValueError('QC26A cap is 20 estimated QPU seconds')
    if auth.get('acknowledge_estimate_not_hard_limit') is not True or auth.get('checked_current_allocation') is not True:
        raise ValueError('owner must acknowledge provider runtime uncertainty and confirm current allocation')
    age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(meta['compiled_at'])).total_seconds()
    if not 0<=age<=900: raise ValueError('frozen compilation older than 15 minutes; rebuild and reauthorize')
    _,backend=backend_connection()
    old=json.loads((out/'calibration_build.json').read_text())
    snap=calibration_snapshot(backend,source='qc26a_submission')
    if not old.get('timestamp') or snap.get('timestamp')!=old['timestamp']:
        raise ValueError('calibration timestamp changed or missing; rebuild and reauthorize')
    for k in ('qubits','edges','basis_gates','two_qubit_gate'):
        # JSON normalization makes tuple/list representation irrelevant.
        if json.dumps(old[k],sort_keys=True)!=json.dumps(snap[k],sort_keys=True):
            raise ValueError(f'calibration {k} changed; rebuild and reauthorize')
    # Verify every stored circuit remains supported by the current target.
    for row,isa in zip(meta['circuits'],isas):
        if abs(duration(isa,backend.target)-row['duration_s'])>1e-12:
            raise ValueError('duration changed; rebuild and reauthorize')
    write_json(out/'calibration_submission.json',snap)
    # Exclusive marker before the irreversible call: do not retry automatically
    # after an ambiguous network error. Check account job history first.
    with (out/'submission_started.json').open('x') as f:
        json.dump({'started_at':now(),'plan_sha256':meta['plan_sha256']},f)
    from qiskit_ibm_runtime import SamplerV2
    sampler=SamplerV2(mode=backend,options={'execution':{'rep_delay':0.00025}})
    job=sampler.run(isas,shots=meta['shots'])
    write_json(out/'job.json',{'job_id':job.job_id(),'plan_sha256':meta['plan_sha256'],'submitted_at':now()})
    print('Submitted job:',job.job_id(),flush=True)
    result=job.result()
    if len(result)!=len(isas): raise RuntimeError('PUB count mismatch')
    counts={}
    for row,entry in zip(meta['circuits'],result):
        c=entry.data.meas.get_counts()
        assert sum(c.values())==row['shots']
        assert all(len(k)==len(row['path']) and set(k)<=set('01') for k in c)
        counts[row['id']]={k:int(v) for k,v in c.items()}
    write_json(out/'counts.json',counts)
    print('Wrote raw counts; inspect provider-reported usage separately.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['build','submit'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--authorization',type=Path)
    a=p.parse_args()
    if a.command=='build':
        if a.authorization: p.error('authorization is for submit only')
        build(a.out)
    else: submit(a.out,a.authorization)
