"""HAD-QFT-QPU01: paired three/four-qubit shell-QFT hardware pilot.
Logical mode needs NumPy only. Hardware mode uses the existing QC23 account,
calibration, mapping and owner-authorization helpers. No automatic submission.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, io, json, math, random, sys
from pathlib import Path
import numpy as np
from recursive import compile_shell, inverse, simulate
from radial_qft import compile_radial, radial_matrix

HERE = Path(__file__).resolve().parent
SHOTS, CAP, REP = 384, 15, .00025


def sha(b): return hashlib.sha256(b).hexdigest()
def digest(x): return sha(json.dumps(x, sort_keys=True, separators=(',', ':')).encode())
def write(p, x): p.write_text(json.dumps(x, indent=2, sort_keys=True)+'\n', encoding='utf8')
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def must(ok, message):
    if not ok: raise ValueError(message)


def core(b, arm):
    if arm == 'recursive':
        g, w = compile_shell(b, 2, 'noanc'); assert w == b; return g
    assert arm == 'gray'
    # Both arms receive binary shell labels, so account for the Gray conversion.
    encode = [('cx', i, i+1) for i in reversed(range(b-1))]
    return encode + compile_radial((1<<b)-1, 2) + encode[::-1]


def prepare(b, a, other=None, phase=0.):
    g = [('x', i) for i in range(b) if a>>(b-1-i)&1]
    if other is None: return g
    assert a < other
    diffs = [i for i in range(b) if (a^other)>>(b-1-i)&1]
    pivot = diffs[0]
    assert not (a>>(b-1-pivot)&1)
    return g + [('h', pivot), ('p', pivot, phase)] + [('cx', pivot, i) for i in diffs[1:]]


def read_basis(b, basis):
    if basis == 'Z': return []
    assert basis in ('X','Y')
    return ([('p', i, -math.pi/2) for i in range(b)] if basis == 'Y' else []) + [('h', i) for i in range(b)]


def route(gates):
    """Remote CNOT on a line, exact 4d-4 CNOT bridge, restores intermediates."""
    out = []
    for g in gates:
        if g[0] != 'cx' or abs(g[1]-g[2]) == 1:
            out.append(g); continue
        _, a, b = g
        p = list(range(a,b+1)) if a < b else list(range(a,b-1,-1))
        e = [('cx', x, y) for x,y in zip(p,p[1:])]
        out += e + e[-2::-1] + e[1:] + e[-2:0:-1]
    return out


def count_cx(g): return sum(x[0]=='cx' for x in g)
def evolved(g, b, v): return simulate([x for x in g if x[0]!='barrier'], b, v)


def plan(b):
    must(b in (3,4), 'initial pilot supports b=3 or b=4')
    d=1<<b; Q=radial_matrix(d-1,2); zero=np.eye(d,dtype=complex)[:,0]
    probes=[('zero',0,None,0.,'X'),('last',d-1,None,0.,'X'),
            ('plus',1,d-2,0.,'X'),('imag',1,d-2,math.pi/2,'Y')]
    rows=[]
    def add(label, arm, probe, basis, gates, expected):
        p=abs(expected)**2
        assert abs(sum(p)-1)<1e-11
        routed=route(gates)
        assert np.max(abs(evolved(gates,b,zero)-expected))<1e-10
        assert np.max(abs(evolved(routed,b,zero)-expected))<1e-10
        rows.append(dict(id=label,arm=arm,probe=probe,basis=basis,
                         ideal=p.tolist(),logical_cx=count_cx(gates),line_cx=count_cx(routed),gates=[list(g) for g in routed]))
    for k in range(d): add(f'spam_{k:02}','spam',str(k),'Z',prepare(b,k),np.eye(d)[:,k])
    for name,a,o,phase,basis in probes:
        prep=prepare(b,a,o,phase); x=evolved(prep,b,zero)
        if o is not None:
            add(f'prep_{name}','prep',name,'echo',prep+[('barrier',)]+inverse(prep),zero)
        for arm in ('gray','recursive'):
            g=core(b,arm)
            for m in ('Z',basis):
                r=read_basis(b,m)
                add(f'{arm}_{name}_{m}',arm,name,m,prep+g+r,evolved(r,b,Q@x))
            if o is not None:
                add(f'{arm}_{name}_echo',arm,name,'echo',prep+g+[('barrier',)]+inverse(g)+inverse(prep),zero)
    # Retain complete roster, fixed shuffle to avoid clustering all trials by arm.
    random.Random(271126+b).shuffle(rows)
    for arm in ('gray','recursive'):
        g=core(b,arm)
        assert np.max(abs(simulate(g,b,np.eye(d))-Q))<1e-10
        assert np.max(abs(simulate(route(g),b,np.eye(d))-Q))<1e-10
    return dict(scope='EXPLORATORY_MATCHED_SHELL_QFT',b=b,N=2**(d-1),shots=SHOTS,cap_seconds=CAP,
                rows=rows,core_counts={a:{'logical_cx':count_cx(core(b,a)),
                    'line_cx':count_cx(route(core(b,a)))} for a in ('gray','recursive')})


def hardware_helpers():
    # qft01/ -> hadamard_qft/ -> experiments/ -> channel_mpo/ibm_qc23
    sys.path.insert(0,str(HERE.parents[1]/'channel_mpo'/'ibm_qc23'))
    from qc23_access import connect, check_authorization
    from qc23_compile import choose_line_calibrated, transpile_on_path, audit_mapping, isa_metrics, qasm3_text, qpy_bytes
    from qc23_noise import calibration_snapshot
    from qc26.qc26a_compile import duration
    return locals()


def connection(h, name):
    service, error = h['connect'](None)
    must(service is not None,'IBM account unavailable on this machine; use saved account')
    backend=service.backend(name)
    must(backend.name==name and backend.status().operational,'backend unavailable')
    return service,backend


def usage(service):
    u=service.usage()
    return {k:u.get(k) for k in ('usage_remaining_seconds','usage_consumed_seconds','usage_limit_seconds','usage_limit_reached','usage_period')}


def enough(u, seconds):
    r=u['usage_remaining_seconds']
    must(isinstance(r,(int,float)) and math.isfinite(r) and r>=seconds and not u['usage_limit_reached'],
         'live allocation unavailable or insufficient; do not submit')


def sources():
    return {f:sha((HERE/f).read_bytes()) for f in ('qpu01.py','recursive.py','radial_qft.py','orbit_qft.py','circuit.py')}


def circuit(row,b,measured=True):
    from qiskit import QuantumCircuit
    q=QuantumCircuit(b,name=row['id'])
    for g in row['gates']:
        if g[0]=='barrier':
            if measured: q.barrier()
        elif g[0] in ('ry','rz','p'): getattr(q,g[0])(g[2],g[1])
        elif g[0]=='phase': q.global_phase+=g[1]
        else: getattr(q,g[0])(*g[1:])
    if measured: q.measure_all()
    return q


def qiskit_check(p):
    from qiskit.quantum_info import Statevector
    b=p['b']; order=[int(f'{i:0{b}b}'[::-1],2) for i in range(1<<b)]
    for row in p['rows']:
        got=Statevector.from_instruction(circuit(row,b,False)).probabilities()[order]
        must(np.max(abs(got-row['ideal']))<1e-10,'Qiskit logical probability mismatch')


def estimate(rows,reset):
    # IBM's ~2 s overhead, plus conservative prior QC23 per-circuit allowance.
    return 2+sum(.1+SHOTS*(REP+max(.0001,r['duration_s']+reset)) for r in rows)


def build(out,b,name):
    h=hardware_helpers(); p=plan(b); qiskit_check(p)
    service,backend=connection(h,name); u=usage(service); enough(u,CAP)
    snap=h['calibration_snapshot'](backend,source='had_qft_qpu01')
    must(bool(snap['timestamp']),'missing calibration timestamp')
    path,cost=h['choose_line_calibrated'](backend,b)
    reset=max(float(backend.target['reset'][(q,)].duration) for q in path)
    rows=[]; isas=[]
    out.mkdir(parents=True,exist_ok=False); (out/'circuits').mkdir()
    write(out/'calibration.json',snap);write(out/'usage_build.json',u)
    for row in p['rows']:
        isa=h['transpile_on_path'](backend,circuit(row,b),path,optimization_level=0)
        mapping=h['audit_mapping'](isa,path,b)
        qasm=h['qasm3_text'](isa).encode(); (out/'circuits'/f"{row['id']}.qasm").write_bytes(qasm)
        rows.append({**row,'mapping':mapping,'metrics':h['isa_metrics'](isa),'duration_s':h['duration'](isa,backend.target),'qasm_sha256':sha(qasm)})
        isas.append(isa)
    qpy=h['qpy_bytes'](isas); (out/'compiled.qpy').write_bytes(qpy)
    meta=dict(backend=name,b=b,shots=SHOTS,cap_seconds=CAP,path=path,layout_cost=cost,compiled_at=now(),
              plan_digest=digest(p),sources=sources(),calibration_sha256=sha((out/'calibration.json').read_bytes()),
              qpy_sha256=sha(qpy),reset_duration=reset,estimated_qpu_seconds=estimate(rows,reset),rows=rows)
    meta['plan_sha256']=digest(meta);write(out/'compiled.json',meta)
    must(meta['estimated_qpu_seconds']<=CAP,'estimate exceeds 15 s ceiling')
    print(json.dumps({k:meta[k] for k in ('backend','b','path','plan_sha256','estimated_qpu_seconds','cap_seconds')},indent=2))
    print('Build only. No job submitted.')


def validate(out,h):
    from qiskit import qpy
    m=read(out/'compiled.json'); k=m.pop('plan_sha256');must(digest(m)==k,'manifest mismatch');m['plan_sha256']=k
    p=plan(m['b']);must(m['plan_digest']==digest(p) and m['sources']==sources(),'plan/source changed')
    must(m['shots']==SHOTS and m['cap_seconds']==CAP,'shot or cap mismatch')
    must(m['calibration_sha256']==sha((out/'calibration.json').read_bytes()),'calibration hash mismatch')
    q=(out/'compiled.qpy').read_bytes();must(m['qpy_sha256']==sha(q),'QPY hash mismatch');isas=qpy.load(io.BytesIO(q))
    must(len(isas)==len(m['rows'])==len(p['rows']),'roster mismatch')
    for row,spec,isa in zip(m['rows'],p['rows'],isas):
        must(all(row[k]==v for k,v in spec.items()),'row changed')
        must(row['mapping']==h['audit_mapping'](isa,m['path'],m['b']),'layout changed')
        must(row['metrics']==h['isa_metrics'](isa),'ISA counts changed')
        must(row['qasm_sha256']==sha(h['qasm3_text'](isa).encode())==sha((out/'circuits'/f"{row['id']}.qasm").read_bytes()),'QASM mismatch')
    must(abs(m['estimated_qpu_seconds']-estimate(m['rows'],m['reset_duration']))<1e-9,'estimate changed')
    return m,isas


def submit(out,authorization):
    h=hardware_helpers();m,isas=validate(out,h)
    ok,why=h['check_authorization'](authorization,backend=m['backend'],sha=m['plan_sha256'],estimated_seconds=m['estimated_qpu_seconds']);must(ok,why)
    a=read(Path(authorization));must(a['max_qpu_seconds']<=CAP and a.get('checked_current_allocation') is True and a.get('acknowledge_estimate_not_hard_limit') is True,'owner acknowledgments/ceiling required')
    age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(m['compiled_at'])).total_seconds();must(0<=age<=900,'build older than 15 minutes')
    service,backend=connection(h,m['backend']);snap=h['calibration_snapshot'](backend,source='had_qft_qpu01');must(snap==read(out/'calibration.json'),'calibration changed')
    must(max(float(backend.target['reset'][(q,)].duration) for q in m['path'])==m['reset_duration'],'reset duration changed')
    for row,isa in zip(m['rows'],isas):must(abs(h['duration'](isa,backend.target)-row['duration_s'])<1e-12,'duration changed')
    before=usage(service);enough(before,a['max_qpu_seconds']);write(out/'usage_before.json',before)
    write(out/'calibration_submission.json',snap)
    with (out/'submission_started.json').open('x') as f:json.dump({'plan_sha256':m['plan_sha256'],'at':now()},f)
    from qiskit_ibm_runtime import SamplerV2
    job=SamplerV2(mode=backend,options={'execution':{'rep_delay':REP}}).run(isas,shots=SHOTS)
    write(out/'job.json',{'job_id':job.job_id(),'plan_sha256':m['plan_sha256'],'submitted_at':now()})
    print('Submitted job:',job.job_id(),flush=True)


def analyze(out):
    m=read(out/'compiled.json');counts=read(out/'counts.json');results=[]
    must(set(counts)=={r['id'] for r in m['rows']},'count roster mismatch')
    for r in m['rows']:
        c=counts[r['id']];must(sum(c.values())==SHOTS and all(len(k)==m['b'] and set(k)<=set('01') and isinstance(v,int) and v>=0 for k,v in c.items()),'invalid counts')
        # Qiskit bitstring order c_(b-1)...c_0, our shell labels q_0...q_(b-1).
        p=np.zeros(1<<m['b'])
        for k,v in c.items(): p[int(k[::-1],2)]+=v/SHOTS
        q=np.asarray(r['ideal']);results.append(dict(id=r['id'],arm=r['arm'],probe=r['probe'],basis=r['basis'],tv=float(abs(p-q).sum()/2),hellinger_fidelity=float(np.sqrt(p*q).sum()**2),probability_zero=float(p[0])))
    summary={}
    for arm in ('gray','recursive'):
        f=[r for r in results if r['arm']==arm and r['basis']!='echo'];e=[r['probability_zero'] for r in results if r['arm']==arm and r['basis']=='echo']
        summary[arm]={'mean_forward_tv':float(np.mean([r['tv'] for r in f])),'mean_echo_return':float(np.mean(e))}
    pairs=[(r,next(s for s in results if s['arm']=='recursive' and s['probe']==r['probe'] and s['basis']==r['basis'])) for r in results if r['arm']=='gray' and r['basis']!='echo']
    report=dict(scope='DESCRIPTIVE_EXPLORATORY_RAW_COUNTS',summary=summary,paired_tv_gray_minus_recursive=[{'probe':a['probe'],'basis':a['basis'],'difference':a['tv']-b['tv']} for a,b in pairs],rows=results)
    write(out/'analysis.json',report);print(json.dumps(summary,indent=2))


def collect(out):
    h=hardware_helpers();m,_=validate(out,h);j=read(out/'job.json');must(j['plan_sha256']==m['plan_sha256'],'job manifest mismatch')
    service,_=connection(h,m['backend']);job=service.job(j['job_id']);result=job.result();must(len(result)==len(m['rows']),'PUB count mismatch')
    counts={r['id']:{k:int(v) for k,v in e.data.meas.get_counts().items()} for r,e in zip(m['rows'],result)}
    write(out/'counts.json',counts);after=usage(service);write(out/'usage_after.json',after)
    try: ju=job.usage()
    except Exception: ju=None
    write(out/'execution_receipt.json',dict(job_id=j['job_id'],plan_sha256=m['plan_sha256'],usage_before=read(out/'usage_before.json'),usage_after=after,job_usage=ju,collected_at=now()))
    analyze(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['logical','qiskit-check','build','submit','collect','analyze']);p.add_argument('--b',type=int,default=3);p.add_argument('--backend',default='ibm_kingston');p.add_argument('--out',type=Path,required=True);p.add_argument('--authorization');a=p.parse_args()
    if a.command in ('logical','qiskit-check'):
        x=plan(a.b)
        if a.command=='qiskit-check':qiskit_check(x)
        a.out.mkdir(parents=True,exist_ok=True);write(a.out/'logical_plan.json',x);print(json.dumps({'b':a.b,'circuits':len(x['rows']),'shots':SHOTS,'core_counts':x['core_counts'],'nominal_estimate_s':estimate([{'duration_s':0} for _ in x['rows']],0)},indent=2))
    elif a.command=='build':build(a.out,a.b,a.backend)
    elif a.command=='submit':submit(a.out,a.authorization)
    elif a.command=='collect':collect(a.out)
    else:analyze(a.out)
if __name__=='__main__':main()
