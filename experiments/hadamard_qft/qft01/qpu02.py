"""HAD-QFT-QPU02, prospective phase-contrast campaign. Build/check is read-only.
Never creates owner authorization; separate submit and collect. CPU logical audit.
"""
from __future__ import annotations
import argparse, datetime as dt, io, json, math, random
from functools import lru_cache
from pathlib import Path
import numpy as np
import qpu01 as old
from compact import mapped, gray_core, recursive_core, simplify, physical_index, synthesize, level_bound
from recursive import simulate
from radial_qft import radial_matrix

HERE=Path(__file__).resolve().parent
SHOTS=512; CAP=30; REP=.00025
ARMS={4:('gray','recursive','compact_exact'),5:('gray','recursive','compact_exact','compact_approx'),6:('recursive','compact_exact','compact_approx')}


def compile_at(b,arm,pos):
    g=gray_core(b,pos) if arm=='gray' else recursive_core(b,pos,arm)
    logical=simplify(g);routed=simplify(old.route(mapped(logical,pos)))
    return old.count_cx(logical),old.count_cx(routed),routed


@lru_cache(None)
def common_layout(b):
    from itertools import permutations
    opt={arm:synthesize(b,arm) for arm in ARMS[b]}
    layouts=permutations(range(b)) if b<=5 else sorted({tuple(r['pos']) for r in opt.values()}|{tuple(range(b))})
    best=None
    for pos in layouts:
        c={arm:compile_at(b,arm,pos) for arm in ARMS[b]}
        # Minimize worst relative routed count, then summed relative count.
        # No device calibration, shots or measured outcomes enter this choice.
        ratios=[c[a][1]/opt[a]['line_cx'] for a in ARMS[b]]
        key=(max(ratios),sum(ratios),pos)
        if best is None or key<best[0]:best=(key,c)
    return best[0][2],best[1],opt


@lru_cache(None)
def plan(b,replicate=0):
    old.must(b in ARMS and replicate>=0,'unsupported stage')
    d=1<<b;n=d-1;pos,cs,opt=common_layout(b);ix=physical_index(b,pos);I=np.eye(d,dtype=complex);Q=radial_matrix(n);rows=[];resources={}
    for arm,(logical,line,g) in cs.items():
        u=simulate(g,b,I[:,ix])[ix,:];bound=sum(level_bound(r) for r in range(4,b)) if arm=='compact_approx' else 0.
        err=float(np.linalg.norm(u-Q,ord=2));old.must(err<=bound+1e-10,'operator error bound')
        resources[arm]=dict(logical_cx=logical,line_cx=line,own_best_line_cx=opt[arm]['line_cx'],operator_error=err,analytic_bound=bound)
    def add(id,arm,probe,state,g,ideal,compiled,target):
        p=abs(ideal)**2;pc=abs(compiled)**2
        old.must(abs(p.sum()-1)<1e-10 and abs(pc.sum()-1)<1e-10,'normalization')
        rows.append(dict(id=id,arm=arm,probe=probe,state=state,basis='Z',target=int(target),ideal=p.tolist(),compiled_ideal=pc.tolist(),gates=[list(x) for x in g],line_cx=old.count_cx(g)))
    probes=[('edge',0),('middle',d//2),('tail',d-2)]
    # The full unitary is independently verified for every arm above. The
    # barrier separates preparation from the input-independent core.
    for name,a in probes:
        k=n-a;target=ix[k]
        for state,phase in [('plus',0.),('minus',math.pi),('imag',math.pi/2),('basis0',None),('basis1',None)]:
            g=old.prepare(b,a+(state=='basis1'),a+1 if phase is not None else None,phase or 0.)
            g=mapped(g,pos);x=simulate(g,b,I[:,0]);ideal=(Q@x[ix])[np.argsort(ix)]
            for arm,(_,_,cg) in cs.items():
                compiled=simulate(cg,b,x);add(f'{arm}_{name}_{state}',arm,name,state,g+[('barrier',)]+cg,ideal,compiled,target)
            if state in ('plus','minus','imag'):
                readout=([('p',pos[-1],-math.pi/2)] if state=='imag' else [])+[('h',pos[-1])]
                y=simulate(readout,b,x);expected=ix[a+(state=='minus')];old.must(abs(y[expected])**2>1-1e-10,'prep control')
                add(f'prep_{name}_{state}','prep',name,state,g+[('barrier',)]+readout,y,y,expected)
    for k in sorted({0,n}|{n-a for _,a in probes}):
        g=mapped(old.prepare(b,k),pos);y=I[:,ix[k]];add(f'spam_{k}','spam',str(k),'basis',g,y,y,ix[k])
    for arm in ARMS[b]:
        for name,a in probes:
            r={v['state']:v for v in rows if v['arm']==arm and v['probe']==name};k=ix[n-a]
            contrast=r['minus']['ideal'][k]-r['plus']['ideal'][k]
            old.must(abs(contrast-(1 if a==0 else 1/math.sqrt(2)))<1e-10,'analytic contrast')
            avg=(np.array(r['basis0']['ideal'])+r['basis1']['ideal'])/2
            old.must(np.max(abs(avg-r['imag']['ideal']))<1e-10,'imaginary control')
    random.Random(271126+100*b+replicate).shuffle(rows)
    return dict(scope='PROSPECTIVE_PHASE_CONTRAST',b=b,N=str(2**n),replicate=replicate,shots=SHOTS,cap_seconds=CAP,pos=list(pos),resources=resources,rows=rows)


def qiskit_check(p):
    from qiskit.quantum_info import Statevector
    b=p['b'];order=[int(f'{i:0{b}b}'[::-1],2) for i in range(1<<b)]
    for row in p['rows']:
        got=Statevector.from_instruction(old.circuit(row,b,False)).probabilities()[order]
        old.must(np.max(abs(got-row['compiled_ideal']))<1e-10,'Qiskit logical mismatch')


def native_check(isa,path,row,b):
    # Simulate only active physical wires, not the full IBM device.
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    q=QuantumCircuit(b);q.global_phase=isa.global_phase
    for inst in isa.data:
        if inst.operation.name in ('barrier','delay','measure'):continue
        loc=[path.index(isa.find_bit(x).index) for x in inst.qubits]
        q.append(inst.operation,loc)
    order=[int(f'{i:0{b}b}'[::-1],2) for i in range(1<<b)]
    got=Statevector.from_instruction(q).probabilities()[order]
    old.must(np.max(abs(got-row['compiled_ideal']))<1e-9,'native circuit ideal mismatch')


def sources():
    return {**old.sources(),**{x:old.sha((HERE/x).read_bytes()) for x in ('compact.py','qpu02.py')}}


def estimate(rows,reset):return 2+sum(.1+SHOTS*(REP+max(.0001,r['duration_s']+reset)) for r in rows)


def build(out,b,backend_name,replicate,path,opt):
    h=old.hardware_helpers();p=plan(b,replicate);qiskit_check(p)
    service,backend=old.connection(h,backend_name);u=old.usage(service);old.enough(u,CAP)
    snap=h['calibration_snapshot'](backend,source='had_qft_qpu02');old.must(bool(snap['timestamp']),'missing calibration timestamp')
    if path is None:path,_=h['choose_line_calibrated'](backend,b)
    old.must(len(path)==b and len(set(path))==b,'invalid path')
    reset=max(float(backend.target['reset'][(q,)].duration) for q in path)
    old.must(math.isfinite(reset) and reset>=0,'invalid reset duration')
    out.mkdir(parents=True,exist_ok=False);(out/'circuits').mkdir();old.write(out/'calibration.json',snap);old.write(out/'usage_build.json',u)
    rows=[];isas=[]
    for row in p['rows']:
        isa=h['transpile_on_path'](backend,old.circuit(row,b),path,optimization_level=opt)
        mapping=h['audit_mapping'](isa,path,b);native_check(isa,path,row,b)
        data=h['qasm3_text'](isa).encode();(out/'circuits'/f"{row['id']}.qasm").write_bytes(data)
        rows.append({**row,'mapping':mapping,'metrics':h['isa_metrics'](isa),'duration_s':h['duration'](isa,backend.target),'qasm_sha256':old.sha(data)});isas.append(isa)
    qpy=h['qpy_bytes'](isas);(out/'compiled.qpy').write_bytes(qpy)
    m=dict(backend=backend_name,b=b,replicate=replicate,path=path,optimization_level=opt,shots=SHOTS,cap_seconds=CAP,compiled_at=old.now(),plan_digest=old.digest(p),sources=sources(),calibration_sha256=old.sha((out/'calibration.json').read_bytes()),qpy_sha256=old.sha(qpy),reset_duration=reset,estimated_qpu_seconds=estimate(rows,reset),resources=p['resources'],pos=p['pos'],rows=rows)
    m['plan_sha256']=old.digest(m);old.write(out/'compiled.json',m)
    old.must(m['estimated_qpu_seconds']<=CAP,'estimate exceeds proposed ceiling')
    print(json.dumps({k:m[k] for k in ('backend','b','replicate','path','optimization_level','plan_sha256','estimated_qpu_seconds','cap_seconds')},indent=2));print('Build only; no job submitted.')


def validate(out,h):
    from qiskit import qpy
    m=old.read(out/'compiled.json');k=m.pop('plan_sha256');old.must(old.digest(m)==k,'manifest mismatch');m['plan_sha256']=k
    p=plan(m['b'],m['replicate']);old.must(m['plan_digest']==old.digest(p) and m['sources']==sources(),'plan/source changed')
    old.must(m['shots']==SHOTS and m['cap_seconds']==CAP and m['estimated_qpu_seconds']<=CAP,'budget mismatch')
    old.must(m['calibration_sha256']==old.sha((out/'calibration.json').read_bytes()),'calibration hash mismatch')
    q=(out/'compiled.qpy').read_bytes();old.must(m['qpy_sha256']==old.sha(q),'QPY mismatch');isas=qpy.load(io.BytesIO(q))
    old.must(len(isas)==len(m['rows'])==len(p['rows']),'roster mismatch')
    for r,s,isa in zip(m['rows'],p['rows'],isas):
        old.must(all(r[k]==v for k,v in s.items()),'row changed')
        old.must(r['mapping']==h['audit_mapping'](isa,m['path'],m['b']) and r['metrics']==h['isa_metrics'](isa),'mapping/metrics changed')
        old.must(r['qasm_sha256']==old.sha(h['qasm3_text'](isa).encode())==old.sha((out/'circuits'/f"{r['id']}.qasm").read_bytes()),'QASM mismatch')
    old.must(abs(m['estimated_qpu_seconds']-estimate(m['rows'],m['reset_duration']))<1e-9,'estimate changed');return m,isas


def submit(out,authorization):
    h=old.hardware_helpers();m,isas=validate(out,h)
    ok,why=h['check_authorization'](authorization,backend=m['backend'],sha=m['plan_sha256'],estimated_seconds=m['estimated_qpu_seconds']);old.must(ok,why)
    a=old.read(Path(authorization));old.must(a['max_qpu_seconds']<=CAP and a.get('checked_current_allocation') is True and a.get('acknowledge_estimate_not_hard_limit') is True,'owner acknowledgments required')
    age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(m['compiled_at'])).total_seconds();old.must(0<=age<=900,'build older than 15 minutes')
    service,backend=old.connection(h,m['backend']);snap=h['calibration_snapshot'](backend,source='had_qft_qpu02');old.must(snap==old.read(out/'calibration.json'),'calibration changed')
    old.must(max(float(backend.target['reset'][(q,)].duration) for q in m['path'])==m['reset_duration'],'reset duration changed')
    for r,isa in zip(m['rows'],isas):old.must(abs(h['duration'](isa,backend.target)-r['duration_s'])<1e-12,'duration changed')
    before=old.usage(service);old.enough(before,a['max_qpu_seconds']);old.write(out/'usage_before.json',before);old.write(out/'calibration_submission.json',snap)
    with (out/'submission_started.json').open('x') as f:json.dump({'plan_sha256':m['plan_sha256'],'at':old.now()},f)
    from qiskit_ibm_runtime import SamplerV2
    job=SamplerV2(mode=backend,options={'execution':{'rep_delay':REP}}).run(isas,shots=SHOTS)
    old.write(out/'job.json',{'job_id':job.job_id(),'plan_sha256':m['plan_sha256'],'submitted_at':old.now()});print('Submitted job:',job.job_id(),flush=True)


def analyze(out):
    m=old.read(out/'compiled.json');counts=old.read(out/'counts.json');b=m['b'];n=m['shots'];rows={};probs={}
    old.must(set(counts)=={r['id'] for r in m['rows']},'count roster mismatch')
    for r in m['rows']:
        c=counts[r['id']];old.must(sum(c.values())==n and all(len(k)==b and set(k)<=set('01') and isinstance(v,int) and v>=0 for k,v in c.items()),'invalid counts')
        p=np.zeros(1<<b)
        for k,v in c.items():p[int(k[::-1],2)]+=v/n
        q=np.array(r['ideal']);qc=np.array(r['compiled_ideal']);probs[r['id']]=p
        rows[r['id']]=dict(target_prob=float(p[r['target']]),ideal_target=float(q[r['target']]),tv=float(abs(p-q).sum()/2),tv_compiled=float(abs(p-qc).sum()/2),hf=float(np.sqrt(p*q).sum()**2))
    contrasts=[];radius=math.sqrt(math.log(40)/n);diff_radius=math.sqrt(2*math.log(40)/n)
    for arm in ARMS[b]:
        for probe in ('edge','middle','tail'):
            id=lambda state:f'{arm}_{probe}_{state}';x=rows[id('minus')];y=rows[id('plus')];c=x['target_prob']-y['target_prob'];ideal=x['ideal_target']-y['ideal_target']
            mix=(probs[id('basis0')]+probs[id('basis1')])/2;mid=(probs[id('plus')]+probs[id('minus')])/2
            contrasts.append(dict(arm=arm,probe=probe,contrast=c,ideal=ideal,normalized=c/ideal,shot_hoeffding95=[max(-1,c-radius),min(1,c+radius)],midpoint_vs_mix_tv=float(abs(mid-mix).sum()/2),imag_vs_mix_tv=float(abs(probs[id('imag')]-mix).sum()/2)))
    comparisons=[]
    for r in contrasts:
        if r['arm']=='recursive':continue
        base=next(v for v in contrasts if v['arm']=='recursive' and v['probe']==r['probe']);d=r['contrast']-base['contrast'];comparisons.append(dict(arm=r['arm'],probe=r['probe'],difference=d,shot_hoeffding95=[max(-2,d-diff_radius),min(2,d+diff_radius)]))
    report=dict(scope='EXPLORATORY_PHASE_CONTRAST_SHOT_ONLY',contrasts=contrasts,comparisons=comparisons,rows=rows,resources=m['resources'],note='Edge contrast is primary; other probes secondary. Marginal Hoeffding intervals assume independent fixed-probability shots, exclude drift and systematics, and are not multiplicity-adjusted.')
    old.write(out/'analysis.json',report);print(json.dumps(contrasts,indent=2))


def collect(out):
    h=old.hardware_helpers();m,_=validate(out,h);j=old.read(out/'job.json');old.must(j['plan_sha256']==m['plan_sha256'],'job manifest mismatch')
    service,_=old.connection(h,m['backend']);job=service.job(j['job_id']);res=job.result();old.must(len(res)==len(m['rows']),'PUB roster mismatch')
    counts={r['id']:{k:int(v) for k,v in e.data.meas.get_counts().items()} for r,e in zip(m['rows'],res)}
    if (out/'counts.json').exists():old.must(counts==old.read(out/'counts.json'),'recollection differs')
    old.write(out/'counts.json',counts);after=old.usage(service);old.write(out/'usage_after.json',after)
    try:ju=job.usage()
    except Exception:ju=None
    old.write(out/'execution_receipt.json',dict(job_id=j['job_id'],plan_sha256=m['plan_sha256'],usage_before=old.read(out/'usage_before.json'),usage_after=after,job_usage=ju,collected_at=old.now()));analyze(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['logical','qiskit-check','build','submit','collect','analyze']);p.add_argument('--b',type=int,choices=[4,5,6],default=5);p.add_argument('--replicate',type=int,default=0);p.add_argument('--backend',default='ibm_kingston');p.add_argument('--path',type=int,nargs='+');p.add_argument('--opt',type=int,choices=[0,1],default=1);p.add_argument('--out',type=Path,required=True);p.add_argument('--authorization');a=p.parse_args()
    if a.command in ('logical','qiskit-check'):
        x=plan(a.b,a.replicate)
        if a.command=='qiskit-check':qiskit_check(x)
        a.out.mkdir(parents=True,exist_ok=True);old.write(a.out/'logical_plan.json',x);print(json.dumps({'b':a.b,'circuits':len(x['rows']),'shots':SHOTS,'resources':x['resources'],'pos':x['pos'],'nominal_estimate_s':estimate([{'duration_s':0} for _ in x['rows']],0)},indent=2))
    elif a.command=='build':build(a.out,a.b,a.backend,a.replicate,a.path,a.opt)
    elif a.command=='submit':submit(a.out,a.authorization)
    elif a.command=='collect':collect(a.out)
    else:analyze(a.out)
if __name__=='__main__':main()
