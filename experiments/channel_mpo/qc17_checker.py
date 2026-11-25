"""QC-17 independent checkpointed, optionally parallel full-witness checker.

All segments must be verified. Workers check exact dyadic local residuals and
schedule from checkpoint inputs; the coordinator anchors genesis and readout,
checks SHA-256 boundary continuity, and aggregates the witness-entry limits.
Checkpoints make segments independently *computable*, not independently sound.
The checker imports only the QC-15/16 standalone arithmetic implementation,
not the simulator or the QC-17 producer. Standard library only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parent))

from qc15_checker import (CertificateError,require,cint,read_proof,read_tensor,_canonical_bytes,
                          realhex,PAULIS,expected_cone,initial_core,checked_chain,
                          sqrt_up,MAX_EVENTS,MAX_TOTAL_WITNESS_ENTRIES)
from qc16_dyadic import Kernel
from qc16_segments import SegmentEvents
from qc17_format import (SCHEMA17,KINDS,CHECKPOINT_PATTERN,SHA,read_checkpoint)


def prepare_circuit(circuit, claimed, *, compute_corrections=True):
    require(isinstance(circuit,dict),'missing circuit')
    n=cint(circuit.get('qubits'),2,64)
    targets=circuit.get('targets')
    require(isinstance(targets,list) and targets and
            targets==list(range(targets[0],targets[-1]+1)) and
            all(type(j) is int and 0<=j<n for j in targets),
            'invalid contiguous targets')
    label=circuit.get('pauli')
    require(isinstance(label,str) and len(label)==len(targets) and
            set(label)<=set(PAULIS),'invalid Pauli observable')
    raw_layers=circuit.get('layers')
    require(isinstance(raw_layers,list) and len(raw_layers)<=64,'invalid layer list')
    layers=[]
    for layer in raw_layers:
        require(isinstance(layer,list),'malformed circuit layer')
        parsed=[]
        for record in layer:
            require(isinstance(record,dict) and set(record)=={'site','gate'},
                    'invalid gate record')
            parsed.append((cint(record['site'],0,n-2),read_tensor(record['gate'])))
        layers.append(parsed)
    raw_ops=circuit.get('kraus')
    require(isinstance(raw_ops,list) and 1<=len(raw_ops)<=10,'bad Kraus set')
    operators=[read_tensor(t) for t in raw_ops]
    require(all(t.shape==(2,2) for t in operators),'bad Kraus shape')
    states=expected_cone(n,layers,targets)
    require(isinstance(claimed,dict) and
            cint(claimed.get('cone_width'),1,64)==len(states[0]),
            'cone width mismatch')

    schedule=[]
    first,last=targets[0],targets[-1]
    for t in range(len(layers)-1,-1,-1):
        after_state=states[t+1]
        for site in range(first,last+1):
            schedule.append(('channel',t,site,(site-first,),None))
        low,high=states[t][0],states[t][-1]
        schedule.append(('grow',t,low,high,None))
        first,last=low,high
        for site,gate in layers[t]:
            if site in after_state or site+1 in after_state:
                j=site-first
                require(0<=j<last-first,'gate outside cone')
                for k in range(j):
                    schedule.append(('left_qr',t,first+k,(k,k+1),None))
                for k in range(last-first,j+1,-1):
                    schedule.append(('right_qr',t,first+k,(k-1,k),None))
                schedule.append(('gate_svd',t,site,(j,j+1),gate))
    require(len(schedule)<=MAX_EVENTS+64,'circuit event count exceeds limit')
    # The independent QC15 scheduler is maintained separately. Differential
    # tests against QC16 catch any order discrepancy in this implementation.
    k=Kernel()
    if compute_corrections:
        d=sqrt_up(k.kraus_defect2(operators))
        require(d<F(1,2),'Kraus polar correction inapplicable')
        channel_alpha=d*(2+d)
        gate_alpha={}
        for t,layer in enumerate(layers):
            for _,gate in layer:
                defect=sqrt_up(k.gate_defect2(gate))
                require(defect<F(1,2),'gate polar correction inapplicable')
                gate_alpha[id(gate)]=defect*(2+defect)
    else:
        channel_alpha=F(0)
        gate_alpha={}
    return {'n':n,'targets':targets,'pauli':label,'schedule':schedule,
            'operators':operators,'channel_alpha':channel_alpha,
            'gate_alpha':gate_alpha,'cone_width':len(states[0]),'final_support':(states[0][0],states[0][-1])}


def load_root(manifest_path):
    path=Path(manifest_path)
    manifest=read_proof(path)
    require(manifest.get('format')==SCHEMA17,'QC17 checkpointed manifest required')
    require(set(manifest)=={'format','circuit','result','segments','checkpoints'},
            'malformed QC17 manifest')
    segment_source=SegmentEvents(manifest,path.parent)
    checkpoints=manifest['checkpoints']
    require(isinstance(checkpoints,list) and
            len(checkpoints)==len(segment_source.segments)+1,
            'checkpoint count does not match segments')
    require(len(checkpoints)<=MAX_EVENTS+65,'too many checkpoints')
    seen=set()
    for j,desc in enumerate(checkpoints):
        require(isinstance(desc,dict) and set(desc)=={
            'filename','event_index','commitment','file_sha256'},
            'invalid checkpoint descriptor')
        fname=desc['filename']
        require(isinstance(fname,str) and CHECKPOINT_PATTERN.fullmatch(fname) is not None
                and fname not in seen,'unsafe or duplicate checkpoint filename')
        seen.add(fname)
        pos=0 if j==0 else (segment_source.segments[j-1]['start_index']+
                              segment_source.segments[j-1]['count'])
        require(type(desc['event_index']) is int and desc['event_index']==pos,
                'checkpoint indices are discontinuous')
        commitment=(segment_source.segments[0]['start_commitment'] if j==0 else
                     segment_source.segments[j-1]['end_commitment'])
        require(desc['commitment']==commitment,'checkpoint/segment linkage broken')
        if j<len(segment_source.segments):
            require(desc['commitment']==segment_source.segments[j]['start_commitment'],
                    'adjacent segment linkage broken')
    plan=prepare_circuit(manifest['circuit'],manifest['result'],compute_corrections=False)
    require(len(plan['schedule'])==len(segment_source),'circuit/schedule event count mismatch')
    return manifest,segment_source,checkpoints,plan


def _check_genesis(point,plan):
    require(point['event_index']==0 and
            (point['first'],point['last'])==(plan['targets'][0],plan['targets'][-1]),
            'checkpoint genesis support mismatch')
    expected=[initial_core(ch) for ch in plan['pauli']]
    require(point['cores']==expected,'checkpoint genesis must be exact Pauli observable')
    require(point['accumulated']==0 and all(x==0 for x in point['tally'].values()),
            'checkpoint genesis has nonzero error or counts')


def _check_final(point,plan,claimed):
    require(point['event_index']==len(plan['schedule']),'wrong final event index')
    require((point['first'],point['last'])==plan['final_support'],
            'final support differs from circuit backward cone')
    k=Kernel()
    value=k.exact_readout(point['cores'])
    reported=realhex(claimed.get('expectation'))
    error=realhex(claimed.get('error_upper'),nonnegative=True)
    require(error>=point['accumulated']+abs(value[0]-reported),
            'end-to-end error bound understated')
    require(abs(value[1])<=error,'imaginary readout exceeds bound')
    require(claimed.get('event_counts')==point['tally'],'final event tallies disagree')
    require(sum(point['tally'].values())==sum(s[0]!='grow' for s in plan['schedule']),
            'incomplete audited-event count')
    return {'proof_level':'checkpointed-parallel-full-witness',
            'reported_expectation':float(reported),
            'verified_bound':float(error),
            'arithmetic_lower_requirement':float(point['accumulated']),
            'event_counts':point['tally'],
            'events_checked':sum(point['tally'].values()),
            'cone_width':plan['cone_width'],
            'target':'polar-corrected binary64 gate/Kraus circuit'}


def verify_one_segment(manifest_path,segment_number,expected_manifest_digest=None):
    """Verify a single chunk conditionally on its checkpoint input.

    A successful return is a local transition statement. Acceptance as a
    circuit certificate also requires verification of *all* chunks and both
    global anchors by verify().
    """
    path=Path(manifest_path)
    manifest,source,checkpoints,plan=load_root(path)
    if expected_manifest_digest is not None:
        require(hashlib.sha256(_canonical_bytes(manifest)).hexdigest()==
                expected_manifest_digest,'manifest changed during parallel verification')
    i=cint(segment_number,0,len(source.segments)-1)
    rec=source.segments[i]
    point=read_checkpoint(path.parent,checkpoints[i])
    end_point=read_checkpoint(path.parent,checkpoints[i+1])
    require(point['commitment']==rec['start_commitment'] and
            end_point['commitment']==rec['end_commitment'],
            'segment input or output commitment mismatch')
    cores=list(point['cores'])
    first,last=point['first'],point['last']
    tally=dict(point['tally'])
    acc=point['accumulated']
    changes=0
    kernel=Kernel()
    # Every segment checks the same physical-map correction preconditions.
    plan=prepare_circuit(manifest['circuit'],manifest['result'])
    start=rec['start_index']
    end=start+rec['count']
    for event_index in range(start,end):
        ev=source[event_index]
        kind,t,site,indices,gate=plan['schedule'][event_index]
        if kind=='grow':
            require(ev=={'kind':'grow','layer':t,'first':site,'last':indices},
                    'grow event differs from independent schedule')
            low,high=site,indices
            require(low<=first and high>=last,'grow shrinks support')
            cores[:0]=[initial_core('I') for _ in range(first-low)]
            cores.extend(initial_core('I') for _ in range(high-last))
            first,last=low,high
            checked_chain(cores)
            continue
        require(isinstance(ev,dict) and ev.get('kind')==kind and
                ev.get('layer')==t and ev.get('site')==site,
                'event kind/layer/site mismatch')
        require(ev.get('indices')==list(indices) and
                isinstance(ev.get('new_cores'),list) and
                len(ev['new_cores'])==len(indices),
                'changed core set differs from independent schedule')
        after=list(cores)
        for j,encoded in zip(indices,ev['new_cores']):
            require(0<=j<len(after),'core index out of range')
            after[j]=read_tensor(encoded)
            changes+=len(after[j].vals)
            require(changes<=MAX_TOTAL_WITNESS_ENTRIES,'segment witness entry cap')
        checked_chain(after)
        if kind=='channel':
            squared=kernel.channel_error2(cores[indices[0]],after[indices[0]],
                                          plan['operators'])
            needed=plan['channel_alpha']
        elif kind=='gate_svd':
            squared=kernel.pair_error2(cores[indices[0]],cores[indices[1]],
                                       after[indices[0]],after[indices[1]],gate)
            needed=plan['gate_alpha'][id(gate)]
        else:
            squared=kernel.pair_error2(cores[indices[0]],cores[indices[1]],
                                       after[indices[0]],after[indices[1]])
            needed=F(0)
        residual=realhex(ev.get('residual_upper'),nonnegative=True)
        context=realhex(ev.get('context_upper'),nonnegative=True)
        polar=realhex(ev.get('polar_upper'),nonnegative=True)
        require(residual*residual>=squared,'invalid exact local residual upper')
        require(context>=kernel.context_bound(after,indices,gate=(kind=='gate_svd')),
                'invalid context norm upper')
        require(polar>=needed,'invalid gate or Kraus polar correction')
        if kind not in ('gate_svd','channel'):
            require(polar==0,'QR event cannot claim polar correction')
        acc+=residual*context+polar*(1+acc)
        cores=after
        tally[kind]+=1
    from qc16_segments import boundary_digest
    from qc17_format import checkpoint_payload
    # Recompute commitment from independently checked exact state.
    digest=boundary_digest(end,first,last,cores,acc,tally)
    require(digest==rec['end_commitment'],'computed segment output boundary mismatch')
    require(digest==end_point['commitment'],'segment output checkpoint mismatch')
    return {'segment':i,'audited_events':sum(tally.values())-sum(point['tally'].values()),
            'changed_entries':changes,'end_commitment':digest}


def verify(manifest_path,workers=1):
    path=Path(manifest_path)
    manifest,segments,checkpoints,plan=load_root(path)
    first=read_checkpoint(path.parent,checkpoints[0])
    _check_genesis(first,plan)
    last=read_checkpoint(path.parent,checkpoints[-1])
    # Read/check every intermediate checkpoint before launching workers.
    for point in checkpoints[1:-1]:
        read_checkpoint(path.parent,point)
    result=_check_final(last,plan,manifest['result'])
    manifest_digest=hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    workers=cint(workers,1,16)
    count=len(segments.segments)
    if workers==1:
        checks=[verify_one_segment(path,i,manifest_digest) for i in range(count)]
    else:
        with ProcessPoolExecutor(max_workers=min(workers,count)) as pool:
            checks=list(pool.map(verify_one_segment,[path]*count,range(count),
                                 [manifest_digest]*count))
    require(len(checks)==count and
            sum(x['audited_events'] for x in checks)==result['events_checked'],
            'not all scheduled transitions were checked')
    total=sum(x['changed_entries'] for x in checks)
    require(total<=MAX_TOTAL_WITNESS_ENTRIES,
            'whole proof exceeds total witness entry cap')
    return {**result,'segments_checked':count,'checkpoints_checked':len(checkpoints),
            'witness_complex_entries':total,
            'workers':min(workers,count),
            'arithmetic_backend':'common-dyadic-integer'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path)
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    try:
        report=verify(a.manifest,workers=a.workers)
    except (CertificateError,ValueError,TypeError,OverflowError,MemoryError,
            OSError,KeyError,IndexError) as exc:
        p.exit(2,'FAIL: '+str(exc)+'\n')
    print(json.dumps({'status':'PASS',**report},sort_keys=True))

if __name__=='__main__':main()
