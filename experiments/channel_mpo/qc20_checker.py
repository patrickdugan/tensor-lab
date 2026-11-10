"""QC-20 packed-core memory-mapped, independently checked witness verifier.

A read-only framed event tape keeps one immutable, OS-shared backing file. Each
worker constructs only the event currently being checked, and keeps an exact
arithmetic verifier plus its current MPO. Every local transformation is still
checked against a separately derived circuit schedule, full boundary states,
and exact dyadic error inequalities. The verifier imports no simulation code.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parent))
from qc15_checker import (CertificateError,require,cint,read_proof,read_tensor,
                          _canonical_bytes,realhex,checked_chain,initial_core,
                          MAX_EVENTS,MAX_TOTAL_WITNESS_ENTRIES)
from qc20_packed import (StreamPackedKernel, StreamFastKernel, read_packed_tensor, pack_tensor, boundary_digest_packed, read_checkpoint_packed)
from qc17_checker import prepare_circuit,_check_genesis,_check_final
from qc17_format import (read_checkpoint,CHECKPOINT_PATTERN,SHA)
from qc18_tape import SCHEMA18,TapeEvents

_STATE=None


def load_root(path,*,corrections=False):
    path=Path(path)
    manifest=read_proof(path)
    require(isinstance(manifest,dict) and set(manifest)=={
        'format','circuit','result','segments','checkpoints','tape'},
        'malformed QC18 manifest')
    require(manifest['format']==SCHEMA18,'QC18 manifest required')
    source=TapeEvents(manifest,path.parent)
    checkpoints=manifest['checkpoints']
    require(isinstance(checkpoints,list) and len(checkpoints)==len(source.segments)+1,
            'checkpoint count does not match segments')
    require(len(checkpoints)<=MAX_EVENTS+65,'too many checkpoints')
    seen=set()
    for j,desc in enumerate(checkpoints):
        require(isinstance(desc,dict) and set(desc)=={
            'filename','event_index','commitment','file_sha256'},
            'invalid checkpoint description')
        name=desc['filename']
        require(isinstance(name,str) and CHECKPOINT_PATTERN.fullmatch(name) and
                name not in seen,'unsafe or repeated checkpoint path')
        seen.add(name)
        pos=0 if j==0 else (source.segments[j-1]['start_index']+
                              source.segments[j-1]['count'])
        require(type(desc['event_index']) is int and desc['event_index']==pos,
                'nonconsecutive checkpoint event indices')
        expected=(source.segments[0]['start_commitment'] if j==0 else
                  source.segments[j-1]['end_commitment'])
        require(desc['commitment']==expected,'checkpoint/segment linkage mismatch')
        if j<len(source.segments):
            require(desc['commitment']==source.segments[j]['start_commitment'],
                    'intermediate checkpoint linkage mismatch')
    plan=prepare_circuit(manifest['circuit'],manifest['result'],
                         compute_corrections=corrections)
    require(len(plan['schedule'])==len(source),'circuit/schedule event count mismatch')
    return manifest,source,checkpoints,plan


# Worker state consists only of read-only mmap/source, circuit metadata and
# finite checkpoint descriptors. Each event is decoded into temporary objects.
def _initialize(path,expected_digest,storage_mode="fast"):
    global _STATE
    if _STATE is not None and _STATE['path']==str(path) and _STATE['digest']==expected_digest and _STATE['storage_mode']==storage_mode:
        return
    manifest,source,checkpoints,plan=load_root(path,corrections=True)
    digest=hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    require(digest==expected_digest,'manifest changed during worker initialization')
    _STATE={'path':str(path),'digest':digest,'manifest':manifest,
            'source':source,'checkpoints':checkpoints,'plan':plan,'storage_mode':storage_mode}


def verify_one_segment(source,checkpoints,plan,directory,i,storage_mode="fast"):
    """Local conditional transition verification, anchored globally by verify()."""
    rec=source.segments[i]
    point=read_checkpoint_packed(directory,checkpoints[i])
    end_point=read_checkpoint_packed(directory,checkpoints[i+1])
    require(point['commitment']==rec['start_commitment'] and
            end_point['commitment']==rec['end_commitment'],
            'segment input or output commitment mismatch')
    end_commitment=end_point['commitment']
    del end_point  # Full endpoint validated and its MPO released before replay.
    cores=point.pop('cores')
    # Packed source cores remain immutable throughout each local transition.
    first,last=point['first'],point['last']
    tally=dict(point['tally'])
    acc=point['accumulated']
    changes=0
    require(storage_mode in ("fast","compact","summary"), "invalid storage mode")
    if storage_mode == 'summary':
        from qc21_kernel import SummaryKernel
        kernel = SummaryKernel()
    else:
        kernel=StreamFastKernel() if storage_mode=="fast" else StreamPackedKernel()
    # Every segment checks the same physical-map correction preconditions.
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
            cores[:0]=[pack_tensor(initial_core('I')) for _ in range(first-low)]
            cores.extend(pack_tensor(initial_core('I')) for _ in range(high-last))
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
        before_changed=tuple(cores[j] for j in indices)
        after=list(cores)
        for j,encoded in zip(indices,ev['new_cores']):
            require(0<=j<len(after),'core index out of range')
            after[j]=read_packed_tensor(encoded)
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
        for retired in before_changed:
            kernel.retire(retired)  # All checks of these tensor versions are complete.
        cores=after
        tally[kind]+=1
    # Recompute commitment from independently checked exact state.
    digest=boundary_digest_packed(end,first,last,cores,acc,tally)
    require(digest==rec['end_commitment'],'computed segment output boundary mismatch')
    require(digest==end_commitment,'segment output checkpoint mismatch')
    return {'segment':i,'audited_events':sum(tally.values())-sum(point['tally'].values()),
            'changed_entries':changes,'end_commitment':digest}

def _worker_segment(i):
    state=_STATE
    require(state is not None,'uninitialized QC20 verifier worker')
    return verify_one_segment(state['source'],state['checkpoints'],state['plan'],
                              Path(state['path']).parent,i,state['storage_mode'])


def verify(manifest_path,workers=1,*,start_method='auto',storage_mode='fast'):
    """Independent audit with bounded-memory mmap-backed event replay.

    'spawn' works across platforms; 'fork' (POSIX) can share initialized
    circuit metadata by copy-on-write, but workers still independently replay.
    """
    require(storage_mode in ('fast','compact','summary'),'invalid storage mode')
    path=Path(manifest_path).resolve()
    workers=cint(workers,1,16)
    require(start_method in ('auto','spawn','fork'),'invalid start method')
    if start_method=='auto':
        start_method='fork' if os.name=='posix' else 'spawn'
    manifest,source,checkpoints,plan=load_root(path,corrections=False)
    digest=hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    if storage_mode == 'summary':
        from qc21_anchor import check_genesis, check_final
        begin=read_checkpoint_packed(path.parent,checkpoints[0])
        check_genesis(begin,plan)
        final=read_checkpoint_packed(path.parent,checkpoints[-1])
        result=check_final(final,plan,manifest['result'])
        del begin, final
        # Each of the scheduled segment workers independently loads, hashes,
        # validates and replays both endpoint checkpoints; together they cover
        # every intermediate checkpoint twice. There is no need to decode a
        # second in-memory copy in the coordinator before launching workers.
        # The combined verifier requires *every* scheduled segment to pass.
    else:
        begin=read_checkpoint(path.parent,checkpoints[0])
        _check_genesis(begin,plan)
        final=read_checkpoint(path.parent,checkpoints[-1])
        result=_check_final(final,plan,manifest['result'])
        del begin, final  # Release decoded boundary tensors before forking workers.
        for desc in checkpoints[1:-1]:
            read_checkpoint(path.parent,desc)
    count=len(source.segments)
    if workers==1:
        global _STATE
        corrected=prepare_circuit(manifest['circuit'],manifest['result'],
                                  compute_corrections=True)
        _STATE={'path':str(path),'digest':digest,'manifest':manifest,
                'source':source,'checkpoints':checkpoints,'plan':corrected,'storage_mode':storage_mode}
        checks=[_worker_segment(i) for i in range(count)]
    else:
        if start_method=='fork':
            # Do circuit decoding and polar-correction audits ONCE before the
            # fork. All children inherit the read-only plan and tape mapping
            # with copy-on-write backing. Exact kernel temporaries stay local.
            corrected=prepare_circuit(manifest['circuit'],manifest['result'],
                                      compute_corrections=True)
            _STATE={'path':str(path),'digest':digest,'manifest':manifest,
                    'source':source,'checkpoints':checkpoints,'plan':corrected,'storage_mode':storage_mode}
        # Safe worker startup on Windows and on Linux with active parent threads.
        context=multiprocessing.get_context(start_method)
        with ProcessPoolExecutor(max_workers=min(workers,count),mp_context=context,
                                 initializer=_initialize,initargs=(path,digest,storage_mode)) as pool:
            checks=list(pool.map(_worker_segment,range(count),chunksize=1))
    require(len(checks)==count and
            sum(x['audited_events'] for x in checks)==result['events_checked'],
            'not all scheduled transitions were checked')
    total=sum(x['changed_entries'] for x in checks)
    require(total<=MAX_TOTAL_WITNESS_ENTRIES,'whole proof entry limit exceeded')
    return {**result,'proof_level':('qc21-summary-cached-checkpointed-full-witness'
                                      if storage_mode=='summary' else
                                      'qc20-packed-dyadic-checkpointed-full-witness'),
            'segments_checked':count,'checkpoints_checked':len(checkpoints),
            'witness_complex_entries':total,'workers':min(workers,count),
            'arithmetic_backend':('exact-dyadic-context-summary' if storage_mode=='summary'
                          else ('exact-dyadic-streamed-pair-fast' if storage_mode=='fast'
                          else 'exact-dyadic-streamed-pair-packed-int')),
            'transport':'read-only-mmap-packed-binary64-one-event-at-a-time',
            'storage_mode':storage_mode}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--start-method',choices=('auto','spawn','fork'),default='auto')
    p.add_argument('--storage-mode',choices=('fast','compact','summary'),default='fast')
    a=p.parse_args()
    try:
        result=verify(a.manifest,workers=a.workers,start_method=a.start_method,storage_mode=a.storage_mode)
    except (CertificateError,ValueError,TypeError,OverflowError,MemoryError,
            OSError,KeyError,IndexError) as exc:
        p.exit(2,'FAIL: '+str(exc)+'\n')
    print(json.dumps({'status':'PASS',**result},sort_keys=True))

if __name__=='__main__':main()
