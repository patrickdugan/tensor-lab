"""QC-17 producer: split QC-15 full witness into checkpointed parallel segments.

Emission is claim replay, with no trust in its arithmetic. The independent
QC-17 verifier rechecks all events and anchors both ends. Producer-only module.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
from pathlib import Path

from qc15_checker import (SCHEMA,require,realhex,initial_core,read_tensor,
                          checked_chain,read_proof,write_proof)
from qc16_segments import split_witness,SCHEMA16
from qc17_format import SCHEMA17,KINDS,checkpoint_payload,write_checkpoint


def _snapshot_generator(proof, checkpoints):
    targets=proof['circuit']['targets']
    cores=[initial_core(label) for label in proof['circuit']['pauli']]
    first,last=targets[0],targets[-1]
    acc=F(0)
    tally={kind:0 for kind in KINDS}
    if 0 in checkpoints:
        yield checkpoint_payload(0,first,last,cores,acc,tally)
    for idx,ev in enumerate(proof['events'],1):
        kind=ev['kind']
        if kind=='grow':
            low,high=ev['first'],ev['last']
            require(low<=first and high>=last,'invalid grow claim')
            cores[:0]=[initial_core('I') for _ in range(first-low)]
            cores.extend(initial_core('I') for _ in range(high-last))
            first,last=low,high
        else:
            for pos,code in zip(ev['indices'],ev['new_cores']):
                cores[pos]=read_tensor(code)
            r=realhex(ev['residual_upper'],nonnegative=True)
            g=realhex(ev['context_upper'],nonnegative=True)
            alpha=realhex(ev['polar_upper'],nonnegative=True)
            acc+=r*g+alpha*(1+acc)
            tally[kind]+=1
        checked_chain(cores)
        if idx in checkpoints:
            yield checkpoint_payload(idx,first,last,cores,acc,tally)


def produce(full_witness: Path,out_dir: Path,events_per_segment:int=256):
    proof=read_proof(Path(full_witness))
    require(proof.get('format')==SCHEMA,'QC-15 full witness required')
    directory=Path(out_dir)
    directory.mkdir(parents=True,exist_ok=True)
    manifest_path=split_witness(proof,directory,events_per_segment=events_per_segment)
    manifest=read_proof(manifest_path)
    require(manifest['format']==SCHEMA16,'QC16 split failed')
    positions=[0]+[rec['start_index']+rec['count'] for rec in manifest['segments']]
    require(positions[-1]==len(proof['events']),'missing final event')
    desc=[]
    for i,payload in enumerate(_snapshot_generator(proof,set(positions))):
        require(payload['event_index']==positions[i],'checkpoint order mismatch')
        rec=write_checkpoint(directory,i,payload)
        require(rec['commitment']==(
            manifest['segments'][0]['start_commitment'] if i==0 else
            manifest['segments'][i-1]['end_commitment']),
            'producer checkpoint claim differs from QC16 boundary')
        desc.append(rec)
    require(len(desc)==len(positions),'wrong checkpoint count')
    manifest['format']=SCHEMA17
    manifest['checkpoints']=desc
    write_proof(manifest,manifest_path)
    return manifest_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('full_witness',type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    p.add_argument('--events-per-segment',type=int,default=256)
    a=p.parse_args()
    print(produce(a.full_witness,a.out_dir,a.events_per_segment))

if __name__=='__main__': main()
