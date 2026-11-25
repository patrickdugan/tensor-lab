"""QC-16 segmented full-witness containers with checked state-boundary digests.

A manifest and ordered, individually checksummed event-segment files replace a
single large JSON record. The boundary digest commits to the exact stored MPO
cores, support coordinates, cumulative *dyadic* error and audited event counts.

Producer computes commitments by replaying the claimed changes without checking
arithmetic; the independent verifier checks arithmetic and recomputes every
boundary from the verified state. A segment in isolation is NOT a proof from
first principles: all segments must link to the initialized circuit witness.
"""
from __future__ import annotations
import hashlib
import re
import struct
import gzip
import io
import json
from fractions import Fraction as F
from pathlib import Path

from qc15_checker import (SCHEMA, CertificateError, require, read_proof,
                          write_proof, realhex, read_tensor, initial_core,
                          checked_chain, MAX_EVENTS, MAX_JSON_BYTES,
                          _json_no_duplicates, _canonical_bytes)

SCHEMA16="qc16-segmented-full-witness-v1"
SEGMENT_SCHEMA="qc16-event-segment-v1"
SEGMENT_PATTERN=re.compile(r"segment_[0-9]{4,5}\.json\.gz\Z")


def _int_hash(h, value:int):
    sign=int(value<0)
    mag=abs(value)
    data=mag.to_bytes(max(1,(mag.bit_length()+7)//8),'big')
    h.update(bytes([sign]))
    h.update(len(data).to_bytes(4,'big'))
    h.update(data)


def boundary_digest(index,first,last,cores,accumulated,tally):
    """SHA-256 of the reconstructed boundary, with length-delimited fields."""
    h=hashlib.sha256(b'qc16-state-boundary-v1\x00')
    for x in (index,first,last,len(cores)):
        _int_hash(h,int(x))
    _int_hash(h,accumulated.numerator)
    _int_hash(h,accumulated.denominator)
    for kind in ('channel','left_qr','right_qr','gate_svd'):
        _int_hash(h,int(tally[kind]))
    for core in cores:
        h.update(bytes([len(core.shape)]))
        for n in core.shape:
            _int_hash(h,n)
        for re,im in core.vals:
            h.update(struct.pack('<dd',float(re),float(im)))
    return h.hexdigest()


def boundary_generator(proof, cut_indices=None):
    """Producer-side claim replay; emit digests only at requested boundaries."""
    cuts=None if cut_indices is None else set(cut_indices)
    circuit=proof['circuit']
    first,last=circuit['targets'][0],circuit['targets'][-1]
    cores=[initial_core(k) for k in circuit['pauli']]
    acc=F(0)
    tally={k:0 for k in ('channel','left_qr','right_qr','gate_svd')}
    if cuts is None or 0 in cuts:
        yield 0,boundary_digest(0,first,last,cores,acc,tally)
    for index,ev in enumerate(proof['events'],1):
        kind=ev['kind']
        if kind=='grow':
            low,high=ev['first'],ev['last']
            require(low<=first and high>=last,'grow shrinks support')
            cores[:0]=[initial_core('I') for _ in range(first-low)]
            cores.extend(initial_core('I') for _ in range(high-last))
            first,last=low,high
        else:
            indices=ev['indices']
            for j,enc in zip(indices,ev['new_cores']):
                cores[j]=read_tensor(enc)
            r=realhex(ev['residual_upper'],nonnegative=True)
            g=realhex(ev['context_upper'],nonnegative=True)
            alpha=realhex(ev['polar_upper'],nonnegative=True)
            acc+=r*g+alpha*(1+acc)
            tally[kind]+=1
        checked_chain(cores)
        if cuts is None or index in cuts:
            yield index,boundary_digest(index,first,last,cores,acc,tally)


def split_witness(proof,output_dir:Path,*,events_per_segment:int=64):
    require(proof.get('format')==SCHEMA,'full QC15 proof required')
    require(type(events_per_segment) is int and 1<=events_per_segment<=1000,'invalid segment size')
    events=proof.get('events')
    require(isinstance(events,list) and 0<len(events)<=MAX_EVENTS+64,'invalid event count')
    directory=Path(output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    marks={i for i in range(0,len(events)+1,events_per_segment)}|{len(events)}
    hashes=dict(boundary_generator(proof,marks))
    entries=[]
    for part_i,start in enumerate(range(0,len(events),events_per_segment)):
        end=min(start+events_per_segment,len(events))
        name=f'segment_{part_i:04d}.json.gz'
        entry={'filename':name,'start_index':start,'count':end-start,
               'start_commitment':hashes[start],'end_commitment':hashes[end]}
        payload={'format':SEGMENT_SCHEMA,**entry,'events':events[start:end]}
        path=directory/name
        write_proof(payload,path)
        entry['file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(entry)
    manifest={'format':SCHEMA16,'circuit':proof['circuit'],
              'result':proof['result'],'segments':entries}
    manifest_path=directory/'manifest.json'
    write_proof(manifest,manifest_path)
    return manifest_path


class SegmentEvents:
    """Sequential, one-segment-at-a-time random-access protocol for checker."""
    def __init__(self,manifest:dict,directory:Path):
        self.directory=Path(directory)
        self.segments=manifest['segments']
        require(isinstance(self.segments,list) and self.segments,'empty segment manifest')
        require(len(self.segments)<=MAX_EVENTS+64,'too many segments')
        pos=0
        seen=set()
        self.boundaries={}
        for rec in self.segments:
            require(isinstance(rec,dict) and set(rec)=={
                'filename','start_index','count','start_commitment','end_commitment','file_sha256'},
                'invalid segment metadata')
            name=rec['filename']
            require(isinstance(name,str) and SEGMENT_PATTERN.fullmatch(name) is not None
                    and name not in seen,'unsafe or duplicate segment filename')
            seen.add(name)
            require(type(rec['start_index']) is int and rec['start_index']==pos,
                    'nonconsecutive segment index')
            count=rec['count']
            require(type(count) is int and 1<=count<=1000,'invalid segment count')
            for field in ('start_commitment','end_commitment','file_sha256'):
                value=rec[field]
                require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value),
                        'invalid digest')
            if pos in self.boundaries:
                require(self.boundaries[pos]==rec['start_commitment'],
                        'disconnected boundaries')
            else:
                self.boundaries[pos]=rec['start_commitment']
            pos+=count
            self.boundaries[pos]=rec['end_commitment']
        require(pos<=MAX_EVENTS+64,'too many total events')
        self.count=pos
        self.cached_part=None
        self.cached_events=None

    def __len__(self):return self.count

    def __getitem__(self,index):
        require(type(index) is int and 0<=index<self.count,'event index out of range')
        # Called strictly in ascending event order by the QC15 scheduler.
        if self.cached_part is not None:
            rec=self.segments[self.cached_part]
            if rec['start_index']<=index<rec['start_index']+rec['count']:
                return self.cached_events[index-rec['start_index']]
        part=next((i for i,v in enumerate(self.segments)
                   if v['start_index']<=index<v['start_index']+v['count']),None)
        require(part is not None,'event gap')
        rec=self.segments[part]
        path=self.directory/rec['filename']
        require(path.is_file() and not path.is_symlink(),'missing or symlinked segment')
        require(path.stat().st_size<=24_000_000,'segment file size cap')
        # Hash and parse ONE immutable byte snapshot, so a concurrent file
        # replacement cannot race the file-integrity check and JSON decode.
        compressed=path.read_bytes()
        require(hashlib.sha256(compressed).hexdigest()==rec['file_sha256'],
                'segment file checksum mismatch')
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(compressed),mode='rb') as stream:
                raw=stream.read(MAX_JSON_BYTES+1)
            require(len(raw)<=MAX_JSON_BYTES,'segment decompression limit')
            payload=json.loads(raw,object_pairs_hook=_json_no_duplicates,
                               parse_constant=lambda x: (_ for _ in ()).throw(
                                   CertificateError('nonfinite JSON')))
        except (OSError,EOFError,UnicodeError,json.JSONDecodeError) as exc:
            raise CertificateError('unreadable segment') from exc
        require(isinstance(payload,dict),'segment must be a JSON object')
        checksum=payload.pop('sha256',None)
        require(isinstance(checksum,str) and len(checksum)==64 and
                hashlib.sha256(_canonical_bytes(payload)).hexdigest()==checksum,
                'segment canonical JSON checksum mismatch')
        require(isinstance(payload,dict) and set(payload)=={
            'format','filename','start_index','count',
            'start_commitment','end_commitment','events'},'bad segment record fields')
        for field in ('filename','start_index','count','start_commitment','end_commitment'):
            require(payload[field]==rec[field],'segment metadata mismatch')
        require(payload['format']==SEGMENT_SCHEMA and isinstance(payload['events'],list)
                and len(payload['events'])==rec['count'],'wrong segment payload')
        self.cached_part=part
        self.cached_events=payload['events']
        return self.cached_events[index-rec['start_index']]


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('full_witness',type=Path,
                   help='QC-15 full witness, JSON or gzipped JSON')
    p.add_argument('--out-dir',required=True,type=Path)
    p.add_argument('--events-per-segment',type=int,default=64)
    a=p.parse_args()
    proof=read_proof(a.full_witness)
    manifest=split_witness(proof,a.out_dir,
                           events_per_segment=a.events_per_segment)
    print(manifest)


if __name__=='__main__':main()
