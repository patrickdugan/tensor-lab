"""QC-18 read-only framed tensor-witness tape; standard library only.

Each length-prefixed frame contains one canonical JSON event. Immutable
read-only mmap pages are shared by the OS across verification workers. The
canonical source circuit and exact-boundary commitments remain the trust anchor;
checksums bind transported bytes and do not substitute for arithmetic replay.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct
import zlib
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parent))

from qc15_checker import (CertificateError,MAX_EVENTS,MAX_JSON_BYTES,
                          _canonical_bytes,_json_no_duplicates,read_proof,
                          write_proof,require)
from qc16_segments import SegmentEvents
from qc17_format import SCHEMA17

SCHEMA18='qc18-mmap-checkpointed-full-witness-v1'
MAGIC_RAW=b'QC18TAPE\x01'
MAGIC_ZLIB=b'QC18TAPE\x02'
MAGIC=MAGIC_ZLIB
TAPE_FILE='events.q18t'
MAX_TAPE_BYTES=256_000_000
MAX_EVENT_BYTES=2_000_000
HEX64=re.compile('[0-9a-f]{64}\\Z')


def _canonical_event(event):
    require(isinstance(event,dict),'event must be a JSON mapping')
    raw=_canonical_bytes(event)
    require(len(raw)<=MAX_EVENT_BYTES,'event exceeds tape frame limit')
    return raw


def emit(qc17_manifest:Path, out_dir:Path, *, compressed=True):
    """Producer-only conversion of QC-17 events; arithmetic never trusted here."""
    source=Path(qc17_manifest)
    old=read_proof(source)
    require(old.get('format')==SCHEMA17,'QC-17 manifest required')
    out=Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    segment_source=SegmentEvents(old,source.parent)
    count=len(segment_source)
    require(0<count<=MAX_EVENTS+64,'tape event count outside limit')
    index=bytearray()
    frames=[]
    segment_records=[]
    offset=0
    for old_segment in segment_source.segments:
        start=old_segment['start_index']
        stop=start+old_segment['count']
        digest=hashlib.sha256()
        for k in range(start,stop):
            raw=_canonical_event(segment_source[k])
            encoded=zlib.compress(raw,level=3) if compressed else raw
            frame=struct.pack('<I',len(encoded))+encoded
            index.extend(struct.pack('<QI',offset+4,len(encoded)))
            offset+=len(frame)
            digest.update(frame)
            frames.append(frame)
        segment_records.append({k:old_segment[k] for k in (
            'start_index','count','start_commitment','end_commitment')}
            | {'events_sha256':digest.hexdigest()})
    header=(MAGIC_ZLIB if compressed else MAGIC_RAW)+struct.pack('<I',count)+bytes(index)
    dest=out/TAPE_FILE
    with dest.open('wb') as handle:
        handle.write(header)
        for frame in frames:
            handle.write(frame)
    require(dest.stat().st_size<=MAX_TAPE_BYTES,'tape too large')
    for desc in old['checkpoints']:
        name=desc['filename']
        require(re.fullmatch(r'checkpoint_[0-9]{4,5}\.json\.gz',name),
                'unsafe checkpoint filename')
        (out/name).write_bytes((source.parent/name).read_bytes())
    with dest.open('rb') as handle:
        digest=hashlib.file_digest(handle,'sha256').hexdigest()
    manifest={'format':SCHEMA18,'circuit':old['circuit'],'result':old['result'],
              'segments':segment_records,'checkpoints':old['checkpoints'],
              'tape':{'filename':TAPE_FILE,'sha256':digest,'events':count,
                      'bytes':dest.stat().st_size,
                      'encoding':'zlib-event-v1' if compressed else 'raw-event-v1'}}
    path=out/'manifest.json'
    write_proof(manifest,path)
    return path


class TapeEvents:
    """Pinned read-only mmap with strict per-event canonical parse, O(1) access."""

    def __init__(self,manifest:dict,directory:Path):
        self.directory=Path(directory)
        tape=manifest.get('tape')
        require(isinstance(tape,dict) and set(tape)=={
            'filename','sha256','events','bytes','encoding'},'invalid tape metadata')
        require(tape['filename']==TAPE_FILE and
                isinstance(tape['sha256'],str) and HEX64.fullmatch(tape['sha256']),
                'invalid tape name or checksum')
        require(type(tape['events']) is int and 1<=tape['events']<=MAX_EVENTS+64,
                'invalid event count')
        require(type(tape['bytes']) is int and 20<=tape['bytes']<=MAX_TAPE_BYTES,
                'invalid tape byte count')
        self.segments=manifest.get('segments')
        require(isinstance(self.segments,list) and 0<len(self.segments)<=MAX_EVENTS+64,
                'invalid segments')
        count=tape['events']; pos=0; prev=None
        for rec in self.segments:
            require(isinstance(rec,dict) and set(rec)=={
                'start_index','count','start_commitment','end_commitment','events_sha256'},
                'invalid segment description')
            require(type(rec['start_index']) is int and rec['start_index']==pos,
                    'nonconsecutive segment index')
            require(type(rec['count']) is int and 1<=rec['count']<=1000,
                    'invalid segment count')
            for key in ('start_commitment','end_commitment','events_sha256'):
                require(isinstance(rec[key],str) and HEX64.fullmatch(rec[key]),
                        'invalid segment digest')
            require(prev is None or rec['start_commitment']==prev,
                    'segment linkage mismatch')
            prev=rec['end_commitment']; pos+=rec['count']
        require(pos==count,'segment events missing')
        require(tape['encoding'] in ('zlib-event-v1','raw-event-v1'),
                'unsupported tape encoding')
        self.compressed=tape['encoding']=='zlib-event-v1'
        self.count=count
        path=self.directory/TAPE_FILE
        require(path.is_file() and not path.is_symlink(),
                'missing or symlinked tape')
        flags=os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)
        fd=os.open(path,flags)
        self.handle=os.fdopen(fd,'rb')
        try:
            require(os.fstat(self.handle.fileno()).st_size==tape['bytes'],
                    'tape size mismatch')
            self.mm=mmap.mmap(self.handle.fileno(),0,access=mmap.ACCESS_READ)
            require(hashlib.sha256(self.mm).hexdigest()==tape['sha256'],
                    'tape checksum mismatch')
            expected_magic=MAGIC_ZLIB if self.compressed else MAGIC_RAW
            require(self.mm[:len(MAGIC)]==expected_magic and
                    struct.unpack_from('<I',self.mm,len(MAGIC))[0]==count,
                    'tape magic/count mismatch')
            table=len(MAGIC)+4
            data_start=table+count*12
            require(data_start<=len(self.mm),'truncated event table')
            entries=[]
            next_position=data_start
            for i in range(count):
                relative,length=struct.unpack_from('<QI',self.mm,table+i*12)
                absolute=data_start+relative
                require(0<length<=MAX_EVENT_BYTES and absolute==next_position+4,
                        'noncontiguous or oversized frame')
                require(absolute+length<=len(self.mm), 'out-of-bounds frame')
                require(struct.unpack_from('<I',self.mm,absolute-4)[0]==length,
                        'frame length mismatch')
                entries.append((absolute,length))
                next_position=absolute+length
            require(next_position==len(self.mm),'trailing tape data')
            self.entries=entries
            # Bound every segment to the exact framed bytes. Workers parse one
            # event at a time; this step streams a digest without JSON objects.
            for rec in self.segments:
                digest=hashlib.sha256()
                for i in range(rec['start_index'],rec['start_index']+rec['count']):
                    p,n=entries[i]
                    digest.update(self.mm[p-4:p+n])
                require(digest.hexdigest()==rec['events_sha256'],
                        'segment framed-data checksum mismatch')
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return self.count

    def __getitem__(self,index):
        require(type(index) is int and 0<=index<self.count,
                'event index out of range')
        pos,length=self.entries[index]
        data=self.mm[pos:pos+length]
        if self.compressed:
            try:
                decoder=zlib.decompressobj()
                raw=decoder.decompress(data,MAX_EVENT_BYTES+1)
                require(len(raw)<=MAX_EVENT_BYTES and decoder.eof and
                        not decoder.unused_data and not decoder.unconsumed_tail,
                        'truncated or oversized zlib event')
                raw+=decoder.flush()
                require(len(raw)<=MAX_EVENT_BYTES,'oversized decoded event')
            except zlib.error as exc:
                raise CertificateError('invalid compressed event frame') from exc
        else:
            raw=data
        try:
            obj=json.loads(raw,object_pairs_hook=_json_no_duplicates,
                           parse_constant=lambda _:(_ for _ in ()).throw(
                               CertificateError('invalid JSON constant')))
        except (ValueError,UnicodeError,json.JSONDecodeError) as exc:
            raise CertificateError('unparseable event frame') from exc
        require(_canonical_event(obj)==raw,'noncanonical event frame')
        return obj

    def close(self):
        mm=getattr(self,'mm',None)
        if mm is not None:
            mm.close()
            self.mm=None
        handle=getattr(self,'handle',None)
        if handle is not None:
            handle.close()
            self.handle=None

    def __del__(self):
        try:self.close()
        except Exception:pass


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('qc17_manifest',type=Path)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--raw-events',action='store_true')
    a=parser.parse_args()
    print(emit(a.qc17_manifest,a.out_dir,compressed=not a.raw_events))

if __name__=='__main__': main()
