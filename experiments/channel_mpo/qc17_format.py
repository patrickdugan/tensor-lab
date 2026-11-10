"""QC-17 checkpoint encoding and validation. Standard library only.

A checkpoint is a *claimed* MPO state. The whole proof is accepted only after
all event segments are independently checked from canonical Pauli genesis to
final bound and every adjacent boundary digest matches. SHA-256 is linkage,
not a proof that any detached checkpoint is true.
"""
from __future__ import annotations
import base64
import gzip
import io
import json
from fractions import Fraction as F
import hashlib
from pathlib import Path
import re
import struct

from qc15_checker import (CertificateError, Tensor, checked_chain, cint,
                          read_tensor, require, write_proof, MAX_EVENTS, MAX_JSON_BYTES,
                          _canonical_bytes, _json_no_duplicates)
from qc16_segments import boundary_digest

SCHEMA17 = 'qc17-checkpointed-full-witness-v1'
CHECKPOINT_SCHEMA = 'qc17-state-checkpoint-v1'
CHECKPOINT_PATTERN = re.compile(r'checkpoint_[0-9]{4,5}\.json\.gz\Z')
HEX = re.compile(r'-?(?:0|[1-9a-f][0-9a-f]*)\Z')
SHA = re.compile(r'[0-9a-f]{64}\Z')
KINDS = ('channel', 'left_qr', 'right_qr', 'gate_svd')


def encode_core(core: Tensor):
    """Canonicalize exact stored dyadics back to IEEE binary64 little endian."""
    raw = b''.join(struct.pack('<dd', float(re),float(im)) for re,im in core.vals)
    return {'shape':list(core.shape),'binary64_le':base64.b64encode(raw).decode('ascii')}


def encode_accumulated(acc):
    acc = F(acc)
    require(acc >= 0 and acc.denominator & (acc.denominator-1) == 0,
            'accumulator must be a nonnegative dyadic')
    return {'numerator_hex':format(acc.numerator,'x'),
            'denominator_exp':acc.denominator.bit_length()-1}


def decode_accumulated(obj):
    require(isinstance(obj,dict) and set(obj)=={'numerator_hex','denominator_exp'},
            'invalid accumulator encoding')
    s=obj['numerator_hex']
    require(isinstance(s,str) and 0<len(s)<=500_000 and HEX.fullmatch(s) is not None,
            'invalid accumulator numerator')
    exp=cint(obj['denominator_exp'],0,2_000_000)
    number=int(s,16)
    require(number>=0,'negative error accumulator')
    return F(number,1<<exp)


def checkpoint_payload(index, first, last, cores, accumulated, tally):
    digest=boundary_digest(index,first,last,cores,accumulated,tally)
    return {'format':CHECKPOINT_SCHEMA, 'event_index':index,
            'first':first, 'last':last, 'cores':[encode_core(c) for c in cores],
            'accumulated':encode_accumulated(accumulated),
            'tally':{k:int(tally[k]) for k in KINDS}, 'commitment':digest}


def _validate_checkpoint_payload(obj):
    require(isinstance(obj,dict) and set(obj)=={
        'format','event_index','first','last','cores','accumulated','tally','commitment'},
        'malformed checkpoint fields')
    require(obj['format']==CHECKPOINT_SCHEMA,'checkpoint format mismatch')
    index=cint(obj['event_index'],0,MAX_EVENTS+64)
    first=cint(obj['first'],0,63)
    last=cint(obj['last'],first,63)
    enc=obj['cores']
    require(isinstance(enc,list) and len(enc)==last-first+1,
            'checkpoint support mismatch')
    cores=[read_tensor(x) for x in enc]
    require(all(len(c.shape)==4 for c in cores),'checkpoint core shape mismatch')
    checked_chain(cores)
    acc=decode_accumulated(obj['accumulated'])
    tally=obj['tally']
    require(isinstance(tally,dict) and set(tally)==set(KINDS), 'checkpoint tally invalid')
    for k in KINDS: cint(tally[k],0,MAX_EVENTS)
    require(sum(tally.values())<=index,'checkpoint tally exceeds event index')
    digest=boundary_digest(index,first,last,cores,acc,tally)
    require(obj['commitment']==digest,'checkpoint MPO boundary digest mismatch')
    return {'event_index':index,'first':first,'last':last,'cores':cores,
            'accumulated':acc,'tally':dict(tally),'commitment':digest}


def read_checkpoint(directory: Path, desc: dict):
    require(isinstance(desc,dict) and set(desc)=={
        'filename','event_index','commitment','file_sha256'},
        'invalid checkpoint descriptor')
    filename=desc['filename']
    require(isinstance(filename,str) and CHECKPOINT_PATTERN.fullmatch(filename),
            'invalid checkpoint filename')
    for key in ('commitment','file_sha256'):
        require(isinstance(desc[key],str) and SHA.fullmatch(desc[key]),
                'invalid checkpoint hash')
    index=cint(desc['event_index'],0,MAX_EVENTS+64)
    path=Path(directory)/filename
    require(path.is_file() and not path.is_symlink(), 'missing or symlinked checkpoint')
    require(path.stat().st_size<=24_000_000,'compressed checkpoint too large')
    compressed=path.read_bytes()
    require(hashlib.sha256(compressed).hexdigest()==desc['file_sha256'],
            'checkpoint file checksum mismatch')
    # Hash and decode the SAME immutable bytes, closing the hash/read race.
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed),mode='rb') as stream:
            raw=stream.read(MAX_JSON_BYTES+1)
        require(len(raw)<=MAX_JSON_BYTES,'checkpoint decompression limit')
        obj=json.loads(raw,object_pairs_hook=_json_no_duplicates,
                       parse_constant=lambda _: (_ for _ in ()).throw(
                           CertificateError('nonfinite checkpoint JSON')))
    except (OSError,EOFError,UnicodeError,json.JSONDecodeError) as exc:
        raise CertificateError('unreadable checkpoint') from exc
    require(isinstance(obj,dict),'checkpoint must be an object')
    digest=obj.pop('sha256',None)
    require(isinstance(digest,str) and SHA.fullmatch(digest) and
            hashlib.sha256(_canonical_bytes(obj)).hexdigest()==digest,
            'checkpoint canonical checksum mismatch')
    point=_validate_checkpoint_payload(obj)
    require(point['event_index']==index and point['commitment']==desc['commitment'],
            'checkpoint manifest disagreement')
    return point


def write_checkpoint(directory: Path, number: int, payload: dict):
    filename=f'checkpoint_{number:04d}.json.gz'
    require(CHECKPOINT_PATTERN.fullmatch(filename) is not None,'too many checkpoints')
    path=Path(directory)/filename
    write_proof(payload,path)
    return {'filename':filename,'event_index':payload['event_index'],
            'commitment':payload['commitment'],
            'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
