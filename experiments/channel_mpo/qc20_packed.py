"""QC-20: compact, immutable binary64 MPO cores for exact dyadic checking.

The witness supplies finite IEEE-754 binary64 pairs. Retain their *exact* bit
patterns in read-only bytes; convert to common-denominator integers only inside
an audited residual calculation. No reduced-precision coefficients are used.

The storage is deliberately independent of NumPy and the simulator. Cache
keys include the source tensor's identity, as in QC-19's bounded-live kernel.
"""
from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
from math import isfinite
import struct

from qc15_checker import (CertificateError, require, cint, MAX_TENSOR_ENTRIES)
from qc16_dyadic import DyadTensor
from qc16_segments import _int_hash
from qc19_kernel import LiveKernel


class _Pairs(Sequence):
    """Read-only, on-demand exact rational interpretation of packed doubles."""
    __slots__ = ('raw',)

    def __init__(self, raw: bytes):
        self.raw = raw

    def __len__(self):
        return len(self.raw) // 16

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if not 0 <= index < len(self):
            raise IndexError(index)
        a, b = struct.unpack_from('<dd', self.raw, 16 * index)
        return (F(a), F(b))

    def __iter__(self):
        for a, b in struct.iter_unpack('<dd', self.raw):
            yield F(a), F(b)


@dataclass(frozen=True, slots=True)
class PackedTensor:
    shape: tuple[int, ...]
    raw: bytes

    def __post_init__(self):
        require(len(self.shape) in (2, 4), 'matrix or MPO core required')
        size = 1
        for d in self.shape:
            size *= cint(d, 1, 4096)
        require(size <= MAX_TENSOR_ENTRIES and len(self.raw) == 16 * size,
                'packed tensor size mismatch')
        if len(self.shape) == 4:
            require(self.shape[1:3] == (2, 2), 'invalid MPO physical dimensions')

    @property
    def vals(self):
        return _Pairs(self.raw)

    def get4(self, l, a, b, r):
        return self.vals[((l*2+a)*2+b)*self.shape[3]+r]

    def get2(self, i, j):
        return self.vals[i*self.shape[1]+j]


def _canonical_raw(raw: bytes) -> bytes:
    """Reject NaN/inf; canonicalize -0.0 as QC-15's Fraction path does."""
    replacement = None
    for i, (re, im) in enumerate(struct.iter_unpack('<dd', raw)):
        require(isfinite(re) and isfinite(im), 'nonfinite tensor coefficient')
        # Fraction(-0.0) equals zero; writing it back as a float yields +0.0.
        for k, number in enumerate((re, im)):
            if number == 0 and raw[16*i + 8*k + 7] & 0x80:
                if replacement is None:
                    replacement = bytearray(raw)
                replacement[16*i + 8*k:16*i + 8*k + 8] = b'\0' * 8
    return bytes(replacement) if replacement is not None else raw


def read_packed_tensor(obj: dict) -> PackedTensor:
    require(isinstance(obj, dict) and set(obj) == {'shape', 'binary64_le'},
            'bad tensor encoding')
    require(isinstance(obj['shape'], list) and len(obj['shape']) in (2, 4),
            'tensor shape')
    shape = tuple(cint(x, 1, 4096) for x in obj['shape'])
    size = 1
    for n in shape: size *= n
    require(size <= MAX_TENSOR_ENTRIES, 'tensor entry cap')
    encoded = obj['binary64_le']
    require(isinstance(encoded, str) and len(encoded) <= MAX_TENSOR_ENTRIES*32,
            'tensor payload cap')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise CertificateError('invalid tensor base64') from exc
    require(len(raw) == 16*size, 'tensor byte-size mismatch')
    return PackedTensor(shape, _canonical_raw(raw))


def pack_tensor(t) -> PackedTensor:
    if isinstance(t, PackedTensor):
        return t
    raw = b''.join(struct.pack('<dd', float(a), float(b)) for a, b in t.vals)
    return PackedTensor(tuple(t.shape), _canonical_raw(raw))


def boundary_digest_packed(index, first, last, cores, accumulated, tally):
    """QC-16 canonical boundary digest on exact IEEE values, without Fraction copies."""
    h = hashlib.sha256(b'qc16-state-boundary-v1\x00')
    for x in (index, first, last, len(cores)):
        _int_hash(h, int(x))
    _int_hash(h, accumulated.numerator)
    _int_hash(h, accumulated.denominator)
    for kind in ('channel', 'left_qr', 'right_qr', 'gate_svd'):
        _int_hash(h, int(tally[kind]))
    for core in cores:
        h.update(bytes([len(core.shape)]))
        for dimension in core.shape:
            _int_hash(h, dimension)
        if isinstance(core, PackedTensor):
            h.update(core.raw)
        else:
            for re, im in core.vals:
                h.update(struct.pack('<dd', float(re), float(im)))
    return h.hexdigest()


class _PackedIntPairs(Sequence):
    __slots__=('data','width')

    def __init__(self,data:bytes,width:int):
        self.data=data
        self.width=width

    def __len__(self):
        return len(self.data)//(2*self.width)

    def __getitem__(self,index):
        if isinstance(index,slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if not 0<=index<len(self):
            raise IndexError(index)
        p=2*self.width*index
        w=self.width
        return (int.from_bytes(self.data[p:p+w],'little',signed=True),
                int.from_bytes(self.data[p+w:p+2*w],'little',signed=True))

    def __iter__(self):
        w=self.width
        for p in range(0,len(self.data),2*w):
            yield (int.from_bytes(self.data[p:p+w],'little',signed=True),
                   int.from_bytes(self.data[p+w:p+2*w],'little',signed=True))


@dataclass(frozen=True, slots=True)
class PackedDyadTensor:
    """Shared exact dyadic denominator and fixed-width signed integer pairs."""
    shape:tuple[int,...]
    exponent:int
    width:int
    data:bytes

    def __post_init__(self):
        require(1<=self.width<=272,'invalid signed-integer packing width')
        size=1
        for dim in self.shape:size*=dim
        require(len(self.data)==2*self.width*size,'packed numerator length mismatch')

    @property
    def vals(self):
        return _PackedIntPairs(self.data,self.width)

    def get4(self,l,a,b,r):
        return self.vals[((l*2+a)*2+b)*self.shape[3]+r]

    def get2(self,i,j):
        return self.vals[i*self.shape[1]+j]


class PackedKernel(LiveKernel):
    """Recompute integer operands from binary64, caching compact exact numerators."""

    def tensor(self,t):
        if not isinstance(t,PackedTensor):
            return super().tensor(t)
        key=id(t)
        found=self.cache.get(key)
        if found is not None and found[0] is t:
            self.cache.move_to_end(key)
            return found[1]
        exponent=0
        for re,im in struct.iter_unpack('<dd',t.raw):
            _,da=re.as_integer_ratio()
            _,db=im.as_integer_ratio()
            exponent=max(exponent,da.bit_length()-1,db.bit_length()-1)
        bit_count=1
        for re,im in struct.iter_unpack('<dd',t.raw):
            for v in (re,im):
                num,den=v.as_integer_ratio()
                number=num << (exponent-(den.bit_length()-1))
                bit_count=max(bit_count,number.bit_length()+1)
        width=max(1,(bit_count+7)//8)
        require(width<=272,'binary64 common-exponent width exceeds bound')
        raw=bytearray()
        for re,im in struct.iter_unpack('<dd',t.raw):
            for v in (re,im):
                num,den=v.as_integer_ratio()
                value=num << (exponent-(den.bit_length()-1))
                raw.extend(value.to_bytes(width,'little',signed=True))
        new=PackedDyadTensor(t.shape,exponent,width,bytes(raw))
        self.cache[key]=(t,new)
        if len(self.cache)>self.cache_limit:
            self.cache.popitem(last=False)
        return new


class FastPackedKernel(PackedKernel):
    """Fast exact integer-pair cache, retaining packed source MPO storage."""

    def tensor(self,t):
        if not isinstance(t,PackedTensor):
            return super().tensor(t)
        key=id(t)
        found=self.cache.get(key)
        if found is not None and found[0] is t:
            self.cache.move_to_end(key)
            return found[1]
        exp=0
        for re,im in struct.iter_unpack('<dd',t.raw):
            _,da=re.as_integer_ratio()
            _,db=im.as_integer_ratio()
            exp=max(exp,da.bit_length()-1,db.bit_length()-1)
        pairs=[]
        for re,im in struct.iter_unpack('<dd',t.raw):
            a,da=re.as_integer_ratio()
            b,db=im.as_integer_ratio()
            pairs.append((a << (exp-(da.bit_length()-1)),
                          b << (exp-(db.bit_length()-1))))
        new=DyadTensor(t.shape,exp,tuple(pairs))
        self.cache[key]=(t,new)
        if len(self.cache)>self.cache_limit:
            self.cache.popitem(last=False)
        return new


class StreamPackedKernel(PackedKernel):
    """One outer-bond block at a time, with exact integer squared residuals.

    QC-16's `_pair` materializes all (l,r) blocks and their sixteen 2-site
    Pauli matrix entries. The streaming variant retains only one block's
    integer intermediates and an exact integer squared-norm accumulator.
    """

    def pair_error2(self, a, b, c, d, gate=None):
        require(a.shape[0] == c.shape[0] and b.shape[-1] == d.shape[-1],
                'pair outer bonds changed')
        a,b,c,d=[self.tensor(t) for t in (a,b,c,d)]
        require(a.shape[-1] == b.shape[0] and c.shape[-1] == d.shape[0],
                'pair bond mismatch')
        g=self.tensor(gate) if gate is not None else None
        if g is not None:
            require(g.shape==(4,4), 'invalid gate matrix')
        el=a.exponent+b.exponent+(2*g.exponent if g is not None else 0)
        er=c.exponent+d.exponent
        exp=max(el,er)
        acc=0
        dl,dr=a.shape[0],b.shape[-1]

        def block(x,y,l,r):
            inner=x.shape[-1]
            result=[]
            for i in range(4):
                xi,yi=divmod(i,2)
                row=[]
                for j in range(4):
                    xj,yj=divmod(j,2)
                    sr=si=0
                    for k in range(inner):
                        p=(x.get4(l,xi,xj,k),y.get4(k,yi,yj,r))
                        z=p[0][0]*p[1][0]-p[0][1]*p[1][1]
                        w=p[0][0]*p[1][1]+p[0][1]*p[1][0]
                        sr+=z;si+=w
                    row.append((sr,si))
                result.append(row)
            return result

        for l in range(dl):
            for r in range(dr):
                old=block(a,b,l,r)
                new=block(c,d,l,r)
                for i in range(4):
                    for j in range(4):
                        if g is None:
                            sr,si=old[i][j]
                        else:
                            sr=si=0
                            for x in range(4):
                                for y in range(4):
                                    ga=g.get2(x,i)
                                    gb=g.get2(y,j)
                                    # conj(ga) * old[x][y] * gb, exactly.
                                    v=old[x][y]
                                    ur=ga[0]*v[0]+ga[1]*v[1]
                                    ui=ga[0]*v[1]-ga[1]*v[0]
                                    sr+=ur*gb[0]-ui*gb[1]
                                    si+=ur*gb[1]+ui*gb[0]
                        pr=(sr << (exp-el))-(new[i][j][0] << (exp-er))
                        pi=(si << (exp-el))-(new[i][j][1] << (exp-er))
                        acc+=pr*pr+pi*pi
        return F(acc,1<<(2*exp))


class StreamFastKernel(StreamPackedKernel,FastPackedKernel):
    """QC-20 fast default: streaming residual blocks and exact cached integers."""



def read_checkpoint_packed(directory, desc):
    """Fully validate checkpoint with packed state, avoiding Fraction-core allocation.

    Mirrors the QC-17 checkpoint validation, including independent hashes of
    compressed bytes, canonical JSON, exact boundary digest and event count.
    """
    import gzip
    import io
    import json
    from pathlib import Path
    from qc15_checker import (MAX_EVENTS, MAX_JSON_BYTES, _canonical_bytes,
                              _json_no_duplicates, checked_chain)
    from qc17_format import (CHECKPOINT_SCHEMA, CHECKPOINT_PATTERN, SHA,
                              KINDS, decode_accumulated)

    require(isinstance(desc,dict) and set(desc)=={
        'filename','event_index','commitment','file_sha256'},
        'invalid checkpoint descriptor')
    filename=desc['filename']
    require(isinstance(filename,str) and CHECKPOINT_PATTERN.fullmatch(filename),
            'invalid checkpoint filename')
    for name in ('commitment','file_sha256'):
        require(isinstance(desc[name],str) and SHA.fullmatch(desc[name]),
                'invalid checkpoint hash')
    index=cint(desc['event_index'],0,MAX_EVENTS+64)
    path=Path(directory)/filename
    require(path.is_file() and not path.is_symlink(),
            'missing or symlinked checkpoint')
    require(path.stat().st_size<=24_000_000,'compressed checkpoint too large')
    compressed=path.read_bytes()
    require(hashlib.sha256(compressed).hexdigest()==desc['file_sha256'],
            'checkpoint file checksum mismatch')
    def reject_constant(_):
        raise CertificateError('nonfinite checkpoint JSON')

    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed),mode='rb') as stream:
            raw=stream.read(MAX_JSON_BYTES+1)
        require(len(raw)<=MAX_JSON_BYTES,'checkpoint decompression limit')
        obj=json.loads(raw,object_pairs_hook=_json_no_duplicates,
                       parse_constant=reject_constant)
    except (OSError,EOFError,UnicodeError,json.JSONDecodeError,ValueError) as exc:
        raise CertificateError('unreadable checkpoint') from exc
    require(isinstance(obj,dict),'checkpoint must be an object')
    payload_sha=obj.pop('sha256',None)
    require(isinstance(payload_sha,str) and SHA.fullmatch(payload_sha) and
            hashlib.sha256(_canonical_bytes(obj)).hexdigest()==payload_sha,
            'checkpoint canonical checksum mismatch')
    require(set(obj)=={'format','event_index','first','last','cores',
                       'accumulated','tally','commitment'},
            'malformed checkpoint fields')
    require(obj['format']==CHECKPOINT_SCHEMA,'checkpoint format mismatch')
    pos=cint(obj['event_index'],0,MAX_EVENTS+64)
    first=cint(obj['first'],0,63)
    last=cint(obj['last'],first,63)
    encoded=obj['cores']
    require(isinstance(encoded,list) and len(encoded)==last-first+1,
            'checkpoint support mismatch')
    cores=[read_packed_tensor(item) for item in encoded]
    require(all(len(c.shape)==4 for c in cores),'checkpoint core shape mismatch')
    checked_chain(cores)
    acc=decode_accumulated(obj['accumulated'])
    tally=obj['tally']
    require(isinstance(tally,dict) and set(tally)==set(KINDS),
            'checkpoint tally invalid')
    for k in KINDS:
        cint(tally[k],0,MAX_EVENTS)
    require(sum(tally.values())<=pos,'checkpoint tally exceeds event index')
    digest=boundary_digest_packed(pos,first,last,cores,acc,tally)
    require(obj['commitment']==digest,'checkpoint MPO boundary digest mismatch')
    require(pos==index and digest==desc['commitment'],
            'checkpoint manifest disagreement')
    return {'event_index':pos,'first':first,'last':last,'cores':cores,
            'accumulated':acc,'tally':dict(tally),'commitment':digest}
