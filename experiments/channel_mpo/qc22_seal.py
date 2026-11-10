"""QC-22: fail-closed Linux memfd sealing of a validated QC-18 event tape.

A read-only file-backed mmap is *not* an immutable snapshot: an external writer
can change its pages between hashing and replay.  Clone the already-open source
FD to an anonymous memfd, verify the clone against the manifest SHA-256, apply
F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL, and reconstruct all
frame offsets and per-segment digests from the sealed mapping.

No simulator imports.  This Linux-specific API deliberately lacks a mutable
file-backed fallback.  Checkpoint readers independently hash and decode the
same in-memory bytes, so their content has no corresponding mmap race.
"""
from __future__ import annotations

import fcntl
import hashlib
import mmap
import os
import struct
import sys
from types import SimpleNamespace
from time import perf_counter

from qc15_checker import CertificateError, require
from qc18_tape import MAX_EVENT_BYTES, MAGIC, MAGIC_RAW, MAGIC_ZLIB, TapeEvents

MANDATORY_SEALS = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                   fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
COPY_BLOCK = 1 << 20


def require_sealing():
    require(sys.platform.startswith('linux') and hasattr(os, 'memfd_create') and
            hasattr(os, 'MFD_ALLOW_SEALING'),
            'QC-22 requires Linux memfd_create with sealing support')
    for name in ('F_ADD_SEALS', 'F_GET_SEALS', 'F_SEAL_WRITE',
                 'F_SEAL_GROW', 'F_SEAL_SHRINK', 'F_SEAL_SEAL'):
        require(hasattr(fcntl, name), 'QC-22 requires Linux file seals')


def check_sealed(fd: int) -> int:
    """Fail closed if any required Linux seal was dropped or unsupported."""
    try:
        got = fcntl.fcntl(fd, fcntl.F_GET_SEALS)
    except OSError as exc:
        raise CertificateError('unable to read Linux file seals') from exc
    require(got & MANDATORY_SEALS == MANDATORY_SEALS,
            'unsealed or partially sealed proof tape')
    return got


def _sealed_layout(source: TapeEvents, expected_digest: str):
    """Recheck framing from sealed bytes, independent of pre-seal offsets."""
    mm = source.mm
    require(hashlib.sha256(mm).hexdigest() == expected_digest,
            'sealed tape checksum mismatch')
    count=source.count
    require(mm[:len(MAGIC)] in (MAGIC_RAW, MAGIC_ZLIB),
            'sealed tape magic mismatch')
    magic = MAGIC_ZLIB if source.compressed else MAGIC_RAW
    require(mm[:len(MAGIC)] == magic and len(mm) >= len(MAGIC) + 4,
            'sealed tape encoding mismatch')
    require(struct.unpack_from('<I', mm, len(MAGIC))[0] == count,
            'sealed tape event count mismatch')
    start = len(MAGIC) + 4 + count*12
    require(start <= len(mm), 'sealed tape table truncated')
    entries=[]
    position=start
    for i in range(count):
        offset,length=struct.unpack_from('<QI', mm,len(MAGIC)+4+12*i)
        absolute=start+offset
        require(0 < length <= MAX_EVENT_BYTES and absolute == position+4,
                'sealed tape noncontiguous or oversized frame')
        require(absolute + length <= len(mm), 'sealed tape frame overrun')
        require(struct.unpack_from('<I',mm,absolute-4)[0]==length,
                'sealed tape length prefix mismatch')
        entries.append((absolute,length))
        position=absolute+length
    require(position==len(mm),'sealed tape trailing bytes')
    for segment in source.segments:
        h=hashlib.sha256()
        first=segment['start_index']
        for j in range(first,first+segment['count']):
            offset,length=entries[j]
            h.update(mm[offset-4:offset+length])
        require(h.hexdigest()==segment['events_sha256'],
                'sealed tape segment digest mismatch')
    return entries


def seal_tape(source: TapeEvents, expected_digest: str):
    """Replace the original mmap with a kernel-sealed, hash-pinned memfd.

    The source descriptor is fixed before cloning; no pathname is reopened.
    If a writer races the copy, the sealed copy's SHA-256 must still match the
    manifest.  Repeated sealing is forbidden.  The returned source supports
    the original TapeEvents indexing API and owns the sealed FD.
    """
    require_sealing()
    require(isinstance(source,TapeEvents) and not getattr(source,'sealed',False),
            'expected a fresh unsealed TapeEvents')
    require(isinstance(expected_digest,str) and len(expected_digest)==64,
            'invalid expected tape digest')
    started=perf_counter()
    old_fd=source.handle.fileno()
    size=len(source.mm)
    memfd=-1
    newmap=None
    newhandle=None
    try:
        memfd=os.memfd_create('qc22-verified-event-tape',
            os.MFD_ALLOW_SEALING | getattr(os,'MFD_CLOEXEC',0))
        digest=hashlib.sha256()
        offset=0
        while offset<size:
            data=os.pread(old_fd,min(COPY_BLOCK,size-offset),offset)
            require(bool(data),'source tape shrank during sealing')
            digest.update(data)
            written=0
            while written<len(data):
                n=os.write(memfd,data[written:])
                require(n>0,'sealed copy failed to advance')
                written+=n
            offset+=len(data)
        require(offset==size and os.fstat(memfd).st_size==size,
                'sealed tape copy length mismatch')
        require(digest.hexdigest()==expected_digest,
                'source tape changed during sealing')
        # No writable mapping has ever been created on memfd.
        fcntl.fcntl(memfd,fcntl.F_ADD_SEALS,MANDATORY_SEALS)
        check_sealed(memfd)
        newmap=mmap.mmap(memfd,0,access=mmap.ACCESS_READ)
        require(hashlib.sha256(newmap).hexdigest()==expected_digest,
                'sealed tape differs from manifest')
        # Validate all indices from the sealed mapping.  Avoid reusing layout
        # metadata that could have been parsed during a source-file race.
        proxy=SimpleNamespace(mm=newmap, count=source.count,
                              compressed=source.compressed,
                              segments=source.segments)
        entries=_sealed_layout(proxy,expected_digest)
        newhandle=os.fdopen(memfd,'rb')
        memfd=-1
        oldmap,oldhandle=source.mm,source.handle
        source.mm=newmap
        source.handle=newhandle
        source.entries=entries
        source.sealed=True
        source.sealed_sha256=expected_digest
        source.seal_seconds=perf_counter()-started
        oldmap.close()
        oldhandle.close()
        return source
    except CertificateError:
        raise
    except (OSError,ValueError,OverflowError,BufferError) as exc:
        raise CertificateError('Linux memfd sealing failed') from exc
    finally:
        # On failure leave the old source accessible only to caller cleanup;
        # verify() will never proceed unless seal_tape returned successfully.
        if memfd>=0:
            os.close(memfd)
        if newhandle is None and newmap is not None:
            newmap.close()
