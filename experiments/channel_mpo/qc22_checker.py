"""QC-22: Linux-sealed witness replay with unchanged exact dyadic arithmetic.

Requires fork-based workers to inherit an already sealed memfd and decoded
circuit metadata.  Any unsupported kernel or sealing failure is fatal. Event
frames are never read again from the mutable source pathname after sealing.
Compressed checkpoints are separately hash-checked and parsed from the same
immutable bytes on each use by QC-20's independent checkpoint verifier.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parent))

import qc20_checker as baseline
from qc15_checker import CertificateError, require, cint, _canonical_bytes, MAX_TOTAL_WITNESS_ENTRIES
from qc17_checker import prepare_circuit
from qc20_packed import read_checkpoint_packed
from qc21_anchor import check_genesis, check_final
from qc22_seal import seal_tape,check_sealed,require_sealing


def _sealed_worker(i):
    state=baseline._STATE
    require(state is not None and state['source'].sealed,
            'worker missing sealed witness state')
    check_sealed(state['source'].handle.fileno())
    return baseline._worker_segment(i)


def verify(manifest_path,workers=1,*,start_method='fork'):
    """Independently verify exact arithmetic against the sealed event tape."""
    require_sealing()
    require(start_method=='fork','QC-22 requires fork to inherit sealed descriptors')
    require(os.name=='posix','QC-22 requires POSIX file descriptors')
    workers=cint(workers,1,16)
    path=Path(manifest_path).resolve()
    manifest,source,checkpoints,plan=baseline.load_root(path,corrections=False)
    prior=baseline._STATE
    try:
        seal_tape(source,manifest['tape']['sha256'])
        digest=hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
        initial=read_checkpoint_packed(path.parent,checkpoints[0])
        check_genesis(initial,plan)
        terminal=read_checkpoint_packed(path.parent,checkpoints[-1])
        result=check_final(terminal,plan,manifest['result'])
        del initial,terminal
        # All intermediates are checked independently as both incoming and
        # outgoing boundary states during segment replay.
        corrected=prepare_circuit(manifest['circuit'],manifest['result'],
                                  compute_corrections=True)
        baseline._STATE={'path':str(path),'digest':digest,'manifest':manifest,
                         'source':source,'checkpoints':checkpoints,'plan':corrected,
                         'storage_mode':'summary'}
        count=len(source.segments)
        if workers==1:
            checks=[_sealed_worker(i) for i in range(count)]
        else:
            ctx=multiprocessing.get_context('fork')
            with ProcessPoolExecutor(max_workers=min(workers,count),
                                     mp_context=ctx) as pool:
                checks=list(pool.map(_sealed_worker,range(count),chunksize=1))
        require(len(checks)==count and
                sum(x['audited_events'] for x in checks)==result['events_checked'],
                'incomplete sealed witness audit')
        total=sum(x['changed_entries'] for x in checks)
        require(total<=MAX_TOTAL_WITNESS_ENTRIES,'proof entry cap exceeded')
        return {**result,
                'proof_level':'qc22-kernel-sealed-full-witness',
                'segments_checked':count,
                'checkpoints_checked':len(checkpoints),
                'witness_complex_entries':total,
                'workers':min(workers,count),
                'arithmetic_backend':'exact-dyadic-context-summary',
                'transport':'sha256-pinned-F_SEAL_WRITE-grow-shrink-seal-memfd',
                'storage_mode':'summary',
                'sealed_bytes':len(source.mm),
                'sealed_sha256':source.sealed_sha256,
                'seal_seconds':source.seal_seconds}
    finally:
        baseline._STATE=prior
        source.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--start-method',choices=['fork','spawn'],default='fork')
    args=parser.parse_args()
    try:
        value=verify(args.manifest,args.workers,start_method=args.start_method)
    except (CertificateError,OSError,ValueError,TypeError,OverflowError,
            MemoryError,IndexError,KeyError) as exc:
        parser.exit(2,f'FAIL: {exc}\n')
    print(json.dumps({'status':'PASS',**value},sort_keys=True))

if __name__=='__main__':
    main()
