"""Independent QC-16 verifier: segmented full witnesses and dyadic arithmetic.

Standard library only. A certificate's segment boundaries are checked against
MPO states reconstructed by the independent QC-15 event arithmetic checker.
The reference Fraction backend remains selectable for cross-checking.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    # In -I isolated CLI mode Python omits the script directory from sys.path.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
from qc15_checker import (SCHEMA,CertificateError,require,read_proof,
                          check_witness,check_certificate)
from qc16_dyadic import Kernel
from qc16_segments import SCHEMA16,SegmentEvents,boundary_digest


def verify_segmented(manifest_path:Path,*,reference=False):
    path=Path(manifest_path)
    manifest=read_proof(path)
    require(manifest.get('format')==SCHEMA16,'QC16 segmented manifest required')
    require(set(manifest)=={'format','circuit','segments','result'},
            'bad manifest fields')
    source=SegmentEvents(manifest,path.parent)
    expected=source.boundaries
    witnessed=set()

    def advance(index,first,last,cores,acc,tally):
        if index in expected:
            value=boundary_digest(index,first,last,cores,acc,tally)
            require(value==expected[index],f'boundary mismatch after event {index}')
            witnessed.add(index)

    proof={'format':SCHEMA,'circuit':manifest['circuit'],
           'result':manifest['result']}
    result=check_witness(proof,kernel=None if reference else Kernel(),
                         event_source=source,on_boundary=advance)
    require(witnessed==set(expected),'unchecked segment boundary')
    return {**result,'proof_level':'segmented-full-witness',
            'segments_checked':len(source.segments),
            'boundaries_checked':len(witnessed),
            'arithmetic_backend':'reference-fraction' if reference else 'common-dyadic-integer'}


def verify_unsegmented(path:Path,*,reference=False):
    if reference:
        return check_certificate(path,require_witness=True)
    proof=read_proof(path)
    return {**check_witness(proof,kernel=Kernel()),
            'arithmetic_backend':'common-dyadic-integer'}


def verify(path:Path,*,reference=False):
    proof=read_proof(path)
    if proof.get('format')==SCHEMA16:
        return verify_segmented(path,reference=reference)
    if proof.get('format')==SCHEMA:
        return verify_unsegmented(path,reference=reference)
    raise CertificateError('full QC15 or segmented QC16 proof required')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('proof',type=Path)
    p.add_argument('--reference',action='store_true',help='use Fraction reference backend')
    a=p.parse_args()
    try:result=verify(a.proof,reference=a.reference)
    except (CertificateError,ValueError,OverflowError,MemoryError,KeyError,TypeError,
            OSError,IndexError) as ex:
        p.exit(2,'FAIL: '+str(ex)+'\n')
    print(json.dumps({'status':'PASS',**result},sort_keys=True))

if __name__=='__main__':main()
