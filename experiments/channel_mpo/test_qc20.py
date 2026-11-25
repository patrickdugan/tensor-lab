"""QC-20 binary64-backed exact tensor storage: parity, replay and tampering."""
from __future__ import annotations

import base64
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

from qc15_checker import (CertificateError, initial_core, read_tensor,
                          read_proof, write_proof)
from qc16_dyadic import Kernel
from qc16_segments import boundary_digest
from qc17_format import encode_core, read_checkpoint
from qc18_tape import TAPE_FILE
from qc19_checker import verify as verify19
from qc20_checker import verify as verify20, verify_one_segment, load_root
from qc20_packed import (PackedTensor, PackedKernel, FastPackedKernel, read_packed_tensor,
                         pack_tensor, boundary_digest_packed, read_checkpoint_packed)

HERE=Path(__file__).resolve().parent
FIXTURE=HERE/'qc18_certificates/10q_d6_mmap.zip'
FIELDS=('verified_bound','arithmetic_lower_requirement','event_counts',
        'events_checked','segments_checked','checkpoints_checked',
        'witness_complex_entries')


def encoded(vals,shape):
    payload=b''.join(struct.pack('<dd',a,b) for a,b in vals)
    return {'shape':list(shape),'binary64_le':base64.b64encode(payload).decode('ascii')}


class PackingTests(unittest.TestCase):
    def test_exact_rational_parity_extreme_binary64(self):
        values=[(0.0,-0.0),(5e-324,-5e-324),
                (2**-103,1.5),(0.25,-2**-100),
                (math.ldexp(1.,1023),-math.ldexp(1.,-1022)),
                (4.0,-6.0),(0.,-1.),(1.,0.),
                (1. / 3.,-1. / 7.),(math.pi,-math.e),
                (2.**-54,-2.**-54),(-3.75,5.5),
                (-0.,0.),(0.,0.),(-1. / 11.,1. / 17.),
                (1.0,1.0)]
        blob=encoded(values,(4,4))
        packed=read_packed_tensor(blob)
        unpacked=read_tensor(blob)
        self.assertEqual(tuple(packed.vals),unpacked.vals)
        left=PackedKernel().tensor(packed)
        right=Kernel().tensor(unpacked)
        self.assertEqual(left.shape,right.shape)
        self.assertEqual(left.exponent,right.exponent)
        self.assertEqual(tuple(left.vals),right.vals)
        self.assertEqual(PackedKernel().frob_norm2(packed),
                         Kernel().frob_norm2(unpacked))
        self.assertEqual(len(packed.raw),16*len(values))

    def test_signed_zero_canonicalized_to_reference(self):
        v=encoded([(-0.,-0.),(0.,0.),(1.,0.),(0.,-1.)],(2,2))
        a=read_packed_tensor(v)
        b=read_tensor(v)
        self.assertEqual(a.raw,b''.join(struct.pack('<dd',float(x),float(y))
                                        for x,y in b.vals))
        self.assertEqual(tuple(a.vals),b.vals)

    def test_nonfinite_rejected(self):
        for v in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(CertificateError):
                read_packed_tensor(encoded([(v,0)]*4,(2,2)))

    def test_malformed_dimensions_and_payload(self):
        valid=encoded([(1.,0.)]*4,(2,2))
        for obj in ({'shape':[2,2], 'binary64_le':'@@@'},
                    {'shape':[2,2], 'binary64_le':'AAAA'},
                    {'shape':[3,3], 'binary64_le':valid['binary64_le']},
                    {'shape':[1,3,3,1], 'binary64_le':valid['binary64_le']},
                    {'shape':[0,4], 'binary64_le':valid['binary64_le']},
                    {'shape':[2,2], 'binary64_le':None},
                    {'shape':[True,2], 'binary64_le':valid['binary64_le']}):
            with self.assertRaises(CertificateError):read_packed_tensor(obj)

    def test_invalid_packed_constructor(self):
        with self.assertRaises(CertificateError):PackedTensor((1,2,3,1),b'\x00'*96)
        with self.assertRaises(CertificateError):PackedTensor((2,2),b'\x00'*16)

    def test_exact_boundary_digest_parity(self):
        ref=[initial_core('Z'),initial_core('X')]
        packed=[pack_tensor(x) for x in ref]
        tally={'channel':4,'left_qr':2,'right_qr':1,'gate_svd':1}
        acc=F(17,1<<82)
        self.assertEqual(boundary_digest(11,3,4,ref,acc,tally),
                         boundary_digest_packed(11,3,4,packed,acc,tally))
        changed=packed.copy()
        changed[0]=read_packed_tensor(encoded([(1.,0.)]*4,(1,2,2,1)))
        self.assertNotEqual(boundary_digest_packed(11,3,4,packed,acc,tally),
                            boundary_digest_packed(11,3,4,changed,acc,tally))

    def test_pair_channel_context_readout_reference(self):
        a=initial_core('X');b=initial_core('Z')
        pa=pack_tensor(a);pb=pack_tensor(b)
        q,r=Kernel(),PackedKernel()
        self.assertEqual(q.pair_error2(a,b,a,b),r.pair_error2(pa,pb,pa,pb))
        self.assertEqual(q.gram_defect2(a,'L'),r.gram_defect2(pa,'L'))
        self.assertEqual(q.context_bound([a,b],(0,)),r.context_bound([pa,pb],(0,)))
        self.assertEqual(q.exact_readout([a]),r.exact_readout([pa]))

    def test_packed_storage_avoids_permanent_fraction_tuple(self):
        n=4096
        packed=read_packed_tensor(encoded([(float(i)/n,0.) for i in range(n)],(64,2,2,16)))
        self.assertEqual(len(packed.raw),n*16)
        self.assertNotIn('vals',getattr(packed,'__slots__',()))
        self.assertEqual(packed.vals[n-1][0],F(n-1,n))

    def test_retired_packed_core_not_retained(self):
        k=PackedKernel()
        a=pack_tensor(initial_core('Z'))
        k.tensor(a)
        self.assertIn(id(a),k.cache)
        k.retire(a)
        self.assertNotIn(id(a),k.cache)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='qc20_fixture_')
        cls.path=Path(cls.temp.name)
        with ZipFile(FIXTURE) as z:z.extractall(cls.path)
        cls.manifest=cls.path/'manifest.json'
        cls.reference=verify19(cls.manifest,workers=1)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def compare(self,result):
        for key in FIELDS:self.assertEqual(self.reference[key],result[key],key)

    def test_checkpoint_packed_matches_reference_parser(self):
        doc=read_proof(self.manifest)
        for desc in doc['checkpoints']:
            reference=read_checkpoint(self.path,desc)
            packed=read_checkpoint_packed(self.path,desc)
            for key in ('event_index','first','last','accumulated','tally','commitment'):
                self.assertEqual(reference[key],packed[key])
            self.assertEqual([c.shape for c in reference['cores']],
                             [c.shape for c in packed['cores']])
            for a,b in zip(reference['cores'],packed['cores']):
                self.assertEqual(tuple(b.vals),a.vals)

    def test_serial_replay_identical_exact_bound(self):
        result=verify20(self.manifest,workers=1)
        self.compare(result)
        self.assertEqual(result['storage_mode'],'fast')

    def test_compact_mode_serial_parity(self):
        result=verify20(self.manifest,workers=1,storage_mode='compact')
        self.compare(result)
        self.assertEqual(result['storage_mode'],'compact')

    def test_fast_integer_cache_parity(self):
        src=pack_tensor(initial_core('X'))
        self.assertEqual(tuple(FastPackedKernel().tensor(src).vals),
                         tuple(PackedKernel().tensor(src).vals))

    def test_fork_two_workers_parity(self):
        self.compare(verify20(self.manifest,workers=2,start_method='fork'))

    def test_spawn_two_workers_parity(self):
        self.compare(verify20(self.manifest,workers=2,start_method='spawn'))

    def test_all_segment_boundaries(self):
        _,source,points,plan=load_root(self.manifest,corrections=True)
        for i in range(len(source.segments)):
            self.assertEqual(verify_one_segment(source,points,plan,self.path,i)['end_commitment'],
                             source.segments[i]['end_commitment'])

    def test_isolated_cli(self):
        cmd=[sys.executable,'-I',str(HERE/'qc20_checker.py'),
             str(self.manifest),'--workers','2','--start-method','spawn']
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
        self.assertEqual(p.returncode,0,p.stderr)
        self.compare(json.loads(p.stdout))

    def test_tampered_event_rejected_with_rehashed_tape(self):
        with tempfile.TemporaryDirectory(prefix='qc20_tamper_') as d:
            dest=Path(d)
            with ZipFile(FIXTURE) as z:z.extractall(dest)
            doc=read_proof(dest/'manifest.json')
            tape=dest/TAPE_FILE
            data=bytearray(tape.read_bytes())
            data[-31]^=1
            tape.write_bytes(data)
            doc['tape']['sha256']=hashlib.sha256(data).hexdigest()
            write_proof(doc,dest/'manifest.json')
            with self.assertRaises(CertificateError):verify20(dest/'manifest.json')

    def test_missing_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory(prefix='qc20_missing_') as d:
            dest=Path(d)
            with ZipFile(FIXTURE) as z:z.extractall(dest)
            doc=read_proof(dest/'manifest.json')
            (dest/doc['checkpoints'][1]['filename']).unlink()
            with self.assertRaises(CertificateError):verify20(dest/'manifest.json')

    def test_standard_library_independence(self):
        source=(HERE/'qc20_checker.py').read_text()+(HERE/'qc20_packed.py').read_text()
        self.assertNotIn('import numpy',source)
        self.assertNotIn('import torch',source)
        self.assertNotIn('from qc14_certified',source)

if __name__=='__main__':unittest.main()
