"""QC-16 differential integer-kernel and segmented-witness tamper tests."""
from __future__ import annotations
import copy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from qc15_checker import (CertificateError,Tensor,check_witness,read_proof,write_proof,
                          pair_error2,channel_error2,gram_defect2,gate_defect2,
                          kraus_defect2,frob_norm2,context_bound,exact_readout)
from qc16_dyadic import Kernel
from qc16_segments import split_witness
from qc16_checker import verify,verify_segmented
from channel_mpo import gate_layers,kraus
from qc15_emit import export_witness

HERE=Path(__file__).resolve().parent


class DyadicKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=read_proof(HERE/'qc15_certificates/full_10q_d6_dephasing_s3_c4.json.gz')

    def test_exact_same_result_all_channels(self):
        for channel in ('depolarizing','dephasing','amplitude_damping'):
            proof=export_witness(3,gate_layers(3,3,7),kraus(channel,.5),(0,),'Z',bond_cap=3)
            with self.subTest(channel=channel):
                self.assertEqual(check_witness(proof),check_witness(proof,kernel=Kernel()))

    def test_existing_ten_qubit_witness_exact_same_result(self):
        self.assertEqual(check_witness(self.fixture),check_witness(self.fixture,kernel=Kernel()))

    def test_pair_and_gram_kernels(self):
        f=lambda x:F(float(x))
        # Two nontrivial MPO cores with complex, signed and unequal dyadic scales.
        a=Tensor((1,2,2,2),tuple((f(i/16-1/8),f(i/64)) for i in range(8)))
        b=Tensor((2,2,2,1),tuple((f((-1)**i/8),f(i/32)) for i in range(8)))
        c=Tensor((1,2,2,2),tuple((f(i/32),f((i-3)/16)) for i in range(8)))
        d=Tensor((2,2,2,1),tuple((f(i/8),f((i-7)/64)) for i in range(8)))
        gate=Tensor((4,4),tuple((f((i%4==i//4)*1.0),f((i-9)/512)) for i in range(16)))
        kernel=Kernel()
        self.assertEqual(kernel.pair_error2(a,b,c,d),pair_error2(a,b,c,d))
        self.assertEqual(kernel.pair_error2(a,b,c,d,gate),pair_error2(a,b,c,d,gate))
        for core in (a,b,c,d):
            for direction in ('L','R'):
                self.assertEqual(kernel.gram_defect2(core,direction),gram_defect2(core,direction))
            self.assertEqual(kernel.frob_norm2(core),frob_norm2(core))
        self.assertEqual(kernel.gate_defect2(gate),gate_defect2(gate))
        self.assertEqual(kernel.context_bound([a,b],[0,1]),context_bound([a,b],[0,1]))
        self.assertEqual(kernel.exact_readout([a,b]),exact_readout([a,b]))

    def test_channel_kernel_unequal_exponents(self):
        f=lambda x:F(float(x))
        core=Tensor((1,2,2,1),tuple((f(x),f(y)) for x,y in
                     ((.5,0),(.125,.25),(.25,-.125),(.375,0))))
        newer=Tensor((1,2,2,1),tuple((f(x),f(0)) for x in (.5,.125,.25,.375)))
        ops=(Tensor((2,2),tuple((f(x),f(0)) for x in (1,0,0,.125))),
             Tensor((2,2),tuple((f(x),f(y)) for x,y in
                    ((0,0),(.0625,0),(.03125,-.125),(.0625,0)))))
        k=Kernel()
        self.assertEqual(k.channel_error2(core,newer,ops),channel_error2(core,newer,ops))
        self.assertEqual(k.kraus_defect2(ops),kraus_defect2(ops))
        self.assertEqual(k.context_bound([core,newer],[0],gate=False),
                         context_bound([core,newer],[0],gate=False))

    def test_subnormal_binary64_coefficients(self):
        f=lambda x:F(float(x))
        tiny=float.fromhex('0x0.0000000000001p-1022')
        v=Tensor((1,2,2,1),tuple((f(x),f(0)) for x in (1,tiny,0,1)))
        k=Kernel()
        self.assertEqual(k.gram_defect2(v,'L'),gram_defect2(v,'L'))
        self.assertEqual(k.frob_norm2(v),frob_norm2(v))
        self.assertEqual(k.exact_readout([v]),exact_readout([v]))


class SegmentedWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full=read_proof(HERE/'qc15_certificates/full_10q_d6_dephasing_s3_c4.json.gz')

    def create(self,tmp,per=10,proof=None):
        return split_witness(self.full if proof is None else proof,Path(tmp),events_per_segment=per)

    def test_segmented_matches_original_and_fraction_checker(self):
        for per in (1,7,10,1000):
            with self.subTest(events_per_segment=per),tempfile.TemporaryDirectory() as tmp:
                path=self.create(tmp,per)
                result=verify(path)
                control=check_witness(self.full)
                for key in ('verified_bound','arithmetic_lower_requirement',
                            'reported_expectation','events_checked','event_counts'):
                    self.assertEqual(result[key],control[key])
                self.assertEqual(result['verified_bound'],verify(path,reference=True)['verified_bound'])
                self.assertEqual(result['segments_checked'],(len(self.full['events'])+per-1)//per)
                self.assertEqual(result['boundaries_checked'],result['segments_checked']+1)

    def test_cli_isolated_standard_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp,9)
            child=subprocess.run([sys.executable,'-I',str(HERE/'qc16_checker.py'),str(path)],
                                 capture_output=True,text=True,timeout=30)
            self.assertEqual(child.returncode,0,child.stderr)
            self.assertEqual(json.loads(child.stdout)['status'],'PASS')

    def test_broken_payload_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path=self.create(tmp,12)
            part=next(Path(tmp).glob('segment_*.json.gz'))
            raw=bytearray(part.read_bytes());raw[-8]^=1;part.write_bytes(raw)
            with self.assertRaises(CertificateError):verify(manifest_path)

    def test_deleted_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp,12)
            sorted(Path(tmp).glob('segment_*.json.gz'))[-1].unlink()
            with self.assertRaises(CertificateError):verify(path)

    def test_disconnected_boundary_with_recomputed_manifest_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp,12)
            obj=read_proof(path)
            forged='e'*64
            obj['segments'][1]['start_commitment']=forged
            write_proof(obj,path)
            with self.assertRaises(CertificateError):verify(path)

    def test_wrong_computed_boundary_with_recomputed_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp,12)
            obj=read_proof(path)
            forged='e'*64
            obj['segments'][0]['end_commitment']=forged
            obj['segments'][1]['start_commitment']=forged
            for entry in obj['segments'][:2]:
                file=Path(tmp)/entry['filename']
                payload=read_proof(file)
                payload['start_commitment']=entry['start_commitment']
                payload['end_commitment']=entry['end_commitment']
                write_proof(payload,file)
                entry['file_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
            write_proof(obj,path)
            with self.assertRaisesRegex(CertificateError,'boundary mismatch'):verify(path)

    def test_altered_core_recalculated_boundaries_still_rejected(self):
        proof=copy.deepcopy(self.full)
        target=next(ev for ev in proof['events'] if ev['kind']=='gate_svd')
        target['residual_upper']=(0.).hex()
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp,10,proof)
            with self.assertRaises(CertificateError):verify(path)

    def test_reordered_events_with_new_hashes_still_rejected(self):
        proof=copy.deepcopy(self.full)
        proof['events'][2],proof['events'][3]=proof['events'][3],proof['events'][2]
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp,10,proof)
            with self.assertRaises(CertificateError):verify(path)

    def test_segment_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp,10)
            obj=read_proof(path)
            obj['segments'][0]['filename']='../out.json.gz'
            write_proof(obj,path)
            with self.assertRaises(CertificateError):verify(path)

    def test_refuse_conditional_ledger(self):
        path=HERE/'qc15_certificates/ledger_48q_d16_dephasing_s3_c8.json.gz'
        with self.assertRaises(CertificateError):verify(path)


if __name__=='__main__':unittest.main()
