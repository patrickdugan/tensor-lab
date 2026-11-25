"""QC-17 independent checkpointed segment proof and adversarial tests."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from fractions import Fraction as F
import subprocess
import sys
import tempfile
import unittest

from qc15_checker import CertificateError,read_proof,write_proof,check_witness
from qc16_checker import verify as verify16
from qc16_dyadic import Kernel
from qc17_emit import produce
from qc17_checker import verify,verify_one_segment
from qc17_format import (read_checkpoint,checkpoint_payload,write_checkpoint,
                         _validate_checkpoint_payload,encode_accumulated)
from channel_mpo import gate_layers,kraus
from qc15_emit import export_witness

HERE=Path(__file__).resolve().parent
FIX=HERE/'qc15_certificates/full_10q_d6_dephasing_s3_c4.json.gz'


class QC17Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full=read_proof(FIX)

    def create(self, root, *, events_per_segment=12, proof=None):
        full=Path(root)/'full.json.gz'
        write_proof(self.full if proof is None else proof,full)
        return produce(full,Path(root)/'segmented',events_per_segment)

    def test_serial_matches_qc16_and_exact_dyadic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp,events_per_segment=9)
            ours=verify(path,workers=1)
            control=check_witness(self.full,kernel=Kernel())
            for field in ('events_checked','event_counts','cone_width',
                          'reported_expectation','verified_bound',
                          'arithmetic_lower_requirement','witness_complex_entries'):
                self.assertEqual(ours[field],control[field],field)
            self.assertEqual(ours['checkpoints_checked'],ours['segments_checked']+1)

    def test_two_worker_same_as_serial(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp,events_per_segment=8)
            serial=verify(path,workers=1)
            parallel=verify(path,workers=2)
            for field in ('events_checked','event_counts','verified_bound',
                          'arithmetic_lower_requirement','witness_complex_entries',
                          'reported_expectation'):
                self.assertEqual(serial[field],parallel[field])

    def test_each_segment_verified_and_anchored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp,events_per_segment=13)
            report=verify(path,workers=2)
            for i in range(report['segments_checked']):
                self.assertEqual(verify_one_segment(path,i)['segment'],i)

    def test_isolated_python_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp,events_per_segment=16)
            process=subprocess.run([sys.executable,'-I',str(HERE/'qc17_checker.py'),
                                    str(path),'--workers','2'],
                                   capture_output=True,text=True,timeout=30)
            self.assertEqual(process.returncode,0,process.stderr)
            self.assertEqual(json.loads(process.stdout)['status'],'PASS')

    def test_cross_channel_small_exact_reference(self):
        for channel in ('depolarizing','dephasing','amplitude_damping'):
            with self.subTest(channel=channel),tempfile.TemporaryDirectory() as tmp:
                proof=export_witness(3,gate_layers(3,3,11),kraus(channel,.5),
                                     (0,), 'Z', bond_cap=3)
                path=self.create(tmp,events_per_segment=5,proof=proof)
                report=verify(path,workers=2)
                oracle=check_witness(proof,kernel=Kernel())
                self.assertEqual(report['verified_bound'],oracle['verified_bound'])
                self.assertEqual(report['arithmetic_lower_requirement'],
                                 oracle['arithmetic_lower_requirement'])

    def test_modified_checkpoint_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp)
            manifest=read_proof(path)
            file=path.parent/manifest['checkpoints'][1]['filename']
            raw=bytearray(file.read_bytes());raw[-10]^=0x10;file.write_bytes(raw)
            with self.assertRaises(CertificateError):verify(path)

    def test_checkpoint_false_digest_with_rehashed_manifest_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp)
            manifest=read_proof(path)
            j=1
            checkpoint=manifest['checkpoints'][j]
            file=path.parent/checkpoint['filename']
            payload=read_proof(file)
            payload['commitment']='a'*64
            write_proof(payload,file)
            checkpoint['commitment']='a'*64
            checkpoint['file_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
            manifest['segments'][j-1]['end_commitment']='a'*64
            manifest['segments'][j]['start_commitment']='a'*64
            # Repair both affected segment envelopes and checksums as well.
            for record in manifest['segments'][j-1:j+1]:
                sfile=path.parent/record['filename']
                seg=read_proof(sfile)
                seg['start_commitment']=record['start_commitment']
                seg['end_commitment']=record['end_commitment']
                write_proof(seg,sfile)
                record['file_sha256']=hashlib.sha256(sfile.read_bytes()).hexdigest()
            write_proof(manifest,path)
            with self.assertRaises(CertificateError):verify(path)


    def test_forged_checkpoint_with_repaired_links_rejected(self):
        # The adversary updates the exact rational error and recalculates the
        # checkpoint hash, both neighboring segment envelopes, and manifest.
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp)
            manifest=read_proof(path)
            j=1
            desc=manifest['checkpoints'][j]
            file=path.parent/desc['filename']
            payload=read_proof(file)
            current=_validate_checkpoint_payload(payload)
            payload['accumulated']=encode_accumulated(current['accumulated']+F(1,8))
            point=checkpoint_payload(current['event_index'],current['first'],
                      current['last'],current['cores'],current['accumulated']+F(1,8),
                      current['tally'])
            payload['commitment']=point['commitment']
            write_proof(payload,file)
            desc['commitment']=point['commitment']
            desc['file_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
            manifest['segments'][j-1]['end_commitment']=point['commitment']
            manifest['segments'][j]['start_commitment']=point['commitment']
            for record in manifest['segments'][j-1:j+1]:
                segment_file=path.parent/record['filename']
                segment=read_proof(segment_file)
                segment['start_commitment']=record['start_commitment']
                segment['end_commitment']=record['end_commitment']
                write_proof(segment,segment_file)
                record['file_sha256']=hashlib.sha256(segment_file.read_bytes()).hexdigest()
            write_proof(manifest,path)
            with self.assertRaisesRegex(CertificateError,'boundary mismatch'):
                verify(path,workers=2)

    def test_pinned_manifest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp)
            with self.assertRaisesRegex(CertificateError,'manifest changed'):
                verify_one_segment(path,0,'0'*64)

    def test_reordered_event_with_all_checksums_recomputed_rejected(self):
        forged=copy.deepcopy(self.full)
        forged['events'][2],forged['events'][3]=forged['events'][3],forged['events'][2]
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp,proof=forged)
            with self.assertRaises(CertificateError):verify(path,workers=2)

    def test_zero_residual_and_all_commitments_recomputed_rejected(self):
        forged=copy.deepcopy(self.full)
        next(x for x in forged['events'] if x['kind']=='gate_svd')['residual_upper']=(0.).hex()
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp,proof=forged)
            with self.assertRaisesRegex(CertificateError,'residual'):verify(path,workers=1)

    def test_understated_final_error_rejected(self):
        forged=copy.deepcopy(self.full)
        forged['result']['error_upper']=(0.).hex()
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp,proof=forged)
            with self.assertRaisesRegex(CertificateError,'bound understated'):verify(path)

    def test_genesis_forge_with_recomputed_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp)
            manifest=read_proof(path)
            desc=manifest['checkpoints'][0]
            file=path.parent/desc['filename']
            data=read_proof(file)
            data['accumulated']['numerator_hex']='1'
            # Digest consistency is checked before genesis; either failure rejects.
            write_proof(data,file)
            desc['file_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
            write_proof(manifest,path)
            with self.assertRaises(CertificateError):verify(path)

    def test_deleted_segment_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp)
            manifest=read_proof(path)
            (path.parent/manifest['segments'][-1]['filename']).unlink()
            with self.assertRaises(CertificateError):verify(path)

    def test_duplicate_checkpoint_filename_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp)
            data=read_proof(path)
            data['checkpoints'][2]['filename']=data['checkpoints'][1]['filename']
            write_proof(data,path)
            with self.assertRaises(CertificateError):verify(path)

    def test_relative_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.create(tmp)
            data=read_proof(path)
            data['checkpoints'][1]['filename']='../escape.json.gz'
            write_proof(data,path)
            with self.assertRaises(CertificateError):verify(path)

    def test_checkpoint_emission_deterministic(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            pa=self.create(a)
            pb=self.create(b)
            aa={x.name:hashlib.sha256(x.read_bytes()).digest() for x in pa.parent.iterdir()}
            bb={x.name:hashlib.sha256(x.read_bytes()).digest() for x in pb.parent.iterdir()}
            self.assertEqual(aa,bb)


if __name__=='__main__':unittest.main()
