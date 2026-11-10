"""QC-18 framed shared-memory witness transport and independent replay tests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from zipfile import ZipFile,ZIP_DEFLATED

from qc15_checker import CertificateError,_canonical_bytes,read_proof,write_proof
from qc17_checker import verify as qc17_verify
from qc18_checker import verify as qc18_verify,verify_one_segment,load_root
from qc18_tape import (emit,TapeEvents,TAPE_FILE,MAGIC_RAW,MAGIC_ZLIB,
                       _canonical_event)

HERE=Path(__file__).resolve().parent
FIXTURE=HERE/'qc17_certificates/10q_d6_checkpointed.zip'


class QC18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=tempfile.TemporaryDirectory(prefix='qc18_reference_')
        cls.ref_dir=Path(cls.reference.name)
        with ZipFile(FIXTURE) as archive:archive.extractall(cls.ref_dir)
        cls.reference_result=qc17_verify(cls.ref_dir/'manifest.json',workers=1)

    @classmethod
    def tearDownClass(cls):cls.reference.cleanup()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='qc18_test_')
        self.root=Path(self.temp.name)
        self.manifest=emit(self.ref_dir/'manifest.json',self.root/'proof')

    def tearDown(self):self.temp.cleanup()

    def compare(self,r):
        for key in ('verified_bound','arithmetic_lower_requirement','events_checked',
                    'event_counts','witness_complex_entries','cone_width'):
            self.assertEqual(r[key],self.reference_result[key],key)

    def test_serial_same_exact_result(self):
        self.compare(qc18_verify(self.manifest,workers=1))

    def test_parallel_fork_same_exact_result(self):
        self.compare(qc18_verify(self.manifest,workers=2,start_method='fork'))

    def test_parallel_spawn_same_exact_result(self):
        self.compare(qc18_verify(self.manifest,workers=2,start_method='spawn'))

    def test_independent_replay_each_segment(self):
        m,s,c,p=load_root(self.manifest,corrections=True)
        for i in range(len(s.segments)):
            result=verify_one_segment(s,c,p,self.manifest.parent,i)
            self.assertEqual(result['segment'],i)

    def test_one_event_at_a_time_no_cached_json_events(self):
        m,s,c,p=load_root(self.manifest,corrections=False)
        self.assertFalse(hasattr(s,'cached_events'))
        self.assertTrue(s.mm.readonly if hasattr(s.mm,'readonly') else True)
        event=s[0]
        self.assertIsInstance(event,dict)
        self.assertEqual(event['kind'],'channel')

    def test_raw_and_compressed_transport_match(self):
        raw=emit(self.ref_dir/'manifest.json',self.root/'raw',compressed=False)
        self.assertGreater((raw.parent/TAPE_FILE).stat().st_size,
                           (self.manifest.parent/TAPE_FILE).stat().st_size)
        self.compare(qc18_verify(raw,workers=1))

    def test_producer_isolated_cli(self):
        output=self.root/'isolated'
        proc=subprocess.run([sys.executable,'-I',str(HERE/'qc18_tape.py'),
                             str(self.ref_dir/'manifest.json'),
                             '--out-dir',str(output)],
                            capture_output=True,text=True,timeout=20)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.compare(qc18_verify(output/'manifest.json',workers=1))

    def test_standard_library_isolated_cli(self):
        proc=subprocess.run([sys.executable,'-I',str(HERE/'qc18_checker.py'),
                             str(self.manifest),'--workers','2'],
                            capture_output=True,text=True,timeout=25)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertEqual(json.loads(proc.stdout)['status'],'PASS')

    def test_corrupt_event_file_rejected(self):
        tape=self.manifest.parent/TAPE_FILE
        data=bytearray(tape.read_bytes());data[-71]^=3;tape.write_bytes(data)
        with self.assertRaisesRegex(CertificateError,'tape checksum'):
            qc18_verify(self.manifest)

    def test_trailing_byte_with_repaired_sha_rejected(self):
        tape=self.manifest.parent/TAPE_FILE
        tape.write_bytes(tape.read_bytes()+b'X')
        manifest=read_proof(self.manifest)
        manifest['tape']['sha256']=hashlib.sha256(tape.read_bytes()).hexdigest()
        manifest['tape']['bytes']=tape.stat().st_size
        write_proof(manifest,self.manifest)
        with self.assertRaisesRegex(CertificateError,'trailing tape data'):
            qc18_verify(self.manifest)

    def test_index_overlap_with_repaired_sha_rejected(self):
        tape=self.manifest.parent/TAPE_FILE
        data=bytearray(tape.read_bytes())
        struct.pack_into('<Q',data,len(MAGIC_ZLIB)+4+12,0)
        tape.write_bytes(data)
        manifest=read_proof(self.manifest)
        manifest['tape']['sha256']=hashlib.sha256(data).hexdigest()
        write_proof(manifest,self.manifest)
        with self.assertRaisesRegex(CertificateError,'noncontiguous'):
            qc18_verify(self.manifest)

    def _rewrite_events(self,mutate):
        manifest=read_proof(self.manifest)
        src=TapeEvents(manifest,self.manifest.parent)
        events=[src[i] for i in range(len(src))]
        src.close()
        mutate(events)
        count=len(events)
        index=bytearray();frames=[];offset=0
        for ev in events:
            raw=_canonical_event(ev)
            encoded=zlib.compress(raw,level=3)
            frame=struct.pack('<I',len(encoded))+encoded
            index.extend(struct.pack('<QI',offset+4,len(encoded)))
            frames.append(frame);offset+=len(frame)
        data=MAGIC_ZLIB+struct.pack('<I',count)+bytes(index)+b''.join(frames)
        tape=self.manifest.parent/TAPE_FILE
        tape.write_bytes(data)
        manifest['tape']['sha256']=hashlib.sha256(data).hexdigest()
        manifest['tape']['bytes']=len(data)
        for rec in manifest['segments']:
            h=hashlib.sha256()
            for frame in frames[rec['start_index']:rec['start_index']+rec['count']]:
                h.update(frame)
            rec['events_sha256']=h.hexdigest()
        write_proof(manifest,self.manifest)

    def test_forged_residual_with_all_checksums_rehashed_rejected(self):
        def tamper(events):
            target=next(x for x in events if x.get('kind')=='gate_svd' and
                        float.fromhex(x['residual_upper'])>0)
            target['residual_upper']=(0.).hex()
        self._rewrite_events(tamper)
        with self.assertRaisesRegex(CertificateError,'residual'):
            qc18_verify(self.manifest,workers=2,start_method='fork')

    def test_forged_event_order_rehashed_rejected(self):
        def mutate(events):
            events[2],events[3]=events[3],events[2]
        self._rewrite_events(mutate)
        with self.assertRaises(CertificateError):
            qc18_verify(self.manifest)

    def test_modified_checkpoint_rejected(self):
        manifest=read_proof(self.manifest)
        p=self.manifest.parent/manifest['checkpoints'][1]['filename']
        raw=bytearray(p.read_bytes());raw[10]^=4;p.write_bytes(raw)
        with self.assertRaisesRegex(CertificateError,'checksum'):
            qc18_verify(self.manifest)

    def test_symlink_tape_rejected(self):
        tape=self.manifest.parent/TAPE_FILE
        moved=self.root/'real.q18t'
        tape.rename(moved)
        tape.symlink_to(moved)
        with self.assertRaises(CertificateError):qc18_verify(self.manifest)

    def test_missing_segment_rejected(self):
        manifest=read_proof(self.manifest)
        manifest['segments']=manifest['segments'][:-1]
        write_proof(manifest,self.manifest)
        with self.assertRaises(CertificateError):qc18_verify(self.manifest)


if __name__=='__main__':unittest.main()
