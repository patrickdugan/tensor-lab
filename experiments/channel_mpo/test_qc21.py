"""QC-21 independent checker tests for norm-summary retention and packed anchors."""
from __future__ import annotations

from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

from qc15_checker import CertificateError,initial_core
from qc17_checker import _check_genesis,_check_final
from qc17_format import read_checkpoint
from qc20_packed import StreamFastKernel,pack_tensor,read_checkpoint_packed
from qc20_checker import load_root,verify as verify20
from qc21_anchor import check_genesis,check_final
from qc21_checker import verify as verify21
from qc21_kernel import SummaryKernel

HERE=Path(__file__).resolve().parent
FIXTURE=HERE/'qc18_certificates/10q_d6_mmap.zip'
MATCH=('verified_bound','arithmetic_lower_requirement','event_counts',
       'events_checked','segments_checked','checkpoints_checked',
       'witness_complex_entries')


class SummaryArithmeticTests(unittest.TestCase):
    def test_exact_norm_and_gram_parity(self):
        k1=StreamFastKernel()
        k2=SummaryKernel(cache_limit=6)
        for name in ('I','X','Y','Z'):
            core=pack_tensor(initial_core(name))
            self.assertEqual(k1.frob_norm2(core),k2.frob_norm2(core))
            for direction in ('L','R'):
                self.assertEqual(k1.gram_defect2(core,direction),
                                 k2.gram_defect2(core,direction))
                self.assertEqual(k1.gram_defect2(core,direction),
                                 k2.gram_defect2(core,direction))
        self.assertGreater(k2.summary_hits,0)
        self.assertGreater(k2.summary_misses,0)

    def test_context_and_residual_parity(self):
        cores=[pack_tensor(initial_core(name)) for name in 'IXYZIX']
        a=StreamFastKernel();b=SummaryKernel(cache_limit=4)
        for i in range(len(cores)):
            self.assertEqual(a.context_bound(cores,(i,)),
                             b.context_bound(cores,(i,)))
            self.assertEqual(a.context_bound(cores,(i,),gate=True),
                             b.context_bound(cores,(i,),gate=True))
        for i in range(len(cores)-1):
            self.assertEqual(a.pair_error2(cores[i],cores[i+1],cores[i],cores[i+1]),
                             b.pair_error2(cores[i],cores[i+1],cores[i],cores[i+1]))
        self.assertGreater(b.summary_hits,8)
        self.assertLessEqual(len(b.cache),4)

    def test_retirement_and_version_safety(self):
        k=SummaryKernel(cache_limit=2)
        a=pack_tensor(initial_core('Z'))
        self.assertEqual(k.frob_norm2(a),F(2))
        self.assertIn(id(a),k.norm_summaries)
        k.retire(a)
        self.assertNotIn(id(a),k.norm_summaries)
        b=pack_tensor(initial_core('X'))
        self.assertEqual(k.frob_norm2(b),F(2))
        self.assertIn(id(b),k.norm_summaries)
        self.assertNotEqual(a.raw,b.raw)
        k.retire(b)
        self.assertFalse(k.norm_summaries)

    def test_orientation_guard_even_when_cached(self):
        k=SummaryKernel()
        a=pack_tensor(initial_core('Z'))
        k.gram_defect2(a,'L')
        with self.assertRaises(ValueError):k.gram_defect2(a,'J')

    def test_cache_bound_under_core_churn(self):
        k=SummaryKernel(cache_limit=3)
        for i in range(20):
            src=pack_tensor(initial_core('Z' if i%2 else 'X'))
            k.frob_norm2(src)
            self.assertLessEqual(len(k.cache),3)
            k.retire(src)
            self.assertLessEqual(len(k.cache),3)
            self.assertFalse(k.norm_summaries)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='qc21_')
        cls.directory=Path(cls.tmp.name)
        with ZipFile(FIXTURE) as zipfile:
            zipfile.extractall(cls.directory)
        cls.manifest=cls.directory/'manifest.json'
        cls.base=verify20(cls.manifest,workers=1)
        cls.manifest_object,cls.source,cls.checkpoints,cls.plan=load_root(cls.manifest)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def assert_same(self,result):
        for k in MATCH:self.assertEqual(self.base[k],result[k],k)

    def test_genesis_and_final_packed_exact_parity(self):
        first=self.checkpoints[0];last=self.checkpoints[-1]
        oldfirst=read_checkpoint(self.directory,first)
        newfirst=read_checkpoint_packed(self.directory,first)
        _check_genesis(oldfirst,self.plan)
        check_genesis(newfirst,self.plan)
        oldlast=read_checkpoint(self.directory,last)
        newlast=read_checkpoint_packed(self.directory,last)
        expected=_check_final(oldlast,self.plan,self.manifest_object['result'])
        actual=check_final(newlast,self.plan,self.manifest_object['result'])
        self.assertEqual(expected,actual)

    def test_corrupted_genesis_rejected(self):
        first=read_checkpoint_packed(self.directory,self.checkpoints[0])
        bad={**first,'cores':[pack_tensor(initial_core('X'))]+first['cores'][1:]}
        if first['cores'][0].raw==bad['cores'][0].raw:
            bad['cores'][0]=pack_tensor(initial_core('I'))
        with self.assertRaises(CertificateError):check_genesis(bad,self.plan)

    def test_final_claimed_bound_cannot_shrink(self):
        last=read_checkpoint_packed(self.directory,self.checkpoints[-1])
        forged={**self.manifest_object['result'],'error_upper':float(0).hex()}
        with self.assertRaises(CertificateError):check_final(last,self.plan,forged)

    def test_serial_equality(self):
        self.assert_same(verify21(self.manifest,workers=1))

    def test_fork_equality(self):
        self.assert_same(verify21(self.manifest,workers=2,start_method='fork'))

    def test_spawn_equality(self):
        self.assert_same(verify21(self.manifest,workers=2,start_method='spawn'))

    def test_isolated_command(self):
        cmd=[sys.executable,'-I',str(HERE/'qc21_checker.py'),
             str(self.manifest),'--workers','2','--start-method','spawn']
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assert_same(json.loads(result.stdout))

    def test_no_site_spawn_isolated_command(self):
        # The standalone verifier can run even when sitecustomize and all
        # installed non-standard packages are absent.
        cmd=[sys.executable,'-I','-S',str(HERE/'qc21_checker.py'),
             str(self.manifest),'--workers','2','--start-method','spawn']
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assert_same(json.loads(result.stdout))

    def test_lost_checkpoint_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory(prefix='qc21_missing_') as d:
            dst=Path(d)
            for p in self.directory.iterdir():
                shutil.copy2(p,dst/p.name)
            name=self.checkpoints[2]['filename']
            (dst/name).unlink()
            with self.assertRaises(CertificateError):verify21(dst/'manifest.json',workers=1)

    def test_forged_context_rejected_after_all_tape_hashes_rebuilt(self):
        import hashlib, shutil, struct, zlib
        from qc15_checker import read_proof,write_proof
        from qc18_tape import TapeEvents,MAGIC_ZLIB,TAPE_FILE,_canonical_event
        with tempfile.TemporaryDirectory(prefix='qc21_tamper_') as d:
            dest=Path(d)
            for item in self.directory.iterdir():
                shutil.copy2(item,dest/item.name)
            manifest=read_proof(dest/'manifest.json')
            tape=TapeEvents(manifest,dest)
            events=[tape[i] for i in range(len(tape))]
            tape.close()
            selected=next(i for i,event in enumerate(events)
                          if event.get('kind') in ('channel','gate_svd','left_qr','right_qr'))
            events[selected]['context_upper']=float(0).hex()
            frames=[];index=bytearray();offset=0
            for event in events:
                data=zlib.compress(_canonical_event(event),level=3)
                frame=struct.pack('<I',len(data))+data
                frames.append(frame)
                index.extend(struct.pack('<QI',offset+4,len(data)))
                offset+=len(frame)
            raw=MAGIC_ZLIB+struct.pack('<I',len(events))+bytes(index)+b''.join(frames)
            (dest/TAPE_FILE).write_bytes(raw)
            manifest['tape']['bytes']=len(raw)
            manifest['tape']['sha256']=hashlib.sha256(raw).hexdigest()
            for segment in manifest['segments']:
                first=segment['start_index'];last=first+segment['count']
                segment['events_sha256']=hashlib.sha256(b''.join(frames[first:last])).hexdigest()
            write_proof(manifest,dest/'manifest.json')
            with self.assertRaisesRegex(CertificateError,'context norm upper'):
                verify21(dest/'manifest.json',workers=1)

    def test_standard_library_checker_only(self):
        for name in ('qc21_kernel.py','qc21_anchor.py','qc21_checker.py'):
            txt=(HERE/name).read_text()
            self.assertNotIn('import numpy',txt)
            self.assertNotIn('import torch',txt)
            self.assertNotIn('import qc14_certified',txt)


if __name__=='__main__':unittest.main()
