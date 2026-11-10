"""QC-19 arithmetic parity, live-cache lifetime, transport, tampering tests."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import weakref
from zipfile import ZipFile

from qc15_checker import (CertificateError, initial_core, read_proof,
                          write_proof, read_tensor)
from qc16_dyadic import Kernel
from qc18_checker import verify as verify18, load_root, verify_one_segment as verify_segment18
from qc19_checker import verify as verify19, verify_one_segment as verify_segment19
from qc19_kernel import LiveKernel
from qc18_tape import TAPE_FILE

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / 'qc18_certificates/10q_d6_mmap.zip'
COMPARE_KEYS = ('verified_bound', 'arithmetic_lower_requirement', 'event_counts',
                'events_checked', 'segments_checked', 'checkpoints_checked',
                'witness_complex_entries')


class LiveKernelTests(unittest.TestCase):
    def test_exact_arithmetic_parity(self):
        a, b = initial_core('X'), initial_core('Z')
        x, y = Kernel(), LiveKernel()
        self.assertEqual(x.frob_norm2(a), y.frob_norm2(a))
        self.assertEqual(x.pair_error2(a,b,a,b), y.pair_error2(a,b,a,b))
        self.assertEqual(x.gram_defect2(a,'L'), y.gram_defect2(a,'L'))
        self.assertEqual(x.context_bound([a,b],(0,)), y.context_bound([a,b],(0,)))
        self.assertEqual(x.exact_readout([a]),y.exact_readout([a]))

    def test_retired_integer_tensor_released(self):
        kernel=LiveKernel()
        source=initial_core('X')
        weak=weakref.ref(source)
        converted=kernel.tensor(source)
        self.assertIn(id(source),kernel.cache)
        key=id(source)
        kernel.retire(source)
        self.assertNotIn(key,kernel.cache)
        del source,converted
        gc.collect()
        self.assertIsNone(weak())

    def test_other_source_cannot_evict(self):
        kernel=LiveKernel()
        a=initial_core('X'); b=initial_core('Z')
        kernel.tensor(a)
        kernel.retire(b)
        self.assertIn(id(a),kernel.cache)

    def test_cache_limit_bounds(self):
        for invalid in (0,257,-1,2.5,True):
            with self.assertRaises(ValueError):LiveKernel(invalid)
        k=LiveKernel(cache_limit=1)
        a=initial_core('X');b=initial_core('Z')
        k.tensor(a); k.tensor(b)
        self.assertEqual(len(k.cache),1)
        self.assertIn(id(b),k.cache)

    def test_repeated_retirement_safe(self):
        k=LiveKernel()
        a=initial_core('X'); x=k.tensor(a)
        k.retire(a); k.retire(a)
        self.assertIsNot(x,k.tensor(a))


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=tempfile.TemporaryDirectory(prefix='qc19_fixture_')
        cls.dir=Path(cls.root.name)
        with ZipFile(FIXTURE) as z:z.extractall(cls.dir)
        cls.manifest=cls.dir/'manifest.json'
        cls.expected=verify18(cls.manifest,workers=1)

    @classmethod
    def tearDownClass(cls):cls.root.cleanup()

    def check_reference(self,result):
        for key in COMPARE_KEYS:
            self.assertEqual(result[key],self.expected[key],key)

    def test_serial_exact_full_parity(self):
        r=verify19(self.manifest,workers=1)
        self.check_reference(r)
        self.assertEqual(r['arithmetic_backend'],'common-dyadic-integer-live-cache')

    def test_two_worker_fork_parity(self):
        self.check_reference(verify19(self.manifest,workers=2,start_method='fork'))

    def test_two_worker_spawn_parity(self):
        self.check_reference(verify19(self.manifest,workers=2,start_method='spawn'))

    def test_all_segment_boundaries_match(self):
        _,source,checkpoints,plan=load_root(self.manifest,corrections=True)
        for i in range(len(source.segments)):
            left=verify_segment18(source,checkpoints,plan,self.dir,i)
            right=verify_segment19(source,checkpoints,plan,self.dir,i)
            self.assertEqual(left,right)

    def test_isolated_cli(self):
        args=[sys.executable,'-I',str(HERE/'qc19_checker.py'),str(self.manifest),
              '--workers','2','--start-method','spawn']
        p=subprocess.run(args,capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stderr)
        self.check_reference(json.loads(p.stdout))

    def test_mutated_witness_rejected_after_rehash(self):
        with tempfile.TemporaryDirectory(prefix='qc19_tamper_') as directory:
            dest=Path(directory)
            with ZipFile(FIXTURE) as z:z.extractall(dest)
            manifest=dest/'manifest.json';tape=dest/TAPE_FILE
            raw=bytearray(tape.read_bytes())
            # Change framed data while keeping its tape digest consistent.
            raw[-41]^=1
            tape.write_bytes(raw)
            doc=read_proof(manifest)
            doc['tape']['sha256']=hashlib.sha256(raw).hexdigest()
            write_proof(doc,manifest)
            with self.assertRaises(CertificateError):verify19(manifest)

    def test_missing_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory(prefix='qc19_missing_') as directory:
            dest=Path(directory)
            with ZipFile(FIXTURE) as z:z.extractall(dest)
            manifest=dest/'manifest.json';doc=read_proof(manifest)
            (dest/doc['checkpoints'][1]['filename']).unlink()
            with self.assertRaises(CertificateError):verify19(manifest)

    def test_standard_library_checker_imports(self):
        src=(HERE/'qc19_checker.py').read_text()
        kernel=(HERE/'qc19_kernel.py').read_text()
        self.assertNotIn('import numpy',src+kernel)
        self.assertNotIn('import torch',src+kernel)


if __name__=='__main__':unittest.main()
