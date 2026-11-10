"""QC-22 kernel-sealed witness regression and adversarial tests (Linux only)."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from qc15_checker import CertificateError, read_proof
from qc18_tape import TapeEvents, TAPE_FILE
from qc20_checker import verify as verify20
from qc22_checker import verify as verify22
import qc22_checker
from qc22_seal import (MANDATORY_SEALS, check_sealed, require_sealing,
                       seal_tape)

HERE=Path(__file__).resolve().parent
FIXTURE=HERE/'qc18_certificates/10q_d6_mmap.zip'
COMPARE=('verified_bound','arithmetic_lower_requirement','event_counts',
         'events_checked','segments_checked','checkpoints_checked',
         'witness_complex_entries')

@unittest.skipUnless(sys.platform.startswith('linux') and hasattr(os,'memfd_create'),
                     'Linux memfd seals required')
class SealedTapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_sealing()
        cls.tmp=tempfile.TemporaryDirectory(prefix='qc22_fixture_')
        cls.directory=Path(cls.tmp.name)
        with ZipFile(FIXTURE) as fixture:
            fixture.extractall(cls.directory)
        cls.manifest=cls.directory/'manifest.json'
        cls.description=read_proof(cls.manifest)
        cls.baseline=verify20(cls.manifest,workers=1,storage_mode='summary')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def scratch(self):
        tmp=tempfile.TemporaryDirectory(prefix='qc22_tamper_')
        root=Path(tmp.name)
        for item in self.directory.iterdir():
            shutil.copy2(item,root/item.name)
        self.addCleanup(tmp.cleanup)
        return root

    def sealed_source(self, root=None):
        root=root or self.directory
        description=read_proof(root/'manifest.json')
        source=TapeEvents(description,root)
        self.addCleanup(source.close)
        return seal_tape(source,description['tape']['sha256'])

    def assert_same(self,result):
        for key in COMPARE:
            self.assertEqual(self.baseline[key],result[key],key)

    def test_serial_exact_result_and_memfd(self):
        result=verify22(self.manifest,workers=1)
        self.assert_same(result)
        self.assertEqual(result['transport'],
            'sha256-pinned-F_SEAL_WRITE-grow-shrink-seal-memfd')
        self.assertEqual(result['sealed_sha256'],self.description['tape']['sha256'])
        self.assertGreater(result['sealed_bytes'],0)

    def test_two_fork_workers_independent_result(self):
        self.assert_same(verify22(self.manifest,workers=2))

    def test_four_fork_workers_independent_result(self):
        self.assert_same(verify22(self.manifest,workers=4))

    def test_full_kernel_seals_and_denied_writes(self):
        source=self.sealed_source()
        fd=source.handle.fileno()
        self.assertEqual(check_sealed(fd)&MANDATORY_SEALS,MANDATORY_SEALS)
        self.assertIn('/memfd:qc22-verified-event-tape',
                      os.readlink(f'/proc/self/fd/{fd}'))
        for attack in (lambda:os.pwrite(fd,b'Z',0),
                       lambda:os.ftruncate(fd,0),
                       lambda:os.ftruncate(fd,len(source.mm)+1),
                       lambda:fcntl.fcntl(fd,fcntl.F_ADD_SEALS,1)):
            with self.assertRaises(OSError) as failure:attack()
            self.assertIn(failure.exception.errno,(errno.EPERM,errno.EACCES))
        self.assertIsInstance(source[0],dict)

    def test_mutable_source_changed_after_sealing(self):
        root=self.scratch()
        source=self.sealed_source(root)
        first=source[0]
        with (root/TAPE_FILE).open('r+b') as handle:
            handle.seek(-1,os.SEEK_END)
            value=handle.read(1)
            handle.seek(-1,os.SEEK_END)
            handle.write(bytes([value[0]^0x80]))
        self.assertEqual(source[0],first)
        self.assertEqual(hashlib.sha256(source.mm).hexdigest(),
                         self.description['tape']['sha256'])
        with self.assertRaises(CertificateError):
            TapeEvents(read_proof(root/'manifest.json'),root)

    def test_full_verifier_uses_sealed_state_after_disk_mutation(self):
        root=self.scratch()
        real_seal=seal_tape
        def seal_then_corrupt(source,digest):
            result=real_seal(source,digest)
            with (root/TAPE_FILE).open('r+b') as stream:
                stream.seek(-1,os.SEEK_END)
                value=stream.read(1)
                stream.seek(-1,os.SEEK_END)
                stream.write(bytes([value[0]^0x80]))
            return result
        with patch.object(qc22_checker,'seal_tape',seal_then_corrupt):
            self.assert_same(verify22(root/'manifest.json',workers=2))

    def test_preseal_mutation_rejected(self):
        root=self.scratch()
        path=root/TAPE_FILE
        with path.open('r+b') as stream:
            stream.seek(-1,os.SEEK_END)
            val=stream.read(1)
            stream.seek(-1,os.SEEK_END)
            stream.write(bytes([val[0]^1]))
        with self.assertRaises(CertificateError):
            verify22(root/'manifest.json')

    def test_checkpoint_rechecked_against_original_manifest(self):
        root=self.scratch()
        real_seal=seal_tape
        checkpoints=self.description['checkpoints']
        def seal_then_rewrite(source,digest):
            result=real_seal(source,digest)
            path=root/checkpoints[-1]['filename']
            with path.open('ab') as stream:
                stream.write(b'!')
            return result
        with patch.object(qc22_checker,'seal_tape',seal_then_rewrite):
            with self.assertRaisesRegex(CertificateError,'checkpoint file checksum'):
                verify22(root/'manifest.json')

    def test_seal_failure_fails_closed(self):
        import qc22_seal
        source=TapeEvents(self.description,self.directory)
        self.addCleanup(source.close)
        old=fcntl.fcntl
        def fail_add(fd,operation,*args):
            if operation==fcntl.F_ADD_SEALS:
                raise OSError(errno.EPERM,'seals unavailable')
            return old(fd,operation,*args)
        with patch.object(qc22_seal.fcntl,'fcntl',side_effect=fail_add):
            with self.assertRaisesRegex(CertificateError,'sealing failed'):
                seal_tape(source,self.description['tape']['sha256'])
        self.assertFalse(getattr(source,'sealed',False))

    def test_required_seals_fails_on_partial_set(self):
        fd=os.memfd_create('qc22-incomplete',os.MFD_ALLOW_SEALING)
        try:
            fcntl.fcntl(fd,fcntl.F_ADD_SEALS,fcntl.F_SEAL_SHRINK)
            with self.assertRaisesRegex(CertificateError,'unsealed or partially'):
                check_sealed(fd)
        finally:os.close(fd)

    def test_repeat_seal_rejected(self):
        source=self.sealed_source()
        with self.assertRaisesRegex(CertificateError,'fresh unsealed'):
            seal_tape(source,self.description['tape']['sha256'])

    def test_spawn_rejected_explicitly(self):
        with self.assertRaisesRegex(CertificateError,'requires fork'):
            verify22(self.manifest,workers=2,start_method='spawn')

    def test_bad_manifest_digest_fails_sealing(self):
        source=TapeEvents(self.description,self.directory)
        self.addCleanup(source.close)
        with self.assertRaisesRegex(CertificateError,'source tape changed'):
            seal_tape(source,'0'*64)

    def test_cli_no_site(self):
        cmd=[sys.executable,'-I','-S',str(HERE/'qc22_checker.py'),
             str(self.manifest),'--workers','2']
        output=subprocess.run(cmd,capture_output=True,text=True,timeout=35)
        self.assertEqual(output.returncode,0,output.stderr)
        self.assert_same(json.loads(output.stdout))

    def test_producer_is_never_imported(self):
        for name in ('qc22_seal.py','qc22_checker.py'):
            source=(HERE/name).read_text()
            self.assertNotIn('import numpy',source)
            self.assertNotIn('import qc15_emit',source)
            self.assertNotIn('import qc14_certified',source)


if __name__=='__main__':unittest.main()
