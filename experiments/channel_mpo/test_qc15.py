"""QC-15 independent-checker and tamper-resistance regressions."""
from __future__ import annotations
import base64
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

from channel_mpo import gate_layers,kraus,I2
from qc14_certified import certify_observable
from qc15_checker import (check_witness,check_ledger,check_certificate,
                          write_proof,read_proof,CertificateError)
from qc15_emit import export_witness,export_ledger


class FullWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layers=gate_layers(3,3,3)
        cls.ks=kraus('dephasing',.5)
        cls.good=export_witness(3,cls.layers,cls.ks,(0,),'Z',bond_cap=4)

    def verify_bad(self,transform):
        changed=copy.deepcopy(self.good)
        transform(changed)
        with self.assertRaises(CertificateError):
            check_witness(changed)

    def test_standalone_replay_matches_qc14(self):
        result=check_witness(self.good)
        qc14=certify_observable(3,self.layers,self.ks,(0,),'Z',bond_cap=4)
        self.assertEqual(result['proof_level'],'full-witness')
        self.assertEqual(result['events_checked'],len(qc14.events))
        self.assertEqual(result['verified_bound'],qc14.reported_error_upper)
        self.assertAlmostEqual(result['reported_expectation'],qc14.reported_expectation)
        self.assertLessEqual(result['arithmetic_lower_requirement'],result['verified_bound'])

    def test_three_noise_channels_and_rank_caps(self):
        for noise in ('depolarizing','dephasing','amplitude_damping'):
            for cap in (1,4):
                with self.subTest(channel=noise,cap=cap):
                    proof=export_witness(3,self.layers,kraus(noise,.5),(0,),'Z',bond_cap=cap)
                    result=check_witness(proof)
                    self.assertGreaterEqual(result['verified_bound'],0)

    def test_10q_cone_and_growing_support(self):
        layers=gate_layers(10,6,3)
        proof=export_witness(10,layers,kraus('dephasing',.5),(0,),'Z',bond_cap=4)
        result=check_witness(proof)
        self.assertEqual(result['cone_width'],6)
        self.assertEqual(result['events_checked'],48)
        self.assertEqual(sum(ev['kind']=='grow' for ev in proof['events']),6)
        self.assertLess(result['verified_bound'],.05)

    def test_polar_nonunitary_gate(self):
        u=np.diag([1.01,1.,1.,1.]).astype(complex)
        proof=export_witness(2,[[(0,u)]],(I2,),(0,),'Z',bond_cap=4)
        checked=check_witness(proof)
        self.assertGreater(checked['verified_bound'],.04)

    def test_forged_residual(self):
        self.verify_bad(lambda q: next(ev for ev in q['events'] if ev['kind']=='gate_svd').update(
            residual_upper=(0.).hex()))

    def test_forged_context(self):
        self.verify_bad(lambda q: next(ev for ev in q['events'] if ev['kind']=='channel').update(
            context_upper=(0.).hex()))

    def test_forged_polar(self):
        self.verify_bad(lambda q: next(ev for ev in q['events'] if ev['kind']=='gate_svd').update(
            polar_upper=(0.).hex()))

    def test_forged_global_bound(self):
        self.verify_bad(lambda q:q['result'].update(error_upper=(0.).hex()))

    def test_missing_or_reordered_event(self):
        self.verify_bad(lambda q:q['events'].pop(2))
        def swap(q):q['events'][0],q['events'][1]=q['events'][1],q['events'][0]
        self.verify_bad(swap)

    def test_forged_new_core(self):
        def mutate(q):
            event=next(e for e in q['events'] if e['kind']=='gate_svd')
            core=event['new_cores'][0]
            raw=bytearray(base64.b64decode(core['binary64_le']))
            # Force the leading complex coefficient to a large exact dyadic value.
            import struct
            struct.pack_into('<d',raw,0,22.0)
            core['binary64_le']=base64.b64encode(raw).decode('ascii')
        self.verify_bad(mutate)

    def test_forged_gate(self):
        def mutate(q):
            gate=q['circuit']['layers'][0][0]['gate']
            gate['binary64_le']=base64.b64encode(bytes(len(base64.b64decode(gate['binary64_le'])))).decode('ascii')
        self.verify_bad(mutate)

    def test_unsigned_payload_recomputed_checksum_cannot_bypass_arithmetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'witness.json.gz'
            changed=copy.deepcopy(self.good)
            changed['result']['error_upper']=(0.).hex()
            write_proof(changed,path)
            with self.assertRaises(CertificateError):
                check_certificate(path,require_witness=True)
            good=Path(tmp)/'good.json.gz'
            write_proof(self.good,good)
            self.assertEqual(check_certificate(good,require_witness=True)['proof_level'],'full-witness')
            # Rehashing is only an integrity check; silently changing a payload
            # while retaining its old digest must fail before proof replay.
            import gzip
            data=json.loads(gzip.decompress(good.read_bytes()))
            data['result']['error_upper']=(0.).hex()
            good.write_bytes(gzip.compress(json.dumps(data).encode(),mtime=0))
            with self.assertRaises(CertificateError):
                check_certificate(good,require_witness=True)

    def test_canonical_sha_and_standard_library_only_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'witness.json.gz'
            write_proof(self.good,target)
            original=target.read_bytes()
            write_proof(self.good,target)
            self.assertEqual(original,target.read_bytes())
            tool=Path(__file__).with_name('qc15_checker.py')
            run=subprocess.run([sys.executable,'-I',str(tool),str(target),'--require-witness'],
                               capture_output=True,text=True,timeout=30)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(json.loads(run.stdout)['status'],'PASS')
            obj=read_proof(target)
            self.assertEqual(obj['format'],self.good['format'])

    def test_fail_closed_incomplete_records(self):
        self.verify_bad(lambda q:q['circuit'].update(qubits=2))
        self.verify_bad(lambda q:q['result'].update(cone_width=999))
        self.verify_bad(lambda q:q['events'][0].update(layer=123))
        self.verify_bad(lambda q:q.update(format='unknown-v99'))


class LedgerTests(unittest.TestCase):
    def test_ledger_only_conditional(self):
        layers=gate_layers(3,3,4)
        proof=export_ledger(3,layers,kraus('amplitude_damping',.5),(0,),'Z',bond_cap=2)
        verified=check_ledger(proof)
        self.assertEqual(verified['proof_level'],'conditional-ledger-only')
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'ledger.json.gz'
            write_proof(proof,file)
            self.assertEqual(check_certificate(file)['proof_level'],'conditional-ledger-only')
            with self.assertRaises(CertificateError):
                check_certificate(file,require_witness=True)
        bad=copy.deepcopy(proof)
        bad['result']['error_upper']=(0.).hex()
        with self.assertRaises(CertificateError):check_ledger(bad)

    def test_scalar_truncation_and_negative_alpha_fail(self):
        proof=export_ledger(3,gate_layers(3,3,4),kraus('dephasing',.2),(0,),'Z')
        broken=copy.deepcopy(proof)
        broken['events'][0]['polar_upper']=(-1.).hex()
        with self.assertRaises(CertificateError):check_ledger(broken)
        broken=copy.deepcopy(proof)
        broken['events'][0]['context_upper']='nan'
        with self.assertRaises(CertificateError):check_ledger(broken)

if __name__=='__main__':unittest.main()
