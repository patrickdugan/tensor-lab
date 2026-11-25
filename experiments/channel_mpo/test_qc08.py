"""QC-08: spectral equivalence, stable streaming gates, and budget invariants."""
from __future__ import annotations
import unittest
from unittest.mock import patch
import numpy as np
from channel_mpo import gate_layers, kraus, apply_channel, apply_operator
from qc07_joint import JointPurifiedChain, spatial_spectra, allocate_joint
from qc06_adaptive import scan_ancilla_spectra
from qc08_streaming import (StreamingJointChain, _gram_singular_values,
                           scan_joint_spectra)
from benchmark_qc08 import run_case
from benchmark_qc07 import trace_distance


class TestQC08(unittest.TestCase):
    def test_gram_singular_values_reference(self):
        rng=np.random.default_rng(15)
        for shape in ((3,9),(12,4),(10,10),(2,2)):
            m=rng.normal(size=shape)+1j*rng.normal(size=shape)
            np.testing.assert_allclose(_gram_singular_values(m),
                                       np.linalg.svd(m,compute_uv=False),rtol=1e-11,atol=1e-11)
        m=np.diag([1.,1e-12]).astype(complex)
        np.testing.assert_allclose(_gram_singular_values(m),[1.,1e-12],rtol=1e-7,atol=1e-13)
        with self.assertRaises(ValueError):_gram_singular_values(np.ones((0,3)))

    def test_one_scan_matches_two_individual_scans(self):
        for n in (3,5):
            source=JointPurifiedChain(n,max_bond=14,max_env=12)
            ops=kraus('depolarizing',.35)
            for i,u in gate_layers(n,1,2)[0]:source.gate(i,u)
            for i in range(n):source.channel(i,ops)
            prior=source.dense_density()
            baseline=JointPurifiedChain(n,max_bond=14,max_env=12)
            baseline.cores=[a.copy() for a in source.cores]
            bs=spatial_spectra(baseline)
            es,_=scan_ancilla_spectra(baseline)
            b2,e2,st=scan_joint_spectra(source)
            self.assertEqual(st.right_canonical_sweeps,1)
            self.assertEqual(st.forward_qrs,n-1)
            for x,y in zip(bs,b2):np.testing.assert_allclose(x,y,atol=1e-11,rtol=1e-10)
            for x,y in zip(es,e2):np.testing.assert_allclose(x,y,atol=1e-11,rtol=1e-10)
            np.testing.assert_allclose(source.dense_density(),prior,atol=1e-12)

    def test_stream_gates_match_sequential_with_truncation(self):
        for n in (4,6):
            orig=JointPurifiedChain(n,max_bond=5,max_env=8)
            stream=StreamingJointChain(n,max_bond=5,max_env=8)
            for layer in gate_layers(n,3,7):
                for i,u in layer:orig.gate(i,u)
                self.assertEqual(stream.gate_layer(layer),len(layer))
                np.testing.assert_allclose(orig.dense_density(),stream.dense_density(),atol=3e-12)
                self.assertAlmostEqual(orig.error_bound_sum,stream.error_bound_sum,places=10)
                for i in range(n):
                    orig.channel(i,kraus('dephasing',.2))
                    stream.channel(i,kraus('dephasing',.2))
                for i in range(n):
                    orig.compress_env(i);stream.compress_env(i)
                orig.normalize();stream.normalize()

    def test_gate_layers_reject_nonmonotonic_or_overlapping(self):
        stream=StreamingJointChain(6)
        before=stream.dense_density()
        with self.assertRaises(ValueError):stream.gate_layer([(2,np.eye(4)),(1,np.eye(4))])
        with self.assertRaises(ValueError):stream.gate_layer([(1,np.eye(4)),(2,np.eye(4))])
        with self.assertRaises(ValueError):stream.gate_layer([(5,np.eye(4))])
        np.testing.assert_allclose(stream.dense_density(),before)

    def test_classic_matches_qc07_full_circuit(self):
        row=run_case(5,3,.2,depth=3,budgets=[150,900,2800])
        self.assertTrue(row['classic_plan_equal_qc07'])
        self.assertLess(row['classic_density_distance_qc07'],1e-10)
        self.assertAlmostEqual(row['classic_trace_distance'],row['qc07_trace_distance'],places=10)
        self.assertEqual(row['classic_final_entries'] if 'classic_final_entries' in row else row['classic_entries'],
                         row['qc07_entries'])

    def test_fused_mode_keeps_physical_state_and_budget(self):
        n=4;ops=kraus('amplitude_damping',.4)
        sim=StreamingJointChain(n,max_bond=4,max_env=7)
        for layer in gate_layers(n,3,5):
            sim.gate_layer(layer)
            for i in range(n):sim.channel(i,ops)
            r=sim.compress_layer_stream(250,ops,bond_cap=4,env_cap=7,order='fused')
            self.assertLessEqual(r.retained_entries,250)
            self.assertGreaterEqual(r.min_local_retention,0)
            self.assertEqual(r.compression_svds,2*n-1)
            self.assertEqual(r.compression_right_qrs,1)
            sim.normalize()
        rho=sim.dense_density()
        self.assertAlmostEqual(np.trace(rho).real,1.,places=10)
        self.assertGreaterEqual(np.linalg.eigvalsh(rho).min(),-1e-10)

    def test_joint_budget_no_oracle_queried(self):
        for order in ('classic','fused'):
            sim=StreamingJointChain(4,max_bond=12,max_env=12)
            ops=kraus('depolarizing',.25)
            sim.gate_layer(gate_layers(4,1,3)[0])
            for i in range(4):sim.channel(i,ops)
            with patch.object(sim,'dense_density',side_effect=AssertionError('oracle leakage')):
                r=sim.compress_layer_stream(600,ops,order=order)
            self.assertLessEqual(r.retained_entries,600)

    def test_untruncated_state_matches_dense(self):
        n=3;ops=kraus('dephasing',.3)
        sim=StreamingJointChain(n,max_bond=30,max_env=30)
        rho=np.zeros((8,8),complex);rho[0,0]=1
        for layer in gate_layers(n,2,1):
            sim.gate_layer(layer)
            for i,u in layer:rho=apply_operator(rho,u,(i,i+1),n)
            for i in range(n):
                sim.channel(i,ops);rho=apply_channel(rho,ops,i,n)
            sim.compress_layer_stream(30000,ops,bond_cap=30,env_cap=30,order='fused')
            sim.normalize()
        np.testing.assert_allclose(sim.dense_density(),rho,atol=5e-11)

    def test_work_cap_rejection_before_mutation(self):
        sim=StreamingJointChain(3,max_bond=4,max_env=6,work_cap_entries=20)
        before=sim.dense_density()
        with self.assertRaises(MemoryError):sim.gate_layer([(0,np.eye(4))])
        np.testing.assert_allclose(sim.dense_density(),before)
        sim2=StreamingJointChain(3,max_bond=4,max_env=6)
        with self.assertRaises(ValueError):sim2.compress_layer_stream(5,kraus('dephasing',.5))

    def test_diagnostic_marginals_optional(self):
        sim=StreamingJointChain(3,max_bond=10,max_env=10)
        ops=kraus('dephasing',.3)
        sim.gate_layer(gate_layers(3,1,0)[0])
        for i in range(3):sim.channel(i,ops)
        result=sim.compress_layer_stream(200,ops,order='classic',track_marginals=True)
        self.assertEqual(len(result.marginal_changes),3)
        self.assertTrue(all(z>=0 for z in result.marginal_changes))


if __name__=='__main__':unittest.main()
