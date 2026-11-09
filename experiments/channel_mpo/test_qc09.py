"""QC-09: independent residual checks, rank budget, physical-state tests."""
from __future__ import annotations
import unittest
from unittest.mock import patch
import numpy as np
from channel_mpo import gate_layers,kraus,apply_operator,apply_channel
from benchmark_qc07 import trace_distance
from qc07_joint import allocate_joint
from qc09_certified import (
  CertifiedPeakChain, certified_gate_factor, gate_live_model,
  projected_gate_peak, projected_next_channel_peak, allocate_peak_safe,
)


class TestQC09(unittest.TestCase):
    def test_independent_residual_certificate_both_methods(self):
        rng=np.random.default_rng(12)
        m=rng.standard_normal((45,32))+1j*rng.standard_normal((45,32))
        for method in ('full','sketch','auto'):
            a,b,c=certified_gate_factor(m,7,method=method,seed=13,chunk_rows=5)
            self.assertEqual(c.matrix_shape,m.shape)
            self.assertLessEqual(c.kept_rank,7)
            self.assertEqual(c.chunks_checked,9)
            self.assertAlmostEqual(c.residual_fro_norm, np.linalg.norm(m-a@b),places=11)
            self.assertAlmostEqual(c.relative_residual, np.linalg.norm(m-a@b)/np.linalg.norm(m),places=12)
            self.assertLess(c.projection_defect,1e-10)
            self.assertGreaterEqual(c.trace_distance_bound,c.relative_residual)

    def test_pure_state_bound_from_generic_residual(self):
        rng=np.random.default_rng(15)
        m=rng.normal(size=(12,20))+1j*rng.normal(size=(12,20))
        for method in ('sketch','full'):
            l,r,cert=certified_gate_factor(m,4,method=method,seed=3)
            psi=m.reshape(-1);phi=(l@r).reshape(-1)
            fidelity=abs(np.vdot(psi,phi))**2/(np.vdot(psi,psi).real*np.vdot(phi,phi).real)
            d=np.sqrt(max(0,1-fidelity))
            self.assertLessEqual(d,cert.trace_distance_bound+1e-12)

    def test_low_rank_sketch_residual_tiny(self):
        rng=np.random.default_rng(12)
        m=rng.normal(size=(50,4)) @ rng.normal(size=(4,35))
        l,r,c=certified_gate_factor(m,4,method='sketch',seed=3,oversample=3)
        self.assertLess(c.relative_residual,1e-12)

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError): certified_gate_factor(np.zeros((0,2)),2)
        with self.assertRaises(ValueError): certified_gate_factor(np.eye(2),0)
        with self.assertRaises(ValueError): certified_gate_factor(np.eye(2),2,method='wrong')
        with self.assertRaises(ValueError): gate_live_model(2,5,12,1,method='invalid')
        with self.assertRaises(FloatingPointError): certified_gate_factor(np.array([[np.nan]]),1)

    def test_peak_allocator_nonbinding_identical(self):
        s=[np.array([4.,3.,1.]),np.array([3.,2.,1.])]
        e=[np.array([4.,2.,1.]),np.array([3.,2.,1.]),np.array([4.,2.,1.])]
        base=allocate_joint(s,e,100,4,4,channel_bond_weight=.8)
        x=allocate_peak_safe(s,e,100,4,4,channel_bond_weight=.8,peak_cap_entries=10_000)
        self.assertEqual(x.joint,base)
        self.assertLessEqual(x.modeled_next_gate_peak_entries,10_000)

    def test_peak_allocator_cap_binding_and_infeasible(self):
        s=[np.array([9.,7.,3.,2.]),np.array([9.,6.,3.,1.])]
        e=[np.array([9.,8.,3.,1.]) for _ in range(3)]
        mini=projected_gate_peak([1,1],[1,1,1],rank_cap=4,oversample=2)
        with self.assertRaises(MemoryError):
            allocate_peak_safe(s,e,200,4,4,peak_cap_entries=mini-1,oversample=2)
        base=allocate_peak_safe(s,e,200,4,4,oversample=2)
        cap=mini+(base.modeled_next_gate_peak_entries-mini)//2
        constrained=allocate_peak_safe(s,e,200,4,4,oversample=2,peak_cap_entries=cap)
        self.assertLessEqual(constrained.modeled_next_gate_peak_entries,cap)
        self.assertLessEqual(constrained.joint.predicted_entries,200)
        self.assertGreaterEqual(constrained.blocked_increments,1)

    def test_channel_envelope_monotone_and_budgeted(self):
        b=(3,4,2);e=(4,5,3,4)
        p=projected_next_channel_peak(b,e,rank_cap=6,kraus_count=4)
        self.assertGreaterEqual(p,projected_next_channel_peak(b,e,rank_cap=6,kraus_count=2))
        self.assertGreaterEqual(projected_next_channel_peak(b,(4,6,3,4),rank_cap=6,kraus_count=4),p)
        self.assertGreaterEqual(p,projected_next_channel_peak(b,(2,2,2,2),rank_cap=6,kraus_count=4))
        s=[np.array([9.,7.,3.,1.]) for _ in range(3)]
        es=[np.array([9.,8.,3.,1.]) for _ in range(4)]
        unbounded=allocate_peak_safe(s,es,170,6,4,kraus_count=4,
                                     account_channel_peak=True,oversample=2)
        cap=int(unbounded.modeled_next_layer_peak_entries*.78)
        constrained=allocate_peak_safe(s,es,170,6,4,kraus_count=4,
                                       account_channel_peak=True,oversample=2,
                                       peak_cap_entries=cap)
        self.assertLessEqual(constrained.modeled_next_layer_peak_entries,cap)
        self.assertLessEqual(constrained.joint.predicted_entries,170)
        self.assertGreaterEqual(constrained.blocked_increments,1)

    def test_peak_plan_48_qubit_workspace_bound(self):
        from benchmark_qc09 import frozen_budgets
        n=48;ops=kraus('depolarizing',.2)
        sim=CertifiedPeakChain(n,max_bond=12,max_env=12,
                gate_method='sketch',oversample=4,peak_gate_cap_entries=550000,
                work_cap_entries=750000)
        for t,layer in enumerate(gate_layers(n,6,3)):
            sim.gate_layer(layer)
            for i in range(n):sim.channel(i,ops)
            r=sim.compress_layer_peak(frozen_budgets(n)[t],ops)
            self.assertLessEqual(r.retained_entries,frozen_budgets(n)[t])
            self.assertLessEqual(sim.modeled_live_peak_entries,550000)
            self.assertLessEqual(r.plan.modeled_next_layer_peak_entries*sim.planner_safety_factor,550000)
        self.assertGreater(sim.trace(),.9999999)
        self.assertLess(sim.trace(),1.0000001)
        self.assertLess(r.normalized_trace_before,.1)

    def test_full_factors_reproduce_qc08_on_small_circuit(self):
        from qc08_streaming import StreamingJointChain
        n=4;ops=kraus('dephasing',.2)
        std=StreamingJointChain(n,max_bond=5,max_env=6)
        cert=CertifiedPeakChain(n,max_bond=5,max_env=6,gate_method='full')
        budgets=[200,500]
        for t,layer in enumerate(gate_layers(n,2,4)):
            for i,u in layer:
                std.gate(i,u);cert.gate(i,u)
            for i in range(n):std.channel(i,ops);cert.channel(i,ops)
            rs=std.compress_layer_stream(budgets[t],ops,order='classic')
            rc=cert.compress_layer_peak(budgets[t],ops)
            std.normalize()
            self.assertEqual((rs.plan.bond_ranks,rs.plan.environment_ranks),
                             (rc.plan.joint.bond_ranks,rc.plan.joint.environment_ranks))
            np.testing.assert_allclose(std.dense_density(),cert.dense_density(),atol=2e-11)
            self.assertLessEqual(rc.retained_entries,budgets[t])

    def test_physical_trace_distance_bounded_one_gate(self):
        n=3;cap=1
        m=CertifiedPeakChain(n,max_bond=cap,max_env=2,gate_method='sketch')
        rho=np.zeros((2**n,2**n),complex);rho[0,0]=1
        site,u=gate_layers(n,1,13)[0][0]
        rho=apply_operator(rho,u,(site,site+1),n)
        m.gate(site,u)
        m.normalize()
        d=trace_distance(rho,m.dense_density())
        self.assertLessEqual(d,m.gate_certificates[-1].trace_distance_bound+1e-11)

    def test_cap_rejects_gate_without_state_change(self):
        m=CertifiedPeakChain(3,max_bond=5,peak_gate_cap_entries=4)
        start=m.dense_density()
        with self.assertRaises(MemoryError):m.gate(0,np.eye(4))
        np.testing.assert_allclose(m.dense_density(),start,atol=1e-12)

    def test_no_dense_oracle_in_allocator(self):
        n=4;ops=kraus('depolarizing',.25)
        m=CertifiedPeakChain(n,max_bond=8,max_env=8,gate_method='full')
        m.gate_layer(gate_layers(n,1,2)[0])
        for i in range(n):m.channel(i,ops)
        with patch.object(m,'dense_density',side_effect=AssertionError('dense oracle')):
            result=m.compress_layer_peak(200,ops)
        self.assertLessEqual(result.retained_entries,200)
        self.assertAlmostEqual(m.trace(),1.,places=10)

    def test_sequential_channel_validity(self):
        n=4;ks=kraus('amplitude_damping',.35)
        sim=CertifiedPeakChain(n,max_bond=9,max_env=9,gate_method='sketch',oversample=4)
        rho=np.zeros((16,16),complex);rho[0,0]=1
        for i,u in gate_layers(n,2,0)[0]:
            sim.gate(i,u);rho=apply_operator(rho,u,(i,i+1),n)
        for i in range(n):sim.channel(i,ks);rho=apply_channel(rho,ks,i,n)
        result=sim.compress_layer_peak(600,ks)
        d=trace_distance(rho,sim.dense_density())
        self.assertLessEqual(d,1.)
        self.assertAlmostEqual(np.trace(sim.dense_density()).real,1.,places=10)
        self.assertGreaterEqual(np.linalg.eigvalsh(sim.dense_density()).min(),-1e-10)
        self.assertGreaterEqual(result.gate_bound_sum,0.)


if __name__=='__main__':unittest.main()
