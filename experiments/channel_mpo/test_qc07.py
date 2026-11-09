"""QC-07 joint allocation, oracle isolation, and physical-state checks."""
from __future__ import annotations
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from channel_mpo import kraus, gate_layers, apply_operator, apply_channel
from purified_mpo import PurifiedChain
from qc07_joint import (JointPurifiedChain, allocate_joint, projected_entries,
                        spatial_spectra, one_site_densities,pauli_transfer_norm)
from benchmark_qc07 import run_case


class JointRankTests(unittest.TestCase):
    def test_marginal_matches_dense(self):
        c=JointPurifiedChain(4)
        for i,u in gate_layers(4,2,3)[0]:c.gate(i,u)
        for i in range(4):c.channel(i,kraus('amplitude_damping',.4))
        rho=c.dense_density()
        marg=one_site_densities(c)
        for i,m in enumerate(marg):
            ref=np.zeros((2,2),complex)
            for row in range(16):
                for col in range(16):
                    if all(((row>>(3-j))&1)==((col>>(3-j))&1) for j in range(4) if j!=i):
                        ref[(row>>(3-i))&1,(col>>(3-i))&1]+=rho[row,col]
            ref=ref/np.trace(rho)
            np.testing.assert_allclose(m,ref,atol=2e-12)

    def test_pauli_transfer_channel_sensitivity(self):
        for p in (.2,.5,.9):
            self.assertAlmostEqual(pauli_transfer_norm(kraus('depolarizing',p)),1-p,places=12)
            self.assertAlmostEqual(pauli_transfer_norm(kraus('dephasing',p)),1.,places=12)
            self.assertAlmostEqual(pauli_transfer_norm(kraus('amplitude_damping',p)),np.sqrt(1-p),places=12)
        with self.assertRaises(ValueError):pauli_transfer_norm((np.eye(2)*2,))

    def test_bond_scan_is_exact_gauge_change(self):
        sim=JointPurifiedChain(4,max_bond=12)
        for i,u in gate_layers(4,1,2)[0]:sim.gate(i,u)
        for i in range(4):sim.channel(i,kraus('depolarizing',.3))
        before=sim.dense_density()
        vals=spatial_spectra(sim)
        self.assertEqual(len(vals),3)
        np.testing.assert_allclose(sim.dense_density(),before,atol=2e-12)

    def test_alloc_joint_cost_and_limits(self):
        s_b=[np.array([1.,.25])]
        s_e=[np.array([1.,.6]),np.array([1.,.6])]
        result=allocate_joint(s_b,s_e,14,2,2,.5)
        self.assertEqual(result.predicted_entries,projected_entries(result.bond_ranks,result.environment_ranks))
        self.assertLessEqual(result.predicted_entries,14)
        self.assertLessEqual(max(result.environment_ranks),2)
        self.assertLessEqual(max(result.bond_ranks),2)
        with self.assertRaises(ValueError):allocate_joint(s_b,s_e,3,2,2)
        with self.assertRaises(ValueError):allocate_joint(s_b,s_e,14,0,2)
        with self.assertRaises(ValueError):allocate_joint(s_b,s_e,14,2,2,1.5)
        with self.assertRaises(ValueError):projected_entries([2],[3])

    def test_no_truncation_exact_density(self):
        n=4
        c=JointPurifiedChain(n,max_bond=32,max_env=32)
        rho=np.zeros((2**n,2**n),complex);rho[0,0]=1.
        ops=kraus('depolarizing',.3)
        for layer in gate_layers(n,2,2):
            for i,u in layer:
                c.gate(i,u)
                rho=apply_operator(rho,u,(i,i+1),n)
            for i in range(n):
                c.channel(i,ops)
                rho=apply_channel(rho,ops,i,n)
            info=c.compress_layer_joint(200000,ops,bond_cap=32,env_cap=32)
            self.assertLess(info.discarded_squared_weight,1e-18)
            c.normalize()
        np.testing.assert_allclose(c.dense_density(),rho,atol=3e-11)

    def test_state_is_positive_and_normalized_after_truncation(self):
        n=4
        c=JointPurifiedChain(n,max_bond=4,max_env=7)
        ops=kraus('amplitude_damping',.5)
        for layer in gate_layers(n,3,5):
            for i,u in layer:c.gate(i,u)
            for i in range(n):c.channel(i,ops)
            r=c.compress_layer_joint(120,ops,bond_cap=4,env_cap=7)
            self.assertLessEqual(r.retained_entries,120)
            self.assertLessEqual(r.plan.predicted_entries,120)
            self.assertGreaterEqual(r.min_local_retention,0)
            c.normalize()
        rho=c.dense_density()
        self.assertAlmostEqual(np.trace(rho).real,1.,places=10)
        self.assertGreaterEqual(np.linalg.eigvalsh(rho).min(),-1e-10)

    def test_trace_bound_covers_exact_small_case(self):
        n=3
        ops=kraus('depolarizing',.4)
        c=JointPurifiedChain(n,max_bond=3,max_env=5)
        rho=np.zeros((8,8),complex);rho[0,0]=1
        for layer in gate_layers(n,2,12):
            for i,u in layer:
                c.gate(i,u);rho=apply_operator(rho,u,(i,i+1),n)
            for i in range(n):
                c.channel(i,ops);rho=apply_channel(rho,ops,i,n)
            c.compress_layer_joint(56,ops,bond_cap=3,env_cap=5)
            c.normalize()
        diff=(rho-c.dense_density())
        dist=.5*sum(abs(np.linalg.eigvalsh((diff+diff.conj().T)/2)))
        self.assertLessEqual(dist,min(1.,c.error_bound_sum)+1e-9)

    def test_oracle_never_queried_by_optimizer(self):
        c=JointPurifiedChain(4,max_bond=8,max_env=8)
        ops=kraus('depolarizing',.25)
        for i,u in gate_layers(4,1,5)[0]:c.gate(i,u)
        for i in range(4):c.channel(i,ops)
        with patch.object(c,'dense_density',side_effect=AssertionError('dense oracle in policy')):
            c.compress_layer_joint(300,ops)

    def test_infeasible_budget_preserves_density(self):
        c=JointPurifiedChain(3,max_bond=4,max_env=8)
        ops=kraus('dephasing',.3)
        for i,u in gate_layers(3,1,9)[0]:c.gate(i,u)
        for i in range(3):c.channel(i,ops)
        before=c.dense_density()
        with self.assertRaises(ValueError):c.compress_layer_joint(3,ops)
        np.testing.assert_allclose(c.dense_density(),before,atol=1e-12)

    def test_workspace_preflight_and_rejection_preserves_state(self):
        c=JointPurifiedChain(3,max_bond=4,max_env=8,work_cap_entries=20)
        before=c.dense_density()
        with self.assertRaises(MemoryError):c.gate(0,np.eye(4))
        np.testing.assert_allclose(c.dense_density(),before)

    def test_frozen_budget_replay_on_holdout(self):
        data=json.loads(Path(__file__).with_name('qc06_budget_schedule.json').read_text())
        r=run_case(6,3,.2,budget_by_layer=data['budgets_by_n']['6'])
        self.assertTrue(r['matched_boundary_budget'])
        self.assertLess(r['distance_qc07'],r['distance_qc06'])
        self.assertLessEqual(r['qc07_modeled_working_peak_entries'],500000)
        self.assertGreaterEqual(r['qc07_trace_bound_sum'],0.)

if __name__=='__main__': unittest.main()
