"""QC-06 rank allocation and purified-channel correctness regression suite."""
from __future__ import annotations
import unittest
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

from channel_mpo import apply_channel, apply_operator, gate_layers, kraus
from purified_mpo import PurifiedChain
from qc06_adaptive import AdaptivePurifiedChain, greedy_rank_allocation, scan_ancilla_spectra


class AdaptiveRankTests(unittest.TestCase):
    def test_knapsack_greedy_prefers_high_information_per_entry(self):
        spectra = [np.array([3., 2., .1]), np.array([3., 1., .9])]
        ranks = greedy_rank_allocation(spectra, [4, 2], 8, 3)
        self.assertEqual(ranks, (1, 2))

    def test_budget_infeasible_and_bad_inputs(self):
        s = [np.array([1., .5])]
        with self.assertRaises(ValueError):greedy_rank_allocation(s, [4], 3, 2)
        with self.assertRaises(ValueError):greedy_rank_allocation(s, [0], 4, 2)
        with self.assertRaises(ValueError):greedy_rank_allocation(s, [4], 4, 0)
        with self.assertRaises(ValueError):greedy_rank_allocation([], [], 4, 2)
        with self.assertRaises(ValueError):greedy_rank_allocation(s, [4], 4.5, 2)
        with self.assertRaises(ValueError):greedy_rank_allocation([np.array([np.nan])], [4], 4, 2)

    def test_no_noise_no_truncation_agrees_with_dense(self):
        sim = AdaptivePurifiedChain(4, max_bond=None)
        exact = np.zeros((16,16), complex);exact[0,0]=1.
        gates = gate_layers(4, 2, 4); ops = kraus('depolarizing', 0.)
        for layer in gates:
            for i,u in layer:sim.gate(i,u);exact=apply_operator(exact,u,(i,i+1),4)
            for i in range(4):sim.channel(i,ops);exact=apply_channel(exact,ops,i,4)
            report=sim.compress_layer_budgeted(20000,max_site_env=20)
            self.assertLessEqual(report.retained_entries,20000)
            sim.normalize()
            np.testing.assert_allclose(sim.dense_density(),exact,atol=1e-11)

    def test_spectra_scan_is_exact_gauge_change(self):
        sim=AdaptivePurifiedChain(4,max_bond=12)
        for i,u in gate_layers(4,1,3)[0]:sim.gate(i,u)
        for i in range(4):sim.channel(i,kraus('depolarizing', .25))
        before=sim.dense_density()
        s, w=scan_ancilla_spectra(sim)
        self.assertEqual(len(s),4)
        self.assertEqual(len(w),4)
        np.testing.assert_allclose(sim.dense_density(),before,atol=1e-12)

    def test_exact_channel_if_all_ranks_kept(self):
        sim=AdaptivePurifiedChain(4,max_bond=None)
        rho=np.zeros((16,16),complex);rho[0,0]=1.
        for i,u in gate_layers(4,2,2)[0]:sim.gate(i,u);rho=apply_operator(rho,u,(i,i+1),4)
        ops=kraus('depolarizing',.4)
        for i in range(4):sim.channel(i,ops);rho=apply_channel(rho,ops,i,4)
        report=sim.compress_layer_budgeted(100000,max_site_env=32)
        sim.normalize()
        np.testing.assert_allclose(sim.dense_density(),rho,atol=2e-11)
        self.assertLess(report.predicted_discarded_weight,1e-20)

    def test_matched_final_entries_and_positive_density(self):
        n=6
        baseline=PurifiedChain(n,12,6)
        adaptive=AdaptivePurifiedChain(n,12,10)
        for layer in gate_layers(n,3,17):
            for i,u in layer:baseline.gate(i,u);adaptive.gate(i,u)
            for i in range(n):
                ops=kraus('depolarizing',.5)
                baseline.channel(i,ops);adaptive.channel(i,ops)
            for i in range(n):baseline.compress_env(i)
            budget=baseline.dimensions()['purified_entries']
            baseline.normalize()
            report=adaptive.compress_layer_budgeted(budget,max_site_env=10)
            adaptive.normalize()
            self.assertLessEqual(report.retained_entries,budget)
            self.assertEqual(len(report.target_ranks),n)
            rho=adaptive.dense_density()
            self.assertAlmostEqual(np.trace(rho).real,1.,places=10)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(rho).min()),-2e-11)

    def test_allocation_does_not_use_dense_oracle(self):
        sim=AdaptivePurifiedChain(4,max_bond=8)
        for i,u in gate_layers(4,1,4)[0]:sim.gate(i,u)
        for i in range(4):sim.channel(i,kraus('amplitude_damping',.3))
        with patch.object(sim,'dense_density',side_effect=AssertionError('oracle')):
            a=sim.compress_layer_budgeted(2000,max_site_env=6)
        self.assertLessEqual(a.retained_entries,2000)


    def test_uniform_rank_cap_recovers_qc02_density(self):
        n=4
        baseline=PurifiedChain(n,12,6)
        optimized=AdaptivePurifiedChain(n,12,6)
        ops=kraus('depolarizing',.5)
        for layer in gate_layers(n,3,11):
            for i,u in layer:
                baseline.gate(i,u)
                optimized.gate(i,u)
            for i in range(n):
                baseline.channel(i,ops)
                optimized.channel(i,ops)
            for i in range(n):baseline.compress_env(i)
            report=optimized.compress_layer_budgeted(
                baseline.dimensions()['purified_entries'],max_site_env=6)
            baseline.normalize();optimized.normalize()
            self.assertEqual(report.retained_entries,
                             baseline.dimensions()['purified_entries'])
            np.testing.assert_allclose(optimized.dense_density(),
                                       baseline.dense_density(),atol=1e-11)

    def test_frozen_calibration_budget_replay(self):
        from benchmark_qc06 import run_case
        schedule=json.loads(Path(__file__).with_name('qc06_budget_schedule.json').read_text())
        budget=schedule['budgets_by_n']['6']
        self.assertEqual(budget,[96,416,1728,4800,7200,7200])
        row=run_case(6,6,4,.2,12,6,12,True,frozen_budgets=budget)
        self.assertTrue(row['frozen_schedule'])
        self.assertTrue(row['strict_matched_budget'])
        self.assertEqual(row['final_budget_entries'],row['final_adaptive_entries'])
        with self.assertRaises(ValueError):
            run_case(6,6,4,.2,12,6,12,False,frozen_budgets=[96]*6)

    def test_infeasible_budget_preserves_physical_state(self):
        sim=AdaptivePurifiedChain(3,max_bond=4)
        for i,u in gate_layers(3,1,2)[0]:sim.gate(i,u)
        for i in range(3):sim.channel(i,kraus('depolarizing',.35))
        before=sim.dense_density()
        with self.assertRaises(ValueError):sim.compress_layer_budgeted(1,max_site_env=4)
        np.testing.assert_allclose(sim.dense_density(),before,atol=1e-12)

    def test_deterministic_allocation(self):
        spectra=[np.array([1.,.5,.2]),np.array([1.,.6,.1])]
        self.assertEqual(greedy_rank_allocation(spectra,[3,3],18,3),
                         greedy_rank_allocation(spectra,[3,3],18,3))

if __name__ == '__main__': unittest.main()
