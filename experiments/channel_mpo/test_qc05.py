"""QC-05 physical-operator, global density and no-oracle regression tests."""
from __future__ import annotations
import unittest
from unittest.mock import patch
import numpy as np

from channel_mpo import kraus, gate_layers
from purified_mpo import PurifiedChain, run_one
from gauge_pair import run_pair
from qc05_global import hs_inner, hs_square_error
from qc05_observables import pauli_expectation, sketch_words, P
from qc05_portfolio import select_global, run_portfolio


class GlobalDensityTests(unittest.TestCase):
    def make_chain(self, n=4, seed=6, p=.2):
        ks = kraus('depolarizing', p)
        chain = PurifiedChain(n, 10, 4)
        for layer in gate_layers(n, 3, seed):
            for i, u in layer:
                chain.gate(i, u)
            for i in range(n):
                chain.channel(i, ks)
            for i in range(n):
                chain.compress_env(i)
            chain.normalize()
        return chain

    def test_inner_product_matches_dense_trace(self):
        a = self.make_chain(4, 1, .2)
        b = self.make_chain(4, 2, .5)
        expected = np.trace(a.dense_density() @ b.dense_density()).real
        self.assertAlmostEqual(hs_inner(a, b), expected, delta=1e-11)
        self.assertAlmostEqual(hs_inner(b, a), expected, delta=1e-11)
        self.assertGreaterEqual(hs_inner(a, a), -1e-10)

    def test_error_matches_dense_frobenius_and_zero_identical(self):
        a, b = self.make_chain(4, 8), self.make_chain(4, 9)
        delta = a.dense_density() - b.dense_density()
        expected = float(np.vdot(delta, delta).real)
        self.assertAlmostEqual(hs_square_error(a, b), expected, delta=1e-10)
        self.assertAlmostEqual(hs_square_error(a, a), 0., delta=1e-10)

    def test_pauli_transfer_matches_dense_and_complex_y(self):
        a = self.make_chain(4, 11, .3)
        rho = a.dense_density()
        for word in ('IIII', 'YXIZ', 'ZZZZ', 'XXXX', 'IZII'):
            matrix = np.array([[1.]], complex)
            for letter in word:
                matrix = np.kron(matrix, P[letter])
            exact = float(np.trace(rho @ matrix).real)
            self.assertAlmostEqual(pauli_expectation(a, word), exact, delta=1e-10)

    def test_sketch_budget_and_invalid_words(self):
        self.assertEqual(len(sketch_words(8, 8)), 8)
        self.assertTrue(all(len(w) == 8 for w in sketch_words(8, 100)))
        with self.assertRaises(ValueError):
            sketch_words(4, 0)
        for word in ('III', 'AAII'):
            with self.assertRaises(ValueError):
                pauli_expectation(self.make_chain(4), word)

    def test_inner_mismatched_site_count_rejected(self):
        with self.assertRaises(ValueError):
            hs_inner(self.make_chain(3), self.make_chain(4))


class SelectionTests(unittest.TestCase):
    def test_selection_never_materializes_dense_density(self):
        a = PurifiedChain(4, 4, 4)
        b = PurifiedChain(4, 4, 4)
        teacher = PurifiedChain(4, 6, 6)
        with patch.object(PurifiedChain, 'dense_density',
                          side_effect=RuntimeError('oracle leakage')):
            for metric in ('hs', 'sketch'):
                name, score_a, score_b, elapsed = select_global(
                    teacher, a, b, metric)
                self.assertEqual(name, 'sequential')
                self.assertAlmostEqual(score_a, score_b, delta=1e-12)
                self.assertGreaterEqual(elapsed, 0)

    def test_invalid_score_and_shape_rejected(self):
        a, b = PurifiedChain(4, 4, 4), PurifiedChain(3, 4, 4)
        with self.assertRaises(ValueError):
            select_global(a, a, a, 'invalid')
        with self.assertRaises(ValueError):
            select_global(a, a, b, 'hs')
        with self.assertRaises(ValueError):
            select_global(a, a, a, 'hs', width=0)
        with self.assertRaises(ValueError):
            run_portfolio(4, 0, 0, .2)
        with self.assertRaises(ValueError):
            run_portfolio(9, 1, 0, .2, validate=True)

    def test_pilot_matches_qc02_and_qc04(self):
        args = (4, 2, 0, 'depolarizing', .2, 8, 3, 0)
        control = run_one(*args, validate=True)
        pair = run_pair(*args, factor_swaps=True, fast_factor_search=True)
        run = run_portfolio(4, 2, 0, .2, 8, 3,
                            teacher_bond=10, teacher_env=6)
        self.assertAlmostEqual(run['distance_seq'],
                               control[-1]['trace_distance'], delta=1e-10)
        self.assertAlmostEqual(run['distance_pair'],
                               pair[-1]['trace_distance'], delta=1e-10)
        self.assertIn(run['selected'], ('sequential', 'pair'))
        key = 'distance_seq' if run['selected']=='sequential' else 'distance_pair'
        self.assertAlmostEqual(run['distance_selected'], run[key], delta=1e-10)
        self.assertGreater(run['seconds_oracle_excluded'], 0)

    def test_small_qc05_normalization_and_distance(self):
        run = run_portfolio(4, 3, 3, .4, 6, 3,
                            teacher_bond=8, teacher_env=4)
        for key in ('distance_seq', 'distance_pair', 'distance_teacher'):
            self.assertGreaterEqual(run[key], -1e-11)
            self.assertLessEqual(run[key], 1+1e-8)
        self.assertGreater(run['retention_pair_min'], 0)
        self.assertGreater(run['retention_teacher_min'], 0)

    def test_larger_n_path_has_no_dense_oracle(self):
        with patch.object(PurifiedChain, 'dense_density',
                          side_effect=RuntimeError('oracle leakage')):
            run = run_portfolio(9, 1, 4, .3, 6, 3,
                                teacher_bond=8, teacher_env=4,
                                validate=False, mid_baseline=False)
        self.assertTrue(np.isnan(run['distance_seq']))
        self.assertGreaterEqual(run['seconds_selection_total'], 0)


if __name__ == '__main__':
    unittest.main()
