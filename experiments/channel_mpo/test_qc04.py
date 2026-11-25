"""QC-04 acceptance tests: admissible pruning and SVD/Gram agreement."""
from __future__ import annotations
import unittest
import numpy as np

from purified_mpo import PurifiedChain
from gauge_pair import (factor_permutation, optimize_factor_permutations,
                        projected_pair_norm, run_pair)
from qc04_gauge import screened_factor_search, projected_pair_norm_gram


def pair_rho(theta):
    l, _, ea, _, eb, r = theta.shape
    m = theta.transpose(0,1,3,5,2,4).reshape(l*4*r,ea*eb)
    return m @ m.conj().T


class ScorerTests(unittest.TestCase):
    def test_gram_scores_match_full_svd_on_sized_and_rank_deficient_tensors(self):
        rng = np.random.default_rng(731)
        shapes = ((2,2,4,2,4,2), (4,2,8,2,8,3), (4,2,16,2,16,4))
        for shape in shapes:
            t = rng.normal(size=shape) + 1j*rng.normal(size=shape)
            for candidate in (t, t*(1+1j), np.ones(shape,dtype=complex)):
                for env in (None, 1, 2, 4):
                    for bond in (None, 2, 12):
                        with self.subTest(shape=shape, env=env, bond=bond):
                            expected = projected_pair_norm(candidate,env,bond)
                            actual = projected_pair_norm_gram(candidate,env,bond)
                            self.assertAlmostEqual(expected,actual,delta=1e-8*max(1,expected))

    def test_bounds_never_overestimate_exact_retention_loss(self):
        from gauge_pair import env_tail
        rng=np.random.default_rng(12)
        for dim in (4,8,16):
            theta=rng.normal(size=(2,2,dim,2,dim,2))+1j*rng.normal(size=(2,2,dim,2,dim,2))
            norm=float(np.vdot(theta,theta).real)
            for cap in (1,2,4):
                lower=max(env_tail(theta,2,cap),env_tail(theta,4,cap))
                for bond in (1,3,12):
                    true_loss=norm-projected_pair_norm(theta,cap,bond)
                    self.assertLessEqual(lower,true_loss+1e-9*norm)

    def test_screen_selects_same_gauge_as_reference(self):
        rng=np.random.default_rng(2026)
        for shape,k,cap,bond in [((2,2,8,2,8,2),2,2,4),
                                  ((2,2,16,2,16,2),4,4,12),
                                  ((1,2,16,2,16,1),4,2,6)]:
            for ix in range(3):
                theta=rng.normal(size=shape)+1j*rng.normal(size=shape)
                theta/=np.linalg.norm(theta)
                expected, legacy=optimize_factor_permutations(theta,cap,k,bond)
                selected,stats=screened_factor_search(theta,cap,k,bond)
                self.assertLessEqual(stats.new_loss,stats.old_loss+1e-11)
                self.assertLessEqual(stats.pruned+stats.evaluated,stats.candidates+1)
                np.testing.assert_allclose(pair_rho(theta),pair_rho(selected),atol=1e-12)
                self.assertAlmostEqual(legacy.new_loss,stats.new_loss,delta=1e-9)
                np.testing.assert_allclose(expected,selected,atol=2e-11,rtol=1e-10)

    def test_guardrails(self):
        rng=np.random.default_rng(100)
        t=rng.normal(size=(1,2,4,2,4,1))+1j*rng.normal(size=(1,2,4,2,4,1))
        with self.assertRaises(ValueError):
            screened_factor_search(t,2,2,4,score_method='broken')
        same,stats=screened_factor_search(t,2,1,4)
        np.testing.assert_equal(same,t)
        self.assertEqual(stats.candidates,0)
        with self.assertRaises(ValueError):
            run_pair(4,1,0,'depolarizing',.5,4,2,0,
                     factor_objective='env', fast_factor_search=True)

    def test_end_to_end_fidelity_unchanged_under_screened_fast_path(self):
        for seed in (0,1):
            for p in (.2,.5):
                legacy=run_pair(4,3,seed,'depolarizing',p,8,3,0,factor_swaps=True)
                fast=run_pair(4,3,seed,'depolarizing',p,8,3,0,
                              factor_swaps=True,fast_factor_search=True)
                self.assertAlmostEqual(fast[-1]['trace_distance'],legacy[-1]['trace_distance'],delta=2e-9)
                self.assertAlmostEqual(fast[-1]['layer_retention'],legacy[-1]['layer_retention'],delta=2e-9)
                self.assertGreaterEqual(fast[-1]['min_eigenvalue'],-1e-10)
                self.assertLessEqual(fast[-1]['trace_distance'],
                                     fast[-1]['certified_trace_distance_bound']+1e-9)
                self.assertGreater(fast[-1]['gauge_exact_scores'],0)

if __name__=='__main__':
    unittest.main()
