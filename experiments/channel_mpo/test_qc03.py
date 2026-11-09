"""QC-03 regression tests: Kraus gauges, projected retention, trajectory oracle."""
import unittest

import numpy as np

from channel_mpo import apply_channel, gate_layers, kraus
from purified_mpo import PurifiedChain
from gauge_pair import (
    env_objective, factor_permutation, mix_two_ancilla_states,
    optimize_ancillas, optimize_factor_permutations, PairPurifiedChain,
    projected_pair_norm, certified_truncation_bound, run_pair,
)
from trajectory_oracle import (
    TrajectoryChain, depolarizing_reset_kraus, unravelling, sample_ensemble,
)


def pair_rho(theta):
    # Keep all four outside bond/physical indices; trace the two ancillas.
    l, _, ea, _, eb, r = theta.shape
    mat = theta.transpose(0,1,3,5,2,4).reshape(l*2*2*r,ea*eb)
    return mat @ mat.conj().T


class GaugeTests(unittest.TestCase):
    def test_two_level_rotation_preserves_physical_state(self):
        rng=np.random.default_rng(9)
        theta=rng.normal(size=(2,2,4,2,4,2))+1j*rng.normal(size=(2,2,4,2,4,2))
        changed=mix_two_ancilla_states(theta,1,14,.41,1j)
        np.testing.assert_allclose(pair_rho(theta),pair_rho(changed),rtol=1e-12,atol=1e-11)
        self.assertAlmostEqual(np.linalg.norm(changed),np.linalg.norm(theta))

    def test_ancillary_factor_permutation_preserves_physical_state(self):
        rng=np.random.default_rng(99)
        theta=rng.normal(size=(2,2,16,2,16,3))+1j*rng.normal(size=(2,2,16,2,16,3))
        for factors in ((0,1),(0,3),(1,2),(2,3)):
            changed=factor_permutation(theta,4,factors)
            np.testing.assert_allclose(pair_rho(theta),pair_rho(changed),rtol=1e-12,atol=2e-11)
        with self.assertRaises(ValueError):factor_permutation(theta,4,(1,1))
        with self.assertRaises(ValueError):factor_permutation(theta,4,(1,7))

    def test_retained_norm_objective_cannot_worsen_on_local_candidates(self):
        rng=np.random.default_rng(7)
        theta=rng.normal(size=(2,2,8,2,8,2))+1j*rng.normal(size=(2,2,8,2,8,2))
        norm0=np.sum(np.abs(theta)**2)
        optimized, stats=optimize_factor_permutations(theta,2,2,3,0,'retention')
        self.assertLessEqual(stats.new_loss,stats.old_loss+1e-10)
        self.assertGreaterEqual(stats.candidate_evaluations,1)
        self.assertGreaterEqual(projected_pair_norm(optimized,2,3),
                                projected_pair_norm(theta,2,3)-1e-9)
        self.assertLessEqual(stats.new_loss,norm0+1e-9)

    def test_greedy_givens_never_increases_surrogate(self):
        rng=np.random.default_rng(6)
        theta=rng.normal(size=(2,2,4,2,4,2))+1j*rng.normal(size=(2,2,4,2,4,2))
        improved,stats=optimize_ancillas(theta,2,rng,4,2)
        self.assertLessEqual(env_objective(improved,2),env_objective(theta,2)+1e-10)
        self.assertGreaterEqual(stats.candidate_evaluations,1)

    def test_pair_compression_exact_when_untruncated(self):
        chain=PairPurifiedChain(4,max_bond=None,max_env=None)
        gate=gate_layers(4,1,5)[0][0]
        chain.gate(*gate)
        for i in range(4):chain.channel(i,kraus('depolarizing',.4))
        before=chain.dense_density()
        chain.compress_layer()
        np.testing.assert_allclose(chain.dense_density(),before,atol=2e-12)
        self.assertAlmostEqual(chain.trace(),1,places=12)

    def test_small_fidelity_and_certificate(self):
        rows=run_pair(4,3,0,'depolarizing',.35,8,3,0,factor_swaps=True)
        for r in rows:
            self.assertGreaterEqual(r['min_eigenvalue'],-1e-11)
            self.assertLessEqual(r['trace_distance'],r['certified_trace_distance_bound']+1e-10)
            self.assertGreater(r['layer_retention'],0)
            self.assertLessEqual(r['max_env'],3)
            self.assertLessEqual(r['max_bond'],8)

    def test_bell_projection_certificate_is_tight(self):
        sim=PurifiedChain(2,max_bond=1)
        bell=np.eye(4,dtype=complex)
        bell[:,0]=np.array([1,0,0,1],complex)/np.sqrt(2)
        bell[:,3]=np.array([1,0,0,-1],complex)/np.sqrt(2)
        sim.gate(0,bell)
        bound,norm=certified_truncation_bound(sim.truncations)
        self.assertAlmostEqual(bound,np.sqrt(.5),places=12)
        self.assertAlmostEqual(norm,.5,places=12)


class TrajectoryTests(unittest.TestCase):
    def test_reset_operators_equal_pauli_channel(self):
        for p in (0,.2,.5,1):
            a=kraus('depolarizing',p)
            b=depolarizing_reset_kraus(p)
            self.assertEqual(len(b),5)
            np.testing.assert_allclose(sum(k.conj().T@k for k in b),np.eye(2),atol=1e-14)
            for rho in (np.array([[1,0],[0,0]],complex),
                        np.array([[.5,.23+.19j],[.23-.19j,.5]],complex)):
                left=sum(k@rho@k.conj().T for k in a)
                right=sum(k@rho@k.conj().T for k in b)
                np.testing.assert_allclose(left,right,atol=1e-14)
        for mix in (0.,.25,.5,1.):
            ks = unravelling('depolarizing',.5,'hybrid',mix)
            np.testing.assert_allclose(sum(k.conj().T@k for k in ks), np.eye(2), atol=1e-14)
            x=np.array([[.5,.25j],[-.25j,.5]])
            expected=sum(k@x@k.conj().T for k in kraus('depolarizing',.5))
            np.testing.assert_allclose(sum(k@x@k.conj().T for k in ks), expected,atol=1e-14)
        with self.assertRaises(ValueError):unravelling('depolarizing',.5,'hybrid',-1)
        with self.assertRaises(ValueError):depolarizing_reset_kraus(-.1)
        with self.assertRaises(ValueError):unravelling('dephasing',.5,'reset')

    def test_explicit_kraus_ensemble_matches_density(self):
        rng=np.random.default_rng(2)
        state=rng.normal(size=4)+1j*rng.normal(size=4)
        state/=np.linalg.norm(state)
        rho=np.outer(state,state.conj())
        for scheme in ('pauli','reset'):
            ks=unravelling('depolarizing',.5,scheme)
            weighted=sum((np.kron(k,np.eye(2))@rho@np.kron(k.conj().T,np.eye(2))
                          for k in ks),np.zeros_like(rho))
            np.testing.assert_allclose(weighted,apply_channel(rho,kraus('depolarizing',.5),0,2),atol=1e-14)

    def test_noiseless_trajectory_matches_dense(self):
        for scheme in ('pauli','reset'):
            res=sample_ensemble(4,3,0,'depolarizing',0,scheme,8,16,exact_observables=True)
            self.assertLess(res['z0_abs_error'],1e-11)
            self.assertLess(res['zmid_abs_error'],1e-11)
            self.assertAlmostEqual(res['min_gate_retention'],1,places=12)

    def test_sampling_and_truncation_sanity(self):
        for scheme in ('pauli','reset'):
            res=sample_ensemble(5,3,1,'depolarizing',.4,scheme,96,16,exact_observables=True)
            # Fixed-seed statistical check: 5 sigma leaves ample margin.
            self.assertLess(abs(res['z0_zscore']),5)
            self.assertLess(abs(res['zmid_zscore']),5)
            self.assertLessEqual(res['max_observed_bond'],16)
            self.assertGreater(res['min_gate_retention'],.9999999)

    def test_smoke_20q_bounded_trajectory(self):
        result=sample_ensemble(20,3,0,'depolarizing',.5,'reset',3,8)
        self.assertLessEqual(result['max_observed_bond'],8)
        self.assertTrue(np.isfinite(result['z0_mean']))
        self.assertTrue(np.isfinite(result['z0_se']))


if __name__=='__main__':unittest.main()
