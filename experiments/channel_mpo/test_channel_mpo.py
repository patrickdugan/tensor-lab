"""Regression tests for quantum-channel rank and MPO approximation."""
import unittest

import numpy as np

from channel_mpo import (X, apply_channel, apply_operator, choi_rank, interleave,
                         kraus, metrics, mpo_ttsvd, stinespring_trace, uninterleave)


class ChannelTests(unittest.TestCase):
    def test_kraus_is_trace_preserving_and_choi_rank(self):
        for name, expected_rank in [("depolarizing", 4), ("dephasing", 2),
                                    ("amplitude_damping", 2)]:
            ks = kraus(name, 0.2)
            total = sum((k.conj().T @ k for k in ks), np.zeros((2, 2), complex))
            np.testing.assert_allclose(total, np.eye(2), atol=1e-14)
            self.assertEqual(choi_rank(ks), expected_rank)
        self.assertEqual(choi_rank(kraus("depolarizing", 0)), 1)

    def test_dilation_partial_trace_matches_channel(self):
        rho = np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex)
        for name in ("depolarizing", "dephasing", "amplitude_damping"):
            ks = kraus(name, 0.3)
            direct = sum((k @ rho @ k.conj().T for k in ks), np.zeros_like(rho))
            np.testing.assert_allclose(stinespring_trace(rho, ks), direct, atol=1e-14)

    def test_local_contractions_match_dense_kron(self):
        n = 3
        vector = np.arange(1, 9, dtype=complex)
        vector /= np.linalg.norm(vector)
        rho = np.outer(vector, vector.conj())
        np.testing.assert_allclose(apply_operator(rho, X, (1,), n),
                                   np.kron(np.kron(np.eye(2), X), np.eye(2)) @ rho @
                                   np.kron(np.kron(np.eye(2), X), np.eye(2)), atol=1e-14)
        np.testing.assert_allclose(apply_channel(rho, kraus("depolarizing", .3), 1, n),
                                   .7 * rho + .3 / 4 * sum(
                                       np.kron(np.kron(np.eye(2), pauli), np.eye(2)) @ rho @
                                       np.kron(np.kron(np.eye(2), pauli), np.eye(2))
                                       for pauli in (np.eye(2), X,
                                                     np.array([[0,-1j],[1j,0]]),
                                                     np.diag([1,-1]))), atol=1e-14)

    def test_interleave_inverts(self):
        rng = np.random.default_rng(2)
        arr = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        np.testing.assert_array_equal(uninterleave(interleave(arr, 3), 3), arr)

    def test_rank_one_global_unitary_can_create_mpo_bond_four(self):
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        pure_entangled = np.outer(bell, bell.conj())
        cores, recon = mpo_ttsvd(pure_entangled, 2, 0)
        self.assertEqual(cores[0].shape[-1], 4)
        np.testing.assert_allclose(recon, pure_entangled, atol=1e-13)
        product_mixed = np.eye(4, dtype=complex) / 4
        self.assertEqual(mpo_ttsvd(product_mixed, 2, 0)[0][0].shape[-1], 1)

    def test_ttsvd_relative_error_bound_and_diagnostics(self):
        rng = np.random.default_rng(3)
        v = rng.normal(size=32) + 1j*rng.normal(size=32)
        v /= np.linalg.norm(v)
        rho = np.outer(v, v.conj())
        for eps in (0.0, 0.02, 0.2):
            cores, rec = mpo_ttsvd(rho, 5, eps)
            self.assertLessEqual(np.linalg.norm(rho - rec)/np.linalg.norm(rho), eps+1e-12)
            self.assertEqual(cores[0].shape[0], 1)
            self.assertEqual(cores[-1].shape[-1], 1)
        result = metrics(rho, 5, 0.02)
        self.assertEqual(result["dense_entries"], 1024)
        self.assertAlmostEqual(result["trace"], 1.0)
        self.assertLess(result["trace_norm_distance"], .1)
        self.assertGreaterEqual(result["approx_trace_error"], 0)

    def test_noise_strength_limits(self):
        with self.assertRaises(ValueError):
            kraus("depolarizing", 1.1)
        with self.assertRaises(ValueError):
            mpo_ttsvd(np.eye(4), 2, -1)


if __name__ == "__main__":
    unittest.main()
