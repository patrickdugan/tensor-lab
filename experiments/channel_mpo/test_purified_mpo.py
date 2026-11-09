"""QC-02 correctness and safety tests, including independent dense CPTP oracle."""
import unittest
from unittest.mock import patch

import numpy as np

from channel_mpo import apply_channel, apply_operator, gate_layers, kraus
from purified_mpo import PurifiedChain, run_one, select_rank, stable_svd


class PurifiedMPOTests(unittest.TestCase):
    def test_initial_product_state(self):
        for n in (2, 3, 6):
            sim = PurifiedChain(n)
            self.assertAlmostEqual(sim.trace(), 1)
            self.assertEqual(sim.z_expectations(), [1.0] * n)
            expected = np.zeros((2**n, 2**n), complex)
            expected[0, 0] = 1
            np.testing.assert_allclose(sim.dense_density(), expected, atol=1e-14)
            self.assertEqual(sim.dimensions()["purified_entries"], 2 * n)

    def test_bell_entanglement_matches_density(self):
        sim = PurifiedChain(2)
        bell = np.eye(4, dtype=complex)
        bell[:, 0] = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
        bell[:, 3] = np.array([1, 0, 0, -1], complex) / np.sqrt(2)
        sim.gate(0, bell)
        rho = sim.dense_density()
        expected = np.outer(bell[:, 0], bell[:, 0].conj())
        np.testing.assert_allclose(rho, expected, atol=1e-14)
        self.assertEqual(sim.dimensions()["max_bond"], 2)
        self.assertEqual(sim.dimensions()["max_env"], 1)

    def test_kraus_complete_and_ancillary_compression_exact(self):
        for name in ("depolarizing", "dephasing", "amplitude_damping"):
            sim = PurifiedChain(3)
            layers = gate_layers(3, 1, 22)
            sim.gate(*layers[0][0])
            before = sim.dense_density()
            ks = kraus(name, .3)
            sim.channel(1, ks)
            sim.compress_env(1)
            ref = apply_channel(before, ks, 1, 3)
            np.testing.assert_allclose(sim.dense_density(), ref, atol=1e-12)
            self.assertAlmostEqual(sim.trace(), 1.0, places=12)

    def test_untruncated_matches_dense_at_every_layer(self):
        for name, p in (("depolarizing", .35), ("dephasing", .8),
                        ("amplitude_damping", .5)):
            rows = run_one(4, 3, 7, name, p, None, None, 0.0, validate=True)
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertLess(row["trace_distance"], 2e-12)
                self.assertLess(row["z0_abs_error"], 2e-12)
                self.assertGreaterEqual(row["min_eigenvalue"], -2e-12)
                self.assertAlmostEqual(row["trace_before_normalize"], 1.0, places=12)

    def test_truncated_remains_positive_and_normalized(self):
        for name in ("depolarizing", "dephasing", "amplitude_damping"):
            rows = run_one(5, 4, 10, name, .4, 3, 2, 1e-7, validate=True)
            for row in rows:
                self.assertLessEqual(row["max_bond"], 3)
                self.assertLessEqual(row["max_env"], 2)
                self.assertGreaterEqual(row["min_eigenvalue"], -1e-12)
                self.assertLessEqual(row["trace_distance"], 1+1e-12)
                self.assertLessEqual(row["z0_abs_error"], 2.0)
                self.assertLessEqual(row["trace_before_normalize"], 1+1e-12)
                self.assertGreater(row["trace_before_normalize"], 0)

    def test_cap_does_not_increase_size_on_heavy_noise(self):
        small = run_one(6, 3, 0, "depolarizing", .5, 8, 2, 0, validate=True)[-1]
        big = run_one(6, 3, 0, "depolarizing", .5, 8, 16, 0, validate=True)[-1]
        self.assertLessEqual(small["purified_entries"], big["purified_entries"])
        self.assertLessEqual(big["trace_distance"], small["trace_distance"] + 1e-12)

    def test_channel_p_zero_keeps_pure_environment(self):
        row = run_one(6, 3, 4, "depolarizing", 0, 16, 16, 0)[-1]
        self.assertEqual(row["max_env"], 1)

    def test_large_product_channel_matches_closed_form(self):
        # An exact 48-qubit check through local observables, with no dense rho.
        sim = PurifiedChain(48, max_bond=12, max_env=4, local_tol=0)
        p = .3
        for depth in range(1, 5):
            for site in range(sim.n):
                sim.channel(site, kraus("depolarizing", p))
            for site in range(sim.n):
                sim.compress_env(site)
            sim.normalize()
            self.assertLess(max(abs(z - (1-p)**depth)
                                for z in sim.z_expectations()), 1e-11)
            self.assertEqual(sim.dimensions()["max_bond"], 1)
            self.assertLessEqual(sim.dimensions()["max_env"], 2)

    def test_qr_svd_retry(self):
        matrix = np.arange(60).reshape(10, 6).astype(complex)
        matrix += 0.01j * matrix[::-1]
        original = np.linalg.svd
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise np.linalg.LinAlgError("injected divide-and-conquer failure")
            return original(*args, **kwargs)

        with patch.object(np.linalg, "svd", side_effect=fail_once):
            u, values, vh = stable_svd(matrix)
        np.testing.assert_allclose((u * values) @ vh, matrix, atol=1e-11)
        self.assertGreaterEqual(calls, 2)

    def test_low_retention_is_flagged(self):
        r = run_one(8, 6, 0, "depolarizing", .5, 12, 4, 0)
        self.assertTrue(r[0]["retention_gate_pass"])
        self.assertFalse(r[-1]["retention_gate_pass"])
        self.assertLess(r[-1]["min_layer_retention_so_far"], .2)

    def test_relative_rank_cap_and_bad_inputs(self):
        s = np.array([4, 2, 0.01, 0], dtype=float)
        self.assertEqual(select_rank(s, 2, 0), 2)
        self.assertEqual(select_rank(s, None, 0.1), 2)
        self.assertEqual(select_rank(s, None, 0), 3)
        with self.assertRaises(ValueError):
            select_rank(s, 0, 0)
        with self.assertRaises(ValueError):
            PurifiedChain(1)
        with self.assertRaises(ValueError):
            PurifiedChain(3, max_env=0)
        with self.assertRaises(ValueError):
            PurifiedChain(3, local_tol=-0.2)
        with self.assertRaises(ValueError):
            PurifiedChain(3).dense_density(max_n=2)
        with self.assertRaises(ValueError):
            PurifiedChain(3).gate(2, np.eye(4))
        with self.assertRaises(ValueError):
            PurifiedChain(3).channel(3, kraus("dephasing", .5))


if __name__ == "__main__":
    unittest.main()
