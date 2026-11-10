"""QC-23 pipeline tests. Offline: Qiskit + stim + numpy + scipy (no Aer, no network)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import qiskit  # noqa: F401
    import stim
    from qiskit_ibm_runtime.fake_provider import FakeFez, FakeSherbrooke
    HAVE = True
except ImportError:  # pragma: no cover
    HAVE = False

if HAVE:
    from qc23_access import check_authorization, estimate_qpu_seconds, plan_hash
    from qc23_compile import (audit_mapping, choose_line_calibrated, measurement_map,
                              transpile_on_path)
    from qc23_ibm import make_echo, observable_bits, wilson_interval, z_estimators
    from qc23_mpo import (layered_clifford_reference, mpo_z, ops_to_layers, max_unitarity_error,
                          effective_depolarizing_p, dense_cone_z)
    from qc23_noise import ModelB, calibration_snapshot, model_a_faults, predict_model_a, predict_model_b, restrict_snapshot
    from qc23_nonclifford import make_kicked_ising, ideal_z
    from qc23_stabilizer import (extract_native_ops, pauli_flip_expectations, Fault, depolarize1, NativeOp,
                                 ops_tableau, verify_ideal_identity, verify_isa_identity)

SNAP = Path(__file__).parent / 'fixtures' / 'fez_calibration_snapshot.json'


def compile_echo(backend, n, d, s):
    path, _ = choose_line_calibrated(backend, n)
    qc = make_echo(n, d, s)
    qc.measure_all()
    return path, transpile_on_path(backend, qc, path)


@unittest.skipUnless(HAVE, 'qiskit/stim/qiskit-ibm-runtime required')
class IdealAndCompilation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fez = FakeFez()
        cls.snap = json.loads(SNAP.read_text())

    def test_observables_are_left_right_central(self):
        self.assertEqual(observable_bits(8), (0, 7, 4))
        self.assertEqual(observable_bits(16), (0, 15, 8))

    def test_independent_stabilizer_proof_for_all_twelve(self):
        for n in (8, 16):
            for d in (2, 4, 8):
                for s in (3, 5):
                    v = verify_ideal_identity(n, d, s)
                    self.assertTrue(v['tableau_is_identity'])
                    self.assertTrue(v['all_zero_deterministic'])
                    self.assertTrue(v['forward_half_is_nontrivial'])

    def test_qiskit_statevector_agrees_with_stim(self):
        from qiskit.quantum_info import Statevector
        probs = Statevector.from_instruction(make_echo(8, 4, 3)).probabilities()
        self.assertAlmostEqual(probs[0], 1.0, places=12)

    def test_transpiled_circuit_is_identity_and_mapping_audited(self):
        for backend in (self.fez, FakeSherbrooke()):
            path, isa = compile_echo(backend, 8, 4, 5)
            audit = audit_mapping(isa, path, 8)
            self.assertEqual(audit['swaps'], 0)
            self.assertEqual(audit['two_qubit_gates'], 2 * sum(
                1 for layer in __import__('qc23_ibm').echo_schedule(8, 4, 5) for op in layer if op[0] == 'cx'))
            self.assertTrue(verify_isa_identity(isa, path, backend.target)['isa_tableau_is_identity'])

    def test_logical_to_physical_measurement_probe(self):
        """X on logical qubit i only -> exactly bit i set, read from physical path[i]."""
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        path, _ = choose_line_calibrated(self.fez, 8)
        for flip in (0, 3, 7):
            qc = QuantumCircuit(8)
            qc.x(flip)
            qc.measure_all()
            isa = transpile_on_path(self.fez, qc, path)
            mm = measurement_map(isa)
            self.assertEqual(mm[flip], path[flip])
            ops = extract_native_ops(isa, path)
            self.assertEqual(sorted(o.logical[0] for o in ops if o.name == 'x'), [flip])

    def test_mapping_audit_rejects_wrong_layout(self):
        path, isa = compile_echo(self.fez, 8, 2, 3)
        with self.assertRaises(AssertionError):
            audit_mapping(isa, list(reversed(path)), 8)

    def test_non_clifford_angle_rejected(self):
        with self.assertRaises(ValueError):
            ops_tableau([NativeOp('rz', (0,), (0,), 0.3, params=(0.3,))], 2)

    def test_ecr_translation_matches_unitary(self):
        from qiskit.circuit.library import ECRGate
        from qiskit.quantum_info import Operator
        t = ops_tableau([NativeOp('ecr', (0, 1), (0, 1))], 2)
        self.assertEqual(t, stim.Tableau.from_unitary_matrix(Operator(ECRGate()).data, endian='little'))


@unittest.skipUnless(HAVE, 'qiskit/stim/qiskit-ibm-runtime required')
class NoiseAndMPO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fez = FakeFez()
        cls.snap = json.loads(SNAP.read_text())

    def test_pauli_propagation_matches_monte_carlo(self):
        path, isa = compile_echo(self.fez, 8, 2, 3)
        ops = extract_native_ops(isa, path)
        params = ModelB(0.01, 0.03, 0.02)
        exact = predict_model_b(ops, params, (0, 4, 7))
        from qc23_noise import model_b_faults
        faults, ro = model_b_faults(ops, params)
        # Monte Carlo with stim noise instructions built from the same fault list.
        circ = stim.Circuit()
        by_after = {}
        for f in faults:
            by_after.setdefault(f.after_op, []).append(f)
        for k, o in enumerate(ops):
            from qc23_stabilizer import stim_instructions
            for name, tg in stim_instructions(o):
                circ.append(name, tg)
            for f in by_after.get(k, []):
                if len(f.qubits) == 1:
                    circ.append('DEPOLARIZE1', list(f.qubits), sum(f.paulis.values()))
                else:
                    circ.append('DEPOLARIZE2', list(f.qubits), sum(f.paulis.values()))
        circ.append('X_ERROR', list(range(8)), params.r)
        circ.append('M', list(range(8)))
        s = circ.compile_sampler(seed=1).sample(400_000)
        for b in (0, 4, 7):
            mc = 1 - 2 * s[:, b].mean()
            self.assertAlmostEqual(mc, exact[b], delta=0.006)

    def test_layers_reproduce_circuit_unitary_non_clifford(self):
        from qiskit.quantum_info import Statevector
        n = 5
        qc = make_kicked_ising(n, 3, 7)
        path, _ = choose_line_calibrated(self.fez, n)
        m = qc.copy()
        m.measure_all()
        isa = transpile_on_path(self.fez, m, path)
        layers, _ = ops_to_layers(extract_native_ops(isa, path), n, drop_trailing_diagonal=False)
        self.assertLess(max_unitarity_error(layers), 1e-12)
        psi = np.zeros(2 ** n, complex)
        psi[0] = 1
        psi = psi.reshape([2] * n)
        for layer in layers:
            for s, u in layer:
                psi = np.moveaxis(np.tensordot(u.reshape(2, 2, 2, 2), psi, axes=([2, 3], [s, s + 1])), [0, 1], [s, s + 1])
        sv = Statevector.from_instruction(qc).data.reshape([2] * n)
        sv = np.transpose(sv, list(range(n))[::-1]).reshape(-1)
        self.assertAlmostEqual(abs(np.vdot(sv, psi.reshape(-1))), 1.0, places=10)

    def test_ecr_layers_reproduce_circuit(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        backend = FakeSherbrooke()
        n = 4
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.rx(0.4 + i, i)
        qc.cx(1, 0)
        qc.cx(1, 2)
        qc.cx(3, 2)
        m = qc.copy()
        m.measure_all()
        path, _ = choose_line_calibrated(backend, n)
        isa = transpile_on_path(backend, m, path)
        layers, _ = ops_to_layers(extract_native_ops(isa, path), n, drop_trailing_diagonal=False)
        psi = np.zeros(2 ** n, complex)
        psi[0] = 1
        psi = psi.reshape([2] * n)
        for layer in layers:
            for s, u in layer:
                psi = np.moveaxis(np.tensordot(u.reshape(2, 2, 2, 2), psi, axes=([2, 3], [s, s + 1])), [0, 1], [s, s + 1])
        sv = np.transpose(Statevector.from_instruction(qc).data.reshape([2] * n), list(range(n))[::-1]).reshape(-1)
        self.assertAlmostEqual(abs(np.vdot(sv, psi.reshape(-1))), 1.0, places=10)

    def test_mpo_matches_exact_pauli_reference_and_dense(self):
        n = 8
        path, isa = compile_echo(self.fez, n, 4, 3)
        ops = extract_native_ops(isa, path)
        faults, ro = model_a_faults(ops, restrict_snapshot(self.snap, path))
        layers, info = ops_to_layers(ops, n)
        p = effective_depolarizing_p(faults, n, info['mpo_layers'])
        self.assertGreater(p, 0)
        for t in (0, 4, 7):
            r = mpo_z(n, layers, p, t, bond_cap=16)
            ref = layered_clifford_reference(n, layers, p, [t])[t]
            self.assertAlmostEqual(r.expectation, ref, places=8)
            self.assertAlmostEqual(r.expectation, dense_cone_z(n, layers, p, t, max_sites=8), places=8)
            self.assertGreaterEqual(r.error_bound, 0)

    def test_noiseless_models_reproduce_ideal(self):
        n = 8
        path, isa = compile_echo(self.fez, n, 2, 3)
        ops = extract_native_ops(isa, path)
        z = predict_model_b(ops, ModelB(1e-12, 1e-12, 0.0), (0, 4, 7))
        for v in z.values():
            self.assertAlmostEqual(v, 1.0, places=9)

    def test_model_a_degrades_monotonically_with_depth(self):
        vals = []
        for d in (1, 4, 8):
            path, isa = compile_echo(self.fez, 8, d, 3)
            ops = extract_native_ops(isa, path)
            vals.append(predict_model_a(ops, restrict_snapshot(self.snap, path), (4,))[4])
        self.assertGreater(vals[0], vals[1])
        self.assertGreater(vals[1], vals[2])

    def test_pauli_flip_single_fault(self):
        ops = [NativeOp('cz', (0, 1), (0, 1))]
        f = Fault(-1, (0,), depolarize1(0.3))
        z = pauli_flip_expectations(ops, 2, [f], [0.0, 0.0])
        self.assertAlmostEqual(z[0], 1 - 2 * 0.2)   # X and Y flip, Z does not
        self.assertAlmostEqual(z[1], 1.0)


@unittest.skipUnless(HAVE, 'qiskit/stim/qiskit-ibm-runtime required')
class Authorization(unittest.TestCase):
    def test_allow_paid_flag_is_not_authorization(self):
        ok, why = check_authorization(None, backend='ibm_x', sha='abc', estimated_seconds=10)
        self.assertFalse(ok)

    def test_authorization_file_checks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'auth.json'
            base = {'authorized': True, 'backend': 'ibm_x', 'plan_sha256': 'abc',
                    'max_qpu_seconds': 60, 'authorized_by': 'owner'}
            p.write_text(json.dumps(base))
            self.assertTrue(check_authorization(str(p), backend='ibm_x', sha='abc', estimated_seconds=30)[0])
            self.assertFalse(check_authorization(str(p), backend='ibm_y', sha='abc', estimated_seconds=30)[0])
            self.assertFalse(check_authorization(str(p), backend='ibm_x', sha='zzz', estimated_seconds=30)[0])
            self.assertFalse(check_authorization(str(p), backend='ibm_x', sha='abc', estimated_seconds=61)[0])
            p.write_text(json.dumps({**base, 'authorized': False}))
            self.assertFalse(check_authorization(str(p), backend='ibm_x', sha='abc', estimated_seconds=30)[0])

    def test_estimate_scales_with_shots(self):
        self.assertAlmostEqual(estimate_qpu_seconds([1e-3, 1e-3], 1000), 2 * 1000 * (250e-6 + 1e-3))


class Statistics(unittest.TestCase):
    def test_wilson_is_nondegenerate_at_boundary(self):
        lo, hi = wilson_interval(0, 2048)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.001)
        lo, hi = wilson_interval(12, 12)
        self.assertLess(lo, 1.0)

    def test_z_estimator_reports_intervals(self):
        z = z_estimators({'0000': 2048}, 4, (0, 3))
        self.assertEqual(z['0']['expectation_z'], 1.0)
        self.assertEqual(z['0']['standard_error'], 0.0)
        self.assertLess(z['0']['wilson95'][0], 1.0)

    def test_bit_order_little_endian(self):
        z = z_estimators({'0001': 100}, 4, (0, 3))
        self.assertEqual(z['0']['expectation_z'], -1.0)
        self.assertEqual(z['3']['expectation_z'], 1.0)


if __name__ == '__main__':
    unittest.main()
