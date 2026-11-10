"""SYNTHETIC QPU SURROGATE - NOT IBM HARDWARE.

When no IBM Quantum account is reachable the pipeline still has to be exercised
end to end. This module builds a deliberately *out-of-model* noisy simulator
from a calibration snapshot: errors are drifted away from the snapshot values,
T1/T2 are shortened, readout is asymmetric, and coherent over-rotations are
added. The analysis stages that consume ``surrogate`` results are labelled as
such everywhere and must never be reported as hardware measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from qc23_noise import _edge, _f


@dataclass(frozen=True)
class Drift:
    gate_scale: float = 1.3          # multiplicative gate-error drift
    t_scale: float = 0.8             # T1/T2 multiplier (<1: faster decoherence)
    readout_scale: float = 1.4
    readout_asymmetry: float = 0.6   # P(1|0) = asym*e, P(0|1) = (2-asym)*e
    zz_overrotation: float = 0.04    # coherent RZZ(delta) after each 2q gate (rad)
    sx_overrotation: float = 0.015   # coherent RZ(eps) after each sx/x (rad)

    def as_dict(self):
        return asdict(self)


def _relaxation(snap, phys, duration, drift):
    from qiskit_aer.noise import thermal_relaxation_error
    rec = snap['qubits'][str(phys)]
    t1 = _f(rec['t1'], 100e-6) * drift.t_scale
    t2 = min(_f(rec['t2'], 100e-6) * drift.t_scale, 2 * t1)
    if duration <= 0:
        return None
    return thermal_relaxation_error(t1, t2, duration)


def build_surrogate_circuit(ops, n: int, snap: dict, drift: Drift = Drift(), *, measure=True):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import get_standard_gate_name_mapping
    from qiskit_aer.noise import depolarizing_error, coherent_unitary_error
    gates = get_standard_gate_name_mapping()
    qc = QuantumCircuit(n, n)
    clock = [0.0] * n
    phys_of = {}
    for o in ops:
        for l, p in zip(o.logical, o.physical):
            phys_of[l] = p

    def add_error(err, qubits):
        if err is not None:
            qc.append(err.to_instruction(), list(qubits))

    def idle(q, gap):
        add_error(_relaxation(snap, phys_of[q], gap, drift), (q,))

    zz = np.diag(np.exp(-0.5j * drift.zz_overrotation * np.array([1, -1, -1, 1])))
    rzs = np.diag(np.exp(-0.5j * drift.sx_overrotation * np.array([1, -1])))
    for o in ops:
        if o.name == 'measure':
            continue
        if o.name == 'barrier':
            tmax = max(clock[q] for q in o.logical)
            for q in o.logical:
                if tmax > clock[q]:
                    idle(q, tmax - clock[q])
                clock[q] = tmax
            qc.barrier(*o.logical)
            continue
        start = max(clock[q] for q in o.logical)
        if o.name == 'rz':
            dur = 0.0
        elif len(o.logical) == 1:
            dur = _f(snap['qubits'][str(o.physical[0])].get(f'{o.name}_duration'), 0.0)
        else:
            dur = _f(_edge(snap, *o.physical)['duration'], 0.0)
        for q in o.logical:
            if start > clock[q]:
                idle(q, start - clock[q])
            clock[q] = start + dur
        g = gates[o.name]
        qc.append(g.base_class(*o.params) if o.params else g, list(o.logical))
        if o.name in ('sx', 'x'):
            e = _f(snap['qubits'][str(o.physical[0])].get(f'{o.name}_error'), 0.0) * drift.gate_scale
            q = o.logical[0]
            add_error(depolarizing_error(min(1.5 * e, 0.75), 1), (q,))
            add_error(_relaxation(snap, o.physical[0], dur, drift), (q,))
            add_error(coherent_unitary_error(rzs), (q,))
        elif len(o.logical) == 2:
            e = _f(_edge(snap, *o.physical)['error'], 0.0) * drift.gate_scale
            add_error(depolarizing_error(min(1.25 * e, 0.9), 2), o.logical)
            for q, p in zip(o.logical, o.physical):
                add_error(_relaxation(snap, p, dur, drift), (q,))
            add_error(coherent_unitary_error(zz), o.logical)
    if measure:
        tmax = max(clock)
        for q in range(n):
            if tmax > clock[q]:
                idle(q, tmax - clock[q])
        qc.barrier()
        for q in range(n):
            qc.measure(q, q)
    return qc


def readout_noise_model(n: int, ops, snap, drift: Drift = Drift()):
    from qiskit_aer.noise import NoiseModel, ReadoutError
    nm = NoiseModel()
    phys_of = {}
    for o in ops:
        for l, p in zip(o.logical, o.physical):
            phys_of[l] = p
    for q in range(n):
        e = _f(snap['qubits'][str(phys_of[q])]['readout_error'], 0.0) * drift.readout_scale
        e0 = min(0.5, drift.readout_asymmetry * e)
        e1 = min(0.5, (2 - drift.readout_asymmetry) * e)
        nm.add_readout_error(ReadoutError([[1 - e0, e0], [e1, 1 - e1]]), [q])
    return nm


def run_surrogate(ops, n: int, snap, shots: int, seed: int, drift: Drift = Drift(),
                  max_parallel_threads: int = 0):
    """Return counts (Qiskit little-endian keys) from the synthetic surrogate."""
    from qiskit_aer import AerSimulator
    qc = build_surrogate_circuit(ops, n, snap, drift)
    sim = AerSimulator(method='statevector', noise_model=readout_noise_model(n, ops, snap, drift),
                       max_parallel_threads=max_parallel_threads)
    job = sim.run(qc, shots=shots, seed_simulator=int(seed))
    return job.result().get_counts()
