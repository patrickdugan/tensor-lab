"""QC-23 Phase 5: a small non-Clifford family compatible with the QC-12/14 stack.

Kicked-Ising quench from |0^n>: each Trotter step applies seeded, jittered
single-qubit RX kicks, small RZ fields, then RZZ couplings on an alternating
nearest-neighbour brickwork. Generic angles make every circuit non-Clifford,
so the stabilizer oracle does not apply; ideal values come from dense
statevectors (n <= 16) and the causal cone grows by 2 sites per step.
"""
from __future__ import annotations

import random

import numpy as np


def kicked_ising_params(n: int, steps: int, seed: int):
    rng = random.Random(seed)
    out = []
    for step in range(steps):
        rx = [0.55 + 0.30 * (rng.random() - 0.5) for _ in range(n)]
        rz = [0.15 * (rng.random() - 0.5) for _ in range(n)]
        rzz = {i: 0.70 + 0.30 * (rng.random() - 0.5) for i in range(step % 2, n - 1, 2)}
        out.append({'rx': rx, 'rz': rz, 'rzz': rzz})
    return out


def make_kicked_ising(n: int, steps: int, seed: int):
    from qiskit import QuantumCircuit
    if n < 2 or steps < 1:
        raise ValueError('n >= 2 and steps >= 1 required')
    qc = QuantumCircuit(n)
    for layer in kicked_ising_params(n, steps, seed):
        for i in range(n):
            qc.rx(layer['rx'][i], i)
            qc.rz(layer['rz'][i], i)
        for i, ang in layer['rzz'].items():
            qc.rzz(ang, i, i + 1)
        qc.barrier()
    return qc


def ideal_z(circuit, bits) -> dict[int, float]:
    """Exact noiseless <Z_i> (dense statevector; n <= 20)."""
    from qiskit.quantum_info import Statevector
    n = circuit.num_qubits
    if n > 20:
        raise MemoryError('dense reference limited to 20 qubits')
    probs = Statevector.from_instruction(circuit).probabilities()
    idx = np.arange(probs.size)
    return {b: float(np.sum(probs * (1 - 2 * ((idx >> b) & 1)))) for b in bits}
