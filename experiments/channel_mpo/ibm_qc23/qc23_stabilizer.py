"""QC-23 independent stabilizer layer (stim).

* ``stim_echo_circuit`` rebuilds the echo straight from the integer schedule,
  without Qiskit, and ``verify_ideal_identity`` checks that its stabilizer
  tableau is the identity. The ideal output is therefore proven (not assumed)
  to be the all-zero string.
* ``extract_native_ops`` reads the *transpiled* ISA circuit in time order,
  keeping physical qubits, logical positions, angles and durations.
* ``ops_tableau`` translates native Clifford ops to stim so the compiled circuit
  itself can be proven equal to the identity.
* ``pauli_flip_expectations`` is an exact Pauli-fault propagation: for any
  Clifford circuit with independent Pauli faults, E[Z_i] is the product over
  fault locations of (1 - 2 P(flip bit i)). No sampling is involved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import stim

from qc23_ibm import echo_schedule, inverse_gate

HALF_PI = math.pi / 2
_ANGLE_TOL = 1e-9
SINGLE = {'x': 'X', 'y': 'Y', 'z': 'Z', 'h': 'H', 's': 'S', 'sdg': 'S_DAG',
          'sx': 'SQRT_X', 'sxdg': 'SQRT_X_DAG', 'id': 'I'}
DOUBLE = {'cx': 'CX', 'cz': 'CZ', 'swap': 'SWAP'}
NON_UNITARY = {'barrier', 'delay', 'measure'}


def stim_echo_circuit(n: int, depth: int, seed: int) -> stim.Circuit:
    """Echo built from the schedule alone; inverse by reversed dagger gates."""
    name = {'h': 'H', 's': 'S', 'x': 'X', 'sx': 'SQRT_X', 'cx': 'CX'}
    dagger = {'H': 'H', 'S': 'S_DAG', 'X': 'X', 'SQRT_X': 'SQRT_X_DAG', 'CX': 'CX'}
    layers = echo_schedule(n, depth, seed)
    c = stim.Circuit()
    for ops in layers:
        for op in ops:
            c.append(name[op[0]], list(op[1:]))
    for ops in reversed(layers):
        for op in reversed(ops):
            c.append(dagger[name[op[0]]], list(op[1:]))
    return c


def verify_ideal_identity(n: int, depth: int, seed: int) -> dict:
    """Stabilizer-tableau proof that the echo is the identity channel on |0^n>."""
    circ = stim_echo_circuit(n, depth, seed)
    tab = circ.to_tableau()
    forward = stim.Circuit()
    layers = echo_schedule(n, depth, seed)
    name = {'h': 'H', 's': 'S', 'x': 'X', 'sx': 'SQRT_X', 'cx': 'CX'}
    for ops in layers:
        for op in ops:
            forward.append(name[op[0]], list(op[1:]))
    nontrivial = forward.to_tableau() != stim.Tableau(n)
    sim = stim.TableauSimulator()
    sim.do(circ)
    z = [sim.peek_z(q) for q in range(n)]
    return {'tableau_is_identity': tab == stim.Tableau(n),
            'forward_half_is_nontrivial': bool(nontrivial),
            'peek_z': z, 'all_zero_deterministic': all(v == 1 for v in z)}


@dataclass(frozen=True)
class NativeOp:
    name: str
    logical: tuple[int, ...]
    physical: tuple[int, ...]
    angle: float | None = None
    duration: float = 0.0
    clbits: tuple[int, ...] = ()
    params: tuple[float, ...] = ()


def extract_native_ops(isa, physical, target=None) -> list[NativeOp]:
    """Time-ordered native operations on logical positions along ``physical``."""
    pos = {p: i for i, p in enumerate(physical)}
    out = []
    for inst in isa.data:
        op = inst.operation
        qs = tuple(isa.find_bit(q).index for q in inst.qubits)
        cl = tuple(isa.find_bit(c).index for c in inst.clbits)
        if any(q not in pos for q in qs):
            raise AssertionError(f'{op.name} outside the selected path {qs}')
        angle = None
        if op.name == 'rz':
            angle = float(op.params[0])
        dur = 0.0
        if op.name == 'delay':
            dur = float(op.duration) * (1.0 if op.unit == 's' else 1.0)
        elif target is not None and op.name in target.operation_names and op.name not in NON_UNITARY:
            props = target[op.name].get(qs)
            if props is not None and props.duration is not None:
                dur = float(props.duration)
        elif target is not None and op.name == 'measure':
            props = target['measure'].get(qs)
            if props is not None and props.duration is not None:
                dur = float(props.duration)
        params = tuple(float(p) for p in op.params) if op.name != 'delay' else ()
        out.append(NativeOp(op.name, tuple(pos[q] for q in qs), qs, angle, dur, cl, params))
    return out


@lru_cache(maxsize=None)
def _clifford_synth(name: str) -> tuple:
    """Stim gate list for a 1q/2q Qiskit Clifford gate, via Qiskit synthesis."""
    from qiskit.circuit.library import ECRGate, DCXGate, iSwapGate
    from qiskit.quantum_info import Clifford
    gate = {'ecr': ECRGate(), 'dcx': DCXGate(), 'iswap': iSwapGate()}[name]
    circ = Clifford(gate).to_circuit()
    seq = []
    for inst in circ.data:
        gname = inst.operation.name
        qs = tuple(circ.find_bit(q).index for q in inst.qubits)
        seq.append((gname, qs))
    return tuple(seq)


def stim_instructions(op: NativeOp):
    """Yield (stim_name, targets) for one native unitary op (Clifford only)."""
    n = op.name
    if n in NON_UNITARY or n == 'id':
        return
    if n == 'rz':
        k = op.angle / HALF_PI
        kr = round(k)
        if abs(k - kr) * HALF_PI > _ANGLE_TOL:
            raise ValueError(f'rz({op.angle}) is not a Clifford angle')
        phase = {0: None, 1: 'S', 2: 'Z', 3: 'S_DAG'}[kr % 4]
        if phase:
            yield phase, list(op.logical)
        return
    if n in SINGLE:
        yield SINGLE[n], list(op.logical)
        return
    if n in DOUBLE:
        yield DOUBLE[n], list(op.logical)
        return
    if n in ('ecr', 'dcx', 'iswap'):
        for gname, local in _clifford_synth(n):
            targets = [op.logical[q] for q in local]
            if gname in SINGLE:
                yield SINGLE[gname], targets
            elif gname in DOUBLE:
                yield DOUBLE[gname], targets
            else:
                raise ValueError(f'unsupported synthesized gate {gname}')
        return
    raise ValueError(f'gate {n!r} has no stim translation')


def ops_tableau(ops, n: int) -> stim.Tableau:
    c = stim.Circuit()
    for op in ops:
        for name, targets in stim_instructions(op):
            c.append(name, targets)
    if not len(c):
        return stim.Tableau(n)
    c.append('I', [n - 1])
    return c.to_tableau()


def verify_isa_identity(isa, physical, target=None) -> dict:
    """Prove the *transpiled* echo is the identity (stabilizer tableau)."""
    n = len(physical)
    ops = extract_native_ops(isa, physical, target)
    tab = ops_tableau(ops, n)
    return {'isa_tableau_is_identity': tab == stim.Tableau(n), 'native_ops': len(ops)}


# ---------------------------------------------------------------- Pauli faults

_PAULI_ID = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}


def depolarize1(p: float) -> dict[str, float]:
    return {'X': p / 3, 'Y': p / 3, 'Z': p / 3}


def depolarize2(p: float) -> dict[str, float]:
    labels = [a + b for a in 'IXYZ' for b in 'IXYZ' if a + b != 'II']
    return {lab: p / 15 for lab in labels}


def thermal_pauli(duration: float, t1: float, t2: float) -> dict[str, float]:
    """Pauli-twirled thermal relaxation (T2 clipped to 2*T1)."""
    if duration <= 0 or not t1 or not t2:
        return {}
    t2 = min(t2, 2 * t1)
    pxy = (1 - math.exp(-duration / t1)) / 4
    pz = (1 - math.exp(-duration / t2)) / 2 - pxy
    out = {'X': pxy, 'Y': pxy, 'Z': max(0.0, pz)}
    return {k: v for k, v in out.items() if v > 0}


@dataclass
class Fault:
    after_op: int                 # fault inserted after ops[after_op]
    qubits: tuple[int, ...]       # logical qubits (1 or 2)
    paulis: dict[str, float]
    tag: str = ''


def pauli_flip_expectations(ops, n: int, faults: list[Fault], readout_flip: list[float],
                            bits=None) -> dict[int, float]:
    """Exact E[Z_i] for |0^n> -> Clifford(ops) with independent Pauli faults.

    A readout flip r_i multiplies the expectation by (1-2 r_i).
    """
    bits = tuple(range(n)) if bits is None else tuple(bits)
    # suffix[k] = tableau of ops[k:], mapping a Pauli present before op k to the end.
    nops = len(ops)
    suffix = [None] * (nops + 1)
    suffix[nops] = stim.Tableau(n)
    for k in range(nops - 1, -1, -1):
        c = stim.Circuit()
        for name, targets in stim_instructions(ops[k]):
            c.append(name, targets)
        if len(c):
            c.append('I', [n - 1])     # pin the tableau width to n
            suffix[k] = c.to_tableau().then(suffix[k + 1])
        else:
            suffix[k] = suffix[k + 1]
    log_acc = {i: 0.0 for i in bits}
    sign = {i: 1.0 for i in bits}
    for f in faults:
        tab = suffix[f.after_op + 1]
        flip = {i: 0.0 for i in bits}
        for lab, prob in f.paulis.items():
            ps = stim.PauliString(n)
            for q, ch in zip(f.qubits, lab):
                ps[q] = _PAULI_ID[ch]
            out = tab(ps)
            for i in bits:
                if out[i] in (1, 2):
                    flip[i] += prob
        for i in bits:
            v = 1 - 2 * flip[i]
            if v <= 0:
                sign[i] *= -1 if v < 0 else 0.0
                v = abs(v) or 1e-300
            log_acc[i] += math.log(v)
    return {i: sign[i] * math.exp(log_acc[i]) * (1 - 2 * readout_flip[i]) for i in bits}
