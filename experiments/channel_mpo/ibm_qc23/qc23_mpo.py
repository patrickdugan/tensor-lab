"""QC-23 bridge from a *transpiled* ISA gate sequence to the QC-12/14 MPO stack.

The MPO stack consumes disjoint nearest-neighbour 4x4 unitary layers followed
by one uniform single-site channel per layer. ``ops_to_layers`` folds every
native single-qubit gate into the next two-qubit gate on that qubit (trailing
single-qubit gates are folded into a nearest-neighbour carrier gate), so the
simulation input is exactly the compiled gate sequence, not the logical one.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from channel_mpo import kraus  # noqa: E402
from qc12_heisenberg import operator_cone_expectation  # noqa: E402
from qc10_cone import backward_cone, exact_cone_density  # noqa: E402

I2 = np.eye(2, dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], complex)
PURE_1Q = {'rz', 'sx', 'x', 'y', 'z', 'h', 's', 'sdg', 'sxdg', 't', 'tdg', 'rx', 'ry', 'p', 'u1', 'id'}
TWO_Q = {'cz', 'cx', 'ecr', 'swap', 'rzz', 'rxx', 'iswap'}


def _matrix(name: str, params) -> np.ndarray:
    from qiskit.circuit.library import get_standard_gate_name_mapping
    gate = get_standard_gate_name_mapping()[name]
    gate = gate.base_class(*params) if params else gate
    return np.array(gate.to_matrix(), dtype=complex)


def two_qubit_big_endian(name: str, params, first: int, second: int) -> np.ndarray:
    """Unitary on (site=min(first,second), site+1), site = most significant."""
    m = _matrix(name, params)           # Qiskit: qubit0 = LSB
    # Qiskit gate acts as (q0=first, q1=second). Convert to MSB = lower site.
    if first < second:
        return SWAP @ m @ SWAP          # first (LSB) must become MSB
    return m                            # second is the lower site, already MSB


def ops_to_layers(ops, n: int, *, drop_trailing_diagonal: bool = True):
    """Fold native ops into disjoint nearest-neighbour two-qubit layers.

    Returns (layers, info). ``layers[t]`` is a list of (site, 4x4 unitary).
    """
    pending = {q: None for q in range(n)}
    free = [0] * n
    layers: list[list] = []

    def place(site: int, unitary: np.ndarray, at_least: int):
        t = max(at_least, free[site], free[site + 1])
        while len(layers) <= t:
            layers.append([])
        layers[t].append((site, unitary))
        free[site] = free[site + 1] = t + 1

    for o in ops:
        if o.name in ('barrier', 'delay', 'id'):
            continue
        if o.name == 'measure':
            continue
        if len(o.logical) == 1:
            if o.name not in PURE_1Q:
                raise ValueError(f'unsupported native 1q gate {o.name}')
            m = _matrix(o.name, o.params)
            q = o.logical[0]
            pending[q] = m if pending[q] is None else m @ pending[q]
            continue
        if len(o.logical) != 2 or o.name not in TWO_Q:
            raise ValueError(f'unsupported native gate {o.name}')
        a, b = o.logical
        if abs(a - b) != 1:
            raise ValueError(f'non-nearest-neighbour {o.name}{o.logical}')
        site = min(a, b)
        g = two_qubit_big_endian(o.name, o.params, a, b)
        pre = np.kron(pending[site] if pending[site] is not None else I2,
                      pending[site + 1] if pending[site + 1] is not None else I2)
        pending[site] = pending[site + 1] = None
        place(site, g @ pre, 0)
    leftovers = {}
    for q, m in pending.items():
        if m is None:
            continue
        if drop_trailing_diagonal and np.allclose(m, np.diag(np.diag(m)), atol=1e-12):
            continue
        leftovers[q] = m
    done = set()
    for q in sorted(leftovers):
        if q in done:
            continue
        if q + 1 < n and (q + 1) in leftovers:
            place(q, np.kron(leftovers[q], leftovers[q + 1]), 0)
            done.update((q, q + 1))
        elif q + 1 < n:
            place(q, np.kron(leftovers[q], I2), 0)
            done.add(q)
        else:
            place(q - 1, np.kron(I2, leftovers[q]), 0)
            done.add(q)
    for layer in layers:
        layer.sort(key=lambda t: t[0])
    return layers, {'mpo_layers': len(layers), 'gates': sum(len(l) for l in layers)}


def max_unitarity_error(layers) -> float:
    return max((float(np.abs(u.conj().T @ u - np.eye(4)).max()) for l in layers for _, u in l), default=0.0)


def cone_width(n: int, layers, target: int) -> int:
    return len(backward_cone(n, layers, (target,)).sites)


def mpo_z(n: int, layers, p: float, target: int, *, bond_cap: int = 16, max_cone: int = 64,
          max_entries: int = 4_000_000, channel: str = 'depolarizing'):
    """QC-12 expectation of Z_target with its local-residual error ledger."""
    ks = kraus(channel, float(p))
    return operator_cone_expectation(n, layers, ks, (target,), 'Z', bond_cap=bond_cap,
                                     max_cone=max_cone, max_entries=max_entries)


def dense_cone_z(n: int, layers, p: float, target: int, *, max_sites: int = 12,
                 channel: str = 'depolarizing') -> float:
    ks = kraus(channel, float(p))
    d = exact_cone_density(n, layers, ks, (target,), max_sites=max_sites)
    Z = np.diag([1.0, -1.0]).astype(complex)
    return float(np.trace(d.reduced_density @ Z).real)


def certified_z(n: int, layers, p: float, target: int, *, bond_cap: int = 16,
                max_cone: int = 64, max_entries: int = 4_000_000, max_events: int = 20000):
    """QC-14 exact-dyadic certified expectation, or None with the failure reason."""
    from qc14_certified import certify_observable
    ks = kraus('depolarizing', float(p))
    try:
        return certify_observable(n, layers, ks, (target,), 'Z', bond_cap=bond_cap,
                                  max_cone=max_cone, max_entries=max_entries,
                                  max_events=max_events), None
    except (MemoryError, ArithmeticError, ValueError) as exc:
        return None, f'{type(exc).__name__}: {exc}'


# ------------------------------------------------ exact Pauli reference (layered)


def layered_clifford_reference(n: int, layers, p: float, targets) -> dict[int, float]:
    """Exact E[Z_t] for the *layered* model with uniform per-layer depolarising p.

    Independent of QC-12/13/14: gates become stim tableaus, each layer is followed
    by single-site depolarising faults on every qubit, and E[Z] is the product of
    (1-2 P(flip)) over fault locations. Valid only if every layer gate is Clifford.
    """
    import stim
    # Stim gate order: apply layers sequentially; faults after each layer.
    tabs = []
    for layer in layers:
        t = stim.Tableau(n)
        for site, u in layer:
            gt = stim.Tableau.from_unitary_matrix(u, endian='big')
            t_full = stim.Tableau(n)
            t_full.append(gt, [site, site + 1])
            t = t.then(t_full)
        tabs.append(t)
    suffix = [None] * (len(tabs) + 1)
    suffix[-1] = stim.Tableau(n)
    for k in range(len(tabs) - 1, -1, -1):
        suffix[k] = tabs[k].then(suffix[k + 1])
    out = {}
    shrink = 1.0 - float(p)
    for tgt in targets:
        # Heisenberg picture: every site depolarises after each layer; the Z flip
        # probability of a fault at (layer k, site s, Pauli P) is whether
        # suffix[k+1](P_s) has X/Y at the target.
        log_sum = 0.0
        sign = 1.0
        for k in range(len(tabs)):
            tab = suffix[k + 1]
            for s in range(n):
                flip = 0.0
                for ch in (1, 2, 3):
                    ps = stim.PauliString(n)
                    ps[s] = ch
                    if tab(ps)[tgt] in (1, 2):
                        flip += p / 4
                v = 1 - 2 * flip
                if v < 0:
                    sign *= -1
                log_sum += math.log(abs(v) if v else 1e-300)
        out[tgt] = sign * math.exp(log_sum)
    return out


def effective_depolarizing_p(faults, n_sites: int, n_layers: int) -> float:
    """Uniform single-site depolarising p matching mean log-shrink per site-layer."""
    from qc23_noise import site_shrink_log
    logs = [site_shrink_log(faults, probe) for probe in 'XYZ']
    mean = sum(logs) / 3.0
    per = mean / max(1, n_sites * max(1, n_layers))
    return float(min(1.0, max(0.0, 1.0 - math.exp(per))))
