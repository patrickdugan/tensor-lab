"""QC-23 compilation: calibration-aware layouts, ISA transpilation, mapping audit.

Everything here works on a backend-like object exposing ``target`` and
``num_qubits`` (a real ``IBMBackend`` or a ``qiskit_ibm_runtime.fake_provider``
snapshot). No network access is performed.
"""
from __future__ import annotations

import hashlib
import io
import math

TWO_QUBIT_NAMES = ('cz', 'ecr', 'cx')


def two_qubit_gate_name(target) -> str:
    for name in TWO_QUBIT_NAMES:
        if name in target.operation_names:
            return name
    raise RuntimeError('backend exposes no cz/ecr/cx native two-qubit gate')


def _err(props, default=0.0):
    if props is None or props.error is None:
        return default
    return float(props.error)


def edge_errors(target) -> dict[tuple[int, int], float]:
    """Undirected two-qubit gate error; the best direction wins."""
    name = two_qubit_gate_name(target)
    out: dict[tuple[int, int], float] = {}
    for (a, b), props in target[name].items():
        key = (min(a, b), max(a, b))
        e = _err(props, 1e-2)
        out[key] = min(out.get(key, 1.0), e)
    return out


def readout_errors(target, n_qubits: int) -> list[float]:
    return [_err(target['measure'].get((q,)), 0.05) for q in range(n_qubits)]


def single_qubit_errors(target, n_qubits: int) -> list[float]:
    name = 'sx' if 'sx' in target.operation_names else 'x'
    return [_err(target[name].get((q,)), 1e-3) for q in range(n_qubits)]


def path_cost(path, edges, readout, oneq) -> float:
    """Negative log of an independence-model success probability."""
    cost = 0.0
    for a, b in zip(path, path[1:]):
        cost += -math.log(max(1e-12, 1 - edges[(min(a, b), max(a, b))]))
    for q in path:
        cost += -math.log(max(1e-12, 1 - readout[q])) - 4 * math.log(max(1e-12, 1 - oneq[q]))
    return cost


def choose_line_calibrated(backend, n: int, *, beam: int = 256):
    """Beam search for a simple path of ``n`` qubits with minimal error cost.

    Deterministic. Returns (path, cost). Topology comes from the two-qubit gate
    support in ``backend.target``; the cost uses the calibration stored there.
    """
    target = backend.target
    edges = edge_errors(target)
    readout = readout_errors(target, backend.num_qubits)
    oneq = single_qubit_errors(target, backend.num_qubits)
    nbr: dict[int, set[int]] = {q: set() for q in range(backend.num_qubits)}
    for a, b in edges:
        nbr[a].add(b)
        nbr[b].add(a)
    frontier = [([q], path_cost([q], edges, readout, oneq)) for q in sorted(nbr) if nbr[q]]
    frontier.sort(key=lambda t: (t[1], t[0]))
    frontier = frontier[:beam]
    for _ in range(n - 1):
        nxt = []
        for path, _c in frontier:
            seen = set(path)
            for o in sorted(nbr[path[-1]] - seen):
                p2 = path + [o]
                nxt.append((p2, path_cost(p2, edges, readout, oneq)))
        if not nxt:
            raise RuntimeError(f'no connected simple path of {n} qubits')
        nxt.sort(key=lambda t: (t[1], t[0]))
        frontier = nxt[:beam]
    frontier = [(p, c) for p, c in frontier if len(p) == n]
    if not frontier:
        raise RuntimeError(f'no connected simple path of {n} qubits')
    path, cost = min(frontier, key=lambda t: (t[1], t[0]))
    # Canonical orientation: lower physical index first (the cost is symmetric).
    if path[0] > path[-1]:
        path = path[::-1]
    return path, cost


def transpile_on_path(backend, circuit, physical, *, optimization_level: int = 0):
    """Transpile at level 0 so forward/inverse echoes are not cancelled."""
    from qiskit.transpiler import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                      initial_layout=list(physical), seed_transpiler=0)
    return pm.run(circuit)


def measurement_map(isa) -> dict[int, int]:
    """clbit index -> physical qubit actually measured in the ISA circuit."""
    out: dict[int, int] = {}
    for inst in isa.data:
        if inst.operation.name == 'measure':
            q = isa.find_bit(inst.qubits[0]).index
            c = isa.find_bit(inst.clbits[0]).index
            if c in out:
                raise AssertionError(f'clbit {c} written twice')
            out[c] = q
    return out


def audit_mapping(isa, physical, n: int) -> dict:
    """Prove logical bit i is read from physical qubit ``physical[i]``.

    Checks (1) every measure op, (2) the transpiler's recorded layout, (3) that
    no gate acts on a qubit outside the path, (4) that every 2q gate is between
    path neighbours (no routing swaps).
    """
    mm = measurement_map(isa)
    if sorted(mm) != list(range(n)):
        raise AssertionError('measurement register is not exactly bits 0..n-1')
    if any(mm[i] != physical[i] for i in range(n)):
        raise AssertionError('measurement mapping differs from initial layout')
    layout = getattr(isa, 'layout', None)
    final_index = None
    if layout is not None:
        init_index = list(layout.initial_index_layout(filter_ancillas=True))
        final_index = list(layout.final_index_layout(filter_ancillas=True))
        if init_index != list(physical):
            raise AssertionError(f'transpiler initial layout {init_index} != requested {list(physical)}')
        if final_index != list(physical):
            raise AssertionError(f'routing permuted the layout: final {final_index}')
    pos = {p: i for i, p in enumerate(physical)}
    swaps = 0
    two_q = 0
    for inst in isa.data:
        name = inst.operation.name
        if name in ('barrier', 'delay'):
            continue
        qs = [isa.find_bit(q).index for q in inst.qubits]
        if any(q not in pos for q in qs):
            raise AssertionError(f'{name} acts outside the selected path: {qs}')
        if name == 'swap':
            swaps += 1
        if len(qs) == 2 and name != 'measure':
            two_q += 1
            if abs(pos[qs[0]] - pos[qs[1]]) != 1:
                raise AssertionError(f'non-neighbour two-qubit gate {name}{qs}')
    return {'measurement_map': {str(c): q for c, q in sorted(mm.items())},
            'initial_layout_logical_to_physical': list(physical),
            'final_index_layout': final_index,
            'two_qubit_gates': two_q, 'swaps': swaps}


def isa_metrics(isa) -> dict:
    ops = {k: int(v) for k, v in isa.count_ops().items()}
    return {'depth': int(isa.depth()),
            'two_qubit_depth': int(isa.depth(lambda i: len(i.qubits) == 2 and i.operation.name != 'measure')),
            'size': int(isa.size()), 'gate_counts': ops,
            'num_two_qubit': int(sum(v for k, v in ops.items() if k in TWO_QUBIT_NAMES))}


def qasm3_text(isa) -> str:
    from qiskit import qasm3
    return qasm3.dumps(isa)


def qpy_bytes(circuits) -> bytes:
    from qiskit import qpy
    buf = io.BytesIO()
    qpy.dump(list(circuits), buf)
    return buf.getvalue()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
