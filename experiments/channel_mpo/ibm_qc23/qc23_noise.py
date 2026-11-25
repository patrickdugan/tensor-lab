"""QC-23 noise models.

Model A  - calibration-derived local Pauli model (per-qubit/per-edge gate error,
           Pauli-twirled T1/T2 idling on an ASAP schedule, per-qubit readout).
Model B  - compact effective channel (p1, p2, r) fitted on a designated
           calibration subset of circuits, never on evaluation circuits.

Both are evaluated *exactly* on Clifford circuits by Pauli-fault propagation
(``qc23_stabilizer.pauli_flip_expectations``). Neither is claimed to equal the
QPU; their discrepancy from hardware is measured, not assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from qc23_compile import two_qubit_gate_name
from qc23_stabilizer import (Fault, NativeOp, depolarize1, depolarize2,
                             pauli_flip_expectations, thermal_pauli)

# --------------------------------------------------------------- calibration


def calibration_snapshot(backend, *, timestamp: str | None = None, source: str = '') -> dict:
    """JSON-serialisable calibration snapshot from ``backend.target``."""
    target = backend.target
    twoq = two_qubit_gate_name(target)
    qubits = {}
    for q in range(backend.num_qubits):
        qp = target.qubit_properties[q] if target.qubit_properties else None
        rec = {'t1': getattr(qp, 't1', None), 't2': getattr(qp, 't2', None),
               'frequency': getattr(qp, 'frequency', None)}
        m = target['measure'].get((q,))
        rec['readout_error'] = None if m is None else m.error
        rec['readout_duration'] = None if m is None else m.duration
        for g in ('sx', 'x'):
            pr = target[g].get((q,)) if g in target.operation_names else None
            rec[f'{g}_error'] = None if pr is None else pr.error
            rec[f'{g}_duration'] = None if pr is None else pr.duration
        qubits[str(q)] = rec
    edges = []
    for qargs, pr in target[twoq].items():
        edges.append({'gate': twoq, 'qargs': list(qargs), 'error': pr.error, 'duration': pr.duration})
    ts = timestamp
    if ts is None:
        try:
            props = backend.properties()
            ts = None if props is None else props.last_update_date.isoformat()
        except Exception:
            ts = None
    return {'backend': backend.name, 'timestamp': ts, 'source': source,
            'num_qubits': backend.num_qubits, 'dt': getattr(backend, 'dt', None),
            'two_qubit_gate': twoq, 'basis_gates': sorted(target.operation_names),
            'qubits': qubits, 'edges': edges}


def restrict_snapshot(snap: dict, physical) -> dict:
    keep = {int(q) for q in physical}
    return {**snap,
            'qubits': {q: v for q, v in snap['qubits'].items() if int(q) in keep},
            'edges': [e for e in snap['edges'] if set(e['qargs']) <= keep]}


def _edge(snap, a, b):
    for e in snap['edges']:
        if e['qargs'] == [a, b]:
            return e
    for e in snap['edges']:
        if e['qargs'] == [b, a]:
            return e
    raise KeyError((a, b))


def _f(x, default):
    return default if x is None else float(x)


# ------------------------------------------------------------------- Model A


def model_a_faults(ops: list[NativeOp], snap: dict, *, error_scale: float = 1.0,
                   include_idle: bool = True):
    """Calibration-derived Pauli faults and readout flips for ``ops``."""
    n = 1 + max(max(o.logical) for o in ops if o.logical)
    clock = [0.0] * n
    faults: list[Fault] = []
    readout = [0.0] * n
    phys_of = {}
    for o in ops:
        for l, p in zip(o.logical, o.physical):
            phys_of[l] = p

    def relax(q, duration):
        rec = snap['qubits'][str(phys_of[q])]
        return thermal_pauli(duration, rec['t1'], rec['t2'])

    for k, o in enumerate(ops):
        if o.name == 'measure':
            for l, p in zip(o.logical, o.physical):
                readout[l] = _f(snap['qubits'][str(p)]['readout_error'], 0.0) * error_scale
            continue
        if o.name == 'barrier':
            tmax = max(clock[q] for q in o.logical)
            for q in o.logical:
                if include_idle and tmax > clock[q]:
                    idle = relax(q, tmax - clock[q])
                    if idle:
                        faults.append(Fault(k - 1, (q,), {a: b * error_scale for a, b in idle.items()}, 'idle'))
                clock[q] = tmax
            continue
        # gate duration
        start = max(clock[q] for q in o.logical)
        if o.name == 'rz':
            dur = 0.0
        elif len(o.logical) == 1:
            dur = _f(snap['qubits'][str(o.physical[0])].get(f'{o.name}_duration'), 0.0) if o.name in ('sx', 'x') else o.duration
        else:
            dur = _f(_edge(snap, *o.physical)['duration'], 0.0)
        if o.name == 'delay':
            dur = o.duration
        for q in o.logical:
            gap = start - clock[q]
            if include_idle and gap > 0:
                idle = relax(q, gap)
                if idle:
                    faults.append(Fault(k - 1, (q,), {a: b * error_scale for a, b in idle.items()}, 'idle'))
            clock[q] = start + dur
        if o.name in ('sx', 'x'):
            e = _f(snap['qubits'][str(o.physical[0])].get(f'{o.name}_error'), 0.0) * error_scale
            rel = relax(o.logical[0], dur)
            pdep = max(0.0, 1.5 * e - sum(rel.values()))
            if pdep > 0:
                faults.append(Fault(k, o.logical, depolarize1(min(pdep, 0.75)), o.name))
            if rel:
                faults.append(Fault(k, o.logical, {a: b * error_scale for a, b in rel.items()}, 'relax'))
        elif len(o.logical) == 2 and o.name not in ('delay',):
            e = _f(_edge(snap, *o.physical)['error'], 0.0) * error_scale
            r0, r1 = relax(o.logical[0], dur), relax(o.logical[1], dur)
            pdep = max(0.0, 1.25 * e - sum(r0.values()) - sum(r1.values()))
            if pdep > 0:
                faults.append(Fault(k, o.logical, depolarize2(min(pdep, 15 / 16)), o.name))
            for q, rel in zip(o.logical, (r0, r1)):
                if rel:
                    faults.append(Fault(k, (q,), {a: b * error_scale for a, b in rel.items()}, 'relax'))
    return faults, readout


def predict_model_a(ops, snap, bits, *, error_scale=1.0, include_idle=True):
    faults, readout = model_a_faults(ops, snap, error_scale=error_scale, include_idle=include_idle)
    n = len(readout)
    return pauli_flip_expectations(ops, n, faults, readout, bits)


# ------------------------------------------------------------------- Model B


@dataclass(frozen=True)
class ModelB:
    p1: float   # depolarising probability after each physical sx/x
    p2: float   # depolarising probability after each physical two-qubit gate
    r: float    # symmetric readout flip probability

    def as_dict(self):
        return asdict(self)


def model_b_faults(ops, params: ModelB):
    n = 1 + max(max(o.logical) for o in ops if o.logical)
    faults = []
    for k, o in enumerate(ops):
        if o.name in ('sx', 'x'):
            faults.append(Fault(k, o.logical, depolarize1(params.p1), o.name))
        elif len(o.logical) == 2 and o.name not in ('measure', 'barrier', 'delay'):
            faults.append(Fault(k, o.logical, depolarize2(params.p2), o.name))
    return faults, [params.r] * n


def predict_model_b(ops, params: ModelB, bits):
    faults, readout = model_b_faults(ops, params)
    return pauli_flip_expectations(ops, len(readout), faults, readout, bits)


def fit_model_b(records, *, start=ModelB(1e-3, 8e-3, 0.02)):
    """Weighted least-squares fit on designated calibration circuits.

    ``records``: list of dicts with ``ops`` (list[NativeOp]), ``z`` (dict
    bit -> measured <Z>), ``se`` (dict bit -> standard error), ``shots``.
    The variance floor 1/shots stops zero-variance bits from dominating.
    """
    from scipy.optimize import least_squares

    def residual(theta):
        params = ModelB(*np.exp(theta))
        res = []
        for rec in records:
            pred = predict_model_b(rec['ops'], params, tuple(rec['z']))
            for b, meas in rec['z'].items():
                sigma = max(rec['se'][b], 1.0 / rec['shots'])
                res.append((pred[int(b) if not isinstance(b, int) else b] - meas) / sigma)
        return np.array(res)

    x0 = np.log([start.p1, start.p2, start.r])
    sol = least_squares(residual, x0, bounds=(np.log([1e-7] * 3), np.log([0.3, 0.45, 0.45])))
    params = ModelB(*np.exp(sol.x))
    # Parameter covariance (log scale) from the Gauss-Newton Jacobian.
    jac = sol.jac
    try:
        cov = np.linalg.inv(jac.T @ jac)
        rel_sd = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        rel_sd = np.full(3, np.nan)
    return params, {'chi2': float(2 * sol.cost), 'dof': int(len(sol.fun) - 3),
                    'log_param_sd': [float(v) for v in rel_sd], 'success': bool(sol.success)}


# ------------------------------------------------- effective layer depolarising


def _anticommute_prob(paulis: dict[str, float], position: int, probe: str) -> float:
    total = 0.0
    for lab, p in paulis.items():
        a = lab[position]
        if a != 'I' and a != probe:
            total += p
    return total


def site_shrink_log(faults: list[Fault], probe: str = 'Z') -> float:
    """Sum over faults of log(1-2P) for the given single-qubit probe Pauli.

    1q fault on a site: factor (1-2 P(anti-commute with probe)).
    2q fault: each of its two sites receives the square root of the pair
    factor (the probe on one site only) - a product-form projection.
    """
    total = 0.0
    for f in faults:
        if len(f.qubits) == 1:
            v = 1 - 2 * _anticommute_prob(f.paulis, 0, probe)
            total += math.log(max(v, 1e-12))
        else:
            v0 = 1 - 2 * _anticommute_prob(f.paulis, 0, probe)
            v1 = 1 - 2 * _anticommute_prob(f.paulis, 1, probe)
            total += math.log(max(v0, 1e-12)) + math.log(max(v1, 1e-12))
    return total
