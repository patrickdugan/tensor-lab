"""QC-23 IBM account discovery and the hardware-authorization gate.

Credentials are never read, printed, logged or serialised here. The runtime
client resolves them from the user's own saved account or environment.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


def connect(instance: str | None = None):
    """Return (service, None) or (None, exact error string)."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(instance=instance) if instance else QiskitRuntimeService()
        return service, None
    except Exception as exc:  # noqa: BLE001 - report the exact failure, once
        return None, f'{type(exc).__name__}: {exc}'


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def backend_summary(backend) -> dict:
    """Topology, basis, calibration medians and queue state of one backend."""
    from qc23_compile import two_qubit_gate_name, edge_errors, readout_errors, single_qubit_errors
    target = backend.target
    status = backend.status()
    try:
        twoq = two_qubit_gate_name(target)
        edges = edge_errors(target)
        coupling = sorted(edges)
    except RuntimeError:
        twoq, edges, coupling = None, {}, []
    out = {'name': backend.name, 'num_qubits': backend.num_qubits,
           'operational': bool(status.operational), 'pending_jobs': int(status.pending_jobs),
           'status_msg': status.status_msg, 'basis_gates': sorted(target.operation_names),
           'two_qubit_gate': twoq, 'coupling_edges': len(coupling), 'coupling_map': coupling,
           'median_two_qubit_error': _median(list(edges.values())),
           'median_readout_error': _median(readout_errors(target, backend.num_qubits)),
           'median_1q_error': _median(single_qubit_errors(target, backend.num_qubits))}
    try:
        out['processor_type'] = str(backend.processor_type)
    except Exception:  # noqa: BLE001
        out['processor_type'] = None
    return out


def discover(service) -> dict:
    """Accessible QPUs plus (if the API exposes it) plan usage. No secrets."""
    info = {'backends': [], 'usage': None, 'usage_error': None, 'instances': None}
    for backend in service.backends(simulator=False):
        try:
            info['backends'].append(backend_summary(backend))
        except Exception as exc:  # noqa: BLE001
            info['backends'].append({'name': getattr(backend, 'name', '?'),
                                     'error': f'{type(exc).__name__}: {exc}'})
    try:
        info['usage'] = service.usage()
    except Exception as exc:  # noqa: BLE001
        info['usage_error'] = f'{type(exc).__name__}: {exc}'
    try:
        info['instances'] = [{k: v for k, v in inst.items() if k in ('name', 'plan', 'tags', 'pricing_type')}
                             for inst in service.instances()]
    except Exception as exc:  # noqa: BLE001
        info['instances'] = f'unavailable: {type(exc).__name__}: {exc}'
    return info


def plan_hash(rows) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def estimate_qpu_seconds(circuit_durations_s, shots: int, rep_delay_s: float = 250e-6) -> float:
    """Upper-ish estimate: shots x (rep delay + circuit duration) summed over circuits."""
    return float(sum(shots * (rep_delay_s + d) for d in circuit_durations_s))


def check_authorization(auth_path: str | None, *, backend: str, sha: str,
                        estimated_seconds: float) -> tuple[bool, str]:
    """Hardware submission needs an authorization file written by the account owner.

    The ``--allow-paid`` flag is deliberately NOT sufficient. The file must name
    this backend, this exact compiled-plan hash and a QPU-seconds ceiling that
    covers the estimate. Without it nothing is submitted.
    """
    if not auth_path:
        return False, 'no --authorization file supplied'
    path = Path(auth_path)
    if not path.is_file():
        return False, f'authorization file not found: {path}'
    try:
        auth = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return False, f'authorization file is not valid JSON: {exc}'
    if auth.get('authorized') is not True:
        return False, 'authorization.authorized is not true'
    if auth.get('backend') != backend:
        return False, f"authorization names backend {auth.get('backend')!r}, not {backend!r}"
    if auth.get('plan_sha256') != sha:
        return False, 'authorization plan_sha256 does not match the compiled plan'
    ceiling = auth.get('max_qpu_seconds')
    if not isinstance(ceiling, (int, float)) or ceiling <= 0:
        return False, 'authorization.max_qpu_seconds must be a positive number'
    if estimated_seconds > ceiling:
        return False, f'estimated {estimated_seconds:.1f}s exceeds authorized {ceiling}s'
    if not auth.get('authorized_by'):
        return False, 'authorization.authorized_by missing'
    return True, 'authorized'
