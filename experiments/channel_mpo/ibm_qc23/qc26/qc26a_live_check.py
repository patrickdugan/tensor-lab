"""QC-26A read-only live check of a frozen build: calibration freshness, SamplerV2 PUB validity, plan usage.

Submits nothing and consumes no QPU time. Uses the locally saved IBM account; prints no credentials.

  python qc26a_live_check.py --build DIR --out live_receipt.json
"""
from __future__ import annotations
import argparse, datetime as dt, io, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import qc26a_compile as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    meta, isas = C.validate_frozen(a.build)
    service, backend = C.backend_connection()
    old = json.loads((a.build / 'calibration_build.json').read_text())
    from qc23_noise import calibration_snapshot
    snap = calibration_snapshot(backend, source='qc26a_live_check')
    same_ts = snap['timestamp'] == old['timestamp']
    same_content = all(json.dumps(old[k], sort_keys=True) == json.dumps(snap[k], sort_keys=True)
                       for k in ('qubits', 'edges', 'basis_gates', 'two_qubit_gate'))
    durations_ok = all(abs(C.duration(isa, backend.target) - r['duration_s']) <= 1e-12
                       for r, isa in zip(meta['circuits'], isas))
    from qiskit.primitives.containers import SamplerPub
    pubs = [SamplerPub.coerce(isa) for isa in isas]   # PUB structure and parameter-free circuits
    from qiskit_ibm_runtime import SamplerV2
    SamplerV2(mode=backend, options={'execution': {'rep_delay': 0.00025}})   # option accepted locally; no job is run
    # Every instruction must be in the live target's basis (the submit path re-checks durations the same way).
    in_target = all(inst.operation.name in set(backend.target.operation_names) | {'barrier', 'measure'}
                    for isa in isas for inst in isa.data)
    rep_ok = backend.configuration().rep_delay_range if hasattr(backend, 'configuration') else None
    try:
        usage = service.usage()
    except Exception as exc:  # noqa: BLE001
        usage = {'error': f'{type(exc).__name__}: {exc}'}
    keep = ('usage_consumed_seconds', 'usage_limit_seconds', 'usage_remaining_seconds', 'usage_period', 'usage_limit_reached')
    status = backend.status()
    now = dt.datetime.now(dt.timezone.utc)
    compiled = dt.datetime.fromisoformat(meta['compiled_at'])
    receipt = {
        'checked_at': now.isoformat(), 'plan_sha256': meta['plan_sha256'],
        'build_calibration_timestamp': old['timestamp'], 'live_calibration_timestamp': snap['timestamp'],
        'calibration_unchanged': same_ts and same_content, 'durations_unchanged': durations_ok,
        'build_age_minutes': (now - compiled).total_seconds() / 60,
        'submission_window_open': 0 <= (now - compiled).total_seconds() <= 900,
        'pubs_coerced': len(pubs), 'sampler_v2_constructed_with_rep_delay_option': True, 'all_ops_in_live_target': in_target,
        'rep_delay_range_s': list(rep_ok) if rep_ok else None,
        'backend_operational': bool(status.operational), 'pending_jobs': int(status.pending_jobs),
        'usage': {k: usage.get(k) for k in keep} if isinstance(usage, dict) else usage,
        'estimated_qpu_seconds': meta['estimated_qpu_seconds'],
    }
    a.out.write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    print(json.dumps(receipt, indent=2, default=str))


if __name__ == '__main__':
    main()
