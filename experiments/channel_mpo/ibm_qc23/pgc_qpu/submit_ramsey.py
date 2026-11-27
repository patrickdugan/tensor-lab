"""Gated submission of one frozen PGC-QPU01 Ramsey block (q23 or q5).

Same safeguards as QC26A (qc26a_compile.submit), applied to a block compiled by compile_readonly.py:
  * frozen manifest hash, proposal hash (ramsey_campaign.json), calibration hash, QPY hash, per-circuit QASM hash,
    gate-count and mapping audit, and estimate recomputation must all reproduce;
  * an owner-written authorization file naming backend, exact plan_sha256 and a QPU-second ceiling covering the
    estimate, capped at the block cap (15 s), with estimate-not-a-hard-limit and current-allocation acknowledgements;
  * compilation at most 15 minutes old, calibration timestamp/content and circuit durations unchanged;
  * remaining plan allocation at least the authorized ceiling, read from the provider immediately before submission;
  * exclusive started-marker before the irreversible call (no automatic retry after an ambiguous failure);
  * raw joint counts, calibration at submission, job id and before/after usage receipts written next to the build.
The authorization file is written by the account owner. This module never creates one.

  python submit_ramsey.py check  --out DIR                   # offline validation of a frozen build
  python submit_ramsey.py submit --out DIR --authorization FILE
"""
from __future__ import annotations
import argparse, datetime as dt, io, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qc26"))
sys.path.insert(0, str(HERE.parent))
import qc26a_compile as qc  # noqa: E402
from qc23_access import check_authorization, plan_hash  # noqa: E402
from qc23_noise import calibration_snapshot  # noqa: E402

CAMPAIGN = HERE / "ramsey_campaign.json"
BLOCK_CAP_S = 15
MAX_AGE_S = 900


def load_block(block_id):
    campaign = json.loads(CAMPAIGN.read_text())
    block = next(b for b in campaign["blocks"] if b["id"] == block_id)
    rows = block["rows"]
    assert len(rows) == 66 and len({r["id"] for r in rows}) == 66 and all(r["shots"] == 384 for r in rows)
    assert block["cap_seconds"] == BLOCK_CAP_S and block["backend"] == "ibm_kingston"
    return campaign, block


def validate_frozen(out: Path):
    from qiskit import qpy
    meta = json.loads((out / "compiled.json").read_text())
    sha = meta.pop("plan_sha256")
    if plan_hash(meta) != sha:
        raise ValueError("manifest hash mismatch")
    meta["plan_sha256"] = sha
    campaign, block = load_block(meta["block"])
    assert meta["backend"] == "ibm_kingston" and meta["cap_seconds"] == BLOCK_CAP_S and meta["shots"] == 384
    assert meta["version"] == campaign["version"]
    assert meta["proposal_sha256"] == qc.hash_bytes(CAMPAIGN.read_bytes()), "campaign manifest changed since compile"
    assert meta["calibration_sha256"] == qc.hash_bytes((out / "calibration_build.json").read_bytes())
    data = (out / "compiled.qpy").read_bytes()
    assert meta["qpy_sha256"] == qc.hash_bytes(data)
    isas = qpy.load(io.BytesIO(data))
    assert len(isas) == len(block["rows"]) == len(meta["circuits"])
    for spec, row, isa in zip(qc.ordered_rows(block["rows"],meta.get("ordering_seed")), meta["circuits"], isas):
        assert all(row[k] == v for k, v in spec.items()), "plan row mismatch"
        qc.assert_counts(row, isa)
        assert qc.audit_mapping(isa, row["path"], len(row["path"])) == row["mapping"]
        assert qc.hash_bytes((out / "circuits" / f"{row['id']}.qasm").read_bytes()) == row["qasm_sha256"]
        assert qc.hash_bytes(qc.qasm3_text(isa).encode()) == row["qasm_sha256"]
    assert abs(qc.estimate(block, meta["circuits"]) - meta["estimated_qpu_seconds"]) < 1e-9
    return meta, isas


def usage_of(service):
    try:
        u = service.usage()
        return {k: u.get(k) for k in ("usage_consumed_seconds", "usage_limit_seconds", "usage_remaining_seconds", "usage_period", "usage_limit_reached")}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def submit(out: Path, authorization, *, connection=None, sampler_cls=None, clock=None):
    """connection() -> (service, backend); sampler_cls and clock are injectable for offline tests."""
    clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))
    connection = connection or qc.backend_connection
    meta, isas = validate_frozen(out)
    if not authorization:
        raise ValueError("owner authorization file is required")
    ok, why = check_authorization(str(authorization), backend=meta["backend"], sha=meta["plan_sha256"],
                                  estimated_seconds=meta["estimated_qpu_seconds"])
    if not ok:
        raise ValueError(why)
    auth = json.loads(Path(authorization).read_text())
    if auth["max_qpu_seconds"] > BLOCK_CAP_S or meta["estimated_qpu_seconds"] > BLOCK_CAP_S:
        raise ValueError(f"block cap is {BLOCK_CAP_S} estimated QPU seconds")
    if auth.get("acknowledge_estimate_not_hard_limit") is not True or auth.get("checked_current_allocation") is not True:
        raise ValueError("owner must acknowledge provider runtime uncertainty and confirm current allocation")
    age = (clock() - dt.datetime.fromisoformat(meta["compiled_at"])).total_seconds()
    if not 0 <= age <= MAX_AGE_S:
        raise ValueError("frozen compilation older than 15 minutes; rebuild and reauthorize")
    service, backend = connection()
    old = json.loads((out / "calibration_build.json").read_text())
    snap = calibration_snapshot(backend, source="pgc_ramsey_submission")
    if not old.get("timestamp") or snap.get("timestamp") != old["timestamp"]:
        raise ValueError("calibration timestamp changed or missing; rebuild and reauthorize")
    for k in ("qubits", "edges", "basis_gates", "two_qubit_gate"):
        if json.dumps(old[k], sort_keys=True) != json.dumps(snap[k], sort_keys=True):
            raise ValueError(f"calibration {k} changed; rebuild and reauthorize")
    for row, isa in zip(meta["circuits"], isas):
        if abs(qc.duration(isa, backend.target) - row["duration_s"]) > 1e-12:
            raise ValueError("duration changed; rebuild and reauthorize")
    before = usage_of(service)
    rem = before.get("usage_remaining_seconds")
    if (before.get("usage_limit_reached") or type(rem) not in (int, float) or not math.isfinite(rem)
            or rem < auth["max_qpu_seconds"]):
        raise ValueError(f"remaining allocation {rem} s is below authorized ceiling {auth['max_qpu_seconds']} s")
    qc.write_json(out / "calibration_submission.json", snap)
    qc.write_json(out / "usage_before.json", {"checked_at": clock().isoformat(), **before})
    with (out / "submission_started.json").open("x") as f:   # exclusive: never auto-retry an ambiguous submit
        json.dump({"started_at": clock().isoformat(), "plan_sha256": meta["plan_sha256"]}, f)
    if sampler_cls is None:
        from qiskit_ibm_runtime import SamplerV2 as sampler_cls
    sampler = sampler_cls(mode=backend, options={"execution": {"rep_delay": 0.00025}})
    job = sampler.run(isas, shots=meta["shots"])
    qc.write_json(out / "job.json", {"job_id": job.job_id(), "plan_sha256": meta["plan_sha256"], "submitted_at": clock().isoformat()})
    print("Submitted job:", job.job_id(), flush=True)
    result = job.result()
    if len(result) != len(isas):
        raise RuntimeError("PUB count mismatch")
    counts = {}
    for row, entry in zip(meta["circuits"], result):
        c = entry.data.meas.get_counts()
        assert sum(c.values()) == row["shots"]
        assert all(len(k) == len(row["path"]) and set(k) <= set("01") for k in c)
        counts[row["id"]] = {k: int(v) for k, v in c.items()}
    qc.write_json(out / "counts.json", counts)
    after = usage_of(service)
    job_usage = None
    try:
        job_usage = job.usage()
    except Exception as exc:  # noqa: BLE001
        job_usage = f"unavailable: {type(exc).__name__}: {exc}"
    consumed = None
    try:
        consumed = after["usage_consumed_seconds"] - before["usage_consumed_seconds"]
    except (KeyError, TypeError):
        pass
    qc.write_json(out / "execution_receipt.json", {
        "job_id": job.job_id(), "plan_sha256": meta["plan_sha256"], "block": meta["block"],
        "estimated_qpu_seconds": meta["estimated_qpu_seconds"], "authorized_ceiling_seconds": auth["max_qpu_seconds"],
        "usage_before": before, "usage_after": after, "consumed_seconds_delta": consumed, "job_usage": job_usage,
        "calibration_timestamp_build": old["timestamp"], "calibration_timestamp_submission": snap["timestamp"],
        "circuits": len(counts), "shots_per_circuit": meta["shots"]})
    print("Wrote raw counts and execution receipt; consumed (provider delta):", consumed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["check", "submit"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--authorization", type=Path)
    a = p.parse_args()
    if a.command == "check":
        if a.authorization:
            p.error("authorization is for submit only")
        meta, _ = validate_frozen(a.out)
        print(json.dumps({k: meta[k] for k in ("block", "plan_sha256", "estimated_qpu_seconds", "compiled_at")}, indent=2))
    else:
        submit(a.out, a.authorization)


if __name__ == "__main__":
    main()
