"""Compile-only PGC-QPU01 Ramsey blocks using the existing QC26A helpers.
No submission command exists here. Requires a configured IBM account and Qiskit
only when run. An estimate cap is not a provider-enforced runtime limit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qc26"))
import qc26a_compile as qc

def build(block_id, out, ordering_seed=None):
    proposal = HERE / "ramsey_campaign.json"
    campaign = json.loads(proposal.read_text())
    block = next(b for b in campaign["blocks"] if b["id"] == block_id)
    rows = qc.ordered_rows(block["rows"], ordering_seed)
    if len(rows) != 66 or len({r["id"] for r in rows}) != 66 or any(r["shots"] != 384 for r in rows):
        raise ValueError("unexpected campaign shape")
    if block["cap_seconds"] != 15 or block["backend"] != "ibm_kingston":
        raise ValueError("unexpected cap/backend")
    _, backend = qc.backend_connection()
    snapshot = qc.calibration_snapshot(backend, source="pgc_qpu01_readonly")
    if snapshot["two_qubit_gate"] != "cz":
        raise ValueError("requires native CZ")
    out.mkdir(parents=True, exist_ok=False)
    (out / "circuits").mkdir()
    qc.write_json(out / "calibration_build.json", snapshot)
    compiled, isas = [], []
    for row in rows:
        logical = qc.logical_circuit(row)
        isa = qc.transpile_on_path(backend, logical, row["path"], optimization_level=0)
        mapping = qc.audit_mapping(isa, row["path"], len(row["path"]))
        qc.assert_counts(row, isa)
        qasm = qc.qasm3_text(isa).encode()
        (out / "circuits" / (row["id"] + ".qasm")).write_bytes(qasm)
        compiled.append({**row, "mapping": mapping, "duration_s": qc.duration(isa, backend.target),
                         "gate_counts": {k: int(v) for k, v in isa.count_ops().items()},
                         "qasm_sha256": hashlib.sha256(qasm).hexdigest()})
        isas.append(isa)
    qpy = qc.qpy_bytes(isas)
    (out / "compiled.qpy").write_bytes(qpy)
    est = qc.estimate(block, compiled)
    meta = {"version": campaign["version"], "block": block_id, "backend": block["backend"],
            "compiled_at": qc.now(), "proposal_sha256": qc.hash_bytes(proposal.read_bytes()),
            "calibration_sha256": qc.hash_bytes((out / "calibration_build.json").read_bytes()),
            "qpy_sha256": qc.hash_bytes(qpy), "circuits": compiled,
            "shots": 384, "estimated_qpu_seconds": est, "cap_seconds": 15,
            "status": "COMPILE ONLY; no submission or authorization"}
    if ordering_seed is not None: meta["ordering_seed"] = ordering_seed
    meta["plan_sha256"] = qc.plan_hash(meta)
    qc.write_json(out / "compiled.json", meta)
    if est > 15:
        raise RuntimeError("estimate exceeds block cap; do not submit")
    print(json.dumps({k: meta[k] for k in ("block", "plan_sha256", "estimated_qpu_seconds", "cap_seconds", "status")}, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block", choices=("q23", "q5"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ordering-seed", type=int)
    a = p.parse_args()
    build(a.block, a.out, a.ordering_seed)
