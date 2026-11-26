"""Read-only IBM preflight: account reachability, plan usage, operational backends, Kingston calibration summary.
Submits nothing and consumes no QPU time. Prints and writes no credentials.

  python preflight.py --out results_local/preflight.json
"""
import argparse, datetime as dt, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from qc23_access import connect, backend_summary  # noqa: E402
from qc23_noise import calibration_snapshot        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    service, err = connect(None)
    if service is None:
        print(json.dumps({"account": "unavailable", "error": err}))
        sys.exit(2)
    rec = {"checked_at": dt.datetime.now(dt.timezone.utc).isoformat(), "backends": []}
    try:
        u = service.usage()
        rec["usage"] = {k: u.get(k) for k in ("usage_consumed_seconds", "usage_limit_seconds", "usage_remaining_seconds",
                                              "usage_period", "usage_limit_reached")}
    except Exception as exc:  # noqa: BLE001
        rec["usage"] = {"error": f"{type(exc).__name__}: {exc}"}
    for b in service.backends(simulator=False):
        try:
            s = backend_summary(b)
            rec["backends"].append({k: s.get(k) for k in ("name", "num_qubits", "operational", "pending_jobs", "two_qubit_gate",
                                                        "median_two_qubit_error", "median_readout_error", "processor_type")})
        except Exception as exc:  # noqa: BLE001
            rec["backends"].append({"name": getattr(b, "name", "?"), "error": f"{type(exc).__name__}: {exc}"})
    kb = service.backend("ibm_kingston")
    snap = calibration_snapshot(kb, source="pgc_preflight")
    q = {str(k): v for k, v in snap["qubits"].items()} if isinstance(snap["qubits"], dict) else snap["qubits"]
    rec["kingston_calibration_timestamp"] = snap["timestamp"]
    rec["kingston_two_qubit_gate"] = snap["two_qubit_gate"]
    rec["kingston_edges_of_interest"] = {e: snap["edges"].get(e) if isinstance(snap["edges"], dict) else None
                                        for e in ("16_23", "22_23", "4_5", "5_6")} if isinstance(snap["edges"], dict) else "edges not a dict"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(json.dumps(rec, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
