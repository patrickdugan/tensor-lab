"""QC-02: small-system fidelity/memory frontier of locally purified truncation.

All rows are compared with the exact density-matrix oracle at the final layer.
The grid is deliberately small enough to run on CPU with no external SDK.
"""
import argparse
import csv
from pathlib import Path

from purified_mpo import run_one


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channel", default="depolarizing")
    ap.add_argument("--p", type=float, default=.5)
    ap.add_argument("--bonds", default="8,16")
    ap.add_argument("--environments", default="2,4,8,16")
    ap.add_argument("--local-tol", type=float, default=1e-9)
    ap.add_argument("--output", type=Path, default=Path("frontier.csv"))
    args = ap.parse_args()
    if not 2 <= args.qubits <= 10:
        ap.error("frontier requires a dense oracle on 2-10 qubits")
    records = []
    for bond in (int(b) for b in args.bonds.split(",")):
        for env in (int(e) for e in args.environments.split(",")):
            rows = run_one(args.qubits, args.depth, args.seed, args.channel,
                           args.p, bond, env, args.local_tol, validate=True)
            last = rows[-1]
            records.append({
                "n": args.qubits, "depth": args.depth, "seed": args.seed,
                "channel": args.channel, "p": args.p, "bond_cap": bond,
                "env_cap": env, "max_bond_observed": last["max_bond"],
                "max_env_observed": last["max_env"],
                "purified_entries": last["purified_entries"],
                "dense_entries": last["dense_entries"],
                "runtime_total_seconds": sum(r["runtime_layer_seconds"] for r in rows),
                "trace_distance": last["trace_distance"],
                "min_layer_retention": last["min_layer_retention_so_far"],
                "z0_abs_error": last["z0_abs_error"],
                "min_eigenvalue": last["min_eigenvalue"],
            })
            print(f"bond={bond:2} env={env:2} "
                  f"entries={last['purified_entries']:6} "
                  f"distance={last['trace_distance']:.6f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
