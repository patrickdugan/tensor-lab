"""QC-02 reproducible wall-time comparison with the QC-01 dense simulator.

Small n only: the entire dense evolution is run independently, while the
purified simulator builds and updates local tensors. Reported times exclude
accuracy reconstruction and CSV writing. Accuracy is final trace distance.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from channel_mpo import apply_channel, apply_operator, gate_layers, kraus
from purified_mpo import run_one


def dense_evolve(n: int, depth: int, seed: int,
                 channel: str, p: float) -> tuple[np.ndarray, float]:
    ks = kraus(channel, p)
    gates = gate_layers(n, depth, seed)
    rho = np.zeros((2**n, 2**n), complex)
    rho[0, 0] = 1
    t0 = perf_counter()
    for layer in gates:
        for i, u in layer:
            rho = apply_operator(rho, u, (i, i + 1), n)
        for i in range(n):
            rho = apply_channel(rho, ks, i, n)
    return rho, perf_counter() - t0


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--qubits", default="6,8,10")
    cli.add_argument("--depth", type=int, default=6)
    cli.add_argument("--seeds", type=int, default=1)
    cli.add_argument("--channel", default="depolarizing")
    cli.add_argument("--strengths", default="0.2,0.5")
    cli.add_argument("--max-bond", type=int, default=12)
    cli.add_argument("--max-env", type=int, default=8)
    cli.add_argument("--local-tol", type=float, default=1e-9)
    cli.add_argument("--output", type=Path, default=Path("head_to_head.csv"))
    args = cli.parse_args()
    qubits = [int(x) for x in args.qubits.split(",")]
    if any(n < 2 or n > 10 for n in qubits):
        cli.error("dense validation is intentionally limited to 2-10 qubits")
    out = []
    for n in qubits:
        for seed in range(args.seeds):
            for p in map(float, args.strengths.split(",")):
                _, dense_seconds = dense_evolve(n, args.depth, seed, args.channel, p)
                rows = run_one(n, args.depth, seed, args.channel, p,
                               args.max_bond, args.max_env, args.local_tol,
                               validate=True)
                result = rows[-1]
                # Timing comes from the direct network update, before dense validation.
                sim_seconds = sum(r["runtime_layer_seconds"] for r in rows)
                row = {
                    "n": n, "depth": args.depth, "seed": seed,
                    "channel": args.channel, "p": p,
                    "bond_cap": args.max_bond, "env_cap": args.max_env,
                    "dense_seconds": dense_seconds, "purified_seconds": sim_seconds,
                    "dense_over_purified_time": dense_seconds / sim_seconds,
                    "dense_complex_entries": 4**n,
                    "purified_complex_entries": result["purified_entries"],
                    "trace_distance": result["trace_distance"],
                    "min_layer_retention": result["min_layer_retention_so_far"],
                    "retention_gate_pass": result["retention_gate_pass"],
                    "z0_abs_error": result["z0_abs_error"],
                    "min_density_eigenvalue": result["min_eigenvalue"],
                }
                out.append(row)
                print(f"n={n:2} p={p:.2f} dense={dense_seconds:.3f}s "
                      f"purified={sim_seconds:.3f}s "
                      f"trace_distance={result['trace_distance']:.5f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
