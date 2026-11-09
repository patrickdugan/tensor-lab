"""Small dense-reference and tensor-train benchmark for noisy quantum circuits.

The local quantum channel is exact; the MPO is a Frobenius-norm TT-SVD
approximation of its density matrix, with physical legs (ket_i, bra_i).
No speed-up claim is implied: this reference constructs the dense state first.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)


def kraus(name: str, p: float) -> tuple[np.ndarray, ...]:
    """Noise conventions: depolarize rho -> (1-p)rho+p*I/2, 0<=p<=1."""
    if not 0 <= p <= 1:
        raise ValueError("noise strength p must be in [0, 1]")
    if name == "depolarizing":
        return (np.sqrt(1 - 3 * p / 4) * I2,
                np.sqrt(p / 4) * X,
                np.sqrt(p / 4) * Y,
                np.sqrt(p / 4) * Z)
    if name == "dephasing":
        return (np.sqrt(1 - p) * I2, np.sqrt(p) * Z)
    if name == "amplitude_damping":
        return (np.diag([1, np.sqrt(1 - p)]).astype(complex),
                np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex))
    raise ValueError(f"unknown channel: {name}")


def choi_rank(operators: tuple[np.ndarray, ...], rtol: float = 1e-10) -> int:
    """Rank of Choi = rank of stacked Kraus vectors (minimal environment)."""
    vectors = np.column_stack([k.reshape(-1, order="F") for k in operators])
    values = np.linalg.svd(vectors, compute_uv=False)
    return int(np.count_nonzero(values > rtol * values[0]))


def apply_operator(rho: np.ndarray, op: np.ndarray,
                   targets: tuple[int, ...], n: int) -> np.ndarray:
    """Apply op*rho*op^dag on qubits in targets, numbered left to right."""
    if len(set(targets)) != len(targets) or any(i < 0 or i >= n for i in targets):
        raise ValueError("invalid qubit targets")
    k = len(targets)
    if op.shape != (2**k, 2**k):
        raise ValueError("operator dimension does not match target count")
    other = [i for i in range(n) if i not in targets]
    perm = list(targets) + other + [n + i for i in targets] + [n + i for i in other]
    reshaped = rho.reshape((2,) * (2 * n)).transpose(perm)
    reshaped = reshaped.reshape(2**k, 2**(n-k), 2**k, 2**(n-k))
    updated = np.einsum("ai,irjs,bj->arbs", op, reshaped, op.conj(), optimize=True)
    return updated.reshape((2,) * (2 * n)).transpose(np.argsort(perm)).reshape(2**n, 2**n)


def apply_channel(rho: np.ndarray, ks: tuple[np.ndarray, ...],
                  target: int, n: int) -> np.ndarray:
    return sum((apply_operator(rho, k, (target,), n) for k in ks),
               np.zeros_like(rho))


def stinespring_trace(rho: np.ndarray, ks: tuple[np.ndarray, ...]) -> np.ndarray:
    """Construct isometry V=sum_a |a> tensor K_a, trace environment."""
    d = rho.shape[0]
    v = np.vstack(ks)
    joint = (v @ rho @ v.conj().T).reshape(len(ks), d, len(ks), d)
    return np.einsum("aiaj->ij", joint)


def interleave(rho: np.ndarray, n: int) -> np.ndarray:
    perm = [index for i in range(n) for index in (i, n + i)]
    return rho.reshape((2,) * (2 * n)).transpose(perm).reshape((4,) * n)


def uninterleave(local: np.ndarray, n: int) -> np.ndarray:
    perm = [index for i in range(n) for index in (i, n + i)]
    return local.reshape((2,) * (2 * n)).transpose(np.argsort(perm)).reshape(2**n, 2**n)


def mpo_ttsvd(rho: np.ndarray, n: int, rel_tol: float) -> tuple[list[np.ndarray], np.ndarray]:
    """TT-SVD with per-cut tail budget, guaranteeing ||error||F <= tol*||rho||F.

    The guarantee follows from orthogonal TT-SVD residuals.  The returned MPO
    is not guaranteed to be positive semidefinite or exactly trace one.
    """
    if rel_tol < 0:
        raise ValueError("rel_tol must be nonnegative")
    arr = interleave(rho, n)
    if n == 1:
        core = arr.reshape(1, 4, 1)
        return [core], rho.copy()
    budget = (rel_tol * np.linalg.norm(rho))**2 / (n - 1)
    cores: list[np.ndarray] = []
    bond = 1
    for _ in range(n - 1):
        left = arr.reshape(bond * 4, -1)
        u, s, vh = np.linalg.svd(left, full_matrices=False)
        tail = np.r_[np.cumsum((s[::-1] ** 2))[::-1], 0.0]
        possible = np.flatnonzero(tail <= budget + 1e-28)
        kept = max(1, int(possible[0])) if len(possible) else len(s)
        cores.append(u[:, :kept].reshape(bond, 4, kept))
        arr = s[:kept, None] * vh[:kept, :]
        bond = kept
    cores.append(arr.reshape(bond, 4, 1))
    local = cores[0]
    for core in cores[1:]:
        local = np.tensordot(local, core, axes=(-1, 0))
    approx = uninterleave(local.reshape((4,) * n), n)
    return cores, approx


def haar_two_qubit(rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q, r = np.linalg.qr(z)
    diagonal = np.diag(r)
    return q * (diagonal / np.abs(diagonal))


def gate_layers(n: int, depth: int, seed: int) -> list[list[tuple[int, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    return [[(i, haar_two_qubit(rng)) for i in range(t % 2, n - 1, 2)]
            for t in range(depth)]


def metrics(rho: np.ndarray, n: int, rel_tol: float) -> dict[str, float | int]:
    cores, approx = mpo_ttsvd(rho, n, rel_tol)
    denominator = np.linalg.norm(rho)
    fro_error = np.linalg.norm(rho - approx) / denominator
    diff = (rho - approx + (rho - approx).conj().T) / 2
    trace_norm_distance = 0.5 * float(np.abs(np.linalg.eigvalsh(diff)).sum())
    return {
        "trace_norm_distance": trace_norm_distance,
        "approx_trace_error": float(abs(np.trace(approx) - 1)),
        "max_bond": max(max(c.shape[0], c.shape[2]) for c in cores),
        "mpo_entries": sum(c.size for c in cores),
        "dense_entries": rho.size,
        "relative_fro_error": float(fro_error),
        "trace": float(np.trace(rho).real),
        "purity": float(np.trace(rho @ rho).real),
        "min_mpo_eigenvalue": float(np.linalg.eigvalsh((approx + approx.conj().T) / 2).min()),
    }


def run(n: int, depth: int, seeds: int, channels: list[str],
        strengths: list[float], rel_tol: float) -> list[dict]:
    results: list[dict] = []
    initial = np.zeros((2**n, 2**n), dtype=complex)
    initial[0, 0] = 1
    for seed in range(seeds):
        layers = gate_layers(n, depth, seed)
        for channel in channels:
            for p in strengths:
                rho = initial.copy()
                ks = kraus(channel, p)
                for t, layer in enumerate(layers, 1):
                    for start, gate in layer:
                        rho = apply_operator(rho, gate, (start, start + 1), n)
                    for target in range(n):
                        rho = apply_channel(rho, ks, target, n)
                    results.append({"n": n, "depth": t, "seed": seed,
                                    "channel": channel, "p": p,
                                    "local_choi_rank": choi_rank(ks),
                                    "relative_tol": rel_tol,
                                    **metrics(rho, n, rel_tol)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--strengths", default="0,0.05,0.2,0.5,0.9")
    parser.add_argument("--channels", default="depolarizing,dephasing,amplitude_damping")
    parser.add_argument("--relative-tol", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("channel_mpo_results.csv"))
    args = parser.parse_args()
    if not 2 <= args.qubits <= 10:
        parser.error("reference implementation supports 2 <= qubits <= 10")
    if args.depth < 1 or args.seeds < 1:
        parser.error("depth and seeds must be positive")
    rows = run(args.qubits, args.depth, args.seeds,
               args.channels.split(","), [float(x) for x in args.strengths.split(",")],
               args.relative_tol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    for ch in args.channels.split(","):
        for p in [float(x) for x in args.strengths.split(",")]:
            rows_at_last_depth = [r for r in rows if r["channel"] == ch and
                                  r["p"] == p and r["depth"] == args.depth]
            print(f"{ch:18} p={p:.2f}  mean max bond="
                  f"{np.mean([r['max_bond'] for r in rows_at_last_depth]):5.1f}  "
                  f"mean entries={np.mean([r['mpo_entries'] for r in rows_at_last_depth]):7.1f}  "
                  f"mean relative error={np.mean([r['relative_fro_error'] for r in rows_at_last_depth]):.4g}")


if __name__ == "__main__":
    main()
