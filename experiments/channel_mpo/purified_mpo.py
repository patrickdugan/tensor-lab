"""QC-02: direct, positivity-preserving locally purified tensor-network simulator.

Each site stores A[left_bond, physical_qubit, local_environment, right_bond]
with rho = Tr_env |A><A|. Operations act on these tensors, never on a
2**n x 2**n density matrix. Only the optional validation reconstructs rho.

The max-bond/max-environment approximations preserve positivity. A global
norm contraction followed by rescaling preserves trace 1. Truncation losses
refer to the immediately preceding *purification*, not an end-to-end bound.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from channel_mpo import Z, apply_channel, apply_operator, gate_layers, kraus


@dataclass
class Truncation:
    kind: str
    site: int
    kept: int
    original_rank: int
    discarded_weight: float


def select_rank(singular_values: np.ndarray, limit: int | None,
                rel_tol: float) -> int:
    """Keep enough local singular values for a per-operation tail budget.

    Machine-zero singular values are discarded at a relative threshold of
    1e-13, avoiding inflated tensor dimensions after a rank-deficient channel.
    The limit is an independent, potentially more aggressive hard cap.
    """
    if rel_tol < 0 or (limit is not None and limit < 1):
        raise ValueError("rel_tol must be >= 0 and finite cap >= 1")
    if len(singular_values) == 0:
        raise ValueError("empty spectrum")
    cutoff = 1e-13 * singular_values[0]
    numerical_rank = max(1, int(np.count_nonzero(singular_values > cutoff)))
    tail = np.r_[np.cumsum((singular_values[::-1] ** 2))[::-1], 0.0]
    budget = float(rel_tol**2 * np.sum(singular_values**2))
    eligible = np.flatnonzero(tail <= budget)
    desired = max(1, int(eligible[0])) if len(eligible) else len(singular_values)
    return min(numerical_rank, desired, limit if limit is not None else len(singular_values))


def stable_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD with a QR-preconditioned retry for occasional LAPACK failures.

    QR preserves singular values and can bypass divide-and-conquer nonconvergence
    on certain near-degenerate small complex matrices. No SciPy dependency.
    """
    if not np.isfinite(matrix).all():
        raise FloatingPointError("nonfinite tensor encountered in SVD")
    try:
        return np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        if matrix.shape[0] >= matrix.shape[1]:
            q, r = np.linalg.qr(matrix, mode="reduced")
            u, s, vh = np.linalg.svd(r, full_matrices=False)
            return q @ u, s, vh
        q, r = np.linalg.qr(matrix.conj().T, mode="reduced")
        u, s, vh = np.linalg.svd(r.conj().T, full_matrices=False)
        return u, s, vh @ q.conj().T


class PurifiedChain:
    def __init__(self, n: int, max_bond: int | None = None,
                 max_env: int | None = None, local_tol: float = 0.0):
        if n < 2:
            raise ValueError("n must be >= 2")
        if local_tol < 0 or not np.isfinite(local_tol):
            raise ValueError("local_tol must be a nonnegative finite number")
        if max_bond is not None and max_bond < 1:
            raise ValueError("max_bond must be >= 1")
        if max_env is not None and max_env < 1:
            raise ValueError("max_env must be >= 1")
        self.cores = [np.array([[[[1.0 + 0j]], [[0.0 + 0j]]]])
                      for _ in range(n)]  # (left=1, phys=2, env=1, right=1)
        self.max_bond, self.max_env = max_bond, max_env
        self.local_tol = local_tol
        self.truncations: list[Truncation] = []
        self.largest_working_entries = 0

    @property
    def n(self) -> int:
        return len(self.cores)

    def _left_qr(self, end: int) -> None:
        """Left-orthogonalize all sites < end, transporting R rightward."""
        for j in range(end):
            a = self.cores[j]
            left, _, env, right = a.shape
            q, r = np.linalg.qr(a.reshape(left * 2 * env, right), mode="reduced")
            self.cores[j] = q.reshape(left, 2, env, q.shape[1])
            self.cores[j + 1] = np.tensordot(r, self.cores[j + 1], axes=(1, 0))

    def _right_qr(self, start: int) -> None:
        """Right-orthogonalize all sites >= start, transporting L leftward."""
        for j in range(self.n - 1, start - 1, -1):
            a = self.cores[j]
            left, _, env, right = a.shape
            q, r = np.linalg.qr(a.reshape(left, 2 * env * right).T, mode="reduced")
            self.cores[j] = q.T.reshape(q.shape[1], 2, env, right)
            self.cores[j - 1] = np.tensordot(self.cores[j - 1], r.T, axes=(3, 0))

    def _canonicalize(self, first: int, last: int) -> None:
        self._left_qr(first)
        self._right_qr(last + 1)

    def gate(self, site: int, unitary: np.ndarray) -> None:
        if not 0 <= site < self.n - 1 or unitary.shape != (4, 4):
            raise ValueError("gate must act on a valid nearest-neighbor pair")
        self._canonicalize(site, site + 1)
        a, b = self.cores[site], self.cores[site + 1]
        left, _, env_a, _ = a.shape
        _, _, env_b, right = b.shape
        theta = np.tensordot(a, b, axes=(3, 0))  # (L,pa,ea,pb,eb,R)
        theta = np.einsum("xyuv,luavbr->lxaybr",
                          unitary.reshape(2, 2, 2, 2), theta, optimize=True)
        mat = theta.reshape(left * 2 * env_a, 2 * env_b * right)
        self.largest_working_entries = max(self.largest_working_entries, mat.size)
        u, s, vh = stable_svd(mat)
        kept = select_rank(s, self.max_bond, self.local_tol)
        loss = float(np.sum(s[kept:] ** 2))
        self.truncations.append(Truncation("bond", site, kept, len(s), loss))
        self.cores[site] = u[:, :kept].reshape(left, 2, env_a, kept)
        self.cores[site + 1] = (s[:kept, None] * vh[:kept]).reshape(
            kept, 2, env_b, right)

    def channel(self, site: int, operators: tuple[np.ndarray, ...]) -> None:
        if not 0 <= site < self.n:
            raise ValueError("site out of bounds")
        if any(k.shape != (2, 2) for k in operators):
            raise ValueError("channel operators must be 2x2")
        a = self.cores[site]
        # The Kraus-label register is appended to this site's local environment.
        enlarged = np.einsum("moi,liar->loamr",
                             np.asarray(operators), a, optimize=True)
        self.cores[site] = enlarged.reshape(a.shape[0], 2,
                                           a.shape[2] * len(operators), a.shape[3])
        self.largest_working_entries = max(self.largest_working_entries,
                                           self.cores[site].size)

    def compress_env(self, site: int) -> None:
        if not 0 <= site < self.n:
            raise ValueError("site out of bounds")
        self._canonicalize(site, site)
        a = self.cores[site]
        left, _, env, right = a.shape
        mat = a.transpose(0, 1, 3, 2).reshape(left * 2 * right, env)
        self.largest_working_entries = max(self.largest_working_entries, mat.size)
        u, s, vh = stable_svd(mat)
        kept = select_rank(s, self.max_env, self.local_tol)
        loss = float(np.sum(s[kept:] ** 2))
        self.truncations.append(Truncation("environment", site, kept, len(s), loss))
        self.cores[site] = (u[:, :kept] * s[:kept]).reshape(
            left, 2, right, kept).transpose(0, 1, 3, 2)

    def trace(self) -> float:
        env = np.ones((1, 1), dtype=complex)
        for a in self.cores:
            env = np.einsum("lm,lpar,mpas->rs", env, a, a.conj(), optimize=True)
        value = complex(env[0, 0])
        if abs(value.imag) > 1e-8 * max(1, abs(value.real)):
            raise ArithmeticError(f"nonreal purification norm: {value}")
        return float(value.real)

    def normalize(self) -> float:
        before = self.trace()
        if before <= 0 or not np.isfinite(before):
            raise ArithmeticError(f"invalid purification norm {before}")
        self.cores[0] /= np.sqrt(before)
        return before

    def z_expectations(self) -> list[float]:
        """Contract all local Z expectations in O(n) environment steps."""
        right = [None] * (self.n + 1)
        right[self.n] = np.ones((1, 1), complex)
        for j in range(self.n - 1, -1, -1):
            a = self.cores[j]
            right[j] = np.einsum("lpar,mpas,rs->lm", a, a.conj(),
                                  right[j + 1], optimize=True)
        left = np.ones((1, 1), complex)
        out = []
        for j, a in enumerate(self.cores):
            value = np.einsum("lm,lpar,pq,mqas,rs->", left, a, Z,
                              a.conj(), right[j + 1], optimize=True)
            out.append(float(np.real(value)))
            left = np.einsum("lm,lpar,mpas->rs", left, a, a.conj(), optimize=True)
        return out

    def dense_density(self, *, max_n: int = 10) -> np.ndarray:
        """Small-system oracle; balanced contraction avoids large intermediates.

        This constructs 4**n entries and is excluded from simulator timings.
        """
        if self.n > max_n:
            raise ValueError("explicit density is limited to small-system validation")
        blocks = []
        for a in self.cores:
            dl, _, _, dr = a.shape
            block = np.einsum("liar,mjas->lmijrs", a, a.conj(), optimize=True)
            blocks.append(block.reshape(dl * dl, 2, 2, dr * dr))
        mid = self.n // 2
        left = blocks[0][0]
        for j in range(1, mid):
            left = np.tensordot(left, blocks[j], axes=(-1, 0))
        right = blocks[-1][..., 0]
        for j in range(self.n - 2, mid - 1, -1):
            right = np.tensordot(blocks[j], right, axes=(-1, 0))
        operator_tensor = (left.reshape(4**mid, -1) @
                           right.reshape(right.shape[0], 4**(self.n - mid)))
        perm = list(range(0, 2 * self.n, 2)) + list(range(1, 2 * self.n, 2))
        return operator_tensor.reshape((2,) * (2 * self.n)).transpose(perm).reshape(
            2**self.n, 2**self.n)

    def dimensions(self) -> dict[str, int]:
        return {
            "max_bond": max(max(a.shape[0], a.shape[3]) for a in self.cores),
            "max_env": max(a.shape[2] for a in self.cores),
            "purified_entries": sum(a.size for a in self.cores),
            "dense_entries": 4**self.n,
            "largest_working_entries": self.largest_working_entries,
        }


def run_one(n: int, depth: int, seed: int, channel_name: str, p: float,
            bond: int | None, env: int | None, local_tol: float,
            validate: bool = False, min_retention: float = 0.95) -> list[dict[str, object]]:
    if depth < 1:
        raise ValueError("depth must be positive")
    if not 0 <= min_retention <= 1:
        raise ValueError("min_retention must be in [0, 1]")
    gates = gate_layers(n, depth, seed)
    ks = kraus(channel_name, p)
    chain = PurifiedChain(n, bond, env, local_tol)
    dense = None
    if validate and n <= 10:
        dense = np.zeros((2**n, 2**n), complex)
        dense[0, 0] = 1.0
    out: list[dict[str, object]] = []
    min_observed_retention = 1.0
    for t, layer in enumerate(gates, 1):
        begin = perf_counter()
        for i, u in layer:
            chain.gate(i, u)
        for i in range(n):
            chain.channel(i, ks)
        for i in range(n):
            chain.compress_env(i)
        trace_before = chain.normalize()
        min_observed_retention = min(min_observed_retention, trace_before)
        elapsed = perf_counter() - begin
        values = chain.z_expectations()
        record: dict[str, object] = {
            "n": n, "depth": t, "seed": seed, "channel": channel_name,
            "p": p, "bond_cap": bond if bond is not None else "none",
            "env_cap": env if env is not None else "none", "local_tol": local_tol,
            "runtime_layer_seconds": round(elapsed, 6),
            "trace_before_normalize": trace_before,
            "min_layer_retention_so_far": min_observed_retention,
            "retention_threshold": min_retention,
            "retention_gate_pass": min_observed_retention >= min_retention - 1e-12,
            "sum_discarded_weight": float(sum(x.discarded_weight for x in chain.truncations)),
            "num_truncations": sum(x.discarded_weight > 1e-20 for x in chain.truncations),
            "z0": values[0], "z_middle": values[n // 2],
            **chain.dimensions(),
        }
        if dense is not None:
            for i, u in layer:
                dense = apply_operator(dense, u, (i, i + 1), n)
            for i in range(n):
                dense = apply_channel(dense, ks, i, n)
            approx = chain.dense_density()
            diff = (dense - approx + (dense - approx).conj().T) / 2
            record["trace_distance"] = 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(diff))))
            diagonal = np.real(np.diag(dense))
            half = 2**(n - 1)
            reference_z0 = float(np.sum(diagonal[:half]) - np.sum(diagonal[half:]))
            record["z0_abs_error"] = float(abs(reference_z0 - values[0]))
            record["min_eigenvalue"] = float(np.min(np.linalg.eigvalsh(
                (approx + approx.conj().T) / 2)))
        else:
            record.update(trace_distance="", z0_abs_error="", min_eigenvalue="")
        out.append(record)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", default="4,6,10,16,20",
                        help="comma-separated qubit sizes")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--channels", default="depolarizing,dephasing,amplitude_damping")
    parser.add_argument("--strengths", default="0.2,0.5,0.9")
    parser.add_argument("--max-bond", type=int, default=16)
    parser.add_argument("--max-env", type=int, default=4)
    parser.add_argument("--local-tol", type=float, default=1e-8)
    parser.add_argument("--validate-small", action="store_true")
    parser.add_argument("--min-retention", type=float, default=0.95,
                        help="flag low retained purification norm; only a diagnostic")
    parser.add_argument("--output", type=Path, default=Path("purified_results.csv"))
    args = parser.parse_args()
    ns = [int(x) for x in args.qubits.split(",")]
    if any(n < 2 for n in ns) or args.seeds < 1:
        parser.error("require n>=2 and seeds>=1")
    records: list[dict[str, object]] = []
    for n in ns:
        for seed in range(args.seeds):
            for channel in args.channels.split(","):
                for p in map(float, args.strengths.split(",")):
                    result = run_one(n, args.depth, seed, channel, p,
                                     args.max_bond, args.max_env, args.local_tol,
                                     validate=args.validate_small,
                                     min_retention=args.min_retention)
                    records.extend(result)
                    last = result[-1]
                    print(f"n={n:2} {channel:20} p={p:4.2f} "
                          f"bond={last['max_bond']:2} env={last['max_env']:2} "
                          f"entries={last['purified_entries']:6} "
                          f"last_layer={last['runtime_layer_seconds']:.3f}s "
                          f"distance={last['trace_distance']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"saved {len(records)} observations to {args.output}")


if __name__ == "__main__":
    main()
