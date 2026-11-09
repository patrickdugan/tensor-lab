"""QC-06: allocate local purification (Kraus-environment) ranks under a hard
layer-boundary storage budget.

Only singular spectra of the current locally purified state are used to choose
ranks. No exact density matrix, future gates, or teacher reference is accessed.

The bound is on *retained* complex tensor entries at the layer boundary, not
on temporary workspace/peak RSS. Interim channels can exceed the budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from purified_mpo import PurifiedChain, Truncation, select_rank, stable_svd


@dataclass(frozen=True)
class Allocation:
    target_ranks: tuple[int, ...]
    retained_entries: int
    budget_entries: int
    conservative_allocated_entries: int
    probe_svds: int
    compression_svds: int
    predicted_discarded_weight: float
    post_compression_loss: float
    seconds: float


def greedy_rank_allocation(spectra: list[np.ndarray], weights: list[int],
                           budget: int, max_site_env: int,
                           min_site_env: int = 1) -> tuple[int, ...]:
    """Take the next singular component with largest (s_j**2)/entry.

    Weights are upper bounds on entry cost for each retained component.
    Greedy is a heuristic for the integer multiple-choice knapsack problem;
    it is not claimed globally optimal. The at-least-one component policy
    ensures a nonzero purification at each site.
    """
    if len(spectra) == 0 or len(spectra) != len(weights):
        raise ValueError("one nonempty spectrum and weight are required per site")
    if max_site_env < 1 or min_site_env < 1 or min_site_env > max_site_env:
        raise ValueError("invalid allowed rank range")
    if not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("budget must be a positive integer")
    if any(w < 1 or not isinstance(w, (int, np.integer)) for w in weights):
        raise ValueError("weights must be positive integers")
    if any(len(s) == 0 or not np.all(np.isfinite(s)) or np.any(s < 0)
           for s in spectra):
        raise ValueError("singular spectra must be nonempty, finite, nonnegative")
    cap = [min(select_rank(s, None, 0), max_site_env) for s in spectra]
    start = [min(c, min_site_env) for c in cap]
    used = sum(int(w) * int(r) for w, r in zip(weights, start))
    if used > budget:
        raise ValueError(f"infeasible minimal retained-entry budget {budget} < {used}")
    ranks = start.copy()
    while True:
        choices = [(float(s[r]**2) / w, i)
                   for i, (s, w, r, c) in enumerate(zip(spectra, weights, ranks, cap))
                   if r < c and used + w <= budget]
        if not choices:
            break
        _, i = max(choices, key=lambda t: (t[0], -t[1]))
        used += int(weights[i])
        ranks[i] += 1
    return tuple(ranks)


def scan_ancilla_spectra(chain: PurifiedChain) -> tuple[list[np.ndarray], list[int]]:
    """Use one right-to-left and one left-to-right QR canonical sweep.

    At each site, both physical sides are orthonormal, so the local SVD
    gives the ancilla Schmidt spectrum for the *untruncated* input chain.
    The QR operations are exact gauge changes on the purification.
    """
    chain._right_qr(1)
    spectra: list[np.ndarray] = []
    weights: list[int] = []
    for i in range(chain.n):
        a = chain.cores[i]
        left, phys, env, right = a.shape
        if phys != 2:
            raise ValueError("qubit physical dimension must be two")
        mat = a.transpose(0, 1, 3, 2).reshape(left * phys * right, env)
        chain.largest_working_entries = max(chain.largest_working_entries, mat.size)
        s = np.linalg.svd(mat, full_matrices=False, compute_uv=False)
        spectra.append(s)
        weights.append(left * phys * right)
        if i + 1 < chain.n:
            q, r = np.linalg.qr(a.reshape(left * phys * env, right), mode="reduced")
            chain.cores[i] = q.reshape(left, phys, env, q.shape[1])
            chain.cores[i + 1] = np.tensordot(r, chain.cores[i + 1], axes=(1, 0))
    return spectra, weights


class AdaptivePurifiedChain(PurifiedChain):
    def compress_layer_budgeted(self, budget_entries: int, *, max_site_env: int = 10,
                                min_site_env: int = 1) -> Allocation:
        """Compress the layer within a strict retained-entry budget.

        Since gauge QR and local environment truncation only decrease the
        neighboring bond dimensions, precompression weights are conservative.
        A ValueError reports genuinely infeasible budgets. The caller should
        treat that as an experimental failure rather than silently overrunning.
        """
        tic = perf_counter()
        spectra, weights = scan_ancilla_spectra(self)
        ranks = greedy_rank_allocation(spectra, weights, budget_entries,
                                       max_site_env, min_site_env)
        predicted_loss = sum(float(np.sum(s[r:]**2)) for s, r in zip(spectra, ranks))
        cost = sum(w * r for w, r in zip(weights, ranks))

        self._right_qr(1)
        actual_loss = 0.0
        for i, target in enumerate(ranks):
            a = self.cores[i]
            left, phys, env, right = a.shape
            mat = a.transpose(0, 1, 3, 2).reshape(left * phys * right, env)
            self.largest_working_entries = max(self.largest_working_entries, mat.size)
            u, s, _ = stable_svd(mat)
            kept = min(int(target), select_rank(s, None, self.local_tol))
            loss = float(np.sum(s[kept:] ** 2))
            actual_loss += loss
            self.truncations.append(Truncation("environment", i, kept, len(s), loss))
            self.cores[i] = (u[:, :kept] * s[:kept]).reshape(
                left, phys, right, kept).transpose(0, 1, 3, 2)
            if i + 1 < self.n:
                updated = self.cores[i]
                q, r = np.linalg.qr(updated.reshape(left * phys * kept, right), mode="reduced")
                self.cores[i] = q.reshape(left, phys, kept, q.shape[1])
                self.cores[i + 1] = np.tensordot(r, self.cores[i + 1], axes=(1, 0))
        retained = self.dimensions()["purified_entries"]
        if retained > budget_entries:
            raise AssertionError(f"retained entries {retained} exceeded budget {budget_entries}")
        return Allocation(ranks, retained, int(budget_entries), int(cost),
                          self.n, self.n, predicted_loss, actual_loss,
                          perf_counter() - tic)
