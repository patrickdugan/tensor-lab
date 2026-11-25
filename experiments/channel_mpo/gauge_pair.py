"""QC-03: two-site ancillary unitary search plus joint local-env compression.

Each tensor A[left, qubit, ancilla, right] purifies a positive operator. A
unitary on two adjacent ancillary spaces leaves the physical state invariant
until a subsequent truncation. The greedy objective is the SUM of the local
ancilla singular-value tails outside the retained rank. This is a local
surrogate, not a guaranteed optimizer of physical trace distance.

Only NumPy and the QC-02 modules are required. This module leaves QC-02 intact.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from purified_mpo import PurifiedChain, Truncation, select_rank, stable_svd
from channel_mpo import apply_channel, apply_operator, gate_layers, kraus


def env_tail(theta: np.ndarray, axis: int, rank_cap: int | None) -> float:
    """Squared Frobenius tail of matricization at an ancillary index."""
    dim = theta.shape[axis]
    if rank_cap is None or rank_cap >= dim:
        return 0.0
    mat = np.moveaxis(theta, axis, 0).reshape(dim, -1)
    gram = mat @ mat.conj().T
    eigenvalues = np.linalg.eigvalsh(gram).real
    return max(0.0, float(eigenvalues[:dim - rank_cap].sum()))


def env_objective(theta: np.ndarray, rank_cap: int | None) -> float:
    return env_tail(theta, 2, rank_cap) + env_tail(theta, 4, rank_cap)


def mix_two_ancilla_states(theta: np.ndarray, a: int, b: int,
                           angle: float, phase: complex) -> np.ndarray:
    """Rotate pair ancilla labels a,b by a 2x2 unitary; rho stays unchanged."""
    l, pa, ea, pb, eb, r = theta.shape
    assert pa == pb == 2
    if not (0 <= a < ea*eb and 0 <= b < ea*eb and a != b):
        raise ValueError("two distinct valid ancillary basis indices required")
    if not np.isclose(abs(phase), 1, atol=1e-12):
        raise ValueError("complex phase must have unit modulus")
    mat = theta.transpose(0, 1, 3, 5, 2, 4).reshape(-1, ea*eb).copy()
    x, y = mat[:, a].copy(), mat[:, b].copy()
    c, s = np.cos(angle), np.sin(angle)
    mat[:, a] = c*x + s*phase*y
    mat[:, b] = -s*np.conjugate(phase)*x + c*y
    return mat.reshape(l, 2, 2, r, ea, eb).transpose(0, 1, 4, 2, 5, 3)


@dataclass
class GaugeStats:
    old_loss: float
    new_loss: float
    candidate_evaluations: int
    accepted: int


def optimize_ancillas(theta: np.ndarray, rank_cap: int | None,
                      rng: np.random.Generator,
                      proposals: int = 8, sweeps: int = 1) -> tuple[np.ndarray, GaugeStats]:
    """Greedy two-level ancillary disentanglers, with strict objective check.

    Every proposed rotation is unitary, so before truncation the physical
    density matrix is unchanged. Scoring assumes the pair is in mixed-canonical
    form; that assumption is established by PairPurifiedChain._canonicalize.
    """
    if proposals < 0 or sweeps < 0:
        raise ValueError("proposals and sweeps must be nonnegative")
    ea, eb = theta.shape[2], theta.shape[4]
    loss = env_objective(theta, rank_cap)
    baseline = loss
    if not proposals or not sweeps or (rank_cap is None) or ea <= 1 or eb <= 1:
        return theta, GaugeStats(loss, loss, 0, 0)
    candidate_pairs = [(a*eb+b, c*eb+d)
                       for a in range(ea) for b in range(eb)
                       for c in range(a+1, ea) for d in range(eb)
                       if b != d]
    if not candidate_pairs:
        return theta, GaugeStats(loss, loss, 0, 0)
    angles = [np.pi/8, -np.pi/8, np.pi/4, -np.pi/4]
    phases = [1.0+0j, 1.0j]
    evaluated, accepted = 0, 0
    for _ in range(sweeps):
        top_theta, top_loss = None, loss
        indexes = rng.choice(len(candidate_pairs), size=min(proposals, len(candidate_pairs)),
                             replace=False)
        for ix in indexes:
            a, b = candidate_pairs[int(ix)]
            for angle in angles:
                for phase in phases:
                    proposed = mix_two_ancilla_states(theta, a, b, angle, phase)
                    score = env_objective(proposed, rank_cap)
                    evaluated += 1
                    if score < top_loss - max(1e-13, 1e-10*baseline):
                        top_theta, top_loss = proposed, score
        if top_theta is None:
            break
        theta, loss = top_theta, top_loss
        accepted += 1
    return theta, GaugeStats(baseline, loss, evaluated, accepted)


def factor_permutation(theta: np.ndarray, new_kraus_dim: int,
                       first_factors: tuple[int, int]) -> np.ndarray:
    """Exact global ancilla basis permutation, including intersite factor swaps.

    After the new noise layer, each ancilla has the form (old_env, fresh_k).
    Regrouping the four factors is allowed if the two resulting local
    dimensions equal the originals, so the operation is a unitary on the
    *entire pair ancilla space*, never on the physical qubits.
    """
    l, _, ea, _, eb, r = theta.shape
    k = new_kraus_dim
    if ea % k or eb % k or len(first_factors) != 2 or len(set(first_factors)) != 2:
        raise ValueError("invalid ancillary factorization")
    dims = (ea//k, k, eb//k, k)
    order = list(first_factors) + [j for j in range(4) if j not in first_factors]
    if any(j not in range(4) for j in order) or dims[order[0]]*dims[order[1]] != ea:
        raise ValueError("permutation must preserve local Hilbert-space dimensions")
    block = theta.reshape(l,2,dims[0],dims[1],2,dims[2],dims[3],r)
    block = block.transpose(0,1,4,7,2,3,5,6)
    block = block.transpose(0,1,2,3,*[4+j for j in order])
    return block.reshape(l,2,2,r,ea,eb).transpose(0,1,4,2,5,3)


def projected_pair_norm(theta: np.ndarray, env_cap: int | None,
                        bond_cap: int | None, local_tol: float = 0.0) -> float:
    """Squared norm after sequential orthogonal Kraus and bond projections.

    The current two-site environment is canonically embedded, so this norm
    equals the globally retained purification weight for one pair update.
    It is still only a local fidelity surrogate for the final density matrix.
    """
    current = theta
    for axis in (2, 4):
        block = np.moveaxis(current, axis, 0)
        matrix = block.reshape(block.shape[0], -1)
        u, s, _ = stable_svd(matrix)
        kept = select_rank(s, env_cap, local_tol)
        current = np.moveaxis(
            (u[:, :kept].conj().T @ matrix).reshape((kept,)+block.shape[1:]), 0, axis)
    dl,_,ea,_,eb,dr = current.shape
    values = np.linalg.svd(current.reshape(dl*2*ea, 2*eb*dr),compute_uv=False)
    kept = select_rank(values, bond_cap, local_tol)
    return float(np.sum(values[:kept]**2))


def optimize_factor_permutations(theta: np.ndarray, env_cap: int | None,
                                 fresh_k: int, bond_cap: int | None = None,
                                 local_tol: float = 0.0,
                                 objective: str = 'retention') -> tuple[np.ndarray, GaugeStats]:
    """Search all valid factor groupings using a measured local loss.

    objective='retention': minimize total locally discarded purification norm
    after both Kraus projections and the intervening bond split. This avoids
    the missing-bond-cost problem of the original environment-tail surrogate.
    objective='env' preserves the exploratory QC-03 alpha control.
    """
    if objective not in ('retention','env'):
        raise ValueError('unknown ancilla permutation objective')
    ea, eb = theta.shape[2], theta.shape[4]
    norm_before = float(np.sum(np.abs(theta)**2))
    def score(candidate):
        if objective == 'env':
            return env_objective(candidate, env_cap)
        return max(0.0, norm_before-projected_pair_norm(
            candidate,env_cap,bond_cap,local_tol))
    old = score(theta)
    if env_cap is None or fresh_k <= 1 or ea % fresh_k or eb % fresh_k:
        return theta, GaugeStats(old, old, 0, 0)
    best, value, evaluated = theta, old, 0
    for subset in itertools.combinations(range(4), 2):
        try:
            trial = factor_permutation(theta, fresh_k, subset)
        except ValueError:
            continue
        proposed = score(trial)
        evaluated += 1
        if proposed < value - max(1e-13, 1e-10*old):
            value, best = proposed, trial
    return best, GaugeStats(old, value, evaluated, int(value < old - 1e-13))


class PairPurifiedChain(PurifiedChain):
    """Jointly compress adjacent local environments after a two-site gauge."""
    def __init__(self, n: int, max_bond: int | None = None,
                 max_env: int | None = None, local_tol: float = 0,
                 proposals: int = 0, sweeps: int = 1, seed: int = 0,
                 factor_swaps: bool = False, fresh_k: int = 1,
                 factor_objective: str = "retention",
                 fast_factor_search: bool = False):
        super().__init__(n, max_bond, max_env, local_tol)
        if proposals < 0 or sweeps < 0:
            raise ValueError("bad proposal/sweep count")
        self.rng = np.random.default_rng(seed)
        self.proposals, self.sweeps = proposals, sweeps
        self.factor_swaps, self.fresh_k = factor_swaps, fresh_k
        self.factor_objective = factor_objective
        self.fast_factor_search = fast_factor_search
        if fast_factor_search and (factor_objective != "retention" or
                                   proposals != 0 or not factor_swaps):
            raise ValueError("screened factor search needs retention objective, factor_swaps and proposals=0")
        self.gauge_pruned = 0
        self.gauge_exact_scores = 0
        self.gauge_evaluations = 0
        self.gauge_acceptances = 0
        self.gauge_surrogate_gain = 0.0
        self.gauge_seconds = 0.0

    def compress_pair(self, site: int) -> GaugeStats:
        if not 0 <= site < self.n - 1:
            raise ValueError("requires two adjacent sites")
        self._canonicalize(site, site+1)
        a, b = self.cores[site], self.cores[site+1]
        dl, _, ea, _ = a.shape
        _, _, eb, dr = b.shape
        theta = np.tensordot(a, b, axes=(3, 0))
        t0 = perf_counter()
        def local_score(candidate):
            if self.factor_objective == 'env':
                return env_objective(candidate, self.max_env)
            n0 = float(np.sum(np.abs(candidate)**2))
            return max(0., n0-projected_pair_norm(
                candidate, self.max_env, self.max_bond, self.local_tol))
        if self.fast_factor_search and self.factor_swaps and not self.proposals:
            from qc04_gauge import screened_factor_search
            theta, screening = screened_factor_search(
                theta, self.max_env, self.fresh_k, self.max_bond, self.local_tol)
            old_score, new_score = screening.old_loss, screening.new_loss
            factor_stats = GaugeStats(old_score, new_score,
                                      screening.evaluated, screening.accepted)
            self.gauge_pruned += screening.pruned
            self.gauge_exact_scores += screening.evaluated
            rotation_stats = GaugeStats(0., 0., 0, 0)
        else:
            old_score = local_score(theta)
            factor_stats = GaugeStats(old_score, old_score, 0, 0)
            if self.factor_swaps:
                theta, factor_stats = optimize_factor_permutations(
                    theta, self.max_env, self.fresh_k, self.max_bond,
                    self.local_tol, self.factor_objective)
            theta, rotation_stats = optimize_ancillas(theta, self.max_env, self.rng,
                                                     self.proposals, self.sweeps)
            new_score = local_score(theta)
        stats = GaugeStats(old_score, new_score,
                           factor_stats.candidate_evaluations + rotation_stats.candidate_evaluations,
                           factor_stats.accepted + rotation_stats.accepted)
        self.gauge_seconds += perf_counter()-t0
        self.gauge_evaluations += stats.candidate_evaluations
        self.gauge_acceptances += stats.accepted
        self.gauge_surrogate_gain += stats.old_loss-stats.new_loss

        # Exact isometric projection onto the local leading Kraus subspace.
        # QR-canonicalized surroundings make local SVD losses meaningful.
        for axis, label in ((2, site), (4, site+1)):
            moved = np.moveaxis(theta, axis, 0)
            mat = moved.reshape(moved.shape[0], -1)
            self.largest_working_entries = max(self.largest_working_entries, mat.size)
            u, s, _ = stable_svd(mat)
            kept = select_rank(s, self.max_env, self.local_tol)
            loss = float(np.sum(s[kept:]**2))
            self.truncations.append(Truncation("environment", label, kept, len(s), loss))
            new = u[:, :kept].conj().T @ mat
            theta = np.moveaxis(new.reshape((kept,)+moved.shape[1:]), 0, axis)
        dl, _, new_ea, _, new_eb, dr = theta.shape
        mat = theta.reshape(dl*2*new_ea, 2*new_eb*dr)
        self.largest_working_entries = max(self.largest_working_entries, mat.size)
        u, s, vh = stable_svd(mat)
        kept = select_rank(s, self.max_bond, self.local_tol)
        self.truncations.append(Truncation("bond", site, kept, len(s),
                                           float(np.sum(s[kept:]**2))))
        self.cores[site] = u[:, :kept].reshape(dl, 2, new_ea, kept)
        self.cores[site+1] = (s[:kept, None] * vh[:kept]).reshape(
            kept, 2, new_eb, dr)
        return stats

    def compress_layer(self) -> list[GaugeStats]:
        # Disjoint pairs, alternating pairings between layers is implemented
        # by run_pair for a controlled comparison of fixed/shifted partitions.
        stats = []
        for i in range(0, self.n-1, 2):
            stats.append(self.compress_pair(i))
        if self.n % 2:
            self.compress_env(self.n-1)
        return stats


def certified_truncation_bound(events: list[Truncation], initial_norm: float = 1.0
                               ) -> tuple[float, float]:
    """Rigorous (often vacuous) trace-distance upper bound for this layer.

    Every recorded SVD truncation is an orthogonal projection of the global
    purification in canonical gauge. If its normalized relative discarded
    weight is d, the resulting normalized *physical* states have distance at
    most sqrt(d), since a purifying state overlap bounds mixed-state fidelity.
    Exact gates and channels are trace-distance contractive. The triangle
    inequality adds these errors across truncation events. The returned bound
    is clipped at one. A huge bound must never be presented as high fidelity.
    """
    norm = float(initial_norm)
    bound = 0.0
    if norm <= 0:
        raise ValueError('initial purification norm must be positive')
    for event in events:
        loss = max(0.0, event.discarded_weight)
        if loss > norm + 1e-8:
            raise ArithmeticError('discarded singular weight exceeds current norm')
        relative = min(1.0, loss/max(norm,1e-300))
        bound += np.sqrt(relative)
        norm = max(0., norm-loss)
    return min(1.,float(bound)), norm


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    diff = rho - sigma
    return 0.5*float(np.sum(abs(np.linalg.eigvalsh((diff+diff.conj().T)/2))))


def run_pair(n: int, depth: int, seed: int, channel: str, p: float,
             bond: int, env: int, proposals: int, sweeps: int = 1,
             validate: bool = True, local_tol: float = 0.0,
             factor_swaps: bool = False, factor_objective: str = "retention",
             fast_factor_search: bool = False) -> list[dict]:
    if n < 2 or depth < 1 or (validate and n > 10):
        raise ValueError("require n>=2, depth>=1, and n<=10 for validation")
    gates = gate_layers(n, depth, seed)
    ops = kraus(channel, p)
    chain = PairPurifiedChain(n,bond,env,local_tol,proposals,sweeps,seed=seed,
                              factor_swaps=factor_swaps, fresh_k=len(ops),
                              factor_objective=factor_objective,
                              fast_factor_search=fast_factor_search)
    exact = np.zeros((2**n, 2**n), complex) if validate else None
    if exact is not None:
        exact[0,0] = 1
    rows=[]
    accumulated_bound = 0.0
    for t, layer in enumerate(gates,1):
        event_start = len(chain.truncations)
        begin = perf_counter()
        for i,u in layer:
            chain.gate(i,u)
        for i in range(n):
            chain.channel(i,ops)
        stats = chain.compress_layer()
        retention = chain.normalize()
        layer_bound, predicted_retention = certified_truncation_bound(
            chain.truncations[event_start:])
        if abs(retention-predicted_retention) > 5e-8:
            raise ArithmeticError('orthogonal truncation ledger fails to reproduce layer norm')
        accumulated_bound = min(1., accumulated_bound + layer_bound)
        elapsed=perf_counter()-begin
        row={"n":n,"depth":t,"seed":seed,"channel":channel,"p":p,
             "bond_cap":bond,"env_cap":env,"proposals":proposals,"sweeps":sweeps,
             "factor_swaps":factor_swaps,"factor_objective":factor_objective,
             "runtime_layer_seconds":elapsed,"layer_retention":retention,
             "certified_trace_distance_bound":accumulated_bound,
             "certified_layer_distance_bound":layer_bound,
             "gauge_evaluations":chain.gauge_evaluations,
             "gauge_acceptances":chain.gauge_acceptances,
             "gauge_surrogate_gain":chain.gauge_surrogate_gain,
             "gauge_seconds":chain.gauge_seconds,
             "gauge_pruned":chain.gauge_pruned,
             "gauge_exact_scores":chain.gauge_exact_scores,
             "fast_factor_search":fast_factor_search,
             "sum_discarded_weight":sum(x.discarded_weight for x in chain.truncations),
             **chain.dimensions()}
        if exact is not None:
            for i,u in layer:
                exact = apply_operator(exact,u,(i,i+1),n)
            for i in range(n):
                exact = apply_channel(exact,ops,i,n)
            approx = chain.dense_density()
            row["trace_distance"] = trace_distance(exact,approx)
            row["min_eigenvalue"] = float(np.linalg.eigvalsh((approx+approx.conj().T)/2)[0])
        rows.append(row)
    return rows
