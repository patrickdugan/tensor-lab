"""QC-04: screened factor-gauge search with exact retained-norm verification.

Physical states are invariant under candidate unitary environment permutations.
Candidates are scored after the SAME sequential environment/bond projections as
QC-03. The lower bound on discarded norm (max of two marginal environment
SVD tails) is admissible: any capped two-site projection cannot retain more
squared weight than either individual optimal local environment projection.

The fast path avoids repeated identity scoring, checks promising gauges first,
and prunes using the bound. Survivors are scored using Gram eigensystems. The
winning candidate is checked against identity with the original full-SVD
objective, giving a conservative acceptance condition. Because the search
ranks candidates with floating-point Gram scores, globally identical gauge
selection is an empirical result of the matched-seed experiments, not a
formal guarantee for arbitrary ill-conditioned input tensors.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np

from gauge_pair import env_tail, factor_permutation, projected_pair_norm

@dataclass
class ScreenStats:
    old_loss: float
    new_loss: float
    candidates: int
    evaluated: int
    pruned: int
    accepted: int
    identity_scores: int


def screened_factor_search(theta: np.ndarray, env_cap: int | None,
                           fresh_k: int, bond_cap: int | None,
                           local_tol: float = 0.0,
                           score_method: str = "gram") -> tuple[np.ndarray, ScreenStats]:
    """Screen retention-loss candidates with an analytic admissible bound.

    `env_tail` is the squared norm discarded by the optimal rank-`env_cap`
    orthogonal projection on one of the two environment axes, hence a lower
    bound on the loss after both environment projections plus bond truncation.
    Assumes full theta is canonicalized outside the pair, as in QC-03.
    """
    if score_method not in ("gram", "svd"):
        raise ValueError("unknown score_method")
    score = projected_pair_norm_gram if score_method == "gram" else projected_pair_norm
    norm = float(np.vdot(theta, theta).real)
    baseline = max(0., norm - score(theta, env_cap, bond_cap, local_tol))
    stats = ScreenStats(baseline, baseline, 0, 1, 0, 0, 1)
    ea, eb = theta.shape[2], theta.shape[4]
    if env_cap is None or fresh_k <= 1 or ea % fresh_k or eb % fresh_k:
        return theta, stats
    eps = max(1e-13, 1e-10*baseline)
    candidates = []
    for indices in combinations(range(4),2):
        if indices == (0,1):
            # The identity factorization is already scored above.
            continue
        try:
            candidate = factor_permutation(theta, fresh_k, indices)
        except ValueError:
            continue
        lower = max(env_tail(candidate,2,env_cap),env_tail(candidate,4,env_cap))
        candidates.append((lower, indices, candidate))
    candidates.sort(key=lambda c: (c[0],c[1]))
    stats.candidates = len(candidates)
    best, best_loss = theta, baseline
    for lower, _, candidate in candidates:
        # Add a small absolute roundoff allowance. If ambiguous, evaluate.
        if lower >= best_loss - eps + 1e-12 * norm:
            stats.pruned += 1
            continue
        candidate_loss = max(0., norm-score(candidate,env_cap,bond_cap,local_tol))
        stats.evaluated += 1
        if candidate_loss < best_loss-eps:
            best, best_loss = candidate, candidate_loss
    if score_method == "gram" and best is not theta:
        # Conservative acceptance: full-SVD certify retained norm against
        # identity, rejecting the candidate if numerics reverse the gain.
        exact_baseline = max(0., norm-projected_pair_norm(theta,env_cap,bond_cap,local_tol))
        exact_final = max(0., norm-projected_pair_norm(best,env_cap,bond_cap,local_tol))
        if exact_final > exact_baseline - max(1e-13,1e-10*exact_baseline):
            best, best_loss = theta, baseline
        else:
            best_loss = exact_final
            baseline = exact_baseline
    stats.old_loss = baseline
    stats.new_loss = best_loss
    stats.accepted = int(best is not theta)
    return best, stats


def projected_pair_norm_gram(theta: np.ndarray, env_cap: int | None,
                             bond_cap: int | None, local_tol: float = 0.0) -> float:
    """Cheap retained-norm score using Gram eigensystems on ancillary legs.

    Full SVD remains the fallback if a Gram matrix is ill-conditioned near
    the selected rank or if a matricization is skinny.
    """
    from purified_mpo import stable_svd, select_rank
    current = theta
    for axis in (2,4):
        block = np.moveaxis(current,axis,0)
        mat = block.reshape(block.shape[0],-1)
        if local_tol == 0 and (env_cap is None or env_cap >= mat.shape[0]):
            # Every ancilla direction is retained. Preserve the basis and
            # avoid an unnecessary eigensystem/SVD altogether.
            continue
        if mat.shape[0] > mat.shape[1]:
            u,s,_=stable_svd(mat)
            keep=select_rank(s,env_cap,local_tol)
        else:
            vals, vectors = np.linalg.eigh(mat @ mat.conj().T)
            vals = np.maximum(0., vals[::-1])
            singulars = np.sqrt(vals)
            keep=select_rank(singulars,env_cap,local_tol)
            # Reject rank pollution from squaring the condition number.
            if (singulars[keep-1] < 1e-6 * singulars[0] or
                (keep < len(singulars) and
                 singulars[keep] > .999999 * singulars[keep-1])):
                u,s,_=stable_svd(mat)
                keep=select_rank(s,env_cap,local_tol)
            else:
                u = vectors[:,::-1]
        current=np.moveaxis((u[:,:keep].conj().T @ mat).reshape((keep,)+block.shape[1:]),0,axis)
    dl,_,ea,_,eb,dr=current.shape
    mat=current.reshape(dl*2*ea,2*eb*dr)
    # A singular-value-only SVD for the bond has no U/V cost.
    sv=np.linalg.svd(mat,compute_uv=False)
    keep=select_rank(sv,bond_cap,local_tol)
    return float(np.sum(sv[:keep]**2))
