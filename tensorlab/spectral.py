"""Support functionals as a method-independent filter on degenerations.

For a tensor with *free* support S (any two support points differ in at least
two coordinates), Strassen's support functional and the Christandl–Vrana–
Zuidam quantum functional coincide:

    log zeta^theta(T) = max_{P prob. on S}  sum_l theta_l H(P_l),

theta in the probability simplex, H the Shannon entropy (natural log) of the
l-th marginal.  The quantum functionals are tensor characters on *all* tensors
(monotone under restriction and degeneration, additive, multiplicative), so

    A degenerates to T   ==>   zeta^theta(A) >= zeta^theta(T) for every theta.

Both C(a,B) and a shared-X sum of convolution blocks have free support, so this
is a necessary condition for *any* degeneration (monomial or not).

Numerics (floating point, this module is NOT part of the exact pipeline):
  * lower bound: the objective at the computed distribution P;
  * upper bound: for any marginals q_l > 0, max_P F(P) <= max_{s in S}
    sum_l theta_l (-log q_l(s_l))  (Gibbs' inequality), evaluated at q = marginals of P.
A violation is reported only if upper(source) < lower(target); "no obstruction"
only if lower(source) >= upper(target).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _index(support: Sequence[Tuple[int, int, int]]):
    sup = np.array(support, dtype=np.int64)
    idx = []
    for l in range(3):
        vals, inv = np.unique(sup[:, l], return_inverse=True)
        idx.append((len(vals), inv))
    return idx


def log_support_functional(support: Sequence[Tuple[int, int, int]], theta: Sequence[float],
                           iters: int = 3000, tol: float = 1e-10):
    """Return (lower, upper, P) for max_P sum_l theta_l H(P_l) over the support."""
    idx = _index(support)
    m = len(support)
    th = np.array(theta, dtype=float)
    P = np.full(m, 1.0 / m)
    lower = -math.inf
    upper = math.inf
    eta = 1.0
    for it in range(iters):
        margs = [np.bincount(inv, weights=P, minlength=n) for n, inv in idx]
        logs = [np.log(np.maximum(mg, 1e-300)) for mg in margs]
        F = -sum(th[l] * float(np.dot(margs[l], logs[l])) for l in range(3))
        g = -sum(th[l] * logs[l][idx[l][1]] for l in range(3))   # sum_l theta_l (-log q_l(s_l))
        lower = max(lower, F)
        upper = min(upper, float(g.max()))
        if upper - lower < tol:
            break
        # damped multiplicative (mirror-ascent) step towards P ~ exp(g)
        newP = P * np.exp(eta * (g - g.max()))
        newP /= newP.sum()
        P = newP
    return lower, upper, P


def conv_support(a: int, B: int) -> List[Tuple[int, int, int]]:
    return [(i, j, i + j) for i in range(a) for j in range(B)]


def rectangle_block_support(p: int, q: int) -> List[Tuple[int, int, int]]:
    """C(p,q) with its output leg on X: cells (u+v, u, v)."""
    return [(u + v, u, v) for u in range(p) for v in range(q)]


def shared_rectangles_log_zeta(p: int, q: int, n: int, theta, iters: int = 3000):
    """log zeta^theta of n identical rectangle blocks sharing X (= R (x) B_X(n)).
    Exact reduction: identical blocks are optimal by concavity and symmetry, so
    log zeta(T_n) = (theta_Y + theta_Z) log n + log zeta(R)."""
    lo, up, _ = log_support_functional(rectangle_block_support(p, q), theta, iters)
    s = (theta[1] + theta[2]) * math.log(n)
    return lo + s, up + s


def simplex_grid(step: float):
    k = int(round(1 / step))
    pts = []
    for i in range(k + 1):
        for j in range(k + 1 - i):
            pts.append((i / k, j / k, (k - i - j) / k))
    return pts


def dense_rectangle_test(a: int, p: int, q: int, ns: Sequence[int], step: float = 0.1, iters: int = 3000):
    """For each n, compare zeta^theta(C(a, n*max(p,q))) with zeta^theta(R(p,q) (x) B_X(n)).
    Returns rows with the worst certified margin over the theta grid."""
    rows = []
    thetas = simplex_grid(step)
    TOL = 1e-7
    for n in ns:
        B = n * max(p, q)
        src = conv_support(a, B)
        worst = None
        worst_interior = None
        for th in thetas:
            s_lo, s_up, _ = log_support_functional(src, th, iters)
            t_lo, t_up = shared_rectangles_log_zeta(p, q, n, th, iters)
            margin_cert = s_lo - t_up       # >= 0 certifies no obstruction at theta
            viol_cert = s_up - t_lo          # < 0 certifies an obstruction at theta
            rec = (margin_cert, viol_cert, th)
            if worst is None or margin_cert < worst[0]:
                worst = rec
            if min(th) > 0 and (worst_interior is None or margin_cert < worst_interior[0]):
                worst_interior = rec
        rows.append({"a": a, "p": p, "q": q, "n": n, "B": B,
                     "worst_certified_margin": float(worst[0]), "worst_violation_bound": float(worst[1]),
                     "theta": worst[2],
                     "worst_interior_margin": float(worst_interior[0]) if worst_interior else None,
                     "theta_interior": worst_interior[2] if worst_interior else None,
                     "obstruction": bool(worst[1] < -TOL),
                     "no_obstruction_certified_on_grid": bool(worst[0] >= -TOL)})
    return rows
