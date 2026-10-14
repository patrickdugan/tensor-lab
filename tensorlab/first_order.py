"""Reduced first-order border system for dense rectangle packings in tensor powers.

A degeneration  T^{(x)N}  <=  2 . S^{(x)N}  of first order exists iff there are
  * an additive function L = alpha(x) + beta(y) + gamma(z) on supp(S^{(x)N})
    that vanishes on every cross term, and
  * a rank-one weight w = rho(x) sigma(y) tau(z)
with L * w = 1 on every retained term (see border_construct.py for the explicit
degeneration).  This module solves that system numerically (Gauss-Newton on the
retained equations, with L constrained to the exact null space of the cross
equations) and reports the residual."""
from __future__ import annotations

import numpy as np

from .border_construct import _support


def reduced_system(p: int, n: int, N: int):
    S, deltas, dims, terms = _support(p, n, N)
    X, Y, Z = dims
    nv = X + Y + Z
    ret = np.array([all(d == 0 for d in dv) for _, dv, _ in terms])
    M = np.zeros((len(terms), nv))
    idx = np.arange(len(terms))
    xyz = np.array([[x, y, z] for (x, y, z), _, _ in terms])
    M[idx, xyz[:, 0]] = 1
    M[idx, X + xyz[:, 1]] = 1
    M[idx, X + Y + xyz[:, 2]] = 1
    Mc, Mr = M[~ret], M[ret]
    u, s, vt = np.linalg.svd(Mc)
    rank_c = int((s > 1e-9 * s.max()).sum())
    V = vt[rank_c:].T
    R = Mr @ V
    return {"R": R, "rows": xyz[ret], "dims": dims, "dimV": V.shape[1], "V": V, "ret": ret, "terms": terms}


def solve_first_order(p: int, n: int, N: int, trials: int = 40, iters: int = 400, seed: int = 0):
    sysd = reduced_system(p, n, N)
    R, rows, (X, Y, Z) = sysd["R"], sysd["rows"], sysd["dims"]
    k, m = R.shape[1], len(rows)
    rng = np.random.default_rng(seed)
    best = None
    for trial in range(trials):
        th = np.concatenate([rng.standard_normal(k), np.abs(rng.standard_normal(X)) + 0.5,
                             np.sign(rng.standard_normal(Y)) * (np.abs(rng.standard_normal(Y)) + .5),
                             np.sign(rng.standard_normal(Z)) * (np.abs(rng.standard_normal(Z)) + .5)])

        def F(th):
            cv = th[:k]; r = th[k:k + X]; s_ = th[k + X:k + X + Y]; t = th[k + X + Y:]
            return (R @ cv) * r[rows[:, 0]] * s_[rows[:, 1]] * t[rows[:, 2]] - 1.0

        def J(th):
            cv = th[:k]; r = th[k:k + X]; s_ = th[k + X:k + X + Y]; t = th[k + X + Y:]
            L = R @ cv
            w = r[rows[:, 0]] * s_[rows[:, 1]] * t[rows[:, 2]]
            Jm = np.zeros((m, len(th)))
            Jm[:, :k] = R * w[:, None]
            Jm[np.arange(m), k + rows[:, 0]] = L * s_[rows[:, 1]] * t[rows[:, 2]]
            Jm[np.arange(m), k + X + rows[:, 1]] = L * r[rows[:, 0]] * t[rows[:, 2]]
            Jm[np.arange(m), k + X + Y + rows[:, 2]] = L * r[rows[:, 0]] * s_[rows[:, 1]]
            return Jm

        lam = 1e-3
        f = F(th); val = float(f @ f)
        for it in range(iters):
            Jm = J(th); H = Jm.T @ Jm; g = Jm.T @ f
            step = np.linalg.solve(H + lam * np.eye(len(th)), -g)
            th2 = th + step; f2 = F(th2); v2 = float(f2 @ f2)
            if v2 < val:
                th, f, val = th2, f2, v2
                lam = max(lam / 3, 1e-14)
            else:
                lam *= 5
                if lam > 1e12:
                    break
            if val < 1e-26:
                break
        if best is None or val < best[0]:
            best = (val, th)
        if best[0] < 1e-26:
            break
    val, th = best
    L = R @ th[:k]
    return {"p": p, "n": n, "N": N, "dimV": sysd["dimV"], "retained": len(rows), "residual": val,
            "min_abs_L": float(np.abs(L).min()), "theta": th}
