"""Numerical search for restrictions T = (A (x) B (x) C) . S between small tensors.

Floating point (numpy), Levenberg–Marquardt on the cubic system
T[x',y',z'] = sum A[x',x] B[y',y] C[z',z] S[x,y,z].  A small residual is only
*evidence*; an exact claim needs a rationalised solution checked exactly
(``exact_check``).  Used for the open question in notes/04: can non-monomial
maps pack rectangle blocks C(p,q)^sigma (x) B_X(n) more densely than monomial
weights allow?
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np

from .tensors import SparseTensor, convolution, dot_product, tensor_product


def dense(t: SparseTensor) -> np.ndarray:
    a = np.zeros(t.dims)
    for (i, j, k), c in t.terms.items():
        a[i, j, k] = float(c)
    return a


def rectangle_target(p: int, q: int, n: int) -> SparseTensor:
    """C(p,q) with its output leg moved to X, tensored with B_X(n):
    sum_{u<p, v<q, i<n} x_{u+v} y_{(u,i)} z_{(v,i)}."""
    rect = convolution(p, q).permute_legs("cycle_XZY")  # new X = old Z (output), new Y = old X, new Z = old Y
    return tensor_product(rect, dot_product(n, 0))


def _apply(A, B, C, S):
    return np.einsum("ai,bj,ck,ijk->abc", A, B, C, S, optimize=True)


def lm_restriction(S: np.ndarray, T: np.ndarray, starts: int = 20, iters: int = 400, seed: int = 0,
                   tol: float = 1e-12):
    """Return the best (residual, A, B, C, max_entry) over random starts."""
    rng = np.random.default_rng(seed)
    a, b, c = T.shape
    i, j, k = S.shape
    nA, nB, nC = a * i, b * j, c * k
    best = None
    for s in range(starts):
        A = rng.standard_normal((a, i))
        B = rng.standard_normal((b, j))
        C = rng.standard_normal((c, k))
        lam = 1e-2
        R = _apply(A, B, C, S) - T
        f = float((R ** 2).sum())
        for it in range(iters):
            # Jacobian columns: d/dA[a',i'] etc.
            SB_C = np.einsum("bj,ck,ijk->ibc", B, C, S, optimize=True)      # (i, b, c)
            SA_C = np.einsum("ai,ck,ijk->ajc", A, C, S, optimize=True)      # (a, j, c)
            SA_B = np.einsum("ai,bj,ijk->abk", A, B, S, optimize=True)      # (a, b, k)
            JA = np.einsum("xa,ibc->xbcai", np.eye(a), SB_C).reshape(a * b * c, nA)
            JB = np.einsum("yb,ajc->aycbj", np.eye(b), SA_C).reshape(a * b * c, nB)
            JC = np.einsum("zc,abk->abzck", np.eye(c), SA_B).reshape(a * b * c, nC)
            J = np.hstack([JA, JB, JC])
            r = R.reshape(-1)
            g = J.T @ r
            H = J.T @ J
            while True:
                step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), -g)
                A2 = A + step[:nA].reshape(a, i)
                B2 = B + step[nA:nA + nB].reshape(b, j)
                C2 = C + step[nA + nB:].reshape(c, k)
                R2 = _apply(A2, B2, C2, S) - T
                f2 = float((R2 ** 2).sum())
                if f2 < f:
                    A, B, C, R, f = A2, B2, C2, R2, f2
                    lam = max(lam / 3, 1e-12)
                    break
                lam *= 4
                if lam > 1e12:
                    break
            if f < tol or lam > 1e12:
                break
        mx = max(np.abs(A).max(), np.abs(B).max(), np.abs(C).max())
        if best is None or f < best[0]:
            best = (f, A, B, C, mx)
    return best


def exact_check(S: SparseTensor, T: SparseTensor, A, B, C) -> bool:
    """Exact check of a rational restriction (A, B, C given as nested lists of Fractions)."""
    from .tensors import Matrix
    MA = Matrix(len(A), len(A[0]), {(r, c): v for r, row in enumerate(A) for c, v in enumerate(row)})
    MB = Matrix(len(B), len(B[0]), {(r, c): v for r, row in enumerate(B) for c, v in enumerate(row)})
    MC = Matrix(len(C), len(C[0]), {(r, c): v for r, row in enumerate(C) for c, v in enumerate(row)})
    return S.restrict(MA, MB, MC) == T


def diagonal_copy_cost(p: int, n: int, N: int, c: int, starts: int = 30, iters: int = 400, seed: int = 0):
    """Least-squares search for c copies of the block form S'^{(x)N}, each scaled by
    diagonal maps on X, Y, Z, whose sum keeps exactly the Delta = 0 terms:
    find a_r(x), d_r(y), e_r(z) with sum_r a_r(x) d_r(y) e_r(z) = 1 on retained
    support terms and 0 on cross terms (free off the support).  Residual 0 means
    T^{(x)N} <= c . C(2p-1, np)^{(x)N} by diagonal maps.  Floating point."""
    from .dense import block_form, power_with_deltas
    S, deltas = block_form(p, n)
    dims, terms = power_with_deltas(S, deltas, N)
    xs = sorted({t[0][0] for t in terms}); ys = sorted({t[0][1] for t in terms}); zs = sorted({t[0][2] for t in terms})
    xi = {v: i for i, v in enumerate(xs)}; yi = {v: i for i, v in enumerate(ys)}; zi = {v: i for i, v in enumerate(zs)}
    rows = np.array([[xi[k[0]], yi[k[1]], zi[k[2]]] for k, dv, _ in terms])
    target = np.array([1.0 if all(d == 0 for d in dv) else 0.0 for _, dv, _ in terms])
    nx, ny, nz = len(xs), len(ys), len(zs)
    rng = np.random.default_rng(seed)
    best = None
    for s in range(starts):
        a = rng.standard_normal((c, nx)); d = rng.standard_normal((c, ny)); e = rng.standard_normal((c, nz))
        lam = 1e-2
        def resid(a, d, e):
            return (a[:, rows[:, 0]] * d[:, rows[:, 1]] * e[:, rows[:, 2]]).sum(0) - target
        r = resid(a, d, e); f = float(r @ r)
        for it in range(iters):
            m = len(target)
            J = np.zeros((m, c * (nx + ny + nz)))
            for k in range(c):
                ak, dk, ek = a[k, rows[:, 0]], d[k, rows[:, 1]], e[k, rows[:, 2]]
                J[np.arange(m), k * nx + rows[:, 0]] = dk * ek
                J[np.arange(m), c * nx + k * ny + rows[:, 1]] = ak * ek
                J[np.arange(m), c * (nx + ny) + k * nz + rows[:, 2]] = ak * dk
            g = J.T @ r; H = J.T @ J
            while True:
                step = np.linalg.solve(H + lam * (np.diag(np.diag(H)) + 1e-9 * np.eye(len(H))), -g)
                a2 = a + step[:c * nx].reshape(c, nx)
                d2 = d + step[c * nx:c * (nx + ny)].reshape(c, ny)
                e2 = e + step[c * (nx + ny):].reshape(c, nz)
                r2 = resid(a2, d2, e2); f2 = float(r2 @ r2)
                if f2 < f:
                    a, d, e, r, f = a2, d2, e2, r2, f2
                    lam = max(lam / 3, 1e-12)
                    break
                lam *= 4
                if lam > 1e12:
                    break
            if f < 1e-20 or lam > 1e12:
                break
        mx = float(max(abs(a).max(), abs(d).max(), abs(e).max()))
        if best is None or f < best[0]:
            best = (f, mx, (a, d, e))
    return {"p": p, "n": n, "N": N, "c": c, "support_terms": len(target), "retained": int(target.sum()),
            "best_residual": best[0], "max_abs_entry": best[1], "solution": best[2]}


def _setup_pattern(p, n, N):
    from .dense import block_form, power_with_deltas
    S, deltas = block_form(p, n)
    dims, terms = power_with_deltas(S, deltas, N)
    xs = sorted({t[0][0] for t in terms}); ys = sorted({t[0][1] for t in terms}); zs = sorted({t[0][2] for t in terms})
    xi = {v: i for i, v in enumerate(xs)}; yi = {v: i for i, v in enumerate(ys)}; zi = {v: i for i, v in enumerate(zs)}
    rows = np.array([[xi[k[0]], yi[k[1]], zi[k[2]]] for k, dv, _ in terms])
    target = np.array([1.0 if all(d == 0 for d in dv) else 0.0 for _, dv, _ in terms])
    return rows, target, (len(xs), len(ys), len(zs)), terms, (xs, ys, zs)


def border_path(p: int, n: int, N: int, c: int, mus=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8),
                starts: int = 20, iters: int = 600, seed: int = 0, pattern=None):
    """Regularised continuation: minimise |W - I|^2 + mu |params|^2 for decreasing
    mu, warm-starting each stage.  A border (degeneration-only) solution shows the
    residual falling towards 0 while the parameter norm grows without bound; an
    exact low-rank solution keeps the norm bounded; no solution keeps the residual
    bounded away from 0.  Returns the (mu, residual, norm) path of the best start.
    ``pattern`` = (rows, target, dims) overrides the block-form support (e.g.
    coarse_pattern(N))."""
    rows, target, (nx, ny, nz) = pattern if pattern is not None else _setup_pattern(p, n, N)[:3]
    m = len(target)
    rng = np.random.default_rng(seed)
    P = c * (nx + ny + nz)

    def unpack(th):
        return th[:c * nx].reshape(c, nx), th[c * nx:c * (nx + ny)].reshape(c, ny), th[c * (nx + ny):].reshape(c, nz)

    def resid(th):
        a, d, e = unpack(th)
        return (a[:, rows[:, 0]] * d[:, rows[:, 1]] * e[:, rows[:, 2]]).sum(0) - target

    def jac(th):
        a, d, e = unpack(th)
        J = np.zeros((m, P))
        for k in range(c):
            ak, dk, ek = a[k, rows[:, 0]], d[k, rows[:, 1]], e[k, rows[:, 2]]
            J[np.arange(m), k * nx + rows[:, 0]] = dk * ek
            J[np.arange(m), c * nx + k * ny + rows[:, 1]] = ak * ek
            J[np.arange(m), c * (nx + ny) + k * nz + rows[:, 2]] = ak * dk
        return J

    best = None
    for s in range(starts):
        th = rng.standard_normal(P)
        path = []
        for mu in mus:
            lam = 1e-3
            r = resid(th); f = float(r @ r) + mu * float(th @ th)
            for it in range(iters):
                J = jac(th)
                g = J.T @ r + mu * th
                H = J.T @ J + mu * np.eye(P)
                while True:
                    step = np.linalg.solve(H + lam * np.eye(P), -g)
                    th2 = th + step
                    r2 = resid(th2); f2 = float(r2 @ r2) + mu * float(th2 @ th2)
                    if f2 < f:
                        th, r, f = th2, r2, f2
                        lam = max(lam / 3, 1e-14)
                        break
                    lam *= 4
                    if lam > 1e14:
                        break
                if lam > 1e14:
                    break
            path.append((mu, float(r @ r), float(np.sqrt(th @ th))))
        final = path[-1][1]
        if best is None or final < best[0]:
            best = (final, path, th)
    return {"p": p, "n": n, "N": N, "c": c, "path": best[1], "theta": best[2]}


def copy_cost_lm(p: int, n: int, N: int, c: int, starts: int = 10, iters: int = 500, seed: int = 0,
                 mu: float = 0.0, init=None, tol: float = 1e-24, verbose: bool = False, pattern=None):
    """Levenberg-Marquardt for diagonal c-copy restrictions of S^{(x)N} onto
    T^{(x)N} (the same system as diagonal_copy_cost) with a structured-sparse
    Jacobian: every equation involves 3c unknowns, so J^T J is accumulated
    directly (no dense m x P Jacobian).  ``mu`` > 0 adds Tikhonov regularisation.
    Returns the best start: residual, parameter norm and (a, d, e).  Floating
    point evidence only.  ``pattern`` overrides the support as in border_path."""
    rows, target, (nx, ny, nz) = pattern if pattern is not None else _setup_pattern(p, n, N)[:3]
    m = len(target)
    P = c * (nx + ny + nz)
    rng = np.random.default_rng(seed)
    cols = np.concatenate([np.stack([k * nx + rows[:, 0] for k in range(c)], 1),
                           np.stack([c * nx + k * ny + rows[:, 1] for k in range(c)], 1),
                           np.stack([c * (nx + ny) + k * nz + rows[:, 2] for k in range(c)], 1)], 1)  # (m, 3c)
    ii = np.repeat(cols, 3 * c, axis=1).reshape(-1)
    jj = np.tile(cols, (1, 3 * c)).reshape(-1)
    flat = ii * P + jj

    def unpack(th):
        return th[:c * nx].reshape(c, nx), th[c * nx:c * (nx + ny)].reshape(c, ny), th[c * (nx + ny):].reshape(c, nz)

    def resid_vals(th):
        a, d, e = unpack(th)
        A, D, E = a[:, rows[:, 0]], d[:, rows[:, 1]], e[:, rows[:, 2]]
        return (A * D * E).sum(0) - target, A, D, E

    best = None
    for s in range(starts):
        th = init.copy() if (init is not None and s == 0) else rng.standard_normal(P)
        r, A, D, E = resid_vals(th)
        f = float(r @ r) + mu * float(th @ th)
        lam = 1e-3
        for it in range(iters):
            vals = np.concatenate([(D * E).T, (A * E).T, (A * D).T], 1)  # (m, 3c)
            H = np.bincount(flat, weights=(vals[:, :, None] * vals[:, None, :]).reshape(-1),
                            minlength=P * P).reshape(P, P)
            g = np.bincount(cols.reshape(-1), weights=(vals * r[:, None]).reshape(-1), minlength=P)
            H += mu * np.eye(P)
            g += mu * th
            improved = False
            while lam < 1e14:
                step = np.linalg.solve(H + lam * np.eye(P), -g)
                th2 = th + step
                r2, A2, D2, E2 = resid_vals(th2)
                f2 = float(r2 @ r2) + mu * float(th2 @ th2)
                if f2 < f:
                    th, r, A, D, E, f = th2, r2, A2, D2, E2, f2
                    lam = max(lam / 3, 1e-15)
                    improved = True
                    break
                lam *= 4
            if not improved or float(r @ r) < tol:
                break
        res = float(r @ r)
        if verbose:
            print(f"  start {s}: residual {res:.3e}, |params| {np.linalg.norm(th):.2f}", flush=True)
        if best is None or res < best[0]:
            best = (res, float(np.linalg.norm(th)), th.copy())
        if res < tol:
            break
    a, d, e = unpack(best[2])
    return {"p": p, "n": n, "N": N, "c": c, "support_terms": m, "retained": int(target.sum()),
            "residual": best[0], "norm": best[1], "a": a, "d": d, "e": e, "theta": best[2]}


def copy_cost_continuation(p: int, n: int, N: int, c: int, starts: int = 8, iters: int = 300, seed: int = 0,
                           mus=(1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 0.0), verbose: bool = False,
                           pattern=None):
    """Regularised continuation with the sparse solver (copy_cost_lm), warm-started
    along a decreasing mu schedule ending with an unregularised polish.  Returns the
    best start's path [(mu, residual, norm)] and final parameters.  Bounded norm with
    residual -> 0 indicates an exact restriction; residual -> 0 with growing norm a
    border (degeneration-only) solution; a residual plateau, no solution."""
    rng = np.random.default_rng(seed)
    best = None
    for s in range(starts):
        th = None
        path = []
        for mu in mus:
            r = copy_cost_lm(p, n, N, c, starts=1, iters=iters, seed=int(rng.integers(1 << 30)), mu=mu, init=th,
                             pattern=pattern)
            th = r["theta"]
            path.append((mu, r["residual"], r["norm"]))
        if verbose:
            print(f"  start {s}: " + " ".join(f"{res:.1e}" for _, res, _ in path) + f" | norm {path[-1][2]:.1f}", flush=True)
        if best is None or path[-1][1] < best[0]:
            best = (path[-1][1], path, r)
        if best[0] < 1e-24:
            break
    out = dict(best[2])
    out["path"] = best[1]
    return out


COARSE_TYPES = None


def coarse_types():
    """The 12 coarse term types of one factor of the p = 2 block form, valid for
    every n >= 2 (all occur once n >= 3): (x, Y, Z, retained) with Y = 2 col(j) +
    par(j), Z = 2 col(k) + par(k), col = block index mod 2, par = index mod 2."""
    out = []
    for c in (0, 1):
        for p_ in (0, 1):
            out.append((1, 2 * c + p_, 2 * c + p_, 1))          # x = 1: k = j
        out.append((2, 2 * c + 0, 2 * c + 1, 1))                 # x = 2, j even: same block
        out.append((0, 2 * c + 1, 2 * c + 0, 1))                 # x = 0, j odd: same block
        out.append((2, 2 * c + 1, 2 * (1 - c) + 0, 0))           # x = 2, j odd: next block (cross)
        out.append((0, 2 * c + 0, 2 * (1 - c) + 1, 0))           # x = 0, j even: previous block (cross)
    return out


def coarse_pattern(N: int):
    """Support pattern (rows, target, dims) of the N-th power of the coarse problem
    (3 x 4 x 4 per factor, 12 specified entries).  A c-term solution lifts to a
    diagonal c-copy restriction T_n^{(x)N} <= c . C(3, 2n)^{(x)N} for every n >= 2,
    by composing with j -> (col(j), par(j)) and k -> (col(k), par(k))."""
    import itertools
    types = coarse_types()
    rows, target = [], []
    for combo in itertools.product(types, repeat=N):
        x = y = z = 0
        r = 1
        for (tx, ty, tz, ret) in combo:
            x, y, z = 3 * x + tx, 4 * y + ty, 4 * z + tz
            r *= ret
        rows.append((x, y, z))
        target.append(float(r))
    return np.array(rows), np.array(target), (3 ** N, 4 ** N, 4 ** N)


def limit_system_search(pattern, r: int, t: int = 1, starts: int = 4, iters: int = 400, seed: int = 0,
                        mus=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 0.0),
                        init=None, verbose: bool = False):
    """Solve the *limit* form of a border decomposition directly:
        target = sum_{i<r} u_i(x) v_i(y) w_i(z)
               + sum_{s<t} [a'_s b_s c_s + a_s b'_s c_s + a_s b_s c'_s]     on the support,
    where each bracket is a tangential term (the eps -> 0 limit of two copies,
    (1/eps)[(a+eps a')(b+eps b')(c+eps c') - abc]).  Unlike the divergent copy
    parametrisation this system has bounded solutions, so a solution can be
    rationalised.  A solution gives a degeneration from r + 2t copies.
    Regularised continuation with a structured-sparse Jacobian; floating point."""
    rows, target, (nx, ny, nz) = pattern
    m = len(target)
    blk = nx + ny + nz
    P = r * blk + t * 2 * blk
    xs, ys, zs = rows[:, 0], rows[:, 1], rows[:, 2]

    def base(k):  # start of block k
        return k * blk

    cols = []
    for i in range(r):
        o = base(i)
        cols += [o + xs, o + nx + ys, o + nx + ny + zs]
    for s in range(t):
        o = r * blk + s * 2 * blk          # a, b, c then a', b', c'
        cols += [o + xs, o + nx + ys, o + nx + ny + zs,
                 o + blk + xs, o + blk + nx + ys, o + blk + nx + ny + zs]
    cols = np.stack(cols, 1)               # (m, K)
    K = cols.shape[1]
    ii = np.repeat(cols, K, axis=1).reshape(-1)
    jj = np.tile(cols, (1, K)).reshape(-1)
    flat = ii * P + jj

    def evaluate(th):
        g = th[cols]                       # (m, K) parameter values at each row
        val = np.zeros(m)
        der = np.zeros((m, K))
        for i in range(r):
            u, v, w = g[:, 3 * i], g[:, 3 * i + 1], g[:, 3 * i + 2]
            val += u * v * w
            der[:, 3 * i], der[:, 3 * i + 1], der[:, 3 * i + 2] = v * w, u * w, u * v
        for s in range(t):
            k = 3 * r + 6 * s
            a, b, c, a2, b2, c2 = (g[:, k + q] for q in range(6))
            val += a2 * b * c + a * b2 * c + a * b * c2
            der[:, k] = b2 * c + b * c2
            der[:, k + 1] = a2 * c + a * c2
            der[:, k + 2] = a2 * b + a * b2
            der[:, k + 3] = b * c
            der[:, k + 4] = a * c
            der[:, k + 5] = a * b
        return val - target, der

    rng = np.random.default_rng(seed)
    best = None
    for st in range(starts):
        th = init.copy() if (init is not None and st == 0) else rng.standard_normal(P)
        path = []
        for mu in mus:
            res, der = evaluate(th)
            f = res @ res + mu * th @ th
            lam = 1e-3
            for it in range(iters):
                H = np.bincount(flat, weights=(der[:, :, None] * der[:, None, :]).reshape(-1),
                                minlength=P * P).reshape(P, P) + mu * np.eye(P)
                g = np.bincount(cols.reshape(-1), weights=(der * res[:, None]).reshape(-1), minlength=P) + mu * th
                ok = False
                while lam < 1e14:
                    th2 = th - np.linalg.solve(H + lam * np.eye(P), g)
                    res2, der2 = evaluate(th2)
                    f2 = res2 @ res2 + mu * th2 @ th2
                    if f2 < f:
                        th, res, der, f = th2, res2, der2, f2
                        lam = max(lam / 3, 1e-15)
                        ok = True
                        break
                    lam *= 4
                if not ok or res @ res < 1e-26:
                    break
            path.append((mu, float(res @ res), float(np.linalg.norm(th))))
        if verbose:
            print(f"  start {st}: " + " ".join(f"{q:.1e}" for _, q, _ in path) + f" | norm {path[-1][2]:.1f}", flush=True)
        if best is None or path[-1][1] < best[0]:
            best = (path[-1][1], path, th.copy())
        if best[0] < 1e-24:
            break
    return {"r": r, "t": t, "residual": best[0], "path": best[1], "theta": best[2], "P": P, "m": m}
