"""Exact realizability of a block configuration by *general* monomial weights.

Setting: the source is C(a,B), whose support is the set of cells (i, j, k=i+j)
with 0 <= i < a, 0 <= j < B.  A monomial weight degeneration assigns integer
weights x_i, u_j, v_k to the three legs; a cell has weight x_i + u_j + v_k.
Given a target retained set R (for instance a union of blocks with disjoint
Y- and Z-coordinates), the question is whether some weights make every cell of
R weight 0 and every other cell weight >= 1.  This is a linear feasibility
problem over Q (rational solutions can be scaled to integers).

By Farkas' lemma exactly one of the following holds:
  (P) weights x, u, v exist (a *realization*);
  (D) there are a nonnegative, nonzero measure mu on the erased cells and a
      signed measure nu on R with the same i-, j- and k-marginals (an
      *obstruction*: summing the weight over mu gives >= |mu| > 0, over nu gives
      0, but equal marginals force equal sums).
Both are computed with the exact simplex and both are checked independently.
``use_x=False`` fixes x = 0 (the sector search's setting); then the i-marginal
condition disappears from (D).

The sector search in ``search.py`` only uses weights that are constant on
sectors and zero on X; this module decides realizability with arbitrary
per-coordinate weights on all three legs, which is strictly more general.
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .lp import LinearProgram

Cell = Tuple[int, int]  # (i, j); k = i + j


def band_cells(a: int, B: int) -> List[Cell]:
    return [(i, j) for i in range(a) for j in range(B)]


def _lcm(x: int, y: int) -> int:
    return x // gcd(x, y) * y


def solve_primal(a: int, B: int, retained: Set[Cell], use_x: bool = True,
                 max_pivots: int = 200000):
    """Return ("feasible", (x, u, v)) with integer weights, or ("infeasible", None)."""
    lp = LinearProgram()
    K = a + B - 1

    def wvars(i, j):
        k = i + j
        d: Dict[str, int] = {}
        if use_x:
            d[f"xp{i}"] = 1
            d[f"xn{i}"] = -1
        d[f"up{j}"] = d.get(f"up{j}", 0) + 1
        d[f"un{j}"] = d.get(f"un{j}", 0) - 1
        d[f"vp{k}"] = d.get(f"vp{k}", 0) + 1
        d[f"vn{k}"] = d.get(f"vn{k}", 0) - 1
        return d

    for (i, j) in band_cells(a, B):
        if (i, j) in retained:
            lp.add(wvars(i, j), "==", 0)
        else:
            lp.add(wvars(i, j), ">=", 1)
    lp.set_objective({})
    status, _, sol = lp.solve(max_pivots=max_pivots)
    if status == "infeasible":
        return "infeasible", None
    if status != "optimal":
        return status, None

    def val(prefix_p, prefix_n, idx):
        return sol.get(f"{prefix_p}{idx}", Fraction(0)) - sol.get(f"{prefix_n}{idx}", Fraction(0))

    x = [val("xp", "xn", i) if use_x else Fraction(0) for i in range(a)]
    u = [val("up", "un", j) for j in range(B)]
    v = [val("vp", "vn", k) for k in range(K)]
    den = 1
    for q in x + u + v:
        den = _lcm(den, q.denominator)
    xi = [int(q * den) for q in x]
    ui = [int(q * den) for q in u]
    vi = [int(q * den) for q in v]
    return "feasible", (xi, ui, vi)


def solve_farkas(a: int, B: int, retained: Set[Cell], use_x: bool = True,
                 max_pivots: int = 200000):
    """Return ("obstruction", (mu, nu)) or ("none", None).  mu: {cell: Fraction >= 0}
    on erased cells with total mass 1; nu: {cell: Fraction} on retained cells."""
    lp = LinearProgram()
    cells = band_cells(a, B)
    marg: Dict[str, Dict[str, int]] = {}
    for (i, j) in cells:
        k = i + j
        keys = [f"J{j}", f"K{k}"] + ([f"I{i}"] if use_x else [])
        if (i, j) in retained:
            for key in keys:
                marg.setdefault(key, {})[f"np_{i}_{j}"] = -1
                marg[key][f"nn_{i}_{j}"] = 1
        else:
            for key in keys:
                marg.setdefault(key, {})[f"m_{i}_{j}"] = 1
    for key, row in marg.items():
        lp.add(row, "==", 0)
    lp.add({f"m_{i}_{j}": 1 for (i, j) in cells if (i, j) not in retained}, "==", 1)
    lp.set_objective({})
    status, _, sol = lp.solve(max_pivots=max_pivots)
    if status == "infeasible":
        return "none", None
    if status != "optimal":
        return status, None
    mu = {}
    nu = {}
    for (i, j) in cells:
        if (i, j) in retained:
            v = sol.get(f"np_{i}_{j}", Fraction(0)) - sol.get(f"nn_{i}_{j}", Fraction(0))
            if v != 0:
                nu[(i, j)] = v
        else:
            v = sol.get(f"m_{i}_{j}", Fraction(0))
            if v != 0:
                mu[(i, j)] = v
    return "obstruction", (mu, nu)


def verify_realization(a: int, B: int, retained: Set[Cell], weights, use_x: bool = True) -> bool:
    x, u, v = weights
    if not use_x and any(x):
        return False
    for (i, j) in band_cells(a, B):
        w = x[i] + u[j] + v[i + j]
        if (i, j) in retained:
            if w != 0:
                return False
        elif w < 1:
            return False
    return True


def verify_obstruction(a: int, B: int, retained: Set[Cell], cert, use_x: bool = True) -> bool:
    mu, nu = cert
    if not mu or any(c < 0 for c in mu.values()) or sum(mu.values()) <= 0:
        return False
    if any(cell in retained for cell in mu) or any(cell not in retained for cell in nu):
        return False
    if any(not (0 <= i < a and 0 <= j < B) for (i, j) in list(mu) + list(nu)):
        return False

    def marginals(meas):
        I, J, K = {}, {}, {}
        for (i, j), c in meas.items():
            I[i] = I.get(i, 0) + c
            J[j] = J.get(j, 0) + c
            K[i + j] = K.get(i + j, 0) + c
        clean = lambda d: {k: v for k, v in d.items() if v != 0}
        return clean(I), clean(J), clean(K)

    mI, mJ, mK = marginals(mu)
    nI, nJ, nK = marginals(nu)
    ok = mJ == nJ and mK == nK
    if use_x:
        ok = ok and mI == nI
    return ok


def decide(a: int, B: int, retained: Set[Cell], use_x: bool = True) -> dict:
    """Run both sides of the Farkas alternative and check that exactly one holds."""
    st_p, w = solve_primal(a, B, retained, use_x)
    st_d, cert = solve_farkas(a, B, retained, use_x)
    out = {"a": a, "B": B, "use_x": use_x, "retained": len(retained)}
    if st_p == "feasible":
        out["realizable"] = True
        out["weights"] = {"X": w[0], "Y": w[1], "Z": w[2]}
        out["weights_verified"] = verify_realization(a, B, retained, w, use_x)
    else:
        out["realizable"] = False
    if st_d == "obstruction":
        out["obstruction"] = {"mu": {f"{i},{j}": str(c) for (i, j), c in sorted(cert[0].items())},
                              "nu": {f"{i},{j}": str(c) for (i, j), c in sorted(cert[1].items())}}
        out["obstruction_verified"] = verify_obstruction(a, B, retained, cert, use_x)
    out["farkas_consistent"] = (st_p == "feasible") != (st_d == "obstruction")
    return out


# ---------------------------------------------------------------------------------
# Block templates in the band of C(a,B)
# ---------------------------------------------------------------------------------

def rectangle_cells(y: int, p: int, z: int, q: int, a: int) -> Optional[Set[Cell]]:
    """All pairs j in [y,y+p), k in [z,z+q) as cells (i=k-j, j); None unless every
    pair lies in the band 0 <= k-j < a.  The block is C(p,q) with its output on X."""
    if z - (y + p - 1) < 0 or (z + q - 1) - y > a - 1:
        return None
    return {(k - j, j) for j in range(y, y + p) for k in range(z, z + q)}


def normal_cells(s: int, m: int, y: int, h: int) -> Set[Cell]:
    """C(m,h) with X carrying the i-range [s,s+m) and Y the range [y,y+h)."""
    return {(i, j) for i in range(s, s + m) for j in range(y, y + h)}


def exchanged_cells(s: int, m: int, z: int, h: int) -> Set[Cell]:
    """C(m,h) with X carrying [s,s+m), Z carrying [z,z+h) and Y the output."""
    return {(i, k - i) for i in range(s, s + m) for k in range(z, z + h)}


def block_legs(cells: Set[Cell]):
    ys = {j for (_, j) in cells}
    zs = {i + j for (i, j) in cells}
    xs = {i for (i, _) in cells}
    return xs, ys, zs


def check_disjoint(blocks: Sequence[Set[Cell]]) -> bool:
    Y, Z = set(), set()
    for c in blocks:
        _, ys, zs = block_legs(c)
        if Y & ys or Z & zs:
            return False
        Y |= ys
        Z |= zs
    return True


def staircase_retained(a: int, p: int, q: int, n: int, period: int, s: int = 0):
    """n rectangles C(p,q) (output on X, i-range [s, s+p+q-2]) at the given period."""
    blocks = []
    for b in range(n):
        y = b * period
        cells = rectangle_cells(y, p, y + p - 1 + s, q, a)
        if cells is None:
            raise ValueError("rectangle leaves the band")
        blocks.append(cells)
    if not check_disjoint(blocks):
        raise ValueError("blocks overlap on Y or Z")
    B = (n - 1) * period + p
    return B, blocks


# ---------------------------------------------------------------------------------
# Generic version: arbitrary term lists (x, y, z) with hashable coordinates
# ---------------------------------------------------------------------------------

def decide_terms(terms: Sequence[Tuple], retained: Set[Tuple], use_x: bool = True,
                 dual_only: bool = False) -> dict:
    """Farkas alternative for an arbitrary support: either weights w(x)+w(y)+w(z)
    equal to 0 on ``retained`` and >= 1 on every other term, or an obstruction
    (mu >= 0 on erased terms, nu on retained terms, equal marginals).  Both sides
    are solved with the exact simplex and checked."""
    terms = [tuple(t) for t in terms]
    tset = set(terms)
    retained = {tuple(t) for t in retained if tuple(t) in tset}
    out = {"terms": len(terms), "retained": len(retained), "realizable": None}
    if dual_only:
        lp = None
    # primal
    lp = LinearProgram() if not dual_only else None
    for (x, y, z) in (terms if not dual_only else []):
        row: Dict[str, int] = {}
        if use_x:
            row[f"xp|{x}"] = 1
            row[f"xn|{x}"] = -1
        row[f"yp|{y}"] = 1
        row[f"yn|{y}"] = -1
        row[f"zp|{z}"] = 1
        row[f"zn|{z}"] = -1
        lp.add(row, "==" if (x, y, z) in retained else ">=", 0 if (x, y, z) in retained else 1)
    if not dual_only:
        lp.set_objective({})
        st, _, sol = lp.solve()
        out["realizable"] = st == "optimal"
    else:
        st = "skipped"
    if st == "optimal":
        def val(p, nn, key):
            return sol.get(f"{p}|{key}", Fraction(0)) - sol.get(f"{nn}|{key}", Fraction(0))
        ok = True
        for (x, y, z) in terms:
            w = (val("xp", "xn", x) if use_x else 0) + val("yp", "yn", y) + val("zp", "zn", z)
            if (x, y, z) in retained:
                ok &= w == 0
            else:
                ok &= w >= 1
        out["weights_verified"] = ok
    # dual
    lp2 = LinearProgram()
    marg: Dict[str, Dict[str, int]] = {}
    for idx, (x, y, z) in enumerate(terms):
        keys = [f"Y|{y}", f"Z|{z}"] + ([f"X|{x}"] if use_x else [])
        if (x, y, z) in retained:
            for k in keys:
                marg.setdefault(k, {})[f"np{idx}"] = -1
                marg[k][f"nn{idx}"] = 1
        else:
            for k in keys:
                marg.setdefault(k, {})[f"m{idx}"] = 1
    for k, row in marg.items():
        lp2.add(row, "==", 0)
    erased_idx = [i for i, t in enumerate(terms) if t not in retained]
    if not erased_idx:
        out["obstruction"] = False
        out["farkas_consistent"] = out["realizable"]
        return out
    lp2.add({f"m{i}": 1 for i in erased_idx}, "==", 1)
    lp2.set_objective({})
    st2, _, sol2 = lp2.solve()
    out["obstruction"] = st2 == "optimal"
    if st2 == "optimal":
        mu = {terms[i]: sol2.get(f"m{i}", Fraction(0)) for i in erased_idx}
        mu = {k: v for k, v in mu.items() if v != 0}
        nu = {}
        for i, t in enumerate(terms):
            if t in retained:
                v = sol2.get(f"np{i}", Fraction(0)) - sol2.get(f"nn{i}", Fraction(0))
                if v != 0:
                    nu[t] = v
        # independent marginal check
        def margs(meas):
            M: Dict[Tuple[str, object], Fraction] = {}
            for (x, y, z), c in meas.items():
                for k in ([("X", x)] if use_x else []) + [("Y", y), ("Z", z)]:
                    M[k] = M.get(k, Fraction(0)) + c
            return {k: v for k, v in M.items() if v != 0}
        out["obstruction_verified"] = (all(v > 0 for v in mu.values()) and margs(mu) == margs(nu))
        out["obstruction_support"] = len(mu)
    out["farkas_consistent"] = (out["realizable"] != out["obstruction"]) if out["realizable"] is not None else None
    if dual_only and out["obstruction"]:
        out["realizable"] = False  # Farkas: an obstruction excludes every weighting
    return out
