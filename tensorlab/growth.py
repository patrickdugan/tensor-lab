"""The inequality engine for the symmetrised profile P(a,b).

Paper conventions (Section 4): for a tensor character lambda with dot-product
exponents (p_X, p_Y, p_Z), t = (p_X+p_Y+p_Z)/3 > 0, and
    P(a,b) = ( prod_{pi in S_3} lambda_pi(C(a,b)) )^{1/(6t)}.
Known facts (paper eqs. 4.5, 4.6, 4.9):
    (S)  P(a,b) = P(b,a) > 0,
    (B)  P(1,b) = b,
    (U)  P(a,b) <= (a+b-1)^{1/t},
    (C)  2 P(a,b) >= P(a,b+1) + P(a,b-1)          (a >= 1, b >= 2),
    (T)  P(a, 3h+a-1) >= 3 P(a,h)                 (a, h >= 1),
and Lemma 5.1: (S),(B),(C),(T) force P(a,a) >= a^{4/3}; with (U) this gives
t <= 3/4 and omega <= 3t <= 9/4.

A shared-leg degeneration C(a,B) ~> (+)_i C(a_i,h_i) (blocks sharing X, matched
sectors on Y and Z) yields, by Corollary 3.2 and symmetrisation,
    P(a,B) >= sup_q e^{H(q)} prod_i P(a_i,h_i)^{q_i} = sum_i P(a_i,h_i),
because P is invariant under leg permutations and the product over S_3 turns
e^{p_X^{(pi)} H(q)} into e^{6t H(q)}.

This module (i) evaluates such candidate inequalities on explicit profiles,
(ii) checks whether they are implied by the known inequalities on a finite grid
via an exact LP, and (iii) reproduces the discrete-growth recurrence.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .lp import LinearProgram

Profile = Callable[[int, int], Fraction]


# ---------------------------------------------------------------------------------
# Explicit profiles
# ---------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def growth_product(m: int) -> Fraction:
    """H_m = prod_{k=1}^{m-1} (1 + 1/(3k)) from the proof of Lemma 5.1."""
    g = Fraction(1)
    for k in range(1, m):
        g *= Fraction(3 * k + 1, 3 * k)
    return g


def extremal_profile(a: int, b: int) -> Fraction:
    """P*(a,b) = (max(a,b) + (min(a,b)-1)/2) * H_{min(a,b)}.

    This is the profile obtained by running the paper's growth argument with
    equality at every step; see notes/extremal_profile.md.  Its diagonal is
    P*(a,a) = (3a-1)/2 * H_a = Theta(a^{4/3})."""
    lo, hi = min(a, b), max(a, b)
    return (Fraction(hi) + Fraction(lo - 1, 2)) * growth_product(lo)


def power_profile(beta_num: int, beta_den: int) -> Profile:
    """A crude comparison profile, used only in tests: P(a,b) = max * min^{beta-1}
    is not rational in general, so this returns min^{beta_num} * max^{beta_den}
    scaled to P(1,b) = b (only meaningful with beta_den = 1).  Kept for experiments."""
    raise NotImplementedError


# ---------------------------------------------------------------------------------
# Known constraints on a finite grid
# ---------------------------------------------------------------------------------

def _key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def var_name(a: int, b: int) -> str:
    a, b = _key(a, b)
    return f"P[{a},{b}]"


class Constraint:
    """sum coeffs * P(.,.) >= rhs   (or ==)."""

    def __init__(self, coeffs: Dict[Tuple[int, int], Fraction], rhs: Fraction, sense: str, tag: str):
        self.coeffs = {_key(*k): Fraction(v) for k, v in coeffs.items()}
        self.rhs = Fraction(rhs)
        self.sense = sense
        self.tag = tag

    def evaluate(self, P: Profile) -> Fraction:
        """Left-hand side minus right-hand side (nonnegative means satisfied for '>=')."""
        return sum((c * P(a, b) for (a, b), c in self.coeffs.items()), Fraction(0)) - self.rhs

    def satisfied(self, P: Profile) -> bool:
        v = self.evaluate(P)
        return v == 0 if self.sense == "==" else v >= 0

    def __repr__(self) -> str:
        terms = " + ".join(f"{c}*P({a},{b})" for (a, b), c in sorted(self.coeffs.items()))
        return f"{terms} {self.sense} {self.rhs}   [{self.tag}]"


def paper_constraints(N: int, include_boundary: bool = True, include_lower_one: bool = True) -> List[Constraint]:
    """All instances of (B), (C), (T) and P >= 1 whose indices lie in [1,N]^2."""
    cons: List[Constraint] = []
    if include_boundary:
        for b in range(1, N + 1):
            cons.append(Constraint({(1, b): 1}, b, "==", f"B:P(1,{b})={b}"))
    for a in range(1, N + 1):
        for b in range(2, N):
            # 2P(a,b) - P(a,b+1) - P(a,b-1) >= 0
            coeffs: Dict[Tuple[int, int], Fraction] = {}
            for k, c in (((a, b), 2), ((a, b + 1), -1), ((a, b - 1), -1)):
                kk = _key(*k)
                coeffs[kk] = coeffs.get(kk, Fraction(0)) + c
            cons.append(Constraint(coeffs, 0, ">=", f"C:a={a},b={b}"))
    for a in range(1, N + 1):
        for h in range(1, N + 1):
            B = 3 * h + a - 1
            if B <= N:
                coeffs = {}
                for k, c in (((a, B), 1), ((a, h), -3)):
                    kk = _key(*k)
                    coeffs[kk] = coeffs.get(kk, Fraction(0)) + c
                cons.append(Constraint(coeffs, 0, ">=", f"T:a={a},h={h}"))
    if include_lower_one:
        for a in range(1, N + 1):
            for b in range(a, N + 1):
                cons.append(Constraint({(a, b): 1}, 1, ">=", f"L:P({a},{b})>=1"))
    return cons


def check_profile(P: Profile, N: int, constraints: Optional[Sequence[Constraint]] = None):
    """Evaluate every constraint on the grid; return (violations, tight) lists."""
    cons = paper_constraints(N) if constraints is None else constraints
    violations, tight = [], []
    for c in cons:
        v = c.evaluate(P)
        if c.sense == "==":
            if v != 0:
                violations.append((c, v))
        else:
            if v < 0:
                violations.append((c, v))
            elif v == 0:
                tight.append(c)
    return violations, tight


# ---------------------------------------------------------------------------------
# Candidate inequalities from degenerations
# ---------------------------------------------------------------------------------

class Candidate:
    """P(source) >= sum_i P(block_i) for a shared-leg degeneration."""

    def __init__(self, source: Tuple[int, int], blocks: Sequence[Tuple[int, int]], origin: str = ""):
        self.source = _key(*source)
        self.blocks = sorted(_key(*b) for b in blocks)
        self.origin = origin

    def constraint(self) -> Constraint:
        coeffs: Dict[Tuple[int, int], Fraction] = {self.source: Fraction(1)}
        for b in self.blocks:
            coeffs[b] = coeffs.get(b, Fraction(0)) - 1
        return Constraint(coeffs, 0, ">=", f"cand:{self}")

    def slack(self, P: Profile) -> Fraction:
        return P(*self.source) - sum((P(*b) for b in self.blocks), Fraction(0))

    def __repr__(self) -> str:
        rhs = " + ".join(f"P({a},{b})" for a, b in self.blocks)
        return f"P({self.source[0]},{self.source[1]}) >= {rhs}"

    def key(self) -> Tuple:
        return (self.source, tuple(self.blocks))


def lp_minimum(objective: Dict[Tuple[int, int], Fraction], N: int,
               extra: Sequence[Constraint] = (), max_pivots: int = 200000):
    """Minimise sum objective[(a,b)] * P(a,b) over profiles satisfying the known
    constraints on the grid [1,N]^2 (plus ``extra``).  Returns (status, value, profile)."""
    # Variables: x(a,b) = P(a,b) - 1 >= 0 for 2 <= a <= b <= N; P(1,b) = b is a constant.
    # This makes the concavity rows feasible at x = 0 (no artificial variables),
    # so Phase 1 only has to repair the tripling rows.
    lp = LinearProgram()

    def transform(coeffs: Dict[Tuple[int, int], Fraction]):
        row: Dict[str, Fraction] = {}
        const = Fraction(0)
        for (a, b), v in coeffs.items():
            a, b = _key(a, b)
            if a == 1:
                const += v * b
            else:
                row[var_name(a, b)] = row.get(var_name(a, b), Fraction(0)) + v
                const += v  # from P = 1 + x
        return row, const

    for c in list(paper_constraints(N, include_boundary=False, include_lower_one=False)) + list(extra):
        row, const = transform(c.coeffs)
        lp.add(row, c.sense, c.rhs - const)
    orow, oconst = transform(objective)
    lp.set_objective(orow)
    status, value, sol = lp.solve(max_pivots=max_pivots)
    if status != "optimal":
        return status, None, None
    prof = {(1, b): Fraction(b) for b in range(1, N + 1)}
    for name, v in sol.items():
        prof[_parse(name)] = v + 1
    return status, value + oconst, prof


def _parse(name: str) -> Tuple[int, int]:
    inner = name[name.index("[") + 1:name.index("]")]
    a, b = inner.split(",")
    return int(a), int(b)


def implied_on_grid(cand: Candidate, N: int, extra: Sequence[Constraint] = ()):
    """Minimum of P(source) - sum P(blocks) over the grid-feasible set.
    A nonnegative minimum proves the candidate is implied by the known
    inequalities (restricted to the grid); a negative minimum exhibits a
    grid-feasible profile violating it (which is NOT a disproof: the full
    infinite system might still imply it)."""
    obj = dict(cand.constraint().coeffs)
    return lp_minimum(obj, N, extra)


def diagonal_lower_bounds(N: int, A_values: Iterable[int], extra: Sequence[Constraint] = ()):
    """Exact LP minimum of P(A,A) on the grid, for each A."""
    out = {}
    for A in A_values:
        status, value, prof = lp_minimum({(A, A): Fraction(1)}, N, extra)
        out[A] = (status, value, prof)
    return out


# ---------------------------------------------------------------------------------
# The discrete growth recurrence (Lemma 5.1), reproduced exactly
# ---------------------------------------------------------------------------------

def lemma_5_1_bounds(A: int) -> List[Tuple[int, Fraction, Fraction]]:
    """For a = 1..A return (a, H_a, D_a lower bound (3a-1)/2 * H_a).
    The paper proves D_a >= a H_a and H_a^3 >= a, hence D_a >= a^{4/3}."""
    rows = []
    for a in range(1, A + 1):
        H = growth_product(a)
        rows.append((a, H, Fraction(3 * a - 1, 2) * H))
    return rows


def cube_check(A: int) -> bool:
    """H_a^3 >= a for a = 1..A (the elementary inequality (1+1/(3m))^3 >= 1+1/m)."""
    return all(growth_product(a) ** 3 >= a for a in range(1, A + 1))


def exponent_transfer(beta: Fraction) -> Fraction:
    """If a^beta <= P(a,a) <= (2a-1)^{1/t} for all a then t <= 1/beta and
    omega <= 3t <= 3/beta.  Returns 3/beta.  (The hypotheses are the paper's (U)
    and normalisation; see Lean/TensorLab/ExponentTransfer.lean for the proof.)"""
    return Fraction(3) / Fraction(beta)


# ---------------------------------------------------------------------------------
# Further true inequalities and hypothetical ones (what-if analysis)
# ---------------------------------------------------------------------------------

def kronecker_constraints(N: int) -> List[Constraint]:
    """Kronecker substitution: C(a,b) (x) C(a',b') is a coordinate sub-tensor of
    C(a + n(a'-1), b + n(b'-1)) with n = a + b - 1, and P is multiplicative on
    tensor products, so P(a + n(a'-1), b + n(b'-1)) >= P(a,b) P(a',b').  These
    are *not* linear in P; they are evaluated on explicit profiles only."""
    out = []
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            n = a + b - 1
            for a2 in range(1, N + 1):
                for b2 in range(1, N + 1):
                    A, Bb = a + n * (a2 - 1), b + n * (b2 - 1)
                    if A <= N and Bb <= N:
                        out.append(((A, Bb), (a, b), (a2, b2)))
    return out


def check_kronecker(P: Profile, N: int):
    """Return violations of P(A,B) >= P(a,b) P(a',b') on the grid."""
    viol, tight, count = [], [], 0
    for (A, Bb), (a, b), (a2, b2) in kronecker_constraints(N):
        count += 1
        lhs, rhs = P(A, Bb), P(a, b) * P(a2, b2)
        if lhs < rhs:
            viol.append(((A, Bb), (a, b), (a2, b2), lhs, rhs))
        elif lhs == rhs:
            tight.append(((A, Bb), (a, b), (a2, b2)))
    return viol, tight, count


def staircase_candidates(N: int, double: bool = False) -> List[Candidate]:
    """Inequalities P(a,B) >= n P(p,q) from staircases of rectangles with the
    output leg on X (realisable, see sectors.staircase_layout), or the
    *hypothetical* double staircase using both interleaved families
    (not known to be realisable; cross terms form a cycle for weights)."""
    out = []
    for a in range(1, N + 1):
        for p in range(1, a + 1):
            for q in range(1, a + 1):
                if p + q - 1 > a:
                    continue
                for n in range(2, N + 1):
                    B = n * (p + q) - q
                    if B > N:
                        break
                    blocks = [(p, q)] * n
                    if double:
                        # second staircase: Y' = Z-ranges of the first, Z' = Y-ranges of the next block
                        blocks = [(p, q)] * n + [(q, p)] * (n - 1)
                    out.append(Candidate((a, B), blocks, origin="staircase" + ("-double(hypothetical)" if double else "")))
    return out


def check_counting_lemma(N: int):
    """Exact check, for 1 <= p, q <= N, of the inequality behind the counting
    obstruction: P*(p,q) <= (q + (p-1)/2) H_p (row value below the linear
    continuation of its row).  Returns the list of violations (expected empty)
    and the list of equality cases.  Proved in Lean: Extremal.P_le_row_line."""
    viol, eq = [], []
    for p in range(1, N + 1):
        for q in range(1, N + 1):
            lhs = extremal_profile(p, q)
            rhs = (Fraction(q) + Fraction(p - 1, 2)) * growth_product(p)
            if lhs > rhs:
                viol.append((p, q))
            elif lhs == rhs:
                eq.append((p, q))
    return viol, eq


def rectangle_density_excess(a: int, p: int, q: int) -> Fraction:
    """P*(p,q) - (p+q)/2 * H_a: positive when a rectangle C(p,q) (output on the
    shared leg, p+q-1 <= a) beats the per-coordinate value of the extremal
    profile of C(a,B).  The counting obstruction does not apply to such blocks."""
    return extremal_profile(p, q) - Fraction(p + q, 2) * growth_product(a)


def whatif_recurrence(N: int, dense_rectangles: bool):
    """Floating-point what-if analysis of the extremal growth recurrence.

    Paper constraints in slope form: g_{a+1} >= 2(D_a + g_a)/(3a) (tripling and the
    diagonal step), D_{a+1} = D_a + g_a + g_{a+1}, g nondecreasing.  With
    ``dense_rectangles`` the *hypothetical* inequality P(2p-1, np) >= n P(p,p) for
    all n (slope form g_{2p-1} >= D_p / p) is added.  Returns the local exponents
    log2(D_{2x}/D_x) for x = 2^8 .. N/2."""
    import math
    D = [0.0, 1.0]
    g = [0.0, 1.0]
    best = 0.0
    for a in range(1, N):
        cand = 2 * (D[a] + g[a]) / (3 * a)
        if dense_rectangles:
            p = (a + 2) // 2
            best = max(best, D[p] / p)
            cand = max(cand, best)
        cand = max(cand, g[a])
        g.append(cand)
        D.append(D[a] + g[a] + cand)
    out = []
    x = 256
    while 2 * x <= N - 1:
        out.append((2 * x, math.log(D[2 * x] / D[x]) / math.log(2)))
        x *= 2
    return out
