"""Exact border (degeneration) constructions for dense rectangle packings in
tensor powers, from two copies of the source.

Mechanism.  If L(x,y,z) = alpha(x) + beta(y) + gamma(z) is additive on the support
of S = (block form)^{(x)N}, vanishes on every cross term, and its values on the
retained terms are reciprocals of a rank-one function w = rho(x) sigma(y) tau(z),
then with rank-one terms

    P_eps = (1/eps) rho e^{eps alpha} . sigma e^{eps beta} . tau e^{eps gamma},
    Q_eps = -(1/eps) rho . sigma . tau,

the sum P_eps + Q_eps = w (e^{eps L} - 1)/eps tends to w L = [retained] on the
support.  Scaling the coordinates of two copies of S by these factors and adding
them is a restriction of S (+) S, so T^{(x)N} is a degeneration of 2 . S^{(x)N}.
(Polynomial version: replace e^{eps a} by 1 + eps a; then P + Q = w L + eps(...)
exactly, a polynomial degeneration of degree <= 2 in eps.)

This module finds L by exact linear algebra for a prescribed sign/rank-one
weight w, and verifies the resulting polynomial degeneration exactly.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .dense import block_form, power_with_deltas


def _support(p: int, n: int, N: int):
    S, deltas = block_form(p, n)
    dims, terms = power_with_deltas(S, deltas, N)
    return S, deltas, dims, terms


def solve_linear(rows: List[Dict[int, Fraction]], rhs: List[Fraction], nvars: int):
    """Exact Gauss-Jordan over Q; returns one solution (free variables = 0) or None."""
    piv_rows = []  # list of (pivot col, row dict, rhs)
    for r, b in zip(rows, rhs):
        r = dict(r)
        b = Fraction(b)
        for col, prow, pb in piv_rows:
            f = r.get(col)
            if f:
                for k, v in prow.items():
                    nv = r.get(k, Fraction(0)) - f * v
                    if nv == 0:
                        r.pop(k, None)
                    else:
                        r[k] = nv
                b -= f * pb
        if not r:
            if b != 0:
                return None
            continue
        col = min(r)
        pv = r[col]
        r = {k: v / pv for k, v in r.items()}
        b = b / pv
        # eliminate col from previous pivot rows (keep reduced form)
        new = []
        for c2, prow, pb in piv_rows:
            f = prow.get(col)
            if f:
                prow = dict(prow)
                for k, v in r.items():
                    nv = prow.get(k, Fraction(0)) - f * v
                    if nv == 0:
                        prow.pop(k, None)
                    else:
                        prow[k] = nv
                pb = pb - f * b
            new.append((c2, prow, pb))
        piv_rows = new + [(col, r, b)]
    x = [Fraction(0)] * nvars
    for col, prow, pb in piv_rows:
        x[col] = pb  # free variables are 0 and every pivot row is fully reduced
    return x


def find_additive(p: int, n: int, N: int, weight: Callable[[Tuple, Tuple, Tuple, Tuple], Fraction]):
    """Find additive L with L = 0 on cross terms and L = 1/weight on retained terms.
    weight(x, y, z, deltas) must be a rank-one function (checked separately)."""
    S, deltas, dims, terms = _support(p, n, N)
    X, Y, Z = dims
    rows, rhs = [], []
    for (x, y, z), dv, coef in terms:
        row = {x: Fraction(1), X + y: Fraction(1), X + Y + z: Fraction(1)}
        rows.append(row)
        if all(d == 0 for d in dv):
            rhs.append(1 / Fraction(weight(x, y, z, dv)))
        else:
            rhs.append(Fraction(0))
    sol = solve_linear(rows, rhs, X + Y + Z)
    if sol is None:
        return None
    return sol[:X], sol[X:X + Y], sol[X + Y:]


def verify_border(p: int, n: int, N: int, alpha, beta, gamma, rho, sigma, tau) -> dict:
    """Exact check of the polynomial degeneration (degree 2 in eps):
       W(eps) = (1/eps)[rho(1+eps alpha) sigma(1+eps beta) tau(1+eps gamma) - rho sigma tau]
             = rho sigma tau [L + eps (alpha beta + beta gamma + gamma alpha) + eps^2 alpha beta gamma].
    Checks on every support term: the eps^0 coefficient equals [retained]."""
    S, deltas, dims, terms = _support(p, n, N)
    ok0 = True
    for (x, y, z), dv, coef in terms:
        w = rho[x] * sigma[y] * tau[z]
        L = alpha[x] + beta[y] + gamma[z]
        want = 1 if all(d == 0 for d in dv) else 0
        if w * L != want:
            ok0 = False
            break
    return {"p": p, "n": n, "N": N, "constant_term_is_target": ok0, "support_terms": len(terms),
            "copies": 2, "degree_in_eps": 2}


def nullspace(rows: List[Dict[int, Fraction]], nvars: int):
    """Exact basis of the null space of a sparse rational matrix (rows as dicts)."""
    piv = {}  # pivot col -> reduced row
    for r in rows:
        r = {k: Fraction(v) for k, v in r.items() if v != 0}
        for col, prow in piv.items():
            f = r.get(col)
            if f:
                for k2, v in prow.items():
                    nv = r.get(k2, Fraction(0)) - f * v
                    if nv == 0:
                        r.pop(k2, None)
                    else:
                        r[k2] = nv
        if not r:
            continue
        col = min(r)
        pv = r[col]
        r = {k2: v / pv for k2, v in r.items()}
        for c2 in list(piv):
            f = piv[c2].get(col)
            if f:
                pr = dict(piv[c2])
                for k2, v in r.items():
                    nv = pr.get(k2, Fraction(0)) - f * v
                    if nv == 0:
                        pr.pop(k2, None)
                    else:
                        pr[k2] = nv
                piv[c2] = pr
        piv[col] = r
    free = [j for j in range(nvars) if j not in piv]
    basis = []
    for fcol in free:
        vec = [Fraction(0)] * nvars
        vec[fcol] = Fraction(1)
        for col, prow in piv.items():
            vec[col] = -prow.get(fcol, Fraction(0))
        basis.append(vec)
    return basis


def linear_slice(p: int, n: int, N: int, rho, sigma):
    """With rho, sigma fixed, solve exactly for additive L (alpha, beta, gamma) and
    eta = 1/tau:  L = 0 on cross terms,  L * rho(x) sigma(y) = eta(z) on retained.
    Returns the null-space basis in the variables (alpha, beta, gamma, eta)."""
    S, deltas, dims, terms = _support(p, n, N)
    X, Y, Z = dims
    nv = X + Y + Z + Z
    rows = []
    for (x, y, z), dv, coef in terms:
        if all(d == 0 for d in dv):
            w = Fraction(rho[x]) * Fraction(sigma[y])
            rows.append({x: w, X + y: w, X + Y + z: w, X + Y + Z + z: Fraction(-1)})
        else:
            rows.append({x: Fraction(1), X + y: Fraction(1), X + Y + z: Fraction(1)})
    return nullspace(rows, nv), dims


def rank_one_reciprocal(values: Dict[Tuple[int, int, int], Fraction]):
    """Given target values on a set of (x, y, z) cells, find rational f, C, A with
    f(x) C(y) A(z) = values[(x,y,z)] on every cell (multiplicative rank one), by
    propagating along the cell graph.  Returns (f, C, A) dicts or None."""
    f, C, A = {}, {}, {}
    cells = list(values.items())
    # BFS over coordinates
    if not cells:
        return f, C, A
    (x0, y0, z0), _ = cells[0]
    f[x0] = Fraction(1); C[y0] = Fraction(1)
    changed = True
    while changed:
        changed = False
        for (x, y, z), v in cells:
            known = (x in f) + (y in C) + (z in A)
            if known == 2:
                if x not in f:
                    f[x] = v / (C[y] * A[z])
                elif y not in C:
                    C[y] = v / (f[x] * A[z])
                else:
                    A[z] = v / (f[x] * C[y])
                changed = True
            elif known == 1 and not changed:
                pass
        if not changed:
            # seed one more unknown coordinate to 1 if some cell still has two unknowns
            for (x, y, z), v in cells:
                known = (x in f) + (y in C) + (z in A)
                if known == 1:
                    if x not in f:
                        f[x] = Fraction(1)
                    elif y not in C:
                        C[y] = Fraction(1)
                    changed = True
                    break
    for (x, y, z), v in cells:
        if (x not in f) or (y not in C) or (z not in A) or f[x] * C[y] * A[z] != v:
            return None
    return f, C, A


def n2_square_certificate():
    """Exact first-order degeneration  T_2^{(x)2} <= 2 . S'^{(x)2}  (S' = block form of C(3,4)).

    Single factor: ell(tau) = 4 - j - k  (j, k = source Y- and Z-indices of the term)
    vanishes exactly on the two cross terms.  L = ell(tau_1) ell(tau_2) is additive:
        L = 16 - 4(j1+j2) - 4(k1+k2) + 2 j1 j2 + 2 k1 k2 - x1 x2,
    and 1/ell is multiplicatively rank one on the retained single-factor terms, so
    w = 1/L is rank one on retained pair terms.  Verifies everything exactly and
    the epsilon-polynomial degeneration on every support term."""
    S, deltas = block_form(2, 2)
    # single-factor data: block-form y index = j (source Y index), z index = k - 1
    def ell(x, y, z):
        return Fraction(4 - y - (z + 1))
    # check single factor: zero on cross, nonzero on retained
    single_ok = all((ell(*key) == 0) == (deltas[key] != 0) for key in S.terms)
    # rank-one reciprocal on retained single-factor terms
    vals = {key: 1 / ell(*key) for key in S.terms if deltas[key] == 0}
    r1 = rank_one_reciprocal(vals)
    if r1 is None:
        return {"ok": False, "reason": "1/ell not rank one"}
    f1, C1, A1 = r1
    # pair support
    dims, terms = power_with_deltas(S, deltas, 2)
    X1, Y1, Z1 = S.dims

    def split(w, base):
        return w // base, w % base

    alpha, beta, gamma = {}, {}, {}
    rho, sigma, tau = {}, {}, {}
    ok_add = True
    ok_target = True
    for (x, y, z), dv, coef in terms:
        x1, x2 = split(x, X1); y1, y2 = split(y, Y1); z1, z2 = split(z, Z1)
        k1, k2 = z1 + 1, z2 + 1
        a = Fraction(-x1 * x2 + 16)
        b = Fraction(2 * y1 * y2 - 4 * (y1 + y2))
        g = Fraction(2 * k1 * k2 - 4 * (k1 + k2))
        alpha[x], beta[y], gamma[z] = a, b, g
        Lval = a + b + g
        prod = ell(x1, y1, z1) * ell(x2, y2, z2)
        if Lval != prod:
            ok_add = False
        retained = all(d == 0 for d in dv)
        if retained:
            rho[x] = f1[x1] * f1[x2]; sigma[y] = C1[y1] * C1[y2]; tau[z] = A1[z1] * A1[z2]
            if rho[x] * sigma[y] * tau[z] * Lval != 1:
                ok_target = False
        elif Lval != 0:
            ok_target = False
    # every coordinate of the support needs rho/sigma/tau (cross-only coordinates may be set to 1)
    for (x, y, z), dv, coef in terms:
        x1, x2 = split(x, X1); y1, y2 = split(y, Y1); z1, z2 = split(z, Z1)
        rho.setdefault(x, f1.get(x1, Fraction(1)) * f1.get(x2, Fraction(1)))
        sigma.setdefault(y, C1.get(y1, Fraction(1)) * C1.get(y2, Fraction(1)))
        tau.setdefault(z, A1.get(z1, Fraction(1)) * A1.get(z2, Fraction(1)))
    # epsilon-polynomial check: W(eps) = rho sigma tau [L + eps e2 + eps^2 e3], constant term = target
    ok_eps = True
    for (x, y, z), dv, coef in terms:
        w = rho[x] * sigma[y] * tau[z]
        const = w * (alpha[x] + beta[y] + gamma[z])
        if const != (1 if all(d == 0 for d in dv) else 0):
            ok_eps = False
    return {"ok": single_ok and ok_add and ok_target and ok_eps,
            "single_factor_ell_vanishes_exactly_on_cross": single_ok,
            "L_equals_product_and_is_additive": ok_add,
            "w_times_L_is_target": ok_target, "eps_constant_term_is_target": ok_eps,
            "support_terms": len(terms), "copies": 2,
            "single_factor_rank_one": {"f": {k: str(v) for k, v in f1.items()},
                                       "C": {k: str(v) for k, v in C1.items()},
                                       "A": {k: str(v) for k, v in A1.items()}}}


def first_order_obstruction(p: int, n: int, N: int) -> dict:
    """Exact test of the first-order (tangential, w e^{eps L} - w) mechanism.

    Computes, over Q, the space V of additive functions L = alpha(x) + beta(y) +
    gamma(z) on supp(S^{(x)N}) that vanish on every cross term, and counts the
    retained terms on which *every* L in V vanishes.  If that count is positive,
    no L in V is non-zero on all retained terms, so w L = [retained] is impossible
    and the first-order two-copy construction does not exist for (p, n, N).
    (It does not exclude two-copy degenerations in which some of the rank-one
    factors vanish; see notes/06.)"""
    S, deltas, (X, Y, Z), terms = _support(p, n, N)
    nv = X + Y + Z
    cross_rows, retained = [], []
    for (x, y, z), dv, _ in terms:
        row = {x: Fraction(1), X + y: Fraction(1), X + Y + z: Fraction(1)}
        if all(d == 0 for d in dv):
            retained.append((x, y, z))
        else:
            cross_rows.append(row)
    basis = nullspace(cross_rows, nv)
    forced = [t for t in retained
              if all(v[t[0]] + v[X + t[1]] + v[X + Y + t[2]] == 0 for v in basis)]
    return {"p": p, "n": n, "N": N, "dim_V": len(basis), "retained": len(retained),
            "cross": len(cross_rows), "forced_zero_retained": len(forced),
            "first_order_possible": len(forced) == 0, "forced_examples": forced[:4]}


def three_copy_square_terms(n: int):
    """Explicit rank-3 restriction  T_n^{(x)2} <= 3 . S^{(x)2}  (p = 2, any n >= 2),
    S = block form of C(3, 2n).  Factor coordinates: x in {0,1,2}, j, k in [0, 2n)
    (block-form Y and Z indices), block m(.) = . // 2, colour col(.) = m(.) mod 2,
    parity par(.) = . mod 2.  Returns three rank-one terms as functions
    (a(x1,x2), b(j1,j2), c(k1,k2)) with values in {0, +1, -1}.

    Copy W: [x2 != 1] . [col(j1) = par(j2)] . [col(k1) != par(k2)]
    Copy 1: [x2 != 2] . [col(j1) = 0] (-1)^{m(j2)} . [col(k1) = 0] (-1)^{m(k2)}
    Copy 2: [x2 != 0] . [col(j1) = 1] (-1)^{m(j2)} . [col(k1) = 1] (-1)^{m(k2)}

    Why it works: adjacent blocks have different colours, so on the support the
    retained indicator R of one factor is A + B with A, B the 'both colour 1' and
    'both colour 0' indicators.  Copies 1 and 2 are B (x) (R_0 + R_1 - K_0) and
    A (x) (R_1 + R_2 - K_2), where R_t is R restricted to x = t and K_t the
    cross terms with x = t; copy W equals B (x) (R_2 + K_0) + A (x) (R_0 + K_2) on the
    support.  The sum is (A + B) (x) R = R (x) R."""
    m = lambda v: v // 2
    col = lambda v: (v // 2) % 2
    par = lambda v: v % 2
    sgn = lambda v: -1 if m(v) % 2 else 1
    W = (lambda x1, x2: int(x2 != 1),
         lambda j1, j2: int(col(j1) == par(j2)),
         lambda k1, k2: int(col(k1) != par(k2)))
    C1 = (lambda x1, x2: int(x2 != 2),
          lambda j1, j2: int(col(j1) == 0) * sgn(j2),
          lambda k1, k2: int(col(k1) == 0) * sgn(k2))
    C2 = (lambda x1, x2: int(x2 != 0),
          lambda j1, j2: int(col(j1) == 1) * sgn(j2),
          lambda k1, k2: int(col(k1) == 1) * sgn(k2))
    return [W, C1, C2]


def verify_three_copy_square(n: int) -> dict:
    """Exact check that the three rank-one terms reproduce [retained] on every
    term of supp(S^{(x)2}) (integer arithmetic).  Since each term is a diagonal
    scaling of one copy of S^{(x)2}, this is a restriction T_n^{(x)2} <= 3 S^{(x)2}."""
    S, deltas = block_form(2, n)
    sup = list(S.terms)
    bad = 0
    count = 0
    for t1 in sup:
        for t2 in sup:
            want = int(deltas[t1] == 0 and deltas[t2] == 0)
            got = sum(a(t1[0], t2[0]) * b(t1[1], t2[1]) * c(t1[2], t2[2])
                      for a, b, c in three_copy_square_terms(n))
            count += 1
            bad += int(got != want)
    return {"n": n, "support_terms": count, "mismatches": bad, "ok": bad == 0}


def three_copy_restriction_from_convolution(n: int) -> dict:
    """Independent exact check of the three-copy theorem, built from scratch out of
    the convolution tensor (not the block-form machinery): with C = C(3, 2n) and the
    target T_n = dense.target(2, n) (n rectangles C(2,2) sharing X), form
    C (x) C and T (x) T with tensors.tensor_product, apply for each of the three
    rank-one terms the linear restriction  X: diag(a),  Y: diag(b),
    Z: (z1, z2) -> (z1 - 1, z2 - 1) then diag(c)  (the Z-shift drops C's two
    outermost outputs), add the three restrictions and compare with T (x) T as
    exact sparse tensors."""
    from .dense import target
    from .tensors import Matrix, convolution, tensor_product
    C = convolution(3, 2 * n)                       # dims (3, 2n, 2n + 2)
    C2 = tensor_product(C, C)
    T = target(2, n)                                # dims (3, 2n, 2n)
    T2 = tensor_product(T, T)
    X1, Y1, Zs, Zt = 3, 2 * n, 2 * n + 2, 2 * n
    total = None
    for a, b, c in three_copy_square_terms(n):
        A = Matrix(X1 * X1, X1 * X1, {(x1 * X1 + x2, x1 * X1 + x2): a(x1, x2)
                                       for x1 in range(X1) for x2 in range(X1)})
        B = Matrix(Y1 * Y1, Y1 * Y1, {(j1 * Y1 + j2, j1 * Y1 + j2): b(j1, j2)
                                       for j1 in range(Y1) for j2 in range(Y1)})
        Cm = Matrix(Zt * Zt, Zs * Zs, {((z1 - 1) * Zt + (z2 - 1), z1 * Zs + z2): c(z1 - 1, z2 - 1)
                                        for z1 in range(1, Zt + 1) for z2 in range(1, Zt + 1)})
        part = C2.restrict(A, B, Cm)
        total = part if total is None else total + part
    total.terms = {k: v for k, v in total.terms.items() if v != 0}
    return {"n": n, "source": f"C(3,{2 * n})^(x)2 (three copies)", "target": f"T_{n}^(x)2",
            "target_terms": len(T2.terms), "equal": total == T2}


def three_copy_square_certificate(n: int = 8) -> dict:
    """Certificate for  T_n^{(x)2} <= 3 . C(3, 2n)^{(x)2}  with explicit integer
    tables (block-form coordinates; the Z map additionally shifts k -> k - 1)."""
    terms = three_copy_square_terms(n)
    tables = []
    for a, b, c in terms:
        tables.append({"a": [[a(x1, x2) for x2 in range(3)] for x1 in range(3)],
                       "b": [[b(j1, j2) for j2 in range(2 * n)] for j1 in range(2 * n)],
                       "c": [[c(k1, k2) for k2 in range(2 * n)] for k1 in range(2 * n)]})
    blockcheck = verify_three_copy_square(n)
    indep = three_copy_restriction_from_convolution(n)
    return {"n": n, "copies": 3, "factors": 2, "block_form_check": blockcheck, "independent_check": indep,
            "ok": bool(blockcheck["ok"] and indep["equal"]), "tables": tables}


def coarse_type_of(x: int, j: int, k: int):
    """Coarse type (x, 2 col(j) + par(j), 2 col(k) + par(k)) of a block-form term (p = 2)."""
    return (x, 2 * ((j // 2) % 2) + j % 2, 2 * ((k // 2) % 2) + k % 2)


def check_coarse_lift(n: int) -> dict:
    """Exact check that every single-factor term of the p = 2 block form of C(3, 2n)
    maps to one of the 12 coarse types with the same retained flag, so any c-term
    solution of the coarse problem lifts to a c-copy restriction for this n."""
    from .numeric_restriction import coarse_types
    table = {(t[0], t[1], t[2]): t[3] for t in coarse_types()}
    S, deltas = block_form(2, n)
    used = set()
    for (x, j, k) in S.terms:
        ct = coarse_type_of(x, j, k)
        if ct not in table or table[ct] != int(deltas[(x, j, k)] == 0):
            return {"n": n, "ok": False, "bad_term": (x, j, k)}
        used.add(ct)
    return {"n": n, "ok": True, "coarse_types_used": len(used)}


def coarse_cyclic_facts() -> dict:
    """The coarse problem is the cyclic one: its support is that of
    Cyc_3 = sum_{x in {0,1,2}, j in Z_4} x_x y_j z_{j+x-1 mod 4}  (the Z_4 group
    tensor with X restricted to the shifts -1, 0, 1), and the retained part is
    exactly T_2 (two rectangles).  Cyc_3 has rank 4 (discrete Fourier transform
    over Z_4; the identity below is exact because every value is a multiple of
    1/4 of a power of i), while T_2 has border rank 6 (Koszul >= 6, two rectangles
    of border rank 3).  Consequence: coarse copy cost -> 1 would give T_2 <~ Cyc_3,
    hence asymptotic rank R~(T_2) <= 4, an instance of Strassen's asymptotic rank
    conjecture for the tight tensor T_2."""
    from .border import flattening_ranks, koszul_bound
    from .dense import target
    from .tensors import SparseTensor
    from .numeric_restriction import coarse_types
    cyc = SparseTensor((3, 4, 4), {(x, j, (j + x - 1) % 4): 1 for x in range(3) for j in range(4)})
    coarse_support = {(t[0], t[1], t[2]) for t in coarse_types()}
    same_support = coarse_support == set(cyc.terms)
    retained = SparseTensor((3, 4, 4), {k: 1 for k in cyc.terms if k[1] // 2 == k[2] // 2})
    T2 = target(2, 2)
    # rank-4 Fourier decomposition, checked entrywise (exact in binary floating point)
    ok_dft = True
    for x in range(3):
        for j in range(4):
            for z in range(4):
                val = sum((1j ** (-k * (x - 1))) * (1j ** (-k * j)) * (1j ** (k * z)) for k in range(4)) / 4
                want = 1.0 if (x, j, z) in cyc.terms else 0.0
                if abs(val - want) != 0.0:
                    ok_dft = False
    kb_T2, _ = koszul_bound(T2, 1, 0)
    return {"coarse_support_is_cyclic": same_support, "retained_part_is_T2": retained == T2,
            "cyc_flattening_ranks": list(flattening_ranks(cyc)), "cyc_rank_4_by_dft": ok_dft,
            "T2_koszul_border_rank_lower_bound": int(kb_T2), "T2_border_rank_upper_bound": 6,
            "ok": bool(same_support and retained == T2 and ok_dft and int(kb_T2) == 6)}


def parity_twist_pair_system(n: int, h):
    """Exact linear system for the two-copy 'parity-twisted pair' ansatz on
    supp(S^{(x)2}), S = block form of C(3, 2n):
        copy 0 = (1/eps) (a + eps a0')(beta h + eps b0')(gamma h + eps c0'),
        copy 1 = (1/eps) (a + eps a1')(h + eps b1')(h + eps c1'),
    with a(x1,x2) = sigma(x1) sigma(x2) (sigma = 1, 0, -1 at x = 0, 1, 2),
    beta(j1,j2) = 1 if j1 odd else -(-1)^{j2},  gamma(k1,k2) = -1 if k1 even else -(-1)^{k2},
    and h(j1, j2) a +-1 (or any nonzero) pattern on the Y- and Z-words.
    On terms with x1, x2 != 1 both parities flip, so beta gamma = -1 and the 1/eps
    terms cancel; the O(1) terms are linear in (a0', a1', b0', b1', c0', c1').
    Returns (rows, rhs, nvars, layout) for solve_linear."""
    S, deltas = block_form(2, n)
    dims, terms = power_with_deltas(S, deltas, 2)
    X1, Y1, Z1 = S.dims
    sig = {0: 1, 1: 0, 2: -1}
    nx, ny, nz = X1 * X1, Y1 * Y1, Z1 * Z1
    off = {'a0': 0, 'a1': nx, 'b0': 2 * nx, 'b1': 2 * nx + ny, 'c0': 2 * nx + 2 * ny, 'c1': 2 * nx + 2 * ny + nz}
    nv = 2 * nx + 2 * ny + 2 * nz

    def beta(y):
        j1, j2 = divmod(y, Y1)
        return 1 if j1 % 2 else -(-1) ** j2

    def gamma(z):
        k1, k2 = divmod(z, Z1)
        return -1 if k1 % 2 == 0 else -(-1) ** k2

    rows, rhs = [], []
    for (x, y, z), dv, _ in terms:
        x1, x2 = divmod(x, X1)
        a = sig[x1] * sig[x2]
        b0, c0, b1, c1 = beta(y) * h(y), gamma(z) * h(z), h(y), h(z)
        if a and b0 * c0 + b1 * c1 != 0:
            raise ValueError('leading terms do not cancel')
        row = {}

        def add(k, v):
            if v:
                row[k] = row.get(k, Fraction(0)) + Fraction(v)
        add(off['a0'] + x, b0 * c0)
        add(off['a1'] + x, b1 * c1)
        add(off['b0'] + y, a * c0)
        add(off['c0'] + z, a * b0)
        add(off['b1'] + y, a * c1)
        add(off['c1'] + z, a * b1)
        rows.append({k: v for k, v in row.items() if v})
        rhs.append(Fraction(int(all(d == 0 for d in dv))))
    return rows, rhs, nv, off


def parity_twist_pair_certificate(n: int, h=None) -> dict:
    """Solve the parity-twisted pair system exactly (h = 1 by default) and verify the
    resulting two-copy degeneration as a polynomial identity in eps on every term
    of supp(S^{(x)2}):  eps * (copy0 + copy1) = eps * target + eps^2 (...) + ...,
    i.e. the 1/eps part cancels and the eps^0 part is [retained]."""
    if h is None:
        h = lambda y: 1
    rows, rhs, nv, off = parity_twist_pair_system(n, h)
    sol = solve_linear(rows, rhs, nv)
    if sol is None:
        return {"n": n, "ok": False, "reason": "linear system unsolvable"}
    S, deltas = block_form(2, n)
    dims, terms = power_with_deltas(S, deltas, 2)
    X1, Y1, Z1 = S.dims
    sig = {0: 1, 1: 0, 2: -1}

    def beta(y):
        j1, j2 = divmod(y, Y1)
        return 1 if j1 % 2 else -(-1) ** j2

    def gamma(z):
        k1, k2 = divmod(z, Z1)
        return -1 if k1 % 2 == 0 else -(-1) ** k2

    def poly_mul(p, q):
        out = {}
        for i, u in p.items():
            for j, v in q.items():
                out[i + j] = out.get(i + j, Fraction(0)) + u * v
        return {k: v for k, v in out.items() if v != 0}

    bad = 0
    for (x, y, z), dv, coef in terms:
        x1, x2 = divmod(x, X1)
        a = Fraction(sig[x1] * sig[x2])
        total = {}
        for i, (bb, cc) in enumerate(((beta(y) * h(y), gamma(z) * h(z)), (h(y), h(z)))):
            ap, bp, cp = sol[off[f'a{i}'] + x], sol[off[f'b{i}'] + y], sol[off[f'c{i}'] + z]
            # eps * copy_i = (a + eps a')(b + eps b')(c + eps c')   (powers of eps as keys)
            p = poly_mul(poly_mul({0: a, 1: ap}, {0: Fraction(bb), 1: bp}), {0: Fraction(cc), 1: cp})
            for k, v in p.items():
                total[k] = total.get(k, Fraction(0)) + v * coef
        want = Fraction(int(all(d == 0 for d in dv)))
        if total.get(0, Fraction(0)) != 0 or total.get(1, Fraction(0)) != want:
            bad += 1
    nz = sum(1 for v in sol if v != 0)
    return {"n": n, "ok": bad == 0, "support_terms": len(terms), "bad_terms": bad,
            "nonzero_corrections": nz, "solution": sol, "layout": off}


def parity_twist_pair_from_convolution(n: int, cert=None) -> dict:
    """Independent check of the two-copy degeneration T_n^{(x)2} <| 2 C(3,2n)^{(x)2}:
    build C(3,2n)^{(x)2} from the convolution tensor, apply for each copy the
    linear maps  X: diag(a + e a_i'),  Y: diag(b_i + e b_i'),  Z: shift by one then
    diag(c_i + e c_i'),  add the two copies, and do this at e = 1, 2, 3, 4 (exact
    rationals).  The maps omit the common factor 1/e, so every entry is a
    polynomial of degree <= 3 in e and the four evaluations determine it; recover the coefficients by exact
    Lagrange interpolation and check that its e^0 coefficient is zero (the 1/e
    terms cancel) and its e^1 coefficient is T_n^{(x)2} = tensor_product(target,
    target), i.e. (1/e)(copy sum) = T_n^{(x)2} + O(e)."""
    from .dense import target
    from .tensors import Matrix, convolution, tensor_product
    cert = cert if cert is not None else parity_twist_pair_certificate(n)
    if "solution" not in cert:
        return {"n": n, "ok": False, "reason": "no block-form certificate"}
    sol, off = cert["solution"], cert["layout"]
    X1, Y1, Zs, Zt = 3, 2 * n, 2 * n + 2, 2 * n
    sig = {0: 1, 1: 0, 2: -1}
    C2 = tensor_product(convolution(3, 2 * n), convolution(3, 2 * n))
    T2 = tensor_product(target(2, n), target(2, n))

    def beta(y):
        j1, j2 = divmod(y, Y1)
        return 1 if j1 % 2 else -(-1) ** j2

    def gamma(z):
        k1, k2 = divmod(z, Zt)
        return -1 if k1 % 2 == 0 else -(-1) ** k2

    def evaluate(e):
        total = None
        for i in range(2):
            A = Matrix(X1 * X1, X1 * X1, {(x, x): sig[x // X1] * sig[x % X1] + e * sol[off[f'a{i}'] + x]
                                           for x in range(X1 * X1)})
            bb = (lambda y: beta(y)) if i == 0 else (lambda y: 1)
            cc = (lambda z: gamma(z)) if i == 0 else (lambda z: 1)
            B = Matrix(Y1 * Y1, Y1 * Y1, {(y, y): bb(y) + e * sol[off[f'b{i}'] + y] for y in range(Y1 * Y1)})
            Cm = Matrix(Zt * Zt, Zs * Zs, {((z1 - 1) * Zt + (z2 - 1), z1 * Zs + z2):
                                           cc((z1 - 1) * Zt + (z2 - 1)) + e * sol[off[f'c{i}'] + (z1 - 1) * Zt + (z2 - 1)]
                                           for z1 in range(1, Zt + 1) for z2 in range(1, Zt + 1)})
            part = C2.restrict(A, B, Cm)
            total = part if total is None else total + part
        return total.terms

    pts = [Fraction(1), Fraction(2), Fraction(3), Fraction(4)]
    vals = [evaluate(e) for e in pts]
    keys = set().union(*[set(v) for v in vals]) | set(T2.terms)
    ok = True
    for key in keys:
        ys = [v.get(key, Fraction(0)) for v in vals]   # evaluate() already returns e * (copy sum): degree <= 3
        # coefficients of e^0 and e^1 by exact Lagrange interpolation
        coeffs = [Fraction(0)] * 4
        for i, xi in enumerate(pts):
            others = [xj for j, xj in enumerate(pts) if j != i]
            denom = Fraction(1)
            for xj in others:
                denom *= (xi - xj)
            # numerator polynomial prod (e - xj): coefficients
            poly = [Fraction(1)]
            for xj in others:
                poly = [Fraction(0)] + poly
                for k in range(len(poly) - 1):
                    poly[k] -= xj * poly[k + 1]
            for k in range(4):
                coeffs[k] += ys[i] * poly[k] / denom
        if coeffs[0] != 0 or coeffs[1] != T2.terms.get(key, Fraction(0)):
            ok = False
            break
    return {"n": n, "ok": ok, "target_terms": len(T2.terms), "entries_checked": len(keys)}


def parity_twist_pair_coarse_solvable() -> bool:
    """The same parity-twisted pair ansatz (h = 1) on the coarse, i.e. cyclic, two-factor
    problem (T_2 inside Cyc_3^{(x)2}): the leading terms still cancel, but the exact
    linear system for the corrections has no solution.  The construction needs the
    boundary of the path of blocks, which the cyclic version wraps around."""
    from .numeric_restriction import coarse_pattern
    rows, target, (nx, ny, nz) = coarse_pattern(2)
    sig = {0: 1, 1: 0, 2: -1}

    def beta(y):
        y1, y2 = divmod(int(y), 4)
        return 1 if y1 % 2 else -(-1) ** (y2 % 2)

    def gamma(z):
        z1, z2 = divmod(int(z), 4)
        return -1 if z1 % 2 == 0 else -(-1) ** (z2 % 2)

    off = {'a0': 0, 'a1': nx, 'b0': 2 * nx, 'b1': 2 * nx + ny, 'c0': 2 * nx + 2 * ny, 'c1': 2 * nx + 2 * ny + nz}
    eqs, rhs = [], []
    for (x, y, z), t in zip(rows, target):
        x1, x2 = divmod(int(x), 3)
        a = sig[x1] * sig[x2]
        b0, c0 = beta(y), gamma(z)
        assert not (a and b0 * c0 + 1 != 0)
        row = {}
        for k, v in ((off['a0'] + int(x), b0 * c0), (off['a1'] + int(x), 1), (off['b0'] + int(y), a * c0),
                     (off['c0'] + int(z), a * b0), (off['b1'] + int(y), a), (off['c1'] + int(z), a)):
            if v:
                row[k] = row.get(k, Fraction(0)) + Fraction(v)
        eqs.append({k: v for k, v in row.items() if v})
        rhs.append(Fraction(int(t)))
    return solve_linear(eqs, rhs, 2 * (nx + ny + nz)) is not None
