"""The dense rectangle packing C(2p-1, np) versus C(p,p)^sigma (x) B_X(n).

Block form.  Put rectangle m on Y_m = [mp, mp+p) and Z_m = [mp+p-1, mp+2p-1)
inside C(a, np), a = 2p-1.  Every pair (j,k) in Y_m x Z_m has 0 <= k-j <= 2p-2,
so the block is a full p x p rectangle: C(p,p) with its output leg on X.  A term
with j in Y_m and k in Z_{m'} has Delta = m'-m in {-1, 0, +1} (for p >= 2), and
only Delta = 0 terms belong to the packing.  Z-coordinates outside the blocks
(the first and last p-1) are dropped by a coordinate restriction.

Results implemented here (all exact):
  * block_form(p, n): the restricted tensor S' with each term's Delta;
  * target(p, n): the shared-X sum of n rectangles, = C(p,p)^sigma (x) B_X(n);
  * two_copy_restriction(p, n): explicit maps realising
        C(p,p)^sigma (x) B_X(n)  <=  C(2p-1, np) (+) C(2p-1, np)
    (the second copy carries the sign (-1)^m on block m of Y and Z, so the
    average keeps exactly the terms with Delta even, i.e. Delta = 0); valid over
    any field of characteristic != 2;
  * cross-term decomposition S' = T + E with E a sum of 2 dot products.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Tuple

from .tensors import Matrix, SparseTensor, convolution, direct_sum, dot_product, tensor_product


def block_form(p: int, n: int):
    """Return (S', deltas): S' is C(2p-1, np) restricted to Z-coordinates
    [p-1, np+p-1) and re-indexed as Y = (m,u) -> m*p+u (u = offset in Y_m),
    Z = (m,v) -> m*p+v (v = offset in Z_m).  deltas maps each term to Delta."""
    a, B = 2 * p - 1, n * p
    src = convolution(a, B)
    S = SparseTensor((a, n * p, n * p))
    deltas: Dict[Tuple[int, int, int], int] = {}
    for (i, j, k), c in src.terms.items():
        kk = k - (p - 1)
        if not (0 <= kk < n * p):
            continue
        m, u = divmod(j, p)
        m2, v = divmod(kk, p)
        S.terms[(i, j, kk)] = c
        deltas[(i, j, kk)] = m2 - m
    return S, deltas


def target(p: int, n: int) -> SparseTensor:
    """Shared-X sum of n rectangles in block-form coordinates:
    sum_m sum_{u,v<p} x_{(p-1) - u + v} y_{(m,u)} z_{(m,v)}.
    (In C(2p-1, np) the x-index is k - j = (p-1) + v - u.)"""
    T = SparseTensor((2 * p - 1, n * p, n * p))
    for m in range(n):
        for u in range(p):
            for v in range(p):
                T.terms[((p - 1) - u + v, m * p + u, m * p + v)] = Fraction(1)
    return T


def target_is_rectangle_tensor(p: int, n: int) -> bool:
    """Check target(p,n) == C(p,p)^sigma (x) B_X(n) up to an explicit coordinate
    permutation: C(p,p)^sigma = sum x_{u'+v} y_{u'} z_v with u' = p-1-u."""
    rect = convolution(p, p).permute_legs("cycle_XZY")  # x_{u+v} y_u z_v
    ref = tensor_product(rect, dot_product(n, 0))        # Y index u*n+m, Z index v*n+m
    T = target(p, n)
    # pull ref back along: X identity; Y (m,u) -> (p-1-u)*n + m; Z (m,v) -> v*n + m
    ymap = [(p - 1 - (y % p)) * n + (y // p) for y in range(n * p)]
    zmap = [(z % p) * n + (z // p) for z in range(n * p)]
    pulled = ref.pullback([list(range(2 * p - 1)), ymap, zmap], (2 * p - 1, n * p, n * p))
    return pulled == T


def retained_part(S: SparseTensor, deltas) -> SparseTensor:
    R = SparseTensor(S.dims)
    for key, c in S.terms.items():
        if deltas[key] == 0:
            R.terms[key] = c
    return R


def two_copy_restriction(p: int, n: int):
    """Return (D, (A,B,C), T): D = C(2p-1,np) (+) C(2p-1,np) and matrices with
    restrict(A,B,C) of D == T = target(p,n).  Exact."""
    a, Bn = 2 * p - 1, n * p
    src = convolution(a, Bn)
    D = direct_sum(src, src)
    dx, dy, dz = src.dims
    A = Matrix(a, 2 * dx, {**{(i, i): 1 for i in range(a)}, **{(i, dx + i): 1 for i in range(a)}})
    Bent, Cent = {}, {}
    for y in range(Bn):
        m = y // p
        Bent[(y, y)] = 1
        Bent[(y, dy + y)] = (-1) ** m
    for z in range(Bn):
        m = z // p
        k = z + (p - 1)                       # source Z coordinate
        Cent[(z, k)] = Fraction(1, 2)
        Cent[(z, dz + k)] = Fraction((-1) ** m, 2)
    B = Matrix(Bn, 2 * dy, Bent)
    C = Matrix(Bn, 2 * dz, Cent)
    return D, (A, B, C), target(p, n)


def verify_two_copy(p: int, n: int) -> bool:
    D, (A, B, C), T = two_copy_restriction(p, n)
    return D.restrict(A, B, C) == T


def cross_terms(p: int, n: int) -> SparseTensor:
    S, deltas = block_form(p, n)
    E = SparseTensor(S.dims)
    for key, c in S.terms.items():
        if deltas[key] != 0:
            E.terms[key] = c
    return E


def power_with_deltas(S: SparseTensor, deltas, N: int):
    """N-fold tensor power with per-factor Delta vectors (coordinates as tuples
    flattened in base dims)."""
    terms = {(): ((), Fraction(1))}
    cur = [((0, 0, 0), (), Fraction(1))]
    dims = (1, 1, 1)
    for _ in range(N):
        nxt = []
        for (x, y, z), dv, c in cur:
            for key, c2 in S.terms.items():
                nxt.append(((x * S.dims[0] + key[0], y * S.dims[1] + key[1], z * S.dims[2] + key[2]),
                            dv + (deltas[key],), c * c2))
        cur = nxt
        dims = tuple(d * e for d, e in zip(dims, S.dims))
    return dims, cur


def exact_degeneration_never_beats_pstar(max_a: int = 40, max_n: int = 200):
    """Exact check: if C(a,B) degenerates to n square rectangles C(p,p)^sigma
    sharing X (a >= 2p-1), then border rank gives a+B-1 >= n(2p-1) and the
    Y-flattening gives B >= np.  For every such (a, p, n, minimal B) in range,
    compare P*(a,B) with n P*(p,p).  Returns the list of violations (expected
    empty: no exact degeneration of this kind can beat the extremal profile)."""
    from .growth import extremal_profile
    viol = []
    checked = 0
    for a in range(3, max_a + 1):
        for p in range(2, (a + 1) // 2 + 1):
            for n in range(1, max_n + 1):
                B = max(n * p, n * (2 * p - 1) - a + 1, 1)
                checked += 1
                if extremal_profile(a, B) < n * extremal_profile(p, p):
                    viol.append((a, p, n, B))
    return viol, checked


def filtered_power_experiment(p: int, n: int, N: int, c: Tuple[int, ...], L: int, dual_only: bool = False):
    """Tensor power S'^{(x)N} of the block form, keep only terms with
    sum_t c_t Delta_t = 0 (mod L) -- what a block-label Fourier filter with L
    copies leaves -- and ask whether ANY monomial weights then keep exactly the
    Delta = 0 terms (the packing T^{(x)N}).  Returns the Farkas decision."""
    from .weights_lp import decide_terms
    S, deltas = block_form(p, n)
    dims, terms = power_with_deltas(S, deltas, N)
    kept = [(key, dv) for key, dv, coef in terms if sum(ci * di for ci, di in zip(c, dv)) % L == 0]
    keys = [k for k, _ in kept]
    retained = {k for k, dv in kept if all(d == 0 for d in dv)}
    surviving_patterns = sorted({dv for _, dv in kept if any(d != 0 for d in dv)})
    res = decide_terms(keys, retained, use_x=True, dual_only=dual_only)
    res.update({"p": p, "n": n, "N": N, "c": list(c), "L": L,
                "surviving_nonzero_patterns": [list(d) for d in surviving_patterns]})
    return res


def koszul_threshold(p: int, n: int, max_extra: int = 40):
    """Smallest B with rank C(2p-1,B)^{wedge (p-1)}_X >= rank (target)^{wedge (p-1)}_X.
    Below it no degeneration C(2p-1, B) => n rectangles C(p,p)^sigma exists."""
    from .border import koszul_rank
    rt = koszul_rank(target(p, n), p - 1, 0)[0]
    start = max(1, n * p - 2 * p)
    for B in range(start, n * (2 * p - 1) + max_extra):
        if koszul_rank(convolution(2 * p - 1, B), p - 1, 0)[0] >= rt:
            return B, rt
    return None, rt


def verdict(p: int, n: int, B: int) -> dict:
    """Exact analysis of 'does C(2p-1, B) degenerate to n rectangles C(p,p)^sigma
    sharing X?' with all supporting computations."""
    from .border import conv_border_rank_certificate, koszul_bound, koszul_rank
    from .sectors import staircase_certificate
    from .verify import verify_certificate
    a = 2 * p - 1
    T = target(p, n)
    rs, _, _ = koszul_rank(convolution(a, B), p - 1, 0)
    rt, _, _ = koszul_rank(T, p - 1, 0)
    thr, _ = koszul_threshold(p, n)
    stair_B = (n - 1) * (2 * p - 1) + p
    stair_ok = verify_certificate(staircase_certificate(a, p, p, n)).ok
    br_src = conv_border_rank_certificate(a, B)
    br_tgt, _ = koszul_bound(T, p - 1, 0)
    out = {
        "question": f"C({a},{B}) => {n} rectangles C({p},{p}) with output on the shared X leg (= C({p},{p})^sigma (x) B_X({n}))",
        "target_dims": list(T.dims), "target_is_rectangle_tensor": target_is_rectangle_tensor(p, n),
        "koszul_s": p - 1, "koszul_rank_source": rs, "koszul_rank_target": rt,
        "degeneration_possible_by_koszul": rs >= rt,
        "koszul_threshold_B": thr, "staircase_B": stair_B, "staircase_certified": stair_ok,
        "staircase_is_optimal": thr == stair_B,
        "max_rectangles_in_source": max(m for m in range(1, n + 1) if (m - 1) * (2 * p - 1) + p <= B) if B >= p else 0,
        "border_rank_source": br_src["border_rank"], "border_rank_target_lower_bound": int(br_tgt),
        "two_copy_restriction_verified": verify_two_copy(p, n) if B == n * p else None,
        "scope": ("Exact statements: (1) Koszul flattening ranks along X cannot increase under a degeneration that keeps "
                  "the target X-concise, so rank_source < rank_target excludes every degeneration and restriction, "
                  "monomial or not; (2) the staircase certificate shows the threshold is attained.  Not covered: "
                  "asymptotic restriction (tensor powers), which is what the character inequality P(a,B) >= n P(p,p) "
                  "actually needs."),
    }
    return out


def koszul_block_types(a: int, h_max: int):
    """All convolution block types that fit a shared X leg of dimension a:
    ('N', m, h): C(m,h), X carries the m-input; ('E', m, h): legs Y,Z exchanged;
    ('R', p, q): output on X (p+q-1 <= a).  Each is built with its X-leg padded
    to dimension a (Koszul ranks are GL(X)-invariant, so only dim matters)."""
    from .border import pad
    out = []
    for m in range(1, a + 1):
        for h in range(1, h_max + 1):
            t = convolution(m, h)                               # x_i y_j z_{i+j}
            out.append((("N", m, h), pad(t, (a,) + t.dims[1:])))
            te = t.permute_legs("swap_YZ")
            out.append((("E", m, h), pad(te, (a,) + te.dims[1:])))
    for p in range(1, a + 1):
        for q in range(p, a + 1):
            if p + q - 1 <= a:
                t = convolution(p, q).permute_legs("cycle_XZY")  # output on X
                out.append((("R", p, q), pad(t, (a,) + t.dims[1:])))
    return out


def koszul_lp(a: int, h_max: int = 12, B: int | None = None, sym_degrees=()):
    """Upper bound on sum P*(blocks) over exact degenerations of C(a,B) into a
    shared-X sum of convolution blocks, by the LP whose resources are the
    additive invariants: Y- and Z-flattening ranks and the X-Koszul ranks for
    s = 1..a-2.  With B=None the per-unit-length (slope) version is solved:
    capacities are the slopes of these invariants in B.  Exact rationals."""
    from .border import koszul_rank, sym_flattening_rank
    from .growth import extremal_profile
    from .lp import LinearProgram
    types = koszul_block_types(a, h_max)
    S_list = list(range(1, a - 1))
    # capacities: fit rank(C(a,B)) = alpha*B + beta from two large B values
    def caps(Bv):
        src = convolution(a, Bv)
        c = {"Y": Bv, "Z": a + Bv - 1}
        for s in S_list:
            c[f"K{s}"] = koszul_rank(src, s, 0)[0]
        for d in sym_degrees:
            c[f"S{d}"] = sym_flattening_rank(src, d, 0)
        return c
    c1, c2 = caps(3 * a + 6), caps(3 * a + 7)
    slope = {k: Fraction(c2[k] - c1[k]) for k in c1}
    cap = slope if B is None else {k: Fraction(v) for k, v in caps(B).items()}
    lp = LinearProgram()
    rows = {k: {} for k in cap}
    obj = {}
    for (kind, u, v), t in types:
        name = f"{kind}_{u}_{v}"
        res = {"Y": t.flattening_rank(1), "Z": t.flattening_rank(2)}
        for s in S_list:
            res[f"K{s}"] = koszul_rank(t, s, 0)[0]
        for d in sym_degrees:
            res[f"S{d}"] = sym_flattening_rank(t, d, 0)
        for k in cap:
            rows[k][name] = res[k]
        obj[name] = -extremal_profile(u, v)
    for k, row in rows.items():
        lp.add(row, "<=", cap[k])
    lp.set_objective(obj)
    st, val, sol = lp.solve()
    best = {k: v for k, v in sol.items() if v != 0} if sol else None
    target = growth_slope(a) if B is None else extremal_profile(a, B)
    return {"a": a, "B": B, "status": st, "lp_max": (-val) if val is not None else None,
            "pstar": target, "beats_pstar": (-val) > target if val is not None else None,
            "optimal_mix": best, "capacities": {k: str(v) for k, v in cap.items()}}


def growth_slope(a: int) -> Fraction:
    from .growth import growth_product
    return growth_product(a)
