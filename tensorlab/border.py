"""Exact border-rank bounds.

Lower bounds: Koszul flattenings (Landsberg–Ottaviani).  For T in A (x) B (x) C
and 0 <= s < dim A, the map

    T^{wedge s}_A : Lambda^s A (x) B^*  ->  Lambda^{s+1} A (x) C,
    e_I (x) beta  |->  sum_{a,c} T(a, beta, c) (e_a wedge e_I) (x) c

has rank at most binom(dim A - 1, s) on a rank-one tensor, so

    border rank(T) >= rank(T^{wedge s}_A) / binom(dim A - 1, s).

For dim A = 3, s = 1 this is Strassen's equation.  Border rank does not
increase under restriction or degeneration, so this obstructs degenerations.

Upper bounds: explicit rank decompositions, checked exactly.  For C(a,B),
evaluation at a+B-1 distinct rational points and Lagrange interpolation gives a
decomposition of length a+B-1; the Z-flattening has rank a+B-1, so
border rank = rank = a+B-1.

All arithmetic is exact (Fraction).
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import comb
from typing import Dict, List, Sequence, Tuple

from .tensors import SparseTensor, convolution, rank_of_rows


def koszul_rank(T: SparseTensor, s: int, leg: int = 0) -> Tuple[int, int, int]:
    """Exact rank of the Koszul flattening T^{wedge s} along ``leg`` (the 'A' leg).
    Returns (rank, number of rows, number of columns)."""
    perm = {0: (0, 1, 2), 1: (1, 0, 2), 2: (2, 0, 1)}[leg]
    alpha = T.dims[perm[0]]
    beta = T.dims[perm[1]]
    gamma = T.dims[perm[2]]
    subsets_s = list(combinations(range(alpha), s))
    col_index = {(I, b): n for n, (I, b) in enumerate((I, b) for I in subsets_s for b in range(beta))}
    # rows indexed by (J, c) with |J| = s+1; build columns as sparse dicts row -> value
    by_ab: Dict[Tuple[int, int], List[Tuple[int, Fraction]]] = {}
    for key, v in T.terms.items():
        a, b, c = key[perm[0]], key[perm[1]], key[perm[2]]
        by_ab.setdefault((a, b), []).append((c, v))
    cols: List[Dict[Tuple[Tuple[int, ...], int], Fraction]] = []
    for I in subsets_s:
        Iset = set(I)
        for b in range(beta):
            col: Dict[Tuple[Tuple[int, ...], int], Fraction] = {}
            for a in range(alpha):
                if a in Iset or (a, b) not in by_ab:
                    continue
                sign = -1 if sum(1 for i in I if i < a) % 2 else 1
                J = tuple(sorted(I + (a,)))
                for c, v in by_ab[(a, b)]:
                    key = (J, c)
                    nv = col.get(key, Fraction(0)) + sign * v
                    if nv == 0:
                        col.pop(key, None)
                    else:
                        col[key] = nv
            cols.append(col)
    r = rank_of_rows(cols)  # rank of the column set = rank of the matrix
    return r, comb(alpha, s + 1) * gamma, comb(alpha, s) * beta


def koszul_bound(T: SparseTensor, s: int, leg: int = 0) -> Tuple[Fraction, int]:
    """Border-rank lower bound rank/binom(dim A - 1, s), rounded up, and the rank."""
    r, _, _ = koszul_rank(T, s, leg)
    alpha = T.dims[leg]
    denom = comb(alpha - 1, s)
    bound = -(-r // denom)  # ceil
    return Fraction(bound), r


def flattening_ranks(T: SparseTensor) -> Tuple[int, int, int]:
    return tuple(T.flattening_rank(l) for l in range(3))  # type: ignore[return-value]


def interpolation_decomposition(a: int, B: int, points: Sequence[int] | None = None):
    """Rank-(a+B-1) decomposition of C(a,B): list of (u, v, w) with
    C(a,B) = sum_r u_r (x) v_r (x) w_r, from evaluation at K = a+B-1 points and
    the inverse Vandermonde matrix.  Exact rationals."""
    K = a + B - 1
    pts = list(points) if points is not None else list(range(K))
    if len(set(pts)) != K:
        raise ValueError("need a+B-1 distinct points")
    # Vandermonde V[r][k] = t_r^k ; p(t_r) = sum_k V[r][k] p_k ; p_k = sum_r W[k][r] p(t_r) with W = V^{-1}
    V = [[Fraction(t) ** k for k in range(K)] for t in pts]
    W = _invert(V)
    terms = []
    for r, t in enumerate(pts):
        u = [Fraction(t) ** i for i in range(a)]
        v = [Fraction(t) ** j for j in range(B)]
        w = [W[k][r] for k in range(K)]
        terms.append((u, v, w))
    return terms


def tensor_from_decomposition(dims, terms) -> SparseTensor:
    T = SparseTensor(dims)
    for u, v, w in terms:
        for i, ui in enumerate(u):
            if ui == 0:
                continue
            for j, vj in enumerate(v):
                if vj == 0:
                    continue
                uv = ui * vj
                for k, wk in enumerate(w):
                    if wk != 0:
                        T.add_term((i, j, k), uv * wk)
    return T


def _invert(M: List[List[Fraction]]) -> List[List[Fraction]]:
    n = len(M)
    A = [list(row) + [Fraction(1 if i == j else 0) for j in range(n)] for i, row in enumerate(M)]
    for col in range(n):
        piv = next(r for r in range(col, n) if A[r][col] != 0)
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        A[col] = [x / pv for x in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [x - f * y for x, y in zip(A[r], A[col])]
    return [row[n:] for row in A]


def conv_border_rank_certificate(a: int, B: int) -> dict:
    """Exact certificate that border rank C(a,B) = a+B-1: a verified decomposition
    of that length and the Z-flattening rank a+B-1."""
    terms = interpolation_decomposition(a, B)
    T = tensor_from_decomposition((a, B, a + B - 1), terms)
    ok = T == convolution(a, B)
    zr = convolution(a, B).flattening_rank(2)
    return {"a": a, "B": B, "decomposition_length": len(terms), "decomposition_verified": ok,
            "z_flattening_rank": zr, "border_rank": a + B - 1 if ok and zr == a + B - 1 else None}


def orbit_tangent_dimension(T: SparseTensor) -> int:
    """Exact dimension of the tangent space of the GL(X) x GL(Y) x GL(Z) orbit of T
    (in its own ambient space): rank of (a, b, c) |-> (a.1.1 + 1.b.1 + 1.1.c) T.
    If T' lies in the orbit closure of T but not in the orbit, then
    dim orbit(T') < dim orbit(T)."""
    X, Y, Z = T.dims
    cols = []
    # a in gl(X): (a T)[i',j,k] = sum_i a[i',i] T[i,j,k]
    by_i, by_j, by_k = {}, {}, {}
    for (i, j, k), c in T.terms.items():
        by_i.setdefault(i, []).append((j, k, c))
        by_j.setdefault(j, []).append((i, k, c))
        by_k.setdefault(k, []).append((i, j, c))
    for ip in range(X):
        for i in range(X):
            col = {}
            for j, k, c in by_i.get(i, ()):
                col[(ip, j, k)] = col.get((ip, j, k), 0) + c
            cols.append({key: v for key, v in col.items() if v != 0})
    for jp in range(Y):
        for j in range(Y):
            col = {}
            for i, k, c in by_j.get(j, ()):
                col[(i, jp, k)] = col.get((i, jp, k), 0) + c
            cols.append({key: v for key, v in col.items() if v != 0})
    for kp in range(Z):
        for k in range(Z):
            col = {}
            for i, j, c in by_k.get(k, ()):
                col[(i, j, kp)] = col.get((i, j, kp), 0) + c
            cols.append({key: v for key, v in col.items() if v != 0})
    return rank_of_rows([c for c in cols if c])


def pad(T: SparseTensor, dims) -> SparseTensor:
    P = SparseTensor(dims)
    P.terms.update(T.terms)
    return P


def sym_flattening_rank(T: SparseTensor, s: int, leg: int = 0) -> int:
    """Rank of the symmetric (Young) flattening Sym^s A (x) B^* -> Sym^{s+1} A (x) C,
    x^I (x) beta |-> sum_{a,c} T(a,beta,c) x_a x^I (x) c.  Functorial in A (for
    surjective maps), additive over shared-A block sums, monotone under
    degeneration onto A-concise targets."""
    from itertools import combinations_with_replacement
    perm = {0: (0, 1, 2), 1: (1, 0, 2), 2: (2, 0, 1)}[leg]
    alpha, beta = T.dims[perm[0]], T.dims[perm[1]]
    by_ab: Dict[Tuple[int, int], List[Tuple[int, Fraction]]] = {}
    for key, v in T.terms.items():
        by_ab.setdefault((key[perm[0]], key[perm[1]]), []).append((key[perm[2]], v))
    cols = []
    for I in combinations_with_replacement(range(alpha), s):
        for b in range(beta):
            col: Dict[Tuple[Tuple[int, ...], int], Fraction] = {}
            for a in range(alpha):
                for c, v in by_ab.get((a, b), ()):
                    key = (tuple(sorted(I + (a,))), c)
                    nv = col.get(key, Fraction(0)) + v
                    if nv == 0:
                        col.pop(key, None)
                    else:
                        col[key] = nv
            if col:
                cols.append(col)
    return rank_of_rows(cols)
