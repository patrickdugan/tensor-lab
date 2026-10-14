"""Exact sparse trilinear tensors over the rationals.

A tensor is a finite trilinear form  sum_{(i,j,k)} c_{ijk} x_i y_j z_k  with
rational coefficients, stored as a dictionary from index triples to
``fractions.Fraction``.  Legs are numbered 0 (X), 1 (Y), 2 (Z).  Everything
here is exact; no floating point is used anywhere in the package.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

Index = Tuple[int, int, int]
LEG_NAMES = ("X", "Y", "Z")

# The six leg permutations, written as the tuple (p0, p1, p2) meaning that
# new leg t carries old leg p_t.  "swap_YZ" = (0, 2, 1) exchanges Y and Z.
PERMUTATIONS: Dict[str, Tuple[int, int, int]] = {
    "identity": (0, 1, 2),
    "swap_YZ": (0, 2, 1),
    "swap_XY": (1, 0, 2),
    "swap_XZ": (2, 1, 0),
    "cycle_XYZ": (1, 2, 0),   # new X = old Y, new Y = old Z, new Z = old X
    "cycle_XZY": (2, 0, 1),
}


def _frac(c) -> Fraction:
    if isinstance(c, Fraction):
        return c
    if isinstance(c, bool):
        raise TypeError("bool is not a coefficient")
    if isinstance(c, int):
        return Fraction(c)
    if isinstance(c, str):
        return Fraction(c)
    raise TypeError(f"unsupported coefficient type {type(c)!r}")


class SparseTensor:
    """Exact sparse trilinear form on coordinate spaces of sizes ``dims``."""

    __slots__ = ("dims", "terms")

    def __init__(self, dims: Sequence[int], terms: Mapping[Index, object] | None = None):
        if len(dims) != 3:
            raise ValueError("a tensor has exactly three legs")
        self.dims: Tuple[int, int, int] = tuple(int(d) for d in dims)  # type: ignore[assignment]
        if any(d < 0 for d in self.dims):
            raise ValueError("dimensions must be nonnegative")
        self.terms: Dict[Index, Fraction] = {}
        if terms:
            for key, c in terms.items():
                self[key] = c

    # -- basic container protocol -------------------------------------------------
    def _check(self, key: Index) -> Index:
        i, j, k = (int(key[0]), int(key[1]), int(key[2]))
        if not (0 <= i < self.dims[0] and 0 <= j < self.dims[1] and 0 <= k < self.dims[2]):
            raise IndexError(f"index {key} out of range for dims {self.dims}")
        return (i, j, k)

    def __getitem__(self, key: Index) -> Fraction:
        return self.terms.get(self._check(key), Fraction(0))

    def __setitem__(self, key: Index, c) -> None:
        key = self._check(key)
        c = _frac(c)
        if c == 0:
            self.terms.pop(key, None)
        else:
            self.terms[key] = c

    def add_term(self, key: Index, c) -> None:
        self[key] = self[key] + _frac(c)

    def __len__(self) -> int:
        return len(self.terms)

    @property
    def nnz(self) -> int:
        return len(self.terms)

    def support(self) -> Iterable[Index]:
        return sorted(self.terms)

    def is_zero(self) -> bool:
        return not self.terms

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SparseTensor):
            return NotImplemented
        return self.dims == other.dims and self.terms == other.terms

    def __hash__(self):  # tensors are mutable; hashing by identity would be a trap
        raise TypeError("SparseTensor is unhashable")

    def __repr__(self) -> str:
        return f"SparseTensor(dims={self.dims}, nnz={self.nnz})"

    def copy(self) -> "SparseTensor":
        t = SparseTensor(self.dims)
        t.terms = dict(self.terms)
        return t

    # -- algebra ----------------------------------------------------------------------
    def __add__(self, other: "SparseTensor") -> "SparseTensor":
        if self.dims != other.dims:
            raise ValueError("cannot add tensors of different dims")
        t = self.copy()
        for key, c in other.terms.items():
            t.add_term(key, c)
        return t

    def __sub__(self, other: "SparseTensor") -> "SparseTensor":
        return self + other.scale(-1)

    def scale(self, c) -> "SparseTensor":
        c = _frac(c)
        t = SparseTensor(self.dims)
        if c != 0:
            t.terms = {key: v * c for key, v in self.terms.items()}
        return t

    def permute_legs(self, perm: str | Tuple[int, int, int]) -> "SparseTensor":
        """Return the tensor with legs permuted; new leg t carries old leg perm[t]."""
        p = PERMUTATIONS[perm] if isinstance(perm, str) else tuple(perm)
        dims = tuple(self.dims[p[t]] for t in range(3))
        t = SparseTensor(dims)
        for key, c in self.terms.items():
            t.terms[(key[p[0]], key[p[1]], key[p[2]])] = c
        return t

    def pullback(self, maps: Sequence[Sequence[int]], dims: Sequence[int]) -> "SparseTensor":
        """Coordinate pullback: new tensor on ``dims`` whose entry at (i',j',k') is
        self[maps[0][i'], maps[1][j'], maps[2][k']].  The maps are arbitrary
        functions of coordinates (lists indexed by the new coordinate)."""
        t = SparseTensor(dims)
        inv = []
        for leg in range(3):
            d: Dict[int, list] = {}
            if len(maps[leg]) != dims[leg]:
                raise ValueError("map length must equal the new dimension")
            for new, old in enumerate(maps[leg]):
                d.setdefault(int(old), []).append(new)
            inv.append(d)
        for (i, j, k), c in self.terms.items():
            for i2 in inv[0].get(i, ()):
                for j2 in inv[1].get(j, ()):
                    for k2 in inv[2].get(k, ()):
                        t.terms[(i2, j2, k2)] = c
        return t

    def restrict_to_support(self, subsets: Sequence[Sequence[int]]) -> "SparseTensor":
        """Keep only terms whose coordinates lie in the given subsets (one per leg),
        re-indexed by position within each subset."""
        maps = [list(s) for s in subsets]
        return self.pullback(maps, [len(m) for m in maps])

    def sub_tensor(self, subsets: Sequence[Sequence[int] | None]) -> "SparseTensor":
        """Zero out every term outside the given coordinate subsets, keeping the
        ambient coordinate spaces.  ``None`` means the whole leg."""
        sets = [None if s is None else set(int(x) for x in s) for s in subsets]
        t = SparseTensor(self.dims)
        for key, c in self.terms.items():
            if all(sets[l] is None or key[l] in sets[l] for l in range(3)):
                t.terms[key] = c
        return t

    def restrict(self, A: "Matrix", B: "Matrix", C: "Matrix") -> "SparseTensor":
        """Linear restriction: substitute x_i -> sum_{i'} A[i', i] X_{i'} etc.
        Each matrix is a dict {(new, old): coefficient} with shape given by
        ``A.rows`` (new dimension) and ``A.cols`` (old dimension)."""
        if (A.cols, B.cols, C.cols) != self.dims:
            raise ValueError("matrix column counts must match the tensor dims")
        t = SparseTensor((A.rows, B.rows, C.rows))
        colA, colB, colC = A.by_column(), B.by_column(), C.by_column()
        for (i, j, k), c in self.terms.items():
            for i2, a in colA.get(i, ()):
                ca = c * a
                for j2, b in colB.get(j, ()):
                    cab = ca * b
                    for k2, cc in colC.get(k, ()):
                        t.add_term((i2, j2, k2), cab * cc)
        return t

    def flattening_rank(self, leg: int) -> int:
        """Rank of the matrix obtained by placing ``leg`` against the other two."""
        rows: Dict[int, Dict[Tuple[int, int], Fraction]] = {}
        for key, c in self.terms.items():
            r = key[leg]
            col = tuple(key[l] for l in range(3) if l != leg)
            rows.setdefault(r, {})[col] = c  # type: ignore[index]
        return rank_of_rows(list(rows.values()))

    def to_json(self) -> dict:
        return {
            "dims": list(self.dims),
            "terms": [[i, j, k, str(c)] for (i, j, k), c in sorted(self.terms.items())],
        }

    @classmethod
    def from_json(cls, data: Mapping) -> "SparseTensor":
        t = cls(data["dims"])
        for i, j, k, c in data["terms"]:
            t[(i, j, k)] = Fraction(c)
        return t


class Matrix:
    """Sparse exact matrix used for linear restrictions: entries {(row, col): c}."""

    def __init__(self, rows: int, cols: int, entries: Mapping[Tuple[int, int], object] | None = None):
        self.rows, self.cols = int(rows), int(cols)
        self.entries: Dict[Tuple[int, int], Fraction] = {}
        if entries:
            for (r, c), v in entries.items():
                v = _frac(v)
                if v != 0:
                    if not (0 <= r < self.rows and 0 <= c < self.cols):
                        raise IndexError((r, c))
                    self.entries[(int(r), int(c))] = v

    @classmethod
    def identity(cls, n: int) -> "Matrix":
        return cls(n, n, {(i, i): 1 for i in range(n)})

    @classmethod
    def from_map(cls, new_dim: int, old_dim: int, f: Mapping[int, int] | Sequence[int]) -> "Matrix":
        """Zero-one matrix with a single 1 in column old=f[new] for each new row."""
        ent = {}
        items = f.items() if isinstance(f, Mapping) else enumerate(f)
        for new, old in items:
            ent[(int(new), int(old))] = 1
        return cls(new_dim, old_dim, ent)

    def by_column(self) -> Dict[int, list]:
        d: Dict[int, list] = {}
        for (r, c), v in self.entries.items():
            d.setdefault(c, []).append((r, v))
        return d


def rank_of_rows(rows: list) -> int:
    """Exact rank of a list of sparse rows (dicts column -> Fraction) by elimination."""
    pivots: Dict[object, Dict[object, Fraction]] = {}
    rank = 0
    for row in rows:
        r = dict(row)
        while r:
            col = min(r, key=repr)
            if col in pivots:
                p = pivots[col]
                f = r[col] / p[col]
                for c2, v in p.items():
                    nv = r.get(c2, Fraction(0)) - f * v
                    if nv == 0:
                        r.pop(c2, None)
                    else:
                        r[c2] = nv
            else:
                pivots[col] = r
                rank += 1
                break
    return rank


# ---------------------------------------------------------------------------------
# Tensor families
# ---------------------------------------------------------------------------------

def convolution(a: int, b: int) -> SparseTensor:
    """C(a,b) = sum_{i<a, j<b} x_i y_j z_{i+j}: polynomial multiplication of a
    degree-(a-1) by a degree-(b-1) polynomial.  Output leg has a+b-1 coordinates."""
    if a < 0 or b < 0:
        raise ValueError
    out = max(a + b - 1, 0)
    t = SparseTensor((a, b, out))
    for i in range(a):
        for j in range(b):
            t.terms[(i, j, i + j)] = Fraction(1)
    return t


def cyclic_convolution(n: int) -> SparseTensor:
    """Structure tensor of the group algebra of Z/n: x_i y_j z_{(i+j) mod n}."""
    t = SparseTensor((n, n, n))
    for i in range(n):
        for j in range(n):
            t.terms[(i, j, (i + j) % n)] = Fraction(1)
    return t


def dot_product(m: int, leg: int = 0) -> SparseTensor:
    """B_X(m) = x * sum_i y_i z_i (``leg``=0); B_Y, B_Z by cyclic permutation."""
    t = SparseTensor((1, m, m))
    for i in range(m):
        t.terms[(0, i, i)] = Fraction(1)
    if leg == 0:
        return t
    if leg == 1:
        return t.permute_legs((2, 0, 1))  # singleton moves to Y
    if leg == 2:
        return t.permute_legs((1, 2, 0))  # singleton moves to Z
    raise ValueError(leg)


def matrix_multiplication(n: int, m: int, p: int) -> SparseTensor:
    """<n,m,p>: sum x_{ij} y_{jk} z_{ki} with i<n, j<m, k<p."""
    t = SparseTensor((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t.terms[(i * m + j, j * p + k, k * n + i)] = Fraction(1)
    return t


def unit_tensor() -> SparseTensor:
    return SparseTensor((1, 1, 1), {(0, 0, 0): 1})


def tensor_product(A: SparseTensor, B: SparseTensor) -> SparseTensor:
    """Kronecker product; coordinate (i, i') on leg l is flattened as i*dimB + i'."""
    dims = tuple(A.dims[l] * B.dims[l] for l in range(3))
    t = SparseTensor(dims)
    for ka, ca in A.terms.items():
        for kb, cb in B.terms.items():
            key = tuple(ka[l] * B.dims[l] + kb[l] for l in range(3))
            t.terms[key] = ca * cb  # type: ignore[index]
    return t


def direct_sum(A: SparseTensor, B: SparseTensor) -> SparseTensor:
    """Full direct sum: disjoint variables on all three legs."""
    dims = tuple(A.dims[l] + B.dims[l] for l in range(3))
    t = SparseTensor(dims)
    t.terms.update(A.terms)
    for kb, cb in B.terms.items():
        t.terms[tuple(kb[l] + A.dims[l] for l in range(3))] = cb  # type: ignore[index]
    return t


def shared_leg_sum(blocks: Sequence[SparseTensor], shared: int = 0) -> SparseTensor:
    """Direct sum of blocks sharing one leg (all blocks must have the same
    dimension on that leg); the other two legs are made disjoint."""
    if not blocks:
        raise ValueError("need at least one block")
    d_shared = blocks[0].dims[shared]
    if any(b.dims[shared] != d_shared for b in blocks):
        raise ValueError("shared leg dimensions differ")
    others = [l for l in range(3) if l != shared]
    dims = [0, 0, 0]
    dims[shared] = d_shared
    for l in others:
        dims[l] = sum(b.dims[l] for b in blocks)
    t = SparseTensor(dims)
    offsets = {l: 0 for l in others}
    for b in blocks:
        for key, c in b.terms.items():
            new = list(key)
            for l in others:
                new[l] += offsets[l]
            t.terms[tuple(new)] = c  # type: ignore[index]
        for l in others:
            offsets[l] += b.dims[l]
    return t


def make_family(spec: Mapping) -> SparseTensor:
    """Build a tensor from a JSON family specification."""
    fam = spec["family"]
    if fam == "convolution":
        return convolution(int(spec["a"]), int(spec["b"]))
    if fam == "cyclic_convolution":
        return cyclic_convolution(int(spec["n"]))
    if fam == "dot_product":
        return dot_product(int(spec["m"]), int(spec.get("leg", 0)))
    if fam == "matrix_multiplication":
        return matrix_multiplication(int(spec["n"]), int(spec["m"]), int(spec["p"]))
    if fam == "unit":
        return unit_tensor()
    if fam == "tensor_product":
        return tensor_product(make_family(spec["left"]), make_family(spec["right"]))
    if fam == "explicit":
        return SparseTensor.from_json(spec["tensor"])
    raise ValueError(f"unknown family {fam!r}")


def family_name(spec: Mapping) -> str:
    fam = spec["family"]
    if fam == "convolution":
        return f"C({spec['a']},{spec['b']})"
    if fam == "cyclic_convolution":
        return f"Cyc({spec['n']})"
    if fam == "dot_product":
        return f"B_{LEG_NAMES[int(spec.get('leg', 0))]}({spec['m']})"
    if fam == "matrix_multiplication":
        return f"<{spec['n']},{spec['m']},{spec['p']}>"
    if fam == "tensor_product":
        return f"{family_name(spec['left'])}(x){family_name(spec['right'])}"
    return fam
