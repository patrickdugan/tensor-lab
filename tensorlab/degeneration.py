"""Weighted degenerations, sector decompositions, and block identification.

A *weight degeneration* (paper, Section 2.2) assigns an integer weight to every
coordinate on every leg, multiplies each term by epsilon^(total weight), and
retains the terms of minimum total weight.  Everything here is combinatorial
and exact, so it is valid over every field; the passage from a degeneration to
a tensor-character inequality (Lemma 2.3 of the paper) is where characteristic
zero / large fields enter, and that step is *not* performed in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

from .tensors import PERMUTATIONS, Index, SparseTensor, convolution

Weights = Sequence[Sequence[int]]  # weights[leg][coordinate]


def term_weight(weights: Weights, key: Index) -> int:
    return int(weights[0][key[0]]) + int(weights[1][key[1]]) + int(weights[2][key[2]])


@dataclass
class Degeneration:
    source: SparseTensor
    weights: List[List[int]]
    min_weight: int
    retained: SparseTensor
    erased: Dict[Index, Tuple[Fraction, int]]  # key -> (coefficient, weight)

    @property
    def max_weight(self) -> int:
        if not self.erased:
            return self.min_weight
        return max(w for _, w in self.erased.values())


def degenerate(source: SparseTensor, weights: Weights) -> Degeneration:
    """Retain the minimum-weight terms of ``source`` under coordinate weights."""
    if any(len(weights[l]) != source.dims[l] for l in range(3)):
        raise ValueError("weights must have one entry per coordinate on each leg")
    if source.is_zero():
        return Degeneration(source, [list(map(int, w)) for w in weights], 0, SparseTensor(source.dims), {})
    wts = {key: term_weight(weights, key) for key in source.terms}
    m = min(wts.values())
    retained = SparseTensor(source.dims)
    erased: Dict[Index, Tuple[Fraction, int]] = {}
    for key, c in source.terms.items():
        if wts[key] == m:
            retained.terms[key] = c
        else:
            erased[key] = (c, wts[key])
    return Degeneration(source, [list(map(int, w)) for w in weights], m, retained, erased)


# ---------------------------------------------------------------------------------
# Sector structure
# ---------------------------------------------------------------------------------

@dataclass
class Block:
    """A block of a shared-leg direct sum: the retained terms whose Y and Z
    coordinates lie in given sectors (X is the shared leg, kept whole)."""
    label: int
    y_coords: List[int]
    z_coords: List[int]
    tensor: SparseTensor  # ambient coordinates of the source

    def local(self) -> SparseTensor:
        """The block re-indexed to (all of X) x y_coords x z_coords."""
        return self.tensor.restrict_to_support([list(range(self.tensor.dims[0])), self.y_coords, self.z_coords])


def sector_labels(partition: Sequence[Sequence[int]], dim: int) -> List[Optional[int]]:
    """Map coordinate -> sector label (None for coordinates outside every sector)."""
    lab: List[Optional[int]] = [None] * dim
    for s, coords in enumerate(partition):
        for c in coords:
            if lab[c] is not None:
                raise ValueError(f"coordinate {c} appears in two sectors")
            lab[c] = s
    return lab


def interval_partition(sizes: Sequence[int]) -> List[List[int]]:
    out, start = [], 0
    for s in sizes:
        out.append(list(range(start, start + s)))
        start += s
    return out


def decompose_shared_leg(retained: SparseTensor, y_sectors: Sequence[Sequence[int]],
                         z_sectors: Sequence[Sequence[int]], shared: int = 0):
    """Split a retained tensor into blocks that share leg ``shared`` and have
    matched sectors on the other two legs.

    The bipartite graph between Y-sectors and Z-sectors has an edge whenever a
    retained term joins them; the connected components are the blocks.  A
    retained term whose Y or Z coordinate lies outside every sector is a
    *stray* term and makes the decomposition invalid.

    Returns (blocks, components, strays) where components lists for each block
    the (Y-sector indices, Z-sector indices) it is built from.
    """
    if shared != 0:
        raise NotImplementedError("only X as the shared leg is implemented; permute legs first")
    ylab = sector_labels(y_sectors, retained.dims[1])
    zlab = sector_labels(z_sectors, retained.dims[2])
    strays = []
    edges = set()
    for key in retained.terms:
        ys, zs = ylab[key[1]], zlab[key[2]]
        if ys is None or zs is None:
            strays.append(key)
        else:
            edges.add((ys, zs))
    # union-find over Y-sectors and Z-sectors
    parent: Dict[Tuple[str, int], Tuple[str, int]] = {}

    def find(u):
        parent.setdefault(u, u)
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    for ys, zs in edges:
        union(("Y", ys), ("Z", zs))
    groups: Dict[Tuple[str, int], Dict[str, List[int]]] = {}
    for ys, zs in edges:
        r = find(("Y", ys))
        g = groups.setdefault(r, {"Y": [], "Z": []})
        if ys not in g["Y"]:
            g["Y"].append(ys)
        if zs not in g["Z"]:
            g["Z"].append(zs)
    blocks, components = [], []
    for label, (_, g) in enumerate(sorted(groups.items(), key=lambda kv: (min(kv[1]["Y"]), min(kv[1]["Z"])))):
        ys_list, zs_list = sorted(g["Y"]), sorted(g["Z"])
        y_coords = sorted(c for s in ys_list for c in y_sectors[s])
        z_coords = sorted(c for s in zs_list for c in z_sectors[s])
        tensor = retained.sub_tensor([None, y_coords, z_coords])
        blocks.append(Block(label, y_coords, z_coords, tensor))
        components.append((ys_list, zs_list))
    return blocks, components, strays


# ---------------------------------------------------------------------------------
# Block identification: is a block a convolution C(p,q) up to leg permutation and
# coordinate bijections?
# ---------------------------------------------------------------------------------

@dataclass
class Identification:
    family: str
    params: Tuple[int, ...]
    permutation: str              # legs of the target are mapped onto the block's legs
    maps: List[List[int]]         # maps[t][target coordinate] = block coordinate, per target leg t
    note: str = ""

    def to_json(self) -> dict:
        d = {"family": self.family}
        if self.family == "convolution":
            d["a"], d["b"] = self.params
        elif self.family == "cyclic_convolution":
            d["n"] = self.params[0]
        return {"target": d, "leg_permutation": self.permutation,
                "maps": {"X": self.maps[0], "Y": self.maps[1], "Z": self.maps[2]}}


def _coordinate_support(t: SparseTensor):
    sup = [set(), set(), set()]
    for key in t.terms:
        for l in range(3):
            sup[l].add(key[l])
    return [sorted(s) for s in sup]


class _Budget(Exception):
    pass


def find_isomorphism(target: SparseTensor, block: SparseTensor, budget: int = 200000) -> Optional[List[List[int]]]:
    """Find injective coordinate maps m_l (target coords -> block coords) with
    block[m_0 i, m_1 j, m_2 k] == target[i, j, k] for all (i,j,k), and such that
    the image contains the whole support of ``block``.  Returns the maps or None.

    Backtracking over the target's terms, with degree pruning.  Only the support
    pattern matters for the pruning; coefficients are compared exactly."""
    bsup = _coordinate_support(block)
    tsup = _coordinate_support(target)
    if block.nnz != target.nnz:
        return None
    if any(len(bsup[l]) != len(tsup[l]) for l in range(3)):
        return None
    # degree signatures: number of terms per coordinate on each leg
    def degrees(t: SparseTensor):
        deg = [dict(), dict(), dict()]
        for key in t.terms:
            for l in range(3):
                deg[l][key[l]] = deg[l].get(key[l], 0) + 1
        return deg
    bdeg, tdeg = degrees(block), degrees(target)
    for l in range(3):
        if sorted(bdeg[l].values()) != sorted(tdeg[l].values()):
            return None
    tterms = sorted(target.terms.items())
    # order target terms so that each new term shares coordinates with earlier ones
    maps: List[Dict[int, int]] = [dict(), dict(), dict()]
    used: List[set] = [set(), set(), set()]
    block_terms = block.terms

    nodes = [0]

    def extend(idx: int) -> bool:
        nodes[0] += 1
        if nodes[0] > budget:
            raise _Budget()
        if idx == len(tterms):
            return True
        key, c = tterms[idx]
        # candidate images for each coordinate
        cands = []
        for l in range(3):
            if key[l] in maps[l]:
                cands.append([maps[l][key[l]]])
            else:
                cands.append([b for b in bsup[l] if b not in used[l] and bdeg[l][b] == tdeg[l][key[l]]])
        for bi in cands[0]:
            for bj in cands[1]:
                for bk in cands[2]:
                    if block_terms.get((bi, bj, bk)) != c:
                        continue
                    assigned = []
                    ok = True
                    for l, b in zip(range(3), (bi, bj, bk)):
                        if key[l] not in maps[l]:
                            maps[l][key[l]] = b
                            used[l].add(b)
                            assigned.append((l, b))
                    if ok and extend(idx + 1):
                        return True
                    for l, b in assigned:
                        del maps[l][key[l]]
                        used[l].discard(b)
        return False

    try:
        if not extend(0):
            return None
    except _Budget:
        IDENTIFY_STATS["budget_exceeded"] += 1
        return None
    out = []
    for l in range(3):
        m = [maps[l].get(i, -1) for i in range(target.dims[l])]
        # coordinates of the target with no term (dimension padding) map to unused block coords if any
        if -1 in m:
            return None
        out.append(m)
    # the maps must cover the block's support
    for l in range(3):
        if set(out[l]) != set(bsup[l]):
            return None
    return out


def _arith_progression(vals: List[int]):
    """Return the common difference of a sorted list that is an arithmetic
    progression (0 for a singleton), or None."""
    if len(vals) == 1:
        return 0
    d = vals[1] - vals[0]
    if d <= 0:
        return None
    for x, y in zip(vals, vals[1:]):
        if y - x != d:
            return None
    return d


def identify_sum_block(block: SparseTensor) -> Optional[Identification]:
    """Exact linear-time recogniser for blocks whose cells satisfy k = i + j
    (sub-tensors of an ordinary convolution with its own coordinates).

    Such a block is isomorphic to some C(p,q) iff its cells form a full product
    on two of the legs, with all coefficients 1, and the two index sets are
    arithmetic progressions with a common difference (for integer sets,
    |A + B| = |A| + |B| - 1 forces exactly this).  The three cases are
    inputs (X,Y) [identity], inputs (X,Z) with Y = Z - X [swap_YZ, X reversed],
    inputs (Y,Z) with X = Z - Y [cycle_XYZ, Y reversed].  Returns None when the
    block is not of this form (then it is not a convolution)."""
    cells = list(block.terms.items())
    if not cells or any(k != i + j for (i, j, k), _ in cells) or any(c != 1 for _, c in cells):
        return None
    n = len(cells)
    X = sorted({i for (i, _, _), _ in cells})
    Y = sorted({j for (_, j, _), _ in cells})
    Z = sorted({k for (_, _, k), _ in cells})
    cases = [  # (input set A, input set B, output set, leg indices of A and B)
        (X, Y, Z, 0, 1),
        (X, Z, Y, 0, 2),
        (Y, Z, X, 1, 2),
    ]
    for A, Bs, O, la, lb in cases:
        if len(A) * len(Bs) != n or len(O) != len(A) + len(Bs) - 1:
            continue
        pairs = {(key[la], key[lb]) for key, _ in cells}
        if len(pairs) != n:
            continue
        da, db = _arith_progression(A), _arith_progression(Bs)
        if da is None or db is None:
            continue
        d = da or db
        if (da and da != d) or (db and db != d):
            continue
        p, q = len(A), len(Bs)
        if (la, lb) == (0, 1):
            # x_{A[u]} y_{B[v]} z_{A[0]+B[0]+d(u+v)}
            maps = [list(A), list(Bs), [A[0] + Bs[0] + d * s for s in range(p + q - 1)]]
            ident = Identification("convolution", (p, q), "identity", maps)
        elif (la, lb) == (0, 2):
            # target (X, Y, Z) = block (X reversed, Z, Y): y = z - x
            maps = [[A[p - 1 - u] for u in range(p)], list(Bs),
                    [Bs[0] - A[p - 1] + d * s for s in range(p + q - 1)]]
            ident = Identification("convolution", (p, q), "swap_YZ", maps)
        else:
            # target (X, Y, Z) = block (Y reversed, Z, X): x = z - y
            maps = [[A[p - 1 - u] for u in range(p)], list(Bs),
                    [Bs[0] - A[p - 1] + d * s for s in range(p + q - 1)]]
            ident = Identification("convolution", (p, q), "cycle_XYZ", maps)
        if check_identification(block, ident):
            return ident
    return None


IDENTIFY_STATS = {"fast": 0, "backtrack": 0, "budget_exceeded": 0}


def identify_block(block: SparseTensor, x_dim: int | None = None,
                   allow_cyclic: bool = False, budget: int = 200000) -> Optional[Identification]:
    """Identify ``block`` (in ambient coordinates) as a convolution C(p,q) with
    some leg permutation.  The returned maps send target coordinates to the
    block's ambient coordinates.  Dimensions are read off from the support sizes.

    Blocks with k = i + j (sub-tensors of an ordinary convolution) are decided
    exactly by ``identify_sum_block``.  Other blocks use backtracking with a
    node budget; exceeding it returns None (a missed identification, never a
    false one) and is counted in IDENTIFY_STATS["budget_exceeded"]."""
    if block.terms and all(k == i + j for (i, j, k) in block.terms):
        IDENTIFY_STATS["fast"] += 1
        return identify_sum_block(block)
    IDENTIFY_STATS["backtrack"] += 1
    sup = _coordinate_support(block)
    sizes = [len(s) for s in sup]
    n = block.nnz
    if n == 0:
        return None
    # A convolution C(p,q) (any orientation) has support sizes {p, q, p+q-1}
    # on its three legs and p*q terms.  Try every assignment of (p,q,out) to legs.
    for perm_name, p in PERMUTATIONS.items():
        # target leg t sits on block leg p[t]
        pp, qq, oo = sizes[p[0]], sizes[p[1]], sizes[p[2]]
        if pp * qq != n or oo != pp + qq - 1:
            continue
        target = convolution(pp, qq)
        # block legs reordered to target order
        reordered = block.permute_legs(tuple(p))
        maps = find_isomorphism(target, reordered)
        if maps is not None:
            return Identification("convolution", (pp, qq), perm_name, maps)
    if allow_cyclic:
        from .tensors import cyclic_convolution
        m = sizes[0]
        if sizes == [m, m, m] and n == m * m:
            for perm_name, p in PERMUTATIONS.items():
                maps = find_isomorphism(cyclic_convolution(m), block.permute_legs(tuple(p)))
                if maps is not None:
                    return Identification("cyclic_convolution", (m,), perm_name, maps)
    return None


def check_identification(block: SparseTensor, ident: Identification) -> bool:
    """Independently re-check an identification: pull the block back along the
    maps (after the leg permutation) and compare with the target exactly."""
    from .tensors import make_family
    target = make_family(ident.to_json()["target"])
    p = PERMUTATIONS[ident.permutation]
    reordered = block.permute_legs(tuple(p))
    pulled = reordered.pullback(ident.maps, target.dims)
    if pulled != target:
        return False
    # the maps must be injective and cover the support of the block
    sup = _coordinate_support(reordered)
    for l in range(3):
        if len(set(ident.maps[l])) != len(ident.maps[l]):
            return False
        if set(ident.maps[l]) != set(sup[l]):
            return False
    return True
