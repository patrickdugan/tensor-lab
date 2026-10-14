"""Automated discovery of shared-leg sector degenerations.

Search space (bounded, exhaustive within the bounds):
  * source tensors C(a,B) (and optionally cyclic convolutions),
  * Y and Z partitioned into at most K contiguous intervals ("sectors"),
  * an order-preserving matching between some Y-sectors and some Z-sectors
    (matched pairs become blocks; unmatched sectors are padding),
  * sector-constant integer weights, which are not enumerated but *solved for*:
    matched pairs must have total weight 0 and every term joining unmatched
    sectors must have positive weight.  Writing w_Z[sigma(s)] = -w_Y[s] turns the
    requirement into difference constraints w_Y[s] - w_Y[s'] >= 1 for every
    cross term from Y_s into Z_{sigma(s')}, which are feasible iff the
    cross-term digraph is acyclic; the longest-path potential is the
    canonical minimal solution.  Padding sectors receive weights large enough
    to erase every term touching them.

For each feasible candidate the min-weight tensor is computed, decomposed into
blocks, and every block is identified (convolution up to leg permutation and
coordinate bijection).  Candidates whose blocks are all identified give the
inequality  P(a,B) >= sum_i P(block_i).  They are canonicalised by (source,
multiset of block types), deduplicated, and scored against the extremal
profile (growth.extremal_profile): a negative slack would mean the candidate is
not implied by the paper's inequalities and would strengthen the growth
exponent.
"""
from __future__ import annotations

import itertools
import json
import time
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .degeneration import check_identification, decompose_shared_leg, degenerate, identify_block
from .growth import Candidate, extremal_profile
from .sectors import build_certificate
from .tensors import SparseTensor, convolution, cyclic_convolution


def compositions(n: int, k: int) -> Iterable[Tuple[int, ...]]:
    """All k-tuples of positive integers summing to n."""
    if k == 1:
        if n >= 1:
            yield (n,)
        return
    for first in range(1, n - k + 2):
        for rest in compositions(n - first, k - 1):
            yield (first,) + rest


def intervals_from_sizes(sizes: Sequence[int]) -> List[Tuple[int, int]]:
    out, s = [], 0
    for x in sizes:
        out.append((s, s + x))
        s += x
    return out


def order_preserving_matchings(ky: int, kz: int, min_pairs: int = 1):
    """Yield lists of (s,t) pairs with s increasing and t increasing."""
    for m in range(min_pairs, min(ky, kz) + 1):
        for ys in itertools.combinations(range(ky), m):
            for zs in itertools.combinations(range(kz), m):
                yield list(zip(ys, zs))


def has_terms(source_kind: str, a: int, B: int, y_iv: Tuple[int, int], z_iv: Tuple[int, int]) -> bool:
    """Does C(a,B) (or Cyc) have a term with j in y_iv and k in z_iv?"""
    y0, y1 = y_iv
    z0, z1 = z_iv
    if source_kind == "convolution":
        # need i = k - j in [0, a) for some j in [y0,y1), k in [z0,z1)
        lo = z0 - (y1 - 1)
        hi = (z1 - 1) - y0
        return max(lo, 0) <= min(hi, a - 1)
    # cyclic: k = (i + j) mod n; with n = a = B, any (j,k) pair occurs
    return True


def solve_weights(cross: Dict[Tuple[int, int], bool], matched: List[Tuple[int, int]], ky: int, kz: int):
    """Difference-constraint solution for matched sector weights.

    cross[(s,t)] is True when Y_s and Z_t share a term.  matched pairs (s,t)
    get w_Y[s] + w_Z[t] = 0.  Returns (wY, wZ) lists over sectors or None."""
    sigma = {s: t for s, t in matched}
    sigma_inv = {t: s for s, t in matched}
    ms = [s for s, _ in matched]
    # edges s -> s' when Y_s has a cross term into Z_{sigma(s')}: need wY[s] >= wY[s'] + 1
    edges = {s: [] for s in ms}
    for s in ms:
        for s2 in ms:
            if s != s2 and cross.get((s, sigma[s2])):
                edges[s].append(s2)
    # longest path (acyclic check via DFS with colours)
    colour = {s: 0 for s in ms}
    order: List[int] = []

    def dfs(u) -> bool:
        colour[u] = 1
        for v in edges[u]:
            if colour[v] == 1:
                return False
            if colour[v] == 0 and not dfs(v):
                return False
        colour[u] = 2
        order.append(u)
        return True

    for s in ms:
        if colour[s] == 0 and not dfs(s):
            return None
    # order is reverse topological (sinks first); potential = longest path to a sink
    pot = {s: 0 for s in ms}
    for u in order:  # sinks first
        for v in edges[u]:
            pot[u] = max(pot[u], pot[v] + 1)
    wY = [0] * ky
    wZ = [0] * kz
    for s in ms:
        wY[s] = pot[s]
        wZ[sigma[s]] = -pot[s]
    hi = max(pot.values()) if pot else 0
    lo = min(pot.values()) if pot else 0
    for s in range(ky):
        if s not in sigma:
            wY[s] = hi + 1          # pad: w_pad + w_Z[t] = hi + 1 - pot >= 1
    for t in range(kz):
        if t not in sigma_inv:
            wZ[t] = 1 - lo          # pad: w_Y[s] + w_pad = pot + 1 - lo >= 1
    return wY, wZ


_IDENT_CACHE: Dict[Tuple, Optional[object]] = {}


def _identify_cached(kind: str, a: int, blk):
    """Identify a block; for convolution sources the block is the band of C(a,B)
    restricted to an interval of Y and an interval of Z, so its isomorphism type
    depends only on (a, |Y|, |Z|, z0 - y0) and the X-support pattern.  The
    identification (family, parameters, permutation) is cached under that key.
    The certificate built afterwards re-identifies the block from scratch."""
    ys, zs = blk.y_coords, blk.z_coords
    contiguous = ys == list(range(ys[0], ys[-1] + 1)) and zs == list(range(zs[0], zs[-1] + 1))
    if kind != "convolution" or not contiguous:
        ident = identify_block(blk.tensor)
        if ident is None or not check_identification(blk.tensor, ident):
            return None
        return ident
    key = (a, len(ys), len(zs), zs[0] - ys[0])
    if key not in _IDENT_CACHE:
        ident = identify_block(blk.tensor)
        if ident is not None and not check_identification(blk.tensor, ident):
            ident = None
        _IDENT_CACHE[key] = ident
    return _IDENT_CACHE[key]


class SearchStats:
    def __init__(self):
        self.sources = 0
        self.structures = 0
        self.weight_feasible = 0
        self.degenerations = 0
        self.all_identified = 0
        self.unique = 0
        self.seconds = 0.0

    def to_json(self):
        return dict(self.__dict__)


def search(max_a: int, max_b: int, max_sectors: int = 3, min_a: int = 1, min_b: int = 1,
           allow_pads: bool = True, sources: Sequence[str] = ("convolution",),
           max_structures_per_source: Optional[int] = None, verbose: bool = False):
    """Bounded exhaustive search.  Returns (results, stats).

    Each result is a dict with the candidate inequality, the certificate, the
    slack on the extremal profile and classification flags."""
    t0 = time.time()
    stats = SearchStats()
    found: Dict[Tuple, dict] = {}
    for kind in sources:
        if kind == "convolution":
            params = [(a, B) for a in range(min_a, max_a + 1) for B in range(min_b, max_b + 1)]
        elif kind == "cyclic":
            params = [(n, n) for n in range(max(min_a, 2), max_a + 1)]
        else:
            raise ValueError(kind)
        for a, B in params:
            stats.sources += 1
            src = convolution(a, B) if kind == "convolution" else cyclic_convolution(a)
            zdim = src.dims[2]
            n_struct = 0
            for ky in range(1, max_sectors + 1):
                for ysz in compositions(B, ky):
                    y_iv = intervals_from_sizes(ysz)
                    for kz in range(1, max_sectors + 1):
                        for zsz in compositions(zdim, kz):
                            z_iv = intervals_from_sizes(zsz)
                            cross = {(s, t): has_terms(kind, a, B, y_iv[s], z_iv[t])
                                     for s in range(ky) for t in range(kz)}
                            for matched in order_preserving_matchings(ky, kz, min_pairs=2):
                                if not allow_pads and len(matched) < max(ky, kz):
                                    continue
                                if any(not cross[(s, t)] for s, t in matched):
                                    continue  # a matched pair with no terms is not a block
                                n_struct += 1
                                stats.structures += 1
                                if max_structures_per_source and n_struct > max_structures_per_source:
                                    break
                                sol = solve_weights(cross, matched, ky, kz)
                                if sol is None:
                                    continue
                                stats.weight_feasible += 1
                                wY_s, wZ_s = sol
                                wY = [wY_s[s] for s in range(ky) for _ in range(y_iv[s][0], y_iv[s][1])]
                                wZ = [wZ_s[t] for t in range(kz) for _ in range(z_iv[t][0], z_iv[t][1])]
                                weights = [[0] * a, wY, wZ]
                                deg = degenerate(src, weights)
                                stats.degenerations += 1
                                if deg.min_weight != 0:
                                    continue
                                y_sectors = [list(range(*y_iv[s])) for s, _ in matched]
                                z_sectors = [list(range(*z_iv[t])) for _, t in matched]
                                blocks, comps, strays = decompose_shared_leg(deg.retained, y_sectors, z_sectors)
                                if strays or len(blocks) != len(matched):
                                    continue
                                idents = []
                                ok = True
                                for blk in blocks:
                                    ident = _identify_cached(kind, a, blk)
                                    if ident is None:
                                        ok = False
                                        break
                                    idents.append(ident)
                                if not ok:
                                    continue
                                stats.all_identified += 1
                                block_types = [tuple(sorted(i.params)) for i in idents]
                                if kind == "convolution":
                                    cand = Candidate((a, B), block_types, origin="search")
                                else:
                                    cand = Candidate((a, B), block_types, origin="search-cyclic")
                                key = (kind,) + cand.key()
                                xsup = [sorted({k[0] for k in blk.tensor.terms}) for blk in blocks]
                                restriction_type = any(len(x) < a for x in xsup)
                                entry = found.get(key)
                                if entry is not None:
                                    entry["multiplicity"] += 1
                                    continue
                                slack = cand.slack(extremal_profile)
                                found[key] = {
                                    "source_kind": kind, "a": a, "B": B,
                                    "candidate": repr(cand), "blocks": block_types,
                                    "y_sizes": list(ysz), "z_sizes": list(zsz), "matched": matched,
                                    "weights_Y": wY, "weights_Z": wZ,
                                    "permutations": [i.permutation for i in idents],
                                    "extremal_slack": str(slack),
                                    "extremal_slack_float": float(slack),
                                    "restriction_type": restriction_type,
                                    "n_blocks": len(blocks),
                                    "multiplicity": 1,
                                    "max_weight": deg.max_weight,
                                    "erased": len(deg.erased),
                                }
            if verbose:
                print(f"  {kind} a={a} B={B}: {n_struct} structures, {len(found)} unique so far, {time.time()-t0:.1f}s")
    stats.unique = len(found)
    stats.seconds = time.time() - t0
    results = sorted(found.values(), key=lambda e: (e["extremal_slack_float"], -e["n_blocks"], e["a"], e["B"]))
    return results, stats


def classify(entry: dict) -> str:
    """Rank classes for the catalogue."""
    s = Fraction(entry["extremal_slack"])
    a, B = entry["a"], entry["B"]
    blocks = [tuple(b) for b in entry["blocks"]]
    if s < 0:
        return "NOVEL: violates the extremal profile (not implied by the known inequalities)"
    if a == 1:
        return "trivial: splitting a dot product C(1,B) = B_X(B) (P(1,b) = b is additive)"
    from .tensors import PERMUTATIONS
    # the target's output leg (Z, index 2) lands on block leg PERMUTATIONS[perm][2]
    if any(PERMUTATIONS[perm][2] == 0 for perm in entry.get("permutations", [])):
        return "contains a rectangle block (output leg on the shared X leg)"
    if entry["restriction_type"]:
        return "restriction-type (a block uses a proper subset of the shared X leg)"
    if all(a in b for b in blocks):
        if s == 0:
            return "known family: alternating C(a,h_i) blocks, tight on the extremal profile"
        return "known family: alternating C(a,h_i) blocks (slack > 0: coordinates wasted)"
    return "other block shapes (slack >= 0: consistent with the extremal profile)"


def certificate_for(entry: dict) -> dict:
    """Rebuild and certify a search result."""
    a, B = entry["a"], entry["B"]
    src = convolution(a, B) if entry["source_kind"] == "convolution" else cyclic_convolution(a)
    y_iv = intervals_from_sizes(entry["y_sizes"])
    z_iv = intervals_from_sizes(entry["z_sizes"])
    y_sectors = [list(range(*y_iv[s])) for s, _ in entry["matched"]]
    z_sectors = [list(range(*z_iv[t])) for _, t in entry["matched"]]
    weights = [[0] * a, entry["weights_Y"], entry["weights_Z"]]
    spec = {"family": "convolution", "a": a, "b": B} if entry["source_kind"] == "convolution" else {"family": "cyclic_convolution", "n": a}
    cert = build_certificate(spec, src, weights, y_sectors, z_sectors, "search_shared_leg",
                             {"a": a, "B": B, "y_sizes": entry["y_sizes"], "z_sizes": entry["z_sizes"],
                              "matched": entry["matched"]},
                             claims=[entry["candidate"]])
    cert["expected"] = {"num_blocks": entry["n_blocks"]}
    return cert


# ---------------------------------------------------------------------------------
# Catalogue post-processing: dominance and ranking
# ---------------------------------------------------------------------------------

def _norm(b):
    p, q = b
    return (min(p, q), max(p, q))


def _dominates(strong, weak) -> bool:
    """Block multiset ``strong`` dominates ``weak`` if there is an injection from
    weak blocks into strong blocks with componentwise >= (normalised) sizes.
    Since P is increasing in each argument (restriction monotonicity), the
    inequality with ``strong`` blocks then implies the one with ``weak`` blocks."""
    strong = [_norm(b) for b in strong]
    weak = [_norm(b) for b in weak]
    if len(weak) > len(strong):
        return False
    used = [False] * len(strong)

    def match(i):
        if i == len(weak):
            return True
        p, q = weak[i]
        for j, (P_, Q_) in enumerate(strong):
            if not used[j] and P_ >= p and Q_ >= q:
                used[j] = True
                if match(i + 1):
                    return True
                used[j] = False
        return False

    return match(0)


def mark_dominance(results: List[dict]) -> List[dict]:
    """Mark each result as dominated if another result with the same first
    argument a and a source B' <= B has a dominating block multiset (and is not
    the identical inequality).  P(a,B) >= P(a,B') by monotonicity, so such a
    result is a trivial consequence (padding, a dropped block, a shrunk block)."""
    by_a: Dict[int, List[dict]] = {}
    for e in results:
        by_a.setdefault((e["source_kind"], e["a"]), []).append(e)
    for key, group in by_a.items():
        for e in group:
            e["dominated_by"] = None
            for f in group:
                if f is e or f["B"] > e["B"]:
                    continue
                same = f["B"] == e["B"] and sorted(map(_norm, f["blocks"])) == sorted(map(_norm, e["blocks"]))
                if same:
                    continue
                if _dominates(f["blocks"], e["blocks"]):
                    e["dominated_by"] = f["candidate"]
                    break
    return results
