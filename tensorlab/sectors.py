"""The three-sector (shifted tripling) construction of the paper, Lemma 4.2,
plus its generalisations with unequal sector sizes, and certificate generation.

Paper (Section 4.2): put B = 3h + a - 1.  Partition the Y-indices of C(a,B) as
[0,h) | [h, 2h+a-1) | [2h+a-1, B) and the Z-indices as [0, h+a-1) |
[h+a-1, 2h+a-1) | [2h+a-1, 3h+2a-2).  Weight +1 on the middle Y-sector, -1 on the
middle Z-sector, 0 elsewhere.  The weight-zero part is C(a,h) (left) + C(a,h)
with legs Y,Z exchanged and X reversed (middle) + C(a,h) translated (right),
all sharing the X leg.  This matches OpenAI's Lean module Sector/Weights.lean.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from . import __version__
from .degeneration import (Block, Identification, check_identification, decompose_shared_leg,
                           degenerate, identify_block, interval_partition)
from .tensors import SparseTensor, convolution


def three_sector_weights(a: int, h: int):
    B = 3 * h + a - 1
    zdim = a + B - 1
    y_sizes = [h, a + h - 1, h]
    z_sizes = [a + h - 1, h, a + h - 1]
    assert sum(y_sizes) == B and sum(z_sizes) == zdim
    wX = [0] * a
    wY = [0] * h + [1] * (a + h - 1) + [0] * h
    wZ = [0] * (a + h - 1) + [-1] * h + [0] * (a + h - 1)
    return [wX, wY, wZ], interval_partition(y_sizes), interval_partition(z_sizes)


def general_sector_weights(a: int, sizes: Sequence[int], kinds: Sequence[str]):
    """Alternating 'normal'/'exchanged' blocks C(a,h_i) in the Y/Z sectors.

    kinds[i] in {"N", "E"}: a normal block uses Y-size h_i and Z-size a+h_i-1,
    an exchanged block uses Y-size a+h_i-1 and Z-size h_i.  The weights give
    +1 to the Y-sectors of exchanged blocks and -1 to their Z-sectors.
    Returns (weights, y_sectors, z_sectors, B) where the source is C(a,B)."""
    # A normal block C(a,h) placed at Y-offset y0 occupies Z-offset y0; an
    # exchanged block needs Z-offset y0 + a - 1.  Whenever the running Z-offset
    # does not match, padding coordinates (weight 2, so every term touching them
    # is erased) are inserted on Z or on Y.  Padding is not part of any sector.
    PAD = 2
    wY: List[int] = []
    wZ: List[int] = []
    y_sectors: List[List[int]] = []
    z_sectors: List[List[int]] = []
    yoff = zoff = 0
    for h, kind in zip(sizes, kinds):
        if kind not in ("N", "E"):
            raise ValueError(kind)
        need = yoff + (a - 1 if kind == "E" else 0)
        if zoff < need:
            wZ += [PAD] * (need - zoff)
            zoff = need
        elif zoff > need:
            wY += [PAD] * (zoff - need)
            yoff += zoff - need
        ys = h if kind == "N" else a + h - 1
        zs = a + h - 1 if kind == "N" else h
        y_sectors.append(list(range(yoff, yoff + ys)))
        z_sectors.append(list(range(zoff, zoff + zs)))
        wY += [1 if kind == "E" else 0] * ys
        wZ += [-1 if kind == "E" else 0] * zs
        yoff += ys
        zoff += zs
    B = yoff
    zdim = a + B - 1
    if zoff > zdim:
        raise ValueError("Z sectors do not fit")
    wZ += [PAD] * (zdim - zoff)
    return [[0] * a, wY, wZ], y_sectors, z_sectors, B


def build_certificate(source_spec: dict, source: SparseTensor, weights, y_sectors, z_sectors,
                      kind: str, parameters: dict, claims: Sequence[str] = (),
                      explicit: Dict[int, Identification] | None = None) -> Dict:
    """Run the degeneration, decompose, identify blocks, and emit a certificate.

    ``explicit`` optionally supplies, per block label, the coordinate maps that
    the certificate should carry (for instance the paper's own maps).  They are
    checked exactly; if they fail, the isomorphism search is used instead."""
    deg = degenerate(source, weights)
    blocks, components, strays = decompose_shared_leg(deg.retained, y_sectors, z_sectors)
    block_entries = []
    for blk, comp in zip(blocks, components):
        ident = None
        if explicit and blk.label in explicit and check_identification(blk.tensor, explicit[blk.label]):
            ident = explicit[blk.label]
        if ident is None:
            ident = identify_block(blk.tensor)
        entry = {
            "label": blk.label,
            "y_sectors": comp[0],
            "z_sectors": comp[1],
            "y_coords": blk.y_coords,
            "z_coords": blk.z_coords,
            "nnz": blk.tensor.nnz,
        }
        if ident is not None and check_identification(blk.tensor, ident):
            entry.update(ident.to_json())
            entry["identified"] = True
        else:
            entry["identified"] = False
        block_entries.append(entry)
    cert = {
        "certificate_version": 1,
        "generator": f"tensorlab {__version__}",
        "kind": kind,
        "parameters": parameters,
        "source": dict(source_spec, dims=list(source.dims), nnz=source.nnz),
        "weights": {"X": list(deg.weights[0]), "Y": list(deg.weights[1]), "Z": list(deg.weights[2])},
        "sectors": {"Y": [list(s) for s in y_sectors], "Z": [list(s) for s in z_sectors]},
        "shared_leg": "X",
        "min_weight": deg.min_weight,
        "max_weight": deg.max_weight,
        "retained_terms": [[i, j, k, str(c)] for (i, j, k), c in sorted(deg.retained.terms.items())],
        "erased_terms": [[i, j, k, str(c), w] for (i, j, k), (c, w) in sorted(deg.erased.items())],
        "stray_terms": [list(s) for s in strays],
        "blocks": block_entries,
        "claims": list(claims),
        "scope": ("Finite exact identity for the stated parameters only. "
                  "It does not by itself establish the identity for other parameters, "
                  "nor the tensor-character inequality, which needs Lemma 2.3 "
                  "(interpolation, characteristic zero) and Corollary 3.2 of the paper."),
    }
    return cert


def three_sector_certificate(a: int, h: int) -> Dict:
    if a < 1 or h < 1:
        raise ValueError("a, h must be positive")
    weights, ys, zs = three_sector_weights(a, h)
    B = 3 * h + a - 1
    source = convolution(a, B)
    claims = [
        f"C({a},{B}) degenerates (minimum weight 0) to three blocks sharing the X leg",
        f"left block == C({a},{h}) with identity maps",
        f"middle block == C({a},{h}) with legs Y,Z exchanged and X reversed",
        f"right block == C({a},{h}) translated by {2*h + a - 1} on Y and Z",
        "every erased term has weight exactly +1; no term has negative weight",
        f"this is Lemma 4.2 (shifted tripling) of the paper at (a,h)=({a},{h}): P({a},{B}) >= 3 P({a},{h}) after Cor. 3.2",
    ]
    # The paper's coordinate identifications (Section 4.2 and Lean Sector/Branches.lean):
    #   left:   x_i y_r z_{i+r}                       (identity maps)
    #   middle: x_{a-1-u} y_{h+u+r} z_{h+a-1+r}       (X reversed, legs Y/Z exchanged)
    #   right:  x_i y_{2h+a-1+r} z_{2h+a-1+i+r}       (translation by 2h+a-1)
    # Maps send target coordinates of C(a,h) (after the leg permutation) to source coordinates.
    s = 2 * h + a - 1
    explicit = {
        0: Identification("convolution", (a, h), "identity",
                          [list(range(a)), list(range(h)), list(range(a + h - 1))]),
        1: Identification("convolution", (a, h), "swap_YZ",
                          [list(range(a - 1, -1, -1)),          # target X i -> source x_{a-1-i}
                           [h + a - 1 + r for r in range(h)],   # target Y r -> source z_{h+a-1+r}
                           [h + t for t in range(a + h - 1)]]), # target Z t -> source y_{h+t}
        2: Identification("convolution", (a, h), "identity",
                          [list(range(a)), [s + r for r in range(h)], [s + t for t in range(a + h - 1)]]),
    }
    cert = build_certificate({"family": "convolution", "a": a, "b": B}, source, weights, ys, zs,
                             "three_sector_shifted_tripling", {"a": a, "h": h, "B": B}, claims,
                             explicit=explicit)
    cert["expected"] = {
        "num_blocks": 3,
        "block_targets": [{"family": "convolution", "a": a, "b": h}] * 3,
        "middle_block_permutation": "swap_YZ",
        "paper_lemma": "4.2",
        "lean_reference": "OAI/LinearAlgebra/MatrixMultiplication/AuxiliarySeparation/Sector/Weights.lean",
    }
    return cert


def general_sector_certificate(a: int, sizes: Sequence[int], kinds: Sequence[str]) -> Dict:
    weights, ys, zs, B = general_sector_weights(a, sizes, kinds)
    source = convolution(a, B)
    claims = [f"C({a},{B}) degenerates to blocks " +
              " + ".join(f"C({a},{h}){'^swap' if k == 'E' else ''}" for h, k in zip(sizes, kinds)) +
              " sharing the X leg"]
    cert = build_certificate({"family": "convolution", "a": a, "b": B}, source, weights, ys, zs,
                             "general_sector_shared_leg", {"a": a, "sizes": list(sizes), "kinds": list(kinds), "B": B},
                             claims)
    cert["expected"] = {"num_blocks": len(sizes),
                        "block_targets": [{"family": "convolution", "a": a, "b": h} for h in sizes]}
    return cert


def staircase_layout(a: int, p: int, q: int, n: int):
    """Blocks C(p,q) with the *output* leg on the shared X leg ("rectangles" in
    the (j,k) plane of C(a,B)).

    Block b uses Y_b = [y_b, y_b + p) and Z_b = [y_b + p - 1, y_b + p + q - 1)
    with y_b = b (p + q - 1).  Inside a block every pair (j,k) has
    0 <= k - j <= p + q - 2, so with p + q - 1 <= a every pair is a term and the
    block is sum x_{k-j} y_j z_k, i.e. C(p,q) with its output leg on X.
    Z_c lies entirely below Y_b for c < b, so every cross term runs from Y_b to
    Z_c with b < c; sector weights w(Y_b) = n - b, w(Z_c) = -(n - c) give such a
    term weight c - b >= 1.  Gap coordinates get a large weight and are erased.
    Returns (weights, y_sectors, z_sectors, B) with B = (n-1)(p+q-1) + p."""
    if p + q - 1 > a:
        raise ValueError("rectangle does not fit in the band: need p + q - 1 <= a")
    if min(p, q, n) < 1:
        raise ValueError("p, q, n must be positive")
    PAD = 2 * n + 2
    period = p + q - 1
    y_sectors = [list(range(b * period, b * period + p)) for b in range(n)]
    z_sectors = [list(range(b * period + p - 1, b * period + p - 1 + q)) for b in range(n)]
    B = y_sectors[-1][-1] + 1
    zdim = a + B - 1
    if z_sectors[-1][-1] >= zdim:
        raise ValueError("Z range exceeds a + B - 1")
    wY = [PAD] * B
    wZ = [PAD] * zdim
    for b in range(n):
        for j in y_sectors[b]:
            wY[j] = n - b
        for k in z_sectors[b]:
            wZ[k] = -(n - b)
    return [[0] * a, wY, wZ], y_sectors, z_sectors, B


def staircase_certificate(a: int, p: int, q: int, n: int) -> Dict:
    weights, ys, zs, B = staircase_layout(a, p, q, n)
    source = convolution(a, B)
    claims = [f"C({a},{B}) degenerates to {n} blocks C({p},{q}) with the output leg on the shared X leg "
              f"(staircase of rectangles); inequality P({a},{B}) >= {n} P({p},{q})"]
    cert = build_certificate({"family": "convolution", "a": a, "b": B}, source, weights, ys, zs,
                             "staircase_rectangles", {"a": a, "p": p, "q": q, "n": n, "B": B}, claims)
    cert["expected"] = {"num_blocks": n, "block_targets": [{"family": "convolution", "a": p, "b": q}] * n}
    return cert
