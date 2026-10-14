"""Realizability experiments with general monomial weights (see weights_lp.py).

E1  staircase spacing: n rectangles C(p,q) of one i-class at period delta.
    Conjecture (proved by a 2-cycle Farkas certificate): realizable iff delta >= p+q-1.
E2  two i-classes of rectangles packed with full Y and Z utilization.
E3  "double staircase": a rectangle staircase plus the rectangles that would
    fill its gaps.
E4  sanity: the paper's three-sector retained set is realizable with x = 0.
E5  bounded exhaustive packing search: every packing of interval templates
    (rectangles, normal and exchanged convolutions) into C(a,B) whose total
    extremal-profile value exceeds P*(a,B), tested for realizability.
"""
from __future__ import annotations

import time
from fractions import Fraction
from typing import Dict, List, Set, Tuple

from .growth import extremal_profile
from .weights_lp import (Cell, block_legs, check_disjoint, decide, exchanged_cells, normal_cells,
                         rectangle_cells, solve_farkas, solve_primal, staircase_retained,
                         verify_obstruction, verify_realization)


def _value(blocks_types) -> Fraction:
    return sum((extremal_profile(p, q) for p, q in blocks_types), Fraction(0))


def e1_staircase_spacing(cases=((5, 3, 3), (7, 3, 3), (6, 2, 4), (6, 3, 2)), n: int = 3) -> List[dict]:
    out = []
    for a, p, q in cases:
        ell = p + q - 1
        for period in range(max(p, q), ell + 2):
            try:
                B, blocks = staircase_retained(a, p, q, n, period)
            except ValueError as e:
                out.append({"a": a, "p": p, "q": q, "period": period, "skipped": str(e)})
                continue
            R: Set[Cell] = set().union(*blocks)
            res = decide(a, B, R, use_x=True)
            rec = {"experiment": "E1", "a": a, "p": p, "q": q, "ell": ell, "period": period, "n": n, "B": B,
                   "realizable": res["realizable"], "farkas_consistent": res["farkas_consistent"],
                   "verified": res.get("weights_verified", res.get("obstruction_verified")),
                   "predicted_realizable": period >= ell,
                   "value": str(_value([(p, q)] * n)), "Pstar_source": str(extremal_profile(a, B))}
            if not res["realizable"]:
                rec["obstruction_support"] = len(res["obstruction"]["mu"])
                rec["obstruction"] = res["obstruction"]
            else:
                rec["weights"] = res["weights"]
            out.append(rec)
    return out


def e2_two_classes(periods: int = 3) -> List[dict]:
    """a = 9; class 1: C(2,3) rectangles with i-range [0,3]; class 2: C(3,2)
    rectangles with i-range [5,8]; both at period 5, interleaved so that Y and Z
    are fully used (Y: {0,1} | {2,3,4} mod 5, Z: {1,2,3} | {4,0} mod 5)."""
    a = 9
    blocks, types = [], []
    for t in range(periods):
        c1 = rectangle_cells(5 * t, 2, 5 * t + 1, 3, a)
        blocks.append(c1)
        types.append((2, 3))
        if t < periods - 1:
            c2 = rectangle_cells(5 * t + 2, 3, 5 * t + 9, 2, a)
            blocks.append(c2)
            types.append((3, 2))
    assert all(b is not None for b in blocks)
    assert check_disjoint(blocks)
    B = max(j for b in blocks for (_, j) in b) + 1
    R = set().union(*blocks)
    res = decide(a, B, R, use_x=True)
    rec = {"experiment": "E2", "a": a, "B": B, "blocks": types, "realizable": res["realizable"],
           "farkas_consistent": res["farkas_consistent"],
           "verified": res.get("weights_verified", res.get("obstruction_verified")),
           "value": str(_value(types)), "Pstar_source": str(extremal_profile(a, B)),
           "density_note": "value per unit length 28/15 per class pair vs extremal slope H_9 ~ 2.30"}
    if not res["realizable"]:
        rec["obstruction"] = res["obstruction"]
    else:
        rec["weights"] = res["weights"]
    return [rec]


def e3_double_staircase(cases=((5, 3, 3), (7, 4, 4)), n: int = 3) -> List[dict]:
    out = []
    for a, p, q in cases:
        ell = p + q - 1
        blocks, types = [], []
        for b in range(n):
            y = b * ell
            blocks.append(rectangle_cells(y, p, y + p - 1, q, a))
            types.append((p, q))
            if b < n - 1 and p >= 2 and q >= 2:
                # gap rectangle: Y-gap [y+p, y+ell), Z-gap [y+ell, y+ell+p-1)
                g = rectangle_cells(y + p, q - 1, y + ell, p - 1, a)
                if g is not None:
                    blocks.append(g)
                    types.append((q - 1, p - 1))
        assert check_disjoint(blocks), (a, p, q)
        B = max(j for blk in blocks for (_, j) in blk) + 1
        R = set().union(*blocks)
        res = decide(a, B, R, use_x=True)
        rec = {"experiment": "E3", "a": a, "p": p, "q": q, "B": B, "blocks": types,
               "realizable": res["realizable"], "farkas_consistent": res["farkas_consistent"],
               "verified": res.get("weights_verified", res.get("obstruction_verified")),
               "value": str(_value(types)), "Pstar_source": str(extremal_profile(a, B))}
        if not res["realizable"]:
            rec["obstruction"] = res["obstruction"]
        out.append(rec)
    return out


def e4_three_sector_sanity(cases=((2, 1), (3, 2), (2, 3))) -> List[dict]:
    from .sectors import three_sector_certificate
    out = []
    for a, h in cases:
        cert = three_sector_certificate(a, h)
        B = cert["parameters"]["B"]
        R = {(i, j) for i, j, k, c in cert["retained_terms"]}
        for use_x in (False, True):
            res = decide(a, B, R, use_x=use_x)
            out.append({"experiment": "E4", "a": a, "h": h, "B": B, "use_x": use_x,
                        "realizable": res["realizable"], "farkas_consistent": res["farkas_consistent"],
                        "verified": res.get("weights_verified", res.get("obstruction_verified"))})
    return out


# ---------------------------------------------------------------------------------
# E5: bounded exhaustive packing search
# ---------------------------------------------------------------------------------

def templates(a: int, B: int):
    """All interval templates in C(a,B): (kind, type (p,q), Yset, Zset, cells)."""
    K = a + B - 1
    out = []
    # rectangles: Y=[y,y+p), Z=[z,z+q), all pairs in band; output on X
    for p in range(1, B + 1):
        for q in range(1, a + 1):
            if p + q - 1 > a:
                continue
            for y in range(0, B - p + 1):
                for z in range(max(0, y + p - 1), min(K - q, y + a - q) + 1):
                    cells = rectangle_cells(y, p, z, q, a)
                    if cells is None:
                        continue
                    out.append(("R", (p, q), frozenset(range(y, y + p)), frozenset(range(z, z + q)), frozenset(cells)))
    # normal: X carries [s,s+m), Y carries [y,y+h); Z = [y+s, y+s+m+h-1)
    for m in range(2, a + 1):
        for s in range(0, a - m + 1):
            for h in range(2, B + 1):
                for y in range(0, B - h + 1):
                    cells = normal_cells(s, m, y, h)
                    _, ys, zs = block_legs(cells)
                    out.append(("N", (m, h), frozenset(ys), frozenset(zs), frozenset(cells)))
    # exchanged: X carries [s,s+m), Z carries [z,z+h); Y = output
    for m in range(2, a + 1):
        for s in range(0, a - m + 1):
            for h in range(2, B + 1):
                for z in range(0, K - h + 1):
                    cells = exchanged_cells(s, m, z, h)
                    if any(not (0 <= j < B) for (_, j) in cells):
                        continue
                    _, ys, zs = block_legs(cells)
                    out.append(("E", (m, h), frozenset(ys), frozenset(zs), frozenset(cells)))
    # deduplicate identical cell sets (e.g. C(1,q) rectangles coincide with other shapes)
    seen = {}
    for t in out:
        key = t[4]
        if key not in seen or _value([t[1]]) > _value([seen[key][1]]):
            seen[key] = t
    return list(seen.values())


def e5_packing_search(a: int, B: int, max_lp: int = 400, min_blocks: int = 2, time_limit: float = 600.0):
    """Enumerate packings (disjoint Y, disjoint Z) with value > P*(a,B); test each
    for realizability by general weights.  Returns a summary dict."""
    t0 = time.time()
    target = extremal_profile(a, B)
    temps = templates(a, B)
    temps.sort(key=lambda t: (min(t[2]), -len(t[2])))
    by_ystart: Dict[int, list] = {}
    for t in temps:
        by_ystart.setdefault(min(t[2]), []).append(t)
    dens = max(_value([t[1]]) / (len(t[2]) + len(t[3])) for t in temps)
    K = a + B - 1
    found = []
    stats = {"templates": len(temps), "packings_above_target": 0, "lp_calls": 0, "realizable_above_target": 0,
             "max_density": str(dens), "truncated": False}
    best_realizable = None

    def dfs(ypos: int, usedZ: frozenset, chosen: list, value: Fraction, zfree: int):
        nonlocal best_realizable
        if time.time() - t0 > time_limit:
            stats["truncated"] = True
            return
        yfree = B - ypos
        if value + dens * (yfree + zfree) <= target:
            return
        if ypos >= B:
            if len(chosen) >= min_blocks and value > target:
                stats["packings_above_target"] += 1
                if stats["lp_calls"] >= max_lp:
                    stats["truncated"] = True
                    return
                R = set().union(*[t[4] for t in chosen])
                stats["lp_calls"] += 1
                st, w = solve_primal(a, B, R, use_x=True)
                rec = {"blocks": [(t[0], t[1], sorted(t[2]), sorted(t[3])) for t in chosen],
                       "value": str(value), "excess": str(value - target), "realizable": st == "feasible"}
                if st == "feasible":
                    rec["weights_verified"] = verify_realization(a, B, R, w, True)
                    stats["realizable_above_target"] += 1
                    if best_realizable is None or value > Fraction(best_realizable["value"]):
                        best_realizable = rec
                else:
                    sd, cert = solve_farkas(a, B, R, use_x=True)
                    rec["obstruction_verified"] = sd == "obstruction" and verify_obstruction(a, B, R, cert, True)
                    rec["obstruction_mass_cells"] = len(cert[0]) if cert else None
                found.append(rec)
            return
        # option 1: leave ypos unused (pad)
        dfs(ypos + 1, usedZ, chosen, value, zfree)
        # option 2: start a template at ypos
        for t in by_ystart.get(ypos, ()):
            if usedZ & t[3]:
                continue
            ymax = max(t[2])
            dfs(ymax + 1, usedZ | t[3], chosen + [t], value + _value([t[1]]), zfree - len(t[3]))

    dfs(0, frozenset(), [], Fraction(0), K)
    stats["seconds"] = time.time() - t0
    return {"experiment": "E5", "a": a, "B": B, "Pstar_source": str(target), "stats": stats,
            "best_realizable_above_target": best_realizable,
            "sample_obstructed": [r for r in found if not r["realizable"]][:10]}
