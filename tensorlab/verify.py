"""Independent certificate verifier.

This module deliberately does NOT import the search/construction code
(``tensors``, ``degeneration``, ``sectors``).  It rebuilds the source tensor
from the family specification with its own code, recomputes every weight,
re-derives the minimum-weight set, re-checks the sector partition, and
re-checks every block identification by pulling the block back along the
stated coordinate maps and comparing with its own copy of the target tensor.
Only integers and ``fractions.Fraction`` are used.

A passing verification establishes the *finite* identity for the parameters
in the certificate and nothing more; see ``scope`` in the certificate.
"""
from __future__ import annotations

import json
from fractions import Fraction
from typing import Dict, List, Tuple

Index = Tuple[int, int, int]

_PERMS = {
    "identity": (0, 1, 2), "swap_YZ": (0, 2, 1), "swap_XY": (1, 0, 2),
    "swap_XZ": (2, 1, 0), "cycle_XYZ": (1, 2, 0), "cycle_XZY": (2, 0, 1),
}


class VerificationError(Exception):
    pass


def _build(spec: dict) -> Tuple[Tuple[int, int, int], Dict[Index, Fraction]]:
    """Own implementation of the tensor families (kept minimal on purpose)."""
    fam = spec["family"]
    if fam == "convolution":
        a, b = int(spec["a"]), int(spec["b"])
        terms = {(i, j, i + j): Fraction(1) for i in range(a) for j in range(b)}
        return (a, b, max(a + b - 1, 0)), terms
    if fam == "cyclic_convolution":
        n = int(spec["n"])
        terms = {(i, j, (i + j) % n): Fraction(1) for i in range(n) for j in range(n)}
        return (n, n, n), terms
    if fam == "dot_product":
        m, leg = int(spec["m"]), int(spec.get("leg", 0))
        base = {(0, i, i): Fraction(1) for i in range(m)}
        if leg == 0:
            return (1, m, m), base
        if leg == 1:
            return (m, 1, m), {(i, 0, i): Fraction(1) for i in range(m)}
        return (m, m, 1), {(i, i, 0): Fraction(1) for i in range(m)}
    if fam == "explicit":
        t = spec["tensor"]
        return tuple(t["dims"]), {(i, j, k): Fraction(c) for i, j, k, c in t["terms"]}
    if fam == "tensor_product":
        dl, tl = _build(spec["left"])
        dr, tr = _build(spec["right"])
        dims = tuple(dl[l] * dr[l] for l in range(3))
        terms = {}
        for kl, cl in tl.items():
            for kr, cr in tr.items():
                terms[tuple(kl[l] * dr[l] + kr[l] for l in range(3))] = cl * cr
        return dims, terms
    raise VerificationError(f"verifier does not know family {fam!r}")


class Report:
    def __init__(self):
        self.checks: List[Tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = ""):
        self.checks.append((name, bool(ok), detail))

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    def summary(self) -> str:
        lines = []
        for name, ok, detail in self.checks:
            lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        lines.append("VERDICT: " + ("PASS" if self.ok else "FAIL"))
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {"ok": self.ok, "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in self.checks]}


def verify_certificate(cert: dict) -> Report:
    rep = Report()
    try:
        _verify(cert, rep)
    except VerificationError as e:
        rep.add("exception", False, str(e))
    except (KeyError, TypeError, ValueError, IndexError) as e:  # malformed certificate
        rep.add("malformed certificate", False, repr(e))
    return rep


def _verify(cert: dict, rep: Report) -> None:
    if cert.get("certificate_version") != 1:
        raise VerificationError("unsupported certificate version")
    dims, source = _build(cert["source"])
    rep.add("source rebuilt from family spec", True, f"dims={dims} nnz={len(source)}")
    if list(dims) != list(cert["source"]["dims"]):
        rep.add("source dims match certificate", False, f"{dims} vs {cert['source']['dims']}")
        return
    rep.add("source dims match certificate", True)
    rep.add("source nnz matches certificate", len(source) == cert["source"]["nnz"])

    # weights
    w = cert["weights"]
    wX, wY, wZ = [list(map(int, w[k])) for k in ("X", "Y", "Z")]
    rep.add("weight vectors have one entry per coordinate",
            (len(wX), len(wY), len(wZ)) == dims, f"{(len(wX), len(wY), len(wZ))} vs {dims}")
    if (len(wX), len(wY), len(wZ)) != dims:
        return
    total = {key: wX[key[0]] + wY[key[1]] + wZ[key[2]] for key in source}
    m = min(total.values())
    rep.add("minimum weight recomputed", m == int(cert["min_weight"]), f"recomputed {m}, stated {cert['min_weight']}")
    rep.add("no supported term has weight below the stated minimum", all(v >= m for v in total.values()))
    retained_stated = {(i, j, k): Fraction(c) for i, j, k, c in cert["retained_terms"]}
    retained = {key: c for key, c in source.items() if total[key] == m}
    rep.add("retained set equals the exact minimum-weight set",
            retained == retained_stated,
            f"{len(retained)} recomputed vs {len(retained_stated)} stated")
    erased_stated = {(i, j, k): (Fraction(c), int(wt)) for i, j, k, c, wt in cert["erased_terms"]}
    erased = {key: (c, total[key]) for key, c in source.items() if total[key] != m}
    rep.add("erased set and weights recomputed", erased == erased_stated,
            f"{len(erased)} recomputed vs {len(erased_stated)} stated")
    rep.add("retained + erased = source (every coefficient accounted for)",
            set(retained) | set(erased) == set(source) and not (set(retained) & set(erased)))
    if "max_weight" in cert:
        mw = max([wt for _, wt in erased.values()], default=m)
        rep.add("maximum weight recomputed", mw == int(cert["max_weight"]), f"{mw}")

    # sectors
    ys = [list(map(int, s)) for s in cert["sectors"]["Y"]]
    zs = [list(map(int, s)) for s in cert["sectors"]["Z"]]
    yflat = [c for s in ys for c in s]
    zflat = [c for s in zs for c in s]
    rep.add("Y sectors are disjoint and within range",
            len(set(yflat)) == len(yflat) and all(0 <= c < dims[1] for c in yflat))
    rep.add("Z sectors are disjoint and within range",
            len(set(zflat)) == len(zflat) and all(0 <= c < dims[2] for c in zflat))
    ylab = {c: s for s, sec in enumerate(ys) for c in sec}
    zlab = {c: s for s, sec in enumerate(zs) for c in sec}
    strays = [key for key in retained if key[1] not in ylab or key[2] not in zlab]
    rep.add("every retained term lies in a Y sector and a Z sector", not strays, f"{len(strays)} stray")
    rep.add("stray list matches", sorted(tuple(s) for s in cert.get("stray_terms", [])) == sorted(strays))

    # blocks
    blocks = cert["blocks"]
    if cert.get("shared_leg", "X") != "X":
        raise VerificationError("only X-shared certificates are supported")
    covered_y, covered_z = set(), set()
    covered_terms: Dict[Index, Fraction] = {}
    ysec_used, zsec_used = set(), set()
    for blk in blocks:
        label = blk["label"]
        ysecs, zsecs = list(blk["y_sectors"]), list(blk["z_sectors"])
        y_coords = sorted(c for s in ysecs for c in ys[s])
        z_coords = sorted(c for s in zsecs for c in zs[s])
        rep.add(f"block {label}: coordinate lists match its sectors",
                y_coords == sorted(blk["y_coords"]) and z_coords == sorted(blk["z_coords"]))
        rep.add(f"block {label}: sectors not reused by another block",
                not (set(ysecs) & ysec_used) and not (set(zsecs) & zsec_used))
        ysec_used |= set(ysecs)
        zsec_used |= set(zsecs)
        yset, zset = set(y_coords), set(z_coords)
        bterms = {key: c for key, c in retained.items() if key[1] in yset and key[2] in zset}
        rep.add(f"block {label}: nnz", len(bterms) == int(blk["nnz"]), f"{len(bterms)} vs {blk['nnz']}")
        # no retained term joins this block's Y sectors to another block's Z sectors
        cross = [key for key in retained if (key[1] in yset) != (key[2] in zset)]
        rep.add(f"block {label}: no cross-sector contamination", not cross, f"{len(cross)} cross terms")
        covered_y |= yset
        covered_z |= zset
        covered_terms.update(bterms)
        if blk.get("identified"):
            tdims, target = _build(blk["target"])
            p = _PERMS[blk["leg_permutation"]]
            maps = [list(map(int, blk["maps"][k])) for k in ("X", "Y", "Z")]
            ok_len = all(len(maps[t]) == tdims[t] for t in range(3))
            rep.add(f"block {label}: map lengths equal target dims", ok_len)
            if not ok_len:
                continue
            # reordered block: new leg t carries block leg p[t]
            reordered = {(key[p[0]], key[p[1]], key[p[2]]): c for key, c in bterms.items()}
            pulled = {}
            for i in range(tdims[0]):
                for j in range(tdims[1]):
                    for k in range(tdims[2]):
                        c = reordered.get((maps[0][i], maps[1][j], maps[2][k]))
                        if c is not None:
                            pulled[(i, j, k)] = c
            rep.add(f"block {label}: pullback along maps equals target {blk['target']}"
                    f" with permutation {blk['leg_permutation']}", pulled == target)
            inj = all(len(set(maps[t])) == len(maps[t]) for t in range(3))
            rep.add(f"block {label}: coordinate maps injective", inj)
            # the images must cover the block's support, so nothing of the block is lost
            sup = [set(), set(), set()]
            for key in reordered:
                for t in range(3):
                    sup[t].add(key[t])
            rep.add(f"block {label}: map images cover the block support",
                    all(set(maps[t]) >= sup[t] for t in range(3)))
            # shared first leg: the target leg mapped onto X must use a bijection of the
            # block's X support; and X is the ambient shared leg (dimension a)
            rep.add(f"block {label}: X-leg map is a bijection onto the shared X support",
                    sorted(maps[p.index(0)]) == sorted(sup[p.index(0)]))
        else:
            rep.add(f"block {label}: identified", False, "block not identified")
    rep.add("blocks partition the retained terms", covered_terms == retained,
            f"{len(covered_terms)} covered vs {len(retained)} retained")
    rep.add("blocks use disjoint Y coordinates",
            sum(len(set(b["y_coords"])) for b in blocks) == len(covered_y))
    rep.add("blocks use disjoint Z coordinates",
            sum(len(set(b["z_coords"])) for b in blocks) == len(covered_z))

    exp = cert.get("expected")
    if exp:
        rep.add("expected number of blocks", len(blocks) == int(exp["num_blocks"]),
                f"{len(blocks)} vs {exp['num_blocks']}")
        tgts = exp.get("block_targets")
        if tgts:
            got = sorted(_canonical_target(b.get("target")) for b in blocks)
            want = sorted(_canonical_target(t) for t in tgts)
            rep.add("expected block targets (C(p,q) ~ C(q,p) by the input-leg swap)", got == want,
                    f"{got} vs {want}")
        if exp.get("middle_block_permutation"):
            mids = [b for b in blocks if b["label"] == 1]
            rep.add("middle block uses the Y/Z exchange",
                    bool(mids) and mids[0].get("leg_permutation") == exp["middle_block_permutation"])
            if mids and cert["kind"] == "three_sector_shifted_tripling":
                a = int(cert["parameters"]["a"])
                xm = list(mids[0]["maps"]["X"])
                rep.add("middle block reverses the first coordinate", xm == list(range(a - 1, -1, -1)), str(xm))
        if cert["kind"] == "three_sector_shifted_tripling":
            a, h = int(cert["parameters"]["a"]), int(cert["parameters"]["h"])
            rep.add("every erased term has weight exactly +1 (paper: no negative weights)",
                    all(wt == 1 for _, wt in erased.values()) and m == 0)
            rep.add("parameters consistent: B = 3h + a - 1",
                    int(cert["parameters"]["B"]) == 3 * h + a - 1 == int(cert["source"]["b"]))


def _canonical_target(t) -> str:
    """C(p,q) and C(q,p) are isomorphic (exchange the two input legs), and the
    verifier has already checked the exact leg permutation of each block, so
    expected targets are compared up to that symmetry."""
    if t is None:
        return "null"
    t = dict(t)
    if t.get("family") == "convolution":
        a, b = sorted((int(t["a"]), int(t["b"])))
        t["a"], t["b"] = a, b
    return json.dumps(t, sort_keys=True)


def verify_file(path: str) -> Report:
    with open(path, "r", encoding="utf-8") as f:
        cert = json.load(f)
    return verify_certificate(cert)
