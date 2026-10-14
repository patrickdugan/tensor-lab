"""Finite shared-leg separation (paper, Proposition 3.1), computed exactly.

Input: a tensor A = sum_{h=1}^M A_h on a common first leg X, where the blocks
A_h live on disjoint Y-sectors Y_h and disjoint Z-sectors Z_h and there is no
term between unmatched sectors (a *certified compatible block decomposition*,
e.g. the retained tensor of a three-sector certificate).

Construction: L = 5M copies of A, indexed by r = 0..L-1, with a primitive L-th
root of unity zeta and target variables X_{a,g}, Y_{h,b,u}, Z_{h,c,v}
(g,u,v in 1..M):

    x_a^(r)   -> sum_g zeta^{2rg}      X_{a,g}
    y_{h,b}^(r) -> sum_u zeta^{r(u-h)}  Y_{h,b,u}
    z_{h,c}^(r) -> (1/L) sum_v zeta^{r(-v-h)} Z_{h,c,v}

Summing over r multiplies each target coefficient by (1/L) sum_r zeta^{r[u-v+2(g-h)]},
the indicator of L | u-v+2(g-h); since |u-v+2(g-h)| <= 3(M-1) < L this is the
indicator of u - v + 2(g-h) = 0.  Weights w(X_{a,g}) = g^2, w(Y_{h,b,u}) = hu - h^2,
w(Z_{h,c,v}) = -hv then give every surviving term weight (g-h)^2, so the
weight-zero tensor is the full direct sum of the A_h (x) B_X(M).

Field assumptions (recorded in the certificate):
  * the root-of-unity filter needs a primitive L-th root of unity and L
    invertible in the field: true over C and over any field whose
    characteristic does not divide L and which contains zeta_L;
  * the weight bookkeeping and the block identification are purely
    combinatorial (valid over every field);
  * the passage to characters (Lemma 2.3) uses interpolation at jL+1 distinct
    nonzero points, hence an infinite (e.g. characteristic-zero) field.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

from .cyclotomic import Cyclotomic, filter_sum
from .tensors import SparseTensor, dot_product, tensor_product


class SeparationResult:
    def __init__(self):
        self.checks: List[Tuple[str, bool, str]] = []
        self.data: Dict = {}

    def add(self, name, ok, detail=""):
        self.checks.append((name, bool(ok), detail))

    @property
    def ok(self):
        return all(ok for _, ok, _ in self.checks)

    def summary(self) -> str:
        out = [f"[{'PASS' if ok else 'FAIL'}] {n}" + (f": {d}" if d else "") for n, ok, d in self.checks]
        out.append("VERDICT: " + ("PASS" if self.ok else "FAIL"))
        return "\n".join(out)


def separation(retained: SparseTensor, y_sectors: Sequence[Sequence[int]],
               z_sectors: Sequence[Sequence[int]], L: int | None = None) -> SeparationResult:
    """Carry out Proposition 3.1 on a block decomposition and verify it exactly.

    Blocks are labelled h = 1..M in the order of the sector lists; sector h of Y
    is matched with sector h of Z.  Terms outside every sector, or joining
    unmatched sectors, make the input invalid."""
    res = SeparationResult()
    M = len(y_sectors)
    if len(z_sectors) != M or M < 1:
        res.add("matched sector counts", False)
        return res
    L = 5 * M if L is None else int(L)
    res.add("Fourier modulus L = 5M", L == 5 * M, f"L={L}, M={M}")
    ylab = {c: h + 1 for h, s in enumerate(y_sectors) for c in s}
    zlab = {c: h + 1 for h, s in enumerate(z_sectors) for c in s}
    bad = [k for k in retained.terms if ylab.get(k[1]) is None or zlab.get(k[2]) is None or ylab[k[1]] != zlab[k[2]]]
    res.add("input is a compatible block decomposition (no unmatched or stray terms)", not bad, f"{len(bad)} bad terms")
    if bad:
        return res
    nx = retained.dims[0]
    # target coordinate spaces
    X_idx = {(a, g): a * M + (g - 1) for a in range(nx) for g in range(1, M + 1)}
    Y_idx: Dict[Tuple[int, int, int], int] = {}
    Z_idx: Dict[Tuple[int, int, int], int] = {}
    for h, sec in enumerate(y_sectors, start=1):
        for b in sec:
            for u in range(1, M + 1):
                Y_idx[(h, b, u)] = len(Y_idx)
    for h, sec in enumerate(z_sectors, start=1):
        for c in sec:
            for v in range(1, M + 1):
                Z_idx[(h, c, v)] = len(Z_idx)
    dims = (len(X_idx), len(Y_idx), len(Z_idx))

    # exact filter identity, checked by summation for every exponent in range
    rng = 3 * (M - 1)
    filt_ok = True
    for e in range(-rng, rng + 1):
        val = filter_sum(L, e)
        want = 1 if e == 0 else 0
        if not (val.is_rational() and val.as_rational() == want):
            filt_ok = False
            break
    res.add(f"root-of-unity filter (1/L) sum_r zeta^(re) = [e=0] for |e| <= {rng}, computed in Q(zeta_{L})", filt_ok)
    res.add("no modular wrap: 3(M-1) < L", rng < L)

    # transformed tensor: sum over the L copies of the substituted terms
    inv_L = Fraction(1, L)
    coeff: Dict[Tuple[int, int, int], Cyclotomic] = {}
    zeta_cache: Dict[int, Cyclotomic] = {}

    def zp(e: int) -> Cyclotomic:
        e %= L
        if e not in zeta_cache:
            zeta_cache[e] = Cyclotomic.zeta_power(L, e)
        return zeta_cache[e]

    # For efficiency, the sum over r of zeta^{r * phase} is computed once per phase.
    phase_sum: Dict[int, Cyclotomic] = {}

    def psum(phase: int) -> Cyclotomic:
        if phase not in phase_sum:
            s = Cyclotomic.zero(L)
            for r in range(L):
                s = s + zp(r * phase)
            phase_sum[phase] = s.scale(inv_L)
        return phase_sum[phase]

    for (a, b, c), coef in retained.terms.items():
        h = ylab[b]
        for g in range(1, M + 1):
            for u in range(1, M + 1):
                for v in range(1, M + 1):
                    phase = u - v + 2 * (g - h)
                    val = psum(phase).scale(coef)
                    if val == 0:
                        continue
                    key = (X_idx[(a, g)], Y_idx[(h, b, u)], Z_idx[(h, c, v)])
                    coeff[key] = coeff.get(key, Cyclotomic.zero(L)) + val
    # every coefficient must be rational and equal to coef * [phase == 0]
    all_rational = all(val.is_rational() for val in coeff.values())
    res.add("every transformed coefficient is rational (phases cancel exactly)", all_rational)
    transformed = SparseTensor(dims)
    for key, val in coeff.items():
        if val.is_rational() and val.as_rational() != 0:
            transformed.terms[key] = val.as_rational()
    expected = SparseTensor(dims)
    for (a, b, c), coef in retained.terms.items():
        h = ylab[b]
        for g in range(1, M + 1):
            for u in range(1, M + 1):
                for v in range(1, M + 1):
                    if u - v + 2 * (g - h) == 0:
                        expected.terms[(X_idx[(a, g)], Y_idx[(h, b, u)], Z_idx[(h, c, v)])] = coef
    res.add("transformed tensor = sum over terms with u - v + 2(g-h) = 0 (eq. 3.2)", transformed == expected,
            f"{transformed.nnz} terms")

    # weights
    wX = [0] * dims[0]
    for (a, g), idx in X_idx.items():
        wX[idx] = g * g
    wY = [0] * dims[1]
    for (h, b, u), idx in Y_idx.items():
        wY[idx] = h * u - h * h
    wZ = [0] * dims[2]
    for (h, c, v), idx in Z_idx.items():
        wZ[idx] = -h * v
    res.add("some individual weights are negative (allowed; total degree stays nonnegative)",
            any(w < 0 for w in wZ) or M == 1)
    inv_X = {idx: ag for ag, idx in X_idx.items()}
    inv_Y = {idx: hbu for hbu, idx in Y_idx.items()}
    inv_Z = {idx: hcv for hcv, idx in Z_idx.items()}
    weights_ok = True
    zero_terms = SparseTensor(dims)
    maxw = 0
    for key, coef in transformed.terms.items():
        w = wX[key[0]] + wY[key[1]] + wZ[key[2]]
        a, g = inv_X[key[0]]
        h, b, u = inv_Y[key[1]]
        h2, c, v = inv_Z[key[2]]
        if w != (g - h) ** 2 or h != h2:
            weights_ok = False
        maxw = max(maxw, w)
        if w == 0:
            zero_terms.terms[key] = coef
    res.add("total weight of every surviving term equals (g-h)^2", weights_ok)
    res.add("minimum total weight is 0 and no surviving term has negative weight",
            all(wX[k[0]] + wY[k[1]] + wZ[k[2]] >= 0 for k in transformed.terms) and
            (zero_terms.nnz > 0))
    res.add(f"maximum total weight {maxw} <= (M-1)^2 = {(M-1)**2}", maxw <= (M - 1) ** 2)
    # weight-zero terms are exactly g = h and u = v
    gz_ok = all(inv_X[k[0]][1] == inv_Y[k[1]][0] and inv_Y[k[1]][2] == inv_Z[k[2]][2] for k in zero_terms.terms)
    res.add("weight-zero terms have g = h and u = v exactly", gz_ok)

    # the weight-zero tensor equals the full direct sum of A_h (x) B_X(M)
    D = SparseTensor(dims)
    legs_used = [set(), set(), set()]
    for h, (ysec, zsec) in enumerate(zip(y_sectors, z_sectors), start=1):
        block = retained.sub_tensor([None, list(ysec), list(zsec)])
        # A_h (x) B_X(M): B_X(M) = x sum_u y_u z_u
        for (a, b, c), coef in block.terms.items():
            for u in range(1, M + 1):
                key = (X_idx[(a, h)], Y_idx[(h, b, u)], Z_idx[(h, c, u)])
                D.terms[key] = coef
                for l in range(3):
                    legs_used[l].add((h, key[l]))
    res.add("weight-zero tensor = (+)_h A_h (x) B_X(M) with explicit coordinate identification", D == zero_terms,
            f"{D.nnz} terms")
    # fullness: distinct h use disjoint coordinates on all three legs
    for l, name in enumerate("XYZ"):
        by_h: Dict[int, set] = {}
        for h, idx in legs_used[l]:
            by_h.setdefault(h, set()).add(idx)
        hs = list(by_h)
        disjoint = all(not (by_h[hs[i]] & by_h[hs[j]]) for i in range(len(hs)) for j in range(i + 1, len(hs)))
        res.add(f"blocks use disjoint {name}-coordinates (full direct sum)", disjoint)
    # reference construction through the tensor library, independent of the index bookkeeping above
    ref_blocks = []
    for h, (ysec, zsec) in enumerate(zip(y_sectors, z_sectors), start=1):
        local = retained.restrict_to_support([list(range(nx)), list(ysec), list(zsec)])
        ref_blocks.append(tensor_product(local, dot_product(M, 0)))
    res.add("block (x) dot-product term counts match the library construction",
            sum(b.nnz for b in ref_blocks) == D.nnz)
    res.data = {
        "M": M, "L": L, "target_dims": list(dims), "source_copies": L,
        "transformed_nnz": transformed.nnz, "weight_zero_nnz": zero_terms.nnz,
        "max_weight": maxw,
        "weights": {"X": wX, "Y": wY, "Z": wZ},
        "x_index": {f"{a},{g}": idx for (a, g), idx in X_idx.items()},
        "field_assumptions": [
            f"primitive {L}-th root of unity present and {L} invertible (char 0, or char p with p not dividing {L})",
            "weights and block identification: valid over every field",
            "character inequality (Lemma 2.3): interpolation over an infinite field",
        ],
    }
    return res


def separation_from_certificate(cert: dict) -> SeparationResult:
    """Apply Proposition 3.1 to the retained tensor of a certificate.

    The certificate is first checked by the independent verifier; only a
    verified, X-shared, fully identified block decomposition is accepted as a
    compatible input.  Blocks are labelled in the certificate's block order."""
    from .verify import verify_certificate
    rep = verify_certificate(cert)
    if not rep.ok:
        res = SeparationResult()
        res.add("input certificate verifies", False, "certificate failed independent verification")
        return res
    dims = tuple(cert["source"]["dims"])
    retained = SparseTensor(dims)
    for i, j, k, c in cert["retained_terms"]:
        retained[(i, j, k)] = Fraction(c)
    ys = [list(b["y_coords"]) for b in cert["blocks"]]
    zs = [list(b["z_coords"]) for b in cert["blocks"]]
    res = separation(retained, ys, zs)
    res.checks.insert(0, ("input certificate verifies", True, f"{len(cert['blocks'])} blocks"))
    return res
