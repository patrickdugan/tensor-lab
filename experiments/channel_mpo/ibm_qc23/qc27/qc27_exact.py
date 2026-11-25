"""QC-27 exact-arithmetic (fractions.Fraction) pipeline for Clifford + dyadic
non-unital single-qubit channels.

Everything here is rational: Pauli coefficients, channel parameters, ledger
entries (l1, l2^2, support sizes, survival factors) and the bound itself.  The
only irrational operation is sqrt, replaced by a rational upper bound
`ceil_sqrt` (error < 2^-64).  An independent Schroedinger-picture density
matrix (complex-rational entries, matrix-unit channel actions) provides the
exact reference for n <= 4.

Also contains: exact Choi PSD (LDL^T) certificates, the witness emitter and
verifier prototype, and `failed_hypotheses()` with exact counterexamples.
"""
from __future__ import annotations

import json
import random
from fractions import Fraction as Fr
from math import isqrt

# ----------------------------------------------------------------------------
# Pauli algebra with exact phases.  codes: 0=I 1=X 2=Z 3=Y  (l = x + 2z)
# ----------------------------------------------------------------------------
def _xz(l):
    return l & 1, (l >> 1) & 1


def _code(x, z):
    return x + 2 * z


def pmul(a, b):
    """Product of Pauli strings (tuples of codes): returns (i-exponent mod 4, string)."""
    ph = 0
    out = []
    for la, lb in zip(a, b):
        x1, z1 = _xz(la)
        x2, z2 = _xz(lb)
        x3, z3 = x1 ^ x2, z1 ^ z2
        ph += x1 * z1 + x2 * z2 + 2 * z1 * x2 - x3 * z3
        out.append(_code(x3, z3))
    return ph % 4, tuple(out)


def _clifford_table_1q():
    return {
        "h": {0: (1, 0), 1: (1, 2), 2: (1, 1), 3: (-1, 3)},
        "s": {0: (1, 0), 1: (-1, 3), 2: (1, 2), 3: (1, 1)},
        "x": {0: (1, 0), 1: (1, 1), 2: (-1, 2), 3: (-1, 3)},
        "z": {0: (1, 0), 1: (-1, 1), 2: (1, 2), 3: (-1, 3)},
    }


_T1 = _clifford_table_1q()


def _gen_images(name):
    """Images (as 2-qubit strings) of X_a, Z_a, X_b, Z_b under the 2q Clifford (a=control/first)."""
    X, Z, I = 1, 2, 0
    if name == "cx":
        return {("x", 0): (X, X), ("z", 0): (Z, I), ("x", 1): (I, X), ("z", 1): (Z, Z)}
    if name == "cz":
        return {("x", 0): (X, Z), ("z", 0): (Z, I), ("x", 1): (Z, X), ("z", 1): (I, Z)}
    raise ValueError(name)


def _clifford_table_2q(name):
    img = _gen_images(name)
    tab = {}
    for la in range(4):
        for lb in range(4):
            cur = (0, 0)
            ph = 0
            for (q, l) in ((0, la), (1, lb)):
                x, z = _xz(l)
                ph += x * z
                if x:
                    p, cur = pmul(cur, img[("x", q)])
                    ph += p
                if z:
                    p, cur = pmul(cur, img[("z", q)])
                    ph += p
            ph %= 4
            if ph not in (0, 2):
                raise AssertionError("non-Hermitian Clifford image")
            tab[(la, lb)] = (1 if ph == 0 else -1, cur)
    return tab


_T2 = {"cx": _clifford_table_2q("cx"), "cz": _clifford_table_2q("cz")}


# ----------------------------------------------------------------------------
# exact channels (dyadic parameters)
# ----------------------------------------------------------------------------
def choi(mu, lam, tau):
    h = Fr(1, 2)
    return [[(1 + lam + tau) * h, 0, 0, mu],
            [0, (1 - lam - tau) * h, 0, 0],
            [0, 0, (1 + tau - lam) * h, 0],
            [mu, 0, 0, (1 - tau + lam) * h]]


def ldl_psd(M):
    """Exact LDL^T of a symmetric matrix; returns (L, D) or None when not PSD."""
    n = len(M)
    A = [[Fr(v) for v in row] for row in M]
    L = [[Fr(int(i == j)) for j in range(n)] for i in range(n)]
    D = []
    for k in range(n):
        d = A[k][k]
        if d < 0:
            return None
        D.append(d)
        if d == 0:
            if any(A[k][j] != 0 for j in range(k + 1, n)):
                return None
            continue
        for i in range(k + 1, n):
            L[i][k] = A[i][k] / d
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] -= L[i][k] * A[k][j]
    return L, D


def ldl_check(M, L, D):
    n = len(M)
    for i in range(n):
        for j in range(n):
            s = sum(L[i][k] * D[k] * L[j][k] for k in range(n))
            if s != Fr(M[i][j]):
                return False
    return all(d >= 0 for d in D) and all(L[i][i] == 1 and all(L[i][j] == 0 for j in range(i + 1, n))
                                          for i in range(n))


class Chan:
    """Single-qubit channel with Heisenberg action I->I, X->mu X, Y->mu Y, Z->lam Z + tau I
    and certified depolarising content kappa (E^dag = (1-kappa) Psi + kappa tau_q)."""

    def __init__(self, mu, lam, tau, kappa=Fr(0), name=""):
        self.mu, self.lam, self.tau, self.kappa = Fr(mu), Fr(lam), Fr(tau), Fr(kappa)
        self.name = name

    @staticmethod
    def ad_dep(m, p):
        """Amplitude damping (sqrt(1-gamma) = m, so gamma = 1-m^2) followed by depolarising p."""
        m, p = Fr(m), Fr(p)
        return Chan((1 - p) * m, (1 - p) * m * m, (1 - p) * (1 - m * m), p, "ad(m=%s)+dep(p=%s)" % (m, p))

    @property
    def unital(self):
        return self.tau == 0

    def choi(self):
        return choi(self.mu, self.lam, self.tau)

    def certificates(self):
        C = self.choi()
        ldl = ldl_psd(C)
        Ck = [[C[i][j] - (self.kappa / 2 if i == j else 0) for j in range(4)] for i in range(4)]
        ldlk = ldl_psd(Ck)
        return dict(cp=ldl, kappa=ldlk)

    def verified(self):
        c = self.certificates()
        return c["cp"] is not None and c["kappa"] is not None

    def schro_units(self):
        """Matrix-unit action (independent of the Pauli form): dict (a,b) -> {(a2,b2): coeff}."""
        h = Fr(1, 2)
        lam, tau, mu = self.lam, self.tau, self.mu
        return {(0, 0): {(0, 0): (1 + lam + tau) * h, (1, 1): (1 - lam - tau) * h},
                (1, 1): {(0, 0): (1 + tau - lam) * h, (1, 1): (1 - tau + lam) * h},
                (0, 1): {(0, 1): mu}, (1, 0): {(1, 0): mu}}


def ad_dep_schro_units(m, p):
    """Physical composition (amplitude damping then depolarising) on matrix units, built from
    the textbook formulas rho00'=rho00+g rho11, rho11'=(1-g) rho11, rho01'=m rho01 -- independent
    of Chan.ad_dep's Pauli parameters."""
    m, p = Fr(m), Fr(p)
    g = 1 - m * m

    def ad(a, b):
        if (a, b) == (0, 0):
            return {(0, 0): Fr(1)}
        if (a, b) == (1, 1):
            return {(0, 0): g, (1, 1): 1 - g}
        return {(a, b): m}
    out = {}
    for ab in ((0, 0), (1, 1), (0, 1), (1, 0)):
        acc = {}
        for (a2, b2), c in ad(*ab).items():
            # depolarising: (1-p) rho + p Tr(rho) I/2
            acc[(a2, b2)] = acc.get((a2, b2), 0) + (1 - p) * c
            if a2 == b2:
                for k in (0, 1):
                    acc[(k, k)] = acc.get((k, k), 0) + p * c / 2
        out[ab] = acc
    return out


# ----------------------------------------------------------------------------
# circuits
# ----------------------------------------------------------------------------
class ExCircuit:
    """steps: list of (ops, trunc).  op = ('cl', name, sites) | ('noise', [(q, Chan), ...])."""

    def __init__(self, n, target, steps):
        self.n, self.target, self.steps = n, target, steps
        self.flat = []
        self.groups = []
        for ops, tr in steps:
            s = len(self.flat)
            self.flat.extend(ops)
            self.groups.append((s, len(self.flat), tr))


def op_mask(op):
    if op[0] == "cl":
        return sum(1 << s for s in op[2])
    return sum(1 << q for q, _ in op[1])


def random_excircuit(n, depth, seed, mset=(Fr(1, 2), Fr(3, 4), Fr(7, 8)), pset=(Fr(1, 16), Fr(1, 8), Fr(1, 4)),
                     target=None):
    rng = random.Random(seed)
    target = n // 2 if target is None else target
    steps = []
    for layer in range(depth):
        ops = []
        for q in range(n):
            c = rng.choice(["i", "h", "s", "hs"])
            if c in ("h", "hs"):
                ops.append(("cl", "h", (q,)))
            if c in ("s", "hs"):
                ops.append(("cl", "s", (q,)))
        steps.append((ops, False))
        steps.append(([("noise", [(q, Chan.ad_dep(rng.choice(mset), rng.choice(pset)))]) for q in range(n)], True))
        pairs = [(a, a + 1) for a in range(layer % 2, n - 1, 2)]
        ops = []
        for (a, b) in pairs:
            ops.append(("cl", rng.choice(["cx", "cz"]), (a, b)))
        steps.append((ops, False))
        ops = [("noise", [(a, Chan.ad_dep(rng.choice(mset), rng.choice(pset))),
                          (b, Chan.ad_dep(rng.choice(mset), rng.choice(pset)))]) for (a, b) in pairs]
        steps.append((ops, True))
    return ExCircuit(n, target, steps)


# ----------------------------------------------------------------------------
# exact Heisenberg propagation
# ----------------------------------------------------------------------------
def _apply_cl(O, name, sites):
    out = {}
    for s, c in O.items():
        if len(sites) == 1:
            q = sites[0]
            sg, l2 = _T1[name][s[q]]
            t = s[:q] + (l2,) + s[q + 1:]
        else:
            a, b = sites
            sg, (la, lb) = _T2[name][(s[a], s[b])]
            t = list(s)
            t[a], t[b] = la, lb
            t = tuple(t)
        out[t] = out.get(t, 0) + sg * c
    return {k: v for k, v in out.items() if v != 0}


def _apply_ch(O, q, ch):
    out = {}
    for s, c in O.items():
        l = s[q]
        if l == 0:
            terms = [(0, c)]
        elif l in (1, 3):
            terms = [(l, ch.mu * c)]
        else:
            terms = [(2, ch.lam * c)]
            if ch.tau:
                terms.append((0, ch.tau * c))
        for l2, v in terms:
            t = s[:q] + (l2,) + s[q + 1:]
            out[t] = out.get(t, 0) + v
    return {k: v for k, v in out.items() if v != 0}


def _supp_mask(s):
    return sum(1 << q for q, l in enumerate(s) if l)


def ceil_sqrt(q, bits=64):
    """Rational upper bound on sqrt(q), q >= 0 rational, within 2^-bits."""
    q = Fr(q)
    if q == 0:
        return Fr(0)
    num = -((-q.numerator * 4 ** bits) // q.denominator)   # ceil(q 4^bits)
    r = isqrt(num)
    if r * r < num:
        r += 1
    return Fr(r, 2 ** bits)


def exact_survival(flat, pos, support_mask):
    """Fraction version of the block tracker (Theorem 3), independent of qc27_bounds.survival."""
    blocks = [1 << q for q in range(support_mask.bit_length()) if (support_mask >> q) & 1]
    sigma = Fr(1)
    for i in range(pos - 1, -1, -1):
        if not blocks:
            break
        op = flat[i]
        S = op_mask(op)
        if not any(b & S for b in blocks):
            continue
        if op[0] == "cl":
            hit, rest = 0, []
            for b in blocks:
                if b & S:
                    hit |= b
                else:
                    rest.append(b)
            rest.append(hit | S)
            blocks = rest
        else:
            kap = {q: ch.kappa for q, ch in op[1]}
            nonunital = sum(1 << q for q, ch in op[1] if not ch.unital)
            for b in blocks:
                if b & ~S == 0:
                    pr = Fr(1)
                    for q in range(b.bit_length()):
                        if (b >> q) & 1:
                            pr *= kap.get(q, 0)
                    sigma *= (1 - pr)
            if nonunital:
                blocks = [b for b in blocks if not (b & nonunital)]
    return sigma


def exact_propagate(circ, eps, track=True):
    n = circ.n
    zero = (0,) * n
    O = {tuple(2 if q == circ.target else 0 for q in range(n)): Fr(1)}
    events = []
    eps = Fr(eps)
    for (start, end, tr) in reversed(circ.groups):
        for i in range(end - 1, start - 1, -1):
            op = circ.flat[i]
            if op[0] == "cl":
                O = _apply_cl(O, op[1], op[2])
            else:
                for q, ch in op[1]:
                    O = _apply_ch(O, q, ch)
        if tr and eps > 0:
            drop = {s: c for s, c in O.items() if abs(c) < eps and s != zero}
            if drop:
                O = {s: c for s, c in O.items() if s not in drop}
                events.append(_exact_event(circ, start, drop, track))
    expect = sum((c for s, c in O.items() if all(l in (0, 2) for l in s)), Fr(0))
    return dict(expectation=expect, events=events, final_terms=len(O), operator=O)


def _exact_event(circ, pos, drop, track):
    groups = {}
    for s, c in drop.items():
        m = _supp_mask(s)
        g = groups.setdefault(m, [Fr(0), Fr(0)])
        g[0] += abs(c)
        g[1] += c * c
    gl = []
    union = 0
    for m, (l1, l2sq) in sorted(groups.items()):
        union |= m
        sg = exact_survival(circ.flat, pos, m) if (track and pos > 0) else Fr(1)
        gl.append(dict(mask=m, k=bin(m).count("1"), l1=l1, l2sq=l2sq, sigma=sg))
    return dict(pos=pos, k=bin(union).count("1"), l1=sum((g["l1"] for g in gl), Fr(0)),
                l2sq=sum((g["l2sq"] for g in gl), Fr(0)), groups=gl)


def bound_from_ledger(events, n, bits=64):
    """The exact-arithmetic bounds (Fractions) computed from ledger entries only."""
    acc = {k: Fr(0) for k in ("old", "cone", "cone_tl", "minl1l2", "grouped", "surv")}
    for ev in events:
        k, l2sq, l1 = ev["k"], ev["l2sq"], ev["l1"]
        acc["old"] += ceil_sqrt(2 ** n * l2sq, bits)
        acc["cone"] += ceil_sqrt(2 ** k * l2sq, bits)
        tl = ceil_sqrt((2 ** k - 1) * l2sq, bits)
        acc["cone_tl"] += tl
        emin = min(l1, tl)
        acc["minl1l2"] += emin
        grp = Fr(0)
        sv = Fr(0)
        for g in ev["groups"]:
            b = g["l1"] if g["k"] == 0 else min(g["l1"], ceil_sqrt((2 ** g["k"] - 1) * g["l2sq"], bits))
            grp += b
            sv += g["sigma"] * b
        acc["grouped"] += min(emin, grp)
        acc["surv"] += min(emin, sv)
    return acc


# ----------------------------------------------------------------------------
# independent exact Schroedinger reference (complex-rational density matrix)
# ----------------------------------------------------------------------------
def _cm(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def _gate_cols(name, sites, n):
    """U|i> = 2^{-s/2} sum_j c_ij |j>; returns (cols, s) with cols[i] = [(j, (re,im))]."""
    cols = []
    s = 0
    for i in range(2 ** n):
        if name == "h":
            q = sites[0]
            b = (i >> q) & 1
            i0 = i & ~(1 << q)
            cols.append([(i0, (Fr(1), Fr(0))), (i0 | (1 << q), (Fr(-1 if b else 1), Fr(0)))])
            s = 1
        elif name == "s":
            q = sites[0]
            b = (i >> q) & 1
            cols.append([(i, (Fr(0), Fr(1)) if b else (Fr(1), Fr(0)))])
        elif name == "x":
            cols.append([(i ^ (1 << sites[0]), (Fr(1), Fr(0)))])
        elif name == "z":
            cols.append([(i, (Fr(-1 if (i >> sites[0]) & 1 else 1), Fr(0)))])
        elif name == "cx":
            c, t = sites
            j = i ^ (1 << t) if (i >> c) & 1 else i
            cols.append([(j, (Fr(1), Fr(0)))])
        elif name == "cz":
            a, b = sites
            ph = -1 if ((i >> a) & 1 and (i >> b) & 1) else 1
            cols.append([(i, (Fr(ph), Fr(0)))])
        else:
            raise ValueError(name)
    return cols, s


def schro_expectation(circ, units_for):
    """Exact <Z_target> after the full forward circuit from |0..0>.  units_for(ch) gives the
    matrix-unit action of a channel object (independent physical formulas)."""
    n = circ.n
    N = 2 ** n
    rho = {(0, 0): (Fr(1), Fr(0))}
    for op in circ.flat:
        if op[0] == "cl":
            cols, s = _gate_cols(op[1], op[2], n)
            new = {}
            for (i, ip), v in rho.items():
                for (j, cj) in cols[i]:
                    for (jp, cjp) in cols[ip]:
                        cc = _cm(cj, (cjp[0], -cjp[1]))
                        w = _cm(cc, v)
                        k = (j, jp)
                        old = new.get(k, (Fr(0), Fr(0)))
                        new[k] = (old[0] + w[0], old[1] + w[1])
            sc = Fr(1, 2 ** s)
            rho = {k: (v[0] * sc, v[1] * sc) for k, v in new.items() if v != (0, 0)}
        else:
            for q, ch in op[1]:
                units = units_for(ch)
                new = {}
                for (i, ip), v in rho.items():
                    a, b = (i >> q) & 1, (ip >> q) & 1
                    bi, bip = i & ~(1 << q), ip & ~(1 << q)
                    for (a2, b2), coef in units[(a, b)].items():
                        k = (bi | (a2 << q), bip | (b2 << q))
                        old = new.get(k, (Fr(0), Fr(0)))
                        new[k] = (old[0] + coef * v[0], old[1] + coef * v[1])
                rho = {k: v for k, v in new.items() if v != (0, 0)}
    tot = Fr(0)
    for (i, ip), v in rho.items():
        if i == ip:
            tot += (-1 if (i >> circ.target) & 1 else 1) * v[0]
    return tot


# ----------------------------------------------------------------------------
# float twin of an exact circuit (for cross-checks with qc27_bounds)
# ----------------------------------------------------------------------------
def to_float_circuit(circ):
    import numpy as np
    import qc27_bounds as B
    ops = []
    CZ = np.diag([1, 1, 1, -1]).astype(complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Z = np.diag([1, -1]).astype(complex)
    for (kind, *rest) in circ.flat:
        if kind == "cl":
            name, sites = rest
            if name == "h":
                ops.append(B.unitary_op(sites, B._H_GATE, "h", True))
            elif name == "s":
                ops.append(B.unitary_op(sites, B._S_GATE, "s", True))
            elif name == "x":
                ops.append(B.unitary_op(sites, X, "x", True))
            elif name == "z":
                ops.append(B.unitary_op(sites, Z, "z", True))
            elif name == "cx":
                c, t = sites
                if c < t:
                    ops.append(B.unitary_op((c, t), B._CNOT_01, "cx", True))
                else:
                    ops.append(B.unitary_op((t, c), B._CNOT_10, "cx", True))
            elif name == "cz":
                ops.append(B.unitary_op(tuple(sorted(sites)), CZ, "cz", True))
        else:
            (chs,) = rest
            ps = [float(ch.kappa) for _, ch in chs]
            # recover gamma from the exact parameters: lam/(1-p) = m^2, gamma = 1 - m^2
            gs = [float(1 - ch.lam / (1 - ch.kappa)) for _, ch in chs]
            ops.append(B.noise_op([q for q, _ in chs], ps, gs))
    return B.Circuit(circ.n, ops, list(circ.groups), circ.target)


# ----------------------------------------------------------------------------
# witness (prototype) -- QC27-NU-WITNESS-v1
# ----------------------------------------------------------------------------
def _fs(x):
    x = Fr(x)
    return "%d/%d" % (x.numerator, x.denominator)


def _fr(s):
    a, b = s.split("/")
    return Fr(int(a), int(b))


def _is_dyadic(x):
    d = Fr(x).denominator
    return d & (d - 1) == 0


def emit_witness(circ, eps):
    res = exact_propagate(circ, eps)
    w = dict(format="qc27-nonunital-dyadic-witness-v1", n=circ.n, target=circ.target, eps=_fs(eps),
             steps=[], events=[], claim={})
    for ops, tr in circ.steps:
        step = dict(trunc=bool(tr), ops=[])
        for op in ops:
            if op[0] == "cl":
                step["ops"].append(dict(kind="cl", name=op[1], sites=list(op[2])))
            else:
                chs = []
                for q, ch in op[1]:
                    cert = ch.certificates()
                    chs.append(dict(q=q, mu=_fs(ch.mu), lam=_fs(ch.lam), tau=_fs(ch.tau), kappa=_fs(ch.kappa),
                                    cp_L=[[_fs(v) for v in r] for r in cert["cp"][0]],
                                    cp_D=[_fs(v) for v in cert["cp"][1]],
                                    kappa_L=[[_fs(v) for v in r] for r in cert["kappa"][0]],
                                    kappa_D=[_fs(v) for v in cert["kappa"][1]]))
                step["ops"].append(dict(kind="noise", chans=chs))
        w["steps"].append(step)
    for ev in res["events"]:
        w["events"].append(dict(pos=ev["pos"], groups=[dict(mask=g["mask"], l1=_fs(g["l1"]), l2sq=_fs(g["l2sq"]),
                                                              sigma=_fs(g["sigma"])) for g in ev["groups"]]))
    b = bound_from_ledger(res["events"], circ.n)
    w["claim"] = dict(expectation=_fs(res["expectation"]), bound_surv=_fs(b["surv"]),
                      bound_old=_fs(b["old"]), sqrt_bits=64)
    return w


def verify_witness(w):
    """Re-execute the witness in exact arithmetic.  Accepts iff: every parameter is dyadic, every
    channel's Choi matrix and (Choi - kappa/2 I) match their LDL^T certificates with D >= 0, and the
    recomputed ledger, expectation and bounds equal the claims.  Returns (ok, reasons)."""
    reasons = []
    if w.get("format") != "qc27-nonunital-dyadic-witness-v1":
        return False, ["bad format"]
    steps = []
    for st in w["steps"]:
        ops = []
        for op in st["ops"]:
            if op["kind"] == "cl":
                ops.append(("cl", op["name"], tuple(op["sites"])))
            else:
                chs = []
                for c in op["chans"]:
                    ch = Chan(_fr(c["mu"]), _fr(c["lam"]), _fr(c["tau"]), _fr(c["kappa"]))
                    for v in (ch.mu, ch.lam, ch.tau, ch.kappa):
                        if not _is_dyadic(v):
                            reasons.append("non-dyadic channel parameter %s" % v)
                    L = [[_fr(v) for v in r] for r in c["cp_L"]]
                    D = [_fr(v) for v in c["cp_D"]]
                    if not ldl_check(ch.choi(), L, D):
                        reasons.append("CP certificate fails on qubit %d" % c["q"])
                    Lk = [[_fr(v) for v in r] for r in c["kappa_L"]]
                    Dk = [_fr(v) for v in c["kappa_D"]]
                    Ck = [[ch.choi()[i][j] - (ch.kappa / 2 if i == j else 0) for j in range(4)] for i in range(4)]
                    if not ldl_check(Ck, Lk, Dk):
                        reasons.append("kappa certificate fails on qubit %d" % c["q"])
                    chs.append((c["q"], ch))
                ops.append(("noise", chs))
        steps.append((ops, st["trunc"]))
    circ = ExCircuit(w["n"], w["target"], steps)
    res = exact_propagate(circ, _fr(w["eps"]))
    if _fs(res["expectation"]) != w["claim"]["expectation"]:
        reasons.append("expectation mismatch")
    if len(res["events"]) != len(w["events"]):
        reasons.append("event count mismatch")
    else:
        for ev, wev in zip(res["events"], w["events"]):
            if ev["pos"] != wev["pos"] or len(ev["groups"]) != len(wev["groups"]):
                reasons.append("event layout mismatch")
                break
            for g, wg in zip(ev["groups"], wev["groups"]):
                if (g["mask"], _fs(g["l1"]), _fs(g["l2sq"]), _fs(g["sigma"])) != \
                        (wg["mask"], wg["l1"], wg["l2sq"], wg["sigma"]):
                    reasons.append("ledger group mismatch at pos %d" % ev["pos"])
    b = bound_from_ledger(res["events"], circ.n, w["claim"].get("sqrt_bits", 64))
    if _fs(b["surv"]) != w["claim"]["bound_surv"]:
        reasons.append("bound mismatch")
    return (not reasons), reasons


# ----------------------------------------------------------------------------
# reports
# ----------------------------------------------------------------------------
def untruncated_equality(circ):
    res = exact_propagate(circ, 0)
    sc = schro_expectation(circ, lambda ch: ad_dep_units_cache[id(ch)])
    return res["expectation"], sc


ad_dep_units_cache = {}


def _register_units(circ, params):
    """Attach the physical (m, p) of each AD+dep channel for the independent Schroedinger run."""
    for (kind, *rest) in circ.flat:
        if kind == "noise":
            for q, ch in rest[0]:
                p = ch.kappa
                m2 = ch.lam / (1 - p)
                m = ch.mu / (1 - p)
                assert m * m == m2
                ad_dep_units_cache[id(ch)] = ad_dep_schro_units(m, p)


def exact_case(n, depth, seed, eps):
    circ = random_excircuit(n, depth, seed)
    _register_units(circ, None)
    full = exact_propagate(circ, 0)
    sch = schro_expectation(circ, lambda ch: ad_dep_units_cache[id(ch)])
    tr = exact_propagate(circ, eps)
    err = abs(sch - tr["expectation"])
    b = bound_from_ledger(tr["events"], n)
    allch = [ch for (k, *r) in circ.flat if k == "noise" for _, ch in r[0]]
    maxden = max((abs(c).denominator.bit_length() for c in full["operator"].values()), default=1)
    return dict(n=n, depth=depth, seed=seed, eps=_fs(eps), heisenberg_equals_schroedinger=(full["expectation"] == sch),
                exact_value=_fs(sch), truncated_value=_fs(tr["expectation"]), exact_error=_fs(err),
                events=len(tr["events"]), channels_verified=all(c.verified() for c in allch),
                bounds={k: _fs(v) for k, v in b.items()}, bounds_float={k: float(v) for k, v in b.items()},
                error_le_all_bounds=all(err <= v for v in b.values()),
                max_denominator_bits=maxden, final_terms_untruncated=full["final_terms"]), circ


EXACT_CASES = [(3, 3, 0, Fr(1, 32)), (3, 3, 6, Fr(1, 128)), (3, 3, 21, Fr(1, 32)), (4, 3, 0, Fr(1, 128)),
               (4, 3, 6, Fr(1, 32)), (4, 4, 0, Fr(1, 32)), (4, 4, 0, Fr(1, 128))]


def exact_report():
    cases = []
    for (n, d, s, eps) in EXACT_CASES:
        c, circ = exact_case(n, d, s, eps)
        # float twin
        import qc27_bounds as B
        fc = to_float_circuit(circ)
        fr = B.propagate(fc, float(eps))
        fb = B.total_bounds(fr["events"], n, fr["rounding"])
        c["float_expectation"] = fr["expectation"]
        c["float_vs_exact_expectation_dev"] = abs(fr["expectation"] - float(_fr(c["truncated_value"])))
        c["float_bounds"] = fb
        c["float_bound_surv_over_exact"] = fb["surv"] / c["bounds_float"]["surv"] if c["bounds_float"]["surv"] else None
        w = emit_witness(circ, eps)
        ok, why = verify_witness(w)
        c["witness_verified"] = ok
        c["witness_reasons"] = why
        # tamper test: change claimed expectation -> must reject
        w2 = json.loads(json.dumps(w))
        w2["claim"]["expectation"] = "1/1"
        c["tampered_witness_rejected"] = (not verify_witness(w2)[0]) if w["claim"]["expectation"] != "1/1" else None
        cases.append(c)
    return dict(cases=cases)


def failed_hypotheses():
    out = {}
    h = Fr(1, 2)
    # H1: l2 norm of the discarded remainder is non-increasing under later steps
    reset = Chan(0, 0, 1, 0, "reset-to-|0>")
    D = {(0, 2): Fr(1), (2, 2): Fr(1)}                 # Z_1 + Z_0 Z_1 (qubit0 first)
    before = sum(c * c for c in D.values())
    D2 = _apply_ch(D, 0, reset)
    out["H1_l2_nonincreasing"] = dict(
        claim="eta_after <= eta_before under a later CPTP step",
        reset_channel_is_CP=reset.certificates()["cp"] is not None,
        eta_sq_before=_fs(before), eta_sq_after=_fs(sum(c * c for c in D2.values())), holds=False)
    # H2: weight-based survival lambda^(wt at discard)
    lam = Fr(3, 4)
    dep = Chan(lam, lam, 0, 1 - lam, "dep(p=1/4)")
    circ = ExCircuit(2, 0, [([("noise", [(0, dep), (1, dep)])], False), ([("cl", "cx", (0, 1))], False)])
    O = {(2, 2): Fr(1)}
    O = _apply_cl(O, "cx", (0, 1))
    O = _apply_ch(_apply_ch(O, 0, dep), 1, dep)
    actual = sum((c for s, c in O.items() if all(l in (0, 2) for l in s)), Fr(0))
    out["H2_weight_survival"] = dict(
        claim="influence of a weight-w discarded string is damped by lambda^w",
        lam=_fs(lam), discarded="Z0 Z1 (w=2) before [dep, CNOT]", actual_influence=_fs(actual),
        claimed_bound=_fs(lam ** 2), violated=actual > lam ** 2,
        tracker_sigma=_fs(exact_survival(circ.flat, len(circ.flat), 0b11)),
        tracker_valid=actual <= exact_survival(circ.flat, len(circ.flat), 0b11))
    # H3: amplitude damping contracts the Z-content by lambda (or mu)
    ad = Chan.ad_dep(h, 0)
    O = _apply_ch({(2,): Fr(1)}, 0, ad)
    actual = sum((c for s, c in O.items() if all(l in (0, 2) for l in s)), Fr(0))
    out["H3_ad_contracts_Z"] = dict(
        claim="op-norm survival of Z-content through amplitude damping is lambda",
        m=_fs(h), lam=_fs(ad.lam), tau=_fs(ad.tau), actual_influence_on_ground_state=_fs(actual),
        claimed_bound=_fs(ad.lam), violated=actual > ad.lam,
        tracker_kappa=_fs(ad.kappa))
    # H4: tracelessness block survives a non-unital channel
    dep4 = Chan(lam, lam, 0, 1 - lam, "dep")
    O = _apply_ch({(2,): Fr(1)}, 0, ad)       # amplitude damping meets the operator first
    O = _apply_ch(O, 0, dep4)
    actual = sum((c for s, c in O.items() if all(l in (0, 2) for l in s)), Fr(0))
    c4 = ExCircuit(1, 0, [([("noise", [(0, dep4)]), ("noise", [(0, ad)])], False)])
    sg = exact_survival(c4.flat, 2, 1)
    out["H4_block_survives_nonunital"] = dict(
        claim="Tr_q A = 0 persists through amplitude damping, so a later depolarising step still gives (1-p)",
        actual_influence=_fs(actual), claimed_bound=_fs(lam), violated=actual > lam,
        tracker_sigma=_fs(sg), tracker_valid=actual <= sg)
    # H5: product of per-qubit lambda over the union support (instead of hitting-set blocks)
    D5 = {(1, 0): Fr(1), (0, 1): Fr(1)}        # X0 + X1
    import itertools
    def opnorm_x(Dd):  # commuting X-type strings: max over sign assignments
        best = Fr(0)
        for signs in itertools.product((1, -1), repeat=2):
            best = max(best, abs(sum(c * (signs[0] if s[0] else 1) * (signs[1] if s[1] else 1) for s, c in Dd.items())))
        return best
    D5b = _apply_ch(_apply_ch(D5, 0, dep4), 1, dep4)
    out["H5_union_support_product"] = dict(
        claim="||F(D)|| <= lam_0 lam_1 ||D|| for D whose strings jointly cover qubits {0,1}",
        opnorm_before=_fs(opnorm_x(D5)), opnorm_after=_fs(opnorm_x(D5b)),
        ratio=_fs(opnorm_x(D5b) / opnorm_x(D5)), claimed_ratio=_fs(lam * lam),
        violated=opnorm_x(D5b) / opnorm_x(D5) > lam * lam)
    # H6: |error| <= eta (no 2^{k/2}) -- needs the maximally mixed state; tightness of sqrt(2^k - 1)
    Dz = {(2, 0): Fr(1, 4), (0, 2): Fr(1, 4), (2, 2): Fr(1, 4)}
    eta2 = sum(c * c for c in Dz.values())
    val = sum(Dz.values())                  # <00| D |00>
    out["H6_drop_cone_factor"] = dict(
        claim="|<D>| <= ||D||_{2,norm} for product inputs",
        D="(ZI+IZ+ZZ)/4 on |00>", eta_sq=_fs(eta2), value=_fs(val), ratio_sq=_fs(val * val / eta2),
        violated=val * val > eta2, equals_2k_minus_1=(val * val / eta2 == 2 ** 2 - 1))
    return out


if __name__ == "__main__":
    print(json.dumps(failed_hypotheses(), indent=1, default=str))
