"""QC-27: tighter truncation bounds for sparse-Pauli Heisenberg propagation.

Self-contained (numpy only).  See QC27_BOUND_NOTE.md for the statements and proofs.

Pauli strings are pairs of uint64 bitmasks (x, z); qubit q is bit q.  Local
Pauli codes are l = x + 2z:  0=I, 1=X, 2=Z, 3=Y.  A two-qubit local index is
4*l(s0) + l(s1) with sites (s0, s1) = (left, right).

Heisenberg convention: an operator O = sum_P c_P P is mapped by the adjoint
E^dagger of a CPTP step; c' = H c with H[a, b] = Tr(P_a E^dagger(P_b)) / d.
Forward ops are listed in circuit order; propagation runs backwards.  After a
group of backward ops (forward indices start..end-1) small coefficients are
discarded; this is a discard event at forward position `pos = start`
(true error contribution Tr(rho_pos D), rho_pos = state after ops[0:pos]).
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

U53 = 2.0 ** -53
INFL = 1.0 + 2.0 ** -38      # outward rounding on ledger-derived quantities
ROUND_FACTOR = 10.0          # per-op rounding allowance = L1(A_in)*(ROUND_FACTOR*u*colL1(H) + 4^m*H_ERR)
H_ERR = 2.0 ** -50           # assumed entrywise distance of stored H from the exact adjoint PTM

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PM = [_I, _X, _Z, _Y]        # indexed by local code l = x + 2 z


# ----------------------------------------------------------------------------
# local maps
# ----------------------------------------------------------------------------
def pauli_basis(m):
    out = []
    for a in range(4 ** m):
        mat = np.array([[1.0 + 0j]])
        for k in range(m):
            l = (a // 4 ** (m - 1 - k)) % 4
            mat = np.kron(mat, PM[l])
        out.append(mat)
    return out


_BASIS = {1: pauli_basis(1), 2: pauli_basis(2)}


def heis_matrix(kraus):
    """H[a,b] = Tr(P_a sum_k K^dag P_b K)/d for Kraus list on m qubits."""
    d = kraus[0].shape[0]
    m = 1 if d == 2 else 2
    B = _BASIS[m]
    H = np.zeros((4 ** m, 4 ** m))
    for b, Pb in enumerate(B):
        img = sum(K.conj().T @ Pb @ K for K in kraus)
        for a, Pa in enumerate(B):
            H[a, b] = (np.trace(Pa @ img) / d).real
    return H


def _nz(H):
    out = []
    for a in range(H.shape[1]):
        bs = np.flatnonzero(np.abs(H[:, a]) > 1e-15)
        out.append((bs, H[bs, a].copy()))
    return out


class Part:
    """One elementary local channel (sites, Kraus) with its Heisenberg matrix."""
    __slots__ = ("sites", "kraus", "H", "nz", "perm", "exact", "colL1")

    def __init__(self, sites, kraus, exact_perm=False):
        self.sites = tuple(sites)
        self.kraus = kraus
        H = heis_matrix(kraus)
        if exact_perm:   # Clifford: signed permutation, entries exactly 0, +-1
            R = np.rint(H)
            if np.max(np.abs(H - R)) > 1e-12:
                raise ValueError("not a Clifford signed permutation")
            H = R
        H[np.abs(H) < 1e-15] = 0.0
        self.H = H
        self.nz = _nz(H)
        cols = np.count_nonzero(H, axis=0)
        rows = np.count_nonzero(H, axis=1)
        self.perm = bool(np.all(cols == 1) and np.all(rows == 1))
        self.exact = bool(exact_perm and self.perm)
        self.colL1 = float(np.max(np.sum(np.abs(H), axis=0)))


class Op:
    """A forward circuit step: 'unitary' (one Part) or 'noise' (Parts on its qubits)."""
    __slots__ = ("kind", "mask", "parts", "kappa", "nonunital_mask", "name")

    def __init__(self, kind, parts, name="", kappa=None, nonunital_mask=0):
        self.kind = kind
        self.parts = parts
        self.name = name
        m = 0
        for p in parts:
            for s in p.sites:
                m |= 1 << s
        self.mask = m
        self.kappa = kappa or {}
        self.nonunital_mask = nonunital_mask


_TEMPLATES = {}


def _part_from(key, sites, kraus, exact_perm=False):
    """Cache Heisenberg matrices for repeated gates/channels; share read-only data."""
    if key is not None and key in _TEMPLATES:
        t = _TEMPLATES[key]
        p = Part.__new__(Part)
        for a in Part.__slots__:
            setattr(p, a, getattr(t, a))
        p.sites = tuple(sites)
        return p
    p = Part(sites, kraus, exact_perm)
    if key is not None:
        _TEMPLATES[key] = p
    return p


def unitary_op(sites, U, name="", clifford=False, key=None):
    return Op("unitary", [_part_from(key, sites, [U], clifford)], name)


_H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
_S_GATE = np.diag([1, 1j]).astype(complex)
_CNOT_01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_CNOT_10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)


def rz(theta):
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def rx(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def amp_damp_kraus(g):
    return [np.array([[1, 0], [0, math.sqrt(1 - g)]], dtype=complex),
            np.array([[0, math.sqrt(g)], [0, 0]], dtype=complex)]


def depol_kraus(p):
    return [math.sqrt(1 - 0.75 * p) * _I, math.sqrt(p / 4) * _X,
            math.sqrt(p / 4) * _Y, math.sqrt(p / 4) * _Z]


def noise_kraus(p, g):
    """Forward order: amplitude damping first, then depolarising (adjoint meets
    depolarising first, which gives kappa = p: E^dag = (1-p) AD^dag + p tau)."""
    ad = amp_damp_kraus(g) if g > 0 else [_I]
    return [D @ K for D in depol_kraus(p) for K in ad]


def noise_op(qubits, p, g):
    """p, g: scalars (same channel on every qubit) or per-qubit sequences."""
    ps = list(p) if hasattr(p, "__len__") else [p] * len(qubits)
    gs = list(g) if hasattr(g, "__len__") else [g] * len(qubits)
    parts = []
    kappa = {}
    nu = 0
    for q, pq, gq in zip(qubits, ps, gs):
        key = ("noise", pq, gq)
        kr = noise_kraus(pq, gq) if key not in _TEMPLATES else None
        parts.append(_part_from(key, (q,), kr))
        kappa[q] = pq
        if gq > 0:
            nu |= 1 << q
    return Op("noise", parts, "noise", kappa=kappa, nonunital_mask=nu)


# ----------------------------------------------------------------------------
# circuits
# ----------------------------------------------------------------------------
class Circuit:
    def __init__(self, n, ops, groups, target, meta=None):
        self.n, self.ops, self.groups, self.target, self.meta = n, ops, groups, target, meta or {}


def build_circuit(n, depth, seed, theta_max=0.3, p1=0.002, g1=0.001, p2=0.01, g2=0.003,
                  target=None):
    """Brickwork 1D: per layer random Clifford 1q (I/H/S/HS), small RZ, RX,
    1q noise, CNOT brickwork, 2q (per-qubit) noise."""
    rng = np.random.default_rng(seed)
    target = n // 2 if target is None else target
    ops, groups = [], []
    cl_gates = {1: ("h", _H_GATE), 2: ("s", _S_GATE), 3: ("hs", _S_GATE @ _H_GATE)}
    cnot_cache = {}
    for layer in range(depth):
        s = len(ops)
        for q in range(n):
            c = int(rng.integers(0, 4))
            if c:
                nm, U = cl_gates[c]
                ops.append(unitary_op((q,), U, nm, clifford=True, key=("cl", nm)))
        groups.append((s, len(ops), False))
        s = len(ops)
        for q in range(n):
            th = rng.uniform(-theta_max, theta_max, size=2)
            ops.append(unitary_op((q,), rz(th[0]), "rz"))
            ops.append(unitary_op((q,), rx(th[1]), "rx"))
        groups.append((s, len(ops), True))
        s = len(ops)
        for q in range(n):
            ops.append(noise_op((q,), p1, g1))
        groups.append((s, len(ops), True))
        s = len(ops)
        pairs = [(a, a + 1) for a in range(layer % 2, n - 1, 2)]
        for (a, b) in pairs:
            flip = int(rng.integers(0, 2))
            ops.append(unitary_op((a, b), _CNOT_10 if flip else _CNOT_01, "cx", clifford=True,
                                  key=("cx", flip)))
        groups.append((s, len(ops), False))
        s = len(ops)
        for (a, b) in pairs:
            ops.append(noise_op((a, b), p2, g2))
        groups.append((s, len(ops), True))
    return Circuit(n, ops, groups, target, dict(depth=depth, seed=seed, theta_max=theta_max,
                                               p1=p1, g1=g1, p2=p2, g2=g2))


# ----------------------------------------------------------------------------
# survival tracker (Theorem 3)
# ----------------------------------------------------------------------------
def survival(ops, pos, support_mask, blocks0=None, _neg=None):
    """Upper bound sigma such that ||F(D)||_op <= sigma ||D||_op for every Hermitian D
    all of whose Pauli strings have support exactly support_mask, where F is the
    Heisenberg composite of ops[pos-1], ..., ops[0].  Returns sigma (float, <=1)."""
    if blocks0 is None:
        blocks = [1 << q for q in range(support_mask.bit_length()) if (support_mask >> q) & 1]
    else:
        blocks = list(blocks0)
    sigma = 1.0
    for i in range(pos - 1, -1, -1):
        if not blocks:
            break
        op = ops[i]
        S = op.mask
        if not any(b & S for b in blocks):
            continue
        if op.kind == "unitary":
            hit = 0
            rest = []
            for b in blocks:
                if b & S:
                    hit |= b
                else:
                    rest.append(b)
            rest.append(hit | S)
            if _neg == "no_merge":      # negative control: pretend the gate keeps the blocks
                rest = [b for b in blocks]
            blocks = rest
        else:  # product of single-qubit channels
            for b in blocks:
                if b & ~S == 0:
                    pr = 1.0
                    for q in range(b.bit_length()):
                        if (b >> q) & 1:
                            pr *= op.kappa.get(q, 0.0)
                    sigma *= (1.0 - pr)
            if op.nonunital_mask and _neg != "keep_nonunital":
                blocks = [b for b in blocks if not (b & op.nonunital_mask)]
    return min(1.0, sigma * INFL)


# ----------------------------------------------------------------------------
# sparse propagator
# ----------------------------------------------------------------------------
def _local_code(x, z, s):
    return ((x >> np.uint64(s)) & np.uint64(1)).astype(np.int64) | \
           (((z >> np.uint64(s)) & np.uint64(1)).astype(np.int64) << 1)


def _set_code(x, z, s, l):
    one = np.uint64(1)
    sh = np.uint64(s)
    keep = ~(one << sh)
    xb = np.uint64(l & 1)
    zb = np.uint64((l >> 1) & 1)
    return (x & keep) | (xb << sh), (z & keep) | (zb << sh)


def _merge(x, z, c):
    if len(c) == 0:
        return x, z, c
    order = np.lexsort((z, x))
    xs, zs, cs = x[order], z[order], c[order]
    new = np.empty(len(xs), dtype=bool)
    new[0] = True
    new[1:] = (xs[1:] != xs[:-1]) | (zs[1:] != zs[:-1])
    idx = np.flatnonzero(new)
    cs = np.add.reduceat(cs, idx)
    xs, zs = xs[idx], zs[idx]
    nzmask = cs != 0.0
    return xs[nzmask], zs[nzmask], cs[nzmask]


def apply_part(x, z, c, part):
    sites = part.sites
    if len(sites) == 1:
        a = _local_code(x, z, sites[0])
    else:
        a = 4 * _local_code(x, z, sites[0]) + _local_code(x, z, sites[1])
    outx, outz, outc = [], [], []
    for av in np.unique(a):
        sel = a == av
        bs, vals = part.nz[int(av)]
        xs, zs, cs = x[sel], z[sel], c[sel]
        for b, v in zip(bs, vals):
            if len(sites) == 1:
                nx, nz_ = _set_code(xs, zs, sites[0], int(b))
            else:
                nx, nz_ = _set_code(xs, zs, sites[0], int(b) // 4)
                nx, nz_ = _set_code(nx, nz_, sites[1], int(b) % 4)
            outx.append(nx)
            outz.append(nz_)
            outc.append(cs * v)
    x2, z2, c2 = np.concatenate(outx), np.concatenate(outz), np.concatenate(outc)
    if part.perm:
        return x2, z2, c2
    return _merge(x2, z2, c2)


def _popcount(m):
    return int(m).bit_count()


class Event:
    __slots__ = ("pos", "k", "eta", "l1", "groups", "ndrop", "idcoef", "disc")

    def __init__(self):
        self.groups = []


def propagate(circ, eps, track_survival=True, keep_discards=False, max_surv_groups=24,
              surv_mass=0.999, obs=None):
    """Truncated backward Heisenberg propagation of Z_target (or obs=(x,z,coef)).
    Returns dict(expectation, events, rounding, nterms_max, ...)."""
    n, ops = circ.n, circ.ops
    if obs is None:
        x = np.zeros(1, dtype=np.uint64)
        z = np.array([1 << circ.target], dtype=np.uint64)
        c = np.ones(1)
    else:
        x, z, c = (np.array([obs[0]], dtype=np.uint64), np.array([obs[1]], dtype=np.uint64),
                   np.array([obs[2]], dtype=float))
    events = []
    rounding = 0.0
    supp = int(x[0] | z[0])
    nmax = 1
    for (start, end, trunc) in reversed(circ.groups):
        for i in range(end - 1, start - 1, -1):
            op = ops[i]
            if not (op.mask & supp):
                continue
            for part in reversed(op.parts) if op.kind == "unitary" else op.parts:
                if not part.exact:
                    rounding += float(np.abs(c).sum()) * (
                        ROUND_FACTOR * U53 * part.colL1 + 4 ** len(part.sites) * H_ERR)
                x, z, c = apply_part(x, z, c, part)
            supp |= op.mask
        nmax = max(nmax, len(c))
        if trunc and eps > 0:
            keep = (np.abs(c) >= eps) | ((x == 0) & (z == 0))
            if not keep.all():
                dx, dz, dc = x[~keep], z[~keep], c[~keep]
                x, z, c = x[keep], z[keep], c[keep]
                ev = _make_event(start, dx, dz, dc, ops, track_survival, max_surv_groups,
                                 surv_mass, keep_discards)
                events.append(ev)
        if len(c):
            supp = int(np.bitwise_or.reduce(x | z))
        else:
            supp = 0
    expect = math.fsum((c[x == 0]).tolist()) if len(c) else 0.0
    rounding += U53 * float(np.abs(c).sum())
    return dict(expectation=expect, events=events, rounding=rounding, nterms_max=nmax,
                nterms_final=len(c))


def _make_event(pos, dx, dz, dc, ops, track, max_groups, mass, keep):
    ev = Event()
    ev.pos = pos
    ev.ndrop = len(dc)
    sq = dc * dc
    ev.eta = math.sqrt(math.fsum(sq.tolist())) * INFL
    ev.l1 = math.fsum(np.abs(dc).tolist()) * INFL
    supp = dx | dz
    ev.idcoef = 0.0
    uniq, inv = np.unique(supp, return_inverse=True)
    l1g = np.bincount(inv, weights=np.abs(dc))
    l2g = np.bincount(inv, weights=sq)
    union = 0
    glist = []
    for m, a, b in zip(uniq.tolist(), l1g.tolist(), l2g.tolist()):
        union |= m
        glist.append([m, _popcount(m), a * INFL, b * INFL, 1.0])
    ev.k = _popcount(union)
    # survival for the biggest groups
    if track and pos > 0:
        def base(g):
            k = g[1]
            return min(g[2], math.sqrt(max(2.0 ** k - 1.0, 0.0) * g[3])) if k else g[2]
        order = sorted(range(len(glist)), key=lambda i: -base(glist[i]))
        tot = sum(base(g) for g in glist) or 1.0
        acc = 0.0
        for cnt, i in enumerate(order):
            if cnt >= max_groups or acc >= mass * tot:
                break
            g = glist[i]
            acc += base(g)
            g[4] = survival(ops, pos, g[0])
    ev.groups = [tuple(g) for g in glist]
    ev.disc = (dx.copy(), dz.copy(), dc.copy()) if keep else None
    return ev


# ----------------------------------------------------------------------------
# bounds
# ----------------------------------------------------------------------------
def kappa_choi(kraus):
    """Depolarising content of a one-qubit channel: kappa = 2 * lambda_min(Choi) (Choi trace 2)."""
    C = np.zeros((4, 4), dtype=complex)
    for K in kraus:
        v = K.T.reshape(-1)
        C += np.outer(v, v.conj())
    return float(2 * np.linalg.eigvalsh(C)[0])


def _cone_tl(k):
    """sup ||D||_op / ||D||_{2,norm} over traceless Hermitian D on k qubits: sqrt(2^k - 1)."""
    return math.sqrt(2.0 ** k - 1.0)


def event_bounds(ev, n):
    """Per-event contributions to the six bounds (dict of floats)."""
    old = 2.0 ** (n / 2) * ev.eta
    cone = 2.0 ** (ev.k / 2) * ev.eta
    cone_tl = _cone_tl(ev.k) * ev.eta
    ev_min = min(ev.l1, cone_tl)
    grp = 0.0
    surv = 0.0
    for (m, k, l1, l2sq, sg) in ev.groups:
        b = l1 if k == 0 else min(l1, _cone_tl(k) * math.sqrt(l2sq))
        grp += b
        surv += sg * b
    grp = min(ev_min, grp * INFL)
    surv = min(ev_min, surv * INFL)
    return dict(old=old, cone=cone, cone_tl=cone_tl, minl1l2=ev_min, grouped=grp, surv=surv)


def total_bounds(events, n, rounding=0.0):
    keys = ("old", "cone", "cone_tl", "minl1l2", "grouped", "surv")
    acc = {k: [] for k in keys}
    for ev in events:
        eb = event_bounds(ev, n)
        for k in keys:
            acc[k].append(eb[k])
    out = {k: math.fsum(v) * INFL + rounding for k, v in acc.items()}
    out["old"] = math.fsum(acc["old"]) * INFL + rounding
    return out


def ledger_stats(events):
    if not events:
        return dict(n_events=0)
    ks = [e.k for e in events]
    return dict(n_events=len(events), k_median=float(np.median(ks)), k_max=max(ks),
                n_dropped=sum(e.ndrop for e in events),
                eta_sum=math.fsum(e.eta for e in events),
                l1_sum=math.fsum(e.l1 for e in events),
                n_groups=sum(len(e.groups) for e in events),
                sigma_min=min((g[4] for e in events for g in e.groups), default=1.0))


# ----------------------------------------------------------------------------
# dense references
# ----------------------------------------------------------------------------
def cone_restrict(circ):
    """Exact backward-cone restriction.  Returns (sites, ops_sub, new_target)."""
    S = 1 << circ.target
    keep = []
    for i in range(len(circ.ops) - 1, -1, -1):
        op = circ.ops[i]
        if op.mask & S:
            keep.append(i)
            S |= op.mask
    keep.reverse()
    sites = [q for q in range(circ.n) if (S >> q) & 1]
    return sites, keep


def _relabel_parts(op, remap):
    return [(tuple(remap[s] for s in p.sites), p) for p in op.parts]


def dense_heisenberg(circ, restrict=True):
    """Exact (untruncated) <0|O(0)|0> by dense Heisenberg Pauli-vector evolution on the
    backward-cone sites (restrict=True) or on all n qubits (restrict=False)."""
    if restrict:
        sites, keep = cone_restrict(circ)
    else:
        sites, keep = list(range(circ.n)), list(range(len(circ.ops)))
    m = len(sites)
    remap = {s: i for i, s in enumerate(sites)}
    t = np.zeros((4,) * m)
    idx = [0] * m
    idx[remap[circ.target]] = 2
    t[tuple(idx)] = 1.0
    for i in reversed(keep):
        op = circ.ops[i]
        plist = _relabel_parts(op, remap)
        for (ss, p) in (reversed(plist) if op.kind == "unitary" else plist):
            k = len(ss)
            Hm = p.H.reshape((4,) * (2 * k))
            t = np.tensordot(Hm, t, axes=(list(range(k, 2 * k)), list(ss)))
            t = np.moveaxis(t, list(range(k)), list(ss))
    sl = tuple(slice(0, 3, 2) for _ in range(m))   # codes 0 (I) and 2 (Z)
    return float(t[sl].sum()), m


def _apply_kraus(rho, n, sites, kraus):
    k = len(sites)
    out = np.zeros_like(rho)
    for K in kraus:
        Kt = K.reshape((2,) * (2 * k))
        r = np.tensordot(Kt, rho, axes=(list(range(k, 2 * k)), list(sites)))
        r = np.moveaxis(r, list(range(k)), list(sites))
        r = np.tensordot(Kt.conj(), r, axes=(list(range(k, 2 * k)), [n + s for s in sites]))
        r = np.moveaxis(r, list(range(k)), [n + s for s in sites])
        out += r
    return out


def dense_schrodinger(circ, snapshots=(), restrict=False):
    """Density-matrix forward evolution from |0..0> with the true Kraus operators.
    Returns (expectation of Z_target, {pos: rho matrix (2^n x 2^n)})."""
    if restrict:
        sites, keep = cone_restrict(circ)
    else:
        sites, keep = list(range(circ.n)), list(range(len(circ.ops)))
    n = len(sites)
    remap = {s: i for i, s in enumerate(sites)}
    rho = np.zeros((2,) * (2 * n), dtype=complex)
    rho[(0,) * (2 * n)] = 1.0
    snaps = {}
    snapset = set(snapshots)
    keepset = set(keep)
    for i in range(len(circ.ops) + 1):
        if i in snapset:
            snaps[i] = rho.reshape(2 ** n, 2 ** n).copy()
        if i == len(circ.ops):
            break
        if i not in keepset:
            continue
        op = circ.ops[i]
        for (ss, p) in _relabel_parts(op, remap):
            rho = _apply_kraus(rho, n, ss, p.kraus)
    mat = rho.reshape(2 ** n, 2 ** n)
    diag = np.real(np.diag(mat)).reshape((2,) * n)
    tq = remap[circ.target]
    marg = np.moveaxis(diag, tq, 0).reshape(2, -1).sum(axis=1)
    return float(marg[0] - marg[1]), snaps


def pauli_expect_dense(rho, n, x, z):
    """Tr(rho P(x,z)); qubit q <-> bit (n-1-q) of the dense index (qubit 0 most significant)."""
    def rev(m):
        r = 0
        for q in range(n):
            if (m >> q) & 1:
                r |= 1 << (n - 1 - q)
        return r
    xr, zr = rev(int(x)), rev(int(z))
    idx = np.arange(2 ** n)
    par = np.zeros(2 ** n, dtype=np.int64)
    v = idx & zr
    while v.any():
        par ^= (v & 1)
        v = v >> 1
    ph = (-1.0) ** par * (1j) ** (_popcount(xr & zr) % 4)
    return complex(np.sum(rho[idx, idx ^ xr] * ph))


def per_event_errors(circ, res):
    """Exact per-event (and per-group) error contributions Tr(rho_pos D) at small n."""
    events = res["events"]
    _, snaps = dense_schrodinger(circ, snapshots=sorted({e.pos for e in events}))
    n = circ.n
    out = []
    for ev in events:
        dx, dz, dc = ev.disc
        rho = snaps[ev.pos]
        tot = 0.0
        per_group = {}
        for xv, zv, cv in zip(dx.tolist(), dz.tolist(), dc.tolist()):
            val = (cv * pauli_expect_dense(rho, n, xv, zv)).real
            tot += val
            per_group[xv | zv] = per_group.get(xv | zv, 0.0) + val
        out.append((tot, per_group))
    return out


# ----------------------------------------------------------------------------
# direct stress test of Theorem 3 (survival factor) on random small circuits
# ----------------------------------------------------------------------------
def _embed(K, sites, n):
    k = len(sites)
    s0 = sites[0]
    return np.kron(np.kron(np.eye(2 ** s0), K), np.eye(2 ** (n - s0 - k)))


def _haar(d, rng):
    z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def survival_stress(trials, n=5, nops=10, seed=0, neg=None):
    """For random circuits (Haar 1q/2q unitaries, noise ops with random depolarising/amplitude
    damping strengths) and random Hermitian D whose strings all have exactly the support S,
    check ||F(D)||_op <= sigma(S) ||D||_op with the dense Heisenberg composite F.
    Returns dict(trials, violations, max_ratio_over_sigma, min_sigma, mean_sigma)."""
    rng = np.random.default_rng(seed)
    viol = 0
    worst = 0.0
    sig_all = []
    for _ in range(trials):
        ops = []
        for _i in range(nops):
            r = rng.random()
            if r < 0.35:
                q = int(rng.integers(0, n))
                ops.append(unitary_op((q,), _haar(2, rng)))
            elif r < 0.7:
                a = int(rng.integers(0, n - 1))
                ops.append(unitary_op((a, a + 1), _haar(4, rng)))
            else:
                a = int(rng.integers(0, n - 1))
                qs = (a, a + 1) if rng.random() < 0.6 else (a,)
                ops.append(noise_op(qs, list(rng.uniform(0.0, 0.9, len(qs))),
                                    list(rng.uniform(0.0, 0.9, len(qs)) * (rng.random(len(qs)) < 0.5))))
        k = int(rng.integers(1, n + 1))
        sup = sorted(rng.choice(n, size=k, replace=False).tolist())
        mask = sum(1 << q for q in sup)
        D = np.zeros((2 ** n, 2 ** n), dtype=complex)
        for _s in range(int(rng.integers(1, 6))):
            letters = [int(rng.integers(1, 4)) if q in sup else 0 for q in range(n)]
            mat = np.array([[1.0 + 0j]])
            for l in letters:
                mat = np.kron(mat, PM[l])
            D += rng.normal() * mat
        A = D
        for op in reversed(ops):
            for p in (op.parts[::-1] if op.kind == "unitary" else op.parts):
                A = sum(_embed(K, p.sites, n).conj().T @ A @ _embed(K, p.sites, n) for K in p.kraus)
        sg = survival(ops, len(ops), mask, _neg=neg)
        nd = np.max(np.abs(np.linalg.eigvalsh(D)))
        na = np.max(np.abs(np.linalg.eigvalsh((A + A.conj().T) / 2)))
        sig_all.append(sg)
        if na > sg * nd * (1 + 1e-9) + 1e-12:
            viol += 1
        worst = max(worst, na / (sg * nd) if sg * nd > 0 else 0.0)
    return dict(trials=trials, n=n, nops=nops, violations=viol, max_ratio_over_sigma=worst,
                min_sigma=float(min(sig_all)), mean_sigma=float(np.mean(sig_all)),
                frac_sigma_below_1=float(np.mean(np.array(sig_all) < 1 - 1e-9)))


# ----------------------------------------------------------------------------
# single case runner
# ----------------------------------------------------------------------------
def run_case(n, depth, seed, eps, ref="auto", max_cone_ref=10, noise=None, theta_max=0.3,
             track_survival=True):
    noise = noise or {}
    t0 = time.time()
    circ = build_circuit(n, depth, seed, theta_max=theta_max, **noise)
    t1 = time.time()
    res = propagate(circ, eps, track_survival=track_survival)
    t2 = time.time()
    b = total_bounds(res["events"], n, res["rounding"])
    sites, _ = cone_restrict(circ)
    row = dict(n=n, depth=depth, seed=seed, eps=eps, cone_width=len(sites),
               expectation=res["expectation"], rounding=res["rounding"],
               nterms_max=res["nterms_max"], t_build=t1 - t0, t_prop=t2 - t1,
               **{"bound_" + k: v for k, v in b.items()}, **ledger_stats(res["events"]))
    row["actual"] = None
    row["reference"] = None
    if ref != "none" and n <= 16 and len(sites) <= max_cone_ref:
        t3 = time.time()
        ex, m = dense_heisenberg(circ, restrict=True)
        row["reference"] = "dense-heisenberg-pauli-vector on %d backward-cone sites" % m
        row["exact"] = ex
        row["actual"] = abs(ex - res["expectation"])
        row["t_ref"] = time.time() - t3
    return row


if __name__ == "__main__":
    import qc27_benchmark
    qc27_benchmark.main(sys.argv[1:])
