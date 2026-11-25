"""QC-26A offline statistical identification of the native SX and conditional-CZ phase experiment.

  python qc26a_identify.py counts.json out.json [--bootstraps N] [--seed S]

No IBM access, no hardware. Input: {circuit_id: {bitstring: count}} (Qiskit little-endian: character
n-1-j is site path[j]). Output: one JSON identification report.

MODEL (all angles in rad; "per application" means per CZ application or per SX repetition)
  Readout, per site: P(read 1 | 0) = e0, P(read 0 | 1) = e1, so E[1-2b] = (e1-e0) + (1-e0-e1) z.
    e0, e1 are EFFECTIVE parameters: spam_1 is prepared with an X gate whose error contaminates e1 (and
    the same X prepares the control state 1 in the CZ circuits), so e1 is not a pure detector error.
  SX site p, repetition m: Z-axis z = v_m cos(m d + phi0), Y-axis z = -v_m sin((m+1) d + phi0);
    d = signed SX offset, phi0 = shared nuisance (preparation) phase, v_m = per-repetition visibility.
  CZ edge control->target, control state s in {0,1}, sigma = +1 (s=0) / -1 (s=1), m even:
    target phase theta = a_s + m (alpha_t + sigma delta); X-axis z = v cos(theta), Y-axis z = v sin(theta).
    The two-qubit residual Hamiltonian per application is H = 1/2 (alpha_c Z_c + alpha_t Z_t + delta Z_c Z_t).
    alpha_t = local_phase_rad, delta = zz_phase_rad of the JS design. alpha_c commutes with the measured
    control Z and acts on a computational-basis control only as a global phase, so it is NOT identifiable
    from this plan (the control's X/Y signal is not measured); the report marks it non-identifiable.
  The fit is a joint binomial (product-form multinomial) likelihood over all 120 (circuit, measured target
  site) observations plus the SPAM circuits, optimised by Gauss-Newton on deviance residuals. The model is a
  product over sites, so the joint-bitstring likelihood factorises into per-site Bernoulli terms; residual
  inter-site correlation is tested separately in the crosstalk diagnostics.

ALIAS SET (exact, derived in the report text and tested numerically)
  CZ: for a state with slope s, theta_m = a + m s is observed at m in {0,2,8}. Indistinguishable slopes
  satisfy 2 D = 0 and 8 D = 0 (mod 2 pi), i.e. D in pi Z (m=2 is the binding constraint; m=8 alone would
  give pi/4 Z but m=2 removes those, up to noise). Hence (alpha_t, delta) -> (alpha_t + j pi/2,
  delta + l pi/2) with j = l (mod 2). The reported representative has minimum alpha_t^2 + delta^2.
  SX: all repetitions m in {0,4,12,24} are even. Translations d -> d + D need m D = 0 (mod 2 pi) on the Z
  axis (D in (pi/2) Z) but the Y axis carries (m+1) D with m+1 = 1 present, forcing D = 0 (mod 2 pi): no pure
  translation alias. The reflection (d, phi0) -> (pi - d, -phi0) IS an exact alias: Z gives cos(m(pi-d) - phi0)
  = cos(m d + phi0) for even m, and Y gives -sin((m+1)(pi-d) - phi0) = -sin((m+1) d + phi0) for even m. It lies
  outside the +-0.1 rad design bound; any other competing minimum is found by the numerical scan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # tiny dense problems: threaded LAPACK is slower than single thread

import numpy as np
from scipy import stats
from scipy.optimize import least_squares
from scipy.special import xlogy

HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / "qc26a_plan.json"
SYN_PATH = HERE / "qc26a_synthetic.json"
VERSION = "QC26A-identify-v1"

SX_MS = (0, 4, 12, 24)
CZ_MS = (0, 2, 8)
SX_D_BOUND = 0.1          # JS grid bound
SX_PH_BOUND = 0.2         # JS nuisance-phase grid bound
V_MAX = 1.1               # JS visibility clip
LOW_VIS = 0.5             # JS low-visibility flag
BOUND_TOL = 1e-3          # |d| >= 0.1 - 0.001 counts as boundary (JS uses 0.099999 on a 0.0005 grid)
Z975 = 1.959963984540054
PI = math.pi


def wrap(x):
    return (np.asarray(x) + PI) % (2 * PI) - PI


def load_plan(path=PLAN_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def canonical_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------
# counts
# ----------------------------------------------------------------------------------------------
class Circ:
    """Joint bitstring distribution of one circuit. bits[u, j] is the bit of site path[j]."""

    __slots__ = ("bits", "cnt", "shots", "k")

    def __init__(self, bits, cnt):
        self.bits = bits
        self.cnt = np.asarray(cnt, dtype=np.int64)
        self.shots = int(self.cnt.sum())
        self.k = bits.T.astype(np.int64) @ self.cnt  # number of 1s per path site


def parse_counts(plan, counts):
    data = {}
    for r in plan["rows"]:
        c = counts.get(r["id"])
        if c is None:
            raise ValueError("missing circuit " + r["id"])
        n = len(r["path"])
        bits = np.zeros((len(c), n), dtype=np.uint8)
        cnt = np.zeros(len(c), dtype=np.int64)
        for i, (s, kk) in enumerate(c.items()):
            if len(s) != n or set(s) - {"0", "1"}:
                raise ValueError(f"bad bitstring {s!r} in {r['id']}")
            if isinstance(kk, float) and kk.is_integer():
                kk = int(kk)
            if not isinstance(kk, (int, np.integer)) or kk < 0:
                raise ValueError(f"bad count in {r['id']}")
            for j in range(n):
                bits[i, j] = 1 if s[n - 1 - j] == "1" else 0
            cnt[i] = kk
        if int(cnt.sum()) != r["shots"]:
            raise ValueError(f"shot mismatch in {r['id']}: {int(cnt.sum())} != {r['shots']}")
        data[r["id"]] = Circ(bits, cnt)
    return data


def resample(data, rng):
    """Joint-bitstring multinomial bootstrap: each circuit's full joint distribution, shots preserved."""
    out = {}
    for cid, c in data.items():
        out[cid] = Circ(c.bits, rng.multinomial(c.shots, c.cnt / c.shots))
    return out


# ----------------------------------------------------------------------------------------------
# model layout
# ----------------------------------------------------------------------------------------------
class Layout:
    def __init__(self, plan):
        self.plan = plan
        rows = plan["rows"]
        self.rows = rows
        self.byid = {r["id"]: r for r in rows}
        self.all_sites = list(self.byid["spam_0"]["path"])
        self.sx_sites = list(plan["sites"])
        self.edges = [tuple(e) for e in plan["edges"]]
        names = []

        def add(nm):
            names.append(nm)
            return len(names) - 1

        self.e0 = {p: add(f"spam.{p}.e0") for p in self.all_sites}
        self.e1 = {p: add(f"spam.{p}.e1") for p in self.all_sites}
        self.sx_d, self.sx_ph, self.sx_v = {}, {}, {}
        for p in self.sx_sites:
            self.sx_d[p] = add(f"sx.{p}.offset_rad")
            self.sx_ph[p] = add(f"sx.{p}.phase_rad")
            for m in SX_MS:
                self.sx_v[(p, m)] = add(f"sx.{p}.v{m}")
        self.cz_phi, self.cz_del, self.cz_a, self.cz_v = {}, {}, {}, {}
        for e in self.edges:
            key = f"{e[0]}_{e[1]}"
            self.cz_phi[e] = add(f"cz.{key}.alpha_target_rad")
            self.cz_del[e] = add(f"cz.{key}.delta_rad")
            for s in (0, 1):
                self.cz_a[(e, s)] = add(f"cz.{key}.intercept{s}_rad")
            for s in (0, 1):
                for m in CZ_MS:
                    self.cz_v[(e, s, m)] = add(f"cz.{key}.v{s}_{m}")
        self.names = names
        self.P = len(names)
        self.index = {nm: i for i, nm in enumerate(names)}
        lb = np.empty(self.P)
        ub = np.empty(self.P)
        th0 = np.empty(self.P)
        for nm, i in self.index.items():
            if nm.endswith(".e0") or nm.endswith(".e1"):
                lb[i], ub[i], th0[i] = 1e-6, 0.45, 0.01
            elif nm.endswith(".offset_rad"):
                lb[i], ub[i], th0[i] = -SX_D_BOUND, SX_D_BOUND, 0.0
            elif nm.startswith("sx.") and nm.endswith(".phase_rad"):
                lb[i], ub[i], th0[i] = -SX_PH_BOUND, SX_PH_BOUND, 0.0
            elif ".v" in nm:
                lb[i], ub[i], th0[i] = 0.0, V_MAX, 0.9
            elif nm.endswith(".alpha_target_rad") or nm.endswith(".delta_rad"):
                lb[i], ub[i], th0[i] = -PI, PI, 0.0
            else:  # CZ intercepts
                lb[i], ub[i], th0[i] = -7.0, 7.0, 0.0
        self.lb, self.ub, self.th0 = lb, ub, th0

        # observation table: SPAM (all sites), SX (all sites of the block), CZ (target site only)
        obs = []
        self.obs_key = {}

        def push(**kw):
            self.obs_key[kw["key"]] = len(obs)
            obs.append(kw)

        for r in rows:
            fam = r["family"]
            if fam == "spam":
                for j, p in enumerate(r["path"]):
                    push(key=("spam", r["state"], p), cid=r["id"], j=j, site=p, kind=0, m=0, quad=0, sigma=0,
                         zfix=1.0 - 2 * r["state"], i1=0, i2=0, i3=0, ia=0, deps=[self.e0[p], self.e1[p]])
            elif fam == "sx":
                for j, p in enumerate(r["path"]):
                    q = 0 if r["axis"] == "Z" else 1
                    push(key=("sx", p, r["m"], q), cid=r["id"], j=j, site=p, kind=1, m=r["m"], quad=q, sigma=0,
                         zfix=0.0, i1=self.sx_d[p], i2=self.sx_ph[p], i3=self.sx_v[(p, r["m"])], ia=0,
                         deps=[self.e0[p], self.e1[p], self.sx_d[p], self.sx_ph[p], self.sx_v[(p, r["m"])]])
            else:
                c, t = r["path"]
                e = (c, t)
                s = r["state"]
                q = 0 if r["axis"] == "X" else 1
                push(key=("cz", e, s, r["m"], q), cid=r["id"], j=1, site=t, kind=2, m=r["m"], quad=q,
                     sigma=1.0 - 2 * s, zfix=0.0, i1=self.cz_phi[e], i2=self.cz_del[e],
                     i3=self.cz_v[(e, s, r["m"])], ia=self.cz_a[(e, s)],
                     deps=[self.e0[t], self.e1[t], self.cz_phi[e], self.cz_del[e], self.cz_a[(e, s)],
                           self.cz_v[(e, s, r["m"])]])
        self.obs = obs
        self.N = len(obs)
        for f in ("kind", "m", "quad", "i1", "i2", "i3", "ia"):
            setattr(self, f, np.array([o[f] for o in obs], dtype=int))
        self.sigma = np.array([o["sigma"] for o in obs], dtype=float)
        self.zfix = np.array([o["zfix"] for o in obs], dtype=float)
        self.site = np.array([o["site"] for o in obs], dtype=int)
        self.ie0 = np.array([self.e0[o["site"]] for o in obs], dtype=int)
        self.ie1 = np.array([self.e1[o["site"]] for o in obs], dtype=int)
        self.deps = [o["deps"] for o in obs]
        self.cids = [o["cid"] for o in obs]
        self.js = np.array([o["j"] for o in obs], dtype=int)
        self.full = ObsSet(self, np.arange(self.N))
        self._sub = {}

    def subset(self, exclude=()):
        key = tuple(sorted(exclude))
        if not key:
            return self.full
        if key not in self._sub:
            sel = np.array([i for i, c in enumerate(self.cids) if c not in key])
            self._sub[key] = ObsSet(self, sel)
        return self._sub[key]

    def block_set(self, kind, ident):
        """Observation subset of one block (SX site or CZ edge) plus the SPAM circuits of its measured site."""
        key = (kind, ident)
        if key not in self._sub:
            if kind == "sx":
                p = ident
                sel = [i for i, o in enumerate(self.obs)
                       if (o["kind"] == 0 and o["site"] == p) or (o["kind"] == 1 and o["site"] == p)]
            else:
                e = ident
                sel = [i for i, o in enumerate(self.obs)
                       if (o["kind"] == 0 and o["site"] == e[1]) or (o["kind"] == 2 and o["key"][1] == e)]
            self._sub[key] = ObsSet(self, np.array(sel))
        return self._sub[key]

    def gather(self, data):
        K = np.array([data[c].k[j] for c, j in zip(self.cids, self.js)], dtype=float)
        Ns = np.array([data[c].shots for c in self.cids], dtype=float)
        return K, Ns


class ObsSet:
    def __init__(self, L, sel):
        self.L = L
        self.sel = np.asarray(sel)
        for f in ("kind", "m", "quad", "sigma", "zfix", "ie0", "ie1", "i1", "i2", "i3", "ia"):
            setattr(self, f, getattr(L, f)[self.sel])
        self.n = len(self.sel)
        self.sx_idx = np.where(self.kind == 1)[0]
        self.cz_idx = np.where(self.kind == 2)[0]
        self.deps = [L.deps[i] for i in self.sel]
        self.free = np.array(sorted(set().union(*[set(d) for d in self.deps])), dtype=int)
        self._color = {}

    def z(self, th):
        z = self.zfix.copy()
        s = self.sx_idx
        if len(s):
            d, ph, v, m = th[self.i1[s]], th[self.i2[s]], th[self.i3[s]], self.m[s]
            z[s] = np.where(self.quad[s] == 0, v * np.cos(m * d + ph), -v * np.sin((m + 1) * d + ph))
        c = self.cz_idx
        if len(c):
            theta = th[self.ia[c]] + self.m[c] * (th[self.i1[c]] + self.sigma[c] * th[self.i2[c]])
            v = th[self.i3[c]]
            z[c] = v * np.where(self.quad[c] == 0, np.cos(theta), np.sin(theta))
        return z

    def p1(self, th):
        e0, e1 = th[self.ie0], th[self.ie1]
        p = 0.5 * (1.0 - (e1 - e0) - (1.0 - e0 - e1) * self.z(th))
        return np.clip(p, 1e-9, 1 - 1e-9)

    def resid(self, th, k, n):
        p = self.p1(th)
        mu = n * p
        dev = 2.0 * (xlogy(k, k / mu) + xlogy(n - k, (n - k) / (n - mu)))
        return np.sign(k - mu) * np.sqrt(np.maximum(dev, 0.0))

    def coloring(self, free):
        key = tuple(free)
        if key not in self._color:
            col_rows = []
            for j in free:
                col_rows.append(np.array([r for r, d in enumerate(self.deps) if j in d], dtype=int))
            used, assign = [], []
            for rows in col_rows:
                rs = set(rows.tolist())
                for ci, u in enumerate(used):
                    if rs.isdisjoint(u):
                        u |= rs
                        assign.append(ci)
                        break
                else:
                    used.append(set(rs))
                    assign.append(len(used) - 1)
            self._color[key] = (col_rows, np.array(assign), len(used))
        return self._color[key]


def fd_jacobian(fun, x, r0, col_rows, assign, ncolors, lb, ub, h=1e-7):
    J = np.zeros((len(r0), len(x)))
    for c in range(ncolors):
        cols = np.where(assign == c)[0]
        sgn = np.where(x[cols] + h > ub[cols], -1.0, 1.0)
        xh = x.copy()
        xh[cols] += sgn * h
        rh = fun(xh)
        for jj, col in enumerate(cols):
            rows = col_rows[col]
            J[rows, col] = (rh[rows] - r0[rows]) / (sgn[jj] * h)
    return J


def fit_obs(L, oset, K, Ns, th0, fixed=None, max_nfev=300):
    """Gauss-Newton (trust region) on deviance residuals. K, Ns are full-length; oset selects rows."""
    k, n = K[oset.sel], Ns[oset.sel]
    th = np.clip(th0, L.lb, L.ub).copy()
    free = oset.free
    if fixed:
        for i, v in fixed.items():
            th[i] = v
        free = np.array([i for i in free if i not in fixed], dtype=int)
    if len(free) == 0:
        r = oset.resid(th, k, n)
        return th, float(r @ r), r
    lb, ub = L.lb[free], L.ub[free]
    col_rows, assign, nc = oset.coloring(free)

    def fun(x):
        t = th.copy()
        t[free] = x
        return oset.resid(t, k, n)

    def jac(x):
        return fd_jacobian(fun, x, fun(x), col_rows, assign, nc, lb, ub)

    x = np.clip(th[free], lb + 1e-9, ub - 1e-9)
    for _ in range(2):   # one restart resets the trust region; the likelihood has a weak SPAM/visibility valley
        res = least_squares(fun, x, jac=jac, bounds=(lb, ub), method="trf", x_scale=1.0,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=max_nfev)
        x = np.clip(res.x, lb + 1e-9, ub - 1e-9)
    th[free] = res.x
    r = res.fun
    return th, float(r @ r), r


def canonicalize(L, th):
    th = th.copy()
    for e in L.edges:
        phi, dl = th[L.cz_phi[e]], th[L.cz_del[e]]
        best = None
        for j in range(-3, 4):
            for l in range(-3, 4):
                if (j - l) % 2:
                    continue
                a, b = phi + j * PI / 2, dl + l * PI / 2
                nrm = a * a + b * b
                if best is None or nrm < best[0] - 1e-12:
                    best = (nrm, a, b)
        th[L.cz_phi[e]], th[L.cz_del[e]] = best[1], best[2]
        for s in (0, 1):
            th[L.cz_a[(e, s)]] = float(wrap(th[L.cz_a[(e, s)]]))
    return th


def align_to(L, th, ref):
    """Map a replicate onto the alias branch / angle branch closest to the reference estimate."""
    th = th.copy()
    for e in L.edges:
        phi, dl = th[L.cz_phi[e]], th[L.cz_del[e]]
        rp, rd = ref[L.cz_phi[e]], ref[L.cz_del[e]]
        best = None
        for j in range(-3, 4):
            for l in range(-3, 4):
                if (j - l) % 2:
                    continue
                a, b = phi + j * PI / 2, dl + l * PI / 2
                dist = (a - rp) ** 2 + (b - rd) ** 2
                if best is None or dist < best[0] - 1e-12:
                    best = (dist, a, b)
        th[L.cz_phi[e]], th[L.cz_del[e]] = best[1], best[2]
        for s in (0, 1):
            i = L.cz_a[(e, s)]
            th[i] = ref[i] + float(wrap(th[i] - ref[i]))
    return th


# ----------------------------------------------------------------------------------------------
# initial values (global grid search on SPAM-corrected means; same idea as the JS estimator)
# ----------------------------------------------------------------------------------------------
def spam_from_counts(L, data):
    e0, e1 = {}, {}
    for p in L.all_sites:
        j = L.byid["spam_0"]["path"].index(p)
        a, b = data["spam_0"], data["spam_1"]
        e0[p] = a.k[j] / a.shots
        e1[p] = 1.0 - b.k[j] / b.shots
    return e0, e1


def initial_guess(L, data, K, Ns):
    th = L.th0.copy()
    e0, e1 = spam_from_counts(L, data)
    for p in L.all_sites:
        th[L.e0[p]] = min(max(e0[p], 1e-4), 0.3)
        th[L.e1[p]] = min(max(e1[p], 1e-4), 0.3)

    def zc(key):
        i = L.obs_key[key]
        p = K[i] / Ns[i]
        e0_, e1_ = th[L.e0[L.site[i]]], th[L.e1[L.site[i]]]
        return ((1 - 2 * p) - (e1_ - e0_)) / (1 - e0_ - e1_)

    dgrid = np.arange(-200, 201) * 0.0005
    pgrid = np.arange(-20, 21) * 0.01
    for p in L.sx_sites:
        loss = np.zeros((len(dgrid), len(pgrid)))
        vs = []
        for m in SX_MS:
            z, y = zc(("sx", p, m, 0)), zc(("sx", p, m, 1))
            c = np.cos(m * dgrid[:, None] + pgrid[None, :])
            s = -np.sin((m + 1) * dgrid[:, None] + pgrid[None, :])
            v = np.clip((z * c + y * s) / (c * c + s * s), 0, V_MAX)
            vs.append(v)
            loss += (z - v * c) ** 2 + (y - v * s) ** 2
        a, b = np.unravel_index(np.argmin(loss), loss.shape)
        th[L.sx_d[p]], th[L.sx_ph[p]] = dgrid[a], pgrid[b]
        for m, v in zip(SX_MS, vs):
            th[L.sx_v[(p, m)]] = max(v[a, b], 1e-3)
    sgrid = np.arange(-393, 394) * 0.004          # slopes covering (-pi/2, pi/2)
    agrid = np.arange(-31, 32) * 0.1
    for e in L.edges:
        slopes = []
        for s in (0, 1):
            loss = np.zeros((len(agrid), len(sgrid)))
            vs = []
            for m in CZ_MS:
                x, y = zc(("cz", e, s, m, 0)), zc(("cz", e, s, m, 1))
                theta = agrid[:, None] + m * sgrid[None, :]
                c, sn = np.cos(theta), np.sin(theta)
                v = np.clip(x * c + y * sn, 0, V_MAX)
                vs.append(v)
                loss += (x - v * c) ** 2 + (y - v * sn) ** 2
            a, b = np.unravel_index(np.argmin(loss), loss.shape)
            slopes.append(sgrid[b])
            th[L.cz_a[(e, s)]] = agrid[a]
            for m, v in zip(CZ_MS, vs):
                th[L.cz_v[(e, s, m)]] = max(v[a, b], 1e-3)
        th[L.cz_phi[e]] = 0.5 * (slopes[0] + slopes[1])
        th[L.cz_del[e]] = 0.5 * (slopes[0] - slopes[1])
    return th


class Fit:
    def __init__(self, L, th, cost, resid, oset, K, Ns):
        self.L, self.th, self.cost, self.resid, self.oset = L, th, cost, resid, oset
        self.K, self.Ns = K, Ns

    def value(self, name):
        return float(self.th[self.L.index[name]])


def fit_all(L, data, th_init=None, exclude=(), K=None, Ns=None):
    if K is None:
        K, Ns = L.gather(data)
    th0 = th_init.copy() if th_init is not None else initial_guess(L, data, K, Ns)
    oset = L.subset(exclude)
    th, cost, r = fit_obs(L, oset, K, Ns, th0)
    th = canonicalize(L, th)
    return Fit(L, th, cost, r, oset, K, Ns)


# ----------------------------------------------------------------------------------------------
# JS-equivalent estimator (exact port of analyze() in qc26a_design.mjs)
# ----------------------------------------------------------------------------------------------
def js_mirror(plan, data):
    means = {}
    for r in plan["rows"]:
        c = data[r["id"]]
        means[r["id"]] = 1.0 - 2.0 * c.k / c.shots
    spam = {}
    a = means["spam_0"]
    b = means["spam_1"]
    for i, p in enumerate(plan["rows"][[r["id"] for r in plan["rows"]].index("spam_0")]["path"]):
        e0, e1 = (1 - a[i]) / 2, (1 + b[i]) / 2
        contrast = 1 - e0 - e1
        if contrast < 0.5:
            raise ValueError("SPAM contrast too low")
        spam[p] = dict(e0=e0, e1=e1, contrast=contrast, bias=e1 - e0)

    def corr(cid, j, p):
        return (means[cid][j] - spam[p]["bias"]) / spam[p]["contrast"]

    out_sx, out_cz = {}, {}
    d = np.arange(-200, 201) * 0.0005
    ph = np.arange(-20, 21) * 0.01
    for i, p in enumerate(plan["sites"]):
        loss = np.zeros((len(d), len(ph)))
        for m in SX_MS:
            z, y = corr(f"sx_{m}_Z", i, p), corr(f"sx_{m}_Y", i, p)
            c = np.cos(m * d[:, None] + ph[None, :])
            s = -np.sin((m + 1) * d[:, None] + ph[None, :])
            v = np.maximum(0, np.minimum(1.1, (z * c + y * s) / (c * c + s * s)))
            loss = loss + ((z - v * c) ** 2 + (y - v * s) ** 2)
        a_, b_ = np.unravel_index(np.argmin(loss), loss.shape)
        out_sx[p] = dict(offset_rad=float(d[a_]), phase_rad=float(ph[b_]), loss=float(loss[a_, b_]),
                         at_boundary=bool(abs(d[a_]) >= 0.099999))
    for c_, t in plan["edges"]:
        states = []
        for state in (0, 1):
            phases, vis = [], []
            for m in CZ_MS:
                x = corr(f"cz_{c_}_{t}_{state}_{m}_X", 1, t)
                y = corr(f"cz_{c_}_{t}_{state}_{m}_Y", 1, t)
                p_ = math.atan2(y, x)
                if phases:
                    while p_ - phases[-1] > PI:
                        p_ -= 2 * PI
                    while p_ - phases[-1] < -PI:
                        p_ += 2 * PI
                phases.append(p_)
                vis.append(math.hypot(x, y))
            mx = sum(CZ_MS) / 3
            my = sum(phases) / 3
            slope = sum((m - mx) * (q - my) for m, q in zip(CZ_MS, phases)) / sum((m - mx) ** 2 for m in CZ_MS)
            states.append(dict(slope_rad=slope, intercept_rad=my - slope * mx, min_visibility=min(vis),
                               phases=phases))
        out_cz[f"{c_}_{t}"] = dict(local_phase_rad=(states[0]["slope_rad"] + states[1]["slope_rad"]) / 2,
                                   zz_phase_rad=(states[0]["slope_rad"] - states[1]["slope_rad"]) / 2,
                                   states=states, low_visibility=any(s["min_visibility"] < 0.5 for s in states))
    return dict(spam=spam, sx=out_sx, cz=out_cz)


# ----------------------------------------------------------------------------------------------
# synthetic generator (port of synthetic()/generate() in qc26a_design.mjs; same LCG so draw-for-draw
# identical to the committed JS trials when seed and truth are the JS ones)
# ----------------------------------------------------------------------------------------------
JS_TRUTH = {
    "sx": {3: 0.004, 5: -0.025, 21: 0.0, 23: 0.04},
    "cz": {"4_5": 0.032, "6_5": -0.022, "16_23": 0.047, "22_23": -0.038, "4_3": 0.0, "22_21": 0.0},
    "local": {"4_5": 0.013, "6_5": -0.012, "16_23": 0.01, "22_23": 0.006, "4_3": -0.008, "22_21": 0.009},
}


class LCG:
    """The JS generator: seed = (imul(seed,1664525)+1013904223)>>>0; u = seed/2^32."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def __call__(self):
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s / 4294967296.0


def synthetic_counts(plan, truth=None, rng=None, zero=False, np_rng=None, sx_phase_scale=0.017,
                     cz_intercepts=(0.035, -0.025), sx_vis=(0.99, 0.0018), cz_vis=(0.97, 0.004)):
    """Counts dict for one synthetic run. rng: callable uniform draw (LCG) or np_rng: numpy Generator."""
    truth = truth or JS_TRUTH
    out = {}
    for r in plan["rows"]:
        path = r["path"]
        z = []
        for i, p in enumerate(path):
            if r["family"] == "spam":
                z.append(1 - 2 * r["state"])
            elif r["family"] == "sx":
                d = 0.0 if zero else truth["sx"][p]
                phase = sx_phase_scale * (i - 1.5)
                v = sx_vis[0] * math.exp(-sx_vis[1] * r["m"])
                z.append(v * math.cos(r["m"] * d + phase) if r["axis"] == "Z"
                         else -v * math.sin((r["m"] + 1) * d + phase))
            else:
                if i == 0:
                    z.append(1 - 2 * r["state"])
                    continue
                key = "_".join(str(q) for q in path)
                d = 0.0 if zero else truth["cz"][key]
                sg = 1 if r["state"] == 0 else -1
                phase = (cz_intercepts[0] if r["state"] == 0 else cz_intercepts[1]) + r["m"] * (
                    truth["local"][key] + sg * d)
                v = cz_vis[0] * math.exp(-cz_vis[1] * r["m"])
                z.append(v * (math.cos(phase) if r["axis"] == "X" else math.sin(phase)))
        probs = []
        for z_, p in zip(z, path):
            e0 = 0.004 + (p % 5) * 0.002
            e1 = 0.008 + (p % 3) * 0.003
            probs.append((1 - ((e1 - e0) + (1 - e0 - e1) * z_)) / 2)
        n = len(path)
        ct = {}
        if rng is not None:
            for _ in range(r["shots"]):
                bits = ["1" if rng() < pr else "0" for pr in probs]
                s = "".join(reversed(bits))
                ct[s] = ct.get(s, 0) + 1
        else:
            u = np_rng.random((r["shots"], n)) < np.array(probs)[None, :]
            for row in u:
                s = "".join("1" if b else "0" for b in row[::-1])
                ct[s] = ct.get(s, 0) + 1
        out[r["id"]] = ct
    return out


# ----------------------------------------------------------------------------------------------
# Hamiltonian-level check: which of (alpha_c, alpha_t, delta) is observable?
# ----------------------------------------------------------------------------------------------
def hamiltonian_prediction(alpha_c, alpha_t, delta, state, m):
    """Exact 2-qubit evolution under H = 1/2 (a_c Z_c + a_t Z_t + d Z_c Z_t) applied m times (unit time),
    control prepared in |state>, target in |+>. Returns (<Z_c>, <X_t>, <Y_t>)."""
    Z = np.diag([1.0, -1.0])
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]])
    I = np.eye(2)
    H = 0.5 * (alpha_c * np.kron(Z, I) + alpha_t * np.kron(I, Z) + delta * np.kron(Z, Z))
    U = np.diag(np.exp(-1j * np.diag(H) * m))
    c = np.zeros(2, dtype=complex)
    c[state] = 1
    t = np.array([1, 1], dtype=complex) / math.sqrt(2)
    psi = U @ np.kron(c, t)

    def ex(O):
        return float(np.real(np.conj(psi) @ O @ psi))

    return ex(np.kron(Z, I)), ex(np.kron(I, X)), ex(np.kron(I, Y))


def alpha_control_sensitivity(alpha_t, delta, h=1e-4):
    """Largest |d prediction / d alpha_c| over states s in {0,1} and m in {0,2,8} (central difference)."""
    worst = 0.0
    for s in (0, 1):
        for m in CZ_MS:
            a = np.array(hamiltonian_prediction(+h, alpha_t, delta, s, m))
            b = np.array(hamiltonian_prediction(-h, alpha_t, delta, s, m))
            worst = max(worst, float(np.max(np.abs(a - b)) / (2 * h)))
    return worst


# ----------------------------------------------------------------------------------------------
# Jacobian / Fisher information / identifiability
# ----------------------------------------------------------------------------------------------
WEAK_SV = 20.0   # whitened sv below which a direction has 1-sigma > 0.05 (rad or probability) = largest effect sought


def information(L, oset, th, Ns, h=1e-6):
    """Whitened central-difference Jacobian of the predicted per-(circuit,site) expectations E[1-2b]."""
    free = oset.free
    col_rows, assign, nc = oset.coloring(free)
    x = th[free]
    n = Ns[oset.sel]

    def pred(xx):
        t = th.copy()
        t[free] = xx
        return 1.0 - 2.0 * oset.p1(t)

    J = np.zeros((oset.n, len(free)))
    for c in range(nc):
        cols = np.where(assign == c)[0]
        xp, xm = x.copy(), x.copy()
        xp[cols] += h
        xm[cols] -= h
        rp, rm = pred(xp), pred(xm)
        for col in cols:
            rows = col_rows[col]
            J[rows, col] = (rp[rows] - rm[rows]) / (2 * h)
    p = oset.p1(th)
    sd = 2.0 * np.sqrt(np.clip(p * (1 - p), 1e-4, None) / n)
    Jw = J / sd[:, None]
    U, s, Vt = np.linalg.svd(Jw, full_matrices=False)
    return dict(Jw=Jw, s=s, V=Vt.T, free=free, names=[L.names[i] for i in free])


def identifiability(L, oset, th, Ns):
    info = information(L, oset, th, Ns)
    s, V, free, names = info["s"], info["V"], info["free"], info["names"]
    tol = max(info["Jw"].shape) * np.finfo(float).eps * s[0]
    keep = s > tol
    var = (V[:, keep] ** 2 / (s[keep] ** 2)[None, :]).sum(1)
    null_w = (V[:, ~keep] ** 2).sum(1) if (~keep).any() else np.zeros(len(free))
    weak_w = (V[:, s < WEAK_SV] ** 2).sum(1)
    cs = info["Jw"] / np.maximum(np.linalg.norm(info["Jw"], axis=0), 1e-300)
    s_cs = np.linalg.svd(cs, compute_uv=False)
    blocks = {}
    for p in L.sx_sites:
        cols = [jj for jj, i in enumerate(free) if L.names[i].startswith(f"sx.{p}.")]
        if not cols:
            continue
        sb = np.linalg.svd(info["Jw"][:, cols], compute_uv=False)
        blocks[f"sx_{p}"] = dict(n_params=len(cols), singular_values=sb, condition=float(sb[0] / max(sb[-1], 1e-300)))
    for e in L.edges:
        cols = [jj for jj, i in enumerate(free) if L.names[i].startswith(f"cz.{e[0]}_{e[1]}.")]
        if not cols:
            continue
        sb = np.linalg.svd(info["Jw"][:, cols], compute_uv=False)
        blocks[f"cz_{e[0]}_{e[1]}"] = dict(n_params=len(cols), singular_values=sb,
                                          condition=float(sb[0] / max(sb[-1], 1e-300)))
    per_param = {}
    for jj, i in enumerate(free):
        per_param[L.names[i]] = dict(
            wald_se=float(math.sqrt(var[jj])) if null_w[jj] < 0.5 else None,
            null_weight=float(null_w[jj]), weak_weight=float(weak_w[jj]),
            weakly_determined=bool(weak_w[jj] >= 0.5), non_identifiable=bool(null_w[jj] >= 0.5),
            at_bound=bool(th[i] <= L.lb[i] + 1e-6 or th[i] >= L.ub[i] - 1e-6))
    return dict(
        n_observations=int(oset.n), n_parameters=int(len(free)), singular_values=s,
        rank_numeric=int(keep.sum()), condition_number=float(s[0] / max(s[-1], 1e-300)),
        condition_number_column_scaled=float(s_cs[0] / max(s_cs[-1], 1e-300)),
        weak_sv_threshold=WEAK_SV, n_weak_directions=int((s < WEAK_SV).sum()),
        weak_direction_params=sorted(n for n, v in per_param.items() if v["weakly_determined"]),
        smallest_sv_dominant_params=[names[j] for j in np.argsort(-np.abs(V[:, -1]))[:5]],
        blocks=blocks, per_parameter=per_param,
        note="Whitened Jacobian of E[1-2b] w.r.t. fitted parameters in natural units (rad, probability). "
             "J^T J is the Fisher information; Wald SE = sqrt(diag((J^T J)^-1)).")


# ----------------------------------------------------------------------------------------------
# profile likelihood (block-conditional: the block's SPAM parameters are nuisance and re-optimised)
# ----------------------------------------------------------------------------------------------
CHI2_95_1 = 3.841458820694124


def profile_interval(grid, dev, crit=CHI2_95_1):
    grid, dev = np.asarray(grid), np.asarray(dev)
    i0 = int(np.argmin(dev))

    def cross(idx, step):
        j = idx
        while 0 <= j + step < len(grid):
            if dev[j + step] > crit:
                g0, d0, g1, d1 = grid[j], dev[j], grid[j + step], dev[j + step]
                return g0 + (crit - d0) * (g1 - g0) / (d1 - d0), False
            j += step
        return grid[j], True

    lo, open_lo = cross(i0, -1)
    hi, open_hi = cross(i0, +1)
    return float(lo), float(hi), bool(open_lo), bool(open_hi)


def local_minima(grid, dev):
    out = []
    for i in range(len(grid)):
        left = dev[i - 1] if i > 0 else np.inf
        right = dev[i + 1] if i < len(grid) - 1 else np.inf
        if dev[i] <= left and dev[i] <= right:
            out.append(i)
    return out


def profile_parameter(L, kind, ident, name, th, K, Ns, local_se, n_local=25, n_coarse=41):
    idx = L.index[name]
    oset = L.block_set(kind, ident)
    th_free, cost_free, _ = fit_obs(L, oset, K, Ns, th)
    est = th_free[idx]
    if kind == "sx" and name.endswith("offset_rad"):
        coarse = np.linspace(-SX_D_BOUND, SX_D_BOUND, n_coarse)
    else:
        coarse = np.linspace(-0.3, 0.3, n_coarse)
    half = float(np.clip(6 * (local_se if local_se else 0.01), 0.004, 0.2))
    local = np.linspace(est - half, est + half, n_local)
    if L.lb[idx] > -PI:
        local = np.clip(local, L.lb[idx], L.ub[idx])

    def sweep(grid):
        order = np.argsort(np.abs(grid - est))
        costs = np.empty(len(grid))
        last = {}
        for pos in order:
            g = float(grid[pos])
            side = g >= est
            t, c, _ = fit_obs(L, oset, K, Ns, last.get(side, th_free), fixed={idx: g})
            last[side] = t
            costs[pos] = c
        return costs

    c_local, c_coarse = sweep(local), sweep(coarse)
    ref = min(cost_free, c_local.min(), c_coarse.min())
    dl, dc = c_local - ref, c_coarse - ref
    lo, hi, olo, ohi = profile_interval(local, dl)
    if olo or ohi:   # widen with the coarse grid
        g = np.concatenate([coarse, local])
        d = np.concatenate([dc, dl])
        o = np.argsort(g)
        lo2, hi2, olo2, ohi2 = profile_interval(g[o], d[o])
        lo, olo = (lo2, olo2) if olo else (lo, olo)
        hi, ohi = (hi2, ohi2) if ohi else (hi, ohi)
    best_i = int(np.argmin(dc))
    secondary = []
    for i in local_minima(coarse, dc):
        if i == best_i:
            continue
        a, b = sorted((i, best_i))
        barrier = float(dc[a:b + 1].max())
        if dc[i] <= 9.0 and barrier - dc[i] >= 1.0:
            secondary.append(dict(value=float(coarse[i]), delta_deviance=float(dc[i]), barrier=barrier))
    flat = bool((dc[0] < 9.0 and dc[-1] < 9.0) or float((dc < CHI2_95_1).mean()) > 0.5)
    return dict(
        parameter=name, estimate_block_fit=float(est),
        profile_ci95=[lo, hi], interval_open_low=olo, interval_open_high=ohi,
        profile_halfwidth=float((hi - lo) / 2), flat=flat, multimodal=bool(secondary),
        secondary_minima=secondary,
        coarse_grid=[float(x) for x in coarse], coarse_delta_deviance=[float(x) for x in dc],
        local_grid=[float(x) for x in local], local_delta_deviance=[float(x) for x in dl],
        excludes_zero=bool(lo > 0 or hi < 0), crit_delta_deviance=CHI2_95_1)


# ----------------------------------------------------------------------------------------------
# alias analysis
# ----------------------------------------------------------------------------------------------
def lattice_candidates(phi, delta, span=2, top=6):
    c = []
    for j in range(-span, span + 1):
        for l in range(-span, span + 1):
            if (j - l) % 2 == 0:
                a, b = phi + j * PI / 2, delta + l * PI / 2
                c.append((math.hypot(a, b), a, b, j, l))
    c.sort()
    return [dict(alpha_target=a, delta=b, shift_j=j, shift_l=l, norm=n) for n, a, b, j, l in c[:top]]


def corrected(L, th, K, Ns, key):
    """SPAM-corrected mean of one observation and its binomial SE (plug-in with +0.5 smoothing)."""
    i = L.obs_key[key]
    e0, e1 = th[L.ie0[i]], th[L.ie1[i]]
    ph = (K[i] + 0.5) / (Ns[i] + 1.0)
    c = 1 - e0 - e1
    return ((1 - 2 * (K[i] / Ns[i])) - (e1 - e0)) / c, 2 * math.sqrt(ph * (1 - ph) / Ns[i]) / c


def cz_alias_scan(L, th, K, Ns):
    sgrid = np.arange(-1571, 1572) * 0.002
    agrid = np.arange(-63, 64) * 0.05
    out = {}
    for e in L.edges:
        states = []
        for s in (0, 1):
            loss = np.zeros((len(agrid), len(sgrid)))
            for m in CZ_MS:
                x, sx_ = corrected(L, th, K, Ns, ("cz", e, s, m, 0))
                y, sy_ = corrected(L, th, K, Ns, ("cz", e, s, m, 1))
                theta = agrid[:, None] + m * sgrid[None, :]
                c, sn = np.cos(theta), np.sin(theta)
                v = np.clip((x * c / sx_ ** 2 + y * sn / sy_ ** 2) / (c * c / sx_ ** 2 + sn * sn / sy_ ** 2), 0, V_MAX)
                loss += ((x - v * c) / sx_) ** 2 + ((y - v * sn) / sy_) ** 2
            prof = loss.min(0)
            best = float(prof.min())
            s_best = float(sgrid[int(np.argmin(prof))])
            s_best -= PI * round(s_best / PI)      # principal representative of the exact alias class
            comp = []
            for i in local_minima(sgrid, prof):
                d = float(prof[i] - best)
                if d < 25.0 and abs(sgrid[i] - s_best) > 0.02:
                    k = (sgrid[i] - s_best) / PI
                    comp.append(dict(slope=float(sgrid[i]), delta_chi2=d, exact_alias=bool(abs(k - round(k)) < 0.02)))
            fine = [c_ for c_ in comp if not c_["exact_alias"] and c_["delta_chi2"] < 9.0]
            states.append(dict(state=s, best_slope_rad=s_best, best_chi2=best, minima=comp,
                               fine_alias_competitors=fine))
        phi, dl = th[L.cz_phi[e]], th[L.cz_del[e]]
        cands = lattice_candidates(phi, dl)
        margin = cands[1]["norm"] - cands[0]["norm"]
        fine_amb = any(len(st["fine_alias_competitors"]) for st in states)
        out[f"{e[0]}_{e[1]}"] = dict(
            states=states, lattice_candidates=cands, alias_margin_rad=float(margin),
            near_alias_boundary=bool(margin < 0.5), fine_alias_ambiguous=bool(fine_amb),
            alias_ambiguous=bool(margin < 0.5 or fine_amb))
    return out


def sx_alias_scan(L, th, K, Ns):
    dgrid = np.arange(-3141, 3142) * 0.001
    pgrid = np.linspace(-PI, PI, 240, endpoint=False)
    inb = np.abs(dgrid) <= SX_D_BOUND + 1e-9
    out = {}
    for p in L.sx_sites:
        obs = {m: (corrected(L, th, K, Ns, ("sx", p, m, 0)), corrected(L, th, K, Ns, ("sx", p, m, 1))) for m in SX_MS}
        prof = np.empty(len(dgrid))
        for a in range(0, len(dgrid), 800):
            d = dgrid[a:a + 800, None]
            loss = np.zeros((d.shape[0], len(pgrid)))
            for m in SX_MS:
                (z, sz), (y, sy) = obs[m]
                c = np.cos(m * d + pgrid[None, :])
                s = -np.sin((m + 1) * d + pgrid[None, :])
                den = c * c / sz ** 2 + s * s / sy ** 2
                v = np.clip((z * c / sz ** 2 + y * s / sy ** 2) / np.maximum(den, 1e-12), 0, V_MAX)
                loss += ((z - v * c) / sz) ** 2 + ((y - v * s) / sy) ** 2
            prof[a:a + 800] = loss.min(1)
        best = float(prof.min())
        best_in = float(prof[inb].min())
        d_in = float(dgrid[inb][int(np.argmin(prof[inb]))])
        d_mirror = float(wrap(PI - d_in))     # exact reflection alias (pi - d, -phi0)
        comp = []
        for i in local_minima(dgrid, prof):
            dd = float(prof[i] - best_in)
            if dd < 25.0 and abs(dgrid[i] - d_in) > 0.02:
                comp.append(dict(offset=float(dgrid[i]), delta_chi2_vs_inbound_best=dd,
                                 inside_bound=bool(abs(dgrid[i]) <= SX_D_BOUND),
                                 exact_reflection_alias=bool(abs(wrap(dgrid[i] - d_mirror)) < 0.02)))
        nonalias = [c for c in comp if not c["exact_reflection_alias"]]
        out[str(p)] = dict(
            best_offset_inside_bound=d_in, reflection_alias_offset=d_mirror,
            reflection_alias_present=bool(any(c["exact_reflection_alias"] and c["delta_chi2_vs_inbound_best"] < 3.0 for c in comp)),
            delta_chi2_global_vs_inbound=float(best_in - best),
            global_minimum_outside_bound=bool(best_in - best > 2.0 and not any(
                c["exact_reflection_alias"] and abs(c["delta_chi2_vs_inbound_best"]) < 3.0 for c in comp)
                and abs(float(dgrid[int(np.argmin(prof))]) - d_mirror) > 0.02),
            competing_minima=comp,
            ambiguous=bool(any(c["delta_chi2_vs_inbound_best"] < 9.0 for c in nonalias)))
    return out


# ----------------------------------------------------------------------------------------------
# reporting helpers and bootstrap
# ----------------------------------------------------------------------------------------------
def derived_names(L):
    nm = []
    for e in L.edges:
        k = f"cz.{e[0]}_{e[1]}"
        nm += [k + ".slope0_rad", k + ".slope1_rad", k + ".intercept_zz_rad"]
    for p in L.all_sites:
        nm += [f"spam.{p}.contrast", f"spam.{p}.bias"]
    return nm


def derived_values(L, th):
    v = []
    for e in L.edges:
        phi, dl = th[L.cz_phi[e]], th[L.cz_del[e]]
        v += [phi + dl, phi - dl, 0.5 * (th[L.cz_a[(e, 0)]] - th[L.cz_a[(e, 1)]])]
    for p in L.all_sites:
        v += [1 - th[L.e0[p]] - th[L.e1[p]], th[L.e1[p]] - th[L.e0[p]]]
    return np.array(v)


def all_values(L, th):
    return np.concatenate([th, derived_values(L, th)])


def flag_vector(L, th):
    """Per-entity flags for one estimate: boundary, low visibility, clipping, alias margin."""
    f = {}
    for p in L.sx_sites:
        vs = np.array([th[L.sx_v[(p, m)]] for m in SX_MS])
        f[f"sx_{p}"] = dict(boundary=bool(abs(th[L.sx_d[p]]) >= SX_D_BOUND - BOUND_TOL),
                            phase_boundary=bool(abs(th[L.sx_ph[p]]) >= SX_PH_BOUND - BOUND_TOL),
                            low_visibility=bool((vs < LOW_VIS).any()), clipped_low=bool((vs < 1e-4).any()),
                            clipped_high=bool((vs > V_MAX - 1e-4).any()))
    for e in L.edges:
        vs = np.array([th[L.cz_v[(e, s, m)]] for s in (0, 1) for m in CZ_MS])
        phi, dl = th[L.cz_phi[e]], th[L.cz_del[e]]
        f[f"cz_{e[0]}_{e[1]}"] = dict(
            large_phase=bool(max(abs(phi), abs(dl)) > 0.25), low_visibility=bool((vs < LOW_VIS).any()),
            clipped_low=bool((vs < 1e-4).any()), clipped_high=bool((vs > V_MAX - 1e-4).any()),
            alias_boundary=bool(abs(phi) + abs(dl) > PI / 2 - 0.3))
    return f


def pct_ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def run_bootstrap(L, data, est_th, n_boot, seed, progress=False):
    rng = np.random.default_rng(seed)
    reps, flags, failures = [], [], 0
    for b in range(n_boot):
        db = resample(data, rng)
        try:
            f = fit_all(L, db)
        except Exception:
            failures += 1
            continue
        reps.append(all_values(L, align_to(L, f.th, est_th)))
        flags.append(flag_vector(L, f.th))
        if progress and (b + 1) % 50 == 0:
            print(f"bootstrap {b + 1}/{n_boot}", file=sys.stderr, flush=True)
    return dict(names=L.names + derived_names(L), reps=np.array(reps), flags=flags, failures=failures,
                n_requested=n_boot)


def summarize_bootstrap(L, est_th, boot):
    reps = boot["reps"]
    n = len(reps)
    if n < 2:
        return dict(n_replicates=int(n), failures=boot["failures"], parameters={}, flag_fractions={},
                    flag_fractions_any_entity={})
    est = all_values(L, est_th)
    params = {}
    for j, nm in enumerate(boot["names"]):
        x = reps[:, j]
        lo, hi = pct_ci(x)
        params[nm] = dict(estimate=float(est[j]), se=float(np.std(x, ddof=1)), ci95=[lo, hi],
                          bias=float(x.mean() - est[j]), ci_excludes_zero=bool(lo > 0 or hi < 0))
    ff, anyf = {}, {}
    for ent in boot["flags"][0]:
        ff[ent] = {k: float(np.mean([f[ent][k] for f in boot["flags"]])) for k in boot["flags"][0][ent]}
    keys = sorted({k for ent in boot["flags"][0].values() for k in ent})
    for k in keys:
        anyf[k] = float(np.mean([any(e[k] for e in f.values() if k in e) for f in boot["flags"]]))
    return dict(n_replicates=int(n), failures=boot["failures"], parameters=params, flag_fractions=ff,
                flag_fractions_any_entity=anyf,
                note="joint-bitstring multinomial bootstrap, SPAM circuits resampled and everything refit "
                     "(global initial grid search each replicate); replicates are mapped to the alias/angle branch "
                     "closest to the point estimate before percentiles are taken")


def wald_ses(L, oset, th, Ns):
    info = identifiability(L, oset, th, Ns)
    return {k: v["wald_se"] for k, v in info["per_parameter"].items()}, info



# ----------------------------------------------------------------------------------------------
# alternative-explanation diagnostics
# ----------------------------------------------------------------------------------------------
PRACTICAL_POP = 0.02       # 2% population change: above typical single-qubit readout error (<~1-3%)
PHASE_CONTEXT = 0.05       # rad: as large as the 0.02-0.05 rad effects the experiment must resolve
ALPHA = 0.05
FAMILIES = 10              # diagnostic families: sx pair corr, cz CMH, control-vs-SPAM, prep-phase context, leakage drift,
                           # relaxation, cz nonlinearity, cz amplitude, sx quadrature, drop-longest
ALPHA_FAM = ALPHA / FAMILIES   # suite-level 5% split across families, then Bonferroni over tests inside a family


def _bonf_z(ntests, alpha=ALPHA_FAM):
    return float(stats.norm.isf(alpha / 2 / max(ntests, 1)))


def _pair_counts(c, a, b):
    N = c.shots
    na = float((c.cnt * c.bits[:, a]).sum())
    nb = float((c.cnt * c.bits[:, b]).sum())
    n11 = float((c.cnt * c.bits[:, a] * c.bits[:, b]).sum())
    E = na * nb / N
    V = na * (N - na) * nb * (N - nb) / (N * N * (N - 1))
    return n11, E, V


def diag_crosstalk(L, data, th, se, rng):
    out = {}
    sites = L.sx_sites
    neigh = {frozenset((3, 5)): 4, frozenset((21, 23)): 22}
    tests = []
    for r in L.rows:
        if r["family"] != "sx":
            continue
        for a in range(len(sites)):
            for b in range(a + 1, len(sites)):
                n11, E, V = _pair_counts(data[r["id"]], a, b)
                if V >= 1.0:
                    tests.append(dict(circuit=r["id"], pair=[sites[a], sites[b]], z=float((n11 - E) / math.sqrt(V)),
                                      shared_neighbour=neigh.get(frozenset((sites[a], sites[b])))))
    zc = _bonf_z(len(tests))
    worst = max(tests, key=lambda t: abs(t["z"])) if tests else None
    bad = [t for t in tests if abs(t["z"]) > zc]
    out["sx_block_pair_correlation"] = dict(
        n_tests=len(tests), z_crit_bonferroni=zc, max_abs_z=abs(worst["z"]) if worst else None, worst=worst,
        flagged=bad, flag=bool(bad),
        max_abs_z_shared_neighbour_pairs=max([abs(t["z"]) for t in tests if t["shared_neighbour"]], default=None),
        threshold_justification="Bonferroni two-sided alpha=0.005 (0.05 split over 10 diagnostic families) over all tested (circuit, pair) cells; "
                                "z = (n11 - E)/sd under independence given the observed marginals (hypergeometric)")
    # CZ control-target association per edge (Cochran-Mantel-Haenszel over the edge's 12 circuits)
    cmh = {}
    for e in L.edges:
        num = den = 0.0
        for s in (0, 1):
            for m in CZ_MS:
                for ax in "XY":
                    n11, E, V = _pair_counts(data[f"cz_{e[0]}_{e[1]}_{s}_{m}_{ax}"], 0, 1)
                    num += n11 - E
                    den += V
        cmh[f"{e[0]}_{e[1]}"] = dict(z=float(num / math.sqrt(den)) if den >= 1.0 else None, total_variance=float(den))
    zc2 = _bonf_z(sum(v["z"] is not None for v in cmh.values()))
    for v in cmh.values():
        v["flag"] = bool(v["z"] is not None and abs(v["z"]) > zc2)
    out["cz_control_target_association_cmh"] = dict(per_edge=cmh, z_crit_bonferroni=zc2,
                                                    flag=any(v["flag"] for v in cmh.values()))
    # control marginal while the target runs Ramsey vs the idle SPAM preparation (neighbour marginal shift)
    shifts = {}
    zc3 = _bonf_z(2 * len(L.edges))
    for e in L.edges:
        for s in (0, 1):
            ids = [f"cz_{e[0]}_{e[1]}_{s}_{m}_{ax}" for m in CZ_MS for ax in "XY"]
            unexp = np.array([(data[i].k[0] if s == 0 else data[i].shots - data[i].k[0]) for i in ids], dtype=float)
            ns = np.array([data[i].shots for i in ids], dtype=float)
            sp = data[f"spam_{s}"]
            j = L.byid[f"spam_{s}"]["path"].index(e[0])
            ksp = float(sp.k[j] if s == 0 else sp.shots - sp.k[j])
            table = [[unexp.sum(), ns.sum() - unexp.sum()], [ksp, sp.shots - ksp]]
            pval = float(stats.fisher_exact(table)[1])
            diff = float(unexp.sum() / ns.sum() - ksp / sp.shots)
            shifts[f"{e[0]}_{e[1]}_state{s}"] = dict(
                control=e[0], pooled_unexpected_rate=float(unexp.sum() / ns.sum()), spam_unexpected_rate=ksp / sp.shots,
                difference=diff, fisher_p=pval, flag=bool(pval < ALPHA_FAM / (2 * len(L.edges)) and abs(diff) > PRACTICAL_POP))
    out["control_marginal_vs_spam"] = dict(
        per_circuit_group=shifts, flag=any(v["flag"] for v in shifts.values()),
        threshold_justification=f"Fisher exact p < 0.005/{2 * len(L.edges)} (Bonferroni, suite-level alpha split over 10 families) AND |difference| > {PRACTICAL_POP} "
                                "(2% population: above typical readout error, enough to mix control-state phases)")
    # preparation-phase context: concurrent SX block vs isolated CZ circuits (same physical qubit)
    ctx = {}
    n_pairs = sum(1 for p in L.sx_sites for e in L.edges if e[1] == p)
    for p in L.sx_sites:
        for e in L.edges:
            if e[1] != p:
                continue
            a0, a1 = th[L.cz_a[(e, 0)]], th[L.cz_a[(e, 1)]]
            amean = 0.5 * (a0 + a1)
            ph = th[L.sx_ph[p]]
            diff = float(wrap(ph - amean))
            vs = [se.get(f"sx.{p}.phase_rad"), se.get(f"cz.{e[0]}_{e[1]}.intercept0_rad"), se.get(f"cz.{e[0]}_{e[1]}.intercept1_rad")]
            if all(v is not None for v in vs):
                sd = math.sqrt(vs[0] ** 2 + 0.25 * vs[1] ** 2 + 0.25 * vs[2] ** 2)
            else:
                sd = None
            ctx[f"site{p}_vs_edge{e[0]}_{e[1]}"] = dict(
                sx_block_prep_phase=float(ph), cz_isolated_mean_intercept=float(amean), difference=diff, se=sd,
                z=None if not sd else diff / sd,
                zz_intercept_half_difference=float(0.5 * wrap(a0 - a1)),
                flag=bool(abs(diff) > PHASE_CONTEXT and sd is not None and abs(diff) / sd > _bonf_z(n_pairs)))
    out["prep_phase_block_vs_isolated"] = dict(
        per_pair=ctx, flag=any(v["flag"] for v in ctx.values()),
        threshold_justification=f"|difference| > {PHASE_CONTEXT} rad (comparable to the 0.02-0.05 rad effects sought) "
                                "AND > the Bonferroni z for 6 pairs at alpha 0.005 (SE assumes independent estimates). The intercept is the m=0 phase "
                                "averaged over control states; the state-dependent half-difference is reported "
                                "separately and is not an alarm by itself.")
    out["flag"] = bool(out["sx_block_pair_correlation"]["flag"] or out["cz_control_target_association_cmh"]["flag"]
                       or out["control_marginal_vs_spam"]["flag"] or out["prep_phase_block_vs_isolated"]["flag"])
    return out


def _trend_test(ks, ns, ms, rng, sims=20000):
    """Exact-conditional trend test: given the total T, counts are Multinomial(T, n_i/sum n) under no drift."""
    T = int(ks.sum())
    q = ns / ns.sum()
    S = float((ms * ks).sum())
    ES = T * float(q @ ms)
    if T == 0:
        return 1.0
    draws = rng.multinomial(T, q, size=sims) @ ms
    return float((np.sum(np.abs(draws - ES) >= abs(S - ES) - 1e-9) + 1) / (sims + 1))


def diag_leakage(L, data, th, rng):
    per = {}
    zc = 2 * len(L.edges)
    for e in L.edges:
        for s in (0, 1):
            ids = [f"cz_{e[0]}_{e[1]}_{s}_{m}_{ax}" for m in CZ_MS for ax in "XY"]
            ms = np.array([int(i.split("_")[4]) for i in ids], dtype=float)
            unexp = np.array([(data[i].k[0] if s == 0 else data[i].shots - data[i].k[0]) for i in ids], dtype=float)
            ns = np.array([data[i].shots for i in ids], dtype=float)
            phat = unexp / ns
            slope = float(np.sum((ms - ms.mean()) * (phat - phat.mean())) / np.sum((ms - ms.mean()) ** 2))
            pval = _trend_test(unexp, ns, ms, rng)
            per[f"{e[0]}_{e[1]}_state{s}"] = dict(
                unexpected_rate_by_m={str(int(m)): float(phat[ms == m].mean()) for m in CZ_MS},
                total_unexpected=int(unexp.sum()), drift_0_to_8=float(8 * slope), trend_p=pval,
                flag=bool(pval < ALPHA_FAM / zc and abs(8 * slope) > PRACTICAL_POP))
    return dict(
        per_circuit_group=per, flag=any(v["flag"] for v in per.values()),
        threshold_justification=f"exact conditional trend test p < 0.005/{zc} (Bonferroni, suite-level alpha split over 10 families) AND |drift m=0->8| > "
                                f"{PRACTICAL_POP}. Counts are the control's unexpected outcomes (1 for state 0, 0 for state 1). "
                                "Population change on the control mixes its two phase branches; leakage out of {0,1} "
                                "shows up as drift or as non-exponential target visibility loss (see relaxation).")


def _decay(ms, vs, ses):
    pts = [(m, v, s) for m, v, s in zip(ms, vs, ses) if s is not None and v > 1e-3 and s > 0]
    if len(pts) < 3:
        return None
    m = np.array([p[0] for p in pts], dtype=float)
    y = np.log([p[1] for p in pts])
    w = 1.0 / (np.array([p[2] / p[1] for p in pts]) ** 2)
    mb = np.sum(w * m) / w.sum()
    yb = np.sum(w * y) / w.sum()
    sxx = np.sum(w * (m - mb) ** 2)
    slope = float(np.sum(w * (m - mb) * (y - yb)) / sxx)
    icpt = float(yb - slope * mb)
    chi2 = float(np.sum(w * (y - icpt - slope * m) ** 2))
    dof = len(pts) - 2
    kappa = -slope
    return dict(decay_per_repetition=float(kappa), decay_per_repetition_se=float(1 / math.sqrt(sxx)),
                apparent_decay_repetitions=float(1 / kappa) if kappa > 1e-9 else None,
                chi2=chi2, dof=dof, p=float(stats.chi2.sf(chi2, dof)), n_points=len(pts))


def diag_relaxation(L, th, se, durations_ns=None):
    per = {}
    ntest = len(L.sx_sites) + 2 * len(L.edges)
    for p in L.sx_sites:
        d = _decay(SX_MS, [th[L.sx_v[(p, m)]] for m in SX_MS], [se.get(f"sx.{p}.v{m}") for m in SX_MS])
        if d:
            if durations_ns and durations_ns.get("sx"):
                d["apparent_decay_time_ns"] = None if not d["apparent_decay_repetitions"] else d["apparent_decay_repetitions"] * durations_ns["sx"]
            d["min_visibility"] = float(min(th[L.sx_v[(p, m)]] for m in SX_MS))
            d["flag"] = bool(d["p"] < ALPHA_FAM / ntest)
            per[f"sx_{p}"] = d
    for e in L.edges:
        for s in (0, 1):
            d = _decay(CZ_MS, [th[L.cz_v[(e, s, m)]] for m in CZ_MS], [se.get(f"cz.{e[0]}_{e[1]}.v{s}_{m}") for m in CZ_MS])
            if d:
                if durations_ns and durations_ns.get("cz"):
                    d["apparent_decay_time_ns"] = None if not d["apparent_decay_repetitions"] else d["apparent_decay_repetitions"] * durations_ns["cz"]
                d["min_visibility"] = float(min(th[L.cz_v[(e, s, m)]] for m in CZ_MS))
                d["flag"] = bool(d["p"] < ALPHA_FAM / ntest)
                per[f"cz_{e[0]}_{e[1]}_state{s}"] = d
    return dict(
        per_block=per, flag=any(v["flag"] for v in per.values()),
        threshold_justification=f"weighted fit of ln(visibility) = ln v0 - kappa m with bootstrap SEs; chi-square "
                                f"lack-of-fit p < 0.005/{ntest} (Bonferroni, suite-level alpha split over 10 families) flags non-single-exponential decay. kappa is the apparent "
                                "per-repetition amplitude decay; it is not separated into T1/Tphi by this plan "
                                "(1/kappa in repetitions; pass duration_ns for a time conversion). Dof is 2 (SX) or 1 (CZ), "
                                "so power is limited.")


def _xy(L, th, K, Ns, e, s, m):
    x, sx_ = corrected(L, th, K, Ns, ("cz", e, s, m, 0))
    y, sy_ = corrected(L, th, K, Ns, ("cz", e, s, m, 1))
    return x, y, sx_, sy_


def diag_quadrature(L, th, K, Ns):
    per_cz, ztests = {}, []
    for e in L.edges:
        for s in (0, 1):
            sl = th[L.cz_phi[e]] + (1 - 2 * s) * th[L.cz_del[e]]
            res, var, amp = {}, {}, {}
            for m in CZ_MS:
                x, y, sx_, sy_ = _xy(L, th, K, Ns, e, s, m)
                r = math.hypot(x, y)
                theta = math.atan2(y, x)
                res[m] = float(wrap(theta - (th[L.cz_a[(e, s)]] + m * sl)))
                var[m] = (x * x * sy_ ** 2 + y * y * sx_ ** 2) / max(r, 1e-6) ** 4
                sr = math.sqrt((x * x * sx_ ** 2 + y * y * sy_ ** 2)) / max(r, 1e-6)
                amp[m] = dict(r=float(r), v_fit=float(th[L.cz_v[(e, s, m)]]), z=float((r - th[L.cz_v[(e, s, m)]]) / sr),
                              r2=float(r * r))
            R = res[8] - 4 * res[2] + 3 * res[0]
            zlin = R / math.sqrt(var[8] + 16 * var[2] + 9 * var[0])
            per_cz[f"{e[0]}_{e[1]}_state{s}"] = dict(phase_residual=res, nonlinearity=float(R), z_nonlinearity=float(zlin),
                                                    amplitude=amp, max_abs_amplitude_z=float(max(abs(a["z"]) for a in amp.values())),
                                                    max_radius=float(max(a["r"] for a in amp.values())))
    zc_lin = _bonf_z(len(per_cz))
    zc_amp = _bonf_z(len(per_cz) * len(CZ_MS))
    for v in per_cz.values():
        v["flag_nonlinear"] = bool(abs(v["z_nonlinearity"]) > zc_lin)
        v["flag_amplitude"] = bool(v["max_abs_amplitude_z"] > zc_amp)
        v["flag_unphysical_radius"] = bool(v["max_radius"] > 1.1)
    per_sx = {}
    for p in L.sx_sites:
        d, ph = th[L.sx_d[p]], th[L.sx_ph[p]]
        rows = {}
        for m in SX_MS:
            z, sz = corrected(L, th, K, Ns, ("sx", p, m, 0))
            y, sy = corrected(L, th, K, Ns, ("sx", p, m, 1))
            v = th[L.sx_v[(p, m)]]
            qm = (v * math.cos(m * d + ph)) ** 2 + (v * math.sin((m + 1) * d + ph)) ** 2
            sq = 2 * math.sqrt(z * z * sz * sz + y * y * sy * sy)
            rows[m] = dict(q_obs=float(z * z + y * y), q_model=float(qm), z=float((z * z + y * y - qm) / max(sq, 1e-9)))
        per_sx[f"sx_{p}"] = dict(by_m=rows, max_abs_z=float(max(abs(r["z"]) for r in rows.values())))
    zc_sx = _bonf_z(len(per_sx) * len(SX_MS))
    for v in per_sx.values():
        v["flag"] = bool(v["max_abs_z"] > zc_sx)
    return dict(
        cz=per_cz, sx=per_sx, z_crit_nonlinear=zc_lin, z_crit_amplitude=zc_amp, z_crit_sx=zc_sx,
        flag=bool(any(v["flag_nonlinear"] or v["flag_amplitude"] or v["flag_unphysical_radius"] for v in per_cz.values())
                  or any(v["flag"] for v in per_sx.values())),
        threshold_justification="CZ: model-free phase theta_m=atan2(y,x) must be linear in m: R = th8 - 4 th2 + 3 th0 "
                                "(about the fit) with delta-method SE, Bonferroni over 12 (edge,state); amplitude "
                                "r=sqrt(x^2+y^2) must match the fitted visibility (Bonferroni over 36), r<=1.1. "
                                "SX: observed z^2+y^2 versus the model value (Bonferroni over 16).")


def diag_drop_longest(L, data, fit, wald_full, se_full):
    ids = ["sx_24_Z", "sx_24_Y"] + [r["id"] for r in L.rows if r["family"] == "cz" and r["m"] == 8]
    f2 = fit_all(L, data, th_init=fit.th, exclude=ids, K=fit.K, Ns=fit.Ns)
    th2 = align_to(L, f2.th, fit.th)
    _, info2 = wald_ses(L, f2.oset, th2, fit.Ns)
    se2 = {k: v["wald_se"] for k, v in info2["per_parameter"].items()}
    out = {}
    zcrit = _bonf_z(16)
    for nm in L.names:
        if not (nm.endswith("offset_rad") or nm.endswith("alpha_target_rad") or nm.endswith("delta_rad")):
            continue
        a, b = float(fit.th[L.index[nm]]), float(th2[L.index[nm]])
        sf, sd = se_full.get(nm), se2.get(nm)
        if sf is None or sd is None:
            out[nm] = dict(full=a, dropped=b, diff=b - a, z=None, flag=True, note="SE unavailable (non-identifiable)")
            continue
        sdiff = math.sqrt(max(sd ** 2 - sf ** 2, (0.2 * sf) ** 2))
        z = (b - a) / sdiff
        out[nm] = dict(full=a, dropped=b, diff=b - a, se_full_wald=sf, se_dropped_wald=sd, z=float(z),
                       diff_over_se_full=float((b - a) / sf), flag=bool(abs(z) > zcrit))
    return dict(
        dropped_circuits=ids, per_parameter=out, flag=any(v["flag"] for v in out.values()),
        deviance_dropped=f2.cost, rank_dropped=info2["rank_numeric"], n_parameters_dropped=info2["n_parameters"],
        threshold_justification=f"nested-estimator difference z = diff / sqrt(SE_dropped^2 - SE_full^2) (Wald SEs, floor "
                                f"0.2 SE_full), |z| > Bonferroni z over 16 parameters at alpha 0.005 ({zcrit:.2f}). With m=8 (CZ) or m=24 (SX) removed the dropped fit has less "
                                "leverage, so only a large, significant shift indicates the longest repetition drives the result.")


def block_flags(L, diag):
    """Entity-level flags the decision rule consumes."""
    fl = {}
    ctx = diag["crosstalk"]["prep_phase_block_vs_isolated"]["per_pair"]
    sxc = diag["crosstalk"]["sx_block_pair_correlation"]
    for p in L.sx_sites:
        involved = any(abs(t["z"]) > sxc["z_crit_bonferroni"] and p in t["pair"] for t in sxc["flagged"])
        fl[f"sx_{p}"] = dict(
            crosstalk=bool(involved or any(v["flag"] for k, v in ctx.items() if k.startswith(f"site{p}_"))),
            control_prep=False, leakage=False,
            relaxation=bool(diag["relaxation"]["per_block"].get(f"sx_{p}", {}).get("flag", False)),
            quadrature=bool(diag["quadrature"]["sx"][f"sx_{p}"]["flag"]),
            drop_longest=bool(diag["drop_longest"]["per_parameter"][f"sx.{p}.offset_rad"]["flag"]))
    for e in L.edges:
        k = f"{e[0]}_{e[1]}"
        fl[f"cz_{k}"] = dict(
            crosstalk=bool(diag["crosstalk"]["cz_control_target_association_cmh"]["per_edge"][k]["flag"]
                           or any(v["flag"] for kk, v in ctx.items() if kk.endswith(f"_edge{k}"))),
            control_prep=bool(any(diag["crosstalk"]["control_marginal_vs_spam"]["per_circuit_group"][f"{k}_state{s}"]["flag"] for s in (0, 1))),
            leakage=bool(any(diag["leakage_population"]["per_circuit_group"][f"{k}_state{s}"]["flag"] for s in (0, 1))),
            relaxation=bool(any(diag["relaxation"]["per_block"].get(f"cz_{k}_state{s}", {}).get("flag", False) for s in (0, 1))),
            quadrature=bool(any(diag["quadrature"]["cz"][f"{k}_state{s}"][f] for s in (0, 1)
                                for f in ("flag_nonlinear", "flag_amplitude", "flag_unphysical_radius"))),
            drop_longest=bool(any(diag["drop_longest"]["per_parameter"][f"cz.{k}.{q}"]["flag"]
                                  for q in ("alpha_target_rad", "delta_rad"))))
    return fl


# ----------------------------------------------------------------------------------------------
# decision rule and comparison to the committed JS synthetic experiments
# ----------------------------------------------------------------------------------------------
def parameter_table(L):
    """(parameter name, entity key, kind, JS-synthetic family, JS key) for the 16 reported quantities."""
    rows = []
    for p in L.sx_sites:
        rows.append((f"sx.{p}.offset_rad", f"sx_{p}", "sx", "sx", str(p)))
    for e in L.edges:
        k = f"{e[0]}_{e[1]}"
        rows.append((f"cz.{k}.delta_rad", f"cz_{k}", "cz_delta", "cz", k))
        rows.append((f"cz.{k}.alpha_target_rad", f"cz_{k}", "cz_alpha", "local", k))
    return rows


def decision_rule(result, replicate=None, max_flag_fraction=0.05):
    """Apply the QC26A decision rule (report, 'Decision rule') to one identification result.

    A nonzero estimate alone is never sufficient. Required: adequate visibility, no boundary / clipping,
    multinomial-bootstrap and profile intervals excluding zero, no alias ambiguity, control-state preparation
    check, X/Y linearity, stability when the longest repetition is dropped, no crosstalk / leakage /
    relaxation flag, and a fresh independently shuffled replicate block (not available in a single dataset).
    """
    boot = result["bootstrap"]
    prof = result["identifiability"].get("profiles", {})
    flags = result["flags_estimate"]
    bflag = boot.get("flag_fractions", {})
    bl = result["block_flags"]
    alias = result["identifiability"].get("alias", {})
    out = {}
    for name, ent, kind, _, _ in result["parameter_table"]:
        c = {}
        bp = boot.get("parameters", {}).get(name)
        c["bootstrap_available"] = bp is not None
        c["bootstrap_ci_excludes_zero"] = bool(bp and bp["ci_excludes_zero"])
        pf = prof.get(name)
        c["profile_ci_excludes_zero"] = bool(pf and pf["excludes_zero"])
        c["profile_available"] = pf is not None
        c["profile_not_flat_not_multimodal"] = bool(pf is not None and not pf["flat"] and not pf["multimodal"]
                                                     and not pf["interval_open_low"] and not pf["interval_open_high"])
        fl = flags[ent]
        bf = bflag.get(ent, {})
        c["visibility_adequate"] = bool(not fl["low_visibility"] and bf.get("low_visibility", 1.0) <= max_flag_fraction)
        bnd = "boundary" if ent.startswith("sx") else "large_phase"
        c["no_boundary_or_clipping"] = bool(not fl[bnd] and not fl["clipped_low"]
                                            and bf.get(bnd, 1.0) <= max_flag_fraction
                                            and bf.get("clipped_low", 1.0) <= max_flag_fraction
                                            and not (ent.startswith("sx") and (fl["phase_boundary"] or bf.get("phase_boundary", 1.0) > max_flag_fraction)))
        if ent.startswith("cz"):
            a = alias.get("cz", {}).get(ent[3:])
            c["no_alias_ambiguity"] = bool(a is not None and not a["alias_ambiguous"] and bf.get("alias_boundary", 1.0) <= max_flag_fraction)
        else:
            a = alias.get("sx", {}).get(ent[3:])
            c["no_alias_ambiguity"] = bool(a is not None and not a["ambiguous"] and not a["global_minimum_outside_bound"])
        b = bl[ent]
        c["control_state_prep_ok"] = not b["control_prep"]
        c["xy_linearity_ok"] = not b["quadrature"]
        c["drop_longest_stable"] = not b["drop_longest"]
        c["no_crosstalk_flag"] = not b["crosstalk"]
        c["no_leakage_flag"] = not b["leakage"]
        c["no_relaxation_flag"] = not b["relaxation"]
        gates = ["bootstrap_available", "profile_available", "profile_not_flat_not_multimodal", "visibility_adequate",
                 "no_boundary_or_clipping", "no_alias_ambiguity", "control_state_prep_ok", "xy_linearity_ok",
                 "drop_longest_stable", "no_crosstalk_flag", "no_leakage_flag", "no_relaxation_flag"]
        failed = [g for g in gates if not c[g]]
        nonzero = c["bootstrap_ci_excludes_zero"] and c["profile_ci_excludes_zero"]
        rep = None
        if replicate is not None:
            rb = replicate["bootstrap"]["parameters"].get(name)
            if rb and bp:
                d = rb["estimate"] - bp["estimate"]
                rep = bool(np.sign(rb["estimate"]) == np.sign(bp["estimate"]) and rb["ci_excludes_zero"]
                           and abs(d) <= 2 * math.sqrt(rb["se"] ** 2 + bp["se"] ** 2))
        if failed:
            status = "INCONCLUSIVE_FLAGS"
        elif nonzero and rep:
            status = "NONZERO_REPLICATED"
        elif nonzero:
            status = "NONZERO_CANDIDATE_PENDING_REPLICATION"
        elif c["bootstrap_ci_excludes_zero"] or c["profile_ci_excludes_zero"]:
            status = "BORDERLINE_INTERVALS_DISAGREE"
        else:
            status = "CONSISTENT_WITH_ZERO"
        out[name] = dict(criteria=c, failed_gates=failed, nonzero_both_intervals=bool(nonzero),
                         replicated=rep, status=status)
    counts = {}
    for v in out.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    return dict(
        per_parameter=out, status_counts=counts,
        replicate_block_required=replicate is None,
        replicate_block_note="single dataset: the report requires a fresh independently shuffled repetition block "
                             "before any nonzero value is accepted, so NONZERO_REPLICATED is unreachable here",
        hardware_claim="none; offline statistical identification of a counts file only",
        multiplicity_note="16 parameters are tested; intervals are marginal 95% intervals without multiplicity correction")


def _arm_stats(vals, truth, est):
    vals = np.asarray(vals, dtype=float)
    sd = float(vals.std(ddof=1))
    return dict(n=int(len(vals)), truth=truth, mean=float(vals.mean()), sd=sd,
                rmse_about_truth=float(np.sqrt(np.mean((vals - truth) ** 2))),
                z_score=float((est - vals.mean()) / sd) if sd > 0 else None,
                percentile=float(100 * (np.mean(vals < est) + 0.5 * np.mean(vals == est))))


def compare_to_synthetic(result, synthetic_json):
    """Where the estimates fall in the committed JS-synthetic injected/null arms (qc26a_synthetic.json)."""
    if isinstance(synthetic_json, (str, Path)):
        with open(synthetic_json, "r", encoding="utf-8") as f:
            synthetic_json = json.load(f)
    syn = synthetic_json
    inj = [t for t in syn["trials"] if not t["zero"]]
    nul = [t for t in syn["trials"] if t["zero"]]
    per = {}
    for name, ent, kind, fam, key in result["parameter_table"]:
        est = result["fit_parameters"][name]
        js_est = result["js_mirror_estimates"].get(name)
        truth_i = syn["truth"][fam][key]
        truth_n = truth_i if fam == "local" else 0.0
        entry = dict(estimate=est, js_estimator_estimate=js_est, family=fam)
        for arm, trials, tr in (("injected_arm", inj, truth_i), ("null_arm", nul, truth_n)):
            vals = [t[fam][key] for t in trials]
            entry[arm] = _arm_stats(vals, tr, est)
            entry[arm]["js_estimator_z_score"] = (None if js_est is None or entry[arm]["sd"] == 0
                                                  else float((js_est - entry[arm]["mean"]) / entry[arm]["sd"]))
        zi, zn = entry["injected_arm"]["z_score"], entry["null_arm"]["z_score"]
        entry["closer_arm"] = "injected" if abs(zi) < abs(zn) else "null"
        entry["inside_central_95_of_arm"] = dict(
            injected=bool(2.5 <= entry["injected_arm"]["percentile"] <= 97.5),
            null=bool(2.5 <= entry["null_arm"]["percentile"] <= 97.5))
        per[name] = entry
    return dict(per_parameter=per, decision=decision_rule(result),
                label="comparison against SYNTHETIC JS trials only; no hardware inference")


# ----------------------------------------------------------------------------------------------
# top level
# ----------------------------------------------------------------------------------------------
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(o) else float(o)
    return o


def summarize_fit(L, fit):
    th = fit.th
    p = fit.oset.p1(th)
    mu = fit.Ns * p
    pearson = float(np.sum((fit.K - mu) ** 2 / np.maximum(mu * (1 - p), 1e-9)))
    dof = int(fit.oset.n - len(fit.oset.free))
    spam = {str(p_): dict(e0=th[L.e0[p_]], e1_effective=th[L.e1[p_]], contrast=1 - th[L.e0[p_]] - th[L.e1[p_]],
                          bias=th[L.e1[p_]] - th[L.e0[p_]],
                          low_contrast=bool(1 - th[L.e0[p_]] - th[L.e1[p_]] < 0.5)) for p_ in L.all_sites}
    sx = {str(p_): dict(offset_rad=th[L.sx_d[p_]], phase_rad=th[L.sx_ph[p_]],
                        visibility={str(m): th[L.sx_v[(p_, m)]] for m in SX_MS}) for p_ in L.sx_sites}
    cz = {}
    for e in L.edges:
        phi, dl = th[L.cz_phi[e]], th[L.cz_del[e]]
        cz[f"{e[0]}_{e[1]}"] = dict(
            alpha_target_rad=phi, delta_rad=dl, slope_state0_rad=phi + dl, slope_state1_rad=phi - dl,
            intercept0_rad=th[L.cz_a[(e, 0)]], intercept1_rad=th[L.cz_a[(e, 1)]],
            visibility={f"state{s}_m{m}": th[L.cz_v[(e, s, m)]] for s in (0, 1) for m in CZ_MS})
    return dict(deviance=fit.cost, pearson_chi2=pearson, dof=dof, deviance_p_value=float(stats.chi2.sf(fit.cost, max(dof, 1))),
                gof_caveat="dof = observations - free parameters; several parameters (visibility near 1, rare SPAM counts) "
                           "make the chi-square reference approximate",
                spam=spam, sx=sx, cz=cz)


def hamiltonian_block(L, th, bparams):
    out = {}
    for e in L.edges:
        k = f"{e[0]}_{e[1]}"
        phi, dl = th[L.cz_phi[e]], th[L.cz_del[e]]
        b1, b2 = bparams.get(f"cz.{k}.alpha_target_rad"), bparams.get(f"cz.{k}.delta_rad")
        out[k] = dict(
            convention="H = 1/2 (alpha_c Z_c + alpha_t Z_t + delta Z_c Z_t) per CZ application; residual after the "
                       "ideal CZ, even m",
            alpha_target_rad=dict(estimate=phi, bootstrap_se=b1 and b1["se"], ci95=b1 and b1["ci95"]),
            delta_rad=dict(estimate=dl, bootstrap_se=b2 and b2["se"], ci95=b2 and b2["ci95"]),
            alpha_control_rad=dict(
                identifiable=False,
                reason="the control is prepared in a computational-basis state and only its Z marginal is measured; "
                       "Z_c commutes with H and with the readout, so alpha_c only multiplies the state by a global "
                       "phase. Its X/Y coherence (which alone would reveal it) is not measured in this plan.",
                max_abs_prediction_sensitivity=alpha_control_sensitivity(phi, dl)),
            alias=dict(equivalent_decompositions=lattice_candidates(phi, dl),
                       rule="(alpha_t, delta) -> (alpha_t + j pi/2, delta + l pi/2), j = l mod 2, exactly "
                            "indistinguishable from m in {0,2,8} (derivation in the module docstring)"))
    return out


def identify(counts, plan=None, bootstraps=400, seed=1, profile=True, n_local=25, n_coarse=41, alias=True,
             synthetic=None, durations_ns=None, progress=False, diag_seed=20260101):
    t0 = time.time()
    plan = plan or load_plan()
    L = Layout(plan)
    data = parse_counts(plan, counts)
    fit = fit_all(L, data)
    K, Ns = fit.K, fit.Ns
    wald, info = wald_ses(L, L.full, fit.th, Ns)
    js = js_mirror(plan, data)
    boot = run_bootstrap(L, data, fit.th, bootstraps, seed, progress) if bootstraps > 0 else None
    bsum = summarize_bootstrap(L, fit.th, boot) if boot else dict(n_replicates=0, parameters={}, flag_fractions={},
                                                                  flag_fractions_any_entity={}, failures=0)
    se = {k: v["se"] for k, v in bsum["parameters"].items()} if boot else {}
    for k, v in wald.items():
        se.setdefault(k, v)
    table = parameter_table(L)
    ident = dict(jacobian=info, wald_se=wald)
    if profile:
        profs = {}
        for name, ent, kind, fam, key in table:
            if kind == "sx":
                profs[name] = profile_parameter(L, "sx", int(ent[3:]), name, fit.th, K, Ns, se.get(name), n_local, n_coarse)
            else:
                e = tuple(int(x) for x in ent[3:].split("_"))
                profs[name] = profile_parameter(L, "cz", e, name, fit.th, K, Ns, se.get(name), n_local, n_coarse)
        ident["profiles"] = profs
    if alias:
        ident["alias"] = dict(cz=cz_alias_scan(L, fit.th, K, Ns), sx=sx_alias_scan(L, fit.th, K, Ns),
                              cz_alias_statement="slope per control state is determined mod pi (m=2 binds; m=8 alone "
                                                 "would give pi/4 but is resolved by m=2 only if the m=2 phase noise is "
                                                 "well below pi/8); (alpha_t, delta) alias lattice generated by "
                                                 "(pi/2, pi/2) and (pi/2, -pi/2)",
                              sx_alias_statement="all m even: exact reflection alias (d, phi0) -> (pi-d, -phi0), outside the +-0.1 "
                                                 "design bound; no translation alias (Y axis carries (m+1)d with m+1=1)")
    rng = np.random.default_rng(diag_seed)
    diag = dict(crosstalk=diag_crosstalk(L, data, fit.th, se, rng), leakage_population=diag_leakage(L, data, fit.th, rng),
                relaxation=diag_relaxation(L, fit.th, se, durations_ns), quadrature=diag_quadrature(L, fit.th, K, Ns),
                drop_longest=diag_drop_longest(L, data, fit, wald, wald))
    diag["any_flag"] = bool(any(diag[k]["flag"] for k in ("crosstalk", "leakage_population", "relaxation", "quadrature",
                                                         "drop_longest")))
    bflags = block_flags(L, diag)
    fitp = {name: float(fit.th[L.index[name]]) for name, *_ in table}
    jsp = {}
    for name, ent, kind, fam, key in table:
        if fam == "sx":
            jsp[name] = js["sx"][int(key)]["offset_rad"]
        elif fam == "cz":
            jsp[name] = js["cz"][key]["zz_phase_rad"]
        else:
            jsp[name] = js["cz"][key]["local_phase_rad"]
    result = dict(
        label="OFFLINE STATISTICAL IDENTIFICATION of a counts file; no IBM access, no hardware claim",
        version=VERSION,
        inputs=dict(counts_sha256=canonical_sha256(counts), plan_sha256=canonical_sha256(plan), bootstraps=bootstraps,
                    bootstrap_seed=seed, numpy=np.__version__, scipy=__import__("scipy").__version__),
        model=dict(
            description="joint binomial (product-form multinomial) likelihood, deviance-residual Gauss-Newton",
            spam_note="e0, e1 are EFFECTIVE parameters: the X preparation error in spam_1 (and in control-state-1 "
                      "preparation) is absorbed into e1; they are not independent detector errors",
            hamiltonian_identifiability="alpha_target and delta identifiable (up to the pi/2 lattice); alpha_control NOT "
                                        "identifiable by design; SPAM contrast and visibility share a weak direction",
            sx_bounds=dict(offset=SX_D_BOUND, phase=SX_PH_BOUND, visibility_max=V_MAX),
            limitations=[
                "Per-site product (independent-readout) likelihood; correlated readout error and crosstalk are only "
                "tested through the diagnostics, not modelled.",
                "SX offset and nuisance phase are bounded as in the JS design (+-0.1, +-0.2 rad); a true offset outside "
                "the bound is reported at the bound and flagged, not estimated.",
                "Visibility and SPAM contrast share a weak likelihood direction (SPAM circuits pin e0, e1 to about +-0.003 "
                "with 384 shots); parameters of interest are not affected at the reported precision.",
                "Profile likelihood is block-conditional: the block's own SPAM parameters are re-optimised, other blocks' "
                "weak information about shared SPAM parameters is ignored (slightly conservative).",
                "alpha_control is not identifiable by design; (alpha_target, delta) are identified only modulo the "
                "(pi/2, pi/2) lattice, and the reported representative is the smallest-norm one.",
                "Diagnostic thresholds are Bonferroni-calibrated under the Bernoulli model; their false-alarm rates were "
                "measured on synthetic data only (receipts) and real hardware noise may differ.",
                "Single dataset: acceptance of any nonzero value additionally requires a fresh, independently shuffled "
                "replicate block (decision rule), which this tool cannot supply."]),
        fit=summarize_fit(L, fit),
        fit_parameters=fitp, js_mirror_estimates=jsp,
        js_mirror=dict(estimates=js, joint_minus_js={n: float(fitp[n] - jsp[n]) for n in fitp}),
        hamiltonian=hamiltonian_block(L, fit.th, bsum["parameters"]),
        bootstrap=bsum, identifiability=ident, diagnostics=diag, block_flags=bflags,
        flags_estimate=flag_vector(L, fit.th), parameter_table=[list(r) for r in table],
        seconds=None)
    result["decision"] = decision_rule(result)
    if synthetic is None and SYN_PATH.exists():
        synthetic = SYN_PATH
    if synthetic:
        result["comparison_to_js_synthetic"] = compare_to_synthetic(result, synthetic)
    result["seconds"] = time.time() - t0
    return jsonable(result)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("counts")
    ap.add_argument("out")
    ap.add_argument("--bootstraps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--no-alias", action="store_true")
    ap.add_argument("--synthetic", default=None, help="qc26a_synthetic.json for the comparison block")
    ap.add_argument("--sx-duration-ns", type=float, default=None)
    ap.add_argument("--cz-duration-ns", type=float, default=None)
    a = ap.parse_args(argv)
    with open(a.counts, "r", encoding="utf-8") as f:
        counts = json.load(f)
    res = identify(counts, bootstraps=a.bootstraps, seed=a.seed, profile=not a.no_profile, alias=not a.no_alias,
                   synthetic=a.synthetic, durations_ns=dict(sx=a.sx_duration_ns, cz=a.cz_duration_ns), progress=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    dec = res["decision"]["status_counts"]
    print(json.dumps(dict(out=a.out, seconds=round(res["seconds"], 1), decision_status_counts=dec,
                          any_diagnostic_flag=res["diagnostics"]["any_flag"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
