#!/usr/bin/env python3
"""PGC Phase 6 (offline): sensitivity-matrix study J_ij = d<O_i>/d theta_j and experimental-design comparison.

OFFLINE LINEARISED SIMULATION.  No hardware data, no IBM/QPU service, no refit of historical counts.

Candidate pool (140 circuits, 196 observables, 14 parameters):
  (a) the 8 kicked-Ising circuits of receipt.json (8 observables each, 2048 shots; observables of one circuit
      share the circuit and are paid for once);
  (b) the 132 Ramsey circuits of ramsey_campaign.json (128 cz probes + 4 SPAM references, 384 shots) with the
      analytic probe model <X>=cos(phi), <Y>=sin(phi), phi = m (alpha_probe + (+1 if state==0 else -1) delta
      + state pi), differentiated by central finite differences in the 14-parameter space at the receipt
      parameter values (SX columns have zero Ramsey sensitivity by construction).
Cost per circuit = shots*(0.00025 + max(1e-5, duration)) + 0.1 s; duration is not available offline, so the
1e-5 s floor applies (0.63248 s per kicked-Ising circuit, 0.19984 s per Ramsey circuit).

Strategies: random, fixed_first, fixed_uniform, dopt (cost-blind greedy log det), dopt_cost (greedy log det
per QPU second), weak_cost (greedy weak-subspace trace reduction per second), bpd (group-adaptive greedy).

  python bpd_study.py            # full study, writes bpd_results.json and BPD_REPORT.md
  python bpd_study.py --check    # reduced rerun with assertions
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PGC = HERE.parent
RECEIPT = PGC / "receipt.json"
CAMPAIGN = PGC / "ramsey_campaign.json"

NAMED_WEAK = ["ZZ_22_23", "ZZ_4_5", "Z_22_23_23", "Z_4_5_5"]
KICKED_SHOTS = 2048
REP_DELAY = 0.00025
DURATION_FLOOR = 1e-5
OVERHEAD = 0.1
VAR_FLOOR = 0.05          # floor on (1 - y0^2) in the per-observable variance model
TRUTH_SD = 0.05           # rad, per-parameter spread of the simulated truth
PRIOR_SD = 0.10           # rad, prior used in both the design objective and the estimator
WEAK_RANK = 4
RECOVERY_TOL = 0.025      # rad, half of TRUTH_SD, per weak coordinate (L-infinity)
STRATEGIES = ["random", "fixed_first", "fixed_uniform", "dopt", "dopt_cost", "weak_cost", "bpd", "bpd_ld"]

DEFAULT_CFG = {
    "seed": 20261126,
    "splits": 10,
    "trials_per_split": 40,
    "holdout_fraction": 0.25,
    "circuit_budgets": [8, 16, 32],
    "seconds_budgets": [3.2, 6.4, 12.8],
    "noise": "binomial",
    "kicked_duration_s": 0.0,
    "nmf_max_groups": 12,
    "nmf_target_rel_error": 0.10,
}


# ------------------------------------------------------------------------------------------------ pool
def circuit_cost(shots, duration=0.0):
    return shots * (REP_DELAY + max(DURATION_FLOOR, duration)) + OVERHEAD


def ramsey_model(theta, rows_meta):
    """Probe expectation for every Ramsey observable at parameter vector theta."""
    th = np.atleast_2d(theta)
    phi = rows_meta["m"][None, :] * (th[:, rows_meta["ip"]] + rows_meta["sgn"][None, :] * th[:, rows_meta["iz"]]
                                     + rows_meta["state"][None, :] * math.pi)
    out = np.where(rows_meta["is_x"][None, :], np.cos(phi), np.sin(phi))
    out[:, rows_meta["is_spam"]] = 0.0
    return out[0] if np.ndim(theta) == 1 else out


def build_pool(kicked_duration=0.0):
    receipt = json.loads(RECEIPT.read_text())
    res = receipt["result"]
    labels = [p["label"] for p in res["parameters"]]
    theta0 = np.array([p["value"] for p in res["parameters"]], float)
    lab = {l: i for i, l in enumerate(labels)}
    camp = json.loads(CAMPAIGN.read_text())

    J_rows, y0, shots, measured = [], [], [], []
    obs_circ, obs_site, obs_kind = [], [], []
    circuits = []

    # (a) kicked-Ising observables grouped by circuit id (shared circuit => one cost)
    order, by_id = [], {}
    for r in res["rows"]:
        if r["id"] not in by_id:
            by_id[r["id"]] = []
            order.append(r["id"])
        by_id[r["id"]].append(r)
    for cid in order:
        ci = len(circuits)
        idxs = []
        for r in by_id[cid]:
            idxs.append(len(J_rows))
            J_rows.append(np.array(r["jacobian"], float))
            y0.append(float(r["prediction"]))
            measured.append(float(r["measured"]))
            shots.append(KICKED_SHOTS)
            obs_circ.append(ci)
            obs_site.append(int(r["physical"]))
            obs_kind.append("kicked")
        circuits.append({"id": cid, "family": "kicked", "shots": KICKED_SHOTS,
                         "cost": circuit_cost(KICKED_SHOTS, kicked_duration), "obs": idxs, "depth": by_id[cid][0]["depth"]})

    # (b) Ramsey circuits in campaign order (q23 block then q5 block)
    meta = {"m": [], "ip": [], "iz": [], "sgn": [], "state": [], "is_x": [], "is_spam": []}
    r_obs_start = len(J_rows)
    for blk in camp["blocks"]:
        for r in blk["rows"]:
            ci = len(circuits)
            k = len(J_rows)
            if r["family"] == "cz":
                meta["m"].append(r["m"])
                meta["ip"].append(lab[f"Z_{r['edge']}_{r['path'][1]}"])
                meta["iz"].append(lab[f"ZZ_{r['edge']}"])
                meta["sgn"].append(1 if r["state"] == 0 else -1)
                meta["state"].append(r["state"])
                meta["is_x"].append(r["axis"] == "X")
                meta["is_spam"].append(False)
                fam = "cz"
            else:
                for key, v in (("m", 0), ("ip", 0), ("iz", 0), ("sgn", 1), ("state", r["state"]), ("is_x", True), ("is_spam", True)):
                    meta[key].append(v)
                fam = "spam"
            J_rows.append(None)
            y0.append(0.0)
            measured.append(float("nan"))
            shots.append(r["shots"])
            obs_circ.append(ci)
            obs_site.append(-1)
            obs_kind.append("ramsey" if fam == "cz" else "spam")
            circuits.append({"id": r["id"], "family": fam, "shots": r["shots"], "cost": circuit_cost(r["shots"]),
                             "obs": [k], "edge": r.get("edge"), "orientation": r.get("orientation"),
                             "state": r["state"], "m": r.get("m"), "axis": r.get("axis"), "block": blk["id"]})
    rm = {k: np.array(v) for k, v in meta.items()}
    rm["m"] = rm["m"].astype(float)
    rm["state"] = rm["state"].astype(float)
    rm["sgn"] = rm["sgn"].astype(float)
    n_r = len(rm["m"])

    # central finite differences of the analytic Ramsey model over all 14 parameters
    h = 1e-5
    JR = np.zeros((n_r, len(labels)))
    for j in range(len(labels)):
        tp, tm = theta0.copy(), theta0.copy()
        tp[j] += h
        tm[j] -= h
        JR[:, j] = (ramsey_model(tp, rm) - ramsey_model(tm, rm)) / (2 * h)
    y_r = ramsey_model(theta0, rm)
    for i in range(n_r):
        J_rows[r_obs_start + i] = JR[i]
        y0[r_obs_start + i] = float(y_r[i]) if not rm["is_spam"][i] else float(1 - 2 * rm["state"][i])

    J = np.array(J_rows)
    y0 = np.array(y0)
    shots = np.array(shots, float)
    var = np.maximum(1.0 - y0 ** 2, VAR_FLOOR) / shots
    Jw = J / np.sqrt(var)[:, None]
    F = np.zeros((len(circuits), len(labels), len(labels)))
    for ci, c in enumerate(circuits):
        Jc = Jw[c["obs"]]
        F[ci] = Jc.T @ Jc
    return {
        "labels": labels, "theta0": theta0, "J": J, "y0": y0, "shots": shots, "var": var, "Jw": Jw,
        "measured": np.array(measured), "obs_circ": np.array(obs_circ), "obs_kind": np.array(obs_kind),
        "circuits": circuits, "F": F, "cost": np.array([c["cost"] for c in circuits]),
        "n_kicked_obs": r_obs_start, "ramsey_meta": rm, "n_ramsey_obs": n_r,
    }


# ------------------------------------------------------------------------------------------------ BPD grouping
def _nndsvd(R, G):
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    n, p = R.shape
    W = np.zeros((n, G))
    H = np.zeros((G, p))
    W[:, 0] = math.sqrt(S[0]) * np.abs(U[:, 0])
    H[0] = math.sqrt(S[0]) * np.abs(Vt[0])
    for j in range(1, G):
        x, y = U[:, j], Vt[j]
        xp, xn = np.maximum(x, 0), np.maximum(-x, 0)
        yp, yn = np.maximum(y, 0), np.maximum(-y, 0)
        mp = np.linalg.norm(xp) * np.linalg.norm(yp)
        mn = np.linalg.norm(xn) * np.linalg.norm(yn)
        if mp >= mn:
            u, v, m = xp / (np.linalg.norm(xp) + 1e-30), yp / (np.linalg.norm(yp) + 1e-30), mp
        else:
            u, v, m = xn / (np.linalg.norm(xn) + 1e-30), yn / (np.linalg.norm(yn) + 1e-30), mn
        s = math.sqrt(S[j] * m)
        W[:, j] = s * u
        H[j] = s * v
    avg = R.mean()
    W[W < 1e-10] = 0.01 * avg
    H[H < 1e-10] = 0.01 * avg
    return W, H


def nmf(R, G, iters=1500):
    """Deterministic NMF (NNDSVD start, multiplicative updates). Returns W (n x G), H (G x p), relative error."""
    W, H = _nndsvd(R, G)
    eps = 1e-12
    for _ in range(iters):
        H *= (W.T @ R) / (W.T @ W @ H + eps)
        W *= (R @ H.T) / (W @ H @ H.T + eps)
    err = np.linalg.norm(R - W @ H) / np.linalg.norm(R)
    return W, H, float(err)


def fit_groups(pool, obs_idx, max_groups=12, target=0.10):
    """BPD grouping: NMF of row-normalised |J| over the given observables (zero rows are excluded)."""
    Jw = pool["Jw"][obs_idx]
    norms = np.linalg.norm(Jw, axis=1)
    keep = norms > 1e-8
    R = np.abs(Jw[keep]) / norms[keep][:, None]
    sweep = {}
    best_G = None
    for G in range(2, max_groups + 1):
        _, _, e = nmf(R, G, iters=400)
        sweep[G] = e
        if best_G is None and e <= target:
            best_G = G
    if best_G is None:
        best_G = max_groups
    W, H, err = nmf(R, best_G)
    dom = np.argmax(W, axis=1)
    used = [g for g in range(best_G) if (dom == g).any()]
    W, H, best_G = W[:, used], H[used], len(used)       # drop groups that dominate no observable
    Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
    # order groups by (first dominant parameter index, then weight) for stable naming
    order = sorted(range(best_G), key=lambda g: (int(np.argmax(Hn[g])), -float(Hn[g].max())))
    W, Hn = W[:, order], Hn[order]
    Wfull = np.zeros((len(obs_idx), best_G))
    Wfull[keep] = W / (W.sum(axis=1, keepdims=True) + 1e-30)
    return {"H": Hn, "W": Wfull, "G": best_G, "rel_error": err, "sweep": sweep, "obs_idx": np.array(obs_idx)}


def circuit_groups(pool, grp, cand):
    """Assign each candidate circuit to the group with largest whitened-norm-weighted loading (-1 if no sensitivity)."""
    pos = {o: i for i, o in enumerate(grp["obs_idx"])}
    out = {}
    for c in cand:
        score = np.zeros(grp["G"])
        for o in pool["circuits"][c]["obs"]:
            if o in pos:
                score += np.linalg.norm(pool["Jw"][o]) * grp["W"][pos[o]]
        out[c] = int(np.argmax(score)) if score.sum() > 1e-12 else -1
    return out


def weak_subspace(pool, r=WEAK_RANK, include_sx=False):
    """Right singular vectors with the smallest singular values of the noise-whitened full-pool Jacobian.
    Primary: the 12 ZZ/Z columns (the two SX columns are weakest in the whole pool, are measured by separate QC26A
    SX circuits, and cannot reach the recovery tolerance with any design); include_sx=True is the literal 14-column
    version.  Vectors are returned as 14-vectors."""
    cols = list(range(14)) if include_sx else list(range(2, 14))
    U, S, Vt = np.linalg.svd(pool["Jw"][:, cols], full_matrices=False)
    Vw = np.zeros((14, r))
    Vw[cols, :] = Vt[-r:].T
    return Vw, S


# ------------------------------------------------------------------------------------------------ selection
def budget_ok(n_sel, cost_sel, c_cost, budget):
    if budget["mode"] == "circuits":
        return n_sel < budget["value"]
    return cost_sel + c_cost <= budget["value"] + 1e-12


def _logdet(A):
    return np.linalg.slogdet(A)[1]


def greedy_select(pool, cand, budget, P0, score_fn, stop_tol=1e-12):
    A = P0.copy()
    sel, cost = [], 0.0
    avail = list(cand)
    while True:
        feas = [c for c in avail if budget_ok(len(sel), cost, pool["cost"][c], budget)]
        if not feas:
            break
        scores = score_fn(A, feas)
        j = int(np.argmax(scores))
        if scores[j] <= stop_tol:
            break
        c = feas[j]
        sel.append(c)
        cost += pool["cost"][c]
        A = A + pool["F"][c]
        avail.remove(c)
    return sel


def score_dopt(pool):
    def f(A, feas):
        base = _logdet(A)
        return np.array([_logdet(A + pool["F"][c]) - base for c in feas])
    return f


def score_dopt_cost(pool):
    def f(A, feas):
        base = _logdet(A)
        return np.array([(_logdet(A + pool["F"][c]) - base) / pool["cost"][c] for c in feas])
    return f


def weak_trace_gain(A, Fc, Vw):
    S0 = np.linalg.inv(A)
    S1 = np.linalg.inv(A + Fc)
    return float(np.trace(Vw.T @ (S0 - S1) @ Vw))


def score_weak_cost(pool, Vw):
    def f(A, feas):
        S0 = np.linalg.inv(A)
        t0 = np.trace(Vw.T @ S0 @ Vw)
        return np.array([(t0 - np.trace(Vw.T @ np.linalg.inv(A + pool["F"][c]) @ Vw)) / pool["cost"][c] for c in feas])
    return f


def bpd_select(pool, cand, budget, P0, Vw, grp, cgroup, objective="weak"):
    """Group-adaptive greedy.  At each step: rank groups by the residual posterior variance of their parameter
    pattern inside the weak subspace, u_g = h_g' (Pw S Pw) h_g; visit the highest-u group that still holds a
    feasible circuit with positive weak-trace reduction and take the circuit of that group maximising the weak-trace
    reduction per estimated QPU second.  If no group offers a weak-subspace gain, fall back to log det per second.
    objective="logdet" keeps the group ordering but scores circuits inside a group by log det gain per second."""
    Pw = Vw @ Vw.T
    A = P0.copy()
    sel, cost = [], 0.0
    avail = list(cand)
    H = grp["H"]
    while True:
        feas = [c for c in avail if budget_ok(len(sel), cost, pool["cost"][c], budget)]
        if not feas:
            break
        S0 = np.linalg.inv(A)
        t0 = np.trace(Vw.T @ S0 @ Vw)
        if objective == "weak":
            gain = {c: (t0 - np.trace(Vw.T @ np.linalg.inv(A + pool["F"][c]) @ Vw)) / pool["cost"][c] for c in feas}
        else:
            base0 = _logdet(A)
            gain = {c: (_logdet(A + pool["F"][c]) - base0) / pool["cost"][c] for c in feas}
        Sw = Pw @ S0 @ Pw
        u = np.array([float(h @ Sw @ h) for h in H])
        pick = None
        for g in np.argsort(-u, kind="stable"):
            members = [c for c in feas if cgroup.get(c, -1) == g and gain[c] > 1e-12]
            if members:
                pick = max(members, key=lambda c: (gain[c], -c))
                break
        if pick is None:
            base = _logdet(A)
            alt = [(_logdet(A + pool["F"][c]) - base) / pool["cost"][c] for c in feas]
            j = int(np.argmax(alt))
            if alt[j] <= 1e-12:
                break
            pick = feas[j]
        sel.append(pick)
        cost += pool["cost"][pick]
        A = A + pool["F"][pick]
        avail.remove(pick)
    return sel


def fixed_orders(pool):
    """Pool order (kicked circuits then Ramsey rows in campaign order) and the hand-fixed uniform Ramsey order."""
    circ = pool["circuits"]
    first = list(range(len(circ)))
    cz = [i for i, c in enumerate(circ) if c["family"] == "cz"]
    key = {(c["edge"], c["orientation"], c["state"], c["m"], c["axis"]): i for i, c in zip(cz, [circ[i] for i in cz])}
    eo = [(e, o) for e in ["16_23", "22_23", "4_5", "5_6"] for o in ["forward", "reverse"]]
    combos = [(0, "X"), (0, "Y"), (1, "X"), (1, "Y")]
    uni = []
    for m in (1, 2, 8, 0):
        for block in range(4):
            for k, (e, o) in enumerate(eo):
                st, ax = combos[(k + block) % 4]
                uni.append(key[(e, o, st, m, ax)])
    assert len(uni) == len(set(uni)) == len(cz)
    return first, uni


def fill_in_order(pool, order, design_set, budget):
    sel, cost = [], 0.0
    for c in order:
        if c not in design_set:
            continue
        if budget_ok(len(sel), cost, pool["cost"][c], budget):
            sel.append(c)
            cost += pool["cost"][c]
        elif budget["mode"] == "circuits":
            break
    return sel


# ------------------------------------------------------------------------------------------------ evaluation
def prior_precision():
    return np.eye(14) / PRIOR_SD ** 2


def design_metrics(pool, sel, Vw, P0):
    A = P0 + (sum(pool["F"][c] for c in sel) if sel else 0.0)
    S = np.linalg.inv(A)
    S0 = np.linalg.inv(P0)
    ig = 0.5 * (_logdet(A) - _logdet(P0))
    W0 = Vw.T @ S0 @ Vw
    W1 = Vw.T @ S @ Vw
    weak_ig = 0.5 * (_logdet(W0) - _logdet(W1))
    return {
        "n_circuits": len(sel),
        "cost_s": float(sum(pool["cost"][c] for c in sel)),
        "shots": float(sum(pool["circuits"][c]["shots"] for c in sel)),
        "logdet_fim_ridge": float(_logdet(A)),
        "info_gain_nats": float(ig),
        "weak_info_gain_nats": float(weak_ig),
        "weak_trace_post": float(np.trace(W1)),
        "n_kicked": int(sum(pool["circuits"][c]["family"] == "kicked" for c in sel)),
    }, A, S


def make_trials(pool, rng, T, noise, ramsey_truth="linear"):
    delta = rng.normal(0.0, TRUTH_SD, size=(T, 14))
    y_true = pool["y0"][None, :] + delta @ pool["J"].T
    if ramsey_truth == "exact":
        n_k = pool["n_kicked_obs"]
        ex = ramsey_model(pool["theta0"][None, :] + delta, pool["ramsey_meta"])
        is_sp = pool["ramsey_meta"]["is_spam"]
        y_true[:, n_k:] = np.where(is_sp[None, :], pool["y0"][None, n_k:], ex)
    n = pool["shots"][None, :]
    if noise == "binomial":
        p = np.clip((1 + y_true) / 2, 0.0, 1.0)
        k = rng.binomial(n.astype(int).repeat(T, 0), p)
        y_obs = 2 * k / n - 1
    elif noise == "gaussian":
        y_obs = y_true + rng.normal(size=y_true.shape) * np.sqrt(pool["var"])[None, :]
    else:
        raise ValueError(noise)
    return delta, y_obs, y_true - pool["y0"][None, :]


def evaluate_trials(pool, sel, A, S, delta, y_obs, shift, tidx, Vw, Vw14, named_idx, held_obs):
    """Linear Gauss-Newton posterior mean for the trials in tidx; returns per-trial arrays."""
    if sel:
        rows = np.concatenate([pool["circuits"][c]["obs"] for c in sel]).astype(int)
        Jr = pool["J"][rows]
        w = 1.0 / pool["var"][rows]
        b = (Jr * w[:, None]).T @ (y_obs[np.ix_(tidx, rows)] - pool["y0"][rows][None, :]).T
        est = (S @ b).T
    else:
        est = np.zeros((len(tidx), 14))
    err = est - delta[tidx]
    ew = err @ Vw
    out = {
        "weak_ok": (np.abs(ew).max(axis=1) <= RECOVERY_TOL),
        "weak14_ok": (np.abs(err @ Vw14).max(axis=1) <= RECOVERY_TOL),
        "named_ok": (np.abs(err[:, named_idx]).max(axis=1) <= RECOVERY_TOL),
        "weak_msq": (ew ** 2).mean(axis=1),
        "param_msq": (err ** 2).mean(axis=1),
        "held_msq": ((est @ pool["J"][held_obs].T - shift[np.ix_(tidx, held_obs)]) ** 2).mean(axis=1),
    }
    return out


def split_design_held(pool, rng, frac):
    circ = pool["circuits"]
    held = []
    for fam in ("kicked", "cz", "spam"):
        ids = [i for i, c in enumerate(circ) if c["family"] == fam]
        k = int(round(frac * len(ids)))
        held += list(rng.choice(ids, size=k, replace=False))
    held = sorted(int(x) for x in held)
    design = [i for i in range(len(circ)) if i not in set(held)]
    return design, held


def make_selections(pool, design, b, P0, Vw, grp, cgroup, first_order, uni_order):
    """All deterministic strategies for one budget; random is drawn per trial in run_study."""
    design_set = set(design)
    return {
        "fixed_first": fill_in_order(pool, first_order, design_set, b),
        "fixed_uniform": fill_in_order(pool, uni_order, design_set, b),
        "dopt": greedy_select(pool, design, b, P0, score_dopt(pool)),
        "dopt_cost": greedy_select(pool, design, b, P0, score_dopt_cost(pool)),
        "weak_cost": greedy_select(pool, design, b, P0, score_weak_cost(pool, Vw)),
        "bpd": bpd_select(pool, design, b, P0, Vw, grp, cgroup, objective="weak"),
        "bpd_ld": bpd_select(pool, design, b, P0, Vw, grp, cgroup, objective="logdet"),
    }


def run_study(cfg, ramsey_truth="linear", verbose=True):
    pool = build_pool(cfg["kicked_duration_s"])
    P0 = prior_precision()
    Vw, S_full = weak_subspace(pool)
    Vw14, _ = weak_subspace(pool, include_sx=True)
    named_idx = [pool["labels"].index(n) for n in NAMED_WEAK]
    first_order, uni_order = fixed_orders(pool)
    budgets = [{"mode": "circuits", "value": k} for k in cfg["circuit_budgets"]] +               [{"mode": "seconds", "value": t} for t in cfg["seconds_budgets"]]
    T = cfg["trials_per_split"]
    acc = {}
    paired = {}
    names = STRATEGIES + ["no_data"]
    for b in budgets:
        acc[bkey(b)] = {s: {"design": [], "weak_ok": [], "weak14_ok": [], "named_ok": [], "weak_msq": [], "param_msq": [], "held_msq": []} for s in names}
        paired[bkey(b)] = {s: {"weak_ok": [], "held_msq": []} for s in names}

    for sp in range(cfg["splits"]):
        rng = np.random.default_rng([cfg["seed"], sp, 1])
        design, held = split_design_held(pool, rng, cfg["holdout_fraction"])
        held_obs = np.concatenate([pool["circuits"][c]["obs"] for c in held]).astype(int)
        held_obs = held_obs[np.linalg.norm(pool["J"][held_obs], axis=1) > 1e-6]
        design_set = set(design)
        rng_t = np.random.default_rng([cfg["seed"], sp, 2])
        delta, y_obs, shift = make_trials(pool, rng_t, T, cfg["noise"], ramsey_truth)
        design_obs = np.concatenate([pool["circuits"][c]["obs"] for c in design]).astype(int)
        grp = fit_groups(pool, design_obs, cfg["nmf_max_groups"], cfg["nmf_target_rel_error"])
        cgroup = circuit_groups(pool, grp, design)
        tidx_all = np.arange(T)
        rng_sel = np.random.default_rng([cfg["seed"], sp, 3])

        def ev_all(sel, tidx):
            m, A, S = design_metrics(pool, sel, Vw, P0)
            return m, evaluate_trials(pool, sel, A, S, delta, y_obs, shift, tidx, Vw, Vw14, named_idx, held_obs)

        for b in budgets:
            k = bkey(b)
            m, ev = ev_all([], tidx_all)
            _store(acc[k]["no_data"], paired[k]["no_data"], m, ev)
            selections = make_selections(pool, design, b, P0, Vw, grp, cgroup, first_order, uni_order)
            for s, sel in selections.items():
                m, ev = ev_all(sel, tidx_all)
                _store(acc[k][s], paired[k][s], m, ev)
            ms, evs = [], []
            for t in range(T):
                order = [design[i] for i in rng_sel.permutation(len(design))]
                sel = fill_in_order(pool, order, design_set, b)
                m, ev = ev_all(sel, np.array([t]))
                ms.append(m)
                evs.append(ev)
            mm = {key: float(np.mean([x[key] for x in ms])) for key in ms[0]}
            mm["max_cost_s"] = float(max(x["cost_s"] for x in ms))
            ev = {key: np.concatenate([e[key] for e in evs]) for key in evs[0]}
            _store(acc[k]["random"], paired[k]["random"], mm, ev)
        if verbose:
            print(f"[{ramsey_truth}] split {sp + 1}/{cfg['splits']} done", file=sys.stderr)
    return pool, Vw, S_full, acc, paired, budgets


def bkey(b):
    return f"{b['mode']}_{b['value']}"


def _store(a, p, m, ev):
    a["design"].append(m)
    for key in ("weak_ok", "weak14_ok", "named_ok", "weak_msq", "param_msq", "held_msq"):
        a[key].append(ev[key])
    p["weak_ok"].append(ev["weak_ok"].astype(float))
    p["held_msq"].append(ev["held_msq"])


def summarise(acc, paired, budgets):
    res = {}
    for b in budgets:
        k = bkey(b)
        res[k] = {}
        for s, a in acc[k].items():
            n = sum(len(x) for x in a["weak_ok"])
            wok = np.concatenate(a["weak_ok"]).astype(float)
            w14 = np.concatenate(a["weak14_ok"]).astype(float)
            nok = np.concatenate(a["named_ok"]).astype(float)
            d = a["design"]
            mean = lambda key: float(np.mean([x[key] for x in d]))
            cost = mean("cost_s")
            held = np.concatenate(a["held_msq"])
            res[k][s] = {
                "trials": int(n),
                "n_circuits": mean("n_circuits"), "n_kicked": mean("n_kicked"), "cost_s": cost,
                "max_cost_s": float(max(x.get("max_cost_s", x["cost_s"]) for x in d)), "shots": mean("shots"),
                "logdet_fim_ridge": mean("logdet_fim_ridge"),
                "info_gain_nats": mean("info_gain_nats"),
                "info_gain_per_s": (mean("info_gain_nats") / cost) if cost > 0 else None,
                "weak_info_gain_nats": mean("weak_info_gain_nats"),
                "weak_info_gain_per_s": (mean("weak_info_gain_nats") / cost) if cost > 0 else None,
                "weak_trace_post": mean("weak_trace_post"),
                "weak_recovery_rate": float(wok.mean()),
                "weak_recovery_se": float(math.sqrt(wok.mean() * (1 - wok.mean()) / n)),
                "weak14_recovery_rate": float(w14.mean()),
                "named_weak_recovery_rate": float(nok.mean()),
                "weak_rmse_rad": float(math.sqrt(np.concatenate(a["weak_msq"]).mean())),
                "param_rmse_rad": float(math.sqrt(np.concatenate(a["param_msq"]).mean())),
                "heldout_rmse": float(math.sqrt(held.mean())),
            }
        res[k]["_paired"] = {}
        for tgt in ("bpd", "bpd_ld"):
            for ref in ("dopt", "dopt_cost", "weak_cost"):
                wb = np.concatenate(paired[k][tgt]["weak_ok"])
                wr = np.concatenate(paired[k][ref]["weak_ok"])
                hb = np.concatenate(paired[k][tgt]["held_msq"])
                hr = np.concatenate(paired[k][ref]["held_msq"])
                dw, dh = wb - wr, hb - hr
                res[k]["_paired"][f"{tgt}|{ref}"] = {
                    "weak_recovery_diff": float(dw.mean()), "weak_recovery_diff_se": float(dw.std(ddof=1) / math.sqrt(len(dw))),
                    "heldout_msq_diff": float(dh.mean()), "heldout_msq_diff_se": float(dh.std(ddof=1) / math.sqrt(len(dh))),
                }
    return res


# ------------------------------------------------------------------------------------------------ analysis helpers
def pool_summary(pool, Vw, S_full):
    labels = pool["labels"]
    J = pool["J"]
    kc = [c for c in pool["circuits"] if c["family"] == "kicked"]
    n_k = pool["n_kicked_obs"]
    Jk = J[:n_k]
    rank = lambda M: int(np.linalg.matrix_rank(M, tol=np.linalg.svd(M, compute_uv=False)[0] * 1e-6))
    sk = np.linalg.svd(Jk, compute_uv=False)
    sp = np.linalg.svd(pool["Jw"], compute_uv=False)
    # linearisation error of the Ramsey model at TRUTH_SD
    rm = pool["ramsey_meta"]
    rng = np.random.default_rng(7)
    lin = {}
    delta = rng.normal(0, TRUTH_SD, size=(500, 14))
    exact = np.array([ramsey_model(pool["theta0"] + d, rm) for d in delta])
    lin_pred = pool["y0"][n_k:][None, :] + delta @ J[n_k:].T
    for m in (1, 2, 8):
        sel = (rm["m"] == m) & (~rm["is_spam"])
        lin[str(m)] = float(np.sqrt(np.mean((exact[:, sel] - lin_pred[:, sel]) ** 2)))
    resid = pool["measured"][:n_k] - pool["y0"][:n_k]
    return {
        "labels": labels,
        "theta0": pool["theta0"].tolist(),
        "n_circuits": len(pool["circuits"]),
        "n_observables": int(J.shape[0]),
        "n_kicked_circuits": len(kc),
        "n_ramsey_cz_circuits": sum(c["family"] == "cz" for c in pool["circuits"]),
        "n_spam_circuits": sum(c["family"] == "spam" for c in pool["circuits"]),
        "kicked_jacobian_shape": list(Jk.shape),
        "kicked_nonzero_rows": int((np.linalg.norm(Jk, axis=1) > 1e-6).sum()),
        "kicked_rank_restricted_6": rank(Jk[:, :6]),
        "kicked_rank_expanded_14": rank(Jk),
        "kicked_singular_values": sk.tolist(),
        "pool_rank_expanded_14": rank(J),
        "pool_unwhitened_singular_values": np.linalg.svd(J, compute_uv=False).tolist(),
        "pool_whitened_singular_values": sp.tolist(),
        "cost_kicked_s": kc[0]["cost"],
        "cost_ramsey_s": next(c["cost"] for c in pool["circuits"] if c["family"] == "cz"),
        "total_pool_cost_s": float(pool["cost"].sum()),
        "kicked_measured_minus_prediction_rms": float(np.sqrt(np.mean(resid ** 2))),
        "kicked_mean_binomial_sd": float(np.sqrt(pool["var"][:n_k]).mean()),
        "ramsey_linearisation_rmse_at_truth_sd": lin,
    }


def weak_description(pool, Vw, S_full):
    labels = pool["labels"]
    named = [labels.index(n) for n in NAMED_WEAK]
    comps = []
    for j in range(Vw.shape[1]):
        v = Vw[:, Vw.shape[1] - 1 - j]
        top = np.argsort(-np.abs(v))[:4]
        comps.append({"sv_whitened": float(S_full[-1 - j]),
                      "top": [{"param": labels[i], "loading": float(v[i])} for i in top if abs(v[i]) > 0.1]})
    Pw = Vw @ Vw.T
    return {"rank": int(Vw.shape[1]), "singular_values": [float(x) for x in S_full[-Vw.shape[1]:][::-1]],
            "components": comps,
            "weak_energy_by_parameter": {labels[i]: float(Pw[i, i]) for i in range(14)},
            "named_weak_energy": {labels[i]: float(Pw[i, i]) for i in named}}


def group_description(pool, grp, Vw, cgroup, design):
    labels = pool["labels"]
    named = [labels.index(n) for n in NAMED_WEAK]
    Pw = Vw @ Vw.T
    pos_group = np.argmax(grp["W"], axis=1)
    nonzero = np.linalg.norm(pool["Jw"][grp["obs_idx"]], axis=1) > 1e-8
    out = []
    for g in range(grp["G"]):
        h = grp["H"][g]
        top = [(labels[i], float(h[i])) for i in np.argsort(-h) if h[i] > 0.15]
        members = [c for c in design if cgroup.get(c) == g]
        fam = {f: sum(pool["circuits"][c]["family"] == f for c in members) for f in ("kicked", "cz")}
        weak_overlap = float(h @ Pw @ h)
        named_share = float(sum(h[i] ** 2 for i in named))
        out.append({
            "group": g, "top_parameters": top,
            "n_observables": int(((pos_group == g) & nonzero).sum()),
            "kicked_circuits": fam["kicked"], "ramsey_circuits": fam["cz"],
            "weak_subspace_overlap": weak_overlap, "named_weak_share": named_share,
            "aligned_with_weak": bool(weak_overlap >= 0.5 or named_share >= 0.5),
        })
    return out


def run_full(cfg, verbose=True, with_exact=True):
    pool, Vw, S_full, acc, paired, budgets = run_study(cfg, "linear", verbose)
    results = summarise(acc, paired, budgets)
    out = {
        "label": "OFFLINE LINEARISED SIMULATION; no hardware evidence",
        "config": cfg,
        "constants": {"truth_sd_rad": TRUTH_SD, "prior_sd_rad": PRIOR_SD, "weak_rank": WEAK_RANK,
                      "recovery_tol_rad": RECOVERY_TOL, "var_floor": VAR_FLOOR, "named_weak": NAMED_WEAK,
                      "kicked_shots": KICKED_SHOTS, "ramsey_shots": 384},
        "pool": pool_summary(pool, Vw, S_full),
        "weak_subspace": weak_description(pool, Vw, S_full),
        "results": results,
    }
    design_all = list(range(len(pool["circuits"])))
    grp = fit_groups(pool, np.arange(pool["J"].shape[0]), cfg["nmf_max_groups"], cfg["nmf_target_rel_error"])
    cg = circuit_groups(pool, grp, design_all)
    out["groups"] = {"G": grp["G"], "rel_error": grp["rel_error"], "sweep": {str(k): v for k, v in grp["sweep"].items()},
                     "table": group_description(pool, grp, Vw, cg, design_all)}
    out["split_ceilings"] = split_ceilings(cfg, pool, Vw)
    if with_exact:
        _, _, _, acc2, paired2, budgets2 = run_study(cfg, "exact", verbose)
        out["results_exact_ramsey"] = summarise(acc2, paired2, budgets2)
    return out


# ------------------------------------------------------------------------------------------------ report
STRAT_NOTE = {
    "random": "random (fresh draw per trial)", "fixed_first": "fixed, pool order", "fixed_uniform": "fixed, hand-uniform Ramsey",
    "dopt": "greedy D-opt (cost-blind)", "dopt_cost": "greedy D-opt per second", "weak_cost": "greedy weak-trace per second",
    "bpd": "BPD group-adaptive (weak trace/s)", "bpd_ld": "BPD group-adaptive (log det/s)", "no_data": "no data (prior only)",
}
METRICS = [("info_gain_per_s", "info gain per QPU s"), ("weak_info_gain_per_s", "weak info gain per QPU s"),
           ("weak_recovery_rate", "weak recovery rate"), ("heldout_rmse", "held-out RMSE"), ("weak_rmse_rad", "weak RMSE")]


def _fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def table(res_b, rows=STRATEGIES + ["no_data"]):
    head = ("| Strategy | circuits | kicked | QPU s | info gain nats/s | weak info gain nats/s | weak recovery | recovery incl. SX | named-weak recovery "
            "| weak RMSE rad | held-out RMSE | param RMSE rad |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    lines = []
    for s in rows:
        r = res_b[s]
        lines.append(f"| {STRAT_NOTE[s]} | {r['n_circuits']:.1f} | {r['n_kicked']:.1f} | {r['cost_s']:.2f} | "
                     f"{_fmt(r['info_gain_per_s'], 2)} | {_fmt(r['weak_info_gain_per_s'], 2)} | "
                     f"{r['weak_recovery_rate']:.3f} +/- {r['weak_recovery_se']:.3f} | {r['weak14_recovery_rate']:.3f} | {r['named_weak_recovery_rate']:.3f} | "
                     f"{r['weak_rmse_rad']:.4f} | {r['heldout_rmse']:.4f} | {r['param_rmse_rad']:.4f} |")
    return head + "\n".join(lines) + "\n"


def verdicts(results, target):
    """Outcome counts of `target` (bpd or bpd_ld) against each reference over all budgets."""
    tally = {}
    for ref in ("dopt", "dopt_cost", "weak_cost"):
        tally[ref] = {m[0]: {"win": 0, "tie": 0, "loss": 0} for m in METRICS}
        for bk, rb in results.items():
            pr = rb["_paired"][f"{target}|{ref}"]
            for key, _ in METRICS:
                a, b = rb[target][key], rb[ref][key]
                if key == "weak_recovery_rate":
                    d, se = pr["weak_recovery_diff"], pr["weak_recovery_diff_se"]
                    outcome = "win" if d > 2 * se else ("loss" if d < -2 * se else "tie")
                elif key == "heldout_rmse":
                    d, se = pr["heldout_msq_diff"], pr["heldout_msq_diff_se"]
                    outcome = "win" if d < -2 * se else ("loss" if d > 2 * se else "tie")
                else:
                    sign = -1 if key == "weak_rmse_rad" else 1
                    d = sign * (a - b)
                    tol = 0.01 * abs(b) if b else 1e-12
                    outcome = "win" if d > tol else ("loss" if d < -tol else "tie")
                tally[ref][key][outcome] += 1
    return tally


def verdict_table(results, target):
    t = verdicts(results, target)
    lines = [f"| Metric ({STRAT_NOTE[target]}) | vs cost-blind D-opt W/T/L | vs D-opt per second W/T/L | vs weak-trace per second W/T/L |", "|---|---|---|---|"]
    for k, name in METRICS:
        lines.append(f"| {name} | " + " | ".join(f"{t[r][k]['win']}/{t[r][k]['tie']}/{t[r][k]['loss']}" for r in ("dopt", "dopt_cost", "weak_cost")) + " |")
    return "\n".join(lines) + "\n", t


def all_budget_table(R):
    s = ["| Budget | " + " | ".join(STRAT_NOTE[x] for x in STRATEGIES) + " |", "|---|" + "---:|" * len(STRATEGIES)]
    for bk, rb in R.items():
        cells = [f"{rb[x]['weak_recovery_rate']:.2f} / {rb[x]['heldout_rmse']:.3f} / {rb[x]['info_gain_per_s']:.1f}" for x in STRATEGIES]
        s.append(f"| {bk} | " + " | ".join(cells) + " |")
    return "\n".join(s) + "\n"


def best_lines(R):
    out = []
    for key, name, fn in (("weak_recovery_rate", "weak recovery", max), ("heldout_rmse", "held-out RMSE", min), ("info_gain_per_s", "information gain per QPU second", max)):
        parts = []
        for bk, rb in R.items():
            v = fn(STRATEGIES, key=lambda x: rb[x][key])
            parts.append(f"{bk}: {STRAT_NOTE[v]} ({rb[v][key]:.3f})")
        out.append(f"- Best {name} per budget: " + "; ".join(parts) + ".")
    return out


def build_report(out):
    R = out["results"]
    RX = out.get("results_exact_ramsey")
    cfg = out["config"]
    pool = out["pool"]
    ws = out["weak_subspace"]
    tr = cfg["splits"] * cfg["trials_per_split"]
    s = []
    s.append("# PGC Phase 6: sensitivity-matrix study and experimental-design comparison (BPD)\n")
    s.append("**Offline linearised simulation. No hardware evidence, no IBM/QPU service calls, no refit of historical counts.** "
             f"Every number below comes from `bpd_study.py` (seed {cfg['seed']}) and is stored in `bpd_results.json`.\n")
    s.append("## Findings (linearised truth, win/tie/loss over all budgets)\n")
    s.append("\n".join(findings(out)) + "\n")
    s.append("## Setup\n")
    s.append(f"- Pool: {pool['n_circuits']} circuits and {pool['n_observables']} observables over 14 parameters: "
             f"{pool['n_kicked_circuits']} kicked-Ising circuits (8 observables each, 2048 shots, {pool['cost_kicked_s']:.5f} s each), "
             f"{pool['n_ramsey_cz_circuits']} Ramsey cz probes plus {pool['n_spam_circuits']} SPAM references (384 shots, {pool['cost_ramsey_s']:.5f} s each). "
             f"Cost = shots*(0.00025 + max(1e-5, duration)) + 0.1 s; durations are unavailable offline, so the 1e-5 s floor applies. "
             f"Whole pool: {pool['total_pool_cost_s']:.2f} s.")
    s.append("- Jacobian: kicked-Ising rows are the receipt rows (central differences from the native engine). Ramsey rows are central finite "
             "differences (h=1e-5) of the analytic probe model in the 14-parameter space at the receipt parameter values; SX columns are zero. "
             f"Kicked-Ising Jacobian {pool['kicked_jacobian_shape']}: restricted (6 columns) rank {pool['kicked_rank_restricted_6']}, "
             f"expanded (14 columns) rank {pool['kicked_rank_expanded_14']}; whole-pool rank {pool['pool_rank_expanded_14']}.")
    s.append("- Noise: binomial at the stated shots (2048 or 384) per observable, independent across observables (observables of one "
             "kicked-Ising circuit share bitstrings in reality; the joint likelihood is not modelled). Estimator weights use var = max(1-y0^2, 0.05)/shots.")
    s.append(f"- Truth: theta = theta0 + delta, delta ~ N(0, {TRUTH_SD}^2 I) rad. Estimator and design prior: N(0, {PRIOR_SD}^2 I) (weaker than the truth spread). "
             "Estimator: linear Gauss-Newton posterior mean. The model is linear, so the Jacobian never changes and 'adaptive' selection reacts to the "
             "accumulated design information, not to measured outcomes. Primary tables generate Ramsey truth from the linearised model; a second run generates it "
             "from the exact analytic probe model (estimator still linear).")
    s.append(f"- Weak subspace: the {ws['rank']} right singular vectors with smallest singular values of the noise-whitened full-pool Jacobian over the 12 ZZ/Z columns "
             f"(whitened singular values {', '.join(f'{x:.1f}' for x in ws['singular_values'])}). The two SX columns are the weakest directions of the whole pool "
             "and no budgeted design reaches the tolerance on them, so they are excluded from the primary subspace; "
             "'recovery incl. SX' reports the literal 14-column definition. "
             f"Recovery = max over weak coordinates of |error| <= {RECOVERY_TOL} rad (half the truth spread). Named-weak recovery applies the same tolerance to "
             "ZZ_22_23, ZZ_4_5, Z_22_23_23, Z_4_5_5 directly.")
    s.append(f"- Protocol: {cfg['splits']} random design/held-out splits (25% of each circuit family held out) x {cfg['trials_per_split']} trials = {tr} trials per "
             "strategy and budget, common random numbers across strategies; random draws a fresh design every trial. Held-out RMSE: predicted <O> of the posterior mean "
             "versus the simulated true <O> on held-out observables with nonzero sensitivity. Designs are chosen from the design pool only; groups are refitted on design-pool rows per split.")
    s.append(f"- Budgets: at most k circuits for k in {cfg['circuit_budgets']} (cost differs per strategy and is reported) and matched QPU-second caps "
             f"{cfg['seconds_budgets']} s (each strategy fills up to the cap; none exceeds it).\n")

    s.append("## Weak directions of the pool\n")
    for i, c in enumerate(ws["components"]):
        tops = ", ".join(f"{t['param']} ({t['loading']:+.2f})" for t in c["top"])
        s.append(f"- weak vector {i + 1} (whitened singular value {c['sv_whitened']:.1f}): {tops}")
    s.append("")
    named = ws["named_weak_energy"]
    s.append("Share of each named weak parameter inside the weak subspace: " + ", ".join(f"{k} {v:.2f}" for k, v in named.items()) +
             ". The kicked-Ising Jacobian alone is rank-deficient exactly on Z_22_23_23 and Z_4_5_5 (rank 12 of 14); in the pool those two are reached only through "
             "the Ramsey forward probes. Parameters with zero share are not in the weak subspace (they are measured by several rows) and stay in the named-weak metric.\n")

    s.append("## BPD grouping\n")
    g = out["groups"]
    s.append(f"NMF of row-normalised |J| over all {sum(r['n_observables'] for r in g['table'])} nonzero-sensitivity observables; G={g['G']} groups kept (smallest G with relative "
             f"reconstruction error <= {cfg['nmf_target_rel_error']}, empty groups dropped; final error {g['rel_error']:.3f}; sweep: " +
             ", ".join(f"G={k}: {v:.3f}" for k, v in g["sweep"].items()) + ").\n")
    s.append("| Group | informs (pattern weight) | obs | kicked circ | Ramsey circ | weak overlap | named-weak share | aligned with weak |\n|---|---|---:|---:|---:|---:|---:|---|")
    for r in g["table"]:
        tp = ", ".join(f"{a} {b:.2f}" for a, b in r["top_parameters"])
        s.append(f"| {r['group']} | {tp} | {r['n_observables']} | {r['kicked_circuits']} | {r['ramsey_circuits']} | {r['weak_subspace_overlap']:.2f} | "
                 f"{r['named_weak_share']:.2f} | {'yes' if r['aligned_with_weak'] else 'no'} |")
    s.append("\nGroups use |J| rows only. Each recurring pattern is an endpoint Z term, or a ZZ term paired with an endpoint Z term of the same edge, because those columns "
             "co-occur in the sensitive rows. Weak overlap is h' P_w h (fraction of the group's unit pattern inside the weak subspace); aligned = overlap >= 0.5 or "
             "named-weak share >= 0.5.\n")

    for bk, title in (("seconds_6.4", "Matched QPU-second cap 6.4 s"), ("circuits_16", "Circuit budget 16")):
        s.append(f"## Strategy comparison: {title} (linearised truth)\n")
        s.append(table(R[bk]))
    s.append("## All budgets: weak recovery / held-out RMSE / info gain per QPU s (linearised truth)\n")
    s.append(all_budget_table(R))
    s.append("## What BPD does and does not beat (linearised truth)\n")
    s.append(f"Win/tie/loss counts over the {len(R)} budgets (recovery and held-out RMSE: paired difference beyond 2 standard errors; the other metrics: 1% margin).\n")
    vt1, t1 = verdict_table(R, "bpd")
    vt2, t2 = verdict_table(R, "bpd_ld")
    s.append(vt1)
    s.append(vt2)
    s.append("\n".join(best_lines(R)) + "\n")
    s.append("\n".join(_narrative(R, t1, t2, "linearised truth")) + "\n")
    if RX:
        s.append("## Robustness: Ramsey truth from the exact probe model\n")
        s.append("Same designs and trials, but the simulated Ramsey expectations use cos/sin of the exact phase while the estimator stays linear. "
                 "Designs that lean on high-m rows lose information here.\n")
        s.append(table(RX["seconds_6.4"]))
        s.append(all_budget_table(RX))
        vx1, tx1 = verdict_table(RX, "bpd")
        vx2, tx2 = verdict_table(RX, "bpd_ld")
        s.append(vx1)
        s.append(vx2)
        s.append("\n".join(best_lines(RX)) + "\n")
        s.append("\n".join(_narrative(RX, tx1, tx2, "exact-Ramsey truth")) + "\n")
    s.append("## Limits\n")
    s.append("- Linearised model: the Gauss-Newton posterior is exact only for the linear surrogate. The RMSE of linear versus exact Ramsey probe expectation at "
             f"{TRUTH_SD} rad spread is " + ", ".join(f"{v:.3f} (m={k})" for k, v in pool["ramsey_linearisation_rmse_at_truth_sd"].items()) +
             ", so m=8 rows are overstated in the linearised-truth tables.")
    s.append(f"- Kicked-Ising residuals: rms(measured - prediction) = {pool['kicked_measured_minus_prediction_rms']:.4f} versus mean binomial sd "
             f"{pool['kicked_mean_binomial_sd']:.4f}; model misfit, SPAM, drift and crosstalk are not simulated.")
    s.append("- Observables within a kicked-Ising circuit are treated as independent in the likelihood (the README requires a joint likelihood for real inference); "
             "the SPAM circuits carry no parameter sensitivity and sit in the pool as given.")
    s.append("- The kicked-Ising duration is unknown offline; a longer duration raises the kicked-Ising cost and moves every cost-aware comparison further toward Ramsey circuits.")
    s.append("- Standard errors are binomial/paired-trial errors; trials within a split share a design split, so they are not split-level confidence intervals.")
    s.append("- Truth is drawn from the same linear (or exact Ramsey) model the estimator assumes. No claim about hardware follows.\n")
    s.append("## Reproduce\n")
    s.append("    python bpd_study.py            # rewrites bpd_results.json and BPD_REPORT.md\n    python bpd_study.py --check    # reduced rerun with assertions\n    python -m pytest test_bpd.py -q")
    return "\n".join(s) + "\n"


def _phrase(t):
    mn = dict(METRICS)
    ahead = [f"{mn[k]} ({v['win']}W/{v['tie']}T/{v['loss']}L)" for k, v in t.items() if v["win"] > v["loss"]]
    behind = [f"{mn[k]} ({v['win']}W/{v['tie']}T/{v['loss']}L)" for k, v in t.items() if v["loss"] > v["win"]]
    level = [f"{mn[k]} ({v['win']}W/{v['tie']}T/{v['loss']}L)" for k, v in t.items() if v["loss"] == v["win"]]
    return f"ahead on {', '.join(ahead) or 'nothing'}; behind on {', '.join(behind) or 'nothing'}; level on {', '.join(level) or 'nothing'}"


def _narrative(R, t1, t2, label):
    lines = []
    names = {"dopt": "cost-blind D-optimal", "dopt_cost": "D-optimal per second", "weak_cost": "ungrouped weak-trace per second"}
    for tag, t in (("BPD (weak trace/s)", t1), ("BPD (log det/s)", t2)):
        for ref in ("dopt", "dopt_cost", "weak_cost"):
            lines.append(f"- {label}, {tag} vs {names[ref]}: {_phrase(t[ref])}.")
    return lines


def split_ceilings(cfg, pool, Vw):
    """Gaussian-approximation ceiling on weak recovery given everything in each split's design pool."""
    from math import erf
    P0 = prior_precision()
    out = []
    for sp in range(cfg["splits"]):
        rng = np.random.default_rng([cfg["seed"], sp, 1])
        design, _ = split_design_held(pool, rng, cfg["holdout_fraction"])
        A = P0 + pool["F"][design].sum(axis=0)
        sd = np.sqrt(np.diag(Vw.T @ np.linalg.inv(A) @ Vw))
        prob = float(np.prod([erf(RECOVERY_TOL / (x * math.sqrt(2))) for x in sd]))
        out.append({"split": sp, "max_weak_posterior_sd": float(sd.max()), "gaussian_recovery_ceiling": prob})
    return out


def findings(out):
    R = out["results"]
    t1 = verdicts(R, "bpd")
    t2 = verdicts(R, "bpd_ld")
    bk = "seconds_6.4"
    informed = ["dopt", "dopt_cost", "weak_cost", "bpd", "bpd_ld"]
    naive = ["random", "fixed_first", "fixed_uniform"]
    rec_i = [R[bk][x]["weak_recovery_rate"] for x in informed]
    rec_n = [R[bk][x]["weak_recovery_rate"] for x in naive]
    ce = out["split_ceilings"]
    f = []
    f.append(f"- At {bk} (matched QPU seconds) weak recovery is {min(rec_n):.2f}-{max(rec_n):.2f} for random and fixed designs and {min(rec_i):.2f}-{max(rec_i):.2f} for the five "
             f"information-driven designs (D-opt, D-opt per second, weak-trace per second, two BPD variants). All information-driven designs beat random and fixed designs; this is a property of using "
             "the Jacobian at all, not of grouping.")
    f.append(f"- Weak recovery saturates: the mean Gaussian ceiling over the {len(ce)} design pools is {np.mean([c['gaussian_recovery_ceiling'] for c in ce]):.2f} "
             f"(worst split max weak posterior sd {max(c['max_weak_posterior_sd'] for c in ce):.3f} rad, because the held-out set can remove the high-m rows that carry Z_22_23_23 and Z_4_5_5). "
             f"The best observed recovery of any strategy at any budget is {max(R[b][x]['weak_recovery_rate'] for b in R for x in STRATEGIES):.2f}; differences between good designs at large budgets are therefore not informative.")
    f.append(f"- BPD (weak trace/s) vs conventional cost-blind D-opt over {len(R)} budgets: {_phrase(t1['dopt'])}.")
    f.append(f"- BPD (weak trace/s) vs ungrouped weak-trace-per-second greedy (same objective, no groups): {_phrase(t1['weak_cost'])}. Group restriction does not add to the objective it optimises.")
    f.append(f"- BPD (log det/s) vs conventional cost-blind D-opt: {_phrase(t2['dopt'])}.")
    return f


# ------------------------------------------------------------------------------------------------ check
def check():
    cfg = dict(DEFAULT_CFG, splits=2, trials_per_split=20, circuit_budgets=[8, 16], seconds_budgets=[3.2, 6.4])
    pool = build_pool()
    s = pool_summary(pool, *weak_subspace(pool))
    assert s["kicked_jacobian_shape"] == [64, 14], s["kicked_jacobian_shape"]
    assert s["kicked_rank_restricted_6"] == 6 and s["kicked_rank_expanded_14"] == 12, s
    assert s["n_circuits"] == 140 and s["n_observables"] == 196
    out1 = run_full(cfg, verbose=False, with_exact=True)
    out2 = run_full(cfg, verbose=False, with_exact=True)
    assert json.dumps(out1, sort_keys=True) == json.dumps(out2, sort_keys=True), "not reproducible"
    for bk, rb in list(out1["results"].items()) + list(out1["results_exact_ramsey"].items()):
        mode, val = bk.split("_")
        for st in STRATEGIES:
            r = rb[st]
            if mode == "circuits":
                assert r["n_circuits"] <= float(val) + 1e-9, (bk, st)
            else:
                assert r["max_cost_s"] <= float(val) + 1e-9, (bk, st, r["max_cost_s"])
        assert rb["dopt"]["logdet_fim_ridge"] > rb["random"]["logdet_fim_ridge"], bk
        assert rb["dopt"]["logdet_fim_ridge"] >= rb["fixed_first"]["logdet_fim_ridge"] - 1e-9, bk
    print("check passed: ranks, budgets, D-opt > random log det, reproducibility")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--splits", type=int, default=DEFAULT_CFG["splits"])
    ap.add_argument("--trials", type=int, default=DEFAULT_CFG["trials_per_split"])
    ap.add_argument("--noise", choices=["binomial", "gaussian"], default="binomial")
    ap.add_argument("--out", type=Path, default=HERE / "bpd_results.json")
    ap.add_argument("--report", type=Path, default=HERE / "BPD_REPORT.md")
    a = ap.parse_args()
    if a.check:
        return check()
    cfg = dict(DEFAULT_CFG, splits=a.splits, trials_per_split=a.trials, noise=a.noise)
    out = run_full(cfg)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=False) + "\n")
    a.report.write_text(build_report(out))
    print(f"wrote {a.out} and {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
