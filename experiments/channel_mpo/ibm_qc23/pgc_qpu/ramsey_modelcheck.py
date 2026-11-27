"""Ramsey model-check companion; preserves ramsey_identify.py as baseline.

Exact probe marginal binomial likelihood with bounded physical visibility.
For b=1, geometric relaxation mixes phase accumulated before and after
control decay. Effective SPAM is held at the preparation-row estimate.
Curvature is an exploratory diagnostic, not a calibrated hardware parameter.
All models use the same shots. Held-out m=8 tests extrapolation from 0,1,2.
No IBM calls. No model is selected for QC-26B by this script.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import xlogy

import ramsey_identify as R


def probe_expectation(m, state, axis, theta, relaxed=False, curved=False):
    """theta=(intercept,alpha,delta,visibility,retention[,gamma][,curvature]).
    A decay after j completed CZs leaves state 0 for the remaining m-j gates.
    A decay after the final gate is phase-equivalent to survival.
    """
    phi, alpha, delta, vis, retention = theta[:5]
    gamma = theta[5] if relaxed else 0.0
    curvature = theta[5 + int(relaxed)] if curved else 0.0

    def phase(j):
        return phi + m * (alpha + delta) + j * (math.pi - 2 * delta) + curvature * m * (m - 1) / 2

    if not state:
        z = np.exp(1j * phase(0))
    else:
        z = (1 - gamma) ** m * np.exp(1j * phase(m))
        for j in range(1, m + 1):
            z += gamma * (1 - gamma) ** (j - 1) * np.exp(1j * phase(j))
    z *= vis * retention ** m
    return float(z.real if axis == "X" else z.imag)


def row_prob(row, theta, spam, relaxed, curved):
    z = probe_expectation(row["m"], row["state"], row["axis"], theta, relaxed, curved)
    e0, e1 = spam["e0"], spam["e1"]
    return float(np.clip(e0 + (1 - e0 - e1) * (1 - z) / 2, 1e-10, 1 - 1e-10))


def deviance(k, n, p):
    mu = n * p
    return 2 * (xlogy(k, k / mu) + xlogy(n-k, (n-k) / (n-mu)))


def fit(rows, data, spam, hint, relaxed=False, curved=False, holdout=None):
    # Two independent preparation intercepts; alpha/delta shared across states.
    train = [r for r in rows if r["m"] != holdout]
    test = [r for r in rows if r["m"] == holdout]
    lower = [-math.pi, -math.pi, -math.pi/2, -math.pi/2, 0.0, 0.8]
    upper = [math.pi, math.pi, math.pi/2, math.pi/2, 1.0, 1.0]
    initial = [0.0, 0.0, hint[0], hint[1], 0.98, 0.99]
    if relaxed:
        lower += [0.0]; upper += [0.2]; initial += [0.005]
    if curved:
        lower += [-0.2]; upper += [0.2]; initial += [0.0]

    def get_counts(r):
        d = data[r["id"]]
        return sum(c for bits, c in d.items() if bits[1]), sum(d.values())

    def terms(rs, th):
        for r in rs:
            k, n = get_counts(r)
            # Pick preparation-specific intercept; all remaining parameters shared.
            p = row_prob(r, [th[r["state"]], *th[2:]], spam, relaxed, curved)
            yield k, n, p

    def residual(th):
        return np.array([np.sign(k - n*p)*math.sqrt(max(0.0, deviance(k,n,p)))
                         for k,n,p in terms(train,th)])

    # Starting from baseline small-error branch. Alternative branches deliberately
    # excluded; this is a conditional diagnostic, not global alias inference.
    sol = least_squares(residual, np.clip(initial, lower, upper),
                        bounds=(lower, upper), max_nfev=1000)
    th = sol.x
    train_d = float(sum(deviance(k,n,p) for k,n,p in terms(train,th)))
    holdout_d = float(sum(deviance(k,n,p) for k,n,p in terms(test,th))) if test else None
    return {"parameters": th.tolist(), "parameter_count": len(th), "converged": bool(sol.success),
            "train_deviance": train_d, "heldout_deviance": holdout_d,
            "aic_relative_to_saturated": train_d + 2*len(th),
            "alpha": float(th[2]), "delta": float(th[3])}


def run(rows, data):
    points, spam = R.estimate(rows, data)
    train_points, _ = R.estimate(rows, data, include=(0, 1, 2))
    out = {}
    for edge, point in points.items():
        out[edge] = {}
        for orientation in ("forward", "reverse"):
            rs = [r for r in rows if r["family"]=="cz" and r["edge"]==edge and r["orientation"]==orientation]
            probe = str(rs[0]["path"][1])
            hint = [point["alpha"][probe], point["delta_"+orientation]]
            train_hint = [train_points[edge]['alpha'][probe], train_points[edge]['delta_'+orientation]]
            out[edge][orientation] = {}
            for name, relaxed, curved in (("constant",False,False),("relaxation",True,False),
                                           ("curvature",False,True),("both",True,True)):
                out[edge][orientation][name] = {
                    "full": fit(rs,data,spam[int(probe)],hint,relaxed,curved),
                    "holdout_m8": fit(rs,data,spam[int(probe)],train_hint,relaxed,curved,holdout=8)}
    return {"label":"EXPLORATORY MODEL CHECK; no replication or model acceptance",
            "limitations":["Probe marginal likelihood conditional on effective SPAM.",
                           "Relaxation after each CZ is a simplified control-decay model.",
                           "m=8 extrapolation uses few train rows; curvature is weakly identified.",
                           "Held-out branch initialization uses only m=0,1,2; all models are conditional on a small-error branch."],
            "edges":out}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compiled",type=Path,required=True)
    ap.add_argument("--counts",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    meta=json.loads(a.compiled.read_text())
    raw=json.loads(a.counts.read_text())
    data=R.load_counts(meta["circuits"],raw)
    result=run(meta["circuits"],data)
    result["plan_sha256"]=meta["plan_sha256"]
    a.out.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
