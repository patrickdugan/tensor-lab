"""PGC-QPU01 four-edge Ramsey estimator with joint-bitstring bootstrap and per-replicate SPAM refit.

Coherent error model after every CZ:  U_err = exp[-i/2 (a_c Z_c + a_t Z_t + d Z_c Z_t)].
For control eigenvalue z_b = (-1)^b the probe accumulates   phi_m = phi0_b + m (a_t + z_b d + b pi)
(the ideal CZ phase m*b*pi is subtracted explicitly), with <X> = V_m cos phi_m and <Y> = V_m sin phi_m,
V_m = v r^m.  Forward (control=a, probe=b) exposes (a_b, d); the reverse row exposes (a_a, d).

Scope: marginal probe expectation per row, SPAM from the two all-site preparation rows, bootstrap that
resamples every row's joint bitstring multiset (SPAM rows included) and refits SPAM each replicate.
Not modelled: crosstalk beyond a joint-readout correlation diagnostic, leakage, drift, non-diagonal error.

  python ramsey_identify.py --compiled results_local/pgc_q23/compiled.json --counts results_local/pgc_q23/counts.json \
      --out results_local/pgc_q23/identify.json --bootstraps 400
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPS = (0, 1, 2, 8)
TWO_PI = 2 * math.pi


def wrap(x):
    return (x + math.pi) % TWO_PI - math.pi


# ---------------------------------------------------------------- counts handling
def split_counts(path, counts):
    """Joint counts as {(bit_site0, bit_site1, ...): n}; qiskit keys are little-endian in circuit qubit order."""
    out = {}
    for key, n in counts.items():
        bits = tuple(int(key[-1 - i]) for i in range(len(path)))
        out[bits] = out.get(bits, 0) + int(n)
    return out


def resample_row(rng, joint):
    keys = list(joint)
    n = sum(joint.values())
    p = np.array([joint[k] for k in keys], float) / n
    draw = rng.multinomial(n, p)
    return {k: int(c) for k, c in zip(keys, draw) if c}


def marginal_z(joint, i):
    n = sum(joint.values())
    return sum(c * (1 - 2 * k[i]) for k, c in joint.items()) / n


def fit_spam(rows, data):
    """Per-site readout flip probabilities from the all-0 and all-1 preparations."""
    spam = {}
    for r in rows:
        if r["family"] != "spam":
            continue
        for i, site in enumerate(r["path"]):
            ones = sum(c for k, c in data[r["id"]].items() if k[i] == 1) / sum(data[r["id"]].values())
            spam.setdefault(site, {})["e0" if r["state"] == 0 else "e1"] = ones if r["state"] == 0 else 1 - ones
    return spam


def spam_joint_correlation(rows, data):
    """Max |P(joint) - prod P(marginals)| over the SPAM rows: a readout-crosstalk diagnostic, not a model input."""
    worst = 0.0
    for r in rows:
        if r["family"] != "spam":
            continue
        j = data[r["id"]]; n = sum(j.values()); k = len(r["path"])
        marg = [sum(c for kk, c in j.items() if kk[i] == 1) / n for i in range(k)]
        for bits in np.ndindex(*(2,) * k):
            prod = np.prod([marg[i] if b else 1 - marg[i] for i, b in enumerate(bits)])
            worst = max(worst, abs(j.get(tuple(bits), 0) / n - prod))
    return worst


# ---------------------------------------------------------------- per (edge, orientation, state) fit
def fit_group(obs, state, reps=REPS, include=None, s_starts=None):
    """obs[(m, axis)] = (z, var) SPAM-corrected. Returns dict(theta, chi2, s)."""
    keys = [(m, a) for m in reps if (include is None or m in include) for a in "XY" if (m, a) in obs]
    z = np.array([obs[k][0] for k in keys]); sd = np.sqrt(np.array([max(obs[k][1], 1e-9) for k in keys]))
    ms = np.array([k[0] for k in keys], float); ax = [k[1] for k in keys]
    sign_b = state * math.pi

    def resid(th, fixed_s=None):
        phi0, s, v, r = th if fixed_s is None else (th[0], fixed_s, th[1], th[2])
        out = []
        for z_i, sd_i, m, a in zip(z, sd, ms, ax):
            ph = phi0 + m * (s + sign_b)
            e = v * r ** m * (math.cos(ph) if a == "X" else math.sin(ph))
            out.append((z_i - e) / sd_i)
        return np.array(out)

    def start_from(s0):
        # phi0 and v from the m=0 pair, decay r from the |m=1|/|m=0| ratio when present
        x0, y0 = obs.get((0, "X"), (1, 0))[0], obs.get((0, "Y"), (0, 0))[0]
        v0 = max(math.hypot(x0, y0), 0.3)
        return [math.atan2(y0, x0), s0, min(v0, 1.2), 0.99]

    best = None
    for s0 in (np.linspace(-math.pi, math.pi, 25, endpoint=False) if s_starts is None else s_starts):
        sol = least_squares(resid, start_from(s0), bounds=([-4 * math.pi, -2 * math.pi, 0, 0.8], [4 * math.pi, 2 * math.pi, 1.5, 1.0]))
        if best is None or sol.cost < best.cost:
            best = sol
    th = best.x.copy(); th[1] = wrap(th[1])
    return {"theta": th, "chi2": 2 * best.cost, "n": len(z), "resid_fn": resid}


def chi2_at_slope(obs, state, s_fixed, reps=REPS, include=None):
    """Profile chi2 with the slope held fixed (others refit): used for alias scans."""
    keys = [(m, a) for m in reps if (include is None or m in include) for a in "XY" if (m, a) in obs]
    z = np.array([obs[k][0] for k in keys]); sd = np.sqrt(np.array([max(obs[k][1], 1e-9) for k in keys]))
    ms = np.array([k[0] for k in keys], float); ax = [k[1] for k in keys]

    def resid(th):
        phi0, v, r = th
        return np.array([(z_i - v * r ** m * (math.cos(phi0 + m * (s_fixed + state * math.pi)) if a == "X" else math.sin(phi0 + m * (s_fixed + state * math.pi)))) / sd_i
                         for z_i, sd_i, m, a in zip(z, sd, ms, ax)])
    best = None
    for p0 in np.linspace(-math.pi, math.pi, 8, endpoint=False):
        sol = least_squares(resid, [p0, 0.9, 0.99], bounds=([-4 * math.pi, 0, 0.8], [4 * math.pi, 1.5, 1.0]))
        if best is None or sol.cost < best.cost:
            best = sol
    return 2 * best.cost


# ---------------------------------------------------------------- campaign-level estimation
def observations(rows, data, spam):
    """SPAM-corrected probe expectations keyed by (edge, orientation, state) -> {(m, axis): (z, var)}."""
    groups = {}
    for r in rows:
        if r["family"] != "cz":
            continue
        probe = r["path"][1]
        e0, e1 = spam[probe]["e0"], spam[probe]["e1"]
        contrast = 1 - e0 - e1
        j = data[r["id"]]; n = sum(j.values())
        zobs = marginal_z(j, 1)
        z = (zobs - (e1 - e0)) / contrast
        var = max(1 - zobs ** 2, 1.0 / n) / n / contrast ** 2
        groups.setdefault((r["edge"], r["orientation"], r["state"]), {})[(r["m"], r["axis"])] = (z, var)
    return groups


def estimate(rows, data, include=None, hint=None):
    spam = fit_spam(rows, data)
    groups = observations(rows, data, spam)
    paths = {}
    for r in rows:
        if r["family"] == "cz":
            paths[(r["edge"], r["orientation"])] = r["path"]
    out = {}
    for edge in sorted({k[0] for k in groups}):
        slopes, vis, chi2 = {}, [], 0.0
        for orient in ("forward", "reverse"):
            for state in (0, 1):
                f = fit_group(groups[(edge, orient, state)], state, include=include,
                              s_starts=None if hint is None else [hint[edge][f'{orient}_{state}']])
                slopes[(orient, state)] = float(f["theta"][1]); chi2 += f["chi2"]; vis.append(float(f["theta"][2]))
        a_f, a_r, d_f, d_r = {}, {}, {}, {}
        for orient in ("forward", "reverse"):
            s0, s1 = slopes[(orient, 0)], slopes[(orient, 1)]
            probe = paths[(edge, orient)][1]
            a_f[orient] = (probe, wrap(0.5 * (s0 + s1)))
            d_f[orient] = wrap(0.5 * (s0 - s1))
        out[edge] = {"alpha": {str(a_f["forward"][0]): a_f["forward"][1], str(a_f["reverse"][0]): a_f["reverse"][1]},
                     "delta_forward": d_f["forward"], "delta_reverse": d_f["reverse"],
                     "delta": 0.5 * (d_f["forward"] + d_f["reverse"]), "delta_gap": d_f["forward"] - d_f["reverse"],
                     "visibility_v": vis, "chi2": chi2, "slopes": {f"{k[0]}_{k[1]}": v for k, v in slopes.items()}}
    return out, spam


def alias_scan(rows, data, edge, orientation="forward"):
    """Profile chi2 at s0 and s0+pi (state 0) with and without odd repetitions; the even-only alias is exact."""
    spam = fit_spam(rows, data)
    g = observations(rows, data, spam)
    res = {}
    for label, inc in (("even_only_0_2_8", (0, 2, 8)), ("with_odd_0_1_2_8", (0, 1, 2, 8))):
        obs0 = g[(edge, orientation, 0)]
        f = fit_group(obs0, 0, include=inc)
        s_hat = float(f["theta"][1])
        c0 = chi2_at_slope(obs0, 0, s_hat, include=inc)
        c1 = chi2_at_slope(obs0, 0, wrap(s_hat + math.pi), include=inc)
        res[label] = {"slope_hat": s_hat, "chi2_at_hat": c0, "chi2_at_hat_plus_pi": c1, "delta_chi2_alias": c1 - c0}
    return res


def bootstrap(rows, data, n_boot, seed, include=None):
    rng = np.random.default_rng(seed)
    hint = {e: v['slopes'] for e, v in estimate(rows, data, include=include)[0].items()}   # alias already resolved on the point fit
    cols = {}
    for _ in range(n_boot):
        d = {rid: resample_row(rng, j) for rid, j in data.items()}   # joint bitstrings; SPAM rows resampled too
        est, _ = estimate(rows, d, include=include, hint=hint)       # SPAM refit inside estimate()
        for e, v in est.items():
            c = cols.setdefault(e, {"delta": [], "delta_gap": []})
            c["delta"].append(v["delta"]); c["delta_gap"].append(v["delta_gap"])
            for site, a in v["alpha"].items():
                c.setdefault("alpha_" + site, []).append(a)
    return cols


def pct(x):
    x = np.asarray(x)
    return {"mean": float(np.mean(x)), "se": float(np.std(x, ddof=1)), "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]}


def run(rows, data, n_boot=200, seed=1):
    est, spam = estimate(rows, data)
    boot = bootstrap(rows, data, n_boot, seed) if n_boot else {}
    out = {"spam": spam, "spam_joint_readout_correlation_max": spam_joint_correlation(rows, data), "edges": {}}
    for e, v in est.items():
        row = {"point": v}
        if boot:
            row["bootstrap"] = {k: pct(x) for k, x in boot[e].items()}
            gap = row["bootstrap"]["delta_gap"]
            row["forward_reverse_agree_95"] = bool(gap["ci95"][0] <= 0 <= gap["ci95"][1])
            for k, b in row["bootstrap"].items():
                b["nonzero_95"] = bool(b["ci95"][0] > 0 or b["ci95"][1] < 0)
        out["edges"][e] = row
    return out


# ---------------------------------------------------------------- synthetic counts (tests and dry runs)
def synthetic_counts(rows, truth, rng, e0=0.01, e1=0.015, vis=0.97, decay=0.004, offsets=(0.035, -0.025), rev_off=0.014):
    """Joint-bitstring counts from the diagonal-error model with independent readout flips (no crosstalk)."""
    data = {}
    for r in rows:
        n = r["shots"]
        if r["family"] == "spam":
            p1 = [e0 if r["state"] == 0 else 1 - e1] * len(r["path"])
            bits = rng.random((n, len(r["path"]))) < np.array(p1)
        else:
            x = truth[r["edge"]]
            a_probe = x["local"][str(r["path"][1])]
            s = a_probe + (1 if r["state"] == 0 else -1) * x["delta"]
            ph = r["m"] * (s + r["state"] * math.pi) + offsets[r["state"]] + (rev_off if r["orientation"] == "reverse" else 0)
            v = vis * math.exp(-decay * r["m"]) * (math.cos(ph) if r["axis"] == "X" else math.sin(ph))
            p_probe1_ideal = (1 - v) / 2
            ctrl_true = r["state"]
            c_read = (rng.random(n) < (e0 if ctrl_true == 0 else 1 - e1)).astype(int)
            pr = p_probe1_ideal * (1 - e1) + (1 - p_probe1_ideal) * e0
            p_read = (rng.random(n) < pr).astype(int)
            bits = np.stack([c_read, p_read], axis=1)
        joint = {}
        for b in map(tuple, bits.astype(int)):
            joint[b] = joint.get(b, 0) + 1
        data[r["id"]] = joint
    return data


def load_counts(rows, counts):
    return {r["id"]: split_counts(r["path"], counts[r["id"]]) for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compiled", type=Path, nargs="+", required=True)
    ap.add_argument("--counts", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstraps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261126)
    a = ap.parse_args()
    if len(a.compiled)!=len(a.counts): ap.error("compiled/count file counts must match")
    result = {"blocks": []}
    for c, k in zip(a.compiled, a.counts):
        meta = json.loads(c.read_text())
        rows = meta["circuits"]
        raw = json.loads(k.read_text())
        if set(raw)!=set(r["id"] for r in rows): raise ValueError("counts/manifest ids differ")
        for r in rows:
            cts=raw[r["id"]]
            if sum(cts.values())!=r["shots"] or any(len(b)!=len(r["path"]) or set(b)-set("01") or type(n) is not int or n<0 for b,n in cts.items()):
                raise ValueError("malformed counts "+r["id"])
        data = load_counts(rows, raw)
        res = run(rows, data, a.bootstraps, a.seed)
        res["block"] = meta["block"]; res["plan_sha256"] = meta["plan_sha256"]
        res["inputs"] = {"counts_sha256":hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
                         "bootstraps":a.bootstraps,"bootstrap_seed":a.seed}
        res["alias"] = {e: alias_scan(rows, data, e) for e in res["edges"]}
        result["blocks"].append(res)
    a.out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps(result, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
