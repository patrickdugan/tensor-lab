"""Generate receipts/qc26a_identify_synthetic.json: run qc26a_identify on SYNTHETIC counts only.

  python qc26a_identify_receipts.py [--out receipts/qc26a_identify_synthetic.json] [--trials 96]
         [--ensemble-bootstraps 200] [--profile-trials 24] [--workers 16] [--full-bootstraps 400]

Datasets (no hardware, no IBM access):
  A  injected: draw-for-draw port of JS trial 0 (LCG seed 260126, JS truth)
  B  null: draw-for-draw port of JS trial 48 (first null trial; SX/CZ truth set to zero)
  C  injected, conditional phase of edge 4->5 set near pi/2 (alias injection)
  D  injected, SX offset of site 5 set to 0.13 rad (outside the +-0.1 design bound)
  Ensemble: --trials independent datasets per arm (numpy generator, same truth) fitted with bootstraps (and
  profile intervals on a subset): bias, empirical SD versus Wald / bootstrap SE, interval coverage, false-alarm
  rates of the diagnostics. Coverage is reported as realised, not tuned.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qc26a_identify as Q  # noqa: E402

_PLAN = None


def _plan():
    global _PLAN
    if _PLAN is None:
        _PLAN = Q.load_plan()
    return _PLAN


def js_lcg_datasets(plan):
    """Counts of JS trial 0 (injected) and JS trial 48 (null), reproduced exactly with the same LCG stream."""
    rng = Q.LCG(260126)
    inj0 = Q.synthetic_counts(plan, rng=rng)
    for _ in range(47):
        Q.synthetic_counts(plan, rng=rng)
    null0 = Q.synthetic_counts(plan, rng=rng, zero=True)
    return inj0, null0


def truth_values(zero):
    t = {}
    for name, ent, kind, fam, key in PARAMS:
        v = Q.JS_TRUTH[fam][int(key) if fam == "sx" else key]
        t[name] = 0.0 if (zero and fam != "local") else float(v)
    return t


_L = Q.Layout(Q.load_plan())
PARAMS = Q.parameter_table(_L)


def _trial(args):
    arm, i, boots, do_profile, seed = args
    plan = _plan()
    rng = np.random.default_rng([seed, 0 if arm == "injected" else 1, i])
    counts = Q.synthetic_counts(plan, np_rng=rng, zero=(arm == "null"))
    res = Q.identify(counts, plan=plan, bootstraps=boots, seed=int(rng.integers(1 << 30)), profile=do_profile,
                     alias=True, synthetic=False)
    out = dict(arm=arm, i=i, est=res["fit_parameters"], js=res["js_mirror_estimates"],
               wald={n: res["identifiability"]["wald_se"][n] for n, *_ in PARAMS},
               status={n: res["decision"]["per_parameter"][n]["status"] for n, *_ in PARAMS},
               failed={n: res["decision"]["per_parameter"][n]["failed_gates"] for n, *_ in PARAMS},
               diag={k: res["diagnostics"][k]["flag"] for k in ("crosstalk", "leakage_population", "relaxation",
                                                                 "quadrature", "drop_longest")},
               diag_components=dict(
                   sx_pair_correlation=res["diagnostics"]["crosstalk"]["sx_block_pair_correlation"]["flag"],
                   cz_control_target_cmh=res["diagnostics"]["crosstalk"]["cz_control_target_association_cmh"]["flag"],
                   control_marginal_vs_spam=res["diagnostics"]["crosstalk"]["control_marginal_vs_spam"]["flag"],
                   prep_phase_context=res["diagnostics"]["crosstalk"]["prep_phase_block_vs_isolated"]["flag"],
                   cz_nonlinear=any(v["flag_nonlinear"] for v in res["diagnostics"]["quadrature"]["cz"].values()),
                   cz_amplitude=any(v["flag_amplitude"] or v["flag_unphysical_radius"] for v in res["diagnostics"]["quadrature"]["cz"].values()),
                   sx_quadrature=any(v["flag"] for v in res["diagnostics"]["quadrature"]["sx"].values())),
               deviance=res["fit"]["deviance"], seconds=res["seconds"])
    if boots:
        bp = res["bootstrap"]["parameters"]
        out["boot_se"] = {n: bp[n]["se"] for n, *_ in PARAMS}
        out["boot_ci"] = {n: bp[n]["ci95"] for n, *_ in PARAMS}
        out["boot_bias"] = {n: bp[n]["bias"] for n, *_ in PARAMS}
        out["boot_flags"] = res["bootstrap"]["flag_fractions_any_entity"]
    if do_profile:
        out["profile_ci"] = {n: res["identifiability"]["profiles"][n]["profile_ci95"] for n, *_ in PARAMS}
        out["profile_flat"] = {n: res["identifiability"]["profiles"][n]["flat"] for n, *_ in PARAMS}
        out["profile_multimodal"] = {n: res["identifiability"]["profiles"][n]["multimodal"] for n, *_ in PARAMS}
    return out


def _summ(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), min=float(x.min()), max=float(x.max()))


def ensemble_summary(trials, arm):
    ts = [t for t in trials if t["arm"] == arm]
    truth = truth_values(arm == "null")
    out = dict(n_trials=len(ts), parameters={})
    for name, ent, kind, fam, key in PARAMS:
        est = np.array([t["est"][name] for t in ts])
        js = np.array([t["js"][name] for t in ts])
        tr = truth[name]
        z = lambda ci: float(np.mean([(c[0] <= tr <= c[1]) for c in ci]))
        e = dict(truth=tr, mean_estimate=float(est.mean()), bias=float(est.mean() - tr),
                 bias_se=float(est.std(ddof=1) / math.sqrt(len(est))),
                 empirical_sd=float(est.std(ddof=1)), rmse=float(np.sqrt(np.mean((est - tr) ** 2))),
                 js_estimator_bias=float(js.mean() - tr), js_estimator_rmse=float(np.sqrt(np.mean((js - tr) ** 2))),
                 mean_wald_se=float(np.mean([t["wald"][name] for t in ts])),
                 wald_ci_coverage=float(np.mean([abs(t["est"][name] - tr) <= Q.Z975 * t["wald"][name] for t in ts])),
                 sign_correct_fraction=None if tr == 0 else float(np.mean(np.sign(est) == np.sign(tr))),
                 )
        st_ts = [t for t in ts if "profile_ci" in t] or ts   # the decision rule needs the profile interval
        e["decision_trials"] = len(st_ts)
        e["nonzero_status_fraction"] = float(np.mean([t["status"][name].startswith("NONZERO") for t in st_ts]))
        e["inconclusive_flag_fraction"] = float(np.mean([t["status"][name] == "INCONCLUSIVE_FLAGS" for t in st_ts]))
        if "boot_se" in ts[0]:
            bse = np.array([t["boot_se"][name] for t in ts])
            e.update(mean_bootstrap_se=float(bse.mean()), bootstrap_se_over_empirical_sd=float(bse.mean() / est.std(ddof=1)),
                     bootstrap_percentile_ci_coverage=z([t["boot_ci"][name] for t in ts]),
                     bootstrap_ci_excludes_zero_fraction=float(np.mean([not (t["boot_ci"][name][0] <= 0 <= t["boot_ci"][name][1])
                                                                       for t in ts])),
                     mean_bootstrap_bias=float(np.mean([t["boot_bias"][name] for t in ts])))
        pt = [t for t in ts if "profile_ci" in t]
        if pt:
            e.update(profile_trials=len(pt), profile_ci_coverage=z([t["profile_ci"][name] for t in pt]),
                     profile_ci_excludes_zero_fraction=float(np.mean([not (t["profile_ci"][name][0] <= 0 <= t["profile_ci"][name][1])
                                                                     for t in pt])),
                     profile_flat_fraction=float(np.mean([t["profile_flat"][name] for t in pt])),
                     profile_multimodal_fraction=float(np.mean([t["profile_multimodal"][name] for t in pt])))
        out["parameters"][name] = e
    ps = out["parameters"]
    zero_params = [n for n, v in ps.items() if v["truth"] == 0.0]
    nz_params = [n for n, v in ps.items() if v["truth"] != 0.0]
    agg = {}
    for key in ("wald_ci_coverage", "bootstrap_percentile_ci_coverage", "profile_ci_coverage"):
        vals = [v[key] for v in ps.values() if key in v]
        if vals:
            agg[key + "_mean_over_parameters"] = float(np.mean(vals))
            agg[key + "_min_over_parameters"] = float(np.min(vals))
    agg["nominal_coverage"] = 0.95
    agg["false_positive_rate_bootstrap_ci_excludes_zero_on_zero_truth"] = (
        float(np.mean([ps[n]["bootstrap_ci_excludes_zero_fraction"] for n in zero_params]))
        if zero_params and "bootstrap_ci_excludes_zero_fraction" in ps[zero_params[0]] else None)
    agg["power_bootstrap_ci_excludes_zero_on_nonzero_truth"] = (
        float(np.mean([ps[n]["bootstrap_ci_excludes_zero_fraction"] for n in nz_params]))
        if nz_params and "bootstrap_ci_excludes_zero_fraction" in ps[nz_params[0]] else None)
    agg["diagnostic_flag_rates"] = {k: float(np.mean([t["diag"][k] for t in ts])) for k in ts[0]["diag"]}
    agg["diagnostic_component_flag_rates"] = {k: float(np.mean([t["diag_components"][k] for t in ts])) for k in ts[0]["diag_components"]}
    agg["diagnostic_flag_any_rate"] = float(np.mean([any(t["diag"].values()) for t in ts]))
    st = {}
    st_ts = [t for t in ts if "profile_ci" in t] or ts
    agg["decision_trials"] = len(st_ts)
    for t in st_ts:
        for s in t["status"].values():
            st[s] = st.get(s, 0) + 1
    agg["decision_status_counts_over_trials_and_parameters"] = st
    gates = {}
    for t in st_ts:
        for fl in t["failed"].values():
            for g in fl:
                gates[g] = gates.get(g, 0) + 1
    agg["failed_gate_counts"] = gates
    out["aggregate"] = agg
    return out


def _fit_only(args):
    arm, i, seed = args
    plan = _plan()
    rng = np.random.default_rng([seed, 7, 0 if arm == "injected" else 1, i])
    data = Q.parse_counts(plan, Q.synthetic_counts(plan, np_rng=rng, zero=(arm == "null")))
    f = Q.fit_all(_L, data)
    w, _ = Q.wald_ses(_L, _L.full, f.th, f.Ns)
    js = Q.js_mirror(plan, data)
    est = {n: float(f.th[_L.index[n]]) for n, *_ in PARAMS}
    jsv = {}
    for name, ent, kind, fam, key in PARAMS:
        jsv[name] = (js["sx"][int(key)]["offset_rad"] if fam == "sx" else
                     js["cz"][key]["zz_phase_rad"] if fam == "cz" else js["cz"][key]["local_phase_rad"])
    return dict(arm=arm, est=est, wald={n: w[n] for n, *_ in PARAMS}, js=jsv)


def calibration_summary(rows, arm):
    ts = [r for r in rows if r["arm"] == arm]
    truth = truth_values(arm == "null")
    out = {}
    for name, *_ in PARAMS:
        e = np.array([t["est"][name] for t in ts])
        w = np.array([t["wald"][name] for t in ts])
        j = np.array([t["js"][name] for t in ts])
        tr = truth[name]
        z = (e - tr) / w
        out[name] = dict(truth=tr, bias=float(e.mean() - tr), bias_se=float(e.std(ddof=1) / math.sqrt(len(e))),
                         empirical_sd=float(e.std(ddof=1)), mean_wald_se=float(w.mean()),
                         wald_se_over_empirical_sd=float(w.mean() / e.std(ddof=1)), z_sd=float(z.std(ddof=1)),
                         wald_ci_coverage=float(np.mean(np.abs(z) <= Q.Z975)),
                         js_estimator_bias=float(j.mean() - tr), js_estimator_sd=float(j.std(ddof=1)),
                         js_estimator_rmse=float(np.sqrt(np.mean((j - tr) ** 2))))
    cov = [v["wald_ci_coverage"] for v in out.values()]
    return dict(n_trials=len(ts), parameters=out, wald_coverage_mean=float(np.mean(cov)), wald_coverage_min=float(np.min(cov)),
                coverage_standard_error_per_parameter=float(math.sqrt(0.95 * 0.05 / len(ts))))


def short_report(res):
    """The decision-relevant slice of a full report, for the alias/boundary datasets."""
    return dict(
        fit_parameters=res["fit_parameters"], flags_estimate=res["flags_estimate"],
        bootstrap_flag_fractions=res["bootstrap"]["flag_fractions"],
        alias_cz={k: dict(alias_margin_rad=v["alias_margin_rad"], alias_ambiguous=v["alias_ambiguous"],
                          lattice_candidates=v["lattice_candidates"][:4]) for k, v in res["identifiability"]["alias"]["cz"].items()},
        alias_sx={k: dict(best_offset_inside_bound=v["best_offset_inside_bound"], global_minimum_outside_bound=v["global_minimum_outside_bound"],
                          reflection_alias_offset=v["reflection_alias_offset"], reflection_alias_present=v["reflection_alias_present"],
                          competing_minima=v["competing_minima"][:4]) for k, v in res["identifiability"]["alias"]["sx"].items()},
        decision={n: dict(status=v["status"], failed_gates=v["failed_gates"]) for n, v in res["decision"]["per_parameter"].items()},
        bootstrap_ci={n: res["bootstrap"]["parameters"][n]["ci95"] for n, *_ in PARAMS})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "receipts" / "qc26a_identify_synthetic.json"))
    ap.add_argument("--trials", type=int, default=96)
    ap.add_argument("--ensemble-bootstraps", type=int, default=200)
    ap.add_argument("--profile-trials", type=int, default=24)
    ap.add_argument("--full-bootstraps", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--calibration-trials", type=int, default=480)
    a = ap.parse_args(argv)
    t0 = time.time()
    plan = _plan()
    inj0, null0 = js_lcg_datasets(plan)
    syn = json.load(open(Q.SYN_PATH, encoding="utf-8"))
    receipt = dict(
        label="SYNTHETIC ONLY; produced by qc26a_identify.py on generated counts; no IBM access, no hardware data, "
              "no hardware claim",
        versions=dict(identify=Q.VERSION, numpy=np.__version__, scipy=__import__("scipy").__version__,
                      python=sys.version.split()[0]),
        plan_sha256=Q.canonical_sha256(plan),
        settings=vars(a))
    full = {}
    for key, counts, desc in (("A_injected_js_trial0", inj0, "draw-for-draw port of JS injected trial 0"),
                              ("B_null_js_trial48", null0, "draw-for-draw port of JS first null trial")):
        res = Q.identify(counts, plan=plan, bootstraps=a.full_bootstraps, seed=11, profile=True, alias=True,
                         synthetic=Q.SYN_PATH, progress=True)
        res["description"] = desc
        full[key] = res
        print(key, "done", round(time.time() - t0), "s", file=sys.stderr, flush=True)
    # Python port vs the committed JS trials (exact LCG stream): every estimate must agree
    js_ports = {}
    for key, counts, idx in (("A_injected_js_trial0", inj0, 0), ("B_null_js_trial48", null0, 48)):
        data = Q.parse_counts(plan, counts)
        js = Q.js_mirror(plan, data)
        tr = syn["trials"][idx]
        diffs = [abs(js["sx"][int(p)]["offset_rad"] - v) for p, v in tr["sx"].items()]
        diffs += [abs(js["cz"][k]["zz_phase_rad"] - v) for k, v in tr["cz"].items()]
        diffs += [abs(js["cz"][k]["local_phase_rad"] - v) for k, v in tr["local"].items()]
        js_ports[key] = dict(committed_trial_index=idx, max_abs_difference_to_committed_js_estimate=float(max(diffs)))
    receipt["python_port_vs_committed_js_trials"] = js_ports
    receipt["datasets"] = full

    T = copy.deepcopy(Q.JS_TRUTH)
    T["cz"]["4_5"] = math.pi / 2 - 0.03
    T["local"]["4_5"] = 0.01
    c_alias = Q.synthetic_counts(plan, truth=T, np_rng=np.random.default_rng(5))
    r = Q.identify(c_alias, plan=plan, bootstraps=100, seed=2, profile=False, alias=True, synthetic=Q.SYN_PATH)
    receipt["C_alias_injection_edge_4_5_delta_pi_over_2_minus_0.03"] = dict(
        truth=dict(delta=T["cz"]["4_5"], alpha_target=T["local"]["4_5"]), **short_report(r))
    T = copy.deepcopy(Q.JS_TRUTH)
    T["sx"][5] = 0.13
    c_bnd = Q.synthetic_counts(plan, truth=T, np_rng=np.random.default_rng(6))
    r = Q.identify(c_bnd, plan=plan, bootstraps=100, seed=2, profile=False, alias=True, synthetic=Q.SYN_PATH)
    receipt["D_boundary_injection_site5_offset_0.13"] = dict(truth=dict(offset_site5=0.13), **short_report(r))
    print("C, D done", round(time.time() - t0), "s", file=sys.stderr, flush=True)

    jobs = []
    for arm in ("injected", "null"):
        for i in range(a.trials):
            jobs.append((arm, i, a.ensemble_bootstraps, i < a.profile_trials, a.seed))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        trials = list(ex.map(_trial, jobs, chunksize=1))
    receipt["ensemble"] = dict(
        description=f"{a.trials} independent synthetic datasets per arm (numpy generator, same truth as the JS "
                    f"synthetic), full joint refit with {a.ensemble_bootstraps} joint-bitstring bootstrap replicates "
                    f"each; profile intervals on the first {a.profile_trials} trials of each arm",
        injected=ensemble_summary(trials, "injected"), null=ensemble_summary(trials, "null"),
        mean_seconds_per_trial=float(np.mean([t["seconds"] for t in trials])))
    cjobs = [(arm, i, a.seed) for arm in ("injected", "null") for i in range(a.calibration_trials)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        crows = list(ex.map(_fit_only, cjobs, chunksize=8))
    receipt["wald_calibration"] = dict(
        description="point-fit only (no bootstrap) on many independent synthetic datasets: bias, empirical SD versus "
                    "the Fisher/Wald SE from the Jacobian, Wald 95% coverage, and the JS estimator for comparison",
        injected=calibration_summary(crows, "injected"), null=calibration_summary(crows, "null"))
    receipt["total_seconds"] = time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(Q.jsonable(receipt), f, indent=1)
        f.write("\n")
    print("wrote", a.out, round(time.time() - t0), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
