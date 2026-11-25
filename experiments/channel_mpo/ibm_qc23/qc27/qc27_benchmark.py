"""QC-27 validity sweeps and benchmark driver.  Writes qc27_bound_benchmark.json.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python qc27_benchmark.py all [--out PATH]
    python qc27_benchmark.py tables [--json PATH]     # markdown tables from the JSON
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qc27_bounds as B  # noqa: E402

NOISE = {
    "weak": dict(p1=0.002, g1=0.001, p2=0.01, g2=0.003),
    "strong": dict(p1=0.01, g1=0.005, p2=0.05, g2=0.01),
}
VARIANTS = ("old", "cone", "cone_tl", "minl1l2", "grouped", "surv")
REF_TOL = 1e-11   # tolerance for the float dense reference itself


def _row_checks(row):
    """Validity of every bound against the actual error (None when no reference)."""
    if row["actual"] is None:
        return None
    ok = {v: bool(row["actual"] <= row["bound_" + v] + REF_TOL) for v in VARIANTS}
    order = all(row["bound_" + a] >= row["bound_" + b] - 1e-15 * max(1.0, row["bound_" + a])
                for a, b in (("old", "cone"), ("cone", "cone_tl"), ("cone_tl", "minl1l2"),
                             ("minl1l2", "grouped"), ("grouped", "surv")))
    return dict(valid=ok, ordered=order)


def validity_sweep(n, depths, seeds, epss, noises, max_cone_ref, schro_every=0):
    rows = []
    cache = {}
    schro_dev = []
    t0 = time.time()
    for noise_name in noises:
        for depth in depths:
            for seed in seeds:
                circ = B.build_circuit(n, depth, seed, **NOISE[noise_name])
                sites, _ = B.cone_restrict(circ)
                if len(sites) > max_cone_ref:
                    continue
                ex, m = B.dense_heisenberg(circ, restrict=True)
                if schro_every and (seed % schro_every == 0):
                    sc, _ = B.dense_schrodinger(circ, restrict=(n > 8))
                    schro_dev.append(abs(sc - ex))
                for eps in epss:
                    res = B.propagate(circ, eps)
                    b = B.total_bounds(res["events"], n, res["rounding"])
                    row = dict(n=n, depth=depth, seed=seed, eps=eps, noise=noise_name,
                               cone_width=len(sites), exact=ex, expectation=res["expectation"],
                               actual=abs(ex - res["expectation"]), rounding=res["rounding"],
                               **{"bound_" + k: v for k, v in b.items()},
                               **B.ledger_stats(res["events"]))
                    row["checks"] = _row_checks(row)
                    rows.append(row)
    sub = [r for r in rows if r["n_events"] > 0]
    summ = dict(
        n=n, circuits_runs=len(rows), runs_with_discards=len(sub),
        valid_pass={v: sum(r["checks"]["valid"][v] for r in sub) for v in VARIANTS},
        ordered_pass=sum(r["checks"]["ordered"] for r in sub),
        max_actual_over_bound={v: max((r["actual"] / r["bound_" + v] for r in sub), default=None)
                               for v in VARIANTS},
        median_bound_over_actual={v: statistics.median([r["bound_" + v] / max(r["actual"], 1e-300)
                                                        for r in sub]) if sub else None
                                  for v in VARIANTS},
        schrodinger_vs_heisenberg_max_dev=max(schro_dev) if schro_dev else None,
        schrodinger_comparisons=len(schro_dev), wall_s=time.time() - t0)
    return dict(rows=rows, summary=summ)


def per_event_check(n, depths, seeds, epss, noise_names):
    """Exact per-event errors via dense Schroedinger snapshots vs per-event bounds
    (telescoping identity, and every event's bound)."""
    out = []
    worst_tele = 0.0
    n_ev = 0
    n_fail = {v: 0 for v in VARIANTS}
    n_group = 0
    n_group_fail = 0
    for nm in noise_names:
        for depth in depths:
            for seed in seeds:
                circ = B.build_circuit(n, depth, seed, **NOISE[nm])
                ex, _ = B.dense_heisenberg(circ, restrict=False)
                for eps in epss:
                    res = B.propagate(circ, eps, keep_discards=True)
                    if not res["events"]:
                        continue
                    errs = B.per_event_errors(circ, res)
                    tot = sum(e for e, _ in errs)
                    worst_tele = max(worst_tele, abs((ex - res["expectation"]) - tot))
                    for ev, (e, pg) in zip(res["events"], errs):
                        eb = B.event_bounds(ev, n)
                        n_ev += 1
                        for v in VARIANTS:
                            if abs(e) > eb[v] + 1e-12:
                                n_fail[v] += 1
                        for (m, k, l1, l2sq, sg) in ev.groups:
                            n_group += 1
                            bg = l1 if k == 0 else min(l1, B._cone_tl(k) * l2sq ** 0.5)
                            if abs(pg.get(m, 0.0)) > sg * bg + 1e-12:
                                n_group_fail += 1
    return dict(n=n, events_checked=n_ev, groups_checked=n_group, event_bound_failures=n_fail,
                group_survival_bound_failures=n_group_fail,
                max_telescoping_residual=worst_tele)


def bench_grid(ns, depths, epss, seeds, noises, max_cone_ref, ref_n_max=16):
    rows = []
    refcache = {}
    t0 = time.time()
    for n in ns:
        for noise_name in noises:
            for depth in depths:
                for seed in seeds:
                    key = (n, noise_name, depth, seed)
                    circ = B.build_circuit(n, depth, seed, **NOISE[noise_name])
                    sites, _ = B.cone_restrict(circ)
                    ex = None
                    t_ref = None
                    if n <= ref_n_max and len(sites) <= max_cone_ref:
                        t1 = time.time()
                        ex, _ = B.dense_heisenberg(circ, restrict=True)
                        t_ref = time.time() - t1
                    for eps in epss:
                        t1 = time.time()
                        res = B.propagate(circ, eps)
                        t_prop = time.time() - t1
                        b = B.total_bounds(res["events"], n, res["rounding"])
                        row = dict(n=n, depth=depth, seed=seed, eps=eps, noise=noise_name,
                                   cone_width=len(sites), expectation=res["expectation"],
                                   nterms_max=res["nterms_max"], t_prop=t_prop,
                                   rounding=res["rounding"],
                                   **{"bound_" + k: v for k, v in b.items()},
                                   **B.ledger_stats(res["events"]))
                        row["exact"] = ex
                        row["actual"] = None if ex is None else abs(ex - res["expectation"])
                        row["reference"] = (None if ex is None else
                                            "dense Heisenberg Pauli-vector on %d backward-cone sites"
                                            % len(sites))
                        row["t_ref"] = t_ref
                        row["checks"] = _row_checks(row)
                        rows.append(row)
    return rows, time.time() - t0


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarize_bench(rows):
    out = {}
    for n in sorted({r["n"] for r in rows}):
        rs = [r for r in rows if r["n"] == n and r["n_events"] > 0]
        d = dict(runs=len(rs), median_k=_med([r["k_median"] for r in rs]),
                 median_cone_width=_med([r["cone_width"] for r in rs]))
        for v in VARIANTS[1:]:
            d["median_old_over_" + v] = _med([r["bound_old"] / r["bound_" + v] for r in rs])
            d["min_old_over_" + v] = min([r["bound_old"] / r["bound_" + v] for r in rs], default=None)
        ref = [r for r in rs if r["actual"] is not None and r["actual"] > 0]
        d["runs_with_reference"] = len(ref)
        for v in VARIANTS:
            d["median_bound_over_actual_" + v] = _med([r["bound_" + v] / r["actual"] for r in ref])
        d["valid_pass_with_reference"] = {v: sum(bool(r["checks"]["valid"][v]) for r in
                                                 [q for q in rs if q["actual"] is not None])
                                          for v in VARIANTS}
        d["ordering_pass"] = sum(bool(r["checks"]["ordered"]) if r["checks"] else True for r in rs)
        d["median_sigma_min"] = _med([r["sigma_min"] for r in rs])
        out[str(n)] = d
    return out


def main(argv):
    cmd = argv[0] if argv else "all"
    args = argv[1:]

    def opt(name, default):
        return args[args.index(name) + 1] if name in args else default

    here = os.path.dirname(os.path.abspath(__file__))
    if cmd == "tables":
        render_tables(opt("--json", os.path.join(here, "qc27_bound_benchmark.json")))
        return
    out_path = opt("--out", os.path.join(here, "qc27_bound_benchmark.json"))
    t_start = time.time()
    quick = "--quick" in args
    data = dict(meta=dict(
        python=sys.version.split()[0], numpy=np.__version__,
        threads="OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s" % (os.environ.get("OMP_NUM_THREADS"),
                                                                  os.environ.get("OPENBLAS_NUM_THREADS")),
        noise_levels=NOISE, ref_tol=REF_TOL,
        circuit="brickwork 1D; per layer: random Clifford 1q (I/H/S/HS), RZ,RX small angle "
                "(theta_max 0.3), 1q noise, CNOT brickwork, per-qubit noise after each CNOT; "
                "noise = amplitude damping then depolarising; observable Z on the central qubit; "
                "input |0..0>; truncation: discard |c| < eps after every branching group.",
        reference_note="actual error = |exact - truncated| where exact is dense Heisenberg "
                       "Pauli-vector evolution with the true channels on the exact backward-cone "
                       "sites (n=8 additionally cross-checked against density-matrix Schroedinger "
                       "evolution with Kraus operators); no reference for n > 16."))
    # validity sweeps
    s8 = validity_sweep(8, [1, 2, 3, 4], range(3 if quick else 10), [3e-2, 1e-2, 3e-3, 1e-3, 1e-4],
                        ["weak", "strong"], 8, schro_every=2)
    data["validity_n8"] = s8
    s12 = validity_sweep(12, [2, 3, 4, 5], range(2 if quick else 5), [1e-2, 3e-3, 1e-3],
                         ["weak", "strong"], 10, schro_every=3)
    data["validity_n12"] = s12
    s16 = validity_sweep(16, [2, 3, 4, 5], range(2 if quick else 4), [1e-2, 3e-3, 1e-3],
                         ["weak", "strong"], 10, schro_every=0)
    data["validity_n16"] = s16
    data["per_event_n8"] = per_event_check(8, [2, 3, 4], range(2 if quick else 6),
                                           [1e-2, 1e-3], ["weak", "strong"])
    pp, gg = 0.1, 0.3
    ad = B.amp_damp_kraus(gg)
    dp = B.depol_kraus(pp)
    deph = [np.sqrt(0.8) * B._I, np.sqrt(0.2) * B._Z]
    data["kappa_choi"] = dict(
        depolarising_p0p1=B.kappa_choi(dp), amplitude_damping_g0p3=B.kappa_choi(ad),
        dephasing_q0p2=B.kappa_choi(deph),
        amp_then_dep=B.kappa_choi([D @ K for D in dp for K in ad]),
        dep_then_amp=B.kappa_choi([K @ D for D in dp for K in ad]),
        dep_then_amp_lower_bound_p_times_1_minus_g=pp * (1 - gg), p=pp, gamma=gg)
    ntr = 200 if quick else 1500
    data["survival_stress"] = dict(
        valid=[B.survival_stress(ntr, n=5, nops=10, seed=11), B.survival_stress(ntr, n=4, nops=14, seed=12)],
        negative_control_no_merge=B.survival_stress(ntr, n=5, nops=10, seed=11, neg="no_merge"),
        negative_control_keep_nonunital=B.survival_stress(ntr, n=5, nops=10, seed=11, neg="keep_nonunital"))
    # unrestricted n=12 dense check of the cone restriction (few shallow circuits)
    unr = []
    if not quick:
        for seed in range(2):
            circ = B.build_circuit(12, 2, seed, **NOISE["strong"])
            a, _ = B.dense_heisenberg(circ, restrict=True)
            t1 = time.time()
            b, _ = B.dense_heisenberg(circ, restrict=False)
            unr.append(dict(seed=seed, restricted=a, unrestricted=b, dev=abs(a - b),
                            t_unrestricted=time.time() - t1))
    data["cone_restriction_check_n12_unrestricted"] = unr
    # benchmark
    rows, wall = bench_grid([12, 16, 24, 32, 48], [4, 6, 8] if not quick else [4],
                            [1e-2, 3e-3, 1e-3] if not quick else [1e-2],
                            range(3), ["weak", "strong"] if not quick else ["weak"], 10)
    data["benchmark"] = dict(rows=rows, wall_s=wall, summary=summarize_bench(rows))
    try:
        import qc27_exact
        data["exact"] = qc27_exact.exact_report()
        data["failed_hypotheses"] = qc27_exact.failed_hypotheses()
    except ImportError:
        pass
    data["meta"]["total_wall_s"] = time.time() - t_start
    with open(out_path, "w") as f:
        json.dump(data, f, indent=1, default=float)
    print("wrote", out_path, "wall %.1f s" % data["meta"]["total_wall_s"])


def _fmt(x, p=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return ("%." + str(p) + "g") % x
    return str(x)


def render_tables(path):
    d = json.load(open(path))
    lines = []
    for key in ("validity_n8", "validity_n12", "validity_n16"):
        s = d[key]["summary"]
        lines.append("| %s | runs with discards | pass old | pass cone | pass cone_tl | pass min(l1,l2) | pass grouped | pass surv | ordering | max actual/bound (surv) | median surv/actual |" % key)
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        vp = s["valid_pass"]
        lines.append("| n=%d | %d | %d | %d | %d | %d | %d | %d | %d | %s | %s |" % (
            s["n"], s["runs_with_discards"], vp["old"], vp["cone"], vp["cone_tl"], vp["minl1l2"],
            vp["grouped"], vp["surv"], s["ordered_pass"], _fmt(s["max_actual_over_bound"]["surv"]),
            _fmt(s["median_bound_over_actual"]["surv"])))
        lines.append("")
    print("\n".join(lines))
    sm = d["benchmark"]["summary"]
    print("| n | runs | median cone width | median k at discard | median old/cone | old/cone_tl | old/min(l1,l2) | old/grouped | old/surv | min old/surv | refs | median old/actual | median surv/actual |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n, s in sm.items():
        print("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %d | %s | %s |" % (
            n, s["runs"], _fmt(s["median_cone_width"]), _fmt(s["median_k"]),
            _fmt(s["median_old_over_cone"]), _fmt(s["median_old_over_cone_tl"]),
            _fmt(s["median_old_over_minl1l2"]), _fmt(s["median_old_over_grouped"]),
            _fmt(s["median_old_over_surv"]), _fmt(s["min_old_over_surv"]),
            s["runs_with_reference"], _fmt(s.get("median_bound_over_actual_old")),
            _fmt(s.get("median_bound_over_actual_surv"))))


if __name__ == "__main__":
    main(sys.argv[1:])
