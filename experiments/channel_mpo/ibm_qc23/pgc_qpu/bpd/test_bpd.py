"""Offline tests for the PGC Phase 6 BPD study (linearised simulation; no hardware, no IBM services)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bpd_study as b  # noqa: E402

SMALL = dict(b.DEFAULT_CFG, splits=2, trials_per_split=20, circuit_budgets=[8, 16], seconds_budgets=[3.2, 6.4])


@pytest.fixture(scope="module")
def pool():
    return b.build_pool()


@pytest.fixture(scope="module")
def ctx(pool):
    rng = np.random.default_rng(123)
    design, held = b.split_design_held(pool, rng, 0.25)
    obs = np.concatenate([pool["circuits"][c]["obs"] for c in design]).astype(int)
    grp = b.fit_groups(pool, obs)
    cg = b.circuit_groups(pool, grp, design)
    Vw, _ = b.weak_subspace(pool)
    first, uni = b.fixed_orders(pool)
    return {"design": design, "held": held, "grp": grp, "cg": cg, "Vw": Vw, "first": first, "uni": uni, "P0": b.prior_precision()}


def _rank(M, rel=1e-6):
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > s[0] * rel).sum())


def test_receipt_jacobian_shape_and_ranks(pool):
    res = json.loads(b.RECEIPT.read_text())["result"]
    J = np.array([r["jacobian"] for r in res["rows"]])
    assert J.shape == (64, 14)
    assert _rank(J[:, :6]) == 6          # all-sites restricted model
    assert _rank(J) == 12                # all-sites expanded model
    assert np.allclose(pool["J"][:64], J)
    assert len(res["parameters"]) == 14 and pool["labels"] == [p["label"] for p in res["parameters"]]


def test_pool_composition_and_costs(pool):
    fam = [c["family"] for c in pool["circuits"]]
    assert fam.count("kicked") == 8 and fam.count("cz") == 128 and fam.count("spam") == 4
    assert pool["J"].shape == (196, 14)
    kicked = [c for c in pool["circuits"] if c["family"] == "kicked"]
    assert all(len(c["obs"]) == 8 for c in kicked)
    assert math.isclose(kicked[0]["cost"], 2048 * (0.00025 + 1e-5) + 0.1)
    ramsey = [c for c in pool["circuits"] if c["family"] != "kicked"]
    camp = json.loads(b.CAMPAIGN.read_text())
    assert math.isclose(sum(c["cost"] for c in ramsey), camp["total_estimated_seconds"])


def test_ramsey_jacobian_matches_analytic_and_sx_zero(pool):
    n_k = pool["n_kicked_obs"]
    JR = pool["J"][n_k:]
    assert np.all(JR[:, :2] == 0)
    rm = pool["ramsey_meta"]
    th = pool["theta0"]
    for i in (3, 17, 60, 100):
        if rm["is_spam"][i]:
            continue
        phi = rm["m"][i] * (th[rm["ip"][i]] + rm["sgn"][i] * th[rm["iz"][i]] + rm["state"][i] * math.pi)
        d = -math.sin(phi) if rm["is_x"][i] else math.cos(phi)
        assert math.isclose(JR[i, rm["ip"][i]], d * rm["m"][i], abs_tol=1e-6)
        assert math.isclose(JR[i, rm["iz"][i]], d * rm["m"][i] * rm["sgn"][i], abs_tol=1e-6)
    spam = [i for i in range(len(rm["m"])) if rm["is_spam"][i]]
    assert np.all(JR[spam] == 0)


def test_whole_pool_is_full_rank_and_weak_subspace(pool, ctx):
    assert _rank(pool["J"]) == 14
    Vw = ctx["Vw"]
    assert Vw.shape == (14, 4)
    assert np.allclose(Vw.T @ Vw, np.eye(4), atol=1e-10)
    assert np.all(Vw[:2] == 0)
    labels = pool["labels"]
    energy = np.diag(Vw @ Vw.T)
    assert energy[labels.index("Z_22_23_23")] > 0.99 and energy[labels.index("Z_4_5_5")] > 0.99


def test_nmf_groups_are_deterministic_and_nonnegative(pool, ctx):
    obs = np.arange(pool["J"].shape[0])
    g1 = b.fit_groups(pool, obs)
    g2 = b.fit_groups(pool, obs)
    assert np.array_equal(g1["H"], g2["H"]) and g1["G"] == g2["G"]
    assert (g1["H"] >= 0).all() and (g1["W"] >= 0).all()
    assert g1["rel_error"] <= 0.12
    assert np.allclose(np.linalg.norm(g1["H"], axis=1), 1.0)


BUDGETS = [{"mode": "circuits", "value": 8}, {"mode": "circuits", "value": 16}, {"mode": "circuits", "value": 32},
           {"mode": "seconds", "value": 3.2}, {"mode": "seconds", "value": 6.4}, {"mode": "seconds", "value": 12.8}]


@pytest.mark.parametrize("budget", BUDGETS, ids=lambda x: f"{x['mode']}{x['value']}")
def test_selection_respects_budget(pool, ctx, budget):
    sels = b.make_selections(pool, ctx["design"], budget, ctx["P0"], ctx["Vw"], ctx["grp"], ctx["cg"], ctx["first"], ctx["uni"])
    assert set(sels) == set(b.STRATEGIES) - {"random"}
    design = set(ctx["design"])
    for name, sel in sels.items():
        assert len(sel) == len(set(sel)), name
        assert set(sel) <= design, name
        cost = float(pool["cost"][sel].sum())
        if budget["mode"] == "circuits":
            assert len(sel) <= budget["value"], name
        else:
            assert cost <= budget["value"] + 1e-9, name
            assert len(sel) > 0, name


def test_dopt_beats_random_on_logdet(pool, ctx):
    P0 = ctx["P0"]
    design = ctx["design"]
    rng = np.random.default_rng(5)
    for k in (8, 16, 32):
        budget = {"mode": "circuits", "value": k}
        dsel = b.greedy_select(pool, design, budget, P0, b.score_dopt(pool))
        assert len(dsel) == k
        ld = np.linalg.slogdet(P0 + pool["F"][dsel].sum(0))[1]
        rand = []
        for _ in range(200):
            sel = list(rng.choice(design, size=k, replace=False))
            rand.append(np.linalg.slogdet(P0 + pool["F"][sel].sum(0))[1])
        assert ld > np.mean(rand) + 3 * np.std(rand), (k, ld, np.mean(rand))
        assert ld > max(rand)


def test_dopt_greedy_gains_are_nonincreasing(pool, ctx):
    P0, design = ctx["P0"], ctx["design"]
    sel = b.greedy_select(pool, design, {"mode": "circuits", "value": 20}, P0, b.score_dopt(pool))
    A, prev, gains = P0.copy(), None, []
    for c in sel:
        g = np.linalg.slogdet(A + pool["F"][c])[1] - np.linalg.slogdet(A)[1]
        gains.append(g)
        A = A + pool["F"][c]
    assert all(gains[i + 1] <= gains[i] + 1e-9 for i in range(len(gains) - 1))


def test_study_strategies_never_exceed_matched_cost():
    pool, Vw, S, acc, paired, budgets = b.run_study(SMALL, "linear", verbose=False)
    res = b.summarise(acc, paired, budgets)
    for bk, rb in res.items():
        mode, val = bk.split("_")
        for st in b.STRATEGIES:
            r = rb[st]
            if mode == "seconds":
                assert r["max_cost_s"] <= float(val) + 1e-9, (bk, st)
            else:
                assert r["n_circuits"] <= float(val) + 1e-9, (bk, st)
        assert rb["no_data"]["cost_s"] == 0.0
        assert rb["dopt"]["logdet_fim_ridge"] > rb["random"]["logdet_fim_ridge"]


def test_seeds_reproduce_and_differ():
    r1 = b.run_full(SMALL, verbose=False, with_exact=False)
    r2 = b.run_full(SMALL, verbose=False, with_exact=False)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    r3 = b.run_full(dict(SMALL, seed=SMALL["seed"] + 1), verbose=False, with_exact=False)
    assert json.dumps(r1["results"], sort_keys=True) != json.dumps(r3["results"], sort_keys=True)


def test_recovery_improves_with_informative_design(pool):
    pool, Vw, S, acc, paired, budgets = b.run_study(SMALL, "linear", verbose=False)
    res = b.summarise(acc, paired, budgets)["seconds_6.4"]
    assert res["dopt"]["weak_recovery_rate"] > res["random"]["weak_recovery_rate"] + 0.3
    assert res["dopt"]["heldout_rmse"] < res["random"]["heldout_rmse"]
    assert res["no_data"]["weak_recovery_rate"] < 0.2


def test_committed_results_are_consistent():
    path = HERE / "bpd_results.json"
    if not path.exists():
        pytest.skip("bpd_results.json not generated")
    out = json.loads(path.read_text())
    assert out["label"].startswith("OFFLINE LINEARISED SIMULATION")
    assert out["pool"]["kicked_rank_restricted_6"] == 6 and out["pool"]["kicked_rank_expanded_14"] == 12
    assert out["config"]["splits"] * out["config"]["trials_per_split"] >= 200
    for bk, rb in out["results"].items():
        mode, val = bk.split("_")
        for st in b.STRATEGIES:
            if mode == "seconds":
                assert rb[st]["max_cost_s"] <= float(val) + 1e-9
            else:
                assert rb[st]["n_circuits"] <= float(val) + 1e-9
