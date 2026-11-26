"""Offline tests for PGC-QPU02: exact checks, noisy-null behaviour, Lean mirror and proposal shape. No IBM access."""
import json, pathlib, sys
import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qpu02_sim as Q  # noqa: E402


def test_all_exact_checks_pass():
    checks, _ = Q.exact_checks()
    assert all(checks.values()), [k for k, v in checks.items() if not v]


def test_intact_response_zero_but_reachable_and_coherent():
    m = Q.metrics("intact")
    assert Q.reachability("intact")
    assert abs(m["influence_basis_0_vs_1"]) < 1e-12 and abs(m["coherence_witness"] - 1) < 1e-12


@pytest.mark.parametrize("variant", ["path_a_only", "path_b_only", "reset_a", "reset_b"])
def test_cut_restores_unit_response(variant):
    assert abs(abs(Q.metrics(variant)["influence_basis_0_vs_1"]) - 1) < 1e-12


def test_dephasing_separates_coherence_from_response():
    m = Q.metrics("dephase_a")
    assert abs(m["influence_basis_0_vs_1"]) < 1e-12 and abs(m["coherence_witness"]) < 1e-12
    assert m["ghz_fidelity_witness_F"] <= 0.5 + 1e-12      # no tripartite entanglement certified


def test_diagonal_coherent_errors_do_not_create_z_influence():
    for noise in ({"zz_coherent": 0.2}, {"crosstalk_zz_to_target": 0.1}):
        assert abs(Q.metrics("intact", noise)["influence_basis_0_vs_1"]) < 1e-12


def test_relaxation_creates_apparent_influence_matched_by_echo_control():
    n = {"t1_gamma_per_cx": 0.004}
    a, e = Q.metrics("intact", n)["influence_basis_0_vs_1"], Q.metrics("echo_intact", n)["influence_basis_0_vs_1"]
    assert a > 0.01 and e > 0.005 and a > e                # state-dependent T1 mimics influence; depth control does not remove it


def test_readout_and_depolarizing_do_not_create_influence():
    assert abs(Q.metrics("intact", None, 0.02, 0.05)["influence_basis_0_vs_1"]) < 1e-12
    assert abs(Q.metrics("intact", {"p2": 0.02})["influence_basis_0_vs_1"]) < 1e-12


def test_context_robustness():
    for ctx in np.ndindex(2, 2, 2):
        assert abs(Q.metrics("intact", ctx=tuple(ctx))["influence_basis_0_vs_1"]) < 1e-12


def test_lean_mirror():
    assert Q.lean_mirror()


def test_plan_shape():
    p = json.loads((HERE / "qpu02_plan.json").read_text())
    assert p["circuits"] == 30 and len({r["id"] for r in p["rows"]}) == 30
    assert all(r["shots"] == 2048 for r in p["rows"])
    intact = next(r for r in p["rows"] if r["variant"] == "intact")
    echo = next(r for r in p["rows"] if r["variant"] == "echo_intact")
    assert echo["native_gate_counts"]["cz"] > intact["native_gate_counts"]["cz"] >= 4   # echo control keeps its CNOTs after compilation
