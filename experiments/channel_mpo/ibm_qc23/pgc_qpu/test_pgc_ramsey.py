"""Offline tests for the PGC-QPU01 Ramsey estimator and the gated submit adapter. No IBM account or network."""
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "qc26"))
sys.path.insert(0, str(HERE.parent))
import ramsey_identify as R  # noqa: E402

CAMPAIGN = json.loads((HERE / "ramsey_campaign.json").read_text())
TRUTH = CAMPAIGN["truth"]
BLOCK = CAMPAIGN["blocks"][0]
ROWS = BLOCK["rows"]


# ------------------------------------------------------------------ model convention against the statevector
def probe_expectation(row, ac, at, d):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    import qc26a_compile as qc
    q = QuantumCircuit(2)
    if row["state"] == 1:
        q.x(0)
    qc.h_native(q, 1)
    for _ in range(row["m"]):
        q.cz(0, 1); q.rz(ac, 0); q.rz(at, 1); q.rzz(d, 0, 1)
    if row["axis"] == "Y":
        q.rz(-math.pi / 2, 1)
    qc.h_native(q, 1)
    p = Statevector(q).probabilities([1])
    return p[0] - p[1]


@pytest.mark.parametrize("state", [0, 1])
@pytest.mark.parametrize("m", [0, 1, 2, 8])
@pytest.mark.parametrize("axis", ["X", "Y"])
def test_ideal_model_matches_statevector(state, m, axis):
    at, d, ac = 0.31, -0.12, 0.2
    s = at + (1 if state == 0 else -1) * d
    phi = m * (s + state * math.pi)
    want = math.cos(phi) if axis == "X" else math.sin(phi)
    got = probe_expectation({"state": state, "m": m, "axis": axis}, ac, at, d)
    assert abs(got - want) < 1e-12


def test_forward_blind_to_control_local_phase_and_reverse_sees_it():
    base = [probe_expectation({"state": b, "m": m, "axis": a}, 0.0, 0.05, 0.1) for b in (0, 1) for m in (1, 2, 8) for a in "XY"]
    shifted = [probe_expectation({"state": b, "m": m, "axis": a}, 0.4, 0.05, 0.1) for b in (0, 1) for m in (1, 2, 8) for a in "XY"]
    assert max(abs(x - y) for x, y in zip(base, shifted)) < 1e-12   # control Z term is invisible to a Z-eigenstate control
    # reversing roles makes the old control the probe, so its local phase appears as the probe slope
    rev = [probe_expectation({"state": b, "m": 2, "axis": "X"}, 0.05, 0.4, 0.1) for b in (0, 1)]
    ref = [probe_expectation({"state": b, "m": 2, "axis": "X"}, 0.05, 0.0, 0.1) for b in (0, 1)]
    assert max(abs(x - y) for x, y in zip(rev, ref)) > 0.1


# ------------------------------------------------------------------ estimator
@pytest.fixture(scope="module")
def synth():
    return R.synthetic_counts(ROWS, TRUTH, np.random.default_rng(11))


def test_recovers_injected_parameters(synth):
    est, _ = R.estimate(ROWS, synth)
    for edge, v in est.items():
        t = TRUTH[edge]
        assert abs(v["delta"] - t["delta"]) < 0.012
        for site, a in v["alpha"].items():
            assert abs(a - t["local"][site]) < 0.02
        assert abs(v["delta_gap"]) < 0.02


def test_null_has_no_effect():
    null = {e: {"local": {k: 0.0 for k in x["local"]}, "delta": 0.0} for e, x in TRUTH.items()}
    data = R.synthetic_counts(ROWS, null, np.random.default_rng(3), offsets=(0.0, 0.0), rev_off=0.0)
    est, _ = R.estimate(ROWS, data)
    for v in est.values():
        assert abs(v["delta"]) < 0.012 and all(abs(a) < 0.02 for a in v["alpha"].values())


def test_odd_repetition_resolves_even_only_alias(synth):
    scan = R.alias_scan(ROWS, synth, "16_23")
    assert abs(scan["even_only_0_2_8"]["delta_chi2_alias"]) < 1e-6          # exact alias without odd m
    assert scan["with_odd_0_1_2_8"]["delta_chi2_alias"] > 100               # broken by m=1


def test_alias_injection_is_recovered_only_with_odd_repetitions():
    alias = {e: {"local": {k: v + math.pi / 2 for k, v in x["local"].items()}, "delta": x["delta"] + math.pi / 2} for e, x in TRUTH.items()}
    data = R.synthetic_counts(ROWS, alias, np.random.default_rng(4))
    est, _ = R.estimate(ROWS, data)
    # delta is only defined modulo pi/2 with the alias shift; the full-rep fit must land on the injected branch
    d = est["16_23"]["delta"]
    assert abs(R.wrap(d - alias["16_23"]["delta"])) < 0.02


def test_bootstrap_refits_spam_every_replicate(synth, monkeypatch):
    calls = {"n": 0}
    orig = R.fit_spam
    def spy(rows, data):
        calls["n"] += 1
        return orig(rows, data)
    monkeypatch.setattr(R, "fit_spam", spy)
    R.bootstrap(ROWS, synth, 5, seed=1)
    # one point estimate for the warm-start hint plus one SPAM refit per replicate
    assert calls["n"] == 1 + 5


def test_bootstrap_resamples_joint_bitstrings(synth):
    rng = np.random.default_rng(0)
    rid = next(r["id"] for r in ROWS if r["family"] == "cz")
    res = R.resample_row(rng, synth[rid])
    assert sum(res.values()) == sum(synth[rid].values())
    assert set(res) <= set(synth[rid])                                # keys are joint (control, probe) outcomes


def test_run_reports_forward_reverse_agreement(synth):
    out = R.run(ROWS, synth, n_boot=30, seed=2)
    for e, v in out["edges"].items():
        assert "forward_reverse_agree_95" in v and v["bootstrap"]["delta"]["se"] > 0
    assert out["spam_joint_readout_correlation_max"] < 0.05


def test_unwrapping_removes_known_cz_phase_for_odd_repetitions():
    # state 1 odd reps: raw phase near pi; the fit subtracts m*pi exactly so the slope is the small residual
    data = R.synthetic_counts(ROWS, TRUTH, np.random.default_rng(8), offsets=(0, 0), rev_off=0.0)
    est, _ = R.estimate(ROWS, data)
    for v in est.values():
        assert all(abs(s) < 0.2 for s in v["slopes"].values())


# ------------------------------------------------------------------ submit adapter (fake backend, fake sampler)
@pytest.fixture()
def frozen(tmp_path, monkeypatch):
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    import qc26a_compile as qc
    import compile_readonly as cr
    backend = FakeKingston()
    state = {"ts": "2026-11-26T00:00:00+00:00"}

    class Svc:
        def __init__(self): self.used = 100.0
        def usage(self):
            return {"usage_consumed_seconds": self.used, "usage_limit_seconds": 600, "usage_remaining_seconds": 600 - self.used,
                    "usage_period": {}, "usage_limit_reached": False}
    svc = Svc()
    monkeypatch.setattr(qc, "backend_connection", lambda: (svc, backend))
    real_snapshot = qc.calibration_snapshot
    monkeypatch.setattr(qc, "calibration_snapshot", lambda b, source="": real_snapshot(b, timestamp=state["ts"], source=source))
    import submit_ramsey as sr
    monkeypatch.setattr(sr, "calibration_snapshot", lambda b, source="": real_snapshot(b, timestamp=state["ts"], source=source))
    out = tmp_path / "q23"
    cr.build("q23", out)
    return sr, out, svc, backend, state


def write_auth(path, sha, **kw):
    a = {"authorized": True, "backend": "ibm_kingston", "plan_sha256": sha, "max_qpu_seconds": 14,
         "authorized_by": "test-owner", "acknowledge_estimate_not_hard_limit": True, "checked_current_allocation": True}
    a.update(kw)
    path.write_text(json.dumps(a))
    return path


class FakeSampler:
    """Refusal-path tests must never reach the sampler."""
    def __init__(self, *a, **k):
        raise AssertionError("sampler constructed on a refused submission")


def test_validate_frozen_reproduces(frozen):
    sr, out, *_ = frozen
    meta, isas = sr.validate_frozen(out)
    assert len(isas) == 66 and abs(meta["estimated_qpu_seconds"] - 13.18944) < 1e-4


def test_submit_refuses_without_valid_authorization(frozen, tmp_path):
    sr, out, *_ = frozen
    meta = json.loads((out / "compiled.json").read_text())
    with pytest.raises(ValueError):
        sr.submit(out, None, sampler_cls=FakeSampler)
    with pytest.raises(ValueError, match="plan_sha256"):
        sr.submit(out, write_auth(tmp_path / "a.json", "0" * 64), sampler_cls=FakeSampler)
    with pytest.raises(ValueError, match="exceeds authorized"):
        sr.submit(out, write_auth(tmp_path / "b.json", meta["plan_sha256"], max_qpu_seconds=5), sampler_cls=FakeSampler)
    with pytest.raises(ValueError, match="cap"):
        sr.submit(out, write_auth(tmp_path / "c.json", meta["plan_sha256"], max_qpu_seconds=30), sampler_cls=FakeSampler)
    with pytest.raises(ValueError, match="acknowledge"):
        sr.submit(out, write_auth(tmp_path / "d.json", meta["plan_sha256"], checked_current_allocation=False), sampler_cls=FakeSampler)
    assert not (out / "submission_started.json").exists()


def test_submit_refuses_stale_compile_changed_calibration_and_low_allocation(frozen, tmp_path):
    sr, out, svc, backend, state = frozen
    meta = json.loads((out / "compiled.json").read_text())
    auth = write_auth(tmp_path / "a.json", meta["plan_sha256"])
    late = lambda: dt.datetime.fromisoformat(meta["compiled_at"]) + dt.timedelta(minutes=16)
    with pytest.raises(ValueError, match="older than 15 minutes"):
        sr.submit(out, auth, sampler_cls=FakeSampler, clock=late)
    soon = lambda: dt.datetime.fromisoformat(meta["compiled_at"]) + dt.timedelta(minutes=1)
    state["ts"] = "2026-11-26T01:00:00+00:00"
    with pytest.raises(ValueError, match="calibration timestamp changed"):
        sr.submit(out, auth, sampler_cls=FakeSampler, clock=soon)
    state["ts"] = "2026-11-26T00:00:00+00:00"
    svc.used = 595.0
    with pytest.raises(ValueError, match="remaining allocation"):
        sr.submit(out, auth, sampler_cls=FakeSampler, clock=soon)
    assert not (out / "submission_started.json").exists()


def test_submit_writes_receipts_and_blocks_a_second_submission(frozen, tmp_path):
    sr, out, svc, backend, state = frozen
    meta = json.loads((out / "compiled.json").read_text())
    auth = write_auth(tmp_path / "a.json", meta["plan_sha256"])
    soon = lambda: dt.datetime.fromisoformat(meta["compiled_at"]) + dt.timedelta(minutes=1)

    class Sampler:
        def __init__(self, mode, options=None): pass
        def run(self, isas, shots):
            svc.used += 12.0
            class E:
                def __init__(s, k): s.data = type("D", (), {"meas": type("M", (), {"get_counts": lambda z: {"0" * k: shots}})()})()
            class J:
                def job_id(s): return "job-fake-1"
                def result(s): return [E(len(r["path"])) for r in meta["circuits"]]
                def usage(s): return 12.0
            return J()
    sr.submit(out, auth, sampler_cls=Sampler, clock=soon)
    receipt = json.loads((out / "execution_receipt.json").read_text())
    assert receipt["job_id"] == "job-fake-1" and abs(receipt["consumed_seconds_delta"] - 12.0) < 1e-9
    assert len(json.loads((out / "counts.json").read_text())) == 66
    assert (out / "usage_before.json").exists() and (out / "calibration_submission.json").exists()
    with pytest.raises(FileExistsError):
        sr.submit(out, auth, sampler_cls=Sampler, clock=soon)


def test_validate_frozen_detects_tampering(frozen):
    sr, out, *_ = frozen
    p = out / "circuits" / next(iter(sorted(x.name for x in (out / "circuits").iterdir())))
    p.write_bytes(p.read_bytes() + b"\n// tamper")
    with pytest.raises(AssertionError):
        sr.validate_frozen(out)
