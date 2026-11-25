"""Offline tests for qc26a_identify.py. Synthetic data only; no IBM account, no hardware.

Run from this directory:  python -m unittest test_qc26a_identify -v   (about 1-2 minutes)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import copy
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qc26a_identify as Q  # noqa: E402

NODE = shutil.which("node") or r"C:/Users/patri/AppData/Local/Programs/nodejs/node.exe"
PI = math.pi


def js_arm_counts(plan):
    """Counts of JS trial 0 (injected) and JS trial 48 (first null), same LCG stream as qc26a_design.mjs."""
    rng = Q.LCG(260126)
    inj0 = Q.synthetic_counts(plan, rng=rng)
    for _ in range(47):
        Q.synthetic_counts(plan, rng=rng)
    null0 = Q.synthetic_counts(plan, rng=rng, zero=True)
    return inj0, null0


class IdentifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = Q.load_plan()
        cls.L = Q.Layout(cls.plan)
        cls.inj0, cls.null0 = js_arm_counts(cls.plan)
        with open(Q.SYN_PATH, encoding="utf-8") as fh:
            cls.syn = json.load(fh)
        kw = dict(bootstraps=40, seed=3, profile=True, n_local=9, n_coarse=11, alias=True, synthetic=Q.SYN_PATH)
        cls.res_inj = Q.identify(cls.inj0, **kw)
        cls.res_null = Q.identify(cls.null0, **kw)
        cls.table = [tuple(r) for r in cls.res_inj["parameter_table"]]

    # ---------------------------------------------------------------- inputs / generator
    def test_input_validation(self):
        bad = copy.deepcopy(self.inj0)
        bad["sx_0_Z"]["0" * 3] = bad["sx_0_Z"].get("0" * 3, 0)   # wrong length for a 4-site circuit
        with self.assertRaises(ValueError):
            Q.parse_counts(self.plan, bad)
        bad = copy.deepcopy(self.inj0)
        k = next(iter(bad["spam_0"]))
        bad["spam_0"][k] += 1
        with self.assertRaises(ValueError):
            Q.parse_counts(self.plan, bad)
        bad = copy.deepcopy(self.inj0)
        del bad["cz_4_5_0_0_X"]
        with self.assertRaises(ValueError):
            Q.parse_counts(self.plan, bad)

    def test_bootstrap_resample_preserves_shots_and_support(self):
        data = Q.parse_counts(self.plan, self.inj0)
        rng = np.random.default_rng(0)
        b = Q.resample(data, rng)
        for cid, c in b.items():
            self.assertEqual(c.shots, data[cid].shots)
            self.assertTrue(np.all(c.cnt[data[cid].cnt == 0] == 0))
        self.assertTrue(any(not np.array_equal(b[c].cnt, data[c].cnt) for c in b))

    def test_python_generator_and_js_mirror_reproduce_committed_js_trials(self):
        for counts, idx in ((self.inj0, 0), (self.null0, 48)):
            js = Q.js_mirror(self.plan, Q.parse_counts(self.plan, counts))
            tr = self.syn["trials"][idx]
            for p, v in tr["sx"].items():
                self.assertAlmostEqual(js["sx"][int(p)]["offset_rad"], v, delta=1e-9)
            for k, v in tr["cz"].items():
                self.assertAlmostEqual(js["cz"][k]["zz_phase_rad"], v, delta=1e-9)
            for k, v in tr["local"].items():
                self.assertAlmostEqual(js["cz"][k]["local_phase_rad"], v, delta=1e-9)

    @unittest.skipUnless(Path(NODE).exists(), "node not available")
    def test_js_mirror_matches_node_analyze(self):
        with tempfile.TemporaryDirectory() as td:
            cp, op = Path(td) / "c.json", Path(td) / "o.json"
            with open(cp, "w") as fh:
                json.dump(self.inj0, fh)
            subprocess.run([NODE, str(HERE / "qc26a_design.mjs"), "analyze", str(cp), str(op)], check=True,
                           capture_output=True, cwd=HERE)
            with open(op) as fh:
                node = json.load(fh)
        js = Q.js_mirror(self.plan, Q.parse_counts(self.plan, self.inj0))
        for p in js["sx"]:
            self.assertAlmostEqual(js["sx"][p]["offset_rad"], node["sx"][str(p)]["offset_rad"], delta=1e-12)
        for k in js["cz"]:
            self.assertAlmostEqual(js["cz"][k]["zz_phase_rad"], node["cz"][k]["zz_phase_rad"], delta=1e-9)
            self.assertAlmostEqual(js["cz"][k]["local_phase_rad"], node["cz"][k]["local_phase_rad"], delta=1e-9)

    # ---------------------------------------------------------------- recovery
    def _se(self, res, name):
        return max(res["bootstrap"]["parameters"][name]["se"], res["identifiability"]["wald_se"][name])

    def test_injected_recovery_signs_and_sizes(self):
        truth = {n: (self.syn["truth"][fam][key]) for n, ent, kind, fam, key in self.table}
        for n, ent, kind, fam, key in self.table:
            est, se = self.res_inj["fit_parameters"][n], self._se(self.res_inj, n)
            self.assertLess(abs(est - truth[n]), 3 * se, f"{n}: est {est:.4f} truth {truth[n]} se {se:.4f}")
            if fam in ("sx", "cz") and abs(truth[n]) >= 0.02:
                self.assertEqual(np.sign(est), np.sign(truth[n]), n)
        # SX offsets specifically (spec: within 3 sigma of truth)
        for p in (3, 5, 21, 23):
            n = f"sx.{p}.offset_rad"
            self.assertLess(abs(self.res_inj["fit_parameters"][n] - self.syn["truth"]["sx"][str(p)]), 3 * self._se(self.res_inj, n))

    def test_null_arm_coverage_is_reported_honestly(self):
        zero = [n for n, ent, kind, fam, key in self.table if fam in ("sx", "cz")]
        boot = [self.res_null["bootstrap"]["parameters"][n]["ci95"] for n in zero]
        prof = [self.res_null["identifiability"]["profiles"][n]["profile_ci95"] for n in zero]
        cb = sum(lo <= 0 <= hi for lo, hi in boot)
        cp = sum(lo <= 0 <= hi for lo, hi in prof)
        print(f"\n[null arm, JS trial 48] zero covered by bootstrap CI {cb}/{len(zero)}, profile CI {cp}/{len(zero)} "
              f"(nominal 95%; one dataset only, realised coverage over 96 datasets is in the receipts)")
        self.assertGreaterEqual(cb, len(zero) - 2)
        self.assertGreaterEqual(cp, len(zero) - 2)
        for st in self.res_null["decision"]["per_parameter"].values():
            self.assertNotEqual(st["status"], "NONZERO_REPLICATED")

    def test_bootstrap_se_consistent_with_wald_and_profile(self):
        for n, ent, kind, fam, key in self.table:
            b = self.res_inj["bootstrap"]["parameters"][n]["se"]
            w = self.res_inj["identifiability"]["wald_se"][n]
            self.assertTrue(0.5 < b / w < 2.0, f"{n}: bootstrap {b} wald {w}")
            lo, hi = self.res_inj["identifiability"]["profiles"][n]["profile_ci95"]
            self.assertTrue(0.6 < (hi - lo) / 2 / (Q.Z975 * w) < 1.6, f"{n}: profile half-width vs Wald")

    def test_fit_close_to_js_estimator(self):
        for n, ent, kind, fam, key in self.table:
            d = self.res_inj["js_mirror"]["joint_minus_js"][n]
            self.assertLess(abs(d), 2.0 * self.res_inj["identifiability"]["wald_se"][n], n)

    def test_goodness_of_fit_sane(self):
        f = self.res_inj["fit"]
        self.assertEqual(f["dof"], 20)
        self.assertGreater(f["deviance_p_value"], 0.001)

    # ---------------------------------------------------------------- identifiability
    def test_jacobian_rank_and_condition(self):
        J = self.res_inj["identifiability"]["jacobian"]
        self.assertEqual(J["rank_numeric"], J["n_parameters"])
        self.assertEqual(J["n_parameters"], 100)
        self.assertEqual(J["n_observations"], 120)
        s = np.array(J["singular_values"])
        self.assertGreater(s.min(), 5.0)
        self.assertLess(J["condition_number"], 1000)
        self.assertLess(J["condition_number_column_scaled"], 100)
        for k, b in J["blocks"].items():
            self.assertLess(b["condition"], 100, k)
        self.assertFalse([k for k, v in J["per_parameter"].items() if v["non_identifiable"]])
        for n, *_ in self.table:
            self.assertFalse(J["per_parameter"][n]["weakly_determined"], n)

    def test_alpha_control_is_not_identifiable_and_conventions(self):
        for k, h in self.res_inj["hamiltonian"].items():
            self.assertFalse(h["alpha_control_rad"]["identifiable"])
            self.assertLess(h["alpha_control_rad"]["max_abs_prediction_sensitivity"], 1e-8)
            self.assertIn("global phase", h["alpha_control_rad"]["reason"])
        # exact 2-qubit evolution agrees with the fit model's phase convention theta = m (alpha_t + sigma delta)
        for s in (0, 1):
            sigma = 1 - 2 * s
            for m in (0, 2, 8):
                zc, x, y = Q.hamiltonian_prediction(0.37, 0.013, 0.032, s, m)
                th = m * (0.013 + sigma * 0.032)
                self.assertAlmostEqual(x, math.cos(th), places=12)
                self.assertAlmostEqual(y, math.sin(th), places=12)
                self.assertAlmostEqual(zc, sigma, places=12)

    def test_alias_algebra_exact(self):
        L = self.L
        th = L.th0.copy()
        for p in L.all_sites:
            th[L.e0[p]], th[L.e1[p]] = 0.01, 0.02
        rng = np.random.default_rng(1)
        for e in L.edges:
            th[L.cz_phi[e]], th[L.cz_del[e]] = rng.uniform(-0.3, 0.3, 2)
            for s in (0, 1):
                th[L.cz_a[(e, s)]] = rng.uniform(-1, 1)
        for p in L.sx_sites:
            th[L.sx_d[p]], th[L.sx_ph[p]] = rng.uniform(-0.05, 0.05), rng.uniform(-0.1, 0.1)
        z0 = L.full.z(th)
        cz = np.where(L.full.kind == 2)[0]
        sx = np.where(L.full.kind == 1)[0]

        def shifted(j, l):
            t = th.copy()
            for e in L.edges:
                t[L.cz_phi[e]] += j * PI / 2
                t[L.cz_del[e]] += l * PI / 2
            return L.full.z(t)

        for j, l in ((1, 1), (1, -1), (2, 0), (0, 2), (-1, 1), (3, 1)):
            self.assertLess(np.max(np.abs(shifted(j, l)[cz] - z0[cz])), 1e-9, (j, l))
        for j, l in ((1, 0), (0, 1), (0.5, 0.5), (0.5, -0.5)):    # not in the lattice
            self.assertGreater(np.max(np.abs(shifted(j, l)[cz] - z0[cz])), 1e-3, (j, l))
        t = th.copy()                                   # SX reflection alias (d, phi0) -> (pi - d, -phi0)
        for p in L.sx_sites:
            t[L.sx_d[p]], t[L.sx_ph[p]] = PI - th[L.sx_d[p]], -th[L.sx_ph[p]]
        self.assertLess(np.max(np.abs(L.full.z(t)[sx] - z0[sx])), 1e-9)
        for dd in (PI / 2, PI, 2 * PI / 24):            # translations are not aliases
            t = th.copy()
            for p in L.sx_sites:
                t[L.sx_d[p]] += dd
            self.assertGreater(np.max(np.abs(L.full.z(t)[sx] - z0[sx])), 1e-3, dd)

    def test_alias_scans_clean_on_ordinary_data_and_find_reflection(self):
        al = self.res_inj["identifiability"]["alias"]
        for k, v in al["cz"].items():
            self.assertFalse(v["alias_ambiguous"], k)
            self.assertGreater(v["alias_margin_rad"], 1.5)
        for k, v in al["sx"].items():
            self.assertFalse(v["ambiguous"], k)
            self.assertFalse(v["global_minimum_outside_bound"], k)
            self.assertTrue(v["reflection_alias_present"], k)    # the exact (pi-d, -phi0) alias is found numerically

    def test_alias_injection_near_pi_over_2_is_detected(self):
        T = copy.deepcopy(Q.JS_TRUTH)
        T["cz"]["4_5"] = PI / 2 - 0.03
        T["local"]["4_5"] = 0.01
        counts = Q.synthetic_counts(self.plan, truth=T, np_rng=np.random.default_rng(5))
        res = Q.identify(counts, bootstraps=20, seed=2, profile=False, alias=True, synthetic=False)
        a = res["identifiability"]["alias"]["cz"]["4_5"]
        self.assertTrue(a["alias_ambiguous"])
        self.assertTrue(a["near_alias_boundary"])
        self.assertLess(a["alias_margin_rad"], 0.2)
        # the reported decomposition is within the lattice of the truth, and a candidate near zero conditional phase exists
        fit = res["fit"]["cz"]["4_5"]
        dist = min(math.hypot(c["alpha_target"] - 0.01, c["delta"] - T["cz"]["4_5"])
                   for c in res["hamiltonian"]["4_5"]["alias"]["equivalent_decompositions"])
        self.assertLess(dist, 0.03)
        self.assertTrue(any(abs(c["delta"]) < 0.1 for c in res["hamiltonian"]["4_5"]["alias"]["equivalent_decompositions"]))
        self.assertEqual(res["decision"]["per_parameter"]["cz.4_5.delta_rad"]["status"], "INCONCLUSIVE_FLAGS")
        self.assertIn("no_alias_ambiguity", res["decision"]["per_parameter"]["cz.4_5.delta_rad"]["failed_gates"])
        for k in ("6_5", "16_23", "22_23", "4_3", "22_21"):
            self.assertFalse(res["identifiability"]["alias"]["cz"][k]["alias_ambiguous"], k)
        self.assertGreater(res["bootstrap"]["flag_fractions"]["cz_4_5"]["alias_boundary"], 0.9)
        self.assertGreater(fit["delta_rad"], 1.4)

    def test_boundary_solution_is_flagged(self):
        T = copy.deepcopy(Q.JS_TRUTH)
        T["sx"][5] = 0.13
        counts = Q.synthetic_counts(self.plan, truth=T, np_rng=np.random.default_rng(6))
        res = Q.identify(counts, bootstraps=20, seed=2, profile=False, alias=True, synthetic=False)
        self.assertAlmostEqual(abs(res["fit"]["sx"]["5"]["offset_rad"]), 0.1, delta=1e-3)
        self.assertTrue(res["flags_estimate"]["sx_5"]["boundary"])
        self.assertGreater(res["bootstrap"]["flag_fractions"]["sx_5"]["boundary"], 0.9)
        a = res["identifiability"]["alias"]["sx"]["5"]
        self.assertTrue(a["global_minimum_outside_bound"])
        self.assertTrue(any(abs(c["offset"] - 0.13) < 0.01 for c in a["competing_minima"]))
        st = res["decision"]["per_parameter"]["sx.5.offset_rad"]
        self.assertEqual(st["status"], "INCONCLUSIVE_FLAGS")
        self.assertIn("no_boundary_or_clipping", st["failed_gates"])
        # unaffected sites are not flagged as boundary
        self.assertFalse(res["flags_estimate"]["sx_23"]["boundary"])

    # ---------------------------------------------------------------- diagnostics
    def test_clean_synthetic_data_raises_no_diagnostic_flag(self):
        for res in (self.res_inj, self.res_null):
            d = res["diagnostics"]
            for k in ("crosstalk", "leakage_population", "relaxation", "quadrature", "drop_longest"):
                self.assertIn("flag", d[k])
            for k in ("leakage_population", "relaxation", "quadrature", "drop_longest"):
                self.assertIn("threshold_justification", d[k])
            for k in ("sx_block_pair_correlation", "cz_control_target_association_cmh", "control_marginal_vs_spam",
                      "prep_phase_block_vs_isolated"):
                self.assertIn("flag", d["crosstalk"][k])
            self.assertFalse(d["leakage_population"]["flag"])
            self.assertFalse(d["relaxation"]["flag"])

    def test_injected_leakage_and_control_shift_are_flagged(self):
        counts = copy.deepcopy(self.inj0)
        for ax in "XY":                                   # control (site 4, char index 1) flips with 15% prob at m=8
            cid = f"cz_4_5_0_8_{ax}"
            new = {}
            for s, k in counts[cid].items():
                mv = int(round(0.15 * k))
                new[s] = new.get(s, 0) + k - mv
                flipped = s[0] + ("1" if s[1] == "0" else "0")
                new[flipped] = new.get(flipped, 0) + mv
            counts[cid] = new
        data = Q.parse_counts(self.plan, counts)
        fit = Q.fit_all(self.L, data)
        rng = np.random.default_rng(0)
        leak = Q.diag_leakage(self.L, data, fit.th, rng)
        self.assertTrue(leak["per_circuit_group"]["4_5_state0"]["flag"])
        self.assertGreater(leak["per_circuit_group"]["4_5_state0"]["drift_0_to_8"], 0.05)
        self.assertFalse(leak["per_circuit_group"]["6_5_state0"]["flag"])
        cross = Q.diag_crosstalk(self.L, data, fit.th, {}, rng)
        self.assertTrue(cross["control_marginal_vs_spam"]["per_circuit_group"]["4_5_state0"]["flag"])

    def test_injected_sx_block_correlation_is_flagged(self):
        counts = copy.deepcopy(self.inj0)
        cid = "sx_24_Y"      # (Z-axis SX circuits are nearly deterministic, so pair tests have no power there)
        # make site 3 and site 21 bits copy each other in 40% of shots
        new = {}
        for s, k in counts[cid].items():
            mv = int(round(0.4 * k))
            new[s] = new.get(s, 0) + k - mv
            b = s[::-1]       # b[j] = bit of path[j]; sites are [3,5,21,23]
            t = list(b)
            t[2] = t[0]
            s2 = "".join(t)[::-1]
            new[s2] = new.get(s2, 0) + mv
        counts[cid] = new
        data = Q.parse_counts(self.plan, counts)
        fit = Q.fit_all(self.L, data)
        cross = Q.diag_crosstalk(self.L, data, fit.th, {}, np.random.default_rng(0))
        self.assertTrue(cross["sx_block_pair_correlation"]["flag"])
        self.assertEqual(cross["sx_block_pair_correlation"]["worst"]["pair"], [3, 21])

    def test_decay_diagnostic_discriminates_exponential_from_not(self):
        ms = (0, 4, 12, 24)
        ok = Q._decay(ms, [0.99 * math.exp(-0.01 * m) for m in ms], [0.01] * 4)
        bad = Q._decay(ms, [0.99, 0.97, 0.55, 0.9], [0.01] * 4)
        self.assertGreater(ok["p"], 0.5)
        self.assertLess(bad["p"], 1e-6)
        self.assertAlmostEqual(ok["decay_per_repetition"], 0.01, places=9)
        self.assertAlmostEqual(ok["apparent_decay_repetitions"], 100.0, places=5)

    def test_drop_longest_reports_each_parameter(self):
        d = self.res_inj["diagnostics"]["drop_longest"]
        self.assertEqual(len(d["per_parameter"]), 16)
        self.assertEqual(sorted(d["dropped_circuits"])[0][:2], "cz")
        self.assertFalse(d["flag"])

    # ---------------------------------------------------------------- comparison, decision, CLI, receipts
    def test_compare_to_synthetic_places_estimates_in_arms(self):
        ci = self.res_inj["comparison_to_js_synthetic"]["per_parameter"]
        cn = self.res_null["comparison_to_js_synthetic"]["per_parameter"]
        for n in ("sx.23.offset_rad", "sx.5.offset_rad", "cz.16_23.delta_rad", "cz.22_23.delta_rad"):
            self.assertEqual(ci[n]["closer_arm"], "injected", n)
            self.assertLess(abs(ci[n]["injected_arm"]["z_score"]), 3.0, n)
            self.assertGreater(abs(ci[n]["null_arm"]["z_score"]), 4.0, n)
            self.assertEqual(cn[n]["closer_arm"], "null", n)
        for n, v in ci.items():
            if n.startswith("sx."):
                self.assertTrue(0.002 < v["injected_arm"]["rmse_about_truth"] < 0.004)
            self.assertEqual(v["injected_arm"]["n"], 48)
            self.assertEqual(v["null_arm"]["n"], 48)
        again = Q.compare_to_synthetic(self.res_inj, self.syn)
        self.assertEqual(again["per_parameter"]["sx.23.offset_rad"]["estimate"], ci["sx.23.offset_rad"]["estimate"])
        self.assertIn("decision", again)

    def test_decision_rule_requires_replication(self):
        dec = self.res_inj["decision"]
        self.assertTrue(dec["replicate_block_required"])
        self.assertNotIn("NONZERO_REPLICATED", dec["status_counts"])
        self.assertIn("NONZERO_CANDIDATE_PENDING_REPLICATION", dec["status_counts"])
        self.assertIn("none", dec["hardware_claim"])
        dec2 = Q.decision_rule(self.res_inj, replicate=self.res_inj)
        self.assertIn("NONZERO_REPLICATED", dec2["status_counts"])     # pathway exists once a second block is supplied
        # a null dataset never produces a candidate for parameters whose truth is zero except by chance
        n_cand = sum(v["status"].startswith("NONZERO") for n, v in self.res_null["decision"]["per_parameter"].items()
                     if n.startswith("sx.") or n.endswith("delta_rad"))
        self.assertLessEqual(n_cand, 1)

    def test_cli_roundtrip_and_seed_reproducibility(self):
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "counts.json"
            with open(cp, "w") as fh:
                json.dump(self.inj0, fh)
            outs = []
            for i in range(2):
                op = Path(td) / f"out{i}.json"
                self.assertEqual(Q.main([str(cp), str(op), "--bootstraps", "6", "--seed", "4", "--no-profile", "--no-alias"]), 0)
                with open(op) as fh:
                    outs.append(json.load(fh))
        self.assertEqual(outs[0]["bootstrap"]["parameters"], outs[1]["bootstrap"]["parameters"])
        self.assertEqual(outs[0]["bootstrap"]["n_replicates"], 6)
        for k in ("fit", "hamiltonian", "bootstrap", "identifiability", "diagnostics", "decision",
                  "comparison_to_js_synthetic", "inputs"):
            self.assertIn(k, outs[0])

    def test_receipt_present_and_reproducible(self):
        path = HERE / "receipts" / "qc26a_identify_synthetic.json"
        self.assertTrue(path.exists(), "run qc26a_identify_receipts.py to create the receipt")
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
        self.assertIn("SYNTHETIC ONLY", r["label"])
        self.assertEqual(r["plan_sha256"], Q.canonical_sha256(self.plan))
        for k in ("A_injected_js_trial0", "B_null_js_trial48"):
            self.assertLess(r["python_port_vs_committed_js_trials"][k]["max_abs_difference_to_committed_js_estimate"], 1e-9)
        a = r["datasets"]["A_injected_js_trial0"]
        data = Q.parse_counts(self.plan, self.inj0)
        self.assertEqual(a["inputs"]["counts_sha256"], Q.canonical_sha256(self.inj0))
        fit = Q.fit_all(self.L, data)
        for n, ent, kind, fam, key in self.table:
            self.assertAlmostEqual(a["fit_parameters"][n], float(fit.th[self.L.index[n]]), delta=2e-4, msg=n)
        for arm in ("injected", "null"):
            agg = r["ensemble"][arm]["aggregate"]
            self.assertIn("bootstrap_percentile_ci_coverage_mean_over_parameters", agg)
            self.assertGreater(r["ensemble"][arm]["n_trials"], 40)


if __name__ == "__main__":
    unittest.main()
