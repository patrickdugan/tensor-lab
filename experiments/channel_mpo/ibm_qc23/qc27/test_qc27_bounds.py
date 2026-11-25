"""QC-27 tests (unittest, < 2 min).  Run:  python -m unittest -q test_qc27_bounds"""
import os
import sys
import unittest
from fractions import Fraction as Fr

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qc27_bounds as B  # noqa: E402
import qc27_exact as E  # noqa: E402

NOISE = {"weak": dict(p1=0.002, g1=0.001, p2=0.01, g2=0.003),
         "strong": dict(p1=0.01, g1=0.005, p2=0.05, g2=0.01)}
VARIANTS = ("old", "cone", "cone_tl", "minl1l2", "grouped", "surv")
TOL = 1e-11


class TestFloatBounds(unittest.TestCase):
    def test_validity_and_ordering_n8(self):
        runs = 0
        for nm in ("weak", "strong"):
            for depth in (2, 3):
                for seed in range(4):
                    circ = B.build_circuit(8, depth, seed, **NOISE[nm])
                    ex, _ = B.dense_heisenberg(circ, restrict=False)
                    for eps in (3e-2, 3e-3, 3e-4):
                        res = B.propagate(circ, eps)
                        if not res["events"]:
                            continue
                        b = B.total_bounds(res["events"], 8, res["rounding"])
                        actual = abs(ex - res["expectation"])
                        runs += 1
                        for v in VARIANTS:
                            self.assertLessEqual(actual, b[v] + TOL, (nm, depth, seed, eps, v))
                        for hi, lo in (("old", "cone"), ("cone", "cone_tl"), ("cone_tl", "minl1l2"),
                                       ("minl1l2", "grouped"), ("grouped", "surv")):
                            self.assertLessEqual(b[lo], b[hi] * (1 + 1e-12), (hi, lo))
        self.assertGreater(runs, 30)

    def test_dense_references_agree_n8(self):
        circ = B.build_circuit(8, 3, 5, **NOISE["strong"])
        h, _ = B.dense_heisenberg(circ, restrict=False)
        hr, m = B.dense_heisenberg(circ, restrict=True)
        s, _ = B.dense_schrodinger(circ)
        self.assertAlmostEqual(h, s, delta=1e-12)
        self.assertAlmostEqual(h, hr, delta=1e-12)
        # no truncation: sparse propagation reproduces the dense value
        res = B.propagate(circ, 0.0)
        self.assertAlmostEqual(res["expectation"], h, delta=1e-12)
        self.assertEqual(res["events"], [])

    def test_per_event_and_group_bounds_n8(self):
        """Telescoping identity and every event/group bound against exact per-event errors."""
        for nm in ("weak", "strong"):
            for seed in range(2):
                circ = B.build_circuit(8, 3, seed, **NOISE[nm])
                ex, _ = B.dense_heisenberg(circ, restrict=False)
                res = B.propagate(circ, 3e-3, keep_discards=True)
                errs = B.per_event_errors(circ, res)
                self.assertAlmostEqual(sum(e for e, _ in errs), ex - res["expectation"], delta=1e-12)
                for ev, (e, pg) in zip(res["events"], errs):
                    eb = B.event_bounds(ev, 8)
                    for v in VARIANTS:
                        self.assertLessEqual(abs(e), eb[v] + 1e-12, v)
                    for (m, k, l1, l2sq, sg) in ev.groups:
                        bg = l1 if k == 0 else min(l1, B._cone_tl(k) * l2sq ** 0.5)
                        self.assertLessEqual(abs(pg[m]), sg * bg + 1e-12)

    def test_cone_gain_wide_register(self):
        circ = B.build_circuit(24, 4, 0, **NOISE["weak"])
        res = B.propagate(circ, 1e-2)
        b = B.total_bounds(res["events"], 24, res["rounding"])
        sites, _ = B.cone_restrict(circ)
        self.assertLessEqual(len(sites), 10)
        for ev in res["events"]:
            self.assertLessEqual(ev.k, len(sites))
        self.assertLess(b["surv"], b["old"] / 100)
        self.assertLessEqual(b["surv"], b["cone"])

    def test_survival_stress_and_negative_controls(self):
        r = B.survival_stress(150, n=5, nops=10, seed=21)
        self.assertEqual(r["violations"], 0)
        self.assertGreater(r["frac_sigma_below_1"], 0.3)
        self.assertGreater(B.survival_stress(150, n=5, nops=10, seed=21, neg="no_merge")["violations"], 0)
        self.assertGreater(B.survival_stress(300, n=5, nops=10, seed=21, neg="keep_nonunital")["violations"], 0)

    def test_survival_bounds_in_unit_interval(self):
        circ = B.build_circuit(12, 3, 1, **NOISE["strong"])
        for mask in (0b1, 0b11, 0b1010, 0b111000):
            s = B.survival(circ.ops, len(circ.ops), mask << 3)
            self.assertTrue(0.0 < s <= 1.0)
        # more blocks can only help
        self.assertLessEqual(B.survival(circ.ops, len(circ.ops), 0b111 << 4),
                             B.survival(circ.ops, len(circ.ops), 0b1 << 5) * (1 + 1e-12))


class TestExact(unittest.TestCase):
    def test_exact_vs_independent_schrodinger_and_bound_recomputation(self):
        for (n, d, s, eps) in E.EXACT_CASES:
            circ = E.random_excircuit(n, d, s)
            E._register_units(circ, None)
            full = E.exact_propagate(circ, 0)
            sch = E.schro_expectation(circ, lambda ch: E.ad_dep_units_cache[id(ch)])
            self.assertEqual(full["expectation"], sch)       # exact Fraction equality
            tr = E.exact_propagate(circ, eps)
            err = abs(sch - tr["expectation"])
            b = E.bound_from_ledger(tr["events"], n)
            for v in b.values():
                self.assertLessEqual(err, v)
            # recompute the bound from the serialised ledger with an independent minimal routine
            led = [dict(k=ev["k"], l1=E._fr(E._fs(ev["l1"])), l2sq=E._fr(E._fs(ev["l2sq"])),
                        groups=[dict(k=g["k"], l1=E._fr(E._fs(g["l1"])), l2sq=E._fr(E._fs(g["l2sq"])),
                                     sigma=E._fr(E._fs(g["sigma"]))) for g in ev["groups"]])
                   for ev in tr["events"]]
            b2 = E.bound_from_ledger(led, n)
            self.assertEqual(b, b2)
            self.assertTrue(all(E._is_dyadic(g["sigma"]) for ev in tr["events"] for g in ev["groups"]))
            # float pipeline on the float twin agrees with the exact bound up to rounding
            import qc27_bounds
            fc = E.to_float_circuit(circ)
            fr = qc27_bounds.propagate(fc, float(eps))
            fb = qc27_bounds.total_bounds(fr["events"], n, fr["rounding"])
            self.assertAlmostEqual(fr["expectation"], float(tr["expectation"]), delta=1e-12)
            for k, v in b.items():
                self.assertLessEqual(float(v), fb[k] * (1 + 1e-9) + 1e-12, k)
                self.assertLessEqual(fb[k], float(v) * (1 + 1e-9) + 1e-12, k)

    def test_exact_survival_matches_float_tracker(self):
        circ = E.random_excircuit(4, 3, 7)
        fc = E.to_float_circuit(circ)
        for mask in range(1, 16):
            se = E.exact_survival(circ.flat, len(circ.flat), mask)
            sf = B.survival(fc.ops, len(fc.ops), mask)
            self.assertLessEqual(float(se), sf + 1e-12)
            self.assertAlmostEqual(float(se), sf, delta=1e-9)

    def test_witness_roundtrip_and_tamper(self):
        circ = E.random_excircuit(3, 3, 0)
        w = E.emit_witness(circ, Fr(1, 32))
        ok, why = E.verify_witness(w)
        self.assertTrue(ok, why)
        import copy
        w2 = copy.deepcopy(w)
        w2["steps"][1]["ops"][0]["chans"][0]["mu"] = "1/1"        # no longer CP / wrong channel
        self.assertFalse(E.verify_witness(w2)[0])
        w3 = copy.deepcopy(w)
        w3["claim"]["bound_surv"] = "1/1"
        self.assertFalse(E.verify_witness(w3)[0])

    def test_cp_certificates(self):
        self.assertTrue(E.Chan.ad_dep(Fr(1, 2), Fr(1, 8)).verified())
        self.assertFalse(E.Chan(Fr(9, 10), Fr(1, 2), Fr(1, 2), 0).verified())   # mu^2 > ((1+lam)^2-tau^2)/4
        self.assertFalse(E.Chan(Fr(1, 2), Fr(1, 2), Fr(1, 4), Fr(1, 2)).verified())  # kappa too large

    def test_failed_hypotheses_are_counterexamples(self):
        h = E.failed_hypotheses()
        for k, v in h.items():
            self.assertTrue(v.get("violated", not v.get("holds", True)), k)
        self.assertTrue(h["H2_weight_survival"]["tracker_valid"])
        self.assertTrue(h["H4_block_survives_nonunital"]["tracker_valid"])
        self.assertTrue(h["H6_drop_cone_factor"]["equals_2k_minus_1"])


if __name__ == "__main__":
    unittest.main()
