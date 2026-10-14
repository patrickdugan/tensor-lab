import copy
from fractions import Fraction

import pytest

from tensorlab.experiments import e1_staircase_spacing, e2_two_classes, e4_three_sector_sanity, e5_packing_search
from tensorlab.fourier import separation_from_certificate
from tensorlab.growth import check_counting_lemma, extremal_profile, growth_product, rectangle_density_excess
from tensorlab.sectors import staircase_certificate, three_sector_certificate
from tensorlab.verify import verify_certificate
from tensorlab.weights_lp import (decide, rectangle_cells, solve_farkas, solve_primal, staircase_retained,
                                  verify_obstruction, verify_realization)


def test_three_sector_realizable_with_and_without_x_weights():
    for r in e4_three_sector_sanity(cases=((2, 1), (3, 2))):
        assert r["realizable"] and r["farkas_consistent"] and r["verified"]


def test_staircase_spacing_threshold():
    res = e1_staircase_spacing(cases=((5, 3, 3), (6, 3, 2)))
    for r in res:
        if "skipped" in r:
            continue
        assert r["farkas_consistent"] and r["verified"]
        assert r["realizable"] == (r["period"] >= r["ell"])


def test_two_class_dense_packing_is_obstructed():
    (r,) = e2_two_classes()
    assert not r["realizable"] and r["farkas_consistent"] and r["verified"]


def test_obstruction_certificate_is_checked_exactly():
    a, p, q = 5, 3, 3
    B, blocks = staircase_retained(a, p, q, 2, period=4)
    R = set().union(*blocks)
    st, cert = solve_farkas(a, B, R)
    assert st == "obstruction" and verify_obstruction(a, B, R, cert)
    mu, nu = cert
    bad = (dict(mu), dict(nu))
    k = next(iter(bad[0]))
    bad[0][k] = bad[0][k] * 2  # break the marginal balance
    assert not verify_obstruction(a, B, R, bad)


def test_realization_is_checked_exactly():
    a, p, q = 5, 3, 3
    B, blocks = staircase_retained(a, p, q, 3, period=5)
    R = set().union(*blocks)
    st, w = solve_primal(a, B, R)
    assert st == "feasible" and verify_realization(a, B, R, w)
    x, u, v = w
    u2 = list(u)
    u2[0] += 1
    assert not verify_realization(a, B, R, (x, u2, v))


def test_packing_search_small_cases_find_nothing_realizable():
    r = e5_packing_search(3, 15, max_lp=60, time_limit=120)
    assert r["stats"]["packings_above_target"] > 0
    assert r["stats"]["realizable_above_target"] == 0
    assert all(o["obstruction_verified"] for o in r["sample_obstructed"])


@pytest.mark.parametrize("a,p,q,n", [(3, 2, 2, 3), (5, 3, 3, 2), (4, 2, 3, 3), (2, 1, 2, 4)])
def test_staircase_certificates(a, p, q, n):
    cert = staircase_certificate(a, p, q, n)
    rep = verify_certificate(cert)
    assert rep.ok, rep.summary()
    assert all(b["leg_permutation"] in ("swap_XZ", "cycle_XYZ", "cycle_XZY", "identity", "swap_XY", "swap_YZ")
               for b in cert["blocks"])


def test_counting_lemma_and_rectangle_excess():
    viol, eq = check_counting_lemma(25)
    assert viol == []
    assert all(q >= p or q == p - 1 for p, q in eq)  # equality on/above the diagonal and at q = p - 1
    assert rectangle_density_excess(5, 3, 3) > 0      # Lean: Extremal.rectangle_exceeds
    assert rectangle_density_excess(3, 1, 3) <= 0


def test_separation_from_certificate():
    res = separation_from_certificate(three_sector_certificate(2, 1))
    assert res.ok, res.summary()
    bad = copy.deepcopy(three_sector_certificate(2, 1))
    bad["retained_terms"].pop()
    assert not separation_from_certificate(bad).ok
