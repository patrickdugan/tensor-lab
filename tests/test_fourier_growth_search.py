from fractions import Fraction

import pytest

from tensorlab.cyclotomic import Cyclotomic, cyclotomic_polynomial, filter_sum
from tensorlab.degeneration import degenerate
from tensorlab.fourier import separation
from tensorlab.growth import (Candidate, check_profile, cube_check, diagonal_lower_bounds, extremal_profile,
                              growth_product, implied_on_grid, paper_constraints)
from tensorlab.lp import LinearProgram
from tensorlab.search import certificate_for, search, solve_weights
from tensorlab.sectors import three_sector_weights
from tensorlab.tensors import convolution
from tensorlab.verify import verify_certificate


# ---- cyclotomic / Fourier ----------------------------------------------------------

def test_cyclotomic_polynomials():
    assert cyclotomic_polynomial(1) == (-1, 1)
    assert cyclotomic_polynomial(2) == (1, 1)
    assert cyclotomic_polynomial(4) == (1, 0, 1)
    assert cyclotomic_polynomial(6) == (1, -1, 1)
    assert cyclotomic_polynomial(15) == (1, -1, 0, 1, -1, 1, 0, -1, 1)


def test_root_of_unity_filter_exact():
    for L in (5, 10, 15):
        for e in range(-2 * L, 2 * L + 1):
            v = filter_sum(L, e)
            assert v.is_rational()
            assert v.as_rational() == (1 if e % L == 0 else 0)
    z = Cyclotomic.zeta_power(15, 1)
    p = Cyclotomic.one(15)
    for _ in range(15):
        p = p * z
    assert p == 1
    assert not Cyclotomic.zeta_power(15, 5).is_rational()


@pytest.mark.parametrize("a,h", [(1, 1), (2, 1), (3, 2)])
def test_finite_separation(a, h):
    w, ys, zs = three_sector_weights(a, h)
    deg = degenerate(convolution(a, 3 * h + a - 1), w)
    res = separation(deg.retained, ys, zs)
    assert res.ok, res.summary()
    assert res.data["M"] == 3 and res.data["L"] == 15
    assert res.data["weight_zero_nnz"] == 3 * (3 * a * h)  # each of the 3 blocks (a*h terms) tensored with B_X(3)


def test_separation_rejects_incompatible_blocks():
    t = convolution(2, 2)  # one block, but give it two 'sectors' with cross terms
    res = separation(t, [[0], [1]], [[0, 1], [2]])
    assert not res.ok


# ---- LP ----------------------------------------------------------------------------

def test_lp_basic():
    lp = LinearProgram()
    lp.add({"x": 1, "y": 2}, ">=", 4)
    lp.add({"x": 3, "y": 1}, ">=", 6)
    lp.set_objective({"x": 1, "y": 1})
    st, v, sol = lp.solve()
    assert st == "optimal" and v == Fraction(14, 5) and sol == {"x": Fraction(8, 5), "y": Fraction(6, 5)}
    lp = LinearProgram()
    lp.add({"x": 1}, ">=", 5)
    lp.add({"x": 1}, "<=", 3)
    lp.set_objective({"x": 1})
    assert lp.solve()[0] == "infeasible"
    lp = LinearProgram()
    lp.add({"x": 1, "y": -1}, "==", 1)
    lp.set_objective({"x": -1})
    assert lp.solve()[0] == "unbounded"


# ---- growth ------------------------------------------------------------------------

def test_growth_product_and_cube():
    assert growth_product(1) == 1 and growth_product(2) == Fraction(4, 3)
    assert cube_check(80)


def test_extremal_profile_satisfies_paper_constraints():
    viol, tight = check_profile(extremal_profile, 30)
    assert viol == []
    assert any(c.tag.startswith("T") for c in tight)   # tripling is tight for h >= a
    assert extremal_profile(1, 7) == 7 and extremal_profile(2, 2) == Fraction(10, 3)


def test_lp_minimum_matches_extremal_on_small_grid():
    res = diagonal_lower_bounds(9, [2, 3])
    assert res[2][1] == extremal_profile(2, 2) == Fraction(10, 3)
    assert res[3][1] == extremal_profile(3, 3) == Fraction(56, 9)
    assert float(res[2][1]) >= 2 ** (4 / 3) and float(res[3][1]) >= 3 ** (4 / 3)


def test_five_sector_is_implied_on_grid():
    cand = Candidate((2, 7), [(2, 1)] * 5)
    st, v, _ = implied_on_grid(cand, 9)
    assert st == "optimal" and v == 0
    assert cand.slack(extremal_profile) == 0


# ---- search ------------------------------------------------------------------------

def test_solve_weights_acyclic_and_cyclic():
    cross = {(0, 0): True, (1, 0): True, (1, 1): True, (1, 2): True, (2, 2): True, (0, 1): False, (0, 2): False,
             (2, 0): False, (2, 1): False}
    wY, wZ = solve_weights(cross, [(0, 0), (1, 1), (2, 2)], 3, 3)
    assert wY == [0, 1, 0] and wZ == [0, -1, 0]
    cyc = dict(cross)
    cyc[(0, 1)] = True  # now Y_0 -> Z_1 and Y_1 -> Z_0: a cycle
    assert solve_weights(cyc, [(0, 0), (1, 1), (2, 2)], 3, 3) is None


def test_search_rediscovers_three_sector_and_nothing_beats_extremal():
    results, stats = search(max_a=2, max_b=5, max_sectors=3)
    cands = {e["candidate"] for e in results}
    assert "P(2,4) >= P(1,2) + P(1,2) + P(1,2)" in cands  # the paper's (a,h)=(2,1) construction
    assert min(float(e["extremal_slack_float"]) for e in results) >= 0
    top = results[0]
    cert = certificate_for(top)
    assert verify_certificate(cert).ok
