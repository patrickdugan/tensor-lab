from fractions import Fraction

from tensorlab.border import koszul_rank, pad
from tensorlab.border_construct import find_additive, n2_square_certificate
from tensorlab.dense import target
from tensorlab.tensors import convolution, tensor_product


def test_square_two_copy_certificate():
    r = n2_square_certificate()
    assert r["ok"]
    assert r["single_factor_ell_vanishes_exactly_on_cross"]
    assert r["L_equals_product_and_is_additive"]
    assert r["w_times_L_is_target"]
    assert r["support_terms"] == 100


def test_one_copy_of_the_square_is_impossible():
    C = convolution(3, 4)
    T = pad(target(2, 2), (3, 4, 6))
    S2, T2 = tensor_product(C, C), tensor_product(T, T)
    assert koszul_rank(T2, 1, 0)[0] > koszul_rank(S2, 1, 0)[0]


def test_product_mechanism_fails_for_three_factors():
    def split3(w, base):
        return w // (base * base), (w // base) % base, w % base

    def weight3(x, y, z, dv):
        prod = Fraction(1)
        for yt, zt in zip(split3(y, 4), split3(z, 4)):
            prod *= Fraction(4 - yt - (zt + 1))
        return 1 / prod

    assert find_additive(2, 2, 3, weight3) is None


def test_three_copy_square_every_n():
    from tensorlab.border_construct import verify_three_copy_square, three_copy_restriction_from_convolution
    for n in range(2, 9):
        assert verify_three_copy_square(n)["ok"]
    r = three_copy_restriction_from_convolution(8)          # C(3,16)
    assert r["equal"] and r["target_terms"] == 1024


def test_three_copy_negative_control():
    """Dropping any one of the three terms breaks the identity."""
    from tensorlab import border_construct as bc
    orig = bc.three_copy_square_terms
    try:
        for drop in range(3):
            bc.three_copy_square_terms = lambda n, d=drop: [t for i, t in enumerate(orig(n)) if i != d]
            assert not bc.three_copy_restriction_from_convolution(3)["equal"]
    finally:
        bc.three_copy_square_terms = orig


def test_first_order_obstruction_exact():
    from tensorlab.border_construct import first_order_obstruction
    assert first_order_obstruction(2, 2, 2)["first_order_possible"]
    r = first_order_obstruction(2, 2, 3)
    assert not r["first_order_possible"] and r["forced_zero_retained"] == 336
    r = first_order_obstruction(2, 8, 2)                     # C(3,16), two factors
    assert not r["first_order_possible"] and r["forced_zero_retained"] == 744


def test_coarse_reduction_lifts():
    from tensorlab.border_construct import check_coarse_lift
    from tensorlab.numeric_restriction import coarse_types
    ts = coarse_types()
    assert len(ts) == 12 and sum(t[3] for t in ts) == 8
    for n in range(2, 10):
        r = check_coarse_lift(n)
        assert r["ok"]
        if n >= 3:
            assert r["coarse_types_used"] == 12


def test_coarse_problem_is_cyclic_and_calibrated():
    from tensorlab.border_construct import coarse_cyclic_facts
    f = coarse_cyclic_facts()
    assert f["ok"] and f["cyc_flattening_ranks"] == [3, 4, 4]


def test_limit_system_solver_sanity():
    """One factor of the coarse problem is exactly one tangential term (two copies);
    two factors are not (the first-order mechanism has forced zeros there)."""
    from tensorlab.numeric_restriction import coarse_pattern, limit_system_search
    assert limit_system_search(coarse_pattern(1), 0, 1, starts=4, iters=200)["residual"] < 1e-20
    assert limit_system_search(coarse_pattern(2), 0, 1, starts=2, iters=150)["residual"] > 1.0


def test_two_copy_square_every_n():
    """Parity-twisted pair: T_n^{(x)2} is a degeneration of two copies of C(3,2n)^{(x)2}."""
    from tensorlab.border_construct import parity_twist_pair_certificate, parity_twist_pair_from_convolution
    for n in range(2, 7):
        assert parity_twist_pair_certificate(n)["ok"]
    assert parity_twist_pair_from_convolution(3)["ok"]
    assert parity_twist_pair_from_convolution(8)["ok"]          # C(3,16)


def test_two_copy_square_negative_controls():
    """Corrupting the certificate breaks the independent check; the coarse (cyclic)
    version of the same ansatz has no solution."""
    from fractions import Fraction
    from tensorlab.border_construct import parity_twist_pair_certificate, parity_twist_pair_from_convolution
    c = parity_twist_pair_certificate(3)
    sol = list(c["solution"])
    k = next(i for i, v in enumerate(sol) if v != 0)
    sol[k] += Fraction(1, 7)
    bad = dict(c, solution=sol)
    assert not parity_twist_pair_from_convolution(3, bad)["ok"]
    from tensorlab.border_construct import parity_twist_pair_coarse_solvable
    assert not parity_twist_pair_coarse_solvable()
