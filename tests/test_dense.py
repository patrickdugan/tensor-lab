import pytest

from tensorlab.border import (conv_border_rank_certificate, interpolation_decomposition, koszul_bound,
                              koszul_rank, orbit_tangent_dimension, pad, tensor_from_decomposition)
from tensorlab.dense import (block_form, cross_terms, filtered_power_experiment, koszul_threshold, retained_part,
                             target, target_is_rectangle_tensor, verdict, verify_two_copy)
from tensorlab.tensors import convolution


def test_border_rank_of_convolution_is_a_plus_b_minus_one():
    c = conv_border_rank_certificate(3, 16)
    assert c["decomposition_verified"] and c["z_flattening_rank"] == 18 and c["border_rank"] == 18
    terms = interpolation_decomposition(2, 3, points=[-2, 0, 1, 5])
    assert tensor_from_decomposition((2, 3, 4), terms) == convolution(2, 3)


@pytest.mark.parametrize("p,n", [(2, 1), (2, 2), (2, 5), (3, 2)])
def test_block_form_and_target(p, n):
    S, deltas = block_form(p, n)
    assert retained_part(S, deltas) == target(p, n)
    assert target_is_rectangle_tensor(p, n)
    assert set(deltas.values()) <= {-1, 0, 1}
    assert cross_terms(p, n).nnz == (0 if n == 1 else (n - 1) * p * (p - 1))


@pytest.mark.parametrize("p,n", [(2, 2), (2, 8), (3, 3)])
def test_two_copy_restriction(p, n):
    assert verify_two_copy(p, n)


def test_koszul_rank_formulas():
    assert all(koszul_rank(convolution(3, B), 1, 0)[0] == 2 * B + 2 for B in range(2, 20))
    assert all(koszul_rank(target(2, n), 1, 0)[0] == 6 * n for n in range(1, 9))
    assert all(koszul_rank(convolution(5, B), 2, 0)[0] == 6 * B + 12 for B in range(3, 12))


def test_border_rank_lower_bound_of_rectangle_packings():
    assert koszul_bound(target(2, 8), 1, 0)[0] == 24
    assert koszul_bound(target(3, 2), 2, 0)[0] == 10


def test_staircase_is_optimal():
    for n in range(1, 7):
        assert koszul_threshold(2, n)[0] == 3 * n - 1
    for n in range(1, 4):
        assert koszul_threshold(3, n)[0] == 5 * n - 2


def test_c316_verdict():
    v = verdict(2, 8, 16)
    assert v["koszul_rank_source"] == 34 and v["koszul_rank_target"] == 48
    assert not v["degeneration_possible_by_koszul"]
    assert v["staircase_is_optimal"] and v["staircase_certified"]
    assert v["max_rectangles_in_source"] == 5
    assert v["two_copy_restriction_verified"]


def test_two_rectangles_in_c34_excluded_but_not_by_orbit_dimension():
    S = convolution(3, 4)
    T = pad(target(2, 2), (3, 4, 6))
    assert orbit_tangent_dimension(T) < orbit_tangent_dimension(S)     # dimension allows it
    assert koszul_rank(T, 1, 0)[0] > koszul_rank(S, 1, 0)[0]           # Koszul forbids it


def test_fourier_filters_need_two_to_the_n_copies():
    assert not filtered_power_experiment(2, 2, 2, (1, 1), 3)["realizable"]
    assert not filtered_power_experiment(2, 2, 2, (1, 2), 3)["realizable"]
    r = filtered_power_experiment(2, 2, 2, (1, 2), 4)
    assert r["realizable"] and r["weights_verified"]
