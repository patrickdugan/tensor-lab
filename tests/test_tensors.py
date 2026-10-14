from fractions import Fraction

from tensorlab.tensors import (Matrix, SparseTensor, convolution, cyclic_convolution, direct_sum,
                               dot_product, make_family, matrix_multiplication, shared_leg_sum,
                               tensor_product)


def test_convolution_shape_and_count():
    t = convolution(2, 4)
    assert t.dims == (2, 4, 5)
    assert t.nnz == 8
    assert t[(1, 3, 4)] == 1 and t[(1, 3, 3)] == 0


def test_convolution_flattening_ranks():
    t = convolution(3, 5)
    assert t.flattening_rank(0) == 3
    assert t.flattening_rank(1) == 5
    assert t.flattening_rank(2) == 7  # a + b - 1 = rank of the output flattening


def test_permute_and_pullback():
    t = convolution(2, 3)
    s = t.permute_legs("swap_YZ")
    assert s.dims == (2, 4, 3)
    assert s[(1, 2, 1)] == 1
    back = s.permute_legs("swap_YZ")
    assert back == t
    sub = t.pullback([[0, 1], [1, 2], [1, 2, 3]], (2, 2, 3))
    assert sub == convolution(2, 2)


def test_restrict_matches_pullback_for_zero_one_maps():
    t = convolution(3, 3)
    A = Matrix.identity(3)
    B = Matrix.from_map(2, 3, [0, 1])
    C = Matrix.from_map(4, 5, [0, 1, 2, 3])
    assert t.restrict(A, B, C) == t.pullback([[0, 1, 2], [0, 1], [0, 1, 2, 3]], (3, 2, 4))


def test_tensor_product_and_sums():
    a, b = convolution(2, 2), dot_product(3)
    p = tensor_product(a, b)
    assert p.nnz == a.nnz * b.nnz and p.dims == (2, 6, 9)
    d = direct_sum(a, a)
    assert d.nnz == 2 * a.nnz and d.dims == (4, 4, 6)
    s = shared_leg_sum([a, a], shared=0)
    assert s.dims == (2, 4, 6) and s.nnz == 8


def test_cyclic_and_matrix_multiplication():
    c = cyclic_convolution(4)
    assert c.nnz == 16 and c[(3, 2, 1)] == 1
    m = matrix_multiplication(2, 2, 2)
    assert m.nnz == 8 and m.dims == (4, 4, 4)
    assert m.flattening_rank(0) == 4


def test_json_roundtrip_and_family():
    t = convolution(2, 3)
    assert SparseTensor.from_json(t.to_json()) == t
    assert make_family({"family": "convolution", "a": 2, "b": 3}) == t
    assert make_family({"family": "tensor_product", "left": {"family": "convolution", "a": 2, "b": 2},
                        "right": {"family": "dot_product", "m": 3}}) == tensor_product(convolution(2, 2), dot_product(3))
