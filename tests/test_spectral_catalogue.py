import math

from tensorlab.search import _dominates, classify, mark_dominance, search
from tensorlab.spectral import (conv_support, dense_rectangle_test, log_support_functional,
                                rectangle_block_support, shared_rectangles_log_zeta)


def test_support_functional_known_values():
    lo, up, _ = log_support_functional(conv_support(1, 7), (1 / 3, 1 / 3, 1 / 3))
    assert abs(lo - 2 / 3 * math.log(7)) < 1e-8 and abs(up - lo) < 1e-8
    lo, up, _ = log_support_functional(conv_support(3, 5), (1.0, 0.0, 0.0))
    assert abs(lo - math.log(3)) < 1e-8


def test_shared_rectangles_reduction_matches_direct_computation():
    # n identical rectangle blocks sharing X, computed directly vs the exact reduction
    p, q, n = 2, 2, 3
    sup = [(x, (u, i), (v, i)) for (x, u, v) in rectangle_block_support(p, q) for i in range(n)]
    # relabel tuple coordinates as integers
    ys = {y: k for k, y in enumerate(sorted({s[1] for s in sup}))}
    zs = {z: k for k, z in enumerate(sorted({s[2] for s in sup}))}
    flat = [(s[0], ys[s[1]], zs[s[2]]) for s in sup]
    th = (0.5, 0.3, 0.2)
    lo, up, _ = log_support_functional(flat, th, iters=6000)
    rlo, rup = shared_rectangles_log_zeta(p, q, n, th, iters=6000)
    assert abs(lo - rlo) < 1e-6 and abs(up - rup) < 1e-6


def test_dense_rectangles_not_obstructed_small():
    rows = dense_rectangle_test(3, 2, 2, [2, 4], step=0.25, iters=3000)
    assert all(r["no_obstruction_certified_on_grid"] and not r["obstruction"] for r in rows)


def test_dominance():
    assert _dominates([(2, 3), (2, 3)], [(1, 3), (2, 2)])
    assert not _dominates([(2, 3)], [(1, 1), (1, 1)])
    results, _ = search(2, 6, 3)
    for e in results:
        e["class"] = classify(e)
    mark_dominance(results)
    nd = [e for e in results if e["dominated_by"] is None]
    assert 0 < len(nd) < len(results)
    assert any(e["candidate"] == "P(2,4) >= P(1,2) + P(1,2) + P(1,2)" for e in nd)
