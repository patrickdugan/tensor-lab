import copy
import json

import pytest

from tensorlab.degeneration import degenerate, identify_block, check_identification
from tensorlab.sectors import general_sector_certificate, three_sector_certificate, three_sector_weights
from tensorlab.tensors import convolution
from tensorlab.verify import verify_certificate


@pytest.mark.parametrize("a,h", [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (4, 3), (2, 5), (5, 1), (5, 4)])
def test_three_sector_certificate_verifies(a, h):
    cert = three_sector_certificate(a, h)
    rep = verify_certificate(cert)
    assert rep.ok, rep.summary()
    assert len(cert["blocks"]) == 3
    assert [b["leg_permutation"] for b in cert["blocks"]] == ["identity", "swap_YZ", "identity"]
    assert cert["blocks"][1]["maps"]["X"] == list(range(a - 1, -1, -1))
    assert cert["min_weight"] == 0 and cert["max_weight"] == (1 if a >= 2 else 0)
    assert len(cert["retained_terms"]) == 3 * a * h
    assert len(cert["erased_terms"]) == a * (3 * h + a - 1) - 3 * a * h


def test_paper_figure_1_case():
    """C(2,4) with a=2,h=1: six retained terms, two erased (y=1,z=1,x0) and (y=2,z=3,x1)."""
    cert = three_sector_certificate(2, 1)
    erased = sorted(tuple(e[:3]) for e in cert["erased_terms"])
    assert erased == [(0, 1, 1), (1, 2, 3)]
    assert cert["weights"] == {"X": [0, 0], "Y": [0, 1, 1, 0], "Z": [0, 0, -1, 0, 0]}
    assert cert["sectors"] == {"Y": [[0], [1, 2], [3]], "Z": [[0, 1], [2], [3, 4]]}


def test_certificate_json_serialisable():
    cert = three_sector_certificate(3, 2)
    text = json.dumps(cert)
    assert verify_certificate(json.loads(text)).ok


def test_verifier_rejects_tampered_retained_set():
    cert = three_sector_certificate(2, 1)
    bad = copy.deepcopy(cert)
    bad["retained_terms"].pop()
    assert not verify_certificate(bad).ok


def test_verifier_rejects_wrong_weights():
    cert = three_sector_certificate(2, 1)
    bad = copy.deepcopy(cert)
    bad["weights"]["Z"][2] = 0
    assert not verify_certificate(bad).ok


def test_verifier_rejects_wrong_block_map():
    cert = three_sector_certificate(3, 2)
    bad = copy.deepcopy(cert)
    bad["blocks"][1]["maps"]["X"] = [0, 1, 2]  # drop the reversal
    assert not verify_certificate(bad).ok


def test_verifier_rejects_cross_contamination():
    cert = three_sector_certificate(2, 1)
    bad = copy.deepcopy(cert)
    # move a Y coordinate between sectors so a retained term crosses blocks
    bad["sectors"]["Y"] = [[0, 1], [2], [3]]
    bad["blocks"][0]["y_coords"] = [0, 1]
    bad["blocks"][1]["y_coords"] = [2]
    assert not verify_certificate(bad).ok


def test_general_sectors_with_padding():
    for a, sizes, kinds in [(2, [1, 2, 1], "NEN"), (3, [1, 2], "EN"), (2, [2, 1], "NE"), (3, [1, 1, 1, 1, 1], "NENEN"),
                            (4, [2, 2], "NN"), (3, [1, 1, 1], "EEE")]:
        cert = general_sector_certificate(a, sizes, kinds)
        rep = verify_certificate(cert)
        assert rep.ok, (a, sizes, kinds, rep.summary())
        assert len(cert["blocks"]) == len(sizes)
        for blk, h in zip(cert["blocks"], sizes):
            assert sorted((blk["target"]["a"], blk["target"]["b"])) == sorted((a, h))


def test_identify_block_finds_convolution_with_permutation():
    t = convolution(3, 2).permute_legs("cycle_XYZ")
    ident = identify_block(t)
    assert ident is not None and ident.family == "convolution" and set(ident.params) == {2, 3}
    assert check_identification(t, ident)


def test_degeneration_min_weight_selection():
    w, ys, zs = three_sector_weights(2, 1)
    deg = degenerate(convolution(2, 4), w)
    assert deg.min_weight == 0 and deg.retained.nnz == 6 and len(deg.erased) == 2
    assert all(wt == 1 for _, wt in deg.erased.values())
