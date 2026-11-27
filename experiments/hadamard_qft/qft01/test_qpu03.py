"""Offline tests for QPU03: workspace compiler, native verification, frozen design, frozen analysis.
No network and no QPU. Synthetic counts are labeled and never reported as hardware data."""
import json
from pathlib import Path
import numpy as np
import pytest
import workspace as W
import topo as T
import qpu03
import qpu01 as old
from recursive import simulate

HERE = Path(__file__).resolve().parent
DES = HERE / 'hardware_results' / 'qpu03_design'


def test_margolus_exact_on_clean_target():
    U = simulate(W.margolus(0, 1, 2), 3, np.eye(8, dtype=complex))
    for a in (0, 1):
        for b in (0, 1):
            col = (a << 2) | (b << 1)
            out = U[:, col]; assert abs(out[col | (a & b)] - 1) < 1e-12


@pytest.mark.parametrize('b', [3, 4, 5])
def test_variants_exact_and_leak_free(b):
    for mode in ('exact', 'approx'):
        for m in (0, 1, 2, 3):
            g, n = W.core(b, mode, m)
            v = W.verify(g, b, m, bound=W.approx_bound(b) if mode == 'approx' else 0.)
            assert v['ok'] and v['max_workspace_leakage'] < 1e-12
            if mode == 'exact': assert v['operator_error'] < 1e-12
            assert v['random_state_error'] <= v['operator_error'] + 1e-12


def test_logical_counts_and_error_budget():
    expect = {(5, 'approx', 0): 70, (5, 'approx', 1): 68, (6, 'approx', 0): 112, (6, 'approx', 2): 98, (6, 'exact', 0): 260}
    for (b, mode, m), cx in expect.items():
        assert W.count_cx(W.core(b, mode, m)[0]) == cx
    assert abs(W.approx_bound(5) - 0.001633280517) < 1e-11 and abs(W.approx_bound(6) - 0.001639601147) < 1e-11
    for b in (5, 6):
        g, n = W.core(b, 'approx', 0)
        assert W.verify(g, b, 0, bound=W.approx_bound(b))['operator_error'] < 0.002


def test_exact_prefix_chain_is_exact_and_clean():
    for b, m, ep, cx in ((4, 3, 'chain', 74), (5, 4, 'chain', 144), (6, 4, 'auto', 206)):
        g, n = W.core(b, 'exact', m, exact_prep=ep)
        v = W.verify(g, b, m)
        assert v['ok'] and v['operator_error'] < 1e-12 and v['max_workspace_leakage'] < 1e-12
        assert W.count_cx(g) == cx
    assert W.chain_cost(5) < W.mux_prep_cost(5) and W.chain_cost(4) > W.mux_prep_cost(4)


def test_approximation_depth_frontier_and_pruning_saves_no_cnot():
    base = W.count_cx(W.core(6, 'approx', 0)[0])
    assert W.count_cx(W.core(6, 'approx', 0, prune={(5, 0)})[0]) == base          # pruning removes rotations only
    for r0, cx in ((4, 112), (3, 96), (2, 92)):
        g, n = W.core(6, 'approx', 0, approx_from=r0)
        v = W.verify(g, 6, 0, bound=W.approx_bound(6, r0))
        assert v['ok'] and W.count_cx(g) == cx


def test_rotation_pruning_bound_is_certified():
    prune = {(5, 0)}
    g, n = W.core(6, 'approx', 0, prune=prune)
    v = W.verify(g, 6, 0, bound=W.approx_bound(6) + W.prune_bound(6, prune))
    assert v['ok'] and v['operator_error'] > W.approx_bound(6)


def test_native_transpile_equals_logical():
    G = T.load_topology(DES / 'design_calibration.json')[0]
    for mode, m in (('approx', 0), ('approx', 1), ('exact', 1)):
        g, n = W.core(4, mode, m)
        shape = T.shapes(G, 4 + m, 68)[-1]
        tc = T.transpile_core(g, n, G, shape, seed=1, level=2)
        ok, err = T.verify_native(tc, g, n); assert ok, err
        assert all(i.operation.name in ('cz', 'sx', 'rz', 'x', 'barrier') for i in tc.data)


def _design(b):
    p = DES / f'design_b{b}.json'
    if not p.exists(): pytest.skip('design not frozen')
    return qpu03.load_design(p)


@pytest.mark.parametrize('b', [5, 6])
def test_frozen_design_rows_native_and_matched(b):
    d = _design(b)
    items, info = qpu03.build_rows(d)
    names = [a['arm'] for a in d['arms']]
    assert len(items) == 13*len(names) and len({r['id'] for r, _, _ in items}) == len(items)
    for r, qc, phys in items:
        qpu03.native_check(r, qc)
        if r['kind'] == 'core': assert r['arm'] in names
    for a in d['arms']:
        rows = [(r, qc) for r, qc, _ in items if r['arm'] == a['arm'] and r['kind'] == 'core']
        # the compiled core is shared: identical CZ skeleton in every probe row
        skel = {tuple(tuple(qc.find_bit(q).index for q in i.qubits) for i in qc.data if i.operation.name == 'cz') for _, qc in rows}
        assert len(skel) == 1
        assert sum(i.operation.name == 'cz' for i in rows[0][1].data) == a['core_metrics']['cz']


def test_frozen_analysis_on_synthetic_counts(tmp_path):
    d = _design(5)
    items, info = qpu03.build_rows(d)
    rng = np.random.default_rng(3)
    rows, counts = [], {}
    for r, qc, phys in items:
        nd, nw = r['n_data'], r['n_wires']
        pd, leak = qpu03.dist_from_local(qc, nd, nw)
        q = np.array(r['compiled_ideal']); q = 0.8*q + 0.2/len(q)       # synthetic depolarization
        full = np.zeros(1 << nw); full[::1 << (nw-nd)] = q
        c = rng.multinomial(qpu03.SHOTS, full/full.sum())
        counts[r['id']] = {f'{i:0{nw}b}'[::-1]: int(v) for i, v in enumerate(c) if v} if False else {}
        for i, v in enumerate(c):
            if v:
                data, anc = i >> (nw-nd), i & ((1 << (nw-nd))-1)
                key = f'{data:0{nd}b}' + (f'{anc:0{nw-nd}b}' if nw > nd else '')
                counts[r['id']][key[::-1]] = int(v)
        rows.append({**{k: v for k, v in r.items() if k != 'gates'}, 'metrics': {'num_two_qubit': 1, 'depth': 1}, 'duration_s': 1e-5})
    out = tmp_path; old.write(out / 'compiled.json', dict(shots=qpu03.SHOTS, design=d, design_sha256=qpu03.design_digest(d), rows=rows)); old.write(out / 'counts.json', counts)
    qpu03.analyze(out); a = old.read(out / 'analysis.json')
    assert len(a['signal']) == 3*len(d['arms']) and len(a['differences']) == 3*len(d['families']['difference'])
    assert all(0.4 < c['normalized'] < 0.95 for c in a['contrasts'] if c['probe'] == 'edge')
    assert all(v['leakage'] == 0 for v in a['rows'].values() if v['arm'] in ('A', 'D'))
    # radii: Bonferroni over the registered family sizes
    ms = len(d['families']['signal']); n = qpu03.SHOTS
    assert abs(a['signal'][0]['radius'] - np.sqrt(np.log(2/(0.05/ms))/n)) < 1e-12


def test_audit_reproduces_five_of_six():
    import audit_qpu02_simultaneous as A
    root = HERE / 'hardware_results'
    if not (root / 'qpu02_b5_r0' / 'counts.json').exists(): pytest.skip('QPU02 data absent')
    rep = A.audit([root / 'qpu02_b5_r0', root / 'qpu02_b5_r1'])
    assert rep['m'] == 6 and rep['n_survive_simultaneous'] == 5 and rep['edge_both_survive']
    assert [o['survives_simultaneous'] for o in rep['comparisons']] == [True, True, False, True, True, True]


def test_b7_design_records_the_native_verification_gate():
    d = _design(7)
    assert [a['arm'] for a in d['arms']] == ['C', 'D'] and d['families'] == {'signal': ['C', 'D'], 'difference': [['C', 'D']]}
    assert d['native_verification_tolerance'] == 1e-8
    assert d['rejected_candidates'] and all(r['native_vs_logical_max_entry_error'] > 1e-8 for r in d['rejected_candidates'])
    assert all(a['core_metrics']['cz'] == 320 for a in d['arms'])
