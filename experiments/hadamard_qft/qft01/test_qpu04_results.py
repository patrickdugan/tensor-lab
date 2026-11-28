"""Tests tying the QPU04 verdict, audits and resource table to the committed raw data (no network)."""
import json
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
R = HERE / 'hardware_results'


def _need():
    if not (R / 'qpu04_b7_r0' / 'counts.json').exists(): pytest.skip('QPU04 data absent')


def test_registered_joint_verdict_from_committed_counts():
    _need()
    import qpu04
    rep = qpu04.joint_audit([R / 'qpu04_b7_r0', R / 'qpu04_b7_r1'])
    assert rep['primary_both_positive'] is False
    lo = [x['interval'][0] for x in rep['regions']]
    assert lo[0] < 0 < lo[1] < 1e-3                       # region 0 fails, region 1 is above zero by 0.0003
    assert [round(x['edge_difference'], 4) for x in rep['regions']] == [0.0713, 0.0928]


def test_independent_replay_agrees_with_registered_evaluator():
    _need()
    import qpu04, qpu04_replay as rp
    rep = qpu04.joint_audit([R / 'qpu04_b7_r0', R / 'qpu04_b7_r1'])
    for stage, reg in zip(('qpu04_b7_r0', 'qpu04_b7_r1'), rep['regions']):
        s = rp.stage_stats(stage)
        assert abs(s['primary']['difference'] - reg['edge_difference']) < 1e-12
        assert abs(s['primary']['joint_interval'][0] - reg['interval'][0]) < 1e-12
        assert all(s['D_minus_A'][p] > 0 for p in rp.PROBES)        # descriptive: D above A at all five probes
    assert abs(rp.RAD_JOINT - 0.0925) < 1e-4


def test_native_audit_gates_and_counts():
    _need()
    for r in (0, 1):
        a = json.loads((R / 'qpu04' / f'audit_native_r{r}.json').read_text())
        assert a['all_gates_pass'] and a['arms']['A']['native_cz'] == 276 and a['arms']['D']['native_cz'] == 199 < 200
        assert all(x['workspace'] == 0 and x['data_wires'] == 7 and x['native_vs_routed_max_entry_discrepancy'] < 1e-12 for x in a['arms'].values())


def test_resource_table_reproduces():
    _need()
    rows = json.loads((R / 'qpu04' / 'audit_resources.json').read_text())
    got = [(r['b'], r['previous_logical'], r['new_logical'], r['previous_routed'], r['new_routed']) for r in rows]
    assert got == [(5, 70, 68, 90, 86), (6, 112, 98, 157, 136), (7, 188, 138, 276, 199), (8, 330, 192, 494, 301)]
