"""Offline tests, no network or QPU. Synthetic counts are labelled separately."""
import json
import numpy as np
import pytest
from qpu02 import plan, qiskit_check, native_check, analyze, ARMS, SHOTS
from compact import synthesize, physical_index
from recursive import simulate
from radial_qft import radial_matrix
import qpu01 as old


def test_compilers():
    for b in (3,4,5,6):
        for arm in ('gray','recursive','compact_exact','compact_approx'):
            r=synthesize(b,arm);assert r['operator_error']<=r['analytic_bound']+1e-10
            assert all(g[0]!='cx' or abs(g[1]-g[2])==1 for g in r['gates'])


def test_roster_and_analysis(tmp_path):
    for b in (4,5,6):
        p=plan(b);assert p==json.loads(json.dumps(p))
        assert len(p['rows'])==len({r['id'] for r in p['rows']})
        assert len(p['rows'])==15*len(ARMS[b])+13
        p2=plan(b,1);assert p['resources']==p2['resources']
        assert {r['id']:r for r in p['rows']}=={r['id']:r for r in p2['rows']}
        counts={};rng=np.random.default_rng(271126+b)
        for row in p['rows']:
            q=np.asarray(row['compiled_ideal']);q=q/q.sum();c=rng.multinomial(SHOTS,q)
            counts[row['id']]={f'{i:0{b}b}'[::-1]:int(v) for i,v in enumerate(c) if v}
        out=tmp_path/str(b);out.mkdir();old.write(out/'compiled.json',p);old.write(out/'counts.json',counts);analyze(out)
        a=old.read(out/'analysis.json')
        assert len(a['contrasts'])==3*len(ARMS[b])
        assert all(x['contrast']>.55 and abs(x['contrast']-x['ideal'])<.15 for x in a['contrasts'])
        assert all(a['rows'][r['id']]['target_prob']==1 for r in p['rows'] if r['arm'] in ('prep','spam'))
        with pytest.raises(ValueError):
            bad=counts.copy();bad.pop(next(iter(bad)));old.write(out/'counts.json',bad);analyze(out)


def test_qiskit_and_fake_backend():
    pytest.importorskip('qiskit')
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import generate_preset_pass_manager
    for b in (4,5,6):
        p=plan(b);qiskit_check(p)
        edges=[e for i in range(b-1) for e in ([i,i+1],[i+1,i])]
        backend=GenericBackendV2(num_qubits=b,basis_gates=['id','rz','sx','x','cx'],coupling_map=edges,seed=271126)
        pm=generate_preset_pass_manager(optimization_level=1,backend=backend,initial_layout=list(range(b)),seed_transpiler=0)
        for row in p['rows']:
            isa=pm.run(old.circuit(row,b));native_check(isa,list(range(b)),row,b)


def test_committed_hardware_counts_reproduce_analysis(tmp_path):
    """Re-analyze the committed raw counts and compare with the committed analysis.json."""
    from pathlib import Path
    root = Path(__file__).resolve().parent / 'hardware_results'
    stages = [d for d in ('qpu02_b5_r0', 'qpu02_b5_r1', 'qpu02_b6_r0') if (root / d / 'counts.json').exists()]
    if not stages:
        pytest.skip('hardware_results not present')
    for d in stages:
        out = tmp_path / d; out.mkdir()
        for f in ('compiled.json', 'counts.json'):
            (out / f).write_bytes((root / d / f).read_bytes())
        analyze(out)
        new, ref = old.read(out / 'analysis.json'), old.read(root / d / 'analysis.json')
        assert new['contrasts'] == ref['contrasts'] and new['comparisons'] == ref['comparisons']
        assert all(sum(c.values()) == SHOTS for c in old.read(out / 'counts.json').values())


def test_phase_blind_null(tmp_path):
    p=plan(5);out=tmp_path;old.write(out/'compiled.json',p)
    # Perfectly uniform phase-insensitive counts, no hardware claim.
    c={f'{i:05b}':SHOTS//32 for i in range(32)}
    old.write(out/'counts.json',{r['id']:c for r in p['rows']});analyze(out)
    a=old.read(out/'analysis.json');assert all(x['contrast']==0 for x in a['contrasts'])
    assert all(x['midpoint_vs_mix_tv']==x['imag_vs_mix_tv']==0 for x in a['contrasts'])
