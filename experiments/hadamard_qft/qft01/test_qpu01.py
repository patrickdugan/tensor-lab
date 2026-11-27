"""Offline tests for the proposed hardware pilot. No runtime calls."""
import json
import numpy as np
from qpu01 import plan, route, simulate, analyze, write, SHOTS

def test_roster_and_routing(tmp_path):
    for b, n in ((3,30),(4,38)):
        p=plan(b)
        assert len(p['rows'])==n and p==json.loads(json.dumps(p))
        assert len({r['id'] for r in p['rows']})==n
        for i in range(b):
            for j in range(b):
                if i!=j:
                    assert np.array_equal(simulate(route([('cx',i,j)]),b,np.eye(1<<b)),
                                          simulate([('cx',i,j)],b,np.eye(1<<b)))
        out=tmp_path/str(b);out.mkdir()
        write(out/'compiled.json',p)
        rng=np.random.default_rng(177+b);counts={}
        for r in p['rows']:
            q=np.array(r['ideal']);q/=q.sum()
            c=rng.multinomial(SHOTS,q)
            counts[r['id']]={f'{i:0{b}b}'[::-1]:int(v) for i,v in enumerate(c) if v}
        write(out/'counts.json',counts);analyze(out)
        a=json.loads((out/'analysis.json').read_text())
        assert all(a['summary'][x]['mean_echo_return']==1 for x in ('gray','recursive'))

def test_qiskit_reference():
    import pytest
    pytest.importorskip('qiskit')
    from qpu01 import qiskit_check
    for b in (3,4): qiskit_check(plan(b))

def test_fake_backend_isa():
    import pytest
    pytest.importorskip('qiskit')
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.quantum_info import Statevector
    from qpu01 import hardware_helpers, circuit
    h=hardware_helpers()
    for b in (3,4):
        edges=[e for i in range(b-1) for e in ([i,i+1],[i+1,i])]
        backend=GenericBackendV2(num_qubits=b,basis_gates=['id','rz','sx','x','cx'],coupling_map=edges,seed=271126)
        order=[int(f'{i:0{b}b}'[::-1],2) for i in range(1<<b)]
        for row in plan(b)['rows']:
            isa=h['transpile_on_path'](backend,circuit(row,b),list(range(b)),optimization_level=0)
            h['audit_mapping'](isa,list(range(b)),b)
            u=isa.remove_final_measurements(inplace=False)
            got=Statevector.from_instruction(u).probabilities()[order]
            assert np.max(abs(got-row['ideal']))<1e-10
