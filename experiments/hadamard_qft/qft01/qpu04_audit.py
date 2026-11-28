"""HAD-QFT-QPU04 independent audits (CPU only; native audit also reads the live backend target).

resources  : reproduce the logical/routed CX table for b=5..8 (selective.py + route_dp.py).
commutator : check the selective-rotation identity with Qiskit's Operator/Statevector (an
             implementation of gate semantics independent of recursive.simulate) on random
             complex superpositions, including relative-phase cancellation, r = 1..8.
sevenqubit : spectral-norm error and analytic bound at b=7 for both compilers.
native     : full native-compilation audit of one frozen design (per arm resources, durations,
             calibration-derived error estimates, unitary discrepancies, acceptance gates).
Outputs go to hardware_results/qpu04/.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import qpu01 as old
from selective import core, count, bound, audit, selective_y, relative_flip
from route_dp import Router
from qpu04_design import REGIONS, EDGES, ORDERS

HERE = Path(__file__).resolve().parent
OUT = HERE / 'hardware_results' / 'qpu04'


def resources():
    rows = []
    edges8 = list(EDGES) + [(4, 7)]
    for b in (5, 6, 7, 8):
        E = edges8 if b == 8 else [e for e in EDGES if e[0] < b and e[1] < b]
        r = Router(b, E); t0 = time.monotonic()
        prev = core(b, 'direct', 'rev')
        new = core(b, 'commutator', 'rev', level_orders=ORDERS if b == 7 else None) if b == 7 else core(b, 'commutator', 'rev')
        if b == 7:   # frozen selection: modified top-boundary order for D, default for A
            prev = core(7, 'direct', 'rev')
            new = core(7, 'commutator', 'rev', level_orders=ORDERS)
        zp = r.route(prev); zn = r.route(new)
        row = dict(b=b, previous_logical=count(prev), new_logical=count(new), previous_routed=count(zp['gates']), new_routed=count(zn['gates']),
                   previous_dp=zp['cost'], new_dp=zn['cost'], seconds=round(time.monotonic()-t0, 1))
        rows.append(row); print(row, flush=True)
    OUT.mkdir(parents=True, exist_ok=True); old.write(OUT / 'audit_resources.json', rows)


# ---- independent gate semantics through Qiskit (wire w -> qiskit qubit n-1-w so indices are MSB-first)
def qiskit_circuit(gates, n):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    m = lambda w: n - 1 - w
    for g in gates:
        k = g[0]
        if k == 'x': qc.x(m(g[1]))
        elif k == 'h': qc.h(m(g[1]))
        elif k == 'p': qc.p(g[2], m(g[1]))
        elif k == 'ry': qc.ry(g[2], m(g[1]))
        elif k == 'rz': qc.rz(g[2], m(g[1]))
        elif k == 'cx': qc.cx(m(g[1]), m(g[2]))
        else: raise ValueError(g)
    return qc


def commutator():
    from qiskit.quantum_info import Operator, Statevector
    rng = np.random.default_rng(20261128)
    rows = []
    for r in range(1, 9):
        n = r + 1; L = 1 << r
        for theta in (-math.pi/2, 0.382, 2.2):
            g = selective_y(list(range(1, r+1)), 0, theta)
            qc = qiskit_circuit(g, n)
            # analytic target: Ry(theta) on wire 0 when wires 1..r are all zero, identity otherwise
            ref = np.eye(1 << n, dtype=complex); c, s = math.cos(theta/2), math.sin(theta/2)
            ref[0, 0] = c; ref[L, L] = c; ref[L, 0] = s; ref[0, L] = -s
            if n <= 8:
                U = Operator(qc).data; err_u = float(np.max(abs(U - ref)))     # full operator including phases
            else:
                err_u = None
            err = 0.
            for _ in range(24):
                x = rng.normal(size=1 << n) + 1j*rng.normal(size=1 << n); x /= np.linalg.norm(x)
                y = np.asarray(Statevector(x).evolve(qc).data)
                err = max(err, float(np.linalg.norm(y - ref @ x)))
            rows.append(dict(r=r, theta=theta, cx=count(g), operator_max_error=err_u, random_superposition_error=err))
            print(rows[-1], flush=True)
    # a relative flip is monomial: |U| is a permutation matrix on all computational states
    mono = []
    for r in range(1, 7):
        n = r + 1; U = Operator(qiskit_circuit(relative_flip(list(range(1, r+1)), 0), n)).data
        ix = np.arange(1 << n); goal = ix.copy(); goal[(ix & ((1 << r)-1)) == 0] ^= 1 << r
        mono.append(dict(r=r, monomial_error=float(np.max(abs(abs(U) - np.eye(1 << n)[:, goal])))))
    print(mono)
    assert all((x['operator_max_error'] is None or x['operator_max_error'] < 1e-11) and x['random_superposition_error'] < 1e-11 for x in rows)
    assert all(x['monomial_error'] < 1e-11 for x in mono)
    OUT.mkdir(parents=True, exist_ok=True); old.write(OUT / 'audit_commutator.json', dict(rows=rows, monomial=mono, passed=True))
    print('commutator audit passed')


def sevenqubit():
    out = {}
    for style, lo in (('direct', None), ('commutator', ORDERS)):
        g = core(7, style, 'rev', level_orders=lo)
        a = audit(g, 7, bound(7)); out[style] = a; print(style, a)
    old.write(OUT / 'audit_sevenqubit.json', out)


def native(design_path, out_name):
    import qpu03 as base, qpu04
    h = old.hardware_helpers(); d = old.read(Path(design_path))
    items, info = qpu04.build_rows(d)
    service, backend = old.connection(h, d['backend'])
    snap = h['calibration_snapshot'](backend, source='had_qft_qpu04_audit'); q = snap['qubits'] if isinstance(snap['qubits'], dict) else dict(snap['qubits'])
    err = {}
    for e in snap['edges']:
        a, b = e['qargs']; k = (min(a, b), max(a, b)); err[k] = min(err.get(k, 1.0), e['error'])
    res = {}
    for arm in d['arms']:
        name = arm['arm']; phys = arm['phys']
        rows = [(r, qc) for r, qc, _ in items if r['arm'] == name]
        ref = next(qc for r, qc in rows if r['id'] == f'{name}_edge_plus')
        isa = base.to_physical(ref, phys, 7)
        ops = isa.count_ops()
        used = sorted({isa.find_bit(x).index for inst in isa.data for x in inst.qubits})
        two = {k: int(v) for k, v in ops.items() if k in ('cz', 'cx', 'ecr', 'rzz', 'swap', 'iswap')}
        support = all(backend.target.instruction_supported(inst.operation.name, tuple(isa.find_bit(x).index for x in inst.qubits)) for inst in isa.data if inst.operation.name != 'barrier')
        esp = 1.0
        for inst in isa.data:
            nm = inst.operation.name; qs = [isa.find_bit(x).index for x in inst.qubits]
            if nm == 'cz': esp *= 1 - err[(min(qs), max(qs))]
            elif nm in ('sx', 'x'): esp *= 1 - q[str(qs[0])]['sx_error']
        meas = {isa.find_bit(i.clbits[0]).index: isa.find_bit(i.qubits[0]).index for i in isa.data if i.operation.name == 'measure'}
        ro = 1.0
        for c, p in meas.items(): ro *= 1 - q[str(p)]['readout_error']
        v = info[name]['ver']
        cm = info[name]['core_metrics']
        res[name] = dict(physical_qubits=phys, used_qubits=used, all_used_are_region_data_qubits=set(used) <= set(phys), data_wires=7, workspace=0,
                         native_cz=cm['cz'], total_native_two_qubit=sum(two.values()), two_qubit_gate_kinds=two, one_qubit=dict(sx=cm['sx'], x=cm['x'], rz=cm['rz']),
                         two_qubit_depth=cm['cz_depth'], depth=cm['depth'], core_size=cm['size'],
                         edge_circuit_duration_us=h['duration'](isa, backend.target)*1e6, estimated_success_core_and_prep=esp, readout_success=ro, estimated_success_total=esp*ro,
                         frozen_routed_cx=arm['routed_cx'], native_not_above_routed=cm['cz'] <= arm['routed_cx'], instruction_support_all=support,
                         native_vs_routed_max_entry_discrepancy=v['native_vs_routed_max_entry_error'], native_operator_error_vs_exact=v['native_operator_error'], analytic_bound=v['analytic_bound'],
                         rows_checked=len(rows))
    gates = dict(seven_physical_data_qubits=all(len(r['physical_qubits']) == 7 for r in res.values()), zero_workspace=all(r['workspace'] == 0 for r in res.values()),
                 native_checks_pass=all(r['native_vs_routed_max_entry_discrepancy'] < 1e-9 and r['native_operator_error_vs_exact'] <= r['analytic_bound'] + 1e-8 for r in res.values()),
                 instructions_supported=all(r['instruction_support_all'] and r['all_used_are_region_data_qubits'] for r in res.values()),
                 native_not_above_routed=all(r['native_not_above_routed'] for r in res.values()), armD_below_200=res['D']['native_cz'] < 200,
                 only_cz_two_qubit=all(set(r['two_qubit_gate_kinds']) == {'cz'} for r in res.values()))
    out = dict(design=Path(design_path).name, calibration_timestamp=snap['timestamp'], arms=res, gates=gates, all_gates_pass=all(gates.values()))
    OUT.mkdir(parents=True, exist_ok=True); old.write(OUT / out_name, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['resources', 'commutator', 'sevenqubit', 'native'])
    ap.add_argument('--design'); ap.add_argument('--out', default='audit_native.json')
    a = ap.parse_args()
    {'resources': resources, 'commutator': commutator, 'sevenqubit': sevenqubit}.get(a.command, lambda: native(a.design, a.out))()
