"""QC-26A independent audit of a frozen build directory (no IBM calls, no QPU use).

  python qc26a_audit.py --build DIR --out receipt.json [--node PATH_TO_NODE]

Checks, each recorded as a pass/fail line in the receipt:
  1. manifest/QPY/QASM/proposal/calibration hashes (reuses qc26a_compile.validate_frozen)
  2. every circuit is confined to its planned physical path; every CZ acts on a calibrated
     coupling-map edge listed in the build snapshot; native gate set and counts
  3. exact-statevector sign conventions: the compiled ISA circuits are re-simulated with an injected
     coherent SX offset d (SX -> Rx(pi/2+d)) and a diagonal CZ error exp(-i/2 (a Z_t + delta Z_c Z_t)),
     then the exact outcome distributions are rounded to 384-shot integer counts and passed through the
     committed JavaScript estimator; recovered offsets must have the injected sign and size
  4. calibration age at audit time and the nominal runtime estimate
"""
from __future__ import annotations
import argparse, datetime as dt, json, math, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import qc26a_compile as C


def local_circuit(isa, path, dsx, cz_err):
    """Re-express an ISA circuit on len(path) qubits, injecting coherent errors after SX and CZ."""
    from qiskit import QuantumCircuit
    n = len(path)
    qc = QuantumCircuit(n, n)
    idx = {p: i for i, p in enumerate(path)}
    for inst in isa.data:
        op = inst.operation
        qs = [isa.find_bit(q).index for q in inst.qubits]
        if op.name == 'barrier':
            continue
        if op.name == 'measure':
            cl = isa.find_bit(inst.clbits[0]).index
            qc.measure(idx[qs[0]], cl)
            continue
        loc = [idx[q] for q in qs]            # raises KeyError if the circuit leaves its path
        if op.name == 'sx':
            qc.rx(math.pi / 2 + dsx.get(qs[0], 0.0), loc[0])
        elif op.name == 'cz':
            qc.cz(loc[0], loc[1])
            a, d = cz_err.get((qs[0], qs[1]), (0.0, 0.0))
            qc.rz(a, loc[1])                   # exp(-i a Z/2)
            qc.rzz(d, loc[0], loc[1])          # exp(-i d Z Z/2)
        elif op.name in ('rz', 'x'):
            qc.append(op, loc)
        else:
            raise ValueError(f'unexpected gate {op.name}')
    return qc


def probs_as_counts(qc, shots):
    from qiskit.quantum_info import Statevector
    u = qc.remove_final_measurements(inplace=False)
    pd = Statevector.from_instruction(u).probabilities_dict()
    keys = sorted(pd)
    raw = np.array([pd[k] * shots for k in keys])
    base = np.floor(raw).astype(int)
    rem = shots - base.sum()
    order = np.argsort(-(raw - base))
    base[order[:rem]] += 1
    return {k: int(v) for k, v in zip(keys, base) if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--node', default='node')
    a = ap.parse_args()
    results = []

    def check(name, ok, detail=''):
        results.append({'check': name, 'pass': bool(ok), 'detail': detail})
        print(('PASS ' if ok else 'FAIL ') + name + (f': {detail}' if detail else ''))

    meta, isas = C.validate_frozen(a.build)
    check('frozen manifest, proposal, calibration, QPY and QASM hashes reproduce', True, meta['plan_sha256'])
    snap = json.loads((a.build / 'calibration_build.json').read_text())

    edges = {tuple(sorted(e['qargs'])) for e in snap['edges']}
    bad_path = []
    for row, isa in zip(meta['circuits'], isas):
        path = set(row['path'])
        for inst in isa.data:
            qs = {isa.find_bit(q).index for q in inst.qubits}
            if inst.operation.name != 'barrier' and not qs <= path:
                bad_path.append(row['id'])
            if inst.operation.name == 'cz' and tuple(sorted(qs)) not in edges:
                bad_path.append(row['id'] + ':cz-not-on-calibrated-edge')
        if row['mapping']['swaps'] != 0 or row['mapping']['measurement_map'] != {str(i): p for i, p in enumerate(row['path'])}:
            bad_path.append(row['id'] + ':mapping')
    check('all 82 circuits stay on their planned physical qubits; CZ only on calibrated edges; zero swaps',
          not bad_path, ','.join(sorted(set(bad_path))[:5]))
    total = {}
    for row in meta['circuits']:
        for k, v in row['gate_counts'].items():
            total[k] = total.get(k, 0) + v
    check('native gate set is {rz, sx, x, cz, measure, barrier}', set(total) <= {'rz', 'sx', 'x', 'cz', 'measure', 'barrier'}, json.dumps(total))

    # Exact-statevector sign audit through the committed estimator.
    plan = C.load_plan()
    inj_sx = {3: 0.004, 5: -0.025, 21: 0.0, 23: 0.04}
    inj_cz = {(4, 5): (0.013, 0.032), (6, 5): (-0.012, -0.022), (16, 23): (0.010, 0.047),
              (22, 23): (0.006, -0.038), (4, 3): (-0.008, 0.0), (22, 21): (0.009, 0.0)}
    counts = {}
    for row, isa in zip(meta['circuits'], isas):
        cz_err = {}
        if row['family'] == 'cz':
            cz_err = {(row['path'][0], row['path'][1]): inj_cz[(row['path'][0], row['path'][1])],
                      (row['path'][1], row['path'][0]): inj_cz[(row['path'][0], row['path'][1])]}
        qc = local_circuit(isa, row['path'], inj_sx if row['family'] == 'sx' else {}, cz_err)
        counts[row['id']] = probs_as_counts(qc, row['shots'])
    with tempfile.TemporaryDirectory() as td:
        cp, ap_ = Path(td) / 'counts.json', Path(td) / 'analysis.json'
        cp.write_text(json.dumps(counts))
        subprocess.run([a.node, str(HERE / 'qc26a_design.mjs'), 'analyze', str(cp), str(ap_)], check=True,
                       stdout=subprocess.DEVNULL)
        est = json.loads(ap_.read_text())
    rec = {}
    ok_sx = True
    for q, d in inj_sx.items():
        got = est['sx'][str(q)]['offset_rad']
        rec[f'sx_{q}'] = {'injected': d, 'recovered': got}
        ok_sx &= abs(got - d) <= 0.003 and (d == 0 or np.sign(got) == np.sign(d))
    check('SX sign convention: recovered offsets match injected within 0.003 rad with correct sign', ok_sx, json.dumps(rec))
    rec = {}
    ok_cz = True
    for (c, t), (loc, zz) in inj_cz.items():
        e = est['cz'][f'{c}_{t}']
        rec[f'cz_{c}_{t}'] = {'zz_injected': zz, 'zz_recovered': e['zz_phase_rad'],
                              'local_injected': loc, 'local_recovered': e['local_phase_rad']}
        ok_cz &= abs(e['zz_phase_rad'] - zz) <= 0.005 and abs(e['local_phase_rad'] - loc) <= 0.005
        ok_cz &= (zz == 0 or np.sign(e['zz_phase_rad']) == np.sign(zz))
    check('CZ sign convention: recovered conditional and local phases match injected within 0.005 rad with correct sign',
          ok_cz, json.dumps(rec))

    ct = dt.datetime.fromisoformat(meta['compiled_at'])
    age_min = (dt.datetime.now(dt.timezone.utc) - ct).total_seconds() / 60
    check('estimate is under the 20 s ceiling', meta['estimated_qpu_seconds'] <= 20, f"{meta['estimated_qpu_seconds']:.5f} s")
    receipt = {'plan_sha256': meta['plan_sha256'], 'qpy_sha256': meta['qpy_sha256'], 'calibration_timestamp': snap['timestamp'],
               'compiled_at': meta['compiled_at'], 'age_minutes_at_audit': age_min,
               'estimated_qpu_seconds': meta['estimated_qpu_seconds'], 'checks': results, 'all_pass': all(r['pass'] for r in results)}
    a.out.write_text(json.dumps(receipt, indent=2) + '\n')
    print('ALL PASS' if receipt['all_pass'] else 'FAILURES PRESENT')
    sys.exit(0 if receipt['all_pass'] else 1)


if __name__ == '__main__':
    main()
