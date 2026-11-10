"""QC-23 experiment driver: sets, compilation artifacts, simulators.

Sub-commands (``python qc23_experiment.py <cmd> ...``):
  compile   transpile every circuit of a set on a backend (live or fixture)
  simulate  run a set on the 'surrogate' (synthetic, NOT hardware) or 'aer'
            (Qiskit Aer with NoiseModel.from_backend) simulator
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from qc23_ibm import make_echo, observable_bits
from qc23_nonclifford import make_kicked_ising
from qc23_compile import (audit_mapping, choose_line_calibrated, isa_metrics, qasm3_text,
                          qpy_bytes, sha256_text, transpile_on_path)
from qc23_noise import calibration_snapshot, restrict_snapshot
from qc23_stabilizer import extract_native_ops

HERE = Path(__file__).resolve().parent

SETS = {
    # Phase 3 evaluation circuits (never used for fitting).
    'eval_echo': dict(family='echo', ns=(8, 16), depths=(2, 4, 8), seeds=(3, 5), shots=2048),
    # Separate designated calibration subset used only to fit Model B.
    'fit_echo': dict(family='echo', ns=(8, 16), depths=(1, 3, 6), seeds=(11, 13), shots=2048),
    # Phase 5 held-out non-Clifford family (n=16 limited to two seeds/steps for cost).
    'nc_eval': dict(family='kicked_ising', ns=(8, 12), depths=(1, 2, 3, 4), seeds=(41, 42, 43, 44), shots=2048,
                    extra=dict(ns=(16,), depths=(2, 4), seeds=(41, 42))),
}


def list_items(set_name: str) -> list[dict]:
    spec = SETS[set_name]
    blocks = [(spec['ns'], spec['depths'], spec['seeds'])]
    if 'extra' in spec:
        e = spec['extra']
        blocks.append((e['ns'], e['depths'], e['seeds']))
    items = []
    for ns, depths, seeds in blocks:
        for n in ns:
            for d in depths:
                for s in seeds:
                    items.append({'id': f"{spec['family']}_n{n}_d{d}_s{s}", 'set': set_name,
                                  'family': spec['family'], 'n': n, 'depth': d, 'seed': s,
                                  'shots': spec['shots'], 'bits': list(observable_bits(n))})
    return items


def logical_circuit(item):
    if item['family'] == 'echo':
        return make_echo(item['n'], item['depth'], item['seed'])
    return make_kicked_ising(item['n'], item['depth'], item['seed'])


class Compiler:
    """Compile items on one backend, caching the best calibrated path per n."""

    def __init__(self, backend, snapshot: dict):
        self.backend, self.snapshot = backend, snapshot
        self._paths: dict[int, list[int]] = {}

    def path(self, n):
        if n not in self._paths:
            self._paths[n], _ = choose_line_calibrated(self.backend, n)
        return self._paths[n]

    def compile(self, item):
        n = item['n']
        physical = self.path(n)
        measured = logical_circuit(item).copy()
        measured.measure_all()
        isa = transpile_on_path(self.backend, measured, physical)
        audit = audit_mapping(isa, physical, n)
        ops = extract_native_ops(isa, physical, self.backend.target)
        return {'item': item, 'logical': logical_circuit(item), 'isa': isa, 'physical': physical,
                'audit': audit, 'ops': ops, 'metrics': isa_metrics(isa)}


def load_backend(name: str | None, fixture: str | None):
    if fixture:
        import qiskit_ibm_runtime.fake_provider as fp
        backend = getattr(fp, fixture)()
        return backend, f'fixture:{fixture}'
    from qc23_access import connect
    service, err = connect()
    if service is None:
        raise RuntimeError(err)
    return service.backend(name), 'live'


def cmd_compile(args):
    backend, mode = load_backend(args.backend, args.fixture)
    snap = calibration_snapshot(backend, source=mode)
    out = Path(args.out)
    (out / 'circuits').mkdir(parents=True, exist_ok=True)
    comp = Compiler(backend, snap)
    records, isas = [], []
    for item in list_items(args.set):
        c = comp.compile(item)
        qasm = qasm3_text(c['isa'])
        (out / 'circuits' / f"{item['id']}.qasm").write_text(qasm)
        records.append({**item, 'physical_path': c['physical'], 'mapping': c['audit'],
                        'metrics': c['metrics'], 'qasm3_sha256': sha256_text(qasm),
                        'forward_two_qubit': None})
        isas.append(c['isa'])
    qpy = qpy_bytes(isas)
    (out / f'{args.set}.qpy').write_bytes(qpy)
    meta = {'set': args.set, 'backend': backend.name, 'mode': mode,
            'calibration_timestamp': snap['timestamp'], 'qpy_sha256': hashlib.sha256(qpy).hexdigest(),
            'circuits': records}
    (out / f'compiled_{args.set}.json').write_text(json.dumps(meta, indent=2) + '\n')
    (out / 'calibration_snapshot.json').write_text(json.dumps(snap) + '\n')
    print(f"compiled {len(records)} circuits of {args.set} on {backend.name} ({mode})")


def cmd_simulate(args):
    from qc23_surrogate import Drift, run_surrogate
    from qc23_ibm import z_estimators
    backend, mode = load_backend(args.backend, args.fixture)
    snap = calibration_snapshot(backend, source=mode)
    comp = Compiler(backend, snap)
    out = Path(args.out) / args.source
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{args.set}_counts.json'
    done = json.loads(path.read_text()) if path.exists() else {}
    shots_override = args.shots
    aer_sim = None
    for item in list_items(args.set):
        if item['id'] in done:
            continue
        if args.max_n and item['n'] > args.max_n:
            continue
        c = comp.compile(item)
        shots = shots_override or item['shots']
        if item['n'] > 12:
            shots = min(shots, args.big_shots)   # 16-qubit trajectory simulation cost cap
        seed = int(hashlib.sha256(item['id'].encode()).hexdigest()[:8], 16)
        if args.source == 'surrogate':
            counts = run_surrogate(c['ops'], item['n'], restrict_snapshot(snap, c['physical']),
                                   shots, seed, Drift())
            meta = {'drift': Drift().as_dict(), 'disclaimer': 'SYNTHETIC SURROGATE - NOT IBM HARDWARE'}
        else:
            if aer_sim is None:
                from qiskit_aer import AerSimulator
                aer_sim = AerSimulator.from_backend(backend, method='statevector')
            sim = aer_sim
            counts = sim.run(c['isa'], shots=shots, seed_simulator=seed).result().get_counts()
            meta = {'noise_model': 'qiskit_aer NoiseModel.from_backend (calibration snapshot)'}
        done[item['id']] = {**item, 'shots': shots, 'counts': counts, 'source': args.source,
                            'physical_path': c['physical'], 'backend': backend.name, **meta,
                            'z': z_estimators(counts, item['n'], tuple(item['bits']))}
        path.write_text(json.dumps(done))
        print('done', item['id'], flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('compile', 'simulate'):
        s = sub.add_parser(name)
        s.add_argument('--set', required=True, choices=sorted(SETS))
        s.add_argument('--backend')
        s.add_argument('--fixture', help='qiskit_ibm_runtime.fake_provider class, e.g. FakeFez')
        s.add_argument('--out', default='results')
    sim = sub.choices['simulate']
    sim.add_argument('--source', required=True, choices=['surrogate', 'aer'])
    sim.add_argument('--shots', type=int)
    sim.add_argument('--max-n', type=int, default=0)
    sim.add_argument('--big-shots', type=int, default=1024)
    args = p.parse_args()
    {'compile': cmd_compile, 'simulate': cmd_simulate}[args.cmd](args)


if __name__ == '__main__':
    main()
