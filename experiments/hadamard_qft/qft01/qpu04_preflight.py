"""HAD-QFT-QPU04 environment and IBM preflight. Read-only at IBM; prints and stores no credentials.

Records package versions, the live allocation counters, backend availability and the fresh
calibration of the two frozen seven-qubit regions. Output: hardware_results/qpu04/preflight.json
"""
from __future__ import annotations
import json, platform, sys
from importlib import metadata
from pathlib import Path
import qpu01 as old
from qpu04_design import REGIONS, EDGES

HERE = Path(__file__).resolve().parent


def versions():
    out = {}
    for p in ('qiskit', 'qiskit-ibm-runtime', 'qiskit-aer', 'numpy', 'scipy', 'networkx', 'matplotlib', 'pytest', 'rustworkx'):
        try: out[p] = metadata.version(p)
        except metadata.PackageNotFoundError: out[p] = None
    return out


def main():
    h = old.hardware_helpers()
    service, backend = old.connection(h, 'ibm_kingston')
    u = old.usage(service)
    st = backend.status()
    snap = h['calibration_snapshot'](backend, source='had_qft_qpu04_preflight')
    t = backend.target
    q = snap['qubits'] if isinstance(snap['qubits'], dict) else dict(snap['qubits'])
    edges = {}
    for e in snap['edges']:
        a, b = e['qargs']; k = (min(a, b), max(a, b)); edges[k] = min(edges.get(k, 1.0), e['error'])
    regions = []
    for ri, reg in enumerate(REGIONS):
        info = dict(region=ri, nodes=list(reg), qubits={}, edges=[], ok=True)
        for n in reg:
            qq = q[str(n)]
            info['qubits'][n] = dict(readout_error=qq['readout_error'], sx_error=qq['sx_error'], t1_us=(qq['t1'] or 0)*1e6, t2_us=(qq['t2'] or 0)*1e6)
        for a, b in EDGES:
            pa, pb = reg[a], reg[b]; k = (min(pa, pb), max(pa, pb))
            supported = bool(t.instruction_supported('cz', (pa, pb)))
            err = edges.get(k)
            info['edges'].append(dict(logical=[a, b], physical=[pa, pb], cz_supported=supported, cz_error=err))
            if not supported or err is None or err >= 0.1: info['ok'] = False
        regions.append(info)
    out = dict(python=sys.version.split()[0], platform=platform.platform(), packages=versions(), backend=dict(name=backend.name, operational=bool(st.operational), pending_jobs=st.pending_jobs, num_qubits=backend.num_qubits),
               usage=u, calibration_timestamp=snap['timestamp'], regions=regions, basis=sorted(t.operation_names))
    dest = HERE / 'hardware_results' / 'qpu04'; dest.mkdir(parents=True, exist_ok=True)
    old.write(dest / 'preflight.json', out)
    print(json.dumps({k: out[k] for k in ('python', 'packages', 'backend', 'usage', 'calibration_timestamp')}, indent=1))
    for r in regions:
        print('region', r['region'], r['nodes'], 'ok', r['ok'], 'cz errors', [round(e['cz_error'], 4) for e in r['edges']], 'readout', [round(v['readout_error'], 4) for v in r['qubits'].values()])


if __name__ == '__main__':
    main()
