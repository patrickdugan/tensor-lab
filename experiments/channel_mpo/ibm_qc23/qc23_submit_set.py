"""Submit a named QC-23 set (e.g. nc_eval) to IBM hardware under the same authorization gate as `qc23_ibm.py submit`.

  python qc23_submit_set.py --set nc_eval --backend ibm_kingston --out DIR            # compile, print plan hash
  python qc23_submit_set.py --set nc_eval --backend ibm_kingston --out DIR --submit --authorization FILE

Counts are written in the schema `qc23_analysis.py --source ibm` reads (`<out>/ibm/<set>_counts.json`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qc23_access import check_authorization, connect, estimate_qpu_seconds, plan_hash
from qc23_compile import qasm3_text, qpy_bytes, sha256_text
from qc23_experiment import Compiler, list_items
from qc23_ibm import z_estimators
from qc23_noise import calibration_snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True)
    ap.add_argument('--backend', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--authorization')
    args = ap.parse_args()
    out = Path(args.out)
    (out / 'circuits').mkdir(parents=True, exist_ok=True)
    service, err = connect(None)
    if service is None:
        ap.error(f'IBM account unavailable: {err}')
    backend = service.backend(args.backend)
    snap = calibration_snapshot(backend, source='live')
    comp = Compiler(backend, snap)
    rows, isas = [], []
    for item in list_items(args.set):
        c = comp.compile(item)
        qasm = qasm3_text(c['isa'])
        (out / 'circuits' / f"{item['id']}.qasm").write_text(qasm)
        rows.append({**item, 'physical_path': c['physical'], 'mapping': c['audit'], 'metrics': c['metrics'],
                     'qasm3_sha256': sha256_text(qasm)})
        isas.append(c['isa'])
    sha = plan_hash(rows)
    shots = rows[0]['shots']
    est = estimate_qpu_seconds([r['metrics']['depth'] * 5e-7 for r in rows], shots)
    meta = {'set': args.set, 'backend': backend.name, 'calibration_timestamp': snap['timestamp'],
            'plan_sha256': sha, 'estimated_qpu_seconds': est, 'circuits': rows}
    (out / 'compiled.json').write_text(json.dumps(meta, indent=2) + '\n')
    (out / 'compiled.qpy').write_bytes(qpy_bytes(isas))
    print(f'Compiled {len(rows)} circuits of {args.set} on {backend.name}; plan_sha256={sha}; estimated {est:.1f} QPU-s')
    if not args.submit:
        return
    ok, why = check_authorization(args.authorization, backend=backend.name, sha=sha, estimated_seconds=est)
    if not ok:
        ap.error(f'hardware execution NOT authorised: {why} (estimated {est:.1f} QPU-s, plan_sha256={sha})')
    from qiskit_ibm_runtime import SamplerV2
    job = SamplerV2(mode=backend).run(isas, shots=shots)
    job_id = job.job_id()
    (out / 'job.json').write_text(json.dumps({'job_id': job_id, 'backend': backend.name, 'plan_sha256': sha}, indent=2) + '\n')
    print('Submitted IBM job:', job_id, flush=True)
    result = job.result()
    done = {}
    for row, entry in zip(rows, result):
        counts = entry.data.meas.get_counts()
        item = {k: row[k] for k in ('id', 'set', 'family', 'n', 'depth', 'seed', 'shots', 'bits')}
        done[row['id']] = {**item, 'counts': counts, 'source': 'ibm', 'physical_path': row['physical_path'],
                           'backend': backend.name, 'job_id': job_id,
                           'z': z_estimators(counts, row['n'], tuple(row['bits']))}
    (out / 'ibm').mkdir(exist_ok=True)
    (out / 'ibm' / f'{args.set}_counts.json').write_text(json.dumps(done))
    print(f'Wrote {out}/ibm/{args.set}_counts.json')


if __name__ == '__main__':
    main()
