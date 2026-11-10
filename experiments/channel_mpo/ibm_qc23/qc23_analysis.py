"""QC-23 analysis: fit on the calibration subset, predict held-out circuits.

Reads only committed artifacts (compiled QPY + manifests, counts files,
calibration snapshot), so the analysis is reproducible offline and works
unchanged on IBM hardware counts: ``--source ibm`` reads
``results/ibm/<set>_counts.json`` produced by ``qc23_ibm.py submit``.

Three error sources are reported separately for every prediction:
  sampling   binomial standard error and 95% Wilson interval of the measured Z
  model      |prediction - measurement| (includes sampling); the noise model is
             an assumption, not something the numerical certificates cover
  numerical  QC-12 local-residual ledger / QC-14 dyadic certificate / MPO-vs-
             exact-reference difference for the *specified layered model only*
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
from pathlib import Path

import numpy as np

from qc23_ibm import observable_bits, wilson_interval, z_estimators
from qc23_noise import (ModelB, fit_model_b, model_a_faults, model_b_faults, predict_model_a,
                        predict_model_b, restrict_snapshot)
from qc23_nonclifford import ideal_z, make_kicked_ising
from qc23_stabilizer import extract_native_ops, pauli_flip_expectations
from qc23_mpo import (certified_z, dense_cone_z, effective_depolarizing_p, layered_clifford_reference,
                      mpo_z, ops_to_layers, cone_width)

THRESHOLD = 0.05           # pre-registered observable-accuracy threshold
DENSE_CONE_LIMIT = 12      # exact-cone density reference limit (sites)


def load_qpy(path):
    from qiskit import qpy
    with open(path, 'rb') as f:
        return qpy.load(f)


def load_set(root: Path, set_name: str):
    meta = json.loads((root / 'compiled' / f'compiled_{set_name}.json').read_text())
    isas = load_qpy(root / 'compiled' / f'{set_name}.qpy')
    return meta, {rec['id']: (rec, isa) for rec, isa in zip(meta['circuits'], isas)}


def load_counts(root: Path, source: str, set_name: str):
    p = root / source / f'{set_name}_counts.json'
    return json.loads(p.read_text()) if p.exists() else {}


def all_bit_estimates(entry):
    n = entry['n']
    z = z_estimators(entry['counts'], n, tuple(range(n)))
    return {int(b): v for b, v in z.items()}


def fit_calibration(root, snap, source):
    meta, items = load_set(root, 'fit_echo')
    counts = load_counts(root, source, 'fit_echo')
    records = []
    for cid, entry in counts.items():
        rec, isa = items[cid]
        ops = extract_native_ops(isa, rec['physical_path'])
        est = all_bit_estimates(entry)
        records.append({'id': cid, 'ops': ops, 'shots': entry['shots'],
                        'z': {b: v['expectation_z'] for b, v in est.items()},
                        'se': {b: v['standard_error'] for b, v in est.items()}})
    if not records:
        raise RuntimeError(f'no calibration counts for source {source}')
    params, diag = fit_model_b(records)
    return params, diag, [r['id'] for r in records]


def mpo_prediction(ops, n, faults, readout, bits, *, p_override=None, certify=False, bond_cap=16,
                   dense=True, clifford_reference=False):
    layers, info = ops_to_layers(ops, n)
    p = effective_depolarizing_p(faults, n, info['mpo_layers']) if p_override is None else p_override
    out = {'p_eff': p, 'mpo_layers': info['mpo_layers'], 'observables': {}}
    for b in bits:
        res = mpo_z(n, layers, p, b, bond_cap=bond_cap)
        ro = 1 - 2 * readout[b]
        o = {'z_mpo': res.expectation * ro, 'z_mpo_pre_readout': res.expectation,
             'qc12_bound': res.error_bound, 'cone_width': res.cone_width, 'max_bond': res.max_bond}
        if dense and res.cone_width <= DENSE_CONE_LIMIT:
            o['dense_cone_pre_readout'] = dense_cone_z(n, layers, p, b, max_sites=DENSE_CONE_LIMIT)
            o['numerical_gap_vs_dense'] = abs(res.expectation - o['dense_cone_pre_readout'])
        if clifford_reference:
            ref = layered_clifford_reference(n, layers, p, [b])[b]
            o['pauli_reference_pre_readout'] = ref
            o['numerical_gap_vs_pauli_reference'] = abs(res.expectation - ref)
        if certify:
            cert, why = certified_z(n, layers, p, b, bond_cap=bond_cap)
            if cert is None or why:
                o['qc14'] = {'supported': False, 'reason': why}
            else:
                o['qc14'] = {'supported': True, 'reported_error_upper': cert.reported_error_upper,
                             'physical_error_upper': cert.physical_error_upper,
                             'events': sum(cert.event_counts.values())}
        out['observables'][b] = o
    return out


def evaluate_set(root, snap, set_name, source, modelB, aer_source='aer', certify=False):
    meta, items = load_set(root, set_name)
    counts = load_counts(root, source, set_name)
    aer = load_counts(root, aer_source, set_name)
    rows = []
    for cid, entry in counts.items():
        rec, isa = items[cid]
        n = rec['n']
        bits = tuple(rec['bits'])
        snap_used = restrict_snapshot(snap, rec['physical_path'])
        ops = extract_native_ops(isa, rec['physical_path'])
        meas = z_estimators(entry['counts'], n, bits)
        if rec['family'] == 'echo':
            ideal = {b: 1.0 for b in bits}
        else:
            ideal = ideal_z(make_kicked_ising(n, rec['depth'], rec['seed']), bits) if n <= 16 else {}
        fa, ra = model_a_faults(ops, snap_used)
        fb, rb = model_b_faults(ops, modelB)
        mpo_a = mpo_prediction(ops, n, fa, ra, bits, certify=certify,
                               clifford_reference=(rec['family'] == 'echo'))
        mpo_b = mpo_prediction(ops, n, fb, rb, bits, dense=False)
        pauli_a = pauli_b = None
        if rec['family'] == 'echo':
            pauli_a = pauli_flip_expectations(ops, n, fa, ra, bits)
            pauli_b = pauli_flip_expectations(ops, n, fb, rb, bits)
        for b in bits:
            m = meas[str(b)]
            row = {'id': cid, 'family': rec['family'], 'n': n, 'depth': rec['depth'], 'seed': rec['seed'],
                   'bit': b, 'shots': m['shots'], 'measured': m['expectation_z'],
                   'sampling_se': m['standard_error'], 'wilson95_lo': m['wilson95'][0],
                   'wilson95_hi': m['wilson95'][1], 'hoeffding95': m['hoeffding95_halfwidth'],
                   'ideal': ideal.get(b), 'mpo_modelA': mpo_a['observables'][b]['z_mpo'],
                   'mpo_modelB': mpo_b['observables'][b]['z_mpo'],
                   'p_eff_A': mpo_a['p_eff'], 'p_eff_B': mpo_b['p_eff'],
                   'qc12_bound_A': mpo_a['observables'][b]['qc12_bound'],
                   'cone_width': mpo_a['observables'][b]['cone_width'],
                   'numerical_gap_dense': mpo_a['observables'][b].get('numerical_gap_vs_dense'),
                   'numerical_gap_pauli_ref': mpo_a['observables'][b].get('numerical_gap_vs_pauli_reference'),
                   'qc14': mpo_a['observables'][b].get('qc14')}
            if pauli_a:
                row['pauli_modelA'] = pauli_a[b]
                row['pauli_modelB'] = pauli_b[b]
            if cid in aer:
                row['aer_noisy'] = aer[cid]['z'][str(b)]['expectation_z']
                row['aer_se'] = aer[cid]['z'][str(b)]['standard_error']
            rows.append(row)
    return rows


PREDICTORS = ('ideal', 'mpo_modelA', 'mpo_modelB', 'pauli_modelA', 'pauli_modelB', 'aer_noisy')


def threshold_report(rows, key, threshold=THRESHOLD, seed=0, boot=2000):
    sel = [r for r in rows if r.get(key) is not None]
    if not sel:
        return None
    ok = [abs(r[key] - r['measured']) <= threshold for r in sel]
    k, t = sum(ok), len(ok)
    lo, hi = wilson_interval(k, t)
    # Cluster bootstrap over circuits (observables of one circuit are correlated).
    by = {}
    for r, o in zip(sel, ok):
        by.setdefault(r['id'], []).append(o)
    ids = sorted(by)
    rng = random.Random(seed)
    fracs = []
    for _ in range(boot):
        pick = [rng.choice(ids) for _ in ids]
        num = sum(sum(by[i]) for i in pick)
        den = sum(len(by[i]) for i in pick)
        fracs.append(num / den)
    fracs.sort()
    errs = [abs(r[key] - r['measured']) for r in sel]
    return {'predictor': key, 'threshold': threshold, 'observables': t, 'circuits': len(ids),
            'within': k, 'fraction': k / t, 'wilson95': [lo, hi],
            'cluster_bootstrap95': [fracs[int(0.025 * boot)], fracs[int(0.975 * boot) - 1]],
            'mean_abs_error': float(np.mean(errs)), 'median_abs_error': float(np.median(errs)),
            'max_abs_error': float(np.max(errs)),
            'sampling_aware_fraction': float(np.mean([abs(r[key] - r['measured']) <= threshold + 2 * r['sampling_se']
                                                       for r in sel]))}


def scale_test(snap, backend, out_n=(24, 32, 48), depths=(6, 10), bond_cap=16):
    """Cone wider than dense simulation: MPO vs exact Pauli reference (Clifford echo)
    and bond-cap convergence on a non-Clifford circuit."""
    from qc23_compile import choose_line_calibrated, transpile_on_path
    from qc23_ibm import make_echo
    rows = []
    for n in out_n:
        path, _ = choose_line_calibrated(backend, n)
        for d in depths:
            qc = make_echo(n, d, 3)
            qc.measure_all()
            isa = transpile_on_path(backend, qc, path)
            ops = extract_native_ops(isa, path)
            snap_used = restrict_snapshot(snap, path)
            fa, ra = model_a_faults(ops, snap_used)
            layers, info = ops_to_layers(ops, n)
            p = effective_depolarizing_p(fa, n, info['mpo_layers'])
            t = n // 2
            res = mpo_z(n, layers, p, t, bond_cap=bond_cap, max_cone=64)
            ref = layered_clifford_reference(n, layers, p, [t])[t]
            rows.append({'n': n, 'depth': d, 'family': 'echo', 'cone_width': res.cone_width,
                         'beyond_dense_limit': res.cone_width > DENSE_CONE_LIMIT, 'p_eff': p,
                         'mpo': res.expectation, 'exact_pauli_reference': ref,
                         'abs_gap': abs(res.expectation - ref), 'qc12_bound': res.error_bound,
                         'max_bond': res.max_bond, 'seconds': res.seconds})
    for n, steps in ((16, 4), (24, 6), (24, 8), (32, 8)):
        qc = make_kicked_ising(n, steps, 41)
        qc.measure_all()
        path, _ = choose_line_calibrated(backend, n)
        isa = transpile_on_path(backend, qc, path)
        ops = extract_native_ops(isa, path)
        fa, ra = model_a_faults(ops, restrict_snapshot(snap, path))
        layers, info = ops_to_layers(ops, n)
        p = effective_depolarizing_p(fa, n, info['mpo_layers'])
        t = n // 2
        vals = {}
        for cap in (4, 8, 16, 32):
            try:
                r = mpo_z(n, layers, p, t, bond_cap=cap, max_cone=64)
                vals[cap] = {'z': r.expectation, 'qc12_bound': r.error_bound, 'seconds': r.seconds,
                             'cone_width': r.cone_width}
            except MemoryError as exc:
                vals[cap] = {'error': str(exc)}
        ok = [v['z'] for v in vals.values() if 'z' in v]
        rows.append({'n': n, 'depth': steps, 'family': 'kicked_ising', 'p_eff': p, 'by_bond_cap': vals,
                     'spread_across_caps': (max(ok) - min(ok)) if ok else None,
                     'beyond_dense_limit': any(v.get('cone_width', 0) > DENSE_CONE_LIMIT for v in vals.values())})
    return rows


def write_csv(rows, path):
    keys = sorted({k for r in rows for k, v in r.items() if not isinstance(v, (dict, list))})
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def make_plots(rows, out: Path, tag: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for fam in ('echo', 'kicked_ising'):
        sel = [r for r in rows if r['family'] == fam]
        if not sel:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        for key, mk in (('mpo_modelA', 'o'), ('mpo_modelB', 's'), ('ideal', '^'), ('aer_noisy', 'x')):
            pts = [(r['measured'], r[key]) for r in sel if r.get(key) is not None]
            if pts:
                ax[0].scatter(*zip(*pts), marker=mk, s=22, label=key, alpha=0.8)
        ax[0].plot([-1, 1], [-1, 1], 'k--', lw=0.8)
        ax[0].fill_between([-1, 1], [-1 - THRESHOLD, 1 - THRESHOLD], [-1 + THRESHOLD, 1 + THRESHOLD], color='grey', alpha=0.15)
        ax[0].set_xlabel('measured <Z> (' + tag + ')')
        ax[0].set_ylabel('predicted <Z>')
        ax[0].legend(fontsize=7)
        depths = sorted({r['depth'] for r in sel})
        for key in ('mpo_modelA', 'mpo_modelB', 'ideal'):
            ys = [np.mean([abs(r[key] - r['measured']) for r in sel if r['depth'] == d and r.get(key) is not None])
                  for d in depths]
            ax[1].plot(depths, ys, marker='o', label=key)
        ax[1].axhline(THRESHOLD, color='k', ls='--', lw=0.8)
        ax[1].set_xlabel('depth / steps')
        ax[1].set_ylabel('mean |pred - measured|')
        ax[1].legend(fontsize=7)
        fig.suptitle(f'{fam} held-out circuits [{tag}]')
        fig.tight_layout()
        fig.savefig(out / f'{fam}_{tag}.png', dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results')
    ap.add_argument('--source', default='surrogate', choices=['surrogate', 'ibm'])
    ap.add_argument('--certify', action='store_true', help='run QC-14 dyadic certificates (slow)')
    ap.add_argument('--scale-test', action='store_true')
    ap.add_argument('--out-dir', help='write outputs here instead of --root')
    args = ap.parse_args()
    root = Path(args.root)
    snap = json.loads((root / 'compiled' / 'calibration_snapshot.json').read_text())
    tag = 'SYNTHETIC-SURROGATE-NOT-HARDWARE' if args.source == 'surrogate' else 'IBM-HARDWARE'
    params, diag, fit_ids = fit_calibration(root, snap, args.source)
    eval_rows = evaluate_set(root, snap, 'eval_echo', args.source, params, certify=args.certify)
    nc_rows = evaluate_set(root, snap, 'nc_eval', args.source, params, certify=False)
    allrows = eval_rows + nc_rows
    reports = {}
    for fam, rows in (('echo', eval_rows), ('kicked_ising', nc_rows)):
        reports[fam] = {k: threshold_report(rows, k) for k in PREDICTORS}
    result = {'data_source': args.source, 'label': tag, 'threshold': THRESHOLD,
              'backend': snap['backend'], 'calibration_timestamp': snap['timestamp'],
              'model_b': params.as_dict(), 'model_b_fit': diag, 'fit_circuits': fit_ids,
              'held_out': reports, 'rows': allrows}
    if args.scale_test:
        import qiskit_ibm_runtime.fake_provider as fp
        result['scale_test'] = scale_test(snap, getattr(fp, 'FakeFez')())
    dest = Path(args.out_dir) if args.out_dir else root
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f'analysis_{args.source}.json').write_text(json.dumps(result, indent=1, default=float) + '\n')
    write_csv(allrows, dest / f'predictions_{args.source}.csv')
    (dest / 'plots').mkdir(exist_ok=True)
    make_plots(allrows, dest / 'plots', args.source)
    print(json.dumps({'label': tag, 'model_b': params.as_dict(),
                      'held_out': {f: {k: (None if v is None else (v['fraction'], v['wilson95'], round(v['mean_abs_error'], 4)))
                                       for k, v in d.items()} for f, d in reports.items()}}, indent=1))


if __name__ == '__main__':
    main()
