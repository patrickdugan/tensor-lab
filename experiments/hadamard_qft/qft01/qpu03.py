"""HAD-QFT-QPU03: workspace-assisted compaction on heavy-hex hardware.

Commands: freeze (offline design selection -> frozen design JSON), logical (CPU audit),
build (read-only at IBM), submit (needs owner authorization of the exact plan hash),
collect (retrieves an existing job only), analyze.
Never creates authorization; never retries a submission.
"""
from __future__ import annotations
import argparse, datetime as dt, io, itertools, json, math, random, sys
from pathlib import Path
import numpy as np
import qpu01 as old
import qpu02
import workspace as W
import topo as T
from recursive import simulate
from radial_qft import radial_matrix

HERE = Path(__file__).resolve().parent
SHOTS, CAP, REP, ALPHA = 1024, 30, .00025, 0.05
ROSTER = {'edge': ['plus', 'minus', 'imag', 'basis0', 'basis1'], 'middle': ['plus', 'minus'], 'tail': ['plus', 'minus']}
PHASE = {'plus': 0., 'minus': math.pi, 'imag': math.pi/2}
ARM_LABEL = {'A': 'existing approx. compact', 'B': 'workspace exact', 'C': 'workspace bounded-error', 'D': 'ancilla-free bounded-error'}


def sha_file(p): return old.sha(Path(p).read_bytes())


# ----------------------------------------------------------------------------- logical plan
def probe_defs(b):
    d = 1 << b
    return [('edge', 0), ('middle', d//2), ('tail', d-2)]


def prep_gates(b, a, state):
    other = a+1 if state in PHASE else None
    g = old.prepare(b, a + (state == 'basis1'), other, PHASE.get(state, 0.))
    assert all(x[0] in ('x', 'h', 'p') for x in g), 'preparation must be single-qubit'
    return g


def readout_gates(b, state):
    return ([('p', b-1, -math.pi/2)] if state == 'imag' else []) + [('h', b-1)]


def ws_rows(b, arm, mode, m, order):
    """Logical rows of a workspace-compiler arm (identity layout, data wires 0..b-1, MSB first)."""
    d = 1 << b; n = d-1; Q = radial_matrix(n); I = np.eye(d, dtype=complex)
    core, nw = W.core(b, mode, m, order=order)
    assert nw == b+m
    ver = W.verify(core, b, m, bound=W.approx_bound(b) if mode == 'approx' else 0.)
    old.must(ver['ok'], f'verification failed for {arm}')
    rows = []

    def run(x):
        xin = np.zeros(1 << nw, dtype=complex); xin[::1 << m] = x
        y = simulate(core, nw, xin)
        mask = np.ones(1 << nw, dtype=bool); mask[::1 << m] = False
        return y[::1 << m], float(np.linalg.norm(y[mask]))

    def add(id, probe, state, gates, ideal, comp, target, kind):
        p, pc = abs(ideal)**2, abs(comp)**2
        old.must(abs(p.sum()-1) < 1e-10 and abs(pc.sum()-1) < 1e-9, 'normalization')
        rows.append(dict(id=id, arm=arm, probe=probe, state=state, kind=kind, basis='Z', target=int(target), ideal=p.tolist(), compiled_ideal=pc.tolist(),
                         gates=[list(x) for x in gates], n_wires=nw, n_data=b))

    for probe, a in probe_defs(b):
        for state in ROSTER[probe]:
            g = prep_gates(b, a, state); x = simulate(g, b, I[:, 0])
            y, leak = run(x); old.must(leak < 1e-9, 'workspace leakage')
            add(f'{arm}_{probe}_{state}', probe, state, g, Q @ x, y, n-a, 'core')
    return rows, core, nw, ver


def control_rows(b, arm, nw):
    """Preparation-only and SPAM rows for one arm: no core, same node set, measure all wires."""
    d = 1 << b; n = d-1; I = np.eye(d, dtype=complex); rows = []
    for state in ('plus', 'minus'):
        g = prep_gates(b, 0, state); x = simulate(g, b, I[:, 0]); r = readout_gates(b, state)
        y = simulate(r, b, x); exp = 0 + (state == 'minus')
        old.must(abs(y[exp])**2 > 1-1e-10, 'prep control')
        rows.append(dict(id=f'{arm}_prep_edge_{state}', arm=arm, probe='edge', state=state, kind='prep', basis='Z', target=exp, ideal=(abs(y)**2).tolist(),
                         compiled_ideal=(abs(y)**2).tolist(), gates=[list(x_) for x_ in g+[('barrier',)]+r], n_wires=nw, n_data=b))
    for name, k in (('zero', 0), ('top', n)):
        g = old.prepare(b, k); y = I[:, k]
        rows.append(dict(id=f'{arm}_spam_{name}', arm=arm, probe='basis', state=name, kind='spam', basis='Z', target=k, ideal=(abs(y)**2).tolist(),
                         compiled_ideal=(abs(y)**2).tolist(), gates=[list(x_) for x_ in g], n_wires=nw, n_data=b))
    return rows


def old_rows(b, replicate=0):
    """Arm A: the QPU02 approximate compact rows and controls (pre-routed on a line, pos mapping)."""
    p = qpu02.plan(b, replicate); rows = []
    by = {r['id']: r for r in p['rows']}
    for probe, a in probe_defs(b):
        for state in ROSTER[probe]:
            r = by[f'compact_approx_{probe}_{state}']; rows.append({**r, 'id': f'A_{probe}_{state}', 'arm': 'A', 'kind': 'core', 'n_wires': b, 'n_data': b})
    n = (1 << b)-1
    for state in ('plus', 'minus'):
        r = by[f'prep_edge_{state}']; rows.append({**r, 'id': f'A_prep_edge_{state}', 'arm': 'A', 'kind': 'prep', 'n_wires': b, 'n_data': b})
    for name, k in (('zero', 0), ('top', n)):
        # spam rows of QPU02 are indexed by shell label k; target in physical order
        r = by[f'spam_{k}']; rows.append({**r, 'id': f'A_spam_{name}', 'arm': 'A', 'kind': 'spam', 'state': name, 'n_wires': b, 'n_data': b})
    return rows, p


# ----------------------------------------------------------------------------- design
def load_design(path):
    return old.read(Path(path))


def design_digest(d): return old.digest(d)


def compile_ws_core(cfg, b):
    gates, nw = W.core(b, cfg['mode'], cfg['m'], order=cfg['order'])
    tc = T.transpile_core(gates, nw, T.load_topology(T_PATH)[0], cfg['shape'], seed=cfg['seed'], level=cfg['level'])
    return gates, nw, tc


T_PATH = HERE / 'hardware_results' / 'qpu03_design' / 'design_calibration.json'


def assemble_ws_rows(rows, b, nw, tc, cfg):
    """Core compiled once (tc); preparation applied on initial positions, readout from final positions."""
    k = tc.num_qubits
    il = list(tc.layout.initial_index_layout()); fl = list(tc.layout.final_index_layout())
    old.must(len(il) == k and sorted(il) == list(range(k)), 'layout')
    out = []
    for r in rows:
        qc = QuantumCircuit_(k, nw)
        gates = [tuple(x) for x in r['gates']]
        if r['kind'] == 'core':
            pre = [(x[0], il[x[1]], *x[2:]) for x in gates]
            qc.compose(T_basis(T.to_qc(pre, k)), inplace=True); qc.barrier(); qc.compose(tc, inplace=True); qc.barrier()
            for v in range(nw): qc.measure(fl[v], v)
        else:
            gg = []
            for x in gates:
                if x[0] == 'barrier': gg.append(('barrier',))
                else: gg.append((x[0], il[x[1]], *x[2:]))
            body = [x for x in gg if x[0] != 'barrier']
            qc.compose(T_basis(T.to_qc(body, k)), inplace=True); qc.barrier()
            for v in range(nw): qc.measure(il[v], v)
        out.append(qc)
    return out, il, fl


def QuantumCircuit_(k, nc):
    from qiskit import QuantumCircuit
    return QuantumCircuit(k, nc)


def T_basis(qc):
    from qiskit import transpile
    return transpile(qc, basis_gates=T.BASIS, optimization_level=1, seed_transpiler=0)


def dist_from_local(qc, n_data, nw):
    """Exact output distribution of a local (k-qubit) ISA row over its classical bits.
    Returns (data distribution with workspace bits 0, probability of any workspace bit = 1)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    k = qc.num_qubits; mm = {}
    bare = QuantumCircuit(k); bare.global_phase = qc.global_phase
    for inst in qc.data:
        nm = inst.operation.name
        if nm == 'measure': mm[qc.find_bit(inst.clbits[0]).index] = qc.find_bit(inst.qubits[0]).index
        elif nm == 'barrier': continue
        else: bare.append(inst.operation, [qc.find_bit(q).index for q in inst.qubits])
    old.must(sorted(mm) == list(range(nw)), 'measurement register')
    pr = Statevector(bare).probabilities()
    pd = np.zeros(1 << n_data); leak = 0.
    for i, p in enumerate(pr):
        if p < 1e-16: continue
        val = [(i >> mm[c]) & 1 for c in range(nw)]
        if any(val[n_data:]): leak += p
        else: pd[sum(val[v] << (n_data-1-v) for v in range(n_data))] += p
    return pd, float(leak)


def to_physical(qc, phys, ncl):
    from qiskit import QuantumCircuit
    big = QuantumCircuit(156, ncl); big.compose(qc, qubits=list(phys), clbits=list(range(ncl)), inplace=True)
    return big


# ----------------------------------------------------------------------------- build
def estimate(rows, reset):
    return 2 + sum(.1 + SHOTS*(REP+max(.0001, r['duration_s']+reset)) for r in rows)


def build_rows(design):
    """Deterministic logical + compiled-local rows for every arm in the design."""
    b = design['b']; G = T.load_topology(T_PATH)[0]; out = []; info = {}
    for arm in design['arms']:
        name = arm['arm']
        if arm['kind'] == 'old':
            rows, p = old_rows(b, design.get('replicate', 0))
            path = arm['nodes']; qcs = []
            for r in rows:
                from qiskit import transpile
                qc = old.circuit({'id': r['id'], 'gates': r['gates']}, b)
                tc = transpile(qc, coupling_map=T.subgraph_cm(G, path), initial_layout=list(range(b)), basis_gates=T.BASIS,
                               optimization_level=arm['level'], seed_transpiler=arm['seed'])
                qcs.append(tc)
            info[name] = dict(phys=list(path), nw=b, ver=None, logical_cx=None)
            out += [(r, qc, path) for r, qc in zip(rows, qcs)]
        else:
            rows, core, nw, ver = ws_rows(b, name, arm['mode'], arm['m'], arm['order'])
            rows += control_rows(b, name, nw)
            gates, nw2, tc = compile_ws_core(arm, b)
            old.must(gates == core, 'core mismatch')
            old.must(T.verify_native(tc, gates, nw)[0], f'native verification failed for {name}')
            qcs, il, fl = assemble_ws_rows(rows, b, nw, tc, arm)
            info[name] = dict(phys=arm['phys'], nw=nw, ver=ver, logical_cx=W.count_cx(gates), il=il, fl=fl, core_metrics=T.metrics(tc))
            out += [(r, qc, arm['phys']) for r, qc in zip(rows, qcs)]
    return out, info


def native_check(r, qc):
    nd, nw = r['n_data'], r['n_wires']
    pd, leak = dist_from_local(qc, nd, nw)
    old.must(float(np.max(abs(pd - np.array(r['compiled_ideal'])))) < 1e-8 and leak < 1e-8, f"native circuit mismatch {r['id']}")


def build(out, design_path, backend_name, replicate):
    h = old.hardware_helpers(); design = load_design(design_path)
    design = {**design, 'replicate': replicate}
    items, info = build_rows(design)
    G, ERR, RO, OQ = T.load_topology(T_PATH)
    service, backend = old.connection(h, backend_name); u = old.usage(service); old.enough(u, CAP)
    snap = h['calibration_snapshot'](backend, source='had_qft_qpu03'); old.must(bool(snap['timestamp']), 'missing calibration timestamp')
    reset_all = {q: float(backend.target['reset'][(q,)].duration) for q in range(backend.num_qubits)} if 'reset' in backend.target.operation_names else {}
    out.mkdir(parents=True, exist_ok=False); (out / 'circuits').mkdir(); old.write(out / 'calibration.json', snap); old.write(out / 'usage_build.json', u)
    rows = []; isas = []
    for r, qc, phys in items:
        native_check(r, qc)
        isa = to_physical(qc, phys, r['n_wires'])
        used = {isa.find_bit(q).index for inst in isa.data for q in inst.qubits}
        old.must(used <= set(phys), f"gate outside node set {r['id']}")
        for inst in isa.data:
            nm = inst.operation.name; qs = tuple(isa.find_bit(q).index for q in inst.qubits)
            old.must(nm in ('cz', 'sx', 'rz', 'x', 'measure', 'barrier'), f'non-native op {nm}')
            if nm != 'barrier': old.must(backend.target.instruction_supported(nm, qs), f'unsupported {nm} {qs}')
        data = h['qasm3_text'](isa).encode(); (out / 'circuits' / f"{r['id']}.qasm").write_bytes(data)
        reset = max(reset_all.get(q, 0.) for q in phys)
        rows.append({**{k: v for k, v in r.items() if k != 'gates'}, 'logical_gates_sha256': old.digest(r['gates']), 'phys': list(phys),
                     'metrics': h['isa_metrics'](isa), 'duration_s': h['duration'](isa, backend.target), 'qasm_sha256': old.sha(data), 'reset': reset}); isas.append(isa)
    order = list(range(len(rows))); random.Random(271128 + 100*design['b'] + replicate).shuffle(order)
    rows = [rows[i] for i in order]; isas = [isas[i] for i in order]
    qpy = h['qpy_bytes'](isas); (out / 'compiled.qpy').write_bytes(qpy)
    reset = max(r['reset'] for r in rows)
    m = dict(backend=backend_name, b=design['b'], replicate=replicate, design_sha256=design_digest(design), design=design, shots=SHOTS, cap_seconds=CAP,
             compiled_at=old.now(), sources=sources(), calibration_sha256=old.sha((out / 'calibration.json').read_bytes()), qpy_sha256=old.sha(qpy),
             estimated_qpu_seconds=estimate(rows, reset), reset_duration=reset,
             arms={k: {kk: vv for kk, vv in v.items() if kk != 'ver'} | {'verification': v['ver']} for k, v in info.items()}, rows=rows)
    m['plan_sha256'] = old.digest(m); old.write(out / 'compiled.json', m)
    old.must(m['estimated_qpu_seconds'] <= CAP, 'estimate exceeds ceiling')
    print(json.dumps({k: m[k] for k in ('backend', 'b', 'replicate', 'plan_sha256', 'estimated_qpu_seconds', 'cap_seconds')}, indent=2))
    for name in sorted(info):
        rr = [r for r in rows if r['arm'] == name and r['kind'] == 'core']
        print(name, 'nodes', info[name]['phys'], 'native 2q', sorted({r['metrics']['num_two_qubit'] for r in rr}), 'depth', sorted({r['metrics']['depth'] for r in rr})[:2], 'dur_us', round(max(r['duration_s'] for r in rr)*1e6, 1))
    print('Build only; no job submitted.')


def sources():
    return {x: old.sha((HERE / x).read_bytes()) for x in ('qpu03.py', 'workspace.py', 'topo.py', 'compact.py', 'qpu02.py', 'qpu01.py', 'recursive.py', 'radial_qft.py')}


def validate(out, h):
    from qiskit import qpy
    m = old.read(out / 'compiled.json'); k = m.pop('plan_sha256'); old.must(old.digest(m) == k, 'manifest mismatch'); m['plan_sha256'] = k
    old.must(m['design_sha256'] == design_digest(m['design']) and m['sources'] == sources(), 'design/source changed')
    old.must(m['shots'] == SHOTS and m['cap_seconds'] == CAP and m['estimated_qpu_seconds'] <= CAP, 'budget mismatch')
    old.must(m['calibration_sha256'] == old.sha((out / 'calibration.json').read_bytes()), 'calibration hash mismatch')
    q = (out / 'compiled.qpy').read_bytes(); old.must(m['qpy_sha256'] == old.sha(q), 'QPY mismatch'); isas = qpy.load(io.BytesIO(q))
    old.must(len(isas) == len(m['rows']), 'roster mismatch')
    for r, isa in zip(m['rows'], isas):
        old.must(r['metrics'] == h['isa_metrics'](isa), 'metrics changed')
        old.must(r['qasm_sha256'] == old.sha(h['qasm3_text'](isa).encode()) == old.sha((out / 'circuits' / f"{r['id']}.qasm").read_bytes()), 'QASM mismatch')
    old.must(abs(m['estimated_qpu_seconds'] - estimate(m['rows'], m['reset_duration'])) < 1e-9, 'estimate changed')
    return m, isas


def submit(out, authorization):
    h = old.hardware_helpers(); m, isas = validate(out, h)
    ok, why = h['check_authorization'](authorization, backend=m['backend'], sha=m['plan_sha256'], estimated_seconds=m['estimated_qpu_seconds']); old.must(ok, why)
    a = old.read(Path(authorization))
    old.must(a['max_qpu_seconds'] <= CAP and a.get('checked_current_allocation') is True and a.get('acknowledge_estimate_not_hard_limit') is True, 'owner acknowledgments required')
    age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(m['compiled_at'])).total_seconds(); old.must(0 <= age <= 900, 'build older than 15 minutes')
    service, backend = old.connection(h, m['backend']); snap = h['calibration_snapshot'](backend, source='had_qft_qpu03'); old.must(snap == old.read(out / 'calibration.json'), 'calibration changed')
    for r, isa in zip(m['rows'], isas): old.must(abs(h['duration'](isa, backend.target) - r['duration_s']) < 1e-12, 'duration changed')
    before = old.usage(service); old.enough(before, a['max_qpu_seconds']); old.write(out / 'usage_before.json', before); old.write(out / 'calibration_submission.json', snap)
    with (out / 'submission_started.json').open('x') as f: json.dump({'plan_sha256': m['plan_sha256'], 'at': old.now()}, f)
    from qiskit_ibm_runtime import SamplerV2
    job = SamplerV2(mode=backend, options={'execution': {'rep_delay': REP}}).run(isas, shots=SHOTS)
    old.write(out / 'job.json', {'job_id': job.job_id(), 'plan_sha256': m['plan_sha256'], 'submitted_at': old.now()}); print('Submitted job:', job.job_id(), flush=True)


def collect(out):
    h = old.hardware_helpers(); m, _ = validate(out, h); j = old.read(out / 'job.json'); old.must(j['plan_sha256'] == m['plan_sha256'], 'job manifest mismatch')
    service, _ = old.connection(h, m['backend']); job = service.job(j['job_id']); res = job.result(); old.must(len(res) == len(m['rows']), 'PUB roster mismatch')
    counts = {r['id']: {k: int(v) for k, v in e.data.c.get_counts().items()} for r, e in zip(m['rows'], res)}
    if (out / 'counts.json').exists(): old.must(counts == old.read(out / 'counts.json'), 'recollection differs')
    old.write(out / 'counts.json', counts); after = old.usage(service); old.write(out / 'usage_after.json', after)
    try: ju = job.usage()
    except Exception: ju = None
    old.write(out / 'execution_receipt.json', dict(job_id=j['job_id'], plan_sha256=m['plan_sha256'], usage_before=old.read(out / 'usage_before.json'), usage_after=after, job_usage=ju, collected_at=old.now()))
    analyze(out)


# ----------------------------------------------------------------------------- frozen analysis
def radius_c(n, alpha): return math.sqrt(math.log(2/alpha)/n)
def radius_d(n, alpha): return math.sqrt(2*math.log(2/alpha)/n)


def decode(counts, nd, nw, shots):
    pd = np.zeros(1 << nd); pdc = np.zeros(1 << nd); leak = 0.; tot = 0
    for k, v in counts.items():
        kr = k[::-1]; old.must(len(kr) == nw and set(kr) <= set('01'), 'bad bitstring'); tot += v
        idx = int(kr[:nd], 2); pd[idx] += v
        if '1' in kr[nd:]: leak += v
        else: pdc[idx] += v
    old.must(tot == shots, 'shot count')
    return pd/shots, pdc, leak/shots


def analyze(out):
    m = old.read(out / 'compiled.json'); counts = old.read(out / 'counts.json'); n = m['shots']; d = m['design']
    old.must(set(counts) == {r['id'] for r in m['rows']}, 'count roster mismatch')
    rows = {}
    for r in m['rows']:
        pd, pdc, leak = decode(counts[r['id']], r['n_data'], r['n_wires'], n)
        q = np.array(r['ideal']); qc = np.array(r['compiled_ideal'])
        ps = pdc/pdc.sum() if pdc.sum() > 0 else pdc
        rows[r['id']] = dict(arm=r['arm'], kind=r['kind'], probe=r['probe'], state=r['state'], target_prob=float(pd[r['target']]), ideal_target=float(q[r['target']]),
                             tv=float(abs(pd-q).sum()/2), tv_compiled=float(abs(pd-qc).sum()/2), hf=float(np.sqrt(pd*q).sum()**2), leakage=leak,
                             target_prob_postselected=float(ps[r['target']]) if pdc.sum() > 0 else None, postselected_shots=int(pdc.sum()),
                             native_2q=r['metrics']['num_two_qubit'], depth=r['metrics']['depth'], duration_us=r['duration_s']*1e6)
    arms = [a['arm'] for a in d['arms']]
    contrasts = []
    for arm in arms:
        for probe in ('edge', 'middle', 'tail'):
            g = lambda s: rows[f'{arm}_{probe}_{s}']
            c = g('minus')['target_prob'] - g('plus')['target_prob']; ideal = g('minus')['ideal_target'] - g('plus')['ideal_target']
            cps = None
            if g('minus')['target_prob_postselected'] is not None: cps = g('minus')['target_prob_postselected'] - g('plus')['target_prob_postselected']
            contrasts.append(dict(arm=arm, probe=probe, contrast=c, ideal=ideal, normalized=c/ideal, contrast_postselected=cps,
                                  leakage_mean=float(np.mean([g('minus')['leakage'], g('plus')['leakage']]))))
    cm = {(c['arm'], c['probe']): c for c in contrasts}
    fam = d['families']
    ms, md = len(fam['signal']), len(fam['difference'])
    signal = []
    for arm in fam['signal']:
        for probe in ('edge', 'middle', 'tail'):
            c = cm[(arm, probe)]['contrast']; rad = radius_c(n, ALPHA/ms)
            signal.append(dict(arm=arm, probe=probe, contrast=c, lower_simultaneous=c-rad, radius=rad, informative=bool(c-rad > 0), primary=probe == 'edge'))
    diffs = []
    for x, y in fam['difference']:
        for probe in ('edge', 'middle', 'tail'):
            dd = cm[(x, probe)]['contrast'] - cm[(y, probe)]['contrast']; rad = radius_d(n, ALPHA/md)
            diffs.append(dict(first=x, second=y, probe=probe, difference=dd, radius_simultaneous=rad, interval_simultaneous=[dd-rad, dd+rad],
                              radius_unadjusted=radius_d(n, ALPHA), interval_unadjusted=[dd-radius_d(n, ALPHA), dd+radius_d(n, ALPHA)],
                              survives_simultaneous=bool(dd-rad > 0), primary=probe == 'edge'))
    ctl = {k: v['target_prob'] for k, v in rows.items() if v['kind'] in ('prep', 'spam')}
    leak_ctl = {k: v['leakage'] for k, v in rows.items() if v['kind'] in ('prep', 'spam') and v['leakage'] > 0 or v['kind'] in ('prep', 'spam') and v['arm'] in ('B', 'C')}
    report = dict(scope='FROZEN_PRIMARY_FAMILIES_SHOT_ONLY', shots=n, alpha=ALPHA, design_sha256=m['design_sha256'], contrasts=contrasts, signal=signal, differences=diffs,
                  rows=rows, control_success=ctl, control_leakage=leak_ctl,
                  note='Raw counts. Hoeffding intervals are finite-shot only; Bonferroni over the registered family sizes. Calibration variability is not included.')
    old.write(out / 'analysis.json', report)
    for s in signal:
        if s['probe'] == 'edge': print(s['arm'], 'edge contrast', round(s['contrast'], 3), 'lower', round(s['lower_simultaneous'], 3))
    for s in diffs:
        if s['probe'] == 'edge': print(s['first'], '-', s['second'], round(s['difference'], 3), 'simultaneous lower', round(s['interval_simultaneous'][0], 3))


# ----------------------------------------------------------------------------- freeze
def enum_paths(G, b):
    import networkx as nx
    P = nx.path_graph(b); gm = nx.isomorphism.GraphMatcher(G, P); seen = {}
    for iso in gm.subgraph_isomorphisms_iter():
        inv = {v: k for k, v in iso.items()}; p = tuple(inv[i] for i in range(b)); seen[p] = True
    return list(seen)


def freeze(select_path, out_path, b, shots=SHOTS):
    """Apply the registered selection rule to the offline selection file and write the frozen design."""
    import qpu03_select as S
    sel = old.read(Path(select_path)); G, ERR, RO, OQ = T.load_topology(T_PATH)
    rows = sel['candidates']
    rejected = []
    def best(mode, ms):
        # Registered rule: highest ESP among candidates whose NATIVE circuit reproduces the logical circuit to 1e-8.
        c = sorted([r for r in rows if r['mode'] == mode and r['m'] in ms and r['esp'] > 0], key=lambda r: (-r['esp'], r['metrics']['cz']))
        for r in c:
            gates, nw = W.core(b, r['mode'], r['m'], order=r['order'])
            tc = T.transpile_core(gates, nw, G, r['shape'], seed=r['seed'], level=r['level'])
            ok, err = T.verify_native(tc, gates, nw)
            if ok: return r
            rejected.append(dict(mode=r['mode'], m=r['m'], order=r['order'], shape=r['shape'], level=r['level'], seed=r['seed'], esp=r['esp'], cz=r['metrics']['cz'], native_vs_logical_max_entry_error=err))
        raise SystemExit('no verified candidate')
    arms = []
    D = best('approx', (0,)); C = best('approx', (1, 2))
    Bx = best('exact', (1, 2)) if b <= 5 else None
    D_nodes = set(D['phys'])
    # arm A: QPU02 circuit on the best-ESP induced path, maximizing overlap with D's nodes first
    if b in qpu02.ARMS:      # the QPU02 line-routed circuit exists only for b = 4, 5, 6
        paths = enum_paths(G, b); pl = qpu02.plan(b, 0); rr = [r for r in pl['rows'] if r['arm'] == 'compact_approx']
        cand = []
        probe_row = next(r for r in rr if r['probe'] == 'edge' and r['state'] == 'plus')
        gl = probe_row['gates']; ib = [i for i, x in enumerate(gl) if x[0] == 'barrier'][0]; core_gates = [tuple(x) for x in gl[ib+1:]]
        for p in paths:
            ov = len(D_nodes & set(p))
            cand.append((ov, p))
        maxov = max(c[0] for c in cand)
        best_path = None
        for ov, p in cand:
            if ov < maxov: continue
            tc = T.transpile_core(core_gates, b, G, list(p), seed=0, level=1, initial_layout=list(range(b)))
            e = S.esp_circuit(tc, list(p), b, b)
            if best_path is None or e > best_path[0]: best_path = (e, list(p), T.metrics(tc))
        arms.append(dict(arm='A', kind='old', nodes=best_path[1], level=1, seed=0, esp=best_path[0], core_metrics=best_path[2], overlap_with_D=maxov))
    def ws(name, r):
        return dict(arm=name, kind='ws', mode=r['mode'], m=r['m'], order=r['order'], shape=r['shape'], phys=r['phys'], level=r['level'], seed=r['seed'],
                    esp=r['esp'], core_metrics=r['metrics'], logical_cx=r['logical_cx'])
    if Bx: arms.append(ws('B', Bx))
    arms.append(ws('C', C)); arms.append(ws('D', D))
    names = [a['arm'] for a in arms]
    cand_pairs = ([('B', 'A')] if 'B' in names and 'A' in names else []) + ([('C', 'A'), ('D', 'A')] if 'A' in names else []) + [('C', 'D')]
    fam = dict(signal=names, difference=[[x, y] for x, y in cand_pairs])
    design = dict(stage=f'b{b}', b=b, backend='ibm_kingston', shots=shots, calibration_timestamp=sel['calibration_timestamp'],
                  selection_file=Path(select_path).name, selection_sha256=sha_file(select_path), arms=arms, families=fam,
                  primary='edge contrast; simultaneous 95% Hoeffding (Bonferroni) over the registered signal and difference families',
                  roster=ROSTER, native_verification_tolerance=1e-8, rejected_candidates=rejected)
    old.write(Path(out_path), design)
    print(json.dumps({a['arm']: dict(kind=a['kind'], m=a.get('m'), esp=round(a['esp'], 4), cz=a['core_metrics'].get('cz', a['core_metrics'].get('cz')), nodes=a.get('phys', a.get('nodes'))) for a in arms}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['freeze', 'logical', 'build', 'submit', 'collect', 'analyze'])
    p.add_argument('--b', type=int, default=5); p.add_argument('--design'); p.add_argument('--select'); p.add_argument('--backend', default='ibm_kingston')
    p.add_argument('--replicate', type=int, default=0); p.add_argument('--out', type=Path, required=True); p.add_argument('--authorization')
    a = p.parse_args()
    if a.command == 'freeze': freeze(a.select, a.out, a.b)
    elif a.command == 'logical':
        design = load_design(a.design); items, info = build_rows(design)
        a.out.mkdir(parents=True, exist_ok=True)
        old.write(a.out / 'logical_summary.json', {k: {kk: vv for kk, vv in v.items() if kk not in ('il', 'fl')} for k, v in info.items()})
        for r, qc, phys in items: native_check(r, qc)
        print('rows', len(items), 'all native checks passed'); print(json.dumps({k: v.get('core_metrics') for k, v in info.items()}, indent=1))
    elif a.command == 'build': build(a.out, a.design, a.backend, a.replicate)
    elif a.command == 'submit': submit(a.out, a.authorization)
    elif a.command == 'collect': collect(a.out)
    else: analyze(a.out)


if __name__ == '__main__':
    main()
