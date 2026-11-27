"""QC-26B Phase 5: frozen-prediction PROSPECTIVE evaluation plan (build / submit).

  python qc26b_prospective_plan.py build  --out DIR [--backend ibm_kingston] [--fixture FakeKingston]
  python qc26b_prospective_plan.py submit --out DIR --frozen frozen.json --params params.json --authorization FILE

`build` connects READ-ONLY (qc23_access.connect) to compile; it never submits.
`submit` is the only path to hardware. It delegates to qc23_access.check_authorization
(owner-written file naming this backend, this exact plan hash and a QPU-seconds ceiling),
requires the build to be at most 15 minutes old, requires a frozen.json whose predictions
verify against this build and the parameter file, and writes an exclusive
submission_started.json marker before the irreversible call. No authorization file is
created or supplied by this module. Credentials are never read or printed here.

Circuits: kicked-Ising (qc23_nonclifford.make_kicked_ising), fresh seeds 51-54 (earlier runs used
41-44), widths 8 and 12, depths 2-4, 2048 shots, same optimization_level=0 route as nc_eval.
Each (width, seed) pair gets one alternative physical path; its three depths share it:
  seed 51  targets_end        5 and 23 plus controls 3 and 21; qubit 5 forced to a path end (edge 5-6 forbidden)
  seed 52  targets_23_end     5 and 23 plus control 3; qubit 23 forced to a path end, 5 in the interior
  seed 53  controls           control 21 alone, qubits 5 and 23 excluded
  seed 54  pure_control       the whole suspect neighbourhood excluded
Earlier nc_eval paths put 5 between (4,6) and 23 between (16,22) at fixed interior positions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from qc23_access import check_authorization, connect, estimate_qpu_seconds, plan_hash  # noqa: E402
from qc23_compile import (audit_mapping, edge_errors, isa_metrics, path_cost, qasm3_text,  # noqa: E402
                          qpy_bytes, readout_errors, single_qubit_errors, transpile_on_path)

BACKEND = 'ibm_kingston'
SET_NAME = 'qc26b_prospective'
WIDTHS = (8, 12)
DEPTHS = (2, 3, 4)
SEEDS = (51, 52, 53, 54)
SHOTS = 2048
TARGETS = (5, 23)
CONTROLS = (3, 21)
FRESHNESS_S = 900
ESTIMATE_CEILING_S = 15.0
PRIOR_COMPILED = HERE.parent / 'results' / 'hw_root' / 'compiled' / 'compiled_nc_eval.json'

CLASSES = {
    51: {'name': 'targets_end', 'include': [[5, 23, 3, 21]], 'exclude': [], 'forbid_edges': [[5, 6]]},
    52: {'name': 'targets_23_end', 'include': [[5, 23, 3]], 'exclude': [], 'forbid_edges': [[22, 23], [23, 24]]},
    # Kingston's 3/4/16 pocket holds at most 5 qubits once 5 and 23 are excluded and 3 and 21 only connect through
    # 23 or 5, so the stand-alone control is 21; control 3 appears in the two target classes.
    53: {'name': 'controls', 'include': [[21]], 'exclude': [5, 23], 'forbid_edges': []},
    54: {'name': 'pure_control', 'include': [[]], 'exclude': [3, 4, 5, 6, 16, 21, 22, 23, 24], 'forbid_edges': []},
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump_json(obj, *, indent=2) -> bytes:
    """LF bytes, sorted keys, trailing newline."""
    return (json.dumps(obj, indent=indent, sort_keys=True) + '\n').encode()


# --------------------------------------------------------------------------- constrained path search

def search_line(nbr, edges, readout, oneq, n, *, include=(), exclude=(), forbid_edges=(), beam=4096):
    """Deterministic constrained modification of qc23_compile.choose_line_calibrated.

    Same independence-model cost (qc23_compile.path_cost) and the same beam search, with
      exclude       physical qubits that may not appear,
      forbid_edges  undirected couplers that may not be used,
      include       qubits that must appear: partial paths are pruned when a missing included
                    qubit is farther (in the allowed graph) than the remaining path slots.
    Returns (path, cost); canonical orientation (lower end index first) as in the original.
    """
    exclude = set(exclude)
    forbid = {(min(a, b), max(a, b)) for a, b in forbid_edges}
    include = list(include)
    if any(q in exclude for q in include):
        raise ValueError('an included qubit is excluded')
    adj = {q: sorted(o for o in nbr[q] if o not in exclude and (min(q, o), max(q, o)) not in forbid)
           for q in sorted(nbr) if q not in exclude}

    def bfs(src):
        dist, frontier = {src: 0}, [src]
        while frontier:
            nxt = []
            for q in frontier:
                for o in adj.get(q, ()):
                    if o not in dist:
                        dist[o] = dist[q] + 1
                        nxt.append(o)
            frontier = nxt
        return dist

    dists = {m: bfs(m) for m in include}

    def feasible(path):
        missing = [m for m in include if m not in path]
        left = n - len(path)
        if len(missing) > left:
            return False
        return all(dists[m].get(path[-1], math.inf) <= left for m in missing)

    frontier = [([q], path_cost([q], edges, readout, oneq)) for q in sorted(adj) if adj[q]]
    frontier = [(p, c) for p, c in frontier if feasible(p)]
    frontier.sort(key=lambda t: (t[1], t[0]))
    frontier = frontier[:beam]
    for _ in range(n - 1):
        nxt = []
        for path, _c in frontier:
            seen = set(path)
            for o in adj[path[-1]]:
                if o in seen:
                    continue
                p2 = path + [o]
                if feasible(p2):
                    nxt.append((p2, path_cost(p2, edges, readout, oneq)))
        if not nxt:
            raise RuntimeError(f'no constrained simple path of {n} qubits (include={include}, exclude={sorted(exclude)})')
        nxt.sort(key=lambda t: (t[1], t[0]))
        frontier = nxt[:beam]
    frontier = [(p, c) for p, c in frontier if len(p) == n and all(m in p for m in include)]
    if not frontier:
        raise RuntimeError(f'no constrained simple path of {n} qubits (include={include}, exclude={sorted(exclude)})')
    path, cost = min(frontier, key=lambda t: (t[1], t[0]))
    if path[0] > path[-1]:
        path = path[::-1]
    return path, cost


def graph_from_backend(backend):
    target = backend.target
    edges = edge_errors(target)
    readout = readout_errors(target, backend.num_qubits)
    oneq = single_qubit_errors(target, backend.num_qubits)
    nbr = {q: set() for q in range(backend.num_qubits)}
    for a, b in edges:
        nbr[a].add(b)
        nbr[b].add(a)
    return nbr, edges, readout, oneq


def graph_from_snapshot(snap):
    """Same graph quantities from a qc23_noise.calibration_snapshot dict (offline tests)."""
    nq = snap['num_qubits']
    edges = {}
    for e in snap['edges']:
        a, b = e['qargs']
        key = (min(a, b), max(a, b))
        edges[key] = min(edges.get(key, 1.0), float(e['error'] if e['error'] is not None else 1e-2))
    nbr = {q: set() for q in range(nq)}
    for a, b in edges:
        nbr[a].add(b)
        nbr[b].add(a)
    readout = [float(snap['qubits'][str(q)]['readout_error'] if snap['qubits'][str(q)]['readout_error'] is not None else 0.05)
               for q in range(nq)]
    oneq = [float(snap['qubits'][str(q)]['sx_error'] if snap['qubits'][str(q)]['sx_error'] is not None else 1e-3)
            for q in range(nq)]
    return nbr, edges, readout, oneq


def prior_paths(compiled=PRIOR_COMPILED):
    meta = json.loads(compiled.read_text())
    paths = {tuple(r['physical_path']) for r in meta['circuits']}
    seeds = {r['seed'] for r in meta['circuits']}
    return sorted(paths), seeds


def describe_targets(path, priors):
    """Position and path-neighbours of each suspect qubit versus every earlier path."""
    out = {}
    for q in TARGETS:
        if q not in path:
            out[str(q)] = None
            continue
        i = path.index(q)
        nb = sorted(path[j] for j in (i - 1, i + 1) if 0 <= j < len(path))
        old = [(p.index(q), sorted(p[j] for j in (p.index(q) - 1, p.index(q) + 1) if 0 <= j < len(p)))
               for p in priors if q in p]
        out[str(q)] = {'position': i, 'neighbours': nb,
                       'prior_positions': sorted({o[0] for o in old}),
                       'prior_neighbour_sets': sorted({tuple(o[1]) for o in old}),
                       'position_differs_from_all_prior': all(i != o[0] for o in old),
                       'neighbours_differ_from_all_prior': all(nb != o[1] for o in old)}
    return out


def select_paths(graph, *, priors, widths=WIDTHS, seeds=SEEDS, beam=4096):
    """{(n, seed): record} for every width and seed, with provenance of the constraints used."""
    nbr, edges, readout, oneq = graph
    out = {}
    for n in widths:
        for seed in seeds:
            cls = CLASSES[seed]
            last = None
            for inc in cls['include']:
                try:
                    path, cost = search_line(nbr, edges, readout, oneq, n, include=inc, exclude=cls['exclude'],
                                             forbid_edges=cls['forbid_edges'], beam=beam)
                except RuntimeError as exc:
                    last = exc
                    continue
                break
            else:
                raise RuntimeError(f'class {cls["name"]} infeasible at n={n}: {last}')
            rec = {'path_class': cls['name'], 'physical_path': path, 'cost': cost, 'include_used': list(inc),
                   'exclude': list(cls['exclude']), 'forbid_edges': [list(e) for e in cls['forbid_edges']],
                   'targets': describe_targets(path, priors),
                   'contains_controls': [q for q in CONTROLS if q in path],
                   'identical_to_prior_path': tuple(path) in set(priors)}
            if cls['name'] == 'pure_control' or cls['name'] == 'controls':
                assert not any(q in path for q in TARGETS), 'control path contains a target'
            else:
                for q in TARGETS:
                    t = rec['targets'][str(q)]
                    assert t is not None, f'class {cls["name"]} path misses target {q}'
                    assert t['position_differs_from_all_prior'] or t['neighbours_differ_from_all_prior'], \
                        f'target {q} at a previously used position with previous neighbours'
            if rec['path_class'] == 'pure_control':
                assert not any(q in path for q in CONTROLS), 'pure control contains a control qubit'
            assert not rec['identical_to_prior_path']
            out[(n, seed)] = rec
    return out


# --------------------------------------------------------------------------- items and compile

def items(selected):
    from qc23_ibm import observable_bits
    out = []
    for n in WIDTHS:
        for seed in SEEDS:
            for d in DEPTHS:
                out.append({'id': f'kicked_ising_n{n}_d{d}_s{seed}', 'set': SET_NAME, 'family': 'kicked_ising',
                            'n': n, 'depth': d, 'seed': seed, 'shots': SHOTS, 'bits': list(observable_bits(n)),
                            'path_class': selected[(n, seed)]['path_class']})
    return out


def asap_duration(isa, target) -> float:
    """Critical-path duration from calibrated gate durations (rz virtual, barriers free)."""
    clock = {}
    for inst in isa.data:
        op = inst.operation.name
        qs = tuple(isa.find_bit(q).index for q in inst.qubits)
        start = max((clock.get(q, 0.0) for q in qs), default=0.0)
        if op in ('barrier', 'rz'):
            dur = 0.0
        else:
            props = target[op].get(qs)
            if props is None or props.duration is None:
                raise ValueError(f'missing duration for {op} {qs}')
            dur = float(props.duration)
        for q in qs:
            clock[q] = start + dur
    return max(clock.values(), default=0.0)


def snapshot_bytes(snap) -> bytes:
    return dump_json(snap, indent=None)


def build(out: Path, backend_name: str, fixture: str | None):
    from qc23_experiment import load_backend, logical_circuit
    from qc23_noise import calibration_snapshot
    backend, mode = load_backend(backend_name, fixture)     # read-only connect via qc23_access.connect
    if fixture is None and backend.name != BACKEND:
        raise RuntimeError(f'expected {BACKEND}, got {backend.name}')
    priors, prior_seeds = prior_paths()
    if set(SEEDS) & prior_seeds or set(SEEDS) & {3, 5, 11, 13}:
        raise RuntimeError('a protocol seed was used by an earlier set')
    snap = calibration_snapshot(backend, source=f'qc26b_prospective_build:{mode}')
    if snap['two_qubit_gate'] != 'cz':
        raise RuntimeError('protocol requires a CZ-native backend')
    selected = select_paths(graph_from_backend(backend), priors=priors)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'circuits').mkdir()
    cal = snapshot_bytes(snap)
    (out / 'calibration_snapshot.json').write_bytes(cal)
    rows, isas, durs = [], [], []
    for item in items(selected):
        n = item['n']
        rec = selected[(n, item['seed'])]
        physical = rec['physical_path']
        measured = logical_circuit(item).copy()
        measured.measure_all()
        isa = transpile_on_path(backend, measured, physical, optimization_level=0)
        audit = audit_mapping(isa, physical, n)
        qasm = qasm3_text(isa).encode()
        (out / 'circuits' / f"{item['id']}.qasm").write_bytes(qasm)
        dur = asap_duration(isa, backend.target)
        durs.append(dur)
        rows.append({**{k: v for k, v in item.items()}, 'physical_path': physical, 'mapping': audit,
                     'metrics': isa_metrics(isa), 'duration_s': dur, 'qasm3_sha256': hash_bytes(qasm)})
        isas.append(isa)
    qpy = qpy_bytes(isas)
    (out / 'compiled.qpy').write_bytes(qpy)
    est_depth = estimate_qpu_seconds([r['metrics']['depth'] * 5e-7 for r in rows], SHOTS)
    est_asap = estimate_qpu_seconds(durs, SHOTS)
    est = max(est_depth, est_asap)
    protocol = {
        'name': 'QC-26B prospective frozen-prediction test', 'version': 1,
        'seeds': list(SEEDS), 'prior_kicked_ising_seeds': sorted(prior_seeds), 'widths': list(WIDTHS),
        'depths': list(DEPTHS), 'targets': list(TARGETS), 'controls': list(CONTROLS),
        'classes': {str(k): v for k, v in CLASSES.items()},
        'paths': [{'n': n, 'seed': s, **rec} for (n, s), rec in sorted(selected.items())],
        'prior_paths': [list(p) for p in priors],
        'estimate_basis': 'qc23_access.estimate_qpu_seconds, shots*(250us + duration); gate = max(depth*5e-7 form, ASAP calibrated duration form)',
        'estimate_depth_form_s': est_depth, 'estimate_asap_form_s': est_asap,
    }
    meta = {'set': SET_NAME, 'backend': backend.name, 'mode': mode, 'compiled_at': now(),
            'calibration_timestamp': snap['timestamp'], 'calibration_sha256': hash_bytes(cal),
            'qpy_sha256': hash_bytes(qpy), 'shots': SHOTS, 'estimated_qpu_seconds': est,
            'protocol': protocol, 'circuits': rows}
    meta['plan_sha256'] = plan_hash(meta)
    (out / 'compiled.json').write_bytes(dump_json(meta))
    print(json.dumps({'plan_sha256': meta['plan_sha256'], 'circuits': len(rows), 'estimated_qpu_seconds': est,
                      'estimate_depth_form_s': est_depth, 'estimate_asap_form_s': est_asap,
                      'calibration_timestamp': snap['timestamp'], 'mode': mode}))
    for (n, s), rec in sorted(selected.items()):
        print(f"n={n} seed={s} {rec['path_class']:<15} contains {[q for q in (5, 23, 3, 21) if q in rec['physical_path']]} "
              f"path={json.dumps(rec['physical_path'])}")
    print('Build only; no QPU job submitted.')
    if est > ESTIMATE_CEILING_S:
        raise RuntimeError(f'estimate {est:.2f}s exceeds {ESTIMATE_CEILING_S}s; do not submit')


# --------------------------------------------------------------------------- validate and submit

def validate_build(out: Path):
    from qiskit import qpy
    meta = json.loads((out / 'compiled.json').read_text())
    sha = meta.pop('plan_sha256')
    if plan_hash(meta) != sha:
        raise ValueError('manifest hash mismatch')
    meta['plan_sha256'] = sha
    if hash_bytes((out / 'calibration_snapshot.json').read_bytes()) != meta['calibration_sha256']:
        raise ValueError('calibration snapshot hash mismatch')
    data = (out / 'compiled.qpy').read_bytes()
    if hash_bytes(data) != meta['qpy_sha256']:
        raise ValueError('qpy hash mismatch')
    isas = qpy.load(io.BytesIO(data))
    if len(isas) != len(meta['circuits']):
        raise ValueError('circuit count mismatch')
    for row, isa in zip(meta['circuits'], isas):
        qasm = (out / 'circuits' / f"{row['id']}.qasm").read_bytes()
        if hash_bytes(qasm) != row['qasm3_sha256'] or hash_bytes(qasm3_text(isa).encode()) != row['qasm3_sha256']:
            raise ValueError(f"qasm mismatch for {row['id']}")
        if audit_mapping(isa, row['physical_path'], row['n']) != row['mapping']:
            raise ValueError(f"mapping audit mismatch for {row['id']}")
    return meta, isas


def find_node():
    import os
    for cand in (os.environ.get('QC26B_NODE'), shutil.which('node'),
                 'C:/Users/patri/AppData/Local/Programs/nodejs/node.exe'):
        if cand and (Path(cand).exists() or shutil.which(cand)):
            return cand
    raise RuntimeError('node not found; set QC26B_NODE')


def verify_frozen(out: Path, frozen: Path, params: Path, plan_sha: str):
    """Run the JS verifier (recomputes every prediction at 1e-12 from the build dir and parameter file)."""
    res = subprocess.run([find_node(), str(HERE / 'qc26b_freeze.mjs'), 'verify', str(out), str(params), str(frozen)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise ValueError(f'frozen.json does not verify against this build: {res.stdout.strip()} {res.stderr.strip()}')
    fz = json.loads(frozen.read_text())
    if fz['build']['plan_sha256'] != plan_sha:
        raise ValueError('frozen.json was made for a different plan hash')
    return fz


def verify_published_freeze(auth, fz):
    """Require a published main commit containing the exact freeze and scoring code."""
    import re
    sha = auth.get("freeze_commit")
    rel = auth.get("freeze_path")
    if fz.get("format") != "qc26b-frozen-v3":
        raise ValueError("live submission requires v3 freeze; v1/v2 are replay-only")
    if auth.get("frozen_sha256") != fz.get("sha256"):
        raise ValueError("authorization frozen_sha256 mismatch")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("authorization needs full freeze_commit SHA")
    if not isinstance(rel, str) or not rel or rel.startswith("/") or ".." in rel.split("/") or "\\" in rel:
        raise ValueError("authorization needs a repository-relative freeze_path")

    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(HERE), *args], check=True,
                                  capture_output=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("published freeze git check failed: " + str(exc)) from exc

    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    remote = git("ls-remote", "--exit-code", "origin", "refs/heads/main").decode().split()
    if len(remote) != 2 or remote[1] != "refs/heads/main":
        raise ValueError("cannot establish published origin/main")
    git("merge-base", "--is-ancestor", sha, remote[0])
    if json.loads(git("show", sha + ":" + rel)) != fz:
        raise ValueError("published frozen file differs from submitted freeze")
    for name, expected in fz["code"]["manifest"].items():
        code_rel = (HERE / name).relative_to(root).as_posix()
        actual = hash_bytes(git("show", sha + ":" + code_rel).replace(b"\r\n", b"\n"))
        if actual != expected:
            raise ValueError("published scoring code differs: " + name)
    return {"freeze_commit": sha, "freeze_path": rel, "published_main_sha": remote[0],
            "frozen_sha256": fz["sha256"]}


def checked_usage(service, ceiling):
    try:
        usage = service.usage()
        remaining = usage["usage_remaining_seconds"]
    except Exception as exc:
        raise ValueError("provider allocation unavailable; refusing submission") from exc
    if (isinstance(remaining, bool) or not isinstance(remaining, (int, float))
            or not math.isfinite(remaining) or remaining < ceiling or usage.get("usage_limit_reached")):
        raise ValueError("remaining provider allocation below authorized ceiling")
    return usage


def submit(out: Path, authorization, frozen: Path, params: Path):
    meta, isas = validate_build(out)
    if meta['mode'] != 'live':
        raise ValueError('only a live-backend build can be submitted')
    if not authorization:
        raise ValueError('owner authorization file is required')
    fz = verify_frozen(out, frozen, params, meta['plan_sha256'])
    ok, why = check_authorization(str(authorization), backend=meta['backend'], sha=meta['plan_sha256'],
                                  estimated_seconds=meta['estimated_qpu_seconds'])
    if not ok:
        raise ValueError(f'hardware execution NOT authorised: {why}')
    auth = json.loads(Path(authorization).read_text(encoding="utf-8-sig"))
    ceiling = auth["max_qpu_seconds"]
    if isinstance(ceiling, bool) or not math.isfinite(ceiling) or ceiling > ESTIMATE_CEILING_S:
        raise ValueError("invalid authorization ceiling")
    if auth.get("acknowledge_estimate_not_hard_limit") is not True or auth.get("checked_current_allocation") is not True:
        raise ValueError("missing owner cost and allocation acknowledgements")
    provenance = verify_published_freeze(auth, fz)
    age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(meta['compiled_at'])).total_seconds()
    if not 0 <= age <= FRESHNESS_S:
        raise ValueError('compilation older than 15 minutes; rebuild, refreeze and reauthorize')
    service, err = connect(None)
    if service is None:
        raise RuntimeError('IBM account unavailable on this machine')
    backend = service.backend(meta['backend'])
    from qc23_noise import calibration_snapshot
    old = json.loads((out / 'calibration_snapshot.json').read_text())
    snap = calibration_snapshot(backend, source='qc26b_prospective_submission')
    if not old.get('timestamp') or snap.get('timestamp') != old['timestamp']:
        raise ValueError('calibration timestamp changed or missing; rebuild, refreeze and reauthorize')
    for k in ('qubits', 'edges', 'basis_gates', 'two_qubit_gate'):
        if json.dumps(old[k], sort_keys=True) != json.dumps(snap[k], sort_keys=True):
            raise ValueError(f'calibration {k} changed; rebuild, refreeze and reauthorize')
    (out / 'calibration_submission.json').write_bytes(snapshot_bytes(snap))
    before = checked_usage(service, ceiling)
    (out / 'usage_before.json').write_bytes(dump_json(before))
    (out / 'freeze_provenance.json').write_bytes(dump_json(provenance))
    # Exclusive marker before the irreversible call; an ambiguous failure must be checked in the account job
    # history by hand and is never retried automatically.
    with (out / 'submission_started.json').open('x') as f:
        json.dump({'started_at': now(), 'plan_sha256': meta['plan_sha256'], 'frozen_sha256': fz['sha256']}, f)
    from qiskit_ibm_runtime import SamplerV2
    job = SamplerV2(mode=backend).run(isas, shots=meta['shots'])     # same pattern as qc23_submit_set.py
    job_id = job.job_id()
    (out / 'job.json').write_bytes(dump_json({'job_id': job_id, 'backend': backend.name,
                                              'plan_sha256': meta['plan_sha256'], 'submitted_at': now(), **provenance}))
    print('Submitted IBM job:', job_id, flush=True)
    result = job.result()
    if len(result) != len(isas):
        raise RuntimeError('PUB count mismatch')
    done = {}
    for row, entry in zip(meta['circuits'], result):
        counts = {k: int(v) for k, v in entry.data.meas.get_counts().items()}
        if sum(counts.values()) != row['shots'] or any(len(k) != row['n'] or set(k) - set('01') for k in counts):
            raise RuntimeError(f"malformed counts for {row['id']}")
        done[row['id']] = {'counts': counts, 'physical_path': row['physical_path'], 'job_id': job_id,
                           'backend': backend.name, 'source': 'ibm'}
    (out / 'counts.json').write_bytes(dump_json(done, indent=None))
    # Preserve counts before attempting optional post-job usage queries.
    try:
        after = service.usage()
    except Exception as exc:
        after = {"error": type(exc).__name__ + ": " + str(exc)}
    try:
        job_usage = job.usage()
    except Exception as exc:
        job_usage = {"error": type(exc).__name__ + ": " + str(exc)}
    (out / 'execution_receipt.json').write_bytes(dump_json({
        "job_id": job_id, "plan_sha256": meta["plan_sha256"], **provenance,
        "usage_before": before, "usage_after": after, "job_usage": job_usage,
        "authorized_ceiling_seconds": ceiling, "circuits": len(done),
        "shots_per_circuit": meta["shots"]
    }))
    print(f'Wrote {out}/counts.json')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--backend', default=BACKEND)
    b.add_argument('--fixture', help='offline qiskit_ibm_runtime fake provider class, e.g. FakeKingston (never submittable)')
    s = sub.add_parser('submit')
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--frozen', type=Path, required=True)
    s.add_argument('--params', type=Path, required=True)
    s.add_argument('--authorization', type=Path)
    a = ap.parse_args(argv)
    if a.cmd == 'build':
        build(a.out, a.backend, a.fixture)
    else:
        try:
            submit(a.out, a.authorization, a.frozen, a.params)
        except ValueError as exc:
            ap.error(str(exc))


if __name__ == '__main__':
    main()
