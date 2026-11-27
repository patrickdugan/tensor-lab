"""HAD-QFT-QPU03 offline design selection (CPU only, no hardware outcomes).

For one encoded size b, compile every workspace variant on heavy-hex subgraphs (SABRE routing,
many seeds), verify the winners exactly, embed them on the chip using the frozen calibration
snapshot (estimated success probability, ESP) and write the frozen design JSON.

Selection rule, fixed before any QPU03 job existed:
  1. per (variant, shape): minimize (native CZ, CZ depth) over levels and seeds; keep the best 5.
  2. embed each kept circuit on every isomorphic copy of its shape on the chip and maximize ESP
     computed from the calibration snapshot (per-edge CZ error, 1q error, readout error);
     edges with error >= 0.5 are unusable.
  3. arm D (strongest ancilla-free) = best-ESP over approx, m=0. Arm C (workspace-assisted,
     bounded error) = best-ESP over approx, m in {1,2}. Arm B (workspace-assisted exact) =
     best-ESP over exact, m in {1,2}. Arm A = QPU02 approximate compact circuit (old routing)
     on the best-ESP 5/6-node induced path.
"""
from __future__ import annotations
import argparse, json, time, itertools, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import networkx as nx
import numpy as np
import workspace as W, topo as T

HERE = Path(__file__).resolve().parent
CAL = HERE / 'hardware_results' / 'qpu03_design' / 'design_calibration.json'
G, ERR, RO, OQ = T.load_topology(CAL)
for k in list(ERR):
    if ERR[k] >= 0.5: pass
CAL_TS = json.load(open(CAL))['calibration_timestamp']
LEVELS = (2, 3)


def variants(b, with_exact_big=False):
    out = []
    for mode in ('approx', 'exact'):
        for m in (0, 1, 2, 3, 4):
            if mode == 'exact' and b >= 6 and m not in (0, 2) and not with_exact_big: continue
            for order in (None, 'rev', 'near'):
                out.append((mode, m, order))
    return out


def sweep_task(a):
    b, mode, m, order, nodes, level, seed = a
    gates, n = W.core(b, mode, m, order=order)
    tc = T.transpile_core(gates, n, G, nodes, seed=seed, level=level)
    mt = T.metrics(tc)
    return (b, mode, m, order, tuple(nodes), level, seed, mt)


def embeddings(shape_nodes, limit=None):
    """All node tuples (in the order of shape_nodes) of induced copies of the shape on the chip."""
    S = G.subgraph(shape_nodes)
    gm = nx.isomorphism.GraphMatcher(G, S)
    seen = {}
    for iso in gm.subgraph_isomorphisms_iter():     # iso: chip node -> shape node
        inv = {v: k for k, v in iso.items()}
        phys = tuple(inv[v] for v in shape_nodes)
        seen[phys] = True
        if limit and len(seen) >= limit: break
    return list(seen)


def esp_circuit(tc, phys_nodes, n_measured_virtual, n_data):
    """ESP of the transpiled core on the embedding phys_nodes (local index i -> phys_nodes[i]).
    Measured qubits: final positions of all virtual wires in use (data + workspace)."""
    fl = tc.layout.final_index_layout() if tc.layout is not None else list(range(tc.num_qubits))
    measured = [fl[v] for v in range(n_measured_virtual)]
    p = 1.0
    for inst in tc.data:
        nm = inst.operation.name; q = [tc.find_bit(x).index for x in inst.qubits]
        if nm == 'cz':
            a, b2 = phys_nodes[q[0]], phys_nodes[q[1]]
            e = ERR.get((min(a, b2), max(a, b2)))
            if e is None or e >= 0.5: return 0.0
            p *= 1 - e
        elif nm in ('sx', 'x'):
            p *= 1 - OQ[phys_nodes[q[0]]]
    for q in measured: p *= 1 - RO[phys_nodes[q]]
    return float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', type=int, required=True)
    ap.add_argument('--seeds', type=int, default=40)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--center', type=int, default=68)
    ap.add_argument('--reuse', type=Path, help='re-embed the compiled candidates of an earlier selection file')
    ap.add_argument('--exclude', type=int, nargs='*', default=[], help='physical qubits that may not be used (replicate on a disjoint region)')
    ap.add_argument('--only', nargs='*', default=['approx:0', 'approx:1', 'approx:2', 'exact:1', 'exact:2'])
    a = ap.parse_args()
    b = a.b; t0 = time.time()
    if a.reuse:
        prev = json.load(open(a.reuse)); excl = set(a.exclude); keep = {(x.split(':')[0], int(x.split(':')[1])) for x in a.only}
        rows = []
        seen = {}
        for r in prev['candidates']:
            if (r['mode'], r['m']) not in keep: continue
            gates, n = W.core(b, r['mode'], r['m'], order=r['order'])
            tc = T.transpile_core(gates, n, G, r['shape'], seed=r['seed'], level=r['level'])
            if T.metrics(tc) != r['metrics']: raise SystemExit('recompile mismatch: determinism lost for ' + str(r['shape']))
            key = tuple(r['shape'])
            embs = seen.setdefault(key, [e for e in embeddings(r['shape']) if not (set(e) & excl)])
            bestesp = (-1, None)
            for phys in embs:
                e = esp_circuit(tc, phys, n, b)
                if e > bestesp[0]: bestesp = (e, phys)
            rows.append({**r, 'esp': bestesp[0], 'phys': list(bestesp[1]) if bestesp[1] else None, 'n_embeddings': len(embs)})
        rows = [r for r in rows if r['phys']]; rows.sort(key=lambda r: -r['esp'])
        out = dict(b=b, calibration_timestamp=CAL_TS, seeds=prev['seeds'], levels=prev['levels'], n_rows=len(rows), excluded_nodes=sorted(excl), reuse=str(a.reuse.name), candidates=rows)
        json.dump(out, open(a.out, 'w'), indent=1); print('re-embedded', len(rows), 'rows in', round(time.time()-t0), 's')
        for mode, ms in (('approx', (0,)), ('approx', (1, 2)), ('exact', (1, 2))):
            c = [r for r in rows if r['mode'] == mode and r['m'] in ms]
            if c: r = c[0]; print(mode, ms, 'ESP', round(r['esp'], 4), 'cz', r['metrics']['cz'], 'm', r['m'], 'phys', r['phys'])
        return
    vs = variants(b)
    jobs = []; shape_cache = {}
    for mode, m, order in vs:
        k = b + m
        shs = shape_cache.setdefault(k, T.shapes(G, k, a.center))
        for nodes in shs:
            for level in LEVELS:
                for seed in range(a.seeds):
                    jobs.append((b, mode, m, order, nodes, level, seed))
    print('variants', len(vs), 'jobs', len(jobs), flush=True)
    with ProcessPoolExecutor(max_workers=19) as ex:
        res = list(ex.map(sweep_task, jobs, chunksize=8))
    print('sweep seconds', round(time.time() - t0), flush=True)
    best = {}
    for b_, mode, m, order, nodes, level, seed, mt in res:
        best.setdefault((mode, m, order, nodes), []).append(((mt['cz'], mt['cz_depth'], mt['depth']), level, seed, mt))
    rows = []
    for key, lst in sorted(best.items(), key=lambda kv: str(kv[0])):
        mode, m, order, nodes = key
        lst.sort(key=lambda x: x[0])
        keep = []; seen_sig = set()
        for sig, level, seed, mt in lst:
            if sig in seen_sig: continue
            seen_sig.add(sig); keep.append((level, seed, mt))
            if len(keep) == 5: break
        embs = embeddings(list(nodes))
        gates, n = W.core(b, mode, m, order=order)
        for level, seed, mt in keep:
            tc = T.transpile_core(gates, n, G, list(nodes), seed=seed, level=level)
            bestesp = (-1, None)
            for phys in embs:
                e = esp_circuit(tc, phys, n, b)
                if e > bestesp[0]: bestesp = (e, phys)
            rows.append(dict(mode=mode, m=m, order=order, shape=list(nodes), level=level, seed=seed, metrics=mt,
                             esp=bestesp[0], phys=list(bestesp[1]) if bestesp[1] else None, n_embeddings=len(embs), logical_cx=W.count_cx(gates)))
    rows.sort(key=lambda r: -r['esp'])
    out = dict(b=b, calibration_timestamp=CAL_TS, seeds=a.seeds, levels=list(LEVELS), n_rows=len(rows), candidates=rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print('seconds', round(time.time() - t0))
    for mode, ms in (('approx', (0,)), ('approx', (1, 2)), ('exact', (1, 2)), ('exact', (0,))):
        c = [r for r in rows if r['mode'] == mode and r['m'] in ms]
        if c:
            r = c[0]; print(mode, ms, 'best ESP', round(r['esp'], 4), 'cz', r['metrics']['cz'], 'depth', r['metrics']['cz_depth'], 'm', r['m'], r['order'], 'level', r['level'], 'seed', r['seed'], 'phys', r['phys'])


if __name__ == '__main__':
    main()
