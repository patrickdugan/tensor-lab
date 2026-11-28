"""Offline resource scaling of the workspace compiler through b=8 (CPU only).

Per (b, mode, m): logical CNOTs, ideal operator error, and the best heavy-hex routed native
CZ count (SABRE, levels 2-3, several seeds, best induced shapes of b+m nodes). Exact variants
are routed only through b=6. Verification of every circuit at b<=7 is by full simulation.
"""
from __future__ import annotations
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import workspace as W, topo as T

HERE = Path(__file__).resolve().parent
G, ERR, RO, OQ = T.load_topology(HERE / 'hardware_results' / 'qpu03_design' / 'design_calibration.json')
SEEDS, SHAPE_CAP, LEVELS = 12, 6, (2, 3)


def task(a):
    b, mode, m, order, nodes, level, seed = a
    gates, n = W.core(b, mode, m, order=order)
    tc = T.transpile_core(gates, n, G, nodes, seed=seed, level=level)
    return (b, mode, m, order, tuple(nodes), level, seed, T.metrics(tc))


def top_shapes(k):
    shs = T.shapes(G, k, 68)
    def deg3(s): return sum(1 for v in s if G.subgraph(s).degree(v) >= 3)
    shs.sort(key=lambda s: (-deg3(s), s))
    return shs[:SHAPE_CAP]


if __name__ == '__main__':
    t0 = time.time(); out = []; jobs = []
    spec = []
    for b in range(3, 9):
        for mode in ('approx', 'exact'):
            for m in (0, 1, 2, 3, 4):
                if mode == 'exact' and m in (1, 3, 4) : continue
                spec.append((b, mode, m))
    cache = {}
    for b, mode, m in spec:
        if mode == 'exact' and b > 6: continue
        if mode == 'approx' and b < 4 and m > 0: continue
        for order in ('near', 'rev'):
            for nodes in cache.setdefault(b+m, top_shapes(b+m)):
                for level in LEVELS:
                    for seed in range(SEEDS):
                        jobs.append((b, mode, m, order, nodes, level, seed))
    print('jobs', len(jobs), flush=True)
    with ProcessPoolExecutor(max_workers=19) as ex:
        res = list(ex.map(task, jobs, chunksize=4))
    best = {}
    for b, mode, m, order, nodes, level, seed, mt in res:
        key = (b, mode, m)
        if key not in best or (mt['cz'], mt['cz_depth']) < (best[key]['cz'], best[key]['cz_depth']):
            best[key] = dict(cz=mt['cz'], cz_depth=mt['cz_depth'], depth=mt['depth'], order=order, level=level, seed=seed, nodes=list(nodes))
    rows = []
    for b, mode, m in spec:
        g, n = W.core(b, mode, m)
        row = dict(b=b, mode=mode, m=m, n_wires=n, logical_cx=W.count_cx(g), bound=W.approx_bound(b) if mode == 'approx' else 0.)
        if b <= 7 and n <= 11:
            v = W.verify(g, b, m, bound=row['bound']); row.update(operator_error=v['operator_error'], leakage=v['max_workspace_leakage'], verified=v['ok'])
        row['routed'] = best.get((b, mode, m))
        rows.append(row)
        print(row['b'], mode, m, 'logical', row['logical_cx'], 'routed', row['routed']['cz'] if row['routed'] else None, 'err', row.get('operator_error'), flush=True)
    json.dump(dict(seeds=SEEDS, levels=list(LEVELS), shape_cap=SHAPE_CAP, rows=rows), open(HERE / 'hardware_results' / 'qpu03_design' / 'scaling.json', 'w'), indent=1)
    print('seconds', round(time.time()-t0))
