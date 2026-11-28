"""Topology-aware synthesis helpers for QPU03: IBM heavy-hex subgraph search, transpilation,
exact verification of the transpiled native circuit, and calibration-based resource estimates.

Everything here is CPU-only and uses only a saved coupling map and calibration snapshot;
no hardware outcome enters any selection.
"""
from __future__ import annotations
import json, warnings
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import Operator, Statevector

warnings.filterwarnings('ignore', category=UserWarning, module='networkx')
BASIS = ['cz', 'sx', 'rz', 'x']


def load_topology(path):
    t = json.load(open(path))
    G = nx.Graph(); G.add_edges_from((a, b) for a, b, e in t['edges'])
    err = {(min(a, b), max(a, b)): e for a, b, e in t['edges']}
    ro = t['readout']; oq = t['oneq']
    if isinstance(ro, dict): ro = {int(k): v for k, v in ro.items()}
    if isinstance(oq, dict): oq = {int(k): v for k, v in oq.items()}
    return G, err, ro, oq


def to_qc(gates, n):
    q = QuantumCircuit(n)
    for g in gates:
        k = g[0]
        if k == 'cx': q.cx(g[1], g[2])
        elif k == 'ry': q.ry(g[2], g[1])
        elif k == 'x': q.x(g[1])
        elif k == 'p': q.p(g[2], g[1])
        elif k == 'h': q.h(g[1])
        elif k == 'barrier': q.barrier()
        elif k == 'phase': q.global_phase += g[1]
        else: raise ValueError(g)
    return q


def shapes(G, k, center, cutoff=9):
    """Connected induced k-node subgraphs up to isomorphism, one embedding each."""
    region = nx.single_source_shortest_path_length(G, center, cutoff=cutoff)
    H = G.subgraph(region)
    level = {}
    for v in sorted(H.nodes):
        level.setdefault(nx.weisfeiler_lehman_graph_hash(H.subgraph([v]), iterations=4), tuple([v]))
    cur = list(level.values())
    for _ in range(k-1):
        nxt = {}
        for S in cur:
            nb = set().union(*(H[v] for v in S)) - set(S)
            for w in sorted(nb):
                T = tuple(sorted(set(S) | {w}))
                h = nx.weisfeiler_lehman_graph_hash(H.subgraph(T), iterations=4)
                nxt.setdefault(h, T)
        cur = list(nxt.values())
    return [list(s) for s in cur]


def subgraph_cm(G, nodes):
    idx = {v: i for i, v in enumerate(nodes)}
    e = [[idx[a], idx[b]] for a, b in G.subgraph(nodes).edges]
    return CouplingMap(e + [[b, a] for a, b in e])


def metrics(tc):
    ops = tc.count_ops()
    two = [i for i in tc.data if i.operation.name == 'cz']
    return dict(cz=int(ops.get('cz', 0)), depth=int(tc.depth()), cz_depth=int(tc.depth(lambda i: i.operation.name == 'cz')),
                sx=int(ops.get('sx', 0)), x=int(ops.get('x', 0)), rz=int(ops.get('rz', 0)), size=int(tc.size()))


def transpile_core(gates, n_wires, G, nodes, seed=0, level=3, initial_layout=None, layout_method=None, routing_method=None):
    """Transpile a logical circuit on the induced subgraph `nodes` (relabeled 0..k-1)."""
    k = len(nodes)
    qc = QuantumCircuit(k)
    qc.compose(to_qc(gates, n_wires), list(range(n_wires)), inplace=True)
    kw = dict(coupling_map=subgraph_cm(G, nodes), basis_gates=BASIS, optimization_level=level, seed_transpiler=seed)
    if initial_layout is not None: kw['initial_layout'] = initial_layout
    if layout_method: kw['layout_method'] = layout_method
    if routing_method: kw['routing_method'] = routing_method
    return transpile(qc, **kw)


def verify_native(tc, logical_gates, n_wires, tol=1e-8):
    """Operator of the transpiled circuit (layout and final permutation applied) against the
    logical circuit on all encoded basis columns; global phase ignored."""
    k = tc.num_qubits
    ref = QuantumCircuit(k); ref.compose(to_qc(logical_gates, n_wires), list(range(n_wires)), inplace=True)
    A = Operator(ref).data
    B = Operator.from_circuit(tc).data
    # Compare on the columns where the wires beyond n_wires (unused workspace) start in |0>.
    # Qiskit is little-endian, so wire w is bit w of the column index.
    cols = [i for i in range(1 << k) if (i >> n_wires) == 0]
    a = A[:, cols]; b = B[:, cols]
    ph = np.vdot(a.ravel(), b.ravel()); ph = ph/abs(ph)
    return float(np.max(abs(a - b/ph))) < tol, float(np.max(abs(a - b/ph)))


def esp(tc, nodes, err, ro, oq, measured=()):
    """Estimated success probability from calibration: product of gate and readout fidelities."""
    p = 1.0
    for inst in tc.data:
        n = inst.operation.name
        q = [tc.find_bit(x).index for x in inst.qubits]
        if n == 'cz':
            a, b = nodes[q[0]], nodes[q[1]]; p *= 1 - err[(min(a, b), max(a, b))]
        elif n in ('sx', 'x'):
            p *= 1 - oq[nodes[q[0]]]
    for q in measured: p *= 1 - ro[nodes[q]]
    return p
