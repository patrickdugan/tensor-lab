"""Outcome-independent chain selection for QPU02 replicates.

Same calibrated cost and beam search as qc23_compile.choose_line_calibrated, but with
the qubits of earlier replicates excluded, so the replicate runs on a disjoint chain.
Read-only at IBM (calibration only). Prints the path as space-separated indices.
"""
import argparse, sys
import qpu01 as old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', type=int, required=True)
    ap.add_argument('--backend', default='ibm_kingston')
    ap.add_argument('--exclude', type=int, nargs='+', required=True)
    ap.add_argument('--beam', type=int, default=256)
    a = ap.parse_args()
    h = old.hardware_helpers()
    import qc23_compile as c
    service, backend = old.connection(h, a.backend)
    t = backend.target
    edges = {k: v for k, v in c.edge_errors(t).items() if k[0] not in a.exclude and k[1] not in a.exclude}
    readout = c.readout_errors(t, backend.num_qubits); oneq = c.single_qubit_errors(t, backend.num_qubits)
    nbr = {q: set() for q in range(backend.num_qubits) if q not in a.exclude}
    for x, y in edges:
        nbr[x].add(y); nbr[y].add(x)
    frontier = sorted(([[q], c.path_cost([q], edges, readout, oneq)] for q in nbr if nbr[q]), key=lambda s: (s[1], s[0]))[:a.beam]
    for _ in range(a.b - 1):
        nxt = []
        for path, _c in frontier:
            for o in sorted(nbr[path[-1]] - set(path)):
                p2 = path + [o]; nxt.append([p2, c.path_cost(p2, edges, readout, oneq)])
        nxt.sort(key=lambda s: (s[1], s[0])); frontier = nxt[:a.beam]
    path, cost = min(frontier, key=lambda s: (s[1], s[0]))
    if path[0] > path[-1]: path = path[::-1]
    print(' '.join(map(str, path))); print('cost', cost, file=sys.stderr)


if __name__ == '__main__':
    main()
