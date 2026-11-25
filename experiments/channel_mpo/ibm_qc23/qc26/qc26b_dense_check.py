"""QC-26B independent dense density-matrix cross-check of the sparse Pauli engine (qc26b_engine.mjs).

  node qc26b_selftest.mjs cases.json && python qc26b_dense_check.py cases.json receipt.json

Evolves rho in the Schroedinger picture with explicit Kraus and unitary maps (no Pauli propagation, no shared
code) in the same gate order, and compares <Z_q> (with symmetric readout error) to the sparse result.
"""
import json
import sys

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
P1 = [I2, X, Y, Z]


def op_on(n, ops):
    """Tensor product with qubit q at bit q (little endian)."""
    m = np.array([[1]], dtype=complex)
    for q in reversed(range(n)):
        m = np.kron(m, ops.get(q, I2))
    return m


def expm_herm(h, theta):
    w, v = np.linalg.eigh(h)
    return (v * np.exp(-1j * theta * w)) @ v.conj().T


def conj(rho, u):
    return u @ rho @ u.conj().T


def pauli_channel(rho, n, q, probs):
    out = np.zeros_like(rho)
    for p, P in zip(probs, P1):
        out += p * conj(rho, op_on(n, {q: P}))
    return out


def gad(rho, n, q, e1, e3):
    """Generalised amplitude damping to |0> with Z contraction e3 and X/Y contraction e1."""
    g = 1 - e3
    lam = 1 - e1 * e1 / e3 if e3 > 0 else 1.0
    amp = [np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex), np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex)]
    deph = [np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex), np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)]
    out = np.zeros_like(rho)
    for a in amp:
        for b in deph:
            out += conj(rho, op_on(n, {q: a @ b}))
    return out


def cz_matrix(n, a, b):
    d = np.ones(2 ** n, dtype=complex)
    for idx in range(2 ** n):
        if (idx >> a) & 1 and (idx >> b) & 1:
            d[idx] = -1
    return np.diag(d)


def run(case, nonunital, coherent):
    n = case['n']
    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rho[0, 0] = 1
    for o in case['seq']:
        nm, q = o['name'], o['q']
        if nm == 'rz':
            rho = conj(rho, op_on(n, {q[0]: np.diag([np.exp(-0.5j * o['angle']), np.exp(0.5j * o['angle'])])}))
        elif nm == 'sx':
            off = case['sx'][str(o['p'][0])] if coherent else 0.0
            rho = conj(rho, op_on(n, {q[0]: expm_herm(X, (np.pi / 2 + off) / 2)}))
        elif nm == 'x':
            rho = conj(rho, op_on(n, {q[0]: X}))
        elif nm == 'cz':
            rho = conj(rho, cz_matrix(n, q[0], q[1]))
        elif nm == 'czerr':
            e = case['cz'].get(f"{min(o['p'])}_{max(o['p'])}")
            if not coherent or not e:
                continue
            ha = e['alpha'].get(str(o['p'][0]), 0.0)
            hb = e['alpha'].get(str(o['p'][1]), 0.0)
            h = 0.5 * (ha * op_on(n, {q[0]: Z}) + hb * op_on(n, {q[1]: Z}) + e['delta'] * op_on(n, {q[0]: Z, q[1]: Z}))
            rho = conj(rho, expm_herm(h, 1.0))
        elif nm == 'relax':
            e = o['e']
            if nonunital:
                rho = gad(rho, n, q[0], e[1], e[3])
            else:
                pxy = (1 - e[3]) / 4
                pz = (1 - e[1]) / 2 - pxy
                rho = pauli_channel(rho, n, q[0], [1 - 2 * pxy - pz, pxy, pxy, pz])
        elif nm == 'diag':
            dp = (1 - o['e'][1]) * 3 / 4
            rho = pauli_channel(rho, n, q[0], [1 - dp, dp / 3, dp / 3, dp / 3])
        elif nm == 'diag2':
            dp = (1 - o['e']) * 15 / 16
            acc = (1 - dp) * rho
            for a in range(4):
                for b in range(4):
                    if a or b:
                        acc = acc + dp / 15 * conj(rho, op_on(n, {q[0]: P1[a], q[1]: P1[b]}))
            rho = acc
        else:
            raise ValueError(nm)
    return [(1 - 2 * case['ro']) * float(np.real(np.trace(rho @ op_on(n, {k: Z})))) for k in range(n)]


def main():
    cases = json.load(open(sys.argv[1]))['cases']
    worst = {'coherent': 0.0, 'plain': 0.0}
    rows = []
    for ci, case in enumerate(cases):
        for res in case['res']:
            d_coh = max(abs(a - b) for a, b in zip(run(case, res['nonunital'], True), res['z']))
            d_no = max(abs(a - b) for a, b in zip(run(case, res['nonunital'], False), res['z_no_coherent']))
            if not (np.isfinite(d_coh) and np.isfinite(d_no)):
                raise SystemExit(f'non-finite difference in case {ci}')
            worst['coherent'] = max(worst['coherent'], d_coh)
            worst['plain'] = max(worst['plain'], d_no)
            rows.append({'case': ci, 'n': case['n'], 'nonunital': res['nonunital'],
                         'max_abs_diff_coherent': d_coh, 'max_abs_diff_plain': d_no})
    receipt = {'cases': len(cases), 'evaluations': len(rows), 'worst': worst,
               'pass': worst['coherent'] < 1e-10 and worst['plain'] < 1e-10, 'rows': rows}
    json.dump(receipt, open(sys.argv[2], 'w'), indent=1)
    print(json.dumps({k: receipt[k] for k in ('cases', 'evaluations', 'worst', 'pass')}))
    sys.exit(0 if receipt['pass'] else 1)


if __name__ == '__main__':
    main()
