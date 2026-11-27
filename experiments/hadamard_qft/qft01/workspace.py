"""HAD-QFT-QPU03: workspace-assisted preparation-equivalent compaction.

Shell transform on 2^b labels, binary encoding, wire 0 is the most significant bit.
Data wires are 0..b-1, clean workspace wires are b..b+m-1 and are returned to |0>.
Gate set: x, ry, cx only (angles are binary64 values of exact analytic angles).

Recursion (see paper, eq. rec and Theorems compact/approx):
  W_{2L} = (I tensor W_L) P C P^dagger,  L = 2^r lower labels.
The boundary rotation C rotates the leading bit h by RY(-pi/2) iff the r lower bits are 0.
Workspace: the all-zero predicate on the lower bits is computed into clean ancillas by a
chain of relative-phase Toffoli (Margolus) gates and uncomputed by their exact inverses.
Every ancilla is a basis-state function of the data wires at the moment it is used as a
control, so the relative phase of the Margolus gate never acts (it only acts on target=1
inputs, and the ancilla target starts and is returned at |0>).
"""
from __future__ import annotations
from math import atan, asin, pi, sqrt
import numpy as np
from recursive import mux, inverse, simulate
from radial_qft import radial_matrix
from compact import simplify, level_bound


def count_cx(g): return sum(x[0] == 'cx' for x in g)


def margolus(a, b, t):
    """Relative-phase Toffoli, 3 CX. Exact Toffoli action when the target is |0>."""
    return [('ry', t, pi/4), ('cx', b, t), ('ry', t, pi/4), ('cx', a, t),
            ('ry', t, -pi/4), ('cx', b, t), ('ry', t, -pi/4)]


def _mux(vals, controls, target, prep=False, order=None):
    """Gray-cycle multiplexer. controls[0] is the most significant bit of the table index.
    prep=True: the target is |0> on every selected input, so the terminal CX is dropped by
    replacing theta with pi-theta on the branches it would flip (preparation-equivalent)."""
    k = len(controls)
    if order == 'rev': cs = list(reversed(controls))
    elif order == 'near':   # most-significant table bit adjacent in wire index to the target
        cs = sorted(controls, key=lambda q: (abs(q-target), q))
    elif order is not None: cs = list(order)
    else: cs = list(controls)
    assert sorted(cs) == sorted(controls)
    out_vals = []
    for x in range(1 << k):
        old = 0
        for q in controls:
            old = 2*old + ((x >> (k-1-cs.index(q))) & 1)
        v = vals[old]
        out_vals.append(pi - v if prep and k and x >> (k-1) else v)
    g = mux(out_vals, cs, target)
    if prep and k:
        assert g[-1] == ('cx', cs[0], target)
        g = g[:-1]
    return g


def boundary(h, low, anc, theta=-pi/2, order=None):
    """C_L with len(anc) clean ancillas (0 = ancilla-free multiplexer)."""
    r = len(low)
    m = min(len(anc), r-1)
    if m == 0:
        return _mux([theta] + [0.]*((1 << r)-1), low, h, order=order)
    flips = [('x', q) for q in low]
    chain = margolus(low[0], low[1], anc[0])
    for j in range(1, m):
        chain += margolus(anc[j-1], low[j+1], anc[j])
    ctrl = [anc[m-1]] + list(low[m+1:])
    vals = [0.]*((1 << len(ctrl))-1) + [theta]          # all ones (zero lower bits flipped)
    core = _mux(vals, ctrl, h, order=order)
    return flips + chain + core + inverse(chain) + flips


def best_m(r, available):
    """Ancilla count minimizing the logical CX of C_L = 2^(r-m) + 6m."""
    costs = [(2**(r-m) + 6*m if m else 2**r, m) for m in range(0, min(available, r-1)+1)]
    return min(costs)[1]


def prep_chain(h, low, anc, order=None):
    """Exact compact preparation with a reversible prefix-predicate chain (len(anc) >= r-1).
    Bit 0: [pi, 2 asin z] controlled by h. Bit k>=1: base [pi, 2 atan z] controlled by h, then an
    exact correction RY(2 asin z - 2 atan z) controlled by flag_k = h AND low[0..k-1] (all ones),
    where flag_k is computed by a Margolus gate right after bit k-1 is prepared and uncomputed in
    reverse order at the end. Cost 1 + 9(r-1) CX."""
    r = len(low); out = []; chain = []
    for k, t in enumerate(low):
        z = 2**(-(1 << (r-1-k))/2); a = 2*atan(z); sp = 2*asin(z)
        if k == 0:
            out += _mux([pi, sp], [h], t, prep=True, order=order)
        else:
            ctrl = h if k == 1 else anc[k-2]
            mg = margolus(ctrl, low[k-1], anc[k-1]); out += mg; chain.append(mg)
            out += _mux([pi, a], [h], t, prep=True, order=order)
            out += _mux([0., sp - a], [anc[k-1]], t, order=order)       # exact CRY correction, 2 CX
    for mg in reversed(chain): out += inverse(mg)
    return out


def chain_cost(r): return 1 + 9*(r-1)
def mux_prep_cost(r): return 2**(r+1) - 2 - r


def core(b, mode='approx', m=0, prune=None, order=None, m_levels=None, approx_from=4, exact_prep='auto'):
    """Return (gates, n_wires). mode: 'exact' or 'approx' (r>=4 geometric approximation).
    m: available clean ancillas (wires b..b+m-1); each level uses best_m unless m_levels
    overrides. prune: set of (r,k) rotations replaced by identity (error budget, see bound)."""
    prune = prune or set()
    anc = list(range(b, b+m))
    m_levels = m_levels or {}

    def wprod(w):
        if len(w) == 1:
            return [('ry', w[0], -pi/2)]
        h, *low = w
        r = len(low); p = []
        for k, t in enumerate(low):
            z = 2**(-(1 << (r-1-k))/2); a = 2*atan(z)
            if mode == 'approx' and r >= approx_from:
                if (r, k) in prune:
                    # bit stays |0>; the leading-bit-zero branch still flips it (pi on h=0 branch)
                    p += _mux([pi, 0.], [h], t, prep=True, order=order)
                else:
                    p += _mux([pi, a], [h], t, prep=True, order=order)
            else:
                v = [pi]*(1 << k) + [a]*(1 << k); v[-1] = 2*asin(z)
                p += _mux(v, [h] + low[:k], t, prep=True, order=order)
        use_chain = (not (mode == 'approx' and r >= approx_from)) and len(anc) >= r-1 and r >= 2 and (
            exact_prep == 'chain' or (exact_prep == 'auto' and chain_cost(r) < mux_prep_cost(r)))
        if use_chain:
            p = prep_chain(h, low, anc, order=order)
        mm = m_levels.get(r, best_m(r, len(anc)))
        c = boundary(h, low, anc[:mm], order=order)
        return inverse(p) + c + p + wprod(low)

    wires = list(range(b))
    g = [('x', i) for i in wires] + wprod(wires)
    return simplify(g), b + m


def prune_bound(b, prune):
    """Certified operator-norm contribution of pruned rotations at approximate levels.
    Each level error is <= ||g - v'||; replacing the bit-k rotation by identity moves the
    product vector by 2 sin(theta_k/2) with tan(theta_k) = z_k (rotation angle 2*atan(z))."""
    tot = 0.
    for (r, k) in prune:
        z = 2**(-(1 << (r-1-k))/2)
        tot += 2*np.sin(atan(z)/2)
    return tot


def approx_bound(b, approx_from=4):
    return sum(level_bound(r) for r in range(approx_from, b))


def exact_matrix(b):
    return radial_matrix((1 << b)-1)


def verify(gates, b, m, tol_exact=1e-9, bound=0., rng_seed=0, n_random=16):
    """Full verification on the encoded space with clean workspace.
    Returns dict. Checks: every basis column, random complex superpositions, workspace
    return to |0> (leakage), complex relative phases (spectral-norm error of the data block)."""
    n = b + m; d = 1 << b; Q = exact_matrix(b)
    cols = np.zeros((1 << n, d), dtype=complex)
    for j in range(d): cols[j << m, j] = 1.
    out = simulate(gates, n, cols)
    block = out[::1 << m, :]                                  # workspace = 0
    mask = np.ones(1 << n, dtype=bool); mask[::1 << m] = False
    leak = np.linalg.norm(out[mask, :], axis=0)               # norm outside the workspace-zero block
    op_err = float(np.linalg.norm(block - Q, ord=2))
    col_err = float(np.max(abs(block - Q)))
    rng = np.random.default_rng(rng_seed)
    x = rng.normal(size=(d, n_random)) + 1j*rng.normal(size=(d, n_random)); x /= np.linalg.norm(x, axis=0)
    xin = np.zeros((1 << n, n_random), dtype=complex); xin[::1 << m, :] = x
    yo = simulate(gates, n, xin)
    rand_err = float(np.max(np.linalg.norm(yo[::1 << m, :] - Q @ x, axis=0)))
    rand_leak = float(np.max(np.linalg.norm(yo[mask, :], axis=0)))
    unitary_dev = float(np.max(abs(block.conj().T @ block - np.eye(d)))) if m == 0 else None
    ok = op_err <= bound + tol_exact and max(float(leak.max()), rand_leak) < 1e-9
    return dict(operator_error=op_err, max_entry_error=col_err, random_state_error=rand_err,
                max_workspace_leakage=max(float(leak.max()), rand_leak), bound=bound, ok=bool(ok),
                logical_cx=count_cx(gates), n_wires=n)


if __name__ == '__main__':
    import json, sys
    rows = []
    for b in (3, 4, 5, 6, 7, 8):
        for mode in ('exact', 'approx'):
            for m in (0, 1, 2, 3, 4):
                g, n = core(b, mode, m)
                if b <= 6 and n <= 11:
                    v = verify(g, b, m, bound=approx_bound(b) if mode == 'approx' else 0.)
                else:
                    v = dict(logical_cx=count_cx(g), n_wires=n)
                rows.append(dict(b=b, mode=mode, m=m, **v))
                print(b, mode, m, v['logical_cx'], v.get('operator_error'), v.get('max_workspace_leakage'), v.get('ok'), flush=True)
