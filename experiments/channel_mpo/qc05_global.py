"""Global density diagnostics for locally purified tensor networks.

The Hilbert-Schmidt overlap contracts double-layer matrix-product operators
without materializing a 2**n by 2**n density matrix.
"""
from __future__ import annotations
import numpy as np
from purified_mpo import PurifiedChain


def density_mpo(chain: PurifiedChain) -> list[np.ndarray]:
    out = []
    for a in chain.cores:
        l, p, e, r = a.shape
        w = np.einsum('lper,mqes->lmpqrs', a, a.conj(), optimize=True)
        out.append(w.reshape(l*l, 2, 2, r*r))
    return out


def hs_inner(a: PurifiedChain, b: PurifiedChain) -> float:
    if a.n != b.n:
        raise ValueError('site count mismatch')
    env = np.ones((1, 1), complex)
    for wa, wb in zip(density_mpo(a), density_mpo(b)):
        tmp = np.einsum('ab,apqc->bpqc', env, wa, optimize=True)
        env = np.einsum('bpqc,bqpd->cd', tmp, wb, optimize=True)
    z = env[0, 0]
    if abs(z.imag) > 1e-7 * max(1, abs(z.real)):
        raise ArithmeticError('HS overlap must be real')
    return float(z.real)


def hs_square_error(target: PurifiedChain, candidate: PurifiedChain) -> float:
    ta, cb = target.trace(), candidate.trace()
    if ta <= 0 or cb <= 0:
        raise ArithmeticError('bad purification trace')
    sq = (hs_inner(candidate, candidate)/(cb*cb)
          - 2*hs_inner(target, candidate)/(ta*cb)
          + hs_inner(target, target)/(ta*ta))
    return max(0., sq)
