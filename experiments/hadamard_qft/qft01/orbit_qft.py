"""Orbit-restricted CRT Fourier transforms; exact modular certificate helpers.
Numerical QFT matrices are reference simulations, not symbolic gate certificates.
"""
from itertools import product
from math import prod, isqrt, gcd
import numpy as np


def orbits(n, group):
    group = set(group)
    assert 1 in group and all(gcd(g,n)==1 for g in group)
    assert all((a*b) % n in group for a in group for b in group)
    todo, out = set(range(n)), []
    while todo:
        o = sorted({min(todo)*g % n for g in group})
        assert set(o) <= todo
        todo.difference_update(o)
        out.append(o)
    return out


def fourier(n):
    k = np.arange(n, dtype=np.int64)
    return np.exp(2j*np.pi*((k[:, None]*k) % n)/n)/np.sqrt(n)


def embedding(n, blocks):
    b = np.zeros((n, len(blocks)))
    for j, o in enumerate(blocks):
        b[o, j] = 1/np.sqrt(len(o))
    return b


def orbit_fourier(n, blocks):
    b = embedding(n, blocks)
    return b.T @ fourier(n) @ b


def crt_layout(factors, blocks):
    """Time CRT labels x_i=x mod n_i; dual labels k_i=k*(N/n_i)^-1 mod n_i."""
    n = prod(factors)
    axes = [dict((x, i) for i, o in enumerate(b) for x in o) for b in blocks]
    dims = tuple(map(len, blocks))
    sizes = np.array([prod(len(blocks[i][j]) for i,j in enumerate(c))
                      for c in product(*(range(d) for d in dims))])
    assert sizes.sum() == n
    time = np.zeros((n, len(sizes)))
    freq = time.copy()
    for x in range(n):
        a = tuple(axis[x % p] for p, axis in zip(factors, axes))
        b = tuple(axis[(x * pow(n//p, -1, p)) % p] for p, axis in zip(factors, axes))
        i = np.ravel_multi_index(a, dims)
        j = np.ravel_multi_index(b, dims)
        time[x, i] = 1/np.sqrt(sizes[i])
        freq[x, j] = 1/np.sqrt(sizes[j])
    return time, freq, sizes


def kron_all(mats):
    a = np.ones((1, 1))
    for b in mats:
        a = np.kron(a, b)
    return a


def paf(rows):
    rows = np.asarray(rows, dtype=np.int64)
    n = rows.shape[1]
    return np.array([np.sum(rows*np.roll(rows, -t, axis=1)) for t in range(n)])


def target_paf(n, bordered=True):
    r = np.full(n, -4 if bordered else 0, dtype=np.int64)
    r[0] = 4*n
    return r


def phase_sample(rows):
    """QFT on the position register of sum_{f,x} rows[f,x]|f,x>/sqrt(4n)."""
    a = np.asarray(rows)
    assert a.shape[0] == 4 and np.all(np.abs(a) == 1)
    return np.sum(np.abs(np.fft.fft(a, axis=1))**2, axis=0)/(4*a.shape[1]**2)


def prime(q):
    return q >= 2 and all(q % i for i in range(2,isqrt(q)+1))


def order_root(n, q):
    assert prime(q) and (q-1)%n == 0
    factors = [p for p in range(2,n+1) if n%p == 0 and prime(p)]
    for a in range(2,q):
        z = pow(a,(q-1)//n,q)
        if pow(z,n,q)==1 and all(pow(z,n//p,q)!=1 for p in factors):
            return z
    raise ValueError('no primitive root of specified order')


def modular_fourier(n, blocks, q):
    """Value-basis transform: unnormalized orbit coefficient values in F_q."""
    z = order_root(n,q)
    return np.array([[sum(pow(z,(o[0]*x)%n,q) for x in b)%q for b in blocks]
                     for o in blocks], dtype=np.int64)


def ntt_symmetric_paf(values, factors, blocks, q=5981):
    """Sum of PAFs of symmetric rows, returned at every dual orbit modulo q.
    Input coefficients are constant on time orbits, values shape (4,*dims).
    The CRT roots are local primitive roots; output labeling may differ by units
    from a selected global primitive root, which preserves target values.
    """
    n = prod(factors)
    assert q>8*n+8 and (q-1)%n==0
    assert all(sorted({(-x)%p for x in o})==o for p,b in zip(factors,blocks) for o in b)
    assert q*q*max(4,*(len(b) for b in blocks)) < 2**63
    values = np.asarray(values)
    assert values.shape == (4, *(len(b) for b in blocks)) and np.all((values==1)|(values==-1))
    values = values.astype(np.int64)
    a = values.copy()
    for i,(p,b) in enumerate(zip(factors,blocks),1):
        t = modular_fourier(p,b,q)
        a = np.moveaxis(a,i,-1)
        a = (a @ t.T) % q
        a = np.moveaxis(a,-1,i)
    return np.sum(a*a,axis=0)%q


def verify_symmetric_b4(values, factors, blocks, q=5981):
    v = ntt_symmetric_paf(values,factors,blocks,q)
    target = np.full(v.shape,4*(prod(factors)+1)%q)
    target[(0,)*v.ndim] = 4
    return bool(np.array_equal(v,target))
