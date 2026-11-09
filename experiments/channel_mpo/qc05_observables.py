"""Physical Pauli observable probes for locally purified tensor networks."""
from __future__ import annotations
import numpy as np
from purified_mpo import PurifiedChain

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1, -1]).astype(complex)
P = {'I': I, 'X': X, 'Y': Y, 'Z': Z}


def pauli_expectation(chain: PurifiedChain, word: str) -> float:
    if len(word) != chain.n or any(c not in P for c in word):
        raise ValueError('Pauli word shape or alphabet invalid')
    result = np.ones((1, 1), complex)
    for a, letter in zip(chain.cores, word):
        result = np.einsum('lm,lpar,qp,mqas->rs', result, a,
                           P[letter], a.conj(), optimize=True)
    value = result[0, 0]/chain.trace()
    if abs(value.imag) > 1e-8:
        raise ArithmeticError('Pauli expectation has nonreal component')
    return float(value.real)


def sketch_words(n: int, width: int = 8) -> tuple[str, ...]:
    if n < 2 or width < 1:
        raise ValueError('require n>=2 and width>=1')
    words = []
    def add(positions, operator):
        letters = ['I']*n
        for i in positions:
            letters[i] = operator
        word = ''.join(letters)
        if word not in words:
            words.append(word)
    for i in (0, n//2, n-1):
        add((i,), 'Z')
    for i in (0, n//2, n-1):
        add((i,), 'X')
    for i in (0, n//2-1, n-2):
        add((i, i+1), 'Z')
    for i in (0, n//2-1, n-2):
        add((i, i+1), 'X')
    add(range(n), 'Z')
    add(range(n), 'X')
    return tuple(words[:width])
