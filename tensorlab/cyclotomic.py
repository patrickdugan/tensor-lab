"""Exact arithmetic in the cyclotomic field Q(zeta_L) = Q[x] / Phi_L(x).

Elements are polynomials of degree < phi(L) with rational coefficients, reduced
modulo the L-th cyclotomic polynomial, which is computed exactly by integer
polynomial division of x^L - 1 by the Phi_d for proper divisors d of L.
This is what makes the finite Fourier filter in ``fourier.py`` an exact
identity rather than a floating-point approximation.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import List, Sequence


def _poly_divmod_exact(num: List[int], den: List[int]) -> List[int]:
    """Exact division of integer polynomials (coefficient lists, low degree
    first); raises if the remainder is nonzero."""
    num = list(num)
    dn = len(den) - 1
    if den[dn] not in (1, -1):
        raise ValueError("divisor must be monic up to sign")
    q = [0] * (len(num) - dn)
    for i in range(len(num) - 1, dn - 1, -1):
        coef = num[i] // den[dn]
        q[i - dn] = coef
        if coef:
            for j in range(dn + 1):
                num[i - dn + j] -= coef * den[j]
    if any(num):
        raise ArithmeticError("non-exact polynomial division")
    return q


@lru_cache(maxsize=None)
def cyclotomic_polynomial(n: int) -> tuple:
    """Integer coefficients of Phi_n(x), low degree first."""
    if n < 1:
        raise ValueError
    if n == 1:
        return (-1, 1)
    poly = [-1] + [0] * (n - 1) + [1]  # x^n - 1
    for d in range(1, n):
        if n % d == 0:
            poly = _poly_divmod_exact(poly, list(cyclotomic_polynomial(d)))
    return tuple(poly)


class Cyclotomic:
    """An element of Q(zeta_L)."""

    __slots__ = ("L", "c")

    def __init__(self, L: int, coeffs: Sequence[Fraction] | None = None):
        self.L = int(L)
        phi = cyclotomic_polynomial(self.L)
        deg = len(phi) - 1
        c = [Fraction(0)] * deg
        if coeffs:
            for i, v in enumerate(coeffs):
                c[i] = Fraction(v)
        self.c = c

    @classmethod
    def zero(cls, L: int) -> "Cyclotomic":
        return cls(L)

    @classmethod
    def one(cls, L: int) -> "Cyclotomic":
        return cls.rational(L, 1)

    @classmethod
    def rational(cls, L: int, q) -> "Cyclotomic":
        z = cls(L)
        if len(z.c) == 0:  # L = 1: field is Q itself, degree 0 -- store in c[0]
            z.c = [Fraction(q)]
        else:
            z.c[0] = Fraction(q)
        return z

    @classmethod
    def zeta_power(cls, L: int, e: int) -> "Cyclotomic":
        """zeta^e, reduced."""
        e %= L
        phi = list(cyclotomic_polynomial(L))
        deg = len(phi) - 1
        # x^e mod Phi_L, by repeated reduction
        poly = [Fraction(0)] * (e + 1)
        poly[e] = Fraction(1)
        return cls(L, _reduce(poly, phi, deg))

    def is_rational(self) -> bool:
        return all(v == 0 for v in self.c[1:])

    def as_rational(self) -> Fraction:
        if not self.is_rational():
            raise ValueError("element is not rational")
        return self.c[0] if self.c else Fraction(0)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, Fraction)):
            return self.is_rational() and self.as_rational() == Fraction(other)
        if not isinstance(other, Cyclotomic):
            return NotImplemented
        return self.L == other.L and self.c == other.c

    def __hash__(self):
        return hash((self.L, tuple(self.c)))

    def __add__(self, other: "Cyclotomic") -> "Cyclotomic":
        if isinstance(other, (int, Fraction)):
            other = Cyclotomic.rational(self.L, other)
        if other.L != self.L:
            raise ValueError("mixed cyclotomic orders")
        return Cyclotomic(self.L, [x + y for x, y in zip(self.c, other.c)])

    __radd__ = __add__

    def __neg__(self) -> "Cyclotomic":
        return Cyclotomic(self.L, [-x for x in self.c])

    def __sub__(self, other: "Cyclotomic") -> "Cyclotomic":
        return self + (-other)

    def scale(self, q) -> "Cyclotomic":
        q = Fraction(q)
        return Cyclotomic(self.L, [x * q for x in self.c])

    def __mul__(self, other) -> "Cyclotomic":
        if isinstance(other, (int, Fraction)):
            return self.scale(other)
        if other.L != self.L:
            raise ValueError("mixed cyclotomic orders")
        phi = list(cyclotomic_polynomial(self.L))
        deg = len(phi) - 1
        prod = [Fraction(0)] * (2 * deg)
        for i, x in enumerate(self.c):
            if x == 0:
                continue
            for j, y in enumerate(other.c):
                if y != 0:
                    prod[i + j] += x * y
        return Cyclotomic(self.L, _reduce(prod, phi, deg))

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return f"Cyclotomic(L={self.L}, {[str(v) for v in self.c]})"


def _reduce(poly: List[Fraction], phi: List[int], deg: int) -> List[Fraction]:
    """Reduce ``poly`` modulo the monic polynomial ``phi`` of degree ``deg``."""
    poly = list(poly)
    for i in range(len(poly) - 1, deg - 1, -1):
        coef = poly[i]
        if coef != 0:
            for j in range(deg + 1):
                poly[i - deg + j] -= coef * phi[j]
    out = poly[:deg]
    out += [Fraction(0)] * (deg - len(out))
    return out


def filter_sum(L: int, e: int) -> Cyclotomic:
    """(1/L) * sum_{r=0}^{L-1} zeta^{r e}, computed by summation (not assumed)."""
    s = Cyclotomic.zero(L)
    for r in range(L):
        s = s + Cyclotomic.zeta_power(L, r * e)
    return s.scale(Fraction(1, L))
