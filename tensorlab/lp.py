"""A small exact (rational) linear-programming solver.

Two-phase simplex on sparse rows of ``Fraction``s.  It is slow compared with
floating-point solvers but every optimum it reports is an exact rational
number, which is what the inequality engine needs: a reported minimum of 0
really is 0, and a reported minimum of -1/720 really is negative.

Pivoting rule: Dantzig (most negative reduced cost) while progress is made,
switching to Bland's rule after a run of degenerate pivots, so termination is
guaranteed.

Problem form:  minimise  c . x   subject to  rows  (a . x  >=/<=/==  b),  x >= 0.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

Row = Dict[int, Fraction]


class LPError(Exception):
    pass


class LinearProgram:
    def __init__(self):
        self.n = 0
        self.names: List[str] = []
        self.index: Dict[str, int] = {}
        self.rows: List[Tuple[Row, str, Fraction]] = []
        self.objective: Row = {}

    def var(self, name: str) -> int:
        if name not in self.index:
            self.index[name] = self.n
            self.names.append(name)
            self.n += 1
        return self.index[name]

    def add(self, coeffs: Dict[str, object], sense: str, rhs) -> None:
        if sense not in (">=", "<=", "=="):
            raise LPError(sense)
        row: Row = {}
        for name, c in coeffs.items():
            c = Fraction(c)
            if c != 0:
                j = self.var(name)
                row[j] = row.get(j, Fraction(0)) + c
        row = {j: c for j, c in row.items() if c != 0}
        rhs = Fraction(rhs)
        if not row:
            # constant row: keep it only as a feasibility check
            ok = (rhs <= 0) if sense == ">=" else (rhs >= 0) if sense == "<=" else (rhs == 0)
            if not ok:
                self.rows.append(({}, sense, rhs))
            return
        self.rows.append((row, sense, rhs))

    def set_objective(self, coeffs: Dict[str, object]) -> None:
        self.objective = {}
        for name, c in coeffs.items():
            c = Fraction(c)
            if c != 0:
                self.objective[self.var(name)] = c

    # --------------------------------------------------------------------------------
    def solve(self, max_pivots: int = 100000) -> Tuple[str, Optional[Fraction], Optional[Dict[str, Fraction]]]:
        """Returns (status, optimal value, solution) with status in
        {"optimal", "infeasible", "unbounded", "pivot_limit"}."""
        n = self.n
        for row, sense, b in self.rows:
            if not row:
                return "infeasible", None, None
        m = len(self.rows)
        ncols = n
        tab: List[Row] = []
        rhs: List[Fraction] = []
        basis: List[int] = []
        artificials: List[int] = []
        for row, sense, b in self.rows:
            r = dict(row)
            slack = None
            if sense == "<=":
                r[ncols] = Fraction(1)
                slack = ncols
                ncols += 1
            elif sense == ">=":
                r[ncols] = Fraction(-1)
                slack = ncols
                ncols += 1
            if b < 0:
                r = {j: -c for j, c in r.items()}
                b = -b
            if slack is not None and r.get(slack) == 1:
                basis.append(slack)
            else:
                r[ncols] = Fraction(1)
                basis.append(ncols)
                artificials.append(ncols)
                ncols += 1
            tab.append(r)
            rhs.append(b)
        total = ncols
        art_set = set(artificials)
        # column index: which rows hold a nonzero in column j (maintained lazily)

        def pivot(pr: int, pc: int) -> None:
            prow = tab[pr]
            pv = prow[pc]
            if pv != 1:
                inv = 1 / pv
                for j in list(prow):
                    prow[j] *= inv
                rhs[pr] *= inv
            items = list(prow.items())
            bpr = rhs[pr]
            for i in range(m):
                if i == pr:
                    continue
                ri = tab[i]
                f = ri.get(pc)
                if not f:
                    continue
                for j, v in items:
                    nv = ri.get(j, 0) - f * v
                    if nv == 0:
                        ri.pop(j, None)
                    else:
                        ri[j] = nv
                rhs[i] -= f * bpr
            basis[pr] = pc

        def reduced_costs(cost: Row) -> Row:
            d: Row = dict(cost)
            for i in range(m):
                cb = cost.get(basis[i])
                if not cb:
                    continue
                for j, v in tab[i].items():
                    nv = d.get(j, 0) - cb * v
                    if nv == 0:
                        d.pop(j, None)
                    else:
                        d[j] = nv
            return d

        def run(cost: Row, allowed_set: set, pivots_left: List[int]) -> str:
            degenerate_run = 0
            while True:
                if pivots_left[0] <= 0:
                    return "pivot_limit"
                d = reduced_costs(cost)
                neg = [(j, v) for j, v in d.items() if j in allowed_set and v < 0]
                if not neg:
                    return "optimal"
                if degenerate_run > 50:
                    enter = min(j for j, _ in neg)  # Bland
                else:
                    enter = min(neg, key=lambda jv: (jv[1], jv[0]))[0]  # Dantzig
                best = None
                for i in range(m):
                    v = tab[i].get(enter)
                    if v and v > 0:
                        ratio = rhs[i] / v
                        if best is None or ratio < best[0] or (ratio == best[0] and basis[i] < basis[best[1]]):
                            best = (ratio, i)
                if best is None:
                    return "unbounded"
                if best[0] == 0:
                    degenerate_run += 1
                else:
                    degenerate_run = 0
                pivot(best[1], enter)
                pivots_left[0] -= 1

        pivots_left = [max_pivots]
        if artificials:
            cost1: Row = {j: Fraction(1) for j in artificials}
            st = run(cost1, set(range(total)), pivots_left)
            if st == "pivot_limit":
                return st, None, None
            val1 = sum((cost1.get(basis[i], Fraction(0)) * rhs[i] for i in range(m)), Fraction(0))
            if val1 > 0:
                return "infeasible", None, None
            for i in range(m):
                if basis[i] in art_set:
                    for j, v in tab[i].items():
                        if j not in art_set and v != 0:
                            pivot(i, j)
                            break
        cost2: Row = {j: c for j, c in self.objective.items()}
        allowed = set(j for j in range(total) if j not in art_set)
        st = run(cost2, allowed, pivots_left)
        if st != "optimal":
            return st, None, None
        x = [Fraction(0)] * total
        for i in range(m):
            x[basis[i]] = rhs[i]
        value = sum((cost2.get(j, Fraction(0)) * x[j] for j in range(total)), Fraction(0))
        sol = {self.names[j]: x[j] for j in range(n)}
        return "optimal", value, sol
