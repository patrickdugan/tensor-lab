# tensor-lab

An experimental-mathematics laboratory for **tensor degenerations and
matrix-multiplication complexity inequalities**, built around the constructions
of OpenAI's preprint *An Upper Bound of 9/4 for the Matrix Multiplication
Exponent* (October 2, 2026; `reference/openai_nine_fourths_2026-10-02.pdf`,
Lean formalisation in `openai/math` under
`lean/OAI/LinearAlgebra/MatrixMultiplication/AuxiliarySeparation/`).

Everything is exact: integers, `fractions.Fraction`, and cyclotomic integers.
No floating point enters any verification.  See `RESEARCH_REPORT.md` for the
results and `notes/` for the mathematics.

## Installation

Python ≥ 3.10.  The exact core has no third-party dependencies; `numpy` is needed only for the two floating-point side experiments (`spectral`, `numeric_restriction`), and `pytest` for the tests.

```bash
git clone <this repository> tensor-lab && cd tensor-lab
python -m pip install numpy pytest   # or: python -m pip install -e ".[test]"
python -m pytest -q                  # 79 tests, ~1 minute
```

Small floating-point solves are much faster single-threaded on a busy machine:
set `OMP_NUM_THREADS=1` (and `OPENBLAS_NUM_THREADS=1`) before the numerical scripts.

## Commands

```bash
python -m tensorlab reproduce --max-a 5 --max-h 5        # three-sector certificates + independent verification
python -m tensorlab verify certificates/three_sector_a2_h1.json
python -m tensorlab search --max-a 5 --max-b 16 --max-sectors 3
python -m tensorlab fourier --cases 2,1 3,2               # finite Fourier separation (Prop. 3.1), exact in Q(zeta_15)
python -m tensorlab growth --grid 60 --lp-grid 12 --diag 2 3 4 5 \
    --candidate "(2,7)>=(2,1)+(2,1)+(2,1)+(2,1)+(2,1)"      # inequality engine (extremal profile, exact LP)
python -m tensorlab realize --packing 3,15 4,14 5,12         # realizability by general monomial weights (exact LP + Farkas)
python -m tensorlab spectral --cases 3,2,2 5,3,3            # support-functional test of dense rectangle packings (float)
python -m tensorlab dense --p 2 --n 8 --B 16               # exact verdict: C(3,16) vs 8 rectangles (Koszul, border rank, two copies)
python -m tensorlab amortize                               # copy-cost amortisation: two copies (sqrt(2)) for every C(3,2n) incl. C(3,16); three-copy restriction; obstructions
python -m tensorlab report                                 # regenerate catalogue/CATALOGUE.md and logs/report.md
```

Results and their evidence levels: `RESEARCH_REPORT.md`.  Mathematics:
`notes/01_reproduction.md`, `notes/02_extremal_profile_and_theta.md`,
`notes/03_obstructions.md`, `notes/04_open_direction_dense_rectangles.md`,
`notes/05_c316_dense_rectangles.md`, `notes/06_asymptotic_restriction.md`.

Lean (WSL Ubuntu-24.04 with elan; reuses the pinned Mathlib checkout by symlink):

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/<path>/tensor-lab/lean/wsl_build.sh
```

`lean/wsl_check.sh <file>` compiles one file quickly against the built
dependencies.  `lean/direct_check.sh <Module>...` compiles the named modules
with `lean` directly (bypassing `lake`) and runs `AxiomCheck.lean`; use it when
`lake` is slow, as it was on this machine late in the session.

## Layout

| path | content |
|---|---|
| `tensorlab/tensors.py` | exact sparse trilinear forms, families (`C(a,b)`, cyclic, `<n,m,p>`, dot products), products, sums, restrictions, flattening ranks |
| `tensorlab/degeneration.py` | weight degenerations (min-weight term selection), sector decomposition into shared-leg blocks, block identification up to leg permutation and coordinate bijection |
| `tensorlab/sectors.py` | the paper's three-sector construction, generalised alternating-sector constructions, certificate generation |
| `tensorlab/verify.py` | **independent** certificate verifier (does not import the construction code) |
| `tensorlab/search.py` | bounded exhaustive search over interval sectors, order-preserving matchings, weights solved by difference constraints; canonicalisation; scoring against the extremal profile |
| `tensorlab/cyclotomic.py`, `tensorlab/fourier.py` | exact `Q(zeta_L)` arithmetic and the finite shared-leg separation (Prop. 3.1) with the `u - v + 2(g-h) = 0` filter and the `(g-h)^2` weights |
| `tensorlab/lp.py`, `tensorlab/growth.py` | exact rational simplex; the profile inequality engine (paper constraints on a grid, extremal profile, candidate implication, Lemma 5.1 recurrence, what-if recurrences) |
| `tensorlab/weights_lp.py`, `tensorlab/experiments.py` | realizability of block configurations by arbitrary per-coordinate weights: integer weights or a Farkas obstruction, both checked exactly; experiments E1–E5 |
| `tensorlab/catalogue.py` | ranked catalogue with dominance filtering, rediscovery table and per-class certificates |
| `tensorlab/border.py`, `tensorlab/dense.py` | exact border-rank tools (Koszul and symmetric flattenings, interpolation decompositions, orbit dimensions) and the dense-rectangle analysis: block form, two-copy restriction, Koszul thresholds, Koszul LP, filtered tensor powers |
| `tensorlab/border_construct.py`, `tensorlab/first_order.py` | exact copy-cost constructions on tensor powers: the two-copy square for `C(3,4)` (border), the parity-twisted two-copy degeneration for every `C(3,2n)` and the three-copy restriction (both independently re-checked from the convolution tensor), the coarse reduction, and the exact first-order obstruction |
| `tensorlab/spectral.py`, `tensorlab/numeric_restriction.py` | floating-point side experiments: support (quantum) functionals, Levenberg–Marquardt restriction search, regularised continuation (exact vs border vs no solution), sparse copy-cost solver, coarse pattern |
| `certificates/` | machine-readable certificates (three-sector cases, Fourier separation, search finds) |
| `catalogue/` | ranked search catalogues |
| `lean/` | Lean 4 + Mathlib project: finite certified identities (`decide`), odd-sector obstruction, extremal-profile feasibility and counting obstruction, θ-growth, exponent transfer |
| `notes/` | mathematical notes (extremal profile, obstruction, search space) |
| `logs/` | experiment logs |
| `tests/` | pytest suite |

## Certificate format (version 1)

A three-sector certificate records the source family (`C(a,B)`), the integer
weight vectors on all three legs, the Y/Z sector partition, the minimum
weight, every retained and erased coefficient with its weight, and for each
block the target tensor, the leg permutation and the explicit coordinate maps
(target coordinate → source coordinate).  `verify.py` rebuilds the source from
the family spec, recomputes everything, and pulls each block back along the
maps to compare with its own copy of the target.  A passing verification is a
finite statement about the listed parameters only (see the `scope` field).
