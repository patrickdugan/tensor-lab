# Reproduction of the three-sector construction (paper Lemma 4.2)

Source: OpenAI, *An Upper Bound of 9/4 for the Matrix Multiplication Exponent*,
October 2, 2026, Section 4.2 and Figure 1 (copy in `reference/`), and the Lean
modules `Sector/Weights.lean`, `Sector/Degeneration.lean`, `Sector/Branches.lean`
of `openai/math` (commit `adc7f124`, checked out under WSL in `~/oai-verify/math`).

## Construction

For positive integers `a, h` put `B = 3h + a - 1` and take
`C(a,B) = Σ_{i<a, j<B} x_i y_j z_{i+j}` (`a·B` terms, coefficient 1).

| leg | left | middle | right |
|---|---|---|---|
| Y | `[0, h)` | `[h, 2h+a-1)` | `[2h+a-1, B)` |
| Z | `[0, h+a-1)` | `[h+a-1, 2h+a-1)` | `[2h+a-1, 3h+2a-2)` |

Weights: `+1` on middle Y, `-1` on middle Z, `0` on X and elsewhere.

## What the certificates check (all exact, integers and `Fraction`)

`python -m tensorlab reproduce --max-a 5 --max-h 5` writes 25 certificates
(`certificates/three_sector_a{a}_h{h}.json`), each verified by the independent
verifier `tensorlab/verify.py`, which does not import the construction code:

1. The source is rebuilt from `{"family": "convolution", "a": a, "b": B}`; dims and nnz match.
2. Every term weight is recomputed; the minimum is 0; the retained set equals the exact
   set of minimum-weight terms; the erased set and each erased weight are recomputed.
3. Retained ∪ erased = support, disjointly (every source coefficient accounted for).
4. Every erased term has weight exactly `+1`, so no supported term has negative weight
   (for `a = 1` nothing is erased).
5. Y and Z sectors are disjoint and in range; every retained term lies in a matched
   (Y-sector, Z-sector) pair; no retained term joins different blocks (no cross-sector
   contamination).
6. Each block is pulled back along explicit coordinate maps and compared exactly with
   the verifier's own copy of `C(a,h)`:
   * left: identity maps;
   * middle: legs Y and Z exchanged (`swap_YZ`), X map `u ↦ a-1-u` (reversal), Y map
     `r ↦ h+a-1+r` (onto source Z), Z map `t ↦ h+t` (onto source Y);
   * right: translation by `2h+a-1` on Y and Z.
   The maps are injective and cover the block support; the X map is a bijection onto
   the shared X support, so all three blocks keep the full shared first leg.
7. The blocks partition the retained terms and use disjoint Y and disjoint Z coordinates.

Counts: retained `3ah`, erased `a(3h+a-1) - 3ah = a(a-1)`, matching the paper's Figure 1
for `(a,h) = (2,1)`: erased terms `x_0 y_1 z_1` and `x_1 y_2 z_3`.

Tampering tests (`tests/test_reproduction.py`) confirm that the verifier rejects a
dropped retained term, a changed weight, a block map without the reversal, and a
sector partition with cross contamination.

## Lean

`lean/TensorLab/Finite.lean` restates the construction for integer coefficient
functions and proves, by kernel evaluation (`decide`, no `native_decide`), for
`(a,h) ∈ {(2,1), (3,2), (2,3), (4,1)}`:

* `retained a h = leftBranch + middleBranch + rightBranch` (exact tensor identity);
* `WeightsZeroOne a h` (every supported term has weight 0 or 1);
* `MiddleIsExchanged a h` (the middle branch, pulled back along the reversal and the
  two translations, is `C(a,h)` with legs Y, Z exchanged) for three of the cases.

These are finite statements.  The statement for all `(a,h)` is OpenAI's
`Sector.retained_eq_three_branches` and `Sector.middleTensor_pullback`, which this
project does not re-prove.

## Scope

* Each certificate is a finite exact identity for its parameters only.
* The tensor identity is combinatorial and holds over every field.
* The passage to `λ(C(a,B)) ≥ 3^{p_X} λ(C)^{2/3} λ(C^σ)^{1/3}` uses the paper's
  Lemma 2.3 (interpolation: infinitely many distinct nonzero points, e.g.
  characteristic zero) and Corollary 3.2 (shared-leg entropy inequality, which uses
  Proposition 3.1 and therefore a primitive `5M`-th root of unity).
* The symmetrised inequality `P(a,3h+a-1) ≥ 3P(a,h)` then needs the product over
  the six leg permutations; that step is OpenAI's, not re-done here.

## Fourier separation (Proposition 3.1)

`python -m tensorlab fourier --cases 1,1 2,1 2,2 3,1 3,2 4,1` applies the finite
separation to the retained tensor of each three-sector certificate (`M = 3`
blocks, `L = 5M = 15` copies), computing in `Q(ζ_15) = Q[x]/Φ_15(x)` exactly
(`Φ_15 = 1 - x + x^3 - x^4 + x^5 - x^7 + x^8`).  Checked:

* `(1/L) Σ_r ζ^{re} = [e = 0]` for every `|e| ≤ 3(M-1) = 6` (computed by summation in the field);
* every transformed coefficient is rational and equals `c · [u - v + 2(g-h) = 0]`;
* with `w(X_{a,g}) = g²`, `w(Y_{h,b,u}) = hu - h²`, `w(Z_{h,c,v}) = -hv`, every surviving
  term has total weight `(g-h)²` (some individual weights are negative);
* the weight-zero terms are exactly `g = h, u = v`, and they form the full direct sum
  `⊕_h A_h ⊗ B_X(M)` with disjoint coordinates on all three legs;
* maximum surviving weight `≤ (M-1)²`.

Field requirements: a primitive 15th root of unity with 15 invertible (any field
of characteristic 0, or characteristic `p ∤ 15` containing `ζ_15`); the weight
bookkeeping and block identification are field-independent.
`fourier.separation_from_certificate` accepts only a certificate that passes the
independent verifier.
