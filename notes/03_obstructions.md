# Obstructions and failed approaches

Every approach below was tried because it could plausibly beat the extremal
profile `P*` (see `02_extremal_profile_and_theta.md`); each one fails, and the
reason is recorded with its evidence level.

Evidence levels: **[Lean]** machine-checked proof; **[exact]** exact finite
computation with an independently verified certificate; **[argument]** written
proof not machine-checked; **[heuristic]** asymptotic estimate.

## O1. More sectors do not help (odd-sector family) — [Lean]

The search finds, for every `e ≥ 1`, alternating `2e+1`-sector degenerations
`C(a, (2e+1)h + e(a-1)) ⇝ (2e+1) × C(a,h)` (normal and leg-exchanged blocks
alternating; certified e.g. by `general_sector_certificate(2, [1]*5, "NENEN")`).
`TensorLab.Obstruction.odd_sector_implied` proves that the resulting inequality

    (2e+1) P(a,h) ≤ P(a, (2e+1)h + e(a-1))

follows from row concavity and shifted tripling alone (chord inequality between
`h` and `h + Λ_e(2h+a-1)`, `Λ_e = (3^e-1)/2`).  Its overhead is again `θ = 1/2`.

## O2. Counting obstruction — [Lean] scalar part, [argument] link (valid for every degeneration)

**Theorem (`TensorLab.Extremal.counting_obstruction`).**  Let `a ≤ B` and let
blocks `C(x_i, r_i)` have their shared X-leg carrying an *input* leg of size
`x_i ≤ a`.  If `Σ_i (x_i + 2 r_i - 1) ≤ 2B + a - 1`, then `Σ_i P*(x_i, r_i) ≤ P*(a,B)`.

**Link to degenerations (complete argument, not machine-checked).**  Suppose
`C(a,B)` degenerates (or restricts) — by *any* linear maps, monomial or not — to
`T = Σ_i T_i` with `T_i ∈ X ⊗ Y_i ⊗ Z_i`, the `Y_i` independent and the `Z_i`
independent (a shared-leg block sum, the input of Corollary 3.2).  The
Y-flattening of `T` is block-diagonal from `⊕ Y_i*` into the independent
subspaces `X ⊗ Z_i`, so `rank_Y(T) = Σ rank_Y(T_i)`; likewise for Z.  Flattening
ranks are lower semicontinuous, hence do not increase under restriction or
degeneration, and `rank_Y C(a,B) = B`, `rank_Z C(a,B) = a+B-1`.  A block
`T_i ≅ C(x_i, r_i)` whose X-leg carries the input `x_i` has
`rank_Y + rank_Z = r_i + (x_i + r_i - 1)`.  Summing gives the hypothesis.

Consequence: **no degeneration of a single `C(a,B)` into a shared-X sum of
convolution blocks whose shared leg carries an input can violate `P*`** — whatever
the weights, basis changes or number of sectors.  In θ-language: overhead
`θ ≥ 1/2` is forced, and the paper's three-sector construction, which is tight for
this bound, is optimal in this class.

The key scalar lemma is `P*(p,q) ≤ (q + (p-1)/2) H_p` for all `p, q ≥ 1`
(`Extremal.P_le_row_line`): every value lies below the linear continuation of its
row.  Exact check on `[1,25]²`: `growth.check_counting_lemma` (equality exactly for
`q ≥ p` and `q = p - 1`).

## O3. Rectangles (output leg on the shared leg), monomial weights — [exact] instances, [argument] spacing lemma, [heuristic] asymptotics

The counting bound fails for blocks `C(p,q)` whose *output* leg is the shared leg
(`Extremal.rectangle_exceeds`: `P*(3,3) > 6/2 · H_5`).  Such a block is a full
`p × q` rectangle of cells in the `(j,k)` band of `C(a,B)`, needs `p + q - 1 ≤ a`,
and an isolated square rectangle with `p ≈ a/2` has value `≈ 0.75 H_p ≈ 0.595 H_a`
per Y- or Z-coordinate it uses — above the `H_a/2` per coordinate of `P*` — so a
dense packing of rectangles *would* violate `P*`.

Realizability was decided exactly with general per-coordinate weights on all
three legs (`tensorlab/weights_lp.py`): the exact simplex returns either integer
weights or a Farkas certificate (`μ ≥ 0` on erased cells, `ν` on retained cells,
equal i-, j-, k-marginals), and both are checked independently; in every case
exactly one of the two exists.

* **Spacing lemma.**  Rectangles of one *i-class* (same diagonal range, length
  `ℓ = p+q-1`) at period `δ`: realizable iff `δ ≥ ℓ`.  Exact for
  `(a,p,q) ∈ {(5,3,3), (7,3,3), (6,2,4), (6,3,2)}` and all `δ` (`logs/realizability.json`, E1).
  Proof idea [argument]: for `δ < ℓ` a forward cross cell and a backward cross cell
  between two rectangles have i-values inside the common i-range, and
  `μ = δ_fwd + δ_bwd` with the two "swapped" retained cells as `ν` (corrected inside
  the rectangles by second differences, which have zero j- and k-marginals) is an
  obstruction; this is the 2-support certificate the LP finds.  So a single class
  uses at most about half of the coordinates.
* **Two i-classes** interleaved with full Y and Z utilisation (`a = 9`, `C(2,3)` on
  i-range `[0,3]` and `C(3,2)` on `[5,8]`): obstructed (E2).
* **Double staircase** (a staircase plus the rectangles that would fill its gaps):
  obstructed for `(a,p,q) = (5,3,3), (7,4,4)` (E3).
* **Exhaustive packing search** (E5): every packing of interval templates
  (rectangles, normal and exchanged convolutions, arbitrary positions and i-ranges)
  into `C(a,B)` with total `P*`-value above `P*(a,B)`, for
  `(a,B) ∈ {(3,15), (3,16), (3,18)*, (4,12), (4,14), (4,16), (5,12), (5,14)}`:
  342 such packings exist combinatorially (320 tested); **none is realizable** by any monomial
  weights (each has a verified Farkas certificate).  `*` the `(3,18)` run tested the
  first 150 of 172.
* [heuristic] Asymptotically a single class reaches about `0.595 H_a / 2` per
  coordinate and two disjoint i-classes about `0.945 H_a / 2`, both below `H_a / 2`.

## O4. Multiplicativity (Kronecker substitution) adds nothing — [exact]

`C(a,b) ⊗ C(a',b')` is a coordinate restriction of `C(a + n(a'-1), b + n(b'-1))`,
`n = a+b-1`, so `P(A,B) ≥ P(a,b) P(a',b')`.  `P*` satisfies all 7568 instances inside
`[1,40]²` (3357 with equality; `growth.check_kronecker`).

## O5. Higher Clebsch–Gordan filtrations reduce to concavity — [argument]

Tensoring with `B_X(m)` instead of `B_X(2)` and splitting `V_{b-1} ⊗ V_{m-1}` by
Clebsch–Gordan would give `m P(a,b) ≥ Σ_s P(a, b+m-1-2s)`; for a concave row this is
already implied (Jensen for concave sequences).  Not implemented as a degeneration.

## O6. Staircases of rectangles — valid but weak — [exact]

`sectors.staircase_certificate(a,p,q,n)`: `C(a, (n-1)(p+q-1)+p) ⇝ n × C(p,q)` with
the output leg on X, verified for several parameter sets.  This is a valid
inequality family not in the paper, but every instance up to `B ≤ 40` has positive
slack on `P*` (17 666 checked).

## O7. Hypothetical double staircases — [exact] obstruction

If the gaps of a rectangle staircase could also be filled, 68 instances up to `40`
would violate `P*`.  They are not realizable (O3), which is why this family appears
in the catalogue only as a rejected candidate.

## O8. Sector-constant weights with x = 0 (the bounded search)

Within the search space of `search.py` (contiguous sectors, order-preserving
matchings, sector-constant weights solved as difference constraints) the cross-term
digraph must be acyclic; every fully identified result has non-negative slack on
`P*`.  See the catalogue for counts.
