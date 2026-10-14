# The asymptotic restriction for dense rectangles (C(3,16) and all C(3,2n))

**Question.**  Is `T_n = C(2,2)^σ ⊗ B_X(n)` (n rectangles sharing X) an asymptotic
restriction of `C(3,2n)`?  For `n = 8` this is `C(3,16)`.  By Strassen's spectral
theorem this is *equivalent* to `λ(C(3,2n)) ≥ n^{p_X} λ(R)` for every tensor
character `λ`, which is exactly what the growth argument needs.  Exact
degenerations are ruled out (`05_c316_dense_rectangles.md`).

**Answer so far.**  Still open, but the copy cost amortises, and for every `n`,
including `C(3,16)`, two copies of the square suffice:

| per-factor copy cost `c_N^{1/N}` | `n = 2` (`C(3,4)`) | every `n ≥ 2` (incl. `C(3,16)`) |
|---|---|---|
| single copy (`N = 1`) | 2 (one copy impossible, Koszul) | 2 |
| exact restriction (no degeneration) | `√3` (Theorem B) | `√3` (Theorem B) |
| **degeneration, exact certificate** | **`√2`** (Theorems A and D) | **`√2`** (Theorem D, checked for `n ≤ 16`) |
| needed for the asymptotic restriction | 1 | 1 |

`python -m tensorlab amortize` regenerates the certificates
(`certificates/amortize_n2_square.json`, `amortize_three_copy_square_n8.json`,
`amortize_two_copy_square_n8.json`) and the logs `first_order_obstruction.json`
and `coarse_cyclic_facts.json`.

> **Correction (2026-10-14).**  An earlier version of this note reported, from
> random-start continuation, that two copies fail at `N = 2` for `n = 3, 4, 8`, and
> concluded that Theorem B was optimal among diagonal schemes.  That was wrong:
> Theorem D below is an exact two-copy degeneration for every `n` checked.  The
> random starts found its basin only rarely (1 of 49 copy starts and 2 of 60
> rank-2 starts for `n = 3`; none of 50 copy starts and 80 rank-2 starts for `n = 4`).  Plateaus from
> random starts are weak evidence of non-existence; this note now says so wherever
> it relies on them.

## 1. What is needed quantitatively

Let `c_N` be the least number of copies of `C(3,2n)^{⊗N}` that degenerate to
`T_n^{⊗N}`.  `c_{N+M} ≤ c_N c_M`, and the character inequality holds iff
`c_N^{1/N} → 1`.  Any bound `c_N ≤ c` gives `λ(T_n) ≤ c^{1/N} λ(C(3,2n))`, i.e.
`P(3,2n) ≥ c^{-1/(Nt)} n P(2,2)` for the symmetrised profile.  For `C(3,16)`,
beating `P*(3,16) = 238/9` with `8 P*(2,2) = 80/3` needs a per-factor cost below
`(120/119)^t`, so only costs very close to 1 matter for the exponent.  The results
below are structural progress on the asymptotic question, not an exponent
improvement.  (Asymptotically in `n`, packing a fraction `κ > 14/15` of the
rectangles already suffices; exact constructions give `κ = 2/3`.)

## 2. Why no known invariant can refute it

`T_n` is literally a sub-support of `C(3,2n)` (the retained terms of the block
form); both are free (tight) tensors.  Support functionals — equal to the quantum
functionals on free tensors, the only known universal spectral points — maximise
an entropy over the support, so `ζ^θ(T_n) ≤ ζ^θ(C(3,2n))` for every `θ` and `n`.
A refutation therefore needs a spectral point that is not support-monotone on free
tensors.

Support-monotonicity of *all* spectral points on free tensors would imply
`ω = 2`: the triangle support of `⟨n,n,n⟩` embeds into the support of `ℂ[ℤ_M]`
(`≅ ⟨M⟩`) with `M = O(n²)` via `a_i = i n`, `b_j = j`, `c_k = k(n+1)`.  So no general
"support deletion is asymptotically free" principle is available; a proof must
use the structure of `C(3,2n)`.

Flattening and Koszul methods do not decide it: Koszul excludes every equal-count
construction tested (`C^{⊗2} → T^{⊗2}`, `r` copies to `r` copies for `r ≤ 3`,
catalysts `⟨2⟩`, `B_X(2)`, `R`), but stops excluding as soon as the source has more
copies than the target.

## 3. Framework: diagonal copy cost

Write `S` for the block form of `C(3,2n)` (a restriction of it) and `Φ` for its
support: retained terms (same block, target 1) and cross terms (adjacent blocks,
target 0).  A sum of `c` diagonally rescaled copies of `S^{⊗N}` is `G ∘ S^{⊗N}`
where `G = Σ_r a_r ⊗ b_r ⊗ c_r` (functions of the X-, Y- and Z-words).  So
**`T^{⊗N} ≤ c · S^{⊗N}` by diagonal maps iff some `G` of rank `c` equals
`[retained]` on `Φ^N`** (entries off `Φ^N` are free).  Degenerations are limits
of such sums *on `Φ^N` only*: the copies may diverge off the support, so the limit
need not be the restriction of a border-rank-`c` tensor (combinatorial
degenerations are the simplest example).  Two further reductions:

* *Border rank 2 off the support* is classified: rank `≤ 2` or tangential,
  `a'⊗b⊗c + a⊗b'⊗c + a⊗b⊗c'`, i.e. `w · L` with `w = abc` rank one and `L` additive
  where `a, b, c ≠ 0` — the "first-order mechanism" `(w e^{εL} − w)/ε → wL`.
* *Coarse reduction* (`numeric_restriction.coarse_pattern`,
  `border_construct.check_coarse_lift`): a term's retained/cross status depends
  only on `x`, the block colour `col = (block index) mod 2` and the parities of the
  Y- and Z-indices.  The resulting 3 × 4 × 4 problem with 12 specified entries per
  factor (8 retained, 4 cross) is the same for every `n ≥ 3`; any solution lifts to
  every `n ≥ 2`.  It is the two-block *cyclic* version of the dense packing (§8),
  and it is strictly harder than the real problem (§6, Theorem D).

## 4. Theorem A (n = 2): cost √2

**`T_2^{⊗2}` is a degeneration of two copies of `C(3,4)^{⊗2}`; one copy is
impossible (Koszul).  Hence `λ(T_2) ≤ √2 · λ(C(3,4))`.**

With `j, k` the source Y- and Z-indices of a term and `x = k − j`, the affine
function `ℓ = 4 − j − k` vanishes exactly on the two cross terms and `1/ℓ` is
multiplicatively rank one on the eight retained terms
(`f = (1, 1, 3/4)`, `C = (2/3, 1, 2, 1)`, `A = (1/2, 1, −1/2, −1/3)`).  On the square

    L = ℓ(τ₁) ℓ(τ₂) = 16 − 4(j₁+j₂) − 4(k₁+k₂) + 2 j₁j₂ + 2 k₁k₂ − x₁x₂

is additive, because the mixed terms combine through `k = j + x` into
`k₁k₂ − j₁j₂ − x₁x₂`.  With `w = 1/L`, two copies scaled by
`ρσt(1+εα)(1+εβ)(1+εγ)/ε` and `−ρσt/ε` sum to `wL + O(ε) → [retained]` on all 100
support terms (exact).  Theorem D gives a second, different construction for
`n = 2`.

## 5. Theorem B (every n): three copies, exact restriction, cost √3

**For every `n ≥ 2`, `T_n^{⊗2}` is a restriction (no limit) of three copies of
`C(3,2n)^{⊗2}`, by diagonal maps with entries `0, ±1`.**  For character bounds it is
superseded by Theorem D; it remains the best *restriction* found.

Coordinates per factor: `x ∈ {0,1,2}`, block-form indices `j, k ∈ [0, 2n)`,
block `m(·) = ·//2`, colour `col = m mod 2`, parity `par = · mod 2`,
sign `s(·) = (−1)^{m(·)}`.  The three rank-one terms are

| copy | X-word `(x₁,x₂)` | Y-word `(j₁,j₂)` | Z-word `(k₁,k₂)` |
|---|---|---|---|
| W | `[x₂ ≠ 1]` | `[col(j₁) = par(j₂)]` | `[col(k₁) ≠ par(k₂)]` |
| 1 | `[x₂ ≠ 2]` | `[col(j₁) = 0] s(j₂)` | `[col(k₁) = 0] s(k₂)` |
| 2 | `[x₂ ≠ 0]` | `[col(j₁) = 1] s(j₂)` | `[col(k₁) = 1] s(k₂)` |

*Why it works.*  Adjacent blocks have different colours, so on one factor's
support the retained indicator is `R = A₀ + A₁` with `A_c = [col(j) = c][col(k) = c]`
(rank one each).  Split the second factor by `x`: `R = R₀ + R₁ + R₂`, and let
`K_t` be the cross terms with `x = t` (`K₀`: `j` even, `K₂`: `j` odd).  The sign
`s(j)s(k)` is `+1` on retained and `−1` on cross terms, so copy 1 is
`A₀ ⊗ (R₀ + R₁ − K₀)` and copy 2 is `A₁ ⊗ (R₁ + R₂ − K₂)`.  Copy W *multiplexes*
two products through the hidden colour of factor 1 and the hidden parity of
factor 2 (on `x ≠ 1` terms `par(j) ≠ par(k)`): on the support it equals
`A₀ ⊗ (R₂ + K₀) + A₁ ⊗ (R₀ + K₂)`, and its colour-mixed parts vanish.  The sum is
`(A₀ + A₁) ⊗ R = R ⊗ R`.  Verified exactly for `n = 2,…,12`, independently from the
convolution tensor for `n = 2,3,4,5,8`, with a negative control.

## 6. Theorem D (every n, including C(3,16)): two copies, cost √2

**For every `n` checked (`2 ≤ n ≤ 16`), `T_n^{⊗2}` is a degeneration of two copies
of `C(3,2n)^{⊗2}`.  Hence `λ(T_n) ≤ √2 · λ(C(3,2n))` for every tensor character; for
`C(3,16)` the per-factor cost is `√2`.**

*Construction (the parity-twisted pair).*  With `σ = (1, 0, −1)` at `x = 0, 1, 2`,

    a(x₁,x₂)  = σ(x₁) σ(x₂)                        (zero unless x₁, x₂ ≠ 1)
    β(j₁,j₂) = 1 if j₁ odd, else −(−1)^{j₂}
    γ(k₁,k₂) = −1 if k₁ even, else −(−1)^{k₂}

    copy 0 = (1/ε) (a + ε a₀′) ⊗ (β + ε b₀′) ⊗ (γ + ε c₀′)
    copy 1 = (1/ε) (a + ε a₁′) ⊗ (1 + ε b₁′) ⊗ (1 + ε c₁′)

(block-form indices; the Z-map also shifts `C(3,2n)`'s output index by one).
On terms with `x₁, x₂ ≠ 1` both parities flip (`par(k) ≠ par(j)`), so `βγ = −1` and
the `1/ε` parts cancel.  The `ε⁰` part, `Σᵢ (aᵢ′ bᵢcᵢ + a bᵢ′cᵢ + a bᵢcᵢ′)`, is
*linear* in the corrections, and the exact rational system "`ε⁰` part = target" is
solvable for every `n` checked.  Where some `xₜ = 1` only the `aᵢ′` terms act, and
`βγ` there is exactly the parity sign that the two-copy sign trick needs; where both
`xₜ ≠ 1` the additive corrections do the work, as in Theorem A.  It evades Theorem C
because `a` vanishes on the `x = 1` coordinates and the two copies have different
(parity-twisted) base points.

*Verification.*  `border_construct.parity_twist_pair_certificate(n)` checks the
polynomial identity `ε·(copy₀ + copy₁) = ε·target + O(ε²)` term by term on the
block form for `n = 2,…,12` and `16`; `parity_twist_pair_from_convolution(n)` builds
`C(3,2n)^{⊗2}` from the convolution tensor, applies the two restrictions at
`ε = 1, 2, 3, 4`, recovers the polynomial coefficients by exact interpolation and
compares with `T_n^{⊗2}` (`n = 2,…,12`, so `C(3,16)` is included).  Negative
controls: perturbing one correction breaks the independent check, and the same
ansatz on the cyclic coarse problem has **no** solution
(`parity_twist_pair_coarse_solvable() == False`) — the construction uses the
boundary of the path of blocks.

*How it was found.*  A random-start rank-2 continuation on `n = 3` found a
descending basin in 2 of 60 starts (residual `8.7·10⁻⁹` at norm 324, steady over
nine decades).  Its Y- and Z-scalings were exact `±1` patterns and its X-scaling
`σσ` plus small terms; fixing those and solving for the corrections gave an exact
rational solution, and the patterns simplify (`h = 1`) for every `n`.

*Limits.*  The same two-copy ansatz cannot work at three factors: on an X-slab
where exactly one factor has `x = 1`, the target is a product of two parity
indicators (four sign characters), the two copies supply only two functions there,
and the additive corrections vanish where `a` does.  Going below `√2` needs a
different mechanism (e.g. `c₄ ≤ 3`, cost `3^{1/4} ≈ 1.32`).

## 7. Theorem C: the first-order mechanism stops at the square

**Exact (rational null-space computation, `first_order_obstruction`).**  Let
`V` be the additive functions on `Φ^N` vanishing on all cross terms.  For
`(n, N) = (2, 3), (3, 2), (4, 2), (5, 2), (8, 2), (3, 3)` every `L ∈ V` vanishes
on a fixed positive fraction of the retained terms (336/512, 84/144, 168/256,
276/400, 744/1024, 1400/1728), so no two-copy degeneration of tangential type with
non-vanishing `a, b, c` exists.  In every obstructed case the forced set is exactly
described by a *leg-avoidance rule*: a retained term escapes only if, on some leg,
all `N` of its coordinates avoid the values used by cross terms.  Theorem D shows
that the hypothesis "non-vanishing `a, b, c`" cannot be dropped.

## 8. Calibration: what cost 1 would imply

The coarse problem is the cyclic version of the packing: its support is exactly
that of `Cyc₃ = Σ_{x ∈ {0,1,2}, j ∈ ℤ₄} x_x y_j z_{j+x−1}` (the `ℤ₄` group tensor with
X restricted to the shifts `−1, 0, 1`), and its retained part is exactly `T₂`
(`border_construct.coarse_cyclic_facts`, exact).  `Cyc₃` has rank 4 (discrete
Fourier transform over `ℤ₄`), while `T₂` has border rank 6 (Koszul ≥ 6).  Hence:

* coarse copy cost → 1 would give `T₂ ≲ Cyc₃` and so asymptotic rank
  `R̃(T₂) ≤ 4` — the instance of **Strassen's asymptotic rank conjecture** for the
  tight 3 × 4 × 4 tensor `T₂`, with no slack;
* the problem we actually need has slack: `T_n ≲ C(3,2n)` only forces
  `R̃(T_n) ≤ R(C(3,2n)) = 2n + 2` (for `C(3,16)`: `R̃(T₈) ≤ 18`, against border rank
  `R̲(T₈) = 24`), and the block form `S` keeps part of it (Koszul gives
  `R̲(S) ≥ 2n + 1`, versus `2n` for the cyclic version).

Theorem D confirms that the boundary matters in practice: the parity-twisted pair
works on the path for every `n` but has no solution on the cyclic version, where
numerically two copies at `N = 2` never succeeded (0 of 150 copy starts, 0 of 150
rank-2 starts; all plateau at `≥ 5.28`).

## 9. Numerical evidence (floating point; continuation detector)

Regularised continuation `min |G∘S − T|² + μ|θ|²` with `μ → 0`
(`numeric_restriction.border_path`, `copy_cost_continuation`, `limit_system_search`):
residual `∝ μ²` at bounded norm signals an exact restriction, residual → 0 with
growing norm a border solution, a plateau only that *no start found one*.

| problem | c = 2 | c = 3 | c = 4 | c = 5 |
|---|---|---|---|---|
| `n = 2, N = 2` | border (Thm A, Thm D) | | | |
| `n = 3, N = 2` | border, rare basin (Thm D); other starts plateau 6–7.5 | exact (Thm B) | | |
| `n = 4, 8, N = 2` | exists (Thm D); no random start found it (`n = 4`: 0 of 50 copy and 0 of 80 rank-2 starts) | exact (Thm B) | | |
| coarse, `N = 2` | no start of 300 (plateaus ≥ 5.28); twisted-pair ansatz unsolvable | exact | | |
| `n = 2, N = 3` | plateau 45 | slow descent, inconclusive | | |
| coarse, `N = 3` | | | plateau 20–40 (10 of 10 starts) | border-type solution (3 of 9 starts) |
| `n = 3, N = 3` | | | plateau 12–19 (first starts) | |

*Coarse problem at `N = 3`.*  The five-copy solution descends to residual
`6.9·10⁻⁹` at norm 1418 (residual `∝ norm^{-4}`).  It is a high-order border
solution: every copy's support values scale like `ε^e` with `e` a multiple of
`1/2`; the diverging pair is of order `1/ε` on exactly 32 cross entries of product
form (factor 1: any `x ≠ 1` term) × (factor 2: an `x = 2` retained term) ×
(factor 3: an `x = 2` cross term); per-coordinate exponents are multiples of `1/4`
up to about 6.  The limit-system shapes "three rank-one plus one tangential" and
"one rank-one plus two tangential" both diverge, and a tangential fit of the
remainder leaves 48–86%, so it is not first order.  Not converted to a
certificate; it would give `5^{1/3} ≈ 1.710`, now worse than Theorem D.

*Reproduction* (from `tensor-lab/`, with `OMP_NUM_THREADS=1`; default BLAS
threading made these small solves about 100x slower on a loaded machine):
`scripts/asymptotic/many_starts.py`, `rank2_screen.py`, `deep_full.py`,
`exponents_full.py` (the `n = 3` two-copy basin), `coarse_run.py`, `full_run.py`,
`deep_continuation.py`, `deep_snapshots.py`, `exponent_structure.py`,
`limit_run.py`, `limit_full_run.py`, `minimal_box.py`, `multiplex_search.py`.
Logs: `logs/many_*.log`, `logs/rank2_*.log`, `logs/deepfull_*`, `logs/coarse_N3_c*.log`,
`logs/deep_N3_c*.log`, `logs/limit*_*.log`.

## 10. Status for C(3,16)

| statement | status |
|---|---|
| `λ(T_8) ≤ 2 λ(C(3,16))` | proved (two-copy restriction, notes/05) |
| `λ(T_8) ≤ √3 λ(C(3,16))` via an exact restriction | proved (Theorem B) |
| **`λ(T_8) ≤ √2 λ(C(3,16))`** | **proved** (Theorem D, exact degeneration certificate) |
| `λ(T_8) ≤ λ(C(3,16))` (the asymptotic restriction) | **open**; not refutable by known spectral points |
| two copies at three factors (cost `2^{1/3}`) | impossible for the parity-twisted family (slab count); open in general |
| first-order (tangential, non-vanishing) two-copy amortisation | impossible at `N = 2` (Theorem C) |

**Next step.**  Beat `√2`: find `c₄ ≤ 3` (cost `≈ 1.32`) or `c₃ ≤ 2` by a mechanism
that, unlike the parity-twisted pair, lets the additive corrections act on X-slabs
where some factor has `x = 1` — for instance leading parts `a` supported on more
X-types, with cancellation patterns built from block colours as well as parities.
The search method that found Theorem D transfers: random-start rank-`c`
continuation on a small `n` with many starts, read off the sign patterns, then an
exact linear solve for the corrections.  In parallel, a lower bound for all
diagonal schemes would show where non-diagonal maps become necessary.
