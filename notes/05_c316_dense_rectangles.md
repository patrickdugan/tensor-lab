# The dense rectangle packing at C(3,16): exact answer and what remains

> **Update (`06_asymptotic_restriction.md`).**  The copy cost amortises on tensor powers:
> `T_n^{⊗2}` is a degeneration of two copies of `C(3,2n)^{⊗2}` for every `n` checked
> (`n ≤ 16`, so `λ(T_8) ≤ √2 λ(C(3,16))`), and a restriction of three copies for every `n`.  The
> Fourier-filter barrier below concerns block-label filters only; diagonal schemes that also
> read `x` and index parities do amortise.  The asymptotic restriction itself is still open.

**Question.**  Does `C(3,16)` degenerate to eight rectangles `C(2,2)` with their
output leg on the shared X leg, i.e. to `T_8 = C(2,2)^σ ⊗ B_X(8)`?  This is the
smallest instance whose inequality `P(3,16) ≥ 8 P(2,2)` would beat the extremal
profile (`notes/04`).  Command: `python -m tensorlab dense` (certificate
`certificates/dense_p2_n8_B16.json`).

## 1. Exact answer: no — and the monomial staircase is optimal

**Theorem.**  `C(3,B)` degenerates (or restricts) to `n` rectangles `C(2,2)^σ`
sharing X **if and only if** `B ≥ 3n - 1`.  So `C(3,16)` holds at most 5 such
rectangles, not 8.

*If:* the monomial staircase (`sectors.staircase_certificate(3,2,2,n)`, verified
for `n ≤ 8`).

*Only if (proof).*  Use the Koszul flattening along X, `T^{∧1}: X ⊗ Y* → Λ²X ⊗ Z`
(Strassen's equations).

* Source.  With `S_i` = multiplication by `t^i` from `Poly_{<B}` to `Poly_{<B+2}`,
  the map sends `(β_0, β_1, β_2)` to `(β_1 - tβ_0, β_2 - t²β_0, tβ_2 - t²β_1)`.  Its
  kernel is `{(β, tβ, t²β) : deg β < B - 2}`, of dimension `B - 2`, so
  `rank = 3B - (B-2) = 2B + 2`.
* Target.  `T_n = R ⊗ B_X(n)` with `R = C(2,2)^σ`, so the Koszul matrix is
  `K_R ⊗ I_n`; `K_R` is a 6×6 invertible matrix, hence `rank = 6n`.
* Monotonicity.  For linear maps `(α, β, γ)` with `α` surjective,
  `(gT)^{∧1} ∘ (α ⊗ β^T) = (Λ²α ⊗ γ) ∘ T^{∧1}`, so the rank cannot increase; in a
  degeneration onto an X-concise target, `α(ε)` is invertible for generic `ε`, and
  rank is lower semicontinuous in the limit.

Hence a degeneration needs `2B + 2 ≥ 6n`, i.e. `B ≥ 3n - 1`.  For `C(3,16)`:
`34 < 48`.  Exact computations: `rank = 2B+2` for `B ≤ 40`, `6n` for `n ≤ 10`.

Remarks.

* Border rank alone (`R̲(C(3,16)) = 18 < 24 ≤ R̲(T_8)`) already excludes `n = 8`, but
  only gives `B ≥ 3n - 2`; the Koszul comparison is sharper and is tight.
* The case `n = 2`, `B = 4` is the only one border rank does not exclude; orbit
  dimensions do not exclude it either (41 < 56), but Koszul does (10 < 12).  The
  numerical restriction search stalls at residual exactly 1 there (and 2 at `n = 3`),
  consistent with this.
* Larger rectangles behave identically (exact computation):
  `rank C(5,B)^{∧2}_X = 6B + 12` against `30n` for `3×3` rectangles, and
  `rank C(7,B)^{∧3}_X = 20B + 60` against `140n` for `4×4` rectangles; in both cases
  the threshold equals the staircase `B = (n-1)(2p-1) + p`.  Conjecture: the same
  for every `p`, with `rank C(2p-1,B)^{∧(p-1)}_X = binom(2p-2,p-1)(B + p - 1)`.
* Consequence: no exact degeneration of `C(2p-1,B)` into square rectangles can
  violate `P*` (also checked directly for `a ≤ 40, n ≤ 200`).

## 1b. All exact block packings at a = 3 are closed (Koszul LP)

Over a shared-X block sum the Y- and Z-flattening ranks and every X-Koszul rank
are additive, and none can increase under degeneration.  So for any exact
degeneration of `C(a,B)` into convolution blocks (normal, leg-exchanged or
rectangles, any sizes, any mix), `Σ P*(blocks)` is at most the optimum of an LP
whose resources are these ranks (`dense.koszul_lp`, `logs/koszul_lp.json`).

| a | LP optimum per unit B | P* slope `H_a` | closes the question? |
|---|---|---|---|
| 3 | 14/9 | 14/9 | **yes** (also every finite `B ≤ 25`) |
| 4 | 187/108 ≈ 1.7315 | 140/81 ≈ 1.7284 | no, 0.18% gap |
| 5 | ≈ 1.8815 | ≈ 1.8724 | no, 0.49% gap |
| 6 | ≈ 2.0036 | ≈ 1.9973 | no, 0.32% gap |

For `a = 3` (the width of `C(3,16)`) this is a complete answer: **no exact
degeneration of `C(3,B)` into any shared-X packing of convolution blocks violates
`P*`**.  For `a ≥ 4` the relaxation's optimal fractional mixes combine small
normal and exchanged blocks with rectangles; symmetric (Young) flattenings of
degrees 1–3 do not close the gap, and the monomial searches found no packing in
it.  Whether a non-monomial construction realises any of it is open (and a gap of
this size would not obviously move the exponent).

## 2. The cheapest dense construction: two copies

`C(2,2)^σ ⊗ B_X(n) ≤ C(3,2n) ⊕ C(3,2n)` — an exact restriction
(`dense.two_copy_restriction`, verified for `n ≤ 8` and also for `3×3` rectangles).
Put the rectangles at period 2; every leftover term joins neighbouring blocks
(`Δ = ±1`).  The second copy carries the sign `(-1)^m` on block `m` of Y and Z,
and averaging the two copies keeps exactly the `Δ = 0` terms.  Valid over any
field of characteristic ≠ 2.  Equivalently `T_n ≤ C(3,2n) ⊕ 2·B_X(n-1)`, since the
cross terms are two dot products.

So the dense packing costs a factor 2 in copies.  That factor is too expensive:
per character it gives `λ(C(3,2n)) ≥ n^{p_X} λ(R) / 2`, weaker than the staircase.

## 3. What the inequality actually needs, and why it is still open

The growth argument needs `λ(C(3,2n)) ≥ n^{p_X} λ(R)` for every tensor character
`λ`.  By Strassen's spectral theorem this holds **iff `T_n` is an asymptotic
restriction of `C(3,2n)`** (`T_n^{⊗N} ≤ C(3,2n)^{⊗(N+o(N))}`).  Exact
degenerations are only one way to get it; Section 1 rules them out, not the
asymptotic statement.

* Known spectral points allow it: support/quantum functionals with a margin of
  order `1/n` (`logs/spectral.json`); the Y-flattening is exactly tight.
* A necessary condition is `R̰(T_n) ≤ R̰(C(3,2n)) = 2n + 2` (asymptotic rank).
  `R̰(T_n) ≥ 2n` from flattenings, and `T_n ≤ ⟨2,n,2⟩` (project the 2×2 output of a
  `2×n` by `n×2` matrix product onto its symmetric part), so
  `R̰(T_n) ≤ 2^{ω(1, log₂ n, 1)}`.  Flattening-type lower bounds cannot exceed the
  maximal flattening `2n`, so current lower-bound methods cannot refute it.
* **Amortisation barrier for the natural construction** (exact, `dense.filtered_power_experiment`):
  amortising the factor 2 over tensor powers with block-label Fourier filters
  followed by arbitrary monomial weights fails.  A linear filter
  `Σ_t c_t Δ_t ≡ 0 (mod L)` always keeps a pattern together with its negative, and
  every such `±Δ` pair carries a verified Farkas obstruction.  At `N = 2` the only
  working filter uses `L = 4 = 2^N` copies, no better than two copies per factor;
  in general a filter that leaves no nonzero pattern needs all `2^N` subset sums of
  `c` distinct mod `L`, so `L ≥ 2^N`.  At `N = 3` the filter `c = (1,2,4)`, `L = 7`
  leaves only the pair `±(1,1,1)`, which carries a verified two-term obstruction;
  `L = 8 = 2³` is the first filter that works (`logs/dense_filter_N3.log`).

## 4. Status of the C(3,16) instance

| statement | status |
|---|---|
| `C(3,16)` degenerates to 8 rectangles sharing X | **false** (Koszul, proved; border rank, proved) |
| `C(3,16)` degenerates to 5 rectangles sharing X | true (staircase certificate) |
| 8 rectangles ≤ two copies of `C(3,16)` | true (exact restriction) |
| 8 rectangles: `T_8^{⊗2}` ≤ three copies of `C(3,16)^{⊗2}` (cost `√3` per factor) | true (exact restriction, `notes/06`) |
| 8 rectangles: `T_8^{⊗2}` ⊴ two copies of `C(3,16)^{⊗2}` (cost `√2` per factor) | true (exact degeneration, `notes/06`) |
| `P(3,16) ≥ 8 P(2,2)` for every character (asymptotic restriction) | **open**; not refutable by known spectral points or flattening methods; not reachable by Fourier filters + monomial weights |

A positive answer requires a genuinely asymptotic construction that is not an
exact degeneration and not a block-label Fourier filter with monomial weights;
a negative answer requires a spectral point beyond the quantum functionals.
Signed diagonal combinations that read `x` and index parities are not covered by
the filter barrier and do amortise the factor 2, to `√3` by restriction and `√2` by degeneration (`notes/06`).
