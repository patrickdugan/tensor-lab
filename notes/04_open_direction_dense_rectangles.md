# The open direction: dense rectangle packings by non-monomial degenerations

> **Update (same day, `05_c316_dense_rectangles.md`).**  Exact degenerations are now ruled out:
> Koszul flattenings show `C(3,B)` holds `n` rectangles `C(2,2)^σ` iff `B ≥ 3n-1` (the monomial
> staircase is optimal among *all* degenerations), so `C(3,16) ⇝ C(2,2)^σ ⊗ B_X(8)` is impossible;
> the same threshold holds for `3×3` and `4×4` rectangles.  Only the *asymptotic* version, which is
> what the character inequality needs, remains open.  The experiments listed at the end of this
> note are superseded by `05`.

## Where the room is

By the counting obstruction (`03_obstructions.md`, O2), a degeneration of one
`C(a,B)` into a shared-X sum of convolution blocks can beat the extremal profile
`P*` only through *rectangle* blocks: `C(p,q)` with its output leg on the shared
leg (`p + q - 1 ≤ a`).  The flattening-rank count allows `Σ (p_i + q_i) ≤ 2B + a - 1`,
i.e. rectangles may use essentially all coordinates, and a square rectangle with
`p ≈ a/2` is worth `≈ 0.595 H_a` per coordinate against `H_a/2` for `P*`.

A shared-X sum of `n` identical blocks `R` is `R ⊗ B_X(n)`.  So the cleanest form of
the question is

> **Q.**  Does `C(2p-1, np)` degenerate to `C(p,p)^σ ⊗ B_X(n)`, where `σ` moves the
> output leg of `C(p,p)` onto X?  (More generally `C(p+q-1, n·max(p,q))` to
> `C(p,q)^σ ⊗ B_X(n)`.)

As a trilinear form the target is `Σ_{u<p, v<q} x_{u+v} ⟨y_u, z_v⟩`, with
`y_u, z_v ∈ ℂ^n`: polynomial multiplication whose input coefficients are vectors
paired by a dot product.

## Evidence so far

* **Monomial degenerations cannot do it** (exact): one i-class of rectangles must be
  spaced at least `p + q - 1` apart (E1); two interleaved classes and the double
  staircase are obstructed (E2, E3); every packing with value above `P*(a,B)` for
  `a ≤ 5`, `B ≤ 18` is obstructed (E5).  All with verified Farkas certificates.
* **No known spectral point forbids it** (floating point, `logs/spectral.json`):
  for `(a,p,q) ∈ {(3,2,2), (5,3,3), (7,4,4)}` and `n ∈ {2, 8, 16, 32}`, the support
  functionals `ζ^θ` — equal to the Christandl–Vrana–Zuidam quantum functionals on
  these free tensors, hence genuine characters on all tensors — satisfy
  `ζ^θ(C(a, np)) ≥ ζ^θ(C(p,p)^σ ⊗ B_X(n))` at every θ of the 0.1-grid, with equality
  on the faces `θ_Y = 0` or `θ_Z = 0` and an interior margin decaying like `1/n`.
  The flattening-rank characters are likewise satisfied, with asymptotic equality.
* **Payoff if Q has a positive answer for all p, n** ([heuristic], exact recurrence in
  floating point, `growth.whatif_recurrence`): adding `P(2p-1, np) ≥ n P(p,p)` to the
  paper's constraints raises the growth exponent of `P(a,a)` from `4/3` to about
  `1.457`, i.e. `ω ≲ 2.06`.  The first instance that beats `P*` is
  `C(3,16) ⇝ C(2,2)^σ ⊗ B_X(8)` (`n > 7` for `p = 2`); for `p = 3` it is
  `C(5,21) ⇝ C(3,3)^σ ⊗ B_X(7)`.

The asymptotic tightness of every known spectral point on Q is itself informative:
either Q fails for a reason that no current spectral point sees (which would make
Q a natural test case for new obstructions, e.g. Koszul/Young flattenings or
moment-polytope arguments), or Q holds and the convolution-profile method goes
well past 9/4.

## Concrete next experiments

1. Restriction feasibility for small `n` (`n = 2, 3, 4`, `p = 2`): solve
   `C(p,p)^σ ⊗ B_X(n) = (A ⊗ B ⊗ C) · C(2p-1, np)` numerically (Levenberg–Marquardt on
   the cubic system), then rationalise and certify exactly with the existing verifier.
2. Border (degeneration) version: search over `ε`-polynomial maps of low degree,
   reusing `weights_lp.py` for the monomial part and adding a small dense basis change
   on the shared leg X (the obstruction in E1 lives entirely in the X-marginal, so
   basis changes on X are the natural place to look).
3. Obstruction side: Koszul–Young flattenings of the target versus the source at the
   threshold instance `C(3,16)` / `C(2,2)^σ ⊗ B_X(8)`.
