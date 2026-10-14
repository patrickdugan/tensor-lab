# The extremal profile, exactness of 4/3, and the overhead parameter θ

## The extremal profile

Let `H_m = ∏_{k=1}^{m-1} (1 + 1/(3k))` (the paper's comparison product) and

    P*(a,b) = (max(a,b) + (min(a,b) - 1)/2) · H_{min(a,b)}.

For `b ≥ a` each row is affine, `P*(a,b) = (b + (a-1)/2) H_a`, with slope `H_a`.

**Theorem (Lean: `TensorLab.Extremal.extremal_feasible`).**  `P*` is positive,
symmetric, has `P*(1,b) = b`, satisfies discrete concavity
`P(a,h-1) + P(a,h+1) ≤ 2P(a,h)` (`a ≥ 1, h ≥ 2`) and shifted tripling
`3P(a,h) ≤ P(a,3h+a-1)` (`a, h ≥ 1`) — every hypothesis of OpenAI's
`ScalarProfile` except the rank bound, in the same index conventions.

Its diagonal is `P*(a,a) = (3a-1)/2 · H_a` (`Extremal.P_diag`), which is exactly
the lower bound that OpenAI's `growthProduct_le_slope_and_diagonal` proves for
every such profile.  Therefore:

> The two published inequalities determine the diagonal exactly.  No argument
> that uses only (S), (B), (C), (T) can give more than `P(a,a) ≥ (3a-1)/2 · H_a ≍ a^{4/3}`.
> A better growth exponent needs a new valid inequality that `P*` violates.

This gives the search engine a one-line acceptance test: a candidate inequality is
*potentially useful* only if it has negative slack on `P*`
(`growth.extremal_profile`, `Candidate.slack`).  The exact LP over finite grids
(`growth.lp_minimum`) confirms it: the LP minimum of `P(a,a)` subject to the paper's
constraints on `[1,12]²` equals `P*(a,a)` for `a = 2, 3, 4`
(`10/3`, `56/9`, `770/81`) and is slightly below it for `a = 5` (`7388/567 < 13.107…`)
only because constraints with indices above 12 are missing from the finite grid.

Exact checks on `[1,60]²`: no constraint violated; 1769 concavity and 170 tripling
instances are tight (`logs/growth_grid60.json`).

## The overhead parameter θ

The tripling inequality packs `k = 3` row-`a` blocks into row `a` with an
*overhead* of `a - 1 = (k-1) · θ · (a-1)` extra coordinates, `θ = 1/2`.  For a
general `k`-block construction

    P(a, kh + (k-1)θ(a-1)) ≥ k P(a,h),

iterating along the concave row gives `P(a,h) ≤ (h + θ(a-1)) g_a` with `g_a` the
limiting row slope (for `k = 3` and `2θ(a-1) ∈ ℕ` this is OpenAI's
`profile_le_affine_slope` with `c = 2θ(a-1)`), so `D_a = P(a,a) ≤ (a + θ(a-1)) g_a`,
while symmetry and concavity give `D_a + g_a + g_{a+1} ≤ D_{a+1}` as in the paper.

**Theorem (Lean: `TensorLab.Theta.theta_growth`).**  For `0 ≤ θ < 1`, these two
scalar facts (with `g_1 ≥ 1`, `D_1 ≥ 1`) imply `D_a ≥ a^{2/(1+θ)}` for all `a ≥ 1`.

Proof: the simultaneous induction of Lemma 5.1 with
`G_{a+1} = G_a (1 + γ/a)`, `γ = (1-θ)/(1+θ)`, and Bernoulli's inequality
`(1 + γ/m)^{1/γ} ≥ 1 + 1/m` (for `θ = 1/2` this is the paper's `(1+1/(3m))^3 ≥ 1+1/m`).

With the rank bound `P(a,a) ≤ (2a-1)^{1/t}` and `TensorLab.exponent_transfer`:

    ω ≤ 3t ≤ 3(1+θ)/2.

| θ | growth exponent `2/(1+θ)` | exponent bound `3(1+θ)/2` |
|---|---|---|
| 1/2 (paper) | 4/3 | 9/4 |
| 2/5 | 10/7 | 2.1 |
| 1/3 | 3/2 | 2 |

So the whole question "can the inequality be strengthened?" reduces, inside this
framework, to "can row-`a` blocks be packed into a single row with overhead
`θ < 1/2`?".  `notes/03_obstructions.md` shows that for shared-leg monomial
degenerations of a single `C(a,B)` into convolution blocks the answer is no:
coordinate counting forces `θ ≥ 1/2` exactly.
