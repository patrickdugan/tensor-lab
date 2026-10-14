import Mathlib.Tactic

/-!
# The odd-sector family is implied by concavity and shifted tripling

The search engine finds, for every `e ≥ 1`, the `(2e+1)`-sector degeneration
`C(a, (2e+1)h + e(a-1)) ⇝ (2e+1) × C(a,h)` (alternating normal and exchanged
blocks).  The corresponding profile inequality

    (2e+1) · P(a,h) ≤ P(a, (2e+1)h + e(a-1))

is however a consequence of the two inequalities already in the paper:
discrete concavity of each row `P(a, ·)` (Lemma 4.1) and shifted tripling
(Lemma 4.2).  So this family cannot improve the growth exponent `4/3`.

The proof: the tripling chain gives `3^j P(a,h) ≤ P(a, h + Λ_j c)` with
`c = 2h + a - 1` and `Λ_j = (3^j - 1)/2`; the point `h + e c` lies between
`h` and `h + Λ_e c`, and the chord inequality of a discretely concave sequence
interpolates.
-/

namespace TensorLab.Obstruction

/-- Increments of a sequence that is discretely concave on indices `≥ 1` are antitone. -/
theorem increment_le {u : ℕ → ℝ}
    (hc : ∀ n, 1 ≤ n → u (n + 1 + 1) + u n ≤ 2 * u (n + 1))
    {m n : ℕ} (hm : 1 ≤ m) (hmn : m ≤ n) :
    u (n + 1) - u n ≤ u (m + 1) - u m := by
  induction n, hmn using Nat.le_induction with
  | base => exact le_rfl
  | succ n hmn ih =>
    have h := hc n (le_trans hm hmn)
    linarith

/-- Tangent bound at the left endpoint. -/
theorem le_tangent {u : ℕ → ℝ}
    (hc : ∀ n, 1 ≤ n → u (n + 1 + 1) + u n ≤ 2 * u (n + 1))
    {x : ℕ} (hx : 1 ≤ x) (d : ℕ) :
    u (x + d) ≤ u x + (d : ℝ) * (u (x + 1) - u x) := by
  induction d with
  | zero => simp
  | succ d ih =>
    have e : x + (d + 1) = x + d + 1 := by omega
    rw [e]
    have h := increment_le hc hx (Nat.le_add_right x d)
    push_cast
    linarith

/-- Bound at the right endpoint: the increments before `x + d` dominate the one after. -/
theorem tangent_le {u : ℕ → ℝ}
    (hc : ∀ n, 1 ≤ n → u (n + 1 + 1) + u n ≤ 2 * u (n + 1))
    {x : ℕ} (hx : 1 ≤ x) (d : ℕ) :
    (d : ℝ) * (u (x + d + 1) - u (x + d)) ≤ u (x + d) - u x := by
  induction d with
  | zero => simp
  | succ d ih =>
    have e : x + (d + 1) = x + d + 1 := by omega
    rw [e]
    have h := increment_le hc (show 1 ≤ x + d by omega) (Nat.le_succ (x + d))
    have h2 := mul_le_mul_of_nonneg_left h (show (0 : ℝ) ≤ (d : ℝ) + 1 by positivity)
    push_cast
    nlinarith [h2, ih]

/-- Chord inequality: for `x ≤ x + p ≤ x + p + q`,
`p · (u (x+p+q) - u (x+p)) ≤ q · (u (x+p) - u x)`. -/
theorem chord {u : ℕ → ℝ}
    (hc : ∀ n, 1 ≤ n → u (n + 1 + 1) + u n ≤ 2 * u (n + 1))
    {x : ℕ} (hx : 1 ≤ x) (p q : ℕ) :
    (p : ℝ) * (u (x + p + q) - u (x + p)) ≤ (q : ℝ) * (u (x + p) - u x) := by
  have h1 := le_tangent hc (show 1 ≤ x + p by omega) q
  have h2 := tangent_le hc hx p
  have hp : (0 : ℝ) ≤ p := by positivity
  have hq : (0 : ℝ) ≤ q := by positivity
  have h1' := mul_le_mul_of_nonneg_left h1 hp
  have h2' := mul_le_mul_of_nonneg_left h2 hq
  nlinarith [h1', h2']

/-- `Λ_j = (3^j - 1)/2`, defined by the recursion `Λ_{j+1} = 3 Λ_j + 1`. -/
def lam : ℕ → ℕ
  | 0 => 0
  | j + 1 => 3 * lam j + 1

theorem lam_ge (j : ℕ) : j ≤ lam j := by
  induction j with
  | zero => simp [lam]
  | succ j ih => simp only [lam]; omega

theorem two_lam_add_one (j : ℕ) : 2 * lam j + 1 = 3 ^ j := by
  induction j with
  | zero => simp [lam]
  | succ j ih => simp only [lam, pow_succ]; omega

/-- Iterated shifted tripling along the row `P a`, written with `a = a' + 1`
to avoid truncated subtraction: `3^j P(a,h) ≤ P(a, h + Λ_j (2h + a'))`. -/
theorem tripling_chain {P : ℕ → ℕ → ℝ} {a' h : ℕ} (hh : 1 ≤ h)
    (htrip : ∀ k, 1 ≤ k → 3 * P (a' + 1) k ≤ P (a' + 1) (3 * k + a')) (j : ℕ) :
    (3 : ℝ) ^ j * P (a' + 1) h ≤ P (a' + 1) (h + lam j * (2 * h + a')) := by
  induction j with
  | zero => simp [lam]
  | succ j ih =>
    have hk : 1 ≤ h + lam j * (2 * h + a') := by omega
    have step := htrip _ hk
    have idx : 3 * (h + lam j * (2 * h + a')) + a' = h + lam (j + 1) * (2 * h + a') := by
      simp only [lam]; ring
    rw [idx] at step
    rw [pow_succ]
    nlinarith [ih, step]

/-- **Obstruction.**  Row concavity and shifted tripling imply the whole odd-sector family
`(2e+1) P(a,h) ≤ P(a, (2e+1)h + e(a-1))` for every `e`. -/
theorem odd_sector_implied {P : ℕ → ℕ → ℝ} {a h : ℕ} (ha : 1 ≤ a) (hh : 1 ≤ h)
    (hconc : ∀ b, 1 ≤ b → P a (b + 1 + 1) + P a b ≤ 2 * P a (b + 1))
    (htrip : ∀ k, 1 ≤ k → 3 * P a k ≤ P a (3 * k + a - 1))
    (e : ℕ) :
    ((2 * e + 1 : ℕ) : ℝ) * P a h ≤ P a ((2 * e + 1) * h + e * (a - 1)) := by
  obtain ⟨a', rfl⟩ : ∃ a', a = a' + 1 := ⟨a - 1, by omega⟩
  have htrip' : ∀ k, 1 ≤ k → 3 * P (a' + 1) k ≤ P (a' + 1) (3 * k + a') := by
    intro k hk
    have := htrip k hk
    have e1 : 3 * k + (a' + 1) - 1 = 3 * k + a' := by omega
    rwa [e1] at this
  -- c = 2h + a', the chord between x = h and x + p + q = h + Λ_e c, with x + p = h + e c
  set c := 2 * h + a' with hc_def
  have hΛ := lam_ge e
  have chain := tripling_chain hh htrip' e
  have idx : (2 * e + 1) * h + e * (a' + 1 - 1) = h + e * c := by
    simp only [hc_def, Nat.add_sub_cancel]; ring
  rw [idx]
  -- the chord inequality with p = e c and q = (Λ_e - e) c
  have hch := chord (u := fun b => P (a' + 1) b) hconc hh (e * c) ((lam e - e) * c)
  have idx2 : h + e * c + (lam e - e) * c = h + lam e * c := by
    have : (lam e - e) * c = lam e * c - e * c := Nat.sub_mul _ _ _
    rw [this]
    have : e * c ≤ lam e * c := Nat.mul_le_mul_right c hΛ
    omega
  rw [idx2] at hch
  have hsub : (((lam e - e) * c : ℕ) : ℝ) = ((lam e : ℝ) - e) * c := by
    rw [Nat.cast_mul, Nat.cast_sub hΛ]
  rw [hsub] at hch
  push_cast at hch chain ⊢
  have h3 : (3 : ℝ) ^ e = 2 * (lam e : ℝ) + 1 := by
    have := two_lam_add_one e
    exact_mod_cast this.symm
  rw [h3] at chain
  -- Let X = P(a,h), M = P(a, h+ec), Y = P(a, h+Λc).  From the chord,
  -- (Λ c) M ≥ (e c) Y + ((Λ-e) c) X, and Y ≥ (2Λ+1) X.
  by_cases hce : (e : ℝ) * c = 0
  · -- then e = 0 (since c ≥ 1), and the claim is trivial
    have he : (e : ℝ) = 0 := by
      have hcpos : (0 : ℝ) < c := by
        have : 1 ≤ c := by omega
        exact_mod_cast this
      rcases mul_eq_zero.mp hce with h0 | h0
      · exact h0
      · exact absurd h0 (ne_of_gt hcpos)
    have he' : e = 0 := by exact_mod_cast he
    subst he'
    simp
  · have hcpos : (0 : ℝ) < c := by
      have : 1 ≤ c := by omega
      exact_mod_cast this
    have hepos : (0 : ℝ) < e := by
      rcases Nat.eq_zero_or_pos e with h0 | h0
      · subst h0; simp at hce
      · exact_mod_cast h0
    have hΛr : (e : ℝ) ≤ lam e := by exact_mod_cast hΛ
    have hΛpos : (0 : ℝ) < lam e := lt_of_lt_of_le hepos hΛr
    -- multiply out: Λ c M ≥ e c Y + (Λ - e) c X ≥ e c (2Λ+1) X + (Λ - e) c X = Λ c (2e+1) X
    have key : (lam e : ℝ) * c * P (a' + 1) (h + e * c) ≥ (lam e : ℝ) * c * ((2 * e + 1) * P (a' + 1) h) := by
      nlinarith [hch, chain, mul_pos hepos hcpos, mul_pos hΛpos hcpos, mul_nonneg hepos.le hcpos.le]
    have hpos : (0 : ℝ) < (lam e : ℝ) * c := mul_pos hΛpos hcpos
    exact le_of_mul_le_mul_left key hpos

end TensorLab.Obstruction
