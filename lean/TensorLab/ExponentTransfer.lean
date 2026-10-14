import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
import Mathlib.Tactic

/-!
# From diagonal growth to the exponent

The paper closes with `a^{4/3} ≤ P(a,a) ≤ (2a-1)^{1/t}`, hence `t ≤ 3/4` and
`ω ≤ 3t ≤ 9/4`.  The same argument with an arbitrary growth exponent `β`
gives `t ≤ 1/β`, i.e. `ω ≤ 3/β`.  This file proves the scalar step: if a
sequence `D` satisfies `a^β ≤ D a ≤ (2a-1)^{1/t}` for all `a ≥ 1` then `t ≤ 1/β`.

The hypotheses are exactly the paper's normalisation (`P(a,b) ≤ (a+b-1)^{1/t}`,
eq. 4.5) on the diagonal and the assumed growth; everything about tensor
characters stays outside this lemma.
-/

open Filter Topology

namespace TensorLab

theorem exponent_transfer {t β : ℝ} (ht : 0 < t) (hβ : 0 < β) (D : ℕ → ℝ)
    (hlow : ∀ a : ℕ, 1 ≤ a → (a : ℝ) ^ β ≤ D a)
    (hup : ∀ a : ℕ, 1 ≤ a → D a ≤ (2 * (a : ℝ) - 1) ^ (1 / t)) :
    t ≤ 1 / β := by
  by_contra hlt
  push Not at hlt
  -- s = β t > 1
  have hs : 1 < β * t := by
    have := (div_lt_iff₀ hβ).mp hlt  -- 1 < t * β
    linarith
  have hs1 : 0 < β * t - 1 := by linarith
  -- for every a ≥ 1: a^(βt) ≤ 2a, hence a^(βt-1) ≤ 2
  have bound : ∀ a : ℕ, 1 ≤ a → (a : ℝ) ^ (β * t - 1) ≤ 2 := by
    intro a ha
    have ha0 : (0 : ℝ) < a := by exact_mod_cast ha
    have h1 : (a : ℝ) ^ β ≤ (2 * (a : ℝ) - 1) ^ (1 / t) := (hlow a ha).trans (hup a ha)
    have h2 : (2 * (a : ℝ) - 1) ^ (1 / t) ≤ (2 * (a : ℝ)) ^ (1 / t) := by
      apply Real.rpow_le_rpow
      · have : (1 : ℝ) ≤ a := by exact_mod_cast ha
        linarith
      · linarith
      · positivity
    have h3 : (a : ℝ) ^ β ≤ (2 * (a : ℝ)) ^ (1 / t) := h1.trans h2
    have h4 : ((a : ℝ) ^ β) ^ t ≤ ((2 * (a : ℝ)) ^ (1 / t)) ^ t :=
      Real.rpow_le_rpow (by positivity) h3 ht.le
    rw [← Real.rpow_mul ha0.le, ← Real.rpow_mul (by positivity), one_div_mul_cancel ht.ne',
      Real.rpow_one] at h4
    -- h4 : a^(β t) ≤ 2 a
    have h5 : (a : ℝ) ^ (β * t - 1) = (a : ℝ) ^ (β * t) / a := Real.rpow_sub_one ha0.ne' _
    rw [h5, div_le_iff₀ ha0]
    linarith
  -- but a^(βt-1) → ∞
  have hlim : Tendsto (fun a : ℕ => (a : ℝ) ^ (β * t - 1)) atTop atTop :=
    (tendsto_rpow_atTop hs1).comp tendsto_natCast_atTop_atTop
  obtain ⟨a, ha2, ha1⟩ : ∃ a : ℕ, 2 < (a : ℝ) ^ (β * t - 1) ∧ 1 ≤ a := by
    have ev := (hlim.eventually (eventually_gt_atTop (2 : ℝ))).and (eventually_ge_atTop 1)
    exact ev.exists
  exact absurd (bound a ha1) (not_le.mpr ha2)

/-- The exponent consequence in the paper's normalisation: `ω ≤ 3t ≤ 3/β`. -/
theorem omega_bound_of_growth {t β ω : ℝ} (ht : 0 < t) (hβ : 0 < β) (D : ℕ → ℝ)
    (hlow : ∀ a : ℕ, 1 ≤ a → (a : ℝ) ^ β ≤ D a)
    (hup : ∀ a : ℕ, 1 ≤ a → D a ≤ (2 * (a : ℝ) - 1) ^ (1 / t))
    (hω : ω ≤ 3 * t) :
    ω ≤ 3 / β := by
  have := exponent_transfer ht hβ D hlow hup
  calc ω ≤ 3 * t := hω
    _ ≤ 3 * (1 / β) := by gcongr
    _ = 3 / β := by ring

end TensorLab
