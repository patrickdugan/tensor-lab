import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Tactic

/-!
# Growth from an overhead-`θ` sector inequality

The paper's tripling `P(a, 3h + a - 1) ≥ 3 P(a,h)` packs three copies of row-`a`
blocks with an overhead of `a - 1 = 2 · (1/2) · (a-1)` coordinates.  More generally
a `k`-block construction `P(a, kh + (k-1)θ(a-1)) ≥ k P(a,h)` has *overhead* `θ`;
iterating it along a concave row bounds the row by its limiting slope `g_a`:

    D_a = P(a,a) ≤ (a + θ(a-1)) g_a                                  (hupper)

(for `k = 3` and `2θ(a-1) ∈ ℕ` this is exactly OpenAI's `profile_le_affine_slope`
with `c = 2θ(a-1)`), while symmetry and concavity give, as in the paper,

    D_a + g_a + g_{a+1} ≤ D_{a+1}.                                   (hincrement)

This file proves that these two scalar facts force `D_a ≥ a^{2/(1+θ)}` for
`0 ≤ θ < 1`, generalising Lemma 5.1 (`θ = 1/2` gives `a^{4/3}`).  With the rank
bound and `TensorLab.exponent_transfer` this yields `ω ≤ 3(1+θ)/2`.  So the
exponent is governed by a single number, the overhead `θ`, and any construction
with `θ < 1/2` would improve `9/4`.
-/

namespace TensorLab.Theta

/-- The comparison sequence: `G 1 = 1`, `G (n+1) = G n · (1 + (1-θ)/((1+θ) n))`. -/
noncomputable def G (θ : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => G θ n * (1 + (1 - θ) / ((1 + θ) * n))

theorem G_succ (θ : ℝ) (n : ℕ) : G θ (n + 1) = G θ n * (1 + (1 - θ) / ((1 + θ) * n)) := rfl

theorem G_one (θ : ℝ) : G θ 1 = 1 := by simp [G]

/-- One step of the simultaneous induction (real index `n ≥ 1`). -/
theorem step {θ n Dprev Dnext gprev gnext Gprev : ℝ} (hθ : 0 ≤ θ) (hn : 1 ≤ n)
    (hupper : Dnext ≤ ((n + 1) + θ * n) * gnext)
    (hincrement : Dprev + gprev + gnext ≤ Dnext)
    (hgprev : Gprev ≤ gprev)
    (hDprev : (n + θ * (n - 1)) * Gprev ≤ Dprev) :
    (1 + (1 - θ) / ((1 + θ) * n)) * Gprev ≤ gnext ∧
      ((n + 1) + θ * n) * ((1 + (1 - θ) / ((1 + θ) * n)) * Gprev) ≤ Dnext := by
  have hc : 0 < (1 + θ) * n := by positivity
  have key : (n + θ * n + 1 - θ) * Gprev ≤ (n + θ * n) * gnext := by nlinarith
  have e : (1 + (1 - θ) / ((1 + θ) * n)) * Gprev = ((n + θ * n + 1 - θ) * Gprev) / ((1 + θ) * n) := by
    field_simp
    ring
  have hfirst : (1 + (1 - θ) / ((1 + θ) * n)) * Gprev ≤ gnext := by
    rw [e, div_le_iff₀ hc]
    nlinarith [key]
  refine ⟨hfirst, ?_⟩
  have e2 : ((n + 1) + θ * n) * ((1 + (1 - θ) / ((1 + θ) * n)) * Gprev) =
      (n + θ * n + 1 - θ) * Gprev + (1 + (1 - θ) / ((1 + θ) * n)) * Gprev := by
    field_simp
    ring
  rw [e2]
  nlinarith [hfirst]

/-- The simultaneous induction: `G θ n ≤ g n` and `(n + θ(n-1)) · G θ n ≤ D n`. -/
theorem induction {θ : ℝ} (hθ : 0 ≤ θ) (D g : ℕ → ℝ)
    (hgbase : 1 ≤ g 1) (hDbase : 1 ≤ D 1)
    (hupper : ∀ n : ℕ, 1 ≤ n → D n ≤ ((n : ℝ) + θ * ((n : ℝ) - 1)) * g n)
    (hincrement : ∀ n : ℕ, 1 ≤ n → D n + g n + g (n + 1) ≤ D (n + 1)) :
    ∀ n : ℕ, 1 ≤ n → G θ n ≤ g n ∧ ((n : ℝ) + θ * ((n : ℝ) - 1)) * G θ n ≤ D n := by
  intro n hn
  induction n, hn using Nat.le_induction with
  | base => simp [G_one]; exact ⟨hgbase, hDbase⟩
  | succ n hn ih =>
    have hn' : (1 : ℝ) ≤ n := by exact_mod_cast hn
    have hu := hupper (n + 1) (by omega)
    push_cast at hu
    have hu' : D (n + 1) ≤ (((n : ℝ) + 1) + θ * n) * g (n + 1) := by
      have : ((n : ℝ) + 1) + θ * ((n : ℝ) + 1 - 1) = ((n : ℝ) + 1) + θ * n := by ring
      rw [this] at hu; exact hu
    have hs := step hθ hn' hu' (hincrement n hn) ih.1 ih.2
    rw [G_succ, mul_comm (G θ n)]
    push_cast
    refine ⟨hs.1, ?_⟩
    have : ((n : ℝ) + 1) + θ * ((n : ℝ) + 1 - 1) = ((n : ℝ) + 1) + θ * n := by ring
    rw [this]
    exact hs.2

theorem G_pos {θ : ℝ} (hθ : 0 ≤ θ) (hθ1 : θ < 1) (n : ℕ) : 0 < G θ n := by
  induction n with
  | zero => simp [G]
  | succ n ih =>
    rw [G_succ]
    have : (0 : ℝ) ≤ (1 - θ) / ((1 + θ) * n) := by
      apply div_nonneg <;> [linarith; positivity]
    positivity

/-- `G θ n ≥ n^γ` with `γ = (1-θ)/(1+θ)`, by Bernoulli's inequality
`(1 + γ/n)^{1/γ} ≥ 1 + 1/n` (the paper's `(1 + 1/(3m))^3 ≥ 1 + 1/m` when `θ = 1/2`). -/
theorem G_lower {θ : ℝ} (hθ : 0 ≤ θ) (hθ1 : θ < 1) (n : ℕ) (hn : 1 ≤ n) :
    (n : ℝ) ^ ((1 - θ) / (1 + θ)) ≤ G θ n := by
  set γ := (1 - θ) / (1 + θ) with hγ
  have hγpos : 0 < γ := by rw [hγ]; apply div_pos <;> linarith
  have hγle : γ ≤ 1 := by rw [hγ, div_le_one (by linarith)]; linarith
  -- claim: n ≤ (G θ n)^(1/γ)
  have claim : ∀ m : ℕ, 1 ≤ m → (m : ℝ) ≤ (G θ m) ^ (1 / γ) := by
    intro m hm
    induction m, hm using Nat.le_induction with
    | base => simp [G_one]
    | succ m hm ih =>
      have hm' : (0 : ℝ) < m := by exact_mod_cast hm
      have hGm := G_pos hθ hθ1 m
      have hfac : (1 - θ) / ((1 + θ) * m) = γ / m := by
        rw [hγ]; field_simp
      rw [G_succ, hfac]
      have hs : (-1 : ℝ) ≤ γ / m := by
        have : 0 ≤ γ / m := by positivity
        linarith
      have hp : (1 : ℝ) ≤ 1 / γ := by rw [le_div_iff₀ hγpos]; linarith
      have bern := one_add_mul_self_le_rpow_one_add hs hp
      -- bern : 1 + (1/γ) * (γ/m) ≤ (1 + γ/m)^(1/γ)
      have hsimp : 1 + (1 / γ) * (γ / m) = 1 + 1 / (m : ℝ) := by field_simp
      rw [hsimp] at bern
      have hpos2 : (0 : ℝ) ≤ 1 + γ / m := by positivity
      rw [Real.mul_rpow hGm.le hpos2]
      push_cast
      have h1 : (m : ℝ) * (1 + 1 / (m : ℝ)) = m + 1 := by field_simp
      calc (m : ℝ) + 1 = (m : ℝ) * (1 + 1 / (m : ℝ)) := h1.symm
        _ ≤ (G θ m) ^ (1 / γ) * (1 + γ / m) ^ (1 / γ) := by
          apply mul_le_mul ih bern (by positivity) (by positivity)
  have hc := claim n hn
  have hGn := G_pos hθ hθ1 n
  have hn0 : (0 : ℝ) ≤ n := by positivity
  calc (n : ℝ) ^ γ ≤ ((G θ n) ^ (1 / γ)) ^ γ := Real.rpow_le_rpow hn0 hc hγpos.le
    _ = G θ n := by
      rw [← Real.rpow_mul hGn.le, one_div_mul_cancel (ne_of_gt hγpos), Real.rpow_one]

/-- **θ-growth.**  The two scalar facts (`hupper` with overhead `θ`, `hincrement`) force
`D n ≥ n^{2/(1+θ)}`.  `θ = 1/2` is the paper's `n^{4/3}`. -/
theorem theta_growth {θ : ℝ} (hθ : 0 ≤ θ) (hθ1 : θ < 1) (D g : ℕ → ℝ)
    (hgbase : 1 ≤ g 1) (hDbase : 1 ≤ D 1)
    (hupper : ∀ n : ℕ, 1 ≤ n → D n ≤ ((n : ℝ) + θ * ((n : ℝ) - 1)) * g n)
    (hincrement : ∀ n : ℕ, 1 ≤ n → D n + g n + g (n + 1) ≤ D (n + 1)) :
    ∀ n : ℕ, 1 ≤ n → (n : ℝ) ^ (2 / (1 + θ)) ≤ D n := by
  intro n hn
  have hind := (induction hθ D g hgbase hDbase hupper hincrement n hn).2
  have hG := G_lower hθ hθ1 n hn
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hn0 : (0 : ℝ) < n := by linarith
  have hexp : 2 / (1 + θ) = 1 + (1 - θ) / (1 + θ) := by field_simp; ring
  rw [hexp, Real.rpow_add hn0, Real.rpow_one]
  have hcoef : (n : ℝ) ≤ (n : ℝ) + θ * ((n : ℝ) - 1) := by nlinarith
  have hGpos := G_pos hθ hθ1 n
  calc (n : ℝ) * (n : ℝ) ^ ((1 - θ) / (1 + θ)) ≤ (n : ℝ) * G θ n :=
        mul_le_mul_of_nonneg_left hG hn0.le
    _ ≤ ((n : ℝ) + θ * ((n : ℝ) - 1)) * G θ n := mul_le_mul_of_nonneg_right hcoef hGpos.le
    _ ≤ D n := hind

/-- The paper's case: overhead `1/2` gives exponent `4/3`. -/
theorem paper_case : (2 : ℝ) / (1 + 1 / 2) = 4 / 3 := by norm_num

end TensorLab.Theta
