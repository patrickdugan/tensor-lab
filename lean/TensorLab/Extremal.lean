import Mathlib.Tactic

/-!
# The extremal profile and the counting obstruction

`H m = ∏_{k=1}^{m-1} (1 + 1/(3k))` is the comparison product of the paper's
Lemma 5.1 (OpenAI's `growthProduct`, shifted by one), and

    P*(a,b) = (max(a,b) + (min(a,b) - 1)/2) · H(min(a,b)).

* `extremal_feasible`: `P*` satisfies every hypothesis of the paper's scalar
  growth lemma — positivity, symmetry, `P(1,b) = b`, discrete concavity of rows,
  shifted tripling — in exactly the index conventions of OpenAI's
  `ScalarProfile` (Growth/Profile.lean), apart from the rank bound.  Its diagonal
  is `P*(a,a) = (3a-1)/2 · H a`, which is exactly the lower bound proved in
  `growthProduct_le_slope_and_diagonal`.  Hence the two published inequalities
  determine the diagonal *exactly*: an improvement of the growth exponent `4/3`
  needs a new inequality that `P*` violates.

* `counting_obstruction`: a shared-leg degeneration of `C(a,B)` (`a ≤ B`) into
  convolution blocks `C(x_i, r_i)` whose shared X-leg carries an *input* leg
  (of size `x_i ≤ a`) uses `x_i + 2 r_i - 1` of the `2B + a - 1` coordinates on
  the other two legs; disjointness of the blocks on those legs gives
  `Σ (x_i + 2 r_i - 1) ≤ 2B + a - 1`, and then `Σ P*(x_i, r_i) ≤ P*(a,B)`.  So
  no such degeneration can improve the exponent.  The disjointness is the
  combinatorial content of a shared-leg block decomposition and is taken as a
  hypothesis here.

* `rectangle_exceeds`: the bound fails for blocks whose *output* leg is the
  shared leg ("rectangles"); these are the only convolution blocks that the
  counting argument does not exclude.
-/

namespace TensorLab.Extremal

/-- `H 0 = 1`, `H (n+1) = H n · (1 + 1/(3n))`; in `ℚ`, `1/0 = 0`, so `H 1 = 1`. -/
def H : ℕ → ℚ
  | 0 => 1
  | n + 1 => H n * (1 + 1 / (3 * n))

theorem H_succ (n : ℕ) : H (n + 1) = H n * (1 + 1 / (3 * n)) := rfl

theorem H_one : H 1 = 1 := by simp [H]

theorem H_pos (n : ℕ) : 0 < H n := by
  induction n with
  | zero => simp [H]
  | succ n ih =>
    rw [H_succ]
    have : (0 : ℚ) ≤ 1 / (3 * n) := by positivity
    positivity

/-- The recursion without division, for `n ≥ 1`. -/
theorem H_rec {n : ℕ} (hn : 1 ≤ n) : 3 * (n : ℚ) * H (n + 1) = (3 * n + 1) * H n := by
  rw [H_succ]
  have hn' : (0 : ℚ) < n := by exact_mod_cast hn
  field_simp

theorem H_le_succ (n : ℕ) : H n ≤ H (n + 1) := by
  rw [H_succ]
  have h := H_pos n
  have : (0 : ℚ) ≤ 1 / (3 * n) := by positivity
  nlinarith

theorem H_mono {m n : ℕ} (h : m ≤ n) : H m ≤ H n := by
  induction n, h using Nat.le_induction with
  | base => exact le_rfl
  | succ n _ ih => exact ih.trans (H_le_succ n)

/-- The extremal profile. -/
def P (a b : ℕ) : ℚ :=
  if a ≤ b then ((b : ℚ) + ((a : ℚ) - 1) / 2) * H a else ((a : ℚ) + ((b : ℚ) - 1) / 2) * H b

theorem P_of_le {a b : ℕ} (h : a ≤ b) : P a b = ((b : ℚ) + ((a : ℚ) - 1) / 2) * H a := by
  simp [P, h]

theorem P_of_ge {a b : ℕ} (h : b ≤ a) : P a b = ((a : ℚ) + ((b : ℚ) - 1) / 2) * H b := by
  rcases lt_or_eq_of_le h with hlt | heq
  · simp [P, not_le.mpr hlt]
  · subst heq; simp [P]

theorem P_symm (a b : ℕ) : P a b = P b a := by
  rcases le_total a b with h | h
  · rw [P_of_le h, P_of_ge h]
  · rw [P_of_ge h, P_of_le h]

theorem P_pos {a b : ℕ} (ha : 1 ≤ a) (hb : 1 ≤ b) : 0 < P a b := by
  rcases le_total a b with h | h
  · rw [P_of_le h]
    have : (1 : ℚ) ≤ a := by exact_mod_cast ha
    have : (1 : ℚ) ≤ b := by exact_mod_cast hb
    have := H_pos a
    have : (0 : ℚ) < (b : ℚ) + ((a : ℚ) - 1) / 2 := by linarith
    positivity
  · rw [P_of_ge h]
    have : (1 : ℚ) ≤ a := by exact_mod_cast ha
    have : (1 : ℚ) ≤ b := by exact_mod_cast hb
    have := H_pos b
    have : (0 : ℚ) < (a : ℚ) + ((b : ℚ) - 1) / 2 := by linarith
    positivity

theorem P_boundary (b : ℕ) (hb : 1 ≤ b) : P 1 b = b := by
  rw [P_of_le hb, H_one]; ring

theorem P_diag (a : ℕ) : P a a = (3 * (a : ℚ) - 1) / 2 * H a := by
  rw [P_of_le le_rfl]; ring

/-- `3p · H q ≤ (2p + q) · H p` for `1 ≤ q ≤ p`. -/
theorem key_g {q : ℕ} (hq : 1 ≤ q) {p : ℕ} (hqp : q ≤ p) :
    3 * (p : ℚ) * H q ≤ (2 * p + q) * H p := by
  induction p, hqp using Nat.le_induction with
  | base => nlinarith [H_pos q]
  | succ p hqp ih =>
    have hp : 1 ≤ p := le_trans hq hqp
    have hrec := H_rec hp
    have hHp := H_pos p
    have hHq := H_pos q
    have hpq : (q : ℚ) ≤ p := by exact_mod_cast hqp
    have hp1 : (1 : ℚ) ≤ p := by exact_mod_cast hp
    push_cast
    -- multiply the goal by 3p > 0
    have h3p : (0 : ℚ) < 3 * p := by linarith
    have step1 : 3 * (p : ℚ) * (3 * ((p : ℚ) + 1) * H q) ≤ 3 * ((p : ℚ) + 1) * ((2 * p + q) * H p) := by
      nlinarith [ih]
    have step2 : 3 * ((p : ℚ) + 1) * ((2 * p + q) * H p) ≤ (2 * (p + 1) + q) * ((3 * p + 1) * H p) := by
      have : 3 * ((p : ℚ) + 1) * (2 * p + q) ≤ (2 * (p + 1) + q) * (3 * p + 1) := by nlinarith
      nlinarith
    have goal3 : 3 * (p : ℚ) * (3 * ((p : ℚ) + 1) * H q) ≤ 3 * (p : ℚ) * ((2 * ((p : ℚ) + 1) + q) * H (p + 1)) := by
      calc 3 * (p : ℚ) * (3 * ((p : ℚ) + 1) * H q) ≤ (2 * (p + 1) + q) * ((3 * p + 1) * H p) := step1.trans step2
        _ = (2 * ((p : ℚ) + 1) + q) * (3 * (p : ℚ) * H (p + 1)) := by rw [hrec]
        _ = 3 * (p : ℚ) * ((2 * ((p : ℚ) + 1) + q) * H (p + 1)) := by ring
    exact le_of_mul_le_mul_left goal3 h3p

/-- `(2p + q - 1) · H q ≤ (2q + p - 1) · H p` for `1 ≤ q ≤ p`. -/
theorem key_f {q : ℕ} (hq : 1 ≤ q) {p : ℕ} (hqp : q ≤ p) :
    (2 * (p : ℚ) + q - 1) * H q ≤ (2 * q + p - 1) * H p := by
  induction p, hqp using Nat.le_induction with
  | base => exact le_of_eq (by ring)
  | succ p hqp ih =>
    have hp : 1 ≤ p := le_trans hq hqp
    have hrec := H_rec hp
    have hg := key_g hq hqp
    have hp1 : (1 : ℚ) ≤ p := by exact_mod_cast hp
    push_cast
    have h3p : (0 : ℚ) < 3 * p := by linarith
    have goal3 : 3 * (p : ℚ) * ((2 * ((p : ℚ) + 1) + q - 1) * H q) ≤
        3 * (p : ℚ) * ((2 * q + ((p : ℚ) + 1) - 1) * H (p + 1)) := by
      have e : 3 * (p : ℚ) * ((2 * q + ((p : ℚ) + 1) - 1) * H (p + 1)) =
          (2 * q + (p : ℚ)) * ((3 * p + 1) * H p) := by rw [← hrec]; ring
      rw [e]
      nlinarith [ih, hg]
    exact le_of_mul_le_mul_left goal3 h3p

/-- Every value lies below the linear continuation of its row beyond the diagonal. -/
theorem P_le_row_line {p q : ℕ} (hp : 1 ≤ p) (hq : 1 ≤ q) :
    P p q ≤ ((q : ℚ) + ((p : ℚ) - 1) / 2) * H p := by
  rcases le_total p q with h | h
  · rw [P_of_le h]
  · rw [P_of_ge h]
    have := key_f hq h
    nlinarith

/-! ### Feasibility for the paper's hypotheses -/

theorem incr_upper {a b : ℕ} (h : a ≤ b) : P a (b + 1) - P a b = H a := by
  rw [P_of_le h, P_of_le (by omega : a ≤ b + 1)]; push_cast; ring

theorem incr_lower {a b : ℕ} (hb : 1 ≤ b) (h : b + 1 ≤ a) :
    3 * (b : ℚ) * (P a (b + 1) - P a b) = (2 * b + a) * H b := by
  rw [P_of_ge h, P_of_ge (by omega : b ≤ a)]
  have hrec := H_rec hb
  push_cast
  have e : 3 * (b : ℚ) * (((a : ℚ) + ((b : ℚ) + 1 - 1) / 2) * H (b + 1) - ((a : ℚ) + ((b : ℚ) - 1) / 2) * H b)
      = ((a : ℚ) + (b : ℚ) / 2) * (3 * b * H (b + 1)) - 3 * b * ((a : ℚ) + ((b : ℚ) - 1) / 2) * H b := by ring
  rw [e, hrec]; ring

/-- Increments of every row are antitone on positive indices. -/
theorem incr_antitone {a b : ℕ} (ha : 1 ≤ a) (hb : 1 ≤ b) :
    P a (b + 1 + 1) - P a (b + 1) ≤ P a (b + 1) - P a b := by
  by_cases hab : a ≤ b
  · rw [incr_upper hab, incr_upper (by omega : a ≤ b + 1)]
  · -- b < a
    replace hab : b < a := by omega
    have hb1' : (1 : ℚ) ≤ b := by exact_mod_cast hb
    have hb3 : (0 : ℚ) < 3 * b := by linarith
    have hlow := incr_lower hb (by omega : b + 1 ≤ a)
    rcases Nat.lt_or_ge (b + 1) a with h2 | h2
    · -- both increments in the lower region
      have hlow2 := incr_lower (by omega : 1 ≤ b + 1) (by omega : b + 1 + 1 ≤ a)
      have hrec := H_rec hb
      have hHb := H_pos b
      have hba : (b : ℚ) + 2 ≤ a := by exact_mod_cast h2
      have hb1 : (1 : ℚ) ≤ b := by exact_mod_cast hb
      push_cast at hlow2
      -- compare 3b(b+1) times each increment
      set d1 := P a (b + 1 + 1) - P a (b + 1)
      set d0 := P a (b + 1) - P a b
      have e1 : 3 * (b : ℚ) * (3 * ((b : ℚ) + 1) * d1) = (2 * ((b : ℚ) + 1) + a) * ((3 * b + 1) * H b) := by
        rw [← hrec]; linear_combination (3 * (b : ℚ)) * hlow2
      have e0 : 3 * (b : ℚ) * (3 * ((b : ℚ) + 1) * d0) = 3 * ((b : ℚ) + 1) * ((2 * b + a) * H b) := by
        linear_combination (3 * ((b : ℚ) + 1)) * hlow
      have hc : (2 * ((b : ℚ) + 1) + a) * (3 * b + 1) ≤ 3 * ((b : ℚ) + 1) * (2 * b + a) := by nlinarith
      have : 3 * (b : ℚ) * (3 * ((b : ℚ) + 1) * d1) ≤ 3 * (b : ℚ) * (3 * ((b : ℚ) + 1) * d0) := by
        rw [e1, e0]; nlinarith
      have hpos : (0 : ℚ) < 3 * (b : ℚ) * (3 * ((b : ℚ) + 1)) := by positivity
      nlinarith
    · -- b + 1 = a: the next increment is the constant slope H a = H (b+1)
      have hab1 : a = b + 1 := by omega
      subst hab1
      rw [incr_upper (le_refl (b + 1))]
      have hrec := H_rec hb
      -- 3b · d0 = (3b + 1) H b = 3b H (b+1)
      push_cast at hlow
      have : 3 * (b : ℚ) * (P (b + 1) (b + 1) - P (b + 1) b) = 3 * (b : ℚ) * H (b + 1) := by
        rw [hlow, hrec]; ring
      have := mul_left_cancel₀ (ne_of_gt hb3) this
      linarith

/-- Discrete concavity of every row, in OpenAI's `ScalarProfile.concave` form. -/
theorem P_concave (a h : ℕ) (ha : 1 ≤ a) (hh : 2 ≤ h) :
    P a (h - 1) + P a (h + 1) ≤ 2 * P a h := by
  obtain ⟨b, rfl⟩ : ∃ b, h = b + 1 := ⟨h - 1, by omega⟩
  have := incr_antitone ha (by omega : 1 ≤ b)
  simp only [Nat.add_sub_cancel] at *
  linarith

/-- Shifted tripling, in OpenAI's `ScalarProfile.tripling` form. -/
theorem P_tripling (a h : ℕ) (ha : 1 ≤ a) (hh : 1 ≤ h) :
    3 * P a h ≤ P a (3 * h + a - 1) := by
  have hle : a ≤ 3 * h + a - 1 := by omega
  rw [P_of_le hle]
  have hcast : ((3 * h + a - 1 : ℕ) : ℚ) = 3 * (h : ℚ) + a - 1 := by
    rw [Nat.cast_sub (by omega : 1 ≤ 3 * h + a)]; push_cast; ring
  rw [hcast]
  have := P_le_row_line ha hh
  nlinarith

/-- **Feasibility.**  `P*` satisfies every hypothesis of the paper's scalar growth lemma except
the rank bound, with the same index conventions as OpenAI's `ScalarProfile`. -/
theorem extremal_feasible :
    (∀ a b : ℕ, 1 ≤ a → 1 ≤ b → 0 < P a b) ∧
    (∀ a b : ℕ, 1 ≤ a → 1 ≤ b → P a b = P b a) ∧
    (∀ b : ℕ, 1 ≤ b → P 1 b = (b : ℚ)) ∧
    (∀ a h : ℕ, 1 ≤ a → 2 ≤ h → P a (h - 1) + P a (h + 1) ≤ 2 * P a h) ∧
    (∀ a h : ℕ, 1 ≤ a → 1 ≤ h → 3 * P a h ≤ P a (3 * h + a - 1)) :=
  ⟨fun a b ha hb => P_pos ha hb, fun a b _ _ => P_symm a b, P_boundary,
    P_concave, P_tripling⟩

/-! ### The counting obstruction -/

/-- A block `C(x, r)` whose shared leg carries the input of size `x ≤ a` has value at most
half its number of coordinates on the other two legs, times `H a`. -/
theorem block_bound {a x r : ℕ} (hx : 1 ≤ x) (hxa : x ≤ a) (hr : 1 ≤ r) :
    P x r ≤ ((x : ℚ) + 2 * r - 1) / 2 * H a := by
  have h1 := P_le_row_line hx hr
  have h2 := H_mono hxa
  have hpos : (0 : ℚ) ≤ ((x : ℚ) + 2 * r - 1) / 2 := by
    have : (1 : ℚ) ≤ x := by exact_mod_cast hx
    have : (1 : ℚ) ≤ r := by exact_mod_cast hr
    linarith
  calc P x r ≤ ((r : ℚ) + ((x : ℚ) - 1) / 2) * H x := h1
    _ = ((x : ℚ) + 2 * r - 1) / 2 * H x := by ring
    _ ≤ ((x : ℚ) + 2 * r - 1) / 2 * H a := mul_le_mul_of_nonneg_left h2 hpos

/-- **Counting obstruction.**  If blocks `C(x_i, r_i)` with input-carrying shared legs
(`x_i ≤ a`) use at most the `2B + a - 1` available coordinates of `C(a,B)`, then their
extremal values sum to at most `P*(a,B)`. -/
theorem counting_obstruction {n a B : ℕ} (ha : 1 ≤ a) (haB : a ≤ B)
    (x r : Fin n → ℕ) (hx : ∀ i, 1 ≤ x i) (hxa : ∀ i, x i ≤ a) (hr : ∀ i, 1 ≤ r i)
    (hcount : ∑ i, ((x i : ℚ) + 2 * r i - 1) ≤ 2 * (B : ℚ) + a - 1) :
    ∑ i, P (x i) (r i) ≤ P a B := by
  have hsum : ∑ i, P (x i) (r i) ≤ ∑ i, ((x i : ℚ) + 2 * r i - 1) / 2 * H a :=
    Finset.sum_le_sum fun i _ => block_bound (hx i) (hxa i) (hr i)
  rw [← Finset.sum_mul, ← Finset.sum_div] at hsum
  rw [P_of_le haB]
  have hHa := H_pos a
  calc ∑ i, P (x i) (r i) ≤ (∑ i, ((x i : ℚ) + 2 * r i - 1)) / 2 * H a := hsum
    _ ≤ (2 * (B : ℚ) + a - 1) / 2 * H a := by
        apply mul_le_mul_of_nonneg_right _ hHa.le
        linarith
    _ = ((B : ℚ) + ((a : ℚ) - 1) / 2) * H a := by ring

/-- Rectangles escape the counting bound: a block `C(3,3)` with its output on the shared leg,
inside `C(5,B)`, uses only `3 + 3` coordinates of the other two legs but has `P*(3,3) > 6/2 · H 5`. -/
theorem rectangle_exceeds : (3 + 3 : ℚ) / 2 * H 5 < P 3 3 := by
  simp [P, H]
  norm_num

end TensorLab.Extremal
