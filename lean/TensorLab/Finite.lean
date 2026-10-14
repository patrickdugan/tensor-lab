import Mathlib.Tactic

/-!
# Finite certified instances of the three-sector degeneration

`conv a b` is the coefficient tensor of `C(a,b) = Σ x_i y_j z_{i+j}` as an
integer-valued function on finite coordinate types.  The weights follow the
paper (Section 4.2) and OpenAI's `Sector/Weights.lean`: `+1` on the middle
Y-sector `[h, 2h+a-1)`, `-1` on the middle Z-sector `[h+a-1, 2h+a-1)`, zero
elsewhere.  `retained a h` keeps the weight-zero coefficients.

The three branches are given by the paper's explicit coordinate
parametrisations (left: `x_u y_r z_{u+r}`; middle: `x_{a-1-u} y_{h+u+r} z_{h+a-1+r}`;
right: `x_u y_{s+r} z_{s+u+r}` with `s = 2h+a-1`), so the equalities below
certify, by exhaustive evaluation (`decide`), the finite identities that the
Python certificates `certificates/three_sector_a*_h*.json` record.

These theorems are *finite* statements for the listed parameters only.  The
statement for all `a, h` is OpenAI's `Sector.retained_eq_three_branches`.
-/

namespace TensorLab.Finite

/-- Coefficient of `x_i y_j z_k` in `C(a,b)`. -/
def conv (a b : ℕ) (i : Fin a) (j : Fin b) (k : Fin (a + b - 1)) : ℤ :=
  if i.val + j.val = k.val then 1 else 0

def sourceWidth (a h : ℕ) : ℕ := 3 * h + a - 1
def rightStart (a h : ℕ) : ℕ := 2 * h + a - 1

/-- Weight on the second leg. -/
def wY (a h j : ℕ) : ℤ := if h ≤ j ∧ j < rightStart a h then 1 else 0
/-- Weight on the third leg. -/
def wZ (a h k : ℕ) : ℤ := if h + a - 1 ≤ k ∧ k < rightStart a h then -1 else 0
/-- Total weight of the term `x_i y_j z_k` (the first leg carries weight 0). -/
def weight (a h : ℕ) (_i j k : ℕ) : ℤ := wY a h j + wZ a h k

abbrev Src (a h : ℕ) :=
  Fin a → Fin (sourceWidth a h) → Fin (a + sourceWidth a h - 1) → ℤ

def source (a h : ℕ) : Src a h := fun i j k => conv a (sourceWidth a h) i j k

/-- The minimum-weight (weight-zero) part of the source. -/
def retained (a h : ℕ) : Src a h :=
  fun i j k => if weight a h i.val j.val k.val = 0 then source a h i j k else 0

/-- Left branch: the translate-free copy of `C(a,h)`. -/
def leftBranch (a h : ℕ) : Src a h :=
  fun i j k => if ∃ u, u < a ∧ ∃ r, r < h ∧ i.val = u ∧ j.val = r ∧ k.val = u + r then 1 else 0

/-- Middle branch: `C(a,h)` with the first coordinate reversed and legs Y, Z exchanged. -/
def middleBranch (a h : ℕ) : Src a h :=
  fun i j k => if ∃ u, u < a ∧ ∃ r, r < h ∧ i.val = a - 1 - u ∧ j.val = h + u + r ∧
      k.val = h + a - 1 + r then 1 else 0

/-- Right branch: the copy of `C(a,h)` translated by `2h+a-1` on both outer legs. -/
def rightBranch (a h : ℕ) : Src a h :=
  fun i j k => if ∃ u, u < a ∧ ∃ r, r < h ∧ i.val = u ∧ j.val = rightStart a h + r ∧
      k.val = rightStart a h + u + r then 1 else 0

/-- Every supported term has weight 0 or 1 (no negative total weights). -/
def WeightsZeroOne (a h : ℕ) : Prop :=
  ∀ i j k, source a h i j k ≠ 0 → weight a h i.val j.val k.val = 0 ∨ weight a h i.val j.val k.val = 1

instance (a h : ℕ) : Decidable (WeightsZeroOne a h) := by
  unfold WeightsZeroOne
  infer_instance

/-- Pullback of the middle branch along the paper's maps is `C(a,h)` with legs exchanged:
`middleBranch (a-1-u) (h+s) (h+a-1+r) = conv a h u r s`, stated on coordinates. -/
def MiddleIsExchanged (a h : ℕ) : Prop :=
  ∀ (u : Fin a) (r : Fin h) (s : Fin (a + h - 1)),
    ∀ (i : Fin a) (j : Fin (sourceWidth a h)) (k : Fin (a + sourceWidth a h - 1)),
      i.val = a - 1 - u.val → j.val = h + s.val → k.val = h + a - 1 + r.val →
        middleBranch a h i j k = conv a h u r s

instance (a h : ℕ) : Decidable (MiddleIsExchanged a h) := by
  unfold MiddleIsExchanged
  infer_instance

/-! ### `(a, h) = (2, 1)`: the paper's Figure 1, `C(2,4) ⇝ 3 × C(2,1)`. -/

theorem retained_2_1 : retained 2 1 = leftBranch 2 1 + middleBranch 2 1 + rightBranch 2 1 := by
  decide

theorem weights_2_1 : WeightsZeroOne 2 1 := by decide

theorem middle_2_1 : MiddleIsExchanged 2 1 := by decide

/-! ### `(a, h) = (3, 2)`: `C(3,8) ⇝ 3 × C(3,2)`. -/

theorem retained_3_2 : retained 3 2 = leftBranch 3 2 + middleBranch 3 2 + rightBranch 3 2 := by
  decide

theorem weights_3_2 : WeightsZeroOne 3 2 := by decide

theorem middle_3_2 : MiddleIsExchanged 3 2 := by decide

/-! ### `(a, h) = (2, 3)`: `C(2,10) ⇝ 3 × C(2,3)`. -/

theorem retained_2_3 : retained 2 3 = leftBranch 2 3 + middleBranch 2 3 + rightBranch 2 3 := by
  decide

theorem weights_2_3 : WeightsZeroOne 2 3 := by decide

/-! ### `(a, h) = (4, 1)`: `C(4,7) ⇝ 3 × C(4,1)`. -/

theorem retained_4_1 : retained 4 1 = leftBranch 4 1 + middleBranch 4 1 + rightBranch 4 1 := by
  decide

theorem weights_4_1 : WeightsZeroOne 4 1 := by decide

theorem middle_4_1 : MiddleIsExchanged 4 1 := by decide

end TensorLab.Finite
