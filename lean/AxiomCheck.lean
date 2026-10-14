import TensorLab.Finite
import TensorLab.Obstruction
import TensorLab.ExponentTransfer
import TensorLab.Extremal
import TensorLab.ThetaGrowth

/-! Prints the axioms used by every theorem; run with `lake env lean AxiomCheck.lean`.
Only `propext`, `Classical.choice`, `Quot.sound` are expected (no `sorryAx`, no `Lean.ofReduceBool`). -/

#print axioms TensorLab.Finite.retained_2_1
#print axioms TensorLab.Finite.weights_2_1
#print axioms TensorLab.Finite.middle_2_1
#print axioms TensorLab.Finite.retained_3_2
#print axioms TensorLab.Finite.weights_3_2
#print axioms TensorLab.Finite.middle_3_2
#print axioms TensorLab.Finite.retained_2_3
#print axioms TensorLab.Finite.weights_2_3
#print axioms TensorLab.Finite.retained_4_1
#print axioms TensorLab.Finite.weights_4_1
#print axioms TensorLab.Finite.middle_4_1
#print axioms TensorLab.Obstruction.chord
#print axioms TensorLab.Obstruction.tripling_chain
#print axioms TensorLab.Obstruction.odd_sector_implied
#print axioms TensorLab.exponent_transfer
#print axioms TensorLab.omega_bound_of_growth
#print axioms TensorLab.Extremal.key_g
#print axioms TensorLab.Extremal.key_f
#print axioms TensorLab.Extremal.P_le_row_line
#print axioms TensorLab.Extremal.P_concave
#print axioms TensorLab.Extremal.P_tripling
#print axioms TensorLab.Extremal.extremal_feasible
#print axioms TensorLab.Extremal.P_diag
#print axioms TensorLab.Extremal.block_bound
#print axioms TensorLab.Extremal.counting_obstruction
#print axioms TensorLab.Extremal.rectangle_exceeds
#print axioms TensorLab.Theta.step
#print axioms TensorLab.Theta.induction
#print axioms TensorLab.Theta.G_lower
#print axioms TensorLab.Theta.theta_growth
#print axioms TensorLab.Theta.paper_case
