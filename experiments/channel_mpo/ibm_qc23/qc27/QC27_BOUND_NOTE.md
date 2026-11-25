# QC-27: tighter truncation bounds for Pauli-propagation (sparse Heisenberg) with exact dyadic form

No hardware, no IBM access, no network. Synthetic circuits only. Code: `qc27_bounds.py` (float propagator, ledger, bounds, dense references), `qc27_exact.py` (Fraction pipeline, witness prototype, counterexamples), `qc27_benchmark.py` (sweeps, JSON), `test_qc27_bounds.py`. Raw data: `qc27_bound_benchmark.json`. Every number in the Results, Exact-case and Failed-hypotheses sections is rendered from that JSON.

## 0. Summary

Setting (QC-25/QC-12): a Pauli observable is evolved in the Heisenberg picture through gate-local CPTP channels; each step discards small Pauli-coefficient mass. The existing bound is `|Delta<O>| <= 2^(n/2) sum_j eta_j` (eta_j = normalised Hilbert-Schmidt norm of what step j discards, n = register width).

Results of this note:

* **Theorem 1 (support at discard).** `||D||_op <= sqrt(2^k - 1) * eta` for the discarded operator D with k = |support of D| at the time of discard (not n, not the final support); the factor is attained, and later steps cannot increase it.
* **Theorem 2 (l1/l2/grouped).** `||D||_op <= sum_g min(l1_g, sqrt(2^{k_g}-1) eta_g)` for any partition of the discarded strings into groups g with supports S_g (k_g=|S_g|); grouping by exact support is the default.
* **Theorem 3 (noise survival).** If every string of D is non-identity on each qubit of a set Sigma, then `||F(D)||_op <= sigma ||D||_op` where sigma is a product of `(1 - prod kappa_q)` factors computed by a block-tracking pass over the remaining circuit; kappa_q is the depolarising content of the channel on qubit q (kappa = 2 lambda_min(Choi)). Non-unital channels give no contraction beyond their depolarising content and destroy the blocks; this is stated precisely in section 4.
* **Theorem 4 (exact form).** For Clifford gates and single-qubit channels with dyadic Heisenberg parameters, certified completely positive by exact LDL^T of the Choi matrix, all coefficients and ledger entries are dyadic rationals and the combined bound is a rational number that can be recomputed with `fractions.Fraction`.
* Measured on synthetic brickwork circuits: median old/new bound ratio (new = survival-weighted) is n=12: 6.82; n=16: 24.6; n=24: 463; n=32: 6.13e+03; n=48: 2.05e+06. All 3372 validity checks against dense references passed (562 runs with discards (n=8: 349, n=12: 117, n=16: 96), 6 bound variants each; all passed: True); the new bounds are never above the old one (ordering old >= cone >= cone_tl >= min(l1,l2) >= grouped >= surv held in 562/562 validity runs and 270/270 benchmark runs).
* Failed hypotheses (section 8), including four tempting survival rules that are false, are recorded with exact counterexamples.

## 1. Setting and the error identity

Forward channels `E_1,...,E_L` (each CPTP, acting on a small block, identity elsewhere) and Heisenberg maps `Phi_j := E_j^dagger` (adjoint in the Hilbert-Schmidt pairing). `Phi_j` is completely positive and **unital** (`Phi_j(I) = I` because `E_j` is trace preserving), including when `E_j` itself is non-unital. State `rho_0`, `rho_j := E_j(rho_{j-1})`. Heisenberg recursion with truncation:

```
O_L := O,   X_j := Phi_j(O_j),   O_{j-1} := X_j - D_j        (D_j Hermitian = what step j discards)
```

Estimate `mu_tilde = Tr(rho_0 O_0)`; truth `mu = Tr(rho_L O) = Tr(rho_0 Phi_1 ... Phi_L (O))`. In the code a *discard event* happens after a block of consecutive backward ops (forward indices `pos .. end-1`); `rho_pos` is the state after `ops[0:pos]`.

**Lemma 0 (error identity).** `mu - mu_tilde = sum_j Tr(rho_{j-1} D_j)`.

*Proof.* Let `F_j := Phi_1 o ... o Phi_j`, `F_0 = id`. Then `F_j(O_j) - F_{j-1}(O_{j-1}) = F_{j-1}(Phi_j(O_j) - O_{j-1}) = F_{j-1}(D_j)`. Summing over j gives `F_L(O) - F_0(O_0) = sum_j F_{j-1}(D_j)`. Apply `Tr(rho_0 .)` and use `Tr(rho_0 F_{j-1}(X)) = Tr(rho_{j-1} X)`. QED

(`test_per_event_and_group_bounds_n8` checks this identity against a density-matrix reference: max residual 3.55e-15.)

Facts used throughout. (N1) A unital positive map `Phi` is an operator-norm contraction on Hermitian `A`: `-||A|| I <= A <= ||A|| I` implies the same sandwich for `Phi(A)`. So `||F_{j-1}(D_j)||_op <= ||D_j||_op` for every channel mix, unital or not, whatever the later support growth. (N2) `|Tr(rho X)| <= ||X||_op` for any state. (N3) `||D||_op^2 <= tr D^2 = 2^n eta^2`, where `eta^2 := tr(D^2)/2^n = sum_P d_P^2` for `D = sum d_P P`. Hence

```
|mu - mu_tilde| <= sum_j ||D_j||_op          (state independent)             (1)
```

and the old bound is (1) with (N3). All improvements below are better upper bounds on `||D_j||_op` **or** on `||F_{j-1}(D_j)||_op`, using only what is known about `D_j` at the moment it is discarded.

In floating point the stored operator `O_j` is propagated, `D_j := Phi_j(O_j) - O_{j-1}` is taken with exact arithmetic on the stored data, and therefore contains rounding as well as truncation; section 5 bounds the rounding part.

## 2. Theorem 1: support at the time of discard

**Theorem 1.** Let `D = sum_{P in Q} d_P P` be Hermitian, `Q` a set of non-identity Pauli strings with union support `S`, `k = |S|`, and `eta^2 = sum d_P^2`. Then `||D||_op <= sqrt(2^k - 1) * eta <= 2^{k/2} eta`. The constant `sqrt(2^k - 1)` cannot be improved.

*Proof.* `D = D_S (x) I`; `tr D_S = 0` (no identity term) so its `d = 2^k` eigenvalues sum to 0. Let `a = ||D_S||_op = |lambda_1|`. The other eigenvalues sum to `-lambda_1`, so by Cauchy-Schwarz `sum_{i>=2} lambda_i^2 >= a^2/(d-1)`. Hence `tr D_S^2 >= a^2 d/(d-1)`. With `tr D_S^2 = 2^k eta^2` this gives `a^2 <= (d-1) eta^2`. *Attained:* `|0^k><0^k| = 2^{-k} prod_q (I+Z_q)`, so `D = |0^k><0^k| - 2^{-k} I` is a sum of the `2^k - 1` non-identity Z-type strings with coefficient `2^{-k}`; `||D||_op = 1 - 2^{-k}` and `eta^2 = (2^k-1)/4^k`, so `||D||^2/eta^2 = 2^k - 1`. For `k = 2` this is `D = (ZI+IZ+ZZ)/4` on `|00>` (exact check in section 8, H6). QED

**Corollary 1.** `|mu - mu_tilde| <= sum_j sqrt(2^{k_j}-1) eta_j`, `k_j` = support size of `D_j` (the union of the supports of the strings discarded at event j). Since `k_j <= |cone_j| <= n`, this is never worse than the old bound.

Why the *discard-time* support. (i) The later maps `Phi_{j-1},...` enlarge the support of the remainder (gates spread it) so converting `l2 -> op` after they act would use a larger k. (ii) The normalised-l2 norm is **not** monotone under later steps once any channel is non-unital: for the reset channel `Z -> I`, `D = Z_1 + Z_0 Z_1 -> 2 Z_1` has `eta^2` going from 2 to 4 (section 8, H1). The conversion to the operator norm must therefore be done *before* the later maps, and (N1) then carries it to the end. That is exactly the accounting of Corollary 1.

Because the constant is attained, the factor `2^{k/2}` cannot be removed without extra information about the state (H6): it is the price of a state-independent bound. Using the Cauchy-Schwarz form `|Tr(rho D)| <= sqrt(Tr(rho D^2))` one gets `||D||_2` with no `2^{k/2}` if `rho` is maximally mixed on the support; that needs the forward state and is not used here.

## 3. Theorem 2: l1, l2 and grouping

**Theorem 2.** For any partition of the discarded strings into groups `g` with `S_g` = union of supports in g, `k_g = |S_g|`, `eta_g^2 = sum_{P in g} d_P^2`, `l1_g = sum_{P in g} |d_P|`:

```
||D||_op <= sum_g min( l1_g , sqrt(2^{k_g} - 1) eta_g )        (2)
```

*Proof.* Triangle inequality over groups; inside a group `||D_g||_op <= l1_g` (each Pauli string has norm 1) and `<= sqrt(2^{k_g}-1) eta_g` by Theorem 1. QED

Special cases: one group = `min(l1, sqrt(2^k-1) eta)`; singleton groups give `l1`; the code partitions by the **exact support set** of each string (all strings in a group are non-identity on every qubit of the group's support, which Theorem 3 needs), and takes the minimum over {l1, Theorem 1, grouped (2)} per event (every term is a valid bound for that event, so the event minimum is valid).

*When each wins.* For m strings of equal magnitude a in a common support of size k: `l1 = m a`, Theorem 1 gives `sqrt(2^k-1) sqrt(m) a`. The l1 bound wins iff `m <= 2^k - 1`. A single string (`m = 1`) wins by `sqrt(2^k - 1)`: weight-w discarded Paulis cost `|d|`, not `2^{w/2}|d|`. Dense discards (many tiny strings, m up to `3^k` with exact support of size k) favour l2. Grouping by support interpolates: it is never above the l1 bound, but it is not comparable with the single-group l2 bound (many groups can each add a cone factor), hence the event-level minimum. Not implemented, but valid: for a *qubit-wise commuting* group the exact `||D_g||_op` is the maximum of a Walsh-Hadamard transform over the group's support.

## 4. Theorem 3: contraction by the remaining noise

Notation. For a qubit q, `tau_q(A) := (Tr_q A / 2) (x) I_q` (completely depolarise q; unital CP). For a set B of qubits, `tau_B` is the product. An operator A **has block B** if `Tr_B A = 0` (as an operator on the complement), i.e. every Pauli string of A is non-identity somewhere on B.

**Channel coefficient.** For a one-qubit CPTP map `E` let `kappa(E) := max{kappa in [0,1] : E - kappa * D is CP}` where `D(rho) = Tr(rho) I/2`. Equivalently `kappa = 2 lambda_min(C_E)` with `C_E` the Choi matrix of trace 2. Then `E^dagger = (1-kappa) Psi + kappa tau_q` with `Psi` unital CP (when `kappa < 1`; at `kappa = 1`, `E^dagger = tau_q`).

Closed forms (checked numerically in the JSON `kappa_choi`: depolarising (p = 0.1) 0.1; amplitude damping 0; dephasing 0; amplitude damping *then* depolarising 0.1; depolarising *then* amplitude damping 0.07 vs the proved lower bound p(1-gamma) = 0.07):

| channel (forward order) | kappa |
|---|---|
| depolarising `(1-p) rho + p Tr(rho) I/2` | p |
| amplitude damping, phase damping / dephasing, any channel with a zero Choi eigenvalue | 0 |
| amplitude damping, then depolarising p (what the benchmark uses; Heisenberg meets depolarising first) | p |
| depolarising p, then amplitude damping gamma | `>= p(1-gamma)` (proof: `C_E = (1-p) C_AD + p I (x) sigma`, `lambda_min(sigma) = (1-gamma)/2`) |

For exact rational parameters `kappa` is certified by exact LDL^T of `C_E - (kappa/2) I` (`qc27_exact.Chan.certificates`).

**Block rules.** Let `Phi` be the Heisenberg map of one forward step and `A` have blocks `B_1,...,B_r` (pairwise disjoint).

* (R1) `Phi = U^dagger . U` unitary (or any unital channel) on qubit set S. Every block meeting S is merged with S: `B' = S u (union of blocks meeting S)`; `Phi(A)` has block `B'`. *Proof:* `Tr_S(U^dagger A U) = Tr_S A` (cyclicity), so `Tr_{B'} (U^dagger A U) = Tr_{B' \ S} Tr_S A = Tr_{B'} A = Tr_{B' \ B_i} Tr_{B_i} A = 0`. Blocks disjoint from S are untouched.
* (R2) Product of one-qubit channels. If channel on q is unital, blocks are unchanged (`Tr_q . E_q^dagger = Tr_q` for trace-preserving `E_q^dagger`). If it is non-unital, every block containing q is dropped; blocks not containing q are kept.

**Kill factor.** Let step `Phi = tensor_{q in S} E_q^dagger` with certified `kappa_q`. Then

```
E^dagger = sum_{T subset S} w_T  ( tensor_{q in T} tau_q ) ( tensor_{q not in T} Psi_q ),   w_T = prod_{q in T} kappa_q prod_{q not in T} (1-kappa_q).
```

Terms with `T` containing a block `B` of A vanish (`tau_T = tau_{T \ B} tau_B` and `tau_B A = 0`); all other terms are products of unital CP maps, hence `||.||_op`-contractions. Therefore

```
||Phi(A)||_op <= s ||A||_op,     s = prod_{B in blocks, B subset S} ( 1 - prod_{q in B} kappa_q )        (3)
```

(independent events for disjoint blocks).

**Theorem 3.** Let `D` be Hermitian whose Pauli strings are all non-identity on every qubit of a set Sigma (e.g. all strings with exact support Sigma). Start with blocks `{q}, q in Sigma`, run the rules R1/R2 over the remaining steps `Phi_{pos-1},...,Phi_1` in Heisenberg order, multiply the factors (3). The product sigma satisfies `||F(D)||_op <= sigma ||D||_op`. *Proof.* Induction on steps: (R1/R2) maintain the block invariant for the whole map, (3) bounds the norm using the mixture decomposition. QED

The event bound is then `sum_g sigma_g min(l1_g, sqrt(2^{k_g}-1) eta_g)`. Group survival is computed only for the largest groups covering 99.9% of the event's grouped mass (at most 24); the rest use sigma = 1, which is always valid. Implementation: `qc27_bounds.survival`; exact twin `qc27_exact.exact_survival`.

**Direct stress test of Theorem 3.** 3000 random circuits (Haar 1q/2q unitaries, noise with random depolarising and amplitude-damping strengths, n = 4, 5) and random Hermitian D with exact common support; dense Heisenberg evolution; `||F(D)||_op <= sigma ||D||_op` violated in 0 cases; max ratio `||F(D)||/(sigma ||D||)` = 1; fraction of trials with sigma < 1: 0.702. Negative controls with one rule removed: gates that fail to merge blocks give 417 violations in 1500 trials, keeping blocks through non-unital channels gives 52; so the test has power.

**What holds and what does not for non-unital maps.**

1. `Phi = E^dagger` is unital CP, so the operator norm never grows (N1). True for every CPTP `E`.
2. `Phi` is **not** trace preserving: `Phi(Z) = lambda Z + tau I` carries an identity offset. The normalised Hilbert-Schmidt norm can grow (reset: factor sqrt 2 per step, H1) and the traceless part's norm is not contracted by `lambda` in operator norm: `||lambda Z + tau I||_op = lambda + tau = 1` when `tau = 1-lambda` (amplitude damping). So no per-string rule "Z-content decays by lambda" is valid (H3).
3. Tracelessness blocks do not survive a non-unital step on a qubit of the block (H4); the code drops them (R2).
4. The only contraction credited from a non-unital channel is its depolarising content `kappa`. Phase damping and pure amplitude damping give `kappa = 0`. A **type-aware** rule (X/Y letters contract by mu, Z letters do not) is valid but needs to track Pauli letters through the gates; it is not implemented (it is exact for Clifford + Pauli-channel circuits, where every string follows one deterministic path).

## 5. Floating-point implementation and rounding allowance

The propagator stores strings as `(x, z)` uint64 bitmasks and coefficients as float64, applies each gate's Heisenberg matrix H (signed permutation for Clifford gates, exact; at most two nonzeros per column for rotations/channels), merges duplicate strings, and truncates `|c| < eps` (the identity string is never discarded). Ledger per event: `eta` (fsum of squares), `l1`, union support k, and per-support groups `(mask, k, l1_g, eta_g^2, sigma_g)`.

Allowance (all added to *every* variant, old included): 

* Ledger quantities are inflated by `1 + 2^-38` (covers `fsum` rounding of squares, a square root, and `2^{k/2}`), and so is each total.
* For every non-exact op, per application, `L1(A_in) * (10 u colL1(H) + 4^m 2^-50)` with `u = 2^-53`, `colL1(H) = max_b sum_a |H_ab|` and m the number of qubits of the op. First term: sums of at most 4 products per output coefficient; second term: stored `H` assumed entrywise within `2^-50` of the exact adjoint matrix of the exact CPTP channel (polar-completed Kraus operators, as in QC-14). Clifford ops (signed permutations, no merges) contribute zero.
* Final readout: `u * L1`.
* sigma values are inflated by `1 + 2^-38`.

Because the rounding error operators are Hermitian and enter through the same telescoping, their operator norm is bounded by their l1 norm and then carried by (N1). This is an a-priori standard-model allowance, **not** an interval-arithmetic certificate and not machine-checked; the Fraction path (section 6) is the exact alternative. In the benchmark the accumulated allowance is negligible: median 3.25e-13, max 1.12e-11.

## 6. Theorem 4: exact arithmetic for Clifford + dyadic channels

**Setting.** Gates: Clifford (`H, S, X, Z, CNOT, CZ`); Heisenberg action on Pauli strings is a signed permutation. Channels: one-qubit maps with Heisenberg action `I -> I, X -> mu X, Y -> mu Y, Z -> lambda Z + tau I` where `mu, lambda, tau, kappa` are dyadic rationals (amplitude damping with `sqrt(1-gamma) = m` dyadic, `lambda = m^2`, `tau = 1 - m^2`; composed with depolarising p: `mu = (1-p)m`, `lambda = (1-p)m^2`, `tau = (1-p)(1-m^2)`, `kappa = p`). Threshold eps dyadic.

**Complete positivity is certified exactly.** The Choi matrix `C = [[(1+lambda+tau)/2,0,0,mu],[0,(1-lambda-tau)/2,0,0],[0,0,(1+tau-lambda)/2,0],[mu,0,0,(1-tau+lambda)/2]]` must satisfy `C = L D L^T` with exact rational `L` unit lower-triangular and `D >= 0` (equivalent conditions: `lambda + tau <= 1`, `tau - lambda >= -1`, `(1+lambda)^2 - tau^2 >= 4 mu^2`); likewise `C - (kappa/2) I`. No Kraus operators, no square roots and no polar correction are needed because the channel is *defined* by its PTM.

**Theorem 4.** (i) Every Pauli coefficient of the Heisenberg operator lies in `Z[1/2^M]`; if the channels have `b`-bit parameters and a string path crosses r channel applications, the denominator divides `2^{b r}`. (ii) All ledger entries `l1_e, l1_g, l2sq_e, l2sq_g, k, sigma_g` are dyadic rationals, computed with no rounding. (iii) With `ceil_sqrt(q) := ceil(2^64 sqrt(q)) / 2^64` (integer square root),

```
B_exact = sum_e min( l1_e , ceil_sqrt((2^{k_e}-1) l2sq_e) , sum_g sigma_g min(l1_g, ceil_sqrt((2^{k_g}-1) l2sq_g)) )
```

is a rational number with `|<O> - <O~>| <= B_exact`; the same recomputation from the serialised ledger gives bit-identical output. The old bound is `sum_e ceil_sqrt(2^n l2sq_e)` in the same arithmetic. *Proof:* (i),(ii) closure of the dyadic ring under the Heisenberg update; (iii) Lemma 0, Theorems 1-3 with `ceil_sqrt >= sqrt`. QED

Independent exactness check. `qc27_exact.schro_expectation` evolves a **complex-rational density matrix** with matrix-unit channel actions built from the textbook formulas (`rho_00 + gamma rho_11`, `m rho_01`, depolarising) and Gaussian-rational gate matrices (Hadamard handled as `M rho M^T / 2`, no sqrt). Without truncation the Heisenberg and Schroedinger expectations are equal as Fractions in every case below; with truncation the exact error is below every exact bound; the float pipeline run on the float twin agrees with the exact bound to 1e-9 relative.

| n | depth | seed | eps | discard events | exact error | exact bound old | exact cone_tl | exact surv | H = Schroedinger (Fractions) | all channels CP+kappa certified | witness verified, tamper rejected | max coefficient denominator bits |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|---:|
| 3 | 3 | 0 | 1/32 | 2 | 0.023 | 0.117 | 0.0631 | 0.0435 | True | True | True/True | 57 |
| 3 | 3 | 6 | 1/128 | 3 | 0.00768 | 0.0409 | 0.0164 | 0.013 | True | True | True/True | 59 |
| 3 | 3 | 21 | 1/32 | 3 | 0.00327 | 0.117 | 0.048 | 0.0384 | True | True | True/True | 80 |
| 4 | 3 | 0 | 1/128 | 2 | 0.00748 | 0.0517 | 0.0186 | 0.0134 | True | True | True/True | 73 |
| 4 | 3 | 6 | 1/32 | 2 | 0.0262 | 0.201 | 0.0872 | 0.0695 | True | True | True/True | 78 |
| 4 | 4 | 0 | 1/32 | 3 | 0.0142 | 0.266 | 0.106 | 0.0766 | True | True | True/True | 133 |
| 4 | 4 | 0 | 1/128 | 4 | 0.0036 | 0.112 | 0.0577 | 0.0225 | True | True | True/True | 133 |

## 7. Extending the QC-14 certification machinery to non-unital heterogeneous channels

What QC-14/15/16 do: literal binary64 inputs are exact dyadics; every local update is audited by exact integer residuals; unitaries/Kraus stacks are replaced by their polar completions so the target is exactly CPTP; the Heisenberg adjoint of a CPTP map is a unital CP contraction, so `E_new <= E + alpha (1+E) + e_i` holds for **non-unital** targets as well (QC-14 already ran amplitude damping this way). So the operator-norm contraction part of the machinery extends verbatim. What changes, and what stays dyadic:

* **Stays dyadic.** A non-unital adjoint sends `Z -> lambda Z + (1-lambda) I`. If lambda, mu, tau are dyadic, all coefficients stay in `Z[1/2^M]`; bit length grows linearly with the number of noisy events on a path (section 6 table: max denominator bits).
* **What breaks.** (a) Physical parametrisation: `mu = exp(-t/T2)` and `lambda = exp(-t/T1)` are irrational, and `sqrt(1-gamma)` in the Kraus operator `K_0` is irrational unless `m` is dyadic. Fix: parametrise by the dyadic contraction factors `(mu, lambda, tau)` directly and state the model error separately: for a stored dyadic channel `E'` and the intended one `E`, `|mu_E - mu_E'| <= sum_c ( 2|Delta mu_c| + |Delta lambda_c| + |Delta tau_c| )` over all channel applications c (telescoping with (N1) and `||(E^dagger - E'^dagger)||_cb <= 2|Delta mu| + |Delta lambda| + |Delta tau|`; the three Pauli components are three rank-one maps `A -> sigma_b (x) Tr_q(sigma_b A)/2`). Not implemented numerically. (b) CP-ness must be *certified*, not assumed: with a PTM-defined channel the polar correction of QC-14 is replaced by the exact LDL^T certificate (alpha = 0). (c) Non-Clifford gates: `cos, sin` of generic angles are not dyadic; Pythagorean triples give rationals (denominators `5^k`), which Fractions handle but the power-of-two integer-shift arithmetic of QC-16 does not; alternatively round to dyadics and keep the QC-14 polar term `alpha = d(2+d)`. (d) l2 ledger entries need square roots: replaced by `ceil_sqrt` or by comparing squares. (e) The normalised HS norm is not monotone for non-unital maps (H1): any certificate that carries an HS norm across later steps is invalid; QC-12's accounting is safe because it converts to the operator norm at the cut (this is Theorem 1).

**Certificate format `qc27-nonunital-dyadic-witness-v1` (prototype, implemented in `qc27_exact.emit_witness / verify_witness`).**

```
{ format, n, target, eps: "a/b",
  steps: [ { trunc: bool, ops: [ {kind:"cl", name, sites} |
            {kind:"noise", chans:[ {q, mu, lam, tau, kappa,          # "num/den", den a power of 2
                                    cp_L, cp_D, kappa_L, kappa_D } ]} ] } ],   # exact LDL^T of Choi and Choi - kappa/2 I
  events: [ {pos, groups:[{mask, l1, l2sq, sigma}]} ],
  claim: { expectation, bound_surv, bound_old, sqrt_bits } }
```

The verifier (stdlib `fractions` only) checks: all parameters dyadic; `L D L^T == Choi` exactly with `D >= 0` for each channel and for the kappa certificate; re-executes the Heisenberg propagation with the stored eps; recomputes every ledger group (support mask, l1, l2sq, sigma by the block tracker) and the bounds; accepts iff all equal the claims. Tampering with a channel parameter or the claimed bound is rejected (tests). For MPO-scale use the same per-event fields would be added to QC-15 `gate_svd`/`channel` events: `kappa` certificate per channel, the discarded-support block list, and the sigma recurrence as an exact rational chain (like `E_{j+1} = E_j + r_j g_j + alpha_j (1+E_j)`), with the segment boundary commitment of QC-16 covering the block family. Not implemented for MPOs.

**Direct consequence for QC-25.** `qc25_channels.mpo_z` uses `2^(n/2)`. Theorem 1 applies per cut with n replaced by the width `w_j` of the window of non-trivial MPO cores at that cut: the discarded `D_j` is supported inside that window, so `bound = sum_j 2^{w_j/2} eta_j` (use `2^{w/2}`, not `sqrt(2^w-1)`, because an SVD discard can contain the identity component). Not applied here (existing files untouched).

## 8. Failed hypotheses (exact counterexamples, all rendered from the JSON)

| id | hypothesis | exact numbers | verdict |
|---|---|---|---|
| H1 | normalised l2 norm of the remainder never grows under later CPTP steps | reset channel (CP certified: True): eta^2 2/1 -> 4/1 | false |
| H2 | influence of a discarded weight-w string is damped by lambda^w | lambda=3/4, D=Z0Z1 before [dep, CNOT]: true influence 3/4 > claimed 9/16; block tracker sigma 15/16 (valid: True) | false (CNOT lowers weight before the noise acts) |
| H3 | amplitude damping contracts Z-content by lambda (op norm) | m=1/2 (lambda=1/4, tau=3/4): influence on |0> 1/1 > claimed 1/4; kappa=0/1 | false |
| H4 | tracelessness block survives a non-unital step, so later depolarising still contracts | true influence 15/16 > claimed 3/4; tracker sigma 1/1 (valid: True) | false |
| H5 | ||F(D)|| <= lam_0 lam_1 ||D|| when the strings of D jointly cover qubits {0,1} | D=X0+X1: op norm 2/1 -> 3/2, ratio 3/4 > claimed 9/16 | false (hitting sets give 1-(1-lam)^2, not lam^2) |
| H6 | |<D>| <= eta without the 2^{k/2} factor for product inputs | D=(ZI+IZ+ZZ)/4 on |00>: eta^2=3/16, value=3/4, value^2/eta^2=3/1 (= 2^k-1: True) | false; also shows Theorem 1 is tight |

Additional negative results. (a) "Charge the l2 norm at the end": invalid (H1) and not tighter even when valid (support grows). (b) The survival factor of the old-style rule `lambda^w` is unsafe because Clifford gates can lower weight before the noise acts (H2). (c) Keeping the product of per-qubit factors on a union support instead of hitting sets is unsafe (H5). (d) Survival from amplitude/phase damping beyond the depolarising content is unsafe (H3). (e) Survival is a modest correction at these noise levels: median bound_grouped/bound_surv = 1.02, max 1.28 over 270 benchmark runs; the survival bound is evaluated for at most 24 groups per event.

## 9. Results

Dense reference: for n = 8 full density-matrix evolution with Kraus operators (Schroedinger) and dense Heisenberg Pauli-vector evolution agree to 1.89e-15; n = 12 and 16 use the same dense Heisenberg Pauli-vector evolution restricted to the exact backward-cone sites (ops outside the cone act as the identity on the evolving operator). For n = 12, the cone-restricted Schroedinger evolution agrees with the dense Heisenberg value to 4.02e-16 over 16 circuits; unrestricted 12-qubit dense Heisenberg versus cone-restricted: max deviation 9.02e-17 over 2 circuits. For n > 16 **no dense reference exists and none is claimed: only bounds and ledger statistics are reported.** n = 16 is feasible only because the cone is small (depth <= 5 gives at most 10 sites).

Circuits: brickwork 1D, layers of random Clifford 1q (I/H/S/HS), RZ and RX with angles uniform in [-0.3, 0.3], 1q noise, CNOT brickwork, per-qubit noise after each CNOT; noise = amplitude damping then depolarising; "weak" (p1, g1, p2, g2) = (0.002, 0.001, 0.01, 0.003), "strong" = (0.01, 0.005, 0.05, 0.01); observable Z on the central qubit; input |0...0>; discard `|c| < eps` after each branching group (rotations, 1q noise, 2q noise). One layer of the circuit grows the backward cone by about two sites.

### 9.1 Validity against dense references

| n | runs with discards | pass old | pass cone | pass cone_tl | pass min(l1,l2) | pass grouped | pass surv | ordering | max actual/bound (surv) | median surv/actual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 349 | 349 | 349 | 349 | 349 | 349 | 349 | 349 | 1 | 60.6 |
| 12 | 117 | 117 | 117 | 117 | 117 | 117 | 117 | 117 | 0.13 | 98.4 |
| 16 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 0.618 | 172 |

(`pass X` = number of runs with at least one discard in which `actual <= bound + 1e-11`; "ordering" = number of runs with `old >= cone >= cone_tl >= min(l1,l2) >= grouped >= surv`; max actual/bound is over the survival-weighted bound; the 1e-11 allowance covers float roundoff of the dense reference itself.) A ratio of 1 appears when a single discarded string `Z` on the last step is exactly the error. Per-event check (n = 8): 406 events and 2839 (event, support-group) pairs checked against exact per-event errors Tr(rho_pos D); bound failures per variant {"old": 0, "cone": 0, "cone_tl": 0, "minl1l2": 0, "grouped": 0, "surv": 0}; group survival-bound failures 0; max telescoping residual 3.55e-15.

### 9.2 Improvement ratios

| n | runs | median cone width | median k at discard | median old/cone | old/cone_tl | old/min(l1,l2) | old/grouped | old/surv | min old/surv |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 54 | 12 | 6 | 4.62 | 4.65 | 6.59 | 6.75 | 6.82 | 1.49 |
| 16 | 54 | 12 | 6 | 12.7 | 12.7 | 23.1 | 23.2 | 24.6 | 4.31 |
| 24 | 54 | 12 | 6 | 292 | 293 | 456 | 458 | 463 | 141 |
| 32 | 54 | 12 | 6 | 3.52e+03 | 3.53e+03 | 5.6e+03 | 5.73e+03 | 6.13e+03 | 1.02e+03 |
| 48 | 54 | 12 | 6 | 1.14e+06 | 1.15e+06 | 2.01e+06 | 2.02e+06 | 2.05e+06 | 4.87e+05 |

Median old/new ratios by cone width and depth (all eps, seeds and both noise levels pooled):

| n | depth | runs | median cone width | median k | median old/cone_tl | old/min(l1,l2) | old/surv | median surv/cone_tl |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 4 | 18 | 8 | 4 | 6.38 | 7.46 | 7.51 | 0.854 |
| 12 | 6 | 18 | 12 | 6 | 4.19 | 6 | 6.54 | 0.648 |
| 12 | 8 | 18 | 12 | 8 | 2.51 | 3.52 | 3.77 | 0.639 |
| 16 | 4 | 18 | 8 | 4 | 41.7 | 46.5 | 46.7 | 0.845 |
| 16 | 6 | 18 | 12 | 6 | 11.5 | 15.2 | 16.6 | 0.686 |
| 16 | 8 | 18 | 16 | 8 | 7.94 | 12.5 | 13.8 | 0.545 |
| 24 | 4 | 18 | 8 | 4 | 432 | 548 | 552 | 0.849 |
| 24 | 6 | 18 | 12 | 6 | 257 | 423 | 429 | 0.682 |
| 24 | 8 | 18 | 16 | 6.5 | 216 | 297 | 316 | 0.598 |
| 32 | 4 | 18 | 8 | 4 | 7.08e+03 | 8.37e+03 | 8.62e+03 | 0.806 |
| 32 | 6 | 18 | 12 | 6 | 2.83e+03 | 4.15e+03 | 4.41e+03 | 0.61 |
| 32 | 8 | 18 | 16 | 8 | 1.4e+03 | 3.45e+03 | 3.97e+03 | 0.46 |
| 48 | 4 | 18 | 8 | 4 | 1.83e+06 | 2.62e+06 | 2.73e+06 | 0.688 |
| 48 | 6 | 18 | 12 | 6 | 1.06e+06 | 1.94e+06 | 1.98e+06 | 0.509 |
| 48 | 8 | 18 | 16 | 8 | 6.35e+05 | 1.22e+06 | 1.29e+06 | 0.369 |

By truncation threshold (pooled over n >= 24, where the cone is small compared with n):

| eps | runs | median old/cone_tl | old/min(l1,l2) | old/grouped | old/surv | median events | median dropped strings | median max terms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 54 | 4.46e+03 | 8.52e+03 | 8.53e+03 | 9.39e+03 | 14 | 7.63e+03 | 3.45e+03 |
| 0.003 | 54 | 3.28e+03 | 4.99e+03 | 5.19e+03 | 5.87e+03 | 14 | 1.83e+04 | 1.06e+04 |
| 0.001 | 54 | 2.27e+03 | 3.45e+03 | 3.6e+03 | 3.83e+03 | 14 | 4.74e+04 | 3.03e+04 |

Tightness where a dense reference exists (n = 12, 16 at depths where the cone has <= 10 sites):

| n | runs with reference | median actual error | median old bound | median surv bound | median old/actual | median cone_tl/actual | median surv/actual | max surv/actual | min surv/actual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 18 | 0.0038 | 4.11 | 0.54 | 1.17e+03 | 185 | 167 | 589 | 26.6 |
| 16 | 18 | 0.000502 | 8.18 | 0.152 | 1.08e+04 | 179 | 147 | 1.53e+04 | 39.1 |

Reading the numbers. The dominant gain is the support factor: old/cone_tl is roughly `2^{(n-k)/2}` with k the support at discard (median k in the table, about half the final cone width because most discards happen while the cone is still growing). It shrinks as the cone fills the register (n = 12 and 16 at depth >= 6). The per-event min(l1,l2) improves the cone_tl bound by a factor 1.4 (median; max 5.47) and grouping by support a further factor 1.01 (median; max 1.12). The noise-survival factor gives a further factor 1.02 (median; max 1.28) at these noise strengths, i.e. it is nearly negligible here. Even the new bound remains well above the actual error (tightness table): it is a worst-case-over-states bound and Theorem 1 shows its l2 part is attained for adversarial discarded operators.

Wall time: validity sweeps and benchmark 695.5 s total on one thread (OMP/OPENBLAS = 1).

## 10. Reproduce

```
cd experiments/channel_mpo/ibm_qc23/qc27
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest -q test_qc27_bounds          # about 15 s
python qc27_benchmark.py all --out qc27_bound_benchmark.json    # about 12 min
python qc27_benchmark.py tables                  # markdown tables from the JSON
```

## 11. Limitations and open doubts

* The float bound is rigorous under the stated rounding model, not interval-certified; the exact path covers only Clifford + dyadic channels, n <= 4 in tests (the code is general, cost grows with the number of kept strings).
* Theorem 3 is proved and stress-tested (including negative controls) but not formally verified; the survival factor only exploits depolarising content and the block structure, so it is close to 1 for weak noise.
* The reference for n = 12, 16 is cone-restricted; the restriction is validated against unrestricted dense evolution for n = 8 (full) and 2 unrestricted n = 12 circuits: max deviation 9.02e-17.
* "Actual error" is one circuit realisation; bounds are worst-case over states, so the ratios say nothing about typical tightness on hardware.
* Event-level survival uses at most 24 groups per event; larger numbers would tighten slightly.
* The MPO bound of QC-25 was not modified (only a recommended patch is described in section 7).
