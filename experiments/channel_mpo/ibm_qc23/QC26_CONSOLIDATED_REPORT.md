# QC-23 to QC-27 consolidated report (2026-11-25)

Status: offline phases complete; **no QC-26A hardware data exists**. QPU seconds used this session for QC-26/27: 0. Remaining plan allocation on 2026-11-25: 376 s of 600 s.

## 1. Integration

PRs #29 (QC-25B), #30 (QC-25C) and #31 (QC-26A) were merged into `main` in dependency order (25B, then 25C, then 26A); 25C is a superset of 25B, so retargeting was unnecessary. The QC-26A commits were rebased onto main, so GitHub may still list #31 as open with an empty diff; closing it needs GitHub authentication that this machine does not use.

| Check | Result |
|---|---|
| QC-25B replay matrix (eval_echo 8/16, nc_eval 8/12/16), run locally from an LF export | all five pass; sparse oracle gap 1.8e-14 |
| QC-25C receipt replays (fit, transfer 8, 12 shallow, 12 deep, 16) | all pass after one fix |
| QC-26A unit tests, plan receipt (byte-identical), synthetic receipt (tolerance 1e-10) | pass |
| Remote GitHub Actions status | not read: the API was rate-limited from this machine |

Fix: `qc25c_verify.py` compared the fit receipt with exact float equality; Node 24 differs from Node 22 by 4.7e-17 in the mean-square error. It now uses a 1e-12 relative tolerance, with offsets and counts still exact.

Preserved negative results: QC-25B paired improvement of native Model A over uniform A is negative on the 16-circuit nc_eval width-12 replay (mean difference -0.0007, CI [-0.0011, -0.0003], 4/16 wins). QC-25C's combined model reaches 6.3% relative MAE gain over the twirled model, below the 10% target, and explains only 10/56 target-site observations within 0.05.

Windows note: Git for Windows `autocrlf=true` rewrites QASM files to CRLF in the working tree and makes `git archive` do the same, which breaks byte-hash checks locally; use `git -c core.autocrlf=false archive`. Committed blobs are LF and CI is unaffected.

## 2. Kingston diagnostic (QC-26A): built and audited, not submitted

Build `ibm_kingston`, calibration 2026-11-25T17:24:39+08:00: 82 circuits, 384 shots, estimate 16.38688 QPU-s, ceiling 20 s, **plan_sha256 `1ff6d3b5d8147de6fadbe83d8d4966b1435d14912063aefab0b87194c10876bb`**, QPY sha256 in `qc26/receipts/qc26a_compiled_build1.json`.

Audit (`qc26a_audit.py`, receipt `qc26a_audit_build1.json`), all pass: manifest, proposal, calibration, QPY and 82 QASM hashes reproduce; every circuit stays on its planned physical qubits with zero swaps, and every CZ lies on a calibrated edge; native gates are only rz, sx, x, cz, measure, barrier (480 sx, 240 cz in total). Sign conventions were tested on the compiled circuits by exact statevector with injected SX offsets (0.004, -0.025, 0, 0.04 rad) and CZ errors (local and conditional phases up to 0.047 rad), passed through the committed estimator: all recovered with correct sign within 0.003 rad (SX) and 0.005 rad (CZ). Live check (`qc26a_live_check_build1.json`): calibration unchanged, durations unchanged, 82 circuits coerce to SamplerV2 publications, all ops in the live target, backend operational, usage 224 of 600 s.

Limits of this audit: SamplerV2 acceptance was checked locally (publication coercion, option construction), not by a job; the audit does not model simultaneous-drive crosstalk.

**The build is stale after 15 minutes, so its hash cannot be used for authorization now.** To run: rebuild with `python qc26a_compile.py build --out results_local/qc26a`, read the new hash and estimate, then the account owner writes the authorization file (QC-23 schema plus `acknowledge_estimate_not_hard_limit: true` and `checked_current_allocation: true`, `max_qpu_seconds` at most 20) and runs `qc26a_compile.py submit` within 15 minutes.

## 3. Statistical identification (Phase 3): tooling ready, synthetic validation only

`qc26a_identify.py` (24 tests pass): SPAM correction, signed SX offsets, CZ local and conditional phases, joint-bitstring bootstrap that refits SPAM each replicate, Jacobian singular values, profile likelihood, aliasing and boundary checks, and diagnostics for crosstalk, leakage, relaxation and quadrature linearity, plus a decision rule and comparison to the 96 committed JS synthetic trials.

Synthetic results (`receipts/qc26a_identify_synthetic.json`): the Python generator and estimator reproduce committed JS trials 0 and 48 to 1e-17; over 480 datasets per arm, bias is at most 8e-4 rad and Wald 95% coverage is 0.949; over 96 datasets per arm with 200 bootstraps, percentile coverage is 0.934 (minimum 0.896, mild undercoverage) and bootstrap intervals exclude zero on zero truth 5.6 to 7.5% of the time. The Jacobian over 120 observations and 100 parameters has full rank, condition number 120, weakest directions in visibility, SPAM contrast and intercepts rather than the angles. The diagnostic suite flags at least one component on 9 to 11% of clean datasets.

Identifiability findings: the control's local phase alpha_control is not observable from this plan by design (only alpha_target and delta are reported); with even repetition counts SX has an exact reflection alias (d, phi) to (pi - d, -phi), found numerically after a first derivation missed it; CZ (alpha_t, delta) is determined only up to shifts of (j, l)·pi/2 with j congruent to l mod 2. An injected delta near pi/2 is detected, and an injected SX offset of 0.13 is flagged as a boundary solution. A single dataset cannot reach the decision `NONZERO_REPLICATED`; the rule requires a fresh independently shuffled replicate block.

## 4. QC-26B heterogeneous channel integration (Phase 4)

Engine `qc26/qc26b_engine.mjs`: sparse Pauli Heisenberg propagation in native gate order with models C0 (gate-local stochastic), C1 (non-unital thermal relaxation), C2 (C1 plus coherent SX offsets), C3 (C2 plus coherent CZ phases exp(-i/2(a_c Z_c + a_t Z_t + d Z_c Z_t)) placed at each CZ).

Independent verification: over 48 random circuits with every gate type, both channel variants, the sparse engine matches a Schrodinger-picture dense density-matrix evolution (Kraus and unitary maps, no shared code) to 4.2e-15 (96 evaluations, `receipts/qc26b_dense_receipt.json`); C0, C1, C2 reproduce the committed QC-25C receipts to 1.1e-15 on 320 rows. A first version of the dense check let non-physical relaxation parameters through and could have hidden NaN comparisons; it now draws valid channels and fails on non-finite values.

Truncation ledger, all 528 observables, C0 to C3: maximum discarded absolute coefficient mass 8.5e-11, which bounds the discarded operator norm since each Pauli has norm 1. The older 2^(n/2) times summed per-step l2 bound reaches 1.0e-9 on the same runs. Both are far below the roughly 0.04 prediction error. Maximum live terms 104,883.

**Retrospective replay of all 528 existing Kingston observables (zero QPU).** Parameters were fitted on previously seen counts, and sites 5 and 23 were chosen after seeing earlier data, so none of this is prospective evidence. SX offsets are the QC-25C fit (site 5: -0.0191, site 23: +0.1 rad, at its bound). The ZZ phases were fitted here on the same eight width-8 depth 1-2 circuits as QC-25C and frozen: edge 16_23 at -0.101 rad, edge 5_6 at +0.111 rad, edges 22_23 and 4_5 at 0. With the search bound relaxed to 0.2 and 0.4 the optimum is interior, so the first-run values at the 0.1 bound were not a runaway (`receipts/qc26b_boundscan.json`). Both phases are two to five times larger than the 0.022 to 0.047 rad conditional phases the QC-26A design injected, so they are effective parameters that may absorb unmodelled physics.

| Held-out nc_eval circuits (28 circuits, 320 observables) | MAE all sites | MAE sites 5, 23 | MAE other sites |
|---|---|---|---|
| ideal | 0.0474 | 0.1245 | 0.0311 |
| C0 | 0.0463 | 0.1238 | 0.0299 |
| C1 | 0.0453 | 0.1240 | 0.0286 |
| C2 | 0.0434 | 0.1137 | 0.0285 |
| C3 | 0.0346 | 0.0655 | 0.0280 |

Relative MAE improvement over C0 on the 28 held-out circuits: C1 2.1%, C2 6.3%, C3 25.3%. Paired circuit bootstrap (20,000 resamples) of the C3 improvement: +0.0132 [+0.0099, +0.0167], 28/28 circuit wins; on sites 5 and 23, +0.0583 [+0.0459, +0.0702], 25/28 wins. Other-site MAE changes by -0.0019 (no degradation above 0.005). On the 12 echo circuits measured a day earlier at different sites, C1 to C3 are identical to C1 and slightly worse than C0 (-0.9%), because those paths do not contain sites 5 or 23; whether the gains persist across calibration cycles is therefore untested. The retrospective C3 result meets the 10% target numerically but cannot count toward it.

## 5. Prospective evaluation (Phase 5): tooling ready, nothing frozen, no fresh counts

`qc26b_prospective_plan.py`, `qc26b_freeze.mjs`, `qc26b_prospective_eval.mjs`, protocol `QC26B_PROSPECTIVE_PROTOCOL.md`, 13 tests pass (tampered parameters, changed predictions, forged hash, tampered QASM, CRLF checkout, malformed counts). A read-only build on `ibm_kingston` produced 24 kicked-Ising circuits with fresh seeds 51 to 54 on alternative physical paths (some containing 5 and 23 at new positions, one containing 21, one avoiding both), 13.86 QPU-s estimate, plan hash `6ab8789e988f584ada7bfb0f5a7cc133eecc2ddbc5df90cf3720bf340ec7aa01`, built stale and not submittable as is. Dry-run receipt with synthetic counts: counts drawn from C3 pass all five criteria for C3 and fail (a) and (b) for C2; counts drawn from C0 fail (a), (b) and (c) for C3. Control site 3 cannot be isolated (its neighbourhood holds at most five qubits once 5 and 23 are excluded). Criterion (c) is per site with no minimum observation count, so sparsely observed sites may trip it on noise.

Order of operations: run QC-26A, fit independent SX and CZ parameters, run the identification tool, freeze parameters and predictions and commit them, then rebuild the prospective plan, obtain owner authorization, run, evaluate. The Fez replication waits on the Kingston diagnostic.

## 6. Truncation bound (Phase 6, `qc27/`)

Theorems (proofs in `QC27_BOUND_NOTE.md`): T1: the operator norm of a traceless discarded term supported on k qubits is at most sqrt(2^k - 1) times its normalized Hilbert-Schmidt norm, and the constant is attained. T2: grouping discards by exact support and taking the minimum of the l1 and l2 bounds per group gives a valid bound. T3: later channels contract the remainder by a survival factor from tracelessness blocks and depolarising content; non-unital channels add no contraction beyond the depolarising content. T4: for Clifford gates with dyadic non-unital channels certified by an exact LDL^T Choi test, the bound is a rational number recomputed in Fractions (tested for at most four qubits).

Measured (270 synthetic runs per variant, n = 12 to 48, depths 4, 6, 8, three thresholds, weak and strong noise, three seeds; recomputed from the JSON): median ratio of the old bound to the new bound is 6.8 (n=12), 24.6 (n=16), 463 (n=24), 6.1e3 (n=32), 2.1e6 (n=48). Support restriction supplies almost all of the gain; the survival factor adds a median 1.02x. No bound violation in 616 dense-reference rows (n = 8, 12, 16). The new bound is still about 140 times above the actual error at n=16. Dense references at n=12 and 16 are restricted to the backward causal cone; the restriction was checked against the unrestricted dense evolution only at n=8 and on two n=12 circuits. Six hypotheses failed with exact counterexamples, among them monotone l2 norm under later steps, lambda^weight survival, and amplitude damping contracting Z content. The QC-25 MPO patch (replace n by the window width at each cut) is described, not applied.

## 7. Blockers and remaining work

1. Owner authorization of a fresh QC-26A build; then Phase 3 on real counts. Estimated 16.39 s of 376 s remaining.
2. Phase 5 needs frozen parameters from item 1, then a second authorized run (about 14 s estimated).
3. Fez replication waits on the Kingston results.
4. Remote CI results were not read; local replays are the evidence.
5. Applying the tighter bound inside the QC-25 MPO engine and extending QC-14 dyadic certificates to heterogeneous non-unital channels beyond the n <= 4 prototype.
