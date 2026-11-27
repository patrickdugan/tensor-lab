# PGC-QPU campaign: consolidated report (2026-11-26 to 2026-11-27)

Evidence labels used below: **exact** (algebra, Lean), **ideal sim**, **noisy sim**, **retrospective hardware** (archived Kingston counts), **prospective hardware** (none yet). No QPU time was consumed in this session. Emergent spacetime and Einstein dynamics are not addressed.

## Status in one table

| Phase | State | Evidence class |
|---|---|---|
| 0 Preflight | Done | live account read-only, exact algebra |
| 1 QC-26A recalibration | Built and live-checked, **awaiting owner authorization** | none yet (hardware pending) |
| 2 Ramsey q23 and q5 | Compiled, estimator and submit adapter tested, **awaiting authorization** | ideal-model synthetic counts, adapter unit tests |
| 3 QC-26B prospective | Extension C3Z implemented offline (see below); blocked on independent parameters from jobs 1-2 | synthetic dry run only |
| 4 PGC-QPU02 | Exact and noisy simulation done; 30-circuit proposal, 18.97 s at 2048 shots; needs its own adapter and approval | exact, noisy sim |
| 5 Q3 holonomy | Exact rank/Lean/minimal-observation results done; no hardware experiment justified | exact, noisy sim |
| 6 BPD design | Offline comparison done (see `bpd/BPD_REPORT.md`) | linearised sim |

## Phase 0: preflight (done)

* Repository: `main` at `4cb948f` when started; PGC-QPU01 files from PR 32's branch were brought onto `main` (`8a0a9ac`); PR 32 itself was not merged or modified.
* CI on `8a0a9ac`: PGC-QPU01 local identifiability (replays, SVD check, 30 + 13 pytest tests) passed; the QC-02 to QC-21 suites listed passed; QC-23 hardware preflight was still running at the last check.
* Local IBM account: loads without printing credentials. Instance plan `open`. Operational backends: ibm_fez (2 pending jobs), ibm_kingston (0 pending, operational), ibm_marrakesh (0 pending).
* Allocation: **376 s remaining**, 224 s consumed of 600 s in a rolling 28-day window; it matches the last recorded value. The usage window ends at query time, so the balance can change only as past jobs age out; it rechecks at each submission.
* Kingston calibration at the build: 2026-11-26T23:30:31+08:00. Median two-qubit error 1.80e-3, median readout error 1.04e-2, native CZ.
* Replays: `audit.mjs --check` PASS; `ramsey_campaign.mjs --check` PASS.
* Independent NumPy SVD of the committed Jacobian rows (`analysis/jacobian_svd_check.py`, threshold 1e-6 relative): target 6 -> rank 4, all 6 -> 6, target 14 -> 6, all 14 -> 12. All four match the expected table. Zero derivative columns: restricted target set ZZ_22_23, ZZ_4_5; expanded all-site set Z_22_23_23 and Z_4_5_5. Weakest nonzero directions of the restricted all-site model: SX_23 (singular value 0.097) and SX_5 (0.141).

## Phase 1: QC-26A

`qc26a_compile.py build` against fresh calibration: 82 circuits, estimate 16.387 s (cap 20 s), compiled and hashed. The build in `receipts/compile_receipts_20261126.json` is a dry run; it will be older than 15 minutes at any authorization time, so a new build and plan hash must be made at approval time (`HARDWARE_RUNBOOK.md`). The existing identification pipeline (`qc26a_identify.py`) and its synthetic receipts are unchanged. Questions 1.1 to 1.5 (signed SX boundary solutions, conditional phase profiles, Jacobian singular values, visibility/SPAM/crosstalk, whether the effective ZZ near +-0.1 rad is independently observed) are **open until counts exist**; nothing in this session answers them.

## Phase 2: PGC four-edge Ramsey

* Both blocks compile with the existing helpers: q23 plan_sha256 `ab72587d...` and q5 `a6745a5e...`, 66 circuits each, 13.18944 s each (cap 15 s).
* New: `ramsey_identify.py` (SPAM from the two preparation rows, per-orientation per-state fit with the ideal CZ phase `m b pi` explicit, joint-bitstring bootstrap that resamples every row including SPAM rows and refits SPAM on every replicate, forward/reverse delta agreement, odd/even alias scan); `submit_ramsey.py` (QC-26A-equivalent gates plus allocation-versus-ceiling check and usage receipts).
* Synthetic replay (ideal-model counts with asymmetric SPAM, intercepts, visibility decay): delta and endpoint phases recovered to 0.012 and 0.02 rad (test tolerances); even-repetition alias has profile chi-square difference 2e-14 (exact), odd repetitions raise it to 8.4e3; an injected `(+pi/2, +pi/2)` alias is recovered on the correct branch; the control local phase is invisible to the forward circuit and visible after role reversal (statevector, 1e-12). Tests: 30 passed.
* Not established: any hardware nonzero effect. This is an exploratory design; replication blocks (independently shuffled) need a new estimate and approval.

## Phase 3: QC-26B

The endpoint-local Z extension **C3Z** is declared separately from the original C3 comparator (`qc26b_engine.mjs` untouched). Freeze format is now `qc26b-frozen-v2` (stores ideal, C0-C3 and C3Z predictions and the C3Z ledger under the self-hash); C3 is delta-only (alpha stripped), C3Z adds the endpoint alpha terms and equals C3 when alpha is 0. The evaluator scores C3Z on criteria (a)-(e) labelled DECLARED EXTENSION, plus the paired bootstrap of MAE(C3)-MAE(C3Z), target-site MAE for 5 and 23, other-site degradation and the C3Z ledger. Old v1 frozen files still verify (C3, C2 only). Verified offline: 22 unit tests pass (13 original, 9 new), `qc26b_selftest.mjs` 48 cases identical to the receipt, dense check 96 evaluations at 4.2e-15. `qc26b_replay.mjs` could not be run in this Windows checkout (CRLF changes QASM hashes; predates this work); it is unaffected by inference only. The dry-run receipt was regenerated, so its C3 numbers changed (synthetic only). Whether independently calibrated coherent errors reproduce the retrospective 25.3% MAE improvement (previous constant-SX model 6.3%) is **unanswered**: it needs QC-26A and Ramsey counts, a committed `frozen.json`, and fresh evaluation counts.

## Phase 4: PGC-QPU02

See `qpu02/` and the MathCan note. Exact results: the intact circuit maps `|s,a0,b0,t0>` to `|s,a0+s,b0+s,t0+a0+b0>` so the target is independent of `s` in all eight contexts; a gate path from `s` to `t` exists; GHZ fidelity witness 1; cutting either path or resetting either middle register restores response magnitude 1; resetting both gives 0 and no structural path; dephasing one middle qubit keeps the response at 0 and drops the coherence witness to 0. Noisy simulation: diagonal coherent ZZ errors, crosstalk phases, depolarizing and symmetric readout error produce no Z-basis influence on the zero-response target; amplitude damping does (intact apparent influence 0.86% Kingston-like, 2.1% pessimistic; echo control 0.56% and 1.4%). With 2048 shots per arm the intact-minus-echo difference stays below 3 standard errors in both combined scenarios. The proposal has 30 circuits; the intact circuit compiles to 10 native CZ on FakeKingston because the 4-cycle does not embed in heavy-hex. A real build must re-select qubits from fresh calibration.

## Phase 5: arithmetic holonomy and minimal observation sets

Exact: rank 5, gauge dimension 7, five cycle degrees of freedom, six unimodular five-face chord minors, all 15 four-face subsets of rank 4 with an indistinguishable configuration pair. Lean: `Q3Curl.lean`, kernel-checked by the `pgc-q3-curl` workflow on MathCan `96c3729` (workflow run passed). Simulation: the sixth face cannot localize a bias; cosine-only data has a global reflection ambiguity; wrapped Bianchi residuals are required (46.6% of configurations carry nonzero integer winding). Hardware receipts exist only for Q2; no Q3 follow-up QPU experiment is warranted by these results.

## Phase 6: BPD

Offline linearised study (`bpd/`, 140-circuit pool, 400 trials per strategy and budget; 17 tests pass). At a 6.4 s matched-cost cap: weak-direction recovery random 0.21, fixed 0.17, D-opt 0.87, D-opt per second 0.89, BPD 0.88; held-out RMSE 0.163, 0.180, 0.070, 0.069, 0.102; information gain 3.8, 3.3, 6.2, 6.2, 5.1 nats/s. **BPD does not beat conventional Fisher D-optimal design** on recovery, held-out error or total information per second; it wins only on weak-subspace information per second (5 of 6 budgets) by choosing cheaper Ramsey circuits, and ungrouped weak-trace greedy matches it, so the grouping adds nothing here. The linear model overstates Ramsey m=8 rows. Information-driven selection beats random and fixed designs by a wide margin.

## Which parameters are identifiable (current answer)

* From existing archived counts and local sensitivity only: with all sites six restricted parameters are locally identifiable; the 14-parameter model has rank 12, with the two target endpoint phases on 22-23 and 4-5 exactly unobserved. Local rank does not establish physical identifiability.
* Everything beyond that depends on counts from QC-26A and the Ramsey blocks.

## Next most informative QPU experiment

The Ramsey q23 block (13.19 s). It is the only planned job that directly measures the two endpoint phases that the existing data cannot observe, resolves the even-repetition alias, and gives forward/reverse agreement on 16-23 and 22-23, the edge where the retrospective ZZ is largest. q5 and QC-26A follow it; QC-26B is last.

## Ledger

| Item | Value |
|---|---|
| QPU seconds consumed this session | 0 |
| Remaining balance | 376 s |
| Job IDs | none |
| tensor-lab commit | `8a0a9ac` (offline phases); later commits listed in the final status |
| MathCan commit | `96c3729` |

## Hardware results (2026-11-27, ibm_kingston, prospective of nothing: exploratory single datasets)

| Job | ID | Circuits | Provider usage | Status |
|---|---|---:|---|---|
| QC-26A | dc4n9ads2omc738fm84g | 82 | 224 s to 240 s consumed (16 s) | counts archived, identified |
| Ramsey q5 | dc4ne0thdvas73bq1hq0 | 66 | 9 s reported | counts archived, identified |
| Ramsey q23 | none | 66 | 0 | refused twice by the calibration-change gate; third attempt blocked by the permission classifier |

Remaining allocation after q5: 345 s. Between the QC-26A receipt (360 s remaining) and the q5 `usage_before` (354 s) 6 s were consumed by something outside these receipts, most likely an earlier build and submit attempt by the owner; it is not attributed here.

**QC-26A (200 bootstraps, 16 parameters, single dataset):** all four SX offsets are consistent with zero (point estimates -0.005 to +0.001 rad, bootstrap SE about 0.003), far below the retrospective fitted SX_5 of -0.019 and SX_23 of 0.1 (the 0.1 was at the fit bound). Conditional phases: 5-6 (stored as 6_5) delta +0.033 [0.021, 0.044]; 16-23 delta -0.026 [-0.038, -0.015]; 4-5, 22-23 and 4-3 consistent with zero; 22-21 flagged (xy-linearity, relaxation) and inconclusive. Endpoint phases of the probe: alpha on qubit 5 (probe of 6->5) +0.176; alpha on qubit 23 (probe of 16->23) -0.075; qubit 23 on 22-23 -0.033. Decision counts: 9 consistent with zero, 5 nonzero candidates pending replication, 2 inconclusive. These are not accepted as nonzero without a fresh independently shuffled block.

**Ramsey q5 (200 joint-bitstring bootstraps with SPAM refit):** edge 5-6 delta +0.032 [0.024, 0.041] and alpha_5 +0.184 [0.172, 0.197]; edge 4-5 delta -0.0005 [-0.009, 0.007] (zero), alpha_4 -0.024 [-0.035, -0.013], alpha_5 0.000 [-0.011, 0.011]. Forward and reverse deltas agree on both edges (gap CIs contain 0). Odd repetitions break the even-only alias (profile chi-square gap 1.4e4 and 3.1e4 versus 0 without them). Readout joint correlation 0.0025.

**Reading:** the independent 5-6 conditional phase from two separate experiments agrees (0.033 and 0.032) and is about a third of the retrospective +0.1; the retrospective +-0.1 effective ZZ is not reproduced as an independent conditional phase on 5-6 or 16-23 (0.033, -0.026). A large endpoint-local phase on qubit 5 along the 5-6 coupler (0.18 rad) is a new, previously unobservable term; the same qubit 5 shows 0.000 on the 4-5 coupler, so the term is coupler-specific rather than a fixed qubit offset. Whether these terms explain the Kingston residuals is untested until q23, replication and the QC-26B prospective run.
