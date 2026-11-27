# PGC-QPU04 campaign report (2026-11-27)

**Outcome: executed end to end on ibm_kingston. The primary C3 endpoint FAILED (criteria a and c). The declared C3Z extension and the C3Z-D extension also failed criterion c. C3Z-D gave no incremental gain over static C3Z.** All numbers below come from archived raw counts; nothing was tuned, re-run or replaced after the evaluation counts arrived.

Registration: `PGC_QPU04_PROTOCOL.md` (registration v1). All receipts: `pgc_qpu/receipts/qpu04_live_20261127/`.

## 1. Execution record

| Step | Job | Submitted (UTC) | Est. s | Ceiling s | Provider usage s | Plan sha256 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| QC-26A (seed 261127041) | dc4qdhqnj2vs73dkhrpg | 15:50:34 | 16.39 | 20 | 11 | 1c788ff3430eb4a7... |
| Ramsey q5 (seed 261127042) | dc4qf65s2omc738fqksg | 15:54:02 | 13.19 | 15 | 9 | 3378d6a93f7ddbe9... |
| Ramsey q23 (seed 261127043) | dc4qftuq4uuc73a99hag | 15:55:38 | 13.19 | 15 | 9 | (see q23/compiled.json) |
| QC-26B seeds 51-54, evaluation | dc4qnoeq4uuc73a99t3g | 16:12:19 | 13.86 | 15 | 15 | 498b263512357f87... |
| **Total** | | | 56.63 | 65 | **44** | |

Allocation before the first job: 296 s remaining; after the last: 252 s (instance plan open, limit 600 s). Every job was built, authorized and submitted one at a time inside the 15-minute freshness gate, with no retries and no ambiguous submissions. Authorization files (`authorization/`) name the exact plan hash, the evaluation file also the frozen sha256 and freeze commit; `authorized_by` records the owner's chat instruction "DO IT" given after the gate procedure was explained. The three fresh jobs all follow the historical round (dc4n9ads..., dc4ne0th..., dc4ngn2n...), which the parameter-pack verifier checks.

Freeze order (verified from timestamps): parameter pack and calibration receipts pushed to main at a4a0efd; frozen.json (six model sets, sha256 `da7bcb56...a99c`) committed and pushed as `1f120d1250d49d6c91aa21b12215e6a90ff6564a` at 16:09:07 UTC; evaluation circuits submitted at 16:12:19 UTC. The submitter recomputed the published freeze and source bytes before submitting.

## 2. Calibration diagnostics and the 16-value parameter pack

All fits used 400 bootstraps (`analysis/`). Parameters are plug-in point estimates by the registered rule, with no thresholding.

| Coordinate | Value (rad) | | Coordinate | Value (rad) |
| --- | ---: | --- | --- | ---: |
| SX q3 | -0.008837 | | delta 4-5 | -0.005374 |
| SX q5 | -0.000461 | | alpha 4-5 (q4, q5) | -0.012796, +0.009069 |
| SX q21 | +0.000668 | | delta 5-6 | +0.023989 |
| SX q23 | -0.007505 | | alpha 5-6 (q5, q6) | +0.177269, +0.011471 |
| delta 16-23 | -0.023158 | | delta 22-23 | -0.006793 |
| alpha 16-23 (q16, q23) | +0.019002, -0.067892 | | alpha 22-23 (q22, q23) | +0.000034, -0.031288 |

Ramsey bootstrap on delta: 4-5 SE 0.0037 (CI includes 0), 5-6 SE 0.0041 (CI excludes 0), 16-23 SE 0.0036 (CI excludes 0; forward/reverse gap also excludes 0), 22-23 SE 0.0043 (CI includes 0). Point chi-square 25.3, 28.4, 14.1, 18.6. QC-26A decision counts: 7 NONZERO_CANDIDATE_PENDING_REPLICATION, 7 CONSISTENT_WITH_ZERO, 2 INCONCLUSIVE_FLAGS, `any_diagnostic_flag` true; SX crosstalk test max |z| 2.04 against the Bonferroni threshold 3.71. These are exploratory plug-in estimates, not independently established nonzero effects, and the residual model misspecification is reported rather than corrected. The q5 alpha of 0.177 rad on edge 5-6 is large for a plug-in value and sits inside the registered +/-0.4 rad bound.

## 3. Results (frozen predictions, evaluation counts, 24 circuits, 240 site observables)

Mean absolute error against measured <Z>:

| Group | n | Ideal | C0 | C1 | C2 | C3 | C3Z | C3Z-D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All sites | 240 | 0.03699 | 0.03627 | 0.03518 | 0.03541 | 0.03495 | 0.02943 | 0.02944 |
| Targets 5, 23 | 24 | 0.09953 | 0.10232 | 0.10089 | 0.10205 | 0.09307 | 0.04362 | 0.04399 |
| Controls 3, 21 | 24 | 0.02413 | 0.02678 | 0.02513 | 0.02603 | 0.02574 | 0.02503 | n/a |
| Other sites | 216 | 0.03004 | 0.02893 | 0.02788 | 0.02801 | 0.02849 | 0.02786 | n/a |
| Site 5 | 12 | 0.08800 | 0.09102 | 0.08884 | 0.08881 | 0.08197 | 0.06286 | n/a |
| Site 23 | 12 | 0.11105 | 0.11361 | 0.11294 | 0.11529 | 0.10417 | 0.02438 | n/a |

Relative MAE improvement over C0 and registered criteria:

| Model | Pooled | Targets | (a) >=10% both | (b) paired CI lower > 0 | (c) no other-site degradation > 0.005 | (d) ledger < 1% MAE | (e) freeze verified | Verdict |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| **C3 (primary)** | 3.64% | 9.04% | FAIL | PASS | FAIL (site 6, +0.0118) | PASS | PASS | **FAIL** |
| C2 | 2.36% | 0.26% | FAIL | FAIL (targets CI [-0.00083, 0.00132]) | PASS | PASS | PASS | FAIL |
| C3Z (extension) | 18.84% | 57.37% | PASS | PASS | FAIL (site 6, +0.0200) | PASS | PASS | FAIL |
| C3Z-D (extension) | 18.82% | 57.00% | PASS | PASS | FAIL | PASS | PASS | FAIL |

Paired bootstrap 95% intervals on MAE improvement (positive means better):

| Comparison | Pooled circuit CI | Targets circuit CI | Pooled path-cluster CI (8) | Targets path-cluster CI (4) | Label |
| --- | --- | --- | --- | --- | --- |
| C3 vs C0 | [0.00059, 0.00270], wins 19/24 | [0.00446, 0.01476], 12/12 | [-0.00014, 0.00300] | [0.00269, 0.01885] | INCONCLUSIVE_OR_NEGATIVE |
| C2 vs C0 | [0.00055, 0.00117], 20/24 | [-0.00083, 0.00132], 6/12 | [0.00060, 0.00106] | [-0.00143, 0.00197] | INCONCLUSIVE_OR_NEGATIVE |
| C3Z vs C0 | [0.00318, 0.01345], 20/24 | [0.03176, 0.08907], 10/12 | [0.00088, 0.01475] | [0.02417, 0.09846] | POSITIVE |
| C3Z vs C3 | [0.00225, 0.01117], 9/24 | [0.02450, 0.07689], 9/12 | | | |
| C3Z-D vs C0 | [0.00318, 0.01343], 20/24 | [0.03142, 0.08864], 10/12 | [0.00089, 0.01469] | [0.02371, 0.09799] | fails criterion c |
| C3Z-D vs static C3Z | [-0.00005, 0.00003], 9/9 ties 6 | [-0.00052, -0.00024], 0/12 | [-0.00005, 0.00003] | [-0.00056, -0.00019] | **INCONCLUSIVE_OR_NEGATIVE** |

Incremental drift label: INCONCLUSIVE_OR_NEGATIVE. The drift-smoothed mean was slightly worse than static C3Z on target sites (0 wins, 12 losses, about -0.0004 MAE), with both interval levels excluding zero in the wrong direction. The temporal regularizer did not help in this block, and no rule adjustment follows.

Site-level degradation: physical qubit 6 (n=6) worsens under C3 by +0.0118 and under C3Z by +0.0200 against C0; it is the only flagged site. Qubit 6 is the neighbour of qubit 5 on edge 5-6, which carries the +0.177 rad alpha estimate. Per-site tables for all 34 sites are in `qpu04_report.txt`.

The primary improvement is concentrated in the target sites and mostly comes from the Ramsey-informed edge terms: C3Z (adding endpoint-local Z corrections) cuts target MAE from 0.0931 (C3) to 0.0436, chiefly on site 23 (0.1042 to 0.0244). This is a declared extension and does not replace the failed primary result.

## 4. Truncation ledgers

Maximum discarded L1 mass and conservative bound per model set, against physical MAE:

| Model | Max L1 discard | Min bound | Ratio to MAE | Criterion (d, < 1%) |
| --- | ---: | ---: | ---: | --- |
| C3 | 5.634e-11 | 5.634e-11 | 1.61e-9 | PASS |
| C2 | 5.634e-11 | 4.902e-11 | 1.38e-9 | PASS |
| C3Z | 6.848e-11 | 6.848e-11 | 2.33e-9 | PASS |
| C3Z-D (worst case over all models) | 6.876e-11 | 6.876e-11 | 2.34e-9 | PASS |

Maximum live terms 53,753. Truncation is irrelevant to the outcome.

## 5. Negative and null results (none withheld)

- C3 primary: FAIL on criterion (a) (pooled 3.64%, targets 9.04%, both below 10%) and (c) (qubit 6 +0.0118). Its pooled path-cluster interval includes zero.
- C2: pooled gain 2.36%, targets 0.26%; target interval includes zero; C2 on site 23 is worse than C0.
- C3Z: strong target gain but fails (c); it is an extension and cannot rescue the primary.
- C3Z-D: no incremental benefit over static C3Z.
- QC-26A produced 2 INCONCLUSIVE_FLAGS decisions and the Ramsey point fits report chi-square of 14 to 28 (section 2) and the model-misspecification error carried by the plug-in parameters is not modelled in the scoring.

## 6. Limits and disclosures

One evaluation block on one device on one day; two diagnostic rounds do not give between-cycle uncertainty on evaluation MAE. The eight path clusters are within-block robustness, not independent calibration cycles. C3Z-D rests on a two-point Gaussian drift model with fixed process noise and uses submission time as a proxy. BPD results remain computational evidence only; the CPU benchmark (dense 224.0 ms, oracle 162.8 ms, learned audited 181.1 ms median, RSS 57.3, 57.5, 59.7 MiB) was run before this campaign and is unchanged. Lean kernel verification and CI status on the final commit were not re-confirmed as part of this run; Linux CI repairs from earlier in the campaign are in commit 36b3c95.

## 7. Offline phase and CI repairs

All offline suites passed before any acquisition (pytest 49, QC-26B prospective 26 tests, QPU04 calibration/drift/pipeline, BPD-QPU02/03/03B replays, QC-26B dense cross-check 48 cases). Two Linux CI failures were repaired first: BPD-QPU02 alias scoring treated exact posterior ties by last-bit rounding (a 1e-9 tolerance rule now applies and `pilot_32.json` was regenerated; only fixed-even and random baseline alias fields changed), and `qc26b_verify.py` used exact float equality for fitted phases (now 1e-9 tolerance).

## 8. Reproduction

Raw counts and scoring inputs are in `pgc_qpu/receipts/qpu04_live_20261127/{qc26a,q5,q23,eval}/counts.json`; `qpu04_report.json` and `qpu04_report.txt` are the deterministic scorer output (re-running reproduced the JSON byte for byte). Command: `node qc26/qc26b_prospective_eval.mjs --frozen <fresh>/frozen.json --build <fresh>/eval --params <fresh>/params.json --counts <fresh>/eval/counts.json --out <fresh>/qpu04_report.json` with the compiled QASM in `eval/circuits/`.
