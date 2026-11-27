# PGC-QPU04 campaign report (2026-11-27)

**Status: hardware phase not started. No new QPU job, no calibration count and no evaluation count exists. No model prediction has been frozen. The offline phase is complete and verified.**

Registration: `PGC_QPU04_PROTOCOL.md` (registration v1). The historical Kingston acquisitions (jobs dc4n9ads2omc738fm84g, dc4ne0thdvas73bq1hq0, dc4ngn2nj2vs73dkdrdg) are unchanged.

## 1. Offline regression suites

All run on the PR branch (`pgc/qpu04-final-prospective-20261127`, PR 36), Windows, Python 3.12.10, Node 24.19.0, qiskit 2.6.2.

| Suite | Result |
| --- | --- |
| `pgc_qpu/audit.mjs --check` | PASS |
| `pgc_qpu/ramsey_campaign.mjs --check` | PASS |
| `analysis/jacobian_svd_check.py` | exit 0 |
| pytest `test_pgc_ramsey`, `test_ramsey_modelcheck`, `qpu02/test_qpu02` | 49 passed |
| `test_qc26b_prospective` | 26 tests OK (370 s) |
| `test_qpu04_calibration` | 3 tests OK |
| `test_qpu04_drift.mjs`, `test_qpu04_pipeline.mjs` | PASS (synthetic) |
| `test_qc26a_identify` | OK |
| BPD-QPU02 / QPU03 / QPU03B replays | passed |
| `qpu04_bpd_benchmark.mjs` | ran, exit 0 |
| `qc26b_selftest` vs dense check | 48 cases, 96 evaluations, worst gap 4.9e-15, receipt byte-identical |

## 2. CI failures on the PR head and their repair

CI on PR head 35fa92d had two failing jobs on ubuntu/Node 22. Both reproduced on Linux (WSL Ubuntu 24.04, Node 18) and neither depended on QPU04 code.

1. **PGC-QPU01 / nonlinear BPD replay.** `bpd_qpu02.mjs` scored alias recovery with `q >= .5`. For the fixed-even and random baselines the posterior is often exactly 0.5 up to the last bit, and the committed receipt had been decided by rounding that differs between runtimes (7 of 960 records flipped on Linux). Scoring now treats a marginal within 1e-9 of 0.5 as recovered, which can only raise baseline alias rates. `pilot_32.json` was regenerated; 219 record fields changed, all `alias_both`/`alias_fraction` for fixed-even and random policies. RMSE fields and every dense, exact-grouped and alias-only result are unchanged, and `BPD_QPU02_REPORT.md` states the convention. The same replay is now green on Windows and Linux.
2. **QC-26B channel engine / replay verify.** `qc26b_verify.py` compared the fitted ZZ phases with exact equality (`0.10000000000000002` vs `0.1`). Its own docstring says floats are compared to 1e-9; the phases now use the same tolerance. This check already failed on main at 78cf47e.

## 3. Live allocation

Read-only query at 2026-11-27 (instance Blot, plan open): consumed 304 s, limit 600 s, **remaining 296 s**, `usage_limit_reached` false. ibm_kingston operational, 0 pending jobs. The authorization ceilings total 65 s against 296 s remaining. The reported usage period ended 14:56 UTC and the query ran after that, so the provider may roll the period; either way the margin is sufficient.

A verification-only QC-26A build with registered seed 261127041 completed (82 circuits, estimate 16.38688 s, cap 20 s, plan sha256 `b6f3b78a...54dd`). It was built locally under `results_local/qpu04_verify_build` and is not for submission: it will exceed the 15-minute freshness gate and the registered run requires a fresh build.

## 4. Why the hardware phase stopped

Every submit path (`qc26a_compile.py submit`, `submit_ramsey.py`, `qc26b_prospective_plan.py submit`) requires an authorization file written by the account owner that names the exact plan sha256 of a just-compiled build and attests `acknowledge_estimate_not_hard_limit` and `checked_current_allocation`. The protocol states that none is produced by the agent. The plan hash exists only after the build, and the build expires after 15 minutes, so the approval cannot be given in advance. Nothing was submitted and nothing was retried.

## 5. Owner procedure to complete the campaign

From `experiments/channel_mpo/ibm_qc23`, one block at a time (build, authorize, submit before the next block), using the registered seeds:

    python qc26/qc26a_compile.py build --out pgc_qpu/results_local/qpu04/qc26a --ordering-seed 261127041

Read `plan_sha256` from the build output, write the authorization (ceiling 20 for QC-26A, 15 for each Ramsey block and for evaluation):

    {"authorized": true, "backend": "ibm_kingston", "plan_sha256": "<hash printed by build>", "max_qpu_seconds": 20, "authorized_by": "<owner>", "acknowledge_estimate_not_hard_limit": true, "checked_current_allocation": true}

then `python qc26/qc26a_compile.py submit --out pgc_qpu/results_local/qpu04/qc26a --authorization <file>`. Repeat with `pgc_qpu/compile_readonly.py --block q5 ... --ordering-seed 261127042` and `--block q23 ... 261127043` plus `pgc_qpu/submit_ramsey.py`. Steps 2 to 4 of the protocol (fit reports with 400 bootstraps, `qpu04_calibration.py prepare/verify`, QC-26B build, freeze, push, owner authorization naming `frozen_sha256` and `freeze_commit`, submit, score) follow unchanged.

## 6. Results tables

Not available. The tables for the calibration blocks, six model predictions, C3 primary endpoint, C3Z-D comparisons, truncation ledgers and provider usage will be filled only from archived raw counts after the gates above are satisfied. No negative or null hardware result exists yet, and none has been withheld.
