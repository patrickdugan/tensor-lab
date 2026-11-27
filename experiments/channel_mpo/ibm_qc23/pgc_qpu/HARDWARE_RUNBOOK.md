# PGC-QPU hardware runbook (owner steps)

Nothing in this directory creates an authorization file or submits a job without one. Each job is its own gate: fresh build, owner-written authorization naming the exact `plan_sha256`, submit within 15 minutes of the build, provider usage recorded before and after.

Historical pre-acquisition balance verified 2026-11-26 15:52 UTC: 224 s consumed of 600 s, 376 s remaining (rolling 28-day window). Target spend 56.63 s across three jobs; each job's ceiling below is a check on this tool's own estimate, not a provider limit.

| Job | Command (from `experiments/channel_mpo/ibm_qc23`) | Estimate | Ceiling to authorize |
|---|---|---:|---:|
| QC-26A diagnostic (82 circuits) | `python qc26/qc26a_compile.py build --out OUT` / `submit` | 16.39 s | 20 |
| PGC Ramsey block q23 (66 circuits) | `python pgc_qpu/compile_readonly.py --block q23 --out OUT` / `python pgc_qpu/submit_ramsey.py submit --out OUT --authorization FILE` | 13.19 s | 15 |
| PGC Ramsey block q5 (66 circuits) | same with `--block q5` | 13.19 s | 15 |
| QC-26B prospective evaluation | see `qc26/QC26B_PROSPECTIVE_PROTOCOL.md`; parameters come from the two jobs above and are committed with `frozen.json` first | 13.86 s | 15 |

## One job end to end (example: q23)

1. Build and read the hash (about 25 s):

```powershell
python pgc_qpu/compile_readonly.py --block q23 --out pgc_qpu/results_local/run_q23
python pgc_qpu/submit_ramsey.py check --out pgc_qpu/results_local/run_q23
```

2. Owner writes the authorization with the printed `plan_sha256` (replace `HASH`; this file stays local and is gitignored under `results_local`):

```powershell
@'
{"authorized": true, "backend": "ibm_kingston", "plan_sha256": "HASH", "max_qpu_seconds": 15,
 "authorized_by": "owner", "acknowledge_estimate_not_hard_limit": true, "checked_current_allocation": true}
'@ | Set-Content -Encoding utf8 pgc_qpu/results_local/auth_q23.json
```

3. Submit within 15 minutes of step 1:

```powershell
python pgc_qpu/submit_ramsey.py submit --out pgc_qpu/results_local/run_q23 --authorization pgc_qpu/results_local/auth_q23.json
```

4. Analyze (two blocks together once both exist):

```powershell
python pgc_qpu/ramsey_identify.py --compiled pgc_qpu/results_local/run_q23/compiled.json pgc_qpu/results_local/run_q5/compiled.json --counts pgc_qpu/results_local/run_q23/counts.json pgc_qpu/results_local/run_q5/counts.json --out pgc_qpu/results_local/identify.json
```

The adapter refuses (before any job exists): missing or mismatched authorization, ceiling above the 15 s block cap or below the estimate, missing acknowledgements, build older than 15 minutes, any calibration change since the build, any changed circuit duration, remaining allocation below the ceiling, any changed artifact hash. A started marker (`submission_started.json`) prevents automatic resubmission after an ambiguous failure: check the account's job list before deleting it.

After each acquisition commit `execution_receipt.json`, `counts.json`, `job.json`, `calibration_build.json`, `calibration_submission.json`, `compiled.json` and `usage_*.json` from the run directory to `receipts/`, and record job id, provider-reported usage and the remaining balance.

## Order and dependencies

1. QC-26A diagnostic first: it supplies independent SX offsets and conditional phases. It must not touch kicked-Ising counts.
2. PGC Ramsey q23, then q5 (separate authorizations). If the exploratory fit supports the model, a fresh independently shuffled replicate block needs a new estimate and approval.
3. QC-26B: write `params.json` from the QC-26A and Ramsey estimates, build, freeze with `qc26b_freeze.mjs`, commit and push `frozen.json` and `params.json`, then authorize and submit. Live submission now requires v3, a published freeze commit, frozen SHA and provider allocation checks; see the updated QC-26B protocol. C3Z (endpoint-local Z) is a declared extension scored beside, never instead of, the original C3.
4. PGC-QPU02 (proposal in `qpu02/qpu02_plan.json`, 30 circuits, 18.97 s at 2048 shots; about 11.0 s at 1024) needs its own adapter and budget review before any build. Not part of the 56.63 s.

Latest archived allocation: 336 s after q23 (November 27); recheck live before each job. Review branches and old v1/v2 freezes are not authorization to acquire. QPU02 requires regeneration of the corrected GHZ lower-bound simulation receipt and a reviewed hardware adapter.
