# PGC-QPU04: final prospective channel reconstruction (registration v1)

**Status: code and offline tests only. No final evaluation counts exist in this campaign.** The historical Kingston acquisitions are pinned to tensor-lab d7c0946 / merge 58e6651. The registered QC-26B primary endpoint and original C0–C3Z definitions are unchanged. C3Z-D (code key C3ZD) is a declared exploratory extension, not a replacement for C3. Freeze before submitting any final evaluation circuit. Publish null and failed results.

## Acquisition

Use IBM Kingston, fresh independently shuffled calibration blocks, no QPU calls from tests. The prior round's three job IDs are dc4n9ads2omc738fm84g, dc4ne0thdvas73bq1hq0, dc4ngn2nj2vs73dkdrdg. For every block the new calibration snapshot timestamp must be later than its historical counterpart; fresh jobs must follow the entire historical round. The three new blocks need not share the exact calibration timestamp: each estimate has its own submission-time proxy, and the final build snapshot must be at least as recent as each source. The final build must be within four hours of every fresh submission. Recheck live usage; 336 s was the archived balance, not a current quote.

| Job | Estimated seconds | Ceiling (owner authorizes separately) |
| --- | ---: | ---: |
| QC-26A | 16.39 | 20 |
| Ramsey q5 | 13.19 | 15 |
| Ramsey q23 | 13.19 | 15 |
| QC-26B unseen seeds 51–54 | 13.86 | 15 |
| Total | 56.63 | 65 |

No assumption that the estimates are provider hard limits. Stop on stale calibration, authorization mismatch, unavailable allocation, ambiguous submission, or failed provenance. Never automatically retry an ambiguous job. Owner must approve the exact newly compiled plan hash, and for evaluation the published freeze hash. No authorization file is produced here.

## Fixed parameter selection

Use four signed SX offsets from QC-26A (physical 3,5,21,23), and bidirectional Ramsey point estimates for delta and both endpoint alpha terms on 4-5,5-6,16-23,22-23. Exactly 16 values. Four Ramsey edges take precedence over QC-26A conditional phase estimates, avoiding double counting. Other couplers have no fitted coherent correction. In particular 4-3 and flagged 22-21 are not added. Use point estimates even if an interval includes zero; no outcome-dependent selection or thresholding. Record diagnostic flags and chi-square; these are exploratory plug-in estimates, not claims of independently established nonzero effects. Refuse nonfinite values, missing bootstrap SE, SX outside +/-0.1 rad or CZ terms outside +/-0.4 rad. Report the residual model misspecification. No fit to kicked-Ising evaluation counts. Use 200 or more bootstrap replicates for each diagnostic (register 400 for new acquisitions).

The primary C0/C1/C2/C3/C3Z models use only the fresh round. Historical counts are used solely by C3Z-D. For each signed phase coordinate, prior mean is the historical estimate; prior variance is max(SE^2, floor^2). Propagate variance for elapsed hours between the independent job submissions by q^2*dt. Update with the fresh estimate, whose observation variance is max(SE^2, floor^2). Fixed process SD per sqrt hour: 0.003 rad SX, 0.010 rad CZ; SE floors 0.003 and 0.005 rad respectively. These are predeclared design choices, not learned physical drift rates. The Kalman posterior mean is used for the same C3Z simulator; forecast variance at build time is reported. This is an independent-coordinate Gaussian small-branch approximation; it omits covariance and model uncertainty. It predicts a single block with no mid-block measurement or adaptation, and uses submission time as a proxy. Neither two rounds nor one evaluation block estimates a population-level drift process. An improvement supports this specific temporal regularizer only.

## Reproduction on the owner workstation

From experiments/channel_mpo/ibm_qc23:

1. Run the offline CI suites before any device acquisition. Confirm current allocation. Build each diagnostic block in a fresh local directory (the registered permutation seeds are mandatory):

       python qc26/qc26a_compile.py build --out pgc_qpu/results_local/qpu04/qc26a --ordering-seed 261127041
       python pgc_qpu/compile_readonly.py --block q5 --out pgc_qpu/results_local/qpu04/q5 --ordering-seed 261127042
       python pgc_qpu/compile_readonly.py --block q23 --out pgc_qpu/results_local/qpu04/q23 --ordering-seed 261127043

   Build/authorize/submit **one at a time**, within the existing freshness gate; do not build all three and let them age. For each, the account owner supplies the required authorization file. Use the existing QC26A and Ramsey submit commands. Do not print credentials. Keep all local compiled artifacts and archive the manifests, job.json, calibration_build.json, calibration_submission.json, execution_receipt.json, usage_before.json and counts.json into separate subdirectories under pgc_qpu/receipts/qpu04_<unique_label>/.
2. Generate the independent fit reports in that receipt root's analysis directory:

       python qc26/qc26a_identify.py <fresh>/qc26a/counts.json <fresh>/analysis/identify_qc26a_live.json --bootstraps 400 --seed 261127041
       python pgc_qpu/ramsey_identify.py --compiled <fresh>/q5/compiled.json --counts <fresh>/q5/counts.json --out <fresh>/analysis/ramsey_q5_identify.json --bootstraps 400 --seed 261127042
       python pgc_qpu/ramsey_identify.py --compiled <fresh>/q23/compiled.json --counts <fresh>/q23/counts.json --out <fresh>/analysis/ramsey_q23_identify.json --bootstraps 400 --seed 261127043
       python qc26/qpu04_calibration.py prepare --fresh <fresh> --out <fresh>/params.json
       python qc26/qpu04_calibration.py verify --params <fresh>/params.json

   QC26A uses the original logical proposal to fit by circuit ID; the compiled permutation is separately verified. Fresh Ramsey reports bind canonical counts hashes. Historic Ramsey reports lack the new count-hash field; their archived job/plan IDs, exact rows, counts and report bytes are checked, without pretending the historical fits were rerun. Archive all raw sources, reports and params before final acquisition. If the diagnostic model check flags a fit, report it. Do not quietly switch fit methods.
3. After fits are finished, build the prospective QC26B circuit set and freeze with the unchanged CLI, using the new parameter pack:

       python qc26/qc26b_prospective_plan.py build --out pgc_qpu/results_local/qpu04_eval
       node qc26/qc26b_freeze.mjs freeze pgc_qpu/results_local/qpu04_eval <fresh>/params.json <fresh>/frozen.json
       node qc26/qc26b_freeze.mjs verify pgc_qpu/results_local/qpu04_eval <fresh>/params.json <fresh>/frozen.json

   The v3 freeze's self-hash covers both model sets and their independently recomputed truncation ledgers; the code manifest includes this protocol, parameter pack verifier and drift implementation. Commit and push frozen.json, params.json, and all referenced source receipts to main before evaluation submission. Store compiled manifest, QASM, calibration snapshot and QPY (or documented artifact location with byte digests). Owner authorization must name exact plan_sha256, frozen_sha256, freeze_commit, freeze_path, max_qpu_seconds (<=15), backend, authorized_by and existing acknowledgements. The live submitter recomputes source-pack contents and checks the published source bytes against the frozen hashes. If build is older than 15 min or calibration changes, rebuild/refreeze/push/reauthorize. Do not change parameters on a recompile.
4. Submit only under the existing gate:

       python qc26/qc26b_prospective_plan.py submit --out pgc_qpu/results_local/qpu04_eval --frozen <fresh>/frozen.json --params <fresh>/params.json --authorization <owner-auth.json>

   Archive raw evaluation counts, job.json, submission calibration, provenance and execution receipt. Score:

       node qc26/qc26b_prospective_eval.mjs --frozen <fresh>/frozen.json --build pgc_qpu/results_local/qpu04_eval --params <fresh>/params.json --counts pgc_qpu/results_local/qpu04_eval/counts.json --out <fresh>/qpu04_report.json

   Verify job ID, plan/frozen hashes and submitted_at after published freeze, and actual provider completion; synthetic dry runs never count as hardware evidence. If final data exist, do not acquire a replacement because the result failed.

## Registered endpoints and paper decision

C3 remains primary. Original criteria (a)-(e) are untouched: >=10% pooled and targeted MAE improvement vs C0, positive equal-circuit paired lower bounds, non-target site degradation <=0.005, numerical truncation bound <1% of MAE, and frozen integrity. Report C2 and C3Z separately and C3Z vs C3. C3Z-D gets the same criteria against C0 plus paired circuit and path-clustered comparisons against fresh static C3Z. Incremental drift label POSITIVE requires strictly positive lower bounds for both pooled and targeted comparisons at both circuit and path-cluster levels and no extra non-target degradation >0.005; otherwise INCONCLUSIVE_OR_NEGATIVE. Path clusters are width/seed combinations with all depths kept together. The eight clusters are a within-block robustness analysis; they are not independent calibration cycles. Pooled point estimates weight sites, the original circuit bootstrap weights circuits, and the path-cluster bootstrap weights observations. No replacement of a primary failure by an exploratory extension. Two independent diagnostic rounds do not give between-cycle uncertainty on evaluation MAE; disclose this.

BPD results are computational evidence only. Before paper submission measure end-to-end CPU and peak memory for dense vs grouped design, including group learning/audit, predictive likelihood table construction and all posterior updates, on the same fixed workload. Preserve the oracle comparator and any negative runtime result. CI and Lean kernel verification must finish before paper claims. A hardware-independent paper can report the exact and offline results if device execution is unavailable.


## Offline evidence already collected

The synthetic parameter pilot passed 15 checks. Across 512 trials per family (16 coordinates per trial), latest vs C3Z-D RMSE in radians was 0.005369 vs 0.004738 (stationary), 0.006438 vs 0.006139 (matched random walk), and 0.006525 vs 0.008375 (unmodelled jump). Nominal 95% forecast coverage was 98.63%, 95.03%, and 83.52%. The jump result documents sensitivity to the fixed process noise; no adjustment to the rule follows from it. See receipts/qpu04_synthetic_parameter_pilot.json.

A synthetic integration smoke using two archived 8-qubit QASM circuits verified the extended freeze and rejected a rehashed modified prediction. These were synthetic parameters and measurements, not new hardware. See receipts/qpu04_synthetic_integration_smoke.json.

The separate fixed-workload CPU benchmark is pgc_qpu/bpd/nonlinear/qpu04_bpd_benchmark.mjs with its adjacent receipt. Five separate process runs per arm, 1024 states, 24 trajectories, eight acquisition decisions per trajectory. Median CPU 224.025 ms dense, 162.841 ms exact oracle, 181.084 ms learned audited grouping. Median RSS 57304, 57496, 59736 KiB respectively. All action digests and posterior checksums matched. This is a 1.24x learned-vs-dense end-to-end workload speedup, with higher memory and no advantage over oracle. Timing includes table construction, learning, audit and posterior updates but excludes Node startup, and is descriptive on one shared Linux EPYC runtime. The benchmark does not measure actual IBM performance or a learned neural BPD.
