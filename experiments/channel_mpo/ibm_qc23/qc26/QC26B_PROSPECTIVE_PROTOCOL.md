# QC-26B prospective protocol (frozen predictions, fresh counts)

Purpose: test the heterogeneous native-gate channel model (engine `qc26b_engine.mjs`, models C0..C3)
on counts that did not exist when the parameters and predictions were fixed. The retrospective replay
(`qc26b_replay.mjs`) fitted on counts it also scored; this protocol removes that.

## Prospective versus retrospective

- Retrospective: any score computed on counts that existed before the parameters were fixed (all 528 QC-23 observables, QC-25C,
  QC-26A calibration counts). Those numbers are supporting evidence only.
- Prospective: a score on counts acquired after `frozen.json` (parameters + per-site predictions + hashes) was committed and pushed, on
  circuits with seeds 51-54 that no earlier run used, scored by code whose hash is in `frozen.json`. Only this counts for acceptance.
- Dry run: `qc26b_synth_counts.py` output. It is generated from the model itself and proves the code path only. A dry-run PASS is
  never evidence about hardware; every dry-run report and receipt carries the label SYNTHETIC DRY RUN.

## Design

ibm_kingston, 24 kicked-Ising circuits (`qc23_nonclifford.make_kicked_ising`), widths 8 and 12, depths 2,3,4, seeds 51-54, 2048 shots,
`transpile_on_path` at optimization_level 0, native `rz/sx/cz`. Every qubit of every circuit is measured and scored (bit k is
`physical_path[k]`). One path per (width, seed), shared by the three depths:

| seed | class | content |
|---|---|---|
| 51 | targets_end | qubits 5, 23 and controls 3, 21; coupler 5-6 forbidden so 5 sits at a path end (earlier runs: 5 interior between 4 and 6) |
| 52 | targets_23_end | qubits 5, 23, control 3; couplers 22-23 and 23-24 forbidden so 23 is an end (earlier: interior between 16 and 22), 5 at a new interior position |
| 53 | controls | control 21, qubits 5 and 23 excluded |
| 54 | pure_control | 3,4,5,6,16,21,22,23,24 excluded |

Control 3 cannot be isolated from 5 and 23 at width 8 or 12 on this topology (its pocket has 5 qubits once 5 and 23 are excluded),
so it appears only in the target classes. Path selection is `search_line` in `qc26b_prospective_plan.py`, the QC-23 beam search with
include/exclude/forbid-edge constraints and the same cost function. The paths and the comparison with earlier positions and
neighbours are recorded in `compiled.json` under `protocol`.

## What is frozen, and when

1. `build` (read-only IBM connection, compile only): `compiled.json` (plan hash, compile time, QASM hashes, estimate), `circuits/*.qasm`
   (LF bytes), `calibration_snapshot.json`, `compiled.qpy`. The calibration snapshot of this build is the one the predictions use.
2. Parameter file `params.json` `{sx:{phys:rad}, cz:{"a_b":{delta, alpha:{phys:rad}}}}` comes from independent calibration data
   (QC-26A style), never from kicked-Ising counts, and is fixed before step 3.
3. `qc26b_freeze.mjs freeze` writes `frozen.json`: parameter contents, per-circuit QASM sha256, snapshot sha256, `compiled.json` sha256,
   plan hash, sha256 of `qc26b_engine.mjs`, predictions (ideal, C0..C3, truncation ledger) for every circuit and site, and a top-level sha256
   over the canonical JSON of everything else. Text is hashed after CRLF to LF normalisation.
4. `frozen.json` and `params.json` are committed and pushed before the job is submitted. The commit hash is the timestamped proof of order.

The evaluator refuses to run (exit 2, criterion (e) fails) unless the frozen file's self-hash, the supplied parameter file, the engine hash,
the build hashes and every prediction (recomputed from the build dir and parameter file, tolerance 1e-12) agree.
The Python `submit` runs the same verification before submitting.

## Acceptance criteria

Evaluated for C3 (primary) and C2 (secondary) against C0 on the fresh counts. MAE is the mean absolute difference between predicted and
measured single-site `<Z>`, one observable per measured qubit.

- (a) Relative MAE improvement over C0 at least 10 percent, both pooled over all sites and over the targeted sites (physical 5 and 23).
- (b) The 95 percent bootstrap CI (paired over circuits, equal circuit weights, 20000 resamples, fixed seed 26112501) of MAE(C0) - MAE(model) has a positive lower bound, both pooled and targeted.
- (c) No physical site other than 5 and 23 has its MAE rise by more than 0.005 relative to C0 (controls 3 and 21 are included in this check).
- (d) Numerical error substantially below physical error: "substantially" means the worst-case truncation-ledger bound (`min_bound`, the smaller of the
  discarded-l1 bound and the 2^(n/2) sum(eta) bound) is below 1 percent of the model's pooled MAE. The report also gives the l1 and old-bound ratios.
- (e) No tuning: the freeze check passed.

The report states PASS or FAIL per criterion and per model. Negative results (a model worse than C0 pooled or on 5/23, flagged sites, CI crossing zero)
are printed as such. The tests in `test_qc26b_prospective.py` confirm that counts drawn from C3 pass and counts drawn from C0 fail (a).

## Command sequence (from `experiments/channel_mpo/ibm_qc23/qc26`)

No authorization file is created or supplied by these tools. The account owner writes it.

```
# 0. offline checks
python -m unittest test_qc26b_prospective -v

# 1. build (read-only IBM connection, compile only; prints plan_sha256 and the QPU estimate, which must be <= 15 s)
python qc26b_prospective_plan.py build --out results_local/qc26b_prospective

# 2. freeze (about one minute) and commit BEFORE any submission
node qc26b_freeze.mjs freeze results_local/qc26b_prospective params.json results_local/qc26b_prospective/frozen.json
node qc26b_freeze.mjs verify results_local/qc26b_prospective params.json results_local/qc26b_prospective/frozen.json
#    commit and push frozen.json, params.json and compiled.json (the small receipt) to main; record the commit hash

# 3. owner step, outside these tools: write authorization JSON with authorized=true, backend=ibm_kingston,
#    plan_sha256=<hash printed in step 1>, max_qpu_seconds>=<estimate>, authorized_by=<owner>

# 4. submit (within 15 minutes of step 1; otherwise rebuild, refreeze, recommit; a new build has a new plan hash)
python qc26b_prospective_plan.py submit --out results_local/qc26b_prospective \
    --frozen results_local/qc26b_prospective/frozen.json --params params.json --authorization /path/to/owner-auth.json
#    refuses stale builds, changed calibration, hash mismatches, missing frozen/params verification, missing authorization,
#    and a second attempt (exclusive submission_started.json). Writes counts.json. Never retried automatically.

# 5. evaluate
node qc26b_prospective_eval.mjs --frozen results_local/qc26b_prospective/frozen.json --build results_local/qc26b_prospective \
    --params params.json --counts results_local/qc26b_prospective/counts.json --out results_local/qc26b_prospective/report.json
```

Dry run of steps 2 and 5 without hardware: `python qc26b_synth_counts.py --frozen frozen.json --out counts_synth.json`, then evaluate. The receipt of one such run is
`receipts/qc26b_prospective_dryrun.json`.

## Limits

- The 15 minute window covers build, freeze, commit, push and the owner's authorization. If any step overruns, rebuild; the parameter file stays the same, the plan hash changes.
- The QPU figure is an estimate (`qc23_access.estimate_qpu_seconds`, 250 us repetition delay), not a provider-enforced limit.
- Predictions use the calibration snapshot of the build; a calibration cycle between build and submit is refused.
- Fresh seeds do not make the circuits independent of the model class: the same coupler pairs and qubits recur. Pure-control paths test transfer to qubits the parameters never touch.
