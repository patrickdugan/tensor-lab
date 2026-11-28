# HAD-QFT-QPU04 results (28 November 2026)

Frozen compiler commit `6c23ab942a44feb5aa50e34ee660a6666cd59c83`; parent QPU03 `9547024409ef7594cfea5230e26125d176dbf8db`. Backend `ibm_kingston`. Seven data qubits (physical), no workspace, encoded dimension 128 of a transform of ambient dimension 2^127 (no 127-qubit register exists). No claim of general computational advantage.

## Verdict

**The registered replicated positive primary result is NOT met.** Primary endpoint: edge phase-contrast difference D - A, joint 95% Bonferroni-Hoeffding shot-only radius sqrt(2 ln(4/0.05)/1024) = 0.0925.

| Region | Job | C_A | C_D | D - A | Joint interval | Lower > 0 |
|---|---|---:|---:|---:|---|---|
| 0 `[71,72,73,58,70,74,79]` | `dc54l02nj2vs73dl0cdg` | 0.0859 | 0.1572 | +0.0713 | [-0.0212, +0.1638] | no |
| 1 `[123,136,143,122,124,142,144]` | `dc54v96q4uuc73a9p2d0` | 0.0605 | 0.1533 | +0.0928 | [+0.0003, +0.1853] | yes (by 0.0003) |

Descriptive (not registered): D exceeds A at all five probes in both regions (10 of 10); the unregistered mean edge difference is +0.082 with an unadjusted shot-only interval [+0.022, +0.142]; mean total variation distance 0.82/0.84 (A) vs 0.73/0.76 (D); mean Hellinger fidelity 0.18/0.16 vs 0.27/0.25; preparation and SPAM control success 0.93-0.95 in both arms. Absolute contrast stays low (D about 0.15 against an ideal 1). Intervals are shot noise only; region and calibration effects are not covered.

## Native compilation (frozen routed circuits, level 0, no remapping)

| | Arm A | Arm D |
|---|---:|---:|
| Native CZ (= total two-qubit gates; only CZ) | 276 | 199 |
| SX / RZ / X | 970 / 1731 / 7 | 630 / 1225 / 19 |
| Two-qubit depth / depth | 263 / 2088 | 193 / 1316 |
| Scheduled edge circuit | 41.1 us | 27.1 us |
| Calibrated success estimate (core+prep / with readout), region 0 | 0.564 / 0.502 | 0.670 / 0.596 |
| Same, region 1 | 0.555 / 0.509 | 0.658 / 0.604 |
| Native vs routed logical unitary, max entry | 6e-14 | 4e-14 |
| Spectral-norm error vs exact shell transform | 0.001240603674 | 0.001240603674 |

Analytic bound 0.001639601244. All acceptance gates passed in both regions: seven physical data qubits, zero workspace, all native operator checks, every instruction and edge supported by the live target, native CZ not above the frozen routed CX, arm D below 200 CZ. QPU03's recorded 320 native CZ at b=7 and QPU04's offline routed 276 and 199 are different quantities; the verified native comparison is 276 versus 199.

## Offline reproduction

24 tests of `test_qpu04.py` pass with none skipped (28 with `test_qpu04_results.py`). Resource table reproduced exactly (previous/new logical, previous/new routed): b=5 70/68, 90/86; b=6 112/98, 157/136; b=7 188/138, 276/199; b=8 330/192, 494/301. Selective-rotation identity checked independently with Qiskit semantics on random complex superpositions and full operators for 1-8 controls (errors below 3e-15; relative flips exactly monomial).

## Budget and receipts

| Stage | Plan SHA-256 | Calibration (+08:00) | Estimate | Charge |
|---|---|---|---:|---:|
| Region 0 | `ceb4ef50b9a51d1a6e0fc1e6bbde2b8565f8aa91ff56aacba71c2bca237a26fb` | 2026-11-28 10:52:47 | 19.419 s | 13 s |
| Region 1 | `6bf1be377485249a88541dbb903f816b47e6b9c3e406516683a6409018120141` | 2026-11-28 11:35:14 | 19.419 s | 13 s |

Total 26 QPU seconds charged (estimates 38.8 s, within the proposed 60 s). Account: 199 s remaining at preflight, 401 consumed before and 427 after, 173 s remaining. No eight-qubit job was run.

A first region-1 build (plan `2e007417028adb185900e7228b4494f07f63eb30c34c7efd4c0e68b3a0dc637f`) was never submitted: the submit step stopped at the 15-minute build-age check (`build older than 15 minutes`) before writing a submission marker; IBM job history confirmed no job existed and the balance was unchanged. It was rebuilt under the new hash above and authorized afresh.

## Authorization record

Region 0: the owner replied "do" to the exact hash and estimate shown; this was treated as authorization and the script wrote the untracked authorization file for that hash. Region 1 (second build): the owner replied "authorized" to the exact hash. Authorization files are untracked and not published. One `submission_started.json` and one `job.json` per stage; no submission was retried.

## Files

`hardware_results/qpu04_b7_r0`, `hardware_results/qpu04_b7_r1` (compiled manifests and hashes, QPY, per-circuit QASM native circuits, calibration snapshots, raw `counts.json`, job receipts, usage), `hardware_results/qpu04/` (`preflight.json`, `audit_resources.json`, `audit_commutator.json`, `audit_sevenqubit.json`, `audit_native_r0.json`, `audit_native_r1.json`, `joint.json`, `replay.json`), code `qpu04_audit.py`, `qpu04_preflight.py`, `qpu04_replay.py`, `analyze_qpu04.py`, `owner_submit_qpu04.ps1`, environment `requirements_qpu04.txt`.

## Reproduce

```
cd experiments/hadamard_qft/qft01
python -m pytest test_qpu04.py test_qpu04_results.py -q
python qpu04_audit.py resources ; python qpu04_audit.py commutator ; python qpu04_audit.py sevenqubit
python qpu04.py joint --stage hardware_results/qpu04_b7_r0 --stage hardware_results/qpu04_b7_r1 --out results_local/qpu04_joint.json
python qpu04_replay.py hardware_results/qpu04/joint.json
python analyze_qpu04.py
```

## Limitations

Two regions, one job each, 1024 shots per circuit, different calibration cycles; shot-only intervals; routing optimum only within the stated grammar and fixed gate order; the control-order search was not budget-matched across arms; the A and D arms differ in compiler only. The three-control relative-phase flip is a known primitive type; the priority and complexity comparison with ancilla-free relative-phase multi-controlled-gate and CNOT-routing literature is outstanding.
