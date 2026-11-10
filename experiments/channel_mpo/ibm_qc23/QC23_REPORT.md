# QC-23 report: IBM Quantum hardware validation

## Bottom line

**No IBM hardware data exists.** `QiskitRuntimeService()` fails on the build machine with `AccountNotFoundError: Unable to find account. Please make sure an account with the channel name 'ibm_quantum_platform' is saved.` (no saved account, no `QISKIT_IBM_*` environment variables; [`results/hw_discovery_error.txt`](results/hw_discovery_error.txt)). QPU time used: **0 s**. Cost: **$0**. Plan allocation: not observable without an account.

Acceptance criteria 1, 2, 4 (as machinery) and 6 are met offline. Criteria 3, 5 and 7 are met only as tooling: there are no hardware measurements to trace, no hardware held-out accuracy, and no QPU usage to account for. Every "measured" number below comes from a **synthetic surrogate simulator** (`results/surrogate/`, labelled `SYNTHETIC-SURROGATE-NOT-HARDWARE` in the analysis JSON and plots). It exists to exercise and validate the pipeline, not to say anything about IBM devices.

## Phase 1: audit of PR #26 and fixes

* Qiskit 2.6.2 / qiskit-ibm-runtime 0.52.0 / stim 1.16.0 / qiskit-aer 0.18.2 (pins in `requirements-qc23.txt`, full freeze in `requirements-qc23-full.freeze`).
* Original QC-23 tests: 6/6 pass. QC-01 to QC-14 regression: 154/154 pass (106 s).
* Fixed or replaced:
  * Observables were Z at five bits (0, 1, n/2, n-2, n-1); the specification is left edge, right edge, centre. Now `observable_bits(n) = (0, n-1, n//2)`; `selected_bits` kept for compatibility.
  * `z_estimators` returned standard error 0 whenever all shots agree. Added Wilson and Hoeffding intervals.
  * Submission used `qiskit_ibm_runtime.executor_sampler.Sampler`. That import does resolve in 0.52, but it is a newer beta path; switched to the documented `SamplerV2`. Neither path could be run against a service here.
  * Path selection was topology-only. Replaced by a deterministic beam search that minimises calibrated error (`choose_line_calibrated`).
  * `--allow-paid` was the only guard. Execution now also needs an owner-written authorization file naming backend, compiled-plan hash and a QPU-seconds ceiling.
  * Metadata said "single-shot"; corrected.
* Unchanged and verified: schedule generator, level-0 transpilation, `measure_all` bit order.

## Phase 2: hardware discovery

Blocked by the missing account (above). Discovery code (`qc23_ibm.py discover`) records QPUs, coupling maps, basis gates, calibration medians, queue depth and plan usage when exposed, and serialises no credentials. A key file was mentioned on the desktop; it was **not read** (credentials stay out of this session). The README gives a command that registers it without the token passing through any log.

Compilation targeted the bundled `FakeFez` snapshot (Heron-class, 156 qubits, native `cz`, `rz`, `sx`, `x`; calibration timestamp 2025-02-26T15:16:25-05:00; median CZ error 3.9e-3 over 352 edges). Also compiled and proven on `FakeSherbrooke` (ECR) in tests.

## Phase 3: Clifford echoes (12 circuits)

| n | physical path (first..last) | depth 2 / 4 / 8: ISA depth | CZ count |
|---|---|---|---|
| 8 | 126, 125, 124, 123, 136, 143, 142, 141 | 43 / 83-85 / 166-167 | 14 / 28 / 56 |
| 16 | 97, 107, 108, 109, 118, 129, 128, 127, 126, 125, 124, 123, 136, 143, 142, 141 | 43-45 / 80-89 / 168-177 | 30 / 60 / 120 |

Per-circuit transpiled OpenQASM 3 (`results/compiled/circuits/*.qasm`), one QPY per set, SHA-256 of each, ISA depth and gate counts, and the measurement mapping are in `results/compiled/compiled_*.json`.

* **Ideal prediction** (all 12): stim stabilizer tableau of the echo built from the integer schedule equals the identity, the forward half does not, and every qubit is deterministically 0. Qiskit statevector agrees for n = 8. The *transpiled* circuits (native `cz`/`sx`/`rz`/`x`, ECR on Sherbrooke) were also converted to stim and proven identity.
* **Mapping**: `audit_mapping` asserts every `measure` op writes bit i from physical `path[i]`, the transpile layout equals the requested path, no gate touches a qubit outside the path, every two-qubit gate is between path neighbours, and zero SWAPs. A probe test (X on one logical qubit) confirms the right physical qubit is flipped. Bit order is little-endian by clbit; tested.

## Phase 4: noise models and held-out comparison (surrogate data)

Surrogate: `FakeFez` calibration, then drifted: gate errors x1.3, T1/T2 x0.8, readout x1.4 and asymmetric, plus coherent RZZ(0.04) after each CZ and RZ(0.015) after each sx/x. Both noise models are deliberately wrong about it. Surrogate shots: 2048 for n <= 12, **1024** for n = 16 (simulation cost). Aer baseline shots: 2048 (n = 8 echoes), 1024 (n <= 12 non-Clifford), 512 (n = 16 echoes).

Models (both evaluated exactly on Clifford circuits by Pauli-fault propagation, cross-checked against a 400k-shot stim Monte Carlo):

* **A, calibration-derived**: per-qubit sx/x and per-edge CZ error, ASAP schedule with Pauli-twirled T1/T2 idling, per-qubit readout. Uses the snapshot only.
* **B, fitted effective channel** (p1, p2, r): fitted on the 12 separate `fit_echo` circuits (seeds 11, 13, depths 1, 3, 6). Fit: p1 = 1.10e-3, p2 = 6.32e-3, r = 3.21e-3, chi2/dof = 351/141 (poor; the surrogate has coherent and non-Pauli errors B cannot represent).
* **MPO (QC-12)**: the *compiled* gate sequence folded into nearest-neighbour 4x4 layers, one uniform depolarising channel per layer with p matched to the model's mean log-shrink, plus a readout factor. Each of A and B gives its own p.
* **Aer baseline**: `NoiseModel.from_backend(FakeFez)`.

Held-out echoes (12 circuits x 3 observables = 36 pairs), threshold 0.05, Wilson 95% (cluster bootstrap over circuits):

| predictor | within 0.05 | Wilson 95% | bootstrap 95% | mean abs err |
|---|---|---|---|---|
| ideal (identity) | 10/36 = 0.28 | 0.16 - 0.44 | 0.08 - 0.53 | 0.139 |
| QC-12 MPO, model A | 22/36 = 0.61 | 0.45 - 0.75 | 0.36 - 0.83 | 0.040 |
| QC-12 MPO, model B | 27/36 = 0.75 | 0.59 - 0.86 | 0.53 - 0.92 | 0.031 |
| exact Pauli, model A | 20/36 = 0.56 | 0.40 - 0.71 | 0.31 - 0.81 | 0.055 |
| exact Pauli, model B | 34/36 = 0.94 | 0.82 - 0.99 | 0.86 - 1.00 | 0.014 |
| Aer from_backend | 19/36 = 0.53 | 0.37 - 0.68 | 0.28 - 0.78 | 0.063 |

Three error sources, reported separately:

| source | size here |
|---|---|
| sampling (binomial SE, mean over rows) | 0.018 |
| model discrepancy (|prediction - measured|) | 0.014 to 0.139 by predictor (table above) |
| numerical, QC-12 ledger | <= 8.9e-12 (echo), <= 1.9e-13 (non-Clifford) |
| numerical, QC-14 dyadic certificate (18 observables, n = 8) | 3.1e-14 to 1.5e-9; all 18 supported |
| numerical, MPO vs exact layered Pauli reference | <= 6.8e-14 |
| numerical, MPO vs dense cone density (136 observables, cone <= 12) | <= 1.4e-14 |
| model projection (MPO uniform depolarising vs exact Pauli, same parameters) | model A: 0.040 vs 0.055 mean abs error (projection happens to help); model B: 0.031 vs 0.014 (projection hurts) |

The certificates bound only the arithmetic of the specified layered model. They are 10 orders of magnitude below the model discrepancy and say nothing about whether that model describes a device. Projecting a model onto "one uniform depolarising channel per layer" changes its predictions by more than any numerical term: for B, MPO reaches 75% against 94% for exact Pauli propagation of the same parameters; for A the projection moves the error the other way (61% against 56%), by luck of the surrogate rather than by design.

QC-14 certificates were run for n = 8 only; n = 16 certification was not attempted in this pass. QC-15 to QC-22 (segmented witnesses, mmap, sealed tapes) were not exercised on these circuits.

## Phase 5: non-Clifford family

Kicked-Ising quench, seeded angle jitter, RX/RZ kicks and RZZ brickwork; n in {8, 12} x steps 1-4 x seeds 41-44, plus n = 16 x steps {2, 4} x seeds {41, 42} (36 circuits, 108 pairs). Ideal values from dense statevectors; the compiled ISA gate sequence (not the logical circuit) is the MPO input, validated against Qiskit statevectors including an ECR backend.

| predictor | within 0.05 | Wilson 95% | bootstrap 95% | mean abs err |
|---|---|---|---|---|
| ideal | 61/108 = 0.57 | 0.47 - 0.65 | 0.50 - 0.63 | 0.050 |
| QC-12 MPO, model A | 68/108 = 0.63 | 0.54 - 0.72 | 0.57 - 0.70 | 0.042 |
| QC-12 MPO, model B | 74/108 = 0.69 | 0.59 - 0.77 | 0.61 - 0.76 | 0.041 |
| Aer from_backend (n <= 12, 96 pairs) | 56/96 = 0.58 | 0.48 - 0.68 | 0.49 - 0.68 | 0.056 |

Negative result: against this surrogate the noise models barely beat the noiseless prediction on the non-Clifford family, and at 4 steps for n = 8 and 12 they are slightly worse than the ideal (mean abs error 0.039-0.043 against 0.029-0.036 for the ideal; at n = 16 they are marginally better). The observables here are small and scrambled, so the 0.05 threshold is comparable to sampling noise (SE 0.018) and to the coherent errors the models omit. The pre-registered threshold fraction is a weak discriminator at these depths.

Beyond dense simulation (`results/scale_test.json`, no measured data):

* Clifford echoes n = 24, 32, 48, depth 6 and 10, cone width up to 38: MPO equals the exact layered Pauli reference to <= 9.1e-14 with bond cap 16 (QC-12 ledger <= 2.4e-8).
* Non-Clifford kicked-Ising n = 24, 32 at 6-8 steps: the spread across bond caps 4/8/16/32 is 0.004 at n = 24, 6 steps but **0.094 (n = 24, 8 steps) and 0.125 (n = 32, 8 steps)**. Bond caps up to 32 are not converged there, so the MPO is not shown to be useful for deep non-Clifford circuits with a wide cone. No external truth exists for these sizes.

## Limitations and failed experiments

* No hardware: all accuracy fractions are against a surrogate designed to differ from both models; they cannot be quoted for IBM devices.
* Surrogate errors (coherent RZZ, asymmetric readout, shorter T1/T2) are an arbitrary choice; different drift gives different fractions.
* Aer baseline is weak because the surrogate drifts from the calibration it uses; with real data the ranking could change.
* Model B fit is poor (chi2/dof 2.5). The 12 fit circuits do not separate p1 from p2 well (log-sd 0.16 and 0.09).
* Idle error placement assumes ASAP scheduling; IBM hardware uses its own timing.
* n = 16 surrogate and Aer shots are lower than the planned 2048.
* The cost of the QC-14 certificate at n = 16 is untested; the pre-registered threshold is applied per (circuit, observable) pair and observables of one circuit are correlated (hence the cluster bootstrap).

## What a hardware run needs

An IBM account on this machine (README, "Run on IBM hardware"), then discover, build, an authorization file for the chosen backend, and `submit`: 12 evaluation circuits plus 12 calibration circuits at 2048 shots. The harness's crude estimate (250 us repetition delay plus circuit duration, 2048 shots) is tens of seconds of QPU time for both sets; confirm the plan and remaining allocation before authorising.
