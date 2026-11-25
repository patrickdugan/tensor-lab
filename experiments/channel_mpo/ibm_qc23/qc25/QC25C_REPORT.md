# QC-25C: native SX offset + non-unital relaxation (retrospective)

## Result
Source: committed Kingston kicked-Ising measurements, original submitted QASM. No new QPU time. Fit only on width 8 depth 1–2 (8 circuits, 16 target-site expectations); freeze parameters; transfer to all other 28 circuits (320 site expectations). Physical sites 5 and 23 and the model family were selected after inspecting earlier hardware data. Transfer is not prospectively sealed and is conditional on one hardware acquisition.

The sparse Heisenberg evaluator preserves native gate order and QC23's calibrated ASAP fault placement. Compare: ideal; original gate-local twirled model; replace thermal twirl by ground-state non-unital amplitude damping with dephasing (same diagonal contractions and same residual depolarizing probabilities); native SX offsets with twirled or non-unital relaxation. Only two SX offsets fit to target-site MSE, with bounds ±0.1 rad. Reuse symmetric calibrated readout. For non-unital relaxation, adjoint Z -> exp(-dt/T1) Z + (1-exp(-dt/T1)) I; X,Y contraction exp(-dt/min(T2,2T1)). An SX pulse is modeled as Rx(pi/2 + delta_phys), not as a logical RX error. Fits are heuristic coordinate searches, not global optimization or independent hardware calibration.

| Transfer model | All MAE | Target MAE | Other MAE | Within .05 |
|---|---:|---:|---:|---:|
| ideal | .047423 | .124521 | .031069 | 222/320 |
| twirled | .046295 | .123799 | .029855 | 228/320 |
| nonunital | .045302 | .124025 | .028603 | 236/320 |
| twirled + SX | .044293 | .113189 | .029679 | 231/320 |
| nonunital + SX | .043368 | .113658 | .028458 | 238/320 |

Combined channel improves pooled MAE 6.32% relative to gate-local twirled and 4.27% relative to non-unital alone. On 84 common published transfer sites, MAEs: legacy uniform A .043778; legacy B .042991; combined .040553. Paired bootstrap (30,000 circuit resamples, equal circuit weights) for twirled minus combined: +.003244 [+.002339,+.004207], 28/28 circuit wins. Against published uniform B: +.002438 [+.000483,+.004442], 17/28 wins. These retrospective intervals are descriptive and do not measure inter-session uncertainty. Combined model fails to explain most anomalous residuals, target within threshold only 10/56. It also does not reach QC24's prospective target of 10% relative MAE gain versus calibrated stochastic.

Fitted twirled SX deltas: q5=-.021875, q23=+.1 rad; nonunital SX deltas: q5=-.019140625, q23=+.1. Both q23 fits hit upper bound. Do not interpret as native gate estimates or equate them with QC24 logical RX offsets. A simple constant SX error plus thermal relaxation is insufficient. The sign can differ from effective logical RX because the compiled gates interfere and cancel.

All 36 circuits/384 site predictions executed in a JS sandbox. Published ideal max replay gap 1.083e-15. Maximum accumulated discarded Pauli coefficient absolute mass 3.60e-11; this is a truncation-only exact-arithmetic operator-norm bound, not floating-point certification. Maximum live sparse terms 57,541. Measurement maps, count totals and published count expectations verified. Original QASM hashes for all circuits are asserted on CLI execution; the sandbox used original QASM but did not independently hash all 36 in this pass. CI reproduces fit, transfers and committed rows.

## Next experiment, before wider runs

QC-26A: independently measure native SX signed angle using short repeated-pulse sequences, and CZ residual conditional phase using Ramsey on a target conditioned on neighbor |0> vs |1>. Include matched controls, terminal asymmetric readout calibration, randomized/interleaved execution, and alternative physical paths. Freeze per-site and per-edge parameters and predictions before fresh count evaluation. Keep leakage/crosstalk/idle timing hypotheses open. Test edge-specific CZ coherent phase in the heterogeneous MPO only when calibration supports it.

20 QPU-seconds is a proposed first diagnostic cap, not authorization. Exact circuits, shots, runtime budget and plan hash need owner approval. No IBM service access in this session; zero additional QPU seconds. Parent QC25B full MPO replay remains pending CI independently.

## Reproduction
From qc25 directory (Node 18+):
```sh
node qc25c_native_audit.mjs fit 8 /tmp/fit.json
node qc25c_native_audit.mjs transfer 8 /tmp/transfer8.json
node qc25c_native_audit.mjs transfer_shallow 12 /tmp/transfer12s.json
node qc25c_native_audit.mjs transfer_deep 12 /tmp/transfer12d.json
node qc25c_native_audit.mjs transfer 16 /tmp/transfer16.json
```
Transfer commands load committed frozen qc25c_fit.json. No credentials or network required. CI compares recomputed rows and fit with receipts.
