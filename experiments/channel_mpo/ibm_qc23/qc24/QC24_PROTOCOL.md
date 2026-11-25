# QC-24 prospective diagnostic protocol (proposal; no authorization)

Parent QC-23 fd27a77b4e1a7f468d380d32b411943ad3e5f69f. Proposed caps only: 376 s remaining per last receipt; check live balance before submission.

## Hardware design

1. QC-24C (20 s cap): physical qubits 5,23 and at least two preregistered matched controls. Collect |0>, |1> SPAM, repeated SX, SX inverse-pair echoes and phase-cycled X/Y Ramsey or error amplification at repetition counts 0,1,2,4,8,16,32. Interleave controls, randomize execution order, record pulse/native gate counts and timing. Estimate signed coherent angle and SPAM separately using exact binomial likelihood, with profile intervals.
2. QC-24D (40 s): fixed kicked-Ising circuits, n=8, depths 2,3,4, fresh seeds 101,103,107,109; compile paths so logical roles migrate away from q5/23 and onto those sites with matched gate counts; compare path reversal where topology allows. Freeze physical paths and QASM hashes before execution.
3. QC-24E (40 s): isolated and simultaneously active neighbor gates, paired matching layouts and seed, include idles matched in duration. Analyze differences in local residual and correlations to test crosstalk. 
4. QC-24F (40 s): Fez independent backend, same logical circuit schedule; preserve new backend calibration and refit only calibration circuits, test untouched evaluation circuits. Repeat baseline on Kingston if timing permits within its own cap; do not pool backend parameters.
5. QC-24G (60 s): fresh sealed kicked-Ising seed family (e.g. 211,223,227,229,233,239), multiple paths, at least two time blocks; measure all sites and selected two-site correlators. Freeze models and selected primary metric before seeing counts.
6. QC-24H (100 s): wide/deep gate only after predictive improvement; compare bond caps, convergence, arithmetic certificates, and physical residual independently. Preserve 76 s reserve.

Every stage is conditional on compiled resource estimates and an explicit account-owner authorization naming backend, plan hash and hard QPU-seconds ceiling. If estimate exceeds cap, reduce circuits/shots before authorization. Save submission-time calibration snapshot verbatim, including source and provider calibration timestamps if exposed; do not substitute post-routing snapshot. Save raw counts, physical layouts, job IDs and exact QASM/QPY hashes. No credentials or authorization files in this proposal.

## Model development

C0: calibrated per-native-operation stochastic channel, local readout. C1: per-qubit SX angle error plus relaxation (amplitude damping and pure dephasing) and asymmetric readout; C2: add edge-specific coherent ZZ error; C3: only with targeted evidence, simultaneous-gate crosstalk. Use CPTP maps in Pauli transfer basis; do not convert all local faults to a scalar per layer. Exact dense channel reference at small sizes; compare heterogeneous MPO with increasing bond cap and separately bounded truncation error. Fit using calibration-only circuits; use Jacobian SVD and profile likelihood, report boundary solutions, and account for correlations among observables from the same shots.

## Prospective acceptance gates

- Signed rotation parameters supported by phase-sensitive repeated-gate data; if absent, retain competing relaxation/readout/crosstalk models.
- For held-out circuits, preregister all-qubit MAE as primary metric. Demand paired cluster bootstrap lower bound above zero for improvement over a calibrated gate-local stochastic baseline; target at least 10% relative MAE improvement with no degradation above 0.005 MAE on other sites. Report all circuits, widths, paths and time blocks. Treat same-session clustering as insufficient for device-wide generalization.
- Separate fit and evaluation by circuit family/seed and acquisition block; no retuning on fresh evaluation. Verify predictions sealed by hash before hardware counts. 
- Require small-system dense agreement and explicit MPO truncation ledger, with bond-cap convergence before widening. Do not infer cryptographic resource estimates from this experiment.

Status: proposed, not preregistered until frozen and approved, and not submitted.
