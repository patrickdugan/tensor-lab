# QC-23 hardware results: ibm_kingston, 2026-11-24

First IBM QPU data for QC-23. This supersedes the "no hardware data" statements in `README.md` and `QC23_REPORT.md`, which describe the offline build.

## What ran

| Item | Value |
|---|---|
| Backend | `ibm_kingston` (156 qubits, median two-qubit error 0.19% at calibration 2026-11-24T21:47 +08:00) |
| Method | unmitigated SamplerV2, optimization level 0, 2048 shots per circuit |
| Calibration job `fit_echo` (12 circuits: n in {8,16}, depths 1,3,6, seeds 11,13) | job `dc2phfinj2vs73dh50cg`, plan hash `99ea998c...e826c` |
| Evaluation job `eval_echo` (12 circuits: n in {8,16}, depths 2,4,8, seeds 3,5) | job `dc2pii6q4uuc73a5of4g`, plan hash `114be0d2...a622` |
| QPU time | 18 s total (plan usage 184 s to 202 s of 600 s) |
| Authorization | `results/authorization_fit.json`, `results/authorization_eval.json`, written on the account owner's instruction, ceiling 120 s each |

The calibration set ran before the evaluation set, as pre-registered. The transpiled circuits recompiled for analysis have the same physical paths and OpenQASM 3 hashes as the submitted ones (asserted in the conversion step). Raw counts: `results/ibm_run_fit/hardware_results.json`, `results/ibm_run_eval/hardware_results.json`. Analysis root: `results/hw_root/` (`analysis_ibm.json`, `predictions_ibm.csv`, `plots/echo_ibm.png`).

## Held-out accuracy (36 observables over 12 evaluation circuits, threshold 0.05 on |prediction - measured Z|)

| Predictor | Within threshold | Wilson 95% | Cluster bootstrap 95% | Mean abs error | Max abs error |
|---|---|---|---|---|---|
| ideal (no noise) | 14/36 = 0.39 | 0.25 to 0.55 | 0.19 to 0.58 | 0.101 | 0.390 |
| MPO, Model A (calibration-derived) | 29/36 = 0.81 | 0.65 to 0.90 | 0.64 to 0.94 | 0.035 | 0.152 |
| MPO, Model B (fitted) | 27/36 = 0.75 | 0.59 to 0.86 | 0.56 to 0.89 | 0.030 | 0.073 |
| Pauli-fault, Model A | 24/36 = 0.67 | 0.50 to 0.80 | 0.47 to 0.83 | 0.041 | 0.169 |
| Pauli-fault, Model B | 31/36 = 0.86 | 0.71 to 0.94 | 0.72 to 0.97 | 0.028 | 0.095 |

Mean measured Z by configuration (Model B Pauli / Model A Pauli in brackets):

| n | depth 2 | depth 4 | depth 8 |
|---|---|---|---|
| 8 | 0.973 (0.974 / 0.980) | 0.931 (0.937 / 0.963) | 0.835 (0.787 / 0.890) |
| 16 | 0.965 (0.974 / 0.972) | 0.902 (0.924 / 0.947) | 0.787 (0.774 / 0.866) |

## Limitations

* Model B's fit is poor: chi2 = 1280 on 141 dof, the two-qubit parameter `p2` sits at its lower bound (1e-7), and its log-parameter standard deviation is 3872, so the fit is degenerate in that direction. Model B's good held-out score does not show that the fitted channel is physically right; it reflects the single-qubit and readout terms absorbing the error.
* Model A overpredicts Z at depth 8 (0.866 vs 0.787 measured at n=16); Model B underpredicts it at n=8 (0.787 vs 0.835).
* The worst single observable measured Z = 0.610 (n=16, depth 8).
* The non-Clifford (`nc_eval`, kicked-Ising, 36 circuits) set was not run on hardware, so the `kicked_ising` rows in the analysis JSON are empty, and the Aer noisy comparison was not run. The evaluation covers Clifford echoes only.
* One backend, one calibration snapshot, one day, 12 evaluation circuits. Drift across days is not measured.
* The README's earlier note that `compiled.json` contains the instance identifier is wrong: it stores `configured-default`. The instance identifier appears only in `results/hw_discovery/discovery.json`, which is gitignored.
