# PGC Phase 6: sensitivity-matrix study and experimental-design comparison (BPD)

**Offline linearised simulation. No hardware evidence, no IBM/QPU service calls, no refit of historical counts.** Every number below comes from `bpd_study.py` (seed 20261126) and is stored in `bpd_results.json`.

## Findings (linearised truth, win/tie/loss over all budgets)

- At seconds_6.4 (matched QPU seconds) weak recovery is 0.14-0.21 for random and fixed designs and 0.87-0.89 for the five information-driven designs (D-opt, D-opt per second, weak-trace per second, two BPD variants). All information-driven designs beat random and fixed designs; this is a property of using the Jacobian at all, not of grouping.
- Weak recovery saturates: the mean Gaussian ceiling over the 10 design pools is 0.95 (worst split max weak posterior sd 0.021 rad, because the held-out set can remove the high-m rows that carry Z_22_23_23 and Z_4_5_5). The best observed recovery of any strategy at any budget is 0.89; differences between good designs at large budgets are therefore not informative.
- BPD (weak trace/s) vs conventional cost-blind D-opt over 6 budgets: ahead on weak info gain per QPU s (5W/1T/0L), weak recovery rate (1W/5T/0L), weak RMSE (2W/3T/1L); behind on info gain per QPU s (1W/1T/4L), held-out RMSE (0W/1T/5L); level on nothing.
- BPD (weak trace/s) vs ungrouped weak-trace-per-second greedy (same objective, no groups): ahead on nothing; behind on info gain per QPU s (0W/0T/6L), weak info gain per QPU s (2W/0T/4L), weak recovery rate (0W/3T/3L), held-out RMSE (2W/0T/4L), weak RMSE (0W/2T/4L); level on nothing. Group restriction does not add to the objective it optimises.
- BPD (log det/s) vs conventional cost-blind D-opt: ahead on weak info gain per QPU s (5W/1T/0L); behind on info gain per QPU s (0W/0T/6L), held-out RMSE (0W/0T/6L); level on weak recovery rate (1W/4T/1L), weak RMSE (2W/2T/2L).

## Setup

- Pool: 140 circuits and 196 observables over 14 parameters: 8 kicked-Ising circuits (8 observables each, 2048 shots, 0.63248 s each), 128 Ramsey cz probes plus 4 SPAM references (384 shots, 0.19984 s each). Cost = shots*(0.00025 + max(1e-5, duration)) + 0.1 s; durations are unavailable offline, so the 1e-5 s floor applies. Whole pool: 31.44 s.
- Jacobian: kicked-Ising rows are the receipt rows (central differences from the native engine). Ramsey rows are central finite differences (h=1e-5) of the analytic probe model in the 14-parameter space at the receipt parameter values; SX columns are zero. Kicked-Ising Jacobian [64, 14]: restricted (6 columns) rank 6, expanded (14 columns) rank 12; whole-pool rank 14.
- Noise: binomial at the stated shots (2048 or 384) per observable, independent across observables (observables of one kicked-Ising circuit share bitstrings in reality; the joint likelihood is not modelled). Estimator weights use var = max(1-y0^2, 0.05)/shots.
- Truth: theta = theta0 + delta, delta ~ N(0, 0.05^2 I) rad. Estimator and design prior: N(0, 0.1^2 I) (weaker than the truth spread). Estimator: linear Gauss-Newton posterior mean. The model is linear, so the Jacobian never changes and 'adaptive' selection reacts to the accumulated design information, not to measured outcomes. Primary tables generate Ramsey truth from the linearised model; a second run generates it from the exact analytic probe model (estimator still linear).
- Weak subspace: the 4 right singular vectors with smallest singular values of the noise-whitened full-pool Jacobian over the 12 ZZ/Z columns (whitened singular values 230.2, 230.2, 260.8, 263.7). The two SX columns are the weakest directions of the whole pool and no budgeted design reaches the tolerance on them, so they are excluded from the primary subspace; 'recovery incl. SX' reports the literal 14-column definition. Recovery = max over weak coordinates of |error| <= 0.025 rad (half the truth spread). Named-weak recovery applies the same tolerance to ZZ_22_23, ZZ_4_5, Z_22_23_23, Z_4_5_5 directly.
- Protocol: 10 random design/held-out splits (25% of each circuit family held out) x 40 trials = 400 trials per strategy and budget, common random numbers across strategies; random draws a fresh design every trial. Held-out RMSE: predicted <O> of the posterior mean versus the simulated true <O> on held-out observables with nonzero sensitivity. Designs are chosen from the design pool only; groups are refitted on design-pool rows per split.
- Budgets: at most k circuits for k in [8, 16, 32] (cost differs per strategy and is reported) and matched QPU-second caps [3.2, 6.4, 12.8] s (each strategy fills up to the cap; none exceeds it).

## Weak directions of the pool

- weak vector 1 (whitened singular value 230.2): Z_22_23_23 (+1.00)
- weak vector 2 (whitened singular value 230.2): Z_4_5_5 (+1.00)
- weak vector 3 (whitened singular value 260.8): Z_22_23_22 (+0.98), ZZ_22_23 (-0.21)
- weak vector 4 (whitened singular value 263.7): Z_4_5_4 (+0.99), ZZ_4_5 (-0.12)

Share of each named weak parameter inside the weak subspace: ZZ_22_23 0.05, ZZ_4_5 0.01, Z_22_23_23 1.00, Z_4_5_5 1.00. The kicked-Ising Jacobian alone is rank-deficient exactly on Z_22_23_23 and Z_4_5_5 (rank 12 of 14); in the pool those two are reached only through the Ramsey forward probes. Parameters with zero share are not in the weak subspace (they are measured by several rows) and stay in the named-weak metric.

## BPD grouping

NMF of row-normalised |J| over all 104 nonzero-sensitivity observables; G=10 groups kept (smallest G with relative reconstruction error <= 0.1, empty groups dropped; final error 0.045; sweep: G=2: 0.713, G=3: 0.628, G=4: 0.537, G=5: 0.439, G=6: 0.317, G=7: 0.240, G=8: 0.136, G=9: 0.131, G=10: 0.109, G=11: 0.085, G=12: 0.076).

| Group | informs (pattern weight) | obs | kicked circ | Ramsey circ | weak overlap | named-weak share | aligned with weak |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | ZZ_22_23 0.71, Z_22_23_23 0.70 | 6 | 0 | 6 | 0.51 | 1.00 | yes |
| 1 | ZZ_4_5 0.71, Z_4_5_4 0.71 | 6 | 0 | 6 | 0.38 | 0.50 | yes |
| 2 | ZZ_5_6 0.71, Z_5_6_6 0.71 | 12 | 0 | 12 | 0.00 | 0.00 | no |
| 3 | Z_16_23_16 0.96, ZZ_16_23 0.29 | 16 | 0 | 12 | 0.00 | 0.00 | no |
| 4 | Z_16_23_23 1.00 | 20 | 2 | 12 | 0.00 | 0.00 | no |
| 5 | Z_22_23_22 0.81, ZZ_22_23 0.59 | 10 | 0 | 6 | 0.43 | 0.35 | no |
| 6 | Z_4_5_4 1.00 | 4 | 0 | 0 | 0.99 | 0.00 | yes |
| 7 | Z_4_5_5 0.71, ZZ_4_5 0.71 | 6 | 0 | 6 | 0.51 | 1.00 | yes |
| 8 | Z_5_6_5 0.72, ZZ_5_6 0.69 | 20 | 6 | 12 | 0.00 | 0.00 | no |
| 9 | Z_5_6_6 1.00 | 4 | 0 | 0 | 0.00 | 0.00 | no |

Groups use |J| rows only. Each recurring pattern is an endpoint Z term, or a ZZ term paired with an endpoint Z term of the same edge, because those columns co-occur in the sensitive rows. Weak overlap is h' P_w h (fraction of the group's unit pattern inside the weak subspace); aligned = overlap >= 0.5 or named-weak share >= 0.5.

## Strategy comparison: Matched QPU-second cap 6.4 s (linearised truth)

| Strategy | circuits | kicked | QPU s | info gain nats/s | weak info gain nats/s | weak recovery | recovery incl. SX | named-weak recovery | weak RMSE rad | held-out RMSE | param RMSE rad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random (fresh draw per trial) | 28.1 | 1.5 | 6.27 | 3.81 | 0.79 | 0.212 +/- 0.020 | 0.037 | 0.237 | 0.0334 | 0.1625 | 0.0386 |
| fixed, pool order | 19.0 | 6.0 | 6.39 | 3.32 | 0.76 | 0.170 +/- 0.019 | 0.037 | 0.085 | 0.0324 | 0.1797 | 0.0371 |
| fixed, hand-uniform Ramsey | 32.0 | 0.0 | 6.39 | 2.58 | 0.67 | 0.142 +/- 0.017 | 0.040 | 0.165 | 0.0339 | 0.1345 | 0.0329 |
| greedy D-opt (cost-blind) | 23.4 | 3.8 | 6.32 | 6.20 | 1.79 | 0.868 +/- 0.017 | 0.035 | 0.853 | 0.0101 | 0.0700 | 0.0469 |
| greedy D-opt per second | 24.0 | 3.5 | 6.31 | 6.22 | 1.80 | 0.887 +/- 0.016 | 0.037 | 0.877 | 0.0098 | 0.0690 | 0.0467 |
| greedy weak-trace per second | 24.3 | 3.2 | 6.24 | 5.16 | 1.84 | 0.890 +/- 0.016 | 0.098 | 0.895 | 0.0094 | 0.0916 | 0.0429 |
| BPD group-adaptive (weak trace/s) | 25.0 | 3.0 | 6.29 | 5.08 | 1.81 | 0.882 +/- 0.016 | 0.068 | 0.890 | 0.0097 | 0.1017 | 0.0443 |
| BPD group-adaptive (log det/s) | 25.8 | 2.6 | 6.28 | 4.77 | 1.81 | 0.880 +/- 0.016 | 0.068 | 0.890 | 0.0098 | 0.1114 | 0.0407 |
| no data (prior only) | 0.0 | 0.0 | 0.00 | n/a | n/a | 0.025 +/- 0.008 | 0.020 | 0.010 | 0.0496 | 0.2489 | 0.0506 |

## Strategy comparison: Circuit budget 16 (linearised truth)

| Strategy | circuits | kicked | QPU s | info gain nats/s | weak info gain nats/s | weak recovery | recovery incl. SX | named-weak recovery | weak RMSE rad | held-out RMSE | param RMSE rad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random (fresh draw per trial) | 16.0 | 0.9 | 3.60 | 4.76 | 0.81 | 0.122 +/- 0.016 | 0.022 | 0.117 | 0.0388 | 0.1795 | 0.0397 |
| fixed, pool order | 16.0 | 6.0 | 5.79 | 3.24 | 0.54 | 0.110 +/- 0.016 | 0.018 | 0.045 | 0.0357 | 0.1870 | 0.0386 |
| fixed, hand-uniform Ramsey | 16.0 | 0.0 | 3.20 | 2.79 | 0.61 | 0.037 +/- 0.009 | 0.015 | 0.050 | 0.0429 | 0.1863 | 0.0409 |
| greedy D-opt (cost-blind) | 16.0 | 2.3 | 4.19 | 8.93 | 2.63 | 0.835 +/- 0.019 | 0.058 | 0.843 | 0.0110 | 0.0796 | 0.0501 |
| greedy D-opt per second | 16.0 | 1.0 | 3.63 | 10.24 | 3.01 | 0.830 +/- 0.019 | 0.090 | 0.848 | 0.0111 | 0.0799 | 0.0383 |
| greedy weak-trace per second | 16.0 | 3.0 | 4.50 | 5.81 | 2.55 | 0.890 +/- 0.016 | 0.120 | 0.892 | 0.0095 | 0.1352 | 0.0324 |
| BPD group-adaptive (weak trace/s) | 16.0 | 0.1 | 3.24 | 5.57 | 3.37 | 0.802 +/- 0.020 | 0.117 | 0.895 | 0.0122 | 0.1785 | 0.0389 |
| BPD group-adaptive (log det/s) | 16.0 | 0.0 | 3.20 | 5.54 | 3.39 | 0.785 +/- 0.021 | 0.120 | 0.895 | 0.0125 | 0.1809 | 0.0391 |
| no data (prior only) | 0.0 | 0.0 | 0.00 | n/a | n/a | 0.025 +/- 0.008 | 0.020 | 0.010 | 0.0496 | 0.2489 | 0.0506 |

## All budgets: weak recovery / held-out RMSE / info gain per QPU s (linearised truth)

| Budget | random (fresh draw per trial) | fixed, pool order | fixed, hand-uniform Ramsey | greedy D-opt (cost-blind) | greedy D-opt per second | greedy weak-trace per second | BPD group-adaptive (weak trace/s) | BPD group-adaptive (log det/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| circuits_8 | 0.05 / 0.206 / 5.3 | 0.05 / 0.206 / 3.6 | 0.02 / 0.215 / 3.3 | 0.68 / 0.112 / 14.7 | 0.68 / 0.112 / 14.7 | 0.85 / 0.175 / 10.8 | 0.77 / 0.185 / 10.4 | 0.76 / 0.186 / 10.4 |
| circuits_16 | 0.12 / 0.180 / 4.8 | 0.11 / 0.187 / 3.2 | 0.04 / 0.186 / 2.8 | 0.83 / 0.080 / 8.9 | 0.83 / 0.080 / 10.2 | 0.89 / 0.135 / 5.8 | 0.80 / 0.179 / 5.6 | 0.79 / 0.181 / 5.5 |
| circuits_32 | 0.28 / 0.133 / 3.6 | 0.16 / 0.172 / 2.6 | 0.14 / 0.134 / 2.6 | 0.89 / 0.067 / 4.4 | 0.89 / 0.067 / 4.4 | 0.89 / 0.083 / 4.8 | 0.89 / 0.073 / 4.6 | 0.88 / 0.098 / 4.1 |
| seconds_3.2 | 0.07 / 0.188 / 4.9 | 0.06 / 0.210 / 4.2 | 0.04 / 0.186 / 2.8 | 0.81 / 0.091 / 11.1 | 0.81 / 0.086 / 11.8 | 0.88 / 0.139 / 8.1 | 0.79 / 0.181 / 5.5 | 0.79 / 0.181 / 5.5 |
| seconds_6.4 | 0.21 / 0.162 / 3.8 | 0.17 / 0.180 / 3.3 | 0.14 / 0.134 / 2.6 | 0.87 / 0.070 / 6.2 | 0.89 / 0.069 / 6.2 | 0.89 / 0.092 / 5.2 | 0.88 / 0.102 / 5.1 | 0.88 / 0.111 / 4.8 |
| seconds_12.8 | 0.60 / 0.096 / 2.6 | 0.28 / 0.169 / 2.1 | 0.70 / 0.082 / 2.7 | 0.89 / 0.065 / 3.1 | 0.89 / 0.065 / 3.1 | 0.89 / 0.083 / 4.8 | 0.89 / 0.065 / 3.1 | 0.89 / 0.071 / 3.0 |

## What BPD does and does not beat (linearised truth)

Win/tie/loss counts over the 6 budgets (recovery and held-out RMSE: paired difference beyond 2 standard errors; the other metrics: 1% margin).

| Metric (BPD group-adaptive (weak trace/s)) | vs cost-blind D-opt W/T/L | vs D-opt per second W/T/L | vs weak-trace per second W/T/L |
|---|---|---|---|
| info gain per QPU s | 1/1/4 | 1/1/4 | 0/0/6 |
| weak info gain per QPU s | 5/1/0 | 3/2/1 | 2/0/4 |
| weak recovery rate | 1/5/0 | 1/5/0 | 0/3/3 |
| held-out RMSE | 0/1/5 | 0/1/5 | 2/0/4 |
| weak RMSE | 2/3/1 | 1/3/2 | 0/2/4 |

| Metric (BPD group-adaptive (log det/s)) | vs cost-blind D-opt W/T/L | vs D-opt per second W/T/L | vs weak-trace per second W/T/L |
|---|---|---|---|
| info gain per QPU s | 0/0/6 | 0/0/6 | 0/0/6 |
| weak info gain per QPU s | 5/1/0 | 3/2/1 | 2/0/4 |
| weak recovery rate | 1/4/1 | 1/4/1 | 0/2/4 |
| held-out RMSE | 0/0/6 | 0/0/6 | 1/0/5 |
| weak RMSE | 2/2/2 | 1/2/3 | 0/1/5 |

- Best weak recovery per budget: circuits_8: greedy weak-trace per second (0.850); circuits_16: greedy weak-trace per second (0.890); circuits_32: greedy weak-trace per second (0.890); seconds_3.2: greedy weak-trace per second (0.882); seconds_6.4: greedy weak-trace per second (0.890); seconds_12.8: greedy D-opt (cost-blind) (0.890).
- Best held-out RMSE per budget: circuits_8: greedy D-opt (cost-blind) (0.112); circuits_16: greedy D-opt (cost-blind) (0.080); circuits_32: greedy D-opt (cost-blind) (0.067); seconds_3.2: greedy D-opt per second (0.086); seconds_6.4: greedy D-opt per second (0.069); seconds_12.8: BPD group-adaptive (weak trace/s) (0.065).
- Best information gain per QPU second per budget: circuits_8: greedy D-opt (cost-blind) (14.696); circuits_16: greedy D-opt per second (10.242); circuits_32: greedy weak-trace per second (4.842); seconds_3.2: greedy D-opt per second (11.806); seconds_6.4: greedy D-opt per second (6.219); seconds_12.8: greedy weak-trace per second (4.842).

- linearised truth, BPD (weak trace/s) vs cost-blind D-optimal: ahead on weak info gain per QPU s (5W/1T/0L), weak recovery rate (1W/5T/0L), weak RMSE (2W/3T/1L); behind on info gain per QPU s (1W/1T/4L), held-out RMSE (0W/1T/5L); level on nothing.
- linearised truth, BPD (weak trace/s) vs D-optimal per second: ahead on weak info gain per QPU s (3W/2T/1L), weak recovery rate (1W/5T/0L); behind on info gain per QPU s (1W/1T/4L), held-out RMSE (0W/1T/5L), weak RMSE (1W/3T/2L); level on nothing.
- linearised truth, BPD (weak trace/s) vs ungrouped weak-trace per second: ahead on nothing; behind on info gain per QPU s (0W/0T/6L), weak info gain per QPU s (2W/0T/4L), weak recovery rate (0W/3T/3L), held-out RMSE (2W/0T/4L), weak RMSE (0W/2T/4L); level on nothing.
- linearised truth, BPD (log det/s) vs cost-blind D-optimal: ahead on weak info gain per QPU s (5W/1T/0L); behind on info gain per QPU s (0W/0T/6L), held-out RMSE (0W/0T/6L); level on weak recovery rate (1W/4T/1L), weak RMSE (2W/2T/2L).
- linearised truth, BPD (log det/s) vs D-optimal per second: ahead on weak info gain per QPU s (3W/2T/1L); behind on info gain per QPU s (0W/0T/6L), held-out RMSE (0W/0T/6L), weak RMSE (1W/2T/3L); level on weak recovery rate (1W/4T/1L).
- linearised truth, BPD (log det/s) vs ungrouped weak-trace per second: ahead on nothing; behind on info gain per QPU s (0W/0T/6L), weak info gain per QPU s (2W/0T/4L), weak recovery rate (0W/2T/4L), held-out RMSE (1W/0T/5L), weak RMSE (0W/1T/5L); level on nothing.

## Robustness: Ramsey truth from the exact probe model

Same designs and trials, but the simulated Ramsey expectations use cos/sin of the exact phase while the estimator stays linear. Designs that lean on high-m rows lose information here.

| Strategy | circuits | kicked | QPU s | info gain nats/s | weak info gain nats/s | weak recovery | recovery incl. SX | named-weak recovery | weak RMSE rad | held-out RMSE | param RMSE rad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random (fresh draw per trial) | 28.1 | 1.5 | 6.27 | 3.81 | 0.79 | 0.203 +/- 0.020 | 0.022 | 0.228 | 0.0333 | 0.1860 | 0.0424 |
| fixed, pool order | 19.0 | 6.0 | 6.39 | 3.32 | 0.76 | 0.145 +/- 0.018 | 0.020 | 0.065 | 0.0324 | 0.1863 | 0.0383 |
| fixed, hand-uniform Ramsey | 32.0 | 0.0 | 6.39 | 2.58 | 0.67 | 0.128 +/- 0.017 | 0.030 | 0.160 | 0.0328 | 0.1590 | 0.0337 |
| greedy D-opt (cost-blind) | 23.4 | 3.8 | 6.32 | 6.20 | 1.79 | 0.840 +/- 0.018 | 0.058 | 0.833 | 0.0109 | 0.1317 | 0.0487 |
| greedy D-opt per second | 24.0 | 3.5 | 6.31 | 6.22 | 1.80 | 0.863 +/- 0.017 | 0.062 | 0.853 | 0.0105 | 0.1309 | 0.0487 |
| greedy weak-trace per second | 24.3 | 3.2 | 6.24 | 5.16 | 1.84 | 0.887 +/- 0.016 | 0.077 | 0.873 | 0.0100 | 0.1343 | 0.0468 |
| BPD group-adaptive (weak trace/s) | 25.0 | 3.0 | 6.29 | 5.08 | 1.81 | 0.882 +/- 0.016 | 0.072 | 0.877 | 0.0103 | 0.1436 | 0.0513 |
| BPD group-adaptive (log det/s) | 25.8 | 2.6 | 6.28 | 4.77 | 1.81 | 0.880 +/- 0.016 | 0.080 | 0.877 | 0.0103 | 0.1421 | 0.0449 |
| no data (prior only) | 0.0 | 0.0 | 0.00 | n/a | n/a | 0.025 +/- 0.008 | 0.020 | 0.010 | 0.0496 | 0.2267 | 0.0506 |

| Budget | random (fresh draw per trial) | fixed, pool order | fixed, hand-uniform Ramsey | greedy D-opt (cost-blind) | greedy D-opt per second | greedy weak-trace per second | BPD group-adaptive (weak trace/s) | BPD group-adaptive (log det/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| circuits_8 | 0.05 / 0.203 / 5.3 | 0.05 / 0.205 / 3.6 | 0.04 / 0.206 / 3.3 | 0.64 / 0.161 / 14.7 | 0.64 / 0.161 / 14.7 | 0.81 / 0.171 / 10.8 | 0.73 / 0.179 / 10.4 | 0.71 / 0.179 / 10.4 |
| circuits_16 | 0.12 / 0.190 / 4.8 | 0.09 / 0.195 / 3.2 | 0.04 / 0.196 / 2.8 | 0.83 / 0.142 / 8.9 | 0.81 / 0.142 / 10.2 | 0.89 / 0.147 / 5.8 | 0.80 / 0.173 / 5.6 | 0.77 / 0.175 / 5.5 |
| circuits_32 | 0.27 / 0.171 / 3.6 | 0.15 / 0.179 / 2.6 | 0.13 / 0.159 / 2.6 | 0.89 / 0.129 / 4.4 | 0.89 / 0.129 / 4.4 | 0.89 / 0.135 / 4.8 | 0.89 / 0.135 / 4.6 | 0.89 / 0.139 / 4.1 |
| seconds_3.2 | 0.06 / 0.194 / 4.9 | 0.04 / 0.206 / 4.2 | 0.04 / 0.196 / 2.8 | 0.75 / 0.154 / 11.1 | 0.81 / 0.149 / 11.8 | 0.87 / 0.148 / 8.1 | 0.77 / 0.175 / 5.5 | 0.77 / 0.175 / 5.5 |
| seconds_6.4 | 0.20 / 0.186 / 3.8 | 0.14 / 0.186 / 3.3 | 0.13 / 0.159 / 2.6 | 0.84 / 0.132 / 6.2 | 0.86 / 0.131 / 6.2 | 0.89 / 0.134 / 5.2 | 0.88 / 0.144 / 5.1 | 0.88 / 0.142 / 4.8 |
| seconds_12.8 | 0.56 / 0.142 / 2.6 | 0.25 / 0.179 / 2.1 | 0.68 / 0.135 / 2.7 | 0.89 / 0.126 / 3.1 | 0.89 / 0.126 / 3.1 | 0.89 / 0.135 / 4.8 | 0.89 / 0.126 / 3.1 | 0.89 / 0.126 / 3.0 |

| Metric (BPD group-adaptive (weak trace/s)) | vs cost-blind D-opt W/T/L | vs D-opt per second W/T/L | vs weak-trace per second W/T/L |
|---|---|---|---|
| info gain per QPU s | 1/1/4 | 1/1/4 | 0/0/6 |
| weak info gain per QPU s | 5/1/0 | 3/2/1 | 2/0/4 |
| weak recovery rate | 2/4/0 | 1/5/0 | 0/3/3 |
| held-out RMSE | 0/0/6 | 0/0/6 | 1/1/4 |
| weak RMSE | 3/2/1 | 2/2/2 | 0/2/4 |

| Metric (BPD group-adaptive (log det/s)) | vs cost-blind D-opt W/T/L | vs D-opt per second W/T/L | vs weak-trace per second W/T/L |
|---|---|---|---|
| info gain per QPU s | 0/0/6 | 0/0/6 | 0/0/6 |
| weak info gain per QPU s | 5/1/0 | 3/2/1 | 2/0/4 |
| weak recovery rate | 2/3/1 | 1/4/1 | 0/3/3 |
| held-out RMSE | 0/1/5 | 0/1/5 | 1/0/5 |
| weak RMSE | 3/1/2 | 2/1/3 | 0/1/5 |

- Best weak recovery per budget: circuits_8: greedy weak-trace per second (0.805); circuits_16: greedy weak-trace per second (0.887); circuits_32: greedy D-opt (cost-blind) (0.887); seconds_3.2: greedy weak-trace per second (0.868); seconds_6.4: greedy weak-trace per second (0.887); seconds_12.8: greedy D-opt (cost-blind) (0.887).
- Best held-out RMSE per budget: circuits_8: greedy D-opt (cost-blind) (0.161); circuits_16: greedy D-opt per second (0.142); circuits_32: greedy D-opt (cost-blind) (0.129); seconds_3.2: greedy weak-trace per second (0.148); seconds_6.4: greedy D-opt per second (0.131); seconds_12.8: greedy D-opt (cost-blind) (0.126).
- Best information gain per QPU second per budget: circuits_8: greedy D-opt (cost-blind) (14.696); circuits_16: greedy D-opt per second (10.242); circuits_32: greedy weak-trace per second (4.842); seconds_3.2: greedy D-opt per second (11.806); seconds_6.4: greedy D-opt per second (6.219); seconds_12.8: greedy weak-trace per second (4.842).

- exact-Ramsey truth, BPD (weak trace/s) vs cost-blind D-optimal: ahead on weak info gain per QPU s (5W/1T/0L), weak recovery rate (2W/4T/0L), weak RMSE (3W/2T/1L); behind on info gain per QPU s (1W/1T/4L), held-out RMSE (0W/0T/6L); level on nothing.
- exact-Ramsey truth, BPD (weak trace/s) vs D-optimal per second: ahead on weak info gain per QPU s (3W/2T/1L), weak recovery rate (1W/5T/0L); behind on info gain per QPU s (1W/1T/4L), held-out RMSE (0W/0T/6L); level on weak RMSE (2W/2T/2L).
- exact-Ramsey truth, BPD (weak trace/s) vs ungrouped weak-trace per second: ahead on nothing; behind on info gain per QPU s (0W/0T/6L), weak info gain per QPU s (2W/0T/4L), weak recovery rate (0W/3T/3L), held-out RMSE (1W/1T/4L), weak RMSE (0W/2T/4L); level on nothing.
- exact-Ramsey truth, BPD (log det/s) vs cost-blind D-optimal: ahead on weak info gain per QPU s (5W/1T/0L), weak recovery rate (2W/3T/1L), weak RMSE (3W/1T/2L); behind on info gain per QPU s (0W/0T/6L), held-out RMSE (0W/1T/5L); level on nothing.
- exact-Ramsey truth, BPD (log det/s) vs D-optimal per second: ahead on weak info gain per QPU s (3W/2T/1L); behind on info gain per QPU s (0W/0T/6L), held-out RMSE (0W/1T/5L), weak RMSE (2W/1T/3L); level on weak recovery rate (1W/4T/1L).
- exact-Ramsey truth, BPD (log det/s) vs ungrouped weak-trace per second: ahead on nothing; behind on info gain per QPU s (0W/0T/6L), weak info gain per QPU s (2W/0T/4L), weak recovery rate (0W/3T/3L), held-out RMSE (1W/0T/5L), weak RMSE (0W/1T/5L); level on nothing.

## Limits

- Linearised model: the Gauss-Newton posterior is exact only for the linear surrogate. The RMSE of linear versus exact Ramsey probe expectation at 0.05 rad spread is 0.003 (m=1), 0.012 (m=2), 0.186 (m=8), so m=8 rows are overstated in the linearised-truth tables.
- Kicked-Ising residuals: rms(measured - prediction) = 0.0244 versus mean binomial sd 0.0151; model misfit, SPAM, drift and crosstalk are not simulated.
- Observables within a kicked-Ising circuit are treated as independent in the likelihood (the README requires a joint likelihood for real inference); the SPAM circuits carry no parameter sensitivity and sit in the pool as given.
- The kicked-Ising duration is unknown offline; a longer duration raises the kicked-Ising cost and moves every cost-aware comparison further toward Ramsey circuits.
- Standard errors are binomial/paired-trial errors; trials within a split share a design split, so they are not split-level confidence intervals.
- Truth is drawn from the same linear (or exact Ramsey) model the estimator assumes. No claim about hardware follows.

## Reproduce

    python bpd_study.py            # rewrites bpd_results.json and BPD_REPORT.md
    python bpd_study.py --check    # reduced rerun with assertions
    python -m pytest test_bpd.py -q
