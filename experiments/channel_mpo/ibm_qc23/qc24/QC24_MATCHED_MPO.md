# QC-24B matched legacy-MPO comparison

Retrospective comparison on the 60 published QC-23 Z observables from the width 12/16 transfer circuits (20 circuits). This is a common-row subset: the original MPO predictions are available only for three sites per circuit. Model parameters are unchanged from QC24_REPORT.md. The local sites and model family were chosen after viewing the data. No new QPU time.

| Predictor | MAE | Within 0.05 / 60 |
|---|---:|---:|
| ideal | 0.054836 | 32 |
| mpo_modelA | 0.049869 | 39 |
| mpo_modelB | 0.049200 | 40 |
| local_rx | 0.047611 | 29 |
| global_local_rx | 0.043946 | 35 |
| readout | 0.048466 | 32 |
| depth_offset | 0.045625 | 32 |

Paired circuit bootstrap (30,000 resamples, LCG seed 58241), difference = comparator MAE minus global_local_rx MAE. Percentile intervals are descriptive, conditional on the same session.

| Comparator | Mean difference | 95% interval | Improved circuits /20 |
|---|---:|---:|---:|
| ideal | 0.010890 | -0.000552 to 0.022765 | 11 |
| mpo_modelA | 0.005923 | -0.006235 to 0.018392 | 11 |
| mpo_modelB | 0.005254 | -0.006772 to 0.017674 | 10 |
| depth_offset | 0.001679 | -0.003573 to 0.007162 | 11 |
| readout | 0.004519 | -0.001959 to 0.011726 | 10 |

Full-register legacy MPO comparison remains outstanding; the effective coherent model is not a replacement for a calibrated stochastic native-operation channel. No prospective model selection or hardware submission is justified by these retrospective intervals alone.
