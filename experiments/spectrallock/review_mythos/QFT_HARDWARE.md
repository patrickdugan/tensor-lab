# Optional HAD-QFT hardware review (frozen at 08274a5f5c6e61198bbfd6e106f0bc19a96b9702)

Paper already contains its IBM section. Our normalized shell QFT restricts to n+1 valuation shells of p^n, with Q R=G_0...G_(n-1), each adjacent rotation cos=sqrt((p-1)/p), sin=1/sqrt(p). Binary recursion W_2L=(I_2 tensor W_L)P_L C_L P_L^dagger when n+1=2^b; native shell encoding, O(n) no-ancilla gates or O(log^2 n) with clean workspace. Character sectors reverse labels with a Gauss phase. Known orbit Fourier matrices predate us; circuit novelty requires a literature audit. The paper's hierarchical hopping application is classically simulable.

Actual Kingston results (one calibration, separate physical lines, 384 shots/circuit; Gray -> recursive):
| Metric | b3 (128 points) | b4 (32768 points) |
|---|---|---|
| native CZ | 62 -> 52 | 368 -> 200 |
| two-qubit depth | 62 -> 52 | 335 -> 192 |
| circuit depth | 395 -> 360 | 2008 -> 1093 |
| duration us | 9.66 -> 9.08 | 41.41 -> 24.29 |
| mean forward TV | .112875 -> .137412 | .478418 -> .441497 |
| paired Gray minus recursive TV (shot-only interval) | -.024538 [-.041454,-.001447] | +.036921 [.013177,.060650] |
| Hellinger fidelity | .948532 -> .938293 | .593234 -> .643680 |
| mean echo return | .574219 -> .552083 | .074219 -> .058594 |
| prep-only echo | .927–.932 | .732–.758 |
| mean SPAM correct | .987956 | .980632 |
| ideal-sampling floor | .040 | .052 |
| uniform TV reference | .445 | .571 |
| IBM charge | 5s | 6s |

Stage A chain 38,49,48; B chain 148,149,150,151. Calibration 2026-11-27 11:57:17 UTC. 68 circuits and 26112 shots. Independent readback recomputed every TV and Hellinger value from raw counts and the compiled ideal, matching to <5e-16. This audit does not rederive the ideal unitaries or the bootstrap intervals. 11s charged to these jobs, 354s left from 600 at last receipt. Account usage counter moved 22s during the campaign; the existing report attributes the difference to other activity.

Interpretation: deterministic resource saving for these specific compiled circuits (16.1%,45.7% fewer native two-qubit gates). Measured forward accuracy worsened at b3 and improved at b4, with large errors and near-random echo return at b4. Different physical lines confound scaling comparisons, and finite-shot bootstrap intervals exclude drift/systematic uncertainty. No demonstrated advantage in preserving coherent interference, no generic QFT acceleration and no cryptographic hardness conclusion. Suggest one better coherence benchmark if appropriate, without executing it.

Paper: https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/manuscript/paper.tex
Hardware record: https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/hardware_results/README.md
Stage A analysis: https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/hardware_results/stageA_b3/analysis_full.json
Stage B analysis: https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/hardware_results/stageB_b4/analysis_full.json
Raw counts and compiled ideal manifests: see URI_INDEX.md.
