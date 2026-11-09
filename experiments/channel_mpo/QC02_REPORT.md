# QC-02: direct locally purified noisy-circuit evolution

**Date:** 2026-11-09. **Status:** CPU pilot, with exact small-system checks and an explicit retention warning at larger sizes. Builds on [QC-01](README.md) in the same `experiments/channel_mpo` directory.

## Question and method

Does directly evolving a locally purified matrix-product density operator (MPDO) offer practical savings over the dense density-matrix baseline, with explicit information about the accuracy lost through truncation?

The site tensor is `A[left_bond, system_qubit, local_environment, right_bond]`, representing a global purification `|Psi>` with `rho = Tr_env |Psi><Psi|`. Every two-qubit gate acts on adjacent physical indices, followed by an SVD across the intersite bond. Single-site Kraus operators append a local environment label. A second SVD then compresses that local environment. QR sweeps make the surrounding tensors orthonormal before either truncation; local discarded singular-value weights are measured in that gauge. After each circuit layer, a norm contraction and scalar rescaling give trace one. The represented state stays positive semidefinite by construction.

**Crucial distinction:** these truncations are optimal *locally in a chosen purification gauge*, and their discarded weights are per-operation losses. There is **no certified bound on end-to-end trace distance** from the cumulative loss. The script reports retained purification norm *before* renormalizing each layer and raises a conservative `retention_gate_pass=False` flag whenever any layer retains less than 95%. Large systems can run while having an unacceptable or unknown approximation error.

Unlike QC-01, the simulation itself never allocates a `2^n x 2^n` density matrix. `dense_density()` is a separate balanced-contraction oracle used only for `n<=10`, and excluded from simulator timing. `stable_svd()` includes a NumPy-only QR-preconditioned retry for a rare complex-LAPACK nonconvergence observed during testing. Only NumPy is required.

## Verified CPU evidence

- **18/18 local tests pass:** 7 QC-01 and 11 QC-02. Tests include direct-versus-dense agreement on four qubits across all three noise types, positive and normalized truncated five-qubit states, the QR/SVD retry, and an **independent analytic 48-qubit product-state check** of `<Z_i>=(1-p)^depth` under depolarization with gates disabled.
- **144 density-oracle checkpoints** in `qc02_validated_small.csv`: 6 and 8 qubits, 6 depths, seeds 0 and 1, three channels, strengths 0.2 and 0.5, hard caps bond=12 and environment=8. All retain positivity to floating-point tolerance. The depth-six mean trace distances for depolarizing `p=.5` were 0.0512 (6q) and 0.0545 (8q); for `p=.2` they were 0.1160 and 0.1465. Higher environment capacity is necessary for tighter fidelity.

### Measured evolution time versus QC-01 dense code

All measurements use one CPU process with `OPENBLAS_NUM_THREADS=1`, seed 0, six brickwork layers, depolarizing noise on every qubit after each layer, maximum bond 12, local environment cap 8, per-operation relative SVD tolerance `1e-9`. Timings measure the evolution algorithm alone; constructing the dense oracle from the MPDO, diagonalization, CSV overhead, and validation costs are excluded from the corresponding evolution timings. Hardware and interpreter affect wall times; these are single runs.

| qubits | p | dense evolution (s) | purified evolution (s) | dense entries | purified entries | final trace distance |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | .2 | .0162 | .0697 | 4,096 | 9,600 | .1077 |
| 6 | .5 | .0172 | .0744 | 4,096 | 9,600 | .0458 |
| 8 | .2 | .1805 | .1248 | 65,536 | 14,208 | .1319 |
| 8 | .5 | .1763 | .1508 | 65,536 | 14,208 | .0465 |
| 10 | .2 | 5.425 | .1635 | 1,048,576 | 18,816 | .1641 |
| 10 | .5 | 5.670 | .1511 | 1,048,576 | 18,816 | .0679 |

The 10-qubit `p=.5` run used **55.7x fewer stored complex tensor entries** and measured **37.5x faster evolution** than this dense baseline, at **trace distance 0.0679**. This is a speed/accuracy tradeoff at fixed tensor caps; it establishes no matched-accuracy speed advantage. The `p=.2` case produces larger errors (0.1641). Several small-system runs have more tensor entries than the dense state.

### Sensitivity to environment cap

`qc02_frontier.csv` fixes n=6, p=.5, depth=6, seed=0. At bond cap 16, raising environment cap from 2 to 4, 8, and 16 gives trace distance 0.3300, 0.0959, 0.0455, and 0.0239, while final stored entries grow from 2,592 to 6,272, 16,896, and 33,792. With bond cap 8, the curve is not strictly monotonic because gauge-dependent truncations change later trajectories.

### Larger-system execution and its limitation

`qc02_scalable_results.csv` contains **360 layer records**, from 8, 12, 20, 32, and 48 qubits; 6 layers; two seeds; three channels; p=.2 and .5; bond cap 12 and environment cap 4. At 48 qubits, depolarizing p=.5 stores **52,352 complex entries**, versus formal dense storage `4^48`. The final-layer update took about **0.26 s** in this small-cap run.

**The 48-qubit output is not fidelity-validated.** Its mean retained purification squared norm before normalization at the last layer was approximately **6.2e-7**. Thus the program had discarded almost all of the pre-normalization purification weight at that layer. The gate correctly flags it as low-retention. This is evidence that simplistic local Kraus truncation scales in execution *only by sacrificing a potentially enormous amount of information*. Retention is a useful alarm rather than a mathematical bound on density-matrix error. The analytic product-state test at 48 qubits verifies the no-entangling-gates limit; it gives no validation of the random-entangling circuit there.

## Reproduction

Python >=3.10, `numpy>=1.24`. Run from `experiments/channel_mpo`:

```bash
python -m pip install 'numpy>=1.24'
OPENBLAS_NUM_THREADS=1 python -m unittest -v test_channel_mpo test_purified_mpo
OPENBLAS_NUM_THREADS=1 python purified_mpo.py --qubits 6,8 --depth 6 --seeds 2 --max-bond 12 --max-env 8 --strengths 0.2,0.5 --validate-small --output local_validation.csv
OPENBLAS_NUM_THREADS=1 python compare_dense.py --qubits 6,8,10 --depth 6 --max-bond 12 --max-env 8 --output local_comparison.csv
OPENBLAS_NUM_THREADS=1 python frontier.py --qubits 6 --depth 6 --bonds 8,16 --environments 2,4,8,16 --output local_frontier.csv
OPENBLAS_NUM_THREADS=1 python purified_mpo.py --qubits 8,12,20,32,48 --depth 6 --seeds 2 --max-bond 12 --max-env 4 --strengths 0.2,0.5 --output local_scaling.csv
```

The supplied frozen CSVs are the runs made in the working container; timing is not a portable performance guarantee. Each `runtime_layer_seconds` measures updates and trace-normalization. The approximate MPDO storage measure is the sum of complex entries in all site tensors (16 bytes each for NumPy `complex128`); intermediate allocations, Python overhead, and operating-system peak RSS are additional costs. `largest_working_entries` is a partial scratch-tensor diagnostic and is not an RSS measurement.

## Research gates

1. **Accuracy gate:** report full trace distance at n<=10 for every parameter change, alongside local observables and retained norm. Require an explicit accuracy target before using n>10 outputs for substantive physics conclusions.
2. **Environment-basis optimization:** implement two-site ancilla rotations and disentangling sweeps (after Müller–Ayral–Bertrand, arXiv:2403.00152), then compare fidelity at the same storage/time budget. The present implementation uses only independent single-site ancilla SVDs.
3. **Scalability study:** add independent local-observable certificates and local light-cone references for larger n; vary noise and gate geometry. Measure actual peak RSS and wall times across three or more seeds and repeated timings on fixed hardware.
4. **Theory:** compare operator-Schmidt spectra and positivity-preserving purification truncation under the contractivity regimes analyzed by Shao et al., arXiv:2606.00474v2, without assuming their operator-MPO bounds apply unchanged to this fixed-Kraus-cap algorithm.

**References:** [Shao et al., noisy-circuit tensor-network complexity (2026)](https://arxiv.org/abs/2606.00474); [Müller, Ayral, Bertrand, positive tensor networks (2024)](https://arxiv.org/abs/2403.00152).
