# QC-13: Exact-dyadic residual auditing and a deep Clifford reference oracle

**Date:** 2026-11-10. **Status:** completed reproducible CPU experiment; 145/145 regression tests passed locally. This PR is stacked on QC-12's Heisenberg-MPO PR #14.

## Abstract

QC-12's operator-MPO sum of local Frobenius residuals is an **ideal-arithmetic mathematical error ledger**. It recomputes its SVD residuals with NumPy and does not include QR-gauge drift, gate/channel contraction roundoff, or the difference between floating Kraus matrices and exact CPTP coefficients. It cannot be presented as a verified end-to-end numerical certificate for arbitrary 48-qubit Haar circuits.

QC-13 delivers two precisely scoped remedies:

1. **Exact-dyadic arithmetic checks.** Every supplied binary64 matrix element and stored SVD core factor is interpreted as an exact dyadic rational. Python arbitrary-precision integer arithmetic evaluates the residual norm *exactly before the last outward rounding*. The same approach encloses the contraction of the *stored* MPO at the product input and the Kraus completeness defect. These are rigorous **component** checks, not a proof of the whole floating-point channel evolution. In six depth-16 Haar circuits at 48 qubits, the audited residual sums reproduce QC-12's numerical sums to displayed precision while establishing that the stored local residuals were not underestimated by an untracked NumPy dot-product error.
2. **An exact deep-circuit reference on a restricted circuit class.** An independent Pauli-transfer oracle evaluates Clifford gates exactly in Gaussian rationals and noise coefficients in the field `Q(sqrt(1-p))`. For 48-qubit depth-16 synthetic Clifford circuits it provides an outward-rounded **end-to-end comparison to an exact analytic expectation**, including amplitude damping. This does **not** generalize to arbitrary Haar gates. Across five calibration seeds with weak amplitude damping, a rank-2 MPO passed the `0.05` exact-oracle error target in 5/5 cases, even though QC-12's numerical residual ledger rejected 5/5. Three further seeds confirmed 3/3 exact-oracle passes, versus 2/3 under the numerical ledger.

A third, very loose, complete fallback bounds a single-site Pauli expectation using the physical spectrum of the *last* channel's Heisenberg adjoint. This requires a normalized input and an intended CPTP circuit and is useful for detecting vacuous numerical ledgers, not for a 0.05 acceptance target.

## 1. Exact-dyadic norm certificate

Binary64 components are rationals with power-of-two denominators. For a locally supplied gate matrix `M` and the **actual stored** split cores `L=U[:, :k]` and `R=fl(s[:k,None] * Vh[:k])`, choose a shared binary exponent `P` such that

\[
M=2^{-P}M_\mathbb Z,\quad L=2^{-P}L_\mathbb Z,\quad R=2^{-P}R_\mathbb Z,
\]

where the entries of the three scaled arrays are Gaussian integers. Then

\[
\|M-LR\|_F^2=2^{-4P}\sum_{i,j}\left|2^P M_{\mathbb Z,ij}-\sum_rL_{\mathbb Z,ir}R_{\mathbb Z,rj}\right|^2.
\]

The numerator is computed exactly with Python big integers. An integer `isqrt` rounded *up* yields an exact rational upper bound on the square root; conversion to binary64 is checked and rounded upward again with `math.nextafter`. Thus the residual certificate does not assume numerical SVD stability. It checks the actual stored factors, including the rounding of `s*vh`. The arithmetic is independent of the QR/SVD algorithms that supplied those factors.

The MPO readout `v = <0...0|O_MPO|0...0>` is contracted as a boundary vector using exact Gaussian integer arithmetic, with an outward-rounded enclosure of its real and imaginary parts. A similar exact accumulator gives `||I - sum K†K||_F` for the supplied 2×2 floating Kraus matrices, and `||I-U†U||_F` for 4×4 gates.

**What is established:** a rigorous finite-float calculation of each local residual, *the stored MPO's final readout*, and its input channel/gate defects. **What is left:** certifying numerical QR gauge transformations and local gate/channel contractions between these verified checkpoints. To close the end-to-end proof, future work needs a rigorously enclosed norm for every such local transition together with verified environment norms and a controlled accumulation rule. The original QC-12 sum remains an ideal-arithmetic diagnostic until then.

## 2. Exact tiny-circuit reference

`qc13_rational.py` independently evolves up to three qubits using exact Gaussian rational matrices. Every coefficient of the supplied NumPy gate or Kraus matrix is interpreted as an exact dyadic value. The reference makes no assumption that these dyadic gates are precisely unitary or that the dyadic Kraus operators are precisely TP; it reports the resulting trace defect.

Five seeds, three channels, strengths 0.2/0.5, depth three, and ranks 1/2/8 give 90 comparisons:

| Bond cap | Mean exact error | Maximum exact error |
|---:|---:|---:|
| 1 | 0.108226 | 0.612652 |
| 2 | 0.043598 | 0.175477 |
| 8 | 2.47e-16 | 9.76e-16 |

The exact error is between the stored floating MPO estimate and the mathematically exact evaluation of the *same supplied floating coefficients*. These are independently recomputed actual discrepancies, not scalable predictive bounds.

## 3. Exact 48-qubit Clifford-family oracle

The synthetic nearest-neighbor gate ensemble has `HH` (Hadamard tensor Hadamard with entries ±1/2), CNOT, CZ, SWAP and a local S phase. Its entries are Gaussian dyadic rationals, and each gate's action on every two-site Pauli pair is verified by exact matrix multiplication:

\[
U^\dagger(P_i\otimes P_j)U=\pm(P_{i'}\otimes P_{j'}).
\]

For single-qubit noise, the Heisenberg Pauli transfers are analytic:

- Depolarizing: `P` maps to `(1-p) P` for `P=X,Y,Z`.
- Dephasing: `X,Y` map to `(1-2p) X,Y`, while `Z` stays `Z`.
- Amplitude damping: `X,Y` map to `sqrt(1-p) X,Y`, and `Z` maps to `(1-p)Z+pI`.

A symbolic coefficient `a+b*sqrt(1-p)` remains in `Q(sqrt(1-p))`. Its final expectation on `|0>^n` is computed exactly in this number field. An integer square-root algorithm brackets `sqrt(1-p)` with rational endpoints before outward binary64 conversion. Sparse Pauli expansion has an explicit `max_terms` limit, returning a memory error rather than a partial answer if it exceeds the cap.

**Calibration grid:** 48 qubits, depth 16, central Z at site 22, seeds 3–7, three channels, strengths p=0.2/0.5, and bond ranks 1,2,4,8: **120 records**. The more difficult amplitude-damping rows are:

| Noise | Rank | Exact-oracle passes / 5 | QC-12 ledger passes / 5 | Maximum exact-oracle error |
|---|---:|---:|---:|---:|
| Amplitude damping p=.2 | 1 | 1 | 0 | 0.447423 |
| Amplitude damping p=.2 | 2 | **5** | 0 | **0.016882** |
| Amplitude damping p=.2 | 4 | **5** | 2 | 0.008022 |
| Amplitude damping p=.2 | 8 | **5** | 4 | 1.12e-6 |
| Amplitude damping p=.5 | 1 | 3 | 0 | 0.176648 |
| Amplitude damping p=.5 | 2 | **5** | 1 | 0.001709 |
| Amplitude damping p=.5 | 4 | **5** | 4 | 4.76e-5 |
| Amplitude damping p=.5 | 8 | **5** | 5 | 3.60e-12 |

**Independent additional seeds 8–10:** 24 amplitude-damping rows; at rank 2, the exact oracle passed 3/3 at p=.2 and 3/3 at p=.5, while QC-12's residual ledger passed 2/3 in each group. The exact oracle gives a literal, independent reference for this Clifford model, so the ranking rule is **oracle-assisted**. This does not establish that a general Haar circuit can be rank-selected from its compression residuals.

The dephasing and depolarizing outputs of this ensemble sometimes vanish exactly under a Pauli-string readout. Their zero-error successes are valid mathematical checks but are considerably easier than generic Haar circuits. The nontrivial amplitude-damping results above are the consequential tests.

## 4. Deep generic-Haar audit and physical-range fallback

Six depth-16, 48-qubit Haar experiments (seed 3, center Z, rank cap 16) were rerun with the exact-dyadic local checker. The input gates were generated by QC-12's existing sampler. These are **not** symbolic Clifford circuits.

| Channel | p | Exact-dyadic local residual sum | Rigorous physical-range bound |
|---|---:|---:|---:|
| Dephasing | .2 | 6.11314 | 1.00001 |
| Dephasing | .5 | 0.00003450 | 1.00000 |
| Depolarizing | .2 | 28.59927 | 0.80002 |
| Depolarizing | .5 | 0.00671061 | 0.50000 |
| Amplitude damping | .2 | 224.56452 | 0.81234 |
| Amplitude damping | .5 | 0.57232 | 0.54205 |

The residual sum rigorously bounds the sum of **the locally supplied SVD matrix reconstruction errors**; it does not by itself bound the complete circuit's observable error. The last column *is* a mathematically rigorous, albeit weak, absolute error bound under an ideal CPTP circuit and a final local Pauli measurement, because the true expectation must lie in the spectrum of `E†(P)`. In particular, for a final Z measurement under amplitude damping at strength p, the expectation lies in `[2p−1,1]`. The reported bound compares that range to the exact-dyadic interval enclosing the stored MPO's output.

These are independent statements with different assumptions. The columns cannot be combined to claim a tighter fully verified gate-by-gate error estimate. The certified readout rounding gap on these six cases stayed below `3.4e-16`; maximum literal gate unitarity defect was below `2.0e-15`, and Kraus completeness defects were below `1.5e-16`. Auditing the SVD factors took about 3.1–3.6 seconds per depth-16 run, compared with approximately 0.2–0.3 seconds for the earlier numerical-only runner. These are single-run CPU measurements on one environment.

**Failure to reach the 0.05 target on generic weak noise:** at depth 16 and cap 32, the numerical ledger was approximately 129 for amplitude damping p=.2 and 10.85 for depolarizing p=.2. Raising rank alone does not provide a useful full-circuit certificate at this budget. QR/channel/gate certification and noise-adapted contraction bounds remain research tasks.

## 5. Reproduction and verification

Python 3.11+, NumPy 1.24–2.x. No MPFI, GPU, or dense 48-qubit oracle required. All **145 tests** across QC-01 through QC-13 passed on the finalized source.

```bash
cd experiments/channel_mpo
python -m pip install 'numpy>=1.24,<3'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 test_qc04 test_qc05 test_qc06 test_qc07 test_qc08 test_qc09 test_qc10 test_qc11 test_qc12 test_qc13
python benchmark_qc13.py --mode tiny --n 3 --depth 3 --seeds 0,1,2,3,4 --output qc13_tiny_rational_repro.csv
python benchmark_qc13.py --mode clifford --n 48 --depth 16 --seeds 3,4,5,6,7 --output qc13_clifford_repro.csv
python benchmark_qc13.py --mode clifford --n 48 --depth 16 --seeds 8,9,10 --channels amplitude_damping --output qc13_clifford_newseeds_repro.csv
python benchmark_qc13.py --mode audit --n 48 --depth 16 --seeds 3 --caps 16 --output qc13_dyadic_haar_repro.csv
```

Tests cover exact squared residual algebra, stored-factor rounding, subnormal inputs, invalid/nonfinite array rejection, exact Kraus completeness, dyadic readout, tiny rational oracles for three channels, Clifford-map identity checks with exact Gaussian rationals, independent small density checks, 48-qubit symbolic oracles, dense-oracle exclusion, and an oracle-assisted threshold selector. Benchmarks are frozen as `qc13_{tiny_rational,clifford_48,clifford_amplitude_newseeds,dyadic_haar_48}.csv`.

## 6. Next research step

To certify generic QC-12 Heisenberg evolution end-to-end, implement independently checked **QR-gauge step defects** and **gate/channel adjoint contractions** with rigorous exterior-environment operator-norm bounds. Such proof obligations can be assembled with directed-rounded interval or ball arithmetic, along with explicit machine checks of the intended channel/gate models. The Clifford reference will then provide a difficult high-qubit, exactly solvable CI benchmark for that verified numerical pipeline. A noise-adapted dual norm or block-adaptive contraction could improve weak-noise/amplitude-damping acceptance once the core arithmetic is certified.

Background on standard deterministic floating-point error budgets: Nicholas J. Higham, *The Accuracy of Floating Point Summation*, SIAM J. Sci. Comput. 14(4), 1993, https://doi.org/10.1137/0914050 ; Nick Higham and Theo Mary, *A New Approach to Probabilistic Rounding Error Analysis*, 2019, https://doi.org/10.1137/18M1226312 (the probabilistic model is NOT used for these certificates).
