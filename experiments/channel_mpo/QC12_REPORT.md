# QC-12: Backward Heisenberg MPOs with observable-specific error bounds

**Date:** 2026-11-10. **Status:** CPU research pilot, independently validated against QC-11 on small cones; deep-cone residual certificates in *ideal arithmetic*. **GitHub base:** QC-11 PR #13. Files are a self-contained addition to `experiments/channel_mpo`.

## Main result

QC-11 computes exact local density matrices when a target's backward causal cone is narrow (up to eight active qubits in its standard benchmark). At depth 16, a central one-site measurement in the 48-qubit nearest-neighbor brickwork already has a 32-qubit cone. At depth 24, it reaches **46 qubits**. Direct dense-cone simulation would require up to `4^46` complex entries; QC-12 propagates a chosen Heisenberg observable backward as an **operator matrix product operator (MPO)**, with rank-limited two-site SVD truncation and an independently recomputed residual for each cut.

For a 48-qubit, **depth-24**, seed-3 central `Z_22`, bond cap 16 and strong local noise `p=0.5`, the results were:

| Noise | Backward cone | Retained MPO entries (complex) | Estimated `Z_22` | Summed operator-error bound | Meets `0.05`? |
|---|---:|---:|---:|---:|---|
| Dephasing | 46 | 42,592 | 5.34e-11 | **0.0001528** | Yes |
| Depolarizing | 46 | 42,592 | -4.37e-10 | **0.006148** | Yes |
| Amplitude damping | 46 | 42,592 | 0.7438 | 0.9502 | No |

These are **ideal-arithmetic a posteriori** error bounds computed from numerical SVD residuals. They establish numerical agreement with local expectations up to those bounds, conditional on faithfully evaluating each residual; they do not certify floating-point rounding. The 46-qubit benchmark has **no independent exact physical reference**. The retained MPO entry count excludes QR/SVD and BLAS temporary buffers and process overhead.

## Assumptions and proof

The initial state is `|0>^{⊗n}`. Each forward time layer is a disjoint set of nearest-neighbor 2-qubit **unitaries** followed by independent 1-qubit **trace-preserving completely positive** (CPTP) noise maps. The implementation validates unitarity, Kraus completeness, layout and contiguous nominated target blocks.

For the target operator `O` (currently a local Pauli string), the backwards Heisenberg recurrence is

`O_t = U_t† [⊗_i E_i†](O_{t+1}) U_t`.

Because `E†` is **unital completely positive**, it contracts operator norm:

`||E†(A)||∞ <= ||A||∞`.

Unitary conjugation preserves operator norm. Adding newly dependent qubits by `A -> A ⊗ I` also preserves it. The geometric backwards cone is computed once from the gate layout and the code evolves **only** sites in the cone, rather than forming an operator on all 48 qubits.

Each compression factorizes a two-site MPO center matrix `M` into `L R` by SVD. QR sweeps first put all outside tensors in a mixed canonical gauge, so the full-operator Hilbert-Schmidt residual for the active cone equals the center residual `δ_j = ||M_j-L_jR_j||F`. The code **recomputes** `δ_j` in row blocks from the original matrix (including measured numerical SVD reconstruction residual), instead of taking only the discarded singular-value tail.

The MPO after the j-th cut differs from its pre-cut value by operator norm at most `δ_j`, because `||A||∞ <= ||A||F`. Under future backward propagation, this discrepancy cannot grow in operator norm by the unital-channel/identity-extension argument. A telescoping sum yields

`|Tr[rho_initial (O_exact - O_approx)]| <= ||O_exact-O_approx||∞ <= Σ_j δ_j`.

This is the **observable-specific a posteriori error ledger**. It does *not* automatically certify the entire output state and may become extremely loose. It is an exact-mathematics guarantee if the `δ_j` were exact upper bounds. NumPy provides ordinary double-precision **measurements**, so the code makes no formal interval or rounding-error claim.

### One- and two-site tomography

For a target of `k<=2` adjacent qubits, QC-12 independently propagates each Pauli basis element. It reconstructs `tilde rho = (1/2^k) Σ_P tilde r_P P`. With exact Pauli coefficients `r_P` and individual error bounds `ε_P`, Pauli orthogonality gives

`D(rho, tilde rho) = (1/2)||rho-tilde rho||1 <= (1/2) sqrt( Σ_{P != I} ε_P^2 )`.

The reconstructed matrix is Hermitian and trace one, but **MPO truncation can make it nonpositive**; the expression above is a trace-norm discrepancy rather than a physical-state fidelity guarantee. No misleading `min(1,.)` clamp is used. For a single POVM event, a physical reconstructed matrix is needed before interpreting arbitrary reconstructed probabilities. For any *individual* Pauli measurement, `ε_P` directly bounds the signed expectation error in ideal arithmetic. A threshold selector tests only the measured residual ledger, with **predeclared** increasing bond caps and error targets, and returns rejection if none passes. It never queries an exact oracle to select rank.

## Independent validation, 72 comparisons

Against QC-11's exact streaming density-matrix oracle for **48-qubit / depth-8** circuits, we used 10-qubit edge-pair cones, one- and two-site regions, both seeds 3 and 5, three channels, noise strengths 0.2 and 0.5 and bond caps 4/8/16. All **72** trace-distance comparisons satisfy the bound to the numerical check tolerance, with zero observed violations.

Selected seed-5 edge-pair (`(0,1)`), noise 0.5:

| Channel | Bond cap | Measured trace distance | Computed trace bound |
|---|---:|---:|---:|
| Dephasing | 8 | 6.69e-7 | 0.000980 |
| Amplitude damping | 16 | 1.91e-5 | 0.02177 |

The first width-10 results directly extend the earlier exact-cone limit. Across the entire exact window grid, the **maximum actual trace distance** was 0.04175, despite some far larger pessimistic residual bounds. This illustrates why the theoretical ledger is a sufficient acceptance check rather than a tight estimator of physical error.

## Fixed-cap deep held-out benchmark

We ran **216** cases at depths `{10,16,20}` and seeds `{3,4,5}`, with 48 qubits, output targets `{0,22}`, noise channels `{dephasing,depolarizing,amplitude_damping}`, strengths `{0.2,0.5}`, and caps `{8,16}`. An additional **24** depth-24 stress cases use seed 3. Total: **240** deep rows, all without dense-oracle calls. The acceptance target `error_bound <= 0.05` is a conservative threshold on the numerical residual ledger, not a claim of end-to-end formally verified arithmetic.

For the **144** cases at depths `{16,20}`, the strong-noise `p=0.5` subset had the following results (12 circuits per channel and cap):

| Noise `p=0.5` | Cap | Bound below 0.05 | Median bound | Median simulation (s) |
|---|---:|---:|---:|---:|
| Dephasing | 8 | 12/12 | 0.00524 | 0.123 |
| Dephasing | 16 | **12/12** | 0.000018 | 0.188 |
| Depolarizing | 8 | 6/12 | 0.0603 | 0.156 |
| Depolarizing | 16 | **12/12** | 0.00226 | 0.216 |
| Amplitude damping | 8 | 0/12 | 0.890 | 0.119 |
| Amplitude damping | 16 | 6/12 | 0.198 | 0.239 |

For weaker `p=0.2` noise at depths 16/20, **no channel met 0.05 at either cap** in this fixed grid. This is a failure of these caps to deliver an informative **error bound**; it is not proof that the output expectation is necessarily inaccurate. In a 46-site cone with cap 16, the 42,592 retained complex entries correspond to about 0.65 MiB of complex128 payload. The measured resident process memory will be higher, and actual resource scaling depends on numerical library work buffers.

### Runtime and algorithmic scope

Single-thread OpenBLAS/OMP, NumPy CPU; timings exclude gate generation and all independent dense-oracle evaluation, and are single-run samples. Gate costs scale with MPO bond dimensions and active cone width. This implementation uses full QR canonicalization before **every** two-site truncation to keep its local residual ledger valid. Faster sweeps and interval-error evaluation remain open optimizations.

Heisenberg propagation handles **chosen local observables** and can become expensive in low-noise, deep entangling circuits if acceptable error requires large bonds. The 48-qubit system size alone is not a general advantage: depth and cone width govern difficulty, with no polynomial worst-case guarantee for arbitrary circuits.

### Isolated-process memory (seed 3)

The separate-process probe (`qc12_peak_rss.csv`) ran 16 48-qubit center-observable cases (depths 16/24; local dephasing/depolarizing strengths 0.2/0.5; bond caps 8/16). On this Linux environment, `resource.getrusage` measured peak RSS ranging **94.89–95.02 MiB at cap 8** and **95.77–96.16 MiB at cap 16**. Across these runs the cap increase costs about 1 MiB of process peak despite the larger MPO state. These are fresh-process, single-observation resource measurements, with substantial common Python/NumPy overhead; they are not comparable to earlier QC-09/QC-11 hardware/runtime snapshots without re-running those baselines in the same environment. The full in-process matrix/SVD memory footprint is not formally bounded by `max_entries`.

## Acceptance discipline

- An entry below the tolerance threshold is a **conditional mathematical a posteriori bound** for the specified Pauli expectation in exact arithmetic, with numerically measured residuals.
- An entry above tolerance is a **rejection**, even when the approximate numerical expectation looks plausible.
- The 72 exact-window comparisons are independent empirical validation of the error ledger; the deep runs do not have exact-state references.
- End-to-end certified computation needs *outward-rounded interval* bounds for QR/SVD and residual evaluation, or another formally validated arithmetic mechanism.
- The cone argument assumes independent local CPTP noise and an initial product state. General correlated noise, long-range gates, and entangled inputs require additional dependency support.

## Reproduce

```bash
cd experiments/channel_mpo
python -m pip install 'numpy>=1.24,<3'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 test_qc04 \
  test_qc05 test_qc06 test_qc07 test_qc08 test_qc09 test_qc10 test_qc11 test_qc12
python benchmark_qc12.py --mode exact --depths 8 --seeds 3,5 \
  --channels dephasing,depolarizing,amplitude_damping --strengths .2,.5 \
  --caps 4,8,16 --output qc12_exact_windows_repro.csv
for d in 10 16 20; do
  python benchmark_qc12.py --mode deep --depths "$d" --seeds 3,4,5 \
    --targets 0,22 --channels dephasing,depolarizing,amplitude_damping \
    --strengths .2,.5 --caps 8,16 --output "qc12_deep_${d}_repro.csv"
done
python benchmark_qc12.py --mode deep --depths 24 --seeds 3 \
  --targets 0,22 --channels dephasing,depolarizing,amplitude_damping \
  --strengths .2,.5 --caps 8,16 --output qc12_deep_24_stress_repro.csv
python benchmark_peak_qc12.py --output qc12_peak_rss_repro.csv
```

## Literature

- Shao, Zhao, Cheng and Liu (2026), *Complexity of tensor network simulation for noisy quantum circuits*, [arXiv:2606.00474v2](https://arxiv.org/abs/2606.00474). Rigorous noise and MPO complexity results; the present work is a distinct finite-circuit numerical error-ledger experiment.
- Shao, Cheng and Liu (2025), *Pauli Propagation: Simulating Quantum Spin Dynamics via Operator Complexity*, [arXiv:2510.22311](https://arxiv.org/abs/2510.22311). Observable-centered Heisenberg simulation via Pauli propagation.
- QC-09 [report](QC09_REPORT.md) and QC-11 [report](QC11_REPORT.md) provide the prior rank/certificate and exact-cone baselines.
