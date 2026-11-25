# QC-09: Residual-certified rank-revealing gates and peak-live rank budgets

**Date:** 2026-11-09. **Status:** complete CPU pilot with exact-density checks at 6/8 qubits; isolated-process memory scaling at 8/24/48 qubits. Stacked on QC-08. The phrase **residual-certified** means an independently measured numerical Frobenius residual paired with a proved mathematical trace-distance inequality. It is **not a formal floating-point/interval-arithmetic certificate**.

## Question and outcome

QC-08 obtains a median 1.43x speedup against QC-07, but its gate schedule grows temporary arrays and real peak RSS. Can a low-rank gate factorization plus a prospective peak-memory-aware joint bond/environment allocation improve speed and measured memory, while exposing its physical-error cost?

**Result.** Across 20 held-out six-layer depolarizing circuits (n=6/8, p=.2/.5, seeds 3–7), the rank-revealing sketch gives **1.77x median speedup** against the QC-08 *sequential gate / classic compression* control. Its mean exact-density trace distance is **0.069897**, compared with **0.069909** for QC-08; single-case states can differ by up to **0.01785** in trace distance. With predeclared peak-live entry caps, the median speedup remains 1.76x, mean distance becomes **0.071200**, and the modeled live-array peak falls 36% versus the baseline. At 48 qubits, actual peak RSS fell by 6–9 MiB in isolated single trials. Large-system fidelity remains **unverified**.

## Method

Each purification site is \(A_i[b_{i-1},2,e_i,b_i]\). The layer boundary entry count is exactly \(C(b,e)=2\sum_i b_{i-1}e_i b_i\). After placing the gate center in an isometric gauge, reshape the two-site contracted tensor to \(M\in\mathbb C^{m\times n}\).

### Gate factorization

* **Control (`full`):** thin deterministic SVD, rank bounded by 12, reproducing QC-08's physical density matrix at small n.
* **Sketch:** take a deterministic seeded complex Gaussian matrix \(\Omega\in\mathbb C^{n\times(k+4)}\), form an orthonormal basis \(Q=\operatorname{orth}(M\Omega)\), perform two subspace-power iterations, SVD the smaller \(Q^*M\), and retain the leading k factors. The approximation is \(LR\), with \(L\) approximately isometric.
* **Independent residual check:** compute \(\|M-LR\|_F\) by streaming row blocks from the original matrix, avoiding a second full-size residual array. Verify projection defect \(|\langle R,L^*M\rangle-\|R\|_F^2|/\|M\|_F^2\) as a diagnostic.

For normalized purifications \(\psi,\phi\), with \(\|\psi-\phi\|_2/\|\psi\|_2=\epsilon\), the reverse triangle inequality gives \(\|\psi/\|\psi\|-\phi/\|\phi\|\|_2\le2\epsilon\). Contractivity of trace distance under partial trace yields

\[
D(\rho_\psi,\rho_\phi)\le\min(1,2\epsilon),\qquad
\epsilon=\frac{\|M-LR\|_F}{\|M\|_F}.
\]

The isometric-gauge property makes the local matrix residual the *global* purification residual. The bound applies to the ideal-real-arithmetic factors; the reported residual is floating-point numerical evidence. Full rigorous computer arithmetic would require verified interval bounds on the complex contractions and norms. Accumulating these local bounds through CPTP steps is possible in principle, but the naive sum saturates at the tested depth. This experiment makes **no end-to-end 48-qubit fidelity claim**.

### Joint rank budget with prospective peak-live entries

QC-07's greedy rank-benefit objective is retained. QC-09 admits a proposed bond or Kraus-environment rank increase only if it fits **both** the exact retained-entry budget and a prospective live-buffer model. The model includes the resident cores, contracted gate matrices, low-rank factors, and row-block residual working space. A second envelope assumes that the next gate layer may expand all spatial bonds to the bond cap, after which the local Kraus labels expand each environment by the number of Kraus operators. It reserves additional room for canonicalization/compression of the largest enlarged site.

The higher of the gate-model and channel-envelope peaks must fit within the configured cap **with an 8% planning margin**. Monotonicity allows any rejected rank increment to be skipped permanently during that allocation pass, avoiding repeated infeasible candidate checks. The retained-entry limit is exact; the peak-live estimate is **a heuristic NumPy-array accounting model**, not a bound on hidden BLAS/LAPACK workspace or OS RSS. An independent `resource.getrusage` subprocess benchmark measures actual RSS. A peak-model admission rejection occurs before the explicit gate contraction; canonical QR sweeps themselves can still allocate transient memory outside this simplified model.

For six/eight-qubit depolarizing circuits, layer budgets are frozen from QC-06 (`qc06_budget_schedule.json`) and peak-live caps are frozen from calibration: **260,000** and **280,000** modeled entries. For the 24/48-qubit scaling-only pilots, illustrative caps are 350,000 and 550,000, respectively. No dense-reference information enters the rank policy.

## Matched 20-case held-out comparison

Six layers; n={6,8}; p={0.2,0.5}; seeds 3–7; per-layer QC-06 frozen retained-entry budgets; 12 maximum spatial/environment ranks. CPU single-thread OpenBLAS. Timing excludes the exact dense oracle. The baseline uses QC-08 `classic_mem`-style sequential gates plus classic bond-then-environment compression.

| Policy | Mean trace distance | Mean time (s) | Median paired speedup | Mean modeled live-array peak (complex entries) | Mean final retained entries |
|---|---:|---:|---:|---:|---:|
| QC-08 sequential (`qc08`) | 0.069909 | 0.18394 | 1.00x | 391,894 | 8,855 |
| QC-09 full-SVD certificate (`qc09_full`) | 0.069909 | 0.17556 | 1.03x | 620,196 | 8,855 |
| QC-09 sketch (`qc09_sketch`) | **0.069897** | **0.10340** | **1.77x** | 285,194 | 8,852 |
| QC-09 sketch + peak budget (`qc09_peak`) | 0.071200 | 0.10691 | 1.76x | **249,444** | 8,509 |

Mean trace-distance change compared with QC-08: unconstrained sketch \(-1.22\times10^{-5}\); peak-budgeted sketch \(+0.001291\). The slight mean sketch improvement is experimental sampling variation, with different individual states; there is **no monotonic fidelity guarantee**. The maximum observed trace distance between QC-08 and QC-09 sketch's *states* was 0.01785, and the capped policy's maximum was 0.03042. The full-SVD control reproduces QC-08's physical states within floating-point precision.

The original QC-08 fast gate schedule is another valid comparison, but the matched control here is its lower-RSS sequential policy. Faster gate-layer scheduling can trade additional temporary memory for CPU time. These benchmarks therefore establish gains against the lower-RSS control.

### Cross-channel checks (8 circuits per channel)

n={6,8}; p={.2,.5}; seeds 3 and 5. The same frozen depolarizing calibration caps are reused.

| Channel | QC-08 mean D | Sketch mean D | Capped sketch mean D |
|---|---:|---:|---:|
| Dephasing | 0.073202 | 0.073838 | 0.077016 |
| Amplitude damping | 0.079706 | 0.079629 | 0.080608 |

Dephasing's capped policy shows the largest mean regression among these checks. The sketch policy is favorable for speed and reduces memory; accuracy still needs independent application-specific acceptance thresholds.

## Isolated actual peak RSS and runtime

Linux `resource.getrusage(RUSAGE_SELF).ru_maxrss`; one fresh process per configuration, 6 layers, seed 3, no dense oracle; NumPy 2.3.5. Timing is a **single observation** per mode and parameter, not a statistical performance interval.

| Qubits / noise | QC-08 RSS (MiB) | Sketch RSS (MiB) | Capped RSS (MiB) | QC-08 time (s) | Capped time (s) |
|---|---:|---:|---:|---:|---:|
| 8 / p=.2 | 108.09 | 100.72 | 100.21 | 0.196 | 0.137 |
| 24 / p=.2 | 108.65 | 102.18 | 101.55 | 0.894 | 0.487 |
| 48 / p=.2 | **114.29** | 107.32 | **105.52** | 2.106 | **1.243** |
| 48 / p=.5 | **111.01** | 107.14 | **104.84** | 2.402 | **1.270** |

Full 24-row RSS dataset includes p=.5 for 8 and 24 qubits and the full-SVD control. The cap reduces modeled live-array entries in the 48-qubit p=.2 run from 634,464 to 477,792; actual peak RSS falls by 8.77 MiB. At 48 qubits, the minimum layer-retention norm is **0.00357** for p=.2 and approximately **2.07e-6** for p=.5 under the cap; accumulated error bounds saturate. Only bounded execution and resource usage are established at this size.

## Verification, reproduction and next research steps

All **90 unit tests** across QC-01 through QC-09 passed locally, including the gate residual/trace-distance inequality, deterministic low-rank reconstruction, nonbinding plan equivalence, binding caps, no use of dense oracle in the allocator, trace-one/positivity, 48-qubit model cap, and cross-module regression.

```bash
cd experiments/channel_mpo
python -m pip install 'numpy>=1.24,<3'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 test_qc04 \
    test_qc05 test_qc06 test_qc07 test_qc08 test_qc09
python benchmark_qc09.py --qubits 6,8 --strengths .2,.5 --seeds 0,1,2 \
    --output qc09_calibration_repro.csv
python benchmark_qc09.py --qubits 6,8 --strengths .2,.5 --seeds 3,4,5,6,7 \
    --output qc09_heldout_repro.csv
python benchmark_qc09.py --qubits 6,8 --strengths .2,.5 --seeds 3,5 \
    --channel dephasing --output qc09_dephasing_repro.csv
python benchmark_qc09.py --qubits 6,8 --strengths .2,.5 --seeds 3,5 \
    --channel amplitude_damping --output qc09_damping_repro.csv
python benchmark_peak_qc09.py --qubits 8,24,48 --strengths .2,.5 \
    --output qc09_peak_rss_repro.csv
```

The next rigorous step is a small interval-arithmetic certificate for the residual check, including explicit floating-point rounding, and a memory allocator that explicitly tracks QR and SVD workspace. To claim high-qubit physical accuracy, use a meaningful operational observable bound or validate against a separate high-fidelity simulator. Any proposed speed gain should be measured at a stated physical-error threshold rather than a fixed rank limit alone.
