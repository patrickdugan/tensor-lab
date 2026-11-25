# QC-08: single-sweep Kraus/bond spectra and memory-aware gate scheduling

**Status:** reproducible CPU pilot on 6/8 physical qubits; large-system storage/RSS stress test to 48 qubits.  Development continues on top of QC-07 PR #7.  Numerical results below use Python 3.13.5 / NumPy 2.3.5 with `OPENBLAS_NUM_THREADS=1`. The CI suite also supports Python 3.11 / NumPy 1.24+.

## Research question

QC-07 jointly allocates spatial bond ranks and local Kraus-environment ranks under fixed retained-entry budgets and considerably improves small-circuit trace-distance error over QC-06. However, QC-07 repeats both canonical QR sweeps and diagnostics in each layer; a 48-qubit run uses roughly 114 MiB of peak process RSS, against QC-06's roughly 103 MiB.

Does a streaming execution preserve QC-07's density matrices while reducing CPU work and bringing peak memory closer to QC-06?  This note separates **faster execution** from **lower actual RSS**, and retains a high-qubit fidelity caveat.

## Numerical method

The purification consists of cores \(A_i\in\mathbb C^{b_{i-1}\times 2\times e_i\times b_i}\), with \(b_0=b_n=1\). The exact layer-boundary storage proxy is

\[
C(b,e)=2\sum_{i=1}^n b_{i-1}e_i b_i.
\]

QC-07's allocation policy maximizes a local squared-singular-value gain per incremental entry, with the spatial-bond gain scaled by the norm of the channel's traceless Pauli transfer block. The **policy itself is unchanged**; the work here concerns spectrum acquisition, gate scheduling and compression order.

For each layer, `scan_joint_spectra` makes one right-canonical QR sweep, then advances the center tensor to the right. It reads the outgoing spatial-bond spectrum from the **R** factor of the QR of the center matricization \(M=QR\): \(M^\dagger M=R^\dagger R\), so the singular values agree. At the same center it reads the environment spectrum from the smaller Hermitian Gram matrix and uses a direct complex SVD if the smallest estimated singular value is less than \(10^{-6}\) times the largest. That fallback prevents floating-point rank-tie changes on the dephasing/damping pilots. The local QR transforms are exact purification gauge changes.

The `classic` policy retains QC-07's *bond-then-environment* compression order. It uses a monotone nearest-neighbor gate-layer sweep and omits per-layer one-site marginal diagnostics unless explicitly requested. This keeps the same physical final density matrix, although the backend's temporary allocation pattern can use **more** RSS. `classic_mem` uses QC-07's sequential gate calls with the faster joint spectral scan, which is more memory-friendly on the tested machine. `fused` truncates each environment followed by its outgoing spatial bond during one left-to-right compression sweep; this deliberately changes the mathematical truncation order and can change accuracy. It is experimental, not the preferred accuracy-preserving policy.

All modes use the same frozen QC-06 per-layer retained-entry budgets, with bond and environment caps 12, depth six, and a modeled workspace admission cap of 750,000 complex entries. The reported RSS is measured separately in one fresh process per policy. Neither the layer-boundary budget nor the explicit array-work model guarantees a bound on NumPy/LAPACK peak RSS. The dense-reference channel is used only for post-run validation with at most eight qubits.

## Held-out exact-density comparison

20 paired six-layer depolarizing circuits: n in {6, 8}, p in {.2, .5}, seeds 3, 4, 5, 6, 7. Both the circuit gates and the per-layer entry limits are identical for all modes.

| Mode | Mean evolution time (s) | Mean trace distance | Median speedup vs QC-07 | Physical state vs QC-07 |
| --- | ---: | ---: | ---: | --- |
| QC-07 | 0.21586 | 0.0699090 | 1.00x | Baseline |
| QC-08 `classic` | 0.14833 | 0.0699090 | **1.43x** | Maximum difference: 4.01e-15 |
| QC-08 `classic_mem` | 0.14932 | 0.0699090 | **1.43x** | Maximum difference: 4.01e-15 |
| QC-08 `fused` | 0.14871 | 0.0702696 | 1.47x | Changes physical state; mean trace-distance increase 0.000361 |

`classic` and `classic_mem` reproduced QC-07's exact rank plans for all 20 depolarizing cases. The fused mode retained the exact plan for 18/20 and improved trace distance in 4/20; its mathematical truncation order changed 16/20 results for the worse. Runtime includes gate evolution, Kraus application, compression, and normalization; it **excludes** the dense oracle. The speedup incorporates the removal of optional one-site diagnostics; it is not attributable to a single optimization alone.

### Cross-channel spot-check

An additional 16 circuits across dephasing and amplitude damping use n in {6,8}, p in {.2,.5}, seeds 3 and 5. The environment-score SVD fallback produced identical rank plans in all 16 cases. The QC-08 `classic` physical density matrix agrees with QC-07 to within 4.4e-15 in trace distance. The fused variant is not covered by that equivalence.

## Actual peak memory, measured in isolated processes

Seed 3, 48 qubits, six layers, depolarizing noise. Times are **single trials**, not confidence intervals.

| Mode | Peak RSS p=.2 (MiB) | Time p=.2 (s) | Peak RSS p=.5 (MiB) | Time p=.5 (s) |
| --- | ---: | ---: | ---: | ---: |
| QC-06 | 102.69 | 0.73 | 104.02 | 0.91 |
| QC-07 | 113.50 | 2.40 | 111.48 | 2.67 |
| QC-08 `classic` | 131.09 | 1.48 | 131.11 | 1.74 |
| QC-08 `classic_mem` | **113.02** | 2.55 | **110.36** | 2.26 |
| QC-08 `fused` | 131.30 | 1.49 | 132.60 | 1.68 |

The `classic_mem` variant comes close to QC-07's process RSS, and in these two trials sometimes runs faster and sometimes slightly slower. **QC-06 retains the lowest observed peak RSS.** The fastest one-pass gate schedule is measurably heavier in temporary memory, despite identical retained-entry counts. Additional 8- and 24-qubit measurements are recorded in `qc08_peak_rss.csv`.

At 48 qubits the smallest pre-normalization layer retention is about 0.00572 for p=.2 and only 3.86e-6 for p=.5 in the exact-preserving QC-08 modes. Since there is no tractable dense-state reference at that size and the cumulative truncation bound is saturated, these runs validate execution and bounded stored tensor entries **only**, without an end-to-end fidelity claim.

## Reproduction and acceptance

```bash
cd experiments/channel_mpo
python -m pip install 'numpy>=1.24,<3'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 test_qc04 test_qc05 test_qc06 test_qc07 test_qc08

OPENBLAS_NUM_THREADS=1 python benchmark_qc08.py --qubits 6,8 --strengths .2,.5 \
  --seeds 3,4,5,6,7 --output qc08_heldout.csv
OPENBLAS_NUM_THREADS=1 python benchmark_qc08.py --qubits 6,8 --strengths .2,.5 \
  --seeds 3,5 --channel dephasing --output qc08_dephasing.csv
OPENBLAS_NUM_THREADS=1 python benchmark_qc08.py --qubits 6,8 --strengths .2,.5 \
  --seeds 3,5 --channel amplitude_damping --output qc08_amplitude_damping.csv

# Separate process per case; split large-n batches for easier diagnostics.
OPENBLAS_NUM_THREADS=1 python benchmark_peak_qc08.py --qubits 8,24 \
  --strengths .2,.5 --output qc08_peak_rss.csv
OPENBLAS_NUM_THREADS=1 python benchmark_peak_qc08.py --qubits 48 \
  --strengths .2,.5 --output qc08_peak_rss_48.csv
```

Run tests through `unittest` only; there is no GPU/IBM hardware requirement. Individual `qc08_peak_rss.csv` measurements were preserved after one orchestration timeout and finished in fresh subprocesses. Their recorded numbers represent one process each; no statistical variance estimate is implied. Any re-run overwrites the target output file.

## Checks

- 77/77 local unit tests across QC-01 through QC-08 passed with the final threshold.
- The combined scan agrees with QC-07's independent spatial and environment spectra within floating-point tolerance and leaves the exact small-n density unchanged.
- Fast gate-layer execution reproduces sequential gates including truncation; positivity and trace-one checks pass after normalization.
- The policy does not query the dense-state oracle. Model work-cap rejection is tested; the actual process-memory caveat remains.
- All three output modes enforce the frozen per-layer entry budget. Both accuracy-preserving modes agree with QC-07's final densities across 36 circuits.

## Next engineering direction

The main remaining expense is full two-site gate SVD and its temporary arrays. A stronger improvement would use rank-revealing, blockwise or randomized reduced SVD with independent certification of the discarded tail, together with a true allocation scheduler for peak live buffers. Any new approximation must be judged by global physical trace distance on the six- and eight-qubit oracle and by actual peak RSS measured in separate processes. Large-qubit results should remain tagged **fidelity-unverified** until a meaningful error bound or a validated observable estimator exists.
