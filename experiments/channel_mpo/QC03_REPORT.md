# QC-03: Environment factor gauges, local loss certificates, and quantum trajectories

**Date:** 2026-11-09  
**Status:** CPU pilot, frozen tests and data. Stacked on QC-02.  
**Scope:** one-dimensional brickwork Haar two-qubit circuits, with single-qubit CPTP noise after every layer. The physical channel is fixed within every controlled comparison.

## Research question

The Stinespring/Kraus representation permits arbitrary unitary changes of basis on the *discarded environment*. Can that freedom lower the cost of simulating a noisy circuit while keeping the represented physical density matrix accurate?

QC-02 directly evolves a locally purified matrix-product density operator (LPDO) but loses most of the purification norm when repeatedly capping local environments. QC-03 tests two separate approaches:

1. **Ancilla factor rerouting:** an exact two-site environment-basis permutation before Kraus and bond compression; optimize the *actual locally discarded norm* rather than only an individual environment spectrum.
2. **Quantum trajectories:** sample one Kraus outcome on each site and evolve a pure-state MPS; compare two different Kraus decompositions of the **same** depolarizing map, plus convex mixtures of those decompositions.

All observable error claims below are bounded to the tested circuits. Timings are single local CPU measurements under `OPENBLAS_NUM_THREADS=1`; they are not controlled multi-machine benchmarks.

## 1. Environment factor gauge

A locally purified site tensor is `A[left, physical_qubit, environment, right]`. Two adjacent environments following a noise layer can be factored as `(old_A, fresh_K_A)` and `(old_B, fresh_K_B)`. The contraction of two adjacent sites is `theta[left, pA, eA, pB, eB, right]`. Regrouping these four environment subfactors is a basis permutation on their *joint* Hilbert space. The physical state is exactly invariant before truncation:

\[
\rho = \mathrm{Tr}_{E_A E_B} |\Psi\rangle\langle\Psi|
     = \mathrm{Tr}_{E_A E_B} [(I\otimes U_E)|\Psi\rangle\langle\Psi|(I\otimes U_E^\dagger)].
\]

We evaluate all valid 2+2 factor groupings that preserve the original local environment dimensions (up to six candidates). Each candidate is compressed with the **same** two local environment caps and intersite bond cap; the selected grouping maximizes the squared purification norm surviving all three orthogonal SVD projections. The identity grouping remains a candidate, so this *local* retention score cannot deteriorate by design. The method does not guarantee monotone **final physical trace distance** after later gates, which act on different approximate density matrices.

`gauge_pair.py` also includes an experimental greedy two-level Givens ancilla rotation based on an environment-only objective. It is disabled in the main comparison (`proposals=0`), since minimizing local environment tails alone sometimes increases bond truncation loss and final trace distance.

### Dense-oracle results

Six brickwork layers; `max_bond=12`, `max_env=4`, relative SVD tolerance 0, seeds 0,1,2. Exact density-matrix evolution provides the reference. Each cell shows the **mean trace distance** across three seeds. `win` means the final trace distance decreased for that seed.

| n | depolarizing p | QC-02 sequential | QC-03 factor rerouting | wins / seeds | mean evolution time, QC-02 -> QC-03 |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.2 | 0.27761 | 0.25485 | 3/3 | 0.0285s -> 0.6525s |
| 6 | 0.5 | 0.13039 | 0.09849 | 3/3 | 0.0294s -> 0.6278s |
| 8 | 0.2 | 0.32359 | 0.32699 | 2/3 | 0.0509s -> 1.0365s |
| 8 | 0.5 | 0.15169 | 0.13605 | 3/3 | 0.0465s -> 1.0548s |

At p=0.5, the six-qubit mean reduction is **24.5%** and the eight-qubit reduction **10.3%**. At p=0.2, eight-qubit mean trace distance **worsens by 0.00340** despite improving two of three seeds. The optimizer costs roughly 20–23x QC-02 runtime on these tiny instances. Stored final tensor entry counts are unchanged under fixed caps (3,968 at six qubits; 6,272 at eight).

A separate environment-only control produced an eight-qubit p=0.5 seed-0 trace distance of **0.24873** (QC-02 baseline 0.13689). Using the *post-bond retained norm* as the selection criterion reduced that same run to **0.13285**. The choice of surrogate objective matters.

**Interpretation:** small-scale evidence that an exact environment gauge improves *some* compressed quantum states; the method is computationally expensive, remains dependent on noise/circuit/seed, and has not solved the retention collapse at 48 qubits. Treat it as an optimizer prototype, not a scalable simulator.

## 2. A conservative trace-distance certificate

We also implement a mathematical bound for the errors introduced by SVD truncation. Let a canonicalized purification have squared norm `w`, and let a particular orthogonal projection discard squared norm `\Delta` (so `\delta=\Delta/w`). Normalize the resulting state. Its squared overlap with the original purification is `1-\delta`. By pure-state trace distance and contractivity under partial trace, the physical states obey

\[
D(\rho,\tilde\rho) \le \sqrt{\delta}.
\]

Subsequent exact quantum channels contract trace distance. By the triangle inequality, across all individually normalized truncation events,

\[
\boxed{D(\rho_{\rm exact},\rho_{\rm simulated}) \le
\min\!\left(1,\sum_{j\in \text{truncations}}\sqrt{\delta_j}\right).}
\]

This applies to QR-canonicalized global-purification SVD projections. It includes the bond split and each local environment projection. Every layer starts normalized, while the implementation accounts for each intermediate *unnormalized* input norm to compute `\delta_j`. Norms predicted from the truncation ledger agree with the observed pre-normalization norm. The environment-only unitary gauge itself creates no physical error.

**Practical limitation:** this simple certificate becomes **1 (vacuous)** rapidly for the low-environment-cap cases. For example, six qubits at p=0.5 reach the vacuous value after layer two. This provides a mathematically meaningful high-accuracy gate in cases with small truncation, while preventing small *observed* errors from being labeled formally certified in heavily truncated runs. `run_pair()` reports `certified_trace_distance_bound` for every layer.

## 3. Kraus representations and quantum trajectories

For the same one-qubit depolarizing channel,

\[
\mathcal D_p(\rho)=(1-p)\rho + \frac p2 I\,\mathrm{Tr}(\rho),
\]

we compare:

**Four-Pauli Kraus:** `sqrt(1-3p/4) I` and `sqrt(p/4) X`, `Y`, `Z`. Each sampled outcome is proportional to a unitary, so local noise does not cut entanglement along a pure trajectory.

**Five-reset Kraus:** `sqrt(1-p) I` and `sqrt(p/2) |r><s|` for `r,s in {0,1}`. These rank-one outcomes measure/reprepare a qubit, breaking its entanglement with the rest of that sampled trajectory. The channel is exactly the same, even though the Kraus representation is nonminimal.

**Hybrid Kraus:** concatenate `sqrt(1-alpha)` times the four Pauli Kraus operators and `sqrt(alpha)` times the five reset operators. This gives the same CPTP map for any `0 <= alpha <= 1`. A tunable number of entanglement-cutting events trades trajectory MPS complexity against observable variance.

Each trajectory is sampled according to the conditional Born probabilities, evolves as a pure-state MPS, and remains exactly positive. Its mean local Pauli-Z is an unbiased estimate of the physical mixed-state expectation **when no bond truncation is applied**. All reported trajectories use unlimited MPS bond rank. The distribution of pure states, and hence the variance of an observable estimate, depends on the choice of Kraus representation.

### Fixed-seed trajectory comparison

Six layers, `p=0.5`, circuit seed 0, CPU, unlimited MPS bond rank. `entries` counts the maximum stored `complex128` tensor entries in any single sampled final MPS; time is complete ensemble evolution. Estimated uncertainty is one empirical standard error, not an exact error bound.

| n | shots | unraveling | max bond | max entries | Z0 standard error | ensemble seconds |
|---:|---:|---|---:|---:|---:|---:|
| 6 | 256 | Pauli | 8 | 168 | 0.01799 | 4.12 |
| 6 | 256 | reset | 4 | 104 | 0.05684 | 4.40 |
| 8 | 256 | Pauli | 16 | 680 | 0.01661 | 6.30 |
| 8 | 256 | reset | 8 | 232 | 0.05578 | 6.17 |
| 20 | 64 | Pauli | 64 | 43,688 | 0.03999 | 14.04 |
| 20 | 64 | reset | 8 | 664 | 0.11107 | 7.25 |
| 48 | 24 | Pauli | 64 | 158,376 | 0.05502 | 31.14 |
| 48 | 24 | reset | 8 | 1,080 | 0.17965 | 12.91 |

At 48 qubits, the reset unraveling used **147x fewer** final complex tensor entries and **2.4x less ensemble wall time** than Pauli, at the *same number of trajectories*. Its larger variance means it may require about **10.7x as many samples** to reach the same standard error for `Z0`. The observed error-per-time proxy `SE^2 * seconds` was **4.4x worse** for reset. This is a valuable memory/variance tradeoff, rather than a matched-accuracy runtime improvement. The maximum observed entries are not peak RSS.

For n=6 and n=8, the **exact density-oracle local Z values lie within two empirical standard errors** for all 256-shot cases at both p=0.2 and p=0.5; all code-level exact-equivalence checks pass. At 20 and 48 qubits no dense oracle is constructed. The 48-qubit QC-02 approximation's `Z0` at p=0.5 (about -0.01015) falls inside the finite-sample Pauli estimate's uncertainty interval; this does **not** certify the full 48-qubit approximate density matrix.

### Hybrid unraveling at p=0.5

| n | reset mix alpha | max bond | **mean** entries per trajectory | Z0 SE | seconds |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.25 | 32 | 4,355 | 0.08594 | 8.4 |
| 20 | 0.50 | 32 | 1,319 | 0.10277 | 7.5 |
| 20 | 0.75 | 16 | 484 | 0.10663 | 7.2 |
| 48 | 0.25 | 64 | 18,668 | 0.12140 | 14.4 |
| 48 | 0.50 | 32 | 4,698 | 0.13122 | 13.2 |
| 48 | 0.75 | 16 | 1,570 | 0.16862 | 12.8 |

This small suite illustrates a continuous memory/variance frontier. The three mixtures were predefined, while the sample-based efficiency statistics have substantial uncertainty at 24 or 64 shots. Further optimization should use replicated seeds and bootstrap confidence intervals.

## Reproduction

Only `numpy>=1.24` is required. From the checked-out repo's `experiments/channel_mpo` directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest -v test_channel_mpo test_purified_mpo test_qc03
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_gauges.py --qubits 6,8 --seeds 3 --depth 6 --strengths 0.2,0.5 --output qc03_gauge_comparison.csv
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_trajectories.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_hybrid.py
```

The last two commands are research-scale CPU jobs and stay outside light CI. Their output is frozen in the committed CSVs. The `trajectory_oracle.py` CLI permits narrower searches. The `--reset-mix` option defines alpha when `--unravellings hybrid` is selected.

## Protocol and decision gates

- **Gauge method:** use exact density-oracle trace distance for n<=10, keep noise/caps/seeds fixed, report both improvements and regressions. Evaluate objective, final fidelity, and extra CPU work separately.
- **Unraveling method:** require equivalent CPTP maps, per-outcome normalized sampling, and no MPS bond truncation for unbiased-estimator claims. Compare `SE^2 * time` at controlled shots and report MPS tensor entries independently of peak RSS.
- **Large-system conclusions:** 48-qubit trajectories provide unbiased local-observable *estimates*, subject to finite-sample uncertainty. The QC-02 fixed-environment simulation at that size remains globally unvalidated because of catastrophic norm loss.
- **Next gate:** optimize environment disentanglers using two-site isometries/DMRG-like sweeps, or measure the accuracy-runtime frontier of selected hybrid unravelings with confidence intervals and multi-seed repetitions. Benchmark fixed memory rather than fixed sample count before selecting a production simulator.

## Related work

- Müller, Ayral, Bertrand, *Enabling large-depth simulation of noisy quantum circuits with positive tensor networks*, arXiv:2403.00152. The environment-basis/disentangling motivation and distinctions between purification gauges and physical states.
- Godinez-Ramirez, Milbradt, Mendl, *Riemannian approach to the Lindbladian dynamics of a locally purified tensor network*, Phys. Rev. A 112, 012224 (2025), arXiv:2409.08127. Optimization over Kraus isometries for locally purified networks.
- Stucchi, Styliaris et al., *Structure and Classification of Matrix Product Quantum Channels*, Phys. Rev. Lett. 137, 120401 (2026). General structural context for locally purified matrix-product channels.

**Nonclaims:** The experiments supply no quantum speedup or cryptographic attack algorithm. They measure classical simulation behavior under chosen quantum noise models.
