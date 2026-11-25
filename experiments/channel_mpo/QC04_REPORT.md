# QC-04: Screened Kraus-gauge search reduces the QC-03 cost

**Status:** CPU experiment with a dense validation oracle for 6 and 8 qubits.
Code: `qc04_gauge.py` plus a backward-compatible `fast_factor_search` switch in
`gauge_pair.py`. Baseline QC-03 stays the default. The frozen observations live
in `qc04_paired.csv`. Benchmark code and tests are included in this change.

## Research question

QC-03's two-site Kraus-environment factor rerouting improved the compressed
physical density matrix for several noisy 6–8 qubit circuits, but its factor
search paid for full singular-value decompositions on every candidate. Can we
accelerate that search without losing the gains seen against QC-02's direct
sequential compression?

## Method

The original QC-03 score is the local discarded squared purification norm
after first projecting environment A, then environment B, and finally splitting
the intersite bond at a fixed rank cap. Let \(T\) be the canonicalized two-site
purification tensor and \(\|T\|_F^2=w\). Write \(d_A\) and \(d_B\) for the
individual optimal rank-\(r\) Frobenius tails of its two environment
matricizations. Any *sequential* local environment truncation followed by a
bond projection retains at most the norm retained by either single optimal
environment projection. Hence the total loss \(L\) obeys:

\[
\boxed{L\geq\max\{d_A,d_B\}.}
\]

The bound is derived from orthogonal projection monotonicity and the
Eckart–Young theorem. No claim is made that the bound equals the combined
projection loss. Candidate two-site factor permutations are exactly unitary on
the combined Kraus environment; they change the subsequent truncation error.

QC-04 therefore:

1. Computes the identity score once and enumerates valid nonidentity factor
   permutations, preserving the complete candidate set.
2. Computes the two inexpensive marginal tails; candidates whose analytic
   lower bound exceeds the best measured loss (with a roundoff allowance) are
   skipped. Candidates are visited in increasing lower-bound order.
3. Scores the others using eigendecompositions of small ancillary Gram matrices
   \(MM^\dagger\) to obtain their leading retained subspaces. The last bond
   uses a singular-values-only SVD. Ill-conditioned ancillary spectra fall
   back to QC-03's full SVD.
4. Verifies any proposed winning gauge with QC-03's original full-SVD
   retained-weight objective relative to the identity before accepting it.
   This conservative check cannot certify global optimality of the
   Gram-ranked candidate set. Mixed precision/ties warrant further tests.

The new switch is accepted only for `factor_swaps=True`, `proposals=0`, and
`factor_objective="retention"`. Existing QC-03 behavior remains available.

## Validation and measured runtime

**Protocol:** 6 and 8 qubits; 6 brickwork layers; depolarizing strengths
\(p\in\{0.2,0.5\}\); 3 circuit seeds (0,1,2); maximum bond 12; maximum
local Kraus environment 4; no local tolerance; NumPy `2.3.5`, Python `3.13.5`,
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1`. Single process, no concurrency.
All policies use the same exact density-matrix reference. Reported circuit
runtime includes gates, channels, compression and normalization, while
excluding dense reference evolution and reconstructing its output. CPU timing
is a single sample per matched configuration; its variance remains unmeasured.

| Configuration | QC-02 mean trace distance | QC-03 and QC-04 mean trace distance | QC-03 mean seconds | QC-04 mean seconds | time ratio |
|---|---:|---:|---:|---:|---:|
| 6q, p=.2 | .277612 | .254845 | .674 | .300 | 2.25x |
| 6q, p=.5 | .130386 | .098493 | .655 | .276 | 2.38x |
| 8q, p=.2 | .323593 | .326994 | 1.167 | .522 | 2.23x |
| 8q, p=.5 | .151687 | .136046 | 1.173 | .538 | 2.18x |

Across 12 paired configurations, the median QC-03/QC-04 runtime ratio was
**2.13x**, with a range **1.88x to 3.44x**. QC-04 final trace distances matched
QC-03 to at most `5.6e-17` in the benchmark, meaning the same candidate gauges
were selected in these cases. The search skipped **378 candidate scores** in
aggregate. Its `exact_candidates_scored` counter counts Gram-scored candidates
including the identity, while QC-03's `gauge_evaluations` counts attempted
nonidentity candidates, so those counters are not directly comparable.

**Fidelity caveat:** QC-03 and QC-04 still regress slightly against QC-02 on
8q at p=.2 (mean trace distance .326994 vs .323593). The observed speed gain
is for QC-03's local optimizer at fixed caps, not an improvement to the
underlying noise simulation accuracy or the 48-qubit retention problem.

**Memory caveat:** QC-04 screens candidate tensors materialized in memory and
holds them during the score pass. Peak RSS, SVD workspace, and GPU performance
have not been measured. Gram eigenvalue truncation squares the condition number,
which is why the fallback and final full-SVD check are part of this implementation.

## Tests

From the existing `experiments/channel_mpo` directory:

```bash
python -m pip install 'numpy>=1.24,<3'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest -q \
  test_channel_mpo test_purified_mpo test_qc03 test_qc04
```

Local result: **35 tests passed**, covering full-SVD/Gram retained-norm
agreement on random and rank-deficient tensors, analytic lower bounds,
candidate selection against the QC-03 oracle, physical-state invariance,
exact density validation, positivity, truncation certificates, and QC-01–03
regressions. GitHub CI uses the same command with Python 3.11. Wall timings
are retained in the frozen CSV rather than asserted in CI.

To regenerate the paired data:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_qc04.py \
  --qubits 6,8 --seeds 3 --strengths 0.2,0.5 --depth 6 \
  --bond 12 --env 4 --output qc04_paired.csv
```

Further work should improve the 8q/p=.2 failure case, replicate timings and
measure peak memory at fixed fidelity, then try a candidate sampler for
larger Kraus decompositions. Gauge permutation search remains a classical
simulation technique and supplies no evidence of quantum acceleration or an
attack on ECDSA.
