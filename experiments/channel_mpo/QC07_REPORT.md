# QC-07: Joint bond/Kraus rank allocation with workspace admission and physical diagnostics

**Status:** CPU pilot completed 2026-11-09. This is a stacked follow-up to QC-06.
The main observation is that joint spatial-bond and local-environment rank selection
reduces exact small-system trace distance at the same retained-entry budgets.
The extra SVD and QR sweeps incur substantially greater running time and peak
resident memory. Accuracy at 24–48 qubits has **not** been established.

## Hypothesis and mathematical method

A locally purified quantum state is represented as

\[
 A_i\in\mathbb C^{b_{i-1}\times2\times e_i\times b_i},\qquad
 \rho=\operatorname{Tr}_{E}|A\rangle\langle A|,
 \quad b_0=b_n=1.
\]

The exact retained complex-entry count is

\[
 C(\mathbf b,\mathbf e)=2\sum_{i=1}^n b_{i-1}e_i b_i\le B_t,
\]

where `B_t` is the existing QC-06 **predeclared** layer-specific entry
budget. The local ancilla and intersite Schmidt spectra are obtained by
purely gauge-changing QR sweeps, *before* any rank truncation. The joint
allocator starts at rank one on every variable. At each step it considers
adding the next singular component, scored by squared singular value divided
by the exact incremental storage cost:

\[
 \Delta C(e_i)=2b_{i-1}b_i,\quad
 \Delta C(b_i)=2(b_{i-1}e_i+e_{i+1}b_{i+1}).
\]

The bond score receives a channel-specific coefficient
\(\eta(\mathcal E)=\lVert T_{\mathcal E}\rVert_2\), where the 3-by-3
traceless Pauli transfer matrix is
\((T_{\mathcal E})_{ab}=\frac12\operatorname{Tr}[\sigma_a\mathcal E(\sigma_b)]\).
For the implemented noise conventions, \(\eta_{\mathrm{depol}}=1-p\),
\(\eta_{\mathrm{dephasing}}=1\), and
\(\eta_{\mathrm{damping}}=\sqrt{1-p}\).
This coefficient makes the allocation **channel-aware**. It does **not**
provide a theorem equating the local score with global physical trace error.
The greedy integer allocator is heuristic; its nonlinear rank costs are
recomputed after each allocation. Deterministic tie-breaking is used.

The code applies spatial SVD truncation followed by environment SVD
truncation, retaining every selected rank up to its available numerical rank.
It checks the exact post-layer storage bound and normalizes the purification,
which preserves positivity. The policy never constructs a dense density matrix,
queries an exact oracle, uses future gates, or launches an additional teacher
rollout. The benchmark constructs exact density matrices only *after* all
approximate rollouts have completed.

Two physical-error diagnostics accompany the heuristic: all one-site reduced
states are contracted directly from the purification before and after each
layer compression; their trace distances quantify local observable changes.
For each orthogonal SVD truncation losing weight \(\delta\) from a purification
of squared norm \(N\), its normalized pure-state trace distance is
\(\sqrt{\delta/N}\). CPTP contractivity and the triangle inequality give the
floating-point diagnostic

\[
 D(\rho_{\rm exact},\rho_{\rm simulated})\ \le\
 \min\left\{1,\sum_{j\in\mathrm{all\ truncations}}\sqrt{\delta_j/N_j}\right\}
\]

in exact arithmetic assuming the initial state is exact and all intervening
channels are CPTP. Gate, bond, and environment truncations are included.
The bound clipped to **1.0 in all measured depth-six small-system cases**,
so it is *vacuous* at that depth. One-site marginal shifts are diagnostics,
not upper bounds on global trace distance.

## Memory definitions

- **Exact layer-boundary budget:** `C(b,e)<=B_t`, enforced after every layer.
  QC-06's frozen calibration schedule sets `B_t`, with bond cap 12 and
  uniform environment cap 6. This permits a fair comparison of final stored
  complex entries at every intermediate layer.
- **Explicit-array workspace admission:** a declared cap on estimated live
  array entries at two-site gates, Kraus insertion, and the compression sweeps.
  The default is 500,000 entries for up to eight qubits, raised to 750,000
  for the 48-qubit execution-only tests. An admission failure raises
  `MemoryError` before a gate/channel operation is attempted. This model
  excludes hidden NumPy/LAPACK workspaces and hence is **not** a hard bound
  on process memory.
- **Peak RSS:** measured by running each policy in a fresh Linux subprocess,
  `resource.ru_maxrss` after completion. These are observed peaks on the
  local test host, not portable limits or a guarantee of future runs.

The benchmark is *matched at layer-boundary entry counts*. Its observed
peak RSS values are **not matched**. QC-07 pays extra peak memory and extra
compute for improved accuracy. Future work should reduce these costs and
rerun against a matched measured-RSS ceiling.

## Frozen protocols

Main study: six-layer nearest-neighbor Haar brickwork, noise after each
layer, depolarizing \(p\in\{0.2,0.5\}\), qubits \(n\in\{6,8\}\), bond
cap 12 for all three methods, local environment cap 6 for the uniform
baseline, cap 12 for QC-06 and QC-07. Both QC-06 and QC-07 use identical
calibration-derived `qc06_budget_schedule.json` values on every layer:

| n | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 | Layer 6 |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 96 | 416 | 1728 | 4800 | 7200 | 7200 |
| 8 | 128 | 608 | 2496 | 7104 | 10656 | 10656 |

Seeds 0–2 serve as calibration, and **3–7 are the held-out test set**.
Rank caps, weighting rule, and budgets are fixed for held-out evaluation.
The `qc07_calibration.csv` and `qc07_heldout.csv` files contain every paired
trace distance, observable error, rank allocation per layer, storage count,
wall time, and retention metric. Timing excludes the exact dense oracle.

### Main depolarizing results

| Split | Circuits | Uniform env6 mean D | QC-06 adaptive-env mean D | QC-07 joint mean D | QC-07 wins vs QC-06 |
|---|---:|---:|---:|---:|---:|
| Calibration, seeds 0–2 | 12 | 0.137525 | 0.131938 | **0.064101** | 12/12 |
| Held-out, seeds 3–7 | **20** | 0.142776 | 0.132490 | **0.069909** | **20/20** |

Held-out relative mean distance reduction vs QC-06: **47.23%** at the same
predeclared layer-boundary entry limits. Mean evolution time was **0.0589s**
for QC-06 versus **0.1997s** for QC-07; these are single process timings
rather than replicated timing distributions. QC-07 averaged 8,855 final
retained entries against a 8,928-entry allowed average. Its largest modeled
explicit-array demand in the 20 held-out tests was 434,016 entries, below
the 500,000-entry admission cap.

| Held-out group | QC-06 mean D | QC-07 mean D |
|---|---:|---:|
| 6q, p=0.2 | 0.152407 | 0.08655 |
| 6q, p=0.5 | 0.072501 | 0.03709 |
| 8q, p=0.2 | 0.197069 | 0.11263 |
| 8q, p=0.5 | 0.107983 | 0.04337 |

### Ablations and interpretation

Running QC-07 with spatial gate cap **16** rather than **12** on the same 20
held-out depolarizing circuits gave mean D **0.069628** versus **0.069909**.
The gate-cap-16 configuration consumed substantially greater modeled working
space, up to **652,656** rather than **434,016** entries. All 20 cases still
beat QC-06 at gate cap 12. The gain is therefore largely due to the **joint
rank allocation**, rather than a looser internal two-qubit gate cap.

Removing the channel contraction coefficient by fixing \(\eta=1\) on the
12 calibration cases produced mean D **0.064115**, compared with
**0.064101** for the channel-aware main policy. The channel-specific weight
is a small effect on this test grid; the largest benefit comes from jointly
reallocating bond and environment rank. There is insufficient evidence of a
separate general advantage from the transfer-matrix weight.

### Channel transfer, exploratory

Dephasing and amplitude damping each receive **independent frozen budget
schedules** generated on seed 0, verified to match seeds 1 and 2, and saved
in `qc07_transfer_budget_schedule.json`. No held-out seed contributes to
these schedules. Held-out seeds are 3–5; qubits and noise strengths are as
in the main study. This is exploratory channel transfer, rather than an
independent preregistered confirmation.

| Channel | Circuits | QC-06 mean D | QC-07 mean D | QC-07 wins |
|---|---:|---:|---:|---:|
| Dephasing | 12 | 0.133635 | **0.070528** | 12/12 |
| Amplitude damping | 12 | 0.133313 | **0.083426** | 12/12 |

The amplitude-damping improvement is notable relative to QC-06's earlier
ancilla-only regression, but is observed only for these particular small
circuits. The reallocated spatial bonds may help preserve correlations that
purely local singular-energy rankings miss.

### Observed peak memory and execution-only scaling

Separate Python processes, seed 3, six layers; each policy is subject to the
same post-layer entry budget. QC-07 uses a 750,000-entry modeled working cap.

| n / p | QC-06 peak RSS | QC-07 peak RSS | QC-06 seconds | QC-07 seconds |
|---|---:|---:|---:|---:|
| 8 / .2 | 99.3 MiB | 107.7 MiB | 0.075 | 0.285 |
| 24 / .2 | 100.5 MiB | 109.0 MiB | 0.337 | 1.011 |
| 48 / .2 | 101.9 MiB | 114.6 MiB | 0.790 | 2.581 |
| 48 / .5 | 104.1 MiB | 111.3 MiB | 0.788 | 2.656 |

Untruncated dense-state validation is infeasible for 24–48 qubits. For 48q
and \(p=.5\), QC-06's minimum per-layer retention was about **1.03e-6**;
QC-07's was about **3.86e-6**. The simulation executes at that scale but the
small retained norm means there is **no supported fidelity claim** there.
The 48-qubit results also show a clear increase in measured peak RSS.

## Reproduction and acceptance

From `experiments/channel_mpo` in the stacked branch:

```sh
python -m pip install 'numpy>=1.24,<3'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 \
    test_qc04 test_qc05 test_qc06 test_qc07
python benchmark_qc07.py --seeds 0,1,2 --output qc07_calibration.csv
python benchmark_qc07.py --seeds 3,4,5,6,7 --output qc07_heldout.csv
python benchmark_qc07.py --seeds 0,1,2 --bond-weight-override 1 \
    --output qc07_weight1_calibration.csv
python benchmark_transfer_qc07.py --output qc07_transfer.csv
python benchmark_peak_qc07.py --qubits 8,24,48 --strengths .2,.5 \
    --work-cap-entries 750000 --output qc07_peak_rss.csv
```

The core tests cover exact untruncated evolution, normalized positivity,
channel transfer coefficients, deterministic budgets, physical one-site
marginals, no-oracle policy execution, modeled-memory admission failure, and
the truncated physical-distance bound on small cases. The full stacked suite
currently contains **67 passing tests**. The lightweight CI job runs tests
plus one fixed 6q/noise benchmark smoke test; it does not reproduce the full
48-qubit or held-out dataset in GitHub Actions.

## Open limitations and next experiments

1. Replace the repeated QR/SVD scans with a streaming locally purified rank
   allocator, aiming to preserve the fidelity improvement while recovering
   QC-06's CPU and RSS costs. A strict allocator should report both the
   calibrated per-process RSS cap and its independent retained-entry limit.
2. Compare the proposed spectral score with an online estimate of a bounded
   set of nonlocal Pauli-observable errors. The present one-site marginals
   and conservative full trace bound do not discriminate well at depth six.
3. Repeat at deeper circuits, different gate ensembles, randomized seeds,
   and calibrated noise models, including coherent errors. Test channel-aware
   weighting on a larger design before treating it as useful.
4. Prioritize improved layer-retention diagnostics for 24–48 qubits; larger
   simulations without exact validation are execution benchmarks only.
