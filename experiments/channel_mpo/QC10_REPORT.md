# QC-10: Exact causal-cone validation of operational observables at 48 qubits

**Date:** 2026-11-09. **Status:** reproducible local CPU experiments. Stacked on QC-09 PR #10. All tests and results concern six-layer, nearest-neighbor brickwork circuits starting in the product state |0...0>, with independent single-qubit CPTP noise after each gate layer. The phrase *exact local reference* means the full mathematical model on a finite dependency cone, computed in NumPy floating-point arithmetic. No interval-arithmetic certificate is claimed.

## Research question

QC-09 reduces time and modeled gate memory but its naive sum of purification-truncation residuals becomes uninformative at 48 qubits. It cannot validate global physical fidelity. Can locality certify **specific operational measurement errors** at 48 qubits without constructing a 2^48-dimensional state, and does the peak-memory policy preserve these observables?

**Answer:** For output measurements near a boundary, the exact numerical reference needs at most 6 qubits for one edge site or 8 qubits for an adjacent edge pair at the tested depth. The 48-qubit compressed simulator has modest errors on many of those marginals despite very low purification retention, but dephasing can produce substantial local regressions. A post-run acceptance gate makes the tolerated local measurement error explicit.

## Mathematical construction and scope

For a qubit channel \(\mathcal E(\rho)=\sum_jK_j\rho K_j^\dagger\), its adjoint obeys

\[
\mathcal E^\dagger(I)=\sum_jK_j^\dagger K_j=I.
\]

Consequently a local channel outside the current Heisenberg support leaves the observable unchanged; a two-qubit gate increases support only if one of its legs is already supported. Beginning with a target set \(T\) at output and traversing circuit layers backwards, for every gate on \(\{i,i+1\}\) intersecting the active set, add both gate sites to that set. This computes the exact backward causal cone \(S_0\).

Under the stated product initial state, the full circuit's reduced output on \(T\) equals the output on \(T\) from the restricted circuit on \(S_0\), retaining gates internal to \(S_0\) and all local channels there. Boundary-crossing gates cannot affect the target observable by the backward-support construction. This is exact in the ideal mathematical circuit model. The implementation checks that each gate layer has disjoint nearest-neighbor pairs and every noise channel is trace-preserving.

For the measured depth-six brickwork schedule on 24 or 48 qubits:

| Output target | Number of cone qubits | Dense cone matrix shape |
|---|---:|---:|
| Leftmost or rightmost qubit | 6 | 64 × 64 |
| Adjacent edge pair | 8 | 256 × 256 |
| Center site in 48-qubit chain | 12 | 4,096 × 4,096 |

The implementation caps the oracle at 8 qubits by default and explicitly refuses wider cones. Its 6/8-qubit comparisons reproduce a complete dense-circuit reference to ~1e-12 trace distance, including nonunital amplitude damping. Full dense simulation of the 48-qubit circuit is never attempted. General nonlocal gates, an unknown correlated initial state, or a non-trace-preserving local operation require additional dependency information.

For a target region \(T\), construct a tensor-network marginal \(\widetilde\rho_T\) by contracting the QC-09 locally purified chain, then compute

\[
D_T=\tfrac12\|\rho_T^{\mathrm{cone}}-\widetilde\rho_T\|_1.
\]

Every POVM **event** on that region obeys \(|P(E)-\widetilde P(E)|\le D_T\), and every Hermitian observable with \(\|O\|_\infty\le1\) obeys \(|\langle O\rangle-\langle O\rangle_{\sim}|\le2D_T\). These are precise finite-dimensional inequalities applied to numerically calculated densities. Floating-point error remains unbounded formally. Local trace distance is also a *lower bound* on the full 48-qubit trace distance, and **is not an upper bound** on global trace distance.

## Frozen protocol

- **Simulator:** QC-09 `CertifiedPeakChain`, rank cap 12 for bond and environment, seeded low-rank gate factors, oversampling 4, two power iterations, retained-entry budgets from `qc06_budget_schedule.json` at every layer. Both variants run the identical gates and channels.
- **Modes:** `sketch` without prospective peak cap, and `peak` with a 350,000-complex-entry modeled peak cap at n=24 or 550,000 at n=48. This is a planning model, not guaranteed peak RSS.
- **Primary held-out grid:** n={24,48}, p={0.2,0.5}, seeds 3–7, depolarizing noise. Twenty independent circuit configurations, four measured regions, and two modes produce 160 rows.
- **Channel-transfer grid:** n=48, p={0.2,0.5}, seeds 3–7, amplitude damping or dephasing; 160 additional rows.
- **Operational gate:** predeclared \(D_T\le0.05\) for each of the four edge regions (two one-site and two adjacent two-site). Prefer the peak-budgeted simulator if every region passes, otherwise fall back to the sketch if every region passes, otherwise reject the run. This decision is *post-run validation*, using the independent reference; it is not an a priori guarantee or a cheaper estimator.

## Depolarizing held-out results

Mean local trace distance across 40 measured region/mode observations per qubit count:

| Qubits | QC-09 sketch | QC-09 peak-budgeted | Worst observed sketch | Worst observed peak |
|---|---:|---:|---:|---:|
| 24 | **0.01878** | 0.02293 | 0.04524 | 0.06384 |
| 48 | **0.02010** | 0.02747 | 0.05591 | 0.06198 |

At 48 qubits, the sketch's mean time was ~1.097 s and the peak mode's ~1.182 s on this CPU; each 6-qubit one-edge oracle took about 0.02 s, and each 8-qubit adjacent-edge oracle about 0.30 s. These timings are from a single-threaded local run and depend on machine. The mean final tensor entry count for 48 qubits was ~79,692 for `sketch` and ~79,727 for `peak`; the peak policy admits different **rank locations**, despite a similar final count. Modeled live-array peak across these runs was at most 656,448 entries for sketch and 483,840 for peak.

The sketch had lower local trace distance in **70/80** paired region observations across the 24/48 grid. Among the 40 measured 48-qubit sketch regions, 39 passed an event-error threshold of 0.05; 36 of 40 peak-budgeted regions passed. Individual failures mattered. At p=0.2, seed 3, the sketch had right-pair D=0.05591 while the peak mode reached 0.06198. At p=0.5, seed 3, the right-pair D values were 0.01825 and 0.05775 respectively.

This is evidence of an application-specific cost to the memory cap, even when its final tensor entry count is similar. No monotonic error or speed guarantee follows for future seeds.

## Channel transfer at 48 qubits

The same five held-out seeds, two strengths, four target regions and two modes:

| Channel | p | Mean local D | Max local D |
|---|---:|---:|---:|
| Dephasing | 0.2 | 0.02480 | 0.06653 |
| Dephasing | 0.5 | 0.04559 | **0.14405** |
| Amplitude damping | 0.2 | 0.01701 | 0.02822 |
| Amplitude damping | 0.5 | 0.01703 | 0.04297 |

Both modes gave numerically identical reduced states in the 80 paired channel-transfer observations (the peak cap was inactive at these parameters). Dephasing at p=0.5 can be substantially less trustworthy on the right boundary than on the left; the four-region acceptance gate catches these cases. Some QC-09 layer-retention numbers are far below 1%, but small retention alone cannot resolve the operational quality of any particular measurement.

### Local operational acceptance at tolerance 0.05

| Family | Configurations | Peak accepted | Sketch fallback | Rejected |
|---|---:|---:|---:|---:|
| Depolarizing, n=24 and 48 | 20 | 15 | 4 | 1 |
| Dephasing, n=48 | 10 | 7 | 0 | 3 |
| Amplitude damping, n=48 | 10 | 10 | 0 | 0 |
| **Total** | **40** | **32** | **4** | **4** |

The rejection list includes depolarizing n=48 p=0.2 seed 3, and dephasing n=48 with (p,seed)=(0.2,7),(0.5,3),(0.5,7). A rejected circuit means **both** candidate modes exceed 0.05 on at least one of the four nominated regions. No rejection is evidence of unsafe behavior outside those regions; the acceptance scope is deliberately narrow.

## Independence of the two boundaries

A second oracle forms the exact *joint* density on the two outermost sites from the tensor product of the two separately evolved local cones, but only after verifying that their backward cones are disjoint. Because the initial state is a product and no intervening gate connects the cones inside their causal history, the exact joint output factorizes.

On the 20 depolarizing 48-qubit runs (five seeds, two strengths, two QC-09 modes), the compressed simulator exhibited spurious independence distance below **1.0e-15** between its two-edge marginal and the product of its own edge marginals. Connected \(Z_0Z_{47}\) correlations were below **1.4e-16** in absolute value. The difference between the exact and approximate remote-edge *joint densities* remained nonzero (mean D=0.01920 for sketch and 0.02696 for peak), dominated by individual-site errors. These observations are specific to the tested shallow noisy circuits and do not prove global factorization under other schedules.

## Verification and reproducibility

The new test module covers: cone widths and overlap rejection, strict gate/channel validation, equality against independently evolved dense density matrices for depolarizing, dephasing and amplitude damping, tensor-network reduced-density contractions, exact two-edge factorization, rejection of wider-than-cap cones, analytical POVM/observable bounds, and a real 48-qubit end-to-end smoke test.

```bash
cd experiments/channel_mpo
python -m pip install 'numpy>=1.24,<3'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 test_qc04 \
  test_qc05 test_qc06 test_qc07 test_qc08 test_qc09 test_qc10
python benchmark_qc10.py --qubits 24,48 --strengths .2,.5 \
  --seeds 3,4,5,6,7 --channels depolarizing --output qc10_depolarizing_repro.csv
python benchmark_qc10.py --qubits 48 --strengths .2,.5 \
  --seeds 3,4,5,6,7 --channels dephasing,amplitude_damping --output qc10_channels_repro.csv
python qc10_acceptance.py --source qc10_depolarizing_repro.csv \
  --tolerance .05 --output qc10_acceptance_depolarizing_repro.csv
python qc10_acceptance.py --source qc10_channels_repro.csv \
  --tolerance .05 --output qc10_acceptance_channels_repro.csv
python benchmark_remote_qc10.py --qubits 48 --strengths .2,.5 --seeds 3,4,5,6,7 \
  --modes sketch,peak --output qc10_remote_repro.csv
```

Frozen CSV files in this PR contain 320 main local observations, 40 accept/reject decisions, and 20 remote-edge observations. Timing excludes oracle work from sim evolution and does not assert cross-machine speed portability.

## Limits and next experiments

The independent local reference gives an operational assessment for nominated shallow-circuit regions even when global truncation bounds saturate. For a center qubit at depth six, the cone is 12 qubits and a dense oracle would require a 4,096×4,096 complex matrix (~256 MiB before working allocations). Extensions should implement a Heisenberg-picture tensor contraction for interior observables and compare its numerical stability with the present Schrödinger-cone oracle. A second line is a **cone-weighted, locally observable-aware rank scheduler**, benchmarking at matched *physical* event-error thresholds and measured RSS. Both extensions must preserve independent reference validation to prevent proxy-only gains.
