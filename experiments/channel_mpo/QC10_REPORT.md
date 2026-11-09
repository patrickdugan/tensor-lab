# QC-10: Exact local-observable validation at 48 qubits

**2026-11-09. CPU numerical experiment. Stacked on QC-09 PR #10.**

## Why

QC-09's accumulated purification-truncation bound saturates at six layers. The 48-qubit simulation executes within its memory budget, but its very low retained norms leave global-state fidelity unresolved. QC-10 independently evaluates precise **local** operational errors using backward causal cones.

## Derivation

For any local CPTP noise channel with Kraus operators K_j, its Heisenberg adjoint obeys

\[
\mathcal E^\dagger(I)=\sum_jK_j^\dagger K_j=I.
\]

A local channel outside the support of an observable leaves that observable unchanged. Traversing nearest-neighbor gate layers in reverse, add both legs of every two-site gate touching the current support. The resulting set S is the backward causal cone of the output target T. When the input is the product |0...0>, a simulation restricted to S with its original internal gates and local noise computes the exact reduced output state on T in ideal arithmetic.

At six brickwork layers, either edge qubit has a **6-qubit** input cone (64x64 density matrix), and either adjacent edge pair has an **8-qubit** cone (256x256). A center qubit in the 48-qubit chain has a 12-qubit cone and exceeds the default oracle cap. The code refuses such cases. Long-range gates and unknown correlated initial states require an appropriately expanded dependency calculation.

For exact local reference rho_T and compressed simulator marginal rho_approx_T, evaluate

\[
D_T=\tfrac12\|\rho_T-\rho_{\mathrm{approx},T}\|_1.
\]

For every POVM **event** E on T, the probability error satisfies |P(E)-P_approx(E)| <= D_T. For every Hermitian O with operator norm at most one, the expectation error is at most 2 D_T. These are analytical inequalities applied to computed densities. The reference uses NumPy floating point and does not claim a rounding-error certificate. **A small local D cannot upper-bound global-state trace distance.**

## Frozen benchmark

The original QC-06 six-layer entry schedules are reused. Both modes retain QC-09's deterministic sketch-gate method (oversampling 4, two power iterations) and bond/environment cap 12. The sketch mode has no prospective peak cap; peak mode uses a modeled 350,000-entry cap at n=24 and 550,000 at n=48.

Primary held-out grid: n={24,48}, depolarizing p={.2,.5}, seeds 3-7, four boundary target regions, two modes, 160 observations total.

| Qubits | Mean local D, sketch | Mean local D, peak | Largest D, sketch | Largest D, peak |
|---|---:|---:|---:|---:|
| 24 | **0.01878** | 0.02293 | 0.04524 | 0.06384 |
| 48 | **0.02010** | 0.02747 | 0.05591 | 0.06198 |

Sketch produced the lower local D in **70 of 80** paired comparisons. At 48 qubits, 39 of 40 sketch regions and 36 of 40 peak regions met a predeclared 0.05 event-error tolerance. Mean evolution times were about 1.10 seconds (sketch) and 1.18 seconds (peak). The 6-qubit one-edge cone reference cost about 0.02 seconds and the 8-qubit two-site cone about 0.30 seconds per result on the same CPU; those timings are excluded from evolution time.

## Other noise channels, held-out 48-qubit circuits

Dephasing and amplitude damping were tested at strengths .2/.5 and seeds 3-7, yielding 160 more per-region observations. Both modes produced numerically identical reduced states on all paired channel-transfer observations: the peak cap was inactive.

| Channel | p | Mean local D | Maximum local D |
|---|---:|---:|---:|
| Dephasing | .2 | 0.02480 | 0.06653 |
| Dephasing | .5 | 0.04559 | **0.14405** |
| Amplitude damping | .2 | 0.01701 | 0.02822 |
| Amplitude damping | .5 | 0.01703 | 0.04297 |

The large right-boundary error for some dephasing circuits is a counterexample to assuming that low-depth strongly noisy simulations always have small local errors.

## Post-run acceptance

Require local D <= 0.05 on **all four** nominated boundary regions. Prefer the peak mode when it passes, fall back to sketch if it passes, and reject if both fail. The exact finite-cone oracle is evaluated separately, so this is a post-run validation gate, not a predictive low-cost estimator.

| Family | Cases | Peak accepted | Sketch fallback | Rejected |
|---|---:|---:|---:|---:|
| Depolarizing, 24/48 qubits | 20 | 15 | 4 | 1 |
| Dephasing, 48 qubits | 10 | 7 | 0 | 3 |
| Amplitude damping, 48 qubits | 10 | 10 | 0 | 0 |
| **Total** | **40** | **32** | **4** | **4** |

Rejections were depolarizing (n48,p.2,seed3), dephasing (n48,p.2,seed7), and dephasing (n48,p.5,seed3 and seed7). Acceptance concerns only those four measured regions.

## Remote-edge independence

The leftmost and rightmost six-layer backward cones are disjoint at n=48. For product initial input, their exact joint reduced output density factorizes into two independent one-site densities. In 20 depolarizing remote-edge tests, the compressed simulator's joint marginal differed from the product of its own edge marginals by less than 1.0e-15 trace distance, with connected ZZ below 1.4e-16. Mean distance to the **exact** remote-edge joint density was 0.01920 for sketch and 0.02696 for peak, driven by local errors.

## Reproduction

From experiments/channel_mpo, using NumPy >=1.24, <3 and one BLAS thread:

~~~bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 test_qc04 test_qc05 test_qc06 test_qc07 test_qc08 test_qc09 test_qc10
python benchmark_qc10.py --qubits 24,48 --strengths .2,.5 --seeds 3,4,5,6,7 --channels depolarizing --output depolarizing.csv
python benchmark_qc10.py --qubits 48 --strengths .2,.5 --seeds 3,4,5,6,7 --channels dephasing,amplitude_damping --output channels.csv
python qc10_acceptance.py --source depolarizing.csv --tolerance .05 --output accepted_depolarizing.csv
python qc10_acceptance.py --source channels.csv --tolerance .05 --output accepted_channels.csv
python benchmark_remote_qc10.py --qubits 48 --strengths .2,.5 --seeds 3,4,5,6,7 --output remote.csv
~~~

**110/110 local regression tests passed** in the source worktree. The frozen raw data comprise 320 per-region local measurements, 40 acceptance decisions, and 20 remote-edge observations; the reproducible research packet contains all those files. Repository aggregate checks are in qc10_summary.json.

## Next

Extend the independent reference toward interior observables with a Heisenberg-picture tensor network, then test a cone-weighted bond/Kraus rank scheduler at equal measured RSS and matched local operational-error tolerances. Formal floating-point/interval verification remains a separate possible extension.
