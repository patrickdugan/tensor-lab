# QC-11: Streaming causal-cone simulation for exact local observables

**Date:** 2026-11-09. **Status:** reproducible CPU pilot. Stacked on QC-10 PR #12 in `patrickdugan/tensor-lab`.

## Research question and outcome

QC-10 made shallow 48-qubit circuit *edge observables* testable by computing their full backwards causal cones. Its eight-qubit density oracle held all cone sites for all six layers. QC-11 asks whether a target-aware resource allocator can produce more accurate observables under the fixed QC-09 memory constraints, and whether strictly causal contraction gives a stronger answer.

**Finding:** the original weighted bond/Kraus-rank heuristic was unreliable on the known dephasing failure cases (see negative ablation below). We implemented `qc11_streaming_cone.py`, which exactly traces away sites as soon as they leave the target's backwards cone. For edge measurements in the tested six-layer nearest-neighbor brickwork circuits, the active density matrix shrinks across layers **8 -> 7 -> 6 -> 5 -> 4 -> 3 -> 2 qubits**. Two maximal regions (the left and right edge pairs) supply all four QC-10 edge observables. Unlike the QC-09 approximation, the local result is mathematically exact under the model assumptions.

Across **50 paired held-out configurations** (24/48 qubits; depolarizing, amplitude-damping and dephasing; `p=0.2,0.5`; seeds 3–7), max trace-distance disagreement against the previous fixed-cone oracle was **1.60e-15**. Median single-process paired CPU speedup was **12.90x**, with observed range **8.03–18.88x**. The gain is relative to QC-10's exact fixed-width cone method; QC-09 is an approximate global purified-chain simulator and answers a broader state-representation problem.

## Mathematical algorithm

Let `S_t` denote the backwards causal support at the input to forward layer `t`, for `t=0,...,T`. The final support `S_T` is the requested output subsystem. The circuit begins with the product state `|0>^n`. A forward layer consists of a set of disjoint nearest-neighbor two-qubit unitaries, followed by independent trace-preserving one-site channels.

1. Build the entire backwards support schedule using topology alone, before any state evolution. The support satisfies `S_{t+1} subset S_t` in forward order.
2. Initialize the density matrix on `S_0` as `|0...0><0...0|`.
3. For each forward layer, apply only gates that intersect `S_{t+1}`. Backwards-cone construction ensures both gate inputs lie in `S_t`.
4. Trace away `S_t \ S_{t+1}` **before** applying noise, then apply the complete Kraus sum on the remaining sites only. Record the active width and resources.
5. Return the density on `S_T`. To compute nested target regions, solve maximal regions once and obtain subset densities by partial trace.

**Exactness argument (ideal arithmetic):** suppose the active density before layer `t` equals the reduction of the full circuit state onto `S_t`. The omitted disjoint gates act entirely on the complement of `S_{t+1}`. They disappear under the partial trace. All gates touching `S_{t+1}` have both endpoints in `S_t`. One-site channels on discarded qubits also disappear under the partial trace because they are trace preserving; on surviving qubits the partial trace and the local channel commute. Consequently the resulting active density is precisely the full state reduced onto `S_{t+1}`. Induction proves equality at the output. This argument requires **an initial product state**, disjoint local unitary layers, and local CPTP noise. The implementation enforces gate unitarity to tolerance and Kraus completeness to tolerance.

For the initial cone width `w_0` and active widths `w_t`, resident dense storage is `O(4^{w_0})` complex entries, and a coarse operation bound follows from summing local gate and Kraus operations on matrices with at most `4^{w_t}` elements. This is independent of *global* qubit count for fixed shallow boundary-local targets. State initialization and user-provided gate generation may still cost `O(n*T)`. NumPy/BLAS transient workspace and OS RSS are measured separately.

**Precision:** the proof is exact in ideal arithmetic; experimental equality is numerical floating-point evidence, **not** formal interval arithmetic. There is no bound on errors from floating-point rounding in the present implementation. There is no claim of accurately reconstructing the entire global 48-qubit state.

## 50-case held-out benchmark

Six-layer circuits, seed set `{3,4,5,6,7}`, strengths `{0.2,0.5}`, 4 output regions per case. The fixed method is QC-10 `exact_cone_density` called four times; the streaming method shares two maximal pair calculations. CPU single-thread OpenBLAS, wall-clock timings excluding interpreter startup. Values are means across 10 configurations per row; measured speedup is paired.

| Qubits | Noise | QC-11 bundle (s) | QC-10 fixed-cone (s) | Median paired speedup |
|---|---|---:|---:|---:|
| 24 | Depolarizing | 0.04524 | 0.67856 | 14.78x |
| 24 | Amplitude damping | 0.03054 | 0.36334 | 12.27x |
| 48 | Depolarizing | 0.04971 | 0.75433 | 16.18x |
| 48 | Dephasing | 0.03220 | 0.34307 | 10.71x |
| 48 | Amplitude damping | 0.03129 | 0.37436 | 11.79x |

Overall max local trace-distance disagreement = **1.5959456e-15**. Overall median paired speedup = **12.8965x**. The QC-11 algorithm needs the full 8-qubit dense matrix only at the earliest step for an edge pair; the active widths shrink with each layer. Both exact methods compute all four marginals with numerical trace-distance agreement to the QC-10 oracle.

### Known QC-10 operational failures

QC-09 global compression previously reached trace distance **0.132305** on the right edge pair (48 qubits, dephasing `p=.5`, seed 3) and **0.144051** on the left edge pair (seed 7). QC-11 returns those local density matrices directly and matches the QC-10 exact reference to about `1e-15`. At a 0.05 local-event-probability threshold, the exact-in-model results therefore pass the independent local benchmark. They are not global-state fidelity certificates.

### Isolated-process peak RSS

All modes measure four edge regions, 48 qubits, six layers, dephasing; two strengths and seeds 3/7. Each row is the **mean of four independent fresh processes**, RSS via Linux `resource.getrusage`. Runtime includes only the numerical simulation, excluding Python startup. OS/Python/NumPy overhead dominates these MiB figures.

| Mode | Mean runtime (s) | Mean peak RSS (MiB) |
|---|---:|---:|
| QC-09 approximate global peak-budgeted chain | 0.8525 | 103.36 |
| QC-10 fixed-cone exact density (four solves) | 0.3525 | 102.23 |
| QC-11 streamed exact densities (two solves) | **0.0280** | **100.21** |

These are local-observable-specific comparisons; QC-09 additionally maintains a reusable approximation to the full state. The memory improvement is small in absolute terms, since all three share a large interpreter/runtime baseline.

## Negative ablation: target-weighted global rank allocation

`qc11_focus.py` rescales the squared singular-value benefits of QC-09's joint bond/Kraus rank allocator by the prescribed backwards-cone site masks. It uses the same QC-06 layer-boundary entry schedules, QC-09 caps, `gate_method='sketch'`, and deterministic gates. Its rank decisions never query QC-10/QC-11 exact densities. The focus masks are selected before rollout.

On known difficult 48-qubit dephasing `p=.5` seeds (selected **because** QC-10 exposed large errors), the weighted heuristic *regressed* in the target region:

| Case, target | QC-09 baseline local trace D | Weighted target local trace D |
|---|---:|---:|
| Seed 3, right edge pair | 0.132305 | 0.142349 |
| Seed 7, left edge pair | 0.144051 | 0.190340 |

The two-seed stress-case pilot is not a generalization study or a basis for selecting a best per-case mode. A higher retained rank in a nominated area does not ensure lower local trace distance: the remaining compression and normalization stages interact. This negative result motivates using exact causal contraction whenever the requested region's cone is tractable. The target-weighted allocator remains an experimental research component for deeper cones.

## Verification and limits

All **121 local unit tests** across QC-01 through QC-11 passed, including 11 new tests for full dense-state agreement on small circuits, cross-channel cone equivalence, physical density constraints, multi-region reuse, trace-preserving noise, strict unitary validation, memory admission checks, focus weights independent of oracle, and rank-policy parity when `off_cone=1`.

The method is applicable only where the backwards cone is tractable; an interior two-site observable after many layers can have a cone much larger than eight qubits. `max_sites` and `max_density_entries` fail early instead of silently returning an approximation. Long-range gates and correlated noise would require an extended dependency graph and explicit channel handling. More general initial states require correct starting reduced densities.

Future research: a dual-Heisenberg contraction for single selected observables, tensor-network contraction inside cones wider than 8, and **observable-specific truncation error controls** for those deeper circuits. Any claimed operational error should be tied to the measured region and independent validation or a rigorous numerical certificate.

## Reproduce

```bash
cd experiments/channel_mpo
python -m pip install 'numpy>=1.24,<3'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 test_qc04 \
  test_qc05 test_qc06 test_qc07 test_qc08 test_qc09 test_qc10 test_qc11
python benchmark_streaming_qc11.py --qubits 48 --channels dephasing \
  --strengths .2,.5 --seeds 3,4,5,6,7 --output /tmp/qc11_dephasing.csv
python benchmark_streaming_qc11.py --qubits 24,48 \
  --channels depolarizing,amplitude_damping --strengths .2,.5 \
  --seeds 3,4,5,6,7 --output /tmp/qc11_other.csv
python benchmark_peak_qc11.py --qubits 48 --channels dephasing \
  --strengths .2,.5 --seeds 3,7 --output /tmp/qc11_peak.csv
python benchmark_qc11.py --qubits 48 --strengths .5 --seeds 3,7 \
  --modes baseline,left,right,edges --output /tmp/qc11_focus.csv
```

Frozen records: `qc11_streaming_heldout.csv` (50 comparisons), `qc11_peak_rss.csv` (12 isolated processes), and `qc11_focus_calibration.csv` (56 rows from two preselected failure cases). Each CSV is raw numerical evidence rather than proof of asymptotic behavior.
