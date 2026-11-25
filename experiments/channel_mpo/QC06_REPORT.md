# QC-06: adaptive Kraus-environment rank allocation at matched storage

**Status:** exploratory CPU study, 2026-11-09. The primary six/eight-qubit
depolarizing experiment improves mean trace distance at a strictly matched
post-layer tensor-entry budget. The advantage is channel-dependent.

## Method

A locally purified tensor network represents a state as

\[
A_i\in\mathbb C^{D_{i-1}\times 2\times e_i\times D_i},\qquad
\rho=\operatorname{Tr}_{\mathrm{anc}}|A\rangle\langle A|.
\]

In the mixed canonical gauge, the singular values of the local environment
matricization are \(s_{i,1}\ge s_{i,2}\ge\cdots\). QC-06 greedily adds the next
singular component with the largest squared weight per retained complex entry:

\[
\frac{s_{i,r_i+1}^2}{2D_{i-1}D_i},\quad
\sum_i (2D_{i-1}D_i)r_i\le B_t,\quad
1\le r_i\le \min(r_i^{\rm numeric},12).
\]

This integer allocation is heuristic. Its local error surrogate can disagree
with physical-state trace distance; spectra at later sites shift when earlier
sites are compressed. The algorithm has no access to a dense reference matrix,
future gates, or a teacher network. QR gauge changes followed by local ancilla
SVDs preserve positivity, and the resulting trace is normalized.

The bound \(B_t\) controls **retained entries at layer boundaries**. It does
not control temporary working arrays or peak process RSS.

## Primary experiment and frozen budget schedules

The setup matches QC-05: six-layer brickwork Haar gates, depolarizing noise
p in {0.2,0.5}, qubits n in {6,8}, bond cap 12. The baseline is sequential
purification with a uniform environment cap of 6. The adaptive maximum
environment rank is 12. Calibration seeds are 0–2; held-out seeds are 3–7.
The environment-entry budget for every layer comes from cap-6 calibration
rollouts and is saved in qc06_budget_schedule.json.

Those six-layer schedules were identical across tested seeds and noise
strengths. The benchmark verifies exact agreement with baseline entries
on each held-out layer. Dense CPTP evolution and physical trace distance
are computed **after** both tensor-network rollouts finish, for n<=8.

| Split | Circuits | Uniform mean trace D | Adaptive mean trace D | Adaptive wins |
|---|---:|---:|---:|---:|
| Calibration seeds 0–2 | 12 | 0.137525 | 0.131938 | 7/12 |
| Held-out seeds 3–7 | **20** | **0.142776** | **0.132490** | **17/20** |

Held-out absolute improvement was 0.010286 and relative improvement 7.20%.
The three regressions occurred with stronger depolarizing noise p=0.5.

| Qubits | p | Uniform trace D | Adaptive trace D | Wins |
|---:|---:|---:|---:|---:|
| 6 | 0.2 | 0.169994 | 0.152407 | 5/5 |
| 6 | 0.5 | 0.080474 | 0.072501 | 4/5 |
| 8 | 0.2 | 0.210291 | 0.197069 | 5/5 |
| 8 | 0.5 | 0.110347 | 0.107983 | 3/5 |

The averaged in-process evolution times on twenty held-out cases were
0.0646 s (uniform) and 0.0651 s (adaptive). Timings are exploratory
single runs. The cap-6 adaptive implementation reproduces the baseline
density matrix to numerical precision; it isolates the faster QR sweep
from rank reallocation.

### Rank-cap ablation, held-out means

| Adaptive per-site cap | Trace D | Wins / 20 |
|---:|---:|---:|
| 6 | 0.142776 | 0 |
| 8 | 0.138580 | 16 |
| **12, primary** | **0.132490** | **17** |
| 16 | 0.135014 | 16 |

Caps 8 and 16 were tested after the primary run and constitute
exploratory sensitivity checks. Some leave entry budget unused.

### Exploratory cross-channel transfer

For six/eight qubits, p=0.2/0.5 and seeds 3–5, using each baseline's
same-layer entry budget:

| Noise | Uniform mean D | Adaptive mean D | Adaptive wins / 12 |
|---|---:|---:|---:|
| Dephasing | 0.140605 | 0.133635 | 9 |
| Amplitude damping | 0.129930 | 0.133313 | 4 |

The amplitude-damping degradation shows why channel-specific validation
matters. This pilot does not establish a uniform benefit across CPTP maps.

## Large-qubit diagnostics

Twelve-, sixteen-, twenty-four-, and forty-eight-qubit circuits were
executed without a dense reference, so physical trace distance at those
sizes remains **unknown**.

Separate Linux processes, six-layer circuits, seed 3:

| n | p | Uniform peak RSS MiB | Adaptive peak RSS MiB | Uniform sec | Adaptive sec |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.2 | 98.4 | 99.3 | 0.079 | 0.080 |
| 24 | 0.2 | 98.6 | 100.7 | 0.529 | 0.354 |
| 48 | 0.2 | 101.3 | 102.7 | 1.650 | 0.835 |
| 48 | 0.5 | 101.1 | 102.2 | 1.538 | 0.792 |

The speed change at large n is substantially attributable to the
single-sweep QR implementation, rather than the choice of ancilla ranks.
Those memory figures include Python/NumPy startup and LAPACK allocations.

A 48-qubit p=0.5 circuit retained only about 10^-6 of the purification
norm at its worst layer. Its output has no established physical accuracy.
The larger bond16/env8 reference from QC-05 still yields lower small-system
error when it is allowed more retained tensor entries.

## Reproduce

Python 3.10+ and numpy>=1.24,<3. From experiments/channel_mpo run:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 test_qc04 test_qc05 test_qc06

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_qc06.py --qubits 6,8 --strengths 0.2,0.5 --seeds 3,4,5,6,7 --frozen-budgets qc06_budget_schedule.json --output qc06_heldout.csv

Local full regression: **56/56 tests passing**, including eleven QC-06
tests. The GitHub workflow runs these tests and a 6q smoke benchmark.

The raw experimental CSVs, peak-RSS runner and sensitivity scripts are
also available in the QC-06 research packet delivered with this study.

## Next direction

An allocator that jointly optimizes the local bond and environment ranks
could operate under a true peak-RSS constraint. A channel-aware
physical-observable error proxy also merits testing on held-out circuits,
particularly to address the amplitude-damping regression.
