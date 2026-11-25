# QC-05: Global density proxy versus wider Kraus environment

Status: completed CPU benchmark, November 9, 2026. Extends QC-01 through
QC-04 on the same structured-noise circuits.

## Question

QC-04's local retained-purification-weight objective occasionally worsened
final trace distance relative to QC-02, especially at 8 qubits, p=0.2.
Can a global physical-density estimate select the better fixed-cap policy
without allocating exponential memory? Does selection outperform simply
allocating more of the memory budget to Kraus environments?

## Method

Three purified chains evolve independently through the *same* Haar brickwork
gates and depolarizing noise: QC-02 sequential at bond 12 / env 4, QC-04
screened-gauge pair compression at bond 12 / env 4, and an approximate teacher
at bond 16 / env 8. At the final layer QC-05 selects between the two fixed-cap
candidates by their squared Frobenius distance to the teacher:

\[
\|\rho_c-\rho_T\|_F^2 =
\operatorname{Tr}(\rho_c^2)-2\operatorname{Tr}(\rho_c\rho_T)
+\operatorname{Tr}(\rho_T^2).
\]

The last term cancels when ranking candidates. QC-05 contracts double-layer
MPO transfer matrices to compute overlaps of the three *approximate*
locally purified tensor networks, requiring no dense 2^n by 2^n matrices.
The exact density oracle is evaluated *after* selection on 6- and 8-qubit
circuits solely to measure error. A unit test patches dense_density() to
raise during selection.

The teacher is approximate. Hilbert–Schmidt ordering does not certify
trace-distance ordering, and the teacher itself may have substantial
truncation error. Each decision scores exactly two candidates, with fixed
rank caps. The teacher simulation and score are counted in selector runtime.

A separate QC-02 bond 12 / env 6 control tests the marginal value of
increasing Kraus capacity without raising the intersite bond cap.

## Frozen protocol

- Qubits: 6 and 8; depth: 6; depolarizing p=0.2 or 0.5.
- Seeds 0–2 calibration (12 runs), seeds 3–7 held out (20 runs).
- Fixed candidate caps: bond 12, env 4.
- Reference caps: bond 16, env 8.
- Additional resource baseline: bond 12, env 6.
- Python 3.13.5, NumPy 2.3.5; BLAS threads pinned to 1.
- Single process, one timing observation per configuration; independent exact
  validation is excluded from wall timings. Neither IBM quantum hardware
  nor an actual quantum speedup is involved.

The accompanying CSV records all 32 configurations, selected policies,
global scores, trace-distance errors, timing and final tensor entries.

## Results

| Split | Count | Correct candidate choice | QC-04 mean D | QC-05 mean D | env6 sequential mean D | env8/bond16 mean D |
|---|---:|---:|---:|---:|---:|---:|
| Calibration | 12 | 11/12 | 0.204095 | 0.203536 | 0.137525 | 0.090198 |
| Held out | 20 | 17/20 | **0.206768** | **0.207256** | **0.142776** | **0.097231** |

The selector picked the worse candidate in three held-out cases:

- 6q p=.5 seed 5: selected sequential D=.176386 vs pair D=.158919.
- 8q p=.2 seed 3: selected sequential D=.344782 vs pair D=.341511.
- 8q p=.5 seed 7: selected sequential D=.134348 vs pair D=.118028.

Although the global proxy often ranked candidates correctly, mean
held-out error is slightly *worse* than QC-04 (by 0.000488).

### Held-out cost and quality

| Policy | Mean trace distance | Mean CPU seconds | Final stored complex entries |
|---|---:|---:|---:|
| QC-02 bond12/env4 | 0.225904 | 0.0366 | 5,120 |
| QC-04 bond12/env4 | 0.206768 | 0.3231 | 5,120 |
| QC-05 three TNs + global HS | 0.207256 | 0.7130 | sum of three TNs |
| Sequential bond12/env6 | **0.142776** | **0.0577** | not recorded in CSV |
| Sequential bond16/env8 | **0.097231** | **0.1671** | 20,992 |

The env6 sequential policy beat QC-04 in **18/20** held-out cases,
ran about **5.6x faster**, and lowered mean trace distance by **31%**.
The env8/bond16 teacher beat QC-04 in **20/20** cases, ran about
**1.93x faster**, and lowered mean trace distance by **53%**.
Its final tensor-entry count is approximately 4.1x the fixed-cap runs.

The bond16/env8 result changes two independent capacity limits, so its
quality gain cannot be assigned solely to environment size. The bond12/env6
comparison isolates the effect of Kraus cap at fixed intersite bond.
These are small-system timing samples, without RSS measurements or timing
confidence intervals. Even the larger teacher exhibits low per-layer
retention in some runs, so no large-system fidelity claims follow.

## Reproduce

From experiments/channel_mpo, with Python 3.11 or newer:

    python -m pip install 'numpy>=1.24,<3'
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 test_qc04 test_qc05
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_qc05.py --qubits 6,8 --strengths 0.2,0.5 --seeds 0,1,2,3,4,5,6,7 --depth 6 --bond 12 --env 4 --teacher-bond 16 --teacher-env 8 --metric hs --output qc05_results.csv

Local QC-01–QC-05 acceptance suite: **45/45 tests passed**. CI runs the same
quick CPU unit suite and skips dense benchmark jobs.

## Verdict and next experiment

The global HS proxy is a valuable diagnostic for gauge choices, yet its
extra compute produces no held-out accuracy gain on this grid. The wider
sequential teacher is itself both more accurate and faster than gauge search.

QC-06 should compare **adaptive local environment rank** to uniform
env6 and env8 baselines at *matched peak memory*, tracking truncation
spectra, errors, measured RSS, runtime, SVD count, and held-out seeds.
Use screened gauge permutations only when they exceed larger-cap
sequential baselines at an equal storage budget.
