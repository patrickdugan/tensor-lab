# QC-20: Packed dyadic source tensors for the independent MPO verifier

**Date:** 2026-11-10. **Status:** experimental results frozen locally; this branch currently contains the research note. The tested checker, unit tests, benchmark runner, and lightweight CI workflow remain pending transfer.

## Objective

QC-19 holds binary64 witness coefficients as nested Python `Fraction` pairs and also caches their common-denominator integer representation. The QC-20 prototype retains checked, immutable IEEE-754 binary64 tensor bytes and derives exact integer operands during each residual check. The producer, witness file, hash commitments, physical-circuit interpretation, and exact dyadic error inequality stay unchanged.

The prototype also implements a streaming two-site exact integer contraction and validates checkpoint boundary SHA-256 commitments directly from the packed binary64 coefficients. Signed zero is canonicalized to the QC-15 reference semantics. A slower optional fixed-width signed integer cache was tested separately.

## Four-worker 48-qubit result

The identical QC-18 dephasing witness was used in every run: depth 16, p=0.5, seed 3, 13 segments, 14 checkpoints, 3,097 transformations and 1,206,052 changed complex entries. Memory measurements include parent and workers, sampled from Linux `/proc/<pid>/smaps_rollup` at 50 ms intervals.

| Repeat | Checker | Wall time | Aggregate RSS | Aggregate PSS | Private USS |
|---|---|---:|---:|---:|---:|
| 0 | QC-19 | 10.765 s | 486.0 MiB | 298.7 MiB | 243.5 MiB |
| 0 | QC-20 fast | 10.100 s | 476.9 MiB | 131.6 MiB | 98.1 MiB |
| 1 | QC-19 | 10.971 s | 486.2 MiB | 299.0 MiB | 243.8 MiB |
| 1 | QC-20 fast | 10.508 s | 476.4 MiB | 131.3 MiB | 98.0 MiB |
| pilot | QC-20 packed numerators | 27.863 s | 477.0 MiB | 126.4 MiB | 98.1 MiB |

Across the two paired measurements, **private memory decreased by approximately 59.8%**; PSS fell by about 56%. RSS includes shared mapped pages counted once per process and therefore falls much less. The fixed-width integer-cache ablation increases runtime markedly with negligible extra USS savings, so the fast cache remains the preferred default. Timings represent two observations, not confidence intervals.

Every variant returned exactly:

```text
verified_bound                 0.011832047006821769
arithmetic_lower_requirement   0.011832047006820182
events_checked                 3097
segments_checked               13
checkpoints_checked            14
witness_complex_entries        1206052
```

## Verification and trust boundary

**252/252 local regression tests passed** in CPU batches of 110, 75, and 67, including 20 new QC-20-specific tests. These cover binary64 extremes and subnormals, signed-zero equality, exact arithmetic and boundary-digest parity, checkpoint-decoder equivalence, serial/fork/spawn replay, corrupted witness and missing-checkpoint rejection, and the original QC-01 through QC-19 suites. The original QC-19 checker remains an independent differential reference.

The certificate covers the polar-corrected binary64 circuit defined in QC-14. The standalone checker's implementation and the mathematical residual inequalities remain the trusted code base; proof-source verification in Lean or an equivalent system remains future work. The read-only event tape is hashed before verification. Stronger sealing is needed against concurrent hostile writers. These observations are on Linux fork workers; cross-platform memory results require new benchmarks.

## Source-transfer status

The tested local source set has seven new files:

- `qc20_packed.py`
- `qc20_checker.py`
- `test_qc20.py`
- `benchmark_qc20.py`
- `QC20_REPORT.md` (long-form, including exact reproduction commands)
- `qc20_memory_benchmark.csv`
- `.github/workflows/qc20-channel-tests.yml`

The artifact-file bridge refused durable uploads during this run, so this note is a **draft transfer marker**, and the branch should not be considered implementation-complete until the seven verified files arrive and CI passes.
