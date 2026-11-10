# QC-20: Compact exact-dyadic source tensors for low-private-memory proof replay

**Date:** November 10, 2026. **Status:** 48-qubit CPU benchmark, two paired fresh-process runs, complete QC-01 through QC-20 regressions. **Base:** QC-19 PR #21 (`research/qc19-live-kernel-memory-20261110`).

## Research question and verdict

QC-19's standalone verifier retains a full set of `Fraction` objects for each decoded MPO core, then constructs a second exact common-denominator integer representation for the same tensor. It also decodes checkpoint tensors into these objects. Can we keep the exact binary64 witness coefficients in compact immutable storage, while maintaining independent residual checking and QC-19's proof semantics?

**Yes for this witness.** In two matched four-worker replays of the *same* pinned 48-qubit, depth-16 dephasing certificate, QC-20 lowered aggregate private memory (USS) from approximately **243.6 MiB to 98.1 MiB** (a **59.7% reduction**) with equivalent run times. The entire exact proof result, all 3,097 locally audited operations and all 13 independently linked segments match QC-19. The only changed component is the *checker-side decoding and integer workspace*; witness transport, boundary commitments, and the original QC-19 checker remain unchanged.

The only source of memory reduction claimed is the absence of persistent per-coefficient `Fraction` pairs in *working MPO cores and worker checkpoint replay*. Source cores are stored as canonical immutable bytes representing the same finite IEEE-754 binary64 values supplied by the witness. Exact dyadic conversion occurs inside each local integer arithmetic check; the default fast path retains the integers in QC-19's liveness-bounded cache until they are no longer used. An explicit compact-numerator mode also exists but sacrifices speed for very little further private-memory reduction.

## Design

* `qc20_packed.PackedTensor(shape, raw)` holds a shape and exactly 16 raw bytes per complex coefficient. It validates dimension caps and excludes nonfinite coefficients. Signed zero is canonicalized to +0.0 because the QC-15 `Fraction` path does likewise when re-encoding a commitment. A lazy `.vals` adapter exists for independent comparisons but the replay does not materialize it.
* `read_packed_tensor()` validates the same external `{shape,binary64_le}` encoding. `read_checkpoint_packed()` independently validates file SHA-256, canonical compressed JSON checksum, support/bond dimensions, cumulative exact dyadic accumulator, event tally, commitment and manifest correspondence. Unlike QC-17's checkpoint decoder, it retains MPO cores in packed bytes after validation.
* `boundary_digest_packed()` recreates **the identical QC-16 boundary hash** directly from checked binary64 coefficient bytes, exact cumulative dyadic fractions and length-delimited index/shape/tally fields. It is compared with the existing implementation in tests and with all fourteen original checkpoints from the frozen 48-qubit proof.
* `StreamFastKernel` uses QC-16's common-denominator integer arithmetic and QC-19's cache retirement, but contracts one outer-bond block at a time rather than materializing all two-site blocks simultaneously. Its numerical results are exact integer/rational equalities. It is the **default**.
* `StreamPackedKernel` additionally stores the cached common-denominator signed numerators in fixed-width little-endian integer buffers; it reconstructs Python integers on demand. Width is chosen from actual values (up to 272 bytes, covering every finite binary64 extreme). This is a performance/memory ablation, selected with `--storage-mode compact`.
* The original standard-library-only circuit schedule verification, local residual and context inequalities, polar correction allowance, independent genesis/final checks, segment linking and tape hash verification remain mandatory. Changes to a signed-zero encoding keep the same exact rational meaning and canonical commitment.

QC-20 *does not* change the claimed physical circuit or the mathematics from QC-14 through QC-19. A full local operand witness remains essential: a scalar ledger alone does not independently establish the residuals. The independent checker's Python implementation, mathematical inequalities and filesystem stability are explicit trust assumptions; the code is not a formal proof in Lean or another proof assistant. The read-only mmap file is hashed before use; preventing malicious concurrent rewrites requires immutable/sealed file descriptors or external filesystem isolation.

## Matched 48-qubit benchmarks

Linux, four **fork** workers, a single 48-qubit depth-16 dephasing `p=0.5` seed-3 proof. The exact same 15 MB read-only mmap tape, 13 segments and 14 checkpoints are used for every run. Fresh verifier processes are paired sequentially. `/proc/<pid>/smaps_rollup` was sampled every 50 ms and includes coordinator and all workers. USS measures private pages, PSS divides shared pages proportionally among processes, while summed RSS counts shared memory repeatedly. These are *observed samples*, not hard upper bounds.

| Repeat | Checker | Wall (s) | Aggregate RSS (MiB) | PSS (MiB) | USS (MiB) |
|---:|---|---:|---:|---:|---:|
| 0 | QC-19 | 10.765 | 486.0 | 298.7 | 243.5 |
| 0 | QC-20 default | 10.100 | 476.9 | 131.6 | **98.1** |
| 1 | QC-19 | 10.971 | 486.2 | 299.0 | 243.8 |
| 1 | QC-20 default | 10.508 | 476.4 | 131.3 | **98.0** |
| pilot | QC-20 compact-integer ablation | 27.863 | 477.0 | 126.4 | 98.1 |

Means across the two paired observations: QC-19 **10.868 s**, **243.65 MiB USS**; QC-20 default **10.304 s**, **98.05 MiB USS**. That is a **59.8% decrease** in USS and a **56.0% decrease** in PSS. Summed RSS fell only about 1.9%, illustrating why it is the wrong headline metric for forked processes sharing a large read-only tape. Timings have too few repetitions for a robust CPU-speed claim.

The optional fixed-width cached-numerator format modestly reduced PSS versus the fast mode but increased wall time roughly 2.8x. We therefore retain the `fast` integer cache as the recommended default and expose `compact` for experimentation only.

All verifier variants agree exactly on these invariant fields:

```
events_checked                 3097
segments_checked               13
checkpoints_checked            14
witness_complex_entries        1206052
verified_bound                 0.011832047006821769
arithmetic_lower_requirement   0.011832047006820182
channel updates                241
gate SVD updates               136
left/right QR updates          1360 / 1360
```

## Verification

The complete **252-test** suite (QC-01 through QC-20) passed in three CPU batches: 110 + 75 + 67 tests. The twenty QC-20-specific tests cover exact-IEEE exponent extremes and subnormals, signed-zero normalization, bit-for-bit dyadic arithmetic parity, input size and finiteness rejection, exact boundary hashes, checkpoint decoder parity, context and readout parity, cache retirement, serial/fork/spawn full-witness replays, tampered event rejection, missing checkpoint rejection, and standard-library-only source independence. Full 48-qubit memory measurements were run in separate fresh subprocesses.

### Commands

```bash
cd experiments/channel_mpo
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pip install 'numpy>=1.24,<3'
python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 \
  test_qc04 test_qc05 test_qc06 test_qc07 test_qc08 test_qc09 test_qc10
python -m unittest -q test_qc11 test_qc12 test_qc13 test_qc14 test_qc15 test_qc16
python -m unittest -q test_qc17 test_qc18 test_qc19 test_qc20
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
with ZipFile('qc18_certificates/10q_d6_mmap.zip') as z:
    z.extractall('/tmp/qc20-small')
PY
python -I qc20_checker.py /tmp/qc20-small/manifest.json --workers 2 --start-method spawn
python -I qc20_checker.py /tmp/qc20-small/manifest.json --workers 2 --start-method fork
python -I qc20_checker.py /tmp/qc20-small/manifest.json --storage-mode compact
```

With the separate 48-qubit QC-18 tape archive unpacked at `/path/to/proof`, run:

```bash
python benchmark_qc20.py --manifest /path/to/proof/manifest.json \
  --workers 4 --repetitions 2 --out qc20_memory_benchmark_repro.csv
python -I qc20_checker.py /path/to/proof/manifest.json \
  --workers 4 --start-method fork --storage-mode compact
```

### Further work

The biggest remaining proof-verifier memory cost comes from exact integer contraction temporaries and the shared circuit plan. QC-21 should test a bounded-memory exact accumulator with chunked integer polynomials and verified nonresident checkpoints, while benchmarking independent workers on hardware with different fork/COW behavior. Do not assume the measured USS saving transfers to Windows spawn or hostile filesystem settings without fresh evidence.
