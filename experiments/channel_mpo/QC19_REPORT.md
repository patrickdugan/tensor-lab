# QC-19: Bounded-live integer workspace for independent MPO witness verification

**Date:** November 10, 2026. **Branch:** stacked on QC-18 PR #20.  
**Status:** local CPU study of the pre-existing 48-qubit, depth-16, dephasing `p=0.5`, seed-3 witness. The *proof statement, protocol, encoded witness and QC-18 format are unchanged*.

## What was changed

QC-18's independent checker caches up to 96 common-denominator integer conversions of source MPO tensors inside each worker's `qc16_dyadic.Kernel`. A core that has already been replaced can remain in that conversion cache through many subsequent updates. The cache holds both the original exact `Fraction` pairs and the converted large integers. QC-19 explicitly retires those cache entries immediately after the corresponding event's local exact residual, surrounding-context bound and polar-correction checks have all passed.

QC-19 also releases decoded input/output checkpoint tensors as soon as the independent boundary checks complete. Each worker holds the current MPO chain and small checked input/output commitments. QC-18's read-only memory-mapped event tape, independent circuit schedule, exact integer arithmetic, boundary commitments and all witness validation remain in force. The original QC-18 checker remains intact for differential testing.

`qc19_kernel.LiveKernel` inherits QC-16's arithmetic and adds an identity-checked `retire(tensor)` operation. `qc19_checker.py` is the independent standalone CLI, with a per-event call to `retire` after all old-version uses have ended. This reduces duplicate *private* integer storage; it does **not** replace unbounded integers with finite-precision approximations. Exact proof results are therefore unchanged.

The cache still holds up to 96 live conversions. A smaller 24-entry cache was tested on the 48-qubit witness. Its memory result was similar, while it ran more slowly, so the 96-entry limit remains the default.

## Matched four-worker proof benchmark

All trials use the *same existing QC-18 frozen memory-mapped witness*, without rewriting or regenerating any event data, on Linux with 4 forked workers. Measurements are fresh verifier processes, paired in sequence, sampled every 50 milliseconds using `/proc/<pid>/smaps_rollup`. Measurements include the coordinator and its worker process tree. RSS double-counts shared mappings; PSS apportions them; USS is the sum of uniquely private pages.

| Repeat | Checker | Wall seconds | Aggregate RSS (MiB) | PSS (MiB) | USS (MiB) |
|---:|---|---:|---:|---:|---:|
| 0 | QC-18 | 11.512 | 553.76 | 355.10 | 293.92 |
| 0 | QC-19 | 14.656 | 504.59 | 308.00 | 252.11 |
| 1 | QC-18 | 14.418 | 553.75 | 354.98 | 294.23 |
| 1 | QC-19 | 10.541 | 504.33 | 309.30 | 251.78 |

**Private memory reduction:** approximately 42 MiB, or **14.3%**, in both repeats. PSS decreases about 13%, and summed RSS decreases about 9%. The wall-time variation across these two sequential repetitions prevents a reliable speedup claim. QC-19 aims to reduce private memory while retaining the existing verified arithmetic.

All runs returned identical exact arithmetic data:

- `events_checked`: **3,097**, spanning 13 segments and 14 checkpoints;
- `witness_complex_entries`: **1,206,052**;
- `verified_bound`: **0.011832047006821769**;
- `arithmetic_lower_requirement`: **0.011832047006820182**;
- event counts: 241 channels, 136 gate SVDs, 1,360 left QR and 1,360 right QR transformations.

The bound pertains to the polar-corrected circuit encoded by the witness, as described in QC-14 through QC-18. These are exact checks of binary64 operands interpreted as dyadic rationals, plus the established mathematical inequalities; they rely on the standalone checker's implementation and the QC-18 concurrent-file-stability assumption. No claim of arbitrary-circuit fault tolerance or formal verification of checker source is made.

## Tests and reproduction

Run from `experiments/channel_mpo`:

```bash
python -m pip install 'numpy>=1.24,<3'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 \
  test_qc04 test_qc05 test_qc06 test_qc07 test_qc08 test_qc09 \
  test_qc10 test_qc11 test_qc12 test_qc13 test_qc14 test_qc15 \
  test_qc16 test_qc17 test_qc18 test_qc19
```

Use the committed QC-18 ten-qubit fixture to check the isolated CLI:

```bash
python - <<'PY'
from zipfile import ZipFile
with ZipFile('qc18_certificates/10q_d6_mmap.zip') as z:
    z.extractall('/tmp/qc19-10q')
PY
python -I qc19_checker.py /tmp/qc19-10q/manifest.json --workers 2 --start-method fork
python -I qc19_checker.py /tmp/qc19-10q/manifest.json --workers 2 --start-method spawn
```

The large witness remains a separately distributed QC-18 fixture. With its `manifest.json` available at `/path/to/qc18/manifest.json`, reproduce the four-worker comparison:

```bash
python benchmark_qc19.py --manifest /path/to/qc18/manifest.json \
  --workers 4 --repetitions 2 --out qc19_deep_benchmark_repro.csv
```

`benchmark_qc19.py` checks exact equality of all seven critical result fields before emitting a CSV row; verification failures abort the run. Linux `/proc` metrics are observational peaks at the declared polling cadence, not formal resident-memory upper bounds. A complete OS-level isolation of hostile concurrent writers would require a sealed read-only file descriptor or an immutable filesystem snapshot.

## Next experiment

The major remaining per-worker costs are decoded binary64-to-Fraction tensors and residual-context loops. A compact checked tensor representation could reduce duplicate rational objects, provided the independent checkpoint commitment and local-error arithmetic continue to use the exact values of the pinned binary64 operands. Benchmark future work on the same proof with measured PSS/USS, wall time, and full tamper regressions.
