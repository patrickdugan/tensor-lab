# QC-18: Sharing witness bytes between parallel exact verifiers

**Date:** 2026-11-10. **Base:** QC-17 PR #19 (`research/qc17-parallel-checkpointed-witness-20261110`). **Status:** verified research implementation with 219 passing local tests and one complete 48-qubit independently replayed witness.

## Question and outcome

Can QC-17's four-worker exact dyadic checker reduce duplicated process memory by decoding each witness event only when needed, while sharing an immutable read-only event source?

QC-18 introduces an indexed, per-event zlib-compressed tape, read-only file mappings, and a fork-first worker pool on POSIX. The parent parses and audits the immutable circuit schedule once before forking; every worker independently rechecks the arithmetic for each segment from the pinned tape and checkpoint inputs. The streamed event transport avoids caching hundreds of decoded JSON events per worker.

**Outcome:** Under a matched four-worker, 48-qubit benchmark, unique process memory fell from **306.1 MiB** to **290.1 MiB** (5.2% reduction). Proportional shared physical memory improved from **349.6 MiB** to **346.4 MiB** (0.9%). Verification time was essentially unchanged: **10.25 seconds** for QC-17 versus **10.39 seconds** for QC-18 in fresh runs. Summed RSS was *higher*, 520.7 versus 533.2 MiB, because it counts shared file mappings once per process. The memory improvement is limited and depends on the process-start method; a portable four-process spawn run used far more memory. The results are per-run measurements, not timing confidence intervals.

## Exact proof reproduction

The 48-qubit, depth-16, strong-dephasing, bond-cap-8 witness contains **13 segments**, **14 checkpoint states**, **3,097 audited transformations**, and **1,206,052 changed complex tensor coefficients**. QC-18's independent checker reproduced QC-17's verified error bound, `0.011832047006821769`, and the lower arithmetic requirement, `0.011832047006820182`, exactly. It verifies the polar-corrected binary64 circuit of QC-14, with the same residual, context, polar-correction, and final-readout checks.

The QC-18 tape occupies about 15 MiB on disk compared with QC-17's approximately 12.52 MiB of compressed events, checkpoints and manifest. The extra disk storage buys random access to individually compressed event records and sharing of the index and tape pages through `mmap`. QC-18 can also transport uncompressed frames for benchmarking; they occupy approximately 25.33 MiB and gave no compelling memory advantage.

## Fresh-process memory measurements

All rows use the same 48-qubit witness on Linux, 3,097 operations, 13 segments, and an 80 ms `/proc` sampling interval. RSS, PSS, and USS aggregate parent and workers; the maxima of the three fields are sampled separately and need not occur in the same instant. `PSS` apportions shared pages; `USS` sums `Private_Clean + Private_Dirty`; RSS double-counts shared mappings. Small variations in process scheduling, allocators, and sampling can affect the results.

| Workers | Backend | Time (s) | Aggregate RSS (MiB) | Aggregate PSS (MiB) | Aggregate USS (MiB) |
|---:|---|---:|---:|---:|---:|
| 1 | QC-17 | 34.012 | 123.2 | 110.5 | 105.3 |
| 1 | QC-18 zlib | 33.472 | 130.6 | 117.8 | 112.7 |
| 2 | QC-17 | 18.299 | 313.6 | 222.3 | 179.3 |
| 2 | QC-18 zlib/fork | 18.662 | 333.7 | 227.1 | 174.0 |
| 4 | QC-17 | 10.251 | 520.7 | 349.6 | 306.1 |
| 4 | QC-18 zlib/fork | 10.388 | 533.2 | 346.4 | 290.1 |
| 4 | QC-18 zlib/spawn | 11.732 | 715.2 | 560.8 | 533.6 |

This test shows a limited benefit in uniquely owned worker memory for four forked workers. It does not establish a major reduction in total resident physical memory. The original QC-17 segment representation is a strong reference baseline, since Linux also shares many of its Python code and library pages across forked workers.

## Verification, adversarial checks, and limitations

All **219** tests pass locally across QC-01 through QC-18. The 17 new tests cover strict frame lengths and contiguity, raw versus zlib encoding, serial versus fork/spawn parallel equivalence, isolated CLI execution, independent segment checks, tampered tensor residuals with all transport checksums recalculated, reordered events, malformed checkpoint files, symlinked tape files, missing segments and noncanonical proof inputs.

The event decoder never imports the simulator or NumPy. Cryptographic hashes bind the exact encoded bytes, and every local transformation is checked by QC-16's independent exact-dyadic integer kernel. The checker is written in Python and has no external formal proof. A valid certificate proves the specified physical observable bound under the QC-14 assumptions, for a polar-corrected circuit, not for an arbitrary uncensored physical device.

The mmapped tape is read-only to the checker; the underlying file must remain stable while the process runs. The compressed event stream improves random-access memory behavior at the expense of slightly larger disk storage and zlib decode overhead. Parallel worker workspaces and large Python exact-integer temporaries remain private. Future work should profile memory allocation inside the exact kernel itself, benchmark repeated independent runs, and test authenticated immutable inputs or sealing for hostile concurrent-writer environments.

## Reproduction

```bash
cd experiments/channel_mpo
python -m pip install 'numpy>=1.24,<3'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest -q \
    test_channel_mpo test_purified_mpo test_qc03 test_qc04 \
    test_qc05 test_qc06 test_qc07 test_qc08 test_qc09 \
    test_qc10 test_qc11 test_qc12 test_qc13 test_qc14 \
    test_qc15 test_qc16 test_qc17 test_qc18
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
with ZipFile('qc18_certificates/10q_d6_mmap.zip') as z:
    z.extractall('/tmp/qc18-10q')
PY
python -I qc18_checker.py /tmp/qc18-10q/manifest.json --workers 2
# For the 48-qubit QC-17 proof directory, generate an equivalent mmap tape:
python -I qc18_tape.py /tmp/qc17-deep48/manifest.json --out-dir /tmp/qc18-deep48
python -I qc18_checker.py /tmp/qc18-deep48/manifest.json --workers 4
python benchmark_qc18.py --qc17 /tmp/qc17-deep48/manifest.json \
  --qc18 /tmp/qc18-deep48/manifest.json --workers 1,2,4 --out qc18_bench.csv
```

The full 48-qubit tape is distributed separately as a reproducible archive rather than committed to GitHub. See `QC18_FORMAT.md` for the binary format, verifier requirements, and security assumptions.
