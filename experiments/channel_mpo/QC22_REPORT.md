# QC-22: Kernel-sealed event tapes for exact-dyadic witness replay

**Date:** 2026-11-10. **Status:** CPU pilot completed and tested, stacked on QC-21.

## Why it matters

QC-18 through QC-21 share a read-only mmap of a source witness file. Read-only
access protects the verifier process from accidental writes through that
mapping; it does not prevent another process with write access to the backing
file from modifying pages while proof events are being replayed. Hashing the
source before replay therefore depends on external file immutability.

QC-22 closes this specific time-of-check/time-of-use gap. It copies the validated
event tape into an anonymous, kernel-sealed Linux `memfd`, pins that snapshot to
the manifest SHA-256, checks the four mandatory seals, and reconstructs the
frame layout and per-segment digests from the sealed mapping. Workers share
this immutable snapshot via `fork`, retaining QC-21's exact dyadic checker.

## Experimental result

Both QC-21 and QC-22 verified the same **48-qubit, depth-16 dephasing** witness:
13 event segments, 14 checkpoint states, 3,097 audited transformations and
1,206,052 changed complex entries. The independent verified physical-circuit
expectation-error upper bound remains

`0.011832047006821769`.

Two paired fresh-process runs used four forked workers and standard-library-only
`python -I -S` interpreters. Process-tree memory is estimated by 50 ms Linux
`/proc/*/smaps_rollup` sampling, with the coordinator and its descendants
summed. These are single observations, not statistically controlled runtime
confidence intervals.

| Method | Runtime trials (s) | Mean runtime (s) | Mean private USS (MiB) | Mean PSS (MiB) | Mean summed RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| QC-21 standard mmap | 7.558, 7.794 | 7.676 | 29.795 | 31.184 | 37.486 |
| QC-22 sealed memfd | 7.751, 7.971 | 7.861 | 32.217 | 33.600 | 39.123 |

The measured incremental runtime was approximately **0.185 s** (2.4%) and
mean private memory increased by **2.42 MiB**. The sealing/copy step itself
finished in under 0.1 seconds on this host. Those are acceptable preliminary
costs for a defense against concurrent tape-file modification; QC-22 is a
security improvement, not a speed or memory improvement.

## Verification

**283/283 local regression tests passed** in three batches (QC-01 through
QC-09: 90; QC-10 through QC-15: 80; QC-16 through QC-22: 113). QC-22 adds
15 adversarial and equivalence checks, covering:

* serial, two-worker and four-worker replay parity;
* seal bit presence; denial of write, shrink, grow and seal-modification;
* original tape mutation **after** sealing, including full two-worker proof
  replay with the original file deliberately corrupted;
* pre-seal tape mutation and altered checkpoint rejection;
* absent or partial seals, failed `F_ADD_SEALS`, double-sealing and spawn
  mode rejection;
* standard-library-only `python -I -S` CLI replay.

Every proof event is still checked against independently derived circuit
scheduling, gate/channel corrections, changed MPO coefficients and checkpoint
boundaries. The event source is the sealed snapshot, never the original path.

## Reproduce

```bash
cd experiments/channel_mpo
python -m unittest -q test_qc20 test_qc21 test_qc22
# Full suite requires NumPy for the earlier simulation tests.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 test_qc04 \
  test_qc05 test_qc06 test_qc07 test_qc08 test_qc09 test_qc10 test_qc11 \
  test_qc12 test_qc13 test_qc14 test_qc15 test_qc16 test_qc17 test_qc18 \
  test_qc19 test_qc20 test_qc21 test_qc22
python -I -S qc22_checker.py /path/to/qc18/manifest.json \
  --workers 4 --start-method fork
python benchmark_qc22.py --manifest /path/to/qc18/manifest.json \
  --workers 4 --repeats 2 --output qc22_deep_benchmark_repro.csv
```

The 48-qubit witness is the existing QC-18/QC-21 pinned artifact, rather than a
duplicated giant binary file. The tiny committed QC-18 fixture supplies CI
inputs. See [QC22_FORMAT.md](QC22_FORMAT.md) for the exact trust and failure
model. Subsequent work could enable descriptor passing for spawn-mode workers
or seal checkpoint byte snapshots as well; such changes would require a fresh
independent-input and threat-model audit.
