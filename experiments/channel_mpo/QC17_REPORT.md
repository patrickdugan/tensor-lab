# QC-17: Parallel independent checking of checkpointed MPO witnesses

**Date:** 2026-11-10  
**Repository:** `patrickdugan/tensor-lab`; PR stacked on QC-16  
**Evidence:** 202 passing QC-01 through QC-17 local regression tests, verified 48-qubit full witness, frozen fresh-process runtime and process-tree RSS benchmarks.

## Research question

QC-16 independently checked all 3,097 operations of a depth-16, 48-qubit Heisenberg-MPO witness in about 35 seconds with one worker. A full witness can be partitioned into claim segments with independently reconstructed and checked state boundaries. How much CPU time does verification save by checking each segment concurrently, and what is the aggregate RAM cost?

**Measured result:** For the recorded 48-qubit proof, QC-17 checks the complete witness with four workers in 11.21 seconds against 33.95 seconds serially (3.03x). The identical expectation-error bound `0.011832047006821769` is accepted by every configuration. Four workers raised sampled peak aggregate process-tree RSS from 126,240 KiB to 529,412 KiB. These are three fresh-process timing observations, not a sampling distribution. A subsequent four-worker replay independently returned `PASS` with the same exact counts and bound.

## Certificate and algorithm

The producer splits the complete QC-15 operand trace into QC-16 event segments and inserts exact, serialized MPO state checkpoints. The 48-qubit example contains 13 segments and 14 checkpoint records. Each checkpoint carries its position in the event schedule, the physical MPO support, the exact dyadic error accumulator, four event counters and the complete stored binary64 MPO cores. A domain-separated SHA-256 commitment binds these fields; individual compressed files also have byte checksums.

The separate QC-17 checker reconstructs the event schedule from the serialized circuit and Pauli target. It verifies genesis and final readout independently. Each worker receives the pinned manifest, the segment index and claimed checkpoint boundaries, then replays all transitions with the QC-16 common-denominator exact-integer kernel. The coordinator requires every worker to finish successfully and checks global event and witness-entry limits.

The exact error recurrence is

\[
 E_{j+1}=E_j+r_j g_j+\alpha_j(1+E_j),
\]

where each residual upper bound `r`, context upper bound `g` and polar-correction bound `alpha` is checked by independent exact arithmetic before being accumulated. The end-to-end statement remains the QC-14 absolute expectation-error bound for the polar-corrected binary64 circuit. The producer's boundary calculations are treated as claims until worker replay.

A detached segment is a conditional transition certificate. Acceptance of the whole circuit requires the exact genesis anchor, verified chain and final readout. The checker imports only its standalone QC-15/16 arithmetic modules and the QC-17 serialization module; it does not import NumPy or simulator code. See `QC17_FORMAT.md` for the schema, invariants and threat model.

## Measured 48-qubit results

Configuration: 48 qubits, 16 circuit layers, dephasing `p=0.5`, bond cap 8, central Pauli expectation. The complete proof contains 3,097 audited operations (`channel=241`, `gate_svd=136`, `left_qr=1360`, `right_qr=1360`) and 1,206,052 changed complex MPO coefficients. The 16 support-growth events are separate schedule records.

| Worker processes | Verification time (s) | Paired speedup | Peak aggregate RSS (KiB) | Approx. MiB |
|---:|---:|---:|---:|---:|
| 1 | 33.953 | 1.00x | 126,240 | 123.3 |
| 2 | 18.550 | 1.83x | 320,676 | 313.2 |
| 4 | **11.211** | **3.03x** | 529,412 | 517.0 |

The benchmark starts a new isolated Python process for each worker count and samples `/proc` resident memory of the parent and its descendants every 30 milliseconds. It reports the largest sampled *sum*, which may underestimate instantaneous peaks between samples. The values include the coordinator and workers; they are neither virtual memory reservations nor individual-worker RSS peaks. The serial and parallel runs all returned the identical accumulated bound, event counts and changed-entry count.

Stored proof sizes: the 13 compressed event files occupy **11,995,527 bytes**, 14 compressed checkpoints **969,983 bytes**, and the manifest **156,679 bytes**, totalling **13,122,189 bytes** (approximately 12.52 MiB). The additional checkpoints cost about 0.93 MiB compressed. The larger proof archive is distributed as a separate reproducible artifact and is not committed to GitHub.

The exact reported expectation is `6.8465099397700234e-09`; the lower required error is `0.011832047006820182`; the accepted upper bound is `0.011832047006821769`. These numbers match QC-16.

## Small-circuit parallelism

The fresh-process CPU comparison covers a three-qubit depth-three fixture, a ten-qubit depth-six fixture and a twelve-qubit depth-eight fixture. All configurations agree on result and event count. Short proofs are dominated by process startup, scheduling and checkpoint validation.

| Circuit | Audited ops | Workers | Serial (s) | Parallel (s) | Speedup |
|---|---:|---:|---:|---:|---:|
| 3 qubits / depth 3 | 11 | 2 | 0.648 | 0.679 | 0.95x |
| 10 qubits / depth 6 | 48 | 4 | 0.774 | 0.715 | 1.08x |
| 12 qubits / depth 8 | 107 | 4 | 1.173 | 0.933 | 1.26x |

Use one worker for tiny proofs; four workers become worthwhile for this deeper witness. These single-shot results are platform dependent.

## Verification and adversarial tests

All **202 tests** passed locally across QC-01 through QC-17. The 17 new QC-17 tests cover serial and parallel agreement with QC-16, independent segment verification, an isolated Python CLI, exact cross-channel small-system checks, and attempts to forge a checkpoint with all neighboring checksums repaired. The suite also rejects missing, duplicated, reordered or path-traversing payloads, understated event residuals, forged manifest pins, altered genesis states and final bounds. The ten-qubit fixture is committed as a zipped archive and checked by lightweight GitHub CI using only the standard library. The complete regression workflow additionally installs NumPy for historical simulator tests.

A separately rerun four-worker replay of the 48-qubit proof returned all 13 accepted segments, 14 verified checkpoints, and the exact bound `0.011832047006821769`.

## Reproduce

```bash
cd experiments/channel_mpo
python -m pip install 'numpy>=1.24,<3'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 \
  test_qc04 test_qc05 test_qc06 test_qc07 test_qc08 test_qc09 \
  test_qc10 test_qc11 test_qc12 test_qc13 test_qc14 test_qc15 \
  test_qc16 test_qc17

python - <<'PY'
from zipfile import ZipFile
with ZipFile('qc17_certificates/10q_d6_checkpointed.zip') as archive:
    archive.extractall('/tmp/qc17_small')
PY
python -I qc17_checker.py /tmp/qc17_small/manifest.json --workers 2

# For a downloaded 48-qubit proof archive extracted to /tmp/qc17_deep:
python -I qc17_checker.py /tmp/qc17_deep/manifest.json --workers 4
python benchmark_qc17.py --manifest /tmp/qc17_deep/manifest.json \
  --workers 1,2,4 --out /tmp/qc17_benchmark.csv
```

## Limits and next directions

Four-worker verification consumes around half a gigabyte of aggregate RAM in the recorded 48-qubit test. Execution uses Python multiprocessing and duplicates the circuit description and some arithmetic state. Better packaging could stream immutable circuit coefficients from a read-only mapping, keep fewer checkpoints resident at once, and share deduplicated MPO cores. Results apply to one selected Pauli expectation and a polar-corrected binary64 physical circuit, under QC-14's inequalities. Independent formal proof of the checker remains open. Additional parallelism may plateau because of file I/O, process creation and uneven segment costs.
