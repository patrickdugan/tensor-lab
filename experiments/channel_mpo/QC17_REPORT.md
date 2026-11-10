# QC-17: Checkpointed parallel full-witness verification

**Date:** 2026-11-10  
**Repository:** `patrickdugan/tensor-lab`  
**Base:** QC-16, stacked on PR #18  
**Evidence:** 202/202 local regression tests, independent 48-qubit verification and fresh-process timing/memory measurements.

## Research question

QC-16 reduced standalone verification of a 48-qubit, depth-16, dephasing Heisenberg-MPO proof from QC-15's recorded 390 seconds to 34.57 seconds using exact dyadic integer arithmetic. The remaining computational cost is distributed over 13 independently delimited segments. Can we check those segments concurrently without trusting their starting states or weakening the error bound?

**Result:** QC-17 produces a checkpointed format with independently checked MPO states at every boundary. Each worker verifies a complete exact-dyadic local transition sequence against a deterministically reconstructed global event schedule. The coordinator verifies genesis, the chain of SHA-256 boundary commitments, the final readout, cumulative error, and the global changed-entry limit. Four workers checked the 48-qubit proof in **11.21 seconds**, compared with **33.95 seconds** for a serial QC-17 run (3.03× faster). The complete witness statement and exact verified error bound were unchanged.

## Certificate specification and security

See [`QC17_FORMAT.md`](QC17_FORMAT.md) for the complete format, parsing bounds, mathematical induction argument and security assumptions. The producer emits the full QC-16 event segments and an added checkpoint for each boundary. Each checkpoint stores the exact tensor coefficients, physical support, accumulated rational error and audited event counts, with the same QC-16 boundary digest. The verifier reconstructs its own event schedule from the circuit, checks all segment arithmetic, and requires each computed output digest to match the next independently decoded checkpoint.

Critically, **a detached segment is not a proof of the full circuit**. The final result is accepted only after every segment has passed, the exact Pauli genesis has been checked and the final readout/bound is validated. The producer is excluded from the verifier's imports. The verifier uses Python's standard library and QC-15/16's independent arithmetic code, without NumPy, QR or SVD libraries.

The segment/checkpoint chain relies on SHA-256 collision resistance for matching states across processes. No external digital signature, trusted execution environment, formal verification of Python, or reduction to a zero-knowledge proof is claimed.

## Deep-circuit verification

- Physical circuit: **48 qubits**, depth **16**, dephasing strength `p=0.5`, central observable, bond cap **8**.
- Source witness: the QC-15 full operand witness; archived 48-qubit segmented QC-16 and QC-17 variants are distributed alongside this report. Source full witness SHA-256 `3bbd436120fee57677fcab1d970e60802c523ad9817e955ab2016ab9f14e8889` (from QC-16).
- Segments: **13**, plus **14** checked state checkpoints.
- Checked transformations: **3,097** (`channel` 241, `gate_svd` 136, `left_qr` 1360, `right_qr` 1360).
- Changed complex MPO coefficients checked: **1,206,052**.
- Reported expectation: `6.8465099397700234e-09`.
- Independently checked expectation error bound: **`0.011832047006821769`**. The error-bound sum preceding final readout rounding is `0.011832047006820182`.
- The 13 unmodified QC-16 event-segment files occupy **11,995,527 bytes**. The QC-17 checkpoints add **969,983 bytes** compressed, and the root manifest is **156,679 bytes**. The complete QC-17 directory is approximately 13 MB.

### Fresh-process resource results

The same verified 48-qubit proof was run under three configurations on this container (5 available logical AMD EPYC cores). `benchmark_qc17.py` samples Linux `/proc` parent PIDs to count the coordinator **and all child verifier processes** when reporting aggregate RSS. Sampling intervals are 30 ms; the measured maxima may miss short transient allocations and are not formal upper bounds.

| Workers | Elapsed time | Paired speedup | Sampled aggregate peak RSS | Sampled largest single process |
|--:|--:|--:|--:|--:|
| 1 | 33.953 s | 1.00× | 126,240 KiB | 126,240 KiB |
| 2 | 18.550 s | 1.83× | 320,676 KiB | 109,156 KiB |
| 4 | **11.211 s** | **3.03×** | 529,412 KiB | 109,800 KiB |

The additional RAM cost is real and substantial. This result establishes lower latency with adequate cores and memory. QC-16's 34.57-second verifier is a separately recorded baseline; the paired QC-17 speedups above use the new serial run. Per-process RSS must not be mistaken for total process-tree RSS.

### Small-fixture measurements

Fresh Python interpreter for each configuration, times include certificate parsing and worker startup. Small proofs show little or no advantage from multiple processes:

| Qubits / depth | Audited events | Serial time | Four-worker setting | Speedup |
|:--|--:|--:|--:|--:|
| 3 / 3 | 11 | 0.648 s | 0.740 s | 0.88× |
| 10 / 6 | 48 | 0.774 s | 0.715 s | 1.08× |
| 12 / 8 | 107 | 1.173 s | 0.933 s | 1.26× |

At three qubits the proof has only one segment, so the four-worker setting starts just one worker. At larger depth, segments contain enough exact contractions to offset the cost of process creation and duplicated metadata parsing.

## Verification and adversarial tests

The full QC-01 through QC-17 regression suite passes: **202/202 tests**. The 17 QC-17 tests cover serial/parallel differential replay, exact agreement with the QC-16 dyadic kernel, isolated `python -I` execution, depolarizing/dephasing/amplitude-damping witnesses, deterministic proof emission, missing or duplicate file records, path escapes, damaged compressed bytes, altered intermediate states with repaired checksums, forged error bounds, reordered events with regenerated commitments, altered local residuals, and coordinator-to-worker manifest pinning.

The repository includes a compact prebuilt 10-qubit checkpointed proof for a standard-library-only CI smoke test. Large witnesses and the 48-qubit checkpoints are published separately from the Git source tree to keep pull-request CI lightweight.

## Commands

```bash
cd experiments/channel_mpo
python -m pip install 'numpy>=1.24,<3'  # only needed for producer/full regression tests
python qc17_emit.py qc15_certificates/full_10q_d6_dephasing_s3_c4.json.gz \
  --out-dir /tmp/qc17-10q --events-per-segment 12
python -I qc17_checker.py /tmp/qc17-10q/manifest.json --workers 1
python -I qc17_checker.py /tmp/qc17-10q/manifest.json --workers 4
python benchmark_qc17.py --small --out /tmp/qc17-small.csv
python benchmark_qc17.py --manifest /path/to/48q/manifest.json \
  --out /tmp/qc17-deep.csv
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest -q test_channel_mpo test_purified_mpo \
  test_qc03 test_qc04 test_qc05 test_qc06 test_qc07 test_qc08 \
  test_qc09 test_qc10 test_qc11 test_qc12 test_qc13 test_qc14 \
  test_qc15 test_qc16 test_qc17
```

## Next experiment

QC-18 should reduce the parallel verifier's aggregate memory without losing its independent-event semantics. Candidate approaches include a process pool that shares immutable checkpoint/circuit pages across workers, arena allocation for dyadic cores, and a compact Rust/Lean checker. Any acceleration should report a paired error bound, elapsed time, **aggregate** memory, and the extra trust introduced by the new execution model.
