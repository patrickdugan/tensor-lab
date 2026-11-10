# QC-21: No-site execution and live exact-norm summaries for proof checking

**Date:** 2026-11-10  
**Status:** CPU verification pilot, frozen four-worker 48-qubit witness comparison; stacked on QC-20.

## Question

QC-20 keeps raw binary64 tensor bytes, streams local integer contractions, and replays a 48-qubit proof in approximately 10 seconds with four workers. Its four-worker aggregate unique set size (USS) on the test host was still approximately 98 MiB. Can the independent checker reduce that footprint while preserving all the QC-15 through QC-20 arithmetic and proof checks?

**Result:** Under this particular Python environment, interpreter startup was an unexpectedly large source of resident memory: `python -I` loaded NumPy through site initialization, while `python -I -S` started with approximately 5.7 MiB RSS and no NumPy. Combined with QC-21's packed checkpoint anchors and live rational summary cache, no-site verification reduced sampled aggregate private memory **from 57.27 MiB (QC-20 no-site) to 34.48 MiB**, a **39.8% reduction** with the same proof and interpreter configuration. Relative to QC-20's former `-I` command, private memory fell **64.9%**, from 98.11 MiB to 34.48 MiB. The no-site comparison gives the better measure of the checker-code improvement.

All runs verified **3,097 transformations** across **13 segments and 14 checkpoints**, checking **1,206,052 changed complex coefficients** and the same end-to-end error bound for the polar-corrected circuit:

```
verified_bound                 0.011832047006821769
arithmetic_lower_requirement   0.011832047006820182
events_checked                 3097
witness_complex_entries        1206052
segments_checked               13
checkpoints_checked            14
event_counts: channel 241, gate_svd 136, left_qr 1360, right_qr 1360
```

## Method

1. **No-site standard-library execution.** Invoke the verifier with `python -I -S qc21_checker.py ...`. `-I` isolates the interpreter, while `-S` avoids `site` initialization. The QC-15 through QC-21 independent checker imports the standard library and its adjacent source files; the simulator and NumPy remain outside the trusted execution path. On this host, the Python site startup imported NumPy automatically. The no-site reduction is **environment-specific**; CPython deployments without heavyweight site initialization will see a smaller startup difference.
2. **Packed checkpoint anchors.** QC-20's coordinator decodes every intermediate checkpoint into tuples of rational values before forking. The QC-21 path validates genesis/final checkpoint hashes, exact packed-Pauli genesis, and exact integer final readout from binary64-backed cores. Every intermediate checkpoint is independently decoded, hashed, checked against the boundary commitment and recomputed after replay by **both adjacent segment workers**. Completion of *all* scheduled segments is mandatory. The previous redundant coordinator pre-decode of intermediate checkpoints is omitted only for QC-21; QC-20's default and compact behaviors remain unchanged.
3. **Live exact-norm summaries.** `SummaryKernel` stores each live core's exact Frobenius-squared norm, left/right Gram defects and their outward-rounded `sqrt_up` factors, keyed by object identity. The original integer operands are cached with a six-entry limit. A changed core's summaries are retired after all its checks are complete. The sequence of `Fraction` multiplications and local residual inequalities matches QC-20. This caching changes computation and storage only; it cannot increase or decrease the declared error bound. A rehashed malicious event with a forged context factor is rejected by the arithmetic checker.

The QC-21 entrypoint is `qc21_checker.py`. An additive `storage_mode='summary'` selector in `qc20_checker.py` reuses the mandatory schedule, segment, correction, local-residual, and independent boundary checks while preserving QC-20's existing default semantics. The cache limit is a resource policy, not a proof parameter.

## Matched four-worker 48-qubit benchmark

Linux `/proc/<pid>/smaps_rollup` sampled every 50 ms over the coordinator and all workers. Two sequential paired fresh-process observations per configuration. Values below are means; the exact raw samples are in `qc21_interpreter_frontier.csv`. Wall times are observations on shared CPU resources, not confidence intervals.

| Checker | Interpreter | Mean wall (s) | Mean summed RSS (MiB) | Mean PSS (MiB) | Mean USS (MiB) |
|---|---|---:|---:|---:|---:|
| QC-20 | `-I` | 10.195 | 474.762 | 131.354 | 98.109 |
| QC-21 | `-I` | 9.216 | 469.279 | 128.073 | 96.850 |
| QC-20 | `-I -S` | 9.448 | 135.570 | 78.543 | 57.268 |
| **QC-21** | **`-I -S`** | **8.306** | **129.812** | **56.442** | **34.484** |

Sampled USS is private process memory. PSS apportions shared pages; summed RSS counts mapped pages once per process. Independent workers compute the same events and exact bound. The no-site environment affects the baseline too, which is why both comparisons are shown. Fewer workers provide a CPU/memory tradeoff; the observed four-worker peak was dominated by interpreter and checkpoint memory before QC-21.

A separate two-worker experiment under `-I` (before adopting `-S`) found QC-20 USS near 98 MiB and QC-21 USS near 97 MiB. This confirmed that the original local peak occurred in coordinator preparation, even with fewer workers.

### Trust and interpretation

The result remains a mathematically checked end-to-end expectation-error bound for the explicitly defined **polar-corrected binary64 circuit**; it is not a global-state fidelity certificate or a Lean-verified implementation. The checker validates hashes and exact dyadic arithmetic. Protection against hostile concurrent rewriting of the underlying mmap file still requires filesystem isolation or a sealed descriptor. The `/proc` sampling interval can miss short-lived memory peaks; all measurements are experimental rather than enforceable limits.

## Reproduction

With a QC-18 read-only tape witness unpacked to `/path/to/witness`:

```bash
cd experiments/channel_mpo
python -I -S qc20_checker.py /path/to/witness/manifest.json --workers 4 --start-method fork
python -I -S qc21_checker.py /path/to/witness/manifest.json --workers 4 --start-method fork
python benchmark_qc21.py --manifest /path/to/witness/manifest.json \
  --workers 4 --repetitions 2 --startups site,no-site \
  --output qc21_interpreter_frontier_replay.csv
```

To run the short standard-library fixture and full regressions:

```bash
python - <<'PY'
from zipfile import ZipFile
with ZipFile('qc18_certificates/10q_d6_mmap.zip') as z:
    z.extractall('/tmp/qc21-ten-qubits')
PY
python -I -S qc21_checker.py /tmp/qc21-ten-qubits/manifest.json --workers 2 --start-method spawn
python -m unittest -q test_qc20 test_qc21
# Run QC-01 through QC-21 tests as three separate CPU batches on constrained runners.
```

## Follow-up

An OS-sealed immutable event tape would close the concurrent-file-modification assumption on Linux. Beyond that, the compact verifier can be reimplemented as a small Rust or formally checked Lean kernel. Its exact dyadic normalization, boundary hash encoding and error inequalities are already expressed in a language-independent fixture format.
