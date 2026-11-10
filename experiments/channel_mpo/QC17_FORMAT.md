# QC17-CHECKPOINTED-FULL-WITNESS-v1

## Claim and scope

QC-17 verifies the same polar-corrected binary64 quantum-circuit expectation claim as QC-15/16. It adds **operand checkpoints** to QC-16's segmented transcript so the segments can be checked in separate processes. A segment verifies only a conditional transition from its supplied input state; **the verifier accepts the circuit only after every segment has been checked**, starting from the exact Pauli initialization and ending at the checked readout.

All core operations are checked over exact dyadic integers using QC-16's independent standard-library `Kernel`. The full proof has the same circuit approximation/correction assumptions and mathematical norm inequalities as QC-14/15; the Python checker is not machine-verified. SHA-256 is assumed collision resistant and supplies commitment continuity and content integrity, without authenticating an author or establishing zero knowledge.

## Manifest

`manifest.json` uses the QC-15 canonical JSON checksum envelope (`sha256` over the sorted, compact, ASCII JSON payload). Its payload contains precisely:

- `format = "qc17-checkpointed-full-witness-v1"`
- `circuit` and `result`, unchanged from the QC-15 full-witness specification
- `segments`, in the QC-16 format, consecutively covering all events
- `checkpoints`, one per segment boundary, with `len(checkpoints) = len(segments) + 1`

Each checkpoint descriptor has exactly `filename`, `event_index`, `commitment`, and `file_sha256`. The filename is `checkpoint_NNNN.json.gz` (four or five digits); indexes must equal successive segment starts and ends; commitments must match both adjacent segment descriptors. `file_sha256` binds the compressed checkpoint bytes.

Each checkpoint uses the same `sha256` canonical JSON envelope and contains exactly:

- `format = "qc17-state-checkpoint-v1"`
- `event_index`, `first`, `last`, the event index and inclusive physical support
- `cores`, each `{shape:[left,2,2,right], binary64_le:<base64>}`, with little-endian complex binary64 entries
- `accumulated = {numerator_hex, denominator_exp}`, expressing the nonnegative dyadic error sum as `int(numerator_hex,16) / 2**denominator_exp`
- `tally = {channel,left_qr,right_qr,gate_svd}`, the audited event totals
- `commitment`, the QC-16 `qc16-state-boundary-v1` digest of the decoded **exact** boundary state

The checkpoint digest includes the event index, support, every tensor coefficient, exact accumulated error, and operation counts. Binary64 signed zeros are normalized by exact-rational decoding. Checkpoint files have a 24 MB compressed-file cap and QC-15's 40 MB decompressed JSON cap. Filenames are restricted to a safe local pattern; symlinks are rejected. The verifier checks a compressed file's SHA-256 and decodes that **same immutable byte snapshot**.

## Verification procedure

1. Validate the root manifest and all segment/checkpoint descriptors, their order, safe filenames, and linked hashes. Parse the circuit and independently construct the full backward causal-cone operation schedule.
2. Read checkpoint zero. Require the exact specified initial Pauli cores, zero accumulated error, zero operation counts, and the exact output observable support.
3. Read/check the final checkpoint, including the exact reconstructed MPO readout, the claimed binary64 expectation/error bound and the total event counts. Every intermediate checkpoint's canonical contents and digest are also checked.
4. For each segment, independently parse and check the root statement, its starting checkpoint and its event bytes. Verify every `grow`, `channel`, `left_qr`, `right_qr`, and `gate_svd` event against the predetermined schedule. Recompute local residuals, outer-context bounds and physical-map polar allowances over exact integers and rational error accumulation.
5. Require the independently computed output boundary digest to equal the segment descriptor and the corresponding next checkpoint. Confirm that **all** segments passed, all non-grow operations were checked, and that the aggregate changed-entry count respects the QC-15 whole-witness cap.

All worker processes must parse the same canonical manifest digest chosen by the coordinator. This pins the common circuit statement if manifest files are changed during a concurrent run. The coordinator accepts the final result only after all worker reports pass. Individual worker results must **never** be mistaken for a full-circuit proof.

## Soundness argument (conditional)

Checkpoint zero is exactly the initial observable. Suppose the checked start boundary of segment `i` equals the corresponding state from a valid prefix. The independent event checker verifies every local update and error inequality in segment `i`, computing its output state and cumulative error. The output digest matches the `i+1` checkpoint digest. Assuming collision resistance, the next checkpoint must encode the same state and error. Induction across all segments establishes the same complete checked transition transcript as QC-16; the final reconstructed expectation and claimed bound are checked separately. A forged input checkpoint, altered local residual, missing event, or reordered event fails one of these checks unless the adversary breaks a hash commitment or the checker/arithmetic assumptions.

Parallelism changes evaluation order **across** segments, while preserving the predetermined event order **inside** each segment and retaining the same mathematical recurrence.

## Commands

```bash
cd experiments/channel_mpo
python qc17_emit.py qc15_certificates/full_10q_d6_dephasing_s3_c4.json.gz \
  --out-dir /tmp/qc17-10q --events-per-segment 12
python -I qc17_checker.py /tmp/qc17-10q/manifest.json --workers 1
python -I qc17_checker.py /tmp/qc17-10q/manifest.json --workers 2
python -m unittest -q test_qc17
python benchmark_qc17.py --small --out /tmp/qc17-small.csv
```

A prebuilt 10-qubit proof archive is included at `qc17_certificates/10q_d6_checkpointed.zip`; extract its files into a single directory. The checker requires Python 3.10+ standard library only. Full simulator and producer test suites require NumPy.
