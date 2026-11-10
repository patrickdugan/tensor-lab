# QC16-SEGMENTED-WITNESS-v1

## Statement and threat model

`qc16-segmented-full-witness-v1` represents one *QC-15 full operand witness* as a root manifest plus ordered independently checksummed segment files. The standalone verifier replays **every** gate, channel and QR/SVD update using exact dyadic integer arithmetic; it reconstructs and checks the MPO state and exact accumulated error at every boundary. Checking a segment alone establishes nothing about the original circuit. The complete chain must be verified from the exact Pauli initialization to the final MPO readout.

The trust assumptions are the arithmetic implementation, the documented norm inequalities, and SHA-256 collision resistance for the boundary/content commitments. Checksums provide **integrity and linkage**, not identity authentication, zero knowledge, or a formal proof of the Python implementation. No NumPy, BLAS or simulator code is imported by the verifier.

## File structure

`manifest.json` is a QC-15 canonical JSON envelope with a `sha256` checksum of its contents (sorted ASCII keys, compact separators, excluding the checksum). It contains exactly:

- `format`: `qc16-segmented-full-witness-v1`
- `circuit`: unchanged QC-15 specification, including complete binary64 gates and Kraus matrices
- `result`: unchanged QC-15 expectation, bound, cone width and audited event counts
- `segments`: a nonempty ordered array of segment descriptors

Each descriptor has `filename` matching `segment_[0-9]{4,5}.json.gz`, `start_index` (zero-based), `count` (1–1000), `start_commitment`, `end_commitment` and `file_sha256`. The `start_index` entries must form a contiguous interval from zero, with no gaps, duplicate filenames or overlaps. The total is at most the QC-15 event budget. Every segment is a `qc16-event-segment-v1` envelope, with the matching descriptor fields apart from `file_sha256`, an ordered `events` array and its own canonical `sha256` checksum. `file_sha256` binds the exact compressed bytes. Relative path escapes and symlinked segment files are rejected.

## State-boundary commitment

For a boundary after event `j`, the SHA-256 input is the ASCII prefix `qc16-state-boundary-v1` followed by a NUL byte, then these length-delimited fields in order:

1. `j`, the physical support's first and last qubit coordinates, and the number of MPO cores, as signed-magnitude integers.
2. Numerator and denominator of the *exact* `fractions.Fraction` accumulated error recurrence. The accumulator begins at zero and follows `E_next = E + r*g + alpha*(1+E)`, interpreting every binary64 hexadecimal scalar as an exact dyadic rational.
3. Counts of `channel`, `left_qr`, `right_qr` and `gate_svd` events in that order.
4. For each MPO core: its number of dimensions, its dimensions as integers, and the row-major core coefficients as `<dd` IEEE-754 little-endian `(real,imag)` pairs, reconstructed from the **verified** exact stored coefficients. Signed zero is normalized by rational decoding.

Each integer is encoded as `sign: one byte`, `byte_length: four-byte big-endian unsigned`, then `magnitude: shortest nonempty big-endian unsigned bytes`, including one zero byte for zero. The boundary digest binds support, tensor coefficients, the accumulated-error value and event count. This prevents mixing two algebraically different traces that happen to have the same hash of their raw event list.

The emission tool computes claimed boundary commitments **without checking** event arithmetic; the independent verifier instead calls back into the exact QC-15 transition checker after each event. Its recomputed state digest must match each segment boundary. The initial state is constructed from the `circuit` observable and is checked, including at boundary index zero. The final state is also checked against the result's readout and error claim. Rehashing tampered JSON cannot override the arithmetic checks.

## Optimized arithmetic

Any literal binary64 scalar `x` has a unique dyadic rational representation `n/2^e`. For a tensor, choose `E = max(e_i)` and store each complex coefficient as a pair of integers with denominator `2^E`. Products then have denominator `2^(E_a+E_b)`; sums of products accumulate Python integers. Before subtracting tensors, align denominators with integer left shifts. The squared residual is converted to a rational *once*, after the integer sum of squares. The same method applies to channel contractions, Gram defects, context Frobenius norms and the final MPO readout.

The fixed-width source representation is **not** used for intermediate arithmetic. Python integers expand without overflow. A 96-tensor LRU cache stores recently converted cores. The QC-15 `Fraction` formulas remain available as `--reference` and are compared exactly in tests. The residual upper bounds themselves are still the original binary64 claims checked against exact squares, with no change to the proved numerical statement.

## Limits and command-line reproduction

The checker preserves QC-15's finite witness event/tensor budgets and adds a 24 MB compressed limit per segment. It loads one segment at a time, verifies both its canonical JSON and compressed-file hash, then checks its events. The total event witness and readout remain algebraically linked even when the in-memory segment changes.

```bash
cd experiments/channel_mpo
python qc16_segments.py qc15_certificates/full_10q_d6_dephasing_s3_c4.json.gz \
    --out-dir /tmp/qc16_example --events-per-segment 16
python -I qc16_checker.py /tmp/qc16_example/manifest.json
python -I qc16_checker.py /tmp/qc16_example/manifest.json --reference
python -I qc16_checker.py qc15_certificates/full_10q_d6_dephasing_s3_c4.json.gz
OPENBLAS_NUM_THREADS=1 python -m unittest -q test_qc15 test_qc16
```

The emitter needs the QC-15 full witness. The checker accepts both the full-witness QC-15 container and the segmented QC-16 container; it **rejects QC-15 conditional scalar ledgers**. Both verifier arithmetic backends use only Python's standard library.
