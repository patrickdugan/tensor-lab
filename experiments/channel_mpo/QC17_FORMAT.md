# QC17-CHECKPOINTED-WITNESS-v1

## Certificate statement

A QC-17 proof is a collection of verified local tensor transitions for one backward Heisenberg-MPO circuit. Every accepted result is tied to a precise target observable, an input circuit, and an absolute expectation-error bound for the polar-corrected binary64 gate/Kraus circuit described in QC-14. The arithmetic is exact over the stored dyadic rationals; the checker implementation and its analytic error inequalities are the remaining trusted elements.

QC-17 allows a segment to be checked in an independent process. A successful *isolated segment* check asserts a transition between its claimed checkpoints. The complete certificate requires all segments, an independently constructed exact Pauli genesis state, a verified final expectation and error bound, and agreement at every neighboring checkpoint. SHA-256 binds the payloads and state transitions. It supplies integrity under collision resistance and does not authenticate authorship.

## File structure

The root `manifest.json` is the canonical checksummed JSON envelope inherited from QC-16, with fields:

- `format`: `qc17-checkpointed-full-witness-v1`.
- `circuit`: the QC-15 binary64 gate and Kraus coefficients, ordered layers, target sites and Pauli observable.
- `result`: the final expectation, error bound, cone width and operation counts.
- `segments`: the QC-16 ordered event segment descriptors, including file hashes, indices and endpoint commitments.
- `checkpoints`: a list of **one more** checkpoint descriptor than the number of segments, from genesis through final readout.

Each checkpoint descriptor contains `filename`, `event_index`, `commitment` and `file_sha256`; filenames follow `checkpoint_[0-9]{4,5}.json.gz`. The referenced compressed checkpoint contains `format`, `event_index`, `first`, `last`, `cores`, `accumulated`, `tally`, `commitment`, and an envelope checksum. Cores use base64-encoded little-endian `(real, imaginary)` IEEE binary64 pairs and tensor shapes. The exact accumulated nonnegative dyadic rational is encoded as a hexadecimal numerator and a base-two denominator exponent. Event counts are recorded for `channel`, `left_qr`, `right_qr` and `gate_svd`.

A checkpoint commitment is recomputed from the tensor values, spatial support, exact accumulated error, event index and counts via QC-16's domain-separated `boundary_digest`. The verifier checks the checkpoint's compressed-byte hash and canonical JSON checksum and recomputes its semantic commitment. Paths, shapes, sizes and event counts are subject to the QC-15/16 resource limits.

## Independent verification procedure

1. Read the full manifest and reconstruct the backward causal-cone event schedule from the stated circuit. Check contiguous segment indices, unique safe filenames and checkpoint-to-segment links.
2. Load every checkpoint and independently recompute its commitment. Construct the exact target Pauli MPO and check the genesis checkpoint; recompute the final MPO readout and require the claimed error bound to cover the accumulated error plus readout discrepancy.
3. For each segment, independently load the claimed starting state. Execute the exact dyadic channel, QR and gate arithmetic for *every* event according to the reconstructed schedule. Require every claimed residual and context bound to enclose the recomputed exact quantity. Carry the error recurrence with exact rational arithmetic, and check the resulting output checkpoint digest.
4. Require every segment to pass and verify the total number of audited events and modified tensor coefficients remains within the global resource caps.

With `--workers N`, step 3 runs concurrently in at most `N` independent Python processes. The coordinator checks both global anchors and aggregates every result. An arbitrary segment may be checked separately using `verify_one_segment`, provided the manifest is pinned; this does not independently establish the final statement.

A different or missing segment, forged boundary, changed coefficient, reordered event, understated residual, invalid checkpoint genesis or altered final error bound causes rejection. Recomputing a JSON checksum cannot circumvent those arithmetic and schedule checks.

## Reproduce the small fixture

```bash
cd experiments/channel_mpo
python -I qc17_checker.py qc17_certificates/10q_d6_checkpointed/manifest.json --workers 2
# Or extract the included 10q_d6_checkpointed.zip into /tmp/qc17_fixture:
python - <<'PY'
from zipfile import ZipFile
with ZipFile('qc17_certificates/10q_d6_checkpointed.zip') as archive:
    archive.extractall('/tmp/qc17_fixture')
PY
python -I qc17_checker.py /tmp/qc17_fixture/manifest.json --workers 2
```

To produce an equivalent checkpointed witness from the archived QC-15 full witness:

```bash
python qc17_emit.py qc15_certificates/full_10q_d6_dephasing_s3_c4.json.gz \
  --out-dir /tmp/qc17_produced --events-per-segment 12
python -I qc17_checker.py /tmp/qc17_produced/manifest.json --workers 1
python -I qc17_checker.py /tmp/qc17_produced/manifest.json --workers 2
```

The checker needs Python >= 3.10 and standard-library modules only. The producer's full experiment suite also uses NumPy. The larger 48-qubit archive is distributed separately to avoid enlarging the repository history.

## Trust and limits

Verification establishes the numerical statement encoded by the complete witness under the documented QC-14 error inequalities. It makes no full-state fidelity claim, and the checker has yet to be formally verified. A compressed proof ledger containing only scalar residuals offers a weaker evidence level. Workers are trusted to execute the checker implementation faithfully; a dishonest worker in a distributed deployment must be handled with redundant verification or an authenticated execution protocol. Process-parallelism does not reduce the total exact-arithmetic work.
