# QC-18 read-only event tape and independent witness format

Schema: `qc18-mmap-checkpointed-full-witness-v1`. The checker is implemented in `qc18_checker.py`, which imports only the QC-15/16/17 independent proof and arithmetic modules. It does not use NumPy or the circuit simulator.

## Payloads

A QC-18 proof directory contains `manifest.json`, `events.q18t`, and checkpoint files named `checkpoint_NNNN.json.gz`. The checkpoint files follow `QC17_FORMAT.md`; the producer copies their exact bytes and the checker rederives their SHA-256 commitments from the decoded MPO cores, dyadic error accumulator, support and event tallies.

The manifest has the exact fields `format`, `circuit`, `result`, `segments`, `checkpoints`, and `tape`. Each segment has `start_index`, `count`, `start_commitment`, `end_commitment`, and `events_sha256`. The tape descriptor contains the literal filename `events.q18t`, `sha256`, `bytes`, `events`, and `encoding` (`zlib-event-v1` or `raw-event-v1`). All integer offsets are bounded and contiguous; segment counts, event indices and checkpoint boundaries must agree with the independently reconstructed circuit schedule.

The tape is binary, little-endian:

```
magic:     9 bytes: ASCII QC18TAPE + 0x02 (zlib) or 0x01 (raw)
count:     unsigned 32-bit event count
index:     count consecutive records of <uint64 payload_offset, uint32 payload_size>
frames:    count consecutive <uint32 payload_size, payload_bytes> records
```

Offsets in the index point to the beginning of each event's payload relative to the first frame's length field. Every frame must be exactly contiguous with the previous one, its embedded length must match the index, and no trailing bytes are permitted. Event payloads are canonical UTF-8 JSON objects serialized with sorted keys, compact separators, and ASCII escaping. In zlib mode each JSON object is independently compressed with RFC 1950 zlib; the verifier imposes a 2 MB decompressed size cap per event. The tape file is limited to 256 MB. A segment's `events_sha256` is SHA-256 of the concatenation of *length-prefixed, encoded* frames belonging to that segment. The whole tape has a separate SHA-256 digest, which the checker computes over the pinned mapping. The manifest and checkpoints retain QC-15/17's canonical JSON checksums.

## Trust and proof scope

Each worker independently replays its assigned segment against a locally reconstructed event schedule, using the exact integer dyadic kernel. The coordinator verifies that genesis is exactly the requested Pauli observable, that every segment and checkpoint boundary links, that every local residual and polar correction is bounded, and that the final MPO readout fits the reported error. A detached segment remains a conditional statement; a valid overall certificate requires the complete anchored chain. SHA-256 protects transport integrity, while arithmetic replay supplies the proof.

The physical statement is an absolute expectation-error bound for the **polar-corrected binary64 gate/Kraus circuit**, under the QC-14 mathematical inequalities. The checker is still unformalized Python. The program assumes proof files remain stable during verification; it hashes the tape when loading each process's mapping. The mapping is read-only to verifier code, while external writers with filesystem permissions may still alter the underlying file. Use an immutable extracted proof directory or content-addressed artifact when adversarial concurrent mutation is in scope.

For CPU memory, a read-only `mmap` shares physical pages through the OS and permits one-event-at-a-time decoding. Linux `fork` additionally shares decoded circuit metadata through copy-on-write. The workers hold mutable exact-integer intermediate states independently, limiting the total memory saved. `spawn` is supported but creates separate Python interpreter and correction objects in each worker.

## Commands

```bash
python -I qc18_tape.py /path/to/qc17/manifest.json --out-dir /tmp/qc18-proof
python -I qc18_checker.py /tmp/qc18-proof/manifest.json --workers 4 --start-method fork
python -I qc18_checker.py /tmp/qc18-proof/manifest.json --workers 2 --start-method spawn
```

The second command is Linux/Unix-focused. The third works on platforms supporting Python process spawning. Both return one JSON object containing `status=PASS` only after every segment has been checked.
