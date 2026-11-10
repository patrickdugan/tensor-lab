# QC-22 sealed event-tape transport

The certificate schema remains **`qc18-mmap-checkpointed-full-witness-v1`**.
The event format, state-boundary commitments, exact dyadic residual checks, and
reported physical-circuit error bound are unchanged. QC-22 is a **verifier
transport mode**, applied to the event-tape file before any arithmetic replay.

## Kernel-sealed event snapshot

On Linux, QC-22 opens the original QC-18 tape through its existing hardened
reader. That reader validates the canonical event frame table, SHA-256 tape
checksum, and each segment digest. QC-22 then copies from the **already-open
source file descriptor** into an anonymous Linux `memfd` using bounded
`pread`/`write` chunks. It hashes the copy as it is written and requires the
hash to equal the tape SHA-256 in the loaded manifest. It applies these seals:

* `F_SEAL_WRITE`: reject writes and new shared writable mappings.
* `F_SEAL_GROW` and `F_SEAL_SHRINK`: reject changes to file length.
* `F_SEAL_SEAL`: prevent subsequent seal-set modification.

The verifier requires `F_GET_SEALS` to show all four bits, maps the memfd
read-only, rehashes the **sealed** mapping, and independently reconstructs the
frame table and segment digests from those sealed bytes. It discards the old
file-backed mapping and handle. Forked worker processes inherit the sealed
mapping and descriptor and re-check the required bits before replay.

This transport is strictly Linux-specific. QC-22 permits `fork` worker startup
only. Unsupported kernels, seal failures, absent bytes, digest mismatches and
wrong replay modes fail closed. It deliberately does **not** fall back to an
unsealed file-backed mapping.

## Other proof inputs

The circuit, segment descriptors, and result live in the validated manifest
object held in memory. Each endpoint checkpoint has a manifest-pinned SHA-256,
and the existing QC-20 reader decodes **the same bytes it hashes** into an
independent boundary state. Every checkpoint is checked at least once by its
adjacent segment, and genesis/final boundary checks are independently repeated
by the coordinator. Mutation of a checkpoint after a successful read may cause
a subsequent replay failure; it cannot silently change accepted checkpoint
arithmetic without matching the pinned hash.

## Explicit limitations

The security model assumes SHA-256 collision resistance and a correct Linux
file-seal implementation. A compromised kernel, malicious verifier code,
untrusted circuit model, or hash collision is outside this claim. The
memfd snapshot is a live-process ephemeral proof input, **not a persistent
sealed proof artifact**. A privileged process able to write the verifier's
address space is outside the model. Forking from a multithreaded process has
usual POSIX caveats; run the supplied standalone `python -I -S` CLI in a fresh
single-threaded process. The independent mathematical checker remains the
QC-15 to QC-21 exact dyadic algorithm. Neither the transport nor checksum
turns the checker into a formally proved program.
