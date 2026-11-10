# QC-16: Fast independently verified segmented tensor-network certificates

**Date:** 2026-11-10  
**Repository:** `patrickdugan/tensor-lab`, stacked on QC-15 PR #17  
**Evidence:** 185 local regression tests; independent 48-qubit segmented full-witness replay; exact cross-backend comparison; frozen small-circuit process benchmarks.

## Research question

QC-15 could independently replay a 48-qubit, depth-16 Heisenberg-MPO witness, including 3,097 audited local updates, but a single 12 MB compressed proof took approximately **390 seconds** and peaked at **205,048 KiB** RSS. Can an equally independent checker use exact dyadic integer arithmetic and segment state commitments to reduce verification cost without weakening the error statement?

**Result:** yes, for the tested full witness. The new checker verifies the same 3,097 updates and exact final error bound of **0.011832047006821769** in **34.57 seconds** and **124,164 KiB** peak RSS on the current CPU. This is **11.3× faster** and roughly **39.5% less peak process memory** than the previously recorded QC-15 result. The prior 390-second baseline was from the earlier QC-15 run, rather than a same-process paired benchmark. Every arithmetic result on the smaller paired cases exactly matches the QC-15 rational checker.

This result bounds the selected Pauli expectation for the QC-14 **polar-corrected binary64 circuit**; it is not a guarantee for arbitrary uncorrected Haar matrices or a full-state trace-distance bound. The separate checker still requires code review or formal verification before treating its implementation as trustworthy.

## Arithmetic

QC-15's original contraction loops construct and reduce a `fractions.Fraction` for nearly every binary64 multiply and sum. QC-16 maps each tensor's complex coefficients to a shared integer denominator `2^E`, performs the complex multiplications and accumulations over integers, then converts the **final squared residual** back to a rational. The checker also accelerates channel contraction, Gram defects, Frobenius context norms and the exact product-input readout. A bounded LRU avoids converting the same immutable core repeatedly.

For tensor coefficients represented as `a_k / 2^A` and `b_k / 2^B`, the exact contraction is

\[
 \sum_k \frac{a_k}{2^A}\frac{b_k}{2^B}
 =\frac{\sum_k a_k b_k}{2^{A+B}}.
\]

With the real and imaginary integer residuals `r_j` and `s_j` at common denominator `2^Q`,

\[
  \|\Delta\|_F^2=\frac{\sum_j(r_j^2+s_j^2)}{2^{2Q}}.
\]

All operations use arbitrary-precision integers. The original QC-15 checker remains the arithmetic reference, selected by `--reference`. In the differential test suite, the final expectation, accumulated bound and event counts are **exactly equal**, rather than agreeing only up to floating tolerance.

## Segmented proof design

The full trace is stored as an ordered manifest plus compressed segment files. Every segment commits to its input and output **actual MPO state**, physical support coordinates, operation counts and accumulated rational error. On replay, QC-16 rechecks all transitions and independently reconstructs the commitments at every segment boundary. It rejects gaps, duplicate segments, changed cores, invalid event arithmetic and regenerated-checksum forgeries.

The 48-qubit full witness contains 3,113 records when the 16 support-growth events are included. It was split into 13 segments of at most 256 records. The segment files range from roughly 0.23 to 1.22 MB compressed, and the manifest adds roughly 150 KB of circuit metadata. The splitter took **3.06 seconds** to emit the set, separate from verification. The independent checker confirmed **14 boundaries** (including genesis and final state), **3,097 audited transformations**, 1,206,052 changed complex MPO entries, and the same final error bound.

The compact QC-15 conditional ledger remains a distinct, weaker format. QC-16's verifier refuses it. Full semantic verification requires every segment from genesis; a detached middle segment's checksum cannot by itself certify a circuit.

## Fresh-process small-circuit benchmarks

Single threaded OpenBLAS, one fresh Python process per arithmetic backend. Times include proof reading and checking. The reference and dyadic backends consume the **same full QC-15 proof**; segmented verification additionally checks boundary commitments.

| Qubits / depth | Audited updates | Fraction ref (s) | Dyadic ref (s) | Segmented (s) | Dyadic speedup |
|:--|--:|--:|--:|--:|--:|
| 3 / 3 | 11 | 0.0569 | 0.00681 | 0.00729 | 8.36× |
| 10 / 6 | 48 | 0.4718 | 0.1173 | 0.1187 | 4.02× |
| 12 / 8 | 107 | 4.5699 | 0.4228 | 0.4527 | 10.81× |

Input full proofs are 4.4, 35.3 and 184.0 KB. Their segmented versions are 5.7, 43.6 and 202.9 KB respectively, reflecting boundary and per-file overhead. For the largest deep witness, segmentation enables independent chunk loading while preserving event continuity.

## Deep-circuit verification record

- Source: the QC-15 compressed full witness (`qc15_full_48qubit_witness.json.gz`), SHA-256 `3bbd436120fee57677fcab1d970e60802c523ad9817e955ab2016ab9f14e8889`.
- Configuration: 48 physical qubits, 16 layers, central Pauli observable, dephasing `p=0.5`, bond cap 8.
- Segments: 13, with 14 independently checked state-boundary commitments.
- Audited local transformations: 3,097 (`channel` 241, `gate_svd` 136, `left_qr` 1360, `right_qr` 1360).
- Full-witness changed complex entries: 1,206,052.
- Reported expectation: `6.8465099397700234e-09`.
- Independently verified error bound: **`0.011832047006821769`**.
- Verification elapsed time: **34.57 seconds**. Maximum resident set: **124,164 KiB**, from `/usr/bin/time -v` in a fresh process.
- Manifest SHA-256: `dc6b3ed7a794c508846e78d66acffa0d45bb15f2af729429d56cfe1b1fb0a7e6`.

A portable archive of the 48-qubit segmented proof is available alongside the PR's research packet. The 12 MB witness files are deliberately **not committed** to the source repository.

## Security regressions and limitations

The 15 new tests exercise exact agreement with the Fraction backend, multi-channel witness replay, unequal per-tensor exponents, complex coefficients, subnormal binary64 entries, safe path handling, missing segments, disconnected boundaries, forged boundary digests with repaired manifest and segment checksums, changed residuals with all commitments recomputed, reordered events, standard-library isolated CLI execution, and refusal of conditional-ledger proofs.

The complete QC-01 to QC-16 local regression suite is **185/185 passing**. The mathematical statements remain conditional on the certified circuit construction and the correctness of both checker code and underlying arithmetic/norm inequalities. SHA-256 is an integrity mechanism, not a public-key authentication mechanism. A future compact proof system could replace the full operand witness while retaining independently checkable transition claims.

## Reproduce

```bash
cd experiments/channel_mpo
python -m pip install 'numpy>=1.24,<3'     # needed for the producer / full regression suite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 \
  test_qc04 test_qc05 test_qc06 test_qc07 test_qc08 test_qc09 \
  test_qc10 test_qc11 test_qc12 test_qc13 test_qc14 test_qc15 test_qc16
python qc16_segments.py qc15_certificates/full_10q_d6_dephasing_s3_c4.json.gz \
  --out-dir /tmp/qc16 --events-per-segment 16
python -I qc16_checker.py /tmp/qc16/manifest.json
python -I qc16_checker.py /tmp/qc16/manifest.json --reference
OPENBLAS_NUM_THREADS=1 python benchmark_qc16.py --out-dir /tmp/qc16_bench
```

The independent checker requires only Python ≥3.10, with no NumPy or scientific libraries. A future experiment could isolate the checker into a tiny Rust binary or formalize the transition invariants in Lean.
