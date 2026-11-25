# QC-15: Portable, independently replayed quantum-channel error certificates

**Date:** 2026-11-10. **Status:** 12 small-system full witnesses and one 48-qubit full witness independently verified; conditional deep ledger also checked. Stacked on QC-14. Standard-library checker in `qc15_checker.py`, simulator-facing emitter in `qc15_emit.py`.

## Research question

Can QC-14's floating-point-independent local residuals and rigorous accumulated inequality be moved out of the simulation implementation, so a second program checks each claimed operation? QC-15 defines a versioned proof witness that gives the checker the *actual binary64 MPO core changes*. This makes it possible to verify local transition errors, mixed-canonical external-context norms, and polar corrections without trusting QR/SVD implementations or importing the simulator.

The checker uses only the Python standard library. It reads float64 components as exactly specified dyadic rationals with `fractions.Fraction`. On a full witness, it computes the exact rational squared residual of every channel, QR or two-site-gate/SVD operation; reconstructs the cone expansion and event schedule; independently bounds the untouched MPO context using Frobenius and Gram defects; checks the matrix polar corrections; then recomputes the final all-zero product-state readout and error recurrence.

The physical target is inherited from QC-14: the mathematical polar completion of the *stored* binary64 gate matrices and Kraus-isometry coefficients. Neither the upstream ideal Haar sampling distribution nor any pre-rounded analog channel is part of this claim. The certificate concerns one nominated Pauli expectation, not the global state.

## Two evidence levels

1. **Full operand witness:** `qc15-full-witness-v1`, `--require-witness` accepted. Each record encodes the changed MPO cores in contiguous binary64, with the previous state reconstructed from all earlier records. A separate checker confirms every provided local residual and the complete backward schedule. A tampered transcript must still satisfy all exact checks to pass. Only the checker and its mathematical inequalities remain trusted.
2. **Conditional scalar ledger:** `qc15-conditional-ledger-v1`. It recomputes the exact rational accumulation and compares a *claimed* exact-dyadic final readout with a claimed reported expectation. Local residual and context bounds, circuit linkage, and the readout itself remain **producer assertions**. The checker labels the output `conditional-ledger-only`, and `--require-witness` rejects it. This compact format supports easy comparison with QC-14's 3,097-event deep-circuit ledger without implying independent local correctness.

The JSON `sha256` field is for accidental corruption detection and deterministic replay. It is not a signature or a substitute for mathematical verification. Binary64 coefficients are encoded as base64 of little-endian complex pairs; `error_upper`, `polar_upper`, `residual_upper` and `context_upper` are exact IEEE-754 hexadecimal strings. Gzip is deterministic, with mtime 0.

## Mathematical verification

For each local event, let \(A,B\) denote the old cores and \(C,D\) their replacements. The checker evaluates, by exact rationals on the stored values,

\[
\begin{aligned}
 r_{\rm QR}^2&\ge\|AB-CD\|_F^2,\\
 r_{\rm gate}^2&\ge\|U^\dagger (AB)U-CD\|_F^2,\\
 r_{\rm channel}^2&\ge\|\sum_k K_k^\dagger A K_k-C\|_F^2.
\end{aligned}
\]

The external norm \(g_j\) is checked against product Frobenius bounds outside a local update. During a gate/SVD cut, the checker independently evaluates each left and right canonical core's Gram defect \(\delta_i\), using \(1+\delta_i\) as a conservative upper bound on its norm. To avoid a floating square root, it rounds exact rational square roots *upward* at 160 binary fractional bits.

With a supplied conservative polar dual correction \(\alpha_j\) computed from \(\|U^\dagger U-I\|_F\) or \(\|W^\dagger W-I\|_F\), the checker requires \(\alpha_j\ge d_j(2+d_j)\) for an independently upper-rounded Gram defect \(d_j<1/2\). Every QR event has \(\alpha=0\). It then evaluates the exact rational recurrence

\[
E_0=0,\qquad E_{j+1}=E_j+r_jg_j+\alpha_j(1+E_j).
\]

Finally it contracts \(\widetilde O\) exactly and checks

\[
\varepsilon_{\rm reported}\ge E_m+\left|\operatorname{Re}\langle0|\widetilde O|0\rangle-\widehat\mu\right|.
\]

This validates the accumulation independently of the original floating implementation. The proof uses operator-norm contractivity of the adjoints of the polar-completed CPTP maps, exact identity insertion when the cone grows, and submultiplicativity of MPO context bounds. The checker is separately implemented but has **not been formally proved correct** in Lean or another proof assistant.

## Frozen full-witness benchmarks

All 12 circuits passed exact residual replay. Locally measured timings with Python 3.13 and NumPy 2.3.5 for the producer, with one-thread BLAS. Runtime excludes loading Python startup unless noted.

| Qubits | Depth | Channel/seed coverage | Audited local operations per run | Compressed witness | Checker CPU time |
|---|---|---|---:|---:|---:|
| 3 | 3 | all 3 channels, seeds 3 & 5 | 11 | 4.4–8.0 KB | 0.04–0.08 s |
| 10 | 6 | all 3 channels, seed 3 | 48 | 35.3–50.1 KB | 0.43–0.65 s |
| 12 | 8 | all 3 channels, seed 5 | 107 | 184–260 KB | 4.08–5.20 s |

The 10-qubit depth-six dephasing case reports \(\widehat\mu=0.01508965327\), with validated bound \(\varepsilon=0.01688419852\). Its full operand transcript contains 3,128 new-core complex values across 48 audited operations.

### Deep 48-qubit full witness and scalar ledger

At 48 qubits, depth 16, dephasing \(p=0.5\), seed 3, measured \(Z_{22}\), cap 8, the full witness contains **3,097 checked transformations**: 241 channel updates, 136 gate/SVD updates, and 1,360 each of left and right QR gauge moves. The checker independently recomputed their exact local squared residuals, external-core norms/isometries, polar corrections, event order, and readout. Its *verified* upper bound on the polar-corrected circuit's nominated Pauli expectation is **0.011832047006821769**, with reported expectation **6.8465099397700234e-9**. All checks passed.

The full witness is **12,072,836 bytes compressed** (26,663,922 bytes uncompressed), holding **1,206,052 changed-core complex coefficients**. Independent replay took **389.96 s wall time**, with **390.17 s user CPU** and **205,048 KiB peak RSS** measured by GNU `/usr/bin/time -v`. This is a correctness demonstration with substantial verification cost. It is not yet practical for every deep-circuit output.

The corresponding **conditional scalar ledger** records the same 3,097 event claims in **45,063 bytes**, checks its accumulated inequalities in **0.45 s**, and reproduces the same QC-14 error bound. Its smaller size and faster replay come from trusting the producer's local residual and context claims. The two files are distinct evidence levels, and the checker explicitly labels them.

Full witness SHA-256: `3bbd436120fee57677fcab1d970e60802c523ad9817e955ab2016ab9f14e8889`. Regeneration uses the same fixed circuit, seed, and versioned binary64 format.

## Adversarial validation and strict rejection

The new tests include fabricated local residuals, zeroed context norms, suppressed gate polar corrections, forged final bounds, altered tensor coefficients, altered gates, missing events, reordered events, invalid cone widths, incorrect gate locations, incorrect SHA-256 digests, forbidden ledger-level acceptance under `--require-witness`, and a deliberately nonunitary gate whose polar correction is essential. All pass as negative tests: every tampered claim is rejected.

The checker can run with `python -I qc15_checker.py`, which excludes user-site imports. It has no dependencies on NumPy, QC-12, QC-13, QC-14, or other simulator modules. The emitter retains those dependencies and is outside the verifier's trusted code base.

## Reproduction

```bash
cd experiments/channel_mpo
python -m pip install 'numpy>=1.24,<3'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 test_qc04 \
  test_qc05 test_qc06 test_qc07 test_qc08 test_qc09 test_qc10 \
  test_qc11 test_qc12 test_qc13 test_qc14 test_qc15
python benchmark_qc15.py --grid small --out-dir /tmp/qc15
python -I qc15_checker.py /tmp/qc15/full_10q_d6_dephasing_s3_c4.json.gz --require-witness
python benchmark_qc15.py --grid deep --out-dir /tmp/qc15  # 48q conditional ledger
```

For the binary format and trust boundary, see `QC15_FORMAT.md`.

## Limitations and next experiment

Full witnesses are larger than scalar ledgers, and exact-rational replay is substantially slower than producer-side NumPy: approximately 390 seconds for the 48-qubit example. The trusted checker itself is not machine-checked; its security relies on exact Python arithmetic, the specified operator inequalities, and input-shape/resource validation. Malicious proof producers can alter their *claimed circuit*, so the original client must bind the circuit specification to the desired task. Signature/authentication and network transport are outside QC-15.

QC-16 should specialize the exact-rational verifier to common-denominator integer kernels or a Lean-checkable arithmetic certificate, then add authenticated boundary commitments for segmented proofs. The difficulty to solve is large-scale full-witness verification within a practical CPU budget while retaining the local residual and context checks.
