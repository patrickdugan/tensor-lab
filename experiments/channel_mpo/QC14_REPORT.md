# QC-14: end-to-end dyadic audit of backward Heisenberg MPO evolution

**Date:** 2026-11-10. **Implementation:** `qc14_certified.py` (standard Python integers + NumPy). **Status:** one nonvacuous generic 48-qubit, depth-16 certificate; three-channel shallow validations; deeper weak-noise failures retained. Stacked on QC-13.

## Research claim and precise target

QC-12 bounded singular-value truncation residuals in a floating-point simulation but omitted rounding from QR sweeps, two-site tensor contractions, channel applications, and final readout. QC-13 audited stored SVD factors and readout with exact dyadic arithmetic, leaving the intermediate computation unchecked. QC-14 supplies an independent exact-dyadic residual check for **every** executed local operator transformation, and propagates those errors through the entire backward circuit.

This is an **end-to-end a posteriori numerical certificate for a specifically defined physical circuit**. For each finite binary64 gate `U`, the target is its mathematically exact polar unitary `V = U(U†U)^(-1/2)`. For the vertically stacked binary64 Kraus coefficients `W`, the target is the exact isometry `W(W†W)^(-1/2)`, regrouped into Kraus operators. Each exact physical map is CPTP, and its Heisenberg adjoint is a contraction in operator norm. QC-14 bounds the discrepancy between those idealized maps and the literal stored coefficients.

The target thus differs slightly from any unspecified, pre-rounded random Haar gate and from analytic continuous-strength Kraus formulas. The certificate encloses the expected Pauli measurement on the **polar-corrected physical circuit** starting in `|0>^n`. It does not certify the global density matrix, arbitrary observables, or the input library's abstract Haar generator. Python integer arithmetic is assumed exact, and the implementation's proof obligations have not been machine-checked in Lean or an independent formal verifier.

## Method

Represent the Heisenberg operator by MPO cores `A_i[l,ket,bra,r]` with physical index dimension four. At every local update, retain copies of the **old** tensors, perform the existing NumPy operation, and independently evaluate the residual from old tensors to stored new tensors using exact integer arithmetic on binary64 components. The local audit includes:

- Single-site adjoint `sum_k K_k† A_i K_k`, including every complex multiply and sum performed by NumPy.
- Pairwise QR gauge moves: the exact difference between contractions of the two cores before and after the QR/tensordot operations. QR orthogonality is independently checked using exact dyadic Gram residuals.
- Two-site gate and SVD: the exact difference between `U† contract(A_i,A_{i+1}) U` and the contraction of the two **stored** truncated cores. This includes contractions, SVD and storage rounding without making any SVD backward-stability assumption.
- Final computational-basis input readout: an exact integer contraction of all selected `A_i[0,0]` slices, independently enclosing the reported floating-point result.

The fixed input arrays are exact dyadic rationals. Every local residual `r_i` is an upward-rounded upper bound on their exact Frobenius difference. Let `g_i` bound the operator norm of the context contraction outside that local region. Submultiplicativity of Hilbert-space contractions gives a global operator-norm error at most `e_i = g_i r_i`. During QR and channel operations, `g_i` is bounded by products of exact-dyadic Frobenius norm enclosures of untouched cores. For a two-site gate, the left and right cores have been put into mixed-canonical form; checking the exact Gram defects supplies tighter bounds `||Q||_2 <= 1 + ||Q†Q-I||_F` (or its right-canonical analogue). These bounds are multiplied across the untouched context. All positive floating-point sums and products are rounded upwards by converting their operands to `fractions.Fraction` and applying `nextafter` when required.

For either a stored gate or stacked Kraus isometry `W`, compute `d >= ||W†W-I||_F` using exact dyadic arithmetic. If `d<1/2`, the polar correction obeys

\[
\|W - W(W^\dagger W)^{-1/2}\|_2 \le d,
\qquad
\|W\|_2 \le 1+d.
\]

The physical Heisenberg dual-map difference is bounded by `alpha = d(2+d)` times the input operator norm. For an ideal Pauli backward observable with operator norm at most 1 and an existing enclosure `E`, each physical gate/channel update satisfies

\[
E_{\mathrm{new}} \le E+\alpha(1+E)+e_i.
\]

For a QR gauge change, `alpha=0`. The final independently enclosed readout gap is added to the accumulated `E`, with outward rounding. Invalid, nonfinite or poorly normalized coefficients fail closed; the interface also caps cone width, retained entries, and the number of audited events.

A rigorous **mathematical enclosure** follows from these inequalities, assuming the code implements the stated integer calculations correctly. It is an executable certificate with a small auditable trusted-code base, rather than a formal proof of the program or a universally efficient algorithm.

## Frozen evaluation

Single-threaded NumPy BLAS on the local CPU. All runs use product-zero input, nearest-neighbor brickwork Haar gates, Pauli `Z` target, and the existing QC-12 backward MPO. QC-14 and QC-12 produce the same reported expectations and stored tensors; the difference is independent arithmetic auditing.

| Dataset | Circuits | Bond cap | QC-14 accepted at `error <= 0.05` |
|---|---:|---:|---:|
| `qc14_quick.csv` (10q depth 6 and 12q depth 8, three channels, seeds 3–5) | 18 | 4 / 8 | **13/18** |
| `qc14_scaling.csv` (48q depth 10, three channels, seeds 3 and 5) | 6 | 4 | **0/6** |
| `qc14_deep.csv` (48q depth 16, dephasing `p=0.5`, seed 3) | 1 | 8 | **1/1** |

The independent dense or streaming-cone reference is available for all 18 quick cases, and their observed local errors fall within QC-14's bounds. The reference is itself numerical and serves as a diagnostic; the certificate is derived independently. Those quick bounds range from `0.0001948` to `0.23665`. The six depth-10 bounds range from about `0.40487` to `3.3178`; each is retained as a failed acceptance case.

### Deep 48-qubit, depth-16 example

- Noise: dephasing `p=0.5`; seed `3`; measured `Z_22`; exact backward cone width `32`; cap `8`.
- Floating result: `6.84650994e-9`.
- QC-12 truncation-only ledger: `0.009199062625`.
- QC-14 total end-to-end upper bound: **`0.011832047007`**.
- Independently bounded QR/channel rounding contribution: `0.002632984382`.
- Polar-normalization correction contribution: `2.94e-13`.
- Readout rounding upper: `5.58e-25`.
- Exactly audited local transformations: **3,097**, retained MPO entries at most **7,072**; runtime **25.57 s** on this local CPU, excluding any global dense oracle.

Thus `|<Z_22>_polar-physical - 6.84650994e-9| <= 0.011832047007`, conditional on exact integer execution and the code's proof obligations. The bound is deliberately conservative and guarantees only this nominated expectation value. In particular, the small reported value is consistent with many possible true values inside that interval.

## Independent checks

The newly added tests compare 3-qubit generic circuits against the exact `Fraction` reference from QC-13, including all three noise channels and several bond ranks. A 10-qubit, depth-6 case compares the bound to QC-10's independently evolved six-qubit cone. Exact-dyadic Clifford gates have also been compared with QC-13's symbolic reference at 48 qubits; errors near `3.5e-15` fall below bounds near `3e-6` for the tested extreme dephasing strengths.

A deliberately corrupted QR factor increases the certified bound in the regression test. Another test uses an intentionally nonunitary stored two-qubit matrix: its polar-completed physical target has expectation exactly 1, and the computed error is enclosed by the coefficient-correction bound. These adversarial checks help detect accidental reliance on numerical stability or a silent assumption of exact unitarity.

**Known limitations:** large exact-dyadic audits cost CPU time and may grow expensive with bond dimension; the summed bound often saturates under weak noise or amplitude damping; the target's polar completion is not an assertion about a continuous random gate before rounding. The 48-qubit result has no independent exact dense-oracle cross-check. No global-state trace-distance claim or machine-checked implementation proof is made.

## Reproduction

```bash
cd experiments/channel_mpo
python -m pip install 'numpy>=1.24,<3'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest -q test_channel_mpo test_purified_mpo test_qc03 test_qc04 \
  test_qc05 test_qc06 test_qc07 test_qc08 test_qc09 test_qc10 test_qc11 \
  test_qc12 test_qc13 test_qc14
python benchmark_qc14.py --grid quick --output /tmp/qc14_quick.csv
python benchmark_qc14.py --grid scaling --output /tmp/qc14_scaling.csv
python benchmark_qc14.py --grid deep --output /tmp/qc14_deep.csv
```

## Next step

Introduce a compact proof-certificate format containing exact square numerators and Gram residuals so a separate, smaller checker can independently replay every logged inequality. That would narrow the trusted code surface and allow interval auditing of a physically specified gate-generation process before its binary64 rounding.
