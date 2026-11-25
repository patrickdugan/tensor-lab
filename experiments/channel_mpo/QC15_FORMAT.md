# QC15-WITNESS-v1 certificate format

## Scope

The `qc15-full-witness-v1` format records the *literal stored binary64* inputs and outputs of every local transformation in a backward Heisenberg operator-MPO. A separate verifier written with Python's standard library reconstructs the circuit's causal-cone event order, verifies that a candidate transition changes exactly the designated MPO cores, and computes the local residual **as an exact rational**. It checks the Frobenius/isometry bounds on the untouched context, the exact Gram defects behind polar-normalization allowances, and the accumulated error inequality. For the final all-zero product input, it evaluates the MPO readout by an exact rational contraction.

The mathematical target is the polar-completed physical circuit from QC-14: each stored 4×4 gate `U` denotes `U(U†U)^{-1/2}`, and the vertically stacked stored Kraus coefficients `W` denote `W(W†W)^{-1/2}`. These polars are defined mathematically, including when the stored binary64 matrices are slightly nonunitary. Operators with Frobenius Gram defect ≥ 1/2 are rejected. The output is a bound on a *nominated Pauli expectation*, not a global-state fidelity guarantee.

Independent verification still assumes correct implementation of the small verifier, exact integer/rational operations, and the stated operator-norm inequalities. This is an executable witness checker, **not a formal proof of its implementation**.

## Container and content

The `.json.gz` is UTF-8 JSON compressed with gzip (deterministic mtime 0). The `sha256` field is SHA-256 of the canonical JSON object **without** `sha256`, using sorted keys, ASCII escaping, and compact separators. This checksum detects accidental corruption; it is not an authentication signature. The verifier recomputes it before using the record.

A top-level full witness has `format`, `circuit`, `events`, `result` and the checksum. The circuit describes `qubits`, contiguous `targets`, the target `pauli` word, nearest-neighbor gate `layers`, and single-site `kraus` matrices. A matrix or core is an object of shape integers and `binary64_le`: RFC 4648 base64 of consecutive little-endian IEEE-754 double pairs `(real,imag)` in row-major order. Original floating coefficients are interpreted *exactly* as dyadic rationals. Tensor cores have shape `[left_bond,2,2,right_bond]`.

Each transformation event contains `kind` (`channel`, `left_qr`, `right_qr`, or `gate_svd`), its `layer`, physical `site`, indices of exactly the changed cores, new-core binary64 data, and three nonnegative binary64 hexadecimal floats: `residual_upper`, `context_upper`, `polar_upper`. A `grow` record inserts exact identity cores and is checked against the causal cone. The verifier obtains every before-state core from previous records; therefore removing, reordering, replacing, or altering an event breaks the proof or its schedule.

The final `result` contains the reported binary64 `expectation`, `error_upper`, the cone width, and event counts. The verifier contracts the *last actual cores* exactly. Its acceptance condition is

\[
  \varepsilon_{\rm report}\;\ge\; E_m + \bigl|\operatorname{Re} \langle 0|\widetilde O|0\rangle - \widehat{\mu}\bigr|,
\]

where the exact rational recurrence is

\[
 E_{j+1}=E_j+r_j g_j+\alpha_j(1+E_j),\quad E_0=0.
\]

For each event the checker verifies \(r_j^2\ge\|\Delta_j\|_F^2\) from the stored source and target tensors, \(g_j\) against independently computed conservative Frobenius/isometry context norms, and \(\alpha_j\) against the exact stored gate/Kraus Gram defect. A rational upward square-root approximation (160 binary bits) is used only for intermediate norm upper bounds; no unverified BLAS or SVD operations occur in the checker.

**Acceptance proves the stated bound for the specified circuit, conditional on the verifier's implementation and the norm inequalities.** The producer can use arbitrary algorithms to choose its MPO cores. An invalid or incomplete transition is rejected independently of whether it came from QC-14.

## Compact conditional ledger

`qc15-conditional-ledger-v1` is a smaller alternative for thousands of events. It contains the producer's scalar `residual_upper`, `context_upper`, `polar_upper`, exact-dyadic *claimed* final readout numerator (hexadecimal), denominator exponent, and total bound. The independent checker replays the rational recurrence and final bound using these claims. This mode **cannot independently validate the local residuals, context norms, readout numerator or the circuit event schedule**, because it omits operand witnesses. Its result is explicitly `conditional-ledger-only`. Add `--require-witness` to the CLI to make this mode fail closed.

## Limits and failure behavior

The independent checker rejects malformed dimensions, missing/out-of-order events, nonfinite binary64 entries, invalid or understated local residuals or context bounds, missing polar corrections, bad checksums, noncontiguous cones and underestimated final bounds. Limits include ≤64 qubits/layers, ≤5,000 audited events, a 40 MB decompressed JSON cap, and a cumulative 3-million-complex-entry witness budget. Larger genuine traces can be split into independently checked segments by a future format revision, with explicit boundary commitments.

## Reproduce

```bash
cd experiments/channel_mpo
OPENBLAS_NUM_THREADS=1 python qc15_emit.py --qubits 10 --depth 6 --noise dephasing --seed 3 --bond-cap 4 --tier witness --output /tmp/qc15_10q.json.gz
python -I qc15_checker.py /tmp/qc15_10q.json.gz --require-witness
OPENBLAS_NUM_THREADS=1 python benchmark_qc15.py --grid small --out-dir /tmp/qc15_bench
python -m unittest -q test_qc15
```

`qc15_emit.py` and `benchmark_qc15.py` need NumPy and the existing QC-14 modules. `qc15_checker.py` can be copied by itself to a new machine with Python ≥3.10; its imports are entirely from the standard library.
