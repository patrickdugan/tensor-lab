# QC-23 | IBM Quantum Clifford echo pilot

**Status:** executable hardware submission harness; **no IBM QPU job submitted**. The IBM account must be configured locally before `build` or `submit`.

## Research question

Can the QC-13/14/15–22 observable-certification apparatus be carried over to real IBM hardware, with a clean separation between (1) circuit-model truncation error, (2) sampling uncertainty, and (3) model–hardware discrepancy?

Start with an ideal identity Clifford echo: forward layers of single-qubit Clifford operations and nearest-neighbor CX gates, followed by the exact inverse. The ideal bitstring is always `0` on every qubit. Any nonzero outcome is physical execution/readout error. This is a calibration/verification **entry point**, not evidence that IBM noise follows a given depolarizing model.

## Experiment

- Logical widths 8, 16, 32, 48. Depth parameters 2, 4, 8; each echo has twice the forward two-qubit layers.
- Fixed seeds 3, 5, 7; 2048 shots per circuit initially. Start with 8 and 16 qubits to fit a free Open Plan.
- Measure all qubits; report Z at left edge, right edge, and middle, including binomial standard error.
- Fit channel parameters on separate circuits, then reserve held-out circuit seeds for testing. Never treat effective gate noise as independently calibrated merely from the same echo data.
- Preserve actual physical qubit path and compiled QPY to permit comparison with calibration data, native-gate topology, and duration.
- Transpilation level 0 is intentional: optimizing a Clifford echo can cancel the forward/inverse gates and remove the test. Check compiled two-qubit operations before spending QPU time.

## Reproduce

```bash
python -m unittest -v test_qc23.py
python qc23_ibm.py plan --out ibm_plan
python -m pip install 'qiskit>=2.2,<3' 'qiskit-ibm-runtime>=0.50,<1'
# An IBM Quantum Cloud account with a locally configured Qiskit Runtime instance is required.
python qc23_ibm.py build --backend YOUR_ACCESSIBLE_QPU --out ibm_build
# Explicit hardware submission, which can use billable QPU time:
python qc23_ibm.py submit --backend YOUR_ACCESSIBLE_QPU --out ibm_run --submit --allow-paid
```

All non-run invocations avoid QPU consumption. `build` contacts IBM for backend metadata. `submit` requires a separate explicit authorization switch for possible charges. Results include job ID and raw counts, with no IBM credential serialized.

## Acceptance and analysis

Baseline reproducibility: the ideal-state prediction is exactly `Z_i=+1`; compare measured `Z_i` with `+1`, report standard errors and readout-calibrated results as separately labeled outputs. A model-based noise prediction should be written to a separate JSON and fixed before the first held-out job. For sampled expectation `m`, a conservative non-asymptotic 95% Hoeffding half-width is `sqrt(2 log(40)/shots)` for a single ±1 observable (union-adjust for multiple queries). Do not attach QC-14's numerical error bound to hardware discrepancy; the bound covers its specified polar-corrected model only.

A future follow-up will compile the **same physical ISA gates** into the QC-13 Pauli-transfer oracle and apply hardware-derived per-gate noise channels, with held-out predictions fixed before execution. That step requires accurate native gate and noise data.

## Limitations

The launch gate currently checks only CLI consent, not the IBM instance's billing plan; users should inspect the selected instance in the IBM dashboard. `choose_line` searches topology, not best-calibrated edges, and large paths may require manual selection. Device availability and queue time vary. Hardware shots have statistical error and device drift; the QC-12 to QC-22 rigorous certificate covers a precisely specified simulated channel, not an arbitrary physical QPU.

Current IBM documentation: https://quantum.cloud.ibm.com/docs/en/guides/get-started-with-sampler ; https://quantum.cloud.ibm.com/docs/en/guides/plans-overview
