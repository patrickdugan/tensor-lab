# HAD-QFT-QPU01: owner-run IBM pilot

This is a prospective exploratory hardware comparison, not a claim of general QFT acceleration or advantage. All calculations here were CPU-only. Use the saved IBM account on the machine used for QC23/QC26. No credential goes into the repository. The owner must authorize the exact compiled plan; no job is submitted automatically, and a started marker prevents retries after ambiguous errors.

The first stage b=3 represents 128 positions in the unit-invariant shell space; b=4 represents 32768. Compare old Gray shell circuit plus binary label conversion with new recursive binary shell circuit, both in the same logical coordinates. Native old core CNOT counts before conversion are 28 and 120. Common-binary counts are 32/28 and 126/70. The explicit nearest-neighbor bridge costs are 62/52 and 368/200 before IBM ISA translation; they are not measured native gate counts. Fixed deterministic shuffle; same calibration-selected physical line, optimization level zero; no adaptive selection on outcomes.

30 circuits for b=3 and 38 for b=4, each 384 shots. Four input probes (first and last basis vectors, real and imaginary two-shell superpositions); Z and X/Y readout for each arm; two coherent inverse echoes per arm, prep-only echoes, full basis SPAM controls. Primary descriptive metric paired forward TV error (mean gray minus recursive); secondary Hellinger fidelity and echo return. No error mitigation or hypothesis-test claim. A single hardware run cannot establish fidelity or resource superiority across calibrations.

Local NumPy validation: all 68 logical circuits agree with analytic shell transform, also after explicit line routing. All 18 directed bridge cases across b3,b4 verified (6+12 distinct directed pairs); JSON round-trip and synthetic multinomial counts/endian analysis pass. Synthetic samples are not hardware data. Qiskit packages absent locally and install failed on DNS; no live calibration, runtime, transpilation or job result yet. Current remaining allocation unverified.

From repository root, in the existing QC23 Python environment:

```powershell
cd experiments/hadamard_qft/qft01
python qpu01.py qiskit-check --b 3 --out results_local/qft_check3
python qpu01.py build --b 3 --backend ibm_kingston --out results_local/qft3
```

Build is read-only at IBM and checks statevector agreement, calibration, fixed layout, hashes and available usage. Estimated 9.032 s (b3) or 10.9072 s (b4) with default duration allowance, including 2 s job overhead and 0.1 s per circuit; use the actual compiled estimate, subject to provider overhead. The estimate is not a hard provider limit. Maximum owner-authorized estimated time is 15 s per stage. Run b=4 separately only after reviewing b=3. IBM usage documentation: https://quantum.cloud.ibm.com/docs/en/ibm-quantum-compute/estimate-job-run-time

After inspecting the fresh compiled manifest and current allocation, the account owner writes `results_local/auth_qft3.json` (UTF-8 without BOM). Substitute the exact printed hash and a ceiling >= compiled estimate and <=15, and personally acknowledge the budget:

```json
{"authorized":true,"backend":"ibm_kingston","plan_sha256":"REPLACE_WITH_EXACT_COMPILED_HASH","max_qpu_seconds":15,"authorized_by":"owner","checked_current_allocation":true,"acknowledge_estimate_not_hard_limit":true}
```

Within 15 minutes, on the authenticated machine:

```powershell
python qpu01.py submit --out results_local/qft3 --authorization results_local/auth_qft3.json
python qpu01.py collect --out results_local/qft3
```

Submit writes the provider job ID immediately; collect retrieves existing job only. If a started marker exists without job ID, inspect the account job history before any manual retry. Never regenerate an authorization to work around a calibration/plan mismatch. Rebuild and obtain fresh authorization instead. Save compiled.json, calibration snapshots, QASM/QPY hashes, raw counts, analysis.json, execution receipt and usage before/after; exclude authorization, account secrets, and local configuration from public commits. Paper remains a working draft; hardware results must be added only after receipt validation.
