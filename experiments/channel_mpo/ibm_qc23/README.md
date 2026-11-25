# QC-23 | IBM Quantum hardware validation

**Status:** the full pipeline (compile, mapping audit, stabilizer proofs, noise models, MPO prediction, held-out analysis) runs offline end to end. **No IBM QPU job has been submitted and no hardware measurement exists in this directory.** The machine this was built on has no saved IBM account (`AccountNotFoundError`, recorded in [`results/hw_discovery_error.txt`](results/hw_discovery_error.txt)). Everything labelled `surrogate` is a synthetic simulator, never hardware.

Read [`QC23_REPORT.md`](QC23_REPORT.md) for results and limitations.

## Layout

| File | Purpose |
|---|---|
| `qc23_ibm.py` | CLI: `plan`, `discover`, `build`, `submit`; echo generator, Z estimators with Wilson/Hoeffding intervals |
| `qc23_access.py` | account discovery (no secrets serialised) and the hardware-authorization gate |
| `qc23_compile.py` | calibration-aware path selection, level-0 transpilation, logical-to-physical mapping audit, QPY/OpenQASM 3 |
| `qc23_stabilizer.py` | independent stim proofs (schedule and transpiled ISA are the identity), exact Pauli-fault propagation |
| `qc23_noise.py` | Model A (calibration-derived) and Model B (fitted effective channel) |
| `qc23_mpo.py` | compiled ISA gates to QC-12/14 layers; exact layered Pauli reference |
| `qc23_nonclifford.py` | kicked-Ising family and dense ideal reference |
| `qc23_surrogate.py` | synthetic out-of-model QPU surrogate (pipeline exerciser only) |
| `qc23_experiment.py` | circuit sets, compile and simulate drivers |
| `qc23_analysis.py` | fit on the calibration subset, evaluate held-out circuits, threshold statistics, plots |
| `results/` | compiled artifacts, counts, analysis JSON/CSV, plots |

## Reproduce offline

```bash
python -m pip install -r requirements-qc23.txt        # exact pins; requirements-qc23-full.freeze is the full freeze
python -m unittest -v test_qc23 test_qc23_pipeline
python qc23_ibm.py plan --out /tmp/plan
python qc23_ibm.py build --fixture FakeFez --out /tmp/build        # compile 12 echoes on a bundled snapshot
python qc23_analysis.py --root results --source surrogate --out-dir /tmp/replay
python qc23_verify_replay.py results/analysis_surrogate.json /tmp/replay/analysis_surrogate.json
```

## Run on IBM hardware

1. Register the account yourself so the token never enters a log or this repo. For a key saved in a file:

   ```bash
   python -c "import pathlib,sys; from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(token=pathlib.Path(sys.argv[1]).read_text().strip(), channel='ibm_quantum_platform', overwrite=True)" PATH_TO_KEY_FILE
   ```

   Add `instance='<CRN>'` to `save_account` if the account has several instances.
2. Discover: `python qc23_ibm.py discover --out results/hw_discovery` (backends, coupling maps, basis gates, calibration medians, queue depth, plan usage when the API exposes it).
3. Compile the plan on the chosen backend: `python qc23_ibm.py build --backend BACKEND --out results/ibm_build`. This prints `plan_sha256` in `compiled.json`.
4. Execution needs an **authorization file written by the account owner**. `--allow-paid` is not enough. Create `authorization.json`:

   ```json
   {"authorized": true, "backend": "BACKEND", "plan_sha256": "<from compiled.json>",
    "max_qpu_seconds": 120, "authorized_by": "your name"}
   ```

   then `python qc23_ibm.py submit --backend BACKEND --out results/ibm_run --submit --authorization authorization.json`. Submission is refused if the backend, hash or estimated QPU seconds do not match.
5. Run the calibration subset (`fit_echo`) **before** the evaluation sets and write the counts to `results/ibm/<set>_counts.json` in the same schema as `results/surrogate/`; then `python qc23_analysis.py --source ibm`.

## Pre-registered choices (fixed before any hardware job)

* Evaluation echoes: n in {8, 16}, depth in {2, 4, 8}, seeds {3, 5}, 2048 shots, observables Z at left edge, right edge, centre.
* Model B is fitted only on `fit_echo` (seeds 11, 13, depths 1, 3, 6). Evaluation circuits and seeds are never used for fitting.
* Accuracy threshold 0.05 on |prediction - measured Z|, reported as a Wilson interval over (circuit, observable) pairs and a cluster bootstrap over circuits.
* Optimization level 0 so that echo gates are not cancelled by the compiler.
