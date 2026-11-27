# HAD-QFT-QPU01 hardware record (27 November 2026)

Backend `ibm_kingston`, Qiskit 2.6.2, qiskit-ibm-runtime 0.52.0, Python 3.12.10. Both stages used 384 shots per circuit, optimization level 0, Sampler V2, `rep_delay` 0.00025 s, and the complete registered roster (30 circuits at b=3, 38 at b=4). The device calibration (27 Nov 2026 11:57:17 UTC) was unchanged between build and submission for both stages.

| Stage | Directory | Job ID | Plan SHA-256 | Chain | Estimate | IBM charge |
|---|---|---|---|---|---|---|
| A, b=3 | `stageA_b3` | `dc4n8nmq4uuc73a94qa0` | `927978a9083fd3feff75bc3d57b96ccce0805b605746922555167670d90c192c` | 38, 49, 48 | 9.032 s | 5 s |
| B, b=4 | `stageB_b4` | `dc4nal6q4uuc73a94sm0` | `c4a9933037ea04594ec167be9077819ba3829156743824cd242991266f2ebab8` | 148, 149, 150, 151 | 10.907 s | 6 s |

QPU seconds consumed by the two jobs: 11 of the 30 reserved. Account usage counter: 224 s before Stage A, 246 s after Stage B (`usage_final.json`: 246 consumed, 354 remaining of 600). The counter moved 22 s over the campaign; the 11 s not charged to these jobs came from other activity on the account.

Each stage directory holds `compiled.json` (compiled plan, per-circuit ISA counts and duration), `compiled.qpy` and `circuits/*.qasm`, `calibration.json` (build) and `calibration_submission.json` (re-read at submit; identical), `counts.json` (raw counts, unmodified), `job.json`, `submission_started.json`, `usage_*.json`, `execution_receipt.json`, `analysis.json` (basic summary written by `qpu01.py`) and `analysis_full.json` (all metrics from `analyze_qpu01.py`). Authorization files and account configuration are not committed.

## Reproduce the analysis (CPU only, no account needed)

```
cd experiments/hadamard_qft/qft01
python -m pytest test_qpu01.py -q
python analyze_qpu01.py --stage hardware_results/stageA_b3 --stage hardware_results/stageB_b4 \
    --figdir manuscript/figures --tables manuscript/tables
cd manuscript && pdflatex paper.tex && pdflatex paper.tex
```

Bootstrap seeds are fixed (`20261127 + b`), so `analysis_full.json` regenerates identically. Re-running hardware needs the saved account and a fresh owner authorization of the printed plan hash (see `QPU01_RUNBOOK.md`); never reuse an authorization after a rebuild.

## Corrections and deviations

- Circuit definitions, routing and mapping logic in `qpu01.py` were not changed. `qiskit-check` passed for b=3 and b=4 and the offline tests passed before any hardware use (4 passed after adding the analysis test).
- `analyze_qpu01.py` was added: Hellinger fidelity, ideal-sampling floors, uniform-noise reference, paired bootstrap differences, parity witness, Wilson echo intervals, SPAM-corrected sensitivity block (secondary, not used in the paper), LaTeX tables and figures. A first draft treated raw counts as probabilities; fixed before any hardware data existed.
- `owner_submit.ps1` wrote the authorization file and ran submit and collect. In its first version a Python stderr warning was treated as a failure, so the Stage A run reported an error after the job had been submitted. `job.json` showed the submission, no resubmission was made, and `collect` was run alone. The script now judges success by exit code and `job.json`. In Stage B its post-submit collect check also misreported an exit code; `collect` was rerun alone and re-read the same job, producing identical raw counts. Neither stage submitted more than once (one `submission_started.json` and one `job.json` each).
- Stage A and Stage B authorizations were given in chat by the account owner against the exact plan hash and were written to `results_local/auth_qft3.json` and `auth_qft4.json` by the script (untracked).
- The Stage A plan was built twice; the first build (hash `c1989ea8…`, chain 48, 49, 50) was discarded unsubmitted because the 15-minute window would have lapsed.
