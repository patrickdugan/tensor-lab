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

# HAD-QFT-QPU02 hardware record (27 November 2026)

Protocol: `../QPU02_RUNBOOK.md` at commit b754b6da117224ba2e543516b5d696c0d770bd26. Backend `ibm_kingston`, Qiskit 2.6.2, qiskit-ibm-runtime 0.52.0, optimization level 1, 512 shots per circuit, Sampler V2, `rep_delay` 0.00025 s. All three jobs ran under one device calibration (27 Nov 2026 12:56:21 UTC), unchanged between build and submission (`calibration.json` equals `calibration_submission.json`).

| Stage | Directory | Job ID | Plan SHA-256 | Chain | Circuits | Estimate | IBM charge |
|---|---|---|---|---|---|---|---|
| b=5, replicate 0 | `qpu02_b5_r0` | `dc4oaqqnj2vs73dkf2kg` | `4c2ef2086e52b80c7f3c761b879d1fc740632f6ec6c25f3f637d6790dde4145c` | 148, 149, 150, 151, 152 | 73 | 22.534 s | 14 s |
| b=5, replicate 1 | `qpu02_b5_r1` | `dc4ock2nj2vs73dkf570` | `13ad51412877f8ea611970be899eb567676acbf7ce37ee850571d8e1f5a6a8ee` | 67, 68, 69, 78, 89 | 73 | 22.534 s | 14 s |
| b=6 stretch | `qpu02_b6_r0` | `dc4oduinj2vs73dkf74g` | `9678e596f7f54be6f0c2261e790923ed82ef91ae96e37049950be8cee3fee672` | 67, 68, 69, 78, 89, 88 | 58 | 18.262 s | 12 s |

QPU seconds charged to the three jobs: 40 (28 of the 60 s reserved for the two b=5 jobs). Account counter at the first QPU02 build: 264 of 600 s consumed; after the last collect: 304 consumed, 296 remaining (`qpu02_b6_r0/usage_after.json`). The counter moved 40 s over the campaign, which equals the three job charges. The 18 s between the QPU01 final reading (246) and the first QPU02 build was other activity on the account.

Replicate 1 chain rule (fixed before replicate 1 existed): the lowest-cost calibrated 5-qubit path avoiding qubits 148-152, using the same cost and beam search as `choose_line_calibrated` (`select_path.py --b 5 --exclude 148 149 150 151 152`, cost 0.0415). It changes the physical chain only; the calibration was the same.

Decision rule for the b=6 stretch (runbook): run only if the five-qubit compact circuits retained informative phase contrast. They did (approximate compact edge contrast 0.189 and 0.309, intervals above zero in both replicates), so b=6 was built and run. It was saturated by noise. The b=4 anchor was not run (it was only prioritized if b=5 had saturated), and the b=6 Gray circuit was never compiled for hardware.

Each stage directory holds `compiled.json`, `compiled.qpy`, `circuits/*.qasm`, both calibration snapshots, raw `counts.json`, `job.json`, `submission_started.json`, `usage_*.json`, `execution_receipt.json` and `analysis.json` (written by `qpu02.py collect`). `qpu02_summary.json` holds the combined tables used in the paper. Authorization files are not committed.

## Reproduce

```
cd experiments/hadamard_qft/qft01
python -m pytest test_qpu02.py -q        # includes re-analysis of the committed raw counts
python analyze_qpu02.py --stage hardware_results/qpu02_b5_r0 --stage hardware_results/qpu02_b5_r1 \
    --stage hardware_results/qpu02_b6_r0 --figdir manuscript/figures --tables manuscript/tables \
    --out hardware_results/qpu02_summary.json
cd manuscript && pdflatex paper.tex && pdflatex paper.tex
```

## Deviations

- No change to `compact.py`, `qpu02.py` or the registered roster. Offline: 8 tests passed (QPU01 and QPU02 suites, including Qiskit statevector and fake-backend checks); `qiskit-check` passed for b=5 and b=6 and the routed CNOT counts matched the runbook (1,122 / 468 / 296 / 218 at b=5; 1,080 / 736 / 434 at b=6).
- The routed and native two-qubit counts were identical at optimization level 1 for every arm.
- `owner_submit_qpu02.ps1` wrote each authorization file for the exact printed plan hash (ceiling 30 s) under the owner's chat pre-approval, then ran submit and collect. Each stage has one `submission_started.json` and one `job.json`; no submission was retried. The files are untracked.
- `analyze_qpu02.py` and `select_path.py` are additions; the runbook's own `analysis.json` (shot-only Hoeffding intervals) is the primary analysis and is unchanged.
