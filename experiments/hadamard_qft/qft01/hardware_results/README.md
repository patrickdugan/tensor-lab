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

# HAD-QFT-QPU03 hardware record (28 November 2026)

Workspace-assisted compaction and topology-aware compilation, `ibm_kingston`, Qiskit 2.6.2, qiskit-ibm-runtime 0.52.0, 1,024 shots per circuit, Sampler V2, `rep_delay` 0.00025 s, no dynamical decoupling, no mitigation. Frozen designs and offline selection (committed before the corresponding job): `qpu03_design/` (`design_b5.json`, `design_b6.json`, `design_b6_r1.json` at commit a66a7fe; `design_b7.json` at 25d9c60), with the calibration snapshot `design_calibration.json` (28 Nov 2026 00:19 +08:00) used for every ESP.

| Stage | Directory | Job ID | Plan SHA-256 | Arms (native CZ) | Estimate | IBM charge |
|---|---|---|---|---|---|---|
| b=5 | `qpu03_b5_r0` | `dc4rl6dhdvas73bq7no0` | `e4082278f0b3a0044b259c2e36318701e23a6b00d517790975e51a78c3c97b88` | A 218, B 175, C 102, D 95 | 25.837 s | 17 s |
| b=6, region 0 | `qpu03_b6_r0` | `dc4rqqts2omc738fsmpg` | `e877ae383ce7ba670ab9692b097ed78883097a347a750cad811072260a11287b` | A 434, C 189, D 178 | 19.878 s | 13 s |
| b=6, region 1 | `qpu03_b6_r1` | `dc4rrneq4uuc73a9bmt0` | `41672754faa781251e3f5279f9beebdcd2b86b400c5d88e60634a849324f292b` | A 434, C 184, D 185 | 19.878 s | 13 s |
| b=7 | `qpu03_b7_r0` | `dc4sidqnj2vs73dkl2og` | `b035c38844947386d27ace813fd61b8f6db276c76c0ce2a3b920579e4cdffc30` | C 320, D 320 | 13.918 s | 10 s |

QPU seconds charged: 53 (estimates 79.5 s, within the registered 90 s; every job under the 30 s ceiling). Account counter: 348 s consumed before the first QPU03 job, 401 after the last (`qpu03_b7_r0/usage_after.json`), 199 of 600 s remaining; the 53 s increase equals the four job charges. Physical qubits: b=5 A 149-153 (path), B and C 149-153 and 138, D 149,150,151,138,152; b=6 region 0 A 148-153, C 148-153 and 138, D 149-153 and 138; region 1 A 1,2,3,16,23,24, C 1-5,16,23, D 1,2,3,4,16,23; b=7 C 138,147-153, D 51,58,69-73.

Authorization: job 1 on the account owner's chat message "authorized" after the exact plan hash and estimate were shown; jobs 2-4 under the owner's later chat statement that the other three jobs were pre-authorized. For each, the script wrote the authorization file for the exact built hash with a 30 s ceiling after checking hash equality and the estimate; the files are untracked. One `submission_started.json` and one `job.json` per stage; no submission was retried.

## Reproduce (CPU only)

```
cd experiments/hadamard_qft/qft01
python -m pytest test_qpu03.py -q
python audit_qpu02_simultaneous.py      # independent reproduction of the QPU02 simultaneous audit
python analyze_qpu03.py design          # tables/figures from select_b*.json and scaling.json
python analyze_qpu03.py stages --stage hardware_results/qpu03_b5_r0 --stage hardware_results/qpu03_b6_r0 \
    --stage hardware_results/qpu03_b6_r1 --stage hardware_results/qpu03_b7_r0
cd manuscript && pdflatex paper.tex && pdflatex paper.tex
```

Offline design runs are deterministic given seeds (re-embedding recompiles every stored candidate and aborts if the compiled metrics change).

## Deviations and unfavorable results

- Roster trimmed from the QPU02 roster so that 1,024 shots fit the 30 s ceiling: 13 circuits per arm (mixture controls and imaginary inputs only on the edge probe; two preparation-only and two SPAM circuits per arm). Fixed before submission.
- Offline selection used the 00:19 calibration snapshot; builds were made later. Qubit and edge properties of the b=5 and b=6 regions were identical at build; for b=7 the readout errors had been refreshed (CZ and single-qubit errors unchanged).
- b=7 was run only because b=6 retained informative contrast (registered condition). Before its design was frozen we added a native-verification gate (1e-8) to the selection rule after the best-ESP ancilla-free candidate failed by a seed-independent 5.4e-6; the next eligible candidate lies on a different region than arm C, so C-versus-D at b=7 is confounded by region. Rejected candidates are recorded in `design_b7.json`.
- QPU02's line routing had inflated the native counts of the same logical circuits (b=6: 434 against 178 after heavy-hex routing); QPU02's recorded counts were correct for what was run.
- Workspace did not reduce routed native counts at b=5, 6 and did not improve phase contrast (C-D not significant at b=5, 6, 7); workspace readout was anomalous at b=5 (B 0.40, C 0.78; a noisy simulation of the exact circuits gives 0.03; independent of every data qubit) and 0.10-0.12 at b=6; the cause is unexplained (an equatorial-idle dephasing explanation was tested and does not fit). Rotation pruning saves no CNOTs.
- No b=7 replicate and no b=4 anchor were run. The independent audit of QPU02 reproduced 5 of 6 simultaneous improvements (both edge ones); the second-replicate middle probe survives by 0.004.

# HAD-QFT-QPU04 hardware record (28 November 2026)

See `../QPU04_RESULTS.md` for the verdict, native audit, receipts and authorization record. Jobs: `dc54l02nj2vs73dl0cdg` (region 0, plan `ceb4ef50b9a5`, 13 s) and `dc54v96q4uuc73a9p2d0` (region 1, plan `6bf1be377485`, 13 s); 26 QPU s charged, 173 of 600 s remaining. The registered replicated positive primary result is not met (region 0 joint lower bound -0.021, region 1 +0.0003). A first region-1 build (`2e007417028a`) expired before submission and was never sent to IBM.
