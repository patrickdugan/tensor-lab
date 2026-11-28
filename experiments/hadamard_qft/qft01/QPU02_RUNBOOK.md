# HAD-QFT-QPU02: preparation-equivalent compaction and phase contrast

Prospective hardware protocol, 27 November 2026. Extends QPU01 at 08274a5f5c6e61198bbfd6e106f0bc19a96b9702. Native shell registers only. No QPU submission or native-compilation claim for QPU02.

## Exact and bounded-error compaction

At a recursive level with L=2^r lower labels, W_(2L)=(I tensor W_L) P C P^dagger. C rotates only |0,0>, |1,0>. Thus the full product depends only on P's images of these two vectors, including their relative phases. Use P|0,0>=|0,L-1>, P|1,0>=|1,g_L>, where g_j=c*d^j for j<L-1 and g_(L-1)=d^(L-1), c=d=1/sqrt(2).

For a target initialized to |0>, a cyclic uniformly controlled RY can omit its terminal controlled X, replacing the rotation angle theta on the affected control branch by pi-theta, since X RY(pi-theta)|0>=RY(theta)|0>. This changes the full multiplexor but preserves the selected preparation vectors. Using the actual new inverse P^dagger preserves the complete conjugated rotation. Fuse leading-zero branch flips into RY(pi). Exact preparation CNOT count is sum_(k=0)^(r-1)(2^(k+1)-1)=2^(r+1)-2-r, and the recursive level uses 5*2^r-4-2r including inverse and boundary rotation. Summing gives 5*2^b-b^2-3b-6 before commuting cancellation (previous 5*2^b+b^2-5b-6). No ancillas. Reorder Gray-cycle controls according to line distance; cancel commuting involutions and fuse same-axis rotations; drop only literal zero angles.

For r>=4, optionally substitute v_j=c*d^j/sqrt(1-d^(2L)). This normalized geometric vector factors over binary digits, using one CNOT per bit in P. The two-level rotation on a=|0,L-1>, b=|1,g> is I+(c-1)(aa^T+bb^T)+d(ab^T-ba^T). For b'=|1,v>, the projector difference has norm <=||g-v|| and the skew term difference has norm ||g-v||, since a is orthogonal to b-b'. Consequently the level error is <=((1-c)+d)||g-v||=||g-v||. The unnormalized geometric vector differs from g only at its last entry by (1-c)d^(L-1); normalization costs 1-sqrt(1-d^(2L))<=d^(2L). Telescoping the unitary product gives

delta_b <= sum_(r=4)^(b-1) [(1-1/sqrt(2))*2^(-(2^r-1)/2)+2^(-2^r)].

This bounds ideal pure-state trace distance and measurement TV; a difference of two target probabilities changes by at most 2 delta_b. Exact arithmetic bound excludes finite angle synthesis and hardware errors. b5 bound .001633280517, observed .001238371115; b6 bound .001639601147, observed .001240603657. Approximate level uses 2^r+2r CNOT before cancellation.

## Offline results

All 16 independent-layout settings b3..6, four arms, verified against full encoded unitary; exact settings maximum operator error 5.49e-14. Layout enumeration exhaustive through b5; b6 restricted to 13 candidates per arm. Counts below use each arm's own best tested layout:
| b | Gray | Recursive | Exact compact | Approx compact |
|---|---:|---:|---:|---:|
|3|54|48|28|28|
|4|272|170|104|104|
|5|1108|458|296|218|
|6|4494|1050|736|418|

Common-layout hardware core counts (pre-native compilation):
| b | Gray | Recursive | Exact compact | Approx compact |
|---|---:|---:|---:|---:|
|4|272|170|104|same as exact|
|5|1122|468|296|218|
|6|CPU only|1080|736|434|

Select common logical-to-line permutation by minimizing maximum relative count versus each arm's own best, then sum of ratios, then lexicographic order. Same mapping, physical line and optimizer level for all arms in a job. Logical permutation is calibration- and outcome-independent. All figures are CPU results, not IBM measurements.

## Frozen prospective campaign

Primary b5: 32 shell labels, ambient 2^31 positions. Four arms, 73 circuits, 512 shots per circuit. For a in {0,D/2,D-2}, D=2^b, n=D-1, measure Z after Q on (|a> +/- |a+1>)/sqrt(2), (|a>+i|a+1>)/sqrt(2), and the two basis endpoints. Only LSB input preparation; nine phase-preparation checks, four basis SPAM controls. Barrier between preparation and input-independent full core. Each row stores ideal exact and ideal compiled probabilities. Compare midpoint of +/- distributions and imaginary phase to average of endpoints (incoherent mixture). Edge contrast C=P(n|minus)-P(n|plus) is primary, ideal 1; middle/tail target n-a has ideal contrast 1/sqrt(2). Primary comparison compact_approx vs recursive; exact compact and Gray controls. Finite-shot marginal Hoeffding intervals radius sqrt(log(40)/512) for contrast, sqrt(2log(40)/512) for difference. Independence/fixed probabilities assumed; excludes drift/systematics, no multiplicity adjustment. Positive contrast is not a process-fidelity certificate. Report TV, Hellinger fidelity, raw phase contrasts, preparation/SPAM, mixture diagnostics and every pair. No outcome-dependent roster changes.

Repeat b5 at a distinct preselected calibration or physical line; shuffle with replicate index. b4 anchor and b6 stretch each 58 circuits; b6 excludes huge Gray hardware circuit and is exploratory. Run b6 only if b5 has informative phase signal; if saturated prioritize native optimization and b4. No advantage claim from one job. Last available QPU01 snapshot 354/600 s remaining, recheck before build and submit.

## Run

Existing QPU01 authenticated environment: Qiskit 2.6.2, Runtime 0.52.0. From repository root:

```powershell
cd experiments/hadamard_qft/qft01
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -m pytest test_qpu02.py -q
python qpu02.py qiskit-check --b 5 --out results_local/qpu02_check5
python qpu02.py build --b 5 --replicate 0 --backend ibm_kingston --opt 1 --out results_local/qpu02_b5_r0
```

Local tests: 3 passed, 1 skipped (Qiskit absent). Mandatory Qiskit logical and fake-backend tests on owner's machine. Build checks native active-wire ideal distributions and fixed mapping, hashes QPY/QASM/source/plan/calibration, records live allocation. Optimization level 1 same for every arm. If native/mapping checks fail, fix offline and document changes before submission; optional opt0 must apply to all arms. Nominal estimate b5 22.3816 s, b4/b6 18.1936 s; actual duration/reset estimate recalculated. Estimated ceiling 30 s per job, not a provider hard limit. Reserve 60 s for two b5 replicates subject to separate owner authorization; optional other stages are separate budget decisions.

Owner must write UTF-8-no-BOM auth file with EXACT fresh hash, current backend and budget:
```json
{"authorized":true,"backend":"ibm_kingston","plan_sha256":"REPLACE_WITH_FRESH_HASH","max_qpu_seconds":30,"authorized_by":"owner","checked_current_allocation":true,"acknowledge_estimate_not_hard_limit":true}
```

Within 15 minutes of build:
```powershell
python qpu02.py submit --out results_local/qpu02_b5_r0 --authorization results_local/auth_qpu02_b5_r0.json
python qpu02.py collect --out results_local/qpu02_b5_r0
```

Fail closed on calibration/usage/hash/duration mismatch. Never automatically create auth or retry a submission; inspect job history if started marker exists without ID. Collect retrieves existing job only. Replicate 1 uses new directory and fresh authorization, different line chosen in advance from calibration via --path or new calibration. Commit safe receipts/counts/calibration/manifests/QPY/QASM/analysis and manuscript section after hardware; exclude account files/auth. No new QPU jobs yet. Full local reproducibility archive SHA256 ae702cc387f326870d5fd349d85bfde2190c433abf414a16517bff8346a4d1a5.

## Execution record

Executed 27 November 2026 on ibm_kingston: b=5 replicate 0 job dc4oaqqnj2vs73dkf2kg (14 QPU s), b=5 replicate 1 on the disjoint chain 67,68,69,78,89 job dc4ock2nj2vs73dkf570 (14 s), and the b=6 stretch job dc4oduinj2vs73dkf74g (12 s): 40 QPU s in total, 296 of 600 s remaining. The b=6 stretch was saturated by noise; the b=4 anchor was not run. Raw counts, receipts and reproduction steps: hardware_results/README.md. Owner step: owner_submit_qpu02.ps1 -Dir <stage dir> -Hash <printed plan hash>.
