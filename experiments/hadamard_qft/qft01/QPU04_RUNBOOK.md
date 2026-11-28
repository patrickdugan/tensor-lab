# HAD-QFT-QPU04: ancilla-free selective rotations and permutation routing

Parent QPU03 hardware and manuscript: 9547024409ef7594cfea5230e26125d176dbf8db. This is a prospective design and CPU-only audit, not new IBM data. Native translation and live hardware preflight are mandatory before submission.

## Selective rotation identity

Partition the r all-zero controls into nonempty disjoint groups S,T. Let A be any monomial unitary acting on the target as a relative-phase X precisely when S is zero, and otherwise diagonally. Let B be Rz(-theta/2) conditional on T being zero. On each fixed control block, A^dagger Z A equals -Z when S is zero and Z otherwise. Thus B^dagger A^dagger B A is Rz(theta) exactly on the intersection of the predicates and I otherwise. Phases cancel, including on arbitrary complex superpositions; no initialized workspace. Conjugate with S H (the phase gate S=diag(1,i)) to turn Z rotations into Y rotations.

A with a controls can be synthesized via a preparation-equivalent Gray multiplexor (2^a-1 CX): the image of target |0> is the required computational basis state on each control branch, so the unitary extension necessarily permutes both basis states up to phase. For a=3, the explicit six-CX relative-phase triple-controlled flip uses a four-CX diagonal phase loop, which is I except for the first two controls both one, where it is iZ. The third-controlled outer conjugation turns Z into an equatorial Pauli when the third control is one. X flips convert to all-zero controls. This gadget is a known type of relative-phase MCX; we do not claim general multi-controlled gate novelty. The recurrence chooses min(2^r, 2(F(a)+C(r-a))) over 1<=a<r, with F(a)=2^a-1 except F(3)=6. Counts C(r) for r=1..8 are 2,4,8,14,20,28,40,52. It improves the shell recursion's largest boundary rotations without changing the QPU02 geometric approximation. Related work on ancilla-free relative-phase multi-controlled gates: https://arxiv.org/abs/2507.00400 (abstract consulted; full priority/complexity comparison outstanding).

## Routing grammar and CPU audit

For a fixed chronological gate sequence on an n-node connected graph, track every permutation of the n logical wires. All initial and final permutations are free. Min-plus DP first closes each layer under adjacent 3-CX swaps, then chooses a CNOT at the current layout (direct or 4d-4 bridge), a fused adjacent CX and output swap (2 CX), or a three-node template. The template table is obtained by BFS over all 168 elements of GL(3,2), with the four directed CNOT generators on a three-node path. This realizes a desired CNOT followed by an output permutation in one shortest local CNOT word. Reconstruct permutations, map every one-qubit gate and simplify commuting cancellations. DP optimum is only within the stated gate-order/routing grammar, before the final simplification; no global synthesis optimality claim. Full matrix verification checks the logical-to-physical initial/final permutations. Related parity routing literature: Dynamic Qubit Routing with CNOT Circuit Synthesis for Quantum Compilation (QPL 2022). We use an independent small-instance DP and template search.

Saved QPU03 calibration defines two disjoint isomorphic H-shaped seven-node regions:
- Region 0: [71,72,73,58,70,74,79].
- Region 1: [123,136,143,122,124,142,144].
- Local edges: (0,1),(1,2),(0,3),(0,4),(2,5),(2,6).

Offline preliminary development searched control orders with earlier gate/router variants (114 distinct order evaluations in the first completed search; later variant search interrupted and excluded). Final matched four-order comparison per arm (reverse/forward prep, default/modified top boundary order) reran with the final code. The modified order was found in development; the final comparison is not an exhaustive equal-total-search-budget compiler contest. The selection uses only routed counts, before any new hardware outcomes.

| b | Previous logical | New logical | Previous routed | New routed |
|---|---:|---:|---:|---:|
|5|70|68|90|86|
|6|112|98|157|136|
|7|188|138|276|199|
|8|330|192|494|301|

b=5/6 use induced prefixes of the local region graph, b=8 appends a neighbor of local node 4 (physical 69 in region 0). These scaling rows use default reversed order except the b7 frozen selection. All routed matrices exactly agree with the corresponding logical simulator matrices in the selected replay (max entry discrepancy 0 in this deterministic simulation; not a floating-point proof). At b7 the new compiler has spectral-norm error 0.0012406036744622255 and analytic bound 0.0016396012437297346, same mathematical transform as the previous bounded-error version. At b8 the error is 0.0012406036744621973. The b7 DP cost is 201, reduced to 199 by commuting cancellation; both arms independently select initial [0,2,6,5,1,4,3] and final [3,4,0,5,6,1,2] local mappings. They therefore use the same physical preparation and readout qubits. 320 was QPU03's recorded b7 native CZ count; 276 and 199 are **offline routed CX** counts. Basis translation to CZ, exact native matrix comparison, and topology/target checks still need to run on the authenticated environment.

## Frozen hardware design

Two seven-data-qubit jobs, no workspace, 38 circuits each, 1024 shots/circuit. A = previous bounded-error logical compiler with the same new router, D = commutator compiler with the new router. Matched region and preparation/readout maps. Both use the same approximation from level 4 onward, same exact target unitary and analytic bound. Core is input-independent. QPU03's edge/middle/tail probes retained, and added adjacent pairs at shell labels 32 and 96 (plus, minus and imaginary-phase). Edge includes both basis endpoints, preparation-only plus/minus and SPAM endpoints. Primary comparison D-A edge contrast. Joint 95% shot-only Bonferroni-Hoeffding interval for each of the two primary differences: radius sqrt(2 log(4/.05)/1024). Both lower endpoints must exceed zero for a replicated positive primary finding. Secondary probes are descriptive/exploratory. No postselection. Region and calibration effects beyond shot noise are not covered. Seven qubits encode dimension 128 of a 2^127-position transform, not an ambient 127-qubit register or computational advantage.

Use the two frozen JSON files. Reuse QPU03's checked build/submit/collect pipeline via qpu04.py without altering QPU03's archived source. Basis translation at optimization level 0, no remapping, full native operator comparison (up to global phase), each row distribution check and live instruction support check. Native count must not exceed the frozen routed count, D must be <200. Fail closed if any verification fails. Do not use python -O (the logical audits use asserts). A native failure is an offline blocker, not a reason to relax tolerances or substitute an unregistered circuit.

## Reproduce and run (owner machine)

From repository root:
```powershell
cd experiments/hadamard_qft/qft01
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
python -m pytest test_qpu04.py -q
python qpu04.py logical --design qpu04_b7_r0.json --out results_local/qpu04_logical0
python qpu04.py native --design qpu04_b7_r0.json --out results_local/qpu04_native0
python qpu04.py build --design qpu04_b7_r0.json --out results_local/qpu04_b7_r0
```
Offline locally: 23 passed, one skipped (Qiskit absent). Both frozen logical plans regenerate with 38 rows. For eight-qubit resource replay use Router(8, list(EDGES)+[(4,7)]) and core(8,style) from selective.py. Re-run native test on owner machine; if needed fix a compatibility bug offline, document changes and rerun all tests before freezing a revised design. The submitted plan hashes cover the design, source hashes, QPY/QASM and calibration snapshot. Last reported QPU balance after QPU03 was 199 seconds; check live before each build and submit. Propose 60 seconds estimated reservation for two individually authorized <=30-second jobs; actual charges may differ, and this is not a provider hard limit. Never submit on implicit approval or automatically generate owner authorization.

For each build, obtain explicit owner authorization for the printed exact fresh hash, then write UTF-8-no-BOM untracked authorization JSON with the QPU03 fields: authorized=true, backend=ibm_kingston, plan_sha256=<exact hash>, max_qpu_seconds<=30, authorized_by=owner, checked_current_allocation=true, acknowledge_estimate_not_hard_limit=true. Within the fresh 15-minute window:
```powershell
python qpu04.py submit --out results_local/qpu04_b7_r0 --authorization results_local/auth_qpu04_b7_r0.json
python qpu04.py collect --out results_local/qpu04_b7_r0
```
Repeat with r1 JSON and new output/auth paths, using the same checks. If a submission is ambiguous, inspect existing job history; never retry automatically. On both successful collects:
```powershell
python qpu04.py joint --stage results_local/qpu04_b7_r0 --stage results_local/qpu04_b7_r1 --out results_local/qpu04_joint.json
```
Preserve all raw counts and receipts; include full native/resource/control results, shot-only limits and any failures. Independently recompute joint results from raw counts before manuscript integration. No b8 hardware job is authorized or included here. No claim of advantage without the paired hardware result. No additional QPU time consumed preparing QPU04.
