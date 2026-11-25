# QC-26A: diagnostic design, no hardware submission

## Purpose and design

The QC-25C constant native SX fit reaches a parameter boundary and leaves the
large local residuals unexplained. This experiment tests a separate native
conditional-phase model without fitting the previous kicked-Ising counts.
The physical targets (5,23) and control sites (3,21) were selected using
retrospective results and the committed Kingston topology. The choice of
control edges is pragmatic, not a matched causal design.

82 circuits at 384 shots = 31,488 shots. Two SPAM circuits prepare all eight
sites in 0 and 1; eight SX circuits measure four sites concurrently using
repetitions 0,4,12,24 in Z and Y; 72 isolated pair CZ Ramsey circuits cover
six directed control/target pairs, two control states, even repeat counts
0,2,8 and X/Y quadratures. Targets (control -> target): 4->5,6->5,16->23,
22->23,4->3,22->21. Deterministic shuffled order. Native SX tests share a
parallel context and cannot rule out simultaneous-drive effects.
Preparation X errors contaminate the SPAM estimates; these are effective
preparation/readout parameters, not independently identified detector errors.

## Phase model and synthetic check

For a pair with control eigenvalue s = ±1, residual target phase after m
even CZ applications is m(phi + s delta) plus a state-dependent intercept.
The ideal CZ contribution is canceled for even m. From measured X+iY,
estimate each slope, then delta=(slope_0-slope_1)/2 and
phi=(slope_0+slope_1)/2. This distinguishes local phase from conditional
phase under the diagonal coherent-error model. Visibility is measured
separately. Non-diagonal coherent errors, population transfer, preparation
errors, readout correlations and pulse-context effects can violate it.
Use a bounded SX grid (±0.1 rad), profile per-repetition visibility and
a common nuisance phase (±0.2 rad); the Y analysis pulse includes its own
SX error. Flag fit boundaries and low visibility.

Synthetic 48 injection and 48 null trials, with Bernoulli finite-shot
sampling, asymmetric readout, decaying visibility and nuisance phases.
All 288 signed target estimates (six injected targets x48) have correct
sign; one of 288 null CZ estimates exceeds |0.02| rad. Target SX RMSE
0.00266–0.00317 rad; target CZ RMSE 0.00524–0.00705 rad. No synthetic
boundary or low-visibility flags. These are conditional simulation
results, not hardware evidence or a formal multiple-comparison test.

## Decision rule

Treat the first batch as exploratory. Keep raw counts and control-state
marginals. A nonzero phase estimate alone is insufficient: require a
fresh independently shuffled repetition block, adequate visibility,
control-state preparation checks, and uncertainty intervals from
multinomial resampling of joint bitstring counts. Check X/Y phase
linearity and aliasing, alternate paths, and the effect of removing
the longest repetition. Freeze parameters and the corresponding
native CPTP channels before comparing to fresh kicked-Ising data.
Do not replace the QC25 baseline using retrospective fit alone.

## QPU gate

Nominal estimate: 16.38688 s using 250 us repetition delay, a floor of
10 us circuit allowance, and 0.1 s overhead per circuit. Actual billed
runtime is provider-dependent; 20 s is an estimate/authorization
ceiling, not a provider-enforced hard timeout. The account owner must
verify current allocation. Build validates native gate counts, layout,
and calibrated durations and freezes the exact QPY, QASM, manifest
and snapshot hashes. The submission command refuses stale (>15 min)
or changed calibration, mismatched hashes, missing owner approval,
and a repeated submission attempt. A failed/ambiguous submission
requires manual account-history inspection. The 15-minute freshness
limit and 20 s ceiling are deliberate conservative design choices.

Local workflow from this directory, on the machine with the saved IBM account:

    python qc26a_compile.py build --out results_local/qc26a
    # Review compiled.json, actual estimate, and account allocation.
    # Owner writes the authorization file separately, using the exact plan hash.
    python qc26a_compile.py submit --out results_local/qc26a --authorization /path/to/owner-auth.json
    node qc26a_design.mjs analyze results_local/qc26a/counts.json /tmp/analysis.json

The authorization schema is the existing QC23 schema plus
acknowledge_estimate_not_hard_limit=true and checked_current_allocation=true.
No authorization is included. Do not commit credentials or local
authorization files. The Python compiler/submission path has not been
executed in this sandbox (Qiskit/IBM account unavailable here).
The JS synthetic estimator was executed and its receipts are committed.
Offline CI is configured to verify ideal circuits, plan and simulation
reproducibility. Live target and runtime behavior remain untested.
No new IBM job and zero additional QPU seconds.
