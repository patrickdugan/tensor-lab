# PGC-QPU01 four-edge Ramsey campaign (proposed, not submitted)

The first audit used the historical C3 fit slice and found two zero derivative columns for the target endpoint phases on 22_23 and 4_5. The earlier 32-circuit supplement breaks the explicit alias on the strongest fitted edges, but relies on collecting the original baseline in the same session. This self-contained alternative covers all four edges and collects its own zero-repeat and SPAM references.

Two separately compiled blocks, each 66 circuits at 384 shots: q23 on 16->23 and 22->23; q5 on 4->5 and 6->5. Both directions, both conditioning states, X and Y, repeat counts 0,1,2,8; two SPAM preparations across the three sites in each block. All four edges have independent forward and reverse phase contrasts. The ideal CZ contribution m*b*pi must be removed before estimating a residual phase when the repetition is odd. The even-only alias shifts (alpha_t,delta) by (pi/2,pi/2); m=1 distinguishes it. The reverse experiment exposes alpha_control. Remaining periodic global equivalences, pulse errors, SPAM and nuisance visibility are not excluded by this model.

Logical rows are shuffled with pinned seeds. Each block has a conservative *estimated* cap of 15 s. Nominal estimate 13.18944 s per block, 26.37888 s total, excluding fresh replicate and existing QC26A SX measurements. Do not append these rows to the older 82-circuit QC26A plan or use its existing authorization. A new gated submit adapter and fresh account-owner authorization of an exact compilation are needed. A fresh independently shuffled replicate is required before a nonzero replicated verdict; this is an exploratory design only. No current quota or actual runtime was checked.

The independent two-qubit amplitude oracle agrees with the analytic X/Y model on 384 comparisons (injected, null and shifted-phase cases), max gap 3.552713678800501e-15. Synthetic trials (256 per arm, each covers four edges) include per-site asymmetric effective readout and preparation estimates, state/orientation intercepts, visibility decay and finite shots. Their delta RMSEs are 16_23: 0.00394; 22_23: 0.00441; 4_5: 0.00439; 5_6: 0.00354 rad. See the receipt for signed counts, local phase errors, null excursions and forward/reverse delta gap diagnostics. These are marginal simulations; no hardware significance or robust coverage claim. The simple unwrapping estimator assumes small residual slopes and can alias under larger deviations or low visibility. Full joint-bitstring refitting and an alias-aware estimator are needed for real counts.

Reproduce offline:

    node ramsey_campaign.mjs --check

On the machine with the configured IBM account, compile only:

    python compile_readonly.py --block q23 --out results_local/pgc_q23
    python compile_readonly.py --block q5 --out results_local/pgc_q5

The compiler reuses QC26A's native logical circuit, path mapping, count and duration checks. The script produces QPY/QASM and calibration/plan hashes, with no submission path. Qiskit and account access were unavailable for this new adapter during preparation; real compilation is pending. The old build cannot authorize it. Use the existing QC26A workflow for independent SX data and the QC26B prospective freeze protocol after independent phase calibration.
