# BPD-QPU02 v0.1: nonlinear grouped Bayesian experiment design

Offline discrete simulation, no IBM job. The 288-state joint posterior includes two local 12-state phase hypotheses and one shared 2-state visibility regime. The prior correlates the aliases. The full joint posterior is updated after every observed count. A candidate is one edge, control state, X/Y probe and repetition count in {1,2,4,8}, with 12 shots. Counts have an exact binomial likelihood under the inference model.

## Exact group theorem

Let θ denote the full latent vector, and Z_a=f_a(θ) a sufficient projection for candidate a, so p(Y_a | θ,D)=p(Y_a | Z_a,D). Then

    I(θ;Y_a | D)=I(Z_a;Y_a | D)

because the conditional mutual information of θ and Y_a given Z_a,D vanishes. Form the posterior marginal p(Z_a|D) once, score all candidates sharing that projection, then update the full θ posterior with the selected likelihood. This preserves all posterior correlations. No mean-field approximation is used.

For two edges, each observation depends only on the local edge state and the shared visibility regime: 24 of 288 hypotheses per score. The exact grouped policy is a conventional sufficient-marginal Bayesian design implementation. Its equivalence to dense design is mathematical; the pilot checks numerical decisions and counts computational work. It is not evidence that a learned BPD grouping beats this exact baseline.

## Pilot

32 paired trials in each of matched, off-grid and drift+relaxation truth families. Two checkpoints, 6.5 and 13 toy cost units, not QPU seconds. Each action can be selected once; all methods are restricted to the same budget. Shared truth and action/step keyed sampling across policies. Prediction target is the noiseless generating probability of held-out repetition counts 3,5,7.

At budget 13, dense/group-exact RMSE is 0.03962, 0.05414, 0.07561 for the three families respectively. Random RMSE is 0.16400, 0.18165, 0.16439. Paired bootstrap intervals are in pilot_comparisons.json. All 96 dense/group-exact trajectories agree; final posterior max gap 0. Group-exact scores use 91,560 modeled accumulation operations versus 1,029,600 for dense scoring (11.245x reduction). This counts predictive likelihood-bin accumulations plus marginalization, not actual wall time or all inference work. Shared updates and likelihood table generation are omitted.

All 96 trials recover both simulated aliases by the first checkpoint under dense, exact grouped and alias-only design. Alias-only optimizes a coarser mutual information objective, with inferior held-out RMSE.

## Limits and next stage

The model is deliberately small and exact dependency sets are supplied. The misspecified truths do not prove hardware transfer. The first drift checkpoint scores the post-change truth even though its observations precede the change. No new controls were added to the historical hardware receipts. Future work: action-level learned sensitivity groups, joint cross-edge observables, approximate grouping error bounds, scaling to many edges, and comparison with optimized sufficient-marginal or factor-graph designs rather than naive dense scoring alone.

Replay: node bpd_qpu02.mjs 32 > /tmp/bpd_qpu02.json
