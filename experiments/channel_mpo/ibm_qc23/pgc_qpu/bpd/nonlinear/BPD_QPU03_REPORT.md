# BPD-QPU03 / QPU03B: cross-edge phase grouping and residual refinement

Evidence: offline finite-state inference simulation. No QPU calls. Both pilots executed in a JavaScript runtime; Node regression checks are supplied.

## QPU03

Three phase coordinates in Z/8, two shared visibility values (1,024 joint hypotheses), six fixed scout quadratures, 26 subsequent candidate quadratures, 10 shots/action, two toy cost checkpoints. The model includes single, pair and triple integer combinations. Certain cross rows acquire an additional nonlinear phase depending on another coordinate. The drift family perturbs truth off grid and changes it after the first budget; inference remains stationary.

Forty model-generated noiseless calibration states are used to recover each sparse integer row by X/Y fitting over the candidate coefficient family {-2,-1,0,1,2}. The other 984 states are evaluated separately. All 16 rows are recovered in each of three model families. This is supervised model compression, not experimental calibration or a trained neural BPD.

For an action a with likelihood p(y|theta), a group map Z_a is sufficient if p(y|theta) depends only on Z_a. In that case I(theta;Y_a|D) = I(Z_a;Y_a|D), and a posterior marginal over Z_a scores the experiment without approximating the full posterior. The exact oracle uses the known residue, visibility, and nonlinear dependency. All policies update the entire posterior with the exact model likelihood.

For an approximate group model q(y|theta), let e(theta)=TV(p,q), delta=E_w e(theta), and K be the number of count outcomes. Entropy continuity gives

    |I_p-I_q| <= f_K(delta) + E_w |H(p(.|theta))-H(q(.|theta))|,
    f_K(t) = h_2(t)+t log(K-1) for t <= 1-1/K, log K otherwise.

Divide by action cost for a score interval. The certified policy evaluates an exact score when an interval could beat the incumbent; tie-breaking uses the same numerical tolerance as dense scoring. This bound assumes the exact inference-model likelihood table, not the true hardware channel. Numerical comparisons on 538 evaluated scores showed maximum positive gap minus bound 4.9e-15. Dense and certified acquisition paths and terminal posteriors agreed on all 72 trials. Uncertified learned paths disagreed in 23/24 nonlinear and 22/24 drift trials.

At budget 10, learned certified design uses 6.82%, 63.04%, 64.67% of dense scoring work in integer, nonlinear and drift families respectively. Oracle groups are cheaper in the latter two. The model-aware random comparator is not outperformed conclusively in the drift family: paired mean random-minus-dense held-out RMSE -0.000675, 95% descriptive interval [-0.00273,0.00105]. All methods saturate alias recovery at the larger budget; the Smith-rank-first policy has the same RMSE as dense in this pilot. Full paired comparisons are in qpu03_analysis.json.

## QPU03B: discover residual group structure

Fit the circular phase residual after removing the learned integer row. Candidate residual features are sin(2pi*x_j/8) and sin(2pi*(x_j+x_k)/8), each with a fitted scalar amplitude. The full finite model likelihood table is then audited for constancy within the inferred residue/feature/visibility group. If any count-bin difference exceeds 1e-12, that action is assigned singleton groups. This is a numerical finite-table sufficiency audit; it is not an analytic proof of equality for continuous phases or unknown physical models.

Five families, 24 paired trials each:
- integer
- one-coordinate nonlinear
- drift/off-grid truth
- nonlinear joint-coordinate residual
- out-of-family mixed sine/cosine residual

All 120 refined-policy paths and posteriors agree with dense selection. Refined scoring work matches the exact oracle in the first four families (14.66x, 7.59x, 7.59x, 7.63x reductions against dense at budget 10). In the mixed residual family, the learned group audit rejects 14 of 32 actions and singleton fallback reduces the advantage to about 1.50x. Oracle grouping is slightly cheaper. The policy's "fallbacks" metric counts on-the-fly dense rescoring in the entropy-bound policy only; rejected refined groups are listed separately in learning[].refined_audit. The full table is available during offline verification, which is an important computational and epistemic limitation.

## Smith aliases

The even scout 2I on Z/8^3 leaves eight integer shifts with each coordinate in {0,4}. Pair sums x1+x2, x2+x3, x1+x3 leave two shifts, (0,0,0) and (4,4,4). Adding the odd triple sum x1+x2+x3 leaves only the zero shift. Exhaustive 512-point kernel enumeration for each stage is in qpu03_analysis.json. Actual quadrature-only observations may retain additional aliases; nonlinear residuals may break the ideal integer symmetry. The pilot tracks the ideal shift kernel, not a full physical identifiability certificate.

## Limits and follow-up

The learned refined group reaches the conventional oracle in designed families. No advantage over an optimized exact grouping algorithm is established. Score-work excludes precomputation and full posterior updates. The calibration feature family was chosen for these experiments; the out-of-family stress demonstrates its failure mode. Scaling requires sparse/structured likelihood certification rather than full-table enumeration, fresh held-out residual families and an end-to-end measured compute budget. The stationary model also needs explicit drift tracking before deployment.

Replay:
    node bpd_qpu03.mjs 24 > /tmp/qpu03.json
    node bpd_qpu03b.mjs 24 > /tmp/qpu03b.json
    node check_qpu03.mjs qpu03_pilot_24.json /tmp/qpu03.json
    node check_qpu03.mjs qpu03b_pilot_24.json /tmp/qpu03b.json
