# Restricted-Observation Quantum Channel Identification: Phase Ambiguities, Audited Bayesian Design, and a Prospective Hardware Test

**Research manuscript, version 0.1 • 28 November 2026**  
Author list and affiliations to be confirmed before submission. This is a working draft, not a published article.

## Abstract

Learning a small coherent error model from restricted quantum observations requires distinguishing algebraic ambiguity, statistical uncertainty, predictive transfer, and the computational cost of choosing experiments. We examine these issues in a common framework using finite phase observations, a four-qubit causal-cancellation circuit, synthetic Bayesian design studies, and a prospective test on IBM Kingston. Integer observation matrices characterize continuous gauge freedom and discrete phase aliases through Smith invariants. A three-coordinate cyclic example has eight ideal phase aliases after an even scout, two after pair-sum observations, and one after an additional odd triple observation. For finite Bayesian design we use sufficient likelihood groups and entropy-continuity bounds to audit learned approximate groups. Certified or refined grouped selection agrees with dense selection in the tested synthetic campaigns; a fixed-workload benchmark yields a modest 1.24-fold end-to-end CPU speedup for learned audited grouping, with higher memory use. The optimized exact grouping oracle is faster. In hardware, bidirectional Ramsey measurements identify candidate coupler-dependent endpoint phases. Predictions for 24 previously unseen circuits are frozen before acquisition. The registered conditional-ZZ model fails: pooled and targeted mean absolute error improve by 3.64% and 9.04% against the stochastic baseline, below the required 10%, and a non-target site worsens. A declared endpoint-phase extension improves pooled and targeted error by 18.84% and 57.37%, but also violates the non-target degradation criterion. A two-round drift regularizer provides no incremental gain. These results establish the importance of restricted-observation design and independent transfer testing while documenting the limits of the tested model family.

**Keywords:** quantum characterization; coherent errors; Ramsey spectroscopy; observation fibers; Bayesian design; Smith normal form; prospective evaluation.

## 1. Introduction

Predictive quantum characterization faces a practical trade-off between the cost of obtaining diagnostic observations and the adequacy of the resulting channel model. An error parameter can be locally sensitive to a measurement while remaining globally ambiguous. A model can estimate its own parameters accurately in simulation and still transfer poorly to an unseen physical circuit. Numerical accuracy of the forward simulator resolves only one part of this problem.

Gate set tomography (GST) provides detailed, predictive and self-consistent characterization of a gate set, including its gauge freedom [1,2]. Robust phase estimation exploits repeated gates to amplify systematic errors [3]. Time-resolved characterization addresses processor instability [4], while nested error models and wildcard error offer frameworks for quantifying unmodeled behavior [5]. Randomized compiling illustrates the distinct behavior of coherent and stochastic errors [6]. Bayesian experimental design provides the decision-theoretic background for selecting experiments under uncertainty [7,8].

Our study addresses a narrower setting than full GST: restricted measurements of selected physical gates and single-site observables, with a small prespecified noise model. We combine exact finite identifiability calculations, likelihood grouping, and a frozen predictive test. These components are related by the observation map, but their evidence is deliberately separated. The algebra is not a new Smith normal form theorem, the grouping identity is a consequence of sufficiency, and the hardware study does not demonstrate a new physical geometry, an entanglement advantage, or a deployed adaptive acquisition policy.

The distinctive empirical question is whether a small independently measured coherent channel model improves predictions on unseen kicked-Ising circuits, without materially worsening non-target observations. The prospective failure is the primary result of the hardware study. The extension results help diagnose where the restricted model is useful and where it fails.

## 2. Restricted observation and phase ambiguities

### 2.1 Observation fibers

Let Theta be a specified model class, let O:Theta -> Y be an observation map, and let g:Theta -> G be a quantity of interest. A decoder on the image of O exists with g = d composed with O if and only if

$$
O(\theta)=O(\theta')\Longrightarrow g(\theta)=g(\theta').
$$

The decoder is then unique on the image. This elementary factorization identifies the appropriate unit of inference: a fiber of the observation map. A finite, noiseless equivalence statement is different from numerical conditioning or statistical precision.

The MathCan formalization includes quotient observation classes and the corresponding decoder theorems. Its original finite Boolean setting also exhibits a nontransitive event-response relation and a four-query lower bound for certifying independence over all two-input Boolean truth tables. These results concern explicitly supplied observations and interventions.

### 2.2 Gauge fixing and Smith invariants

Consider a connected finite cell complex with V vertices and E oriented edges. A spanning-tree gauge fixes V-1 independent vertex differences and leaves g=E-V+1 edge coordinates. Normalize phases to T=R/Z. Selected oriented face sums define an integer matrix A and a torus homomorphism

$$
A:\mathbb T^g\longrightarrow\mathbb T^f.
$$

If the nonzero Smith invariant factors are d_1,...,d_r, there exist unimodular integer basis transformations U,V with UAV=D, where D is diagonal with those factors. These basis transformations induce automorphisms on the torus and on every finite cyclic phase grid. Consequently,

$$
\ker A\simeq \mathbb T^{g-r}\times \prod_{i=1}^r \mathbb Z/d_i\mathbb Z,
\qquad
|\ker(A\bmod M)|=M^{g-r}\prod_{i=1}^r\gcd(M,d_i).
$$

The first expression distinguishes continuous invisible directions from discrete phase aliases. The second follows by counting solutions to d_i x=0 modulo M for each diagonal coordinate. Real rank alone misses the finite factors. For a full cell-complex face map, the gauge quotient has the familiar interpretation through first cohomology with circle coefficients; a restricted set of faces instead defines its own observation problem.

The cubic Q3 example has twelve edges, eight vertices, six faces, five independent chord coordinates and a single signed Bianchi relation. Every five-face chord minor is unimodular; every four-face subset leaves one free coordinate. The explicit integer checks and finite rank computations are archived in MathCan. In the general small-complex audit, 28 fixtures and 152 finite-modulus comparisons agreed with the formula above across 271,796 enumerated assignments. The executable audit uses determinantal divisors (an expensive small-matrix reference method), rather than a scalable Smith reduction algorithm.

**Example.** On (Z/8)^3, the even scout 2I leaves eight shifts, each coordinate equal to zero or four. Adding x1+x2, x2+x3 and x1+x3 leaves precisely (0,0,0) and (4,4,4). Adding x1+x2+x3 removes the remaining shift. These are statements about ideal phase measurements; a single quadrature, finite shot noise and nonlinear responses may create or remove additional ambiguities.

Formal scope: the Lean development checks finite integer facts for Q3, the finite cyclic alias chain, and general fiber equivalences conditional on certified invertible basis maps. The full Smith existence theorem and the general topological gauge reduction are standard mathematical arguments supplied here, rather than claims of new Lean formalization.

### 2.3 Operational causal cancellation

Consider the unitary circuit

$$
U=\mathrm{CX}_{b,t}\mathrm{CX}_{a,t}
\mathrm{CX}_{s,b}\mathrm{CX}_{s,a}.
$$

On computational-basis inputs it maps (s,a,b,t) to (s,a XOR s,b XOR s,t XOR a XOR b). The source has gate paths to the target, but the two source contributions cancel. Heisenberg conjugation yields

$$
U^\dagger X_t U=X_t,\quad
U^\dagger Z_t U=Z_aZ_bZ_t,\quad
U^\dagger Y_t U=Z_aZ_bY_t.
$$

Linearity extends the identity-source-support property to every target operator. For any initial density operator, including correlated initial states, a local trace-preserving operation on the source before U leaves the target's reduced state unchanged: the adjoint of that source operation fixes the source identity. Postselection and interventions inside the circuit fall outside this statement. The Lean proof certifies the finite Pauli-support pullback; the general operator statement follows by the above argument. The circuit is an exact observation example, not a hardware demonstration of causal cancellation.

For the ideal input |+000>, the output is GHZ+ on (s,a,b) with t in |0>. A two-setting lower bound on unconditional reduced GHZ fidelity is

$$
F_{\rm GHZ+}\geq\max\{0,P_{000,111}+(\langle XXX\rangle-1)/2\}.
$$

Indeed, the complementary subspace contributes at most 1-P to XXX, and the GHZ subspace contributes twice the real off-diagonal coherence. The earlier expression (P+<XXX>)/2 was invalid as a general witness: |+> tensor |Phi+> gives 0.75 despite biseparability. We corrected that expression before the present campaign; historical witness receipts are superseded. No noisy hardware entanglement certification is asserted.

## 3. Audited Bayesian experiment selection

### 3.1 Exact grouping

Let w(theta)=p(theta|D) be a finite joint posterior and p_a(y|theta) the likelihood of a candidate experiment, with cost c_a>0. The design score is mutual information per cost,

$$
S_a=\frac{H(\sum_\theta w_\theta p_a(\cdot|\theta))-
\sum_\theta w_\theta H(p_a(\cdot|\theta))}{c_a}.
$$

If Z_a=f_a(theta) is sufficient for the candidate likelihood, the numerator equals I(Z_a;Y_a|D). Score computation can use the marginal of Z_a, then update the full joint posterior using the selected experiment's likelihood. This preserves global correlations. It follows directly from the chain rule for mutual information and conditional independence, and is conventional exact marginalization, not evidence of a novel learning advantage.

### 3.2 Approximate grouping and certified fallback

For an approximate grouped likelihood q_a, define e_theta=TV(p_a(.|theta),q_a(.|theta)), delta=sum_theta w_theta e_theta, and K as the number of observation outcomes. Entropy continuity for finite distributions [9] implies

$$
|I_p-I_q|\le f_K(\delta)+
\sum_\theta w_\theta|H(p_a(\cdot|\theta))-H(q_a(\cdot|\theta))|,
$$

where f_K(t)=h_2(t)+t log(K-1) for t<=1-1/K, and log K otherwise. This follows by the triangle inequality, convexity of TV under mixing, and the entropy continuity bound applied to the mixture distributions. Dividing by cost gives score intervals. An incumbent exact score can eliminate candidates whose upper bounds are below it; overlapping candidates receive exact scores, with the registered numerical tie rule.

This certificate is conditional on the supplied finite inference likelihoods and floating-point tolerance. It does not certify physical model correctness. In the refinement policy we fit additional residual dependencies and audit every finite likelihood group. A group with within-group bin differences above 1e-12 is rejected in favor of singleton scoring. The full joint posterior remains exact within the discretized model.

### 3.3 Computational experiments

The first information-design study used a local linearized model and matched simulated QPU-cost caps. At the 6.4-second cap, conventional D-optimal per-second selection achieved held-out RMSE 0.0690, versus 0.1017 for BPD weak-trace grouping, with similar weak-direction recovery (88.7% and 88.2%). Ungrouped weak-trace selection matched or exceeded the grouped method. Group restriction offered no demonstrated benefit here.

The nonlinear BPD-QPU02 pilot maintained 288 hypotheses over two local phase variables and shared visibility. Thirty-two paired trials per family compared dense expected information gain (EIG), sufficient-marginal grouping, random and alias-focused selection at fixed toy costs. Across 96 paired trajectories, exact grouped and dense policies selected the same actions and produced identical terminal posteriors. At the larger budget, modeled design-scoring work was reduced by a factor of 11.245. This count omits shared setup and posterior-update work.

| Truth family | Random RMSE | Dense / exact grouped RMSE |
|---|---:|---:|
| Matched | 0.1640 | 0.0396 |
| Off-grid | 0.1816 | 0.0541 |
| Drift and relaxation | 0.1644 | 0.0756 |

BPD-QPU03 used 1,024 joint hypotheses, three phase coordinates, and 40 model-generated noiseless calibration states to infer integer observation rows. All 48 rows across three families were recovered. Yet the uncertified learned policy differed from dense design on 23/24 nonlinear and 22/24 drift trajectories, demonstrating that recovering the integer projection does not recover a sufficient nonlinear observation. The entropy-bound policy matched dense selection in all 72 paired trials. Its modeled scoring work fell to 6.82%, 63.04% and 64.67% of dense scoring in the integer, nonlinear and drift cases, respectively. Under stationary inference with drifted off-grid truth, dense EIG did not conclusively beat random: RMSE 0.1264 versus 0.1258, with a paired interval including zero.

BPD-QPU03B learned residual coordinate features, audited the entire finite likelihood table, and matched dense selection across 120 paired trajectories.

| Model family | Scoring-work reduction | Matched trajectories |
|---|---:|---:|
| Integer | 14.66x | 24/24 |
| One-coordinate nonlinear | 7.59x | 24/24 |
| Drift/off-grid truth | 7.59x | 24/24 |
| Joint-coordinate nonlinear | 7.63x | 24/24 |
| Out-of-family residual | 1.50x | 24/24 |

The last family rejected 14 of 32 candidate-action groups and used singleton fallback. This controlled stress demonstrates a safeguard, with a computational penalty. The oracle knows the model's exact dependencies; the learned residual families were selected for these synthetic experiments. No neural grouping, large-scale continuous inference, or hardware adaptive selection has been established.

### 3.4 Measured end-to-end cost

A separate archived Node 22 benchmark measured setup, group learning and audit, likelihood construction, score calculation and full posterior updates for 1,024 hypotheses and 24 eight-decision trajectories. Five fresh processes were used per arm on a Linux AMD EPYC runtime.

| Policy | Median process CPU (ms) | Median peak RSS (KiB) |
|---|---:|---:|
| Dense | 224.025 | 57,304 |
| Exact oracle | 162.841 | 57,496 |
| Learned audited | 181.084 | 59,736 |

All fifteen runs have matching action digests and posterior checksums in the archived receipt. Learned grouping reduced median CPU time by 19.2% (1.24x speedup) with 4.2% higher peak RSS; it remained slower than the oracle. The timing is descriptive for one synthetic workload, excludes process startup, and is not a hardware speedup. The large scoring-work reductions above should not be conflated with end-to-end performance. This benchmark uses the earlier archived measurement; the final hardware campaign did not rerun it.

## 4. Hardware protocol

### 4.1 Coherent model and Ramsey identifiability

The native controlled-Z (CZ) coherent error is modeled as a diagonal rotation

$$
U_{ij}^{\rm err}=\exp[-i(\alpha_{i|ij}Z_i+\alpha_{j|ij}Z_j+\delta_{ij}Z_iZ_j)/2].
$$

For a Z-eigenstate control b and a probe, the idealized repeated-CZ phase has slope alpha_probe+(-1)^b delta+b*pi. The control-local term is unobservable in that orientation. Role reversal gives access to the other endpoint. With only even repetition counts the slope is periodic modulo pi, creating the exact simultaneous half-pi alias in endpoint and conditional phases. An odd repetition separates that alias; the remaining simultaneous pi shifts leave the observable unchanged. The analysis therefore uses a specified small-error branch rather than claiming globally unique Hamiltonian identification.

The historical local sensitivity analysis of the 64-by-14 Jacobian found rank four of six for targeted observations, rank six of six for all sites, and rank twelve of fourteen for the expanded all-site model at the examined point. Two endpoint terms had zero derivatives. These numerical local ranks motivated Ramsey acquisition, not a global identifiability certificate.

The Ramsey designs use m=0,1,2,8, both probe quadratures, both control preparations, forward/reverse roles and SPAM reference circuits. The registered estimator fits phase slopes and a decaying visibility envelope to SPAM-corrected probe marginals; joint-bitstring bootstrap resampling includes the SPAM rows. The QC-26A diagnostic estimates signed SX offsets and one-orientation coherent phase terms. Each new fit used 400 bootstrap replicates. Preparation and measurement effects are only approximately separated by these measurements. Control relaxation, context dependence and non-diagonal coherent errors can invalidate the fit.

### 4.2 Registered models

The prospective model family operates on compiled native RZ/SX/CZ circuits, with backend calibration and specified gate-local channels. The six frozen model sets are:

| Model | Definition |
|---|---|
| C0 | Gate-local twirled stochastic noise |
| C1 | C0 plus non-unital relaxation model |
| C2 | C1 plus coherent SX angle offsets |
| C3 | C2 plus conditional CZ ZZ phases, with endpoint-local phases set to zero |
| C3Z | C3 plus measured endpoint-local Z phases |
| C3Z-D | C3Z with a prespecified two-round temporal regularizer |

C3 is the primary endpoint. C2 is secondary and C3Z and C3Z-D are declared extensions. The parameter pack uses four SX offsets and four edge triples (one ZZ and two endpoint phases per edge), for sixteen fitted coordinates. Ramsey estimates take precedence over the QC-26A conditional phases on the overlapping edges, avoiding duplicate use. Estimates are included as registered plug-in values even when their intervals include zero; other couplers receive no fitted coherent correction. Phase bounds and uncertainty checks precede freezing.

For C3Z-D, each phase is treated as an independent Gaussian random walk with fixed process SD q per square-root hour and a fixed SE floor. With historical mean mu0, variance P0=max(SE0^2,floor^2), fresh estimate z and variance R=max(SE1^2,floor^2), the propagated variance is P-=P0+q^2 dt. The updated mean and variance are

$$
K=P^-/(P^-+R),\quad
\mu=\mu_0+K(z-\mu_0),\quad P=(1-K)P^-.
$$

The model uses q=0.003 rad/sqrt(hour) for SX, 0.010 for CZ, and SE floors 0.003 and 0.005 rad respectively. The forecast variance is reported, and the predictive circuit calculation uses the plug-in mean. These constants were fixed before evaluation. Two rounds do not identify a physical drift process or between-cycle evaluation uncertainty.

A synthetic parameter stress study (512 trials in each of three families) gave latest versus drift RMSE of 0.005369 vs 0.004738 rad (stationary), 0.006438 vs 0.006139 (matched random walk), and 0.006525 vs 0.008375 (unmodeled jump). Nominal 95% forecast coverage was 98.6%, 95.0%, and 83.5%, respectively. The unmodeled-jump result warned against excessive historical shrinkage.

### 4.3 Acquisition and freezing

The November 27, 2026 UTC historical diagnostic round comprised QC-26A, q5 Ramsey and q23 Ramsey: 214 circuits at 384 shots, 82,176 shots and 34 provider-reported seconds. The fresh round and evaluation comprised:

| Job | Circuits | Shots/circuit | Provider seconds |
|---|---:|---:|---:|
| QC-26A | 82 | 384 | 11 |
| q5 Ramsey | 66 | 384 | 9 |
| q23 Ramsey | 66 | 384 | 9 |
| QC-26B evaluation | 24 | 2,048 | 15 |
| Total | 238 | | 44 |

The fresh calibration blocks were independently shuffled with registered seeds 261127041–261127043 and later calibration snapshots. Each was built, authorized, and submitted separately under a 15-minute freshness gate. The two acquisition rounds account for 213,504 shots and 78 seconds of attributed provider usage; six other historical seconds were not attributed in the earlier ledger. This is provider-reported allocation consumption, not wall-clock runtime.

The evaluation set comprises widths 8 and 12, depths 2–4, and four fresh circuit seeds 51–54. It yields 240 single-site Z observables, of which 24 are on targets 5 and 23, 24 on controls 3 and 21, and six on qubit 6. Paths include target and control classes; three depths reuse each of eight width/seed path combinations. No evaluation shot was used for fitting.

A v3 self-hashed freeze binds the parameters, predictions, circuit hashes, calibration snapshot, engine, scoring code and numerical ledgers. The published freeze commit is 1f120d1250d49d6c91aa21b12215e6a90ff6564a, timestamped 2026-11-27 16:09:07 UTC. The evaluation job dc4qnoeq4uuc73a99t3g was submitted at 16:12:19 UTC (November 28 local time, GMT+8). The archived submission receipt binds the exact plan and frozen hash to that commit. All four fresh jobs were executed once, with no replacement of the final dataset.

The publication audit retrieved the archived manifests and all four raw count files, checked the circuit IDs, shot counts, bitstrings and job/plan links, and reconstructed the evaluation Z means directly from counts. All 240 measured observables and frozen predictions matched the archived rows. The frozen file has the same blob at its original publication and final campaign commits. The provider job records have not been separately authenticated against a live IBM API in this manuscript audit.

### 4.4 Statistical and numerical acceptance

The observable error is the absolute difference between predicted and observed single-site Z expectation. For a model M, the relative improvement is (MAE(C0)-MAE(M))/MAE(C0).

The primary verdict requires all of the registered criteria: (a) at least 10% improvement pooled and targeted; (b) positive lower limits of paired 95% circuit-bootstrap intervals for pooled and targeted improvements; (c) no other physical site worsening by more than 0.005; (d) maximum numerical truncation ledger below 1% of pooled MAE; (e) frozen integrity verification. The bootstrap uses 20,000 draws and a fixed seed, keeping sites from the same circuit together. The original circuit-level estimand weights circuits equally, while pooled point estimates weight observables equally. The additional path-cluster bootstrap keeps all depths of a width/seed path together and weights observations. With eight pooled path clusters and four targeted clusters, these intervals are within-block sensitivity analyses, not evidence of repeatability across calibration cycles.

The Heisenberg simulator propagates a sparse Pauli expansion. Discarded Pauli coefficients give an L1 bound on the operator norm of each discarded piece; adjoints of the stipulated CPTP channels are unital positive maps and contract the norm of Hermitian observables. Summed discarded contributions thus provide a truncation bound conditional on exact model operations. The implemented ledger also records a conservative alternative norm bound and live term count. These ledgers do not bound channel misspecification, shot noise, or every floating-point rounding error.

## 5. Prospective results

### 5.1 Independent calibration

The historical 5–6 and 16–23 Ramsey conditional phase estimates were +0.03164 and -0.02496 rad. The fresh estimates are +0.02399 and -0.02316 rad. The earlier retrospective effective ZZ magnitudes of approximately 0.1 rad were therefore not reproduced by independent diagnostics. On 5–6 the qubit-5 endpoint phase persists near +0.18 rad, while the qubit-5 endpoint on 4–5 is small.

| Edge | Fresh delta (rad) | Fresh endpoint alpha (rad; ordered endpoints) |
|---|---:|---|
| 4–5 | -0.005374 | -0.012796, +0.009069 |
| 5–6 | +0.023989 | +0.177269, +0.011471 |
| 16–23 | -0.023158 | +0.019002, -0.067892 |
| 22–23 | -0.006793 | +0.000034, -0.031288 |

Fresh SX offsets at 3,5,21,23 are -0.008837, -0.000461, +0.000668 and -0.007505 rad. QC-26A reported seven nonzero candidates pending replication, seven consistent-with-zero decisions, and two inconclusive flags. Ramsey chi-square values across the four edges were 25.3, 28.4, 14.1, 18.6; for 16–23 the forward/reverse delta discrepancy interval excluded zero. Some fitted visibilities exceed one under the effective-SPAM convention. These limitations warn against interpreting the fitted coefficients as perfectly isolated physical Hamiltonian terms. The registered model retains them unchanged.

### 5.2 Primary and extension outcomes

**The registered C3 endpoint failed.** Table 1 gives the complete frozen point results; each evaluated extension also fails at least one registered condition.

**Table 1.** Prospective mean absolute error and improvement versus C0. Values are unitless Z-expectation errors.

| Model | Pooled MAE | Target MAE | Pooled improvement | Target improvement | Verdict |
|---|---:|---:|---:|---:|---|
| Ideal | 0.03699 | 0.09953 | | | reference |
| C0 | 0.03627 | 0.10232 | | | baseline |
| C1 | 0.03518 | 0.10089 | | | descriptive |
| C2 | 0.03541 | 0.10205 | 2.36% | 0.26% | FAIL |
| C3 primary | 0.03495 | 0.09307 | 3.64% | 9.04% | FAIL |
| C3Z | 0.02943 | 0.04362 | 18.84% | 57.37% | FAIL |
| C3Z-D | 0.02944 | 0.04399 | 18.82% | 57.00% | FAIL |

C3 fails the 10% requirement and the non-target guardrail. C2 fails the improvement and paired targeted interval requirements. C3Z and C3Z-D exceed the relative-improvement requirements and have positive circuit and path-cluster intervals against C0, yet both violate the guardrail. They cannot replace the primary verdict.

**Table 2.** Paired 95% bootstrap intervals on absolute MAE improvement. Positive means lower error in the first-named candidate relative to the comparator.

| Comparison | Pooled circuit | Target circuit | Pooled path | Target path |
|---|---|---|---|---|
| C2 vs C0 | [0.00055,0.00117] | [-0.00083,0.00132] | [0.00060,0.00106] | [-0.00143,0.00197] |
| C3 vs C0 | [0.00059,0.00270] | [0.00446,0.01476] | [-0.00014,0.00300] | [0.00269,0.01885] |
| C3Z vs C0 | [0.00318,0.01345] | [0.03176,0.08907] | [0.00088,0.01475] | [0.02417,0.09846] |
| C3Z vs C3 | [0.00225,0.01117] | [0.02450,0.07689] | [0.00118,0.01176] | [0.02134,0.07961] |
| C3Z-D vs C0 | [0.00318,0.01343] | [0.03142,0.08864] | [0.00089,0.01469] | [0.02371,0.09799] |
| C3Z-D vs C3Z | [-0.00005,0.00003] | [-0.00052,-0.00024] | [-0.00005,0.00003] | [-0.00056,-0.00019] |

The C3 pooled path interval contains zero. The endpoint-local extension improves target MAE from 0.09307 to 0.04362, with positive incremental circuit and path-cluster intervals. Nevertheless, the registered extension verdict is failure. The two-round drift estimate slightly worsens targeted error relative to C3Z (0 wins, 12 losses across target-bearing circuits); the incremental drift verdict is inconclusive or negative.

### 5.3 Localization of gains and degradation

**Table 3.** Site-level MAE for the two intended targets and the only flagged non-target.

| Physical site | n | C0 | C3 | C3Z | C3Z-D |
|---|---:|---:|---:|---:|---:|
| 5 | 12 | 0.09102 | 0.08197 | 0.06286 | 0.06354 |
| 23 | 12 | 0.11361 | 0.10417 | 0.02438 | 0.02445 |
| 6 | 6 | 0.02155 | 0.03334 | 0.04159 | 0.04161 |

Qubit 23 supplies much of the observed benefit. As a post hoc descriptive decomposition, the two targets (10% of observables) contribute approximately 85.9% of the net C3Z pooled absolute-error reduction; qubit 23 alone contributes about 65.3%. At qubit 6, C3 and C3Z worsen MAE by 0.01179 and 0.02004 respectively. The latter exceeds the allowed increase by a factor of four. Site 6 is a neighbor of site 5 on the 5–6 coupler with the large measured endpoint phase; proximity motivates model diagnostics but does not establish the cause of the degradation.

The finding supports limited predictive transfer of the declared endpoint-phase model in this block, with heterogeneous errors and insufficient evidence for safe non-target transfer.

### 5.4 Numerical error and reproducibility

For C3, C2, C3Z and C3Z-D the largest recorded truncation bounds relative to pooled MAE are approximately 1.61e-9, 1.38e-9, 2.33e-9 and 2.34e-9, respectively. The maximum live Pauli term count was 53,753. Truncation cannot explain the failed registered criteria under the stipulated simulator model. The execution report records byte-identical re-scoring of the archived data; the independent manuscript audit separately reconstructed all measured means and matched all frozen rows. No replacement dataset or model adjustment is made following the results.

## 6. Discussion

The primary scientific outcome is a failed prospective test of a restricted coherent model. A locally identifiable parameter vector, successful repeated-gate fits and an accurate numerical simulator were insufficient to satisfy the transfer and non-target criteria. Endpoint-local phase corrections improved predictive accuracy substantially on the intended target sites, especially qubit 23. The failure at qubit 6 prevents a device-wide improvement claim. The forward/reverse disagreement and diagnostic flags indicate limitations of the model and measurement assumptions.

The algebraic examples explain why diagnostic designs should be assessed globally as well as locally. Smith factors identify exact discrete aliases even in rank-complete phase systems. Role reversal and odd repetitions directly address particular invisibilities of the Ramsey design. Neither guarantees that the fitted diagonal model is a complete physical description.

The grouping studies establish equivalence of sufficient-marginal scoring to dense scoring under a finite model and demonstrate a numerical fallback strategy for learned approximations. Conventional exact grouping remains a strong comparator. The measured end-to-end savings are modest. Inference drift remains difficult: the stationary synthetic inference model did not reliably outperform random design under misspecification, and the physical two-round drift extension did not improve the latest-calibration predictor.

The results motivate future evaluation under additional independent calibration cycles, diagnostics for spectator and context-dependent channels, and a comparison to optimized model-based experiment design. Such work should be separately registered with a new evaluation block. The present failed endpoint remains fixed.

### Limitations

The hardware test uses one evaluation block on one device and 34 physical sites; path-cluster intervals have few clusters. Two diagnostic rounds do not give between-cycle variance for evaluation MAE. The design uses plug-in fitted values, marginal effective SPAM, restricted local channels, and a small-error branch. It does not learn general non-Markovian noise or the covariance of all parameters. The evaluated hardware acquisitions used registered fixed diagnostic blocks; the synthetic BPD policy was not deployed on them. We make no claim of experimental quantum causal cancellation, entanglement certification, emergent spacetime, gravity or quantum advantage. The formal proofs cover stated finite algebra and conditional equivalences, not general physical model validity.

## 7. Conclusion

Restricted-observation quantum characterization benefits from explicit ambiguity analysis and a strict separation between calibration and evaluation. The finite algebra and grouped design experiments illustrate how those principles can be implemented and audited. On IBM Kingston, the independent coherent-phase model fails the registered prospective endpoint; its endpoint-local extension gives a meaningful target-site improvement with a disqualifying non-target deterioration. A drift regularizer supplies no incremental improvement. This complete record, including the failures, provides a reproducible basis for future model development.

## Data, code, and formal evidence

The results are pinned to [tensor-lab 5244802](https://github.com/patrickdugan/tensor-lab/commit/5244802c3e3f8317a359096d8ef2f1ba5f1cd7c6). The [prospective freeze](https://github.com/patrickdugan/tensor-lab/commit/1f120d1250d49d6c91aa21b12215e6a90ff6564a) predates evaluation. See the [registered protocol](https://github.com/patrickdugan/tensor-lab/blob/1f120d1250d49d6c91aa21b12215e6a90ff6564a/experiments/channel_mpo/ibm_qc23/qc26/PGC_QPU04_PROTOCOL.md), [final campaign report](https://github.com/patrickdugan/tensor-lab/blob/5244802c3e3f8317a359096d8ef2f1ba5f1cd7c6/experiments/channel_mpo/ibm_qc23/qc26/PGC_QPU04_CAMPAIGN_REPORT.md), [hardware receipts](https://github.com/patrickdugan/tensor-lab/tree/5244802c3e3f8317a359096d8ef2f1ba5f1cd7c6/experiments/channel_mpo/ibm_qc23/pgc_qpu/receipts/qpu04_live_20261127), and [Bayesian design studies](https://github.com/patrickdugan/tensor-lab/tree/5244802c3e3f8317a359096d8ef2f1ba5f1cd7c6/experiments/channel_mpo/ibm_qc23/pgc_qpu/bpd/nonlinear).

MathCan's finite proof modules are in [Pregeometry](https://github.com/patrickdugan/MathCan/tree/685b98980a5ee4fd0f1429a7d4c1b3dfa0be39d4/MathCan/Pregeometry). The [dedicated Lean workflow](https://github.com/patrickdugan/MathCan/actions/runs/46033283951) checked Q3 curl, the quantum diamond support theorem after repair, Smith observation fiber equivalences and finite aliases, as well as the exact Node and Python replay. The axiom reports for those modules contain no sorryAx or native-evaluation axioms. The broader historical extension receipt's entrypoint hash also required refreshing after the new imports; its final CI status is a pre-submission item, together with the final tensor-lab PGC replay.

Re-score from the repository's experiments/channel_mpo/ibm_qc23 directory, using the pinned source:
    
    node qc26/qc26b_prospective_eval.mjs \
      --frozen pgc_qpu/receipts/qpu04_live_20261127/frozen.json \
      --build pgc_qpu/receipts/qpu04_live_20261127/eval \
      --params pgc_qpu/receipts/qpu04_live_20261127/params.json \
      --counts pgc_qpu/receipts/qpu04_live_20261127/eval/counts.json \
      --out /tmp/qpu04_report.json

The hardware record contains job identifiers, raw counts, compiled QASM, authorization and calibration provenance, fit reports and the scoring output.

## References

1. E. Nielsen, J. K. Gamble, K. Rudinger, T. Scholten, K. Young and R. Blume-Kohout. Gate Set Tomography. *Quantum* **5**, 557 (2021). https://doi.org/10.22331/q-2021-10-05-557
2. R. Blume-Kohout et al. Demonstration of qubit operations below a rigorous fault tolerance threshold with gate set tomography. *Nature Communications* **8**, 14485 (2017). https://doi.org/10.1038/ncomms14485
3. S. Kimmel, G. H. Low and T. J. Yoder. Robust calibration of a universal single-qubit gate set via robust phase estimation. *Physical Review A* **92**, 062315 (2015); erratum **104**, 069901 (2021). https://doi.org/10.1103/PhysRevA.92.062315
4. T. Proctor et al. Detecting and tracking drift in quantum information processors. *Nature Communications* **11**, 5396 (2020). https://doi.org/10.1038/s41467-020-19074-4
5. E. Nielsen, K. Rudinger, T. Proctor, K. Young and R. Blume-Kohout. Efficient flexible characterization of quantum processors with nested error models. *New Journal of Physics* **23**, 093020 (2021). https://arxiv.org/abs/2103.02188
6. J. J. Wallman and J. Emerson. Noise tailoring for scalable quantum computation via randomized compiling. *Physical Review A* **94**, 052325 (2016). https://doi.org/10.1103/PhysRevA.94.052325
7. K. Chaloner and I. Verdinelli. Bayesian experimental design: A review. *Statistical Science* **10**, 273–304 (1995). https://doi.org/10.1214/ss/1177009939
8. C. E. Granade, C. Ferrie, N. Wiebe and D. G. Cory. Robust online Hamiltonian learning. *New Journal of Physics* **14**, 103013 (2012). https://doi.org/10.1088/1367-2630/14/10/103013
9. K. M. R. Audenaert. A sharp continuity estimate for the von Neumann entropy. *Journal of Physics A: Mathematical and Theoretical* **40**, 8127–8136 (2007). https://doi.org/10.1088/1751-8113/40/28/S18

## Pre-submission checklist

- Confirm final tensor-lab PGC and MathCan CI, preserving exact workflow references.
- Confirm authorship, affiliations, funding, and competing-interest disclosures with the research owner.
- Independently verify all final manuscript tables and figure captions against the pinned receipts; keep raw primary verdict unchanged.
- Review bibliography and prior work for scope and novelty; convert to the chosen journal's LaTeX format after the source draft is reviewed.
