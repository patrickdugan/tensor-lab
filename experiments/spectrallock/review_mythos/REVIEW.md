# Mythos Max: SpectralLock security review and construction brainstorm

Frozen inputs: SpectralLock SL01–04 at 702fe98779644f7a2b4cb7e403ce7ba16c15384e; HAD-QFT manuscript and hardware at 08274a5f5c6e61198bbfd6e106f0bc19a96b9702. Date: 2026-11-27. This packet is self-contained; retrieve additional files only where a disputed claim or proposed construction requires them. Synthetic research only. No deployed cryptography, no Nova, no live-key analysis, no new hardware jobs.

## Objective

Find a defensible new lattice-based construction or a genuinely useful implementation theorem informed by our Hadamard/CRT/QFT work. Prefer an exact equivalence or a reduction with the right distribution over a superficially hard new problem. An explicit no-go result is a valuable outcome. Spend most reasoning on the first two tasks below; QPU review is optional and logically separate.

## 1. Established negative results

For a prime ell, N=ell^n, binary valuation-shell signs s_0,...,s_n, the unnormalized integer Fourier transform ordered by descending additive order has
P_0=s_0; t_j=ell^(j-1); F_j=P_(j-1)-t_j s_j; P_j=P_(j-1)+(ell-1)t_j s_j.
The final coefficient is P_n. In natural shell order T^2=N I. When N is a unit in the coefficient ring, T identifies the shell-constant convolution algebra with a product of n+1 scalar rings. This identity is distinct from the normalized quantum amplitude transform.

Rounding in all toy tests is floor(B*(z mod q)/q), B<q. For prime q>4N, independent nonzero uniform spectral multipliers and B>4, merge prefix witnesses with equal P_j. A wrong unrounded coefficient survives with probability <=2/B; at most two extensions match the exact planted coefficient. E K_j<=2+(4/B) E K_(j-1), hence expected linear candidate checks. Fixed modulus q permits at most 2nq checks by prefix-residue merging. Fixed-width per-frequency block mixing and cross-level causal mixing with fresh independent measurements have analogous bounds, subject to coverage/independence. Globally noncausal mixing is outside those bounds. These count extensions, not worst-case bit complexity.
Single-row full squared spectra are prefix-invertible up to sign ambiguities. Distinct spectra count: 2^(n-1)+1 for ell=2, 2^n for odd ell, n>=1. Multirow k=12 search caps establish no hardness.
For sparse public S and rounded STs, omitting any dyadic Fourier coordinate gives explicit collisions. For F_j use all-plus first j signs, then minus, and negate that entire prefix; P_j=0 so only F_j changes. If DC is omitted use opposite constant vectors. Four randomly selected coordinates per row, m=d=256: support omission probability .9914649. Uniform dense A remains uniform after an invertible transform; sparse S is a different distribution. Full coverage removes this particular collision witness. Collision resistance and random planted preimage hardness are different properties.

## 2. Frozen experiments

| Stage | Result |
|---|---|
| SL01 | All 1536 scalar and 192 block-rounded synthetic cases inverted. Full power: k=1,4,8 each 48/48; k=12 24/48, 24 capped. |
| SL02 | Cross-shell 256/288 hit, 32 globally sparse capped; fixed-q diagonal 96/96 through 1024 bits. |
| SL03 | 102 LLL reductions over 78 distinct inputs; 71 hits, 31 misses, all hits original. delta .75: dense 23/36, sparse 21/36; delta .99 fixed q: dense 10/12, sparse 12/12. Regular full coverage d28: 5/6. 15/16 earlier capped dyadic matrices admit explicit collisions. |
| SL04 | Stronger LLL, q12289, scalar dimensions 16/24/32, m=d, B8/32: dense 14/16, module 15/16. All 29 hits original, 32 full rank. |

LLL here is SymPy integer LLL on toy synthetic instances. No BKZ or concrete security estimate. An LLL miss is not a lower bound. The exact embedding and verification are in SL03 source. Raw per-case receipts are in the local archives and can regenerate from the pinned source; the Git repository contains summaries and executable reproduction code.

## 3. Surviving direction: coefficient-domain lattice arithmetic

SL04 uses R_N=F_q[X]/(X^N+1), uniform coefficient-domain public A, iid sign secrets, coefficientwise rounding AFTER inverse NTT. 192 arithmetic settings, 262080 coefficient comparisons and 376740 NTT round-trip coordinates. Prepared-public NumPy NTT at degree2048 rank3: 1.198ms vs direct convolution 22.686ms; optimized native baseline untested.

Subring warning: N=rd, Y=X^r. If s=s_0(Y), decompose a=sum_h X^h a_h(Y); deinterleave each rounded equation to obtain r smaller-ring equations. Uniform public entries become independent uniform smaller-ring entries. Secret dimension kd, sample count mr for rank k and original m samples. If A also lies in the subring, other cosets vanish. 192 tests, 97920 coefficient equalities.
Unrestricted s=sum_g X^g s_g(Y): output h=sum_g Y^[h<g] a_((h-g) mod r)(Y)s_g(Y). Keeping every sector permutes the full scalar matrix, secret coefficients and rounding coordinates; the smaller-ring blocks are correlated. 180 tests, 177408 matrix entries. No extra security or asymptotic speed claimed. Restriction of scalars and subring restriction are different.

## 4. Highest-value external lead

Radical Ring-LWR: https://ia.cr/2026/2078
Abstract mirror: https://eprint.iacr.asia/2026/2078
Foundational radical rings: https://eprint.iacr.org/2026/098
Reference implementation: https://github.com/NXP-Research/RadicalRingLWR/tree/ba0a745d360455fa9a1e29f086b307e4516de4aa

The abstract claims a distortion bound for error sampling and relates its KEM and signatures to radical-ring assumptions. Reference README explicitly says power-of-two moduli and optional NTT implementation. Verify actual ring definition, embedding, distribution, reduction hypotheses, parameter selection, and failure/rejection bounds from the paper and code. Our toy q12289 differs from their power-of-two moduli. Do not assume an NTT over their actual coefficient modulus; auxiliary primes/CRT or other multiplication algorithms may be necessary. Treat preprint results as claims until audited.

## 5. Requested output, in priority order

A. Audit our theorems. Find missing hypotheses, indexing mistakes, distribution changes, or counterexamples. In particular inspect the prefix recurrence, omitted-coordinate collision, and exact subring equivalence. Check whether full scalar-lattice isometry extends to the relevant radical-ring canonical embedding and norm; coefficient permutations alone need not prove canonical-embedding isometry.

B. Read Radical Ring-LWR and derive its actual assumption/reduction chain. Separate decisional hardness, search/OWF hardness, collision resistance/SIS, KEM IND-CCA, and signature EUF-CMA. Explain how sample limits, rounding conventions, secret/error law, embedding distortion and q affect each link. State exactly which assumptions can be inherited by an implementation-only transform. Consider noncyclic rings, twisted/polyphase multiplication, full character-sector decomposition, symmetry-aware scheduling and known factorizations that retain the full secret. Look for a new provable construction or a tight no-go theorem, with novelty compared against standard NTT/CRT and radical-ring multiplication. If proposing an alternative, specify ring, public distribution, secret law, output, security game, reduction target and unresolved assumptions.

C. Give a minimal prospective falsification and benchmarking plan. Use synthetic inputs and matched parameters. Include primal/dual/hybrid attack models, exact toy checks, BKZ when available, and pinned estimator/cost model versions. An LWE estimator using rounded-error variance as an approximation is a heuristic for LWR; characterize error correlation/bias and sample bounds explicitly. Compare optimized native arithmetic as well as setup, memory, constant-time behavior, end-to-end encoding and key sizes. No numeric bit-security claim without justified inputs and model. Rank at most three candidates by proof prospects, implementation benefit, and strongest known obstruction. Recommend one next experiment.

D. Optional short QFT peer review using hardware section below. It is a separate mathematical/circuit contribution, not evidence for one-wayness.

Return a concise technical memo with evidence labels PROVED / COMPUTED / CONDITIONAL / OPEN; include exact source theorem/section references; preserve failed approaches. Target <=2500 output words, with a compact experiment specification. Do not regenerate the full earlier reports, burn tokens on generic cryptography explanations, propose production deployment, or run expensive hardware/training. Use one review pass.
