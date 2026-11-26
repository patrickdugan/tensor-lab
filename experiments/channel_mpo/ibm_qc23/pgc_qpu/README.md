# PGC-QPU01: native channel observability and Ramsey augmentation

Base: 4cb948f58195fe837caf3000a067a2901584097a. Zero additional QPU seconds. This is a retrospective local sensitivity calculation on the actual eight Kingston width-8 depth-1/2 fit circuits and a proposed ideal Ramsey supplement. No refit was performed. All 64 measured expectations were reconstructed from archived bitstring counts; the C3 predictions were replayed with the existing native engine. The fitted parameters and model family were selected on historical data; no fresh hardware claim follows.

## Native local sensitivity

Central differences at 0.2 and 0.1 mrad, fixed calibrated stochastic/relaxation model. The restricted model has two SX and four ZZ terms; the expanded model adds two endpoint-local Z terms on each of the four edges. Each column is measured in radians. The Gram matrix is diagonalized with a symmetric Jacobi iteration; numerical rank uses a relative singular-value threshold of 1e-6. This is unweighted marginal sensitivity, not a joint-bitstring likelihood or a global identifiability proof. Near-zero singular values are subject to roundoff. Finite differences at bound values are intentionally two-sided to diagnose the unconstrained model.

| Rows | Restricted rank | Expanded rank | Restricted condition | Expanded condition |
| --- | ---: | ---: | ---: | ---: |
| all 64 | 6/6 | 12/14 | 36.219 | singular |
| 16 target rows | 4/6 | 6/14 | singular | singular |

Max C3 replay gap 8.881784197001252e-16; measured gap 0; max derivative step-halving gap 3.2812363937040345e-8. The receipt retains row Jacobians, spectra, parameter ordering, column norms and weakest right vectors. The l1 ledger bounds truncation in exact arithmetic; floating-point error is separate. Rows within one circuit share the shot record, so Fisher inference would need a joint or correlated likelihood. The fit-only slice excludes larger circuits, and the restricted parameter family excludes SPAM, drift, crosstalk and other physical effects. A local full-rank result cannot identify the physical cause globally.

## PGC observer design: resolve exact aliases

QC26A probes the target while the control sits in a Z eigenstate. The diagonal coherent error is exp(-i(alpha_c Z_c + alpha_t Z_t + delta Z_c Z_t)/2). For control bit b, target phase is m(alpha_t + (-1)^b delta + b*pi), with the known ideal CZ phase retained. The control's local Z term is globally invisible to that experiment. Even repeat counts admit the exact alias (alpha_t,delta) -> (alpha_t+pi/2,delta+pi/2). The supplement targets the two largest absolute fitted ZZ edges, 16_23 and 5_6: add odd m=1 in both quadratures and both control states; reverse roles and include m=0,1,2. All 32 proposed additional circuits use 384 shots. The analytic ideal model raises phase-Jacobian rank from two to three for each edge and separates the two specified aliases. This does not exclude all global periodic equivalences or non-diagonal and SPAM confounding. The m=0 references should be collected in the same session. Use a separate independently shuffled replicate to establish a nonzero hardware effect; never fit new diagnostic parameters to the old kicked-Ising counts.

The proposed extra allocation estimate is 6.39488 s by the existing simple QC26A allowances. No current balance, live backend target, compile or owner authorization was checked. Use the existing freshness and authorization gates. The rows in receipt.json are a logical manifest, not a submit-ready artifact. An additional independent acquisition is necessary for prospective evaluation under QC26B.

## Reproduce

From this directory: `node audit.mjs --check`. CI replays the eight circuits with the existing engine and verifies QASM hashes. The initial receipt was produced in a JS sandbox using the pinned engine and fetched text; the Node filesystem and hash replay is a separate CI gate. The two phases of the experiment use the finite observation-fiber principle from [MathCan Pregeometry](https://github.com/patrickdugan/MathCan/tree/379bccdf6ba613307441fa11fb464ca7f455ba58/MathCan/Pregeometry). The observed record and physical model class are declared. No claim of emergent spacetime.
