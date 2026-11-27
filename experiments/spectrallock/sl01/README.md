# SpectralLock-01: prefix reconstruction obstructions in valuation-shell cryptography

Research prototype 2026-11-27. Synthetic secrets only. No production security parameters, live-system attack, formal verification, QPU, GPU, or Fable. Related to HAD-QFT-01/04; separate branch from its open paper PR. Both rounded constructions fail as one-way candidates.

Let ell be prime, N=ell^n; sign s_j constant on the additive-order ell^j shell. Ramanujan sums give P_0=s0, t_j=ell^(j-1), F_j=P_(j-1)-t_j s_j, P_j=P_(j-1)+(ell-1)t_j s_j, and DC=P_n. Natural-order unnormalized Fourier value matrix T has T^2=N I by inversion symmetry. Code reverses frequency order to process from highest additive order to DC. This is integer arithmetic on shell values, not a quantum-normalized amplitude transformation. See https://eprint.iacr.org/2015/1186 for the classical Ramanujan identity and https://eprint.iacr.org/2015/769 for LWR hardness (which does not apply automatically to the new distribution).

## Diagonal rounding theorem

Define R_B(z)=floor(B*(z mod q)/q), q prime >4N. Output R_B(a_j F_j) with independent uniform nonzero a_j, and similarly DC. For a target keep one prefix witness per reachable integer sum; all future checks depend only on that sum. Each level at most two transitions match the exact planted F_j because the next sign fixes the predecessor. A nonzero coefficient difference is uniform nonzero after the public multiplier; same rounding bin requires difference in +/-{1,...,ceil(q/B)-1}. Thus alpha<=2 floor((q-1)/B)/(q-1)<=2/B. Conditioning on earlier multipliers gives E K_j<=2+2 alpha E K_(j-1), K_0=2; expected <=4n/(1-4/B) prefix checks for B>4. No uniform bound proved here for B4. This is a count of arithmetic operations, not unit bit-cost or worst-case time; witness tuples can be replaced by parent pointers.

1536 planted cases, ell=2,3, n15,63,127,255, B4,8,16, 64 roots/cell. All valid preimages, 1501 equal original, 446434 prefix checks. At n255 median/max checks B4 774/890, B8 598/658, B16 550/584. Eight split primes q=kN+1 certified by Pocklington with F=N>sqrt(q), root order verified.

## Block-causal rounding theorem

At each level mix k parallel Fourier channels by independent uniform m-by-k public matrices over F_q, round each output. A nonzero vector difference has uniform scalar inner product with each independent matrix row. beta=(1+2 floor((q-1)/B))/q bounds same-bin probability. At most 2^k exactly matching transitions, so E K_j <= 2^k + 2^k beta^m E K_(j-1). For fixed k,m,B with 2^k beta^m<1, expected O(n) transition checks; may be exponential in k. No result for dense cross-level coupling.

192 planted cases, ell2,3; n63,127,255; k=m=2,4; 16 roots/cell; B8. All valid preimages, 191 original, 322504 checks. k4,n255: 1024 signs, 32/32 preimages, median4400 and max4672 checks, peak16 states. Both rounding designs rejected.

## Power spectra

One binary row, full exact squared spectrum: with initial sign fixed, each next sign is determined when P!=0 by the difference of squared candidates 4Pt. For odd ell prefix mass is odd, so no zero. For ell2 zero occurs only at the first sign deviation (next weight equals the preceding total weight), never again. If that zero precedes another step, complementary suffixes yield the same power and are the sole additional ambiguity. Thus 2^(n-1)+1 spectra at ell2, multiplicities2 or4, and 2^n spectra for odd ell, multiplicity2. This is a linear-step preimage algorithm. Example (1,-1,1,1) and (1,-1,-1,-1). Exact exhaustive counts for ell2,3,5 n1..12.

For k rows require sum P_f x_f = (sum P_f^2 + k t^2 - y_j)/(2t). Enumerate sign counts among equal nonnegative prefix sums, normalize each updated row sign and sort, keep one witness per absolute prefix vector. This preserves future feasibility and past power. 192 planted cases ell2,3, n15,31,63, k1,4,8,12, eight roots/cell. k1/4/8: each48/48 hits; k12:24/48 hits,24 capped. Cap5000 accepted transitions, not all sign-search nodes. No security inference from capped instances. No result for partial spectra, other symmetry families or growing k.

## Reproduce

Python3.10+, NumPy. Run python run.py, python run_block.py, python audit_exact.py. 768 direct FFT coefficients; 1920 exhaustive rounded targets; 482 exhaustive/corrupted multichannel targets; 640 exhaustive block-rounded targets; 4491 exact integer involution checks; 36 exact collision-count families; 121 field-column bijection checks; eight prime/root certificates. Seeded PRNG is for reproducibility only. Complete per-case summaries regenerate from source. Artifact SHA256 e65d7d2574064a0c081381cb80709e783a4319fdb2bff51f3a88646894a8b63e. Receipt SHA256 AUDIT.json 279315ea6a2a486e92e6628c025d1049d2fe925eba641b02553fd9a2f1de78ee; BLOCK_AUDIT.json 8b70a55467e746ca0595f76ac901b92a1b4508325a134685527e86bc6ce251d9; EXACT_AUDIT.json 74b8eedfabdaff33fe818882914184c1d6a71096d861b8a6810690a3ca1f5ba4.

Next test noncausal cross-shell mixing against standard dense LWR and Module-LWR; require a defensible reduction before security claims. A truly uniform matrix A remains uniform under invertible T, so R_B(T A s) is a reparameterization of standard LWR, with no automatic speed or security gain. For a near-term ZK application retain reviewed commitments and prove the spectral equations separately. Do not use these experiments to secure funds or credentials.
