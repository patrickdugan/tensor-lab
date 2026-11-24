# 07: What the November 2026 Riemann and VP != VNP claims mean for Bitcoin

Written 2026-11-24. Evidence level: **press and aggregator summaries of two
unreviewed claims.** The primary papers and Lean repositories were not read for
this note. Everything under "Claims" is as reported, not independently checked.
The author of this note is an Anthropic model, and one of the claims is from
Anthropic, so it is not an independent party on that claim.

## Claims (as reported)

| Claim | Source of claim | Status on 2026-11-24 |
|---|---|---|
| Riemann hypothesis proved | Levent Alpoge (Anthropic) with Claude, announced 2026-11-20; Lean proof, no human-readable proof yet | Lean statement and proof reported correct by the Lean/Mathlib community (2026-11-21); no natural-language proof; Clay still lists the problem open |
| VP != VNP over every field | Anthropic repository, 70-page paper credited to Claude, Lean 4 formalization (324 files, 72,932 lines); a competing OpenAI repository claims the same for exact circuits | README says no mathematician has checked it; Lean build passes, which certifies the formal statement, not that the statement matches Valiant's conjecture |

Reported scope limit: the VP != VNP paper says it does not settle P vs NP, and
an algebraic separation is not known to imply a Boolean one.

## Consequences for Bitcoin, assuming both claims are correct

**VP != VNP: no effect.** It is a lower bound for arithmetic circuits over a
field. Bitcoin's assumptions are Boolean and concrete: hardness of the elliptic
curve discrete log on secp256k1 (ECDSA, Schnorr/Taproot) and preimage and
collision resistance of SHA-256 and RIPEMD-160. None of these is known to be
NP-complete. Even P != NP would not give one-way functions, and VP != VNP does
not give P != NP. It neither breaks nor secures anything Bitcoin uses.

**Riemann hypothesis: no effect on the signature or hash layer.** RH sharpens
the error terms in prime counting and prime gaps. The consequences are for
number-theoretic algorithms: Miller's primality test becomes deterministic
under GRH, and bounds on least non-residues and class groups tighten. It does
not give a factoring algorithm or an ECDLP algorithm. Bitcoin does not use
RSA, and its field prime is fixed.

**What does matter, in order:**

1. **Quantum (Shor) on secp256k1** is the concrete threat to ECDSA and Schnorr.
   Grover against SHA-256 is only quadratic, so hashing stays near 128-bit
   strength and Bitcoin's hash layer is not the issue. The exposed surface is
   outputs whose public key is already on chain: P2PK, reused addresses, and
   Taproot key-path outputs. Hashed, never-reused addresses expose the key only
   at spend time. Migration proposals (post-quantum output types, for example
   BIP-360) are the relevant policy track.
2. **The capability signal.** If an AI system can settle RH and VP != VNP in
   Lean, the realistic update is that machine-found cryptanalysis (new attacks
   on ECDLP, SHA-256 reduced rounds, implementation flaws) is more plausible
   than it was. That is a statement about search power, not about these two
   theorems.
3. **Verification culture.** Lean-checked proofs change what "unverified"
   means, but the statement-fidelity gap remains until specialists confirm the
   formal statements match the intended problems.

## Link to the IBM work (QC-23)

QC-23 measures how well a noise model predicts Clifford-echo circuits on real
IBM hardware. It does not test any cryptographic claim. Its relevance is
indirect: measured two-qubit error and coherence data are the inputs to
resource estimates for when fault-tolerant Shor on a 256-bit curve becomes
feasible. That is the number to track for Bitcoin, and it does not depend on RH
or VP vs VNP.

Hardware status on 2026-11-24: **blocked on credentials.** `~/.qiskit/qiskit-ibm.json`
exists but holds no saved account (the file was checked for account names and
channels only; no values were printed). `qc23_ibm.py discover` fails with
`AccountNotFoundError` for channel `ibm_quantum_platform`. QPU time used: 0 s.
The pinned QC-23 requirements install and import cleanly (qiskit-ibm-runtime
0.52.0).

To unblock, from `experiments/channel_mpo/ibm_qc23` with the API key in a file
(token never touches a log):

```powershell
python -c "import pathlib,sys; from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(token=pathlib.Path(sys.argv[1]).read_text().strip(), channel='ibm_quantum_platform', overwrite=True)" PATH_TO_KEY_FILE
python qc23_ibm.py discover --out results/hw_discovery
python qc23_ibm.py build --backend BACKEND --out results/ibm_build
```

Then write `authorization.json` (account owner only, per the QC-23 README) and
run `submit`.

## What would change this assessment

* A human-readable RH proof showing a new method that touches elliptic curve
  L-functions or point counting over finite fields: reassess ECDLP.
  (Standard RH-type results give no attack by themselves.)
* A Boolean P != NP or one-way function result: still would not break
  Bitcoin, but would clarify what assumptions remain.
* Published fault-tolerant resource estimates for secp256k1 dropping into the
  range of announced QPU roadmaps: act on migration, independent of the above.
