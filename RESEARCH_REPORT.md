# Tensor Degeneration Discovery Lab — research report

Date: 2026-10-12.  Code: `tensor-lab/` (Python 3.12, no third-party dependencies
except `numpy` for two floating-point side experiments; Lean 4 v4.34.1 + Mathlib
`d13f23b7`).  Source material: OpenAI, *An Upper Bound of 9/4 for the Matrix
Multiplication Exponent* (October 2, 2026; `reference/`), and the Lean modules under
`OAI/LinearAlgebra/MatrixMultiplication/AuxiliarySeparation/` of `openai/math`
(local clone at commit `adc7f124`).

## Summary

1. **Reproduced exactly.** The three-sector construction (paper Lemma 4.2) is
   certified for all `a, h ≤ 5` (25 certificates) by an independent verifier, and
   in Lean for four cases.  The finite Fourier separation (Proposition 3.1) is
   reproduced in exact cyclotomic arithmetic `Q(ζ_15)` for six cases.
2. **Rediscovered automatically.** A bounded search over sector degenerations
   (1.84 million structures for `a ≤ 5, B ≤ 16`; 3.28 million four-sector
   structures for `a ≤ 4, B ≤ 10`) rediscovers every instance of the paper's
   construction inside its bounds (17 of 17, each re-certified).
3. **No improvement found, and a reason why.** None of the 5 381 + 1 723 distinct
   inequalities from the two searches, the 17 666 staircase instances or the 342
   rectangle packings beats the published bound.  Three results explain the negative outcome:
   * the paper's two inequalities determine the diagonal *exactly* — the
     extremal profile `P*` satisfies all of them and its diagonal equals the
     paper's lower bound (**Lean**);
   * the exponent is governed by one number, the overhead `θ` of a sector
     packing: `P(a,a) ≥ a^{2/(1+θ)}`, `ω ≤ 3(1+θ)/2`; the paper has `θ = 1/2`
     (**Lean**);
   * any degeneration of a single `C(a,B)` into a shared-leg sum of convolution
     blocks whose shared leg carries an input has `θ ≥ 1/2` and cannot beat `P*`
     (**Lean** for the scalar part, a short flattening-rank argument for the link).
4. **The open door, examined (follow-up).** The only blocks the counting argument
   misses are "rectangles" (a convolution with its *output* on the shared leg); a
   dense packing would raise the growth exponent to about `1.457` (`ω ≲ 2.06`,
   heuristic).  The follow-up (`notes/05_c316_dense_rectangles.md`) settles the
   exact version: by Koszul flattenings, `C(3,B)` degenerates to `n` rectangles
   `C(2,2)^σ` sharing X **iff `B ≥ 3n-1`**, so the monomial staircase is optimal
   among all degenerations and `C(3,16)` holds 5 rectangles, not 8 (proved; same
   threshold computed for `3×3` and `4×4` rectangles).  For `a = 3` a Koszul LP
   shows that no exact packing of *any* convolution blocks beats `P*`.  What
   remains is the asymptotic version — equivalent, by Strassen's spectral theorem,
   to the character inequality itself — which no known spectral point refutes; the
   cheapest dense construction on one copy level is a restriction from two copies
   of `C(3,2n)`, and block-label Fourier filters cannot amortise that factor.
5. **The asymptotic restriction, attacked (second follow-up, `notes/06_asymptotic_restriction.md`).**
   The copy cost *does* amortise on tensor powers.  **Theorem D (exact
   degeneration certificate):** for every `n` checked (`2 ≤ n ≤ 16`), `T_n^{⊗2}` is a
   degeneration of **two** copies of `C(3,2n)^{⊗2}` (the *parity-twisted pair*), so
   `λ(T_n) ≤ √2 λ(C(3,2n))` for every character; for `C(3,16)` the per-factor cost
   drops from 2 to `√2`.  It is verified as an `ε`-polynomial identity on the block
   form and independently from the convolution tensor (`n ≤ 12`), with negative
   controls.  **Theorem B:** three copies give an exact *restriction* (`√3`).
   **Theorem A:** a different two-copy construction for `C(3,4)`.  **Theorem C:**
   the first-order mechanism with non-vanishing factors has forced zeros beyond
   that square; Theorem D evades it with factors that vanish on `x = 1` and
   parity-twisted base points.  *Correction:* an earlier version of this report
   said, from random-start continuation, that two copies fail for `n ≥ 3`; the
   basin of Theorem D is narrow (found by 2 of 60 starts for `n = 3`), and the
   claim is withdrawn.  The exponent needs cost 1, so the asymptotic restriction
   stays open.  Calibration: the n-independent coarse (cyclic) version — `T₂`
   inside the rank-4 tensor `Cyc₃` — cannot reach cost 1 without proving
   `R̃(T₂) ≤ 4`, a case of Strassen's asymptotic rank conjecture; the
   parity-twisted pair indeed has no solution there, so the path boundary of
   `C(3,2n)` is what the construction uses.

## A. Working code

| command | what it does |
|---|---|
| `python -m tensorlab reproduce` | three-sector certificates for `a, h ≤ 5`, each independently verified |
| `python -m tensorlab verify <cert.json>…` | independent verifier (does not import the construction code) |
| `python -m tensorlab search --max-a 5 --max-b 16` | bounded exhaustive sector search, canonicalised, ranked, top results certified |
| `python -m tensorlab fourier --cases 2,1 3,2` | finite separation, exact in `Q(ζ_15)` |
| `python -m tensorlab growth --grid 60 --lp-grid 12 --diag 2 3 4 5 --whatif` | inequality engine: extremal profile, exact LP minima, candidate implication, what-if recurrence |
| `python -m tensorlab realize --packing 3,15 4,14 5,12` | realizability by general monomial weights (exact LP + Farkas certificates) |
| `python -m tensorlab spectral` | support-functional test of dense rectangle packings (floating point) |
| `python -m tensorlab dense --n 8 --B 16` | exact verdict on dense rectangle packings: Koszul ranks, border ranks, staircase optimality, two-copy restriction |
| `python -m tensorlab report` | regenerates `catalogue/CATALOGUE.md` and `logs/report.md` |

Tests: `python -m pytest -q` (54 tests, about one minute), including tampering
tests that the verifier must reject.

Design points:

* All verification is exact (`int`, `Fraction`, cyclotomic polynomials).  The only
  floating-point modules are `spectral.py` and `numeric_restriction.py`, and their
  outputs are reported as numerical evidence only.
* `verify.py` rebuilds the source from its family specification with its own code,
  recomputes weights and the minimum-weight set, and pulls every block back along
  the certificate's coordinate maps.
* Block identification is exact and linear-time for sub-tensors of a convolution
  (full product on two legs with arithmetic-progression index sets — the equality
  case of `|A+B| ≥ |A|+|B|-1`), with a budgeted backtracking fallback for other tensors.
* `weights_lp.py` decides realizability of a block configuration by *arbitrary*
  per-coordinate weights on all three legs and returns either integer weights or a
  Farkas obstruction; both are checked independently and exactly one always exists.

## B. Reproduction certificate

`certificates/three_sector_a{a}_h{h}.json` (25 files, all PASS) and
`notes/01_reproduction.md`.  Each certificate checks every source coefficient,
the weights (`+1` middle Y, `-1` middle Z), that the retained set is exactly the
minimum-weight set (weight 0), that every erased term has weight exactly `+1`,
the absence of cross-sector terms, and explicit coordinate maps for the three
blocks — identity, the middle block with legs Y and Z exchanged and X reversed
(`u ↦ a-1-u`), and the translation by `2h+a-1` — with the full shared X-leg kept.
Figure 1 of the paper (`a = 2, h = 1`, erased `x_0y_1z_1`, `x_1y_2z_3`) is a test.

Scope: finite identities for the listed parameters; the general statement is
OpenAI's Lean theorem.  The character inequality needs the paper's Lemma 2.3
(characteristic zero) and Corollary 3.2.  The Fourier separation
(`certificates/fourier_separation_*.json`) needs a primitive 15th root of unity
with 15 invertible; the weight bookkeeping is field-independent.

## C. Candidate catalogue

`catalogue/CATALOGUE.md` (generated), backed by `catalogue/search_*.json`.

| search | structures | distinct inequalities | non-dominated | negative slack on `P*` |
|---|---|---|---|---|
| `a ≤ 5, B ≤ 16`, ≤ 3 sectors | 1 842 160 | 5 381 | 956 | **0** |
| `a ≤ 4, B ≤ 10`, ≤ 4 sectors | 3 279 354 | 1 723 | 373 | **0** |

Classes: tight members of the paper's alternating family (155 non-dominated),
packings containing a rectangle block (85), wasteful alternating packings (333),
restriction-type packings (196), trivial dot-product splittings (187).
Duplicates are removed by canonical (source, block multiset) keys; *dominated*
entries (padding, a dropped or shrunk block) are marked.  Rejected candidates
with their evidence: double staircases and interleaved rectangle classes
(Farkas certificates), and every packing above `P*` in E5.

**Strongest new candidate.**  No candidate improves the bound.  The most
interesting *new valid* family is the rectangle staircase
`P(a, (n-1)(p+q-1)+p) ≥ n P(p,q)` (`p+q-1 ≤ a`), whose blocks have their output leg on
the shared leg — a block orientation the paper does not use.  It is certified
(`sectors.staircase_certificate`) but has positive slack on `P*` in all 17 666
instances with `B ≤ 40`.  The most valuable *new statement* is the counting
obstruction (D, O2), which identifies rectangles as the only possible source of
improvement.

## D. Formal proof artefacts (Lean)

Build: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/<path>/tensor-lab/lean/wsl_build.sh`
(mirrors `lean/` to `~/work/tensorlab-lean`, reuses the pinned Mathlib by symlink).
No `sorry`, `admit`, custom axioms or `native_decide`; `lean/AxiomCheck.lean`
prints the axioms of all 31 listed theorems (only `propext`, `Classical.choice`,
`Quot.sound` occur; see `logs/lean_axioms_direct.log`).  `lake build` succeeded
for `Finite`, `Obstruction`, `ExponentTransfer` and `Extremal`; `ThetaGrowth` was
compiled with `lean/direct_check.sh` because `lake` became very slow on this
machine late in the session, so the next `wsl_build.sh` run will rebuild the root
module once to include it.

| module | theorem | statement |
|---|---|---|
| `Finite` | `retained_2_1`, `retained_3_2`, `retained_2_3`, `retained_4_1` | weight-zero part of `C(a,3h+a-1)` = the three branches (finite, `decide`) |
| `Finite` | `weights_*`, `middle_*` | weights are 0 or 1 on the support; the middle branch is `C(a,h)` with Y, Z exchanged |
| `Obstruction` | `odd_sector_implied` | `(2e+1)P(a,h) ≤ P(a,(2e+1)h+e(a-1))` follows from row concavity + shifted tripling |
| `Extremal` | `extremal_feasible`, `P_diag` | `P*` satisfies positivity, symmetry, `P(1,b)=b`, concavity, tripling; `P*(a,a) = (3a-1)/2 · H_a` |
| `Extremal` | `counting_obstruction` | input-on-X blocks using ≤ `2B+a-1` coordinates have `Σ P* ≤ P*(a,B)` |
| `Extremal` | `rectangle_exceeds` | the per-coordinate bound fails for a rectangle block |
| `ThetaGrowth` | `theta_growth` | overhead-`θ` slope bound + diagonal step ⟹ `D_a ≥ a^{2/(1+θ)}` |
| `ExponentTransfer` | `exponent_transfer`, `omega_bound_of_growth` | `a^β ≤ D_a ≤ (2a-1)^{1/t}` ⟹ `t ≤ 1/β` ⟹ `ω ≤ 3/β` |

Semantic notes and remaining obligations:

* `Finite` works with integer coefficient functions on `Fin` types; it is a finite
  restatement, not a link to OpenAI's `Tensor` type.
* `Extremal` matches OpenAI's `ScalarProfile` index conventions field by field
  (concavity with `h - 1`, tripling with `3h + a - 1`); the rank bound is not claimed.
* `counting_obstruction` takes the coordinate count as a hypothesis.  Discharging it
  for an actual degeneration is the flattening-rank argument in
  `notes/03_obstructions.md` (not formalised; it needs lower semicontinuity of
  matrix rank under degeneration).
* `theta_growth` takes the slope bound `D_a ≤ (a + θ(a-1)) g_a` as a hypothesis.  For
  3-sector constructions with `2θ(a-1) ∈ ℕ` it is OpenAI's `profile_le_affine_slope`.
* `ExponentTransfer` assumes `ω ≤ 3t`, which is OpenAI's detecting-character step.

## E. Findings in detail

### Reproduced

Lemma 4.2 (three sectors), Figure 1, Proposition 3.1 (Fourier filter
`u - v + 2(g-h) = 0` and weights collapsing to `(g-h)²`), and the scalar growth
argument of Lemma 5.1 (`H_a³ ≥ a` exactly for `a ≤ 80`, diagonal bound
`(3a-1)/2 · H_a`).

### Verified computationally (exact)

* 25 three-sector certificates, 17 rediscovered by search, 15 class representatives
  and 6 Fourier separations, all PASS.
* `P*` violates no paper constraint on `[1,60]²`; the exact LP minimum of the
  diagonal on `[1,12]²` equals `P*(a,a)` for `a ≤ 4`.
* `P*` satisfies all 7 568 Kronecker-substitution inequalities inside `[1,40]²`
  (3 357 with equality), so multiplicativity adds nothing.
* Realizability with general weights (`logs/realizability.json`): staircase spacing
  threshold `δ ≥ p+q-1` in 14 cases; two-class and double-staircase packings
  obstructed; 342 packings above `P*` for `a ≤ 5`, `B ≤ 18`, of which 320 tested and
  none realizable.  Every claim carries a checked certificate.

### Counterexamples and failed approaches (see `notes/03_obstructions.md`)

* More sectors (odd-sector family): implied, Lean.
* Any input-on-X convolution blocks, any degeneration: blocked by counting, Lean.
* Rectangles via monomial weights: blocked by Farkas certificates; via *any* exact degeneration:
  blocked by Koszul flattenings (staircase optimal, `notes/05`).
* Kronecker substitution, Clebsch–Gordan generalisations of concavity: add nothing.
* Restriction (not degeneration) to dense rectangles at the smallest size:
  `C(3,4) → C(2,2)^σ ⊗ B_X(2)` does not exist (numerical residual stalls at exactly 1;
  short proof: the X-map must be invertible and the kernel of the Z-map would have to
  be divisible by every quadratic on a conic).  Degenerations are not excluded.

### The theory in one line

Within shared-leg degenerations of one convolution tensor, the exponent equals
`3(1+θ)/2` with `θ` the packing overhead; counting forces `θ ≥ 1/2` for every
block orientation except rectangles, which monomial weights cannot pack densely.

## Next experiments

The original recommendation — decide whether a non-monomial degeneration packs
rectangles densely, starting at `C(3,16)` — has been carried out
(`notes/05_c316_dense_rectangles.md`, `python -m tensorlab dense`):

| statement | status |
|---|---|
| `C(3,B) ⇝ n` rectangles `C(2,2)^σ` sharing X | **iff `B ≥ 3n-1`** (Koszul rank `2B+2` vs `6n`; staircase certificate) |
| `C(3,16) ⇝ 8` rectangles | **impossible** (also border rank `18 < 24`) |
| `C(5,B)`, `C(7,B)` with `3×3`, `4×4` rectangles | threshold = staircase (exact computation) |
| any exact packing of convolution blocks in `C(3,B)` beating `P*` | **impossible** (Koszul LP, slope and `B ≤ 25`) |
| same for `a = 4, 5, 6` | Koszul LP leaves a 0.2–0.5% gap; open |
| `n` rectangles `≤ C(3,2n) ⊕ C(3,2n)` | **true**, exact restriction (sign trick) |
| squares: `T_n^{⊗2} ⊴ 2·C(3,2n)^{⊗2}`, every `n ≤ 16` (incl. `C(3,16)`) | **true**, exact degeneration (parity-twisted pair); cost `√2` per factor (`notes/06`, `python -m tensorlab amortize`) |
| squares: `T_n^{⊗2} ≤ 3·C(3,2n)^{⊗2}`, every `n` | **true**, exact restriction; cost `√3` |
| first-order two-copy amortisation with non-vanishing factors beyond `C(3,4)` | **impossible** (exact forced zeros) |
| same parity-twisted pair on the cyclic (coarse) version | **no solution** (exact); the path boundary is needed |
| asymptotic restriction `C(2,2)^σ ⊗ B_X(n) ≲ C(3,2n)` (the character inequality) | **open**; quantum functionals allow it; block-label Fourier filters + monomial weights cannot amortise the factor 2, signed diagonal schemes reach `√3` |

**Highest-value direction now: beat `√2`.**  The best bound for `C(3,16)` is
`λ(T₈) ≤ √2 λ(C(3,16))` (Theorem D).  Lowering the per-factor cost further needs
`c₄ ≤ 3` (cost `≈ 1.32`) or `c₃ ≤ 2`, by a mechanism in which the additive
corrections also act on X-slabs where some factor has `x = 1`; the two-copy
parity-twisted ansatz provably cannot do this at three factors.  The search method
that produced Theorem D — many-start rank-`c` continuation on a small `n`, read off
the sign patterns, exact linear solve for the corrections — is the tool to point
at `N = 3, 4`.  The coarse (cyclic) version is not the right proxy: cost 1 there
would prove `R̃(T₂) ≤ 4` (a case of Strassen's asymptotic rank conjecture), and
Theorem D's ansatz fails there while working on the path.
Exact degenerations fail by Koszul, block-label filters fail by Farkas, the
first-order mechanism fails by forced zeros, and known spectral points do not
refute the statement.  A cheaper parallel track is to point the Koszul
LP at other sources (tensor products `C(a,b) ⊗ C(a',b')`, cyclic convolutions) as
a fast filter before any construction search, and to test whether the small
`a ≥ 4` LP gaps are realisable.

## F. Next-generation agent architecture (optional, downstream)

Keep the exact engine as the single source of truth and put agents around it:

* **Generator** proposes block configurations and degeneration ansätze (sector
  structures, weight families, basis-change templates) as JSON specs.
* **Critic** runs the cheap filters first — slack on `P*`, the counting bound,
  support functionals — and rejects anything that cannot improve the bound before
  any expensive search.
* **Proof engineer** turns verified finite families into Lean statements (finite
  `decide` instances, then parametric lemmas modelled on `Extremal` and
  `ThetaGrowth`) and tracks the five-level status of each claim.
* **Experiment controller** schedules bounded searches, LP and Farkas runs,
  caches identifications by shape, and records failures with their certificates.

Every agent writes certificates, never conclusions; promotion of a claim to a
higher status level requires the corresponding verifier or Lean build to pass.

## Status ladder of the main claims

| claim | 1 finite identity | 2 general degeneration | 3 character inequality | 4 growth | 5 exponent |
|---|---|---|---|---|---|
| three-sector (paper) | yes (ours) | yes (OpenAI Lean) | yes (OpenAI) | yes (OpenAI) | yes (OpenAI) |
| odd-sector family | yes | argument | via Cor. 3.2 | implied, no gain (Lean) | no gain |
| rectangle staircase | yes | argument | via Cor. 3.2 | no gain (`P*`) | no gain |
| dense rectangle packing, exact | **impossible** (Koszul) | — | — | — | — |
| dense rectangle packing, two copies | yes (exact restriction) | yes | `λ(C) ≥ n^{p_X}λ(R)/2`, too weak | no gain | no gain |
| dense rectangle packing, three copies of the square | yes (exact restriction, all `n`) | yes | `λ(C) ≥ n^{p_X}λ(R)/√3`, too weak | no gain | no gain |
| dense rectangle packing, two copies of the square | yes (exact degeneration, `n ≤ 16`) | yes | `λ(C) ≥ n^{p_X}λ(R)/√2`, too weak | no gain | no gain |
| dense rectangle packing, asymptotic | — | **open** | open | conditional `≈ a^{1.457}` | conditional `ω ≲ 2.06` |
| θ-growth | — | — | — | Lean (scalar, conditional on θ) | Lean (with `ω ≤ 3t`) |

## Reproduce everything

```bash
cd tensor-lab
python -m pytest -q
python -m tensorlab reproduce --max-a 5 --max-h 5
python -m tensorlab fourier --cases 1,1 2,1 2,2 3,1 3,2 4,1
python -m tensorlab search --max-a 5 --max-b 16 --max-sectors 3
python -m tensorlab search --max-a 4 --max-b 10 --max-sectors 4
python -m tensorlab growth --grid 60 --lp-grid 12 --diag 2 3 4 5 --whatif
python -m tensorlab realize --packing 3,15 3,16 3,18 4,12 4,14 4,16 5,12 5,14 --max-lp 150
python -m tensorlab spectral --cases 3,2,2 5,3,3 7,4,4 --n 2 8 16 32
python -m tensorlab dense --p 2 --n 8 --B 16
python -m tensorlab amortize
python -m tensorlab report
```

Approximate times on this machine: searches 4 and 7 minutes, `realize` 10 minutes,
`spectral` 3 minutes, Lean build 15–30 minutes (the `Extremal` module alone about 11).
