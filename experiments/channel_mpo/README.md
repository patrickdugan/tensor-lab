# QC-01: Kraus rank versus noisy-circuit MPO compressibility

A small dense-reference benchmark for a specific question prompted by the
Stinespring/Kraus construction: **does a channel's minimal environment size
predict the cost of representing the resulting many-qubit state?**

**Verdict (small-system evidence):** Kraus/Choi rank and the spatial operator
Schmidt ranks of the output density matrix measure different structure.
Correlations across cuts and noise contraction control the observed MPO
compression. Strong depolarizing noise sharply compresses the output; dephasing
has a non-monotonic dependence on its phase-flip probability. These numbers
are specific to a six-qubit brickwork pilot.

## Exact distinctions

For a local channel E(rho) = sum_a K_a rho K_a^dagger, the **minimal** pure
Stinespring environment dimension is rank(J_E), the rank of its Choi matrix,
which equals the rank of the matrix with columns vec(K_a). This is a
*single-channel* quantity, independent of circuit layout.

An n-qubit density matrix, rearranged as an order-n tensor with local physical
indices `(ket_i, bra_i)`, has an MPO bond across cut j equal to the *operator
Schmidt rank* across that cut when represented exactly. Approximate bond
requirements depend on the singular-value tails and a chosen error norm.

Counterexamples implemented and tested:

- A Bell-state projector is created from |+0> by a two-qubit CNOT, a **rank-one
  unitary channel**, yet its spatial operator Schmidt rank is **4**.
- I/2 tensor I/2 has exact MPO bond **1** even though the **fully
  depolarizing** single-qubit channel has Choi rank **4**.

These are rank comparisons, not a general simulation-speed reduction. The
reference constructs the full dense density matrix before compressing.

## Method

- n = 6; initial state |000000>; depth 1-6 of alternating nearest-neighbor
  brickwork gates sampled by seeded complex-Gaussian QR (Haar U(4)).
- Three seeds, `0,1,2`. The gates are **identical across noise models**.
- After *every* layer, noise independently acts on all six qubits.
- Channels: depolarizing `(1-p)rho + p I/2`, phase flip `(1-p)rho + p ZrhoZ`,
  amplitude damping with excited-state decay probability p.
- p in `0, .05, .2, .5, .9`. For phase flip, p=.5 is **complete dephasing**;
  p=1 is the coherent unitary Z channel, so increasing p past .5 can restore
  operator correlations.
- TT-SVD rearranges the density matrix into interleaved local `(ket,bra)`
  indices and assigns each cut `epsilon^2 ||rho||_F^2/(n-1)` discarded-square
  budget. The resulting **relative Frobenius error** is <= epsilon (here .01).
- We record the number of complex entries in the truncated MPO, maximum bond,
  actual relative Frobenius error, trace-norm distance `||rho-rho_MPO||_1/2`,
  trace drift, and minimum eigenvalue of the Hermitian MPO reconstruction.

**Caveat:** Frobenius closeness does not guarantee small error in every quantum
observable as n grows. TT-SVD can violate positivity or trace preservation,
which is why trace-norm, trace, and minimum eigenvalue appear in the output.
The channel implementation itself is CPTP.

## Reproduce

```bash
python -m pip install 'numpy>=1.24'
cd experiments/channel_mpo
OPENBLAS_NUM_THREADS=1 python -m unittest -v test_channel_mpo.py
OPENBLAS_NUM_THREADS=1 python channel_mpo.py --qubits 6 --depth 6 --seeds 3 \
  --relative-tol 0.01 --output results.csv
```

Python >= 3.10. The experiment needs no GPU or quantum SDK. The CPU script
uses NumPy and `unittest`; `results.csv` is the frozen pilot.

## Depth-six results (average over 3 seeds)

| Local channel | p | Choi rank | Maximum MPO bond | MPO complex entries | Mean relative Frobenius error |
|:--|--:|--:|--:|--:|--:|
| Depolarizing | 0 | 1 | 55.3 | 7627 | .00428 |
| Depolarizing | .05 | 4 | 55.3 | 7627 | .00400 |
| Depolarizing | .2 | 4 | 23.3 | 3451 | .00512 |
| Depolarizing | .5 | 4 | 1.0 | 24 | .00320 |
| Depolarizing | .9 | 4 | 1.0 | 24 | < .000001 |
| Phase flip | .05 | 2 | 54.3 | 7499 | .00396 |
| Phase flip | .2 | 2 | 12.0 | 1497 | .00692 |
| Phase flip | .5 | 2 | 2.0 | 52 | .00351 |
| Phase flip | .9 | 2 | 42.0 | 5920 | .00435 |
| Amplitude damping | .05 | 2 | 56.0 | 7712 | .00399 |
| Amplitude damping | .2 | 2 | 41.3 | 5835 | .00427 |
| Amplitude damping | .5 | 2 | 8.3 | 527 | .00578 |
| Amplitude damping | .9 | 2 | 3.7 | 61 | .00453 |

The dense 6-qubit density matrix contains 4^6 = **4096 complex entries**. The
uncompressed/weak-noise MPO can therefore cost *more* storage than dense.
At depth six, the largest observed trace-norm distance across all grid cells
was < .01, the largest trace drift was < .007, and the most negative MPO
eigenvalue across all depths was about -0.00422. No physicality guarantee is
claimed for TT-SVD alone.

## Research significance / next gates

1. **Actual scalable MPO engine:** update cores directly with local gates and
   Kraus channels (ideally a locally purified/CP-preserving method); benchmark
   CPU time and memory against the present dense baseline at 8-20 qubits.
2. **Error contract:** track trace-distance or rigorous local-observable error
   in addition to relative Hilbert-Schmidt error, including when rho is near
   maximally mixed. Test positivity repair and truncation-induced bias.
3. **Architectures:** compare nearest-neighbor with long-range and correlated
   noise, and distinguish average bond from worst-cut bond. Repeat over larger
   seed grids and depth/n sweeps.
4. **Compactification stress test:** use operator-Schmidt singular-value tails
   across gate/noise layers as a measurable contraction-gate diagnostic. Choi
   rank alone is insufficient for an asymptotic complexity claim.

## Relevant literature

- Y. Shao et al., *Complexity of tensor network simulation for noisy quantum
  circuits*, arXiv:2606.00474 (v2, November 4, 2026): polynomial/constant
  bond results under specified local-noise contractivity regimes.
  https://arxiv.org/abs/2606.00474
- A. Müller, T. Ayral, C. Bertrand, *Enabling large-depth simulation of noisy
  quantum circuits with positive tensor networks*, arXiv:2403.00152:
  locally purified MPDOs and environment-basis optimization.
  https://arxiv.org/abs/2403.00152
- E. Granet, K. Hémery, H. Dreyer, *Analog simulation of noisy quantum
  circuits*, Phys. Rev. Research 7, 013213 (2025): low-variance trajectory
  methods for selected noise channels.
  https://doi.org/10.1103/PhysRevResearch.7.013213
