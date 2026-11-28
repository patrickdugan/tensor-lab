# HAD-QFT-04: recursive valuation-shell QFTs

Research working draft, 2026-11-27. Extends the orbit/CRT work in tensor-lab PR #33 (baseline 499c563bc6260889e37b561b7482bc1670ed538c). The paper is a draft; mathematical identities are proved in the text, quantum numerical verification is CPU-only, novelty and physical resource savings remain to be established.

## Reproduce

Python 3.10+ with NumPy and SciPy. From this directory:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit.py
sha256sum -c SHA256SUMS
```

The checksum manifest covers frozen sources, manuscript, QASM and audit output, not wall-time logs. On this host the audit JSON and console receipt were byte-identical in two consecutive runs. To regenerate the paper: `pdflatex paper.tex` twice (LaTeX with amsmath, geometry, microtype, lmodern, hyperref, booktabs). PDF compilation may change embedded metadata, so verify source hashes separately. `AUDIT.json` records every test, all generated gate counts through b=12 and the separately simulated subset. The timings are descriptive, not hardware measurements.

## Main results

- Exact shell QFT for prime powers; adjacent rotations and a binary recursion using the truncated geometric row of the smaller product. For n+1=2^b, noanc CNOT count 5*2^b+b^2-5*b-6; clean count (b-1)*(23*b-38), with max(0,b-2) clean work qubits. Hybrid chooses minimum per recursion level. This is an encoded transform, not a general-purpose QFT speedup.
- Nontrivial character sectors transform by shell reversal, conjugate character and normalized Gauss phase. 490 nontrivial sectors, 769 FFT columns in 17 small prime-power settings. The mod4 two-sector coherent compiler uses 2b total wires, including clean work bits, and O(b^2) logical gates. 248 basis-column checks; direct FFT at N=8,128,32768.
- Uniform nested-subgroup hopping with linear shell eigenvalues; b phase gates between two shell transforms. Compared with a direct full Walsh-prefix circuit, not merely the full QFT. Eight small application settings, six alternating potential/hopping steps; max repeated amplitude error 1.222e-15. Classical simulation in O(n) amplitudes is possible. There is no quantum-advantage claim.
- Phase-ramp exclusion: the unit-invariant projection retains 1/orbit_size at an exact Fourier grid frequency. 237 numerical checks; standard phase estimation is not generally compressible by this method.

## Contents

`paper.tex`, `paper.pdf`: manuscript, references and proofs. `recursive.py`: new compiler and independent gate simulation, including elementary decomposition controls. `hierarchical.py`: compressed and full direct Walsh circuits. `character_blocks.py`, `partial_symmetry.py`: general conductor theorem controls and coherent mod4 circuit. `audit.py`, `AUDIT.json`, `run.log`, `replay.log`: deterministic reproducibility. `radial_qft.py`, `circuit.py`, `orbit_qft.py`: prior source copies required by the audit. QASM files are emitted logical circuits with arbitrary rotation angles. QASM parser, routing, fault-tolerant synthesis, hardware and general encoder are untested. No Fable consultations, QPU submissions or learned-model training. Initial local syntax typo in boundary multiplexer was corrected before any search or numerical results; final source passes all checks.
