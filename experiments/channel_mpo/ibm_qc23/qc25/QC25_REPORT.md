# QC-25A: Heterogeneous CPTP Heisenberg-MPO contraction (synthetic validation)

Parent main: aacff07d99e79ac5b035c654b90bb544b3730ac7. No QPU time. No new hardware predictions in this receipt.

## Mathematical model

For a local channel E with Kraus operators K_j satisfying sum K_j^* K_j = I,
use T_ab = Tr(P_a E(P_b))/d in the normalized Pauli coefficient convention
O = sum_p c_p P_p. The adjoint acts as T^T on local coefficients. The code
keeps each one/two-site map separately and reverses the channel order in
Heisenberg evolution. For asymmetric readout, terminal Z becomes
(e1-e0) I + (1-e0-e1) Z. General Kraus maps support nonunital relaxation.

Before each two-site truncation, the external coefficient-MPS environments are
made isometric by QR. Discarded singular values have norm eta_j equal to
the global normalized Hilbert-Schmidt residual in exact arithmetic. For
Hermitian error D, ||D||_op <= 2^(n/2) ||D||_(HS,normalized).
A CPTP adjoint is positive and unital, hence contracts operator norm on
Hermitian operators. Telescoping bounds the final expectation error by
2^(n/2) sum_j eta_j. This is an exact-arithmetic truncation theorem;
the floating-point ledger is not an interval certificate or a roundoff bound.
The rank threshold 1e-14 discards tiny singular values into the same ledger.
The bound can be loose for wide circuits or small caps. No clamping.

## Executed tests

Local Linux Python/NumPy 2.3.5, OpenBLAS and OMP limited to one thread.
72 dense density-matrix comparisons: widths 2,4,6; seeds 41,43; 2 targets;
6 caps 1,2,4,8,16,64. All errors <= ledger + 1e-10; maximum cap-64
difference 1.27675647831893e-15. Also passed analytical one-site
amplitude-damping check independently against the MPO and dense methods.
The largest positive error-minus-ledger residual was 5.70e-16
(roundoff; not a strict floating-point certificate).

Synthetic kicked-Ising-like heterogeneous channels, center Z, seed 41:

| n | depth | cap | Z | truncation ledger | change from previous cap |
|---:|---:|---:|---:|---:|---:|
|12|4|4|0.085431052532565|2.2982e-2|--|
|12|4|8|0.085518796220786|3.3361e-14|8.7744e-5|
|16|6|8|-0.051563931153958|8.7158e-1|3.5695e-3|
|16|6|16|-0.051690946688801|1.5927e-5|1.2702e-4|
|16|6|32|-0.051690946592605|3.3642e-13|9.6197e-11|
|24|8|16|-0.126960036415639|1.2932|7.5399e-5|
|24|8|32|-0.126935865206757|3.3069e-4|2.4171e-5|
|24|8|48|-0.126935857819135|2.8756e-7|7.3876e-9|
|24|8|64|-0.126935857817572|1.3547e-9|1.5629e-12|

n=24 depth=8 cap=64 took approximately 0.094 s in this local run,
excluding channel construction; timing is machine-specific. These are
synthetic convergence data with no dense reference at n=12,16,24.

## Native-operation bridge and limitations

native_channels preserves QC-23 Model A fault placement and gate order,
including reversing asymmetric Pauli labels for reversed two-site logical
order. It inherits QC-23's ASAP schedule and Pauli-twirled thermal model;
the standalone thermal() map supports nonunital relaxation but is not yet
wired to native IBM calibration. Optional native SX offsets are zero by
default. QC-24's fitted effective logical RX offsets are not native SX
calibrations and must not be substituted. Qiskit and Stim were unavailable
in the local sandbox, so the native bridge was not executed here.
QC-25 CI will run a separate native bridge regression using existing QC-23
dependencies and offline committed circuits.

The current implementation is restricted to contiguous one/two-site
channels, initial |0...0>, and single-site Z observables. Crosstalk beyond
nearest-neighbor pairs, exact-dyadic interval certification, and actual
hardware predictive improvement remain untested. No merge or hardware
spending should be inferred from the synthetic benchmark.

Reproduce: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python qc25_bench.py /tmp/qc25.json
