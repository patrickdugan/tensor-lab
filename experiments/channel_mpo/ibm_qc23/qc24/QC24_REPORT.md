# QC-24A/B: retrospective hardware-local coherent-kick diagnostic

Source: QC-23 pinned commit fd27a77b4e1a7f468d380d32b411943ad3e5f69f, Kingston kicked-Ising raw counts. Zero QPU seconds; no new hardware data.

## Results

The independent complex statevector implementation reproduces 384 measured single-site expectations across 36 circuits. On the 108 published observable locations the ideal maximum difference is 9.658940314238862e-15, count-derived maximum difference 0.

Fit only n=8 (16 circuits); evaluate n=12 and n=16 (20 circuits). Physical qubits 5 and 23 were selected retrospectively using all widths, so the transfer set is not a sealed holdout. Fit minimizes all-qubit MSE, with offsets bounded to ±0.3 rad. This is a logical RX effective model, not a fit of physical native SX pulses. Compare ideal, readout-only local CPTP channel, empirical linear-depth residual correction, two local coherent kick offsets, and global-plus-local coherent kick offsets. All comparators fit on n=8 only.

| Model | Transfer all MAE | Target qubits MAE | Other qubits MAE | All within 0.05 |
|---|---:|---:|---:|---:|
| ideal | 0.04212 | 0.10480 | 0.03052 | 72.66% |
| local_rx | 0.03504 | 0.05145 | 0.03200 | 73.05% |
| global_local_rx | 0.03235 | 0.05083 | 0.02893 | 76.56% |
| readout | 0.03523 | 0.06071 | 0.03052 | 77.34% |
| depth_offset | 0.03390 | 0.05215 | 0.03052 | 76.17% |

Local offsets (radians): q5=0.0825, q23=-0.09125000000000001. Global+local fit: {"5":0.0925,"23":-0.08218750000000001,"global":-0.0071874999999999994}. Readout fits: {"5":{"e0":0.098,"e1":0},"23":{"e0":0,"e1":0.234}}. Depth slopes: {"5":-0.044753523375041425,"23":0.059467814856312466}.

The exact statevector is the reference; this experiment does not yet implement a heterogeneous MPO. Offset estimates are phenomenological; coherent, dissipative, crosstalk and compilation-dependent effects remain possible. Adjacent qubits and physical path are confounded by the reused chain. A readout-only arm is deliberately flexible but constrained to stochastic flip probabilities. The depth-offset arm is empirical and is not a quantum channel. All metrics are descriptive; no independent session uncertainty was estimated. No claim of hardware-native gate calibration.

## Reproduction

From this directory, run `node qc24_audit.mjs /tmp/qc24_results.json` (Node 18+; no dependencies). The script checks count totals, state norm, and the published ideal/count replay. It writes all 384 rows, cell means and split metrics. The JSON in this directory was produced by executing the same audit function in a JavaScript sandbox. CLI invocation in Node and repo CI remain to be checked.

## Next gate

Read QC24_PROTOCOL.md before spending QPU time. Model selection and site selection are exploratory; freeze a fresh seed family and independent physical paths before submission. No account-owner authorization has been created.
