# QC-25B: matched hardware replay protocol

Parent 959e73ad60dd51954e7cd5f9900ba10e208555f6 (QC-25A merged). Existing Kingston counts only. No authorization and no QPU execution.

Compare ideal, QC-23 uniform Model A (original cap 16), and gate-local Model A using QC-25's heterogeneous PTMs, exactly the same calibrated faults and readout. QC-23 Model B is a secondary published-site reference only. No coherent parameters fitted on these counts. This tests the effect of uniform projection and preserves the earlier ASAP/Pauli-twirled idle approximation. No new model of native amplitude damping or crosstalk.

Replay all 12 echo and 36 kicked-Ising circuits, every measured bit (528 observables: 144 echo, 384 kicked-Ising); also report original 144 published sites separately. Compare raw bitstrings and original predictions, and check exact Clifford oracle for all echo sites. Primary retrospective metric is MAE; paired circuit bootstrap estimates differences, conditional on this single hardware acquisition. Report all cases, including bad sites 5 and 23. Use cap ladder 16/32/64 and report unresolved bounds; the ledger is exact-arithmetic truncation only.

Provenance gate: SHA-256 of QPY, QASM re-export, original submitted QASM and measurement mapping must match the pinned manifest. The shared hw_root/compiled/circuits/kicked_ising_n8_d1_s41.qasm references the earlier echo path, while submitted ibm_run_nc/circuits/... uses the manifest path. Never use stale shared NC QASM as the submitted circuit. Save list of stale shared files. Counts and calibration hashes in each receipt. The NC analysis snapshot is five minutes newer than routing-time; this experiment uses the same snapshot as published to isolate projection error. Independent snapshot provenance and prospective native coherent calibration remain required.

Execution: five CI shards, offline. Results are pending until completed logs are retrieved. No claim of improved hardware prediction until actual receipts are available.
