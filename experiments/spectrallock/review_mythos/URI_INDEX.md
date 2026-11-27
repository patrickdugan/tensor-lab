# Selective retrieval index

All project code links are raw, commit-pinned HTTPS. Prefer text retrieval to opening GitHub directory listings. The tensor-lab repository is public. If a raw host is blocked, use the corresponding GitHub blob URL (replace raw.githubusercontent.com with github.com and insert /blob/ before the commit). Fetch the small summaries first; skip sources unless needed for a particular review task.

## Priority 1 — lattice review
- https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl04/README.md
- https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl03/README.md
- https://ia.cr/2026/2078
- https://eprint.iacr.asia/2026/2078 (abstract mirror)
- https://eprint.iacr.org/2026/098
- https://raw.githubusercontent.com/NXP-Research/RadicalRingLWR/ba0a745d360455fa9a1e29f086b307e4516de4aa/README.md
- https://github.com/NXP-Research/RadicalRingLWR/tree/ba0a745d360455fa9a1e29f086b307e4516de4aa (actual ring/parameter/multiplication source, select files)

## Priority 2 — only for checking a theorem or reproducing results
- SL01: https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl01/README.md ; https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl01/AUDIT_SUMMARY.json
- SL02: https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl02/README.md ; https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl02/AUDIT_SUMMARY.json
- SL03: https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl03/README.md ; https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl03/AUDIT_SUMMARY.json
- SL04: https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl04/README.md ; https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl04/AUDIT_SUMMARY.json
- Prefix rounding/power: https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl01/spectrallock.py
- Block rounding: https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl01/block.py
- Cross-level and fixed-q: https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl02/cross_shell.py
- Centered embedding: https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl03/lattice_pilot.py
- Collision theorem checks: https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl03/collision_audit.py
- Ring arithmetic: https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl04/module_lwr.py
- Subring/full-sector layout: https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl04/polyphase.py
- Matched module lattice tests: https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl04/lattice_compare.py
- Arithmetic verification: https://raw.githubusercontent.com/patrickdugan/tensor-lab/702fe98779644f7a2b4cb7e403ce7ba16c15384e/experiments/spectrallock/sl04/audit.py

## External foundations and estimator (select as needed)
- https://eprint.iacr.org/2015/769 (small-modulus LWR reductions)
- https://eprint.iacr.org/2016/589 (dimension-preserving module reductions)
- https://eprint.iacr.org/2022/727 (limits of rounding reductions)
- https://eprint.iacr.org/2018/230 (Saber)
- https://github.com/malb/lattice-estimator/tree/1d5c3f5c2365ea41c536805626d3f937c4412e55 (pin estimator revision; generic LWE estimate is not a proof of structured LWR hardness)

## Optional QFT/hardware review
- https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/manuscript/paper.tex (includes hardware evaluation)
- https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/PAPER_RECURSIVE.md
- https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/recursive.py
- https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/hardware_results/README.md
- https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/analyze_qpu01.py
- stageA_b3: https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/hardware_results/stageA_b3/analysis_full.json
  - https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/hardware_results/stageA_b3/counts.json
  - https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/hardware_results/stageA_b3/compiled.json
  - https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/hardware_results/stageA_b3/execution_receipt.json
- stageB_b4: https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/hardware_results/stageB_b4/analysis_full.json
  - https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/hardware_results/stageB_b4/counts.json
  - https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/hardware_results/stageB_b4/compiled.json
  - https://raw.githubusercontent.com/patrickdugan/tensor-lab/08274a5f5c6e61198bbfd6e106f0bc19a96b9702/experiments/hadamard_qft/qft01/hardware_results/stageB_b4/execution_receipt.json

Do not recursively retrieve every file or binary QPY artifact. Local archives have the full raw synthetic receipts and checksums; reproduce specific cases if needed. Full ePrint paper retrieval and external novelty review remain outstanding.
