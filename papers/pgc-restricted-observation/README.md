# Restricted-observation quantum channel identification

Manuscript 0.2, working draft, 28 November 2026. The canonical source is `PAPER.md`. The deterministic renderer `render_paper.mjs` generates `paper.tex`; the separately committed `results_figure.tex` is derived from the pinned prospective report. All experiment files remain unchanged.

## Build

From this directory, with Node 22 and a TeX Live installation including latex-extra, fonts-recommended, and pictures:

```sh
node render_paper.mjs
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

The hosted paper workflow checks that the generated TeX agrees with the committed source and uploads the PDF after a successful build. The document uses a journal-neutral single-column article layout; no author identity, institutional affiliation, funding statement or venue has been assumed. Those require confirmation before submission.

## Frozen evidence

- Hardware and scoring: tensor-lab `5244802c3e3f8317a359096d8ef2f1ba5f1cd7c6`; evaluation freeze `1f120d1250d49d6c91aa21b12215e6a90ff6564a`.
- Formal finite results and repaired receipt: MathCan `27eb7296a8d7670903ab8789921168e9d3ff1661`.
- Hosted PGC replay: [46032571852](https://github.com/patrickdugan/tensor-lab/actions/runs/46032571852); machine-readable timing replay in `benchmark_ci_receipt.json`.
- Primary C3 verdict: **FAIL**. C3Z, C3Z-D and C2 also fail their registered criteria. The manuscript must preserve those verdicts.
- Authorship, funding and competing interests are unresolved; complete them before submission.
