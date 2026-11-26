"""Independent NumPy SVD check of the committed PGC-QPU01 Jacobian rows (exact math on archived rows; no hardware)."""
import json, sys, pathlib
import numpy as np

here = pathlib.Path(__file__).resolve().parent.parent
R = json.loads((here / "receipt.json").read_text())["result"]
labels = [p["label"] for p in R["parameters"]]
rows = R["rows"]
J_all = np.array([r["jacobian"] for r in rows], float)
target = np.array([r["physical"] in (5, 23) for r in rows])
EXPECT = {("target", 6): 4, ("all", 6): 6, ("target", 14): 6, ("all", 14): 12}
out = {"labels": labels, "cases": {}}
ok = True
for sel_name, sel in (("target", target), ("all", np.ones(len(rows), bool))):
    for npar in (6, 14):
        J = J_all[sel][:, :npar]
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        tol = 1e-6 * s[0]
        rank = int((s > tol).sum())
        zero_cols = [labels[j] for j in range(npar) if np.max(np.abs(J[:, j])) == 0.0]
        weak = []
        for k in range(len(s) - 1, -1, -1):
            if s[k] <= tol or len(weak) < 2:
                v = Vt[k]
                top = np.argsort(-np.abs(v))[:3]
                weak.append({"sv": float(s[k]), "dominant": [(labels[i], round(float(v[i]), 4)) for i in top]})
            if len(weak) >= 6:
                break
        good = rank == EXPECT[(sel_name, npar)]
        ok &= good
        out["cases"][f"{sel_name}_{npar}"] = {
            "rows": int(sel.sum()), "rank": rank, "expected": EXPECT[(sel_name, npar)], "match": good,
            "singular_values": [float(x) for x in s], "zero_derivative_columns": zero_cols,
            "weakest_directions": weak[:4],
        }
print(json.dumps(out, indent=1))
(here / "analysis" / "jacobian_svd_check.json").write_text(json.dumps(out, indent=1))
sys.exit(0 if ok else 1)
