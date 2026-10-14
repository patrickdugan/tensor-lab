"""Summarise certificates, catalogues and logs into a Markdown status report."""
from __future__ import annotations

import glob
import json
import os
from fractions import Fraction


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_report(root: str) -> str:
    lines = ["# tensorlab status report", ""]
    # three-sector certificates
    certs = sorted(glob.glob(os.path.join(root, "certificates", "three_sector_a*_h*.json")))
    lines.append(f"## Three-sector reproduction certificates ({len(certs)})")
    if certs:
        lines.append("")
        lines.append("| a | h | B | retained | erased | blocks | verdict |")
        lines.append("|---|---|---|---|---|---|---|")
        for p in certs:
            c = _load(p)
            v = c.get("verification", {})
            lines.append(f"| {c['parameters']['a']} | {c['parameters']['h']} | {c['parameters']['B']} | "
                         f"{len(c['retained_terms'])} | {len(c['erased_terms'])} | {len(c['blocks'])} | "
                         f"{'PASS' if v.get('ok') else 'FAIL/unverified'} |")
    lines.append("")
    # fourier
    fcerts = sorted(glob.glob(os.path.join(root, "certificates", "fourier_separation_*.json")))
    lines.append(f"## Fourier separation certificates ({len(fcerts)})")
    for p in fcerts:
        c = _load(p)
        d = c["data"]
        lines.append(f"- {os.path.basename(p)}: M={d['M']} L={d['L']} target dims {d['target_dims']} "
                     f"weight-zero terms {d['weight_zero_nnz']} max weight {d['max_weight']} -> "
                     f"{'PASS' if c['ok'] else 'FAIL'}")
    lines.append("")
    # catalogues
    cats = sorted(glob.glob(os.path.join(root, "catalogue", "search_*.json")))
    lines.append(f"## Search catalogues ({len(cats)})")
    for p in cats:
        try:
            c = _load(p)
        except (json.JSONDecodeError, OSError):
            lines.append(f"- {os.path.basename(p)}: unreadable (incomplete run), skipped")
            continue
        st = c["stats"]
        lines.append(f"- {os.path.basename(p)}: {st['structures']} structures, {st['weight_feasible']} weight-feasible, "
                     f"{st['all_identified']} fully identified, {st['unique']} unique inequalities, "
                     f"novel={c['novel']}, {st['seconds']:.0f}s")
        for k, v in sorted(c["classes"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    - {v}: {k}")
        neg = [e for e in c["results"] if Fraction(e["extremal_slack"]) < 0]
        if neg:
            lines.append("    - NEGATIVE-SLACK CANDIDATES:")
            for e in neg[:20]:
                lines.append(f"        - {e['candidate']} (slack {e['extremal_slack']})")
    lines.append("")
    # growth logs
    logs = sorted(glob.glob(os.path.join(root, "logs", "growth_*.json")))
    lines.append(f"## Growth / inequality logs ({len(logs)})")
    for p in logs:
        c = _load(p)
        ep = c.get("extremal_profile", {})
        lines.append(f"- {os.path.basename(p)}: grid {c['grid']}, extremal-profile violations: {len(ep.get('violations', []))}, "
                     f"tight: {ep.get('tight_by_type')}")
        for A, d in c.get("lp_diagonal", {}).items():
            lines.append(f"    - LP min P({A},{A}) = {d['min']} vs extremal {d['extremal']}")
        for d in c.get("candidates", []):
            lines.append(f"    - {d['candidate']}: extremal slack {d['extremal_slack']}, LP min {d['lp_min']} ({d['lp_status']})")
    lines.append("")
    # lean
    lines.append("## Lean")
    ax = os.path.join(root, "logs", "lean_axioms_direct.log")
    if os.path.exists(ax):
        with open(ax, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
        dep = [l for l in txt.splitlines() if "depends on axioms" in l or "does not depend on any axioms" in l]
        bad = [l for l in dep if "sorryAx" in l or "ofReduceBool" in l]
        lines.append(f"- axiom report (`logs/lean_axioms_direct.log`): {len(dep)} theorems checked, "
                     f"{len(bad)} using `sorryAx` or `ofReduceBool`")
        for l in dep:
            lines.append(f"    - {l.strip()}")
    else:
        lines.append("- no axiom report found (run lean/direct_check.sh or lean/wsl_build.sh)")
    return "\n".join(lines) + "\n"
