"""Build the ranked candidate catalogue (catalogue/CATALOGUE.md) from the
search catalogues, the realizability log and the structured families."""
from __future__ import annotations

import glob
import json
import os
from fractions import Fraction
from typing import Dict, List

from .growth import Candidate, extremal_profile, staircase_candidates


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


RANK = [
    "NOVEL",
    "known family: alternating C(a,h_i) blocks, tight",
    "contains a rectangle block",
    "other block shapes",
    "known family: alternating C(a,h_i) blocks (slack > 0",
    "restriction-type",
    "trivial",
]


def _rank(cls: str) -> int:
    for i, prefix in enumerate(RANK):
        if cls.startswith(prefix):
            return i
    return len(RANK)


def build_catalogue(root: str) -> str:
    out: List[str] = ["# Candidate catalogue", ""]
    out.append("Every entry is an inequality `P(a,B) >= sum P(blocks)` coming from a shared-leg "
               "degeneration of `C(a,B)` found by the bounded search (`search.py`).  Slack is measured "
               "on the extremal profile `P*` (`notes/02_extremal_profile_and_theta.md`): only a negative "
               "slack could improve the growth exponent.  An entry is *dominated* when another entry with "
               "the same `a` and a source `B' <= B` has blocks that are componentwise at least as large "
               "(then it is a trivial consequence: padding, a dropped block or a shrunk block).")
    out.append("")
    cats = sorted(glob.glob(os.path.join(root, "catalogue", "search_*.json")))
    merged: Dict[str, dict] = {}
    from .search import classify, mark_dominance
    for p in cats:
        try:
            c = _load(p)
        except (json.JSONDecodeError, OSError) as exc:
            out.append(f"## Search `{os.path.basename(p)}`: unreadable ({exc.__class__.__name__}; incomplete run), skipped")
            out.append("")
            continue
        # re-derive classes and dominance with the current rules
        for e in c["results"]:
            e["class"] = classify(e)
        mark_dominance(c["results"])
        c["classes"], c["nondominated_classes"] = {}, {}
        for e in c["results"]:
            c["classes"][e["class"]] = c["classes"].get(e["class"], 0) + 1
            if e["dominated_by"] is None:
                c["nondominated_classes"][e["class"]] = c["nondominated_classes"].get(e["class"], 0) + 1
        c["nondominated"] = sum(c["nondominated_classes"].values())
        c["novel"] = sum(1 for e in c["results"] if Fraction(e["extremal_slack"]) < 0)
        st = c["stats"]
        out.append(f"## Search `{os.path.basename(p)}`")
        out.append("")
        args = c.get("arguments", {})
        out.append(f"Bounds: a <= {args.get('max_a')}, B <= {args.get('max_b')}, at most {args.get('max_sectors')} "
                   f"sectors per leg.  {st['structures']} sector structures, {st['weight_feasible']} with feasible "
                   f"sector weights, {st['all_identified']} with every block identified, **{st['unique']} distinct "
                   f"inequalities**, {c.get('nondominated', '?')} non-dominated, **{c['novel']} with negative slack**; "
                   f"{st['seconds']:.0f} s.")
        out.append("")
        out.append("| class | all | non-dominated |")
        out.append("|---|---|---|")
        nd = c.get("nondominated_classes", {})
        for k, v in sorted(c["classes"].items(), key=lambda kv: _rank(kv[0])):
            out.append(f"| {k} | {v} | {nd.get(k, 0)} |")
        out.append("")
        for e in c["results"]:
            merged.setdefault(e["candidate"] + "|" + e["source_kind"], e)
    if merged:
        entries = sorted(merged.values(), key=lambda e: (_rank(e["class"]), Fraction(e["extremal_slack"]),
                                                          -e["n_blocks"], e["a"], e["B"]))
        nd = [e for e in entries if e.get("dominated_by") is None]
        out.append("## Ranked non-dominated candidates")
        out.append("")
        out.append("Classes in rank order; within a class, smallest slack first, then larger `a`.  "
                   "No entry has negative slack, so none can improve the exponent.")
        out.append("")
        out.append("| rank | inequality | slack on P* | class | block legs |")
        out.append("|---|---|---|---|---|")
        r = 0
        for cls in sorted({e["class"] for e in nd}, key=_rank):
            group = sorted([e for e in nd if e["class"] == cls],
                           key=lambda e: (Fraction(e["extremal_slack"]), -e["a"], e["B"]))
            for e in group[:8]:
                r += 1
                out.append(f"| {r} | `{e['candidate']}` | {e['extremal_slack']} | {e['class']} | "
                           f"{', '.join(e['permutations'])} |")
        out.append("")
        certified = [e for e in entries if e.get("certificate_file")]
        if certified:
            out.append("### Certified search results (independent verifier)")
            out.append("")
            for e in certified:
                out.append(f"- `{e['candidate']}`: `{e['certificate_file']}` {e.get('certificate_verdict', '')}")
            out.append("")
    main_cat = os.path.join(root, "catalogue", "search_a5_b16_k3.json")
    if os.path.exists(main_cat):
        out.append(rediscovery_and_certification(root, main_cat))
    # structured families
    out.append("## Structured families")
    out.append("")
    out.append("| family | status | evidence |")
    out.append("|---|---|---|")
    out.append("| three-sector `P(a,3h+a-1) >= 3P(a,h)` (paper Lemma 4.2) | reproduced, tight on P* | 25 certificates, Lean `Finite` (4 cases) |")
    out.append("| odd-sector `P(a,(2e+1)h+e(a-1)) >= (2e+1)P(a,h)` | valid, **implied** by (C)+(T) | Lean `Obstruction.odd_sector_implied` |")
    out.append("| any input-on-X convolution blocks in one `C(a,B)` | cannot violate P* | Lean `Extremal.counting_obstruction` |")
    sc = staircase_candidates(40)
    out.append(f"| rectangle staircase `P(a,(n-1)(p+q-1)+p) >= n P(p,q)` | valid, slack > 0 on all {len(sc)} instances with B <= 40 | `sectors.staircase_certificate` |")
    dc = staircase_candidates(40, double=True)
    neg = [c for c in dc if c.slack(extremal_profile) < 0]
    out.append(f"| double staircase (gaps filled) | **rejected**: not realizable ({len(neg)} of {len(dc)} hypothetical instances would beat P*) | Farkas certificates, `logs/realizability.json` E3 |")
    out.append("| two interleaved rectangle classes | **rejected**: not realizable | Farkas certificate, E2 |")
    out.append("| dense rectangles `C(3,2n) => n x C(2,2)^sigma` (e.g. `C(3,16)`, n=8) | **rejected for every degeneration**: needs `B >= 3n-1` (Koszul), staircase optimal | `certificates/dense_p2_n8_B16.json`, `notes/05` |")
    out.append("| dense rectangles from two copies `C(3,2n) (+) C(3,2n)` | valid restriction, too weak (factor 2) | `dense.two_copy_restriction` |")
    out.append("| dense rectangles, squares from two copies `T_2^(x)2 <= 2 C(3,4)^(x)2` | valid **degeneration** (one copy impossible): cost sqrt(2) per factor | `certificates/amortize_n2_square.json`, `notes/06` |")
    out.append("| dense rectangles, squares from two copies `T_n^(x)2 <| 2 C(3,2n)^(x)2`, every n <= 16 (incl. `C(3,16)`) | valid **degeneration** (parity-twisted pair): cost sqrt(2) per factor; still far from the cost 1 the exponent needs | `certificates/amortize_two_copy_square_n8.json`, `notes/06` |")
    out.append("| dense rectangles, squares from three copies `T_n^(x)2 <= 3 C(3,2n)^(x)2`, every n | valid **restriction** (no limit): cost sqrt(3) per factor | `certificates/amortize_three_copy_square_n8.json`, `notes/06` |")
    out.append("| first-order (tangential, non-vanishing factors) two-copy amortisation beyond the C(3,4) square | **rejected** (exact forced zeros, e.g. 744/1024 for `C(3,16)`, N=2) | `logs/first_order_obstruction.json` |")
    out.append("| any exact block packing in `C(3,B)` | cannot beat P* (Koszul LP) | `logs/koszul_lp.json` |")
    out.append("")
    real = os.path.join(root, "logs", "realizability.json")
    if os.path.exists(real):
        d = _load(real)
        out.append("## Rejected packings (general monomial weights, exact LP)")
        out.append("")
        out.append("| a | B | P*(a,B) | packings with value > P* | tested | realizable | all sampled obstructions verified |")
        out.append("|---|---|---|---|---|---|---|")
        for r in d.get("E5_packing_search", []):
            s = r["stats"]
            obs = r["sample_obstructed"]
            out.append(f"| {r['a']} | {r['B']} | {r['Pstar_source']} | {s['packings_above_target']} | {s['lp_calls']} | "
                       f"{s['realizable_above_target']} | {all(o.get('obstruction_verified') for o in obs) if obs else 'n/a'} |")
        out.append("")
    return "\n".join(out) + "\n"


def certify_entries(root: str, entries: List[dict]) -> List[dict]:
    """Rebuild, independently verify and save certificates for search entries."""
    from .search import certificate_for
    from .verify import verify_certificate
    out = []
    d = os.path.join(root, "certificates", "search")
    os.makedirs(d, exist_ok=True)
    for e in entries:
        cert = certificate_for(e)
        rep = verify_certificate(cert)
        cert["verification"] = rep.to_json()
        fname = f"search_a{e['a']}_B{e['B']}_{'_'.join(f'{x}x{y}' for x, y in e['blocks'])}.json"
        with open(os.path.join(d, fname), "w", encoding="utf-8") as f:
            json.dump(cert, f, indent=1)
        out.append({"candidate": e["candidate"], "file": f"certificates/search/{fname}",
                    "verdict": "PASS" if rep.ok else "FAIL", "class": e["class"]})
    return out


def rediscovery_and_certification(root: str, catalogue_file: str, per_class: int = 3) -> str:
    """Markdown section: (1) is every paper instance P(a,3h+a-1) >= 3P(a,h) inside the
    search bounds found by the search?  (2) certificates for the top entries of each class."""
    from .search import classify, mark_dominance
    c = _load(catalogue_file)
    res = c["results"]
    for e in res:
        e["class"] = classify(e)
    mark_dominance(res)
    args = c.get("arguments", {})
    max_a, max_b = args.get("max_a", 5), args.get("max_b", 16)
    lines = [f"## Rediscovery of the paper's construction (`{os.path.basename(catalogue_file)}`)", ""]
    lines.append("| a | h | B = 3h+a-1 | found by the search | blocks / legs | certificate |")
    lines.append("|---|---|---|---|---|---|")
    found_all = True
    to_cert = []
    for a in range(2, max_a + 1):
        for h in range(1, max_b + 1):
            B = 3 * h + a - 1
            if B > max_b:
                break
            want = sorted([tuple(sorted((a, h)))] * 3)
            hit = [e for e in res if e["source_kind"] == "convolution" and e["a"] == a and e["B"] == B
                   and sorted(tuple(sorted(b)) for b in e["blocks"]) == want]
            found_all &= bool(hit)
            if hit:
                to_cert.append(hit[0])
            lines.append(f"| {a} | {h} | {B} | {'yes' if hit else '**no**'} | "
                         f"{', '.join(hit[0]['permutations']) if hit else ''} | |")
    certs = certify_entries(root, to_cert)
    by_cand = {x["candidate"]: x for x in certs}
    # fill certificate column
    new_lines = []
    for ln in lines:
        if ln.startswith("| ") and ln.count("|") == 7 and "yes" in ln:
            parts = ln.split("|")
            a, h = int(parts[1]), int(parts[2])
            B = 3 * h + a - 1
            cand = next((x for x in to_cert if x["a"] == a and x["B"] == B), None)
            if cand is not None and cand["candidate"] in by_cand:
                cc = by_cand[cand["candidate"]]
                parts[6] = f" `{cc['file']}` {cc['verdict']} "
                ln = "|".join(parts)
        new_lines.append(ln)
    lines = new_lines
    lines.append("")
    lines.append(f"All instances inside the bounds rediscovered: **{found_all}**.")
    lines.append("")
    # top entries per class
    nd = [e for e in res if e.get("dominated_by") is None]
    pick = []
    for cls in sorted({e["class"] for e in nd}, key=_rank):
        group = sorted([e for e in nd if e["class"] == cls],
                       key=lambda e: (Fraction(e["extremal_slack"]), -e["a"], e["B"]))
        pick += group[:per_class]
    certs2 = certify_entries(root, pick)
    lines.append("## Certified representatives of each class")
    lines.append("")
    lines.append("| inequality | class | certificate | verifier |")
    lines.append("|---|---|---|---|")
    for x in certs2:
        lines.append(f"| `{x['candidate']}` | {x['class']} | `{x['file']}` | {x['verdict']} |")
    lines.append("")
    return "\n".join(lines) + "\n"
