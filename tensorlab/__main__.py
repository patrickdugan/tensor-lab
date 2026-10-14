"""Command line interface.

    python -m tensorlab reproduce [--max-a A] [--max-h H] [--out DIR]
    python -m tensorlab search --max-a 5 --max-b 16 [--max-sectors 3] [--out DIR]
    python -m tensorlab verify <certificate.json> [...]
    python -m tensorlab fourier [--a 2 --h 1] [--out DIR]
    python -m tensorlab growth [--grid N] [--diag A ...] [--out DIR]
    python -m tensorlab report [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction

from . import __version__

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CERT_DIR = os.path.join(HERE, "certificates")
CAT_DIR = os.path.join(HERE, "catalogue")
LOG_DIR = os.path.join(HERE, "logs")


def _dump(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=False)


def cmd_reproduce(args) -> int:
    from .sectors import three_sector_certificate
    from .verify import verify_certificate
    out = args.out or CERT_DIR
    summary = []
    all_ok = True
    for a in range(1, args.max_a + 1):
        for h in range(1, args.max_h + 1):
            cert = three_sector_certificate(a, h)
            rep = verify_certificate(cert)
            path = os.path.join(out, f"three_sector_a{a}_h{h}.json")
            cert["verification"] = rep.to_json()
            _dump(path, cert)
            ok = rep.ok
            all_ok &= ok
            summary.append({"a": a, "h": h, "B": cert["parameters"]["B"], "verdict": "PASS" if ok else "FAIL",
                            "retained": len(cert["retained_terms"]), "erased": len(cert["erased_terms"]),
                            "file": os.path.relpath(path, HERE)})
            print(f"(a,h)=({a},{h}): C({a},{cert['parameters']['B']}) -> 3 x C({a},{h})  "
                  f"retained={len(cert['retained_terms'])} erased={len(cert['erased_terms'])}  {'PASS' if ok else 'FAIL'}")
            if not ok:
                print(rep.summary())
    _dump(os.path.join(out, "three_sector_summary.json"),
          {"generator": f"tensorlab {__version__}", "range": {"max_a": args.max_a, "max_h": args.max_h},
           "all_pass": all_ok, "cases": summary,
           "scope": "Each case is a finite exact identity (Lemma 4.2 of the paper at those parameters). "
                    "The family statement for all (a,h) is OpenAI's Lean theorem Sector.retained_eq_three_branches; "
                    "this tool only checks the listed cases."})
    print(f"{len(summary)} cases, all_pass={all_ok}; certificates in {out}")
    return 0 if all_ok else 1


def cmd_verify(args) -> int:
    from .verify import verify_file
    rc = 0
    for path in args.certificates:
        rep = verify_file(path)
        print(f"== {path}")
        print(rep.summary())
        rc |= 0 if rep.ok else 1
    return rc


def cmd_search(args) -> int:
    from .search import certificate_for, classify, search
    from .verify import verify_certificate
    out = args.out or CAT_DIR
    results, stats = search(args.max_a, args.max_b, max_sectors=args.max_sectors,
                            min_a=args.min_a, min_b=args.min_b, allow_pads=not args.no_pads,
                            sources=tuple(args.sources), verbose=args.verbose)
    from .search import mark_dominance
    for e in results:
        e["class"] = classify(e)
    mark_dominance(results)
    certified = 0
    novel = 0
    for e in results:
        if Fraction(e["extremal_slack"]) < 0 or certified < args.certify:
            cert = certificate_for(e)
            rep = verify_certificate(cert)
            e["certificate_verdict"] = "PASS" if rep.ok else "FAIL"
            fname = f"search_a{e['a']}_B{e['B']}_{'_'.join(f'{x}x{y}' for x, y in e['blocks'])}.json"
            cert["verification"] = rep.to_json()
            _dump(os.path.join(CERT_DIR, "search", fname), cert)
            e["certificate_file"] = os.path.join("certificates", "search", fname)
            certified += 1
            if Fraction(e["extremal_slack"]) < 0:
                novel += 1
    tag = f"a{args.max_a}_b{args.max_b}_k{args.max_sectors}"
    classes = {}
    for e in results:
        classes[e["class"]] = classes.get(e["class"], 0) + 1
    nondominated = [e for e in results if e["dominated_by"] is None]
    classes_nd = {}
    for e in nondominated:
        classes_nd[e["class"]] = classes_nd.get(e["class"], 0) + 1
    arguments = {k: v for k, v in vars(args).items() if k != "fn"}
    payload = {"generator": f"tensorlab {__version__}", "arguments": arguments, "stats": stats.to_json(),
               "classes": classes, "nondominated_classes": classes_nd, "nondominated": len(nondominated),
               "novel": novel, "results": results}
    path = os.path.join(out, f"search_{tag}.json")
    _dump(path, payload)
    print(json.dumps(stats.to_json()))
    print("classes:")
    for k, v in sorted(classes.items(), key=lambda kv: -kv[1]):
        print(f"  {v:6d}  {k}")
    print(f"non-dominated: {len(nondominated)} of {len(results)}")
    for k, v in sorted(classes_nd.items(), key=lambda kv: -kv[1]):
        print(f"  {v:6d}  {k}  (non-dominated)")
    print(f"novel (negative slack on the extremal profile): {novel}")
    print(f"catalogue written to {path}")
    return 0


def cmd_fourier(args) -> int:
    from .degeneration import degenerate
    from .fourier import separation
    from .sectors import three_sector_weights
    from .tensors import convolution
    out = args.out or CERT_DIR
    rc = 0
    for a, h in args.cases:
        w, ys, zs = three_sector_weights(a, h)
        B = 3 * h + a - 1
        deg = degenerate(convolution(a, B), w)
        t0 = time.time()
        res = separation(deg.retained, ys, zs)
        print(f"(a,h)=({a},{h}) M=3 L=15: {'PASS' if res.ok else 'FAIL'}  target dims={res.data.get('target_dims')} "
              f"transformed nnz={res.data.get('transformed_nnz')} weight-zero nnz={res.data.get('weight_zero_nnz')} "
              f"max weight={res.data.get('max_weight')}  [{time.time()-t0:.1f}s]")
        if not res.ok or args.verbose:
            print(res.summary())
        cert = {"certificate_version": 1, "kind": "fourier_separation", "generator": f"tensorlab {__version__}",
                "input": {"three_sector": {"a": a, "h": h, "B": B}},
                "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in res.checks],
                "ok": res.ok, "data": {k: v for k, v in res.data.items() if k != "x_index"},
                "scope": "Finite exact instance of Proposition 3.1 (A^{(+)5M} ~> (+)_h A_h (x) B_X(M)) for the "
                         "retained tensor of the three-sector certificate; valid over any field with a primitive "
                         "15th root of unity and 15 invertible."}
        _dump(os.path.join(out, f"fourier_separation_a{a}_h{h}.json"), cert)
        rc |= 0 if res.ok else 1
    return rc


def cmd_growth(args) -> int:
    from .growth import (Candidate, check_profile, cube_check, diagonal_lower_bounds, extremal_profile,
                         growth_product, implied_on_grid, lemma_5_1_bounds)
    out = args.out or LOG_DIR
    log = {"generator": f"tensorlab {__version__}", "grid": args.grid}
    t0 = time.time()
    viol, tight = check_profile(extremal_profile, args.grid)
    tags = {}
    for c in tight:
        k = c.tag.split(":")[0]
        tags[k] = tags.get(k, 0) + 1
    log["extremal_profile"] = {
        "definition": "P*(a,b) = (max(a,b) + (min(a,b)-1)/2) * H_min, H_m = prod_{k<m} (1+1/(3k))",
        "violations": [[repr(c), str(v)] for c, v in viol], "tight_by_type": tags,
        "diagonal": {str(a): str(extremal_profile(a, a)) for a in range(1, args.grid + 1)},
        "seconds": time.time() - t0,
    }
    print(f"extremal profile on [1,{args.grid}]^2: {len(viol)} violations, tight: {tags}  [{time.time()-t0:.1f}s]")
    print("cube check H_a^3 >= a:", cube_check(args.grid))
    log["lemma_5_1"] = [[a, str(H), str(D)] for a, H, D in lemma_5_1_bounds(args.grid)]
    if args.diag:
        t0 = time.time()
        res = diagonal_lower_bounds(args.lp_grid, args.diag)
        log["lp_diagonal"] = {}
        for A, (st, v, prof) in res.items():
            log["lp_diagonal"][str(A)] = {"status": st, "min": None if v is None else str(v),
                                          "extremal": str(extremal_profile(A, A)),
                                          "profile": None if prof is None else {f"{a},{b}": str(x) for (a, b), x in sorted(prof.items())}}
            print(f"LP grid {args.lp_grid}: min P({A},{A}) = {v} ({float(v) if v is not None else None}),"
                  f" extremal P*({A},{A}) = {float(extremal_profile(A, A)):.4f}, {A}^(4/3) = {A ** (4/3):.4f}  [{st}]")
        log["lp_seconds"] = time.time() - t0
    if args.candidate:
        log["candidates"] = []
        for spec in args.candidate:
            src, blocks = spec.split(">=")
            a, b = map(int, src.strip("() ").split(","))
            bl = [tuple(map(int, x.strip("() ").split(","))) for x in blocks.split("+")]
            cand = Candidate((a, b), bl, origin="cli")
            slack = cand.slack(extremal_profile)
            st, v, _ = implied_on_grid(cand, args.lp_grid)
            print(f"{cand}: extremal slack {slack}; LP min over grid {args.lp_grid}: {st} {v}")
            log["candidates"].append({"candidate": repr(cand), "extremal_slack": str(slack), "lp_status": st,
                                      "lp_min": None if v is None else str(v)})
    if args.whatif:
        from .growth import whatif_recurrence
        log["whatif"] = {}
        for dense in (False, True):
            ex = whatif_recurrence(1 << 16, dense)
            key = "dense_rectangles" if dense else "paper_only"
            log["whatif"][key] = ex
            print(f"what-if ({key}): local exponent of P(a,a) at a={ex[-1][0]}: {ex[-1][1]:.4f}, "
                  f"omega <= {3 / ex[-1][1]:.4f}")
    _dump(os.path.join(out, f"growth_grid{args.grid}.json"), log)
    return 0


def cmd_realize(args) -> int:
    """Realizability of block configurations by general monomial weights (exact LP + Farkas)."""
    from .experiments import (e1_staircase_spacing, e2_two_classes, e3_double_staircase,
                              e4_three_sector_sanity, e5_packing_search)
    out = args.out or os.path.join(LOG_DIR, "realizability.json")
    log = {"generator": f"tensorlab {__version__}"}
    t0 = time.time()
    if not args.skip_structured:
        log["E4_three_sector_sanity"] = e4_three_sector_sanity()
        log["E1_staircase_spacing"] = e1_staircase_spacing()
        log["E2_two_classes"] = e2_two_classes()
        log["E3_double_staircase"] = e3_double_staircase()
        for key in ("E4_three_sector_sanity", "E1_staircase_spacing", "E2_two_classes", "E3_double_staircase"):
            for r in log[key]:
                brief = {k: r.get(k) for k in ("a", "p", "q", "h", "period", "B", "use_x", "realizable",
                                               "predicted_realizable", "farkas_consistent", "verified") if k in r}
                print(key, brief)
    log["E5_packing_search"] = []
    for spec in args.packing:
        a, B = (int(x) for x in spec.split(","))
        r = e5_packing_search(a, B, max_lp=args.max_lp, time_limit=args.time_limit)
        log["E5_packing_search"].append(r)
        best = r["best_realizable_above_target"]
        print(f"E5 a={a} B={B}: P*={r['Pstar_source']} {r['stats']} best realizable above target: "
              f"{None if best is None else best['value']}")
        _dump(out, log)
    log["seconds"] = time.time() - t0
    _dump(out, log)
    print(f"written to {out}")
    return 0


def cmd_spectral(args) -> int:
    """Support-functional (quantum-functional) test of dense rectangle packings."""
    from .spectral import dense_rectangle_test
    out = args.out or os.path.join(LOG_DIR, "spectral.json")
    log = {"generator": f"tensorlab {__version__}", "step": args.step,
           "note": "floating point; necessary condition for ANY degeneration C(a,n*p) ~> R(p,q) (x) B_X(n)",
           "rows": []}
    t0 = time.time()
    for spec in args.cases:
        a, p, q = (int(x) for x in spec.split(","))
        rows = dense_rectangle_test(a, p, q, args.n, step=args.step, iters=args.iters)
        for r in rows:
            print(f"a={r['a']} p={r['p']} q={r['q']} n={r['n']}: worst margin {r['worst_certified_margin']:.2e} "
                  f"at theta={r['theta']}, worst interior margin {r['worst_interior_margin']:.4f} at "
                  f"{r['theta_interior']}, obstruction={r['obstruction']}")
        log["rows"] += rows
    log["seconds"] = time.time() - t0
    _dump(out, log)
    print(f"written to {out}")
    return 0


def cmd_dense(args) -> int:
    """Exact verdict on dense rectangle packings (default: the C(3,16) instance)."""
    from .dense import verdict
    out = args.out or os.path.join(CERT_DIR, f"dense_p{args.p}_n{args.n}_B{args.B}.json")
    B = args.B if args.B is not None else args.n * args.p
    v = verdict(args.p, args.n, B)
    v["generator"] = f"tensorlab {__version__}"
    for k, val in v.items():
        if k != "scope":
            print(f"{k}: {val}")
    _dump(out, v)
    print(f"written to {out}")
    return 0


def cmd_amortize(args) -> int:
    """Exact two-copy degeneration of the square: T_2^{(x)2} <= 2 . C(3,4)^{(x)2}."""
    from .border_construct import n2_square_certificate
    out = args.out or os.path.join(CERT_DIR, "amortize_n2_square.json")
    r = n2_square_certificate()
    r["generator"] = f"tensorlab {__version__}"
    r["statement"] = ("Two copies of C(3,4)^{(x)2} degenerate (first order in eps, polynomial of degree 2) to "
                      "T_2^{(x)2}, T_2 = two rectangles C(2,2) sharing X.  One copy is impossible (Koszul).  "
                      "Consequence: lambda(T_2) <= sqrt(2) lambda(C(3,4)) for every tensor character.")
    r["construction"] = ("ell(tau) = 4 - j - k on single-factor terms (zero exactly on the two cross terms); "
                         "L = ell(tau1) ell(tau2) = 16 - 4(j1+j2) - 4(k1+k2) + 2 j1 j2 + 2 k1 k2 - x1 x2 (additive); "
                         "w = 1/L on retained terms is rank one; copies scaled by rho sigma tau (1 + eps alpha)(1 + eps beta)(1 + eps gamma) "
                         "and -rho sigma tau, divided by eps.")
    for k, v in r.items():
        if k not in ("single_factor_rank_one", "construction", "statement"):
            print(f"{k}: {v}")
    print(r["statement"])
    _dump(out, r)
    print(f"written to {out}")
    ok = bool(r["ok"])

    # Three copies of the square for every n (exact restriction, +-1 diagonal scalings).
    from .border_construct import (check_coarse_lift, first_order_obstruction, three_copy_square_certificate,
                                   verify_three_copy_square)
    sweep = [verify_three_copy_square(n) for n in range(2, args.max_n + 1)]
    cert = three_copy_square_certificate(8)
    cert["generator"] = f"tensorlab {__version__}"
    cert["statement"] = ("T_n^{(x)2} is a restriction of three copies of C(3,2n)^{(x)2} for every n >= 2 "
                         "(diagonal maps with entries 0, +1, -1); for n = 8 this is C(3,16).  Consequence: "
                         "lambda(T_n) <= sqrt(3) lambda(C(3,2n)) for every tensor character.")
    cert["sweep_block_form"] = sweep
    cert["coarse_lift"] = [check_coarse_lift(n) for n in range(2, args.max_n + 1)]
    out3 = os.path.join(CERT_DIR, "amortize_three_copy_square_n8.json")
    _dump(out3, cert)
    ok = ok and cert["ok"] and all(s["ok"] for s in sweep) and all(c["ok"] for c in cert["coarse_lift"])
    print(f"three-copy square: n = 2..{args.max_n} block-form checks ok = {all(s['ok'] for s in sweep)}; "
          f"independent C(3,16)^(x)2 check equal = {cert['independent_check']['equal']}")
    print(cert["statement"])
    print(f"written to {out3}")

    # Two copies of the square for every n: parity-twisted pair (exact degeneration).
    from .border_construct import parity_twist_pair_certificate, parity_twist_pair_from_convolution
    sweep2 = []
    for n_ in range(2, args.max_n + 1):
        c_ = parity_twist_pair_certificate(n_)
        sweep2.append({"n": n_, "ok": c_["ok"], "support_terms": c_.get("support_terms"),
                       "nonzero_corrections": c_.get("nonzero_corrections")})
    c8 = parity_twist_pair_certificate(8)
    ind8 = parity_twist_pair_from_convolution(8, c8)
    sol = c8["solution"]; lay = c8["layout"]
    keys = list(lay)
    cert2 = {"n": 8, "copies": 2, "factors": 2, "generator": f"tensorlab {__version__}",
             "statement": ("T_n^{(x)2} is a degeneration of two copies of C(3,2n)^{(x)2} for every n checked "
                           f"(2..{args.max_n}; exact eps-polynomial identity on the block form, and independently "
                           "from the convolution tensor for n = 8).  For n = 8 this is C(3,16).  Consequence: "
                           "lambda(T_n) <= sqrt(2) lambda(C(3,2n)) for every tensor character."),
             "construction": ("copy0 = (1/eps)(a + eps a0')(beta + eps b0')(gamma + eps c0'), "
                              "copy1 = (1/eps)(a + eps a1')(1 + eps b1')(1 + eps c1'); "
                              "a(x1,x2) = sigma(x1) sigma(x2) with sigma = (1, 0, -1) at x = 0, 1, 2; "
                              "beta(j1,j2) = 1 if j1 odd else -(-1)^j2; gamma(k1,k2) = -1 if k1 even else -(-1)^k2 "
                              "(block-form indices; Z shifted by one from C(3,2n)); rational corrections below."),
             "block_form_identity_ok": c8["ok"], "independent_check": ind8, "sweep": sweep2,
             "corrections": {k: [str(v) for v in sol[lay[k]:(lay[keys[i + 1]] if i + 1 < len(keys) else len(sol))]]
                             for i, k in enumerate(keys)},
             "ok": bool(c8["ok"] and ind8["ok"] and all(r["ok"] for r in sweep2))}
    out2 = os.path.join(CERT_DIR, "amortize_two_copy_square_n8.json")
    _dump(out2, cert2)
    ok = ok and cert2["ok"]
    print(f"two-copy square (parity-twisted pair): n = 2..{args.max_n} identities ok = {all(r['ok'] for r in sweep2)}; "
          f"independent C(3,16)^(x)2 check ok = {ind8['ok']}")
    print(cert2["statement"])
    print(f"written to {out2}")

    # Exact obstruction to the first-order (single additive L) two-copy mechanism.
    obs = []
    for (n, N) in [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (4, 2), (5, 2), (8, 1), (8, 2), (3, 3)]:
        r_ = first_order_obstruction(2, n, N)
        r_.pop("forced_examples", None)
        obs.append(r_)
        print(f"first-order mechanism n={n} N={N}: dim V = {r_['dim_V']}, forced zeros "
              f"{r_['forced_zero_retained']}/{r_['retained']} -> possible = {r_['first_order_possible']}")
    _dump(os.path.join(HERE, "logs", "first_order_obstruction.json"), obs)

    # The coarse (n-independent) problem is the cyclic one; calibrate its difficulty.
    from .border_construct import coarse_cyclic_facts
    facts = coarse_cyclic_facts()
    _dump(os.path.join(HERE, "logs", "coarse_cyclic_facts.json"), facts)
    print(f"coarse problem = T_2 inside Cyc_3 (Z_4 group tensor, shifts -1,0,1): support match "
          f"{facts['coarse_support_is_cyclic']}, retained part = T_2 {facts['retained_part_is_T2']}, "
          f"rank(Cyc_3) = 4 by DFT {facts['cyc_rank_4_by_dft']}, border rank T_2 = "
          f"{facts['T2_koszul_border_rank_lower_bound']}; coarse cost -> 1 would force asymptotic rank(T_2) <= 4")
    ok = ok and facts["ok"]
    return 0 if ok else 1


def cmd_report(args) -> int:
    from .catalogue import build_catalogue
    from .report import build_report
    cat = build_catalogue(HERE)
    with open(os.path.join(CAT_DIR, "CATALOGUE.md"), "w", encoding="utf-8") as f:
        f.write(cat)
    text = build_report(HERE)
    path = args.out or os.path.join(HERE, "logs", "report.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"[written to {path}]")
    return 0


def main(argv=None) -> int:
    try:  # the Windows console is cp1252; never crash on a stray Unicode symbol
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(prog="tensorlab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"tensorlab {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reproduce", help="reproduce the three-sector construction and certify it")
    r.add_argument("--max-a", type=int, default=5)
    r.add_argument("--max-h", type=int, default=5)
    r.add_argument("--out")
    r.set_defaults(fn=cmd_reproduce)
    v = sub.add_parser("verify", help="independently verify certificate files")
    v.add_argument("certificates", nargs="+")
    v.set_defaults(fn=cmd_verify)
    s = sub.add_parser("search", help="bounded exhaustive search for shared-leg degenerations")
    s.add_argument("--max-a", type=int, default=5)
    s.add_argument("--max-b", type=int, default=16)
    s.add_argument("--min-a", type=int, default=1)
    s.add_argument("--min-b", type=int, default=1)
    s.add_argument("--max-sectors", type=int, default=3)
    s.add_argument("--no-pads", action="store_true", help="require every sector to be matched")
    s.add_argument("--sources", nargs="+", default=["convolution"], choices=["convolution", "cyclic"])
    s.add_argument("--certify", type=int, default=10, help="number of top candidates to certify and verify")
    s.add_argument("--verbose", action="store_true")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_search)
    f = sub.add_parser("fourier", help="finite Fourier separation (Prop. 3.1) on three-sector outputs")
    f.add_argument("--cases", type=lambda s: tuple(int(x) for x in s.split(",")), nargs="+", default=[(2, 1)],
                   help="a,h pairs, e.g. --cases 2,1 3,2")
    f.add_argument("--verbose", action="store_true")
    f.add_argument("--out")
    f.set_defaults(fn=cmd_fourier)
    g = sub.add_parser("growth", help="inequality engine: extremal profile, LP minima, candidate implication")
    g.add_argument("--grid", type=int, default=40)
    g.add_argument("--lp-grid", type=int, default=9)
    g.add_argument("--diag", type=int, nargs="*", default=[])
    g.add_argument("--candidate", nargs="*", default=[], help='e.g. "(2,4)>=(2,1)+(2,1)+(2,1)"')
    g.add_argument("--whatif", action="store_true", help="what-if growth recurrence with hypothetical dense rectangles")
    g.add_argument("--out")
    g.set_defaults(fn=cmd_growth)
    rz = sub.add_parser("realize", help="general-weight realizability experiments (exact LP with Farkas certificates)")
    rz.add_argument("--packing", nargs="*", default=[], help="a,B pairs for the exhaustive packing search")
    rz.add_argument("--max-lp", type=int, default=400)
    rz.add_argument("--time-limit", type=float, default=900.0)
    rz.add_argument("--skip-structured", action="store_true")
    rz.add_argument("--out")
    rz.set_defaults(fn=cmd_realize)
    sp = sub.add_parser("spectral", help="support-functional test of dense rectangle packings (floating point)")
    sp.add_argument("--cases", nargs="+", default=["3,2,2", "5,3,3"], help="a,p,q triples")
    sp.add_argument("--n", type=int, nargs="+", default=[2, 4, 8, 16])
    sp.add_argument("--step", type=float, default=0.1)
    sp.add_argument("--iters", type=int, default=4000)
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_spectral)
    dn = sub.add_parser("dense", help="exact verdict on C(2p-1,B) => n rectangles C(p,p) sharing X (default C(3,16), n=8)")
    dn.add_argument("--p", type=int, default=2)
    dn.add_argument("--n", type=int, default=8)
    dn.add_argument("--B", type=int, default=16)
    dn.add_argument("--out")
    dn.set_defaults(fn=cmd_dense)
    am = sub.add_parser("amortize", help="copy-cost amortisation: two-copy square (n = 2), three-copy square (all n), "
                                         "first-order obstruction")
    am.add_argument("--out")
    am.add_argument("--max-n", type=int, default=12)
    am.set_defaults(fn=cmd_amortize)
    rp = sub.add_parser("report", help="summarise certificates, catalogues and logs")
    rp.add_argument("--out")
    rp.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
