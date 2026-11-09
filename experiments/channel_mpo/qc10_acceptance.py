"""QC-10: post-run local operational error acceptance gate.

A selection covers only the nominated local regions and the tested circuit.
Numerical finite-cone references serve as independent acceptance data.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np

REQUIRED_REGIONS = ("left_1", "left_2", "right_1", "right_2")

def select_acceptable(rows: list[dict], tolerance: float = .05,
                      *, required_regions: tuple[str, ...] = REQUIRED_REGIONS) -> list[dict]:
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("operational tolerance must lie in [0,1]")
    if not required_regions or len(set(required_regions)) != len(required_regions):
        raise ValueError("required regions must be unique")
    grouped = {}
    for r in rows:
        key = (int(r["n"]), str(r["channel"]), float(r["p"]),
               int(r["seed"]), int(r["depth"]))
        index = (str(r["mode"]), str(r["region"]))
        if index in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate observation: {key}, {index}")
        grouped[key][index] = r
    decisions = []
    for key, records in sorted(grouped.items()):
        for mode in ("peak", "sketch"):
            if not set(required_regions) <= {region for m, region in records if m == mode}:
                raise ValueError(f"missing observable region in {key}: {mode}")
        worst = {mode: max(float(records[(mode, region)]["local_trace_distance"])
                           for region in required_regions)
                 for mode in ("peak", "sketch")}
        if any(not np.isfinite(d) or d < 0 or d > 1 + 1e-8 for d in worst.values()):
            raise ValueError("invalid trace distance")
        chosen = ("peak" if worst["peak"] <= tolerance else
                  "sketch" if worst["sketch"] <= tolerance else "reject")
        decision = {"n":key[0], "channel":key[1], "p":key[2],
                    "seed":key[3], "depth":key[4],
                    "event_error_tolerance":tolerance,
                    "decision":chosen, "max_d_peak":worst["peak"],
                    "max_d_sketch":worst["sketch"],
                    "max_d_selected":worst.get(chosen,float("nan"))}
        if chosen != "reject":
            selected = records[(chosen,required_regions[0])]
            decision["modeled_live_peak_selected"] = selected["modeled_live_peak_entries"]
            decision["retained_entries_selected"] = selected["retained_entries"]
            decision["simulation_seconds_selected"] = selected["sim_seconds"]
        decisions.append(decision)
    return decisions

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=.05)
    parser.add_argument("--output", type=Path, default=Path("qc10_acceptance.csv"))
    args = parser.parse_args()
    with args.source.open(newline="") as inp:
        records = list(csv.DictReader(inp))
    decisions = select_acceptable(records, args.tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w",newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    counts = {name: sum(row["decision"] == name for row in decisions)
              for name in ("peak","sketch","reject")}
    print(f"{len(decisions)} cases: {counts}; saved {args.output}")

if __name__ == "__main__":
    main()
