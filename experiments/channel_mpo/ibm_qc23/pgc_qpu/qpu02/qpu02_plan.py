"""PGC-QPU02 hardware proposal (logical circuits, routing audit on a fake Kingston, QPU-second estimate). Submits nothing.

Circuits keep every echo/control CNOT because compilation uses optimization_level=0 with barriers between layers.
The 4-cycle s-a-t-b does not embed in heavy-hex, so one CNOT needs routing; the audit picks the physical 4-set and labelling
with the fewest native CZ after routing. The real build must re-select qubits from fresh calibration and take its own plan hash.

  python qpu02_plan.py            # writes qpu02_plan.json
"""
from __future__ import annotations
import itertools, json, math, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qpu02_sim as Q  # noqa: E402

SHOTS = 2048
EST = {"shot_repetition_delay_s": 0.00025, "per_shot_circuit_allowance_s": 0.00001, "per_circuit_overhead_allowance_s": 0.1}

HW_VARIANTS = ["intact", "path_a_only", "reset_a", "reset_ab", "echo_intact", "path_a_echo_b", "no_source_link"]
COHERENCE_VARIANTS = ["intact", "reset_a", "echo_intact", "no_source_link"]


def rows():
    out = []
    for v in HW_VARIANTS:
        for s in (0, 1):
            out.append({"id": f"qpu02_{v}_s{s}_Z_ctx000", "variant": v, "input": str(s), "basis": "Z", "ctx": [0, 0, 0], "shots": SHOTS})
    for v in COHERENCE_VARIANTS:
        for k in ("+", "-"):
            out.append({"id": f"qpu02_{v}_{'p' if k == '+' else 'm'}_XXXZ_ctx000", "variant": v, "input": k, "basis": "XXXZ", "ctx": [0, 0, 0], "shots": SHOTS})
        out.append({"id": f"qpu02_{v}_p_Z_ctx000", "variant": v, "input": "+", "basis": "Z", "ctx": [0, 0, 0], "shots": SHOTS})
    for ctx in ((1, 0, 0), (1, 1, 1)):                    # context robustness of the cancellation (Lean quantifies over all contexts)
        for s in (0, 1):
            out.append({"id": f"qpu02_intact_s{s}_Z_ctx{''.join(map(str, ctx))}", "variant": "intact", "input": str(s), "basis": "Z", "ctx": list(ctx), "shots": SHOTS})
    assert len({r["id"] for r in out}) == len(out)
    return out


def logical(row):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4, 4)
    ctx = row["ctx"]
    for q, bit in zip((Q.A, Q.B, Q.T), ctx):
        if bit:
            qc.x(q)
    if row["input"] == "1":
        qc.x(Q.S)
    elif row["input"] == "+":
        qc.h(Q.S)
    elif row["input"] == "-":
        qc.x(Q.S); qc.h(Q.S)
    qc.barrier()
    for kind, a, b in Q.VARIANTS[row["variant"]]:
        if kind == "cx":
            qc.cx(a, b)
        elif kind == "reset":
            qc.reset(a)
            if b == 1:
                qc.x(a)
        else:
            raise ValueError("dephase is a simulation-only intervention")
        qc.barrier()
    if row["basis"] == "XXXZ":
        for q in (Q.S, Q.A, Q.B):
            qc.h(q)
    qc.measure(range(4), range(4))
    return qc


def duration(isa, target):
    clock = {}
    for inst in isa.data:
        op = inst.operation.name
        qs = tuple(isa.find_bit(q).index for q in inst.qubits)
        start = max((clock.get(q, 0.0) for q in qs), default=0.0)
        if op in ("barrier", "rz"):
            d = 0.0
        else:
            props = target[op].get(qs) if op in target.operation_names else None
            d = float(props.duration) if props is not None and props.duration is not None else 1.0e-6   # reset: allowance when unlisted
        for q in qs:
            clock[q] = start + d
    return max(clock.values(), default=0.0)


def audit_layout():
    from qiskit import transpile
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    backend = FakeKingston()
    cm = backend.coupling_map
    adj = {}
    for a, b in cm.get_edges():
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    probe = {"id": "p", "variant": "intact", "input": "0", "basis": "Z", "ctx": [0, 0, 0], "shots": 1}
    qc = logical(probe)
    best = None
    seeds = [q for q in sorted(adj) if len(adj[q]) == 3][:6]
    cands = set()
    for c in seeds:
        nb = sorted(adj[c])
        for x in nb:
            for y in sorted(adj[x]):
                if y != c:
                    cands.add(tuple(sorted({c, x, y, nb[0] if nb[0] != x else nb[1]})))
    for quad in sorted(cands):
        if len(quad) != 4:
            continue
        for perm in itertools.permutations(quad):
            t = transpile(qc, backend, initial_layout=list(perm), optimization_level=0, seed_transpiler=7)
            ops = t.count_ops()
            key = (ops.get("cz", 0), t.depth())
            if best is None or key < best[0]:
                best = (key, perm, quad, ops)
    return backend, best


def build():
    from qiskit import transpile
    backend, best = audit_layout()
    (_, depth), perm, quad, _ = best
    rs = rows()
    comp, total = [], 0.0
    for r in rs:
        isa = transpile(logical(r), backend, initial_layout=list(perm), optimization_level=0, seed_transpiler=7)
        d = duration(isa, backend.target)
        ops = {k: int(v) for k, v in isa.count_ops().items()}
        total += r["shots"] * (EST["shot_repetition_delay_s"] + max(EST["per_shot_circuit_allowance_s"], d)) + EST["per_circuit_overhead_allowance_s"]
        comp.append({**r, "duration_s": d, "native_gate_counts": ops})
    plan = {"version": "PGC-QPU02-PLAN-v1", "status": "proposal only; fake-backend routing audit; not compiled against the live device, not authorized",
            "layout_audit": {"fake_backend": "FakeKingston", "physical_qubits_for_s_a_b_t": list(perm), "native_cz_in_intact": comp[0]["native_gate_counts"].get("cz"),
                             "logical_two_qubit_gates_intact": 4},
            "circuits": len(rs), "shots_per_circuit": SHOTS, "estimated_qpu_seconds": total, "estimate_model": EST,
            "rows": comp,
            "interpretation": {"structural_reachability": "directed gate path s->t not cut by reset (qpu02_sim.reachability)",
                               "operational_target_distinguishability": "P(t=1|s=1) - P(t=1|s=0) from Z readout",
                               "joint_coherence": "Coherence contrast (XXX_plus - XXX_minus)/2; GHZ+ fidelity lower bound max(0, P(sab equal) + (XXX_plus - 1)/2) with t marginalized, valid detector model and confidence bounds required",
                               "restoration": "influence in path_a_only/reset_a relative to intact; echo controls match two-qubit gate count",
                               "apparent_influence_null": "intact and echo_intact have ideal response 0; excess over echo isolates depth-independent leakage"}}
    (HERE / "qpu02_plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({k: plan[k] for k in ("layout_audit", "circuits", "shots_per_circuit", "estimated_qpu_seconds")}, indent=1))


if __name__ == "__main__":
    build()
