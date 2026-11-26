"""PGC-QPU02 quantum causal cancellation: exact statevector and noisy density-matrix simulation (no hardware).

Four logical qubits (s, a, b, t). Intact circuit: layer 1 CNOT(s->a), CNOT(s->b); layer 2 CNOT(a->t), CNOT(b->t).
Computational-basis map |s,a0,b0,t0> -> |s, a0^s, b0^s, t0^a0^b0>, so for every context the target is independent of s
(the cancellation is context-robust); with a0=b0=t0=0 it is |s,0,0,0> -> |s,s,s,0>.
Mirrors the Lean theorem MathCan.Pregeometry.event_response_not_transitive (diamond step F = (0,x0,x0,x1 xor x2)).

Evidence classes produced here: exact (statevector/density matrix) and noisy simulation. Nothing here is hardware evidence.

  python qpu02_sim.py            # writes qpu02_sim_results.json; exits nonzero if an exact check fails
"""
from __future__ import annotations
import itertools, json, math, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
S, A, B, T = 0, 1, 2, 3
N = 4
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)


def op1(u, q):
    m = np.array([[1]], complex)
    for k in range(N):
        m = np.kron(m, u if k == q else I2)
    return m


def cnot(c, t):
    d = 2 ** N
    m = np.zeros((d, d))
    for i in range(d):
        bits = [(i >> (N - 1 - k)) & 1 for k in range(N)]
        if bits[c]:
            bits[t] ^= 1
        j = sum(b << (N - 1 - k) for k, b in enumerate(bits))
        m[j, i] = 1
    return m


def basis_state(bits):
    v = np.zeros(2 ** N, complex)
    v[sum(b << (N - 1 - k) for k, b in enumerate(bits))] = 1
    return v


def source_state(kind, ctx=(0, 0, 0)):
    """Source register kinds '0','1','+','-'; a, b, t start in the computational context ctx."""
    v0 = basis_state((0,) + tuple(ctx))
    v1 = basis_state((1,) + tuple(ctx))
    return {"0": v0, "1": v1, "+": (v0 + v1) / math.sqrt(2), "-": (v0 - v1) / math.sqrt(2)}[kind]


LAYER1 = [("cx", S, A), ("cx", S, B)]
VARIANTS = {
    "intact": LAYER1 + [("cx", A, T), ("cx", B, T)],
    "path_a_only": LAYER1 + [("cx", A, T)],
    "path_b_only": LAYER1 + [("cx", B, T)],
    "reset_a": LAYER1 + [("reset", A, 0), ("cx", A, T), ("cx", B, T)],       # Lean setRegister: clamp one middle register
    "reset_b": LAYER1 + [("reset", B, 0), ("cx", A, T), ("cx", B, T)],
    "reset_ab": LAYER1 + [("reset", A, 0), ("reset", B, 0), ("cx", A, T), ("cx", B, T)],
    "dephase_a": LAYER1 + [("dephase", A, 0), ("cx", A, T), ("cx", B, T)],    # measure-and-forget a: kills coherence only
    "overwrite_a1": LAYER1 + [("reset", A, 1), ("cx", A, T), ("cx", B, T)],
    # matched two-qubit-gate-count controls
    "echo_intact": LAYER1 + [("cx", A, T), ("cx", A, T), ("cx", B, T), ("cx", B, T)],   # layer 2 is identity: depth control
    "path_a_echo_b": LAYER1 + [("cx", A, T), ("cx", B, T), ("cx", B, T)],               # one live path plus echo pad
    "no_source_link": [("cx", A, T), ("cx", B, T), ("cx", A, T), ("cx", B, T)],         # source never copied: negative control
}


def density(v):
    return np.outer(v, v.conj())


def apply_u(rho, u):
    return u @ rho @ u.conj().T


def kraus_apply(rho, ks):
    return sum(k @ rho @ k.conj().T for k in ks)


def amp_damp(q, g):
    e0 = np.array([[1, 0], [0, math.sqrt(1 - g)]], complex)
    e1 = np.array([[0, math.sqrt(g)], [0, 0]], complex)
    return [op1(e0, q), op1(e1, q)]


def dephase(q, p):
    return [math.sqrt(1 - p) * np.eye(2 ** N), math.sqrt(p) * op1(Z, q)]


def depol2(c, t, p):
    paulis = [I2, X, Y, Z]
    ks = [math.sqrt(1 - p) * np.eye(2 ** N)]
    for a, b in itertools.product(range(4), repeat=2):
        if a == b == 0:
            continue
        m = np.array([[1]], complex)
        for k in range(N):
            m = np.kron(m, paulis[a] if k == c else paulis[b] if k == t else I2)
        ks.append(math.sqrt(p / 15) * m)
    return ks


def reset_channel(rho, q, value):
    k0 = op1(np.array([[1, 0], [0, 0]], complex), q)
    k1 = op1(np.array([[0, 1], [0, 0]], complex), q)
    out = k0 @ rho @ k0.conj().T + k1 @ rho @ k1.conj().T
    if value == 1:
        out = op1(X, q) @ out @ op1(X, q)
    return out


def zzrot(c, t, th):
    d = np.ones(2 ** N, complex)
    for i in range(2 ** N):
        zc = 1 - 2 * ((i >> (N - 1 - c)) & 1)
        zt = 1 - 2 * ((i >> (N - 1 - t)) & 1)
        d[i] = np.exp(-0.5j * th * zc * zt)
    return np.diag(d)


def apply_step(rho, step, noise):
    kind, a, b = step
    if kind == "cx":
        rho = apply_u(rho, cnot(a, b))
        if noise:
            if noise.get("zz_coherent"):
                rho = apply_u(rho, zzrot(a, b, noise["zz_coherent"]))
            if noise.get("p2"):
                rho = kraus_apply(rho, depol2(a, b, noise["p2"]))
            if noise.get("t1_gamma_per_cx"):
                for q in range(N):
                    rho = kraus_apply(rho, amp_damp(q, noise["t1_gamma_per_cx"]))
            if noise.get("crosstalk_zz_to_target") and b != T:
                rho = apply_u(rho, zzrot(a, T, noise["crosstalk_zz_to_target"]))
    elif kind == "reset":
        rho = reset_channel(rho, a, b)
    elif kind == "dephase":
        rho = kraus_apply(rho, dephase(a, 0.5))
    return rho


def readout_probs(rho, basis="Z", e0=0.0, e1=0.0):
    if basis == "XXXZ":
        rho = apply_u(rho, op1(H, S) @ op1(H, A) @ op1(H, B))
    p = np.real(np.diag(rho)).clip(0, 1)
    p = p / p.sum()
    conf = np.array([[1 - e0, e1], [e0, 1 - e1]])
    full = np.array([[1.0]])
    for _ in range(N):
        full = np.kron(full, conf)
    return full @ p


def marg(p, q):
    return sum(p[i] for i in range(2 ** N) if (i >> (N - 1 - q)) & 1)


def parity_expect(p, qs):
    return sum(p[i] * (-1) ** sum((i >> (N - 1 - q)) & 1 for q in qs) for i in range(2 ** N))


def run_state(variant, kind, noise=None, ctx=(0, 0, 0)):
    rho = density(source_state(kind, ctx))
    for st in VARIANTS[variant]:
        rho = apply_step(rho, st, noise)
    return rho


def metrics(variant, noise=None, e0=0.0, e1=0.0, ctx=(0, 0, 0)):
    out = {}
    rhos = {k: run_state(variant, k, noise, ctx) for k in ("0", "1", "+", "-")}
    pz = {k: readout_probs(r, "Z", e0, e1) for k, r in rhos.items()}
    px = {k: readout_probs(r, "XXXZ", e0, e1) for k, r in rhos.items() if k in ("+", "-")}
    pt = {k: marg(v, T) for k, v in pz.items()}
    out["P_t1_given_s"] = {k: float(v) for k, v in pt.items()}
    out["influence_basis_0_vs_1"] = float(pt["1"] - pt["0"])
    out["influence_superposition_plus_vs_minus"] = float(pt["+"] - pt["-"])
    xxx = {k: parity_expect(px[k], (S, A, B)) for k in ("+", "-")}
    out["XXX_plus"] = float(xxx["+"])
    out["XXX_minus"] = float(xxx["-"])
    out["coherence_witness"] = float((xxx["+"] - xxx["-"]) / 2)
    p = pz["+"]
    pop = sum(p[i] for i in range(2 ** N)
              if ((i >> 3) & 1) == ((i >> 2) & 1) == ((i >> 1) & 1) and (i & 1) == 0)
    out["ghz_population_sab_equal_given_t0"] = float(pop)
    out["ghz_fidelity_witness_F"] = float(0.5 * (pop + xxx["+"]))     # F > 1/2 certifies tripartite entanglement of s,a,b

    def red_t(rho):
        return np.einsum("abcdabce->de", rho.reshape([2] * 8))
    d = red_t(rhos["1"]) - red_t(rhos["0"])
    out["target_state_trace_distance_0_vs_1"] = float(0.5 * np.abs(np.linalg.eigvalsh(d)).sum())
    return out


def reachability(variant):
    """Structural reachability: a time-ordered gate path from s to t that no reset cuts."""
    dep = {S}
    for kind, a, b in VARIANTS[variant]:
        if kind == "cx" and a in dep:
            dep.add(b)
        elif kind == "reset":
            dep.discard(a)
    return T in dep


def lean_mirror():
    """Exhaustive Boolean mirror of the Lean diamond F(x0..x3)=(0,x0,x0,x1^x2): source->target response absent over
    all contexts, source->middle and middle->target present, clamping the second middle register restores source response."""
    F = lambda s: (0, s[0], s[0], s[1] ^ s[2])
    two = lambda s: F(F(s))
    ctxs = list(itertools.product((0, 1), repeat=3))
    absent = all(two((0,) + c)[3] == two((1,) + c)[3] for c in ctxs)
    s2m = any(F((0,) + c)[1] != F((1,) + c)[1] for c in ctxs)
    m2t = any(F((0, 0, 0) + (c[2],))[3] != F((0, 1, 0) + (c[2],))[3] for c in ctxs)
    clamp = all(F(tuple(0 if i == 2 else v for i, v in enumerate(F((b,) + c))))[3] == b for b in (0, 1) for c in ctxs)
    return bool(absent and s2m and m2t and clamp)


def exact_checks():
    c = {}
    ok = True
    for ctx in itertools.product((0, 1), repeat=3):
        for kind in ("0", "1"):
            p = np.real(np.diag(run_state("intact", kind, None, ctx)))
            s = int(kind)
            a0, b0, t0 = ctx
            want = basis_state((s, a0 ^ s, b0 ^ s, t0 ^ a0 ^ b0))
            ok &= bool(np.allclose(p, np.abs(want) ** 2))
    c["intact_basis_map_all_8_contexts"] = ok
    c["intact_target_independent_of_s_all_contexts"] = all(
        abs(metrics("intact", ctx=ctx)["influence_basis_0_vs_1"]) < 1e-12 for ctx in itertools.product((0, 1), repeat=3))
    m = {v: metrics(v) for v in VARIANTS}
    c["intact_influence_zero"] = abs(m["intact"]["influence_basis_0_vs_1"]) < 1e-12
    c["intact_superposition_influence_zero"] = abs(m["intact"]["influence_superposition_plus_vs_minus"]) < 1e-12
    c["intact_ghz_coherence_one"] = abs(m["intact"]["coherence_witness"] - 1) < 1e-12 and abs(m["intact"]["ghz_fidelity_witness_F"] - 1) < 1e-12
    c["target_decoupled_from_ghz"] = m["intact"]["target_state_trace_distance_0_vs_1"] < 1e-12
    c["single_path_restores_influence_magnitude_one"] = all(abs(abs(m[v]["influence_basis_0_vs_1"]) - 1) < 1e-12
                                              for v in ("path_a_only", "path_b_only", "reset_a", "reset_b", "overwrite_a1"))
    c["overwrite_a1_response_is_sign_inverted"] = abs(m["overwrite_a1"]["influence_basis_0_vs_1"] + 1) < 1e-12
    c["reset_both_leaves_target_unresponsive"] = abs(m["reset_ab"]["influence_basis_0_vs_1"]) < 1e-12
    c["dephasing_kills_coherence_not_target_response"] = (abs(m["dephase_a"]["influence_basis_0_vs_1"]) < 1e-12
                                                         and abs(m["dephase_a"]["coherence_witness"]) < 1e-12)
    c["reset_destroys_coherence"] = abs(m["reset_a"]["coherence_witness"]) < 1e-12
    c["echo_controls_match_intact_and_single_path_ideals"] = (abs(m["echo_intact"]["influence_basis_0_vs_1"]) < 1e-12
                                                              and abs(m["path_a_echo_b"]["influence_basis_0_vs_1"] - 1) < 1e-12)
    c["no_source_link_zero_response_zero_coherence"] = (abs(m["no_source_link"]["influence_basis_0_vs_1"]) < 1e-12
                                                        and abs(m["no_source_link"]["coherence_witness"]) < 1e-12)
    c["reachable_yet_unresponsive_intact"] = reachability("intact") and abs(m["intact"]["influence_basis_0_vs_1"]) < 1e-12
    c["reset_ab_not_reachable"] = not reachability("reset_ab")
    c["lean_diamond_mirror"] = lean_mirror()
    return c, m


SCENARIOS = {
    "ideal": {},
    "depol_p2_1pct": {"noise": {"p2": 0.01}},
    "t1_gamma_0p004_per_cx": {"noise": {"t1_gamma_per_cx": 0.004}},
    "readout_asym_e0_1pct_e1_3pct": {"e0": 0.01, "e1": 0.03},
    "coherent_zz_0p1": {"noise": {"zz_coherent": 0.1}},
    "crosstalk_zz_0p05_to_target": {"noise": {"crosstalk_zz_to_target": 0.05}},
    "kingston_like_combined": {"noise": {"p2": 0.004, "t1_gamma_per_cx": 0.0015, "zz_coherent": 0.03}, "e0": 0.008, "e1": 0.02},
    "pessimistic_combined": {"noise": {"p2": 0.012, "t1_gamma_per_cx": 0.004, "zz_coherent": 0.08, "crosstalk_zz_to_target": 0.03},
                             "e0": 0.015, "e1": 0.04},
}
KEYS = ("influence_basis_0_vs_1", "influence_superposition_plus_vs_minus", "coherence_witness",
        "ghz_fidelity_witness_F", "target_state_trace_distance_0_vs_1")


def noisy_table():
    table, full = {}, {}
    for sc, cfg in SCENARIOS.items():
        full[sc] = {v: metrics(v, cfg.get("noise"), cfg.get("e0", 0.0), cfg.get("e1", 0.0)) for v in VARIANTS}
        table[sc] = {v: {k: full[sc][v][k] for k in KEYS} for v in VARIANTS}
    return table, full


def apparent_influence(table, rows_by_scenario, shots=2048):
    """Is the intact circuit's apparent influence (ideal 0) distinguishable from its matched echo control (ideal 0)?
    Standard error uses the model's own P(t=1 | s) in the four arms (two circuits x two source values)."""
    out = {}
    for sc, rows in table.items():
        a = rows["intact"]["influence_basis_0_vs_1"]
        e = rows["echo_intact"]["influence_basis_0_vs_1"]
        arms = [rows_by_scenario[sc][v]["P_t1_given_s"][k] for v in ("intact", "echo_intact") for k in ("0", "1")]
        se = math.sqrt(sum(p * (1 - p) for p in arms) / shots)
        out[sc] = {"intact_apparent_influence": a, "echo_control_apparent_influence": e, "intact_minus_echo": a - e,
                   "se_binomial_model": se, "shots_per_arm": shots, "intact_over_3se": bool(abs(a) > 3 * se),
                   "intact_minus_echo_over_3se": bool(abs(a - e) > 3 * se),
                   "single_path_true_influence": rows["path_a_only"]["influence_basis_0_vs_1"]}
    return out


def main():
    checks, ideal = exact_checks()
    table, full = noisy_table()
    res = {"evidence_class": "exact statevector/density matrix and noisy simulation; no hardware",
           "variants": {k: [list(s) for s in v] for k, v in VARIANTS.items()},
           "two_qubit_gate_counts": {k: sum(1 for s in v if s[0] == "cx") for k, v in VARIANTS.items()},
           "structural_reachability_s_to_t": {v: reachability(v) for v in VARIANTS},
           "exact_checks": checks, "ideal_metrics": ideal, "noisy": table, "apparent_influence": apparent_influence(table, full)}
    (HERE / "qpu02_sim_results.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
