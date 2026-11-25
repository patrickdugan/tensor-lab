"""QC-05: global Hilbert-Schmidt selection with a bounded TN reference.

QC-02 sequential and QC-04 screened-gauge simulations run independently at
the same caps. A third, higher-capacity locally purified network supplies an
approximate global density reference. Selection uses only MPO contractions,
never a dense reference matrix. For small systems the exact density is used
after selection to validate physical trace distance. The teacher itself is an
additional simulation cost and is also benchmarked as a stand-alone baseline.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np

from channel_mpo import gate_layers, kraus, apply_operator, apply_channel
from purified_mpo import PurifiedChain
from gauge_pair import PairPurifiedChain, trace_distance
from qc05_observables import pauli_expectation, sketch_words
from qc05_global import hs_inner


def select_global(teacher: PurifiedChain, sequential: PurifiedChain,
                  pair: PurifiedChain, metric: str = 'hs', width: int = 8,
                  tie_epsilon: float = 1e-13) -> tuple[str, float, float, float]:
    """Choose between two bounded outputs using physical density scores."""
    if metric not in ('hs', 'sketch') or width < 1 or tie_epsilon < 0:
        raise ValueError('unsupported metric, width or tolerance')
    if sequential.n != pair.n or pair.n != teacher.n:
        raise ValueError('mismatched number of sites')
    t0 = perf_counter()
    if metric == 'hs':
        teacher_trace = teacher.trace()
        def score(candidate):
            norm = candidate.trace()
            return (hs_inner(candidate, candidate)/norm**2
                    - 2*hs_inner(teacher, candidate)/(teacher_trace*norm))
        s0, s1 = score(sequential), score(pair)
    else:
        words = sketch_words(teacher.n, width)
        target = np.array([pauli_expectation(teacher, w) for w in words])
        def score(candidate):
            values = np.array([pauli_expectation(candidate, w) for w in words])
            return float(np.mean((target-values)**2))
        s0, s1 = score(sequential), score(pair)
    decision = 'pair' if s1 < s0-tie_epsilon else 'sequential'
    return decision, float(s0), float(s1), perf_counter()-t0


def run_portfolio(n: int, depth: int, seed: int, p: float,
                  bond: int = 12, env: int = 4,
                  teacher_bond: int = 16, teacher_env: int = 8,
                  metric: str = 'hs', width: int = 8, validate: bool = True,
                  channel: str = 'depolarizing',
                  mid_baseline: bool = True) -> dict:
    """Run fixed-cap policies and teacher, with an independent cap-6 control.

    The exact small-system oracle is evaluated only after the policy choice.
    Timings exclude all dense-reference work. Only two candidate scores are
    evaluated. The cap-6 control is excluded from selector cost.
    """
    if n < 2 or depth < 1 or min(bond, env, teacher_bond, teacher_env) < 1:
        raise ValueError('invalid circuit or rank cap')
    if validate and n > 8:
        raise ValueError('dense validation capped at eight qubits')
    ops = kraus(channel, p)
    gates = gate_layers(n, depth, seed)
    chains = {
        'sequential': PurifiedChain(n, bond, env),
        'pair': PairPurifiedChain(n, bond, env, 0, proposals=0, seed=seed,
                                 factor_swaps=True, fresh_k=len(ops),
                                 fast_factor_search=True),
        'teacher': PurifiedChain(n, teacher_bond, teacher_env),
    }
    if mid_baseline:
        chains['mid_env6'] = PurifiedChain(n, bond, max(env, 6))
    seconds = {name: 0. for name in chains}
    min_retention = {name: 1. for name in chains}
    for layer in gates:
        for name, chain in chains.items():
            tic = perf_counter()
            for i, u in layer:
                chain.gate(i, u)
            for i in range(n):
                chain.channel(i, ops)
            if name == 'pair':
                chain.compress_layer()
            else:
                for i in range(n):
                    chain.compress_env(i)
            retained = chain.normalize()
            min_retention[name] = min(min_retention[name], retained)
            seconds[name] += perf_counter()-tic
    decision, s0, s1, score_seconds = select_global(
        chains['teacher'], chains['sequential'], chains['pair'], metric, width)
    oracle_seconds = 0.
    distances = {name: float('nan') for name in chains}
    if validate:
        tic = perf_counter()
        ref = np.zeros((2**n, 2**n), complex)
        ref[0, 0] = 1.
        for layer in gates:
            for i, u in layer:
                ref = apply_operator(ref, u, (i, i+1), n)
            for i in range(n):
                ref = apply_channel(ref, ops, i, n)
        distances = {
            name: trace_distance(ref, chain.dense_density())
            for name, chain in chains.items()
        }
        oracle_seconds = perf_counter()-tic
    return {
        'n': n, 'depth': depth, 'seed': seed, 'channel': channel, 'p': p,
        'bond_cap': bond, 'env_cap': env,
        'teacher_bond_cap': teacher_bond, 'teacher_env_cap': teacher_env,
        'metric': metric, 'width': width, 'selected': decision,
        'score_seq': s0, 'score_pair': s1, 'decision_margin': s0-s1,
        'distance_selected': distances[decision],
        'distance_seq': distances['sequential'],
        'distance_pair': distances['pair'],
        'distance_teacher': distances['teacher'],
        'distance_mid_env6': distances.get('mid_env6', float('nan')),
        'correct_relative_choice': (
            float(distances[decision] <= min(distances['sequential'],
                                            distances['pair'])+1e-10)
            if validate else float('nan')),
        'seconds_seq': seconds['sequential'],
        'seconds_pair': seconds['pair'],
        'seconds_teacher': seconds['teacher'],
        'seconds_mid_env6': seconds.get('mid_env6', float('nan')),
        'seconds_score': score_seconds,
        'seconds_selection_total': (seconds['sequential']+seconds['pair']
                                    + seconds['teacher']+score_seconds),
        'seconds_oracle_excluded': oracle_seconds,
        'retention_seq_min': min_retention['sequential'],
        'retention_pair_min': min_retention['pair'],
        'retention_teacher_min': min_retention['teacher'],
        'entries_seq': chains['sequential'].dimensions()['purified_entries'],
        'entries_pair': chains['pair'].dimensions()['purified_entries'],
        'entries_teacher': chains['teacher'].dimensions()['purified_entries'],
    }
