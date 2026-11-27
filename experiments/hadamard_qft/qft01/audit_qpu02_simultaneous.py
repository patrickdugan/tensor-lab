"""Independent reproduction of the QPU02 simultaneous-confidence audit from raw counts.

Primary comparison family (fixed in advance): approximate-compact minus recursive phase
contrast, three probes (edge, middle, tail) in each of the two b=5 replicates, so m = 6.

Hoeffding for a contrast C = P(n|minus) - P(n|plus) built from two independent n-shot samples:
a sum of 2n independent terms each of range 1/n, so
    P(|C - EC| >= t) <= 2 exp(-n t^2).
For a difference of two such contrasts (4n terms): P(|D - ED| >= t) <= 2 exp(-n t^2 / 2).
Bonferroni over m simultaneous statements at level alpha uses alpha/m in each.
No code is shared with qpu02.analyze; counts are read directly.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def target_prob(counts, b, target, shots):
    # Qiskit bitstring is c_(b-1)...c_0 and classical bit i holds wire i; the shell label has
    # wire 0 as its most significant bit, so the label is the bitstring reversed.
    tot = sum(v for k, v in counts.items() if int(k[::-1], 2) == target)
    return tot / shots


def contrast(rows, counts, arm, probe, b, shots):
    res = {}
    for state in ('minus', 'plus'):
        r = next(x for x in rows if x['arm'] == arm and x['probe'] == probe and x['state'] == state)
        res[state] = target_prob(counts[r['id']], b, r['target'], shots)
    return res['minus'] - res['plus']


def radius_c(n, alpha): return math.sqrt(math.log(2/alpha) / n)
def radius_d(n, alpha): return math.sqrt(2*math.log(2/alpha) / n)


def audit(stage_dirs, alpha=0.05):
    stats = []
    for d in stage_dirs:
        m = json.load(open(d / 'compiled.json')); counts = json.load(open(d / 'counts.json'))
        b, n = m['b'], m['shots']
        for probe in ('edge', 'middle', 'tail'):
            ca = contrast(m['rows'], counts, 'compact_approx', probe, b, n)
            cr = contrast(m['rows'], counts, 'recursive', probe, b, n)
            stats.append(dict(stage=d.name, probe=probe, approx=ca, recursive=cr, diff=ca - cr, n=n))
    mtests = len(stats)
    out = []
    for s in stats:
        r1 = radius_d(s['n'], alpha); rb = radius_d(s['n'], alpha / mtests)
        out.append({**s, 'radius_unadjusted': r1, 'radius_bonferroni': rb,
                    'survives_unadjusted': s['diff'] - r1 > 0, 'survives_simultaneous': s['diff'] - rb > 0,
                    'lower_bound_simultaneous': s['diff'] - rb})
    return dict(alpha=alpha, m=mtests, family='compact_approx minus recursive, edge/middle/tail, replicates 0 and 1',
                comparisons=out, n_survive_simultaneous=sum(o['survives_simultaneous'] for o in out),
                edge_both_survive=all(o['survives_simultaneous'] for o in out if o['probe'] == 'edge'))


if __name__ == '__main__':
    root = HERE / 'hardware_results'
    rep = audit([root / 'qpu02_b5_r0', root / 'qpu02_b5_r1'])
    out = root / 'qpu02_simultaneous_audit.json'
    out.write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n', encoding='utf8')
    for o in rep['comparisons']:
        print(o['stage'], o['probe'], f"diff {o['diff']:+.4f}", f"radius_bonf {o['radius_bonferroni']:.4f}", f"lower {o['lower_bound_simultaneous']:+.4f}", 'SURVIVES' if o['survives_simultaneous'] else 'does not survive')
    print('survive:', rep['n_survive_simultaneous'], 'of', rep['m'], '| both edge survive:', rep['edge_both_survive'])
