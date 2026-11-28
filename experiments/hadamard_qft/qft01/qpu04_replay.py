"""HAD-QFT-QPU04 independent statistical replay from raw shot counts.

Shares no analysis code with qpu03/qpu04: counts.json and compiled.json (target index, ideal
distributions) are read directly. Verifies the registered joint evaluator, then reports secondary
probes, controls and exploratory comparisons (labelled as such). Output: hardware_results/qpu04/replay.json
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'hardware_results'
SHOTS, ALPHA = 1024, 0.05
PROBES = ('edge', 'middle', 'tail', 'quarter', 'threequarter')
RAD_JOINT = math.sqrt(2*math.log(4/ALPHA)/SHOTS)          # registered: two primary differences, Bonferroni
RAD_SINGLE_D = math.sqrt(2*math.log(2/ALPHA)/SHOTS)       # one unadjusted difference of two contrasts (descriptive)
RAD_AVG = math.sqrt(math.log(2/ALPHA)/SHOTS)              # mean of two independent differences (exploratory)


def load(stage):
    d = ROOT / stage
    return json.loads((d/'compiled.json').read_text()), json.loads((d/'counts.json').read_text()), json.loads((d/'execution_receipt.json').read_text())


def dist(counts, b=7):
    p = np.zeros(1 << b); tot = 0
    for k, v in counts.items():
        assert len(k) == b and set(k) <= {'0', '1'}
        p[int(k[::-1], 2)] += v; tot += v          # clbit v = wire v, wire 0 is the most significant label bit
    assert tot == SHOTS
    return p/SHOTS


def stage_stats(stage):
    m, counts, rc = load(stage)
    rows = {r['id']: r for r in m['rows']}
    P = {rid: dist(c) for rid, c in counts.items()}
    out = dict(stage=stage, job_id=rc['job_id'], plan_sha256=m['plan_sha256'], job_usage=rc['job_usage'], arms={})
    for arm in ('A', 'D'):
        a = dict(contrast={}, tv={}, hellinger={}, controls={})
        for pr in PROBES:
            mi, pl = rows[f'{arm}_{pr}_minus'], rows[f'{arm}_{pr}_plus']
            t = mi['target']; assert t == pl['target']
            a['contrast'][pr] = float(P[mi['id']][t] - P[pl['id']][t])
        ph = [rid for rid, r in rows.items() if r['arm'] == arm and r['kind'] == 'core' and r['state'] in ('plus', 'minus', 'imag')]
        for rid in ph:
            q = np.array(rows[rid]['ideal']); p = P[rid]
            a['tv'][rid] = float(abs(p-q).sum()/2); a['hellinger'][rid] = float(np.sqrt(p*q).sum()**2)
        for rid, r in rows.items():
            if r['arm'] == arm and r['kind'] in ('prep', 'spam'): a['controls'][rid] = float(P[rid][r['target']])
        a['mean_tv'] = float(np.mean(list(a['tv'].values()))); a['mean_hellinger'] = float(np.mean(list(a['hellinger'].values())))
        a['mean_control_success'] = float(np.mean(list(a['controls'].values())))
        # incoherent-mixture check at the edge: average of the two basis endpoints against the imaginary-phase row
        mix = (P[f'{arm}_edge_basis0'] + P[f'{arm}_edge_basis1'])/2
        a['edge_imag_vs_mixture_tv'] = float(abs(P[f'{arm}_edge_imag'] - mix).sum()/2)
        out['arms'][arm] = a
    diffs = {pr: out['arms']['D']['contrast'][pr] - out['arms']['A']['contrast'][pr] for pr in PROBES}
    out['D_minus_A'] = diffs
    d0 = diffs['edge']; out['primary'] = dict(difference=d0, joint_interval=[d0-RAD_JOINT, d0+RAD_JOINT], lower_positive=bool(d0-RAD_JOINT > 0))
    out['secondary_unadjusted_intervals'] = {pr: [diffs[pr]-RAD_SINGLE_D, diffs[pr]+RAD_SINGLE_D] for pr in PROBES}
    return out


def main():
    s0, s1 = stage_stats('qpu04_b7_r0'), stage_stats('qpu04_b7_r1')
    reg = json.loads((Path(sys.argv[1]) if len(sys.argv) > 1 else HERE/'results_local'/'qpu04_joint.json').read_text())
    for s, r in zip((s0, s1), reg['regions']):
        assert abs(s['primary']['difference'] - r['edge_difference']) < 1e-12 and abs(s['primary']['joint_interval'][0] - r['interval'][0]) < 1e-12
        for arm in ('A', 'D'):
            for pr in PROBES: assert abs(s['arms'][arm]['contrast'][pr] - r['contrast'][arm][pr]) < 1e-12
    both = s0['primary']['lower_positive'] and s1['primary']['lower_positive']
    assert both == reg['primary_both_positive']
    mean_edge = (s0['D_minus_A']['edge'] + s1['D_minus_A']['edge'])/2
    expl = dict(label='EXPLORATORY, not registered', mean_edge_D_minus_A=mean_edge, unadjusted_interval=[mean_edge-RAD_AVG, mean_edge+RAD_AVG], radius=RAD_AVG,
                n_probe_region_pairs=10, n_D_above_A=sum(1 for s in (s0, s1) for pr in PROBES if s['D_minus_A'][pr] > 0))
    # exploratory: native entangling count versus edge contrast across all campaigns
    pts = []
    q2 = json.loads((ROOT/'qpu02_summary.json').read_text())
    for st in q2['stages']:
        for arm, r in st['resources'].items():
            c = next(x for x in st['contrasts'] if x['arm'] == arm and x['probe'] == 'edge')
            pts.append(dict(campaign='QPU02', stage=st['label'], arm=arm, b=st['b'], native_cz=r['native_2q'], edge=c['contrast']))
    q3 = json.loads((ROOT/'qpu03_summary.json').read_text())
    for st in q3['stages']:
        for arm, r in st['arms'].items():
            c = next(x for x in st['signal'] if x['arm'] == arm and x['probe'] == 'edge')
            pts.append(dict(campaign='QPU03', stage=st['label'], arm=arm, b=st['b'], native_cz=r['native_2q'], edge=c['contrast']))
    nat = {'A': 276, 'D': 199}
    for s in (s0, s1):
        for arm in ('A', 'D'): pts.append(dict(campaign='QPU04', stage=s['stage'], arm=arm, b=7, native_cz=nat[arm], edge=s['arms'][arm]['contrast']['edge']))
    from scipy.stats import spearmanr
    rho, pval = spearmanr([p['native_cz'] for p in pts], [p['edge'] for p in pts])
    seven = [p for p in pts if p['b'] == 7]
    rho7 = spearmanr([p['native_cz'] for p in seven], [p['edge'] for p in seven])
    expl['spearman_all_campaigns'] = dict(rho=float(rho), p=float(pval), n=len(pts), label='exploratory; unregistered; points are not independent')
    expl['seven_qubit_points'] = seven
    out = dict(regions=[s0, s1], registered_primary_both_positive=both, registered_radius=RAD_JOINT, exploratory=expl, points=pts)
    dest = ROOT/'qpu04'; dest.mkdir(parents=True, exist_ok=True); (dest/'replay.json').write_text(json.dumps(out, indent=2, sort_keys=True)+'\n', encoding='utf8')
    print('registered radius', round(RAD_JOINT, 4), '| both lower bounds > 0:', both)
    for s in (s0, s1):
        print(s['stage'], 'job', s['job_id'], 'D-A edge', round(s['D_minus_A']['edge'], 4), 'joint interval', [round(x, 4) for x in s['primary']['joint_interval']])
        print('   D-A all probes', {k: round(v, 3) for k, v in s['D_minus_A'].items()})
        for arm in ('A', 'D'):
            a = s['arms'][arm]; print('   ', arm, 'edge', round(a['contrast']['edge'], 3), 'meanTV', round(a['mean_tv'], 3), 'HF', round(a['mean_hellinger'], 3), 'controls', round(a['mean_control_success'], 3), 'imag-vs-mix TV', round(a['edge_imag_vs_mixture_tv'], 3))
    print('exploratory mean edge D-A', round(mean_edge, 4), [round(x, 4) for x in expl['unadjusted_interval']], '| D>A in', expl['n_D_above_A'], 'of 10 probe-region pairs')
    print('spearman native CZ vs edge contrast (all campaigns, n=%d): rho=%.3f p=%.3g' % (len(pts), rho, pval))


if __name__ == '__main__':
    main()
