"""HAD-QFT-QPU01 hardware analysis: raw-count metrics, paired comparison, figures, LaTeX tables.

Inputs per stage directory: compiled.json, counts.json (raw counts, never modified),
execution_receipt.json. Outputs: analysis_full.json, tables_<stage>.tex, figures.
Primary metrics use raw counts only. The SPAM-corrected block is a labeled secondary
sensitivity analysis. Uncertainty from finite shots is a multinomial bootstrap on the
observed counts plus an ideal-sampling floor; it does not include drift or other
systematic hardware error, which is the quantity the comparison is trying to measure.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

SHOTS = 384
NBOOT = 4000
ARMS = ('gray', 'recursive')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, sort_keys=True) + '\n', encoding='utf8')


def to_prob(counts, b):
    """Qiskit bitstring c_(b-1)..c_0 -> shell label index with q_0 as most significant."""
    p = np.zeros(1 << b)
    for k, v in counts.items(): p[int(k[::-1], 2)] += v
    return p / p.sum()


def tv(p, q): return float(abs(p - q).sum() / 2)
def hfid(p, q): return float(np.sqrt(p * q).sum() ** 2)


def parity(p, b):
    s = np.array([(-1) ** bin(i).count('1') for i in range(1 << b)])
    return float((s * p).sum())


def ideal_floor(q, rng, n=4000):
    """Distribution of TV(empirical, ideal) when hardware were perfect (shot noise only)."""
    c = rng.multinomial(SHOTS, q, size=n) / SHOTS
    t = abs(c - q).sum(1) / 2
    h = np.sqrt(c * q).sum(1) ** 2
    return dict(tv_mean=float(t.mean()), tv_q95=float(np.quantile(t, .95)),
                hf_mean=float(h.mean()), hf_q05=float(np.quantile(h, .05)))


def boot(counts_vec, fn, rng, n=NBOOT):
    p = counts_vec / counts_vec.sum()
    s = rng.multinomial(int(counts_vec.sum()), p, size=n) / counts_vec.sum()
    return np.array([fn(x) for x in s])


def ci(x): return [float(np.quantile(x, .025)), float(np.quantile(x, .975))]


def wilson(k, n, z=1.96):
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [max(0., c - h), min(1., c + h)]


def spam_correct(rows, probs, b):
    """Secondary only: invert the measured basis-state confusion matrix with NNLS."""
    from scipy.optimize import nnls
    d = 1 << b
    A = np.zeros((d, d))
    for r in rows:
        if r['arm'] == 'spam': A[:, int(r['probe'])] = probs[r['id']]
    out = {}
    W = 1e3
    for r in rows:
        if r['arm'] in ARMS and r['basis'] != 'echo' and r['basis'] == 'Z':
            M = np.vstack([A, W * np.ones(d)]); y = np.append(probs[r['id']], W)
            x, _ = nnls(M, y); out[r['id']] = x / x.sum()
    return out


def analyze_stage(stage_dir, label):
    stage_dir = Path(stage_dir)
    m = read(stage_dir / 'compiled.json'); counts = read(stage_dir / 'counts.json'); b = m['b']
    rng = np.random.default_rng(20261127 + b)
    rows = m['rows']
    assert set(counts) == {r['id'] for r in rows}
    probs, cvec = {}, {}
    for r in rows:
        c = counts[r['id']]
        assert sum(c.values()) == SHOTS
        probs[r['id']] = to_prob(c, b)
        cvec[r['id']] = np.rint(probs[r['id']] * SHOTS)
    res = {}
    for r in rows:
        p, q = probs[r['id']], np.asarray(r['ideal']); cv = cvec[r['id']]
        e = dict(id=r['id'], arm=r['arm'], probe=r['probe'], basis=r['basis'],
                 tv=tv(p, q), hf=hfid(p, q), p0=float(p[0]),
                 cz=r['metrics']['num_two_qubit'], depth=r['metrics']['depth'],
                 cz_depth=r['metrics']['two_qubit_depth'], duration_us=r['duration_s'] * 1e6,
                 logical_cx=r['logical_cx'], line_cx=r['line_cx'])
        t = boot(cv, lambda x: tv(x, q), rng); e['tv_ci'] = ci(t)
        h = boot(cv, lambda x: hfid(x, q), rng); e['hf_ci'] = ci(h)
        e['floor'] = ideal_floor(q, rng)
        e['excess_tv'] = e['tv'] - e['floor']['tv_mean']
        if r['basis'] in ('X', 'Y'):
            e['parity_obs'] = parity(p, b); e['parity_ideal'] = parity(q, b)
            pb = boot(cv, lambda x: parity(x, b), rng); e['parity_ci'] = ci(pb)
        if r['basis'] == 'echo':
            k = int(round(p[0] * SHOTS)); e['p0_ci'] = wilson(k, SHOTS)
        res[r['id']] = e
    # resource table
    def arm_core(a):
        r = res[f'{a}_zero_Z']
        return dict(logical_cx=r['logical_cx'], line_cx=r['line_cx'], cz=r['cz'], depth=r['depth'],
                    cz_depth=r['cz_depth'], duration_us=r['duration_us'])
    resources = {a: arm_core(a) for a in ARMS}
    # paired comparisons (gray minus recursive); positive favors recursive for tv, negative for hf/echo
    pairs = []
    keys = sorted({(r['probe'], r['basis']) for r in rows if r['arm'] == 'gray'})
    for probe, basis in keys:
        g = res[f'gray_{probe}_{basis}']; rc = res[f'recursive_{probe}_{basis}']
        if basis == 'echo':
            cg, cr = cvec[g['id']], cvec[rc['id']]
            d = boot(cg, lambda x: x[0], rng) - boot(cr, lambda x: x[0], rng)
            pairs.append(dict(probe=probe, basis=basis, metric='echo_return', gray=g['p0'], recursive=rc['p0'],
                              diff_rec_minus_gray=rc['p0'] - g['p0'],
                              diff_ci=ci(-d), cz_gray=g['cz'], cz_rec=rc['cz'],
                              dur_gray_us=g['duration_us'], dur_rec_us=rc['duration_us']))
        else:
            qg, qr = np.asarray(next(x for x in rows if x['id'] == g['id'])['ideal']), np.asarray(next(x for x in rows if x['id'] == rc['id'])['ideal'])
            dt = boot(cvec[g['id']], lambda x: tv(x, qg), rng) - boot(cvec[rc['id']], lambda x: tv(x, qr), rng)
            pairs.append(dict(probe=probe, basis=basis, metric='tv', gray=g['tv'], recursive=rc['tv'],
                              diff_gray_minus_rec=g['tv'] - rc['tv'], diff_ci=ci(dt),
                              hf_gray=g['hf'], hf_rec=rc['hf'], cz_gray=g['cz'], cz_rec=rc['cz'],
                              dur_gray_us=g['duration_us'], dur_rec_us=rc['duration_us']))
    fw = [p for p in pairs if p['metric'] == 'tv']; ec = [p for p in pairs if p['metric'] == 'echo_return']
    summary = {}
    for a in ARMS:
        f = [v for v in res.values() if v['arm'] == a and v['basis'] != 'echo']
        z = [v for v in f if v['basis'] == 'Z']; c = [v for v in f if v['basis'] in ('X', 'Y')]
        e = [v for v in res.values() if v['arm'] == a and v['basis'] == 'echo']
        summary[a] = dict(mean_tv=float(np.mean([v['tv'] for v in f])), mean_tv_Z=float(np.mean([v['tv'] for v in z])),
                          mean_tv_coherent=float(np.mean([v['tv'] for v in c])),
                          mean_hf=float(np.mean([v['hf'] for v in f])), mean_hf_coherent=float(np.mean([v['hf'] for v in c])),
                          mean_echo=float(np.mean([v['p0'] for v in e])), mean_floor_tv=float(np.mean([v['floor']['tv_mean'] for v in f])),
                          n_forward=len(f), n_echo=len(e))
    sp = [v for v in res.values() if v['arm'] == 'spam']; pe = [v for v in res.values() if v['arm'] == 'prep']
    controls = dict(spam_mean_correct=float(np.mean([v['p0'] if v['probe'] == '0' else
                                                     probs[v['id']][int(v['probe'])] for v in sp])),
                    spam_min_correct=float(min(probs[v['id']][int(v['probe'])] for v in sp)),
                    prep_echo_return={v['probe']: v['p0'] for v in pe})
    # whole-probe-set paired difference with bootstrap (finite-shot only)
    mean_diff = [p['diff_gray_minus_rec'] for p in fw]
    boots = []
    for _ in range(NBOOT // 4):
        s = 0.
        for p in fw:
            g = res[f"gray_{p['probe']}_{p['basis']}"]; rc = res[f"recursive_{p['probe']}_{p['basis']}"]
            qg = np.asarray(next(x for x in rows if x['id'] == g['id'])['ideal']); qr = np.asarray(next(x for x in rows if x['id'] == rc['id'])['ideal'])
            xg = rng.multinomial(SHOTS, probs[g['id']] / probs[g['id']].sum()) / SHOTS
            xr = rng.multinomial(SHOTS, probs[rc['id']] / probs[rc['id']].sum()) / SHOTS
            s += tv(xg, qg) - tv(xr, qr)
        boots.append(s / len(fw))
    paired_mean = dict(mean_gray_minus_recursive_tv=float(np.mean(mean_diff)), ci95_shot_only=ci(np.array(boots)),
                       n_pairs=len(fw), n_recursive_better=int(sum(d > 0 for d in mean_diff)),
                       n_gray_better=int(sum(d < 0 for d in mean_diff)), n_ties=int(sum(d == 0 for d in mean_diff)))
    sc = spam_correct(rows, probs, b)
    sens = {k: dict(tv_raw=res[k]['tv'], tv_spam_corrected=tv(v, np.asarray(next(x for x in rows if x['id'] == k)['ideal']))) for k, v in sc.items()}
    receipt = read(stage_dir / 'execution_receipt.json') if (stage_dir / 'execution_receipt.json').exists() else {}
    out = dict(stage=label, b=b, backend=m['backend'], plan_sha256=m['plan_sha256'], shots=SHOTS, path=m['path'],
               calibration_timestamp=read(stage_dir / 'calibration.json').get('timestamp'),
               resources=resources, summary=summary, paired_mean=paired_mean, pairs=pairs, controls=controls,
               spam_corrected_secondary=sens, rows=list(res.values()), job_id=receipt.get('job_id'),
               usage_before=receipt.get('usage_before'), usage_after=receipt.get('usage_after'), job_usage=receipt.get('job_usage'),
               note='Primary metrics are raw counts. CI are multinomial finite-shot only and exclude drift and systematics.')
    write(stage_dir / 'analysis_full.json', out)
    return out


def fmt(x, n=3): return f'{x:.{n}f}'


def latex_tables(a, path):
    b = a['b']; r = a['resources']; s = a['summary']
    L = []
    L.append(r'\begin{table}[htbp]\centering\small')
    L.append(rf'\caption{{Stage $b={b}$ compiled resources for the core transform on {a["backend"]} (physical qubits {a["path"]}). Counts are IBM ISA native gates at optimization level 0; duration is the calibrated schedule length of one shot.}}')
    L.append(rf'\label{{tab:hw-res-b{b}}}\begin{{tabular}}{{lrr}}\toprule Quantity & Gray shell & Recursive\\\midrule')
    for k, n in (('logical_cx', 'Logical CNOT (with label conversion)'), ('line_cx', 'Nearest-neighbor CNOT after bridging'),
                 ('cz', 'Native two-qubit gates'), ('cz_depth', 'Two-qubit depth'), ('depth', 'Circuit depth'),
                 ('duration_us', 'Duration ($\\mu$s)')):
        g, c = r['gray'][k], r['recursive'][k]
        L.append(f'{n} & {fmt(g,2) if k=="duration_us" else int(g)} & {fmt(c,2) if k=="duration_us" else int(c)}\\\\')
    L.append(r'\bottomrule\end{tabular}\end{table}')
    L.append('')
    L.append(r'\begin{table}[htbp]\centering\small')
    L.append(rf'\caption{{Stage $b={b}$ paired forward-transform results, raw counts, 384 shots per circuit. $\Delta=\mathrm{{TV}}_{{\rm Gray}}-\mathrm{{TV}}_{{\rm rec}}$ (positive favors the recursive circuit); brackets are 95\% multinomial bootstrap intervals from finite shots only. The ideal-sampling floor is the mean TV a perfect device would show at this shot count.}}')
    L.append(rf'\label{{tab:hw-pairs-b{b}}}\begin{{tabular}}{{llrrrcrr}}\toprule Probe & Basis & TV Gray & TV rec & $\Delta$ & 95\% interval & $F_H$ Gray & $F_H$ rec\\\midrule')
    for p in a['pairs']:
        if p['metric'] != 'tv': continue
        L.append(f"{p['probe']} & {p['basis']} & {fmt(p['gray'])} & {fmt(p['recursive'])} & {p['diff_gray_minus_rec']:+.3f} & [{p['diff_ci'][0]:+.3f},{p['diff_ci'][1]:+.3f}] & {fmt(p['hf_gray'])} & {fmt(p['hf_rec'])}\\\\")
    pm = a['paired_mean']
    L.append(r'\midrule')
    L.append(rf"Mean & & {fmt(s['gray']['mean_tv'])} & {fmt(s['recursive']['mean_tv'])} & {pm['mean_gray_minus_recursive_tv']:+.3f} & [{pm['ci95_shot_only'][0]:+.3f},{pm['ci95_shot_only'][1]:+.3f}] & {fmt(s['gray']['mean_hf'])} & {fmt(s['recursive']['mean_hf'])}\\")
    L.append(rf"Ideal floor & & {fmt(s['gray']['mean_floor_tv'])} & {fmt(s['recursive']['mean_floor_tv'])} & & & &\\")
    L.append(r'\bottomrule\end{tabular}\end{table}')
    L.append('')
    L.append(r'\begin{table}[htbp]\centering\small')
    L.append(rf'\caption{{Stage $b={b}$ inverse-transform echoes and controls (ideal return probability 1; 95\% Wilson intervals). Prep-only rows contain the state preparation and its inverse without the transform.}}')
    L.append(rf'\label{{tab:hw-echo-b{b}}}\begin{{tabular}}{{lrrc}}\toprule Circuit & Probe & Return probability & 95\% interval\\\midrule')
    for v in a['rows']:
        if v['basis'] == 'echo':
            L.append(f"{v['arm']} & {v['probe']} & {fmt(v['p0'])} & [{v['p0_ci'][0]:.3f},{v['p0_ci'][1]:.3f}]\\\\")
    c = a['controls']
    L.append(r'\midrule')
    L.append(rf"SPAM, mean correct readout & all {1<<b} basis states & {fmt(c['spam_mean_correct'])} & min {fmt(c['spam_min_correct'])}\\")
    L.append(r'\bottomrule\end{tabular}\end{table}')
    Path(path).write_text('\n'.join(L) + '\n', encoding='utf8')


def figures(stages, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
                         'font.family': 'serif', 'axes.linewidth': .6})
    col = {'gray': '#D55E00', 'recursive': '#0072B2'}; mk = {'gray': 's', 'recursive': 'o'}; hat = {'gray': '//', 'recursive': ''}
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    # Fig 1: resources
    fig, ax = plt.subplots(1, len(stages), figsize=(6.4, 2.4), squeeze=False)
    for j, a in enumerate(stages):
        r = a['resources']; labs = ['Native 2q gates', '2q depth', 'Duration (us)']
        keys = ['cz', 'cz_depth', 'duration_us']; x = np.arange(len(keys)); w = .36
        for i, arm in enumerate(ARMS):
            vals = [r[arm][k] for k in keys]
            bars = ax[0, j].bar(x + (i - .5) * w, vals, w, color=col[arm], hatch=hat[arm], edgecolor='white' if arm == 'recursive' else 'black', linewidth=.5, label='Gray shell' if arm == 'gray' else 'Recursive')
            for bb, v in zip(bars, vals): ax[0, j].text(bb.get_x() + bb.get_width() / 2, v, f'{v:.0f}' if v > 20 else f'{v:.1f}', ha='center', va='bottom', fontsize=7)
        ax[0, j].set_xticks(x); ax[0, j].set_xticklabels(labs); ax[0, j].set_title(f'$b={a["b"]}$ ({2**(2**a["b"]-1)} positions)')
        ax[0, j].set_yscale('log')
    ax[0, 0].legend(frameon=False, loc='upper left'); ax[0, 0].set_ylabel('Count (log scale)')
    fig.tight_layout(); fig.savefig(outdir / 'hw_resources.pdf'); fig.savefig(outdir / 'hw_resources.png', dpi=200); plt.close(fig)
    # Fig 2: paired TV per probe with CI, plus floor
    fig, ax = plt.subplots(1, len(stages), figsize=(6.4, 2.9), squeeze=False)
    for j, a in enumerate(stages):
        fw = [p for p in a['pairs'] if p['metric'] == 'tv']; x = np.arange(len(fw))
        for arm in ARMS:
            ys = [a_['tv'] for a_ in a['rows'] if False] or None
        for i, arm in enumerate(ARMS):
            ys = [p[arm] for p in fw]
            lo = [a_row['tv_ci'] for a_row in (next(v for v in a['rows'] if v['id'] == f"{arm}_{p['probe']}_{p['basis']}") for p in fw)]
            ax[0, j].errorbar(x + (i - .5) * .18, ys, yerr=[[y - l[0] for y, l in zip(ys, lo)], [l[1] - y for y, l in zip(ys, lo)]],
                              fmt=mk[arm], color=col[arm], ms=4.5, capsize=2, lw=.8, label='Gray shell' if arm == 'gray' else 'Recursive')
        fl = np.mean([v['floor']['tv_mean'] for v in a['rows'] if v['arm'] in ARMS and v['basis'] != 'echo'])
        ax[0, j].axhline(fl, color='0.45', lw=.7, ls=':'); ax[0, j].text(len(fw) - .5, fl, 'ideal-sampling floor', ha='right', va='bottom', fontsize=7, color='0.35')
        ax[0, j].set_xticks(x); ax[0, j].set_xticklabels([f"{p['probe']}\n{p['basis']}" for p in fw], fontsize=7)
        ax[0, j].set_title(f'$b={a["b"]}$'); ax[0, j].set_ylim(0, 1.0)
    ax[0, 0].set_ylabel('Total variation from ideal'); ax[0, 0].legend(frameon=False, loc='upper left', fontsize=7)
    fig.tight_layout(); fig.savefig(outdir / 'hw_paired_tv.pdf'); fig.savefig(outdir / 'hw_paired_tv.png', dpi=200); plt.close(fig)
    # Fig 3: echo return and controls
    fig, ax = plt.subplots(1, len(stages), figsize=(6.4, 2.6), squeeze=False)
    for j, a in enumerate(stages):
        eps = sorted({v['probe'] for v in a['rows'] if v['basis'] == 'echo' and v['arm'] == 'gray'})
        x = np.arange(len(eps) + 1); w = .26
        for i, arm in enumerate(('gray', 'recursive', 'prep')):
            ys, es = [], []
            for pr in eps:
                v = next(v for v in a['rows'] if v['basis'] == 'echo' and v['probe'] == pr and v['arm'] == arm)
                ys.append(v['p0']); es.append(v['p0_ci'])
            if arm == 'prep': pass
            c = {'gray': col['gray'], 'recursive': col['recursive'], 'prep': '0.5'}[arm]
            ax[0, j].bar(x[:len(eps)] + (i - 1) * w, ys, w, color=c, hatch='//' if arm == 'gray' else ('..' if arm == 'prep' else ''), edgecolor='black' if arm != 'recursive' else 'white', linewidth=.5,
                         yerr=[[y - e[0] for y, e in zip(ys, es)], [e[1] - y for y, e in zip(ys, es)]], error_kw=dict(lw=.7, capsize=2),
                         label={'gray': 'Gray shell', 'recursive': 'Recursive', 'prep': 'Prep only'}[arm])
        sp = a['controls']['spam_mean_correct']
        ax[0, j].bar(x[-1], sp, .5, color='0.8', edgecolor='black', linewidth=.5, label='SPAM (mean correct)')
        ax[0, j].set_xticks(x); ax[0, j].set_xticklabels([f'echo {e}' for e in eps] + ['SPAM'], fontsize=7)
        ax[0, j].set_ylim(0, 1.3); ax[0, j].set_yticks([0, .25, .5, .75, 1.]); ax[0, j].set_title(f'$b={a["b"]}$')
    ax[0, 0].set_ylabel('Return / success probability'); ax[0, 0].legend(frameon=False, fontsize=6.5, loc='upper left', ncol=2)
    fig.tight_layout(); fig.savefig(outdir / 'hw_echo.pdf'); fig.savefig(outdir / 'hw_echo.png', dpi=200); plt.close(fig)
    # Fig 4: coherent readout parity
    fig, ax = plt.subplots(1, len(stages), figsize=(6.4, 2.6), squeeze=False)
    for j, a in enumerate(stages):
        rows = [v for v in a['rows'] if v['arm'] in ARMS and 'parity_obs' in v]
        pr = sorted({(v['probe'], v['basis']) for v in rows}); x = np.arange(len(pr))
        ax[0, j].scatter(x, [next(v for v in rows if (v['probe'], v['basis']) == k)['parity_ideal'] for k in pr], marker='_', s=300, color='k', lw=1, label='Ideal', zorder=3)
        for i, arm in enumerate(ARMS):
            vv = [next(v for v in rows if v['arm'] == arm and (v['probe'], v['basis']) == k) for k in pr]
            ys = [v['parity_obs'] for v in vv]
            ax[0, j].errorbar(x + (i - .5) * .18, ys, yerr=[[y - v['parity_ci'][0] for y, v in zip(ys, vv)], [v['parity_ci'][1] - y for y, v in zip(ys, vv)]],
                              fmt=mk[arm], color=col[arm], ms=4.5, capsize=2, lw=.8, label='Gray shell' if arm == 'gray' else 'Recursive')
        ax[0, j].set_xticks(x); ax[0, j].set_xticklabels([f'{k[0]}\n{k[1]}' for k in pr], fontsize=7); ax[0, j].set_title(f'$b={a["b"]}$')
        ax[0, j].axhline(0, color='0.7', lw=.5)
    ax[0, 0].set_ylabel(r'Parity $\langle \sigma_{\rm readout}^{\otimes b}\rangle$'); ax[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(outdir / 'hw_coherence.pdf'); fig.savefig(outdir / 'hw_coherence.png', dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', action='append', required=True, help='stage directory, repeatable')
    ap.add_argument('--figdir', type=Path, required=True)
    ap.add_argument('--tables', type=Path, required=True, help='directory for tables_<b>.tex')
    ap.add_argument('--save-index', type=Path, help='write combined json')
    a = ap.parse_args()
    out = []
    for s in a.stage:
        out.append(analyze_stage(s, Path(s).name))
    a.tables.mkdir(parents=True, exist_ok=True)
    for x in out: latex_tables(x, a.tables / f'hw_tables_b{x["b"]}.tex')
    figures(out, a.figdir)
    if a.save_index: write(a.save_index, {x['stage']: {k: x[k] for k in ('b', 'plan_sha256', 'job_id', 'resources', 'summary', 'paired_mean', 'controls')} for x in out})
    print(json.dumps({x['stage']: {'summary': x['summary'], 'paired_mean': x['paired_mean']} for x in out}, indent=2))


if __name__ == '__main__':
    main()
