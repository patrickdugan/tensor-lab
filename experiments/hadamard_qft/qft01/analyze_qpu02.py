"""HAD-QFT-QPU02 summary tables and figures from collected stage directories.

Inputs per stage: compiled.json, counts.json (raw), analysis.json (from qpu02.py collect),
execution_receipt.json. Primary metric: edge phase contrast C = P(n|minus) - P(n|plus).
Intervals are the registered marginal Hoeffding intervals (finite shots only; no drift or
systematics; no multiplicity adjustment). Nothing here is error-mitigated.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

ARM_NAME = {'gray': 'Gray', 'recursive': 'Recursive', 'compact_exact': 'Exact compact', 'compact_approx': 'Approx. compact'}
COL = {'gray': '#D55E00', 'recursive': '#0072B2', 'compact_exact': '#009E73', 'compact_approx': '#CC79A7'}
MK = {'gray': 's', 'recursive': 'o', 'compact_exact': '^', 'compact_approx': 'D'}


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, sort_keys=True) + '\n', encoding='utf8')


def stage_summary(d, label):
    d = Path(d); m = read(d / 'compiled.json'); a = read(d / 'analysis.json'); rc = read(d / 'execution_receipt.json')
    arms = [x for x in ARM_NAME if any(r['arm'] == x for r in m['rows'])]
    res = {}
    for arm in arms:
        rows = [r for r in m['rows'] if r['arm'] == arm]
        ref = next(r for r in rows if r['id'] == f'{arm}_edge_plus')
        phase = [a['rows'][r['id']] for r in rows if r['state'] in ('plus', 'minus', 'imag')]
        res[arm] = dict(logical_cx=m['resources'][arm]['logical_cx'], line_cx=m['resources'][arm]['line_cx'],
                        own_best_line_cx=m['resources'][arm]['own_best_line_cx'],
                        operator_error=m['resources'][arm]['operator_error'], analytic_bound=m['resources'][arm]['analytic_bound'],
                        native_2q=ref['metrics']['num_two_qubit'], native_2q_depth=ref['metrics']['two_qubit_depth'],
                        depth=ref['metrics']['depth'], duration_us=ref['duration_s'] * 1e6,
                        mean_tv=float(np.mean([x['tv'] for x in phase])), mean_hf=float(np.mean([x['hf'] for x in phase])),
                        mean_tv_compiled=float(np.mean([x['tv_compiled'] for x in phase])))
    ctl = {k: v['target_prob'] for k, v in a['rows'].items() if k.startswith(('prep_', 'spam_'))}
    uni = {}
    for arm in arms:
        # TV between ideal and uniform for the phase rows: the fully scrambled reference.
        t = [0.5 * np.abs(np.array(r['ideal']) - 1 / len(r['ideal'])).sum() for r in m['rows'] if r['arm'] == arm and r['state'] in ('plus', 'minus', 'imag')]
        uni[arm] = float(np.mean(t))
    return dict(label=label, b=m['b'], replicate=m['replicate'], path=m['path'], plan_sha256=m['plan_sha256'],
                job_id=rc['job_id'], job_usage=rc['job_usage'], usage_before=rc['usage_before'], usage_after=rc['usage_after'],
                estimated_qpu_seconds=m['estimated_qpu_seconds'], optimization_level=m['optimization_level'],
                calibration=read(d / 'calibration.json')['timestamp'], resources=res, contrasts=a['contrasts'], comparisons=a['comparisons'],
                prep_spam_target_prob=ctl, mean_prep=float(np.mean([v for k, v in ctl.items() if k.startswith('prep_')])),
                mean_spam=float(np.mean([v for k, v in ctl.items() if k.startswith('spam_')])), min_spam=float(min(v for k, v in ctl.items() if k.startswith('spam_'))),
                uniform_tv=uni, shots=m['shots'])


def f(x, n=3): return f'{x:.{n}f}'
def iv(x, n=3): return f'[{x[0]:+.{n}f},{x[1]:+.{n}f}]'


def tex_tables(stages, path):
    L = []; seen = set()
    for s in stages:
        tag = f"b{s['b']}r{s['replicate']}"
        if s['b'] not in seen:
            seen.add(s['b'])
            L.append(r'\begin{table}[htbp]\centering\small')
            L.append(rf"\caption{{QPU02, $b={s['b']}$: compiled resources on \texttt{{ibm\_kingston}} (IBM native gates, optimization level 1; identical for every replicate at this $b$). Routed CNOTs are the registered compiler output on the common layout; native two-qubit gates, depth and duration are for the edge phase circuit including preparation. The operator error is the ideal spectral-norm distance from the exact shell transform.}}\label{{tab:q2res-b{s['b']}}}")
            L.append(r'\begin{tabular}{lrrrrrr}\toprule Arm & Routed CNOT & Native 2q & 2q depth & Depth & Duration ($\mu$s) & Operator error\\\midrule')
            for arm, r in s['resources'].items():
                err = f"{r['operator_error']:.1e}"
                L.append(f"{ARM_NAME[arm]} & {r['line_cx']} & {r['native_2q']} & {r['native_2q_depth']} & {r['depth']} & {f(r['duration_us'],1)} & {err}\\\\")
            L.append(r'\bottomrule\end{tabular}\end{table}'); L.append('')
        L.append(r'\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}')
        L.append(rf"\caption{{QPU02, $b={s['b']}$, replicate {s['replicate']}: phase contrast $C=P(n\mid\psi_-)-P(n\mid\psi_+)$, 512 shots per circuit, raw counts. Ideal $C$ is 1 (edge) or $1/\sqrt2$ (middle, tail). Intervals are marginal 95\% Hoeffding intervals from finite shots only. $\Delta$ is the contrast minus the recursive arm's, with its interval; TV and $F_H$ are means over the plus, minus and imaginary circuits of each arm (uniform-noise TV reference {f(np.mean(list(s['uniform_tv'].values())),3)}).}}\label{{tab:q2c-{tag}}}")
        L.append(r'\begin{tabular}{llrcrcrr}\toprule Arm & Probe & $C$ & 95\% interval & $C/C_{\rm ideal}$ & $\Delta$ vs recursive & mean TV & mean $F_H$\\\midrule')
        for arm in s['resources']:
            for probe in ('edge', 'middle', 'tail'):
                c = next(x for x in s['contrasts'] if x['arm'] == arm and x['probe'] == probe)
                cmp_ = next((x for x in s['comparisons'] if x['arm'] == arm and x['probe'] == probe), None)
                dd = f"{cmp_['difference']:+.3f} {iv(cmp_['shot_hoeffding95'])}" if cmp_ else '---'
                r = s['resources'][arm]
                L.append(f"{ARM_NAME[arm]} & {probe} & {f(c['contrast'])} & {iv(c['shot_hoeffding95'])} & {f(c['normalized'])} & {dd} & {f(r['mean_tv']) if probe=='edge' else ''} & {f(r['mean_hf']) if probe=='edge' else ''}\\\\")
            L.append(r'\midrule') if arm != list(s['resources'])[-1] else None
        L.append(rf"\multicolumn{{8}}{{l}}{{Preparation-only mean success {f(s['mean_prep'])}; SPAM mean {f(s['mean_spam'])}, min {f(s['min_spam'])}.}}\\")
        L.append(r'\bottomrule\end{tabular}\end{table}'); L.append('')
    Path(path).write_text('\n'.join(L) + '\n', encoding='utf8')


def figures(stages, scaling, outdir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'font.family': 'serif', 'axes.linewidth': .6})
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    # 1. CPU scaling of routed CNOTs, own best layout, b=3..6 (compiler output, not hardware).
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for arm in ARM_NAME:
        bs = [b for b in sorted(scaling) if arm in scaling[b]]
        ax.plot(bs, [scaling[b][arm] for b in bs], marker=MK[arm], color=COL[arm], lw=1, ms=4.5, label=ARM_NAME[arm])
    ax.set_yscale('log'); ax.set_xticks([3, 4, 5, 6]); ax.set_xlabel('Encoded qubits $b$'); ax.set_ylabel('Routed CNOTs (best layout)')
    ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(outdir / 'q2_scaling.pdf'); fig.savefig(outdir / 'q2_scaling.png', dpi=200); plt.close(fig)
    # 2. Edge contrast per arm and stage with intervals, ideal 1.
    fig, ax = plt.subplots(1, len(stages), figsize=(6.4, 2.7), sharey=True, squeeze=False)
    for j, s in enumerate(stages):
        arms = list(s['resources']); x = np.arange(3)
        for i, arm in enumerate(arms):
            cs = [next(c for c in s['contrasts'] if c['arm'] == arm and c['probe'] == p) for p in ('edge', 'middle', 'tail')]
            ys = [c['contrast'] / c['ideal'] for c in cs]
            lo = [(c['contrast'] - c['shot_hoeffding95'][0]) / c['ideal'] for c in cs]; hi = [(c['shot_hoeffding95'][1] - c['contrast']) / c['ideal'] for c in cs]
            ax[0, j].errorbar(x + (i - (len(arms) - 1) / 2) * .13, ys, yerr=[lo, hi], fmt=MK[arm], color=COL[arm], ms=4.2, capsize=1.8, lw=.8, label=ARM_NAME[arm])
        ax[0, j].axhline(1, color='k', lw=.6, ls='--'); ax[0, j].axhline(0, color='0.6', lw=.6)
        ax[0, j].set_xticks(x); ax[0, j].set_xticklabels(['edge', 'middle', 'tail']); ax[0, j].set_title(f"$b={s['b']}$, rep. {s['replicate']}" )
        ax[0, j].set_ylim(-0.2, 1.1)
    ax[0, 0].set_ylabel('Contrast / ideal contrast'); ax[0, 0].legend(frameon=False, fontsize=6.5, loc='upper right')
    fig.tight_layout(); fig.savefig(outdir / 'q2_contrast.pdf'); fig.savefig(outdir / 'q2_contrast.png', dpi=200); plt.close(fig)
    # 3. Edge contrast versus native two-qubit gates across stages.
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for s in stages:
        for arm, r in s['resources'].items():
            c = next(c for c in s['contrasts'] if c['arm'] == arm and c['probe'] == 'edge')
            ax.errorbar(r['native_2q'], c['contrast'], yerr=[[c['contrast'] - c['shot_hoeffding95'][0]], [c['shot_hoeffding95'][1] - c['contrast']]],
                        fmt=MK[arm], color=COL[arm], ms=4.2, capsize=1.6, lw=.7, mfc=COL[arm] if s['b'] == 5 else 'white')
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax.axhline(0, color='0.6', lw=.6); ax.set_xscale('log'); ax.set_xticks([200, 300, 500, 1000])
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlabel('Native two-qubit gates'); ax.set_ylabel('Edge phase contrast (ideal 1)')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=MK[a], color=COL[a], ls='', ms=4.5, label=ARM_NAME[a]) for a in ARM_NAME] + [Line2D([], [], marker='o', color='k', ls='', mfc='k', ms=4, label='$b=5$'), Line2D([], [], marker='o', color='k', ls='', mfc='white', ms=4, label='$b=6$')]
    ax.legend(handles=h, frameon=False, fontsize=6.5, ncol=2, loc='upper right'); ax.set_ylim(-0.15, 0.55)
    fig.tight_layout(); fig.savefig(outdir / 'q2_contrast_vs_gates.pdf'); fig.savefig(outdir / 'q2_contrast_vs_gates.png', dpi=200); plt.close(fig)


def cpu_scaling():
    import compact
    out = {}
    for b in (3, 4, 5, 6):
        out[b] = {arm: compact.synthesize(b, arm)['line_cx'] for arm in ARM_NAME}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', action='append', required=True)
    ap.add_argument('--figdir', type=Path, required=True); ap.add_argument('--tables', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    stages = [stage_summary(s, Path(s).name) for s in a.stage]
    scaling = cpu_scaling()
    a.tables.mkdir(parents=True, exist_ok=True); tex_tables(stages, a.tables / 'q2_tables.tex'); figures(stages, scaling, a.figdir)
    write(a.out, dict(stages=stages, cpu_scaling_own_best_routed_cx={str(k): v for k, v in scaling.items()}))
    for s in stages:
        e = {arm: next(c for c in s['contrasts'] if c['arm'] == arm and c['probe'] == 'edge')['contrast'] for arm in s['resources']}
        print(s['label'], {k: round(v, 3) for k, v in e.items()})


if __name__ == '__main__':
    main()
