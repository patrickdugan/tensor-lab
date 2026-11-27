"""HAD-QFT-QPU03 tables and figures: offline design/scaling results and hardware stages.

design mode: select_b5.json, select_b6.json, scaling.json  -> tables/q3_design_tables.tex, figures/q3_scaling.pdf
stage mode : stage directories (compiled.json, analysis.json, execution_receipt.json) -> tables/q3_stage_tables.tex,
             figures/q3_contrast.pdf, q3_contrast_vs_gates.pdf, hardware_results/qpu03_summary.json
Raw counts only. Intervals are finite-shot Hoeffding bounds with Bonferroni over the registered family sizes.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
COL = {'A': '#D55E00', 'B': '#009E73', 'C': '#CC79A7', 'D': '#0072B2'}
MK = {'A': 's', 'B': '^', 'C': 'D', 'D': 'o'}
NAME = {'A': 'A: QPU02 approx. compact', 'B': 'B: workspace exact', 'C': 'C: workspace bounded-error', 'D': 'D: ancilla-free bounded-error'}


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def f(x, n=3): return f'{x:.{n}f}'
def iv(x, n=3): return f'[{x[0]:+.{n}f},{x[1]:+.{n}f}]'


# ----------------------------------------------------------------------------- offline design
def design_tables(sel5, sel6, scaling, out):
    L = []
    for b, sel in ((5, sel5), (6, sel6)):
        L.append(r'\begin{table}[htbp]\centering\small')
        L.append(rf"\caption{{Workspace trade-off at $b={b}$ (offline, calibration snapshot {sel['calibration_timestamp'][:16]}): for each exact or bounded-error variant with $m$ clean ancillas, the logical CNOT count, the best native CZ count after heavy-hex SABRE routing (best of {sel['seeds']} seeds, levels 2--3, all induced shapes of $b+m$ nodes, three control orders), its CZ depth and the best calibrated estimated success probability (ESP) over all embeddings on \texttt{{ibm\_kingston}}.}}\label{{tab:q3-ws-b{b}}}")
        L.append(r'\begin{tabular}{llrrrrr}\toprule Level & $m$ & Logical CX & Native CZ & CZ depth & Depth & ESP\\\midrule')
        for mode in ('approx', 'exact'):
            for m in (0, 1, 2, 3, 4):
                c = [r for r in sel['candidates'] if r['mode'] == mode and r['m'] == m]
                if not c: continue
                bz = min(c, key=lambda r: (r['metrics']['cz'], r['metrics']['cz_depth']))
                be = max(c, key=lambda r: r['esp'])
                L.append(f"{'bounded-error' if mode == 'approx' else 'exact'} & {m} & {bz['logical_cx']} & {bz['metrics']['cz']} & {bz['metrics']['cz_depth']} & {bz['metrics']['depth']} & {f(be['esp'])}\\\\")
            L.append(r'\midrule') if mode == 'approx' else None
        L.append(r'\bottomrule\end{tabular}\end{table}'); L.append('')
    rows = scaling['rows']
    L.append(r'\begin{table}[htbp]\centering\small')
    L.append(r"\caption{Offline resource scaling of the workspace compiler through $b=8$ encoded qubits (ambient dimension $2^{2^b-1}$). Logical CNOTs and best native CZ after heavy-hex routing (12 seeds, levels 2--3, six induced shapes, two control orders) for the bounded-error compiler with $m$ clean ancillas; the exact compiler is routed through $b=6$ only, and its $m{=}b{-}2$ column (prefix-predicate chain active) is a logical count. Every circuit through $b=7$ was verified by full simulation (workspace returned to $\ket0$). The ideal operator error of the bounded-error circuits is below $1.7\cdot10^{-3}$ throughout.}\label{tab:q3-scaling}")
    L.append(r'\begin{tabular}{r rrrrr rrr}\toprule & \multicolumn{5}{c}{Bounded-error, logical (native CZ)} & \multicolumn{3}{c}{Exact, logical (native CZ)}\\ $b$ & $m{=}0$ & $m{=}1$ & $m{=}2$ & $m{=}3$ & $m{=}4$ & $m{=}0$ & $m{=}2$ & $m{=}b{-}2$\\\midrule')
    import workspace as W
    def cell(b, mode, m):
        if mode == 'exact' and m == 'full':          # exact, m = b-2: prefix-predicate chain active; logical count only
            return f"{W.count_cx(W.core(b, 'exact', max(b-2, 0))[0])} (--)"
        r = next((x for x in rows if x['b'] == b and x['mode'] == mode and x['m'] == m), None)
        if r is None: return '---'
        rz = r['routed']['cz'] if r.get('routed') else '--'
        return f"{r['logical_cx']} ({rz})"
    for b in range(3, 9):
        L.append(f"{b} & " + ' & '.join(cell(b, 'approx', m) for m in range(5)) + ' & ' + ' & '.join(cell(b, 'exact', m) for m in (0, 2, 'full')) + r'\\')
    L.append(r'\bottomrule\end{tabular}\end{table}')
    Path(out).write_text('\n'.join(L) + '\n', encoding='utf8')


def scaling_figure(scaling, outdir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'font.family': 'serif', 'axes.linewidth': .6})
    rows = scaling['rows']
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8))
    def series(mode, m, key):
        bs = sorted({r['b'] for r in rows if r['mode'] == mode and r['m'] == m})
        vals = []
        for b in bs:
            r = next(x for x in rows if x['b'] == b and x['mode'] == mode and x['m'] == m)
            v = r['logical_cx'] if key == 'logical_cx' else (r['routed']['cz'] if r.get('routed') else None)
            vals.append(v)
        return [(b, v) for b, v in zip(bs, vals) if v is not None]
    # old line routing of the approximate compact circuit (QPU02 CPU counts, own best layout)
    old_line = {3: 28, 4: 104, 5: 218, 6: 418}
    for a, key, title in ((ax[0], 'logical_cx', 'Logical CNOTs'), (ax[1], 'routed', 'Native CZ after heavy-hex routing')):
        for (mode, m, c, lab, ls) in (('approx', 0, '#0072B2', 'bounded-error, $m{=}0$', '-'), ('approx', 2, '#CC79A7', 'bounded-error, $m{=}2$', '-'), ('approx', 4, '#D55E00', 'bounded-error, $m{=}4$', '-'),
                                      ('exact', 0, '#009E73', 'exact, $m{=}0$', '--')):
            s = series(mode, m, key)
            if s: ax[0 if key == 'logical_cx' else 1].plot([x for x, _ in s], [y for _, y in s], ls, color=c, marker='o', ms=3.5, lw=1, label=lab)
        if key == 'routed':
            a.plot(list(old_line), list(old_line.values()), ':', color='0.4', marker='s', ms=3.5, lw=1, label='QPU02 line routing')
            a.axhline(220, color='0.6', lw=.6, ls='--'); a.text(7.9, 240, 'target 220', fontsize=7, color='0.4', ha='right')
        a.set_yscale('log'); a.set_xlabel('Encoded qubits $b$'); a.set_title(title, fontsize=9)
    ax[0].set_ylabel('Gates (log scale)'); ax[1].legend(frameon=False, fontsize=6.3, loc='lower right')
    fig.tight_layout(); outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / 'q3_scaling.pdf'); fig.savefig(outdir / 'q3_scaling.png', dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------- hardware stages
def stage_summary(d):
    d = Path(d); m = read(d / 'compiled.json'); a = read(d / 'analysis.json'); rc = read(d / 'execution_receipt.json')
    arms = {}
    for arm in [x['arm'] for x in m['design']['arms']]:
        rows = [r for r in m['rows'] if r['arm'] == arm and r['kind'] == 'core']
        ref = next(r for r in rows if r['probe'] == 'edge' and r['state'] == 'plus')
        info = m['arms'][arm]; cfg = next(x for x in m['design']['arms'] if x['arm'] == arm)
        ph = [a['rows'][r['id']] for r in rows if r['state'] in ('plus', 'minus', 'imag')]
        ctl = [a['rows'][r['id']] for r in m['rows'] if r['arm'] == arm and r['kind'] in ('prep', 'spam')]
        arms[arm] = dict(m=cfg.get('m', 0), mode=cfg.get('mode', 'old'), logical_cx=info.get('logical_cx'), native_2q=ref['metrics']['num_two_qubit'], two_q_depth=ref['metrics']['two_qubit_depth'],
                         depth=ref['metrics']['depth'], duration_us=ref['duration_s']*1e6, esp=cfg['esp'], nodes=cfg.get('phys', cfg.get('nodes')),
                         mean_tv=float(np.mean([x['tv'] for x in ph])), mean_hf=float(np.mean([x['hf'] for x in ph])),
                         leakage_core=float(np.mean([x['leakage'] for x in ph])), control_success=float(np.mean([x['target_prob'] for x in ctl])),
                         control_leakage=float(np.mean([x['leakage'] for x in ctl])))
    return dict(label=d.name, b=m['b'], replicate=m['replicate'], plan_sha256=m['plan_sha256'], job_id=rc['job_id'], job_usage=rc['job_usage'], usage_before=rc['usage_before'],
                usage_after=rc['usage_after'], estimated=m['estimated_qpu_seconds'], shots=m['shots'], calibration=read(d / 'calibration.json')['timestamp'], arms=arms,
                contrasts=a['contrasts'], signal=a['signal'], differences=a['differences'], families=m['design']['families'])


def stage_tables(stages, out):
    L = []
    for s in stages:
        tag = f"b{s['b']}r{s['replicate']}"
        L.append(r'\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}')
        L.append(rf"\caption{{QPU03, $b={s['b']}$, replicate {s['replicate']}: compiled resources and phase contrast, {s['shots']} shots per circuit, raw counts. Native counts, depth and duration are for the edge phase circuit. $C$ is the edge contrast (ideal 1) with its simultaneous lower bound (Hoeffding, Bonferroni over {len(s['families']['signal'])} arms, 95\%). Leakage is the mean probability of a workspace readout of 1 in the plus and minus circuits (ideal 0); the control leakage is the same quantity with no transform.}}\label{{tab:q3-{tag}}}")
        L.append(r'\begin{tabular}{llrrrrrrrrr}\toprule Arm & $m$ & Native 2q & 2q depth & Dur. ($\mu$s) & ESP & $C$ & lower bound & mean TV & Leakage & Ctl.\ leak.\\\midrule')
        for arm, r in s['arms'].items():
            sg = next(x for x in s['signal'] if x['arm'] == arm and x['probe'] == 'edge')
            L.append(f"{arm} & {r['m']} & {r['native_2q']} & {r['two_q_depth']} & {f(r['duration_us'],1)} & {f(r['esp'],2)} & {f(sg['contrast'])} & {sg['lower_simultaneous']:+.3f} & {f(r['mean_tv'])} & {f(r['leakage_core'])} & {f(r['control_leakage'])}\\\\")
        L.append(r'\bottomrule\end{tabular}\end{table}'); L.append('')
        L.append(r'\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}')
        L.append(rf"\caption{{QPU03, $b={s['b']}$, replicate {s['replicate']}: registered edge-contrast differences. Simultaneous intervals use Hoeffding with Bonferroni over the {len(s['families']['difference'])} registered comparisons; unadjusted 95\% intervals are shown for reference. Middle and tail probes are secondary.}}\label{{tab:q3d-{tag}}}")
        L.append(r'\begin{tabular}{llrcc}\toprule Comparison & Probe & Difference & Simultaneous interval & Unadjusted interval\\\midrule')
        for x in s['differences']:
            L.append(f"{x['first']} $-$ {x['second']} & {x['probe']} & {x['difference']:+.3f} & {iv(x['interval_simultaneous'])} & {iv(x['interval_unadjusted'])}\\\\")
        L.append(r'\bottomrule\end{tabular}\end{table}'); L.append('')
    Path(out).write_text('\n'.join(L) + '\n', encoding='utf8')


def stage_figures(stages, outdir, qpu02_points):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'font.family': 'serif', 'axes.linewidth': .6})
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, len(stages), figsize=(2.9*len(stages)+.6, 2.7), sharey=True, squeeze=False)
    for j, s in enumerate(stages):
        arms = list(s['arms']); x = np.arange(3)
        for i, arm in enumerate(arms):
            cs = [next(c for c in s['contrasts'] if c['arm'] == arm and c['probe'] == p) for p in ('edge', 'middle', 'tail')]
            sg = {p: next(z for z in s['signal'] if z['arm'] == arm and z['probe'] == p) for p in ('edge', 'middle', 'tail')}
            ys = [c['contrast']/c['ideal'] for c in cs]; rad = [sg[p]['radius']/c['ideal'] for p, c in zip(('edge', 'middle', 'tail'), cs)]
            ax[0, j].errorbar(x + (i-(len(arms)-1)/2)*.13, ys, yerr=rad, fmt=MK[arm], color=COL[arm], ms=4.2, capsize=1.8, lw=.8, label=NAME[arm])
        ax[0, j].axhline(1, color='k', lw=.6, ls='--'); ax[0, j].axhline(0, color='0.6', lw=.6)
        ax[0, j].set_xticks(x); ax[0, j].set_xticklabels(['edge', 'middle', 'tail']); ax[0, j].set_title(f"$b={s['b']}$, rep. {s['replicate']}"); ax[0, j].set_ylim(-0.3, 1.15)
    ax[0, 0].set_ylabel('Contrast / ideal contrast'); ax[0, 0].legend(frameon=False, fontsize=6, loc='upper right')
    fig.tight_layout(); fig.savefig(outdir / 'q3_contrast.pdf'); fig.savefig(outdir / 'q3_contrast.png', dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    for s in stages:
        for arm, r in s['arms'].items():
            sg = next(z for z in s['signal'] if z['arm'] == arm and z['probe'] == 'edge')
            ax.errorbar(r['native_2q'], sg['contrast'], yerr=sg['radius'], fmt=MK[arm], color=COL[arm], ms=4.2, capsize=1.6, lw=.7, mfc=COL[arm] if s['b'] == 5 else 'white')
    for (g, c, lab) in qpu02_points:
        ax.plot(g, c, 'x', color='0.5', ms=4)
    ax.axhline(0, color='0.6', lw=.6); ax.set_xscale('log'); ax.set_xlabel('Native two-qubit gates'); ax.set_ylabel('Edge phase contrast (ideal 1)')
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax.set_xticks([100, 200, 300, 500]); ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=MK[a], color=COL[a], ls='', ms=4.5, label=NAME[a]) for a in 'ABCD'] + [Line2D([], [], marker='x', color='0.5', ls='', ms=4, label='QPU02 (all arms)')]
    ax.legend(handles=h, frameon=False, fontsize=6, loc='upper right')
    fig.tight_layout(); fig.savefig(outdir / 'q3_contrast_vs_gates.pdf'); fig.savefig(outdir / 'q3_contrast_vs_gates.png', dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['design', 'stages'])
    ap.add_argument('--stage', action='append', default=[])
    ap.add_argument('--figdir', type=Path, default=HERE/'manuscript'/'figures'); ap.add_argument('--tables', type=Path, default=HERE/'manuscript'/'tables')
    ap.add_argument('--out', type=Path, default=HERE/'hardware_results'/'qpu03_summary.json')
    a = ap.parse_args(); a.tables.mkdir(parents=True, exist_ok=True)
    root = HERE/'hardware_results'/'qpu03_design'
    if a.mode == 'design':
        design_tables(read(root/'select_b5.json'), read(root/'select_b6.json'), read(root/'scaling.json'), a.tables/'q3_design_tables.tex')
        scaling_figure(read(root/'scaling.json'), a.figdir); print('design tables and scaling figure written')
    else:
        stages = [stage_summary(s) for s in a.stage]
        q2 = read(HERE/'hardware_results'/'qpu02_summary.json')
        pts = []
        for st in q2['stages']:
            for arm, r in st['resources'].items():
                c = next(x for x in st['contrasts'] if x['arm'] == arm and x['probe'] == 'edge'); pts.append((r['native_2q'], c['contrast'], f"{st['label']}:{arm}"))
        stage_tables(stages, a.tables/'q3_stage_tables.tex'); stage_figures(stages, a.figdir, pts)
        a.out.write_text(json.dumps(dict(stages=stages), indent=2, sort_keys=True)+'\n', encoding='utf8')
        for s in stages:
            print(s['label'], {arm: round(next(x for x in s['signal'] if x['arm'] == arm and x['probe'] == 'edge')['contrast'], 3) for arm in s['arms']})


if __name__ == '__main__':
    main()
