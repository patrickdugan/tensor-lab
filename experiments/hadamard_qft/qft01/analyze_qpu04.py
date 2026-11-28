"""HAD-QFT-QPU04 tables and figures from hardware_results/qpu04 (audits, replay) and the stage data."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / 'hardware_results'
Q4 = R / 'qpu04'
PR = ('edge', 'middle', 'tail', 'quarter', 'threequarter')
PN = {'edge': 'edge', 'middle': 'middle', 'tail': 'tail', 'quarter': 'quarter (32)', 'threequarter': 'three-quarter (96)'}


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def f(x, n=3): return f'{x:.{n}f}'
def iv(x, n=3): return f'[{x[0]:+.{n}f},{x[1]:+.{n}f}]'


def tables(out):
    res = read(Q4/'audit_resources.json'); n0, n1 = read(Q4/'audit_native_r0.json'), read(Q4/'audit_native_r1.json'); rp = read(Q4/'replay.json')
    L = []
    L.append(r'\begin{table}[htbp]\centering\small')
    L.append(r"\caption{Ancilla-free logical and routed CNOT counts of the previous bounded-error compiler (direct boundary rotations) and the selective-rotation commutator compiler, both routed with the same permutation-state dynamic program on the induced prefix of the seven-node H-shaped region ($b=8$ appends a neighbor of local node 4). Routed counts are optimal only within the stated grammar and fixed chronological gate order; $b=7$ uses the frozen control order. CPU results, reproduced from the committed code.}\label{tab:q4-resources}")
    L.append(r'\begin{tabular}{rrrrr}\toprule $b$ & Logical (prev.) & Logical (new) & Routed (prev.) & Routed (new)\\\midrule')
    for r in res: L.append(f"{r['b']} & {r['previous_logical']} & {r['new_logical']} & {r['previous_routed']} & {r['new_routed']}\\\\")
    L.append(r'\bottomrule\end{tabular}\end{table}'); L.append('')
    L.append(r'\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}')
    L.append(r"\caption{Native compilation audit of the frozen seven-qubit circuits (basis CZ, SX, RZ, X; optimization level 0; no remapping). Counts are for the input-independent core including routing; duration is the scheduled length of the edge phase circuit (preparation, core, measurement) from the live backend target. The success estimates multiply calibrated CZ, single-qubit and readout fidelities (core and preparation, and with readout). Discrepancy is the largest entry difference between the native and routed logical unitaries; the last column is the spectral-norm distance of the native operator to the exact shell transform (analytic bound $1.6396\cdot10^{-3}$).}\label{tab:q4-native}")
    L.append(r'\begin{tabular}{llrrrrrrrrrr}\toprule Region & Arm & CZ & SX & RZ & X & 2q depth & Depth & Dur. ($\mu$s) & Est.\ success & Discrepancy & Error\\\midrule')
    for ri, a in enumerate((n0, n1)):
        for arm in ('A', 'D'):
            x = a['arms'][arm]
            L.append(f"{ri} & {arm} & {x['native_cz']} & {x['one_qubit']['sx']} & {x['one_qubit']['rz']} & {x['one_qubit']['x']} & {x['two_qubit_depth']} & {x['depth']} & {f(x['edge_circuit_duration_us'],1)} & {f(x['estimated_success_core_and_prep'],3)}/{f(x['estimated_success_total'],3)} & {x['native_vs_routed_max_entry_discrepancy']:.0e} & {x['native_operator_error_vs_exact']:.6f}\\\\")
    L.append(r'\bottomrule\end{tabular}\end{table}'); L.append('')
    L.append(r'\begin{table}[htbp]\centering\small')
    L.append(r"\caption{QPU04 primary endpoint: edge phase contrast $C=P(n\mid\psi_-)-P(n\mid\psi_+)$ (ideal 1), 1024 shots per circuit, raw counts, no postselection. The joint 95\% Bonferroni--Hoeffding interval (shot noise only, radius $\sqrt{2\ln(4/0.05)/1024}=0.0925$ for each of the two registered differences, no independence assumption across regions) applies to $D-A$. A replicated positive result needs both lower endpoints above zero.}\label{tab:q4-primary}")
    L.append(r'\begin{tabular}{lrrrcl}\toprule Region (job) & $C_A$ & $C_D$ & $D-A$ & Joint interval & Lower $>0$\\\midrule')
    for ri, s in enumerate(rp['regions']):
        p = s['primary']
        L.append(f"{ri} (\\texttt{{{s['job_id']}}}) & {f(s['arms']['A']['contrast']['edge'])} & {f(s['arms']['D']['contrast']['edge'])} & {p['difference']:+.4f} & {iv(p['joint_interval'], 4)} & {'yes' if p['lower_positive'] else 'no'}\\\\")
    L.append(r'\midrule')
    L.append(rf"\multicolumn{{6}}{{l}}{{Replicated positive primary result (both lower endpoints above zero): \textbf{{{'yes' if rp['registered_primary_both_positive'] else 'no'}}}}}\\")
    L.append(r'\bottomrule\end{tabular}\end{table}'); L.append('')
    L.append(r'\begin{table}[htbp]\centering\small')
    L.append(r"\caption{Secondary and exploratory probes: $D-A$ contrast difference with the unadjusted 95\% shot-only interval (radius $0.0849$), descriptive only and not part of the registered primary family.}\label{tab:q4-secondary}")
    L.append(r'\begin{tabular}{llrrrrr}\toprule Region & Probe & $C_A$ & $C_D$ & $D-A$ & Interval & \\\midrule')
    for ri, s in enumerate(rp['regions']):
        for pr in PR:
            d = s['D_minus_A'][pr]; ci = s['secondary_unadjusted_intervals'][pr]
            L.append(f"{ri} & {PN[pr]} & {f(s['arms']['A']['contrast'][pr])} & {f(s['arms']['D']['contrast'][pr])} & {d:+.3f} & {iv(ci)} & \\\\")
        L.append(r'\midrule' if ri == 0 else '')
    L.append(r'\bottomrule\end{tabular}\end{table}'); L.append('')
    L.append(r'\begin{table}[htbp]\centering\small')
    L.append(r"\caption{QPU04 distribution-level diagnostics and controls. Mean total variation distance and Hellinger fidelity are over the plus, minus and imaginary circuits of the arm; control success is the mean success probability of the two preparation-only and two SPAM circuits; the last column is the total variation distance between the edge imaginary-phase circuit and the average of the two endpoint circuits.}\label{tab:q4-controls}")
    L.append(r'\begin{tabular}{llrrrr}\toprule Region & Arm & Mean TV & Mean $F_H$ & Control success & Imag.\ vs mixture TV\\\midrule')
    for ri, s in enumerate(rp['regions']):
        for arm in ('A', 'D'):
            a = s['arms'][arm]
            L.append(f"{ri} & {arm} & {f(a['mean_tv'])} & {f(a['mean_hellinger'])} & {f(a['mean_control_success'])} & {f(a['edge_imag_vs_mixture_tv'])}\\\\")
    L.append(r'\bottomrule\end{tabular}\end{table}')
    Path(out).write_text('\n'.join(L)+'\n', encoding='utf8')


def figures(figdir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'font.family': 'serif', 'axes.linewidth': .6})
    rp = read(Q4/'replay.json'); figdir = Path(figdir)
    rad = rp['registered_radius']
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw=dict(width_ratios=[1, 1.5]))
    for i, s in enumerate(rp['regions']):
        d = s['D_minus_A']['edge']
        ax[0].errorbar(i, d, yerr=rad, fmt='o', color='#0072B2', capsize=3, ms=5)
    ax[0].axhline(0, color='k', lw=.6); ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(['region 0', 'region 1']); ax[0].set_xlim(-.5, 1.5)
    ax[0].set_ylabel(r'$D-A$ edge contrast'); ax[0].set_title('Registered primary (joint 95%)', fontsize=9)
    rad1 = np.sqrt(2*np.log(2/.05)/1024)
    for i, s in enumerate(rp['regions']):
        for j, pr in enumerate(('edge', 'middle', 'tail', 'quarter', 'threequarter')):
            ax[1].errorbar(j + (i-.5)*.2, s['D_minus_A'][pr], yerr=rad1, fmt='os'[i], color=['#0072B2', '#D55E00'][i], capsize=2, ms=4, lw=.8, label=f'region {i}' if j == 0 else None)
    ax[1].axhline(0, color='k', lw=.6); ax[1].set_xticks(range(5)); ax[1].set_xticklabels(['edge', 'middle', 'tail', '32', '96'])
    ax[1].set_title('All probes (unadjusted 95%, descriptive)', fontsize=9); ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(figdir/'q4_primary.pdf'); fig.savefig(figdir/'q4_primary.png', dpi=200); plt.close(fig)
    res = read(Q4/'audit_resources.json')
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    bs = [r['b'] for r in res]
    for a, k, t in ((ax[0], 'logical', 'Logical CNOTs'), (ax[1], 'routed', 'Routed CNOTs (H-region)')):
        a.plot(bs, [r[f'previous_{k}'] for r in res], 's--', color='#D55E00', ms=4, lw=1, label='previous (direct)')
        a.plot(bs, [r[f'new_{k}'] for r in res], 'o-', color='#0072B2', ms=4, lw=1, label='commutator')
        a.set_xlabel('Encoded qubits $b$'); a.set_title(t, fontsize=9); a.set_xticks(bs)
    ax[1].axhline(200, color='0.6', lw=.6, ls=':'); ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(figdir/'q4_scaling.pdf'); fig.savefig(figdir/'q4_scaling.png', dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    col = {'QPU02': '0.65', 'QPU03': '#009E73', 'QPU04': '#D55E00'}
    for c in ('QPU02', 'QPU03', 'QPU04'):
        pts = [p for p in rp['points'] if p['campaign'] == c]
        ax.scatter([p['native_cz'] for p in pts], [p['edge'] for p in pts], s=18, color=col[c], label=c, zorder=3 if c == 'QPU04' else 2)
    ax.axhline(0, color='0.6', lw=.6); ax.set_xscale('log'); ax.set_xlabel('Native CZ gates'); ax.set_ylabel('Edge phase contrast (ideal 1)')
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax.set_xticks([100, 200, 300, 500, 1000]); ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(figdir/'q4_contrast_vs_gates.pdf'); fig.savefig(figdir/'q4_contrast_vs_gates.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    t = HERE/'manuscript'/'tables'; fg = HERE/'manuscript'/'figures'; t.mkdir(exist_ok=True); fg.mkdir(exist_ok=True)
    tables(t/'q4_tables.tex'); figures(fg); print('q4 tables and figures written')
