"""QC-06 sensitivity study: env rank cap 6 (equivalent control), 8, 12, 16.

Heldout cap variants are exploratory ablations, not model selection; the
preregistered QC-06 primary comparison uses cap 12.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from benchmark_qc06 import run_case


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds',default='0,1,2')
    ap.add_argument('--caps',default='6,8,12,16')
    ap.add_argument('--qubits',default='6,8')
    ap.add_argument('--strengths',default='0.2,0.5')
    ap.add_argument('--no-validate',action='store_true')
    ap.add_argument('--depth',type=int,default=6)
    ap.add_argument('--schedule',type=Path,default=Path('qc06_budget_schedule.json'))
    ap.add_argument('--output',type=Path,default=Path('qc06_ablation.csv'))
    a=ap.parse_args()
    budgets={int(k):v for k,v in json.loads(a.schedule.read_text())['budgets_by_n'].items()}
    rows=[]
    for n in map(int,a.qubits.split(',')):
        for p in map(float,a.strengths.split(',')):
            for seed in map(int,a.seeds.split(',')):
                for cap in map(int,a.caps.split(',')):
                    row=run_case(n,a.depth,seed,p,12,6,cap,not a.no_validate,
                                 frozen_budgets=budgets[n])
                    rows.append(row)
                    print(f'n={n} p={p} seed={seed} cap={cap} '
                          f'dist={row["distance_adaptive"]:.5f} '
                          f'delta={row["distance_delta"]:+.5f} '
                          f'time={row["seconds_adaptive"]:.3f}s',flush=True)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('saved',a.output)


if __name__=='__main__':main()
