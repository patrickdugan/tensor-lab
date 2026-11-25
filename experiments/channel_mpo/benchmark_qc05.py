"""Reproduce QC-05 calibration and held-out quantum-channel evaluations."""
from __future__ import annotations
import argparse
import csv
import platform
from pathlib import Path
import numpy as np
from qc05_portfolio import run_portfolio


def run_grid(ns=(6, 8), strengths=(.2, .5), seeds=range(8),
             depth=6, bond=12, env=4, teacher_bond=16, teacher_env=8,
             metric='hs', width=8):
    rows = []
    for n in ns:
        for p in strengths:
            for seed in seeds:
                x = run_portfolio(n, depth, seed, p, bond, env,
                                  teacher_bond, teacher_env, metric, width)
                x['split'] = 'calibration' if seed < 3 else 'heldout'
                x['python'] = platform.python_version()
                x['numpy'] = np.__version__
                rows.append(x)
                print(f'{n}q p={p} seed={seed} selected={x["selected"]} '
                      f'seq={x["distance_seq"]:.5f} '
                      f'pair={x["distance_pair"]:.5f} '
                      f'selected={x["distance_selected"]:.5f} '
                      f'env6={x["distance_mid_env6"]:.5f} '
                      f'teacher={x["distance_teacher"]:.5f}', flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qubits', default='6,8')
    parser.add_argument('--strengths', default='0.2,0.5')
    parser.add_argument('--seeds', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--bond', type=int, default=12)
    parser.add_argument('--env', type=int, default=4)
    parser.add_argument('--teacher-bond', type=int, default=16)
    parser.add_argument('--teacher-env', type=int, default=8)
    parser.add_argument('--metric', choices=('hs', 'sketch'), default='hs')
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('qc05_results.csv'))
    args = parser.parse_args()
    rows = run_grid(tuple(map(int, args.qubits.split(','))),
                    tuple(map(float, args.strengths.split(','))),
                    tuple(map(int, args.seeds.split(','))),
                    args.depth, args.bond, args.env,
                    args.teacher_bond, args.teacher_env,
                    args.metric, args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('wrote', len(rows), 'records to', args.output)
