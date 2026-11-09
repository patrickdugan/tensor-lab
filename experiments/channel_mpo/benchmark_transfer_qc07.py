"""QC-07 dephasing/amplitude-damping transfer with calibration-only budgets."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from channel_mpo import gate_layers,kraus
from purified_mpo import PurifiedChain
from benchmark_qc07 import run_case


def frozen_channel_schedule(n:int,p:float,channel:str,seed:int=0,
                            depth:int=6)->list[int]:
    chain=PurifiedChain(n,12,6)
    ops=kraus(channel,p)
    budgets=[]
    for layer in gate_layers(n,depth,seed):
        for i,u in layer:chain.gate(i,u)
        for i in range(n):chain.channel(i,ops)
        for i in range(n):chain.compress_env(i)
        budgets.append(chain.dimensions()['purified_entries'])
        chain.normalize()
    return budgets


def run_transfer(channels=('dephasing','amplitude_damping'),
                 ns=(6,8),ps=(.2,.5),seeds=(3,4,5),
                 budget_path:Path|None=None, validate=True):
    if budget_path is None:
        budget_path=Path(__file__).with_name('qc07_transfer_budget_schedule.json')
    if budget_path.exists():
        config=json.loads(budget_path.read_text())
    else:
        config={'protocol':'QC07-transfer: calibration seed 0 with uniform bond12/env6; verified on seeds 1,2',
                'depth':6,'schedules':{}}
        for name in channels:
            for n in ns:
                for p in ps:
                    key=f'{name}/{n}/{p:g}'
                    s=frozen_channel_schedule(n,p,name,seed=0)
                    for candidate in (1,2):
                        if frozen_channel_schedule(n,p,name,seed=candidate)!=s:
                            raise ValueError(f'calibration schedule seed-dependent for {key}')
                    config['schedules'][key]=s
        budget_path.write_text(json.dumps(config,indent=2)+'\n')
    rows=[]
    for name in channels:
        for n in ns:
            for p in ps:
                key=f'{name}/{n}/{p:g}'
                schedule=config['schedules'][key]
                for seed in seeds:
                    r=run_case(n,seed,p,channel=name,budget_by_layer=schedule,
                               validate=validate)
                    rows.append(r)
                    print(f'{key} seed={seed} D(QC06/QC07): '
                          f'{r["distance_qc06"]:.5f}/{r["distance_qc07"]:.5f}',flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds',default='3,4,5')
    ap.add_argument('--channels',default='dephasing,amplitude_damping')
    ap.add_argument('--qubits',default='6,8')
    ap.add_argument('--strengths',default='.2,.5')
    ap.add_argument('--no-validate',action='store_true')
    ap.add_argument('--budgets',type=Path,default=Path(__file__).with_name('qc07_transfer_budget_schedule.json'))
    ap.add_argument('--output',type=Path,default=Path('qc07_transfer.csv'))
    a=ap.parse_args()
    rows=run_transfer(tuple(a.channels.split(',')),tuple(int(x) for x in a.qubits.split(',')),
                      tuple(float(x) for x in a.strengths.split(',')),
                      tuple(int(x) for x in a.seeds.split(',')),a.budgets,not a.no_validate)
    if not rows:return
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('WROTE',len(rows),'records',a.output)

if __name__=='__main__':main()
