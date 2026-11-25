"""QC-07 Linux subprocess peak-RSS comparison without dense-state allocation.

Run each candidate in its own Python process, with the QC-06 predeclared
boundary budgets and a QC-07 explicit-array workspace admission cap. RSS
includes interpreter, NumPy, and hidden BLAS/LAPACK workspaces, whereas the
admission cap cannot include these hidden allocations. One execution per
condition; no confidence intervals are implied.
"""
from __future__ import annotations
import argparse,csv,json,os,platform,subprocess,sys,resource
from pathlib import Path
from time import perf_counter
from channel_mpo import gate_layers,kraus
from purified_mpo import PurifiedChain
from qc06_adaptive import AdaptivePurifiedChain
from qc07_joint import JointPurifiedChain


def child_run(n:int,seed:int,p:float,policy:str,channel:str='depolarizing',
              work_cap:int=750000)->dict:
    budget=json.loads(Path(__file__).with_name('qc06_budget_schedule.json').read_text())['budgets_by_n'][str(n)]
    operators=kraus(channel,p)
    chain=({'uniform':lambda:PurifiedChain(n,12,6),
            'qc06':lambda:AdaptivePurifiedChain(n,12,12),
            'qc07':lambda:JointPurifiedChain(n,max_bond=12,max_env=12,work_cap_entries=work_cap)}[policy])()
    min_ret=1.; tic=perf_counter(); trajectory=[]
    for t,layer in enumerate(gate_layers(n,6,seed)):
        for i,u in layer:chain.gate(i,u)
        for i in range(n):chain.channel(i,operators)
        if policy=='uniform':
            for i in range(n):chain.compress_env(i)
        elif policy=='qc06':
            chain.compress_layer_budgeted(budget[t],max_site_env=12)
        else:
            chain.compress_layer_joint(budget[t],operators,bond_cap=12,env_cap=12)
        dims=chain.dimensions()
        trajectory.append(dims['purified_entries'])
        if dims['purified_entries']>budget[t]:
            raise AssertionError('layer budget exceeded')
        min_ret=min(min_ret,chain.normalize())
    secs=perf_counter()-tic
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.startswith('linux'):
        rss_mib=float(rss)/1024
    else:
        rss_mib=float(rss)/(1024*1024)
    return dict(n=n,seed=seed,p=p,channel=channel,policy=policy,
                seconds=secs,peak_rss_mib=rss_mib,
                min_layer_retention=min_ret,
                max_boundary_budget_entries=max(budget),
                max_retained_entries=max(trajectory),
                final_retained_entries=trajectory[-1],
                modeled_working_peak_entries=getattr(chain,'predicted_working_peak_entries',None),
                q07_work_cap_entries=work_cap,
                python=platform.python_version())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qubits',default='8,24,48')
    ap.add_argument('--strengths',default='.2,.5')
    ap.add_argument('--seed',type=int,default=3)
    ap.add_argument('--channel',default='depolarizing')
    ap.add_argument('--work-cap-entries',type=int,default=750000)
    ap.add_argument('--output',type=Path,default=Path('qc07_peak_rss.csv'))
    ap.add_argument('--child',action='store_true')
    ap.add_argument('--n',type=int,default=8)
    ap.add_argument('--p',type=float,default=.2)
    ap.add_argument('--policy',choices=('uniform','qc06','qc07'),default='qc07')
    args=ap.parse_args()
    if args.child:
        print(json.dumps(child_run(args.n,args.seed,args.p,args.policy,args.channel,args.work_cap_entries)))
        return
    rows=[]
    for n in [int(x) for x in args.qubits.split(',')]:
        for p in [float(x) for x in args.strengths.split(',')]:
            for policy in ('uniform','qc06','qc07'):
                argv=[sys.executable,str(Path(__file__).resolve()),'--child',
                      '--n',str(n),'--p',str(p),'--seed',str(args.seed),
                      '--policy',policy,'--channel',args.channel,
                      '--work-cap-entries',str(args.work_cap_entries)]
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                         MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                result=subprocess.run(argv,env=env,capture_output=True,text=True,check=True)
                row=json.loads(result.stdout.strip())
                rows.append(row)
                print(f'{n}q p={p} {policy}: {row["peak_rss_mib"]:.1f} MiB; '
                      f'{row["seconds"]:.3f}s; entries={row["final_retained_entries"]}',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('WROTE',len(rows),'records',args.output)

if __name__=='__main__':main()
