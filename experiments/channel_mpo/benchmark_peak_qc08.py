"""QC-08: per-process peak-RSS comparison with frozen retained-entry budgets.

Measured RSS includes NumPy and LAPACK allocations; explicit work admission is
only an accounting estimate. Each candidate gets a fresh Python interpreter.
All paths execute six layers and report minimum pre-normalization retention.
"""
from __future__ import annotations
import argparse,csv,json,os,resource,subprocess,sys,platform
from pathlib import Path
from time import perf_counter
from channel_mpo import gate_layers,kraus
from qc06_adaptive import AdaptivePurifiedChain
from qc07_joint import JointPurifiedChain
from qc08_streaming import StreamingJointChain


def child_run(n:int,seed:int,p:float,policy:str,channel:str='depolarizing',
              work_cap_entries:int=750000)->dict:
    budget=json.loads(Path(__file__).with_name('qc06_budget_schedule.json').read_text())['budgets_by_n'][str(n)]
    ks=kraus(channel,p)
    if policy=='qc06':
        chain=AdaptivePurifiedChain(n,max_bond=12,max_env=12)
    elif policy=='qc07':
        chain=JointPurifiedChain(n,max_bond=12,max_env=12,work_cap_entries=work_cap_entries)
    elif policy in ('classic','classic_mem','fused'):
        chain=StreamingJointChain(n,max_bond=12,max_env=12,work_cap_entries=work_cap_entries)
    else: raise ValueError('invalid policy')
    start=perf_counter();retain=[];entries=[]
    for t,layer in enumerate(gate_layers(n,6,seed)):
        if policy in ('qc06','qc07','classic_mem'):
            for i,u in layer:chain.gate(i,u)
        else:chain.gate_layer(layer)
        for i in range(n):chain.channel(i,ks)
        if policy=='qc06':chain.compress_layer_budgeted(budget[t],max_site_env=12)
        elif policy=='qc07':chain.compress_layer_joint(budget[t],ks,bond_cap=12,env_cap=12)
        else:chain.compress_layer_stream(budget[t],ks,bond_cap=12,env_cap=12,order='classic' if policy=='classic_mem' else policy)
        current=chain.dimensions()['purified_entries']
        if current>budget[t]:raise AssertionError('entry budget overflow')
        entries.append(current)
        retain.append(chain.normalize())
    seconds=perf_counter()-start
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib=peak/1024 if sys.platform.startswith('linux') else peak/(1024**2)
    return {'n':n,'seed':seed,'p':p,'channel':channel,'policy':policy,
            'seconds':seconds,'peak_rss_mib':peak_mib,
            'max_boundary_entries':max(entries),
            'final_boundary_entries':entries[-1],
            'max_budget_entries':max(budget),
            'min_layer_retention':min(retain),
            'modeled_peak_entries':getattr(chain,'predicted_working_peak_entries',None),
            'max_work_cap_entries':work_cap_entries,
            'numpy':__import__('numpy').__version__,
            'python':platform.python_version()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--child',action='store_true')
    p.add_argument('--qubits',default='8,24,48')
    p.add_argument('--strengths',default='.2,.5')
    p.add_argument('--seed',type=int,default=3)
    p.add_argument('--channel',default='depolarizing')
    p.add_argument('--policy',default='classic',choices=('qc06','qc07','classic','classic_mem','fused'))
    p.add_argument('--n',type=int,default=8)
    p.add_argument('--p',type=float,default=.2)
    p.add_argument('--work-cap-entries',type=int,default=750000)
    p.add_argument('--output',type=Path,default=Path('qc08_peak_rss.csv'))
    a=p.parse_args()
    if a.child:
        print(json.dumps(child_run(a.n,a.seed,a.p,a.policy,a.channel,a.work_cap_entries)))
        return
    rows=[]
    for n in map(int,a.qubits.split(',')):
        for strength in map(float,a.strengths.split(',')):
            for policy in ('qc06','qc07','classic','classic_mem','fused'):
                argv=[sys.executable,str(Path(__file__).resolve()),'--child','--n',str(n),
                      '--p',str(strength),'--seed',str(a.seed),'--policy',policy,
                      '--channel',a.channel,'--work-cap-entries',str(a.work_cap_entries)]
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                         MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                outcome=subprocess.run(argv,env=env,capture_output=True,text=True,check=True,timeout=15)
                row=json.loads(outcome.stdout.strip());rows.append(row)
                # Persist each isolated observation so interrupted large-N runs
                # leave useful partial evidence with explicit row counts.
                a.output.parent.mkdir(parents=True,exist_ok=True)
                with a.output.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                print(f"{n}q p={strength} {policy}: {row['peak_rss_mib']:.1f} MiB, "
                      f"{row['seconds']:.3f}s, retention={row['min_layer_retention']:.3g}",flush=True)
    print('saved',len(rows),'rows to',a.output)

if __name__=='__main__': main()
