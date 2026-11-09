"""QC-11 isolated-process peak RSS and runtime for observable extraction.

Contrast a QC-09 approximate 48q purification, QC-10 fixed-cone exact
simulation, and QC-11 shrinking-cone exact simulation. Fresh processes keep
allocator histories separate. Linux ru_maxrss measures total RSS peak,
including Python, NumPy, BLAS and interpreter overhead.
"""
from __future__ import annotations
import argparse,csv,json,os,platform,resource,subprocess,sys
from pathlib import Path
from time import perf_counter
from channel_mpo import gate_layers,kraus
from benchmark_qc09 import frozen_budgets
from benchmark_qc10 import CAPS,target_regions
from qc09_certified import CertifiedPeakChain
from qc10_cone import exact_cone_density
from qc11_streaming_cone import streaming_region_bundle

MODES=('qc09_peak','qc10_fixed','qc11_stream')


def child(n,channel,p,seed,mode):
    if mode not in MODES:raise ValueError('unknown mode')
    layers=gate_layers(n,6,seed)
    ks=kraus(channel,p)
    regions=target_regions(n)
    start=perf_counter()
    if mode=='qc09_peak':
        sim=CertifiedPeakChain(n,max_bond=12,max_env=12,work_cap_entries=750_000,
             gate_method='sketch',oversample=4,power_iters=2,peak_gate_cap_entries=CAPS[n])
        for t,gates in enumerate(layers):
            sim.gate_layer(gates)
            for i in range(n):sim.channel(i,ks)
            sim.compress_layer_peak(frozen_budgets(n)[t],ks)
        result_count=4 # Four marginals are available from one evolved chain.
        output_entries=sim.dimensions()['purified_entries']
    elif mode=='qc10_fixed':
        results={key:exact_cone_density(n,layers,ks,target)
                 for key,target in regions.items()}
        result_count=len(results)
        output_entries=sum(r.reduced_density.size for r in results.values())
    else:
        results,solves=streaming_region_bundle(n,layers,ks,regions)
        result_count=len(solves)
        output_entries=sum(x.size for x in results.values())
    elapsed=perf_counter()-start
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'n':n,'channel':channel,'p':p,'seed':seed,'method':mode,
           'seconds':elapsed,'peak_rss_mib':rss/1024 if sys.platform.startswith('linux') else rss/(1024*1024),
           'result_count':result_count,'retained_output_complex_entries':output_entries,
           'numpy_python_environment':platform.python_version()}


def run_grid(ns=(48,),channels=('dephasing',),strengths=(.2,.5),seeds=(3,7),modes=MODES):
    rows=[];script=Path(__file__).resolve()
    for n in ns:
      for channel in channels:
       for p in strengths:
        for seed in seeds:
         for mode in modes:
          env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
          cmd=[sys.executable,str(script),'--child','--qubits',str(n),
                '--channels',channel,'--strengths',str(p),'--seeds',str(seed),'--modes',mode]
          r=subprocess.run(cmd,capture_output=True,text=True,check=True,env=env)
          item=json.loads(r.stdout.strip())
          rows.append(item)
          print(n,channel,p,seed,mode,round(item['seconds'],4),
                round(item['peak_rss_mib'],2),flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--child',action='store_true')
    ap.add_argument('--qubits',default='48')
    ap.add_argument('--channels',default='dephasing')
    ap.add_argument('--strengths',default='.2,.5')
    ap.add_argument('--seeds',default='3,7')
    ap.add_argument('--modes',default=','.join(MODES))
    ap.add_argument('--output',type=Path,default=Path('qc11_peak_rss.csv'))
    a=ap.parse_args()
    ns=tuple(map(int,a.qubits.split(',')))
    strengths=tuple(map(float,a.strengths.split(',')))
    seeds=tuple(map(int,a.seeds.split(',')))
    channels=tuple(a.channels.split(','))
    modes=tuple(a.modes.split(','))
    if a.child:
        if len(ns)!=len(strengths)!=len(seeds) or any(len(x)!=1 for x in (ns,strengths,seeds,channels,modes)):
            raise ValueError('child expects exactly one of every field')
        print(json.dumps(child(ns[0],channels[0],strengths[0],seeds[0],modes[0])))
        return
    rows=run_grid(ns,channels,strengths,seeds,modes)
    if not rows:raise ValueError('empty grid')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('wrote',len(rows),'rows',a.output)

if __name__=='__main__':main()
