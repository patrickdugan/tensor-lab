"""QC-11 paired CPU benchmark: shrinking cone vs fixed-cone exact simulation.

Both procedures are exact-in-model local calculations; this script provides
independent numerical cross-checks. All seeds and channels are chosen in CLI
arguments, and no QC-09 approximate states influence either algorithm.
"""
from __future__ import annotations
import argparse,csv,platform
from pathlib import Path
from time import perf_counter
import numpy as np
from channel_mpo import gate_layers,kraus
from qc10_cone import exact_cone_density,trace_distance
from qc11_streaming_cone import streaming_region_bundle
from benchmark_qc10 import target_regions


def run_case(n:int,p:float,seed:int,*,channel:str='dephasing',depth:int=6,
             max_sites:int=8)->dict:
    if depth!=6 or n<8 or not 0<=p<=1:
        raise ValueError('QC-11 benchmark targets 6 layers with >=8 qubits')
    layers=gate_layers(n,depth,seed)
    operators=kraus(channel,p)
    regions=target_regions(n)
    start=perf_counter()
    streaming,source=streaming_region_bundle(n,layers,operators,regions,max_sites=max_sites)
    streaming_elapsed=perf_counter()-start
    start=perf_counter()
    fixed={label:exact_cone_density(n,layers,operators,sites,max_sites=max_sites)
           for label,sites in regions.items()}
    fixed_elapsed=perf_counter()-start
    errors={label:trace_distance(fixed[label].reduced_density,streaming[label]) for label in regions}
    max_error=max(errors.values())
    if max_error>5e-10:
        raise AssertionError(f'exact-cone disagreement {max_error}')
    return {'n':n,'depth':depth,'channel':channel,'p':p,'seed':seed,
          'split':'heldout' if seed>=3 else 'calibration',
          'streaming_seconds':streaming_elapsed,'fixed_cone_seconds':fixed_elapsed,
          'paired_speedup':fixed_elapsed/streaming_elapsed,
          'streaming_exact_solves':len(source),'fixed_exact_solves':len(fixed),
          'max_local_trace_distance':max_error,
          'left_1_delta':errors['left_1'],'left_2_delta':errors['left_2'],
          'right_1_delta':errors['right_1'],'right_2_delta':errors['right_2'],
          'max_matrix_complex_entries':max(r.peak_density_entries for r in source.values()),
          'max_cone_width':max(len(r.cone.sites) for r in source.values()),
          'streaming_gate_count':sum(r.relevant_gates for r in source.values()),
          'streaming_noise_site_updates':sum(r.kraus_site_updates for r in source.values()),
          'streaming_discarded_sites':sum(r.discarded_sites for r in source.values()),
          'numpy':np.__version__,'python':platform.python_version()}


def run_grid(ns=(48,),strengths=(.2,.5),seeds=(3,4,5,6,7),*,
             channels=('dephasing',)):
    rows=[]
    for n in ns:
      for channel in channels:
       for p in strengths:
        for seed in seeds:
         r=run_case(n,p,seed,channel=channel)
         rows.append(r)
         print(n,channel,p,seed,'D',f"{r['max_local_trace_distance']:.2e}",
              'speedup',round(r['paired_speedup'],2),flush=True)
    return rows


def save(rows:list[dict],path:Path):
    if not rows:raise ValueError('empty run')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qubits',default='48')
    ap.add_argument('--channels',default='dephasing')
    ap.add_argument('--strengths',default='.2,.5')
    ap.add_argument('--seeds',default='3,4,5,6,7')
    ap.add_argument('--output',type=Path,default=Path('qc11_streaming_heldout.csv'))
    args=ap.parse_args()
    rows=run_grid(tuple(map(int,args.qubits.split(','))),
                  tuple(map(float,args.strengths.split(','))),
                  tuple(map(int,args.seeds.split(','))),
                  channels=tuple(args.channels.split(',')))
    save(rows,args.output)
    print(f'wrote {len(rows)} rows to {args.output}',flush=True)

if __name__=='__main__':main()
