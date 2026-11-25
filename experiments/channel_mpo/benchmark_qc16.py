"""QC-16 independent proof verification benchmark.

Examples:
  OPENBLAS_NUM_THREADS=1 python benchmark_qc16.py --out-dir /tmp/qc16-bench
  python benchmark_qc16.py --worker fast /tmp/qc16-bench/fixture.json.gz

Each worker runs in a fresh process. Linux ru_maxrss is measured in KiB.
The reference and optimized backends verify the same full witness; the
segmented mode independently validates continuity commitments and math.
"""
from __future__ import annotations
import argparse
import csv
import json
try:
    import resource
except ImportError:  # Available on Linux and macOS; benchmark also runs on Windows.
    resource=None
import subprocess
import sys
from pathlib import Path
from time import perf_counter


def worker(mode,path):
    from qc15_checker import check_certificate
    from qc16_checker import verify
    t0=perf_counter()
    if mode=='reference':
        result=check_certificate(path,require_witness=True)
    elif mode=='dyadic':
        result=verify(path)
    elif mode=='segmented':
        result=verify(path)
        if result['proof_level']!='segmented-full-witness':
            raise ValueError('segmented witness required')
    else:raise ValueError('unknown worker mode')
    seconds=perf_counter()-t0
    return {'backend':mode,'seconds':seconds,'peak_rss_kib':(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                       if resource is not None else None),**result}


def fresh_worker(mode,path):
    p=subprocess.run([sys.executable,__file__,'--worker',mode,str(path)],
                     capture_output=True,text=True,check=True)
    return json.loads(p.stdout)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path)
    parser.add_argument('--worker',choices=['reference','dyadic','segmented'])
    parser.add_argument('proof',nargs='?',type=Path)
    a=parser.parse_args()
    if a.worker:
        if a.proof is None:parser.error('worker needs certificate filename')
        print(json.dumps(worker(a.worker,a.proof),sort_keys=True))
        return
    if a.out_dir is None:parser.error('--out-dir required')
    from qc15_checker import read_proof,write_proof
    from qc15_emit import export_witness
    from qc16_segments import split_witness
    from channel_mpo import gate_layers,kraus
    a.out_dir.mkdir(parents=True,exist_ok=True)
    refs=[
        (3,3,3,4,a.out_dir/'full_3q_d3_dephasing_s3_c4.json.gz'),
        (10,6,3,4,a.out_dir/'full_10q_d6_dephasing_s3_c4.json.gz'),
        (12,8,5,8,a.out_dir/'full_12q_d8_dephasing_s5_c8.json.gz')
    ]
    rows=[]
    for n,depth,seed,cap,file in refs:
        if not file.exists():
            obj=export_witness(n,gate_layers(n,depth,seed),
                               kraus('dephasing',.5),(0,),'Z',bond_cap=cap)
            write_proof(obj,file)
        full=read_proof(file)
        directory=a.out_dir/(f'qc16_{n}q_d{depth}_segments')
        manifest=split_witness(full,directory,events_per_segment=16)
        reference=fresh_worker('reference',file)
        fast=fresh_worker('dyadic',file)
        segmented=fresh_worker('segmented',manifest)
        for r in (fast,segmented):
            for key in ('verified_bound','arithmetic_lower_requirement',
                        'events_checked','reported_expectation'):
                assert r[key]==reference[key],(n,key,r[key],reference[key])
        compressed=sum(p.stat().st_size for p in directory.glob('segment_*.json.gz'))
        row={'qubits':n,'depth':depth,'seed':seed,'events_checked':reference['events_checked'],
             'original_bytes':file.stat().st_size,
             'segments':len(list(directory.glob('segment_*.json.gz'))),
             'segmented_bytes':compressed+manifest.stat().st_size,
             'fraction_seconds':reference['seconds'],'dyadic_seconds':fast['seconds'],
             'segmented_seconds':segmented['seconds'],
             'dyadic_speedup':reference['seconds']/fast['seconds'],
             'segmented_speedup':reference['seconds']/segmented['seconds'],
             'fraction_peak_kib':reference['peak_rss_kib'],
             'dyadic_peak_kib':fast['peak_rss_kib'],
             'segmented_peak_kib':segmented['peak_rss_kib'],
             'verified_bound':segmented['verified_bound'],
             'status':'PASS'}
        rows.append(row)
        print(f'{n}q d{depth} events={row["events_checked"]} '
              f'{row["fraction_seconds"]:.3f}s -> '
              f'{row["dyadic_seconds"]:.3f}s integer, '
              f'{row["segmented_seconds"]:.3f}s segmented',flush=True)
    out=a.out_dir/'qc16_small_benchmark.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(out)

if __name__=='__main__':main()
