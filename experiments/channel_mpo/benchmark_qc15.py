"""Generate QC-15 proof witnesses, check them independently and freeze metrics.

Reproduction:
  OPENBLAS_NUM_THREADS=1 python benchmark_qc15.py --grid small --out-dir /tmp/qc15
  OPENBLAS_NUM_THREADS=1 python benchmark_qc15.py --grid deep --out-dir /tmp/qc15
  python qc15_checker.py /tmp/qc15/full_10q_d6_dephasing_s3_c4.json.gz --require-witness
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from time import perf_counter
from channel_mpo import gate_layers,kraus
from qc15_emit import export_witness,export_ledger
from qc15_checker import check_certificate,write_proof


def configs(grid):
    if grid=='small':
        for n,d,c,seeds in ((3,3,4,(3,5)),(10,6,4,(3,)),(12,8,8,(5,))):
            for noise in ('dephasing','depolarizing','amplitude_damping'):
                for seed in seeds:
                    yield ('full',n,d,noise,.5,seed,c)
    elif grid=='deep':
        yield ('ledger',48,16,'dephasing',.5,3,8)
    else:
        raise ValueError('unknown grid')


def run_one(config,out_dir:Path):
    tier,n,depth,noise,p,seed,cap=config
    name=f'{tier}_{n}q_d{depth}_{noise}_s{seed}_c{cap}.json.gz'
    path=out_dir/name
    layers=gate_layers(n,depth,seed)
    ops=kraus(noise,p)
    t0=perf_counter()
    if tier=='full':
        obj=export_witness(n,layers,ops,(0 if n<=12 else 22,),'Z',bond_cap=cap)
    else:
        obj=export_ledger(n,layers,ops,(22,),'Z',bond_cap=cap)
    emit_seconds=perf_counter()-t0
    write_proof(obj,path)
    t0=perf_counter()
    r=check_certificate(path,require_witness=(tier=='full'))
    verify_seconds=perf_counter()-t0
    event_count=r['events_checked']
    row=dict(tier=tier,n=n,depth=depth,noise=noise,p=p,seed=seed,bond_cap=cap,
             filename=name,bytes=path.stat().st_size,events=event_count,
             emit_seconds=emit_seconds,verify_seconds=verify_seconds,
             reported_expectation=r['reported_expectation'],
             verified_bound=r['verified_bound'],
             independently_checked_residuals=(tier=='full'),
             certificate_status='PASS',
             provenance='polar-corrected literal binary64 gates and Kraus operators')
    print(f"{n}q d{depth} {noise} seed={seed} {tier} "
          f"events={event_count} bytes={row['bytes']} "
          f"emit={emit_seconds:.2f}s verify={verify_seconds:.2f}s "
          f"bound={row['verified_bound']:.6g}",flush=True)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid',choices=('small','deep'),default='small')
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    rows=[run_one(c,a.out_dir) for c in configs(a.grid)]
    filename=a.out_dir/f'qc15_{a.grid}_bench.csv'
    with filename.open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    print('Wrote',filename,flush=True)

if __name__=='__main__':main()
