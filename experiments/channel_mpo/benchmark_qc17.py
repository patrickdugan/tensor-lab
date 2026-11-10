"""QC-17 independent-process verification timings and *aggregate* RSS sampling.

Example:
  python benchmark_qc17.py --manifest /path/to/manifest.json --out /tmp/qc17.csv
  python benchmark_qc17.py --small --out /tmp/qc17-small.csv

Linux /proc supports aggregate parent+worker peak RSS; otherwise field is blank.
Each configuration starts a new Python interpreter with isolated `-I` mode.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
PAGE_KIB=4096/1024


def _descendants(pid):
    """Enumerate all child processes via Linux /proc parent PIDs.

    ProcessPoolExecutor may fork from a non-main thread, so reading only
    /proc/<pid>/task/<pid>/children misses workers. Check all thread parents.
    """
    children={}
    for f in Path('/proc').glob('[0-9]*/stat'):
        try:
            body=f.read_text()
            fields=body[body.rfind(')')+1:].split()
            ppid=int(fields[1])
            children.setdefault(ppid,[]).append(int(f.parent.name))
        except (OSError,IndexError,ValueError):
            continue
    result=[]
    todo=[pid]
    seen=set()
    while todo:
        current=todo.pop()
        if current in seen: continue
        seen.add(current)
        result.append(current)
        todo.extend(children.get(current,()))
    return result


def _rss_kib(pid):
    try:
        resident_pages=int(Path(f'/proc/{pid}/statm').read_text().split()[1])
        return resident_pages*PAGE_KIB
    except (OSError,ValueError,IndexError):return 0.0


def fresh_verify(manifest,workers):
    cmd=[sys.executable,'-I',str(HERE/'qc17_checker.py'),str(manifest),
         '--workers',str(workers)]
    start=time.perf_counter()
    with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) as child:
        peak_tree=0.0
        max_single=0.0
        while child.poll() is None:
            if Path('/proc/self/statm').exists():
                memory=[_rss_kib(pid) for pid in _descendants(child.pid)]
                peak_tree=max(peak_tree,sum(memory))
                max_single=max(max_single,max(memory,default=0.0))
            time.sleep(0.03)
        stdout,stderr=child.communicate()
        if child.returncode:
            raise RuntimeError(f'QC17 worker failed: {stderr[-2000:]}')
        result=json.loads(stdout)
    elapsed=time.perf_counter()-start
    return {**result,'seconds':elapsed,'peak_tree_rss_kib':int(peak_tree) if peak_tree else None,
            'sampled_max_single_rss_kib':int(max_single) if max_single else None}


def benchmark(manifest,counts=(1,2,4)):
    records=[]
    baseline=None
    for count in counts:
        result=fresh_verify(manifest,count)
        if baseline is None:baseline=result
        for key in ('verified_bound','arithmetic_lower_requirement','events_checked',
                    'reported_expectation','witness_complex_entries','event_counts'):
            assert result[key]==baseline[key],key
        records.append({'workers':count,'seconds':result['seconds'],
            'relative_speedup':baseline['seconds']/result['seconds'],
            'peak_tree_rss_kib':result['peak_tree_rss_kib'],
            'sampled_max_single_rss_kib':result['sampled_max_single_rss_kib'],
            'event_count':result['events_checked'],
            'segments':result['segments_checked'],
            'verified_bound':result['verified_bound'],
            'witness_complex_entries':result['witness_complex_entries']})
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--small',action='store_true')
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--workers',default='1,2,4')
    a=parser.parse_args()
    counts=tuple(int(x) for x in a.workers.split(','))
    if a.manifest is None and not a.small:parser.error('specify --manifest or --small')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='qc17bench_') as d:
        manifest_paths=[]
        if a.manifest is not None:
            manifest_paths=[('provided',a.manifest)]
        else:
            from qc17_emit import produce
            for n,depth,seed,cap in ((3,3,3,4),(10,6,3,4),(12,8,5,8)):
                fname=f'full_{n}q_d{depth}_dephasing_s{seed}_c{cap}.json.gz'
                full=HERE/'qc15_certificates'/fname
                if not full.is_file():
                    full=HERE.parent.parent.parent/'bench'/fname
                if not full.is_file():
                    from qc15_emit import export_witness
                    from qc15_checker import write_proof
                    from channel_mpo import gate_layers,kraus
                    full=Path(d)/fname
                    write_proof(export_witness(n,gate_layers(n,depth,seed),
                           kraus('dephasing',.5),(0,),'Z',bond_cap=cap),full)
                dest=Path(d)/f'{n}q'
                manifest_paths.append((f'{n}q_d{depth}',produce(full,dest,events_per_segment=16)))
        for label,path in manifest_paths:
            for record in benchmark(path,counts=counts):
                rows.append({'case':label,**record})
                print(f"{label} workers={record['workers']} seconds={record['seconds']:.3f} "
                      f"speedup={record['relative_speedup']:.2f}x",flush=True)
    require_out=a.out
    require_out.parent.mkdir(parents=True,exist_ok=True)
    with require_out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    print(require_out)

if __name__=='__main__':main()
