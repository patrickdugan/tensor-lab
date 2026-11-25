"""Fresh-process QC-17/QC-18 comparison with RSS, PSS and USS sampling.

RSS sums double-count shared mappings. PSS apportions each shared page among
processes and USS measures unique private pages. Each run is a single sample.
"""
from __future__ import annotations
import argparse,csv,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def _tree(root):
    children={}
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            s=p.read_text(); fields=s[s.rfind(')')+1:].split()
            children.setdefault(int(fields[1]),[]).append(int(p.parent.name))
        except (OSError,ValueError,IndexError):continue
    todo=[root];visited=set()
    while todo:
        pid=todo.pop()
        if pid in visited:continue
        visited.add(pid);todo.extend(children.get(pid,()))
    return visited

def _memory(pid):
    try:
        d={}
        for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
            k,_,v=line.partition(':')
            if v.strip().endswith('kB'):
                try:d[k]=int(v.split()[0])
                except ValueError:pass
        return d.get('Rss',0),d.get('Pss',0),d.get('Private_Clean',0)+d.get('Private_Dirty',0)
    except (OSError,ValueError):return 0,0,0

def run(manifest,backend,workers,sample=.08):
    cmd=[sys.executable,'-I',str(HERE/('qc18_checker.py' if backend=='qc18' else 'qc17_checker.py')),str(manifest),'--workers',str(workers)]
    if backend=='qc18':cmd+=['--start-method','auto']
    start=time.perf_counter()
    with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                          env={**os.environ,'OPENBLAS_NUM_THREADS':'1'}) as child:
        peak=[0,0,0]
        while child.poll() is None:
            if Path('/proc/self/smaps_rollup').exists():
                mm=[_memory(pid) for pid in _tree(child.pid)]
                for k in range(3):peak[k]=max(peak[k],sum(x[k] for x in mm))
            time.sleep(sample)
        stdout,stderr=child.communicate()
        if child.returncode:raise RuntimeError(stderr[-2500:])
        result=json.loads(stdout)
    return {**result,'seconds':time.perf_counter()-start,
            'max_tree_rss_kib':peak[0],'max_tree_pss_kib':peak[1],
            'max_tree_uss_kib':peak[2]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qc17',type=Path,required=True)
    p.add_argument('--qc18',type=Path,required=True)
    p.add_argument('--qc18-raw',type=Path)
    p.add_argument('--workers',default='1,2,4')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    backends=[('qc17',a.qc17),('qc18-zlib',a.qc18)]
    if a.qc18_raw:backends.append(('qc18-raw',a.qc18_raw))
    rows=[];control=None
    for workers in map(int,a.workers.split(',')):
        for label,manifest in backends:
            result=run(manifest,'qc17' if label=='qc17' else 'qc18',workers)
            if control is None:control=result
            for key in ('verified_bound','arithmetic_lower_requirement',
                        'event_counts','events_checked','witness_complex_entries'):
                if result[key]!=control[key]:raise AssertionError((label,key))
            rows.append({k:result[k] for k in ('seconds','max_tree_rss_kib',
                          'max_tree_pss_kib','max_tree_uss_kib','verified_bound',
                          'events_checked','witness_complex_entries')}|
                        {'backend':label,'workers':workers})
            print(f'{label} workers={workers}: {result["seconds"]:.3f}s '
                  f'RSS={result["max_tree_rss_kib"]/1024:.1f} MiB '
                  f'PSS={result["max_tree_pss_kib"]/1024:.1f} MiB '
                  f'USS={result["max_tree_uss_kib"]/1024:.1f} MiB',flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
