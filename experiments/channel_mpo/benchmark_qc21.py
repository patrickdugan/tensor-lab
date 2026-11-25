"""Paired peak-memory benchmark of QC-20 and QC-21 independent checkers."""
from __future__ import annotations
import argparse,csv,json,os,subprocess,sys,time
from pathlib import Path
from benchmark_qc18 import _tree,_memory
HERE=Path(__file__).resolve().parent
FIELDS=('verified_bound','arithmetic_lower_requirement','event_counts','events_checked','witness_complex_entries','segments_checked','checkpoints_checked')

def measure(manifest, checker, workers, sampling=.05, *, no_site=False):
    if checker not in ('qc20','qc21') or not 1<=workers<=16: raise ValueError('bad configuration')
    cmd=[sys.executable,'-I']+(['-S'] if no_site else [])+[str(HERE/f'{checker}_checker.py'),str(manifest),'--workers',str(workers),'--start-method','fork']
    start=time.perf_counter();peak=[0,0,0]
    with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'}) as p:
        while p.poll() is None:
            values=[_memory(pid) for pid in _tree(p.pid)]
            for i in range(3): peak[i]=max(peak[i],sum(row[i] for row in values))
            time.sleep(sampling)
        out,err=p.communicate()
        if p.returncode: raise RuntimeError(f'{checker}: {err[-2000:]}')
    result=json.loads(out)
    if result.get('status')!='PASS':raise RuntimeError('failed proof')
    return {'checker':checker,'startup':('no-site' if no_site else 'site'),'workers':workers,'seconds':time.perf_counter()-start,'tree_rss_kib':peak[0], 'tree_pss_kib':peak[1], 'tree_uss_kib':peak[2],**{key:result[key] for key in FIELDS}}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--workers',default='4')
    ap.add_argument('--repetitions',type=int,default=2)
    ap.add_argument('--startups',default='site,no-site')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if not 1<=args.repetitions<=10:ap.error('repetitions must be 1..10')
    rows=[]
    for workers in map(int,args.workers.split(',')):
      for rep in range(args.repetitions):
        for startup in args.startups.split(','):
          if startup not in ('site','no-site'): raise ValueError('unknown startup mode')
          no_site=startup=='no-site'
          base=measure(args.manifest,'qc20',workers,no_site=no_site)
          candidate=measure(args.manifest,'qc21',workers,no_site=no_site)
          for key in FIELDS:
              if base[key]!=candidate[key]:raise AssertionError((key,base[key],candidate[key]))
          for row in (base,candidate):
              rows.append({'repeat':rep,**row})
              print(row['checker'],startup,workers,rep,f"{row['seconds']:.3f}s USS {row['tree_uss_kib']/1024:.1f}MiB",row['verified_bound'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
if __name__=='__main__':main()
