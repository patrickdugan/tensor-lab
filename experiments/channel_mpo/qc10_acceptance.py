"""QC-10: post-run operational-error gate for nominated local regions.

The local reference is an exact finite causal-cone simulation in ideal
arithmetic, computed numerically. Selection does not certify the global
48-qubit state, hidden regions, or floating-point round-off. All modes must
be evaluated using the same gates, channels, and frozen retained-entry budget.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np

REQUIRED_REGIONS=('left_1','left_2','right_1','right_2')


def select_acceptable(rows:list[dict], tolerance:float=0.05,*,
                      required_regions:tuple[str,...]=REQUIRED_REGIONS)->list[dict]:
    if not np.isfinite(tolerance) or tolerance<0 or tolerance>1:
        raise ValueError('operational tolerance must be in [0,1]')
    if not required_regions or len(set(required_regions))!=len(required_regions):
        raise ValueError('required region names must be unique')
    grouped={}
    for r in rows:
        key=(int(r['n']),str(r['channel']),float(r['p']),int(r['seed']),int(r['depth']))
        ident=(str(r['mode']),str(r['region']))
        if ident in grouped.setdefault(key,{}):raise ValueError(f'duplicate observation: {key} {ident}')
        grouped[key][ident]=r
    result=[]
    for key,rec in sorted(grouped.items()):
        modes=set(m for m,_ in rec)
        if not {'peak','sketch'} <= modes:
            raise ValueError(f'missing mode for {key}')
        for mode in ('peak','sketch'):
            if not set(required_regions) <= {region for m,region in rec if m==mode}:
                raise ValueError(f'missing observable region in {key}: {mode}')
        worst={mode:max(float(rec[(mode,label)]['local_trace_distance']) for label in required_regions)
               for mode in ('peak','sketch')}
        if any(not np.isfinite(x) or x<0 or x>1+1e-8 for x in worst.values()):
            raise ValueError('invalid density distance')
        chosen='peak' if worst['peak']<=tolerance else 'sketch' if worst['sketch']<=tolerance else 'reject'
        selected=rec[(chosen if chosen!='reject' else 'sketch',required_regions[0])]
        result.append({'n':key[0],'channel':key[1],'p':key[2],
          'seed':key[3],'depth':key[4],'event_error_tolerance':tolerance,
          'decision':chosen,'max_d_peak':worst['peak'],'max_d_sketch':worst['sketch'],
          'max_d_selected':worst.get(chosen,float('nan')),
          'modeled_live_peak_selected':selected['modeled_live_peak_entries'] if chosen!='reject' else '',
          'retained_entries_selected':selected['retained_entries'] if chosen!='reject' else '',
          'simulation_seconds_selected':selected['sim_seconds'] if chosen!='reject' else ''})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--tolerance',type=float,default=.05)
    parser.add_argument('--output',type=Path,default=Path('qc10_acceptance.csv'))
    args=parser.parse_args()
    with args.source.open(newline='') as inp:rows=list(csv.DictReader(inp))
    decisions=select_acceptable(rows,args.tolerance)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(decisions[0]))
        writer.writeheader();writer.writerows(decisions)
    counts={name:sum(row['decision']==name for row in decisions) for name in ('peak','sketch','reject')}
    print(f'{len(decisions)} cases: {counts}; saved {args.output}')


if __name__=='__main__':main()
