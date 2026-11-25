"""QC-03 pilot matrix. Fixed shots per size; exact local Z oracle at n<=10.

The stdout is flushed per completed arm. Hardware timing is single-run only.
"""
import csv
from pathlib import Path
from trajectory_oracle import sample_ensemble


def main():
    cases = [(6,.2,256),(6,.5,256),(8,.2,256),(8,.5,256),
             (20,.5,64),(48,.5,24)]
    out=Path('qc03_trajectory_comparison.csv')
    rows=[]
    for n,p,shots in cases:
        for scheme in ('pauli','reset'):
            row=sample_ensemble(n,6,0,'depolarizing',p,scheme,shots,None,
                                exact_observables=n<=10)
            rows.append(row)
            print(f"n={n} p={p} {scheme} shots={shots} z0={row['z0_mean']:+.5f} "
                  f"se={row['z0_se']:.5f} maxbond={row['max_observed_bond']} "
                  f"entries={row['max_stored_complex_entries']} "
                  f"time={row['seconds']:.1f}s",flush=True)
            with out.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                w.writeheader();w.writerows(rows)
    print(f'wrote {len(rows)} rows to {out}',flush=True)

if __name__=='__main__':main()
