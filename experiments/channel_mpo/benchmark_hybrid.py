"""QC-03: optional Kraus unravelling mixture, memory/variance frontier.

Every mixture is the same physical depolarizing quantum channel.
"""
import csv
from pathlib import Path
from trajectory_oracle import sample_ensemble


def main():
    cases=[(20,64),(48,24)]
    mixtures=(.25,.5,.75)
    output=Path('qc03_hybrid_frontier.csv')
    rows=[]
    for n,shots in cases:
        for mix in mixtures:
            row=sample_ensemble(n,6,0,'depolarizing',.5,'hybrid',shots,None,
                                reset_mix=mix)
            row['z0_variance_per_cpu_s_proxy']=row['z0_se']**2*row['seconds']
            rows.append(row)
            with output.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
            print(f"n={n} α={mix:.2f} bond={row['max_observed_bond']} "
                  f"mean_entries={row['mean_stored_complex_entries']:.0f} "
                  f"z0se={row['z0_se']:.5f} time={row['seconds']:.1f}s",flush=True)
    print('wrote',output,flush=True)

if __name__=='__main__':main()
