"""QC-19 isolated CPU benchmark; samples total process-tree RSS, PSS and USS.

Use a fixed already-pinned QC-18 witness as input. Every verifier must produce
identical exact results before its timings are compared. Samples are Linux
/proc statistics (may miss short-lived peaks); report repetitions explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_qc18 import _tree, _memory

HERE = Path(__file__).resolve().parent
FIELDS = ('verified_bound', 'arithmetic_lower_requirement', 'event_counts',
          'events_checked', 'witness_complex_entries', 'segments_checked',
          'checkpoints_checked')


def run(manifest: Path, *, checker: str, workers: int, sampling: float = .05):
    if checker not in ('qc18', 'qc19') or not 1 <= workers <= 16:
        raise ValueError('invalid checker or worker count')
    cmd = [sys.executable, '-I', str(HERE / f'{checker}_checker.py'),
           str(manifest), '--workers', str(workers), '--start-method', 'fork']
    start = time.perf_counter()
    peak = [0, 0, 0]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'}) as p:
        while p.poll() is None:
            sampled = [_memory(pid) for pid in _tree(p.pid)]
            for i in range(3):
                peak[i] = max(peak[i], sum(row[i] for row in sampled))
            time.sleep(sampling)
        stdout, stderr = p.communicate()
        if p.returncode:
            raise RuntimeError(f'{checker} verifier failed: {stderr[-3000:]}')
    result = json.loads(stdout)
    return {**result, 'checker': checker, 'workers': workers,
            'seconds': time.perf_counter() - start,
            'tree_rss_kib': peak[0], 'tree_pss_kib': peak[1],
            'tree_uss_kib': peak[2]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--workers', default='1,2,4')
    ap.add_argument('--repetitions', type=int, default=1)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.repetitions < 1 or a.repetitions > 10:
        ap.error('repetitions must be 1 to 10')
    records = []
    for w in (int(s) for s in a.workers.split(',')):
        for rep in range(a.repetitions):
            baseline = run(a.manifest, checker='qc18', workers=w)
            candidate = run(a.manifest, checker='qc19', workers=w)
            for key in FIELDS:
                if baseline[key] != candidate[key]:
                    raise AssertionError(f'{key}: {baseline[key]} != {candidate[key]}')
            for record in (baseline, candidate):
                records.append({key: record[key] for key in
                    ('checker', 'workers', 'seconds', 'tree_rss_kib',
                     'tree_pss_kib', 'tree_uss_kib', *FIELDS)} | {'repeat': rep})
                print(record['checker'], w, rep,
                      round(record['seconds'],3),
                      round(record['tree_uss_kib']/1024,1),
                      record['verified_bound'], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


if __name__ == '__main__':
    main()
