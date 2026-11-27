"""Exploratory repair of missing-coordinate collisions, six matched toy cases."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from lattice_pilot import job
if __name__=='__main__':
    specs=[(28,12289,B,root,'regular','fixed',99) for B in (8,32) for root in range(3)]
    with ProcessPoolExecutor(max_workers=4) as pool:records=list(pool.map(job,specs))
    for r in records:print(r['B'],r['root'],r['status'],r['rank'],r['cpu_seconds'])
    Path(__file__).with_name('AUDIT_regular.json').write_text(json.dumps(dict(scope='EXPLORATORY_FULL_COVERAGE_SPARSE',records=records),indent=2)+'\n')
