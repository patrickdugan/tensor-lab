"""Cone-wider-than-dense scale test; writes results/scale_test.json (no QPU data involved)."""
import json
import sys
from pathlib import Path

import qiskit_ibm_runtime.fake_provider as fp

from qc23_analysis import scale_test

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'results')
snap = json.loads((root / 'compiled' / 'calibration_snapshot.json').read_text())
rows = scale_test(snap, fp.FakeFez())
(root / 'scale_test.json').write_text(json.dumps(rows, indent=1, default=float) + '\n')
for r in rows:
    print({k: v for k, v in r.items() if k != 'by_bond_cap'})
