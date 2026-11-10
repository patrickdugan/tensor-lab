"""QC-21 CLI: independently checked witness replay with live norm summaries.

Reuses QC-20's proof-format parsing, mandatory event checks, and exact integer
residual code. Changes only the integer-core cache eviction strategy and saves
exact squared-norm summaries for unchanged live cores. Supports fork and spawn.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from qc15_checker import CertificateError
from qc20_checker import verify as _verify


def verify(manifest_path, workers=1, *, start_method='auto'):
    return _verify(manifest_path, workers=workers,
                   start_method=start_method, storage_mode='summary')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--start-method', choices=('auto', 'fork', 'spawn'), default='auto')
    args = parser.parse_args()
    try:
        result = verify(args.manifest, workers=args.workers,
                        start_method=args.start_method)
    except (CertificateError, ValueError, TypeError, OverflowError,
            MemoryError, OSError, KeyError, IndexError) as exc:
        parser.exit(2, f'FAIL: {exc}\n')
    print(json.dumps({'status': 'PASS', **result}, sort_keys=True))


if __name__ == '__main__':
    main()
