"""SYNTHETIC DRY RUN counts for the QC-26B prospective pipeline. NOT HARDWARE DATA.

  python qc26b_synth_counts.py --frozen frozen.json --out counts.json [--model C3] [--seed 1]

Each measured bit k of circuit c is drawn independently with P(bit=1) = (1 - z)/2, where z is the frozen
prediction of the chosen model (default C3) for that site, so a shot is Bernoulli noise around the model's
own marginal. Correlations between sites are not modelled (the evaluator only uses single-site marginals).
Output follows the counts.json schema of the evaluator and is labelled synthetic in every entry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LABEL = 'SYNTHETIC DRY RUN - generated from frozen model predictions plus Bernoulli shot noise; NOT HARDWARE DATA'


def synth_counts(frozen: dict, model: str = 'C3', seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for cid in sorted(frozen['predictions']):
        info = frozen['build']['circuits'][cid]
        n, shots = info['n'], info['shots']
        z = np.array([s[model] for s in frozen['predictions'][cid]])
        if len(z) != n:
            raise ValueError(f'site count mismatch for {cid}')
        p1 = np.clip((1.0 - z) / 2.0, 0.0, 1.0)
        bits = rng.random((shots, n)) < p1                     # column k is bit k
        counts: dict[str, int] = {}
        for row in bits:
            key = ''.join('1' if row[k] else '0' for k in range(n - 1, -1, -1))   # little-endian: bit k at index n-1-k
            counts[key] = counts.get(key, 0) + 1
        out[cid] = {'counts': counts, 'physical_path': info['physical_path'], 'synthetic': True,
                    'label': LABEL, 'generated_from_model': model, 'seed': seed}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--frozen', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', default='C3', choices=['ideal', 'C0', 'C1', 'C2', 'C3', 'C3Z'])
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args(argv)
    frozen = json.loads(a.frozen.read_text())
    a.out.write_bytes((json.dumps(synth_counts(frozen, a.model, a.seed), sort_keys=True) + '\n').encode())
    print(f'{LABEL}: wrote {a.out}')


if __name__ == '__main__':
    main()
