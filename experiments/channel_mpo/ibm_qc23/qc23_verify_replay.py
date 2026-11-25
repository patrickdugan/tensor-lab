"""Compare a committed analysis JSON with a freshly replayed one (CI helper)."""
import json
import math
import sys


def close(a, b, tol=1e-6):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    return a == b


def main(committed, replay):
    a, b = json.load(open(committed)), json.load(open(replay))
    bad = []
    for k in ('data_source', 'backend', 'calibration_timestamp'):
        if a[k] != b[k]:
            bad.append(k)
    for k, v in a['model_b'].items():
        if not close(v, b['model_b'][k], 1e-5):
            bad.append(f'model_b.{k}')
    for fam, preds in a['held_out'].items():
        for key, rep in preds.items():
            other = b['held_out'][fam][key]
            if (rep is None) != (other is None):
                bad.append(f'{fam}.{key}')
            elif rep:
                for f in ('fraction', 'mean_abs_error', 'within', 'observables'):
                    if not close(rep[f], other[f], 1e-5):
                        bad.append(f'{fam}.{key}.{f}')
    if len(a['rows']) != len(b['rows']):
        bad.append('row count')
    if bad:
        print('REPLAY MISMATCH:', bad)
        raise SystemExit(1)
    print('replay matches committed analysis:', len(a['rows']), 'rows')


if __name__ == '__main__':
    main(*sys.argv[1:3])
