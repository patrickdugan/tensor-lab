"""Compare a freshly recomputed QC-26B replay against the committed receipt (floats to 1e-9, structure exact)."""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])


def close(x, y, tol=1e-9):
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(close(x[k], y[k], tol) for k in x)
    if isinstance(x, list):
        return len(x) == len(y) and all(close(u, v, tol) for u, v in zip(x, y))
    if isinstance(x, float) or isinstance(y, float):
        return x is not None and y is not None and abs(x - y) <= tol * max(1.0, abs(x), abs(y))
    return x == y


assert a['counts'] == b['counts'] == {'circuits': 48, 'observables': 528}
assert close(a['fit_report']['delta'], b['fit_report']['delta']), 'fitted ZZ phases differ'
assert a['sx_offsets_rad'] == b['sx_offsets_rad']
assert len(a['rows']) == len(b['rows']) == 528
for x, y in zip(a['rows'], b['rows']):
    assert (x['id'], x['bit'], x['set']) == (y['id'], y['bit'], y['set'])
    assert all(abs(x[k] - y[k]) < 1e-9 for k in ('measured', 'ideal', 'C0', 'C1', 'C2', 'C3')), (x['id'], x['bit'])
assert close(a['metrics'], b['metrics']) and close(a['engine_verification'], b['engine_verification'], 1e-6)
assert a['engine_verification']['c0_c1_c2_vs_qc25c_receipts']['max_abs_gap'] < 1e-9
print('QC26B replay receipt verified: 528 observables')
