"""Greedy search for a small sub-box of supp(S^{(x)2}) (block form, p = 2) on which
two diagonal copies already fail (no completion of border rank <= 2).

A sub-box keeps subsets of the X-, Y- and Z-coordinates; its support is the part
of the full support inside the box, with the same targets.  Restricting a
completion G to a sub-box is a restriction, so infeasibility on a sub-box implies
infeasibility on the whole problem (and, for n = 3, on every n >= 3).

Feasibility test (floating point): regularised continuation with 2 copies down to
mu = 1e-10 from several starts; 'infeasible' when every start stays above a
residual threshold.  Run from tensor-lab/ with OMP_NUM_THREADS=1.
Usage: python scripts/asymptotic/minimal_box.py n [starts] [threshold]"""
import json
import sys
import time

sys.path.insert(0, '.')
import numpy as np

from tensorlab.numeric_restriction import _setup_pattern, border_path

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
STARTS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
THRESH = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-2
MUS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10)

rows_full, target_full, dims, _, _ = _setup_pattern(2, n, 2)


def sub_pattern(keep):
    kx, ky, kz = keep
    mask = np.isin(rows_full[:, 0], kx) & np.isin(rows_full[:, 1], ky) & np.isin(rows_full[:, 2], kz)
    r = rows_full[mask]
    # re-index to compact coordinates
    mx = {v: i for i, v in enumerate(sorted(kx))}
    my = {v: i for i, v in enumerate(sorted(ky))}
    mz = {v: i for i, v in enumerate(sorted(kz))}
    rr = np.array([[mx[a], my[b], mz[c]] for a, b, c in r]) if len(r) else np.zeros((0, 3), int)
    return rr, target_full[mask], (len(kx), len(ky), len(kz))


def best_residual(keep, starts=STARTS):
    pat = sub_pattern(keep)
    if len(pat[1]) == 0 or pat[1].sum() == 0:
        return 0.0
    out = border_path(2, n, 2, 2, mus=MUS, starts=starts, iters=200, pattern=pat)
    path = out['path']
    final, mid = path[-1][1], path[-3][1]          # mu = 1e-10 and mu = 1e-6
    # count as infeasible only on a flat tail (a plateau, not a slow border descent)
    return final if final > 0.5 * mid else min(final, THRESH / 10)


def prune(keep):
    """Drop coordinates that touch no support term of the box."""
    kx, ky, kz = (set(k) for k in keep)
    rr = rows_full[np.isin(rows_full[:, 0], list(kx)) & np.isin(rows_full[:, 1], list(ky)) & np.isin(rows_full[:, 2], list(kz))]
    return (sorted(set(rr[:, 0]) & kx), sorted(set(rr[:, 1]) & ky), sorted(set(rr[:, 2]) & kz))


if __name__ == '__main__':
    t0 = time.time()
    keep = (list(range(dims[0])), list(range(dims[1])), list(range(dims[2])))
    r0 = best_residual(keep)
    print(f'n={n}: full box dims {dims}, two-copy residual {r0:.3e} [{time.time() - t0:.0f}s]', flush=True)
    assert r0 > THRESH, 'full problem looks feasible; nothing to minimise'
    changed = True
    while changed:
        changed = False
        for leg in (1, 2, 0):
            for v in list(keep[leg]):
                trial = [list(k) for k in keep]
                trial[leg].remove(v)
                trial = prune(tuple(trial))
                if not all(trial):
                    continue
                r = best_residual(trial)
                if r > THRESH:
                    keep = trial
                    changed = True
                    sz = len(sub_pattern(keep)[1])
                    print(f'  dropped leg {leg} coord {v}: box {tuple(len(k) for k in keep)}, support {sz}, residual {r:.3e} [{time.time() - t0:.0f}s]', flush=True)
    pat = sub_pattern(keep)
    final = best_residual(keep, starts=3 * STARTS)
    print(f'MINIMAL box {tuple(len(k) for k in keep)}, support {len(pat[1])} '
          f'(retained {int(pat[1].sum())}), residual with {3 * STARTS} starts {final:.3e}', flush=True)
    with open(f'logs/minimal_box_n{n}.json', 'w') as f:
        json.dump({'n': n, 'keep': [list(map(int, k)) for k in keep], 'residual': final,
                   'support': len(pat[1]), 'retained': int(pat[1].sum())}, f, indent=1)
    print('DONE', flush=True)
